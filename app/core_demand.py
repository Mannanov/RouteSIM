"""Восстановление остановочной матрицы спроса.

Исходные CSV:
- matrix_mobile: source, target, period, demand;
- ref_tr_distr: id_tr_distr, wkt, опционально pt_share_origin;
- ref_stops: id_stop, wkt;
- los: source, target, all_vars, los_speed, los_carry.
- carry (опционально): rnd, h5, h6, ..., h23, day;
- ref_links (для маршрутной калибровки): link, RNDs;
- route_boardings (для маршрутной калибровки): route, flow.

source и target в matrix_mobile — транспортные районы, а в los —
остановочные пункты. period задается номером часа от 5 до 23.
"""

import ast
import logging
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from flask import current_app, session
from scipy import sparse

from serv_funcs import path_gen, select_utm_crs, build_stop_access


logger = logging.getLogger('rsim.demand')


class _DemandLogger(logging.LoggerAdapter):
    """Добавляет пользователя и проект к этапам расчета в журнале приложения."""

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            self.logger.log(level, '%s ' + msg, self.extra['prefix'], *args, **kwargs)


HOUR_COLUMNS = [f'h{hour}' for hour in range(5, 24)]
ROUTE_PRIOR_WEIGHT = 0.10
ROUTE_MAX_ITERATIONS = 200
ROUTE_CONVERGENCE_TOLERANCE = 1e-6
ROUTE_STATIONARITY_TOLERANCE = 1e-6
ROUTE_CONVERGENCE_STREAK = 5


def _require_columns(df, columns, name):
    """Проверяет наличие обязательных столбцов в таблице"""
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f'{name}: отсутствуют столбцы {sorted(missing)}')


def _hour_name(value):
    """Проверяет целочисленный номер часа и возвращает внутреннее имя hN."""
    value = str(value).strip()
    if value.lower().startswith('h'):
        raise ValueError('В period номер часа задается без префикса h')
    numeric_hour = float(value)
    if not numeric_hour.is_integer():
        raise ValueError(f'Час должен быть целым числом: {value}')
    hour = int(numeric_hour)
    if hour not in range(5, 24):
        raise ValueError(f'Некорректный час: {value}')
    return f'h{hour}'


def _prepare_demand_inputs(
    demand,
    ref_districts,
    ref_stops,
    los,
    calculation_period,
    demand_source='source',
    demand_target='target',
    demand_period='period',
    demand_value='demand',
):
    """Проверяет исходные таблицы и приводит спрос к выбранному периоду."""
    _require_columns(
        demand, [demand_source, demand_target, demand_period, demand_value],
        'demand',
    )
    _require_columns(ref_districts, ['id_tr_distr', 'wkt'], 'ref_districts')
    _require_columns(ref_stops, ['id_stop', 'wkt'], 'ref_stops')
    _require_columns(
        los, ['source', 'target', 'all_vars', 'los_speed', 'los_carry'], 'los',
    )
    if ref_districts['id_tr_distr'].duplicated().any():
        raise ValueError('В справочнике районов есть повторяющиеся id')
    if ref_stops['id_stop'].duplicated().any():
        raise ValueError('В справочнике остановок есть повторяющиеся id')

    demand = demand[[
        demand_source, demand_target, demand_period, demand_value,
    ]].rename(columns={
        demand_source: 'district_source',
        demand_target: 'district_target',
        demand_period: 'period',
        demand_value: 'mobile_demand',
    })
    demand['period'] = demand['period'].map(_hour_name)
    demand['mobile_demand'] = pd.to_numeric(
        demand['mobile_demand'], errors='raise',
    )
    if (
        demand['mobile_demand'].isna().any()
        or (demand['mobile_demand'] < 0).any()
    ):
        raise ValueError('Спрос должен быть заполненным неотрицательным числом')
    district_ids = set(ref_districts['id_tr_distr'])
    if (
        ~demand['district_source'].isin(district_ids)
        | ~demand['district_target'].isin(district_ids)
    ).any():
        raise ValueError('Матрица спроса содержит неизвестные районы')
    stop_ids = set(ref_stops['id_stop'])
    if (~los['source'].isin(stop_ids) | ~los['target'].isin(stop_ids)).any():
        raise ValueError('LOS содержит неизвестные остановки')

    demand = demand.groupby(
        ['district_source', 'district_target', 'period'], as_index=False,
    )['mobile_demand'].sum()
    if calculation_period != 'day':
        demand = demand.loc[
            demand['period'] == _hour_name(calculation_period)
        ].reset_index(drop=True)
        if demand.empty:
            raise ValueError(
                f'В matrix_mobile отсутствуют данные за час {calculation_period}'
            )
    return demand


def _validate_calibration_parameters(
    calibration_mode, total_boardings_period, transfer_factor,
):
    """Проверяет параметры общего и маршрутного объема поездок."""
    if calibration_mode not in {'pt_share', 'total_boardings', 'route_boardings'}:
        raise ValueError('Неизвестный режим калибровки')
    if calibration_mode == 'total_boardings' and total_boardings_period is None:
        raise ValueError('Количество поездок должно быть больше 0')
    if calibration_mode in {'total_boardings', 'route_boardings'}:
        if total_boardings_period is not None and total_boardings_period <= 0:
            raise ValueError('Количество поездок должно быть больше 0')
        if transfer_factor is None or not 1 <= transfer_factor <= 2:
            raise ValueError('Коэффициент пересадочности должен быть от 1 до 2')


def _prepare_hourly_carry(carry, require_positive=False):
    """Проверяет часовые значения carry и возвращает числовую таблицу."""
    _require_columns(carry, HOUR_COLUMNS, 'carry')
    hourly = carry[HOUR_COLUMNS].apply(pd.to_numeric, errors='raise')
    if (hourly < 0).any().any():
        raise ValueError('Провозная способность не может быть отрицательной')
    if require_positive and hourly.sum().sum() <= 0:
        raise ValueError('Суммарная провозная способность должна быть больше 0')
    return hourly


def _prepare_los(los):
    """Оставляет уникальные транспортные LOS-пары с положительными весами."""
    los = los.drop_duplicates(['source', 'target']).copy()
    for column in ['los_speed', 'los_carry']:
        los[column] = pd.to_numeric(los[column], errors='coerce')
    los = los.loc[
        (los['all_vars'].astype(str).str.strip() != 'W')
        & (los['los_speed'] > 0)
        & (los['los_carry'] > 0)
    ]
    if los.empty:
        raise ValueError('LOS не содержит положительных транспортных связей')
    # Районная диагональ допустима, остановочная — не является обычной поездкой.
    # Если остаются только такие связи, спрос сохраняется как необслуженный.
    return los.loc[los['source'] != los['target']].copy()


def validate_demand_files(
    demand_path,
    districts_path,
    stops_path,
    los_path,
    calculation_period='day',
    carry_path=None,
    demand_profile_mode='mobile_only',
    calibration_mode='pt_share',
    ref_links_path=None,
    route_boardings_path=None,
):
    """Проверяет структуру и ссылочную целостность комплекта исходных CSV."""
    # Читаем только исходные таблицы, не выполняя геометрические вычисления.
    demand = pd.read_csv(demand_path, sep=';')
    districts = pd.read_csv(districts_path, sep=';')
    stops = pd.read_csv(stops_path, sep=';')
    los = pd.read_csv(los_path, sep=';')
    _prepare_demand_inputs(demand, districts, stops, los, calculation_period)

    # Carry обязателен только для суточного профиля mobile_carry.
    if demand_profile_mode == 'mobile_carry' and not carry_path:
        raise ValueError(
            'Для выбранного профиля требуется файл провозной способности'
        )
    if carry_path:
        carry = pd.read_csv(carry_path, sep=';')
        _prepare_hourly_carry(
            carry, require_positive=demand_profile_mode == 'mobile_carry',
        )

    # Маршрутный режим требует LOS-связи с rnd и хотя бы одно наблюдение.
    if calibration_mode == 'route_boardings':
        if not ref_links_path or not route_boardings_path:
            raise ValueError(
                'Для маршрутной калибровки нужны ref_links и route_boardings'
            )
        ref_links = pd.read_csv(ref_links_path, sep=';')
        _prepare_link_rnds(ref_links)
        route_boardings = pd.read_csv(route_boardings_path, sep=';')
        _prepare_route_observations(route_boardings)


def _prepare_access(
    ref_districts,
    ref_stops,
    access_radius_km,
    min_overlap_share,
):
    """Общий расчёт доступности остановок для demand и forecast."""
    return build_stop_access(ref_districts, ref_stops, access_radius_km, min_overlap_share).rename(
        columns={'id_tr_distr': 'tr_district', 'overlap_share': 'access_weight'})


def _prepare_zone_pairs(
    los,
    access,
):
    """Формирует допустимые остановочные пары и их доли внутри районных OD"""

    # Готовим веса подхода отдельно для отправления и прибытия
    origin_access = access.rename(columns={
        'tr_district': 'district_source',
        'id_stop': 'source',
        'access_weight': 'origin_access',
    })
    target_access = access.rename(columns={
        'tr_district': 'district_target',
        'id_stop': 'target',
        'access_weight': 'target_access',
    })

    # Привязываем подготовленные транспортные LOS-связи к районам.
    pairs = (
        los.merge(origin_access, on='source')
        .merge(target_access, on='target')
    )

    # Объединяем доступность остановок, привлекательность и силу LOS-связи
    pairs['pair_weight'] = (
        pairs['origin_access']
        * pairs['target_access']
        * pairs['los_speed']
        * pairs['los_carry']
    )
    pairs = pairs.loc[pairs['pair_weight'] > 0]

    # Нормируем веса до долей отдельно для каждой пары районов
    totals = pairs.groupby(
        ['district_source', 'district_target']
    )['pair_weight'].transform('sum')
    pairs['pair_share'] = pairs['pair_weight'] / totals

    return pairs[[
        'district_source',
        'district_target',
        'source',
        'target',
        'pair_share',
    ]]


def _parse_list(value, field_name):
    """Разбирает строковое представление списка из LOS-справочников."""
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f'Некорректное значение {field_name}: {value}') from error
    if not isinstance(parsed, list):
        raise ValueError(f'{field_name} должен содержать список')
    return parsed


def _parse_link_rnds(value):
    """Возвращает список rnd остановочной связи либо признак ходьбы."""
    if str(value).strip() == 'W':
        return [None]
    parsed = _parse_list(value, 'RNDs')
    rnds = [str(item).strip() for item in parsed if str(item).strip()]
    if not rnds:
        raise ValueError('RNDs содержит пустой список направлений')
    for rnd in rnds:
        _route_from_rnd(rnd)
    return rnds


def _route_from_rnd(rnd):
    """Возвращает маршрут из rnd формата ``маршрут-направление``."""
    value = str(rnd).strip()
    if '-' not in value:
        raise ValueError(f'Некорректный rnd без разделителя "-": {value}')
    route, direction = value.rsplit('-', 1)
    if not route.strip() or not direction.strip():
        raise ValueError(f'Некорректный rnd: {value}')
    return route.strip()


def _route_counts(rnd_counts):
    """Агрегирует число поездок из направлений до уровня маршрута."""
    counts = Counter()
    for rnd, count in rnd_counts.items():
        counts[_route_from_rnd(rnd)] += count
    return dict(counts)


def _minimal_route_segments(links, candidate_sets):
    """Строит непрерывные реализации с минимальным числом поездок."""
    # Состояние хранит завершенные участки и текущий незавершенный участок.
    states = {(tuple(), None, None, None, 0)}
    for link, candidates in zip(links, candidate_sets):
        try:
            source, target = str(link).split('>', 1)
        except ValueError as error:
            raise ValueError(
                f'Некорректная остановочная связь: {link}'
            ) from error
        next_states = set()
        for completed, current, start, end, count in states:
            for rnd in candidates:
                segments = completed
                if rnd is None:
                    if current is not None:
                        segments += ((current, start, end),)
                    next_states.add((segments, None, None, None, count))
                elif rnd == current:
                    next_states.add((
                        segments, current, start, target, count
                    ))
                else:
                    if current is not None:
                        segments += ((current, start, end),)
                    next_states.add((
                        segments, rnd, source, target, count + 1
                    ))

        # Для одинакового текущего rnd более дорогая история уже не может
        # стать минимальной на оставшейся части цепочки.
        minimum_by_current = {}
        for state in next_states:
            current = state[1]
            minimum_by_current[current] = min(
                minimum_by_current.get(current, state[4]), state[4]
            )
        states = {
            state for state in next_states
            if state[4] == minimum_by_current[state[1]]
        }

    completed_states = []
    for completed, current, start, end, count in states:
        if current is not None:
            completed += ((current, start, end),)
        completed_states.append((completed, count))
    if not completed_states:
        return set()
    minimum = min(count for _, count in completed_states)
    return {
        segments for segments, count in completed_states
        if count == minimum and segments
    }


def _prepare_link_rnds(ref_links):
    """Проверяет справочник связей и разбирает его направления."""
    _require_columns(ref_links, ['link', 'RNDs'], 'ref_links')
    if ref_links['link'].duplicated().any():
        raise ValueError('В ref_links есть повторяющиеся link')
    return {
        str(row.link): _parse_link_rnds(row.RNDs)
        for row in ref_links.itertuples()
    }


def _prepare_route_options(los, access, ref_links):
    """Строит допустимые реализации подготовленного LOS с участками и районами."""
    link_rnds = _prepare_link_rnds(ref_links)
    missing_links = set()
    stop_options = []

    for row in los.itertuples():
        variants = _parse_list(row.all_vars, 'all_vars')
        if variants and all(isinstance(item, str) for item in variants):
            variants = [variants]
        for variant_number, variant in enumerate(variants):
            if not isinstance(variant, list) or not variant:
                raise ValueError('all_vars содержит некорректный вариант пути')
            links = [str(link) for link in variant]
            candidate_sets = []
            incomplete = False
            for link in links:
                if link not in link_rnds:
                    missing_links.add(link)
                    incomplete = True
                    break
                candidate_sets.append(link_rnds[link])

            # Неполный путь сохраняет спрос, но не участвует в rnd-калибровке.
            if incomplete:
                stop_options.append({
                    'source': row.source,
                    'target': row.target,
                    'variant_number': variant_number,
                    'segments': tuple(),
                    'route_counts': {},
                    'attributed': False,
                    'los_speed': float(row.los_speed),
                    'los_carry': float(row.los_carry),
                })
                continue

            # Строим только непрерывные реализации без лишних пересадок.
            transport_segments = _minimal_route_segments(
                links, candidate_sets
            )
            if not transport_segments:
                continue
            for segments in transport_segments:
                stop_options.append({
                    'source': row.source,
                    'target': row.target,
                    'variant_number': variant_number,
                    'segments': segments,
                    'route_counts': dict(Counter(
                        segment[0] for segment in segments
                    )),
                    'attributed': True,
                    'los_speed': float(row.los_speed),
                    'los_carry': float(row.los_carry),
                })

    options = pd.DataFrame(stop_options)
    if options.empty:
        return pd.DataFrame(columns=[
            'district_source', 'district_target', 'source', 'target',
            'variant_number', 'segments', 'route_counts', 'attributed',
            'los_speed', 'los_carry',
        ]), missing_links, {}
    options = options.drop_duplicates(
        ['source', 'target', 'segments', 'attributed']
    ).reset_index(drop=True)

    # В маршрутном режиме пересечение буфера задает только принадлежность.
    origin_access = access.rename(columns={
        'tr_district': 'district_source',
        'id_stop': 'source',
    })[['district_source', 'source']]
    target_access = access.rename(columns={
        'tr_district': 'district_target',
        'id_stop': 'target',
    })[['district_target', 'target']]
    options = options.merge(origin_access, on='source').merge(
        target_access, on='target'
    )
    options = options.reset_index(drop=True)
    relevant_rnds = {
        rnd
        for counts in options.loc[options['attributed'], 'route_counts']
        for rnd in counts
    }
    return options, missing_links, _group_route_rnds(relevant_rnds)


def _group_route_rnds(rnds):
    """Однократно группирует направления по маршрутам."""
    route_rnds = {}
    for rnd in rnds:
        route_rnds.setdefault(_route_from_rnd(rnd), []).append(rnd)
    return route_rnds


def _prepare_route_observations(route_boardings):
    """Приводит маршрутные поездки к словарю выбранного периода расчета."""
    _require_columns(route_boardings, ['route', 'flow'], 'route_boardings')
    if set(route_boardings.columns) != {'route', 'flow'}:
        raise ValueError('route_boardings должен содержать только route и flow')
    route_boardings = route_boardings.copy()
    if route_boardings['route'].isna().any():
        raise ValueError('В route_boardings есть пустые route')
    route_boardings['route'] = route_boardings['route'].astype(str).str.strip()
    if route_boardings['route'].eq('').any():
        raise ValueError('В route_boardings есть пустые route')
    if route_boardings['route'].duplicated().any():
        raise ValueError('В route_boardings есть повторяющиеся route')

    values = pd.to_numeric(route_boardings['flow'], errors='coerce')
    invalid = route_boardings['flow'].notna() & values.isna()
    if invalid.any() or (values.dropna() < 0).any():
        raise ValueError(
            'Поездки по маршрутам должны быть неотрицательными числами'
        )
    observations = dict(zip(
        route_boardings.loc[values.notna(), 'route'],
        values.dropna().astype(float),
    ))
    if not observations:
        raise ValueError('route_boardings не содержит наблюдений выбранного периода')
    return observations


def _round_largest_remainder(allocated):
    """Округляет потоки до сотых, сохраняя итог районной OD за час"""
    allocated = allocated.copy()
    if allocated.empty:
        allocated['flow'] = pd.Series(index=allocated.index, dtype='float64')
        return allocated
    codes, _ = pd.factorize(allocated['_demand_id'], sort=False)
    raw_cents = allocated['raw_flow'].to_numpy(dtype='float64') * 100
    cents = np.floor(raw_cents).astype('int64')
    totals = allocated.groupby('_demand_id', sort=False)['raw_flow'].sum()
    remainders = (
        np.floor(totals.to_numpy() * 100 + 0.5).astype('int64')
        - np.bincount(codes, weights=cents).astype('int64')
    )
    # При равных остатках сохраняем исходный порядок, как Series.nlargest.
    order = np.lexsort((np.arange(len(codes)), -(raw_cents - cents), codes))
    sorted_codes = codes[order]
    starts = np.flatnonzero(np.r_[True, np.diff(sorted_codes) != 0])
    ranks = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    selected = order[ranks < remainders[sorted_codes]]
    cents[selected] += 1
    allocated['flow'] = cents / 100
    return allocated


def _round_series_to_total(values, total):
    """Округляет ряд до сотых с сохранением заданной общей суммы"""
    raw_cents = values * 100
    rounded_cents = np.floor(raw_cents).astype('int64')
    target_cents = int(np.floor(total * 100 + 0.5))
    remainder = target_cents - int(rounded_cents.sum())
    if remainder > 0:
        fractions = raw_cents - rounded_cents
        selected = fractions.nlargest(remainder).index
        rounded_cents.loc[selected] += 1
    return rounded_cents / 100


def _apply_demand_profile(
    demand,
    target_period_volume,
    demand_profile_mode,
    carry_totals,
):
    """Распределяет целевой объем по районным OD и часам."""
    demand = demand.copy()
    demand['profile_weight'] = demand['base_pt_demand']
    if demand_profile_mode == 'mobile_carry':
        carry_shares = carry_totals / carry_totals.sum()
        demand['profile_weight'] *= demand['period'].map(carry_shares)

    profile_total = demand['profile_weight'].sum()
    if target_period_volume > 0 and profile_total <= 0:
        if demand['base_pt_demand'].sum() <= 0:
            raise ValueError(
                'Невозможно распределить положительный целевой объем: '
                'спрос на ОТ после применения долей равен нулю'
            )
        raise ValueError('Спрос задан только для часов без предложения')
    if profile_total > 0:
        demand['pt_demand'] = (
            demand['profile_weight'] * target_period_volume / profile_total
        )
    else:
        demand['pt_demand'] = 0.0
    demand['pt_demand'] = _round_series_to_total(
        demand['pt_demand'], target_period_volume,
    )
    return demand


def _route_carry_values(carry, calibration_period):
    """Возвращает провозную способность по rnd за период калибровки."""
    if carry is None or 'rnd' not in carry:
        return {}
    if carry['rnd'].duplicated().any():
        raise ValueError('В carry есть повторяющиеся rnd')
    carry = carry.copy()
    carry['rnd'] = carry['rnd'].astype(str).str.strip()
    if calibration_period == 'day':
        if 'day' in carry:
            values = pd.to_numeric(carry['day'], errors='coerce')
        else:
            # Часовые значения уже проверены и преобразованы при подготовке carry.
            values = carry[HOUR_COLUMNS].sum(axis=1)
    else:
        _require_columns(carry, [calibration_period], 'carry')
        values = pd.to_numeric(carry[calibration_period], errors='coerce')
    invalid = values.isna() | (values < 0)
    if invalid.any():
        raise ValueError('Провозная способность должна быть числом от 0')
    return dict(zip(carry['rnd'], values.astype(float)))


def _prepare_route_weights(
    observations,
    route_rnds,
    carry,
    target_period_volume,
    transfer_factor,
    calibration_period,
):
    """Формирует веса rnd из маршрутных поездок и провозной способности."""
    carry_values = _route_carry_values(carry, calibration_period)
    route_weights = {
        route: float(observations[route])
        for route in route_rnds
        if route in observations
    }
    sources = {route: 'flow' for route in route_weights}
    expected = float(target_period_volume) * transfer_factor
    observed_total = sum(route_weights.values())
    residual = max(expected - observed_total, 0.0)
    warnings = []
    if observed_total > expected + 0.01:
        warnings.append(f'{calibration_period}: поездки превышают ожидаемый объем')

    # Для ненаблюдаемых маршрутов суммируем carry обоих направлений.
    missing = [route for route in route_rnds if route not in observations]
    available = {}
    for route in missing:
        route_carry = sum(carry_values.get(rnd, 0.0) for rnd in route_rnds[route])
        if route_carry > 0:
            available[route] = route_carry
    carry_total = sum(available.values())
    if residual > 0 and carry_total > 0:
        for route, value in available.items():
            route_weights[route] = residual * value / carry_total
            sources[route] = 'carry_estimate'
    elif residual > 0 and missing:
        warnings.append(
            f'{calibration_period}: нет провозной способности для части маршрутов '
            'без наблюдений'
        )

    # Внутри маршрута carry задает только относительный вес направлений.
    weights = {}
    for route, route_weight in route_weights.items():
        rnds = route_rnds[route]
        positive_carry = {
            rnd: carry_values.get(rnd, 0.0)
            for rnd in rnds
            if carry_values.get(rnd, 0.0) > 0
        }
        if positive_carry:
            # Для направления без значения используем среднее известных.
            mean_carry = sum(positive_carry.values()) / len(positive_carry)
            direction_basis = {
                rnd: positive_carry.get(rnd, mean_carry) for rnd in rnds
            }
        else:
            direction_basis = {rnd: 1.0 for rnd in rnds}
        basis_total = sum(direction_basis.values())
        for rnd, value in direction_basis.items():
            weights[rnd] = route_weight * value / basis_total
    return weights, sources, warnings


def _los_route_option_weights(options):
    """Делит вес LOS-пары поровну между переданными реализациями."""
    pair_keys = ['_demand_id', 'source', 'target']
    variant_count = options.groupby(pair_keys)['segments'].transform('size')
    return (
        options['los_speed'].astype('float64')
        * options['los_carry'].astype('float64') / variant_count
    )


def _route_count_patterns(counts):
    """Кодирует повторяющиеся наборы направлений без повторного разбора rnd."""
    return pd.factorize(counts.map(lambda value: tuple(value.items())), sort=False)


def _expand_route_allocation(
    active_demand,
    route_options,
    route_weights,
    observations,
):
    """Формирует положительный LOS-приор по остановочным маршрутным вариантам."""
    allocated = active_demand.merge(
        route_options,
        how='left',
        on=['district_source', 'district_target'],
        indicator=True,
    )
    available = allocated['_merge'] == 'both'
    allocated = allocated.loc[available].copy()
    # Явный ноль запрещает реализацию целиком, включая пути с пересадкой.
    # Исключаем ее до деления LOS-веса: оставшиеся реализации сохраняют
    # базовый вес пары. Отсутствующее наблюдение не является запретом.
    if allocated.empty:
        allocated['raw_flow'] = pd.Series(dtype='float64')
        return allocated

    codes, patterns = _route_count_patterns(allocated['route_counts'])
    zero_routes = {route for route, value in observations.items() if value == 0}
    blocked_patterns = np.array([
        bool(zero_routes.intersection(_route_counts(dict(pattern))))
        for pattern in patterns
    ])
    blocked = allocated['attributed'].to_numpy(dtype=bool) & blocked_patterns[codes]
    allocated = allocated.loc[~blocked].copy()
    codes = codes[~blocked]
    if allocated.empty:
        allocated['raw_flow'] = pd.Series(dtype='float64')
        return allocated

    # Число допустимых реализаций не увеличивает вес пары. После исключения
    # запрещенных вариантов делим LOS-основу до схлопывания неатрибутированных.
    allocated['los_option_weight'] = _los_route_option_weights(allocated)
    positive = [value for value in route_weights.values() if value > 0]
    weight_scale = float(np.mean(positive)) if positive else 1.0

    factors = np.ones(len(patterns), dtype='float64')
    weighted = np.zeros(len(patterns), dtype=bool)
    for index, pattern in enumerate(patterns):
        values = []
        for rnd, count in pattern:
            value = route_weights.get(rnd)
            if value is None or value <= 0:
                # Ненаблюдаемый и невзвешенный путь остается неатрибутированным.
                break
            # Относительные веса rnd безразмерны. Для неатрибутированного
            # варианта поправка равна 1, а не провозной способности в мест/ч.
            values.extend([value / weight_scale] * count)
        else:
            if values:
                factors[index] = float(np.exp(np.mean(np.log(values))))
                weighted[index] = True
    allocated['attributed'] = allocated['attributed'].to_numpy(dtype=bool) & weighted[codes]
    allocated['option_weight'] = allocated['los_option_weight'] * np.where(
        allocated['attributed'], factors[codes], 1.0,
    )
    allocated.loc[~allocated['attributed'], 'segments'] = allocated.loc[
        ~allocated['attributed'], 'segments'
    ].map(lambda _: tuple())
    allocated.loc[~allocated['attributed'], 'route_counts'] = allocated.loc[
        ~allocated['attributed'], 'route_counts'
    ].map(lambda _: {})
    allocated = allocated.loc[allocated['option_weight'] > 0].copy()

    # Схлопываем неатрибутированные реализации одной пары, сохраняя сумму
    # их долей LOS-веса, чтобы не переносить ее на известные маршруты.
    unattr = ~allocated['attributed']
    dedupe = ['_demand_id', 'source', 'target', 'attributed']
    unattributed = allocated.loc[unattr].copy()
    unattributed['option_weight'] = unattributed.groupby(dedupe)[
        'option_weight'
    ].transform('sum')
    allocated = pd.concat([
        allocated.loc[~unattr],
        unattributed.drop_duplicates(dedupe),
    ], ignore_index=True)
    totals = allocated.groupby('_demand_id')['option_weight'].transform('sum')
    allocated['raw_flow'] = (
        allocated['pt_demand'] * allocated['option_weight'] / totals
    )
    # Вспомогательные величины храним компактно после вычисления приора.
    # Потоки, нормирование и последующий оптимизатор сохраняют float64.
    auxiliary = ['los_speed', 'los_carry', 'los_option_weight', 'option_weight']
    allocated[auxiliary] = allocated[auxiliary].astype('float32')
    return allocated


def _build_mobile_route_baseline(demand, route_options):
    """Оценивает поездки по маршрутам из мобильного спроса и LOS до калибровки."""
    # Базовый расчет сохраняет долю неатрибутированных вариантов в общем LOS.
    pairs = ['district_source', 'district_target']
    base_demand = demand.loc[demand['base_pt_demand'] > 0].groupby(
        pairs, as_index=False,
    )['base_pt_demand'].sum()
    base_demand['_demand_id'] = np.arange(len(base_demand))
    baseline = base_demand.merge(
        route_options,
        how='inner',
        on=pairs,
    )
    if baseline.empty:
        return {}

    # Вес остановочной пары задается LOS, а все ее реализации делят этот вес
    # поровну, чтобы их число не увеличивало вес пары.
    baseline['option_weight'] = _los_route_option_weights(baseline)
    weight_total = baseline.groupby('_demand_id')['option_weight'].transform(
        'sum'
    )
    baseline['baseline_flow'] = (
        baseline['base_pt_demand'] * baseline['option_weight'] / weight_total
    )
    # Конечная поездка создает по одной поездке на каждом маршрутном участке.
    attributed = baseline.loc[baseline['attributed']]
    codes, patterns = _route_count_patterns(attributed['route_counts'])
    flows = np.bincount(codes, weights=attributed['baseline_flow'])
    boardings = Counter()
    for pattern, flow in zip(patterns, flows):
        for route, count in _route_counts(dict(pattern)).items():
            boardings[route] += float(flow) * count
    return dict(boardings)


def _optimize_route_allocation(allocated, observations):
    """Балансирует маршрутные варианты зеркальным спуском с OD-нормированием."""
    if allocated.empty:
        # При нулевых наблюдениях пустое решение является точным; положительные
        # поездки без назначаемого LOS остаются несогласованными.
        trivial_solution = not any(value > 0 for value in observations.values())
        return allocated, {
            'objective_value': 0.0 if trivial_solution else np.nan,
            'iterations': 0,
            'objective_relative_change': 0.0 if trivial_solution else np.nan,
            'max_variant_share_change': 0.0 if trivial_solution else np.nan,
            'stationarity_error': 0.0 if trivial_solution else np.nan,
            'converged': trivial_solution,
            'termination_reason': (
                'converged' if trivial_solution else 'no_assignable_los'
            ),
        }

    # Строки наблюдений включают также маршруты без назначаемого варианта LOS.
    key_index = {route: index for index, route in enumerate(observations)}
    observed_values = np.array(list(observations.values()), dtype='float64')
    rows = []
    cols = []
    data = []
    codes, patterns = _route_count_patterns(allocated['route_counts'])
    for column, pattern in enumerate(patterns):
        for route, count in _route_counts(dict(pattern)).items():
            if route in key_index:
                rows.append(key_index[route])
                cols.append(column)
                data.append(float(count))
    # Коэффициенты — целые кратности участков; float32 хранит их точно.
    # Умножение на потоки float64 и все вычисления решателя остаются float64.
    incidence = sparse.csr_matrix(
        (np.asarray(data, dtype='float32'), (rows, cols)),
        shape=(len(observations), len(patterns)),
    )[:, codes]

    q = allocated['raw_flow'].to_numpy(dtype='float64')
    x = q.copy()
    total = float(q.sum())
    group_codes, _ = pd.factorize(allocated['_demand_id'], sort=False)
    group_targets = np.bincount(group_codes, weights=q)
    denominators = np.maximum(observed_values, 1.0)

    def objective(values):
        """Вычисляет маршрутную ошибку и KL-отклонение от LOS-приора."""
        modeled = incidence @ values
        route_error = (
            float(np.mean(((modeled - observed_values) / denominators) ** 2))
            if len(observed_values) else 0.0
        )
        prior = float(np.sum(
            values * np.log(values / q) - values + q
        ) / total) if total else 0.0
        return route_error + ROUTE_PRIOR_WEIGHT * prior

    def scaled_gradient(values):
        """Вычисляет полный градиент в координатах долей общего объема."""
        modeled = incidence @ values
        if len(observed_values):
            residual = (modeled - observed_values) / denominators
            route_gradient = (
                2.0 / len(observed_values)
                * (incidence.T @ (residual / denominators))
            )
        else:
            route_gradient = np.zeros_like(values)
        # Умножение на total компенсирует масштаб обоих компонентов градиента.
        return (
            total * np.asarray(route_gradient).ravel()
            + ROUTE_PRIOR_WEIGHT * np.log(values / q)
        )

    def stationarity_error(gradient):
        """Проверяет равенство градиентов вариантов внутри каждой OD-группы."""
        if not np.isfinite(gradient).all():
            return np.inf
        group_min = np.full(len(group_targets), np.inf)
        group_max = np.full(len(group_targets), -np.inf)
        np.minimum.at(group_min, group_codes, gradient)
        np.maximum.at(group_max, group_codes, gradient)
        return float(np.max(group_max - group_min))

    current_objective = objective(x)
    gradient = scaled_gradient(x)
    current_stationarity = stationarity_error(gradient)
    # При равной цели сохраняем первое решение, а не последнюю итерацию.
    best_x = x.copy()
    best_objective = current_objective
    best_stationarity = current_stationarity
    step = 1.0
    streak = 0
    relative_change = np.nan
    share_change = np.nan
    converged = False
    iteration = 0
    termination_reason = 'max_iterations'

    for iteration in range(1, ROUTE_MAX_ITERATIONS + 1):
        if not np.isfinite(current_objective) or not np.isfinite(gradient).all():
            termination_reason = 'non_finite_objective_or_gradient'
            break

        # Обратный поиск принимает только невозрастающую целевую функцию.
        trial_step = step
        candidate = x
        candidate_objective = current_objective
        accepted = False
        for _ in range(30):
            candidate = x * np.exp(np.clip(-trial_step * gradient, -50, 50))
            sums = np.bincount(group_codes, weights=candidate)
            # Нефинитные и нулевые потоки не допускаются в KL-приор.
            if (
                np.isfinite(candidate).all() and (candidate > 0).all()
                and np.isfinite(sums).all() and (sums > 0).all()
            ):
                candidate *= group_targets[group_codes] / sums[group_codes]
                if np.isfinite(candidate).all() and (candidate > 0).all():
                    candidate_objective = objective(candidate)
                    if (
                        np.isfinite(candidate_objective)
                        and candidate_objective <= current_objective
                    ):
                        accepted = True
                        break
            trial_step *= 0.5
        if not accepted:
            if current_stationarity > ROUTE_STATIONARITY_TOLERANCE:
                termination_reason = 'line_search_stalled'
                break
            candidate = x.copy()
            candidate_objective = current_objective

        group_total_for_row = group_targets[group_codes]
        share_change = float(np.max(np.abs(
            candidate / group_total_for_row - x / group_total_for_row
        )))
        relative_change = abs(current_objective - candidate_objective) / max(
            abs(current_objective), 1e-12
        )
        x = candidate
        current_objective = candidate_objective
        # Этот градиент проверяет принятое решение и используется на следующем шаге.
        gradient = scaled_gradient(x)
        current_stationarity = stationarity_error(gradient)
        step = min(trial_step * 1.2, 10.0)
        if current_objective < best_objective:
            best_objective = current_objective
            best_x = x.copy()
            best_stationarity = current_stationarity
        if (
            relative_change <= ROUTE_CONVERGENCE_TOLERANCE
            and share_change <= ROUTE_CONVERGENCE_TOLERANCE
            and current_stationarity <= ROUTE_STATIONARITY_TOLERANCE
            and best_stationarity <= ROUTE_STATIONARITY_TOLERANCE
        ):
            streak += 1
            if streak >= ROUTE_CONVERGENCE_STREAK:
                converged = True
                termination_reason = 'converged'
                break
        else:
            streak = 0

    allocated = allocated.copy()
    allocated['raw_flow'] = best_x
    diagnostics = {
        'objective_value': round(float(best_objective), 6),
        'iterations': iteration,
        'objective_relative_change': round(float(relative_change), 8),
        'max_variant_share_change': round(float(share_change), 8),
        'stationarity_error': float(best_stationarity),
        'converged': converged,
        'termination_reason': termination_reason,
    }
    return allocated, diagnostics


def _route_stop_control(modeled_value, stop_flow_total):
    """Сверяет модельные поездки с округленными остановочными потоками."""
    modeled_rounded = round(float(modeled_value), 2)
    stop_flow_total = round(float(stop_flow_total), 2)
    balance = round(modeled_rounded - stop_flow_total, 2)
    status = 'OK' if abs(balance) <= 0.01 else 'ERROR'
    return modeled_rounded, stop_flow_total, balance, status


def _normalize_available_od(demand, allocated, unserved, target_volume):
    """Резервный алгоритм нормирования; временно не вызывается при расчете."""
    excluded_ids = set(unserved['_demand_id'])
    excluded_mask = demand['_demand_id'].isin(excluded_ids)
    requested = float(demand['pt_demand'].sum())
    excluded = float(demand.loc[excluded_mask, 'pt_demand'].sum())
    excluded_share = excluded / requested if requested else 0.0
    if excluded_share > 0.5 + 1e-12:
        raise ValueError(
            'Нормирование невозможно: доля недоступных OD-пар '
            f'{excluded_share:.1%} превышает 50%'
        )

    # Весь целевой объем пропорционально переносится на доступные OD-пары.
    available = requested - excluded
    if target_volume > 0 and available <= 0:
        raise ValueError('Нормирование невозможно: нет доступных OD-пар')
    factor = float(target_volume) / available if available else 1.0
    demand = demand.copy()
    available_mask = ~excluded_mask
    original_available = demand.loc[available_mask, 'pt_demand'].copy()
    normalized_available = _round_series_to_total(
        original_available * factor, target_volume
    )
    demand.loc[available_mask, 'pt_demand'] = normalized_available
    demand.loc[excluded_mask, 'pt_demand'] = 0.0

    # При суточной маршрутной калибровке варианты еще не развернуты по часам.
    # Иначе для каждой OD-пары тем же коэффициентом изменяем ее LOS-варианты.
    if allocated is not None:
        normalized_by_id = demand.loc[available_mask].set_index('_demand_id')[
            'pt_demand'
        ]
        original_by_id = original_available.copy()
        original_by_id.index = demand.loc[available_mask, '_demand_id'].to_numpy()
        factor_by_id = normalized_by_id / original_by_id
        allocated = allocated.copy()
        row_factors = allocated['_demand_id'].map(factor_by_id)
        allocated['pt_demand'] *= row_factors
        allocated['raw_flow'] *= row_factors
    return demand, allocated, {
        'normalization_factor': factor,
        'excluded_pt_demand': excluded,
        'excluded_od_periods': len(excluded_ids),
        'excluded_od_share': excluded_share,
    }


def _restore_route_hours(demand, allocated):
    """Восстанавливает почасовые потоки из общих суточных долей вариантов."""
    pairs = ['district_source', 'district_target']
    options = allocated[[
        *pairs, 'source', 'target', 'segments', 'attributed', 'raw_flow', 'pt_demand',
    ]].copy()
    options['variant_share'] = options['raw_flow'] / options['pt_demand']
    options = options.drop(columns=['raw_flow', 'pt_demand'])
    active = demand.loc[demand['pt_demand'] > 0]
    for period, hourly in active.groupby('period', sort=False):
        restored = hourly[['_demand_id', *pairs, 'pt_demand']].merge(
            options, on=pairs, how='inner',
        )
        if restored.empty:
            continue
        restored['period'] = period
        restored['raw_flow'] = restored['pt_demand'] * restored.pop('variant_share')
        yield restored


def _route_segment_flows(allocated):
    """Агрегирует участки до маршрутных остановочных OD за каждый час."""
    segment_rows = allocated.loc[
        allocated['attributed'].astype(bool), ['period', 'segments', 'raw_flow'],
    ]
    # Один и тот же путь относится к нескольким районным OD. Сначала
    # складываем их потоки, чтобы не разворачивать одинаковые участки снова.
    segment_rows = segment_rows.groupby(
        ['period', 'segments'], as_index=False, sort=False,
    )['raw_flow'].sum().explode('segments', ignore_index=True).dropna(subset=['segments'])
    keys = ['demand_period', 'rnd', 'source', 'target']
    if segment_rows.empty:
        return pd.DataFrame(columns=[*keys, 'raw_flow'])
    segment_values = pd.DataFrame(
        segment_rows.pop('segments').tolist(),
        columns=['rnd', 'source', 'target'], index=segment_rows.index,
    )
    segment_rows = pd.concat([segment_rows, segment_values], axis=1).rename(
        columns={'period': 'demand_period'},
    )
    return segment_rows.groupby(keys, as_index=False)['raw_flow'].sum()


def _build_route_outputs(
    allocated,
    observations,
    route_sources,
    baseline_boardings,
    route_rnds,
    missing_links,
    optimizer_stats,
    calibration_period,
    route_warnings,
    input_transfer_factor,
    route_full_coverage,
    period_segment_flows=None,
):
    """Формирует маршрутные OD, построчную калибровку и сводную диагностику."""
    attributed = allocated['attributed'].astype(bool)
    attributed_trips = float(allocated.loc[attributed, 'raw_flow'].sum())

    # Суточный расчет передает уже агрегированные почасовые участки.
    period_route_stops = (
        _route_segment_flows(allocated)
        if period_segment_flows is None else period_segment_flows.copy()
    )
    modeled_route = {}
    if not period_route_stops.empty:
        rnd_routes = {
            rnd: route for route, rnds in route_rnds.items() for rnd in rnds
        }
        period_route_stops['route'] = period_route_stops['rnd'].map(rnd_routes)
        modeled_route = period_route_stops.groupby('route')['raw_flow'].sum().to_dict()

    # Совместно округляем часовые потоки всех направлений маршрута. Поэтому
    # сумма h5-h23 сохраняет модельный маршрутный итог с точностью 0,01.
    stop_flow_totals = {}
    if not period_route_stops.empty:
        period_route_stops['rounded_flow'] = 0.0
        for route, index in period_route_stops.groupby('route').groups.items():
            period_route_stops.loc[index, 'rounded_flow'] = (
                _round_series_to_total(
                    period_route_stops.loc[index, 'raw_flow'],
                    modeled_route[route],
                )
            )
        stop_flow_totals = (
            period_route_stops.groupby('route')['rounded_flow']
            .sum().round(2).to_dict()
        )

        # Разворачиваем фактические часы спроса в формат основной OD-матрицы.
        route_stops = period_route_stops.pivot_table(
            index=['rnd', 'source', 'target'],
            columns='demand_period',
            values='rounded_flow',
            aggfunc='sum',
            fill_value=0,
        ).reset_index()
        route_stops.columns.name = None
        for hour in HOUR_COLUMNS:
            if hour not in route_stops:
                route_stops[hour] = 0.0
        route_stops[HOUR_COLUMNS] = route_stops[HOUR_COLUMNS].astype(float)
        route_stops['day'] = route_stops[HOUR_COLUMNS].sum(axis=1).round(2)
        route_stops = route_stops[
            ['rnd', 'source', 'target', *HOUR_COLUMNS, 'day']
        ]
        route_stops = route_stops.sort_values(
            ['rnd', 'source', 'target']
        ).reset_index(drop=True)
    else:
        route_stops = pd.DataFrame(
            columns=['rnd', 'source', 'target', *HOUR_COLUMNS, 'day']
        )

    calibration_rows = []
    relative_errors = []
    calibration_errors = []
    baseline_errors = []
    relevant_routes = set(route_rnds)
    all_observed_routes = set(observations)
    period_label = (
        calibration_period[1:]
        if calibration_period.startswith('h') else calibration_period
    )
    routes = sorted(relevant_routes | all_observed_routes)
    for route in routes:
        observed = observations.get(route)
        modeled_value = float(modeled_route.get(route, 0.0))
        baseline_value = float(baseline_boardings.get(route, 0.0))
        calibration_change = modeled_value - baseline_value
        calibration_change_ratio = (
            calibration_change / baseline_value
            if baseline_value > 0 else None
        )
        error = modeled_value - observed if observed is not None else None
        assignable_observation = (
            observed is not None and route in relevant_routes
        )
        if assignable_observation:
            calibration_errors.append(error)
            baseline_errors.append(baseline_value - observed)
        absolute_error = abs(error) if error is not None else None
        relative_error = (
            error / observed if observed is not None and observed > 0
            else None
        )
        if relative_error is not None and assignable_observation:
            relative_errors.append(abs(relative_error))
        if observed is None:
            status = 'NO_OBSERVATION'
        elif route not in relevant_routes:
            status = 'NO_ASSIGNABLE_LOS'
        elif observed == 0:
            status = 'OK' if modeled_value <= 0.01 else 'WARNING'
        else:
            status = (
                'OK' if abs(relative_error) <= 0.10 else 'WARNING'
            )
        source = route_sources.get(route, 'none')
        if observed is not None:
            source = 'flow'
        (
            modeled_rounded,
            stop_flow_total,
            stop_flow_balance,
            stop_flow_status,
        ) = _route_stop_control(
            modeled_value,
            stop_flow_totals.get(route, 0.0),
        )
        calibration_rows.append({
            'route': route,
            'period': period_label,
            'observed_boardings': observed,
            'baseline_boardings': round(baseline_value, 2),
            'modeled_boardings': modeled_rounded,
            'calibration_change': round(calibration_change, 2),
            'calibration_change_ratio': (
                round(calibration_change_ratio, 4)
                if calibration_change_ratio is not None else None
            ),
            'error': round(error, 2) if error is not None else None,
            'absolute_error': (
                round(absolute_error, 2)
                if absolute_error is not None else None
            ),
            'relative_error': (
                round(relative_error, 4)
                if relative_error is not None else None
            ),
            'weight_source': source,
            'stop_flow_total': stop_flow_total,
            'stop_flow_balance': stop_flow_balance,
            'stop_flow_status': stop_flow_status,
            'status': status,
        })
    route_calibration = pd.DataFrame(calibration_rows)
    baseline_rounded_total = round(
        float(route_calibration['baseline_boardings'].sum()), 2
    )
    modeled_rounded_total = round(
        float(route_calibration['modeled_boardings'].sum()), 2
    )

    served_trips = float(allocated.groupby('_demand_id')['raw_flow'].sum().sum())
    observed_total = float(sum(observations.values()))
    assignable_observed_total = float(sum(
        value
        for route, value in observations.items()
        if route in relevant_routes
    ))
    wmape = (
        sum(abs(error) for error in calibration_errors)
        / assignable_observed_total
        if assignable_observed_total > 0 else None
    )
    baseline_wmape = (
        sum(abs(error) for error in baseline_errors)
        / assignable_observed_total
        if assignable_observed_total > 0 else None
    )
    bias = (
        sum(calibration_errors) / assignable_observed_total
        if assignable_observed_total > 0 else None
    )
    carry_estimated = {
        route
        for route, source in route_sources.items()
        if source == 'carry_estimate'
    }
    unweighted = relevant_routes - all_observed_routes - carry_estimated
    stop_flow_error = (
        not route_calibration.empty
        and route_calibration['stop_flow_status'].eq('ERROR').any()
    )
    warning = bool(
        route_warnings
        or missing_links
        or unweighted
        or not route_full_coverage
        or not optimizer_stats['converged']
        or (relative_errors and max(relative_errors) > 0.10)
    )
    route_stats = {
        'calibration_scope': (
            'day' if calibration_period == 'day' else 'hourly'
        ),
        'los_rnd_count': sum(len(rnds) for rnds in route_rnds.values()),
        'los_route_count': len(relevant_routes),
        'observed_route_count': len(all_observed_routes),
        'carry_estimated_route_count': len(carry_estimated),
        'unweighted_route_count': len(unweighted),
        'attributed_trip_share': round(
            attributed_trips / served_trips if served_trips else 0.0, 4
        ),
        'observed_boardings_total': round(observed_total, 2),
        'assignable_observed_boardings_total': round(
            assignable_observed_total, 2
        ),
        'baseline_boardings_total': baseline_rounded_total,
        'modeled_boardings_total': modeled_rounded_total,
        'baseline_wmape': (
            round(baseline_wmape, 4)
            if baseline_wmape is not None else None
        ),
        'wmape': round(wmape, 4) if wmape is not None else None,
        'wmape_improvement': (
            round(baseline_wmape - wmape, 4)
            if baseline_wmape is not None and wmape is not None else None
        ),
        'bias': round(bias, 4) if bias is not None else None,
        'input_transfer_factor': input_transfer_factor,
        'modeled_transfer_factor': round(
            modeled_rounded_total / attributed_trips
            if attributed_trips else 0.0,
            4,
        ),
        'mean_absolute_relative_error': (
            round(float(np.mean(relative_errors)), 4)
            if relative_errors else None
        ),
        'max_absolute_relative_error': (
            round(float(max(relative_errors)), 4)
            if relative_errors else None
        ),
        'missing_ref_link_count': len(missing_links),
        'optimizer': 'mirror_descent',
        'prior_weight': ROUTE_PRIOR_WEIGHT,
        **optimizer_stats,
        'calibration_status': (
            'ERROR' if stop_flow_error else ('WARNING' if warning else 'OK')
        ),
    }
    return route_stops, route_calibration, pd.DataFrame([route_stats])


def build_stop_demand_matrix(
    demand,
    ref_districts,
    ref_stops,
    los,
    carry=None,
    pt_share=0.2,
    calculation_period='day',
    demand_profile_mode='mobile_only',
    calibration_mode='pt_share',
    total_boardings_period=None,
    transfer_factor=1.1,
    access_radius_km=0.7,
    min_overlap_share=0.15,
    normalize_target_volume=False,
    ref_links=None,
    route_boardings=None,
    demand_source='source',
    demand_target='target',
    demand_period='period',
    demand_value='demand',
    return_access=False,
    return_control=False,
    return_supply=False,
    return_route=False,
    pt_share_intrazonal=0.05,
    log=logger,
):
    """Восстанавливает остановочную OD-матрицу за сутки или выбранный час."""

    # Параметр сохранен для совместимости; перераспределение временно отключено.
    if normalize_target_volume:
        log.warning('normalize_target_volume temporarily disabled; request ignored')
    normalize_target_volume = False

    log.info(
        'build_stop_demand_matrix started; period=%s; calibration_mode=%s',
        calculation_period, calibration_mode,
    )
    if not 0.1 <= pt_share <= 0.6:
        raise ValueError('pt_share должен быть от 0.1 до 0.6')
    if not 0 <= pt_share_intrazonal <= 0.3:
        raise ValueError('pt_share_intrazonal должен быть от 0 до 0.3')
    if demand_profile_mode not in {'mobile_only', 'mobile_carry'}:
        raise ValueError('Неизвестный режим временного профиля')
    _validate_calibration_parameters(
        calibration_mode, total_boardings_period, transfer_factor,
    )
    if calculation_period != 'day':
        calculation_period = _hour_name(calculation_period)[1:]
        if demand_profile_mode != 'mobile_only':
            raise ValueError('В часовом расчете доступен только mobile_only')
    if demand_profile_mode == 'mobile_carry' and carry is None:
        raise ValueError(
            'Для выбранного профиля требуется файл провозной способности'
        )
    carry_totals = None
    if carry is not None:
        hourly_carry = _prepare_hourly_carry(
            carry, require_positive=demand_profile_mode == 'mobile_carry',
        )
        carry = carry.copy()
        carry[HOUR_COLUMNS] = hourly_carry
        carry_totals = hourly_carry.sum(axis=0)
    if calibration_mode == 'route_boardings':
        if ref_links is None or route_boardings is None:
            raise ValueError(
                'Для маршрутной калибровки нужны ref_links и route_boardings'
            )
    if not 0.3 <= access_radius_km <= 1.2:
        raise ValueError('access_radius_km должен быть от 0.3 до 1.2')
    if not 0.15 <= min_overlap_share <= 0.3:
        raise ValueError('min_overlap_share должен быть от 0.15 до 0.3')
    demand = _prepare_demand_inputs(
        demand, ref_districts, ref_stops, los, calculation_period,
        demand_source, demand_target, demand_period, demand_value,
    )

    # Применяем долю ОТ района отправления, а при ее отсутствии общую долю
    district_shares = pd.Series(dtype='float64')
    if 'pt_share_origin' in ref_districts:
        raw_shares = ref_districts.set_index('id_tr_distr')[
            'pt_share_origin'
        ]
        converted_shares = pd.to_numeric(raw_shares, errors='coerce')
        if (raw_shares.notna() & converted_shares.isna()).any():
            raise ValueError('pt_share_origin должен быть числом')
        district_shares = converted_shares.dropna()
        if not district_shares.between(0.1, 0.6).all():
            raise ValueError('pt_share_origin должен быть от 0.1 до 0.6')
    demand['origin_pt_share'] = (
        demand['district_source'].map(district_shares).fillna(pt_share)
    )
    # Единая внутрирайонная доля имеет приоритет над районной долей отправления.
    intrazonal = demand['district_source'] == demand['district_target']
    demand.loc[intrazonal, 'origin_pt_share'] = pt_share_intrazonal
    demand['base_pt_demand'] = (
        demand['mobile_demand'] * demand['origin_pt_share']
    )

    los = _prepare_los(los)

    log.info('prepare_inputs pass; demand_rows=%s; los_rows=%s', len(demand), len(los))

    # Рассчитываем доступность остановок и доли LOS выбранного среза
    access = _prepare_access(
        ref_districts,
        ref_stops,
        access_radius_km,
        min_overlap_share,
    )
    log.info('prepare_access pass; assignments=%s', len(access))

    # В маршрутном режиме восстанавливаем варианты rnd и покрытие маршрутов.
    route_options = pd.DataFrame()
    route_observations = {}
    route_sources = {}
    route_warnings = []
    missing_ref_links = set()
    route_rnds = {}
    calibration_period = (
        'day' if calculation_period == 'day' else f'h{calculation_period}'
    )
    route_full_coverage = False
    optimizer_observations = {}
    if calibration_mode == 'route_boardings':
        route_options, missing_ref_links, route_rnds = _prepare_route_options(
            los, access, ref_links
        )
        if not route_rnds and not los.empty:
            raise ValueError('LOS и ref_links не содержат допустимых rnd')
        route_observations = _prepare_route_observations(route_boardings)
        relevant_routes = set(route_rnds)
        # Полным считается только точное совпадение наборов. Дополнительные
        # наблюдаемые маршруты без допустимого LOS не должны увеличивать
        # автоматически определяемый целевой объем поездок.
        route_full_coverage = relevant_routes == set(route_observations)
        # Неатрибутируемые наблюдения сохраняются для диагностики, но не
        # включаются в оптимизацию и потому не меняют остановочную матрицу.
        optimizer_observations = {
            route: value for route, value in route_observations.items()
            if route in relevant_routes
        }
        log.info(
            'prepare_route_options pass; options=%s; routes=%s; full_coverage=%s',
            len(route_options), len(route_rnds), route_full_coverage,
        )

    # Определяем объем пассажирских поездок выбранного периода.
    mobile_total = float(demand['mobile_demand'].sum())
    base_period_volume = float(demand['base_pt_demand'].sum())
    target_period_volume = (
        float(total_boardings_period) / transfer_factor
        if (
            calibration_mode in {'total_boardings', 'route_boardings'}
            and total_boardings_period is not None
        )
        else base_period_volume
    )
    if calibration_mode == 'route_boardings' and route_full_coverage:
        target_period_volume = sum(route_observations.values()) / transfer_factor
    if mobile_total <= 0 and target_period_volume > 0:
        raise ValueError('Положительный объем задан при нулевом мобильном спросе')
    effective_pt_share = (
        target_period_volume / mobile_total if mobile_total else 0.0
    )
    if effective_pt_share > 1:
        raise ValueError('Расчетный объем превышает мобильный спрос периода')
    effective_status = (
        'OK' if 0.1 <= effective_pt_share <= 0.6 else 'WARNING'
    )
    demand = _apply_demand_profile(
        demand,
        target_period_volume,
        demand_profile_mode,
        carry_totals,
    )
    demand['_demand_id'] = np.arange(len(demand))
    active_demand = demand.loc[demand['pt_demand'] > 0]
    log.info(
        'apply_demand_profile pass; profile=%s; target_period_trip_volume=%s',
        demand_profile_mode, target_period_volume,
    )

    route_stops = pd.DataFrame()
    route_calibration = pd.DataFrame()
    route_stats = pd.DataFrame()
    baseline_boardings = {}
    normalization_stats = {
        'normalization_factor': 1.0,
        'excluded_pt_demand': 0.0,
        'excluded_od_periods': 0,
        'excluded_od_share': 0.0,
    }
    if calibration_mode == 'route_boardings':
        # Независимо оцениваем маршрутные поездки из мобильного спроса и LOS
        # до применения фактических потоков и нормирования доступных OD.
        baseline_boardings = _build_mobile_route_baseline(
            demand, route_options
        )
        log.info('build_mobile_route_baseline pass')
        route_weights, route_sources, route_warnings = _prepare_route_weights(
            route_observations,
            route_rnds,
            carry,
            float(demand['pt_demand'].sum()),
            transfer_factor,
            calibration_period,
        )
        log.info('prepare_route_weights pass')
        daily_routes = calculation_period == 'day'
        pairs = ['district_source', 'district_target']
        if daily_routes:
            # Суточные наблюдения и маршрутные веса одинаковы для всех часов.
            # Оптимальные доли вариантов одной OD поэтому также общие.
            route_demand = active_demand.groupby(pairs, as_index=False)[
                'pt_demand'
            ].sum()
            route_demand['_demand_id'] = np.arange(len(route_demand))
            route_demand['period'] = 'day'
        else:
            route_demand = active_demand
        allocated = _expand_route_allocation(
            route_demand,
            route_options,
            route_weights,
            route_observations,
        )
        del route_options
        log.info('expand_route_allocation pass; variants=%s', len(allocated))
        served_ids = set(allocated['_demand_id']) if len(allocated) else set()
        if daily_routes:
            served_pairs = route_demand.loc[
                route_demand['_demand_id'].isin(served_ids), pairs,
            ]
            served_ids = set(active_demand.merge(served_pairs, on=pairs)['_demand_id'])
            hours_per_pair = active_demand.groupby(pairs).size()
            hourly_variants = int(
                allocated.groupby(pairs).size().mul(hours_per_pair, fill_value=0).sum()
            )
            log.info(
                'compact_daily_routes pass; od_pairs=%s; variants=%s; hourly_variants=%s',
                len(route_demand), len(allocated), hourly_variants,
            )
        unserved = active_demand.loc[
            ~active_demand['_demand_id'].isin(served_ids),
            [
                '_demand_id', 'district_source', 'district_target', 'period',
                'pt_demand',
            ],
        ].copy()
        # Нормирование временно отключено. Код сохранен для восстановления.
        # if normalize_target_volume:
        #     if daily_routes:
        #         demand, _, normalization_stats = _normalize_available_od(
        #             demand, None, unserved, target_period_volume,
        #         )
        #         # Сначала округляем нормированные OD-часы по прежним правилам,
        #         # затем переносим их точные суточные суммы в компактную задачу.
        #         normalized = route_demand[pairs].merge(
        #             demand.groupby(pairs, as_index=False)['pt_demand'].sum(), on=pairs,
        #         )['pt_demand'].to_numpy()
        #         new_targets = normalized[allocated['_demand_id'].to_numpy()]
        #         allocated['raw_flow'] *= new_targets / allocated['pt_demand'].to_numpy()
        #         allocated['pt_demand'] = new_targets
        #         allocated = allocated.loc[allocated['pt_demand'] > 0].copy()
        #     else:
        #         demand, allocated, normalization_stats = _normalize_available_od(
        #             demand, allocated, unserved, target_period_volume
        #         )
        #     log.info(
        #         'normalize_available_od pass; normalization_factor=%s',
        #         normalization_stats['normalization_factor'],
        #     )
        log.info('optimize_route_allocation started')
        allocated, optimizer_stats = _optimize_route_allocation(
            allocated, optimizer_observations,
        )
        log.info(
            'optimize_route_allocation pass; iterations=%s; converged=%s; termination_reason=%s',
            optimizer_stats['iterations'], optimizer_stats['converged'],
            optimizer_stats['termination_reason'],
        )
        period_segments = None
        if daily_routes:
            hourly_parts = []
            segment_parts = []
            for hourly in _restore_route_hours(demand, allocated):
                segment_parts.append(_route_segment_flows(hourly))
                rounded = _round_largest_remainder(hourly)
                hourly_parts.append(rounded.groupby(
                    ['_demand_id', 'source', 'target', 'period'],
                    as_index=False, sort=False,
                )['flow'].sum())
            flow_columns = ['_demand_id', 'source', 'target', 'period', 'flow']
            hourly_allocated = (
                pd.concat(hourly_parts, ignore_index=True)
                if hourly_parts else pd.DataFrame(columns=flow_columns)
            )
            segment_columns = ['demand_period', 'rnd', 'source', 'target', 'raw_flow']
            period_segments = (
                pd.concat(segment_parts, ignore_index=True)
                if segment_parts else pd.DataFrame(columns=segment_columns)
            )
            log.info(
                'restore_route_hours pass; rows=%s; segment_rows=%s',
                len(hourly_allocated), len(period_segments),
            )
        route_stops, route_calibration, route_stats = _build_route_outputs(
            allocated,
            route_observations,
            route_sources,
            baseline_boardings,
            route_rnds,
            missing_ref_links,
            optimizer_stats,
            calibration_period,
            route_warnings,
            transfer_factor,
            route_full_coverage,
            period_segment_flows=period_segments,
        )
        log.info('build_route_outputs pass')
        allocated = (
            hourly_allocated if daily_routes else _round_largest_remainder(allocated)
        )
    else:
        zone_pairs = _prepare_zone_pairs(los, access)
        log.info('prepare_zone_pairs pass; pairs=%s', len(zone_pairs))
        allocated = active_demand.merge(
            zone_pairs,
            how='left',
            on=['district_source', 'district_target'],
            indicator=True,
        )
        unserved = allocated.loc[
            allocated['_merge'] == 'left_only',
            [
                '_demand_id', 'district_source', 'district_target', 'period',
                'pt_demand',
            ],
        ].drop_duplicates()
        allocated = allocated.loc[allocated['_merge'] == 'both'].copy()
        allocated['raw_flow'] = allocated['pt_demand'] * allocated['pair_share']
        # Нормирование временно отключено. Код сохранен для восстановления.
        # if normalize_target_volume:
        #     demand, allocated, normalization_stats = _normalize_available_od(
        #         demand, allocated, unserved, target_period_volume
        #     )
        #     log.info(
        #         'normalize_available_od pass; normalization_factor=%s',
        #         normalization_stats['normalization_factor'],
        #     )
        allocated = _round_largest_remainder(allocated)

    log.info(
        'allocate_demand pass; allocated_rows=%s; unserved_od_periods=%s',
        len(allocated), len(unserved),
    )
    unserved['pt_demand'] = unserved['pt_demand'].round(2)

    # Классифицируем основные причины отсутствия распределения
    access_districts = set(access['tr_district'])
    unserved['reason'] = np.select(
        [
            ~unserved['district_source'].isin(access_districts),
            ~unserved['district_target'].isin(access_districts),
        ],
        ['NO_ACCESS_STOPS', 'NO_EGRESS_STOPS'],
        default='NO_LOS_PAIR',
    )

    # Контролируем баланс каждой районной корреспонденции отдельно по часам
    distributed = allocated.groupby('_demand_id')['flow'].sum()
    unserved_ids = set(unserved['_demand_id'])
    control = demand[[
        '_demand_id', 'district_source', 'district_target', 'period',
        'pt_demand',
    ]].copy()
    control['requested_pt_demand'] = control['pt_demand'].round(2)
    control['distributed_pt_demand'] = (
        control['_demand_id'].map(distributed).fillna(0).round(2)
    )
    control['unserved_pt_demand'] = np.where(
        control['_demand_id'].isin(unserved_ids),
        control['requested_pt_demand'],
        0.0,
    )
    # Баланс проверяется внутри расчета, но не дублируется в выходном файле.
    row_balance = (
        control['requested_pt_demand']
        - control['distributed_pt_demand']
        - control['unserved_pt_demand']
    ).round(2)
    row_control_errors = int((row_balance.abs() > 0.01).sum())
    max_row_balance_error = (
        float(row_balance.abs().max()) if len(row_balance) else 0.0
    )
    control = control[[
        'district_source',
        'district_target',
        'period',
        'requested_pt_demand',
        'distributed_pt_demand',
        'unserved_pt_demand',
    ]]
    # Во внешней контрольной таблице час записывается без префикса h
    control['period'] = control['period'].str[1:].astype('int64')

    # Собираем строки stop-to-stop и разворачиваем часы в столбцы h5-h23
    if allocated.empty:
        matrix = pd.DataFrame(
            columns=['link', 'source', 'target', *HOUR_COLUMNS, 'day']
        )
    else:
        flows = allocated.groupby(
            ['source', 'target', 'period'], as_index=False
        )['flow'].sum()
        matrix = flows.pivot(
            index=['source', 'target'],
            columns='period',
            values='flow',
        ).fillna(0).reset_index()
        for hour in HOUR_COLUMNS:
            if hour not in matrix:
                matrix[hour] = 0
        # Идентификаторы остановок должны попадать в link без десятичной части.
        matrix[['source', 'target']] = matrix[['source', 'target']].apply(
            pd.to_numeric, errors='raise'
        ).astype('int64')
        matrix['link'] = (
            matrix['source'].astype(str) + '>' + matrix['target'].astype(str)
        )
        matrix[HOUR_COLUMNS] = (
            matrix[HOUR_COLUMNS].astype('float64').round(2)
        )
        matrix['day'] = matrix[HOUR_COLUMNS].sum(axis=1).round(2)
        matrix = matrix[
            ['link', 'source', 'target', *HOUR_COLUMNS, 'day']
        ].sort_values(['source', 'target']).reset_index(drop=True)
    log.info('build_matrix pass; rows=%s', len(matrix))

    # Независимо проверяем сумму часов в day и общий баланс периода.
    day_balance_error = (
        float((matrix[HOUR_COLUMNS].sum(axis=1) - matrix['day']).abs().max())
        if len(matrix) else 0.0
    )
    requested_total = float(control['requested_pt_demand'].sum())
    served_total = float(control['distributed_pt_demand'].sum())
    unserved_total = float(control['unserved_pt_demand'].sum())
    period_balance_error = max(
        abs(round(target_period_volume, 2) - requested_total),
        abs(requested_total - served_total - unserved_total),
    )
    day_control_status = 'OK' if day_balance_error <= 0.01 else 'ERROR'
    period_control_status = (
        'OK' if period_balance_error <= 0.01 else 'ERROR'
    )
    log.info(
        'check_demand_balance pass; row_control_errors=%s; '
        'day_control_status=%s; period_control_status=%s',
        row_control_errors, day_control_status, period_control_status,
    )

    # Формируем агрегированную диагностику спроса и предложения по часам
    supply = pd.DataFrame()
    if carry is not None:
        hourly_requested = control.groupby('period')[
            'requested_pt_demand'
        ].sum()
        supply = pd.DataFrame({
            'period': list(range(5, 24)),
            'requested_pt_demand': [
                hourly_requested.get(hour, 0.0) for hour in range(5, 24)
            ],
            'carry_total': carry_totals.to_numpy(),
        })
        supply['demand_supply_index'] = np.where(
            supply['carry_total'] > 0,
            supply['requested_pt_demand'] / supply['carry_total'],
            np.nan,
        )
        if calculation_period != 'day':
            supply = supply.loc[
                supply['period'] == int(calculation_period)
            ].reset_index(drop=True)
        log.info('build_supply pass; rows=%s', len(supply))

    # Формируем компактные контрольные показатели расчета
    aggregate_control_errors = int(day_control_status == 'ERROR') + int(
        period_control_status == 'ERROR'
    )
    # Маршрутные выходы формируются до округления конечной OD-матрицы.
    # Окончательный статус учитывает также все последующие жесткие балансы.
    if calibration_mode == 'route_boardings' and (
        row_control_errors or aggregate_control_errors
    ):
        route_stats.loc[:, 'calibration_status'] = 'ERROR'
    reported_boardings = total_boardings_period
    if calibration_mode == 'route_boardings' and route_full_coverage:
        reported_boardings = sum(route_observations.values())
    summary = {
        'calculation_period': calculation_period,
        'demand_profile_mode': demand_profile_mode,
        'calibration_mode': calibration_mode,
        'normalize_target_volume': bool(normalize_target_volume),
        'pt_share': pt_share,
        'pt_share_intrazonal': pt_share_intrazonal,
        'district_pt_share_count': int(len(district_shares)),
        'total_boardings_period': reported_boardings,
        'transfer_factor': (
            transfer_factor if calibration_mode != 'pt_share' else None
        ),
        'base_period_trip_volume': round(base_period_volume, 2),
        'target_period_trip_volume': round(target_period_volume, 2),
        'effective_pt_share': round(effective_pt_share, 4),
        'effective_pt_share_status': effective_status,
        'mobile_demand': float(demand['mobile_demand'].sum()),
        'requested_pt_demand': round(
            float(control['requested_pt_demand'].sum()), 2
        ),
        'served_pt_demand': round(
            float(control['distributed_pt_demand'].sum()), 2
        ),
        'unserved_pt_demand': round(
            float(control['unserved_pt_demand'].sum()), 2
        ),
        'unserved_od_periods': (
            0 if normalize_target_volume else int(len(unserved))
        ),
        'excluded_pt_demand': round(
            normalization_stats['excluded_pt_demand'], 2
        ),
        'excluded_od_periods': normalization_stats['excluded_od_periods'],
        'excluded_od_share': round(
            normalization_stats['excluded_od_share'], 4
        ),
        'normalization_factor': round(
            normalization_stats['normalization_factor'], 6
        ),
        'access_assignments': int(len(access)),
        'districts_with_stops': int(access['tr_district'].nunique()),
        'districts_without_stops': int(
            len(ref_districts) - access['tr_district'].nunique()
        ),
        'control_errors': (
            row_control_errors + aggregate_control_errors
        ),
        'max_balance_error': round(max(
            max_row_balance_error,
            day_balance_error,
            period_balance_error,
        ), 2),
        'day_balance_error': round(day_balance_error, 2),
        'day_control_status': day_control_status,
        'period_balance_error': round(period_balance_error, 2),
        'period_control_status': period_control_status,
        'los_speed_min': round(float(los['los_speed'].min()), 2),
        'los_speed_median': round(float(los['los_speed'].median()), 2),
        'los_speed_max': round(float(los['los_speed'].max()), 2),
        'los_carry_min': round(float(los['los_carry'].min()), 2),
        'los_carry_median': round(float(los['los_carry'].median()), 2),
        'los_carry_max': round(float(los['los_carry'].max()), 2),
    }
    if calibration_mode == 'route_boardings':
        route_summary = route_stats.iloc[0].to_dict()
        summary.update({
            'route_full_coverage': route_full_coverage,
            'route_calibration_status': route_summary[
                'calibration_status'
            ],
            'attributed_trip_share': route_summary['attributed_trip_share'],
            'modeled_transfer_factor': route_summary[
                'modeled_transfer_factor'
            ],
            'route_wmape': route_summary['wmape'],
            'route_baseline_wmape': route_summary['baseline_wmape'],
            'route_wmape_improvement': route_summary['wmape_improvement'],
            'route_bias': route_summary['bias'],
        })
    unserved_result = unserved.drop(columns='_demand_id').reset_index(
        drop=True
    )
    if len(unserved_result):
        unserved_result['period'] = (
            unserved_result['period'].str[1:].astype('int64')
        )
    results = [matrix, unserved_result, summary]
    if return_access:
        access_result = access.rename(columns={
            'tr_district': 'id_tr_distr',
            'access_weight': 'overlap_share',
        })
        access_result['access_radius_km'] = access_radius_km
        access_result['min_overlap_share'] = min_overlap_share
        results.append(access_result)
    if return_control:
        results.append(control)
    if return_supply:
        results.append(supply)
    if return_route:
        results.extend([route_stops, route_calibration, route_stats])
    log.info(
        'build_stop_demand_matrix pass; served_pt_demand=%s; '
        'unserved_pt_demand=%s; control_errors=%s',
        summary['served_pt_demand'], summary['unserved_pt_demand'],
        summary['control_errors'],
    )
    return tuple(results)


def recover_demand_files(
    demand_path,
    districts_path,
    stops_path,
    los_path,
    carry_path,
    output_path,
    unserved_path,
    access_output_path,
    control_output_path,
    supply_output_path=None,
    ref_links_path=None,
    route_boardings_path=None,
    route_stops_output_path=None,
    route_calibration_output_path=None,
    route_stats_output_path=None,
    pt_share=0.2,
    access_radius_km=0.7,
    baseline_output_path=None,
    baseline_diagnostic_path=None,
    baseline_stats_path=None,
    pt_share_intrazonal=0.05,
    log=logger,
    **kwargs,
):
    """Запускает восстановление из CSV и сохраняет итоговые таблицы.

    unserved_path задает диагностический файл итогового необслуженного спроса.
    Нормирование по целевому объему временно отключено.
    """

    # Базовая оценка сохраняется комплектом: матрица, диагностика и сводка.
    baseline_paths = [baseline_output_path, baseline_diagnostic_path, baseline_stats_path]
    if any(baseline_paths) and not all(baseline_paths):
        raise ValueError('Для базовой оценки нужны пути матрицы, диагностики и сводки')

    # Читаем общие исходные таблицы один раз для обоих расчетов.
    demand = pd.read_csv(demand_path, sep=';')
    districts = pd.read_csv(districts_path, sep=';')
    stops = pd.read_csv(stops_path, sep=';')
    los = pd.read_csv(los_path, sep=';')
    carry = (
        pd.read_csv(carry_path, sep=';')
        if carry_path and Path(carry_path).is_file()
        else None
    )
    ref_links = (
        pd.read_csv(ref_links_path, sep=';')
        if ref_links_path and Path(ref_links_path).is_file()
        else None
    )
    route_boardings = (
        pd.read_csv(route_boardings_path, sep=';')
        if route_boardings_path and Path(route_boardings_path).is_file()
        else None
    )
    log.info(
        'read_demand_files pass; demand_rows=%s; districts=%s; stops=%s; '
        'los_rows=%s; carry=%s; ref_links=%s; route_boardings=%s',
        len(demand), len(districts), len(stops), len(los),
        carry is not None, ref_links is not None, route_boardings is not None,
    )
    (
        matrix,
        unserved,
        summary,
        access,
        control,
        supply,
        route_stops,
        route_calibration,
        route_stats,
    ) = build_stop_demand_matrix(
        demand,
        districts,
        stops,
        los,
        carry,
        ref_links=ref_links,
        route_boardings=route_boardings,
        pt_share=pt_share,
        pt_share_intrazonal=pt_share_intrazonal,
        access_radius_km=access_radius_km,
        return_access=True,
        return_control=True,
        return_supply=True,
        return_route=True,
        log=log,
        **kwargs,
    )

    if baseline_output_path:
        # Локальный импорт исключает цикл с общими функциями core_demand.
        from core_demand_bl import build_stop_demand_matrix as build_baseline

        baseline_matrix, baseline_unserved, baseline_stats, baseline_control = build_baseline(
            demand, districts, stops, los,
            pt_share=pt_share,
            pt_share_intrazonal=pt_share_intrazonal,
            calculation_period=kwargs.get('calculation_period', 'day'),
            access_radius_km=access_radius_km,
            min_overlap_share=kwargs.get('min_overlap_share', 0.15),
            stop_district_access=access,
            return_control=True,
            log=log,
        )
        # Диагностика содержит все районные балансы и причины необслуживания.
        baseline_diagnostic = baseline_control.merge(
            baseline_unserved[['district_source', 'district_target', 'period', 'reason']],
            on=['district_source', 'district_target', 'period'], how='left',
        ).fillna({'reason': ''})
        for path, frame in [
            (baseline_output_path, baseline_matrix),
            (baseline_diagnostic_path, baseline_diagnostic),
        ]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, sep=';', index=False, encoding='utf-8', float_format='%.1f')
        Path(baseline_stats_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([baseline_stats]).to_csv(
            baseline_stats_path, sep=';', index=False, encoding='utf-8',
        )
        log.info('baseline save_results pass')

    # Записываем матрицу, диагностику недоступных OD и привязку к районам.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(unserved_path).parent.mkdir(parents=True, exist_ok=True)
    Path(access_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(control_output_path).parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(
        output_path,
        sep=';',
        index=False,
        encoding='utf-8',
        float_format='%.2f',
    )
    unserved.to_csv(
        unserved_path,
        sep=';',
        index=False,
        encoding='utf-8',
        float_format='%.2f',
    )
    access.to_csv(
        access_output_path, sep=';', index=False, encoding='utf-8'
    )
    control.to_csv(
        control_output_path,
        sep=';',
        index=False,
        encoding='utf-8',
        float_format='%.2f',
    )
    if supply_output_path and not supply.empty:
        Path(supply_output_path).parent.mkdir(parents=True, exist_ok=True)
        supply.to_csv(
            supply_output_path,
            sep=';',
            index=False,
            encoding='utf-8',
            float_format='%.2f',
        )
    route_mode = kwargs.get('calibration_mode') == 'route_boardings'
    if route_mode and route_stops_output_path:
        Path(route_stops_output_path).parent.mkdir(parents=True, exist_ok=True)
        route_stops.to_csv(
            route_stops_output_path,
            sep=';',
            index=False,
            encoding='utf-8',
            float_format='%.2f',
        )
    if route_mode and route_calibration_output_path:
        Path(route_calibration_output_path).parent.mkdir(
            parents=True, exist_ok=True
        )
        route_calibration.to_csv(
            route_calibration_output_path,
            sep=';',
            index=False,
            encoding='utf-8',
            float_format='%.4f',
        )
    if route_mode and route_stats_output_path:
        Path(route_stats_output_path).parent.mkdir(parents=True, exist_ok=True)
        route_stats.to_csv(
            route_stats_output_path,
            sep=';',
            index=False,
            encoding='utf-8',
        )
    log.info('save_demand_results pass')
    return summary


def calculate_demand(
    ver,
    pt_share=0.3,
    calculation_period='day',
    demand_profile_mode='mobile_only',
    calibration_mode='pt_share',
    total_boardings_period=None,
    transfer_factor=None,
    access_radius_km=0.7,
    min_overlap_share=0.15,
    normalize_target_volume=False,
    pt_share_intrazonal=0.3,
    **kwargs,
):
    """Рассчитывает спрос для demand-проекта текущего пользователя
    Исходные файлы читает из каталога input, результаты сохраняет в output
    Имена файлов формируются по принятому в проекте шаблону с версией
    """

    # Нормализуем версию и получаем стандартные каталоги demand-проекта
    ver = ver if str(ver).startswith('_') else f'_{ver}'
    project_name = ver[1:]
    username = session['username']
    path_src, path_out = path_gen(
        username, 'demand', project_name, False
    )

    _validate_calibration_parameters(
        calibration_mode, total_boardings_period, transfer_factor,
    )
    log = _DemandLogger(current_app.logger, {
        'prefix': f'{username} - demand project {project_name}',
    })
    # Сохраняем совместимость вызовов и единые имена файлов необслуженного спроса.
    if normalize_target_volume:
        log.warning('normalize_target_volume temporarily disabled; request ignored')
    normalize_target_volume = False
    log.info(
        'calculate_demand started; pt_share=%s; '
        'pt_share_intrazonal=%s; '
        'period=%s; profile=%s; calibration_mode=%s; '
        'total_boardings_period=%s; '
        'transfer_factor=%s; '
        'access_radius_km=%s; min_overlap_share=%s; '
        'normalize_target_volume=%s',
        pt_share,
        pt_share_intrazonal,
        calculation_period,
        demand_profile_mode,
        calibration_mode,
        total_boardings_period,
        transfer_factor,
        access_radius_km,
        min_overlap_share,
        normalize_target_volume,
    )

    try:
        # Запускаем расчет на загруженных файлах проекта.
        summary = recover_demand_files(
            demand_path=path_src + f'matrix_mobile{ver}.csv',
            districts_path=path_src + f'ref_tr_distr{ver}.csv',
            stops_path=path_src + f'ref_stops{ver}.csv',
            los_path=path_src + f'los{ver}.csv',
            carry_path=(
                path_src + f'carry{ver}.csv'
                if Path(path_src + f'carry{ver}.csv').is_file()
                else None
            ),
            ref_links_path=(
                path_src + f'ref_links{ver}.csv'
                if Path(path_src + f'ref_links{ver}.csv').is_file()
                else None
            ),
            route_boardings_path=(
                path_src + f'route_boardings{ver}.csv'
                if Path(path_src + f'route_boardings{ver}.csv').is_file()
                else None
            ),
            output_path=path_out + f'demand_matrix{ver}.csv',
            unserved_path=path_out + (
                f'excluded_demand{ver}.csv' if normalize_target_volume
                else f'unserved_demand{ver}.csv'
            ),
            access_output_path=(
                path_out + f'stop_district_access{ver}.csv'
            ),
            control_output_path=path_out + f'demand_control{ver}.csv',
            baseline_output_path=path_out + f'demand_matrix{ver}_bl.csv',
            baseline_diagnostic_path=path_out + f'baseline{ver}.csv',
            baseline_stats_path=path_out + f'demand_stats{ver}_bl.csv',
            supply_output_path=path_out + f'demand_supply{ver}.csv',
            route_stops_output_path=(
                path_out + f'route_stops_matrix{ver}.csv'
            ),
            route_calibration_output_path=(
                path_out + f'route_calibration{ver}.csv'
            ),
            route_stats_output_path=(
                path_out + f'route_calibration_stats{ver}.csv'
            ),
            pt_share=pt_share,
            pt_share_intrazonal=pt_share_intrazonal,
            calculation_period=calculation_period,
            demand_profile_mode=demand_profile_mode,
            calibration_mode=calibration_mode,
            total_boardings_period=(
                total_boardings_period
                if calibration_mode in {'total_boardings', 'route_boardings'}
                else None
            ),
            transfer_factor=transfer_factor,
            access_radius_km=access_radius_km,
            min_overlap_share=min_overlap_share,
            normalize_target_volume=normalize_target_volume,
            log=log,
            **kwargs,
        )

        # Сохраняем контрольные показатели отдельным CSV
        stats_path = path_out + f'demand_stats{ver}.csv'
        pd.DataFrame([summary]).to_csv(
            stats_path, sep=';', index=False, encoding='utf-8'
        )
        log.info('save_demand_stats pass')
        log.info(
            'calculate_demand pass; '
            'requested_pt_demand=%s; served_pt_demand=%s; '
            'unserved_pt_demand=%s',
            summary['requested_pt_demand'],
            summary['served_pt_demand'],
            summary['unserved_pt_demand'],
        )
        return summary
    except Exception:
        current_app.logger.exception(
            '%s - demand project %s calculate error',
            username,
            project_name,
        )
        raise
