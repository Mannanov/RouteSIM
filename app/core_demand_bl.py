"""Базовая OD-оценка: общие доли ОТ, веса los_carry, точность 0,1.

Исходные таблицы и общие параметры передает вызывающий основной модуль.
Загрузку и сохранение файлов выполняет основной модуль; маршрутная калибровка
в базовой оценке не применяется.
"""

import geopandas as gpd
import numpy as np
import pandas as pd

from core_demand import HOUR_COLUMNS, _hour_name, _prepare_demand_inputs, logger
from serv_funcs import build_stop_access


OD_COLUMNS = ['district_source', 'district_target']


def _round_tenths(values, total):
    """Возвращает целые десятые методом наибольших остатков."""
    # Переводим потоки в десятые и округляем каждую строку вниз.
    scaled = values * 10
    units = np.floor(scaled).astype('int64')

    # Остаток общего объема отдаем строкам с наибольшими дробными частями.
    remainder = int(round(float(total) * 10)) - int(units.sum())
    if remainder:
        selected = (scaled - units).nlargest(remainder).index
        units.loc[selected] += 1
    return units


def _prepare_inputs(
    demand, ref_districts, ref_stops, los,
    pt_share, calculation_period, access_radius_km, min_overlap_share,
    pt_share_intrazonal,
):
    """Проверяет таблицы и параметры, формирует округленный районный спрос."""
    # Проверяем общие параметры и приводим выбранный час к единому формату.
    for name, value, lower, upper in [
        ('pt_share', pt_share, 0.1, 0.6),
        ('pt_share_intrazonal', pt_share_intrazonal, 0, 0.3),
        ('access_radius_km', access_radius_km, 0.3, 1.2),
        ('min_overlap_share', min_overlap_share, 0.15, 0.3),
    ]:
        if not lower <= value <= upper:
            raise ValueError(f'{name} должен быть от {lower} до {upper}')
    if calculation_period != 'day':
        calculation_period = _hour_name(calculation_period)[1:]

    # Общая проверка структуры и ссылок также агрегирует мобильный спрос периода.
    prepared = _prepare_demand_inputs(
        demand, ref_districts, ref_stops, los, calculation_period,
    )
    for frame, columns in [
        (demand, ['source', 'target']), (los, ['source', 'target', 'all_vars']),
        (ref_districts, ['id_tr_distr']), (ref_stops, ['id_stop']),
    ]:
        for column in columns:
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq('').any():
                raise ValueError(f'{column} не должен содержать пустые значения')

    # Не допускаем бесконечный спрос; районные доли проверяем, но не применяем.
    mobile = pd.to_numeric(demand['demand'], errors='raise')
    if not np.isfinite(mobile).all() or not np.isfinite(prepared['mobile_demand']).all():
        raise ValueError('Спрос должен быть конечным числом')
    if 'pt_share_origin' in ref_districts:
        raw = ref_districts['pt_share_origin']
        shares = pd.to_numeric(raw, errors='raise')
        if (raw.notna() & shares.isna()).any() or not shares.dropna().between(0.1, 0.6).all():
            raise ValueError('pt_share_origin должен быть от 0.1 до 0.6')

    # Проверяем LOS, сохраняем первые записи пар и исключаем пешие и нулевые связи.
    los = los[['source', 'target', 'all_vars', 'los_speed', 'los_carry']].copy()
    for column in ['los_speed', 'los_carry']:
        values = pd.to_numeric(los[column], errors='raise')
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'{column} должен быть конечным неотрицательным числом')
        los[column] = values
    los = los.drop_duplicates(['source', 'target'])
    los = los.loc[
        los['all_vars'].astype(str).str.strip().ne('W')
        & los['los_carry'].gt(0) & los['source'].ne(los['target']),
        ['source', 'target', 'los_carry'],
    ]

    # Общие доли ОТ и сортировка обеспечивают объем и порядок равных остатков.
    prepared['period'] = prepared['period'].astype(str).str[1:].astype('int64')
    prepared = prepared.sort_values([*OD_COLUMNS, 'period']).reset_index(drop=True)
    prepared['_demand_id'] = prepared.index
    shares = np.where(
        prepared['district_source'] == prepared['district_target'],
        pt_share_intrazonal, pt_share,
    )
    raw_demand = prepared['mobile_demand'] * shares
    target_volume = float(raw_demand.sum())
    if not np.isfinite(target_volume):
        raise ValueError('Общий спрос должен быть конечным числом')
    prepared['requested_units'] = _round_tenths(raw_demand, target_volume)
    return prepared, los, calculation_period, int(round(target_volume * 10))


def _prepare_membership(ref_districts, ref_stops, access_radius_km, min_overlap_share):
    """Использует общий с forecast расчёт доступности остановок."""
    access = build_stop_access(ref_districts, ref_stops, access_radius_km, min_overlap_share)
    return access[['id_tr_distr', 'id_stop']].rename(columns={'id_tr_distr': 'tr_district'})


def _prepare_pairs(los, membership, active_od):
    """Формирует остановочные пары только для OD с положительным спросом."""
    # Привязываем LOS к районам и исключаем OD, не участвующие в расчете.
    pairs = los.merge(
        membership.rename(columns={'tr_district': 'district_source', 'id_stop': 'source'}),
        on='source',
    ).merge(
        membership.rename(columns={'tr_district': 'district_target', 'id_stop': 'target'}),
        on='target',
    ).merge(active_od, on=OD_COLUMNS)

    # Масштабирование внутри OD сохраняет пропорции и предотвращает переполнение суммы.
    pairs['pair_weight'] = pairs['los_carry'] / pairs.groupby(OD_COLUMNS)['los_carry'].transform('max')
    pairs['pair_share'] = pairs['pair_weight'] / pairs.groupby(OD_COLUMNS)['pair_weight'].transform('sum')
    return pairs[[*OD_COLUMNS, 'source', 'target', 'pair_share']]


def _round_stop_flows(allocated):
    """Округляет все OD-группы одновременно, возвращает положительные потоки."""
    # Считаем сразу в целых десятых; порядок остановок разрешает равные остатки.
    allocated = allocated.sort_values(['_demand_id', 'source', 'target'], ignore_index=True)
    scaled = allocated['pair_share'] * allocated['requested_units']
    allocated['flow_units'] = np.floor(scaled).astype('int64')
    allocated['_fraction'] = scaled - allocated['flow_units']

    # Групповое ранжирование заменяет цикл и сохраняет объем каждой OD и часа.
    groups = allocated.groupby('_demand_id', sort=False)
    rest = allocated['requested_units'] - groups['flow_units'].transform('sum')
    rank = groups['_fraction'].rank(method='first', ascending=False)
    allocated['flow_units'] += rank.le(rest).astype('int64')
    return allocated.loc[allocated['flow_units'] > 0].drop(columns='_fraction')


def _allocate_demand(prepared, pairs):
    """Распределяет спрос по часам, ограничивая размер промежуточного слияния."""
    columns = ['_demand_id', 'source', 'target', 'period', 'flow_units']
    parts = []
    # Одновременно держим развернутые остановочные пары только одного часа.
    active = prepared.loc[prepared['requested_units'] > 0]
    for hour, demand in active.groupby('period', sort=False):
        allocated = demand[['_demand_id', *OD_COLUMNS, 'requested_units']].merge(
            pairs, on=OD_COLUMNS,
        )
        allocated = _round_stop_flows(allocated)
        allocated['period'] = hour
        parts.append(allocated[columns])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def _build_matrix(allocated):
    """Собирает остановочную матрицу и возвращает ее объем в целых десятых."""
    columns = ['link', 'source', 'target', *HOUR_COLUMNS, 'day']
    if allocated.empty:
        return pd.DataFrame(columns=columns), 0

    # Суммируем положительные потоки и часы до перевода в число поездок.
    flows = allocated.groupby(['source', 'target', 'period'])['flow_units'].sum()
    hourly = flows.unstack('period', fill_value=0).reindex(columns=range(5, 24), fill_value=0)
    matrix_units = int(hourly.to_numpy().sum())
    hourly.columns = HOUR_COLUMNS
    hourly['day'] = hourly.sum(axis=1)
    matrix = (hourly / 10).reset_index()
    matrix['link'] = matrix['source'].astype(str) + '>' + matrix['target'].astype(str)
    matrix = matrix[columns]
    matrix.columns.name = None
    return matrix, matrix_units


def _build_results(prepared, allocated, membership, district_count, target_units,
                   calculation_period, pt_share, return_control, pt_share_intrazonal):
    """Формирует необслуженный спрос, балансы и итоговую сводку."""
    # Недоступные корреспонденции сохраняем без перераспределения их объема.
    distributed = allocated.groupby('_demand_id')['flow_units'].sum()
    prepared['distributed_units'] = prepared['_demand_id'].map(distributed).fillna(0).astype('int64')
    prepared['unserved_units'] = prepared['requested_units'].where(
        ~prepared['_demand_id'].isin(distributed.index), 0,
    )
    unserved = prepared.loc[prepared['unserved_units'] > 0, [
        *OD_COLUMNS, 'period', 'unserved_units',
    ]].rename(columns={'unserved_units': 'pt_demand'})
    unserved['pt_demand'] = unserved['pt_demand'] / 10
    districts_with_stops = set(membership['tr_district'])
    unserved['reason'] = np.select(
        [~unserved['district_source'].isin(districts_with_stops),
         ~unserved['district_target'].isin(districts_with_stops)],
        ['NO_ACCESS_STOPS', 'NO_EGRESS_STOPS'], default='NO_LOS_PAIR',
    )

    # Проверяем районные балансы, общий объем периода и итог остановочной матрицы.
    matrix, matrix_units = _build_matrix(allocated)
    unit_columns = ['requested_units', 'distributed_units', 'unserved_units']
    requested, served, lost = (int(prepared[column].sum()) for column in unit_columns)
    row_error = prepared['requested_units'] - prepared['distributed_units'] - prepared['unserved_units']
    if ((row_error != 0).any() or requested != target_units
            or requested != served + lost or matrix_units != served):
        raise ValueError('Нарушен баланс базовой матрицы спроса')

    # Возвращаем объемы и показатели доступности в формате основного модуля.
    summary = {
        'calculation_period': calculation_period,
        'demand_profile_mode': 'mobile_only', 'calibration_mode': 'pt_share',
        'normalize_target_volume': False, 'pt_share': pt_share,
        'pt_share_intrazonal': pt_share_intrazonal,
        'district_pt_share_count': 0,
        'mobile_demand': float(prepared['mobile_demand'].sum()),
        'base_period_trip_volume': requested / 10,
        'target_period_trip_volume': requested / 10,
        'requested_pt_demand': requested / 10,
        'served_pt_demand': served / 10, 'unserved_pt_demand': lost / 10,
        'unserved_od_periods': len(unserved), 'access_assignments': len(membership),
        'districts_with_stops': len(districts_with_stops),
        'districts_without_stops': district_count - len(districts_with_stops),
        'control_errors': 0, 'max_balance_error': 0.0,
        'day_balance_error': 0.0, 'day_control_status': 'OK',
        'period_balance_error': 0.0, 'period_control_status': 'OK',
    }
    results = [matrix, unserved.reset_index(drop=True), summary]
    if return_control:
        # Контрольную таблицу создаем только по запросу, переводя десятые в поездки.
        control = prepared[[*OD_COLUMNS, 'period', *unit_columns]].copy()
        control[unit_columns] = control[unit_columns] / 10
        control = control.rename(columns=dict(zip(unit_columns, [
            'requested_pt_demand', 'distributed_pt_demand', 'unserved_pt_demand',
        ])))
        results.append(control)
    return tuple(results)


def build_stop_demand_matrix(
    demand, ref_districts, ref_stops, los, *, pt_share=0.3,
    calculation_period='day', access_radius_km=0.7, min_overlap_share=0.15,
    return_control=False, stop_district_access=None,
    pt_share_intrazonal=0.3,
    log=logger,
):
    """Возвращает матрицу, необслуженный спрос, сводку и опционально контроль.

    Имена входных полей и выходных столбцов совпадают с core_demand.
    stop_district_access позволяет использовать готовую привязку основного расчета.
    Входные таблицы не изменяются.
    """
    log.info('baseline build_stop_demand_matrix started; period=%s', calculation_period)
    # Проверяем входные данные и определяем районный спрос.
    prepared, los, calculation_period, target_units = _prepare_inputs(
        demand, ref_districts, ref_stops, los,
        pt_share, calculation_period, access_radius_km, min_overlap_share,
        pt_share_intrazonal,
    )
    log.info('baseline prepare_inputs pass; demand_rows=%s', len(prepared))
    # Привязываем остановки и формируем допустимые пары с весами los_carry.
    if stop_district_access is None:
        membership = _prepare_membership(
            ref_districts, ref_stops, access_radius_km, min_overlap_share,
        )
    else:
        membership = stop_district_access[['id_tr_distr', 'id_stop']].rename(
            columns={'id_tr_distr': 'tr_district'},
        )
    log.info(
        'baseline prepare_membership pass; assignments=%s; reused=%s',
        len(membership), stop_district_access is not None,
    )
    active_od = prepared.loc[prepared['requested_units'] > 0, OD_COLUMNS].drop_duplicates()
    pairs = _prepare_pairs(los, membership, active_od)
    log.info('baseline prepare_pairs pass; pairs=%s', len(pairs))
    # Распределяем спрос по часам и собираем проверенные результаты.
    allocated = _allocate_demand(prepared, pairs)
    log.info('baseline allocate_demand pass; allocated_rows=%s', len(allocated))
    results = _build_results(
        prepared, allocated, membership, len(ref_districts), target_units,
        calculation_period, pt_share, return_control,
        pt_share_intrazonal,
    )
    summary = results[2]
    log.info(
        'baseline build_results pass; served_pt_demand=%s; '
        'unserved_pt_demand=%s; control_errors=%s',
        summary['served_pt_demand'], summary['unserved_pt_demand'],
        summary['control_errors'],
    )
    return results
