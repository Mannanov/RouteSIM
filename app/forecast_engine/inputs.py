"""Validate and aggregate forecast tables while preserving directed hourly units."""
import geopandas as gpd
import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, HOUR_COLUMNS, _validate_config
from .contracts import ForecastCalculationError, failure



def _require_columns(frame, columns, name):
    """Проверяет наличие обязательных столбцов и сообщает имя входной таблицы."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{name}: отсутствуют столбцы {missing}')


def _numeric(frame, columns, name, nonnegative=True):
    """Преобразует выбранные столбцы в числа и проверяет их пригодность.
    Пустые и бесконечные значения запрещены; отрицательные допускаются только явно.
    """
    values = frame[list(columns)].apply(pd.to_numeric, errors='coerce')
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f'{name}: числовые поля содержат пустые или бесконечные значения')
    if nonnegative and (values < 0).any().any():
        raise ValueError(f'{name}: отрицательные значения не допускаются')
    return values


def _require_nonempty_keys(frame, columns, name):
    """Проверяет, что таблица непуста, а ее ключи не содержат пропусков и пустых строк."""
    if frame.empty:
        raise ValueError(f'{name}: файл не должен быть пустым')
    for column in columns:
        values = frame[column]
        blank = values.isna() | values.astype(str).str.strip().eq('')
        if blank.any():
            raise ValueError(f'{name}: ключевой столбец {column} содержит пустые значения')


def _validate_district_geometry(districts):
    """Разбирает WKT районов в EPSG:4326.

    Возвращает GeoSeries только с непустыми валидными Polygon или MultiPolygon.
    """
    try:
        geometry = gpd.GeoSeries.from_wkt(districts['wkt'], crs='EPSG:4326')
    except Exception as error:
        raise ValueError('ref_tr_distr: некорректная WKT-геометрия') from error
    if geometry.isna().any() or geometry.is_empty.any() or (~geometry.is_valid).any():
        raise ValueError('ref_tr_distr: геометрия пуста или невалидна')
    if not geometry.geom_type.isin({'Polygon', 'MultiPolygon'}).all():
        raise ValueError('ref_tr_distr: допускаются только Polygon и MultiPolygon')
    return geometry


def _normalize_districts(districts, config):
    """Проверяет характеристики, геометрию и состояния старых и новых районов.

    Для старого района без forecast копирует base и сохраняет запись наследования.
    Возвращает нормализованные состояния и отдельную таблицу наследования.
    """
    name = 'ref_tr_distr'
    characteristics = [
        'population', 'employment', 'school', 'higher_education', 'hospital',
        'local_medical', 'shopping', 'public_services',
    ]
    if 'scenario' in districts:
        raise ValueError(
            'ref_tr_distr: столбец scenario больше не используется; '
            'один проект соответствует одному варианту прогноза'
        )
    # Проверяем ключи и постоянные свойства района до разделения на состояния.
    required = ['id_tr_distr', 'is_new', 'state', 'wkt', *characteristics]
    _require_columns(districts, required, name)
    result = districts.copy()
    _require_nonempty_keys(result, ['id_tr_distr', 'state'], name)
    if result.duplicated(['id_tr_distr', 'state']).any():
        raise ValueError(f'{name}: повторяется ключ района/состояния')
    if not set(result['state']).issubset({'base', 'forecast'}):
        raise ValueError('state должен принимать значения base или forecast')
    if not set(result['is_new']).issubset({0, 1, False, True}):
        raise ValueError('is_new должен принимать значения 0 или 1')
    if result.groupby('id_tr_distr')['is_new'].nunique().gt(1).any():
        raise ValueError('is_new должен быть одинаковым во всех строках района')
    if result.groupby('id_tr_distr')['wkt'].nunique().gt(1).any():
        raise ValueError('Геометрия должна быть одинаковой во всех строках района')
    result[characteristics] = _numeric(result, characteristics, name)
    _validate_district_geometry(result.drop_duplicates('id_tr_distr'))

    # Отделяем наблюдаемую базу от новых территорий: у новых районов состояния base нет.
    old_ids = set(result.loc[~result['is_new'].astype(bool), 'id_tr_distr'])
    new_ids = set(result.loc[result['is_new'].astype(bool), 'id_tr_distr'])
    if result.loc[
        result['id_tr_distr'].isin(new_ids) & result['state'].eq('base')
    ].shape[0]:
        raise ValueError('Новые районы не должны содержать состояние base')

    base = result.loc[result['state'] == 'base'].set_index('id_tr_distr')
    forecast = result.loc[result['state'] == 'forecast'].set_index('id_tr_distr')
    missing = old_ids - set(base.index)
    if missing:
        raise ValueError(f'Нет base-характеристик старых районов: {sorted(missing)}')
    missing = new_ids - set(forecast.index)
    if missing:
        raise ValueError(f'Нет forecast-характеристик новых районов: {sorted(missing)}')

    # Достраиваем отсутствующий forecast только у старых районов и фиксируем наследование.
    records = []
    inherited = []
    for district_id in sorted(old_ids | new_ids):
        if district_id in forecast.index:
            row = forecast.loc[district_id].copy()
        else:
            row = base.loc[district_id].copy()
            row['state'] = 'forecast'
            inherited.append({
                'id_tr_distr': district_id,
                'inheritance_source': 'STATE_BASE',
            })
        row['id_tr_distr'] = district_id
        records.append(row.to_dict())
    for district_id, row in base.iterrows():
        row = row.copy()
        row['id_tr_distr'] = district_id
        row['state'] = 'base'
        records.append(row.to_dict())
    normalized = pd.DataFrame(records)[[
        'id_tr_distr', 'is_new', 'state', 'wkt',
        *characteristics,
    ]]
    inheritance = pd.DataFrame(inherited, columns=[
        'id_tr_distr', 'inheritance_source',
    ])
    return normalized, inheritance


def _normalize_matrix_input(matrix):
    """Проверяет уникальность остановочных OD-пар и неотрицательность спроса.

    Сверяет day с суммой h5–h23 с допуском исходного округления 0,01 поездки.
    """
    matrix = matrix.copy()
    _require_columns(matrix, ['source', 'target', *HOUR_COLUMNS, 'day'],
                     'matrix_stops')
    _require_nonempty_keys(matrix, ['source', 'target'], 'matrix_stops')
    if matrix.duplicated(['source', 'target']).any():
        raise ValueError('matrix_stops: повторяются пары source/target')
    matrix[HOUR_COLUMNS + ['day']] = _numeric(
        matrix, HOUR_COLUMNS + ['day'], 'matrix_stops'
    )
    # Допуск учитывает округление источника; существенное расхождение не исправляем автоматически.
    if (matrix[HOUR_COLUMNS].sum(axis=1) - matrix['day']).abs().gt(0.01).any():
        raise ValueError('matrix_stops: day не совпадает с суммой h5-h23')
    return matrix


def _normalize_stop_districts_input(stop_districts):
    """Проверяет единственное назначение каждой остановки транспортному району."""
    stop_districts = stop_districts.copy()
    _require_columns(stop_districts, ['id_stop', 'id_tr_distr'], 'stop_districts')
    _require_nonempty_keys(
        stop_districts, ['id_stop', 'id_tr_distr'], 'stop_districts'
    )
    if stop_districts['id_stop'].duplicated().any():
        raise ValueError('Каждая остановка должна принадлежать ровно одному району')
    return stop_districts


def _normalize_los_input(los):
    """Согласует поля los/los_by_carry и отмечает пригодность направленных времен.

    Непригодное время сохраняется для диагностики и резервной оценки; явный запрет
    помечается отдельно и не может использоваться как наблюдаемый LOS.
    """
    los = los.copy()
    if 'los_by_carry' not in los and 'los' in los:
        los = los.rename(columns={'los': 'los_by_carry'})
    elif 'los_by_carry' in los and 'los' in los:
        left = pd.to_numeric(los['los_by_carry'], errors='coerce')
        right = pd.to_numeric(los['los'], errors='coerce')
        if not np.allclose(left, right, equal_nan=True):
            raise ValueError('los и los_by_carry содержат разные значения')
    _require_columns(los, ['agg_source', 'agg_target', 'los_by_carry'],
                     'los_districts')
    _require_nonempty_keys(
        los, ['agg_source', 'agg_target'], 'los_districts'
    )
    if los.duplicated(['agg_source', 'agg_target']).any():
        raise ValueError('Районный LOS содержит дублирующиеся пары')
    los['los_by_carry'] = pd.to_numeric(los['los_by_carry'], errors='coerce')
    # Отсутствие времени требует резервной оценки, а is_forbidden исключает саму связь.
    los['los_valid'] = np.isfinite(los['los_by_carry']) & (los['los_by_carry'] > 0)
    if 'is_forbidden' in los:
        if not set(los['is_forbidden']).issubset({0, 1, False, True}):
            raise ValueError('los_districts.is_forbidden должен принимать 0 или 1')
        los['is_forbidden'] = los['is_forbidden'].astype(bool)
    else:
        los['is_forbidden'] = False
    los.loc[los['is_forbidden'], 'los_valid'] = False
    return los


def _identifier_key(value):
    """Приводит числовые ID к тексту, сохраняя значащие нули текстовых ID."""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value)


def _coverage_report(matrix, aggregation, config):
    """Измеряет потери назначения по часам и суткам и проверяет заданный порог."""
    records = []
    for hour in HOUR_COLUMNS + ['day']:
        source = float(matrix[hour].sum())
        unmapped = float(aggregation['excluded'][hour].sum())
        share = unmapped / source if source else 0.0
        # Полная потеря положительного спроса недопустима даже при пороге 100%.
        absolute_limit = config['unmapped_hour_loss_tolerance'] if hour != 'day' else 0.0
        complete_loss = source > 0 and share >= 1
        failed = (share > config['unmapped_max_trip_share'] and unmapped > absolute_limit) or complete_loss
        records.append({'stage': 'STOP_ASSIGNMENT', 'hour': hour, 'source_total': source,
                        'unmapped_total': unmapped, 'unmapped_share': share,
                        'max_trip_share': config['unmapped_max_trip_share'],
                        'absolute_loss_tolerance': absolute_limit,
                        'complete_loss': complete_loss,
                        'status': 'UNMAPPED_DEMAND_EXCEEDED' if failed else ('WARNING' if unmapped else 'OK'),
                        'reason': 'Спрос с неназначенным концом поездки исключен из районной матрицы' if unmapped else ''})
    report = pd.DataFrame(records)
    if report.status.eq('UNMAPPED_DEMAND_EXCEEDED').any():
        raise ForecastCalculationError('Превышен порог неотнесенного спроса', report)
    return report


def aggregate_methodology_matrix(matrix, stop_districts):
    """Агрегирует остановочные потоки по транспортным районам.

    Возвращает межрайонные, внутрирайонные, исключенные неназначенные потоки
    и контроль суточного баланса. Исключенная поездка учитывается один раз.
    """
    # Числовой 101 и текстовый «101» обозначают одну остановку; «00101» сохраняет нули.
    keys = stop_districts.id_stop.map(_identifier_key)
    if keys.duplicated().any():
        raise ValueError('stop_districts: повторяется нормализованный идентификатор остановки')
    mapping = pd.Series(stop_districts.id_tr_distr.to_numpy(), index=keys)
    data = matrix.copy()
    if pd.DataFrame({'source': data.source.map(_identifier_key),
                     'target': data.target.map(_identifier_key)}).duplicated().any():
        raise ValueError('matrix_stops: повторяется нормализованная пара остановок')
    data['district_source'] = data['source'].map(_identifier_key).map(mapping)
    data['district_target'] = data['target'].map(_identifier_key).map(mapping)
    # Если не назначен хотя бы один конец поездки, исключаем всю строку один раз.
    excluded = data.loc[data['district_source'].isna() | data['district_target'].isna()].copy()
    mapped = data.dropna(subset=['district_source', 'district_target']).copy()
    group = ['district_source', 'district_target']
    aggregated = mapped.groupby(group, as_index=False)[HOUR_COLUMNS + ['day']].sum()
    # Внутрирайонные перемещения отделяем от межрайонной обучающей матрицы.
    diagonal = aggregated['district_source'] == aggregated['district_target']
    intrazonal = aggregated.loc[diagonal].reset_index(drop=True)
    interzonal = aggregated.loc[~diagonal].reset_index(drop=True)
    # Проверяем разложение исходного объема на три непересекающиеся части.
    observed = float(matrix['day'].sum())
    values = {
        'interzonal_demand': float(interzonal['day'].sum()),
        'intrazonal_demand': float(intrazonal['day'].sum()),
        'excluded_demand': float(excluded['day'].sum()),
    }
    balance = observed - sum(values.values())
    controls = [{
        'control_type': 'AGGREGATION', 'stop_demand': observed, **values,
        'balance': balance, 'status': 'OK' if abs(balance) <= 0.01 else 'ERROR',
    }]
    return {'interzonal': interzonal, 'intrazonal': intrazonal,
            'excluded': excluded, 'control': pd.DataFrame(controls)}


def _state(districts, state):
    """Выбирает base или forecast и использует идентификатор района как индекс."""
    selected = districts.loc[districts['state'] == state].copy()
    return selected.set_index('id_tr_distr')


def _matrix_arrays(frame, ids, values):
    """Суммирует несколько числовых столбцов таблицы в плотные матрицы N×N.
    Порядок осей задается ids; неизвестные районы и диагональные пары исключаются.
    Отсутствующие пары остаются нулевыми, поэтому покрытие проверяется на входе.
    """
    values = list(values)
    size = len(ids)
    result = {
        value: np.zeros((size, size), dtype=float) for value in values
    }
    if frame.empty or not size:
        return result

    # Переводим идентификаторы в позиции единого порядка районов и исключаем диагональ.
    id_index = pd.Index(ids)
    source_index = id_index.get_indexer(frame['source'])
    target_index = id_index.get_indexer(frame['target'])
    valid = (
        (source_index >= 0)
        & (target_index >= 0)
        & (source_index != target_index)
    )
    if not valid.any():
        return result

    # Кодируем пару одной позицией: bincount суммирует повторные вклады без цикла по строкам.
    flat_index = source_index[valid] * size + target_index[valid]
    data = frame.loc[valid, values].to_numpy(dtype=float)
    for column_index, value in enumerate(values):
        result[value] = np.bincount(
            flat_index,
            weights=data[:, column_index],
            minlength=size * size,
        ).reshape(size, size)
    return result


def _matrix_array(frame, ids, value='day'):
    """Возвращает одну плотную OD-матрицу выбранного показателя в порядке ids."""
    return _matrix_arrays(frame, ids, [value])[value]


def _observed_arrays(day_matrix, ids):
    """Преобразует районную таблицу наблюдений в суточную матрицу и словарь часов."""
    renamed = day_matrix.rename(columns={'district_source': 'source',
                                         'district_target': 'target'})
    arrays = _matrix_arrays(renamed, ids, ['day', *HOUR_COLUMNS])
    return arrays['day'], {hour: arrays[hour] for hour in HOUR_COLUMNS}


def validate_forecast_upload(frame, file_prefix):
    """Проверяет структуру одного загружаемого файла по его типу.
    Согласованность идентификаторов между файлами проверяется на следующем этапе.
    """
    validators = {'matrix_stops': _normalize_matrix_input,
                  'stop_districts': _normalize_stop_districts_input,
                  'los_districts': _normalize_los_input,
                  'ref_tr_distr': lambda f: _normalize_districts(f, {}),
                  'constraints': _normalize_constraints}
    if file_prefix not in validators:
        raise ValueError(f'Неизвестный или устаревший тип файла прогноза: {file_prefix}')
    validators[file_prefix](frame)


def _normalize_constraints(frame):
    """Нормализует направленные запреты с необязательной областью действия по часу.
    Пустой hour означает все часы; повторяющиеся ключи пары/часа запрещены.
    Если таблица не передана, возвращает пустую таблицу с установленными столбцами.
    """
    if frame is None:
        return pd.DataFrame(columns=['source', 'target', 'hour', 'is_forbidden'])
    _require_columns(frame, ['source', 'target', 'is_forbidden'], 'constraints')
    result = frame.copy()
    if result[['source', 'target']].isna().any().any():
        raise ValueError('constraints: отсутствует идентификатор района')
    if not set(result['is_forbidden']).issubset({0, 1, False, True}):
        raise ValueError('constraints.is_forbidden должен быть 0 или 1')
    result['is_forbidden'] = result['is_forbidden'].astype(bool)
    if 'hour' not in result:
        result['hour'] = None
    result['hour'] = result['hour'].astype(object).where(result['hour'].notna(), None)
    for hour in result['hour']:
        if hour is not None and hour not in HOUR_COLUMNS:
            raise ValueError('constraints.hour должен быть h5-h23 или пустым')
    if result.duplicated(['source', 'target', 'hour']).any():
        raise ValueError('constraints: повторяется ключ пары/часа')
    return result[['source', 'target', 'hour', 'is_forbidden']]


def normalize_methodology_inputs(matrix, stop_districts, los, districts,
                                 config=None, constraints=None):
    """Проверяет четыре обязательные таблицы, ограничения и конфигурацию вместе.
    Проверяет требования покрытия матрицы и сопоставимости нарезки районов;
    согласует идентификаторы и переносит общие запреты в LOS. Возвращает набор входов.
    """
    config = _validate_config(config)
    # Только подтвержденное полное покрытие позволяет трактовать отсутствующие пары как нулевые.
    if config['matrix_coverage'] != 'full_sparse':
        raise failure('UNCONFIRMED_MATRIX_COVERAGE', 'INPUT_VALIDATION',
                      'Требуется матрица с полным покрытием: отсутствующие пары должны означать нулевой поток')
    if not config['zoning_comparable']:
        raise failure('UNCONFIRMED_ZONING', 'INPUT_VALIDATION', 'Требуются сопоставимые правила нарезки старых и новых районов')
    matrix = _normalize_matrix_input(matrix)
    stops = _normalize_stop_districts_input(stop_districts)
    los = _normalize_los_input(los)
    districts, inheritance = _normalize_districts(districts, config)
    constraints = _normalize_constraints(constraints)
    # Проверяем согласованность районных идентификаторов между независимыми входными файлами.
    ids = set(districts.id_tr_distr)
    if not set(stops.id_tr_distr).issubset(ids):
        raise ValueError('Назначение остановок содержит неизвестные районы')
    for frame, source, target in [(los, 'agg_source', 'agg_target'),
                                   (constraints, 'source', 'target')]:
        if not (set(frame[source]) | set(frame[target])).issubset(ids):
            raise ValueError('LOS или ограничения содержат неизвестные районы')
    # Общие запреты добавляем в LOS, чтобы ни наблюдения, ни доноры не восстановили запрещенную связь.
    global_forbidden = constraints.loc[constraints.is_forbidden & constraints.hour.isna()]
    for row in global_forbidden.itertuples():
        mask = los.agg_source.eq(row.source) & los.agg_target.eq(row.target)
        if mask.any():
            los.loc[mask, ['is_forbidden', 'los_valid']] = [True, False]
        else:
            los = pd.concat([los, pd.DataFrame([{'agg_source': row.source, 'agg_target': row.target,
                         'los_by_carry': np.nan, 'los_valid': False, 'is_forbidden': True}])], ignore_index=True)
    count = districts.loc[~districts.is_new.astype(bool), 'id_tr_distr'].nunique()
    config['analog_count'] = min(config['analog_count'], count) if count else config['analog_count']
    aggregation = aggregate_methodology_matrix(matrix, stops)
    # Проверяем сохранение спроса сразу после агрегации, до диагностики
    # покрытия, обучения и внешней проверки. Записи переиспользуются в отчете.
    aggregation_controls = []
    for hour in HOUR_COLUMNS + ['day']:
        total = float(matrix[hour].sum())
        inter = float(aggregation['interzonal'][hour].sum())
        intra = float(aggregation['intrazonal'][hour].sum())
        excluded = float(aggregation['excluded'][hour].sum())
        residual = total - inter - intra - excluded
        aggregation_controls.append({'stage': 'AGGREGATION', 'hour': hour, 'source_total': total,
                                     'interzonal_total': inter, 'intrazonal_total': intra,
                                     'unmapped_total': excluded, 'max_residual': abs(residual),
                                     'status': 'OK' if abs(residual) <= config['ipf_tolerance'] else 'ERROR'})
    failed_controls = [r for r in aggregation_controls if r['status'] != 'OK']
    if failed_controls:
        raise ForecastCalculationError('AGGREGATION_CONTROL_FAILED', failed_controls)
    coverage = _coverage_report(matrix, aggregation, config)
    # Эти свойства источника невозможно установить только по структуре CSV.
    # Отделяем принятые предпосылки от результатов автоматических проверок.
    requirements = [
        {'stage': 'INPUT_REQUIREMENTS', 'status': 'ASSUMPTION',
         'source': config['input_requirements_source'], 'parameter': 'matrix_coverage',
         'value': config['matrix_coverage'],
         'reason': 'Принято полное покрытие часов и территории; отсутствующие пары означают нулевой поток. Проверка CSV не подтверждает полноту наблюдений.'},
        {'stage': 'INPUT_REQUIREMENTS', 'status': 'ASSUMPTION',
         'source': config['input_requirements_source'], 'parameter': 'zoning_comparable',
         'value': config['zoning_comparable'],
         'reason': 'Приняты сопоставимые правила нарезки районов. Размер, периферийность и сочетания характеристик проверяются отдельно; структура CSV не устанавливает сопоставимость правил.'},
    ]
    return {'matrix': matrix, 'stop_districts': stops, 'los': los, 'districts': districts,
            'constraints': constraints, 'inheritance': inheritance, 'config': config,
            'aggregation': aggregation, 'aggregation_controls': aggregation_controls, 'coverage': coverage,
            'input_requirements': requirements}


def _observed_marginals(interzonal, district_ids):
    """Суммирует наблюдения в почасовые отправления и прибытия выбранных районов.
    Районы без строк получают нулевые итоги и сохраняют заданный порядок индекса.
    """
    outgoing = interzonal.groupby('district_source')[HOUR_COLUMNS].sum().reindex(district_ids, fill_value=0.0)
    incoming = interzonal.groupby('district_target')[HOUR_COLUMNS].sum().reindex(district_ids, fill_value=0.0)
    return outgoing, incoming


def _subset_matrix(frame, ids):
    """Оставляет связи, у которых оба конца входят в обучающий набор районов.
    Это исключает утечку проверочных потоков через противоположное направление.
    """
    return frame.loc[frame.district_source.isin(ids) & frame.district_target.isin(ids)]
