"""Почасовая генерация, гравитационное распределение и проверка по районам.

Расчетное ядро не зависит от Flask и хранения проектов. Идентификаторы районов
и остановок сохраняются строками. Правила расчета описаны в Forecast_baseline_2.md.
"""

import logging
import math
import warnings
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import logsumexp
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from forecast_engine.config import FEATURE_COLUMNS, HOUR_COLUMNS
from forecast_engine.balancing import prepare_support
from forecast_engine.contracts import ForecastCalculationError
from forecast_engine.target_reconciliation import reconcile_targets
from forecast_engine.reconciliation_diagnostics import reconciliation_delta_tables
from forecast_engine.correction_diagnostics import correction_tables
from forecast_engine.spatial_validation import validation_splits, nearest_analogue_baseline, baseline_matrix
from forecast_engine.inputs import (
    _identifier_key, _normalize_districts,
    _numeric, _require_columns, _require_nonempty_keys, _validate_district_geometry,
    aggregate_methodology_matrix,
)
from forecast_engine.los import _district_geodata
from forecast_engine.spatial_aggregation import aggregate_spatial_matrix
from serv_funcs import stop_geodata, validate_access_parameters


logger = logging.getLogger(__name__)


LOS_MIN_SPEED_KMH = 4.5
GENERATION_ALPHA_GRID = (.01, .1, 1.0, 10.0, 100.0, 1000.0)


# Допуски калибровки задаются отдельно от строгой балансировки модельных матриц.
DEFAULT_CONFIG = {
    'hours': HOUR_COLUMNS, 'recalculation_mode': 'recalculate_all',
    'new_connection_mode': 'all',
    'include_intrazonal': True, 'intrazonal_speed_kmh': 5.0, 'los_neighbor_radius_km': 1.5,
    'los_fallback_percentile': 75, 'access_radius_km': .7, 'min_overlap_share': .15, 'validation_folds': 5, 'random_state': 42,
    'generation_alpha': 'auto', 'beta_max': 60.0,
    'target_tolerance': 1e-9, 'target_max_iterations': 500,
    'ipf_tolerance': 1e-4, 'ipf_max_iterations': 1000,
    'calibration_mass_tolerance': 1e-3, 'calibration_time_tolerance_seconds': 1.0,
    'calibration_max_iterations': 1000, 'calibration_verification_max_iterations': 10000,
    'calibration_search_iterations': 16,
}
# Доли верхней границы beta определяют воспроизводимую начальную сетку поиска.
CALIBRATION_GRID = (0.0, .125, .25, .5, 1.0)
CALIBRATION_TRIAL_COLUMNS = ['hour', 'phase', 'beta', 'status', 'iterations', 'residual',
                             'mass_residual', 'model_mean_hours', 'difference_seconds', 'reason']
FILE_TYPES = ('matrix_stops', 'ref_stops', 'ref_tr_distr', 'los_districts')
# Фиксированный порядок колонок нужен при дописывании отчетов разных контекстов.
STATUS_COLUMNS = ['hour', 'state', 'status', 'iterations', 'residual', 'reason', 'Q_C',
                  'correction_share', 'limited_count', 'machine_count', 'machine_volume',
                  'control_error', 'Q_final', 'warnings', 'target_status', 'support_status']


class InputError(ValueError):
    """Ошибка входных данных с проблемными строками для диагностического отчета."""

    def __init__(self, message, diagnostics=None):
        """Сохраняет сообщение и таблицу проблемных строк, если она передана."""
        super().__init__(message)
        self.diagnostics = diagnostics if diagnostics is not None else pd.DataFrame()


def validate_config(config=None):
    """Дополняет параметры значениями по умолчанию и проверяет их допустимость."""
    supplied = dict(config or {})
    if set(supplied) - set(DEFAULT_CONFIG):
        raise ValueError(f'Неизвестные параметры: {sorted(set(supplied) - set(DEFAULT_CONFIG))}')
    result = {**DEFAULT_CONFIG, **supplied}
    hours = result['hours']
    if (not isinstance(hours, (tuple, list)) or not hours
            or any(h not in HOUR_COLUMNS for h in hours) or len(set(hours)) != len(hours)):
        raise ValueError('hours: требуется непустой список уникальных h5–h23')
    result['hours'] = [h for h in HOUR_COLUMNS if h in hours]
    if result['recalculation_mode'] not in ('recalculate_all', 'preserve_old'):
        raise ValueError('Неизвестный recalculation_mode')
    if result['new_connection_mode'] not in ('all', 'analogues_only'):
        raise ValueError('Неизвестный new_connection_mode')
    if not isinstance(result['include_intrazonal'], bool):
        raise ValueError('include_intrazonal должен быть логическим значением')
    # Численные допуски должны быть конечными, а бюджеты итераций — целыми.
    if result['generation_alpha'] != 'auto':
        value = result['generation_alpha']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('generation_alpha: требуется auto или положительное конечное число')
        result['generation_alpha'] = float(value)
    for name in ('intrazonal_speed_kmh', 'los_neighbor_radius_km', 'beta_max', 'ipf_tolerance',
                 'calibration_mass_tolerance', 'calibration_time_tolerance_seconds', 'target_tolerance'):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name}: требуется положительное конечное число')
        result[name] = float(value)
    if not 1.5 <= result['los_neighbor_radius_km'] <= 3.5:
        raise ValueError('Радиус аналогов должен быть от 1,5 до 3,5 км')
    validate_access_parameters(result['access_radius_km'], result['min_overlap_share'])
    if result['target_tolerance'] > result['ipf_tolerance']:
        raise ValueError('target_tolerance должен быть не больше ipf_tolerance')
    if result['calibration_mass_tolerance'] >= 1:
        raise ValueError('calibration_mass_tolerance должен быть меньше 1')
    if isinstance(result['los_fallback_percentile'], bool) or result['los_fallback_percentile'] not in (70, 75, 80, 85, 90, 95):
        raise ValueError('Процентиль LOS: 70, 75, 80, 85, 90 или 95')
    for name, minimum in (('target_max_iterations', 1), ('validation_folds', 3), ('random_state', 0), ('ipf_max_iterations', 1),
                          ('calibration_max_iterations', 1), ('calibration_verification_max_iterations', 1),
                          ('calibration_search_iterations', 1)):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name}: требуется целое число не меньше {minimum}')
    if result['validation_folds'] > 10:
        raise ValueError('validation_folds: требуется целое число от 3 до 10')
    if result['calibration_verification_max_iterations'] < result['calibration_max_iterations']:
        raise ValueError('Бюджет перепроверки должен быть не меньше бюджета калибровки')
    if result['random_state'] >= 2**32:
        raise ValueError('random_state должен быть меньше 2**32')
    return result


def _keys(frame, columns, name, allow_empty=False):
    """Проверяет ключевые поля и приводит идентификаторы к строковому виду."""
    _require_columns(frame, columns, name)
    if not allow_empty or not frame.empty:
        _require_nonempty_keys(frame, columns, name)
    result = frame.copy()
    for column in columns:
        result[column] = result[column].map(_identifier_key)
    return result


def validate_upload(frame, file_type):
    """Проверяет обязательный или дополнительный CSV и возвращает нормализованную таблицу.

    Связи между разными файлами проверяются позже, при подготовке всего набора.
    """
    if file_type == 'matrix_stops':
        result = _keys(frame, ['source', 'target'], file_type)
        columns = [*HOUR_COLUMNS, 'day']
        _require_columns(result, columns, file_type)
        result[columns] = _numeric(result, columns, file_type)
        hourly = result[HOUR_COLUMNS].sum(axis=1)
        tolerance = 1e-8 * np.maximum(1, np.maximum(hourly, result['day']))
        bad = (result['day'] - hourly).abs() > tolerance
        if bad.any():
            raise InputError('matrix_stops: day не совпадает с суммой h5–h23', result.loc[bad])
        # Повторные остановочные пары разрешены: после построчной проверки их потоки суммируются.
        return result[['source', 'target', *columns]]
    if file_type == 'ref_stops':
        result = _keys(frame, ['id_stop'], file_type, allow_empty=True)
        _require_columns(result, ['wkt'], file_type)
        if result.id_stop.duplicated().any():
            raise ValueError('ref_stops: повторяется идентификатор остановки')
        stop_geodata(result)
        return result[['id_stop', 'wkt']]
    if file_type == 'stop_districts':
        result = _keys(frame, ['id_stop'], file_type, allow_empty=True)
        _require_columns(result, ['id_tr_distr'], file_type)
        if result.id_stop.duplicated().any():
            raise ValueError('Каждая остановка должна принадлежать ровно одному району')
        result['id_tr_distr'] = result.id_tr_distr.map(
            lambda value: _identifier_key(value) if pd.notna(value) and str(value).strip() else pd.NA)
        return result[['id_stop', 'id_tr_distr']]
    if file_type == 'ref_tr_distr':
        result = _keys(frame, ['id_tr_distr', 'state'], file_type)
        _require_columns(result, ['is_new'], file_type)
        result['is_new'] = pd.to_numeric(result['is_new'], errors='coerce')
        normalized, _ = _normalize_districts(result, {})
        if not normalized['is_new'].eq(0).any():
            raise ValueError('Нужен хотя бы один существующий район base')
        bounds = _validate_district_geometry(normalized).bounds.to_numpy()
        if (not np.isfinite(bounds).all() or (bounds[:, [0, 2]] < -180).any()
                or (bounds[:, [0, 2]] > 180).any() or (bounds[:, [1, 3]] < -90).any()
                or (bounds[:, [1, 3]] > 90).any()):
            raise ValueError('Геометрия должна иметь координаты EPSG:4326')
        return normalized
    if file_type == 'los_districts':
        result = _keys(frame, ['agg_source', 'agg_target'], file_type, allow_empty=True)
        _require_columns(result, ['los'], file_type)
        if result.duplicated(['agg_source', 'agg_target']).any():
            raise ValueError('LOS содержит повторные направленные пары')
        result['los'] = _numeric(result, ['los'], file_type)['los']
        if result['los'].le(0).any():
            raise ValueError('LOS должен быть положительным')
        return result[['agg_source', 'agg_target', 'los']]
    raise ValueError(f'Неизвестный тип файла: {file_type}')


def prepare_data(matrix, stop_districts, districts, los, config=None):
    """Готовит согласованные таблицы, почасовой факт, LOS и расстояния в километрах.

    Сначала идут существующие районы, затем новые; проекция выбирается по базе.
    Отсутствующий межрайонный LOS означает запрет; диагональ можно оценить по площади.
    """
    config = validate_config(config)
    # Legacy direct callers may replay archived assignment files. New project uploads require coordinates.
    spatial = 'wkt' in stop_districts.columns
    stop_type = 'ref_stops' if spatial else 'stop_districts'
    frames = {name: validate_upload(frame, name) for name, frame in zip(
        ('matrix_stops', stop_type, 'ref_tr_distr', 'los_districts'),
        (matrix, stop_districts, districts, los)
    )}
    # Нормализованные состояния определяют общий порядок ID для таблиц и массивов.
    districts, stops = frames['ref_tr_distr'], frames[stop_type]
    base = districts.loc[districts.state.eq('base')].set_index('id_tr_distr').sort_index()
    future = districts.loc[districts.state.eq('forecast')].set_index('id_tr_distr')
    ids = list(base.index) + sorted(set(future.index) - set(base.index))
    future = future.loc[ids]
    # Неизвестный или незаданный район означает потерю назначения, а не отказ расчета.
    if not spatial:
        stops = stops.loc[stops.id_tr_distr.isin(base.index)]
    matrix = frames['matrix_stops']
    times = frames['los_districts']
    unknown = (set(times.agg_source) | set(times.agg_target)) - set(base.index)
    if unknown:
        raise ValueError(f'LOS ссылается не на существующие районы: {sorted(unknown)}')
    # Единая метрическая проекция сохраняется для основного расчета и всех фолдов.
    _, crs = _district_geodata(base.reset_index())
    geometry = _validate_district_geometry(future.reset_index())
    metric = gpd.GeoDataFrame({'id_tr_distr': ids}, geometry=geometry).to_crs(crs)
    area = metric.geometry.area.to_numpy() / 1e6
    centers = metric.geometry.centroid
    xy = np.column_stack([centers.x, centers.y])
    if not np.isfinite(area).all() or (area <= 0).any() or not np.isfinite(xy).all():
        raise ValueError('Геометрия должна иметь конечные центроиды и положительную площадь')
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=2) / 1000
    if ((distance == 0) & ~np.eye(len(ids), dtype=bool)).any():
        raise ValueError('Центроиды разных районов совпадают')
    # Внутрирайонная расчетная длина равна корню из площади в квадратных километрах.
    np.fill_diagonal(distance, np.sqrt(area))
    # Общий агрегатор требует уникальные пары; объединяем их после проверки отдельных строк.
    matrix = matrix.groupby(['source', 'target'], as_index=False)[[*HOUR_COLUMNS, 'day']].sum()
    aggregation = (aggregate_spatial_matrix(matrix, stops, base.reset_index(), config) if spatial
                   else aggregate_methodology_matrix(matrix, stops))
    observed = pd.concat([aggregation['interzonal'], aggregation['intrazonal']], ignore_index=True)
    n = len(base)
    # Отсутствующие потоки равны нулю; отсутствие времени LOS сохраняем как NaN.
    demand = np.zeros((len(HOUR_COLUMNS), n, n))
    positions = {key: i for i, key in enumerate(ids)}
    for row in observed.itertuples(index=False):
        demand[:, positions[row.district_source], positions[row.district_target]] = [getattr(row, h) for h in HOUR_COLUMNS]
    los_array = np.full((n, n), np.nan)
    for row in times.itertuples(index=False):
        los_array[positions[row.agg_source], positions[row.agg_target]] = row.los
    # Отключенная диагональ исключается одновременно из спроса и транспортной маски.
    if not config['include_intrazonal']:
        demand[:, np.arange(n), np.arange(n)] = 0
        np.fill_diagonal(los_array, np.nan)
    # Spatial reassignment may create observed pairs absent from the old district LOS.
    # Estimate those pairs later, inside each context, using only that context's donors.
    forbidden = ~np.isfinite(los_array)
    if config['include_intrazonal']:
        np.fill_diagonal(forbidden, False)
    i, j = np.where((demand > 0).any(axis=0) & forbidden)
    if len(i) and not spatial:
        records = [{'source': ids[a], 'target': ids[b], **dict(zip(HOUR_COLUMNS, demand[:, a, b])),
                    'day': demand[:, a, b].sum()} for a, b in zip(i, j)]
        raise InputError('Положительный поток на запрещенных связях LOS', pd.DataFrame(records))
    feature_columns = list(FEATURE_COLUMNS)
    return {'stop_assignment': 'buffer_overlap' if spatial else 'legacy_assignment',
            'config': config, 'ids': ids, 'n_existing': n, 'base': base, 'forecast': future,
            'feature_columns': feature_columns,
            'observed': demand, 'observed_los': los_array, 'distance': distance,
            'metric_crs': crs, 'aggregation': aggregation, 'inputs': frames}


def fit_generation(features, observed, alpha):
    """Обучает 38 неотрицательных Ridge-регрессий отправлений и прибытий.

    Все 19 часов используют общие масштабы признаков, оцененные на обучающих районах.
    """
    targets = np.stack((observed.sum(axis=2), observed.sum(axis=1)))
    return _train_generation(features, targets, alpha)


def _train_generation(features, targets, alpha):
    """Select a common penalty using only the supplied training districts and labels."""
    active = np.any(features != 0, axis=0)
    z = StandardScaler(with_mean=False).fit_transform(features)[:, active]
    singular = np.linalg.svd(z, compute_uv=False)
    rank = int(np.linalg.matrix_rank(z)) if active.any() else 0
    count = int(active.sum())
    # Each inner training set must contain at least p + 1 districts; at least six overall.
    minimum = max(6, math.ceil(3 * (count + 1) / 2))
    diagnostics = {'training_districts': len(features), 'active_features': count, 'feature_rank': rank,
                   'condition_number': float(singular[0] / singular[-1]) if rank == count and count else None,
                   'minimum_tuning_districts': minimum, 'requested_alpha': alpha, 'warnings': []}
    if not count:
        diagnostics['warnings'].append('NO_ACTIVE_FEATURES')
    if rank < count:
        diagnostics['warnings'].append('RANK_DEFICIENT_FEATURES')
    elif diagnostics['condition_number'] is not None and diagnostics['condition_number'] > 1e8:
        diagnostics['warnings'].append('ILL_CONDITIONED_FEATURES')
    if len(features) < minimum:
        diagnostics['warnings'].append('INSUFFICIENT_TUNING_DISTRICTS')
    trials = []
    selected = 1.0 if alpha == 'auto' else float(alpha)
    status = 'FIXED'
    if alpha == 'auto':
        status = 'FALLBACK_INSUFFICIENT_DATA' if count else 'FALLBACK_NO_ACTIVE_FEATURES'
        if count and len(features) >= minimum:
            splits = list(KFold(3, shuffle=True, random_state=42).split(features))
            scores = []
            for candidate in GENERATION_ALPHA_GRID:
                candidate_errors = []
                for fold, (training, hidden) in enumerate(splits, 1):
                    row = {'alpha': candidate, 'inner_fold': fold, 'training_districts': len(training),
                           'hidden_districts': len(hidden), 'status': 'OK', 'absolute_error': None,
                           'fact_volume': float(targets[:, :, hidden].sum()), 'reason': ''}
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            fitted = _fit_generation_targets(features[training], targets[:, :, training], candidate)
                        predicted = predict_generation(fitted, features[hidden])['raw']
                        if not np.isfinite(predicted).all():
                            raise FloatingPointError('Non-finite inner generation predictions')
                        if caught:
                            row.update(status='FIT_WARNING', reason='; '.join(str(item.message) for item in caught))
                        else:
                            row['absolute_error'] = float(np.abs(predicted - targets[:, :, hidden]).sum())
                            candidate_errors.append(row['absolute_error'])
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                        row.update(status='FIT_FAILED', reason=str(error) or type(error).__name__)
                    trials.append(row)
                if len(candidate_errors) == len(splits):
                    scores.append((sum(candidate_errors), candidate))
            if scores:
                # Grid order (smaller alpha) resolves exact ties; no outer-fold results participate.
                selected = min(scores)[1]
                status = 'SELECTED'
            else:
                status = 'FALLBACK_TUNING_FAILED'
                diagnostics['warnings'].append('TUNING_FAILED')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = _fit_generation_targets(features, targets, selected)
    diagnostics['fit_warnings'] = [f'{item.category.__name__}: {item.message}' for item in caught]
    if caught:
        diagnostics['warnings'].append('FINAL_FIT_WARNING')
    for item in caught:
        warnings.warn(item.message, item.category)
    diagnostics.update(selected_alpha=selected, selection_status=status)
    model.update(diagnostics=diagnostics, alpha_trials=trials)
    return model


def _fit_generation_targets(features, targets, alpha):
    """Обучает ту же генерацию на заданных районных итогах (направление, час, район)."""
    scaler = StandardScaler(with_mean=False).fit(features)
    # Полностью нулевые обучающие признаки не участвуют в оценке коэффициентов.
    active = np.any(features != 0, axis=0)
    z = scaler.transform(features)
    coefficients = np.zeros((2, len(HOUR_COLUMNS), features.shape[1]))
    for direction, target in enumerate(targets):
        if active.any():
            model = Ridge(alpha=alpha, positive=True, fit_intercept=False, solver='lbfgs', tol=1e-10)
            model.fit(z[:, active], target.T)
            coefficients[direction][:, active] = model.coef_
    return {'coefficients': coefficients, 'scale': scaler.scale_, 'active': active,
            'min': features.min(axis=0), 'max': features.max(axis=0),
            'zero_hours': (targets == 0).all(axis=(0, 2)),
            'training_targets': targets.transpose(0, 2, 1)}


def predict_generation(model, features):
    """Вычисляет исходную генерацию и согласованные почасовые цели для IPF."""
    raw = np.einsum('dhk,nk->dhn', model['coefficients'], features / model['scale'])
    totals = raw.sum(axis=2)
    goals = np.full_like(raw, np.nan)
    factors = np.full_like(totals, np.nan)
    statuses = []
    # Часы согласуются независимо; неопределимые цели остаются NaN с явным статусом.
    for h in range(len(HOUR_COLUMNS)):
        o, d = totals[:, h]
        if o == 0 and d == 0:
            goals[:, h] = 0
            factors[:, h] = 0
            statuses.append('ZERO_OBSERVED_HOUR' if model['zero_hours'][h] else 'ZERO_PREDICTED_HOUR')
        elif o == 0 or d == 0:
            statuses.append('GENERATION_UNRECONCILABLE')
        else:
            # Общий объем часа — среднее суммарных предсказанных отправлений и прибытий.
            factors[:, h] = ((o + d) / 2) / totals[:, h]
            goals[:, h] = raw[:, h] * factors[:, h, None]
            statuses.append('OK')
    return {'raw': raw, 'goals': goals, 'totals': totals, 'factors': factors, 'statuses': statuses}


def preprocess_los(distance, observed_los):
    """Raise speeds below 4.5 km/h by shortening LOS; retain missing connections."""
    result = observed_los.copy()
    corrected = np.isfinite(result) & (result > distance / LOS_MIN_SPEED_KMH)
    result[corrected] = distance[corrected] / LOS_MIN_SPEED_KMH
    return result, corrected


def _nearest_los_donor(distance, known, left, right, source, target, ids):
    """Choose the nearest directed observed interdistrict pair, never a filled time."""
    candidates = []
    for u in left:
        for v in right:
            if u == v or not np.isfinite(known[u, v]) or known[u, v] <= 0 or distance[u, v] <= 0:
                continue
            offset = ((0. if source == u else distance[source, u])
                      + (0. if target == v else distance[target, v]))
            candidates.append((float(offset), ids[u], ids[v], u, v))
    if not candidates:
        return None
    offset, _, _, u, v = min(candidates)
    return u, v, offset


def build_los(distance, observed_los, training, ids, config, *, required_pairs=None):
    """Строит времена поездок в часах и направленную маску доступности связей.

    Доноры скорости берутся только из наблюдаемого обучающего блока R → R.
    Новые времена переносятся с ближайших наблюдаемых аналогов с поправкой на длину.
    Общая модель скорости — резерв при отсутствии подходящей донорской связи.
    required_pairs разрешает оценку отсутствующих времён положительной буферной опоры.
    """
    training = np.asarray(training, dtype=int)
    n = len(ids)
    required_pairs = (np.zeros((n, n), dtype=bool) if required_pairs is None
                      else np.asarray(required_pairs, dtype=bool))
    if required_pairs.shape != (n, n):
        raise ValueError('required_pairs: размер должен совпадать с числом районов')
    raw_known = np.full((n, n), np.nan)
    raw_known[np.ix_(training, training)] = observed_los
    known, corrected = preprocess_los(distance, raw_known)
    mask = np.isfinite(known)
    minimum_speed = short_speed = LOS_MIN_SPEED_KMH
    intrazonal_speed = max(config['intrazonal_speed_kmh'], minimum_speed)
    observed_i, observed_j = np.where(mask)
    preprocessing = pd.DataFrame({
        'source': [ids[i] for i in observed_i], 'target': [ids[j] for j in observed_j],
        'distance_km': distance[observed_i, observed_j],
        'raw_los': raw_known[observed_i, observed_j], 'los': known[observed_i, observed_j],
        'raw_speed_kmh': distance[observed_i, observed_j] / raw_known[observed_i, observed_j],
        'speed_kmh': distance[observed_i, observed_j] / known[observed_i, observed_j],
        'corrected': corrected[observed_i, observed_j]})
    # Preserve the fitted curve; join the short branch at its boundary speed.
    donor_i, donor_j = np.where(mask & (distance >= 1.5))
    lengths = distance[donor_i, donor_j]
    speed = lengths / known[donor_i, donor_j]
    a = b = fitted_b = None
    if len(lengths) >= 3 and len(np.unique(lengths)) >= 2:
        x = np.log(lengths)
        a = max(0.0, float(np.dot(x - x.mean(), speed - speed.mean()) / np.sum((x - x.mean())**2)))
        fitted_b = float(speed.mean() - a * x.mean())
        b = fitted_b
        short_speed = max(minimum_speed, float(a * np.log(1.5) + b))
    threshold = float(np.percentile(known[mask], config['los_fallback_percentile'], method='linear')) if mask.any() else None
    # Оценочная диагональ разрешена без аналогов и не служит наблюдаемым донором LOS.
    if config['include_intrazonal']:
        np.fill_diagonal(mask, True)
    # Сам район не является своим аналогом: диагональ расстояний содержит корень из площади.
    analogues = {i: sorted(
                    [int(r) for r in training if r != i and distance[i, r] <= config['los_neighbor_radius_km']],
                    key=lambda r: (distance[i, r], ids[r]))
                 for i in range(n) if i not in training}
    times = known.copy()
    records, errors = [], []
    evaluated_lengths = list(lengths)
    training_set = set(training)
    for i in range(n):
        for j in range(n):
            record = {'source': ids[i], 'target': ids[j], 'distance_km': float(distance[i, j]),
                      'allowed': False, 'los': None, 'time_method': None, 'speed_floor': False,
                      'raw_observed_los': float(raw_known[i, j]) if np.isfinite(raw_known[i, j]) else None,
                      'los_preprocessed': bool(corrected[i, j]),
                      'extrapolated': False, 'witness_source': None, 'witness_target': None,
                      'donor_los': None, 'donor_distance_km': None, 'donor_offset_km': None,
                      'distance_ratio': None, 'donor_los_preprocessed': False, 'fallback_reason': None}
            if i == j and not config['include_intrazonal']:
                record['reason'] = 'diagonal_disabled'
            elif i == j and not np.isfinite(known[i, i]):
                value = float(distance[i, i] / intrazonal_speed)
                times[i, i] = value
                record.update(allowed=True, reason='intrazonal_estimate', los=value,
                              time_method='intrazonal_area_speed',
                              speed_floor=bool(config['intrazonal_speed_kmh'] < minimum_speed))
            elif (i in training_set and j in training_set
                  and (mask[i, j] or not required_pairs[i, j])):
                # Обучающие времена и запреты сохраняем без повторной оценки.
                record.update(allowed=bool(mask[i, j]), reason='observed' if mask[i, j] else 'no_observed_connection',
                              los=float(known[i, j]) if mask[i, j] else None,
                              time_method='observed' if mask[i, j] else None)
            else:
                d = distance[i, j]
                donor = None
                if i not in training_set or j not in training_set:
                    left = [i] if i in training_set else analogues[i]
                    right = [j] if j in training_set else analogues[j]
                    donor = _nearest_los_donor(distance, known, left, right, i, j, ids)
                    if donor is None:
                        record['fallback_reason'] = ('no_analogue_in_radius' if not left or not right
                                                     else 'no_observed_analogue_connection')
                        if config['new_connection_mode'] == 'analogues_only':
                            record['reason'] = record['fallback_reason']
                            records.append(record)
                            continue
                evaluated_lengths.append(d)
                if donor is not None:
                    u, v, offset = donor
                    ratio = float(d / distance[u, v])
                    value, method = float(known[u, v] * ratio), 'analogue_distance_scaled'
                    record.update(witness_source=ids[u], witness_target=ids[v],
                                  donor_los=float(known[u, v]), donor_distance_km=float(distance[u, v]),
                                  donor_offset_km=offset, distance_ratio=ratio,
                                  donor_los_preprocessed=bool(corrected[u, v]))
                elif d < 1.5:
                    value, method = d / short_speed, 'short_fixed'
                    record['speed_floor'] = bool(a is not None and a * np.log(1.5) + b < minimum_speed)
                elif a is None:
                    value, method = None, None
                    errors.append({'source': ids[i], 'target': ids[j], 'status': 'LOS_MODEL_UNAVAILABLE'})
                else:
                    v = a * np.log(d) + b
                    value, method = d / max(minimum_speed, v), 'global_log_speed'
                    record.update(speed_floor=bool(v < minimum_speed),
                                  extrapolated=bool(d < lengths.min() or d > lengths.max()))
                record.update(los=float(value) if value is not None else None, time_method=method)
                if value is None:
                    record['reason'] = 'LOS_MODEL_UNAVAILABLE'
                else:
                    record.update(allowed=True, reason=(
                        'analogue_transfer' if donor is not None else
                        'regression_observed_support' if i in training_set and j in training_set
                        else 'regression_transfer'))
                    mask[i, j], times[i, j] = True, value
            records.append(record)
    # The anchored branches meet exactly; retain the separate nonmonotonicity diagnostic.
    warnings = []
    right_time = None if a is None else 1.5 / short_speed
    # Время убывает на активной логарифмической ветви при v_min < A × ln(d) + B < A.
    if a is not None and a > minimum_speed and evaluated_lengths:
        low, high = max(1.5, min(evaluated_lengths)), max(evaluated_lengths)
        if max(a * np.log(low) + b, minimum_speed) < min(a * np.log(high) + b, a):
            warnings.append('LOS_NONMONOTONE_TIME')
    donors = pd.DataFrame({'source': [ids[i] for i in donor_i], 'target': [ids[j] for j in donor_j],
                           'distance_km': lengths, 'los': known[donor_i, donor_j], 'speed_kmh': speed})
    return {'times': times, 'mask': mask, 'pairs': pd.DataFrame(records), 'donors': donors,
            'preprocessing': preprocessing,
            'analogues': {ids[i]: [ids[r] for r in values] for i, values in analogues.items()},
            'model': {'A': a, 'B': b, 'B_fitted': fitted_b,
                      'boundary_distance_km': 1.5, 'boundary_speed_kmh': short_speed,
                      'short_speed_source': 'minimum_without_regression' if a is None else 'fitted_curve_at_boundary',
                      'threshold_hours': threshold,
                      'threshold_minutes': None if threshold is None else threshold * 60,
                      'percentile': config['los_fallback_percentile'], 'short_speed_kmh': short_speed,
                      'minimum_speed_kmh': minimum_speed,
                      'minimum_speed_source': 'fixed_preprocessing_threshold',
                      'preprocessed_pairs': int(corrected.sum()),
                      'observed_pairs': len(observed_i),
                      'intrazonal_speed_kmh': intrazonal_speed,
                      'min_distance_km': float(lengths.min()) if len(lengths) else None,
                      'max_distance_km': float(lengths.max()) if len(lengths) else None,
                      'left_boundary_hours': 1.5 / short_speed, 'right_boundary_hours': right_time,
                      'zero_slope': a == 0, 'warnings': warnings,
                      'new_connection_mode': config['new_connection_mode'],
                      'transfer_policy': ('OBSERVED_ANALOGUES_ONLY' if config['new_connection_mode'] == 'analogues_only'
                                          else 'NEAREST_OBSERVED_ANALOGUE_THEN_REGRESSION'),
                      'analogue_blocked_pairs': sum(row['reason'] in (
                          'no_analogue_in_radius', 'no_observed_analogue_connection') for row in records),
                      'analogue_transferred_pairs': sum(row['reason'] == 'analogue_transfer' for row in records),
                      'regression_transferred_pairs': sum(row['reason'] == 'regression_transfer' for row in records),
                      'observed_support_estimated_pairs': sum(
                          row['reason'] == 'regression_observed_support' for row in records),
                      'percentile_applied': False, 'los_slice': 'ONE_SLICE_FOR_ALL_HOURS'}, 'errors': errors}


def _prepare_ipf_support(mask, outgoing, incoming, config):
    """Adapt absolute flow feasibility tolerance to the core's rowwise criterion."""
    o, d = np.asarray(outgoing, float), np.asarray(incoming, float)
    absolute = config['ipf_tolerance'] * min(np.maximum(o, 1).min(), np.maximum(d, 1).min())
    support, boundary = prepare_support(mask, o, d, absolute)
    return {'mask': mask.copy(), 'outgoing': o.copy(), 'incoming': d.copy(),
            'support': support, 'boundary': boundary}


def balance_ipf(times, mask, outgoing, incoming, beta, config, *, mass_tolerance=None,
                prepared_support=None):
    """Балансирует гравитационные веса методом IPF в логарифмической шкале.

    Для модельных матриц проверяет максимальную районную невязку; mass_tolerance
    включает критерий относительной суммарной ошибки калибровки. При отказе matrix=None.
    """
    o, d = np.asarray(outgoing, dtype=float), np.asarray(incoming, dtype=float)
    tolerance = config['ipf_tolerance']
    result = {'status': 'IPF_INFEASIBLE', 'iterations': 0, 'residual': None,
              'matrix': None, 'outgoing': o.tolist(), 'incoming': d.tolist(), 'reason': '', 'mass_residual': None}
    def fail(reason):
        """Возвращает причину невыполнимости без успешной модельной матрицы."""
        result['reason'] = reason
        return result
    if not np.isfinite(o).all() or not np.isfinite(d).all() or (o < 0).any() or (d < 0).any():
        return fail('Invalid targets')
    if abs(o.sum() - d.sum()) > tolerance * max(1, o.sum(), d.sum()):
        return fail('Total mismatch')
    if not o.any() and not d.any():
        return {**result, 'status': 'OK', 'matrix': np.zeros_like(times), 'residual': 0.0, 'mass_residual': 0.0,
                'support_status': 'OK', 'boundary': []}
    # Нулевые цели исключаем из делений, не отсекая малые положительные потоки.
    support = mask & (o[:, None] > 0) & (d[None, :] > 0)
    if ((o > 0) & ~support.any(axis=1)).any() or ((d > 0) & ~support.any(axis=0)).any():
        return fail('Isolated positive target')
    n = len(o)
    # Отправления и прибытия — разные вершины; проверка компонент необходима, но не достаточна.
    graph = bmat([[None, csr_matrix(support)], [csr_matrix(support.T), None]], format='csr')
    _, labels = connected_components(graph, directed=False)
    for label in np.unique(labels):
        so, sd = o[labels[:n] == label].sum(), d[labels[n:] == label].sum()
        if abs(so - sd) > tolerance * max(1, so, sd):
            return fail(f'Component mismatch: {label}')
    try:
        if prepared_support is None:
            prepared_support = _prepare_ipf_support(mask, o, d, config)
        elif not (np.array_equal(prepared_support['mask'], mask)
                  and np.array_equal(prepared_support['outgoing'], o)
                  and np.array_equal(prepared_support['incoming'], d)):
            raise ValueError('Prepared IPF support does not match mask and targets')
    except ForecastCalculationError as error:
        return fail(str(error))
    support = prepared_support['support']
    result['support_status'] = 'OK'
    result['boundary'] = prepared_support['boundary']
    # Балансируем логарифмы весов, чтобы снизить риск численной потери малых значений.
    log_t = np.full(times.shape, -np.inf)
    log_t[support] = -beta * times[support]
    positive_o, positive_d = o > 0, d > 0
    volume = max(o.sum(), d.sum())
    for iteration in range(1, config['ipf_max_iterations'] + 1):
        row_sum = logsumexp(log_t, axis=1)
        if not np.isfinite(row_sum[positive_o]).all():
            return fail('No positive row weight')
        log_t[positive_o] += (np.log(o[positive_o]) - row_sum[positive_o])[:, None]
        col_sum = logsumexp(log_t, axis=0)
        if not np.isfinite(col_sum[positive_d]).all():
            return fail('No positive column weight')
        log_t[:, positive_d] += (np.log(d[positive_d]) - col_sum[positive_d])[None, :]
        matrix = np.exp(log_t)
        # Оба критерия сохраняем для диагностики, но завершаем цикл по выбранному допуску.
        error_o, error_d = np.abs(matrix.sum(axis=1) - o), np.abs(matrix.sum(axis=0) - d)
        residual = max(np.max(error_o / np.maximum(o, 1)), np.max(error_d / np.maximum(d, 1)))
        mass_residual = max(error_o.sum(), error_d.sum()) / volume
        converged = residual <= tolerance if mass_tolerance is None else mass_residual <= mass_tolerance
        if converged:
            return {**result, 'status': 'OK', 'matrix': matrix, 'iterations': iteration,
                    'residual': float(residual), 'mass_residual': float(mass_residual)}
    return {**result, 'status': 'IPF_NOT_CONVERGED', 'iterations': iteration,
            'residual': float(residual), 'mass_residual': float(mass_residual), 'diagnostic_matrix': matrix}


def calibrate_beta(observed, times, mask, config):
    """Подбирает beta в 1/час по среднему наблюдаемому времени и проверяет устойчивость.

    Поиск имеет ограниченный бюджет; его допуски не ослабляют финальный IPF.
    Численный отказ оставляет beta недоступным для ненулевых модельных расчетов.
    """
    total = observed.sum()
    trials = []
    result = {'beta': None, 'usable': False, 'observed_mean_hours': None,
              'model_mean_hours': None, 'difference_hours': None, 'boundary': False,
              'status': 'CALIBRATION_FAILED', 'reason': '', 'coarse_beta': None,
              'beta_change': None, 'numerical_error_seconds': None,
              'trial_count': 0, 'total_ipf_iterations': 0, 'trials': trials}
    # Нулевой час и одинаковые времена активных связей не требуют численного поиска.
    if total == 0:
        result.update(beta=0.0, usable=True, status='ZERO_OBSERVED_HOUR')
        return result
    mean = float(np.sum(observed[mask] * times[mask]) / total)
    o, d = observed.sum(axis=1), observed.sum(axis=0)
    result['observed_mean_hours'] = mean
    try:
        prepared_support = _prepare_ipf_support(mask, o, d, config)
    except ForecastCalculationError as error:
        result['reason'] = str(error)
        return result
    active = mask & (o[:, None] > 0) & (d[None, :] > 0)
    if np.ptp(times[active]) == 0:
        result.update(beta=0.0, usable=True, status='BETA_UNIDENTIFIED',
                      model_mean_hours=mean, difference_hours=0.0, numerical_error_seconds=0.0)
        return result
    cache = {}
    time_tolerance = config['calibration_time_tolerance_seconds']

    def evaluate(beta, phase):
        """Оценивает одну пробу заданной точности и кэширует диагностику без матрицы."""
        key = (phase, float(beta))
        if key in cache:
            return cache[key]
        # Перепроверка имеет допуск в десять раз строже и отдельный бюджет итераций.
        verify = phase == 'verification'
        local_config = {**config, 'ipf_max_iterations': config[
            'calibration_verification_max_iterations' if verify else 'calibration_max_iterations']}
        fitted = balance_ipf(times, mask, o, d, beta, local_config,
                             mass_tolerance=config['calibration_mass_tolerance'] / (10 if verify else 1),
                             prepared_support=prepared_support)
        model_mean = None if fitted['status'] != 'OK' else float(
            np.sum(fitted['matrix'][mask] * times[mask]) / total)
        trial = {'phase': phase, 'beta': float(beta), 'status': fitted['status'],
                 'iterations': fitted['iterations'], 'residual': fitted['residual'],
                 'mass_residual': fitted['mass_residual'], 'model_mean_hours': model_mean,
                 'difference_seconds': None if model_mean is None else (model_mean - mean) * 3600,
                 'reason': fitted['reason']}
        trials.append(trial)
        cache[key] = trial
        return trial

    def search(phase):
        """Ищет коэффициент на сетке и уточняет допустимый интервал делением пополам."""
        grid = [evaluate(config['beta_max'] * fraction, phase) for fraction in CALIBRATION_GRID]
        successful = [trial for trial in grid if trial['status'] == 'OK']
        complete = len(successful) == len(grid)
        if not successful:
            return None, None, 'IPF failed on calibration grid'
        # Неполная сетка не подтверждает практическую неопределимость коэффициента.
        if np.ptp([trial['model_mean_hours'] for trial in successful]) * 3600 <= time_tolerance:
            return (grid[0], 'BETA_UNIDENTIFIED', '') if complete else (
                None, None, 'Incomplete grid cannot establish beta identifiability')
        best = min(successful, key=lambda trial: (abs(trial['difference_seconds']), trial['beta']))
        if abs(best['difference_seconds']) <= time_tolerance:
            return best, 'BETA_BOUNDARY' if best['beta'] in (0.0, config['beta_max']) else 'OK', ''
        # Через отказавшую точку не интерполируем даже при успешных более далеких пробах.
        for left, right in zip(grid, grid[1:]):
            if left['status'] != 'OK' or right['status'] != 'OK':
                continue
            if left['difference_seconds'] * right['difference_seconds'] >= 0:
                continue
            for _ in range(config['calibration_search_iterations']):
                trial = evaluate((left['beta'] + right['beta']) / 2, phase)
                if trial['status'] != 'OK':
                    return None, None, 'IPF failed during interval refinement'
                if abs(trial['difference_seconds']) <= time_tolerance:
                    return trial, 'OK', ''
                if left['difference_seconds'] * trial['difference_seconds'] > 0:
                    left = trial
                else:
                    right = trial
            return None, None, 'Calibration search iteration limit reached'
        # Недостижимое среднее допускает граничный beta только при полной успешной сетке.
        if complete and best['beta'] in (0.0, config['beta_max']):
            return best, 'BETA_BOUNDARY', ''
        return None, None, 'No reliable mean-time bracket'

    # Сравниваем среднее при том же beta, затем используем результат более точного поиска.
    coarse, coarse_status, reason = search('coarse')
    if coarse is not None:
        result['coarse_beta'] = coarse['beta']
        checked = evaluate(coarse['beta'], 'verification')
        fine, fine_status, reason = search('verification')
        if fine is None or checked['status'] != 'OK':
            result.update(status='CALIBRATION_VERIFICATION_FAILED', reason=reason or 'Candidate IPF verification failed')
        else:
            numerical_error = abs(checked['model_mean_hours'] - coarse['model_mean_hours']) * 3600
            result.update(beta_change=fine['beta'] - coarse['beta'], numerical_error_seconds=numerical_error)
            if numerical_error > time_tolerance or ((coarse_status == 'BETA_UNIDENTIFIED') !=
                                                    (fine_status == 'BETA_UNIDENTIFIED')):
                result.update(status='CALIBRATION_UNSTABLE', reason='Mean time or beta identifiability changed beyond tolerance')
            else:
                result.update(beta=fine['beta'], usable=True, status=fine_status,
                              model_mean_hours=fine['model_mean_hours'],
                              difference_hours=fine['difference_seconds'] / 3600,
                              boundary=fine_status == 'BETA_BOUNDARY')
    else:
        result['reason'] = reason
    result.update(trial_count=len(trials), total_ipf_iterations=sum(trial['iterations'] for trial in trials))
    return result


def assemble(observed, base, forecast, mode):
    """Собирает итог по выбранному режиму без последующей балансировки.

    Сохраняет фактический старый блок либо добавляет к факту модельное изменение
    с ограничением отрицательных потоков; возвращает также диагностику коррекции.
    """
    n, size = len(observed), len(forecast)
    anchor = np.zeros((size, size))
    anchor[:n, :n] = observed
    correction = np.zeros_like(anchor)
    unbounded = None
    # В этом режиме старый блок копируется точно, а новые связи берутся из прогноза.
    if mode == 'preserve_old':
        final = forecast.copy()
        final[:n, :n] = observed
        eps = np.zeros_like(anchor)
        expected_outgoing, expected_incoming = final.sum(axis=1), final.sum(axis=0)
    else:
        padded = np.zeros_like(anchor)
        padded[:n, :n] = base
        # Поправка равна M_forecast − M_base; ее отрицательный выход за факт компенсирует C.
        unbounded = anchor + (forecast - padded)
        correction = np.maximum(0, -unbounded)
        final = np.maximum(0, unbounded)
        eps = 64 * np.finfo(float).eps * np.maximum.reduce([np.ones_like(anchor), anchor, padded, forecast])
        expected_outgoing = anchor.sum(axis=1) + forecast.sum(axis=1) - padded.sum(axis=1) + correction.sum(axis=1)
        expected_incoming = anchor.sum(axis=0) + forecast.sum(axis=0) - padded.sum(axis=0) + correction.sum(axis=0)
    # Машинный порог разделяет виды диагностики, но не отсекает саму коррекцию.
    substantive = correction > eps
    machine = (correction > 0) & ~substantive
    q = float(correction.sum())
    return {'matrix': final, 'correction': correction, 'unbounded': unbounded,
            'limited_count': int((correction > 0).sum()), 'Q_C': q,
            'O_C': correction.sum(axis=1), 'D_C': correction.sum(axis=0),
            'correction_share': q / float(final.sum()) if final.sum() > 0 else None,
            'machine_count': int(machine.sum()), 'machine_volume': float(correction[machine].sum()),
            'control_error': float(max(np.max(np.abs(final.sum(axis=1) - expected_outgoing)),
                                       np.max(np.abs(final.sum(axis=0) - expected_incoming)))),
            'warnings': ['PIVOT_FLOW_LIMITED'] if substantive.any() else []}


def run_context(prepared, training, *, validation=False, log=logger, context='main'):
    """Рассчитывает независимый основной контекст или фолд проверки.

    В фолде признаки скрытых районов доступны, но спрос и LOS для обучения
    берутся только из подматрицы обучающих районов R → R.
    """
    context_started = perf_counter()
    config = prepared['config']
    training = np.asarray(training, dtype=int)
    ids = prepared['ids'][:prepared['n_existing']] if validation else prepared['ids']
    size = len(ids)
    observed = prepared['observed'][:, training][:, :, training]
    training_los = prepared['observed_los'][np.ix_(training, training)]
    columns = prepared['feature_columns']
    base_features = prepared['base'][columns].to_numpy(dtype=float)
    features = base_features if validation else prepared['forecast'][columns].to_numpy(dtype=float)
    # Один обученный набор коэффициентов и масштабов используется для обоих состояний.
    if not validation:
        log.info('%s fit_generation started; training_districts=%s', context, len(training))
    model = fit_generation(base_features[training], observed, config['generation_alpha'])
    model['feature_columns'] = columns
    generation = {'base': predict_generation(model, base_features[training]),
                  'forecast': predict_generation(model, features)}
    generation_finished = perf_counter()
    if not validation:
        log.info('%s generation pass; forecast_districts=%s', context, size)
        log.info('%s build_los started', context)
    required_pairs = np.zeros((size, size), dtype=bool)
    if prepared.get('stop_assignment') == 'buffer_overlap':
        # Hidden demand must not decide which missing training times get filled.
        required_pairs[np.ix_(training, training)] = (observed > 0).any(axis=0)
    los = build_los(prepared['distance'][:size, :size], training_los, training, ids, config,
                    required_pairs=required_pairs)
    los_finished = perf_counter()
    if not validation:
        log.info('%s build_los pass; allowed_pairs=%s; errors=%s; warnings=%s',
                 context, int(los['mask'].sum()), len(los['errors']), los['model']['warnings'])
    # Калибровка и базовый IPF используют ту же оценочную диагональ, что и прогноз.
    training_los = los['times'][np.ix_(training, training)]
    base_mask = np.isfinite(training_los)
    same_state = size == len(training) and np.array_equal(features, base_features[training])
    # Reconcile every generation hour, independent of the selected OD export.
    for state in ('forecast', 'base'):
        gen = generation[state]
        gen['adjusted_goals'] = np.full_like(gen['goals'], np.nan)
        gen['reconciliation'] = []
        mask = los['mask'] if state == 'forecast' else base_mask
        for h, hour in enumerate(HOUR_COLUMNS):
            if state == 'base' and config['recalculation_mode'] == 'preserve_old':
                item = {'status': 'NOT_REQUIRED', 'reason': '', 'elapsed_seconds': 0., 'solve_executed': False}
            elif state == 'base' and same_state:
                item = {**generation['forecast']['reconciliation'][h],
                        'elapsed_seconds': 0., 'solve_executed': False, 'reused_from': 'forecast'}
                gen['adjusted_goals'][:, h] = generation['forecast']['adjusted_goals'][:, h]
            elif not np.isfinite(gen['goals'][:, h]).all():
                item = {'status': 'GENERATION_UNRECONCILABLE', 'reason': 'Invalid generation totals'}
            elif state == 'forecast' and los['errors'] and gen['goals'][:, h].any():
                item = {'status': 'LOS_UNAVAILABLE', 'reason': 'LOS estimates unavailable'}
            else:
                solve_started = perf_counter()
                item = reconcile_targets(mask, *gen['goals'][:, h],
                    tolerance=config['target_tolerance'], max_iterations=config['target_max_iterations'])
                item.update(elapsed_seconds=perf_counter() - solve_started, solve_executed=True)
                if item['outgoing'] is not None:
                    gen['adjusted_goals'][:, h] = (item['outgoing'], item['incoming'])
            item.setdefault('elapsed_seconds', 0.)
            item.setdefault('solve_executed', False)
            item.setdefault('reused_from', '')
            gen['reconciliation'].append(item)
    hours = {}
    started = perf_counter()
    if not validation:
        log.info('%s hourly matrices started; hours=%s; mode=%s',
                 context, len(config['hours']), config['recalculation_mode'])
    for hour in config['hours']:
        h = HOUR_COLUMNS.index(hour)
        calibration = calibrate_beta(observed[h], training_los, base_mask, config)
        states = {}
        for state in ('forecast', 'base'):
            if state == 'base' and config['recalculation_mode'] == 'preserve_old':
                states[state] = {'status': 'NOT_REQUIRED', 'matrix': None}
                continue
            if state == 'base' and same_state:
                # Общий массив обеспечивает точно нулевое модельное изменение при совпадении состояний.
                states[state] = states['forecast']
                continue
            gen = generation[state]
            times, mask = (los['times'], los['mask']) if state == 'forecast' else (training_los, base_mask)
            reconciliation = gen['reconciliation'][h]
            goals = gen['adjusted_goals'][:, h]
            if not np.isfinite(goals).all():
                states[state] = {'status': reconciliation['status'], 'matrix': None,
                                 'reason': reconciliation.get('reason', '')}
            elif not goals.any():
                states[state] = balance_ipf(times, mask, *goals, 0, config)
            elif not calibration['usable']:
                states[state] = {'status': calibration['status'], 'matrix': None, 'reason': calibration['reason']}
            else:
                states[state] = balance_ipf(times, mask, *goals, calibration['beta'], config)
            states[state]['target_status'] = reconciliation['status']
        assembly = {'status': 'UNAVAILABLE', 'matrix': None}
        if states['forecast']['status'] == 'OK' and (
                config['recalculation_mode'] == 'preserve_old' or states['base']['status'] == 'OK'):
            # Для сборки ставим обучающий блок первым, затем восстанавливаем общий порядок ID.
            order = list(training) + [i for i in range(size) if i not in set(training)]
            forecast = states['forecast']['matrix'][np.ix_(order, order)]
            assembly = assemble(observed[h], states['base']['matrix'], forecast, config['recalculation_mode'])
            reverse = np.argsort(order)
            for name in ('matrix', 'correction', 'unbounded'):
                if assembly[name] is not None:
                    assembly[name] = assembly[name][np.ix_(reverse, reverse)]
            for name in ('O_C', 'D_C'):
                assembly[name] = assembly[name][reverse]
            assembly['status'] = 'OK'
        hours[hour] = {**states, 'assembly': assembly, 'calibration': calibration}
    if not validation:
        log.info('%s hourly matrices finished; successful_assembly_hours=%s; expected_hours=%s; duration_seconds=%.3f',
                 context, sum(item['assembly']['status'] == 'OK' for item in hours.values()),
                 len(hours), perf_counter() - started)
    return {'ids': ids, 'training': training.tolist(), 'generation_model': model,
            'generation': generation, 'los': los, 'hours': hours,
            'timings': {'generation_seconds': generation_finished - context_started,
                        'los_seconds': los_finished - generation_finished,
                        'reconciliation_seconds': started - los_finished,
                        'hourly_seconds': perf_counter() - started,
                        'total_seconds': perf_counter() - context_started,
                        'generation_hours': len(HOUR_COLUMNS), 'selected_od_hours': len(hours),
                        'successful_assembly_hours': sum(item['assembly']['status'] == 'OK' for item in hours.values()),
                        'reconciliation_calls': sum(item['solve_executed'] for gen in generation.values()
                                                    for item in gen['reconciliation'])},
            'zero_observed_day': bool(model['zero_hours'].all())}


def regression_metrics(fact, predicted):
    """Считает MAE, RMSE, WAPE в процентах и R² с учетом неопределенных знаменателей."""
    fact, predicted = np.asarray(fact), np.asarray(predicted)
    if not len(fact):
        return {'status': 'NO_TEST_PAIRS', 'N': 0, 'MAE': None, 'RMSE': None, 'WAPE': None, 'R2': None}
    error = predicted - fact
    variance = float(np.sum((fact - fact.mean())**2))
    return {'status': 'OK', 'N': len(fact), 'MAE': float(np.abs(error).mean()),
            'RMSE': float(np.sqrt(np.mean(error**2))),
            'WAPE': 100 * float(np.abs(error).sum() / fact.sum()) if fact.sum() > 0 else None,
            'R2': 1 - float(np.sum(error**2)) / variance if len(fact) > 1 and variance > 0 else None}


def _failed_context(prepared, training, error, *, validation=False):
    """Создает контекст отказа с причиной и пропусками вместо недоступных прогнозов."""
    training = list(training)
    ids = prepared['ids'][:prepared['n_existing']] if validation else prepared['ids']
    reason = str(error) or type(error).__name__
    hours = {}
    for hour in prepared['config']['hours']:
        hours[hour] = {
            'forecast': {'status': 'CONTEXT_FAILED', 'matrix': None, 'reason': reason},
            'base': {'status': 'NOT_REQUIRED' if prepared['config']['recalculation_mode'] == 'preserve_old' else 'CONTEXT_FAILED',
                     'matrix': None, 'reason': reason},
            'assembly': {'status': 'UNAVAILABLE', 'matrix': None, 'reason': reason},
        }
    return {'ids': ids, 'training': training, 'error': reason, 'hours': hours,
            'generation_model': None, 'generation': {'forecast': {
                'raw': np.full((2, 19, len(ids)), np.nan), 'goals': np.full((2, 19, len(ids)), np.nan),
                'adjusted_goals': np.full((2, 19, len(ids)), np.nan),
            }}, 'los': None, 'zero_observed_day': None}


def validate_generation(prepared, *, scheme='random', log=logger):
    """Отдельно проверяет сырую генерацию полных итогов, без OD, LOS и IPF.

    Обучающие метки районов R включают связи с H. Это допустимо только в данном
    протоколе проверки районных итогов; в run_context эти метки не передаются.
    """
    config, n = prepared['config'], prepared['n_existing']
    k = config['validation_folds']
    if n < k:
        return {'protocol': 'full_marginals', 'scheme': scheme, 'status': 'VALIDATION_UNAVAILABLE',
                'reason': f'Районов {n}, требуется не меньше {k}',
                'expected_district_hours': n * 19, 'successful_district_hours': 0, 'tables': {}}
    observed = prepared['observed']
    targets = np.stack((observed.sum(axis=2), observed.sum(axis=1)))
    features = prepared['base'][prepared['feature_columns']].to_numpy(dtype=float)
    periods = [(hour, [h]) for h, hour in enumerate(HOUR_COLUMNS)] + [('day', list(range(19)))]
    predictions, coverage, statuses, diagnostics, trials = [], [], [], [], []
    for fold, (training, hidden) in enumerate(validation_splits(prepared, scheme), 1):
        rr = observed[:, training][:, :, training]
        rr_targets = np.stack((rr.sum(axis=2), rr.sum(axis=1)))
        full_targets = targets[:, :, training]
        # Полные итоги ниже — только диагностика усечения обучающих меток OD-протокола.
        for hour, hs in periods:
            full, retained = full_targets[:, hs].sum(axis=1), rr_targets[:, hs].sum(axis=1)
            for direction, label in enumerate(('outgoing', 'incoming')):
                for j, i in enumerate(training):
                    total, value = full[direction, j], retained[direction, j]
                    coverage.append({'fold_id': fold, 'hour': hour, 'district_id': prepared['ids'][i],
                                     'direction': label, 'full_target': total, 'rr_target': value,
                                     'excluded_target': total - value,
                                     'retained_share': value / total if total > 0 else None})
        reason, status, fit_messages = '', 'OK', []
        raw = np.full((2, 19, len(hidden)), np.nan)
        try:
            with warnings.catch_warnings(record=True) as fit_warnings:
                warnings.simplefilter('always')
                model = _train_generation(features[training], full_targets, config['generation_alpha'])
            diagnostics.append({'fold_id': fold, **model['diagnostics']})
            trials.extend({'fold_id': fold, **row} for row in model['alpha_trials'])
            for item in fit_warnings:
                fit_messages.append(f'{item.category.__name__}: {item.message}')
                warnings.warn(item.message, item.category)
            if fit_messages:
                status = 'GENERATION_FIT_WARNING'
            raw = predict_generation(model, features[hidden])['raw']
            if not np.isfinite(raw).all():
                raise FloatingPointError('Non-finite generation predictions')
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            log.exception('full_marginals fold_%s generation failed', fold)
            reason, status = str(error) or type(error).__name__, 'GENERATION_FIT_FAILED'
            raw[:] = np.nan
        statuses.append({'fold_id': fold, 'protocol': 'full_marginals', 'status': status, 'reason': reason,
                         'warnings': fit_messages,
                         'training_districts': len(training), 'hidden_districts': len(hidden)})
        for hour, hs in periods:
            fact = targets[:, hs].sum(axis=1)
            predicted = raw[:, hs].sum(axis=1)
            for j, i in enumerate(hidden):
                fo, fd = fact[:, i]
                po, pd_ = predicted[:, j]
                predictions.append({'fold_id': fold, 'protocol': 'full_marginals', 'hour': hour,
                                    'district_id': prepared['ids'][i], 'stage': 'raw', 'status': status,
                                    'fact_outgoing': fo, 'fact_incoming': fd, 'fact_net': fo - fd,
                                    'predicted_outgoing': po, 'predicted_incoming': pd_, 'predicted_net': po - pd_})
    district = pd.DataFrame(predictions)
    metrics = []
    for fold in [0, *range(1, k + 1)]:
        subset = district if fold == 0 else district.loc[district.fold_id.eq(fold)]
        for hour, rows in subset.groupby('hour'):
            for direction in ('outgoing', 'incoming', 'net'):
                valid = rows.dropna(subset=[f'predicted_{direction}'])
                count = len(valid)
                errors = regression_metrics(valid[f'fact_{direction}'].to_numpy(),
                                            valid[f'predicted_{direction}'].to_numpy())
                if direction == 'net':
                    errors['WAPE'] = None  # Сальдо знакопеременно, сумма не является объемом.
                metrics.append({'fold_id': fold, 'protocol': 'full_marginals', 'hour': hour, 'stage': 'raw',
                                'direction': direction, 'expected': len(rows),
                                'coverage_status': 'VALIDATION_COMPLETE' if count == len(rows) else (
                                    'VALIDATION_PARTIAL' if count else 'VALIDATION_UNAVAILABLE'),
                                **errors})
    coverage = pd.DataFrame(coverage)
    summary = pd.concat([coverage, coverage.assign(fold_id=0)], ignore_index=True).groupby(
        ['fold_id', 'hour', 'direction'], as_index=False)[['full_target', 'rr_target', 'excluded_target']].sum()
    summary['retained_share'] = summary.rr_target / summary.full_target.where(summary.full_target > 0)
    hourly = district.loc[district.hour.ne('day')]
    completed = int(hourly[['predicted_outgoing', 'predicted_incoming']].notna().all(axis=1).sum())
    for table in (district, coverage, summary):
        table['validation_scheme'] = scheme
    return {'protocol': 'full_marginals', 'scheme': scheme,
            'status': 'VALIDATION_COMPLETE' if completed == len(hourly) else (
                'VALIDATION_PARTIAL' if completed else 'VALIDATION_UNAVAILABLE'),
            'reason': '', 'expected_district_hours': len(hourly), 'successful_district_hours': completed,
            'warning_folds': sum(item['status'] == 'GENERATION_FIT_WARNING' for item in statuses),
            'tables': {'generation_predictions': district, 'generation_metrics': pd.DataFrame(metrics),
                       'generation_fold_statuses': pd.DataFrame(statuses),
                       'generation_fit_diagnostics': pd.DataFrame(diagnostics),
                       'generation_fit_alpha_trials': pd.DataFrame(trials),
                       'training_coverage': coverage, 'training_coverage_summary': summary}}


def _benchmark_comparison(predictions, benchmark, expectation, k):
    """Compare all available predictions and the identical available pairs separately."""
    records = []
    keys = ['fold_id', 'hour', 'source', 'target']
    paired = predictions[keys + ['predicted']].merge(
        benchmark, on=keys, suffixes=('_model', '_benchmark'), validate='one_to_one')
    for fold in [0, *range(1, k + 1)]:
        planned = expectation if fold == 0 else expectation.loc[expectation.fold_id.eq(fold)]
        actual = predictions if fold == 0 else predictions.loc[predictions.fold_id.eq(fold)]
        naive = benchmark if fold == 0 else benchmark.loc[benchmark.fold_id.eq(fold)]
        common = paired if fold == 0 else paired.loc[paired.fold_id.eq(fold)]
        # Partition once rather than rescan every OD record for each stratum.
        partitions = [dict(tuple(table.groupby(['hour', 'group'], sort=False)))
                      for table in (actual, naive, common)]
        empty = [table.iloc[:0] for table in (actual, naive, common)]
        for (hour, group, analogue_group), count in planned.groupby(
                ['hour', 'group', 'analogue_group'])['expected'].sum().items():
            def select(index):
                rows = partitions[index].get((hour, group), empty[index])
                return rows if analogue_group == 'all' else rows.loc[rows.analogue_group.eq(analogue_group)]
            row = {'fold_id': fold, 'hour': hour, 'group': group, 'analogue_group': analogue_group,
                   'expected': int(count)}
            for index, prefix in enumerate(('model', 'benchmark')):
                valid = select(index).dropna(subset=['predicted'])
                row.update({f'{prefix}_{key}': value for key, value in regression_metrics(
                    valid.fact.to_numpy(), valid.predicted.to_numpy()).items()})
                row[f'{prefix}_coverage_status'] = ('NO_TEST_PAIRS' if count == 0 else 'VALIDATION_COMPLETE'
                    if len(valid) == count else 'VALIDATION_PARTIAL' if len(valid) else 'VALIDATION_UNAVAILABLE')
            valid = select(2).dropna(subset=['predicted_model', 'predicted_benchmark'])
            row['common_N'] = len(valid)
            row['common_coverage_status'] = ('NO_TEST_PAIRS' if count == 0 else 'VALIDATION_COMPLETE'
                if len(valid) == count else 'VALIDATION_PARTIAL' if len(valid) else 'VALIDATION_UNAVAILABLE')
            for name in ('model', 'benchmark'):
                row.update({f'common_{name}_{key}': value for key, value in regression_metrics(
                    valid.fact.to_numpy(), valid[f'predicted_{name}'].to_numpy()).items()})
            a, b = row['common_model_WAPE'], row['common_benchmark_WAPE']
            row['common_WAPE_gain_pp'] = b - a if a is not None and b is not None else None
            records.append(row)
    return pd.DataFrame(records)


def validate_transfer(prepared, *, scheme='random', context_sink=None, log=logger):
    """Проверяет перенос методом K-fold по районам с полным переобучением фолдов.

    Учитывает все запланированные проверочные пары в покрытии. При переданном
    context_sink сохраняет контексты через обработчик, не накапливая их матрицы.
    """
    config, n = prepared['config'], prepared['n_existing']
    assignments, predictions, district_rows, statuses, los_rows, expected = [], [], [], [], [], []
    benchmark_predictions, benchmark_donors, benchmark_expected = [], [], []
    contexts = {}
    k = config['validation_folds']
    log.info('validate_transfer started; folds=%s; districts=%s', k, n)
    if n < k:
        log.warning('validate_transfer unavailable; districts=%s; required=%s', n, k)
        return {'protocol': 'territory_expansion', 'scheme': scheme, 'status': 'VALIDATION_UNAVAILABLE',
                'reason': f'Районов {n}, требуется не меньше {k}',
                'contexts': {}, 'tables': {}, 'successful_forecast_hours': 0,
                'expected_forecast_hours': k * len(config['hours']), 'successful_assembly_hours': 0,
                'generation_validation': {key: value for key, value in validate_generation(prepared, scheme=scheme, log=log).items()
                                          if key != 'tables'}}
    generation_validation = validate_generation(prepared, scheme=scheme, log=log)
    # Разбиение фиксировано по ID районов и не зависит от потоков или успеха расчета.
    keys = np.asarray(prepared['ids'][:n], dtype=object)
    for fold, (training, hidden_array) in enumerate(validation_splits(prepared, scheme), 1):
        hidden = np.isin(np.arange(n), hidden_array)
        # Тестом служат все пары хотя бы с одним скрытым концом, включая нулевой факт.
        testing = hidden[:, None] | hidden[None, :]
        if not config['include_intrazonal']:
            np.fill_diagonal(testing, False)
        test_i, test_j = np.where(testing)
        groups = np.where(hidden[test_i] & hidden[test_j], 'new_new',
                          np.where(hidden[test_i], 'new_existing', 'existing_new'))
        counts = {group: int(np.count_nonzero(groups == group))
                  for group in ('new_existing', 'existing_new', 'new_new')}
        baseline = nearest_analogue_baseline(prepared, training)
        benchmark_donors.append(baseline['diagnostics'].assign(fold_id=fold))
        missing_analogue = hidden & (baseline['analogue_counts'] == 0)
        analogue_groups = np.where(missing_analogue[test_i] | missing_analogue[test_j],
                                   'without_analogue', 'with_analogue')
        assignments.extend({'district_id': prepared['ids'][i], 'fold_id': fold} for i in hidden_array)
        try:
            label = f'fold_{fold}' if scheme == 'random' else f'spatial_fold_{fold}'
            context = run_context(prepared, training, validation=True, log=log, context=label)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            log.exception('fold_%s run_context failed', fold)
            context = _failed_context(prepared, training, error, validation=True)
        if context_sink is None:
            contexts[str(fold)] = context

        def record_predictions(hour, fact, forecast):
            """Записывает доступные прогнозы проверочных OD-пар текущего фолда."""
            hs = list(range(19)) if hour == 'day' else [HOUR_COLUMNS.index(hour)]
            naive = baseline_matrix(prepared, baseline, hs)
            benchmark_predictions.append(pd.DataFrame({
                'fold_id': fold, 'hour': hour, 'source': keys[test_i], 'target': keys[test_j],
                'group': groups, 'analogue_group': analogue_groups,
                'fact': fact[test_i, test_j], 'predicted': naive[test_i, test_j]}))
            for group, count in counts.items():
                for category in ('all', 'with_analogue', 'without_analogue'):
                    number = count if category == 'all' else np.count_nonzero((groups == group) & (analogue_groups == category))
                    benchmark_expected.append({'fold_id': fold, 'hour': hour, 'group': group,
                                               'analogue_group': category, 'expected': int(number)})
            if forecast is not None:
                predictions.append(pd.DataFrame({
                    'fold_id': fold, 'protocol': 'territory_expansion', 'hour': hour,
                    'source': keys[test_i], 'target': keys[test_j], 'group': groups,
                    'analogue_group': analogue_groups,
                    'fact': fact[test_i, test_j], 'predicted': forecast[test_i, test_j],
                    'observed_allowed': np.isfinite(prepared['observed_los'][test_i, test_j]),
                    'predicted_allowed': context['los']['mask'][test_i, test_j],
                }))

        for hour in config['hours']:
            h = HOUR_COLUMNS.index(hour)
            state = context['hours'][hour]
            forecast = state['forecast']['matrix']
            statuses.append({'fold_id': fold, 'hour': hour, 'M_forecast': state['forecast']['status'],
                             'M_base': state['base']['status'], 'T_final': state['assembly']['status'],
                             'reason': context.get('error', ''),
                             **{key: context['generation']['forecast'].get('reconciliation', [{}] * 19)[h].get(key)
                                for key in ('status', 'Q_model', 'Q_feasible', 'Q_reduced', 'retained_share')},
                             'Q_C': state['assembly'].get('Q_C'),
                             'Q_final': float(state['assembly']['matrix'].sum()) if state['assembly']['status'] == 'OK' else None,
                             'limited_count': state['assembly'].get('limited_count'),
                             'machine_count': state['assembly'].get('machine_count'),
                             'machine_volume': state['assembly'].get('machine_volume'),
                             'pivot_limited': 'PIVOT_FLOW_LIMITED' in state['assembly'].get('warnings', [])})
            record_predictions(hour, prepared['observed'][h], forecast)
            expected.extend({'fold_id': fold, 'hour': hour, 'group': group, 'expected': count}
                            for group, count in counts.items())
        if len(config['hours']) == 19:
            complete = all(item['forecast']['status'] == 'OK' for item in context['hours'].values())
            daily = sum(item['forecast']['matrix'] for item in context['hours'].values()) if complete else None
            fact_day = prepared['observed'].sum(axis=0)
            record_predictions('day', fact_day, daily)
            expected.extend({'fold_id': fold, 'hour': 'day', 'group': group, 'expected': count}
                            for group, count in counts.items())
        # Районные ошибки разделяем на исходную генерацию, цели IPF и достигнутые итоги OD.
        periods = [(hour, [HOUR_COLUMNS.index(hour)]) for hour in HOUR_COLUMNS] + [('day', list(range(19)))]
        for hour, hs in periods:
            fact = prepared['observed'][hs].sum(axis=0)
            forecast = None
            if all(HOUR_COLUMNS[h] in context['hours'] and context['hours'][HOUR_COLUMNS[h]]['forecast']['status'] == 'OK' for h in hs):
                forecast = sum(context['hours'][HOUR_COLUMNS[h]]['forecast']['matrix'] for h in hs)
            generation = context['generation']['forecast']
            for i in hidden_array:
                fo, fd = fact[i].sum(), fact[:, i].sum()
                for stage in ('raw', 'goals', 'adjusted', 'od'):
                    if stage == 'od' and (hour not in config['hours'] and not (hour == 'day' and len(config['hours']) == 19)):
                        continue
                    if stage == 'od':
                        po, pd_ = (None, None) if forecast is None else (forecast[i].sum(), forecast[:, i].sum())
                    else:
                        values = generation['adjusted_goals' if stage == 'adjusted' else stage][:, hs, i]
                        po, pd_ = values.sum(axis=1) if np.isfinite(values).all() else (None, None)
                    district_rows.append({'fold_id': fold, 'protocol': 'territory_expansion',
                                          'hour': hour, 'district_id': prepared['ids'][i],
                                          'analogue_count': int(baseline['analogue_counts'][i]),
                                          'stage': stage, 'fact_outgoing': fo, 'fact_incoming': fd,
                                          'predicted_outgoing': po, 'predicted_incoming': pd_,
                                          'fact_net': fo - fd, 'predicted_net': None if po is None else po - pd_,
                                          'fact_day_volume': prepared['observed'][:, i].sum(),
                                          'fact_intrazonal_share': fact[i, i] / fo if fo > 0 and config['include_intrazonal'] else None,
                                          'predicted_intrazonal_share': forecast[i, i] / po if stage == 'od' and po is not None and po > 0 and config['include_intrazonal'] else None,
                                          'diagonal_status': 'INCLUDED' if config['include_intrazonal'] else 'EXCLUDED'})
        # Качество LOS проверяем и при отказе спроса; в ошибки входят только доступные оценки.
        i, j = np.where(testing & np.isfinite(prepared['observed_los']))
        predicted_los = np.full(len(i), np.nan) if context['los'] is None else np.where(
            context['los']['mask'][i, j], context['los']['times'][i, j], np.nan)
        methods = (np.full(len(i), 'unavailable', dtype=object) if context['los'] is None else
                   context['los']['pairs'].set_index(['source', 'target']).time_method.reindex(
                       pd.MultiIndex.from_arrays([keys[i], keys[j]])).fillna('unavailable').to_numpy())
        los_rows.append(pd.DataFrame({
            'fold_id': fold, 'source': keys[i], 'target': keys[j],
            'length_group': np.where(prepared['distance'][i, j] < 1.5, 'short', 'long'),
            'fact': prepared['observed_los'][i, j],
            'predicted': predicted_los,
            'time_method': methods,
        }))
        # После сохранения диагностики матрицы фолда можно освободить до следующего обучения.
        if context_sink is not None:
            context_sink(str(fold), context)
        del context
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(
        columns=['fold_id', 'protocol', 'hour', 'source', 'target', 'group', 'analogue_group', 'fact',
                 'predicted', 'observed_allowed', 'predicted_allowed'])
    expectation = pd.DataFrame(expected)
    metrics = []
    # fold_id=0 объединяет проверочные записи, а не усредняет готовые метрики фолдов.
    for fold in [0, *range(1, k + 1)]:
        subset = pred if fold == 0 else pred.loc[pred.fold_id.eq(fold)]
        coverage = expectation if fold == 0 else expectation.loc[expectation.fold_id.eq(fold)]
        for (hour, group), count in coverage.groupby(['hour', 'group'])['expected'].sum().items():
            rows = subset.loc[subset.hour.eq(hour) & subset.group.eq(group)]
            fact, model = rows.fact.to_numpy(dtype=float), rows.predicted.to_numpy(dtype=float)
            metrics.append({'fold_id': fold, 'protocol': 'territory_expansion',
                            'hour': hour, 'group': group, 'expected': int(count),
                            **regression_metrics(fact, model),
                            'coverage_status': 'VALIDATION_COMPLETE' if len(rows) == count else ('VALIDATION_PARTIAL' if len(rows) else 'VALIDATION_UNAVAILABLE'),
                            'false_allowed_flow_share': float(rows.loc[~rows.observed_allowed.astype(bool), 'predicted'].sum() / model.sum()) if model.sum() > 0 else None,
                            'lost_fact_flow_share': float(rows.loc[~rows.predicted_allowed.astype(bool), 'fact'].sum() / fact.sum()) if fact.sum() > 0 else None})
    # Районные ошибки и покрытие считаем отдельно для каждой стадии расчета генерации.
    district = pd.DataFrame(district_rows)
    district_metrics = []
    for (hour, stage), rows in district.groupby(['hour', 'stage']):
        for direction in ('outgoing', 'incoming', 'net'):
            valid = rows.dropna(subset=[f'predicted_{direction}'])
            district_metrics.append({'protocol': 'territory_expansion', 'hour': hour, 'stage': stage, 'direction': direction,
                                     'expected': len(rows), **regression_metrics(
                                         valid[f'fact_{direction}'].to_numpy(), valid[f'predicted_{direction}'].to_numpy())})
        if stage == 'od':
            valid = rows.dropna(subset=['fact_intrazonal_share', 'predicted_intrazonal_share'])
            district_metrics.append({'protocol': 'territory_expansion', 'hour': hour, 'stage': stage, 'direction': 'intrazonal_share',
                                     'expected': len(rows), **regression_metrics(
                                         valid.fact_intrazonal_share.to_numpy(), valid.predicted_intrazonal_share.to_numpy())})
    # Короткие и длинные LOS оцениваем раздельно, сохраняя полный ожидаемый объем проверки.
    los_frame = pd.concat(los_rows, ignore_index=True)
    los_metrics = []
    for group in ('short', 'long'):
        rows = los_frame.loc[los_frame.length_group.eq(group)]
        for method, subset in [('all', rows), *rows.groupby('time_method')]:
            valid = subset.dropna(subset=['predicted'])
            error = valid.predicted - valid.fact
            los_metrics.append({'length_group': group, 'time_method': method, 'expected': len(subset),
                                **regression_metrics(valid.fact.to_numpy(), valid.predicted.to_numpy()),
                                'mean_error_minutes': float(error.mean() * 60) if len(valid) else None,
                                'relative_bias_pct': float(error.sum() / valid.fact.sum() * 100)
                                if len(valid) and valid.fact.sum() > 0 else None})
    status_frame = pd.DataFrame(statuses)
    completed = int(status_frame.M_forecast.eq('OK').sum())
    planned = k * len(config['hours'])
    benchmark = pd.concat(benchmark_predictions, ignore_index=True)
    comparison = _benchmark_comparison(pred, benchmark, pd.DataFrame(benchmark_expected), k)
    all_cases = comparison.loc[comparison.fold_id.eq(0) & comparison.hour.ne('day') & comparison.analogue_group.eq('all')]
    tables = {**generation_validation['tables'],
              'fold_assignments': pd.DataFrame(assignments), 'test_predictions': pred,
              'metrics': pd.DataFrame(metrics), 'district_predictions': district,
              'district_metrics': pd.DataFrame(district_metrics), 'fold_statuses': status_frame,
              'validation_los': los_frame, 'validation_los_metrics': pd.DataFrame(los_metrics),
              'benchmark_predictions': benchmark, 'benchmark_donors': pd.concat(benchmark_donors, ignore_index=True),
              'benchmark_comparison': comparison}
    for table in tables.values():
        table['validation_scheme'] = scheme
    successful = status_frame.loc[status_frame.T_final.eq('OK')]
    total_correction, total_final = successful.Q_C.sum(), successful.Q_final.sum()
    log.info('validate_transfer finished; successful_forecast_hours=%s; expected_forecast_hours=%s; successful_assembly_hours=%s',
             completed, planned, len(successful))
    return {'protocol': 'territory_expansion', 'scheme': scheme,
            'status': 'VALIDATION_COMPLETE' if completed == planned else ('VALIDATION_PARTIAL' if completed else 'VALIDATION_UNAVAILABLE'),
            'reason': '', 'contexts': contexts, 'successful_forecast_hours': completed,
            'expected_forecast_hours': planned, 'successful_assembly_hours': int(status_frame.T_final.eq('OK').sum()),
            'assembly_success_share': len(successful) / planned,
            'pivot_limited_hours': int(successful.pivot_limited.sum()),
            'pivot_limited_share': float(successful.pivot_limited.mean()) if len(successful) else None,
            'Q_C': float(total_correction),
            'correction_share': float(total_correction / total_final) if total_final > 0 else None,
            'limited_count': int(successful.limited_count.sum()),
            'target_reconciliation': _reconciliation_summary(statuses),
            'generation_validation': {key: value for key, value in generation_validation.items() if key != 'tables'},
            'benchmark': {'expected_pairs': int(all_cases.expected.sum()),
                          'available_pairs': int(all_cases.benchmark_N.sum()),
                          'common_pairs': int(all_cases.common_N.sum()),
                          'without_analogue_districts': int(tables['benchmark_donors'].analogue_count.eq(0).sum()),
                          'hidden_districts': n},
            'acceptance': {'status': 'CRITERIA_NOT_SET'}, 'tables': tables}


def forecast_baseline(prepared, *, context_sink=None, log=logger):
    """Выполняет проверку по районам, затем переобучает основной прогноз на всей базе."""
    log.info('forecast_baseline started; hours=%s', prepared['config']['hours'])
    validation = validate_transfer(prepared, context_sink=context_sink, log=log)
    spatial_sink = (lambda label, context: context_sink(f'spatial_{label}', context)) if context_sink else None
    spatial = validate_transfer(prepared, scheme='spatial', context_sink=spatial_sink, log=log)
    validation['spatial_validation'] = {key: value for key, value in spatial.items() if key not in ('contexts', 'tables')}
    validation['tables'].update({f'spatial_{name}': table for name, table in spatial['tables'].items()})
    validation['contexts'].update({f'spatial_{label}': context for label, context in spatial['contexts'].items()})
    try:
        main = run_context(prepared, range(prepared['n_existing']), log=log)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        log.exception('main run_context failed')
        main = _failed_context(prepared, range(prepared['n_existing']), error)
    # Готовность сценария определяется сборкой; готовность проверки переноса хранится отдельно.
    completed = sum(value['assembly']['status'] == 'OK' for value in main['hours'].values())
    log.info('forecast_baseline finished; successful_assembly_hours=%s; expected_hours=%s; validation=%s',
             completed, len(prepared['config']['hours']), validation['status'])
    return {'config': prepared['config'], 'metric_crs': prepared['metric_crs'],
            'status': 'OK' if completed == len(prepared['config']['hours']) else ('PARTIAL' if completed else 'FAILED'),
            'main': main, 'validation': validation}


def iter_matrix_tables(context, prepared, *, artifacts=None):
    """Выдает таблицы матриц порциями, сохраняя направления, строковые ID и нули."""
    ids = context['ids']
    training = context['training']
    training_ids = [ids[i] for i in training]

    def blocks(matrix, state_ids, hour):
        """Разбивает матрицу на порции из полных строк источников для экономного экспорта."""
        keys = np.asarray(state_ids, dtype=object)
        n = len(keys)
        # Подбираем число строк источников под бюджет 50 000 OD-ячеек, минимум одна строка.
        rows = max(1, 50_000 // n)
        for start in range(0, n, rows):
            stop = min(n, start + rows)
            yield pd.DataFrame({'source': np.repeat(keys[start:stop], n),
                                'target': np.tile(keys, stop - start), 'hour': hour,
                                'trips': matrix[start:stop].reshape(-1)})

    for hour, states in context['hours'].items():
        for state, label in (('forecast', 'M_forecast'), ('base', 'M_base'), ('assembly', 'T_final')):
            if artifacts is not None and label not in artifacts:
                continue
            matrix = states[state]['matrix']
            if matrix is not None:
                state_ids = training_ids if state == 'base' else ids
                for table in blocks(matrix, state_ids, hour):
                    yield label, table
        if artifacts is None or 'B' in artifacts:
            observed = prepared['observed'][HOUR_COLUMNS.index(hour)][np.ix_(training, training)]
            for table in blocks(observed, training_ids, hour):
                yield 'B', table
    # Суточные модельные матрицы выдаем только при успехе каждого из 19 часов состояния.
    if len(prepared['config']['hours']) == 19:
        for state, label in (('forecast', 'M_forecast_day'), ('base', 'M_base_day'), ('assembly', 'T_final_day')):
            if artifacts is not None and label not in artifacts:
                continue
            if all(item[state]['status'] == 'OK' for item in context['hours'].values()):
                daily = sum(item[state]['matrix'] for item in context['hours'].values())
                state_ids = training_ids if state == 'base' else ids
                for table in blocks(daily, state_ids, 'day'):
                    yield label, table


def _reconciliation_summary(items):
    """Volume-weighted reconciliation summary; missing results never become zero."""
    good = [item for item in items if item.get('status') in (
        'TARGETS_UNCHANGED', 'TARGETS_REDUCED', 'ZERO_FEASIBLE_VOLUME')]
    result = {'expected_hours': len(items), 'successful_hours': len(good),
              'complete': len(good) == len(items), 'Q_model': None, 'Q_feasible': None,
              'Q_reduced': None, 'retained_share': None, 'zero_hours': sum(
                  item.get('status') == 'ZERO_FEASIBLE_VOLUME' for item in good)}
    if good:
        for key in ('Q_model', 'Q_feasible', 'Q_reduced'):
            result[key] = float(sum(item[key] for item in good))
        if result['Q_model'] > 0:
            result['retained_share'] = result['Q_feasible'] / result['Q_model']
    result['status'] = ('NOT_REQUIRED' if items and all(item.get('status') == 'NOT_REQUIRED' for item in items)
                        else 'INCOMPLETE' if not result['complete'] or not items
                        else 'ZERO_FEASIBLE_VOLUME' if result['Q_feasible'] == 0
                        else 'TARGETS_REDUCED' if result['Q_reduced'] > 0 else 'TARGETS_UNCHANGED')
    return result


def context_tables(context, prepared, *, include_matrices=True):
    """Формирует отчеты контекста, сохраняя все 19 часов генерации и причины отказов.

    include_matrices=False исключает полные OD-таблицы: сервис пишет их отдельно
    порциями через iter_matrix_tables, чтобы не удерживать весь экспорт в памяти.
    """
    # При общем отказе контекста сохраняем только статусы, не создавая фиктивных прогнозов.
    if 'error' in context:
        return {'statuses': pd.DataFrame([
            {'hour': hour, 'state': label, 'status': states[state]['status'], 'reason': states[state]['reason']}
            for hour, states in context['hours'].items()
            for state, label in (('forecast', 'M_forecast'), ('base', 'M_base'), ('assembly', 'T_final'))
        ], columns=STATUS_COLUMNS)}
    ids, model = context['ids'], context['generation_model']
    training_ids = [ids[i] for i in context['training']]
    generation_rows, coefficient_rows, status_rows = [], [], []
    margins, corrections, target_rows, support_rows = [], [], [], []
    # Коэффициенты в исходной шкале дополняют масштабы и диапазоны обучающих признаков.
    for direction, label in enumerate(('outgoing', 'incoming')):
        for h, hour in enumerate(HOUR_COLUMNS):
            for k, feature in enumerate(model['feature_columns']):
                coefficient_rows.append({'direction': label, 'hour': hour, 'feature': feature,
                                         'coefficient_scaled': model['coefficients'][direction, h, k],
                                         'coefficient_original': model['coefficients'][direction, h, k] / model['scale'][k],
                                         'scale': model['scale'][k], 'active': model['active'][k],
                                         'training_min': model['min'][k], 'training_max': model['max'][k]})
    # Генерация содержит все 19 часов; суточные цели получаем их суммой, без новой модели.
    for state, generation in context['generation'].items():
        state_ids = ids if state == 'forecast' else training_ids
        for h, hour in enumerate(HOUR_COLUMNS):
            target_rows.append({'state': state, 'hour': hour, **{
                key: value for key, value in generation['reconciliation'][h].items()
                if key not in ('outgoing', 'incoming', 'R_O', 'R_D')},
                **{key: [state_ids[i] for i in generation['reconciliation'][h].get(key, [])]
                   for key in ('isolated_outgoing', 'isolated_incoming')},
                'cut_certificates': [{**certificate,
                    'rows': [state_ids[i] for i in certificate['rows']],
                    'columns': [state_ids[i] for i in certificate['columns']],
                    'component_rows': [state_ids[i] for i in certificate['component_rows']],
                    'component_columns': [state_ids[i] for i in certificate['component_columns']]}
                    for certificate in generation['reconciliation'][h].get('cut_certificates', [])]})
        target_rows.append({'state': state, 'hour': 'day',
                            **_reconciliation_summary(generation['reconciliation'])})
        for i, key in enumerate(state_ids):
            for h, hour in enumerate(HOUR_COLUMNS):
                generation_rows.append({'state': state, 'district_id': key, 'hour': hour,
                                        'O_hat': generation['raw'][0, h, i], 'D_hat': generation['raw'][1, h, i],
                                        'O_star': generation['goals'][0, h, i], 'D_star': generation['goals'][1, h, i],
                                        'O_adj': generation['adjusted_goals'][0, h, i],
                                        'D_adj': generation['adjusted_goals'][1, h, i],
                                        'R_O': generation['goals'][0, h, i] - generation['adjusted_goals'][0, h, i],
                                        'R_D': generation['goals'][1, h, i] - generation['adjusted_goals'][1, h, i],
                                        'target_status': generation['reconciliation'][h]['status'],
                                        'S_O': generation['totals'][0, h], 'S_D': generation['totals'][1, h],
                                        'factor_O': generation['factors'][0, h], 'factor_D': generation['factors'][1, h],
                                        'status': generation['statuses'][h]})
            goals = generation['goals'][:, :, i]
            adjusted = generation['adjusted_goals'][:, :, i]
            generation_rows.append({'state': state, 'district_id': key, 'hour': 'day',
                                    'O_hat': generation['raw'][0, :, i].sum(), 'D_hat': generation['raw'][1, :, i].sum(),
                                    'O_star': goals[0].sum() if np.isfinite(goals).all() else None,
                                    'D_star': goals[1].sum() if np.isfinite(goals).all() else None,
                                    'O_adj': adjusted[0].sum() if np.isfinite(adjusted).all() else None,
                                    'D_adj': adjusted[1].sum() if np.isfinite(adjusted).all() else None,
                                    'R_O': (goals[0] - adjusted[0]).sum() if np.isfinite(adjusted).all() else None,
                                    'R_D': (goals[1] - adjusted[1]).sum() if np.isfinite(adjusted).all() else None,
                                    'status': 'OK' if np.isfinite(goals).all() else 'INCOMPLETE'})
    # По рассчитанным состояниям сохраняем невязки, районные итоги и ограниченную поправку.
    for hour, states in context['hours'].items():
        for state, label in (('forecast', 'M_forecast'), ('base', 'M_base'), ('assembly', 'T_final')):
            result = states[state]
            status_rows.append({'hour': hour, 'state': label, 'status': result['status'],
                                'iterations': result.get('iterations'), 'residual': result.get('residual'),
                                'reason': result.get('reason'), 'Q_C': result.get('Q_C'),
                                'correction_share': result.get('correction_share'), 'limited_count': result.get('limited_count'),
                                'machine_count': result.get('machine_count'), 'machine_volume': result.get('machine_volume'),
                                'control_error': result.get('control_error'),
                                'Q_final': float(result['matrix'].sum()) if result['matrix'] is not None else None,
                                'warnings': result.get('warnings', []),
                                'target_status': result.get('target_status'),
                                'support_status': result.get('support_status')})
            state_ids = training_ids if state == 'base' else ids
            for item in result.get('boundary', []):
                support_rows.append({'hour': hour, 'state': label,
                    'source': state_ids[item['source']], 'target': state_ids[item['target']],
                    'reason': item['reason']})
            matrix = result['matrix']
            if matrix is None:
                continue
            state_ids = training_ids if state == 'base' else ids
            for i, key in enumerate(state_ids):
                outgoing, incoming = matrix[i].sum(), matrix[:, i].sum()
                margins.append({'hour': hour, 'state': label, 'district_id': key, 'outgoing': outgoing,
                                'incoming': incoming, 'net': outgoing - incoming,
                                'intrazonal_share': matrix[i, i] / outgoing if outgoing > 0 and prepared['config']['include_intrazonal'] else None,
                                'diagonal_status': 'INCLUDED' if prepared['config']['include_intrazonal'] else 'EXCLUDED',
                                'O_C': result['O_C'][i] if state == 'assembly' else None,
                                'D_C': result['D_C'][i] if state == 'assembly' else None})
            if state == 'assembly':
                i, j = np.where(result['correction'] > 0)
                if len(i):
                    keys = np.asarray(ids, dtype=object)
                    corrections.append(pd.DataFrame({'source': keys[i], 'target': keys[j], 'hour': hour,
                                                     'U': result['unbounded'][i, j], 'C': result['correction'][i, j]}))
    # Суточные контроли допустимы только для полного набора успешных часовых матриц.
    daily_controls = []
    if len(prepared['config']['hours']) == 19:
        for state, label in (('forecast', 'M_forecast'), ('base', 'M_base'), ('assembly', 'T_final')):
            if not all(item[state]['status'] == 'OK' for item in context['hours'].values()):
                daily_controls.append({'state': label, 'status': 'UNAVAILABLE'})
                continue
            daily = sum(item[state]['matrix'] for item in context['hours'].values())
            state_ids = training_ids if state == 'base' else ids
            if state in ('forecast', 'base'):
                goals = context['generation'][state]['adjusted_goals'].sum(axis=1)
                residual = max(np.max(np.abs(daily.sum(axis=1) - goals[0]) / np.maximum(goals[0], 1)),
                               np.max(np.abs(daily.sum(axis=0) - goals[1]) / np.maximum(goals[1], 1)))
            else:
                residual = None
            correction = sum(item['assembly']['Q_C'] for item in context['hours'].values()) if state == 'assembly' else None
            daily_controls.append({'state': label, 'status': 'OK', 'residual': residual,
                                   'total': float(daily.sum()), 'Q_C': correction,
                                   'correction_share': correction / float(daily.sum()) if correction is not None and daily.sum() > 0 else None})
            for i, key in enumerate(state_ids):
                margins.append({'hour': 'day', 'state': label, 'district_id': key,
                                'outgoing': daily[i].sum(), 'incoming': daily[:, i].sum(),
                                'net': daily[i].sum() - daily[:, i].sum(),
                                'intrazonal_share': daily[i, i] / daily[i].sum() if daily[i].sum() > 0 and prepared['config']['include_intrazonal'] else None,
                                'diagonal_status': 'INCLUDED' if prepared['config']['include_intrazonal'] else 'EXCLUDED',
                                'O_C': sum(item['assembly']['O_C'][i] for item in context['hours'].values()) if state == 'assembly' else None,
                                'D_C': sum(item['assembly']['D_C'][i] for item in context['hours'].values()) if state == 'assembly' else None})
    training_rows = [{'district_id': key, 'hour': hour, 'outgoing': model['training_targets'][0, i, h],
                      'incoming': model['training_targets'][1, i, h]} for i, key in enumerate(training_ids)
                     for h, hour in enumerate(HOUR_COLUMNS)]
    tables = {}
    # Полные таблицы нужны прямому вызову API; файловый сервис использует потоковый экспорт.
    if include_matrices:
        matrices = {name: [] for name in ('M_forecast', 'M_base', 'T_final', 'B',
                                         'M_forecast_day', 'M_base_day', 'T_final_day')}
        for name, table in iter_matrix_tables(context, prepared):
            matrices[name].append(table)
        tables = {name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=['source', 'target', 'hour', 'trips']) for name, parts in matrices.items()}
    # Исходную наблюдаемую маску показываем рядом с расчетной для проверки изменения доступности.
    los_pairs = context['los']['pairs'].copy()
    positions = {key: i for i, key in enumerate(prepared['ids'])}
    n_existing = prepared['n_existing']
    los_pairs['observed_allowed'] = [
        positions[source] < n_existing and positions[target] < n_existing
        and bool(np.isfinite(prepared['observed_los'][positions[source], positions[target]]))
        for source, target in zip(los_pairs.source, los_pairs.target)
    ]
    delta_summary, delta_districts = reconciliation_delta_tables(context, prepared)
    correction_summary, correction_districts = correction_tables(context, prepared)
    tables.update(target_reconciliation=pd.DataFrame(target_rows),
                  correction_diagnostics=correction_summary, correction_districts=correction_districts,
                  reconciliation_delta=delta_summary, reconciliation_delta_districts=delta_districts,
                  context_timings=pd.DataFrame([context['timings']]),
                  ipf_support=pd.DataFrame(support_rows, columns=['hour', 'state', 'source', 'target', 'reason']),
                  generation_diagnostics=pd.DataFrame([model['diagnostics']]),
                  generation_alpha_trials=pd.DataFrame(model['alpha_trials']),
                  generation=pd.DataFrame(generation_rows), coefficients=pd.DataFrame(coefficient_rows),
                  statuses=pd.DataFrame(status_rows, columns=STATUS_COLUMNS), margins=pd.DataFrame(margins),
                  corrections=pd.concat(corrections, ignore_index=True) if corrections else pd.DataFrame(
                      columns=['source', 'target', 'hour', 'U', 'C']),
                  training_targets=pd.DataFrame(training_rows), los=los_pairs,
                  los_preprocessing=context['los']['preprocessing'],
                  los_donors=context['los']['donors'],
                  daily_controls=pd.DataFrame(daily_controls, columns=['state', 'status', 'residual', 'total', 'Q_C', 'correction_share']),
                  calibration=pd.DataFrame([{'hour': hour, **{key: value for key, value in states['calibration'].items()
                                                               if key != 'trials'}}
                                            for hour, states in context['hours'].items()]),
                  calibration_trials=pd.DataFrame([{'hour': hour, **trial}
                                                   for hour, states in context['hours'].items()
                                                   for trial in states['calibration']['trials']],
                                                  columns=CALIBRATION_TRIAL_COLUMNS))
    return tables
