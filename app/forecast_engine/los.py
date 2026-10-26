"""District geometry, directed LOS and donor transfer for a single data context."""
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .inputs import _matrix_array, _validate_district_geometry



def _district_geodata(districts):
    """Строит метрический слой районов, площади в км² и центроиды.
    Проекция UTM определяется по геометрии; ее обозначение возвращается для отчета.
    """
    unique = districts.drop_duplicates('id_tr_distr')
    geometry = _validate_district_geometry(unique)
    geographic = gpd.GeoDataFrame(
        unique[['id_tr_distr']].copy(), geometry=geometry, crs='EPSG:4326'
    )
    # Переводим угловые координаты в метры перед вычислением площадей и расстояний.
    metric_crs = geographic.estimate_utm_crs()
    if metric_crs is None:
        raise ValueError('Не удалось определить метрическую проекцию районов')
    metric = geographic.to_crs(metric_crs)
    metric['area_km2'] = metric.geometry.area / 1_000_000
    metric['centroid'] = metric.geometry.centroid
    return metric.set_index('id_tr_distr'), str(metric_crs)


def _pair_distances(geodata):
    """Возвращает центроидные расстояния в километрах для всех направленных пар.
    Диагональ исключается; расстояние симметрично, но последующий LOS может различаться.
    """
    ids = geodata.index.to_numpy()
    x = np.asarray([point.x for point in geodata['centroid']], dtype=float)
    y = np.asarray([point.y for point in geodata['centroid']], dtype=float)
    distances = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) / 1000
    # Сохраняем обе ориентации каждой связи; внутрирайонную диагональ не моделируем.
    i, j = np.where(~np.eye(len(ids), dtype=bool))
    return pd.DataFrame({'source': ids[i], 'target': ids[j],
                         'distance_km': distances[i, j]})


def _fit_time_function(day_los, pair_distances, old_ids):
    """Оценивает время как неотрицательные свободный член и наклон по расстоянию.
    Использует пригодные наблюдения только между переданными старыми районами.
    Возвращает параметры, ошибку в минутах и диапазон обучающих расстояний.
    """
    observed = day_los.loc[day_los['los_valid']].merge(
        pair_distances,
        left_on=['agg_source', 'agg_target'], right_on=['source', 'target'],
        how='inner',
    )
    observed = observed.loc[
        observed['agg_source'].isin(old_ids) & observed['agg_target'].isin(old_ids)
        & (observed['agg_source'] != observed['agg_target'])
    ]
    if observed.empty:
        raise ValueError('Нет пригодных межрайонных LOS для оценки времени')
    # NNLS ограничивает свободный член и наклон нулем снизу; это резервный перенос LOS.
    design = np.column_stack([np.ones(len(observed)), observed['distance_km']])
    params, _ = nnls(design, observed['los_by_carry'].to_numpy(dtype=float))
    predicted = design @ params
    error_minutes = np.abs(predicted - observed['los_by_carry']).mean() * 60
    return float(params[0]), float(params[1]), {
        'los_observations': len(observed), 'time_intercept_hours': float(params[0]),
        'time_slope_hours_per_km': float(params[1]),
        'time_mae_minutes': float(error_minutes),
        'distance_min_km': float(observed['distance_km'].min()),
        'distance_max_km': float(observed['distance_km'].max()),
    }


def _build_pair_los(pair_distances, day_los, state, old_ids, config):
    """Формирует направленный LOS из наблюдений, ближайших доноров и резервной функции.
    Радиус и число доноров ограничивают перенос времени. Явные запреты сохраняются.
    Возвращает LOS всех пар, использованных доноров, отбор кандидатов и модель времени.
    """
    intercept, slope, fit = _fit_time_function(day_los, pair_distances, old_ids)
    distance_lookup = pair_distances.set_index(
        ['source', 'target']
    )['distance_km'].to_dict()
    valid = day_los.loc[day_los['los_valid']].set_index(
        ['agg_source', 'agg_target']
    )['los_by_carry']
    valid_map = valid.to_dict()
    forbidden = set(day_los.loc[day_los['is_forbidden'], [
        'agg_source', 'agg_target'
    ]].itertuples(index=False, name=None))
    new_ids = set(state.index) - set(old_ids)
    old_with_los = set(day_los.loc[day_los['los_valid'], 'agg_source']) | set(
        day_los.loc[day_los['los_valid'], 'agg_target']
    )
    # Отбираем географических доноров в заданном радиусе; при нехватке радиус не расширяем.
    candidates = {}
    candidate_diagnostics = []
    for new_id in sorted(new_ids, key=str):
        nearby = []
        for old_id in sorted(set(old_ids) & old_with_los, key=str):
            distance = float(distance_lookup.get((new_id, old_id), np.inf))
            if distance <= config['los_neighbor_radius_km']:
                nearby.append((distance, old_id))
        nearby.sort(key=lambda item: (item[0], str(item[1])))
        selected = nearby[:config['los_neighbor_count']]
        candidates[new_id] = selected
        candidate_diagnostics.append({
            'new_district': new_id,
            'radius_km': config['los_neighbor_radius_km'],
            'candidate_count_in_radius': len(nearby),
            'selected_candidate_count': len(selected),
            'farthest_candidate_km': selected[-1][0] if selected else np.nan,
            'warning': 'NO_NEARBY_CANDIDATES' if not selected
                       else ('LESS_THAN_5_CANDIDATES' if len(selected) < 5 else ''),
        })

    # Все пары начинают с резервной глобальной оценки. Наблюдаемые,
    # запрещенные и донорные связи ниже заменяются векторными масками либо
    # коротким циклом только по New ↔ Old.
    result = pair_distances[
        ['source', 'target', 'distance_km']
    ].copy().reset_index(drop=True)
    pair_index = pd.MultiIndex.from_frame(result[['source', 'target']])
    observed_time = valid.reindex(pair_index).to_numpy(dtype=float)
    forbidden_index = pd.MultiIndex.from_tuples(
        list(forbidden), names=['source', 'target']
    )
    forbidden_mask = pair_index.isin(forbidden_index)
    source_old = result['source'].isin(old_ids).to_numpy()
    target_old = result['target'].isin(old_ids).to_numpy()
    source_new = result['source'].isin(new_ids).to_numpy()
    target_new = result['target'].isin(new_ids).to_numpy()
    new_pair = source_new | target_new

    distance = result['distance_km'].to_numpy(dtype=float)
    result['travel_time'] = np.maximum(
        config['time_floor_hours'], intercept + slope * distance
    )
    result['los_source'] = 'ESTIMATED_LOS'
    result['los_method'] = 'global_distance'
    result['donor_count'] = 0
    distance_min, distance_max = fit['distance_min_km'], fit['distance_max_km']
    result['is_extrapolation'] = (
        (distance < distance_min) | (distance > distance_max)
    )
    result['is_forbidden'] = False
    result['warning'] = np.where(
        new_pair, 'NO_DONORS', 'MISSING_OBSERVED_LOS'
    )

    # Пригодные наблюдения между старыми районами имеют приоритет над резервной оценкой.
    observed_mask = (
        np.isfinite(observed_time) & source_old & target_old & ~forbidden_mask
    )
    result.loc[observed_mask, 'travel_time'] = observed_time[observed_mask]
    result.loc[observed_mask, 'los_source'] = 'OBSERVED_LOS'
    result.loc[observed_mask, 'los_method'] = 'observed'
    result.loc[observed_mask, 'warning'] = ''

    # Явный направленный запрет сохраняем даже при наличии времени или подходящих доноров.
    result.loc[forbidden_mask, 'travel_time'] = np.nan
    result.loc[forbidden_mask, 'los_source'] = 'OBSERVED_LOS'
    result.loc[forbidden_mask, 'los_method'] = 'forbidden'
    result.loc[forbidden_mask, 'is_extrapolation'] = False
    result.loc[forbidden_mask, 'is_forbidden'] = True
    result.loc[forbidden_mask, 'warning'] = ''

    donor_records = []
    transfer_updates = []
    transfer_mask = (
        ((source_new & target_old) | (source_old & target_new))
        & ~forbidden_mask
    )
    for pair in result.loc[transfer_mask].itertuples():
        source = pair.source
        target = pair.target
        pair_distance = float(pair.distance_km)
        donor_values = []
        # Для нового источника используем donor→target, для новой цели — source→donor.
        # Обратное направление не подменяет отсутствующее наблюдение.
        if source in new_ids:
            for donor_distance, donor in candidates.get(source, []):
                if donor == target:
                    continue
                donor_time = valid_map.get((donor, target), np.nan)
                if np.isfinite(donor_time):
                    donor_target_distance = float(distance_lookup[(donor, target)])
                    adjusted = float(donor_time) + slope * (
                        pair_distance - donor_target_distance
                    )
                    weight = 1 / max(donor_distance, config['distance_floor_km'])
                    donor_values.append((donor, donor_time, donor_distance, weight, adjusted))
        else:
            for donor_distance, donor in candidates.get(target, []):
                if donor == source:
                    continue
                donor_time = valid_map.get((source, donor), np.nan)
                if np.isfinite(donor_time):
                    source_donor_distance = float(distance_lookup[(source, donor)])
                    adjusted = float(donor_time) + slope * (
                        pair_distance - source_donor_distance
                    )
                    weight = 1 / max(donor_distance, config['distance_floor_km'])
                    donor_values.append((donor, donor_time, donor_distance, weight, adjusted))
        # Поправляем донорские времена на различие расстояний и усредняем с обратными весами.
        # Минимальное время защищает оценку от неположительного результата поправки.
        if donor_values:
            weights = np.asarray([item[3] for item in donor_values])
            adjusted = np.asarray([item[4] for item in donor_values])
            travel_time = max(
                config['time_floor_hours'],
                float(np.average(adjusted, weights=weights)),
            )
            donor_count = len(donor_values)
            transfer_updates.append((
                pair.Index,
                travel_time,
                donor_count,
                'LOW_DONOR_COUNT' if donor_count < 3 else '',
            ))
            for donor, donor_time, donor_distance, weight, adjusted_time in donor_values:
                donor_records.append({
                    'source': source, 'target': target,
                    'donor_district': donor, 'donor_time': donor_time,
                    'donor_distance_km': donor_distance, 'weight': weight,
                    'adjusted_time': adjusted_time,
                })
    # Обновляем только пары с реально найденными донорами; остальные сохраняют резервную оценку.
    if transfer_updates:
        update_index = [item[0] for item in transfer_updates]
        result.loc[update_index, 'travel_time'] = [
            item[1] for item in transfer_updates
        ]
        result.loc[update_index, 'los_method'] = 'neighbor_transfer'
        result.loc[update_index, 'donor_count'] = [
            item[2] for item in transfer_updates
        ]
        result.loc[update_index, 'warning'] = [
            item[3] for item in transfer_updates
        ]
    return result, pd.DataFrame(donor_records), pd.DataFrame(candidate_diagnostics), fit


def _prepare_los_arrays(pair_los, ids, constraints):
    """Готовит время, общую опору и индексы почасовых запретов одного этапа.
    Результат принадлежит конкретным LOS и порядку районов; между разбиениями
    обучения он не переиспользуется. Почасовые маски строятся отдельными копиями.
    """
    times = _matrix_array(pair_los, ids, 'travel_time')
    support = np.isfinite(times) & (times > 0)
    np.fill_diagonal(support, False)
    positions = {district: p for p, district in enumerate(ids)}
    hourly = {}
    for row in constraints.loc[constraints.is_forbidden].itertuples():
        if row.source not in positions or row.target not in positions:
            continue
        pair = (positions[row.source], positions[row.target])
        if pd.isna(row.hour):
            support[pair] = False
        else:
            hourly.setdefault(row.hour, []).append(pair)
    return {'times': times, 'support': support, 'hourly_forbidden': hourly}


def _los_arrays(pair_los, ids, constraints, hour, prepared=None):
    """Преобразует LOS в матрицу времен и допустимую направленную опору одного часа.
    Исключает диагональ, непригодные времена и запреты с общей либо выбранной временной областью.
    """
    if prepared is None:
        prepared = _prepare_los_arrays(pair_los, ids, constraints)
    support = prepared['support'].copy()
    for pair in prepared['hourly_forbidden'].get(hour, []):
        support[pair] = False
    return prepared['times'], support
