"""Spatial folds and an independent nearest-district transfer benchmark."""

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import FEATURE_COLUMNS
from .inputs import _validate_district_geometry


def validation_splits(prepared, scheme):
    """Return identical district folds for generation, OD and the benchmark."""
    n, config = prepared['n_existing'], prepared['config']
    k = config['validation_folds']
    if scheme == 'random':
        return list(KFold(n_splits=k, shuffle=True, random_state=config['random_state']).split(range(n)))
    if scheme != 'spatial':
        raise ValueError(f'Unknown validation scheme: {scheme}')
    base = prepared['base'].reset_index()
    geometry = _validate_district_geometry(base)
    metric = gpd.GeoDataFrame(geometry=geometry).to_crs(prepared['metric_crs'])
    centers = metric.geometry.centroid
    xy = np.column_stack((centers.x, centers.y))
    sizes = [n // k + (i < n % k) for i in range(k)]
    ids = np.asarray(prepared['ids'][:n])

    def partition(indices, counts):
        if len(counts) == 1:
            return [np.sort(indices)]
        axis = int(np.argmax(np.ptp(xy[indices], axis=0)))
        # ID resolves equal coordinates independently of input row order.
        order = np.lexsort((ids[indices], xy[indices, axis]))
        indices = indices[order]
        middle = len(counts) // 2
        boundary = sum(counts[:middle])
        return partition(indices[:boundary], counts[:middle]) + partition(indices[boundary:], counts[middle:])

    return [(np.setdiff1d(np.arange(n), hidden), hidden)
            for hidden in partition(np.arange(n), sizes)]


def nearest_analogue_baseline(prepared, training):
    """Copy R→R flows with one geographic donor and feature-size factors per H.

    Size is the sum of features divided by their positive means on R. An unknown
    positive size with a zero-size nearest donor is unavailable, never silently zero.
    No hidden flows, LOS, fitted Ridge coefficients or IPF enter this benchmark.
    """
    n = prepared['n_existing']
    training = np.asarray(training, dtype=int)
    hidden = np.setdiff1d(np.arange(n), training)
    distance = prepared['distance'][:n, :n]
    features = prepared['base'][FEATURE_COLUMNS].to_numpy(dtype=float)
    means = features[training].mean(axis=0)
    active = means > 0
    size = (features[:, active] / means[active]).sum(axis=1)
    donors, scale = np.arange(n), np.ones(n)
    records = []
    radius = prepared['config']['los_neighbor_radius_km']
    analogue_counts = np.zeros(n, dtype=int)
    for i in hidden:
        donor = training[np.argmin(distance[i, training])]
        donors[i] = donor
        analogue_counts[i] = np.count_nonzero(distance[i, training] <= radius)
        scale[i] = size[i] / size[donor] if size[donor] > 0 else (0 if size[i] == 0 else np.nan)
        records.append({'district_id': prepared['ids'][i], 'donor_id': prepared['ids'][donor],
                        'donor_distance_km': distance[i, donor], 'analogue_count': analogue_counts[i],
                        'has_analogue': bool(analogue_counts[i]), 'district_size': size[i],
                        'donor_size': size[donor], 'scale': scale[i],
                        'status': 'OK' if np.isfinite(scale[i]) else 'ZERO_DONOR_SIZE'})
    return {'donors': donors, 'scale': scale, 'analogue_counts': analogue_counts,
            'diagnostics': pd.DataFrame(records)}


def baseline_matrix(prepared, baseline, hours):
    """Predict only from copied R→R cells; keep unknown endpoint scales as NaN."""
    donor, scale = baseline['donors'], baseline['scale']
    # Map before taking the period sum: no original H flow is ever read.
    values = prepared['observed'][hours][:, donor][:, :, donor].sum(axis=0)
    result = values * scale[:, None] * scale[None, :]
    if not prepared['config']['include_intrazonal']:
        np.fill_diagonal(result, 0)
    return result
