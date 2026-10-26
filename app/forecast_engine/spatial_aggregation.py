"""Mass-preserving aggregation using normalized stop catchment overlaps."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from .config import HOUR_COLUMNS
from .inputs import _identifier_key
from serv_funcs import build_stop_access


def aggregate_spatial_matrix(matrix, stops, districts, config):
    access = build_stop_access(districts, stops, config['access_radius_km'], config['min_overlap_share'])
    access['weight'] = (access.overlap_share / access.groupby('id_stop').overlap_share.transform('sum'))
    access['access_radius_km'] = config['access_radius_km']
    access['min_overlap_share'] = config['min_overlap_share']
    # Normalize each end independently; every assigned OD trip retains total weight 1.
    stop_ids = list(stops.id_stop.map(_identifier_key))
    district_ids = list(districts.id_tr_distr)
    si, di = {z:i for i,z in enumerate(stop_ids)}, {z:i for i,z in enumerate(district_ids)}
    eligible = set(access.id_stop.map(_identifier_key))
    sources, targets = matrix.source.map(_identifier_key), matrix.target.map(_identifier_key)
    included = sources.isin(eligible) & targets.isin(eligible)
    excluded = matrix.loc[~included].copy()
    weights = coo_matrix((access.weight.to_numpy(dtype=float),
                         ([si[_identifier_key(z)] for z in access.id_stop],
                          [di[z] for z in access.id_tr_distr])),
                        shape=(len(si), len(di))).tocsr()
    sr = [si[z] for z in sources.loc[included]]
    tr = [si[z] for z in targets.loc[included]]
    arrays = []
    for hour in HOUR_COLUMNS:
        flow = coo_matrix((matrix.loc[included, hour].to_numpy(), (sr,tr)), shape=(len(si),len(si))).tocsr()
        arrays.append((weights.T @ flow @ weights).toarray())
    values = np.stack(arrays)
    i,j = np.where((values != 0).any(axis=0))
    result = pd.DataFrame({'district_source':[district_ids[k] for k in i],
                           'district_target':[district_ids[k] for k in j],
                           **{h: values[k,i,j] for k,h in enumerate(HOUR_COLUMNS)}})
    result['day'] = result[HOUR_COLUMNS].sum(axis=1)
    diagonal = result.district_source.eq(result.district_target)
    inter, intra = result.loc[~diagonal].copy(), result.loc[diagonal].copy()
    total = float(matrix.day.sum())
    totals = {'interzonal_demand':float(inter.day.sum()), 'intrazonal_demand':float(intra.day.sum()),
              'excluded_demand':float(excluded.day.sum())}
    balance = total - sum(totals.values())
    control = pd.DataFrame([{'control_type':'AGGREGATION', 'stop_demand':total, **totals,
                             'balance':balance, 'status':'OK' if abs(balance) <= .01 else 'ERROR'}])
    return {'interzonal':inter, 'intrazonal':intra, 'excluded':excluded, 'control':control, 'stop_access':access}
