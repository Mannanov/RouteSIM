"""Accounting for marginal changes; no counterfactual OD matrices are inferred."""
import numpy as np
import pandas as pd

from .config import HOUR_COLUMNS


def reconciliation_delta_tables(context, prepared):
    """Separate raw generation, total balancing and feasibility reductions in delta."""
    periods = [(hour, [h]) for h, hour in enumerate(HOUR_COLUMNS)] + [('day', list(range(19)))]
    if prepared['config']['recalculation_mode'] == 'preserve_old':
        return pd.DataFrame([{'hour': hour, 'status': 'NOT_APPLICABLE',
                              'reason': 'preserve_old does not use M_forecast - M_base'}
                             for hour, _ in periods]), pd.DataFrame()
    ids, training = context['ids'], context['training']
    forecast, base = context['generation']['forecast'], context['generation']['base']
    shape = forecast['goals'].shape

    def padded(values):
        result = np.zeros(shape)
        result[:, :, training] = values
        return result

    base_goals, base_adjusted = padded(base['goals']), padded(base['adjusted_goals'])
    values = {'raw_delta': forecast['raw'] - padded(base['raw']),
              'target_delta': forecast['goals'] - base_goals,
              'reduction_base': base_goals - base_adjusted,
              'reduction_forecast': forecast['goals'] - forecast['adjusted_goals'],
              'adjusted_delta': forecast['adjusted_goals'] - base_adjusted}
    values['balance_effect'] = values['target_delta'] - values['raw_delta']
    values['reduction_effect'] = values['reduction_base'] - values['reduction_forecast']
    values['identity_error'] = values['adjusted_delta'] - values['target_delta'] - values['reduction_effect']
    od = np.full(shape, np.nan)
    observed, final, correction = (np.full(19, np.nan) for _ in range(3))
    for hour, states in context['hours'].items():
        h = HOUR_COLUMNS.index(hour)
        f, b = states['forecast']['matrix'], states['base']['matrix']
        if f is not None and b is not None:
            od[:, h] = np.stack((f.sum(axis=1), f.sum(axis=0)))
            od[:, h, training] -= np.stack((b.sum(axis=1), b.sum(axis=0)))
        assembly = states['assembly']
        if assembly['matrix'] is not None:
            observed[h] = prepared['observed'][h][np.ix_(training, training)].sum()
            final[h], correction[h] = assembly['matrix'].sum(), assembly['Q_C']
    values['od_delta'] = od
    values['od_margin_error'] = od - values['adjusted_delta']
    summary, districts = [], []
    for hour, hs in periods:
        # Ordinary sums intentionally preserve failed/missing hours, including in day.
        totals = {key: array[:, hs].sum(axis=1) for key, array in values.items()}
        valid = np.isfinite(totals['adjusted_delta']).all()
        row = {'hour': hour, 'status': 'OK' if valid else 'TARGETS_UNAVAILABLE',
               'od_status': 'OK' if np.isfinite(totals['od_delta']).all() else 'UNAVAILABLE',
               'expected_hours': len(hs),
               'available_target_hours': sum(np.isfinite(values['adjusted_delta'][:, h]).all() for h in hs),
               'available_od_hours': sum(np.isfinite(od[:, h]).all() for h in hs)}
        for key in ('raw_delta', 'target_delta', 'balance_effect', 'reduction_base',
                    'reduction_forecast', 'reduction_effect', 'adjusted_delta', 'od_delta'):
            row['Q_' + key] = float(totals[key].sum() / 2)
        row['identity_error_max'] = float(np.abs(totals['identity_error']).max())
        row['od_margin_error_max'] = float(np.abs(totals['od_margin_error']).max())
        row['L1_reduction_effect'] = float(np.abs(totals['reduction_effect']).sum() / 2)
        # Counts describe directed district marginals, not OD cells or trip volumes.
        t, a = totals['target_delta'], totals['adjusted_delta']
        eps = prepared['config']['target_tolerance'] * np.maximum(1, np.maximum(np.abs(t), np.abs(a)))
        row['sign_reversals'] = int(((t * a < 0) & (np.abs(t) > eps) & (np.abs(a) > eps)).sum()) if valid else None
        row['zero_target_changes'] = int(((np.abs(t) <= eps) & (np.abs(a) > eps)).sum()) if valid else None
        row['Q_observed'], row['Q_final'], row['Q_C'] = (float(array[hs].sum()) for array in (observed, final, correction))
        row['assembly_error'] = row['Q_final'] - row['Q_observed'] - row['Q_od_delta'] - row['Q_C']
        summary.append(row)
        for direction, label in enumerate(('outgoing', 'incoming')):
            for i, district in enumerate(ids):
                districts.append({'hour': hour, 'district_id': district, 'direction': label,
                                  'status': row['status'], 'od_status': row['od_status'],
                                  **{key: array[direction, i] for key, array in totals.items()}})
    return pd.DataFrame(summary), pd.DataFrame(districts)
