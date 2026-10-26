"""Audit clipping C without changing the assembled matrix or inventing acceptance limits."""
import numpy as np
import pandas as pd

from .config import HOUR_COLUMNS


SUM_FIELDS = ('Q_observed', 'Q_model_delta', 'Q_positive_delta', 'Q_negative_delta',
              'Q_unbounded', 'Q_C', 'Q_final', 'Q_C_existing', 'Q_C_new', 'Q_C_zero_fact',
              'Q_C_positive_fact', 'machine_volume', 'clipped_cells', 'machine_cells',
              'clipped_zero_fact_cells', 'row_insufficient_count', 'column_insufficient_count',
              'row_shortfall', 'column_shortfall')
SUMMARY_COLUMNS = ['hour', 'status', 'expected_hours', 'available_hours', 'criteria_status',
                   *SUM_FIELDS, 'correction_share', 'correction_to_positive_delta',
                   'correction_to_abs_model_delta', 'Q_actual_delta', 'correction_to_actual_growth',
                   'growth_sign_reversed', 'growth_from_zero_model_change', 'row_volume_sufficient',
                   'column_volume_sufficient', 'joint_compensation_status', 'identity_error_max']
DISTRICT_FIELDS = ('observed', 'model_delta', 'unbounded', 'final', 'C', 'compensation_shortfall',
                   'insufficient_hours')
DISTRICT_COLUMNS = ['hour', 'district_id', 'direction', 'status', 'expected_hours', 'available_hours',
                    *DISTRICT_FIELDS, 'correction_share']


def correction_tables(context, prepared):
    """Hourly C, weighted period ratios and necessary row/column volume conditions.

    Day is available only with all 19 assemblies. The selected-hours subtotal is
    explicitly partial on failures; its feasibility flags retain every hourly deficit.
    """
    ids, training = context['ids'], context['training']
    tolerance = prepared['config']['target_tolerance']
    applicable = prepared['config']['recalculation_mode'] == 'recalculate_all'
    existing = np.zeros((len(ids), len(ids)), dtype=bool)
    existing[np.ix_(training, training)] = True
    def margins(matrix):
        return np.stack((matrix.sum(axis=1), matrix.sum(axis=0)))
    hourly = {}
    for hour, states in context['hours'].items():
        assembly = states['assembly']
        if not applicable or assembly['matrix'] is None:
            continue
        final, c, u = assembly['matrix'], assembly['correction'], assembly['unbounded']
        anchor = np.zeros_like(final)
        anchor[np.ix_(training, training)] = prepared['observed'][HOUR_COLUMNS.index(hour)][np.ix_(training, training)]
        base = np.zeros_like(final)
        base[np.ix_(training, training)] = states['base']['matrix']
        delta = states['forecast']['matrix'] - base
        observed_m, delta_m, u_m, f_m, c_m = [margins(matrix) for matrix in (anchor, delta, u, final, c)]
        shortfall = np.maximum(0, c_m - f_m)
        eps = tolerance * np.maximum(1, np.maximum(c_m, f_m))
        insufficient = shortfall > eps
        values = {'Q_observed': float(anchor.sum()), 'Q_model_delta': float(delta.sum()),
                  'Q_positive_delta': float(np.maximum(delta, 0).sum()),
                  'Q_negative_delta': float(np.maximum(-delta, 0).sum()),
                  'Q_unbounded': float(u.sum()), 'Q_C': float(c.sum()), 'Q_final': float(final.sum()),
                  'Q_C_existing': float(c[existing].sum()), 'Q_C_new': float(c[~existing].sum()),
                  'Q_C_zero_fact': float(c[anchor == 0].sum()),
                  'Q_C_positive_fact': float(c[anchor > 0].sum()),
                  'machine_volume': assembly['machine_volume'], 'clipped_cells': assembly['limited_count'],
                  'machine_cells': assembly['machine_count'],
                  'clipped_zero_fact_cells': int(((c > 0) & (anchor == 0)).sum()),
                  'row_insufficient_count': int(insufficient[0].sum()),
                  'column_insufficient_count': int(insufficient[1].sum()),
                  'row_shortfall': float(shortfall[0].sum()), 'column_shortfall': float(shortfall[1].sum())}
        identity_error = float(np.abs(f_m - observed_m - delta_m - c_m).max())
        hourly[hour] = (values, dict(zip(DISTRICT_FIELDS,
                         (observed_m, delta_m, u_m, f_m, c_m, shortfall, insufficient.astype(int)))), identity_error)
    periods = [(hour, [hour], False) for hour in context['hours']]
    periods += [('day', list(HOUR_COLUMNS), False),
                ('selected_successful_hours', list(context['hours']), True)]
    summaries, districts = [], []
    for label, hours, allow_partial in periods:
        available = [hourly[h] for h in hours if h in hourly]
        usable = bool(available) and (len(available) == len(hours) or allow_partial)
        status = ('NOT_APPLICABLE' if not applicable else 'UNAVAILABLE' if not usable else
                  'PARTIAL' if len(available) < len(hours) else 'OK')
        row = dict.fromkeys(SUMMARY_COLUMNS)
        row.update(hour=label, status=status, expected_hours=len(hours), available_hours=len(available),
                   criteria_status='CRITERIA_NOT_SET' if applicable else 'NOT_APPLICABLE',
                   joint_compensation_status='NOT_EVALUATED' if applicable else 'NOT_APPLICABLE')
        local = None
        if usable:
            row.update({key: sum(item[0][key] for item in available) for key in SUM_FIELDS})
            for key, denominator in [('correction_share', row['Q_final']),
                                     ('correction_to_positive_delta', row['Q_positive_delta']),
                                     ('correction_to_abs_model_delta', abs(row['Q_model_delta']))]:
                row[key] = row['Q_C'] / denominator if denominator > 0 else None
            growth = row['Q_final'] - row['Q_observed']
            row['Q_actual_delta'] = growth
            row['correction_to_actual_growth'] = row['Q_C'] / growth if growth > 0 else None
            eps = tolerance * max(1, abs(row['Q_model_delta']), abs(growth))
            row['growth_sign_reversed'] = bool(row['Q_model_delta'] < -eps and growth > eps)
            row['growth_from_zero_model_change'] = bool(abs(row['Q_model_delta']) <= eps and growth > eps)
            row['row_volume_sufficient'] = row['row_insufficient_count'] == 0
            row['column_volume_sufficient'] = row['column_insufficient_count'] == 0
            row['identity_error_max'] = max(item[2] for item in available)
            local = {key: sum(item[1][key] for item in available) for key in DISTRICT_FIELDS}
        summaries.append(row)
        for direction, name in enumerate(('outgoing', 'incoming')):
            for i, district in enumerate(ids):
                item = dict.fromkeys(DISTRICT_COLUMNS)
                item.update(hour=label, district_id=district, direction=name, status=status,
                            expected_hours=len(hours), available_hours=len(available))
                if local is not None:
                    item.update({key: array[direction, i] for key, array in local.items()})
                    item['correction_share'] = item['C'] / item['final'] if item['final'] > 0 else None
                districts.append(item)
    return pd.DataFrame(summaries, columns=SUMMARY_COLUMNS), pd.DataFrame(districts, columns=DISTRICT_COLUMNS)
