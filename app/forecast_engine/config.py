"""Forecast versions and configuration; no calculation or Flask dependencies."""
import copy
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Historical versions distinguish calculation contracts without rewriting results.
METHOD_VERSION = '2026-10-10-hourly-target-policy'
COMPONENT_VALIDATION_METHOD_VERSION = '2026-10-10-hourly-component-validation'
VERIFIED_CALIBRATION_METHOD_VERSION = '2026-10-10-hourly-verified-calibration'
VALIDATION_PROTOCOL_VERSION = '2026-10-10-components-full-marginals'
OPTIONAL_THRESHOLD_METHOD_VERSION = '2026-10-09-hourly-optional-threshold'
UNBOUNDED_PIVOT_METHOD_VERSION = '2026-10-08-hourly-unbounded-pivot'
PREVIOUS_METHOD_VERSION = '2026-10-02-hourly-additive'
BOUNDED_PIVOT_METHOD_VERSION = '2026-10-08-hourly-change-threshold'
PREVIOUS_HOURLY_METHOD_VERSIONS = {PREVIOUS_METHOD_VERSION, BOUNDED_PIVOT_METHOD_VERSION,
                                 UNBOUNDED_PIVOT_METHOD_VERSION, OPTIONAL_THRESHOLD_METHOD_VERSION,
                                 VERIFIED_CALIBRATION_METHOD_VERSION, COMPONENT_VALIDATION_METHOD_VERSION}
HOURLY_METHOD_VERSIONS = PREVIOUS_HOURLY_METHOD_VERSIONS | {METHOD_VERSION}
LEGACY_METHOD_VERSION = '2026-09-18-chains'
CHARACTERISTIC_CHANGE_THRESHOLD_MIN = .01
CHARACTERISTIC_CHANGE_THRESHOLD_MAX = .10

# Axis order is shared by matrices, coefficients and exported tables.
HOUR_COLUMNS = [f'h{hour}' for hour in range(5, 24)]
FEATURE_COLUMNS = [
    'population', 'employment', 'school', 'higher_education',
    'hospital', 'local_medical', 'shopping', 'public_services',
]


def _integer(value, name):
    """Возвращает положительное целое без округления дробного значения.
    Логическое True не считается целым параметром количества.
    """
    try:
        result = int(value)
        valid = not isinstance(value, bool) and float(value) == result and result > 0
    except (TypeError, ValueError, OverflowError):
        valid = False
        result = 0
    if not valid:
        raise ValueError(f'{name} должен быть положительным целым числом')
    return result


def _positive(value, name, allow_zero=False):
    """Возвращает конечное положительное число либо ноль при allow_zero=True."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = np.nan
    if not np.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        qualifier = 'неотрицательным' if allow_zero else 'положительным'
        raise ValueError(f'{name} должен быть конечным {qualifier} числом')
    return result


def _at_least(value, name, minimum):
    """Проверяет положительное число и установленную нижнюю границу параметра."""
    result = _positive(value, name)
    if result < minimum:
        raise ValueError(f'{name} должен быть не меньше {minimum}')
    return result


def _boolean(value, name):
    """Принимает только bool или numpy.bool_, исключая неявное приведение строк."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} должен быть true или false')
    return bool(value)


@dataclass(frozen=True)
class ParameterSpec:
    """Canonical contract plus the existing editable form projection."""

    default: object
    kind: str
    unit: str = 'dimensionless'
    role: str = 'API_CANONICAL'
    minimum: object = None
    maximum: object = None
    minimum_closed: bool = True
    maximum_closed: bool = True
    allow_zero: bool = False
    disabled_zero: bool = False
    optional: bool = False
    step: object = None
    choices: tuple = ()
    range_error: object = None
    minimum_error: object = None
    field: object = None
    scale: float = 1.
    ui_minimum: object = None
    ui_maximum: object = None
    group: str = 'internal'


# Order matches the persisted parameter registry and configuration serialization.
PARAMETER_SPECS = {
    'methodology_version': ParameterSpec(METHOD_VERSION, 'fixed', choices=(METHOD_VERSION,),
        range_error='Несовместимая версия методики: создайте расчет новой версии', group='input_contract'),
    'characteristic_change_threshold': ParameterSpec(0., 'number', unit='share',
        minimum=CHARACTERISTIC_CHANGE_THRESHOLD_MIN, maximum=CHARACTERISTIC_CHANGE_THRESHOLD_MAX,
        allow_zero=True, disabled_zero=True, field='characteristic_change_threshold_percent', scale=100,
        range_error='characteristic_change_threshold: от 0.01 до 0.10; 0 только для расчёта без порога', group='main'),
    'characteristic_change_absolute_tolerance': ParameterSpec(0., 'number', unit='trips',
        role='INTERNAL_TECHNICAL', minimum=0, allow_zero=True),
    'recalculation_mode': ParameterSpec('preserve_old', 'choice', choices=('preserve_old', 'recalculate_all'),
        range_error='Допустимы только preserve_old и recalculate_all', field='recalculation_mode', group='main'),
    'district_target_policy': ParameterSpec('direct_ratio', 'choice', choices=('direct_ratio', 'residual_change'),
        range_error='district_target_policy: допустимы direct_ratio и residual_change',
        role='INTERNAL_TECHNICAL'),
    'analog_count': ParameterSpec(10, 'integer', minimum=1, maximum=30, field='analog_count', group='main'),
    'los_neighbor_radius_km': ParameterSpec(3.5, 'number', unit='km', minimum=1, maximum=5, step=.5,
        range_error='los_neighbor_radius_km: от 1 до 5 км с шагом 0.5', field='los_neighbor_radius_km', group='main'),
    'los_neighbor_count': ParameterSpec(7, 'integer', minimum=1, maximum=30, field='los_neighbor_count', group='main'),
    'distance_floor_km': ParameterSpec(.1, 'number', unit='km', role='INTERNAL_TECHNICAL', minimum=.001),
    'time_floor_hours': ParameterSpec(1 / 600, 'number', unit='hours', role='INTERNAL_TECHNICAL', minimum=.001 / 60),
    'ipf_tolerance': ParameterSpec(.01, 'number', unit='trips', role='INTERNAL_TECHNICAL', minimum=.000001),
    'ipf_max_iterations': ParameterSpec(1000, 'integer', role='INTERNAL_TECHNICAL', minimum=1, maximum=10000),
    'resistance_ipf_relative_tolerance': ParameterSpec(1e-10, 'number', unit='share', minimum=1e-14, maximum=1e-4,
        range_error='resistance_ipf_relative_tolerance: допустимо от 1e-14 до 1e-4'),
    'resistance_gradient_tolerance': ParameterSpec(1e-6, 'number', unit='hours', minimum=1e-14, maximum=1e-4,
        range_error='resistance_gradient_tolerance: допустимо от 1e-14 до 1e-4'),
    'resistance_max_iterations': ParameterSpec(100, 'integer', unit='iterations', minimum=1, maximum=10000),
    'resistance_beta_interval_tolerance': ParameterSpec(1e-6, 'number', unit='per_hour', minimum=1e-14, maximum=1e-2,
        range_error='resistance_beta_interval_tolerance: допустимо от 1e-14 до 1e-2'),
    'validation_enabled': ParameterSpec(True, 'boolean', group='input_contract'),
    'matrix_coverage': ParameterSpec('unconfirmed', 'choice', choices=('unconfirmed', 'full_sparse'),
        range_error='Неполное покрытие не позволяет получить полные районные итоги; нужен подтвержденный комплект', group='input_contract'),
    'matrix_unit': ParameterSpec('passenger_trips', 'fixed', choices=('passenger_trips',),
        range_error='Для прогноза на ОТ требуется матрица пассажирских поездок на ОТ, а не посадок или всех видов транспорта', group='input_contract'),
    'zoning_comparable': ParameterSpec(False, 'boolean', group='input_contract'),
    'input_requirements_source': ParameterSpec('explicit_configuration', 'choice',
        choices=('explicit_configuration', 'web_input_contract'),
        range_error='Неизвестный источник предпосылок исходного комплекта', group='input_contract'),
    'regularization_grid': ParameterSpec([[.001, .001], [.01, .01], [.1, .1]], 'grid'),
    'zero_mean_max_observation_share': ParameterSpec(.05, 'number', unit='share', minimum=0, maximum=1,
        allow_zero=True, range_error='zero_mean_max_observation_share: недопустимая доля'),
    'zero_mean_max_trip_share': ParameterSpec(.01, 'number', unit='share', minimum=0, maximum=1,
        allow_zero=True, range_error='zero_mean_max_trip_share: недопустимая доля'),
    'validation_blocks': ParameterSpec(4, 'integer', role='INTERNAL_TECHNICAL', minimum=2, maximum=20,
        minimum_error='validation_blocks должен быть не меньше 2'),
    'validation_buffer_km': ParameterSpec(0., 'number', unit='km', role='INTERNAL_TECHNICAL', minimum=0, allow_zero=True),
    'size_quantile': ParameterSpec(.8, 'number', unit='quantile', role='INTERNAL_TECHNICAL', minimum=0, maximum=1,
        minimum_closed=False, maximum_closed=False, allow_zero=True,
        range_error='size_quantile: недопустимая доля'),
    'validation_max_wape': ParameterSpec(None, 'number', unit='share', role='INTERNAL_TECHNICAL', minimum=0, allow_zero=True,
        optional=True),
    'merge_pair_count': ParameterSpec(3, 'integer', minimum=1, maximum=10000),
    'new_new_pair_count': ParameterSpec(3, 'integer', minimum=1, maximum=10000),
    'unmapped_max_trip_share': ParameterSpec(.01, 'number', unit='share', role='INTERNAL_TECHNICAL', minimum=0, maximum=1,
        allow_zero=True, range_error='unmapped_max_trip_share: недопустимая доля'),
    'unmapped_hour_loss_tolerance': ParameterSpec(10., 'number', unit='trips', role='INTERNAL_TECHNICAL', minimum=0,
        allow_zero=True),
    'sensitivity_enabled': ParameterSpec(False, 'boolean', role='INTERNAL_TECHNICAL'),
    'sensitivity_relative_change': ParameterSpec(.1, 'number', unit='share', role='INTERNAL_TECHNICAL', minimum=0, maximum=1,
        minimum_closed=False, maximum_closed=False, allow_zero=True,
        range_error='sensitivity_relative_change: недопустимая доля'),
}

INTERNAL_LOS_PARAMETER_KEYS = ('distance_floor_km', 'time_floor_hours')
DEFAULT_CONFIG = {key: copy.deepcopy(spec.default) for key, spec in PARAMETER_SPECS.items()}
METHODOLOGY_DEFAULT_CONFIG = DEFAULT_CONFIG
WEB_PARAMETER_FIELDS = {spec.field: key for key, spec in PARAMETER_SPECS.items() if spec.field is not None}
THRESHOLD_ENABLED_FIELD = 'characteristic_change_threshold_enabled'
WEB_FORM_FIELDS = (*WEB_PARAMETER_FIELDS, THRESHOLD_ENABLED_FIELD)
WEB_INPUT_CONTRACT = {'methodology_version': METHOD_VERSION, 'matrix_coverage': 'full_sparse',
    'matrix_unit': 'passenger_trips', 'zoning_comparable': True,
    'input_requirements_source': 'web_input_contract', 'validation_enabled': True}
OBSOLETE_PIVOT_KEYS = frozenset({'ratio_min', 'ratio_max', 'pivot_ratio_min', 'pivot_ratio_max'})

# Preserve validation order for common single-field failures and canonical types.
VALIDATION_ORDER = (
    'methodology_version', 'characteristic_change_threshold', 'characteristic_change_absolute_tolerance',
    'recalculation_mode', 'district_target_policy', 'analog_count', 'los_neighbor_count', 'ipf_max_iterations', 'validation_blocks',
    'merge_pair_count', 'new_new_pair_count', 'resistance_max_iterations', 'los_neighbor_radius_km',
    'distance_floor_km', 'time_floor_hours', 'ipf_tolerance', 'validation_buffer_km',
    'resistance_ipf_relative_tolerance', 'resistance_gradient_tolerance', 'resistance_beta_interval_tolerance',
    'validation_enabled', 'sensitivity_enabled', 'zoning_comparable', 'input_requirements_source',
    'zero_mean_max_observation_share', 'zero_mean_max_trip_share', 'size_quantile', 'unmapped_max_trip_share',
    'sensitivity_relative_change', 'validation_max_wape', 'unmapped_hour_loss_tolerance',
    'matrix_coverage', 'matrix_unit', 'regularization_grid',
)
assert set(VALIDATION_ORDER) == set(PARAMETER_SPECS)


def _validate_parameter(key, value):
    spec = PARAMETER_SPECS[key]
    if spec.optional and value is None:
        return None
    if spec.kind == 'boolean':
        return _boolean(value, key)
    if spec.kind in {'choice', 'fixed'}:
        valid = value in set(spec.choices) if spec.kind == 'choice' else value == spec.choices[0]
        if not valid:
            raise ValueError(spec.range_error)
        return value
    if spec.kind == 'grid':
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError('regularization_grid должен содержать пары штрафов')
        normalized = []
        for pair in value:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError('regularization_grid: требуется пара ridge/smoothing')
            normalized.append([_positive(item, 'regularization', True) for item in pair])
        return normalized
    result = _integer(value, key) if spec.kind == 'integer' else _positive(value, key, spec.allow_zero)
    if spec.disabled_zero and result == 0:
        return result
    below = spec.minimum is not None and (result < spec.minimum or
             (not spec.minimum_closed and result == spec.minimum))
    above = spec.maximum is not None and (result > spec.maximum or
             (not spec.maximum_closed and result == spec.maximum))
    off_step = spec.step is not None and not np.isclose(result / spec.step, round(result / spec.step), rtol=1e-12, atol=0)
    if below or above or off_step:
        if below and spec.minimum_error:
            message = spec.minimum_error
        elif spec.range_error:
            message = spec.range_error
        elif spec.kind == 'integer':
            message = f'{key}: допустимо от 1 до {spec.maximum}'
        else:
            message = f'{key} должен быть не меньше {spec.minimum}'
        raise ValueError(message)
    return result


def _validate_config(config=None, existing_count=0):
    """Normalize the canonical configuration using the parameter metadata."""
    if config is not None and not isinstance(config, dict):
        raise ValueError('config должен быть словарем параметров')
    supplied = dict(config or {})
    obsolete = OBSOLETE_PIVOT_KEYS & set(supplied)
    if obsolete:
        raise ValueError(f'Устаревшие границы pivot: {sorted(obsolete)}; эти параметры больше не поддерживаются')
    if 'time_floor_minutes' in supplied:
        hours = _positive(supplied.pop('time_floor_minutes'), 'time_floor_minutes') / 60
        if 'time_floor_hours' in supplied and not np.isclose(
                hours, _positive(supplied['time_floor_hours'], 'time_floor_hours'), rtol=1e-12, atol=0):
            raise ValueError('time_floor_minutes и time_floor_hours задают разные значения')
        supplied['time_floor_hours'] = hours
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f'Неизвестные или устаревшие параметры прогноза: {sorted(unknown)}')
    result = {**copy.deepcopy(DEFAULT_CONFIG), **supplied}
    for key in VALIDATION_ORDER:
        result[key] = _validate_parameter(key, result[key])
    if existing_count:
        result['analog_count'] = min(existing_count, result['analog_count'])
    return result


def _parameter_registry(config):
    """Keep the existing CSV schema and row order, sourcing units/roles from metadata."""
    rows = [{'parameter': key, 'unit': spec.unit, 'default_json': json.dumps(DEFAULT_CONFIG[key]),
             'used_json': json.dumps(config[key]), 'role': spec.role}
            for key, spec in PARAMETER_SPECS.items()]
    rows.append({'parameter': 'time_floor_minutes', 'unit': 'minutes',
                 'default_json': json.dumps(DEFAULT_CONFIG['time_floor_hours'] * 60),
                 'used_json': json.dumps(config['time_floor_hours'] * 60),
                 'role': 'LEGACY_API_ALIAS', 'conversion': 'time_floor_hours = time_floor_minutes / 60'})
    rows.append({'parameter': 'characteristic_change_threshold_percent', 'unit': 'percent',
                 'default_json': json.dumps(DEFAULT_CONFIG['characteristic_change_threshold'] * 100),
                 'used_json': json.dumps(config['characteristic_change_threshold'] * 100),
                 'role': 'UI_PERCENT', 'conversion': 'characteristic_change_threshold = characteristic_change_threshold_percent / 100'})
    return pd.DataFrame(rows)


def web_parameter_default(field):
    key = WEB_PARAMETER_FIELDS[field]
    spec = PARAMETER_SPECS[key]
    value = copy.deepcopy(DEFAULT_CONFIG[key])
    return value * spec.scale if value is not None and spec.scale != 1 else value


def web_parameter_bounds(field):
    spec = PARAMETER_SPECS[WEB_PARAMETER_FIELDS[field]]
    lower = spec.ui_minimum if spec.ui_minimum is not None else spec.minimum
    upper = spec.ui_maximum if spec.ui_maximum is not None else spec.maximum
    return {'min': lower * spec.scale if lower is not None and spec.scale != 1 else lower,
            'max': upper * spec.scale if upper is not None and spec.scale != 1 else upper}


def load_saved_config(payload):
    """Parse a stored object without silently discarding invalid configuration."""
    if payload is None or payload == '':
        return {}
    if isinstance(payload, dict):
        return copy.deepcopy(payload)
    try:
        saved = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError('Сохранённая конфигурация повреждена: требуется корректный JSON-объект') from error
    if not isinstance(saved, dict):
        raise ValueError('Сохранённая конфигурация должна быть JSON-объектом')
    return saved


def web_form_values(saved):
    """Project explicitly saved values into the current form, including zero/false."""
    result = {}
    for field, key in WEB_PARAMETER_FIELDS.items():
        if key not in saved:
            continue
        value = saved[key]
        spec = PARAMETER_SPECS[key]
        if spec.kind == 'boolean':
            value = _boolean(value, key)
        elif spec.scale != 1:
            try:
                value = round(float(value) * spec.scale, 12)
            except (TypeError, ValueError) as error:
                raise ValueError('Сохранённый порог некорректен. Задайте значение от 1 до 10 %.') from error
        result[field] = value
    if 'characteristic_change_threshold' in saved:
        value = result['characteristic_change_threshold_percent']
        result[THRESHOLD_ENABLED_FIELD] = value != 0
        if value == 0:
            result['characteristic_change_threshold_percent'] = None
    return result


def build_web_config(saved, form_values):
    """Defaults -> saved values -> explicit form edits -> web input contract."""
    supplied = load_saved_config(saved)
    unknown = set(form_values) - set(WEB_FORM_FIELDS)
    if unknown:
        raise ValueError(f'Неизвестные поля формы прогноза: {sorted(unknown)}')
    # A partial edit keeps the saved activation; an explicit percent is an active edit.
    enabled = (_boolean(form_values[THRESHOLD_ENABLED_FIELD], THRESHOLD_ENABLED_FIELD)
               if THRESHOLD_ENABLED_FIELD in form_values else
               'characteristic_change_threshold_percent' in form_values or
               supplied.get('characteristic_change_threshold', 0.) != 0)
    for field, value in form_values.items():
        if field == THRESHOLD_ENABLED_FIELD or (not enabled and field == 'characteristic_change_threshold_percent'):
            continue
        key = WEB_PARAMETER_FIELDS[field]
        spec = PARAMETER_SPECS[key]
        if spec.kind in {'number', 'integer'} and value is not None:
            value = _integer(value, field) if spec.kind == 'integer' else _positive(value, field, spec.allow_zero)
            bounds = web_parameter_bounds(field)
            if ((bounds['min'] is not None and value < bounds['min']) or
                    (bounds['max'] is not None and value > bounds['max'])):
                raise ValueError(f'{field}: значение вне диапазона формы')
        if value is not None and spec.scale != 1:
            value = float(value) / spec.scale
        supplied[key] = value
    if not enabled:
        supplied['characteristic_change_threshold'] = 0.
    elif supplied.get('characteristic_change_threshold') is None or supplied.get('characteristic_change_threshold', 0.) == 0:
        raise ValueError('Для включённого порога задайте конечное значение от 1 до 10 %.')
    supplied.update(WEB_INPUT_CONTRACT)
    return _validate_config(supplied)
