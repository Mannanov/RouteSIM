"""Shared forecast failures and diagnostic columns, independent of web context."""
from dataclasses import dataclass

import numpy as np
import pandas as pd



FAILURE_COLUMNS = [
    'stage', 'state', 'hour', 'district', 'direction', 'target',
    'achieved', 'residual', 'iterations', 'status', 'reason',
]


class ForecastCalculationError(ValueError):
    """Ошибка обязательного этапа расчета с таблицей диагностики.
    Позволяет сохранить причины отказа и районные остатки отдельно от результата.
    """

    def __init__(self, message, diagnostics=None):
        """Сохраняет текст ошибки и приводит записи диагностики к DataFrame."""
        super().__init__(message)
        self.diagnostics = pd.DataFrame(diagnostics if diagnostics is not None else [])


def failure(status, stage, reason, **context):
    """Создает ошибку с одной диагностической записью.
    status задает машинный код, stage — этап расчета, context — час, район и другие детали.
    """
    return ForecastCalculationError(reason, [{
        'status': status, 'stage': stage, 'reason': reason, **context,
    }])


def _diagnostics(status, rows, columns, achieved_rows, achieved_columns, context, iterations=0):
    """Создает записи отправлений и прибытий с целями, достижением и остатками.
    Добавляет этап, час, состояние и число итераций из контекста балансировки.
    """
    common = {**(context or {}), 'status': status, 'reason': status, 'iterations': iterations}
    row_labels = common.pop('row_labels', range(len(rows)))
    column_labels = common.pop('column_labels', range(len(columns)))
    return [{**common, 'district': label, 'direction': direction,
             'target': float(target), 'achieved': float(achieved),
             'residual': float(target - achieved)}
            for direction, labels, targets, achieved_values in (
                ('outgoing', row_labels, rows, achieved_rows),
                ('incoming', column_labels, columns, achieved_columns))
            for label, target, achieved in zip(labels, targets, achieved_values)]


@dataclass(frozen=True, kw_only=True)
class ForecastReportData:
    """Calculated data passed from the pipeline to report assembly."""

    inputs: dict
    base: pd.DataFrame
    future: pd.DataFrame
    geodata: pd.DataFrame
    metric_crs: str
    fit: dict
    base_model: dict
    forecast_model: dict
    observed: np.ndarray
    result: pd.DataFrame
    changes: pd.DataFrame
    margins: list
    controls: list
    boundary: list
    exceptions: list
    pair_los: pd.DataFrame
    donors: pd.DataFrame
    candidates: pd.DataFrame
    time_model: dict
    characteristic_changes: pd.DataFrame
    effective_characteristics: pd.DataFrame
    suppressed_changes: pd.DataFrame
    input_diagnostics: list
    retain: bool
    retain_reason: str
    pivot_records: list
    target_records: list
    validation: dict
    legacy_predictor: object


@dataclass(frozen=True, kw_only=True)
class ForecastReportChecks:
    """Control and applicability results used by report assembly."""

    diagnostics: list
    controls_frame: pd.DataFrame
    applicability: pd.DataFrame
    analogs: pd.DataFrame
    transfer_status: str
