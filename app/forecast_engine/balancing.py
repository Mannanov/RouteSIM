"""Feasible transport support, forced zeros and logarithmic IPF."""
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import logsumexp

from .contracts import ForecastCalculationError, _diagnostics


def transport_flow(support, rows, columns):
    """Continuous maximum flow with a residual min-cut certificate.

    Capacities retain their floating point values, including small positives.
    The matrix is a feasibility witness, not an OD prediction.
    """
    rows, columns = np.asarray(rows, float), np.asarray(columns, float)
    support = np.asarray(support, bool)
    nr, nc = support.shape
    ri, ci = np.nonzero(support)
    total = max(float(rows.sum()), float(columns.sum()))
    # Строим сеть: исток→отправления→прибытия→сток.
    # емкости крайних ребер равны районным целям, внутренних — общему объему.
    source, sink = nr + nc, nr + nc + 1
    graph = [[] for _ in range(sink + 1)]

    def edge(start, end, capacity):
        """Добавляет прямое ребро с емкостью и обратное ребро остаточной сети.
        Индексы парных ребер позволяют обновлять остаток при увеличении потока.
        """
        forward = [end, len(graph[end]), float(capacity)]
        backward = [start, len(graph[start]), 0.0]
        graph[start].append(forward)
        graph[end].append(backward)
        return forward

    re = [edge(source, i, x) for i, x in enumerate(rows)]
    transports = [edge(int(i), nr + int(j), total) for i, j in zip(ri, ci)]
    ce = [edge(nr + j, sink, x) for j, x in enumerate(columns)]
    flow = 0.0
    # Максимальный поток ищем алгоритмом Диница: BFS строит уровни, send увеличивает поток.
    while True:
        levels = [-1] * len(graph)
        levels[source] = 0
        queue = deque([source])
        while queue:
            vertex = queue.popleft()
            for e in graph[vertex]:
                if e[2] > 0 and levels[e[0]] < 0:
                    levels[e[0]] = levels[vertex] + 1
                    queue.append(e[0])
        if levels[sink] < 0:
            break
        positions = [0] * len(graph)

        def send(vertex, amount):
            """Продвигает поток по уровневой сети алгоритма Диница.

            Возвращает отправленный объем и обновляет емкости прямых и обратных ребер.
            """
            if vertex == sink:
                return amount
            while positions[vertex] < len(graph[vertex]):
                e = graph[vertex][positions[vertex]]
                if e[2] > 0 and levels[e[0]] == levels[vertex] + 1:
                    pushed = send(e[0], min(amount, e[2]))
                    if pushed > 0:
                        e[2] -= pushed
                        graph[e[0]][e[1]][2] += pushed
                        return pushed
                positions[vertex] += 1
            return 0.0

        while True:
            pushed = send(source, float('inf'))
            if pushed <= 0:
                break
            flow += pushed
    matrix = np.zeros((nr, nc))
    matrix[ri, ci] = [graph[e[0]][e[1]][2] for e in transports]
    reachable = np.zeros(len(graph), dtype=bool)
    reachable[source] = True
    queue = deque([source])
    while queue:
        vertex = queue.popleft()
        for e in graph[vertex]:
            if e[2] > 0 and not reachable[e[0]]:
                reachable[e[0]] = True
                queue.append(e[0])
    cut_rows, cut_columns = reachable[:nr], reachable[nr:nr + nc]
    cut_value = float(rows[~cut_rows].sum() + columns[cut_columns].sum())
    return matrix, cut_rows, cut_columns, cut_value


def prepare_support(support, rows, columns, tolerance, context=None):
    """Проверяет достижимость районных целей на разрешенных OD-связях.
    При полной достижимости максимального потока выявляет вынужденные нули
    через остаточную сеть. Возвращает уточненную опору и BOUNDARY_SUPPORT;
    невозможные цели приводят к ошибке с районными остатками.
    """
    rows, columns = np.asarray(rows, float), np.asarray(columns, float)
    support = np.asarray(support, bool).copy()
    if (not np.isfinite(rows).all() or not np.isfinite(columns).all()
            or np.any(rows < 0) or np.any(columns < 0)):
        raise ValueError('Цели балансировки должны быть конечными и неотрицательными')
    if support.shape != (len(rows), len(columns)):
        raise ValueError('Опора не соответствует размеру районных целей')
    if abs(rows.sum() - columns.sum()) > tolerance:
        raise ForecastCalculationError('IPF_TOTAL_MISMATCH', _diagnostics(
            'IPF_TOTAL_MISMATCH', rows, columns, np.zeros_like(rows),
            np.zeros_like(columns), context))
    # Строки и столбцы с нулевой целью не входят в активную транспортную опору.
    support &= (rows > 0)[:, None] & (columns > 0)[None, :]
    total = float(rows.sum())
    if total == 0:
        return support, []
    # Для полной опоры положительных целей достижимость известна заранее.
    # Для полной опоры без диагонали используем строгие условия Холла.
    nr, nc = len(rows), len(columns)
    ri, ci = np.nonzero(support)
    active_r, active_c = np.flatnonzero(rows > 0), np.flatnonzero(columns > 0)
    if len(ri) == len(active_r) * len(active_c):
        return support, []
    numerical = np.finfo(float).eps * max(1.0, total) * max(nr + nc, 8) * 8
    shared = np.intersect1d(active_r, active_c)
    if (nr == nc and not np.any(ri == ci)
            and len(ri) == len(active_r) * len(active_c) - len(shared)
            and np.all(rows[shared] + columns[shared] < total - numerical)):
        return support, []
    witness, _, _, _ = transport_flow(support, rows, columns)
    flow = float(witness.sum())
    # Недостижимые цели возвращаем с фактически проведенным объемом и остатками.
    achieved_r = witness.sum(axis=1)
    achieved_c = witness.sum(axis=0)
    if total - flow > tolerance:
        raise ForecastCalculationError('INFEASIBLE_SUPPORT', _diagnostics(
            'INFEASIBLE_SUPPORT', rows, columns, achieved_r, achieved_c, context))
    # Вынужденные нули определяем только при потоке, полном до машинной точности.
    # Достижение пользовательского допуска само по себе не дает права сужать опору.
    if total - flow > numerical:
        return support, []
    # В остаточной сети прямое ребро допускает увеличение, обратное — уменьшение текущего потока.
    values = witness[ri, ci]
    # A computed maximum flow can leave round-off on a boundary edge.
    # Scale its numerical-zero check by the local possible flow, preserving
    # genuinely small goals instead of applying a global OD intensity floor.
    edge_error = np.finfo(float).eps * max(nr + nc, 8) * 8 * np.minimum(rows[ri], columns[ci])
    residual_from, residual_to = [], []
    for i, j, value, error in zip(ri, ci, values, edge_error):
        if total - value > 0:
            residual_from.append(int(i)); residual_to.append(nr + int(j))
        if value > error:
            residual_from.append(nr + int(j)); residual_to.append(int(i))
    residual = csr_matrix((np.ones(len(residual_from)), (residual_from, residual_to)),
                          shape=(nr + nc, nr + nc))
    # Нулевое ребро между разными сильными компонентами не входит в цикл перераспределения.
    # Такой ноль вынужден целями; исключаем его перед IPF, сохраняя диагностику границы.
    _, labels = connected_components(residual, directed=True, connection='strong')
    forced = (values <= edge_error) & (labels[ri] != labels[nr + ci])
    support[ri[forced], ci[forced]] = False
    row_labels = (context or {}).get('row_labels', range(nr))
    col_labels = (context or {}).get('column_labels', range(nc))
    diagnostics = [{
        **{k: v for k, v in (context or {}).items() if not k.endswith('_labels')},
        'source': row_labels[i], 'target': col_labels[j],
        'status': 'BOUNDARY_SUPPORT', 'reason': 'FORCED_ZERO',
    } for i, j in zip(ri[forced], ci[forced])]
    # Найденный полный поток не использует удаленные ребра: достижимость целей
    # сохраняется после исключения вынужденных нулей.
    return support, diagnostics


def balance(log_seed, rows, columns, tolerance=.01, max_iterations=1000,
            context=None, prepared=None):
    """Балансирует логарифмическую затравку IPF по целям отправлений и прибытий.
    Нули опоры задаются -inf; достижимость проверяется либо передается в prepared.
    Возвращает матрицу, число итераций, максимальный остаток и вынужденные нули.
    Достижение лимита итераций без допустимого остатка завершает расчет ошибкой.
    """
    rows, columns = np.asarray(rows, float), np.asarray(columns, float)
    log_seed = np.asarray(log_seed, float)
    # Для повторных условных расчетов можно переиспользовать опору с теми же целями.
    if prepared is None:
        prepared = prepare_support(np.isfinite(log_seed), rows, columns, tolerance, context)
    support, boundary = prepared
    if rows.sum() == 0:
        return np.zeros_like(log_seed), 0, 0.0, boundary
    # Балансируем только положительные цели, чтобы не вычислять log(0).
    active_r, active_c = np.flatnonzero(rows > 0), np.flatnonzero(columns > 0)
    sub = np.ix_(active_r, active_c)
    kernel = np.where(support[sub], log_seed[sub], -np.inf)
    # Постоянные сдвиги строк поглощаются множителями IPF и улучшают числовой диапазон.
    row_offset = np.max(kernel, axis=1)
    if not np.isfinite(row_offset).all():
        raise ForecastCalculationError('INFEASIBLE_SUPPORT', _diagnostics(
            'INFEASIBLE_SUPPORT', rows, columns, np.zeros_like(rows),
            np.zeros_like(columns), context))
    kernel = kernel - row_offset[:, None]
    log_r, log_c = np.log(rows[active_r]), np.log(columns[active_c])
    v = np.zeros(len(active_c))
    result = np.zeros_like(log_seed)
    # Чередуем нормировку строк и столбцов через logsumexp, избегая потери ядра exp(-beta·time).
    for iteration in range(1, max_iterations + 1):
        u = log_r - logsumexp(kernel + v[None, :], axis=1)
        v = log_c - logsumexp(kernel + u[:, None], axis=0)
        values = np.exp(kernel + u[:, None] + v[None, :])
        # Успех определяет максимальный абсолютный районный остаток в поездках.
        error = max(np.abs(values.sum(axis=1) - rows[active_r]).max(initial=0),
                    np.abs(values.sum(axis=0) - columns[active_c]).max(initial=0))
        if error <= tolerance:
            result[sub] = values
            return result, iteration, float(error), boundary
    # Лимит итераций не означает успех: сохраняем последнюю матрицу только для диагностики ошибки.
    result[sub] = values
    raise ForecastCalculationError('IPF_NOT_CONVERGED', _diagnostics(
        'IPF_NOT_CONVERGED', rows, columns, result.sum(axis=1),
        result.sum(axis=0), context, max_iterations))


def _ipf(seed, row_targets, column_targets, tolerance, max_iterations, context=None):
    """Обертка IPF для обычной неотрицательной затравки.
    Преобразует положительные элементы в логарифмы, нули в -inf и возвращает
    матрицу, число итераций и максимальный остаток в компактном формате.
    """
    logs = np.full_like(np.asarray(seed, float), -np.inf)
    np.log(seed, out=logs, where=np.asarray(seed) > 0)
    fitted, iterations, error, _ = balance(logs, row_targets, column_targets, tolerance, max_iterations, context)
    return fitted, iterations, error
