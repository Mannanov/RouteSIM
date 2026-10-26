"""Maximum distributable mass and least relative reduction of model margins."""
import numpy as np
from scipy.optimize import least_squares, linprog, minimize
from scipy.sparse import bmat, coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

from .balancing import transport_flow


class _ReconciliationFailure(ValueError):
    """Expected failure of a numerical certificate or optimization budget."""

    def __init__(self, message, stage='quadratic'):
        super().__init__(message)
        self.stage = stage


def _component_targets(mask, outgoing, incoming, tolerance, max_iterations):
    """Solve in marginal fractions; add violated transport cuts as needed.

    The QP has at most nr+nc variables, rather than one variable per OD pair.
    A feasible proportional solution attains an analytic lower bound directly.
    """
    scale = max(float(outgoing.sum()), float(incoming.sum()))
    o, d = outgoing / scale, incoming / scale
    witness, cut_r, cut_c, cut = transport_flow(mask, o, d)
    maximum = float(witness.sum())
    certificate = {'rows': np.flatnonzero(cut_r).tolist(),
                   'columns': np.flatnonzero(cut_c).tolist(), 'capacity': cut * scale,
                   'flow': maximum * scale}
    if abs(maximum - cut) > tolerance:
        raise _ReconciliationFailure('Maximum-flow certificate failed', stage='max_flow')
    nr, nc = mask.shape
    caps = np.r_[o, d]
    proportional = np.r_[np.full(nr, maximum / o.sum()), np.full(nc, maximum / d.sum())]
    fractions = proportional.copy()
    cuts = []
    equalities = np.zeros((2, nr + nc))
    equalities[0, :nr], equalities[1, nr:] = o, d
    # Every maximum flow saturates the source/sink edges of this min cut.
    # Eliminate fixed fractions before optimizing on the maximum-flow face.
    fixed = np.r_[~cut_r, cut_c]
    free = ~fixed
    pinned = fixed.astype(float)
    equality = equalities[:, free]
    rhs = maximum - equalities @ pinned
    nonempty = np.any(equality != 0, axis=1)
    equality, rhs = equality[nonempty], rhs[nonempty]
    eq_scale = np.max(abs(equality), axis=1) if len(equality) else np.empty(0)
    equality, rhs = equality / eq_scale[:, None], rhs / eq_scale

    def expand(z):
        value = pinned.copy()
        value[free] = z
        return value
    last_gap = 0.0
    for iteration in range(max_iterations + 1):
        rows, columns = o * fractions[:nr], d * fractions[nr:]
        flow, reachable_r, reachable_c, flow_cut = transport_flow(mask, rows, columns)
        # The check concerns the requested margins, not just the global mass.
        residual = max(np.max(abs(flow.sum(axis=1) - rows) / np.maximum(rows, 1e-15)),
                       np.max(abs(flow.sum(axis=0) - columns) / np.maximum(columns, 1e-15)))
        if residual <= tolerance:
            if not cuts:
                # Constant fractions minimize the separable quadratic subject
                # to the component's fixed mass; feasibility certifies optimality.
                return rows * scale, columns * scale, maximum * scale, cut * scale, 0., iteration, certificate
            # Independent first-order certificate over the entire transport
            # polytope: a small Frank-Wolfe gap bounds objective suboptimality.
            ri, ci = np.nonzero(mask)
            edges = len(ri)
            incidence = coo_matrix((np.ones(2 * edges), (np.r_[ri, nr + ci],
                np.tile(np.arange(edges), 2))), shape=(nr + nc, edges)).tocsr()
            gradient = 2 * (fractions - 1)
            oracle = linprog(gradient[ri] + gradient[nr + ci], A_ub=incidence, b_ub=caps,
                A_eq=np.ones((1, edges)), b_eq=[maximum], bounds=(0, None), method='highs')
            if not oracle.success:
                raise _ReconciliationFailure('Quadratic optimality certificate unavailable: ' + oracle.message)
            if (np.max(incidence @ oracle.x - caps) > tolerance
                    or abs(oracle.x.sum() - maximum) > tolerance
                    or np.min(oracle.x) < -tolerance):
                raise _ReconciliationFailure('Invalid quadratic optimality witness')
            signed_gap = float(gradient @ np.r_[rows, columns] - oracle.fun)
            if signed_gap < -tolerance:
                raise _ReconciliationFailure('Invalid quadratic lower bound')
            last_gap = max(0., signed_gap)
            if last_gap <= tolerance:
                return rows * scale, columns * scale, maximum * scale, cut * scale, last_gap, iteration, certificate
            raise _ReconciliationFailure(f'Quadratic optimality gap: {last_gap}')
        if iteration == max_iterations:
            break
        # A min cut separates infeasible requested margins: r(S) <= c(N(S)).
        constraint = np.r_[o * reachable_r, -d * reachable_c]
        if any(np.array_equal(constraint, previous) for previous in cuts):
            raise _ReconciliationFailure('Repeated infeasible transport cut')
        cuts.append(constraint)
        inequalities = np.asarray(cuts)
        ineq_scale = np.max(abs(inequalities), axis=1)
        inequalities = inequalities / ineq_scale[:, None]
        # Box-constrained primal QP has a smooth dual in the multipliers
        # of the marginal equalities and transport cuts. Analytic clipping
        # avoids singular active-boundary subproblems in a primal QP.
        matrix = np.vstack([equality, inequalities[:, free]])
        constant = np.r_[-rhs, inequalities @ pinned]
        weights = caps[free]
        def dual(multipliers):
            linear = matrix.T @ multipliers
            z = np.clip(1 - linear / (2 * weights), 0, 1)
            value = np.sum(weights * (1 - z) ** 2) + linear @ z + constant @ multipliers
            return -float(value), -(matrix @ z + constant)
        result = minimize(dual, np.zeros(len(constant)), jac=True, method='L-BFGS-B',
            bounds=[(None, None)] * len(equality) + [(0., None)] * len(cuts),
            options={'ftol': 1e-15, 'gtol': tolerance * .001,
                     'maxiter': max_iterations, 'maxls': 100})
        if not np.isfinite(result.x).all():
            raise _ReconciliationFailure('Quadratic target reconciliation failed: ' + result.message)
        def optimality(multipliers, jacobian=False):
            z = np.clip(1 - (matrix.T @ multipliers) / (2 * weights), 0, 1)
            gradient = -(matrix @ z + constant)
            active = (z > 0) & (z < 1)
            hessian = (matrix[:, active] / (2 * weights[active])) @ matrix[:, active].T
            for k in range(len(equality), len(constant)):
                if multipliers[k] < gradient[k]:
                    gradient[k] = multipliers[k]
                    hessian[k] = 0
                    hessian[k, k] = 1
            return hessian if jacobian else gradient
        refined = least_squares(optimality, result.x, jac=lambda v: optimality(v, True),
            ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=max_iterations)
        fractions = expand(np.clip(1 - (matrix.T @ refined.x) / (2 * weights), 0, 1))
    raise _ReconciliationFailure('Target reconciliation cut iteration limit reached')


def reconcile_targets(mask, outgoing, incoming, *, tolerance=1e-9, max_iterations=500):
    """Return independently checked feasible margins or an explicit failure.

    Raw model targets and transport restrictions are never modified in place.
    Only valid model targets may be reduced; invalid input remains an error.
    """
    o, d = np.asarray(outgoing, float), np.asarray(incoming, float)
    mask = np.asarray(mask, bool)
    if (o.ndim != 1 or d.ndim != 1 or mask.shape != (len(o), len(d))
            or not np.isfinite(o).all() or not np.isfinite(d).all()
            or np.any(o < 0) or np.any(d < 0)):
        raise ValueError('Invalid reconciliation targets or mask')
    total = float((o.sum() + d.sum()) / 2)
    if abs(o.sum() - d.sum()) > tolerance * max(total, np.finfo(float).tiny):
        raise ValueError('Reconciliation target totals differ')
    result = {'status': 'TARGET_RECONCILIATION_FAILED', 'outgoing': None, 'incoming': None,
              'Q_model': total, 'Q_max': None, 'Q_feasible': None, 'Q_reduced': None,
              'retained_share': None, 'objective': None, 'optimality_gap': None,
              'cut_capacity': None, 'residual': None, 'iterations': 0, 'reason': '',
              'tolerance': tolerance, 'max_iterations': max_iterations,
              'max_flow_status': 'NOT_RUN', 'quadratic_status': 'NOT_RUN'}
    adjusted_o, adjusted_d = np.zeros_like(o), np.zeros_like(d)
    support = mask & (o > 0)[:, None] & (d > 0)[None, :]
    graph = bmat([[None, csr_matrix(support)], [csr_matrix(support.T), None]], format='csr')
    _, labels = connected_components(graph, directed=False)
    capacity, gap, iterations = 0., 0., 0
    certificates = []
    try:
        for label in np.unique(labels):
            rows = np.flatnonzero((labels[:len(o)] == label) & (o > 0))
            columns = np.flatnonzero((labels[len(o):] == label) & (d > 0))
            if not len(rows) or not len(columns):
                continue
            ro, rd, maximum, cut, error, steps, certificate = _component_targets(
                support[np.ix_(rows, columns)], o[rows], d[columns], tolerance, max_iterations)
            adjusted_o[rows], adjusted_d[columns] = ro, rd
            certificates.append({**certificate, 'component_rows': rows.tolist(),
                                 'component_columns': columns.tolist(), 'rows': rows[certificate['rows']].tolist(),
                                 'columns': columns[certificate['columns']].tolist()})
            capacity += cut
            gap = max(gap, error)
            iterations += steps
    except _ReconciliationFailure as error:
        result.update(reason=str(error), max_flow_status='FAILED' if error.stage == 'max_flow' else 'PARTIAL',
                      quadratic_status='FAILED' if error.stage == 'quadratic' else 'NOT_RUN')
        return result
    feasible = float(adjusted_o.sum())
    violation = max(np.max(np.maximum(adjusted_o - o, 0) / np.maximum(o, 1)),
                    np.max(np.maximum(adjusted_d - d, 0) / np.maximum(d, 1)),
                    abs(adjusted_o.sum() - adjusted_d.sum()) / max(total, np.finfo(float).tiny),
                    abs(feasible - capacity) / max(total, np.finfo(float).tiny))
    if violation > tolerance:
        result['reason'] = f'Reconciliation constraint residual: {violation}'
        return result
    # Preserve original feasible targets exactly; the preparation of the IPF
    # support will still check them independently before scaling weights.
    reduced = max(0., total - feasible)
    if total > 0 and reduced <= np.finfo(float).eps * total * 32:
        adjusted_o, adjusted_d = o.copy(), d.copy()
        feasible, reduced = total, 0.
    adjusted_o, adjusted_d = np.minimum(adjusted_o, o), np.minimum(adjusted_d, d)
    loss_o, loss_d = o - adjusted_o, d - adjusted_d
    objective = float(np.sum(np.divide(loss_o ** 2, o, out=np.zeros_like(o), where=o > 0))
                      + np.sum(np.divide(loss_d ** 2, d, out=np.zeros_like(d), where=d > 0)))
    result.update(status='ZERO_FEASIBLE_VOLUME' if feasible == 0 else (
                      'TARGETS_REDUCED' if reduced > 0 else 'TARGETS_UNCHANGED'),
        outgoing=adjusted_o, incoming=adjusted_d, R_O=loss_o, R_D=loss_d,
        Q_max=capacity, Q_feasible=feasible, Q_reduced=reduced,
        retained_share=feasible / total if total > 0 else None, objective=objective,
        optimality_gap=gap, cut_capacity=capacity, residual=float(violation), iterations=iterations,
        max_flow_status='OK', quadratic_status='OK', cut_certificates=certificates,
        optimality_gap_units='max_component_normalized_objective',
        isolated_outgoing=np.flatnonzero((o > 0) & ~support.any(axis=1)).tolist(),
        isolated_incoming=np.flatnonzero((d > 0) & ~support.any(axis=0)).tolist())
    return result
