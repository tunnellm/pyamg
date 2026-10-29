"""Budgeted adaptive K/AMLI applications.

Work is charged once per completed operation, across the entire recursive call.
Schwarz uses configured native-loop counts. GS, coarse solves and norm monitoring
retain explicitly named arithmetic models; these are not hardware counters.
"""
from collections import Counter
from dataclasses import dataclass
import math
import numbers

import numpy as np
from .util.sparse_blas import with_serial_blas


class ApplicationWorkLimit(Exception):
    def __init__(self, stage, required, spent, limit):
        self.stage, self.required, self.spent, self.limit = stage, required, spent, limit
        super().__init__(f'{stage}: need {required}, remaining {limit-spent}')


class Ledger:
    def __init__(self, limit=math.inf):
        if math.isnan(limit) or limit < 0:
            raise ValueError('work limit must be nonnegative')
        self.limit = limit
        self.spent = 0
        self.completed = Counter()
        self.failed_operation = None

    def perform(self, stage, count, function):
        count = int(count)
        if count < 0:
            raise ValueError('negative operation cost')
        if count > self.limit-self.spent:
            raise ApplicationWorkLimit(stage, count, self.spent, self.limit)
        try:
            result = function()
        except Exception:
            # A kernel that raises may have executed only part of its work.
            self.failed_operation = dict(stage=stage, reserved_work=count)
            raise
        self.spent += count
        self.completed[stage] += count
        return result


class Operations:
    def __init__(self, ledger):
        self.ledger = ledger

    def mv(self, A, x):
        return self.ledger.perform('spmv', A.nnz, lambda: _matvec(A, x))

    def inner(self, x, y):
        return self.ledger.perform('dot', x.size, lambda: np.inner(x, y))

    def subtract(self, x, y):
        return self.ledger.perform('vector_add', x.size, lambda: x-y)

    def difference_scaled(self, x, alpha, y):
        return self.ledger.perform('axpy', x.size, lambda: x-alpha*y)

    def update(self, y, x, alpha=None, subtract=False):
        def apply():
            if alpha is None:
                y[:] += x
            elif subtract:
                y[:] -= alpha*x
            else:
                y[:] += alpha*x
        self.ledger.perform('vector_add' if alpha is None else 'axpy', y.size, apply)

    def norm(self, x):
        def stable_norm():
            if not np.all(np.isfinite(x)):
                raise np.linalg.LinAlgError('nonfinite inner residual')
            scale = float(np.max(np.abs(x), initial=0.0))
            if scale == 0:
                return 0.0
            scaled = x.ravel()/scale
            return scale*math.sqrt(float(np.inner(scaled, scaled)))
        # Fixed conservative charge for this norm implementation: scaling,
        # squared-norm reduction, sqrt, and rescaling. Comparisons are not FMA.
        return self.ledger.perform('monitor_norm_model', 2*x.size+2, stable_norm)



@dataclass
class ApplicationResult:
    status: str
    solution: object
    work: int
    components: dict
    accepted_ratios: list
    blocked: object = None


def _matvec(A, x):
    # Import lazily to avoid a multilevel/budgeted_cycle import cycle.
    from .multilevel import _matvec as matvec
    return matvec(A, x)


class BudgetedCycle:
    """A configured, real-FP64 adaptive application; use one active config per hierarchy."""

    def __init__(self, ml, cycle, tolerance, num_cycles=1):
        if cycle not in ('K', 'AMLI'):
            raise ValueError('budgeted cycles must be K or AMLI')
        if isinstance(tolerance, (bool, np.bool_)) or not math.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError('explicit inner tolerance must be in (0, 1)')
        if isinstance(num_cycles, (bool, np.bool_)) or not isinstance(num_cycles, numbers.Integral) or num_cycles < 1:
            raise ValueError('positive integer num_cycles required')
        self.ml, self.cycle, self.tolerance, self.num_cycles = ml, cycle, tolerance, num_cycles
        self.inner_steps = getattr(ml, '_k_cg_iters', 2) if cycle == 'K' else 2
        if isinstance(self.inner_steps, (bool, np.bool_)) or not isinstance(self.inner_steps, numbers.Integral) or self.inner_steps < 1:
            raise ValueError('positive integer inner iteration count required')
        # Cache only configuration-dependent counts, never RHS-dependent work.
        ml._precompute_k_costs()
        self.coarse_cost = ml._k_coarse_cost
        self.smoothers = []
        for i, level in enumerate(ml.levels[:-1]):
            self.smoothers.append((
                self._smoother(level.presmoother, level.A, i, 'pre'),
                self._smoother(level.postsmoother, level.A, i, 'post')))
        from .multilevel import _get_coarse_solver_name
        name = _get_coarse_solver_name(ml.coarse_solver)
        # A variable iterative coarse solve cannot be priced as a single SpMV.
        if name not in ('pinv', 'pinv2', 'lu', 'cholesky', 'splu'):
            raise ValueError('budgeted cycles require a supported direct coarse solver')

    @staticmethod
    def _smoother(smoother, A, level, phase):
        work = getattr(smoother, '_cappa_apply_work', None)
        if work is not None:
            cost = sum(work.values())
            stage = f'schwarz_{level}_{phase}'
        elif getattr(smoother, '__name__', '') == 'gauss_seidel' and hasattr(smoother, 'keywords'):
            opts = smoother.keywords
            cost = A.nnz * opts.get('iterations', 1)
            cost *= 2 if opts.get('sweep', 'forward') == 'symmetric' else 1
            stage = f'gauss_seidel_model_{level}_{phase}'
        else:
            raise ValueError('budgeted cycles currently support configured Schwarz and GS smoothers only')
        if not math.isfinite(cost) or cost < 0 or int(cost) != cost:
            raise ValueError('invalid configured smoother work')
        return smoother, stage, int(cost)

    @with_serial_blas
    def apply(self, b, limit=math.inf):
        b = np.asarray(b)
        if b.dtype != np.float64 or b.ndim != 1 or b.size != self.ml.levels[0].A.shape[0]:
            raise ValueError('budgeted application requires a matching real FP64 vector')
        if not np.all(np.isfinite(b)):
            raise np.linalg.LinAlgError('nonfinite application RHS')
        ledger = Ledger(limit)
        ops = Operations(ledger)
        accepted = []
        x = np.zeros_like(b)

        def ratio(numerator, denominator):
            if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
                raise np.linalg.LinAlgError(f'{self.cycle} inner breakdown: invalid direction product')
            def divide():
                with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                    value = numerator / denominator
                if not np.isfinite(value):
                    raise np.linalg.LinAlgError(f'{self.cycle} inner breakdown: nonfinite step')
                return value
            return ledger.perform('scalar_division', 1, divide)

        def accept_fresh(initial, actual):
            relative = actual/initial if initial else (0.0 if actual == 0 else math.inf)
            if math.isfinite(relative) and relative <= self.tolerance:
                accepted.append(relative)
                return True
            return False

        def done(A, y, rhs, residual):
            initial = ops.norm(rhs)
            current = ops.norm(residual)
            if not (current == 0 if initial == 0 else current/initial <= self.tolerance):
                return False
            actual = ops.norm(ops.subtract(rhs, ops.mv(A, y)))
            return accept_fresh(initial, actual)

        def coarse(A, rhs):
            return ledger.perform('coarse_apply_model', self.coarse_cost,
                                  lambda: self.ml.coarse_solver(A, rhs))

        def solve(lvl, y, rhs):
            level = self.ml.levels[lvl]
            A = level.A
            if lvl == len(self.ml.levels)-1:
                y[:] = coarse(A, rhs)
                return
            pre, post = self.smoothers[lvl]
            ledger.perform(pre[1], pre[2], lambda: pre[0](A, y, rhs))
            residual = ops.subtract(rhs, ops.mv(A, y))
            # This residual is already fresh. Do not create a new coarse
            # problem after smoothing has met this level's relative target.
            if accept_fresh(ops.norm(rhs), ops.norm(residual)):
                return
            coarse_b = ops.mv(level.R, residual)
            coarse_x = np.zeros_like(coarse_b)
            Ac = self.ml.levels[lvl+1].A
            if lvl == len(self.ml.levels)-2:
                coarse_x[:] = coarse(Ac, coarse_b)
            elif self.cycle == 'AMLI':
                p = np.zeros((self.inner_steps, coarse_b.size), dtype=coarse_b.dtype)
                original_rhs = coarse_b.copy()
                for k in range(self.inner_steps):
                    if done(Ac, coarse_x, original_rhs, coarse_b):
                        break
                    # An inner correction is a zero-start solve, not a solve
                    # with the unrelated all-ones initial guess.
                    solve(lvl+1, p[k, :], coarse_b)
                    for j in range(k):
                        beta = ratio(ops.inner(p[j, :], ops.mv(Ac, p[k, :])),
                                     ops.inner(p[j, :], ops.mv(Ac, p[j, :])))
                        ops.update(p[k, :], p[j, :], beta, subtract=True)
                    Ap = ops.mv(Ac, p[k, :])
                    alpha = ratio(ops.inner(p[k, :], coarse_b), ops.inner(p[k, :], Ap))
                    ops.update(coarse_x, p[k, :], alpha)
                    ops.update(coarse_b, Ap, alpha, subtract=True)
            else:
                r = coarse_b.copy()
                w = np.zeros_like(r)
                for k in range(self.inner_steps):
                    if done(Ac, coarse_x, coarse_b, r):
                        break
                    w[:] = 0
                    solve(lvl+1, w, r)
                    if k == 0:
                        d = w.copy()
                    else:
                        beta = ratio(ops.inner(w, Ad), dAd)
                        d = ops.difference_scaled(w, beta, d)
                    Ad = ops.mv(Ac, d)
                    dAd = ops.inner(d, Ad)
                    alpha = ratio(ops.inner(d, r), dAd)
                    ops.update(coarse_x, d, alpha)
                    ops.update(r, Ad, alpha, subtract=True)
            ops.update(y, ops.mv(level.P, coarse_x))
            ledger.perform(post[1], post[2], lambda: post[0](A, y, rhs))

        try:
            for _ in range(self.num_cycles):
                solve(0, x, b)
        except ApplicationWorkLimit as error:
            return ApplicationResult('WORK_LIMIT', None, ledger.spent,
                                     dict(ledger.completed), accepted,
                                     dict(stage=error.stage, required=error.required,
                                          remaining=error.limit-error.spent))
        if not np.all(np.isfinite(x)):
            raise np.linalg.LinAlgError('nonfinite application output')
        return ApplicationResult('COMPLETE', x, ledger.spent, dict(ledger.completed), accepted)


def apply(ml, b, cycle, tolerance, limit=math.inf, num_cycles=1):
    """Convenience entry point; wrappers should retain a configured BudgetedCycle."""
    return BudgetedCycle(ml, cycle, tolerance, num_cycles).apply(b, limit)
