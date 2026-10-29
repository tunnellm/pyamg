"""Adaptive application budgets must not publish a partial direction."""
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import eye, diags

import pyamg
from pyamg.gallery import poisson
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.budgeted_cycle import BudgetedCycle, Ledger, ApplicationWorkLimit


@pytest.mark.parametrize('cycle', ['K', 'AMLI'])
@pytest.mark.parametrize('builder', [pyamg.ruge_stuben_solver,
    pyamg.smoothed_aggregation_solver, pyamg.rootnode_solver, pyamg.pairwise_solver])
@pytest.mark.parametrize('smoother', ['gauss_seidel', 'schwarz', 'strength_based_schwarz'])
def test_budget_boundaries(builder, cycle, smoother):
    np.random.seed(42)
    ml = builder(poisson((8, 8), format='csr'), max_coarse=2)
    setting = (smoother, dict(sweep='symmetric', iterations=1))
    change_smoothers(ml, setting, setting)
    driver = BudgetedCycle(ml, cycle, 1e-3, num_cycles=2)
    b = np.linspace(0.1, 2, 64)
    saved = b.copy()
    full = driver.apply(b)
    assert full.status == 'COMPLETE'
    assert full.work == sum(full.components.values())
    assert all(r <= 1e-3 for r in full.accepted_ratios)
    for limit in (0, 1, full.work//2, full.work-1):
        partial = driver.apply(b, limit)
        assert partial.status == 'WORK_LIMIT'
        assert partial.solution is None
        assert partial.work <= limit
        assert sum(partial.components.values()) == partial.work
        assert partial.blocked['remaining'] == limit-partial.work
        assert partial.blocked['required'] > partial.blocked['remaining']
    exact = driver.apply(b, full.work)
    assert exact.status == 'COMPLETE'
    np.testing.assert_array_equal(full.solution, exact.solution)
    np.testing.assert_array_equal(b, saved)


@pytest.mark.parametrize('cycle', ['K', 'AMLI'])
@pytest.mark.parametrize('smoother', ['schwarz', 'strength_based_schwarz'])
def test_schwarz_counts_match_executed_native_loops(cycle, smoother):
    ml = pyamg.ruge_stuben_solver(poisson((8, 8), format='csr'), max_coarse=2)
    setting = (smoother, dict(sweep='symmetric', iterations=2))
    change_smoothers(ml, setting, setting)
    driver = BudgetedCycle(ml, cycle, 1e-3)
    native = pyamg.amg_core.overlapping_schwarz_csr
    measured = []

    def traced(*args):
        Ap, rows, ptr = args[0], args[7], args[8]
        cost = 0
        for d in range(*map(int, args[11:14])):
            domain = rows[ptr[d]:ptr[d+1]]
            cost += sum(int(Ap[r+1]-Ap[r]) for r in domain)
            cost += len(domain)**2 + 2*len(domain)
        measured.append(cost)
        return native(*args)

    with patch.object(pyamg.amg_core, 'overlapping_schwarz_csr', traced):
        result = driver.apply(np.ones(64))
    assert sum(v for k, v in result.components.items() if k.startswith('schwarz_')) == sum(measured)
    with patch.object(pyamg.amg_core, 'overlapping_schwarz_csr', side_effect=AssertionError):
        result = driver.apply(np.ones(64), 0)
    assert result.work == 0


@pytest.mark.parametrize('cycle', ['K', 'AMLI'])
def test_zero_and_underflow_are_distinct(cycle):
    levels = []
    for i in range(3):
        level = SimpleNamespace(A=eye(2, format='csr'))
        if i < 2:
            level.P = level.R = eye(2, format='csr')
            def noop(A, x, b):
                pass
            noop._cappa_apply_work = dict(residual_fma=0, dense_fma=0, scalar_adds=0)
            level.presmoother = level.postsmoother = noop
        levels.append(level)
    ml = pyamg.multilevel.MultilevelSolver(levels, coarse_solver='pinv')
    driver = BudgetedCycle(ml, cycle, 1e-3)
    zero = driver.apply(np.zeros(2))
    assert zero.status == 'COMPLETE'
    np.testing.assert_array_equal(zero.solution, np.zeros(2))
    assert zero.accepted_ratios and all(r == 0 for r in zero.accepted_ratios)
    with pytest.raises(np.linalg.LinAlgError):
        driver.apply(np.full(2, 1e-170))


def test_ledger_and_configuration_rejection():
    ledger = Ledger(5)
    assert ledger.perform('first', 3, lambda: 42) == 42
    with pytest.raises(ApplicationWorkLimit):
        ledger.perform('second', 3, lambda: pytest.fail('must not execute'))
    assert ledger.spent == 3
    assert dict(ledger.completed) == {'first': 3}
    ml = pyamg.ruge_stuben_solver(poisson((8, 8), format='csr'), max_coarse=2)
    for tolerance in (0, 1, -1, np.nan, True):
        with pytest.raises(ValueError):
            BudgetedCycle(ml, 'K', tolerance)
    with pytest.raises(ValueError):
        BudgetedCycle(ml, 'V', 1e-3)
    ml.coarse_solver = lambda A, b: b
    with pytest.raises(ValueError, match='direct coarse solver'):
        BudgetedCycle(ml, 'K', 1e-3)


def test_application_scopes_blas_and_restores_on_all_exits():
    from threadpoolctl import threadpool_info, threadpool_limits
    import pyamg.budgeted_cycle as budgeted
    ml = pyamg.ruge_stuben_solver(poisson((8, 8), format='csr'), max_coarse=2)
    driver = BudgetedCycle(ml, 'K', 1e-3)
    counts = lambda: {p['filepath']: p['num_threads'] for p in threadpool_info()
                      if p['user_api'] == 'blas'}
    native = budgeted._matvec
    visited = []

    def traced(A, x):
        visited.append(True)
        assert counts() and set(counts().values()) == {1}
        return native(A, x)

    with threadpool_limits(limits=2, user_api='blas'):
        before = counts()
        with patch.object(budgeted, '_matvec', traced):
            assert driver.apply(np.ones(64)).status == 'COMPLETE'
            assert driver.apply(np.ones(64), 0).status == 'WORK_LIMIT'
            with pytest.raises(np.linalg.LinAlgError):
                driver.apply(np.full(64, np.nan))
        assert visited
        assert counts() == before


@pytest.mark.parametrize('cycle', ['K', 'AMLI'])
def test_deep_chain_returns_before_underflow(cycle):
    n = 10000
    A = diags([-np.ones(n-1), 4*np.ones(n), -np.ones(n-1)], [-1, 0, 1], format='csr')
    ml = pyamg.ruge_stuben_solver(A, max_coarse=4)
    assert len(ml.levels) >= 10
    ml._k_cg_iters = 2
    b = np.random.default_rng(1).normal(size=n)
    result = BudgetedCycle(ml, cycle, 1e-3).apply(b)
    assert result.status == 'COMPLETE'
    assert all(np.isfinite(result.solution))
    assert result.accepted_ratios and all(r <= 1e-3 for r in result.accepted_ratios)
    assert 'gauss_seidel_model_2_pre' in result.components
    assert 'gauss_seidel_model_3_pre' not in result.components
    assert 'gauss_seidel_model_2_post' not in result.components
    assert 'coarse_apply_model' not in result.components
    assert result.work == sum(result.components.values())
    assert np.linalg.norm(b-A@result.solution) < np.linalg.norm(b)


@pytest.mark.parametrize('cycle', ['K', 'AMLI'])
@pytest.mark.parametrize('relative_error', [0.0, 2**-10, 2**-9])
def test_presmoothing_checkpoint_is_relative_fresh_and_budgeted(cycle, relative_error):
    n = 4
    A = eye(n, format='csr')
    calls = []

    def pre(A, x, b):
        calls.append('pre')
        x[:] = (1-relative_error)*b

    def post(A, x, b):
        pytest.fail('postsmoothing must not run after acceptance')

    pre._cappa_apply_work = post._cappa_apply_work = {'fixture': n}
    ml = pyamg.multilevel.MultilevelSolver([
        SimpleNamespace(A=A, P=A, R=A, presmoother=pre, postsmoother=post),
        SimpleNamespace(A=A)], coarse_solver='pinv')
    driver = BudgetedCycle(ml, cycle, 2**-10)
    # The matrix application must read the post-smoothing candidate. Reaching
    # restriction means the local check rejected it; no further solve needed.
    import pyamg.budgeted_cycle as budgeted
    native = budgeted._matvec
    applications = []

    def traced(A, x):
        applications.append(x.copy())
        if len(applications) == 2:
            raise RuntimeError('restriction reached')
        return native(A, x)

    for scale in (1.0, 2**-500):
        b = np.full(n, scale)
        applications.clear()
        with patch.object(budgeted, '_matvec', traced):
            if relative_error > 2**-10:
                with pytest.raises(RuntimeError, match='restriction reached'):
                    driver.apply(b)
                continue
            result = driver.apply(b)
        assert result.status == 'COMPLETE'
        np.testing.assert_array_equal(applications[0], result.solution)
        np.testing.assert_array_equal(result.solution, (1-relative_error)*b)
        assert result.accepted_ratios == [relative_error]
        # pre + fresh SpMV + subtraction + two stable norms; no transfer,
        # coarse application, or postsmoothing is billed.
        assert result.work == 3*n + 2*(2*n+2)
        partial = driver.apply(b, result.work-1)
        assert partial.status == 'WORK_LIMIT' and partial.solution is None
        assert partial.blocked['stage'] == 'monitor_norm_model'
        assert partial.work == result.work-(2*n+2)
        assert driver.apply(b, result.work).status == 'COMPLETE'
