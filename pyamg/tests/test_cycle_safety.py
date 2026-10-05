"""Regression coverage for failures observed in the CAPPA optioned campaign."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_array, eye_array

import pyamg
from pyamg.multilevel import MultilevelSolver, _checked_inner_ratio
from pyamg.relaxation.relaxation import (
    _schwarz_block_layout, _schwarz_setup_work, schwarz_parameters,
)
from pyamg.relaxation.smoothing import setup_strength_based_schwarz


@pytest.mark.parametrize('domains', [2, 3])
def test_schwarz_offsets_reject_both_negative_and_positive_wrap(domains):
    # No dense allocations: the old int32 total wrapped negative for two domains
    # and positive for three, allowing an undersized buffer to reach C++.
    size = 40000
    indices = np.tile(np.arange(size, dtype=np.int32), domains)
    ptr = np.arange(domains + 1, dtype=np.int32) * size
    with pytest.raises(OverflowError, match='dense blocks'):
        _schwarz_block_layout(indices, ptr, size)


def test_schwarz_setup_work_uses_wide_arithmetic():
    assert _schwarz_setup_work(np.array([0, 2000], dtype=np.int32)) == 2000**3 // 3
    assert _schwarz_setup_work(np.array([0, 3000000], dtype=np.int64)) == 3000000**3 // 3


@pytest.mark.parametrize('indices,ptr', [
    ([0], [0, 0, 1]),       # empty subdomain
    ([0], [1, 1]),          # nonzero origin
    ([0], [0, 2]),          # truncated indices
    ([0, 1], [0, 2, 1]),    # decreasing pointer
    ([2], [0, 1]),          # out-of-range index
    ([-1], [0, 1]),
    ([1, 0], [0, 2]),       # unsorted
    ([0, 0], [0, 2]),       # duplicate
    ([0.5], [0, 1]),        # must not silently truncate floats
])
def test_schwarz_rejects_invalid_domains_before_native(monkeypatch, indices, ptr):
    def unsafe_call(*args):
        pytest.fail('invalid layout reached native code')
    monkeypatch.setattr(pyamg.amg_core, 'extract_subblocks', unsafe_call)
    with pytest.raises(ValueError):
        schwarz_parameters(eye_array(2, format='csr'), np.asarray(indices), np.asarray(ptr))


def test_schwarz_precomputed_blocks_are_validated():
    A = eye_array(2, format='csr')
    with pytest.raises(ValueError, match='inverse blocks'):
        schwarz_parameters(A, np.array([0, 1]), np.array([0, 1, 2]),
                           np.ones(1), np.array([0, 1, 1]))


def test_schwarz_valid_blocks_and_cache():
    A = csr_array([[4., -1.], [-1., 3.]])
    indices = np.array([0, 1, 0, 1], dtype=np.int64)
    ptr = np.array([0, 2, 4], dtype=np.int64)
    params = schwarz_parameters(A, indices, ptr)
    assert_allclose(params[2].reshape(2, 2, 2), np.tile(np.linalg.inv(A.toarray()), (2, 1, 1)))
    assert_array_equal(params[3], [0, 4, 8])
    assert schwarz_parameters(A, indices, ptr) is params


def test_strength_schwarz_rejects_invalid_layout_at_configuration():
    lvl = MultilevelSolver.Level()
    lvl.A = eye_array(2, format='csr')
    lvl.C = csr_array((2, 2))
    with pytest.raises(ValueError, match='nonempty'):
        setup_strength_based_schwarz(lvl)


@pytest.mark.parametrize('cycle', ['V', 'W', 'F', 'K', 'AMLI'])
@pytest.mark.parametrize('coarse', ['pinv', 'lu', 'cholesky'])
def test_direct_single_level_cycle(cycle, coarse):
    A = pyamg.gallery.poisson((4, 4), format='csr')
    ml = pyamg.ruge_stuben_solver(A, max_coarse=20, coarse_solver=coarse)
    b = np.arange(1., 17.)
    x = np.zeros_like(b)
    assert len(ml.levels) == 1
    for _ in range(2):
        cost = ml._MultilevelSolver__solve(0, x, b, cycle)
        assert_allclose(A @ x, b, rtol=1e-12, atol=1e-12)
        assert cost == (16**2 if cycle == 'K' else 0)


def identity_hierarchy():
    levels = []
    for i in range(3):
        lvl = MultilevelSolver.Level()
        lvl.A = eye_array(2, format='csr')
        if i < 2:
            lvl.P = lvl.R = eye_array(2, format='csr')
            lvl.presmoother = lvl.postsmoother = lambda A, x, b: None
        levels.append(lvl)
    ml = MultilevelSolver(levels)
    ml._k_cg_iters = 2
    ml._k_early_exit = 0.0
    return ml


@pytest.mark.parametrize('cycle', ['K', 'AMLI'])
@pytest.mark.parametrize('zero', [True, False])
def test_exact_inner_completion_without_ratio_tolerance(cycle, zero):
    ml = identity_hierarchy()
    b = np.zeros(2) if zero else np.array([1., 2.])
    x = np.zeros(2)
    with np.errstate(all='raise'):
        cost = ml._MultilevelSolver__solve(0, x, b, cycle)
    assert_array_equal(x, b)
    if cycle == 'K':
        # Zero RHS uses only the outer level; an exact first inner update must
        # not charge a second recursive solve or orthogonalization.
        expected = ml._k_level_costs[0]
        if not zero:
            expected += ml._k_level_costs[1] + ml._k_coarse_cost + 2 + 4*2
        assert cost == expected


@pytest.mark.parametrize('cycle', ['K', 'AMLI'])
def test_nonzero_residual_with_zero_direction_is_breakdown(cycle):
    ml = identity_hierarchy()
    ml.coarse_solver = lambda A, b: np.zeros_like(b)
    with pytest.raises(np.linalg.LinAlgError, match='inner breakdown'):
        ml._MultilevelSolver__solve(0, np.zeros(2), np.array([1., 2.]), cycle)


def test_invalid_k_iteration_count_does_not_return_zero_correction():
    ml = identity_hierarchy()
    ml._k_cg_iters = 0
    with pytest.raises(ValueError, match='iteration count'):
        ml._MultilevelSolver__solve(0, np.zeros(2), np.ones(2), 'K')


@pytest.mark.parametrize('num,den', [(0., 0.), (1., 0.), (np.nan, 1.),
                                   (1., np.inf), (1e308, 1e-308)])
def test_invalid_inner_ratios_raise(num, den):
    with pytest.raises(np.linalg.LinAlgError):
        _checked_inner_ratio(num, den, 'K')


@pytest.mark.parametrize('cycle', ['V', 'K', 'AMLI'])
def test_multilevel_zero_rhs(cycle):
    A = pyamg.gallery.poisson((12, 12), format='csr')
    ml = pyamg.smoothed_aggregation_solver(A, max_coarse=2)
    x = np.zeros(A.shape[0])
    with np.errstate(all='raise'):
        ml._MultilevelSolver__solve(0, x, x.copy(), cycle)
    assert_array_equal(x, np.zeros_like(x))
