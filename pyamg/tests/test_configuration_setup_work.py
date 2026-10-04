"""Configuration costs must not depend on previously selected smoothers."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pyamg
from pyamg.multilevel import coarse_grid_solver
from pyamg.relaxation.smoothing import change_smoothers, rho_D_inv_A
from pyamg.util.linalg import approximate_spectral_radius
from pyamg.util.setup_work import SetupWork


GS = ('gauss_seidel', {'sweep': 'symmetric'})
JAC = ('jacobi', {'omega': 2/3, 'iterations': 2})
CHEB = ('chebyshev', {'degree': 3})


class TestConfigurationSetupWork(unittest.TestCase):
    def test_cached_dependency_and_plain_counter(self):
        A = pyamg.gallery.poisson((12, 12), format='csr')
        work = SetupWork()
        rho_D_inv_A(A, work=work)
        expected = tuple(work)
        self.assertGreater(expected[0], 0)
        rho_D_inv_A(A, work=work)
        self.assertEqual(tuple(work), expected)
        other = SetupWork()
        rho_D_inv_A(A, work=other)
        self.assertEqual(tuple(other), expected)
        physical = [0, 0]
        rho_D_inv_A(A, work=physical)
        self.assertEqual(physical, [0, 0])

    def test_builders_and_switches(self):
        for builder in (pyamg.ruge_stuben_solver, pyamg.smoothed_aggregation_solver,
                        pyamg.rootnode_solver, pyamg.pairwise_solver):
            with self.subTest(builder=builder.__name__):
                np.random.seed(7)
                A = pyamg.gallery.poisson((12, 12), format='csr')
                ml = builder(A, presmoother=GS, postsmoother=GS)
                baseline = ml.setup_work_breakdown()
                self.assertEqual(baseline['smoother'], (0, 0))
                rng_after_hierarchy = np.random.get_state()
                change_smoothers(ml, JAC, JAC)
                jac = ml.setup_work_breakdown()['smoother']
                jac_image = ml.aspreconditioner() @ np.ones(A.shape[0])
                np.random.set_state(rng_after_hierarchy)
                change_smoothers(ml, CHEB, CHEB)
                cheb = ml.setup_work_breakdown()['smoother']
                cheb_image = ml.aspreconditioner() @ np.ones(A.shape[0])
                for spec, expected, image in ((JAC, jac, jac_image), (CHEB, cheb, cheb_image)):
                    np.random.seed(7)
                    fresh = builder(pyamg.gallery.poisson((12, 12), format='csr'),
                                    presmoother=spec, postsmoother=spec)
                    self.assertEqual(fresh.setup_work_breakdown()['hierarchy'], baseline['hierarchy'])
                    self.assertEqual(fresh.setup_work_breakdown()['smoother'], expected)
                    np.testing.assert_array_equal(fresh.aspreconditioner() @ np.ones(A.shape[0]), image)
                with patch('pyamg.util.linalg._approximate_eigenvalues',
                           side_effect=AssertionError('cached estimator recomputed')):
                    change_smoothers(ml, JAC, CHEB)
                    self.assertEqual(ml.setup_work_breakdown()['smoother'],
                                     tuple(jac[i] + cheb[i] for i in range(2)))
                    change_smoothers(ml, GS, GS)
                    self.assertEqual(ml.setup_work_breakdown(), baseline)
                    change_smoothers(ml, JAC, JAC)
                    self.assertEqual(ml.setup_work_breakdown()['smoother'], jac)
                copied = deepcopy(ml)
                change_smoothers(copied, CHEB, CHEB)
                self.assertEqual(copied.setup_work_breakdown()['smoother'], cheb)
                parts = ml.setup_work_breakdown()
                self.assertEqual(ml.setup_complexity(),
                                 tuple(sum(p[i] for p in parts.values()) / A.nnz for i in range(2)))

    def test_hierarchy_dependency_is_not_charged_twice(self):
        A = pyamg.gallery.poisson((12, 12), format='csr')
        ml = pyamg.ruge_stuben_solver(A)
        work = SetupWork()
        for level in ml.levels[:-1]:
            approximate_spectral_radius(level.A, work=work)
            rho_D_inv_A(level.A, work=work)
        ml.record_hierarchy_setup(work)
        hierarchy = tuple(work)
        change_smoothers(ml, JAC, CHEB)
        self.assertEqual(ml.setup_work_breakdown()['hierarchy'], hierarchy)
        self.assertEqual(ml.setup_work_breakdown()['smoother'], (0, 0))

    def test_coarse_replacement_and_k_cache(self):
        ml = pyamg.ruge_stuben_solver(pyamg.gallery.poisson((12, 12), format='csr'))
        original = ml.setup_work_breakdown()
        n = ml.levels[-1].A.shape[0]
        for name, cost in (('lu', n**3//3), ('cholesky', n**3//6), ('pinv', 3*n**3)):
            ml.coarse_solver = coarse_grid_solver(name)
            parts = ml.setup_work_breakdown()
            self.assertEqual(parts['coarse'], (cost, 0))
            self.assertEqual(parts['hierarchy'], original['hierarchy'])
        ml._k_level_costs = [999]
        change_smoothers(ml, CHEB, CHEB)
        self.assertFalse(hasattr(ml, '_k_level_costs'))


if __name__ == '__main__':
    unittest.main()
