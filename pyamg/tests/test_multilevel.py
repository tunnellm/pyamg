"""Test MultilevelSolver class."""
import numpy as np
from numpy.testing import TestCase, assert_almost_equal, assert_equal
from scipy import sparse

from pyamg.gallery import poisson
from pyamg.multilevel import coarse_grid_solver, MultilevelSolver


def precon_norm(v, ml):
    """Calculate preconditioner norm of v."""
    v = np.ravel(v)
    w = ml.aspreconditioner()*v
    return np.sqrt(np.dot(v.conjugate(), w))


class TestMultilevel(TestCase):
    def test_coarse_grid_solver(self):
        cases = []

        cases.append(sparse.csr_array(np.diag(np.arange(1, 5, dtype=float))))
        cases.append(poisson((4,), format='csr'))
        cases.append(poisson((4, 4), format='csr'))

        from pyamg.krylov import cg

        def fn(A, b):
            return cg(A, b)[0]

        # method should be almost exact for small matrices
        for A in cases:
            for solver in ['splu', 'pinv', 'pinv2', 'lu', 'cholesky',
                           'cg', fn]:
                s = coarse_grid_solver(solver)

                b = np.arange(A.shape[0], dtype=A.dtype)

                x = s(A, b)
                assert_almost_equal(A@x, b)

                # subsequent calls use cached data
                x = s(A, b)
                assert_almost_equal(A@x, b)

    def test_aspreconditioner(self):
        from pyamg import smoothed_aggregation_solver
        from scipy.sparse.linalg import cg
        from pyamg.krylov import fgmres
        np.random.seed(1331277597)

        A = poisson((50, 50), format='csr')
        b = np.random.rand(A.shape[0])

        ml = smoothed_aggregation_solver(A)

        for cycle in ['V', 'W', 'F']:
            M = ml.aspreconditioner(cycle=cycle)
            x, _info = cg(A, b, M=M, rtol=1e-8, maxiter=30, atol=0)
            # cg satisfies convergence in the preconditioner norm
            assert precon_norm(b - A@x, ml) < 1e-8*precon_norm(b, ml)

        for cycle in ['AMLI']:
            M = ml.aspreconditioner(cycle=cycle)
            res = []
            x, _info = fgmres(A, b, tol=1e-8, maxiter=30, M=M, residuals=res)
            # fgmres satisfies convergence in the 2-norm
            assert np.linalg.norm(b - A@x) < 1e-8*np.linalg.norm(b)

    def test_accel(self):
        from pyamg import smoothed_aggregation_solver
        from pyamg.krylov import cg, bicgstab
        np.random.seed(30459128)

        A = poisson((50, 50), format='csr')
        b = np.random.rand(A.shape[0])

        ml = smoothed_aggregation_solver(A)

        # cg halts based on the preconditioner norm
        for accel in ['cg', cg]:
            residuals = []
            x = ml.solve(b, maxiter=30, tol=1e-8, residuals=residuals, accel=accel)
            assert precon_norm(b - A@x, ml) < 1e-8*precon_norm(b, ml)
            assert_almost_equal(precon_norm(b - A@x, ml), residuals[-1])

        # cgs and bicgstab use the Euclidean norm
        for accel in ['bicgstab', 'cgs', bicgstab]:
            residuals = []
            x = ml.solve(b, maxiter=30, tol=1e-8, residuals=residuals, accel=accel)
            assert np.linalg.norm(b - A@x) < 1e-8*np.linalg.norm(b)
            assert_almost_equal(np.linalg.norm(b - A@x), residuals[-1])

    def test_cycle_complexity(self):
        # four levels
        levels = []
        levels.append(MultilevelSolver.Level())
        levels[0].A = sparse.csr_array(np.ones((10, 10)))
        levels[0].P = sparse.csr_array(np.ones((10, 5)))
        levels.append(MultilevelSolver.Level())
        levels[1].A = sparse.csr_array(np.ones((5, 5)))
        levels[1].P = sparse.csr_array(np.ones((5, 3)))
        levels.append(MultilevelSolver.Level())
        levels[2].A = sparse.csr_array(np.ones((3, 3)))
        levels[2].P = sparse.csr_array(np.ones((3, 2)))
        levels.append(MultilevelSolver.Level())
        levels[3].A = sparse.csr_array(np.ones((2, 2)))

        # one level hierarchy
        mg = MultilevelSolver(levels[:1])
        assert_equal(mg.cycle_complexity(cycle='V'), 100.0/100.0)
        assert_equal(mg.cycle_complexity(cycle='W'), 100.0/100.0)
        assert_equal(mg.cycle_complexity(cycle='AMLI'), 100.0/100.0)
        assert_equal(mg.cycle_complexity(cycle='F'), 100.0/100.0)

        # two level hierarchy
        # Cost: smooth(2*100) + residual(100) + transfers(50+50) + coarse(25) = 425
        mg = MultilevelSolver(levels[:2])
        assert_equal(mg.cycle_complexity(cycle='V'), 425.0/100.0)
        assert_equal(mg.cycle_complexity(cycle='W'), 425.0/100.0)
        assert_equal(mg.cycle_complexity(cycle='AMLI'), 425.0/100.0)
        assert_equal(mg.cycle_complexity(cycle='F'), 425.0/100.0)

        # three level hierarchy
        # Level costs: L0=400, L1=105 (smooth 50 + res 25 + trans 30), coarse=9
        # AMLI overhead at L1: 4*nnz(A1) + 11*n1 = 4*25 + 11*5 = 155
        mg = MultilevelSolver(levels[:3])
        assert_equal(mg.cycle_complexity(cycle='V'), 514.0/100.0)
        assert_equal(mg.cycle_complexity(cycle='W'), 628.0/100.0)
        assert_equal(mg.cycle_complexity(cycle='AMLI'), 783.0/100.0)
        assert_equal(mg.cycle_complexity(cycle='F'), 628.0/100.0)

        # four level hierarchy
        # Level costs: L0=400, L1=105, L2=39 (smooth 18 + res 9 + trans 12), coarse=4
        # AMLI overhead at L2: 4*9+11*3=69, at L1: 4*25+11*5=155
        mg = MultilevelSolver(levels[:4])
        assert_equal(mg.cycle_complexity(cycle='V'), 548.0/100.0)
        assert_equal(mg.cycle_complexity(cycle='W'), 782.0/100.0)
        assert_equal(mg.cycle_complexity(cycle='AMLI'), 1075.0/100.0)
        assert_equal(mg.cycle_complexity(cycle='F'), 739.0/100.0)  # 2,4,6,3


class TestComplexMultilevel(TestCase):
    def test_coarse_grid_solver(self):
        cases = []

        cases.append(sparse.csr_array(np.diag(np.arange(1, 5))))
        cases.append(poisson((4,), format='csr'))
        cases.append(poisson((4, 4), format='csr'))

        # Make cases complex
        cases = [G+1e-5j*G for G in cases]
        cases = [0.5*(G + G.T.conjugate()) for G in cases]

        # method should be almost exact for small matrices
        for A in cases:
            for solver in ['splu', 'pinv', 'pinv2', 'lu', 'cholesky', 'cg']:
                s = coarse_grid_solver(solver)

                b = np.arange(A.shape[0], dtype=A.dtype)

                x = s(A, b)
                assert_almost_equal(A@x, b)

                # subsequent calls use cached data
                x = s(A, b)
                assert_almost_equal(A@x, b)
