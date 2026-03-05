"""Generic AMG solver."""
from warnings import warn

import scipy as sp
from scipy.linalg import pinv
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import LinearOperator
import numpy as np

from . import krylov
from .util.utils import to_type
from .util.params import set_tol
from .relaxation import smoothing
from .util import upcast


def _get_coarse_solver_name(solver):
    """Extract solver name from coarse_solver repr like "coarse_grid_solver('pinv')"."""
    if solver is None:
        return None
    solver_str = str(solver)
    for quote in ("'", '"'):
        if quote in solver_str:
            start = solver_str.find(quote) + 1
            end = solver_str.find(quote, start)
            if end > start:
                return solver_str[start:end]
    return None


class MultilevelSolver:
    """Stores multigrid hierarchy and implements the multigrid cycle.

    The class constructs the cycling process and points to the methods for
    coarse grid solves.  A MultilevelSolver object is typically returned from a
    particular AMG method (see ``ruge_stuben_solver`` or ``smoothed_aggregation_solver``
    for example).  A call to ``MultilevelSolver.solve()`` is a typical access
    point.  The class also defines methods for constructing operator, cycle, and
    grid complexities.

    Parameters
    ----------
    levels : list of Level
        Array of level objects that contain A, R, and P.
    coarse_solver : str, callable, tuple
        The solver method is either (1) a string such as 'splu' or 'pinv'
        of a callable object which receives only parameters (A, b) and
        returns an (approximate or exact) solution to the linear system Ax
        = b, or (2) a callable object that takes parameters (A,b) and
        returns an (approximate or exact) solution to Ax = b, or (3) a
        tuple of the form (str|callable, args), where args is a
        dictionary of arguments to be passed to the function denoted by
        string or callable.

        Sparse direct methods:

        * splu         : sparse LU solver

        Sparse iterative methods:

        * any method in scipy.sparse.linalg or pyamg.krylov (e.g. 'cg').
        * Methods in pyamg.krylov take precedence.
        * relaxation method, such as 'gauss_seidel' or 'jacobi',

        Dense methods:

        * pinv     : pseudoinverse (SVD)
        * lu       : LU factorization
        * cholesky : Cholesky factorization

    Attributes
    ----------
    levels : level array
        Array of level objects that contain A, R, and P.
    coarse_solver : str
        String passed to coarse_grid_solver indicating the solve type

    Methods
    -------
    aspreconditioner()
        Create a preconditioner using this multigrid cycle
    cycle_complexity()
        A measure of the cost of a single multigrid cycle.
    grid_complexity()
        A measure of the rate of coarsening.
    operator_complexity()
        A measure of the size of the multigrid hierarchy.
    setup_complexity()
        A measure of the cost to construct the hierarchy.
    solve()
        Iteratively solves a linear system for the right hand side.
    change_solve_matrix(A)
        Change matrix solve/preconditioning matrix.
        This also changes the corresponding relaxation routines on the fine
        grid.  This can be used, for example, to precondition a
        quadratic finite element discretization with AMG built from
        a linear discretization on quadratic quadrature points.

    Notes
    -----
    If not defined, the R attribute on each level is set to
    the transpose of P.

    Examples
    --------
    >>> # manual construction of a two-level AMG hierarchy
    >>> from pyamg.gallery import poisson
    >>> from pyamg.multilevel import MultilevelSolver
    >>> from pyamg.strength import classical_strength_of_connection
    >>> from pyamg.classical.interpolate import direct_interpolation
    >>> from pyamg.classical.split import RS
    >>> # compute necessary operators
    >>> A = poisson((100, 100), format='csr')
    >>> C = classical_strength_of_connection(A)
    >>> splitting = RS(A)
    >>> P = direct_interpolation(A, C, splitting)
    >>> R = P.T
    >>> # store first level data
    >>> levels = []
    >>> levels.append(MultilevelSolver.Level())
    >>> levels.append(MultilevelSolver.Level())
    >>> levels[0].A = A
    >>> levels[0].C = C
    >>> levels[0].splitting = splitting
    >>> levels[0].P = P
    >>> levels[0].R = R
    >>> # store second level data
    >>> levels[1].A = R @ A @ P                      # coarse-level matrix
    >>> # create MultilevelSolver
    >>> ml = MultilevelSolver(levels, coarse_solver='splu')
    >>> print(ml)
    MultilevelSolver
    Number of Levels:     2
    Operator Complexity:   1.891
    Grid Complexity:       1.500
    Coarse Solver:        'splu'
      level   unknowns     nonzeros
         0       10000        49600 [52.88%]
         1        5000        44202 [47.12%]
    <BLANKLINE>

    """

    class Level:
        """Stores one level of the multigrid hierarchy.

        All level objects will have an 'A' attribute referencing the matrix
        of that level.  All levels, except for the coarsest level, will
        also have 'P' and 'R' attributes referencing the prolongation and
        restriction operators that act between each level and the next
        coarser level.

        Attributes
        ----------
        A : csr_array
            Problem matrix for Ax=b
        R : csr_array
            Restriction matrix between levels (often R = P.T)
        P : csr_array
            Prolongation or Interpolation matrix.

        Notes
        -----
        The functionality of this class is a struct

        """

        def __init__(self):
            """Level construct (empty)."""
            self.A = None

    class level(Level):  # noqa: N801
        """Deprecated level class."""

        def __init__(self):
            """Raise deprecation warning on use, not import."""
            super().__init__()
            warn('level() is deprecated.  use Level()',
                 category=DeprecationWarning, stacklevel=2)

    def __init__(self, levels, coarse_solver='pinv'):
        """Initialize the cycle and ensure complete list of levels.

        Parameters
        ----------
        levels : list of Level
            Array of level objects that contain A, R, and P.
        coarse_solver : str, callable, tuple
            The coarsest level solver. (See the class documentation).

        """
        self.symmetric_smoothing = False  # force change_smoothers to set to True
        self.levels = levels
        self.coarse_solver = coarse_grid_solver(coarse_solver)

        for level in levels[:-1]:
            if not hasattr(level, 'R'):
                level.R = level.P.T.conjugate()

    def __repr__(self):
        """Print basic statistics about the multigrid hierarchy.

        Returns
        -------
        str
            Information about each level of the hierarchy.

        """
        output = 'MultilevelSolver\n'
        output += f'Number of Levels:     {len(self.levels)}\n'
        output += f'Operator Complexity:  {self.operator_complexity():6.3f}\n'
        output += f'Grid Complexity:      {self.grid_complexity():6.3f}\n'
        if hasattr(self, '_setup_work'):
            num_work, graph_work = self.setup_complexity()
            output += f'Setup Complexity:     {num_work:6.3f} (numerical), {graph_work:6.3f} (graph)\n'
        output += f'Coarse Solver:        {self.coarse_solver.name()}\n'

        total_nnz = sum(level.A.nnz for level in self.levels)

        #          123456712345678901 123456789012 123456789
        #               0       10000        49600 [52.88%]
        output += '  level   unknowns     nonzeros\n'
        for n, level in enumerate(self.levels):
            A = level.A
            ratio = 100 * A.nnz / total_nnz
            output += f'{n:>6} {A.shape[1]:>11} {A.nnz:>12} [{ratio:2.2f}%]\n'

        return output

    def cycle_complexity(self, cycle='V'):
        """Cycle complexity of V, W, AMLI, and F(1,1) cycle.

        Cycle complexity is an approximate measure of the number of
        floating point operations (FLOPs) required to perform a single
        multigrid cycle relative to the cost a single smoothing operation.

        Parameters
        ----------
        cycle : {'V','W','F','AMLI'}
            Type of multigrid cycle to perform in each iteration.

        Returns
        -------
        float
            Defined as F_sum / F_0, where
            F_sum is the total number of nonzeros in the matrix on all
            levels encountered during a cycle and F_0 is the number of
            nonzeros in the matrix on the finest level.

        Notes
        -----
        Accounts for smoother iterations, Chebyshev polynomial degree,
        block matrix overhead, residual computation, and grid transfers.

        Block smoothers on BSR matrices have additional cost proportional
        to blocksize^2 for block inversions.

        If no smoothers are defined, assumes cost of 1 per nnz for each
        of pre- and post-smoothing (total cost 2 per level).

        """
        import functools

        def smoother_cost(smoother, A):
            """Compute cost multiplier for a smoother relative to nnz(A)."""
            if smoother is None:
                return 1  # default cost when no smoother specified

            iterations = 1
            degree = 1
            sweep_factor = 1

            if isinstance(smoother, functools.partial):
                iterations = smoother.keywords.get('iterations', 1)
                sweep = smoother.keywords.get('sweep', 'forward')
                if sweep == 'symmetric':
                    sweep_factor = 2  # forward + backward
            elif callable(smoother):
                # Check for closure variables (Chebyshev, gauss_seidel_ne, etc.)
                if hasattr(smoother, '__closure__') and smoother.__closure__:
                    for varname, cell in zip(smoother.__code__.co_freevars,
                                             smoother.__closure__):
                        if varname == 'coefficients':
                            degree = len(cell.cell_contents)
                        elif varname == 'iterations':
                            iterations = cell.cell_contents
                        elif varname == 'sweep':
                            if cell.cell_contents == 'symmetric':
                                sweep_factor = 2

            # Block smoother overhead: blocksize^2 for block inversions
            blocksize = getattr(A, 'blocksize', (1, 1))[0]
            block_factor = blocksize * blocksize if blocksize > 1 else 1

            # Normal equations methods (gauss_seidel_ne/nr, jacobi_ne, cgne, cgnr)
            # operate on A @ A.H or A.H @ A, requiring 2 SpMVs per iteration.
            ne_factor = 1
            smoother_name = getattr(smoother, '__name__', '')
            if smoother_name.endswith(('_ne', '_nr', 'cgne', 'cgnr')):
                ne_factor = 2

            # GMRES has additional Arnoldi orthogonalization cost O(n*k^2)
            if smoother_name == 'gmres':
                # Extract maxiter from closure
                maxiter = 1
                if hasattr(smoother, '__closure__') and smoother.__closure__:
                    for varname, cell in zip(smoother.__code__.co_freevars,
                                             smoother.__closure__):
                        if varname == 'maxiter':
                            maxiter = cell.cell_contents
                            break
                # Cost: k SpMVs + n*k^2 Arnoldi orthogonalization
                n = A.shape[0]
                return maxiter + n * maxiter * maxiter / A.nnz

            # Schwarz: cost is sum of subdomain_size^2 for dense block solves
            if smoother_name == 'schwarz':
                subdomain_ptr = None
                schwarz_iters = 1
                schwarz_sweep = 1
                if hasattr(smoother, '__closure__') and smoother.__closure__:
                    for varname, cell in zip(smoother.__code__.co_freevars,
                                             smoother.__closure__):
                        if varname == 'subdomain_ptr':
                            subdomain_ptr = cell.cell_contents
                        elif varname == 'iterations':
                            schwarz_iters = cell.cell_contents
                        elif varname == 'sweep':
                            if cell.cell_contents == 'symmetric':
                                schwarz_sweep = 2
                if subdomain_ptr is not None:
                    sizes = subdomain_ptr[1:] - subdomain_ptr[:-1]
                    total_cost = np.sum(sizes * sizes) * schwarz_iters * schwarz_sweep
                    return total_cost / A.nnz
                # Fallback: assume average subdomain size is nnz/n
                return A.nnz / A.shape[0] * iterations * sweep_factor

            return iterations * degree * sweep_factor * block_factor * ne_factor

        cycle = str(cycle).upper()

        # Compute per-level costs: smoothing + residual + grid transfers
        costs = []
        for level in self.levels[:-1]:
            pre_cost = smoother_cost(getattr(level, 'presmoother', None), level.A)
            post_cost = smoother_cost(getattr(level, 'postsmoother', None), level.A)
            smooth_work = (pre_cost + post_cost) * level.A.nnz

            # Residual computation: r = b - A @ x
            residual_work = level.A.nnz

            # Grid transfers: restriction R @ r and interpolation P @ e
            R_nnz = level.R.nnz if hasattr(level, 'R') else 0
            P_nnz = level.P.nnz if hasattr(level, 'P') else 0
            transfer_work = R_nnz + P_nnz

            costs.append(smooth_work + residual_work + transfer_work)

        # Coarsest level cost
        coarse_A = self.levels[-1].A
        coarse_n = coarse_A.shape[0]
        coarse_solver = getattr(self, 'coarse_solver', None)

        # Dense solvers: O(n^2) per solve (forward/back substitution or mat-vec)
        # Sparse solvers: O(nnz) per solve
        dense_solvers = {'pinv', 'pinv2', 'lu', 'cholesky'}
        solver_name = _get_coarse_solver_name(coarse_solver)
        if solver_name in dense_solvers:
            coarse_cost = coarse_n * coarse_n
        else:
            coarse_cost = coarse_A.nnz

        def V(level):
            if len(self.levels) == 1:
                return coarse_cost

            if level == len(self.levels) - 2:
                return costs[level] + coarse_cost

            return costs[level] + V(level + 1)

        def W(level):
            if len(self.levels) == 1:
                return coarse_cost

            if level == len(self.levels) - 2:
                return costs[level] + coarse_cost

            return costs[level] + 2 * W(level + 1)

        def F(level):
            if len(self.levels) == 1:
                return coarse_cost

            if level == len(self.levels) - 2:
                return costs[level] + coarse_cost

            return costs[level] + F(level + 1) + V(level + 1)

        def AMLI(level):
            """AMLI cycle: like W-cycle but with orthogonalization overhead.

            With nAMLI=2, each AMLI application does 2 recursive solves
            plus Gram-Schmidt orthogonalization in the A-norm:
              k=0: 1 SpMV(Ac) + 4*n_c (step size + update)
              k=1: 3 SpMV(Ac) + 7*n_c (orthog + step size + update)
            Total overhead: 4*nnz(Ac) + 11*n_c per application.
            """
            if len(self.levels) == 1:
                return coarse_cost

            if level == len(self.levels) - 2:
                return costs[level] + coarse_cost

            # Orthogonalization overhead at coarse level (level+1)
            Ac = self.levels[level + 1].A
            amli_overhead = 4 * Ac.nnz + 11 * Ac.shape[0]

            return costs[level] + 2 * AMLI(level + 1) + amli_overhead

        def K(level):
            """K-cycle: nK FCG iterations (Alg 3.1, m_i=1) at each level.

            nK recursive K-cycle applies,
            nK SpMVs on A_c (one per FCG iteration for A@d),
            Iteration 0: 4*n_c vector ops (2 dots + 2 axpys),
            Iterations 1..nK-1: 6*n_c each (3 dots + 3 axpys).
            """
            if len(self.levels) == 1:
                return coarse_cost

            if level == len(self.levels) - 2:
                return costs[level] + coarse_cost

            nK = getattr(self, '_k_cg_iters', 2)
            Ac = self.levels[level + 1].A
            n_c = Ac.shape[0]
            recursive_cost = nK * K(level + 1)
            spmv_cost = nK * Ac.nnz
            vector_cost = 4 * n_c + max(0, nK - 1) * 6 * n_c
            return costs[level] + recursive_cost + spmv_cost + vector_cost

        if cycle == 'V':
            flops = V(0)
        elif cycle == 'W':
            flops = W(0)
        elif cycle == 'AMLI':
            flops = AMLI(0)
        elif cycle == 'F':
            flops = F(0)
        elif cycle == 'K':
            flops = K(0)
        else:
            raise TypeError(f'Unrecognized cycle type ({cycle})')

        return float(flops) / float(self.levels[0].A.nnz)

    def _precompute_k_costs(self):
        """Precompute and cache per-level fixed costs for K-cycle accounting.

        Stores _k_level_costs[lvl] = smoothing + residual + transfer cost
        at each level, and _k_coarse_cost for the coarsest level solve.
        These are absolute flop counts (not normalized by nnz).
        """
        import functools

        def smoother_cost(smoother, A):
            """Compute cost multiplier for a smoother relative to nnz(A)."""
            if smoother is None:
                return 1
            iterations = 1
            sweep_factor = 1
            degree = 1
            if isinstance(smoother, functools.partial):
                iterations = smoother.keywords.get('iterations', 1)
                sweep = smoother.keywords.get('sweep', 'forward')
                if sweep == 'symmetric':
                    sweep_factor = 2
            elif callable(smoother):
                if hasattr(smoother, '__closure__') and smoother.__closure__:
                    for varname, cell in zip(smoother.__code__.co_freevars,
                                             smoother.__closure__):
                        if varname == 'coefficients':
                            degree = len(cell.cell_contents)
                        elif varname == 'iterations':
                            iterations = cell.cell_contents
                        elif varname == 'sweep':
                            if cell.cell_contents == 'symmetric':
                                sweep_factor = 2
            blocksize = getattr(A, 'blocksize', (1, 1))[0]
            block_factor = blocksize * blocksize if blocksize > 1 else 1
            return iterations * degree * sweep_factor * block_factor

        self._k_level_costs = []
        for level in self.levels[:-1]:
            pre_cost = smoother_cost(getattr(level, 'presmoother', None),
                                     level.A)
            post_cost = smoother_cost(getattr(level, 'postsmoother', None),
                                      level.A)
            smooth_work = (pre_cost + post_cost) * level.A.nnz
            residual_work = level.A.nnz
            R_nnz = level.R.nnz if hasattr(level, 'R') else 0
            P_nnz = level.P.nnz if hasattr(level, 'P') else 0
            transfer_work = R_nnz + P_nnz
            self._k_level_costs.append(smooth_work + residual_work
                                       + transfer_work)

        coarse_A = self.levels[-1].A
        coarse_n = coarse_A.shape[0]
        dense_solvers = {'pinv', 'pinv2', 'lu', 'cholesky'}
        solver_name = _get_coarse_solver_name(
            getattr(self, 'coarse_solver', None))
        if solver_name in dense_solvers:
            self._k_coarse_cost = coarse_n * coarse_n
        else:
            self._k_coarse_cost = coarse_A.nnz

    def operator_complexity(self):
        """Operator complexity of this multigrid hierarchy.

        Defined as::

            Number of nonzeros in the matrix on all levels /
            Number of nonzeros in the matrix on the finest level

        Returns
        -------
        scalar
            Measure of the operator complexity.

        """
        return sum(level.A.nnz for level in self.levels) /\
            float(self.levels[0].A.nnz)

    def grid_complexity(self):
        """Grid complexity of this multigrid hierarchy.

        Defined as::

            Number of unknowns on all levels /
            Number of unknowns on the finest level

        Returns
        -------
        scalar
            Measure of the grid complexity.

        """
        return sum(level.A.shape[0] for level in self.levels) /\
            float(self.levels[0].A.shape[0])

    def setup_complexity(self):
        """Setup complexity of building this multigrid hierarchy.

        Returns a measure of the computational work expended during the
        construction of the multigrid hierarchy, relative to the number
        of nonzeros on the finest level.

        Returns
        -------
        tuple (float, float)
            A tuple of (numerical_work, graph_work) where:
            - numerical_work: Floating point operations normalized by nnz(A_0).
              This includes sparse matrix arithmetic, interpolation construction,
              prolongation smoothing, and Galerkin products.
            - graph_work: Integer/graph operations normalized by nnz(A_0).
              This includes strength of connection pattern traversal,
              aggregation/C-F splitting, and sparse matrix structure operations.

        Notes
        -----
        This is only a rough estimate of the true setup complexity. The
        estimate assumes that:
        - Sparse matrix-matrix products cost proportional to the sum of
          input and output nonzeros
        - Strength of connection computation costs proportional to nnz(A)
        - Aggregation/splitting costs proportional to nnz(C)
        - Prolongation smoothing costs depend on the method used

        The work estimates are tracked during hierarchy construction in
        the solver routines (ruge_stuben_solver, smoothed_aggregation_solver,
        rootnode_solver).

        """
        if not hasattr(self, '_setup_work'):
            return (0.0, 0.0)

        nnz0 = float(self.levels[0].A.nnz)
        numerical_work, graph_work = self._setup_work

        # Add smoother setup work: spectral radius estimation for relaxation.
        # Chebyshev: approximate_spectral_radius(A) -> rho_matvecs SpMVs
        # Jacobi/block_jacobi/jacobi_ne: get_diagonal O(n) + scale_rows O(nnz)
        #   + approximate_spectral_radius(D_inv_A) -> rho_D_inv_matvecs SpMVs
        for level in self.levels[:-1]:
            if hasattr(level.A, 'rho_D_inv'):
                matvecs = getattr(level.A, 'rho_D_inv_matvecs', 15)
                numerical_work += (2 + matvecs) * level.A.nnz
            elif hasattr(level.A, 'rho'):
                matvecs = getattr(level.A, 'rho_matvecs', 15)
                numerical_work += matvecs * level.A.nnz

        # Add coarse solver factorization cost
        # Dense factorization costs (FMA convention: each a -= b*c is 1 FMA):
        #   Cholesky (dpotrf): n³/6 FMAs
        #   LU (dgetrf):       n³/3 FMAs
        #   SVD/pinv: ~3n³ FMAs (rough estimate; SVD not purely FMA-dominated)
        # Sparse factorization (splu): O(nnz) to O(n*nnz) depending on fill-in
        coarse_A = self.levels[-1].A
        coarse_n = coarse_A.shape[0]
        coarse_solver = getattr(self, 'coarse_solver', None)

        solver_name = _get_coarse_solver_name(coarse_solver)
        if solver_name == 'cholesky':
            numerical_work += coarse_n * coarse_n * coarse_n // 6
        elif solver_name == 'lu':
            numerical_work += coarse_n * coarse_n * coarse_n // 3
        elif solver_name in ('pinv', 'pinv2'):
            numerical_work += 3 * coarse_n * coarse_n * coarse_n
        elif solver_name == 'splu':
            # Sparse LU: estimate as O(nnz) for well-structured matrices
            numerical_work += coarse_A.nnz

        return (numerical_work / nnz0, graph_work / nnz0)

    def change_solve_matrix(self, A):
        """Change matrix solve/preconditioning matrix.

        Parameters
        ----------
        A : csr_array
            Target solution matrix.

        Notes
        -----
        This also changes the corresponding relaxation routines on the fine
        grid.  This can be used, for example, to precondition a
        quadratic finite element discretization with linears.

        """
        self.levels[0].A = A

        smoothing.rebuild_smoother(self.levels[0])

    def psolve(self, b):
        """Legacy solve interface.

        Parameters
        ----------
        b : array
            Right-hand side.

        Returns
        -------
        array
            Solution after one iteration.

        """
        return self.solve(b, maxiter=1)

    def aspreconditioner(self, cycle='V'):
        """Create a preconditioner using this multigrid cycle.

        Parameters
        ----------
        cycle : {'V','W','F','AMLI','K'}
            Type of multigrid cycle to perform in each iteration.

        Returns
        -------
        LinearOperator
            Preconditioner suitable for the iterative solvers in defined in
            the scipy.sparse.linalg module (e.g. cg, gmres) and any other
            solver that uses the LinearOperator interface.  Refer to the
            LinearOperator documentation in :obj:`scipy.sparse.linalg`.

        See Also
        --------
        MultilevelSolver.solve
        scipy.sparse.linalg.LinearOperator

        Examples
        --------
        >>> from pyamg.aggregation import smoothed_aggregation_solver
        >>> from pyamg.gallery import poisson
        >>> from scipy.sparse.linalg import cg
        >>> import scipy as sp
        >>> import numpy as np
        >>> A = poisson((100, 100), format='csr')          # matrix
        >>> b = np.random.rand(A.shape[0])                 # random RHS
        >>> ml = smoothed_aggregation_solver(A)            # AMG solver
        >>> M = ml.aspreconditioner(cycle='V')             # preconditioner
        >>> x, info = cg(A, b, rtol=1e-8, maxiter=30, M=M) # solve with CG

        """
        shape = self.levels[0].A.shape
        dtype = self.levels[0].A.dtype

        def matvec(b):
            return self.solve(b, maxiter=1, cycle=cycle, tol=1e-12)

        return LinearOperator(shape, matvec, dtype=dtype)

    def solve(self, b, x0=None, tol=1e-5, maxiter=100, cycle='V', accel=None,
              callback=None, residuals=None, cycles_per_level=1, return_info=False):
        """Execute multigrid cycling.

        Parameters
        ----------
        b : array
            Right hand side.
        x0 : array
            Initial guess.
        tol : float
            Stopping criteria: relative residual r[k]/||b|| tolerance.
            If `accel` is used, the stopping criteria is set by the Krylov method.
        maxiter : int
            Stopping criteria: maximum number of allowable iterations.
        cycle : {'V','W','F','AMLI'}
            Type of multigrid cycle to perform in each iteration.
        accel : str, function
            Defines acceleration method.  Can be a string such as 'cg'
            or 'gmres' which is the name of an iterative solver in
            pyamg.krylov (preferred) or scipy.sparse.linalg.
            If accel is not a string, it will be treated like a function
            with the same interface provided by the iterative solvers in SciPy.
        callback : function
            User-defined function called after each iteration.  It is
            called as callback(xk) where xk is the k-th iterate vector.
        residuals : list
            List to contain residual norms at each iteration.  The residuals
            will be the residuals from the Krylov iteration -- see the `accel`
            method to see verify whether this ||r|| or ||Mr|| (as in the case of
            GMRES).
        cycles_per_level : int, default 1
            Number of V-cycles on each level of an F-cycle.
        return_info : bool
            If true, will return ``(x, info)``.
            If false, will return ``x`` (default).

        Returns
        -------
        array
            Approximate solution to Ax=b after k iterations.

        str
            Halting status::

                 0: successful exit
                >0: convergence to tolerance not achieved
                    return iteration count instead.


        See Also
        --------
        aspreconditioner

        Examples
        --------
        >>> from numpy import ones
        >>> from pyamg import ruge_stuben_solver
        >>> from pyamg.gallery import poisson
        >>> A = poisson((100, 100), format='csr')
        >>> b = A @ ones(A.shape[0])
        >>> ml = ruge_stuben_solver(A, max_coarse=10)
        >>> residuals = []
        >>> x = ml.solve(b, tol=1e-12, residuals=residuals) # standalone solver

        """
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = np.array(x0)  # copy

        A = self.levels[0].A

        cycle = str(cycle).upper()

        # AMLI cycles require hermitian matrix
        if (cycle == 'AMLI') and hasattr(A, 'symmetry'):
            if A.symmetry != 'hermitian':
                raise ValueError('AMLI cycles require \
                    symmetry to be hermitian')

        if accel is not None:

            # Check for symmetric smoothing scheme when using CG
            if (accel == 'cg') and (not self.symmetric_smoothing):
                warn('Incompatible non-symmetric multigrid preconditioner '
                     'detected, due to presmoother/postsmoother combination. '
                     'CG requires SPD preconditioner, not just SPD matrix.')

            # Check for AMLI compatibility
            if (accel != 'fgmres') and (cycle == 'AMLI'):
                raise ValueError('AMLI cycles require acceleration (accel) '
                                 'to be fgmres, or no acceleration')

            # Acceleration is being used
            kwargs = {}
            if isinstance(accel, str):
                kwargs = {}
                if hasattr(krylov, accel):
                    accel = getattr(krylov, accel)
                else:
                    accel = getattr(sla, accel)

            M = self.aspreconditioner(cycle=cycle)

            try:  # try PyAMG style interface which has a residuals parameter
                x, info = accel(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M,
                                callback=callback, residuals=residuals, **kwargs)
                if return_info:
                    return x, info
                return x
            except TypeError:
                # try the scipy.sparse.linalg style interface,
                # which requires a callback function if a residual
                # history is desired

                if residuals is not None:
                    residuals[:] = [np.linalg.norm(b - A @ x)]

                    def callback_wrapper(x):
                        if np.isscalar(x):
                            residuals.append(x)
                        else:
                            residuals.append(np.linalg.norm(b - A @ x))
                        if callback is not None:
                            callback(x)
                else:
                    callback_wrapper = callback

                # for scipy solvers, see if rtol is available
                kwargs['rtol'] = tol
                kwargs['atol'] = 0

                x, info = accel(A, b, x0=x0, maxiter=maxiter, M=M,
                                callback=callback_wrapper, **kwargs)
                if return_info:
                    return x, info
                return x

        else:
            # Scale tol by normb
            # Don't scale tol earlier. The accel routine should also scale tol
            normb = np.linalg.norm(b)
            if normb == 0.0:
                normb = 1.0  # set so that we have an absolute tolerance

        # Start cycling (no acceleration)
        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals[:] = [normr]  # initial residual

        # Create uniform types for A, x and b
        # Clearly, this logic doesn't handle the case of real A and complex b
        tp = upcast(b.dtype, x.dtype, A.dtype)
        [b, x] = to_type(tp, [b, x])
        b = np.ravel(b)
        x = np.ravel(x)

        it = 0

        while True:  # it <= maxiter and normr >= tol:
            if len(self.levels) == 1:
                # hierarchy has only 1 level
                x = self.coarse_solver(A, b)
            else:
                self.__solve(0, x, b, cycle, cycles_per_level)

            it += 1

            normr = np.linalg.norm(b - A @ x)
            if residuals is not None:
                residuals.append(normr)

            if callback is not None:
                callback(x)

            if normr < tol * normb:
                if return_info:
                    return x, 0
                return x

            if it == maxiter:
                if return_info:
                    return x, it
                return x

    def __solve(self, lvl, x, b, cycle, cycles_per_level=1):
        """Multigrid cycling.

        Parameters
        ----------
        lvl : int
            Solve problem on level ``lvl``.
        x : numpy array
            Initial guess ``x``.
        b : numpy array
            Right-hand side for ``Ax=b``.
        cycle : {'V','W','F','AMLI','K'}
            Recursively called cycling function.  The
            Defines the cycling used::

                cycle='V':    V-cycle
                cycle='W':    W-cycle
                cycle='F':    F-cycle
                cycle='AMLI': AMLI-cycle
                cycle='K':    K-cycle (Notay's recursive Krylov cycle)

        Returns
        -------
        int
            Flop cost of this cycle application.  For fixed cycles (V, W, F,
            AMLI) returns 0 (use cycle_complexity instead).  For K-cycles,
            returns the actual flop count since the adaptive iteration count
            makes cost variable.

        cycles_per_level : int, default 1
            Number of V-cycles on each level of an F-cycle.

        """
        A = self.levels[lvl].A

        self.levels[lvl].presmoother(A, x, b)

        residual = b - A @ x

        coarse_b = self.levels[lvl].R @ residual
        coarse_x = np.zeros_like(coarse_b)

        flops = 0  # accumulated for K-cycle; 0 for fixed cycles

        if lvl == len(self.levels) - 2:
            coarse_x[:] = self.coarse_solver(self.levels[-1].A, coarse_b)
            if cycle == 'K':
                if not hasattr(self, '_k_level_costs'):
                    self._precompute_k_costs()
                flops = self._k_level_costs[lvl] + self._k_coarse_cost
        elif cycle == 'V':
            self.__solve(lvl + 1, coarse_x, coarse_b, 'V')
        elif cycle == 'W':
            self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
            self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
        elif cycle == 'F':
            self.__solve(lvl + 1, coarse_x, coarse_b, cycle, cycles_per_level)
            for _ in range(0, cycles_per_level):
                self.__solve(lvl + 1, coarse_x, coarse_b, 'V', 1)
        elif cycle == 'AMLI':
            # Run nAMLI AMLI cycles, which compute "optimal" corrections by
            # orthogonalizing the coarse-grid corrections in the A-norm
            nAMLI = 2
            Ac = self.levels[lvl + 1].A
            p = np.zeros((nAMLI, coarse_b.shape[0]), dtype=coarse_b.dtype)
            beta = np.zeros((nAMLI, nAMLI), dtype=coarse_b.dtype)
            for k in range(nAMLI):
                # New search direction --> M^{-1}@residual
                p[k, :] = 1
                self.__solve(lvl + 1, p[k, :].reshape(coarse_b.shape),
                             coarse_b, cycle)

                # Orthogonalize new search direction to old directions
                for j in range(k):  # loops from j = 0...(k-1)
                    beta[k, j] = np.inner(p[j, :].conj(), Ac @ p[k, :]) /\
                            np.inner(p[j, :].conj(), Ac @ p[j, :])
                    p[k, :] -= beta[k, j] * p[j, :]

                # Compute step size
                Ap = Ac @ p[k, :]
                alpha = np.inner(p[k, :].conj(), np.ravel(coarse_b)) /\
                        np.inner(p[k, :].conj(), Ap)

                # Update solution
                coarse_x += alpha * p[k, :].reshape(coarse_x.shape)

                # Update residual
                coarse_b -= alpha * Ap.reshape(coarse_b.shape)
        elif cycle == 'K':
            # K-cycle: up to ℓ iterations of FCG (Algorithm 3.1, m_i=1) at
            # each level, preconditioned by recursive K-cycle on next level.
            # Reference: Notay & Vassilevski, "Recursive Krylov-based multigrid
            # cycles", Numer. Linear Algebra Appl. 15:473-487, 2008.
            #
            # _k_cg_iters: max FCG iterations per level (paper's ℓ)
            # _k_early_exit: residual ratio threshold for early exit, or 0
            #   to force exactly _k_cg_iters iterations.

            # Ensure per-level fixed costs are precomputed
            if not hasattr(self, '_k_level_costs'):
                self._precompute_k_costs()

            Ac = self.levels[lvl + 1].A
            n_c = Ac.shape[0]
            nK = getattr(self, '_k_cg_iters', 2)
            early_exit = getattr(self, '_k_early_exit', 0.0)

            # Start with this level's fixed costs (smoothing + residual + transfers)
            level_cost = self._k_level_costs[lvl]

            # FCG Algorithm 3.1 with m_i=1 to solve Ac @ y = coarse_b
            r = coarse_b.copy()
            norm_r0_sq = np.inner(r.conj(), r) if early_exit > 0 else 0

            # Iteration 0: w = K^{-1} r, d = w (no previous direction)
            w = np.zeros_like(r)
            level_cost += self.__solve(lvl + 1, w, r, 'K')

            d = w.copy()
            Ad = Ac @ d                         # 1 SpMV
            level_cost += Ac.nnz
            dAd = np.inner(d.conj(), Ad)
            alpha = np.inner(d.conj(), r) / dAd
            coarse_x += alpha * d
            r -= alpha * Ad
            level_cost += 4 * n_c               # 2 dots + 2 axpys

            for _ in range(1, nK):
                # Early exit: skip remaining iterations if residual is small
                if early_exit > 0:
                    norm_r_sq = np.inner(r.conj(), r)
                    if norm_r_sq < early_exit * early_exit * norm_r0_sq:
                        break

                # w = K^{-1} r (preconditioner apply)
                w[:] = 0
                level_cost += self.__solve(lvl + 1, w, r, 'K')

                # A-orthogonalize: d_new = w - (w'Ad / d'Ad) d
                # By symmetry: w'Ad = (Ac@w)'d, but we already have Ad
                beta = np.inner(w.conj(), Ad) / dAd
                d = w - beta * d
                level_cost += 2 * n_c           # 1 dot + 1 axpy

                # Step size and update
                Ad = Ac @ d                     # 1 SpMV
                level_cost += Ac.nnz
                dAd = np.inner(d.conj(), Ad)
                alpha = np.inner(d.conj(), r) / dAd
                coarse_x += alpha * d
                r -= alpha * Ad
                level_cost += 4 * n_c           # 2 dots + 2 axpys

            flops = level_cost
        else:
            raise TypeError(f'Unrecognized cycle type ({cycle})')

        x += self.levels[lvl].P @ coarse_x   # coarse grid correction

        self.levels[lvl].postsmoother(A, x, b)

        return flops


def coarse_grid_solver(solver):
    """Return a coarse grid solver suitable for MultilevelSolver.

    Parameters
    ----------
    solver : str, callable, tuple
        The solver method is either (1) a string such as 'splu' or 'pinv' of a
        callable object which receives only parameters (A, b) and returns an
        (approximate or exact) solution to the linear system Ax = b, or (2) a
        callable object that takes parameters (A,b) and returns an (approximate
        or exact) solution to Ax = b, or (3) a tuple of the form
        (string|callable, args), where args is a dictionary of arguments to
        be passed to the function denoted by string or callable.

        The set of valid string arguments is:
            - Sparse direct methods:
                + splu : sparse LU solver
            - Sparse iterative methods:
                + the name of any method in scipy.sparse.linalg or
                  pyamg.krylov (e.g. 'cg').
                  Methods in pyamg.krylov take precedence.
                + relaxation method, such as 'gauss_seidel' or 'jacobi',
                  present in pyamg.relaxation
            - Dense methods:
                + pinv     : pseudoinverse (SVD)
                + lu       : LU factorization
                + cholesky : Cholesky factorization

    Returns
    -------
    GenericSolver
        A class for use as a standalone or coarse grids solver.

    Examples
    --------
    >>> import numpy as np
    >>> from pyamg.gallery import poisson
    >>> from pyamg import coarse_grid_solver
    >>> A = poisson((10, 10), format='csr')
    >>> b = A @ np.ones(A.shape[0])
    >>> cgs = coarse_grid_solver('lu')
    >>> x = cgs(A, b)

    """

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}

    solver, kwargs = unpack_arg(solver)

    if solver in ['pinv', 'pinv2']:
        def solve(self, A, b):
            if not hasattr(self, 'P'):
                self.P = pinv(A.toarray(), **kwargs)
            return np.dot(self.P, b)

    elif solver == 'lu':
        def solve(self, A, b):
            if not hasattr(self, 'LU'):
                self.LU = sp.linalg.lu_factor(A.toarray(), **kwargs)
            return sp.linalg.lu_solve(self.LU, b)

    elif solver == 'cholesky':
        def solve(self, A, b):
            if not hasattr(self, 'L'):
                self.L = sp.linalg.cho_factor(A.toarray(), **kwargs)
            return sp.linalg.cho_solve(self.L, b)

    elif solver == 'splu':
        def solve(self, A, b):
            if not hasattr(self, 'LU'):
                # for multiple candidates in B, A will often have a couple zero
                # rows/columns that must be removed
                Acsc = A.tocsc()
                Acsc.eliminate_zeros()
                diffptr = Acsc.indptr[:-1] - Acsc.indptr[1:]
                nonzero_cols = (diffptr != 0).nonzero()[0]
                Map = sp.sparse.eye_array(Acsc.shape[0], Acsc.shape[1], format='csc')
                Map = Map[:, nonzero_cols]
                Acsc = Map.T.tocsc() @ Acsc @ Map
                self.LU = sp.sparse.linalg.splu(Acsc, **kwargs)
                self.LU_Map = Map

            return self.LU_Map @ self.LU.solve(np.ravel(self.LU_Map.T @ b))

    elif solver in ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'qmr', 'minres']:
        if hasattr(krylov, solver):
            fn = getattr(krylov, solver)
        else:
            fn = getattr(sla, solver)

        def solve(_, A, b):
            if 'tol' not in kwargs:
                kwargs['tol'] = set_tol(A.dtype)

            return fn(A, b, **kwargs)[0]

    elif solver in ['gauss_seidel', 'jacobi', 'block_gauss_seidel', 'schwarz',
                    'block_jacobi', 'richardson', 'sor', 'chebyshev',
                    'jacobi_ne', 'gauss_seidel_ne', 'gauss_seidel_nr']:

        if 'iterations' not in kwargs:
            kwargs['iterations'] = 10

        def solve(_, A, b):

            lvl = MultilevelSolver.Level()
            lvl.A = A
            fn = getattr(smoothing, 'setup_' + str(solver))
            relax = fn(lvl, **kwargs)
            x = np.zeros_like(b)
            relax(A, x, b)

            return x

    elif solver is None:
        # No coarse grid solve
        def solve(_, __, b):
            return 0 * b  # should this return b instead?

    elif callable(solver):
        def solve(_, A, b):
            return solver(A, b, **kwargs)

    else:
        raise ValueError(f'unknown solver: {solver}')

    class GenericSolver:
        """Generic solver class."""

        def __call__(self, A, b):
            # make sure x is same dimensions and type as b
            b = np.asanyarray(b)

            if A.nnz == 0:
                # if A.nnz = 0, then we expect no correction
                x = np.zeros(b.shape)
            else:
                x = solve(self, A, b)

            if isinstance(b, np.ndarray):
                x = np.asarray(x)
            elif isinstance(b, np.matrix):
                # convert to ndarray
                b = np.asarray(b)
                x = np.asarray(x)
            else:
                raise ValueError('unrecognized type')

            return x.reshape(b.shape)

        def __repr__(self):
            return 'coarse_grid_solver(' + repr(solver) + ')'

        @classmethod
        def name(cls):
            """Return the coarse solver name."""
            return repr(solver)

    return GenericSolver()


class multilevel_solver(MultilevelSolver):  # noqa: N801
    """Deprecated level class.

    .. deprecated:: 4.2.3
              Use :class:`MultilevelSolver` instead.
    """

    def __init__(self, *args, **kwargs):
        """Raise deprecation warning on use, not import."""
        super().__init__(*args, **kwargs)
        warn('multilevel_solver is deprecated.  use MultilevelSolver()',
             category=DeprecationWarning, stacklevel=2)
