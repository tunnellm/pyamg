"""Classical AMG (Ruge-Stuben AMG)."""


from warnings import warn
from scipy.sparse import csr_array, issparse, SparseEfficiencyWarning
import numpy as np

from pyamg.multilevel import MultilevelSolver
from pyamg.relaxation.smoothing import change_smoothers
from pyamg.strength import classical_strength_of_connection, \
    symmetric_strength_of_connection, evolution_strength_of_connection, \
    distance_strength_of_connection, energy_based_strength_of_connection, \
    algebraic_distance, affinity_distance
from pyamg.classical.interpolate import direct_interpolation, classical_interpolation
from . import split
from .cr import CR
from ..util.utils import asfptype, spmm_work, spmm_graph_work
from ..util.sparse_blas import rap_counted, rap_compatible as _rap_ok, \
    with_serial_blas as _with_serial_blas


@_with_serial_blas
def ruge_stuben_solver(A,
                       strength=('classical', {'theta': 0.25}),
                       CF=('RS', {'second_pass': False}),
                       interpolation='classical',
                       presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                       postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                       max_levels=30, max_coarse=10, keep=False, **kwargs):
    """Create a multilevel solver using Classical AMG (Ruge-Stuben AMG).

    Parameters
    ----------
    A : csr_array
        Square matrix in CSR format.
    strength : str
        Valid strings are ['symmetric', 'classical', 'evolution', 'distance',
        'algebraic_distance','affinity', 'energy_based', None].
        Method used to determine the strength of connection between unknowns
        of the linear system.  Method-specific parameters may be passed in
        using a tuple, e.g. strength=('symmetric',{'theta': 0.25 }). If
        strength=None, all nonzero entries of the matrix are considered strong.
    CF : str or tuple, default 'RS'
        Method used for coarse grid selection (C/F splitting).
        Supported methods are RS, PMIS, PMISc, CLJP, CLJPc, and CR.
    interpolation : str, default 'classical'
        Method for interpolation. Options include 'direct', 'classical'.
    presmoother : str or dict
        Method used for presmoothing at each level.  Method-specific parameters
        may be passed in using a tuple, e.g.
        presmoother=('gauss_seidel',{'sweep':'symmetric}), the default.
    postsmoother : str or dict
        Postsmoothing method with the same usage as presmoother.
    max_levels : int, default 30
        Maximum number of levels to be used in the multilevel solver.
    max_coarse : int, default 20
        Maximum number of variables permitted on the coarse grid.
    keep : bool, default False
        Flag to indicate keeping strength of connection (C) in the
        hierarchy for diagnostics.
    **kwargs : dict
        Extra keywords passed to MultilevelSolver class.

    Returns
    -------
    MultilevelSolver
        Multigrid hierarchy of matrices and prolongation operators.

    See Also
    --------
    aggregation.smoothed_aggregation_solver, MultilevelSolver,
    aggregation.rootnode_solver

    Notes
    -----
    "coarse_solver" is an optional argument and is the solver used at the
    coarsest grid.  The default is a pseudo-inverse.  Most simply,
    coarse_solver can be one of ['splu', 'lu', 'cholesky, 'pinv',
    'gauss_seidel', ... ].  Additionally, coarse_solver may be a tuple
    (fn, args), where fn is a string such as ['splu', 'lu', ...] or a callable
    function, and args is a dictionary of arguments to be passed to fn.
    See [1]_ for additional details.

    References
    ----------
    .. [1] Trottenberg, U.; Oosterlee, C. W. & Schüller, A. (2001),
           Multigrid, Vol. 33, Academic Press.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg import ruge_stuben_solver
    >>> A = poisson((10,),format='csr')
    >>> ml = ruge_stuben_solver(A,max_coarse=3)

    """
    levels = [MultilevelSolver.Level()]

    # convert A to csr
    if not issparse(A) or A.format != 'csr':
        try:
            A = csr_array(A)
            warn('Implicit conversion of A to CSR', SparseEfficiencyWarning)
        except Exception as e:
            raise TypeError('Argument A must have type csr_array, '
                            'or be convertible to csr_array') from e
    # preprocess A
    A = asfptype(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    levels[-1].A = A

    # Track work: [numerical_work, graph_work]
    work = [0, 0]

    while len(levels) < max_levels and levels[-1].A.shape[0] > max_coarse:
        bottom = _extend_hierarchy(levels, strength, CF, interpolation, keep, work)

        if bottom:
            break

    ml = MultilevelSolver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother, work=work)
    ml._setup_work = tuple(work)
    return ml


# internal function
def _extend_hierarchy(levels, strength, CF, interpolation, keep, work=None):
    """Extend the multigrid hierarchy."""
    if work is None:
        work = [0, 0]

    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}

    A = levels[-1].A

    # Compute the strength-of-connection matrix C, where larger
    # C[i,j] denote stronger couplings between i and j.
    fn, kwargs = unpack_arg(strength)
    n = A.shape[0]
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(A, work=work, **kwargs)
    elif fn == 'classical':
        C = classical_strength_of_connection(A, work=work, **kwargs)
    elif fn == 'distance':
        C = distance_strength_of_connection(A, **kwargs)
    elif fn in ('ode', 'evolution'):
        C = evolution_strength_of_connection(A, work=work, **kwargs)
    elif fn == 'energy_based':
        C = energy_based_strength_of_connection(A, work=work, **kwargs)
    elif fn == 'algebraic_distance':
        C = algebraic_distance(A, work=work, **kwargs)
    elif fn == 'affinity':
        C = affinity_distance(A, work=work, **kwargs)
    elif fn is None:
        C = A
    else:
        raise ValueError(f'Unrecognized strength of connection method: {fn}')

    # Generate the C/F splitting
    # All coarsening algorithms are iterative graph algorithms over C.
    # Numerical: nnz(C) + n for measure computation (lambda + random augmentation)
    # Graph: C_iter * 2*nnz(C) for IS selection iterations (traverse C and C^T)
    # C_iter varies: RS~2, PMIS/CLJP~3 iterations on average
    fn, kwargs = unpack_arg(CF)
    if fn == 'RS':
        splitting = split.RS(C, work=work, **kwargs)
        # RS work counted inside split.RS
    elif fn == 'PMIS':
        splitting = split.PMIS(C, work=work, **kwargs)
        # PMIS work counted inside split.PMIS (preprocessing + MIS iterations)
    elif fn == 'PMISc':
        splitting = split.PMISc(C, work=work, **kwargs)
        # PMISc work counted inside split.PMISc (preprocessing + coloring + MIS iterations)
    elif fn == 'CLJP':
        splitting = split.CLJP(C, work=work, **kwargs)
        # CLJP work counted inside split.CLJP (preprocessing + iterations)
    elif fn == 'CLJPc':
        splitting = split.CLJPc(C, work=work, **kwargs)
        # CLJPc work counted inside split.CLJPc (delegates to CLJP with color=True)
    elif fn == 'CR':
        splitting = CR(C, work=work, **kwargs)
        # CR work counted inside CR (actual GS iterations + cr_helper)
    else:
        raise ValueError(f'Unknown C/F splitting method {CF}')

    # Make sure all points were not declared as C- or F-points
    # Return early, do not add another coarse level
    num_fpts = np.sum(splitting)
    if (num_fpts == len(splitting)) or (num_fpts == 0):
        return True

    # Generate the interpolation matrix that maps from the coarse-grid to the
    # fine-grid
    fn, kwargs = unpack_arg(interpolation)
    # Estimate nnz(C_FF) ≈ nnz(C)/2 (half of strong connections are F-F)
    # Estimate nnz(A_F) ≈ nnz(A)/2, nnz(C_F) ≈ nnz(C)/2 (half the rows are F-points)
    nnz_C_FF = C.nnz // 2
    nnz_A_F = A.nnz // 2
    nnz_C_F = C.nnz // 2
    if fn == 'classical':
        P = classical_interpolation(A, C, splitting, **kwargs)
        # Pre-processing (Python wrapper): NOT counted (memory copies, scipy overhead)
        # remove_strong_FF_connections: nnz(C_FF)*nnz(C)/n graph (search S_j for common C-pts)
        ff_search_cost = nnz_C_FF * C.nnz // n
        work[1] += ff_search_cost  # graph: FF connection removal search
        # Pass 1: traverse C for F-points = nnz(C_F) graph
        # Pass 2 for each F-point i:
        #   - Sum A row for denominator: nnz(A_F) adds
        #   - Subtract strong connections from denominator: nnz(S_F) ≈ nnz(C)/2 subtracts
        #   - For each strong F-neighbor k:
        #     - Search A_k for a_kj: nnz(A)/n graph per search
        #     - Inner denominator: search A_k for each C-neighbor: nnz(A)/n graph
        #     - Arithmetic (a_ik * a_kj / denom, accumulate): 2 numerical ops per search
        #   - Final weight = -numerator/denominator: 2 ops per P entry
        # Graph: search cost ≈ nnz(C_FF) * nnz(A) / n
        # Numerical: 2× search cost (a_kj lookup + inner_denom) + 2*(nnz(P)-n_C) normalization
        n_C = P.shape[1]
        search_cost = nnz_C_FF * A.nnz // n
        work[1] += 2 * n + nnz_C_F + search_cost + P.nnz  # graph: pass1 + remap(n+nnz_P) + A-row searches
        work[0] += nnz_A_F + nnz_C_F + 2 * search_cost + 2 * (P.nnz - n_C)  # numerical
    elif fn == 'direct':
        P = direct_interpolation(A, C, splitting, **kwargs)
        # Pre-processing (Python wrapper): NOT counted (memory copies, scipy overhead)
        # Pass 1: traverse C for F-points = nnz(C_F) graph
        # Pass 2 for each F-point i:
        #   - Sum strong pos/neg over S row: nnz(S_F) ≈ nnz(C)/2 comparisons + adds
        #   - Sum all pos/neg over A row: nnz(A_F) ≈ nnz(A)/2 adds
        #   - 4 divisions per F-point for alpha, beta, neg_coeff, pos_coeff: n_F
        #   - 1 mul per P entry for weight: nnz(P) - n_C
        # Final remap: n + nnz(P) graph
        n_C = P.shape[1]
        n_F = n - n_C
        work[1] += 2 * n + nnz_C_F + P.nnz  # graph: pass1 traversal + remap(n+nnz_P)
        work[0] += nnz_A_F + nnz_C_F + 4 * n_F + (P.nnz - n_C)  # numerical
    else:
        raise ValueError(f'Unknown interpolation method {interpolation}')

    # Generate the restriction matrix that maps from the fine-grid to the
    # coarse-grid
    R = P.T.tocsr()
    # Work: nnz(P) graph for transpose
    work[1] += P.nnz

    # Store relevant information for this level
    if keep:
        levels[-1].C = C                           # strength of connection matrix

    levels[-1].splitting = splitting.astype(bool)  # C/F splitting
    levels[-1].P = P                               # prolongation operator
    levels[-1].R = R                               # restriction operator

    # Fused RAP via the threaded kernel; rap_counted returns the exact FMA
    # count (== spmm_work(R,A) + spmm_work(RA,P)) without materializing RA.
    levels.append(MultilevelSolver.Level())
    if _rap_ok(R, A, P):
        A, fma, graph = rap_counted(R, A, P)
    else:
        RA = R @ A
        fma = spmm_work(R, A) + spmm_work(RA, P)
        graph = spmm_graph_work(R, A) + spmm_graph_work(RA, P)
        A = RA @ P
    work[0] += fma    # numerical
    work[1] += graph  # graph
    levels[-1].A = A
    return False
