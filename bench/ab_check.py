"""Capture a fingerprint of RS-AMG hierarchies (with second_pass=True) and
AIR hierarchies on representative problems. Compare before/after a kernel
change to verify byte-identical algorithmic output.

Usage:
    python bench/ab_check.py BEFORE > /tmp/before.json
    # ... apply change, rebuild ...
    python bench/ab_check.py AFTER  > /tmp/after.json
    diff /tmp/before.json /tmp/after.json
"""
import hashlib
import json
import sys
import numpy as np
import pyamg
from pyamg.gallery import poisson


def hash_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def csr_canonical(M):
    """Return a CSR copy with sorted indices and explicit zeros eliminated.
    Hashing matrices in this canonical form gives an order-independent
    equality check (scipy SpGEMM doesn't guarantee within-row sort order)."""
    C = M.tocsr().copy()
    C.sum_duplicates()
    C.sort_indices()
    return C


def fingerprint(ml):
    levels_fp = []
    for lvl in ml.levels:
        d = {}
        d['shape'] = list(lvl.A.shape)
        d['nnz'] = int(lvl.A.nnz)
        A = csr_canonical(lvl.A)
        d['A_indptr_hash'] = hash_array(A.indptr)
        d['A_indices_hash'] = hash_array(A.indices)
        d['A_data_hash'] = hash_array(A.data)
        if hasattr(lvl, 'P') and lvl.P is not None:
            P = csr_canonical(lvl.P)
            d['P_nnz'] = int(P.nnz)
            d['P_indptr_hash'] = hash_array(P.indptr)
            d['P_indices_hash'] = hash_array(P.indices)
            d['P_data_hash'] = hash_array(P.data)
        if hasattr(lvl, 'R') and lvl.R is not None:
            R = csr_canonical(lvl.R)
            d['R_nnz'] = int(R.nnz)
            d['R_indptr_hash'] = hash_array(R.indptr)
            d['R_indices_hash'] = hash_array(R.indices)
            d['R_data_hash'] = hash_array(R.data)
        if hasattr(lvl, 'splitting') and lvl.splitting is not None:
            s = np.asarray(lvl.splitting)
            d['splitting_hash'] = hash_array(s)
            d['n_C'] = int((s == 1).sum() if s.dtype != bool else s.sum())
        levels_fp.append(d)
    return levels_fp


def converge(ml, b, x0, tol, maxiter):
    res = []
    ml.solve(b, x0=x0, tol=tol, maxiter=maxiter, residuals=res)
    return [float(r) for r in res]


def run(problem_name, A):
    rng = np.random.default_rng(0)
    b = rng.random(A.shape[0])
    x0 = np.zeros(A.shape[0])

    out = {'problem': problem_name, 'A_nnz': int(A.nnz)}
    # Seed legacy numpy RNG so SA's approximate_spectral_radius
    # (random Lanczos init) is reproducible across runs.
    np.random.seed(42)
    ml = pyamg.ruge_stuben_solver(
        A, max_coarse=10,
        CF=('RS', {'second_pass': True}),
        keep=True,
    )
    out['rs_secondpass'] = {
        'fp': fingerprint(ml),
        'res': converge(ml, b, x0, 1e-10, 30),
    }
    ml = pyamg.ruge_stuben_solver(
        A, max_coarse=10,
        CF=('RS', {'second_pass': False}),
        keep=True,
    )
    out['rs_no_secondpass'] = {
        'fp': fingerprint(ml),
        'res': converge(ml, b, x0, 1e-10, 30),
    }
    try:
        from pyamg.classical import air_solver
        ml = air_solver(A, max_coarse=10, keep=True)
        out['air'] = {
            'fp': fingerprint(ml),
            'res': converge(ml, b, x0, 1e-10, 30),
        }
    except Exception as e:  # noqa: BLE001
        out['air'] = {'error': repr(e)}

    # Smoothed aggregation paths (exercises aggregation.py + smooth.py RAP).
    try:
        np.random.seed(42)
        ml = pyamg.smoothed_aggregation_solver(A, max_coarse=10, keep=True)
        out['sa'] = {
            'fp': fingerprint(ml),
            'res': converge(ml, b, x0, 1e-10, 30),
        }
    except Exception as e:  # noqa: BLE001
        out['sa'] = {'error': repr(e)}

    # Rootnode (exercises rootnode.py RAP).
    try:
        np.random.seed(42)
        ml = pyamg.rootnode_solver(A, max_coarse=10, keep=True)
        out['rootnode'] = {
            'fp': fingerprint(ml),
            'res': converge(ml, b, x0, 1e-10, 30),
        }
    except Exception as e:  # noqa: BLE001
        out['rootnode'] = {'error': repr(e)}

    # Pairwise (exercises pairwise.py + aggregate.py intermediate RAP).
    try:
        np.random.seed(42)
        ml = pyamg.pairwise_solver(A, max_coarse=10)
        out['pairwise'] = {
            'fp': fingerprint(ml),
            'res': converge(ml, b, x0, 1e-10, 30),
        }
    except Exception as e:  # noqa: BLE001
        out['pairwise'] = {'error': repr(e)}
    return out


def main(label):
    cases = [
        ('Poisson 1D 1000', poisson((1000,), format='csr')),
        ('Poisson 2D 30x30', poisson((30, 30), format='csr')),
        ('Poisson 2D 50x50', poisson((50, 50), format='csr')),
        ('Poisson 3D 10^3', poisson((10, 10, 10), format='csr')),
    ]
    results = [run(name, A) for name, A in cases]
    print(json.dumps({'label': label, 'results': results}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'unknown')
