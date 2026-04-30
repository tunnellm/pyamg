// Counted variant of rap_pass1.
//
// Same algorithm as rap_pass1 (rap.h), with an additional int64
// accumulator that returns the second-stage SpGEMM block-pattern count:
//
//   sum_{(I, Q) in RA_blocks} bnnz(P[Q, :])
//
// For CSR (bs=1) this equals spmm_work(RA, P) directly. For BSR with
// square bs x bs blocks, the wrapper multiplies by bs^3 to recover the
// scalar FMA count. Combined with spmm_work(R, A) this gives the exact
// FMA count of the fused RAP without ever materializing RA. The counter
// has near-zero overhead — one int64 add per q-chain entry, OMP-reduced
// across threads at the end of the parallel region.
//
// This file lives only in the counting fork; the upstream rap.h is kept
// counter-free for the upstream PR.
//
#ifndef PYAMG_SPARSE_BLAS_RAP_COUNTED_H
#define PYAMG_SPARSE_BLAS_RAP_COUNTED_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstdint>
#include <vector>

#include "rap.h"  // RapPass1Scratch

namespace pyamg {
namespace sparse_blas {

// Pass 1 (symbolic) with second-stage work counting.
// Cp must have length n_row + 1; on return holds (block-)row pointers of C.
// work_RAP_out receives the block-pattern count
//   sum_{(I, Q) in RA} bnnz(P[Q, :]).
// For CSR (bs=1) this is spmm_work(RA, P) in scalar units; for BSR with
// bs x bs blocks the caller multiplies by bs^3.
template <class I>
void rap_pass1_counted(const I n_row,
                       const I n_inner,
                       const I n_col,
                       const I Rp[], const I Rj[],
                       const I Ap[], const I Aj[],
                       const I Pp[], const I Pj[],
                       I Cp[],
                       std::int64_t* work_RAP_out)
{
    std::int64_t accum = 0;

    #pragma omp parallel reduction(+ : accum) if(omp_get_max_threads() > 1)
    {
        static thread_local RapPass1Scratch<I> spa;
        if (spa.q_mask.size() < static_cast<std::size_t>(n_inner)) {
            spa.q_mask.assign(n_inner, -1);
        }
        if (spa.k_mask.size() < static_cast<std::size_t>(n_col)) {
            spa.k_mask.assign(n_col, -1);
        }
        const long long q_base = spa.q_epoch; spa.q_epoch += n_row;
        const long long k_base = spa.k_epoch; spa.k_epoch += n_row;

        #pragma omp for schedule(dynamic, 64)
        for (I i = 0; i < n_row; ++i) {
            const long long qtag = q_base + i;
            const long long ktag = k_base + i;
            spa.q_chain.clear();

            // 1a: discover RA[i, :] pattern.
            for (I jj = Rp[i]; jj < Rp[i + 1]; ++jj) {
                const I p = Rj[jj];
                for (I kk = Ap[p]; kk < Ap[p + 1]; ++kk) {
                    const I q = Aj[kk];
                    if (spa.q_mask[q] != qtag) {
                        spa.q_mask[q] = qtag;
                        spa.q_chain.push_back(q);
                    }
                }
            }

            // 1b: discover RAP[i, :] pattern AND accumulate
            // sum_{q in RA[i,:]} nnz(P[q,:]) = row's contribution to
            // spmm_work(RA, P).
            I row_nnz = 0;
            std::int64_t row_work = 0;
            for (std::size_t idx = 0; idx < spa.q_chain.size(); ++idx) {
                const I q = spa.q_chain[idx];
                row_work += static_cast<std::int64_t>(Pp[q + 1] - Pp[q]);
                for (I kk = Pp[q]; kk < Pp[q + 1]; ++kk) {
                    const I k = Pj[kk];
                    if (spa.k_mask[k] != ktag) {
                        spa.k_mask[k] = ktag;
                        ++row_nnz;
                    }
                }
            }
            Cp[i + 1] = row_nnz;
            accum += row_work;
        }
    }

    Cp[0] = 0;
    for (I i = 0; i < n_row; ++i) Cp[i + 1] += Cp[i];

    if (work_RAP_out) *work_RAP_out = accum;
}

} // namespace sparse_blas
} // namespace pyamg

#endif // PYAMG_SPARSE_BLAS_RAP_COUNTED_H
