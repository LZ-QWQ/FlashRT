// SPDX-License-Identifier: Apache-2.0
// NVFP4 GEMM tile forms for the Alpamayo 1.5 prefill shapes (SM100/SM110): D_bf16[M, N] = A[M, K] @ B[N, K]^T with
// e2m1 operands and UE4M3 block-16 scales in the CUTLASS Sm1xx tile-interleaved layout. tile 0 = 1-SM 128x128x256,
// 1 = 2-SM 256x128x256, 2 = 2-SM 256x256x256 (2 x 1 clusters). Workspace from alpamayo15_nvfp4_gemm_workspace. D rows
// ldd elements apart (0 = N; a column block of a wider row-major tensor); alpha_ptr = optional device float(s) multiplied
// into the output: one scalar (alpha_per_row 0: the two operands' per-tensor scales) or one per row of D (alpha_per_row 1:
// M floats, 16-byte aligned, the activations' per-row scales times the weight's scale); nullptr = 1.
#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace flash_rt {
namespace fp4 {

size_t alpamayo15_nvfp4_gemm_workspace(int M, int N, int K, int tile);
int alpamayo15_nvfp4_gemm_bf16(void const* A, void const* SFA, void const* B, void const* SFB, void* D, int M, int N, int K,
                               int tile, void* workspace, size_t workspace_bytes, cudaStream_t stream, int ldd = 0,
                               const float* alpha_ptr = nullptr, int alpha_per_row = 0);

}  // namespace fp4
}  // namespace flash_rt
