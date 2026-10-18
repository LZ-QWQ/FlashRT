// SPDX-License-Identifier: Apache-2.0
//
// cuBLASLt BF16 GEMM with fused epilogues for the Alpamayo 1.5 vision tower on Thor: D = act(A @ W^T + bias) + C in
// one kernel (FP32 accumulation, one BF16 rounding), the row-major operands presented to cuBLASLt in the column-major
// formulation PyTorch uses (D^T = W^T-as-N x K times A^T), which is the one the Thor library has fast kernels for.
// Descriptors and the heuristic's algorithm are cached per shape; the handle and a 32 MB workspace are created by
// alpamayo15_cublaslt_init() (call it outside CUDA graph capture).

#include "../kernels/alpamayo15_kernels.cuh"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

namespace flash_rt {
namespace kernels {

namespace {

struct LtPlan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
  cublasLtMatmulAlgo_t algo{};
};

cublasLtHandle_t g_lt = nullptr;
void* g_ws = nullptr;
constexpr size_t kWorkspace = 32u << 20;
std::mutex g_mu;
std::map<std::tuple<int, int, int, int, int, int, int, int>, LtPlan> g_plans;

bool make_plan(LtPlan& p, int M, int N, int K, int ldc, int ldd, int act, bool has_bias, bool has_c) {
  if (cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_DEFAULT;
  if (act == 1) epi = has_bias ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_GELU;
  else if (has_bias) epi = CUBLASLT_EPILOGUE_BIAS;
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
  if (has_bias) {
    const cudaDataType_t bt = CUDA_R_16BF;
    cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt));
  }
  // column-major views: A' = W (K x N, ld K) transposed -> N x K; B' = A (K x M, ld K); C' / D' = N x M (ld ldc / ldd)
  if (cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, K) != CUBLAS_STATUS_SUCCESS) return false;
  if (cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, M, K) != CUBLAS_STATUS_SUCCESS) return false;
  if (cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, N, M, has_c ? ldc : ldd) != CUBLAS_STATUS_SUCCESS) return false;
  if (cublasLtMatrixLayoutCreate(&p.d, CUDA_R_16BF, N, M, ldd) != CUBLAS_STATUS_SUCCESS) return false;
  cublasLtMatmulPreference_t pref = nullptr;
  if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return false;
  cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspace, sizeof(kWorkspace));
  cublasLtMatmulHeuristicResult_t res[4];
  int n = 0;
  const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(g_lt, p.desc, p.a, p.b, p.c, p.d, pref, 4, res, &n);
  cublasLtMatmulPreferenceDestroy(pref);
  if (st != CUBLAS_STATUS_SUCCESS || n == 0) return false;
  p.algo = res[0].algo;
  return true;
}

}  // namespace

int alpamayo15_cublaslt_init() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_lt && cublasLtCreate(&g_lt) != CUBLAS_STATUS_SUCCESS) return 1;
  if (!g_ws && cudaMalloc(&g_ws, kWorkspace) != cudaSuccess) return 2;
  return 0;
}

int alpamayo15_cublaslt_gemm_bf16(const void* A, const void* W, const void* bias, const void* C, void* D, int M, int N,
                                  int K, int ldc, int ldd, int act, cudaStream_t stream) {
  if (!A || !W || !D) return 1;
  if (M <= 0 || N <= 0 || K <= 0 || (K % 8) != 0 || (N % 8) != 0 || act < 0 || act > 1) return 2;
  if (ldd < N || (ldd % 8) != 0 || (C && (ldc < N || (ldc % 8) != 0))) return 2;
  if ((reinterpret_cast<uintptr_t>(A) & 15u) || (reinterpret_cast<uintptr_t>(W) & 15u) || (reinterpret_cast<uintptr_t>(D) & 15u) ||
      (C && (reinterpret_cast<uintptr_t>(C) & 15u)) || (bias && (reinterpret_cast<uintptr_t>(bias) & 15u)))
    return 5;
  if (!g_lt || !g_ws) return 3;      // alpamayo15_cublaslt_init() first (outside graph capture)
  LtPlan* plan;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    const auto key = std::make_tuple(M, N, K, C ? ldc : 0, ldd, act, bias ? 1 : 0, C ? 1 : 0);
    auto it = g_plans.find(key);
    if (it == g_plans.end()) {
      LtPlan p;
      if (!make_plan(p, M, N, K, ldc, ldd, act, bias != nullptr, C != nullptr)) return 6;
      it = g_plans.emplace(key, p).first;
    }
    plan = &it->second;
  }
  if (bias) cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
  const float alpha = 1.0f, beta = C ? 1.0f : 0.0f;
  const cublasStatus_t st = cublasLtMatmul(g_lt, plan->desc, &alpha, W, plan->a, A, plan->b, &beta, C ? C : D, plan->c, D, plan->d,
                                           &plan->algo, g_ws, kWorkspace, stream);
  return st == CUBLAS_STATUS_SUCCESS ? 0 : 4;
}

}  // namespace kernels
}  // namespace flash_rt
