// SPDX-License-Identifier: Apache-2.0
//
// cuBLASLt GEMMs of the Alpamayo 1.5 engine on Thor, called from C++ with no framework in between:
//   BF16: D = act(A @ W^T + bias) (+ C), FP32 accumulation, one BF16 rounding (act 0 identity, 1 GELU tanh);
//   FP8 : D = bf16((A8 @ W8^T) * a_scale * w_scale), e4m3 operands with per-tensor scales read from device memory
//         (static scales, so the call is graph-capturable and needs no host synchronization).
// Row-major operands are presented to cuBLASLt in the column-major TN formulation (D^T = W x A^T), the one the Thor
// library has its fastest kernels for (nvjet). Descriptors and the heuristic's first algorithm are cached per shape;
// the handle and a 32 MB workspace are created by alpamayo15_cublaslt_init(), which must run outside graph capture,
// and every new shape must be seen once outside capture (the engines' warm-up passes do this).

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
// key: kind (0 bf16, 1 fp8), M, N, K, ldc, ldd, act, bias, C
std::map<std::tuple<int, int, int, int, int, int, int, int, int>, LtPlan> g_plans;

bool make_plan(LtPlan& p, int kind, int M, int N, int K, int ldc, int ldd, int act, bool has_bias, bool has_c) {
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
  const cudaDataType_t in_t = kind == 1 ? CUDA_R_8F_E4M3 : CUDA_R_16BF;
  // column-major views: A' = W (K x N, ld K) transposed -> N x K; B' = A (K x M, ld K); C' / D' = N x M
  if (cublasLtMatrixLayoutCreate(&p.a, in_t, K, N, K) != CUBLAS_STATUS_SUCCESS) return false;
  if (cublasLtMatrixLayoutCreate(&p.b, in_t, K, M, K) != CUBLAS_STATUS_SUCCESS) return false;
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

LtPlan* plan_for(int kind, int M, int N, int K, int ldc, int ldd, int act, bool has_bias, bool has_c) {
  std::lock_guard<std::mutex> lock(g_mu);
  const auto key = std::make_tuple(kind, M, N, K, has_c ? ldc : 0, ldd, act, has_bias ? 1 : 0, has_c ? 1 : 0);
  auto it = g_plans.find(key);
  if (it == g_plans.end()) {
    LtPlan p;
    if (!make_plan(p, kind, M, N, K, ldc, ldd, act, has_bias, has_c)) return nullptr;
    it = g_plans.emplace(key, p).first;
  }
  return &it->second;
}

inline bool al16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

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
  // one output row (M = 1, e.g. the LM head of one token) takes any N: its column-major view is a single column
  if (M <= 0 || N <= 0 || K <= 0 || (K % 8) != 0 || (M > 1 && (N % 8) != 0) || act < 0 || act > 1) return 2;
  if (ldd < N || (M > 1 && (ldd % 8) != 0) || (C && (ldc < N || (ldc % 8) != 0))) return 2;
  if (!al16(A) || !al16(W) || !al16(D) || (C && !al16(C)) || (bias && !al16(bias))) return 5;
  if (!g_lt || !g_ws) return 3;      // alpamayo15_cublaslt_init() first (outside graph capture)
  LtPlan* plan = plan_for(0, M, N, K, ldc, ldd, act, bias != nullptr, C != nullptr);
  if (!plan) return 6;
  if (bias) cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
  const float alpha = 1.0f, beta = C ? 1.0f : 0.0f;
  const cublasStatus_t st = cublasLtMatmul(g_lt, plan->desc, &alpha, W, plan->a, A, plan->b, &beta, C ? C : D, plan->c, D, plan->d,
                                           &plan->algo, g_ws, kWorkspace, stream);
  return st == CUBLAS_STATUS_SUCCESS ? 0 : 4;
}

int alpamayo15_cublaslt_gemm_fp8(const void* A8, const void* W8, const float* a_scale, const float* w_scale, void* D, int M,
                                 int N, int K, int ldd, cudaStream_t stream) {
  if (!A8 || !W8 || !a_scale || !w_scale || !D) return 1;
  if (M <= 0 || N <= 0 || K <= 0 || (K % 16) != 0 || (N % 16) != 0 || ldd < N || (ldd % 8) != 0) return 2;
  if (!al16(A8) || !al16(W8) || !al16(D)) return 5;
  if (!g_lt || !g_ws) return 3;
  LtPlan* plan = plan_for(1, M, N, K, 0, ldd, 0, false, false);
  if (!plan) return 6;
  // cuBLASLt's "A" is the weight here (column-major formulation), its "B" the activations
  cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_scale, sizeof(w_scale));
  cublasLtMatmulDescSetAttribute(plan->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_scale, sizeof(a_scale));
  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t st = cublasLtMatmul(g_lt, plan->desc, &alpha, W8, plan->a, A8, plan->b, &beta, D, plan->c, D, plan->d,
                                           &plan->algo, g_ws, kWorkspace, stream);
  return st == CUBLAS_STATUS_SUCCESS ? 0 : 4;
}

}  // namespace kernels
}  // namespace flash_rt
