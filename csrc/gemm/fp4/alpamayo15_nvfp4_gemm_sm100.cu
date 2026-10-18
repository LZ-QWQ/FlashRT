// SPDX-License-Identifier: Apache-2.0
//
// NVFP4 (e2m1 + UE4M3 block-16 scales) GEMM forms for the Alpamayo 1.5 prefill shapes on Thor (SM100/SM110), to measure
// the kind::f4 rate of the 1-SM and the 2-SM UMMA tiles: D_bf16[M, N] = A[M, K] @ B[N, K]^T (A row-major, B K-major,
// scale factors in the CUTLASS Sm1xx tile-interleaved layout). FP32 accumulation; the epilogue multiplies by alpha (one
// scalar, or one value per row of D: the activations' per-row scales times the weights' per-tensor scale).

#include "gemm/fp4/alpamayo15_nvfp4_gemm_sm100.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include <cstdint>
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"

namespace flash_rt {
namespace fp4 {
namespace a15_nvfp4 {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

template <class Tile, bool k2Sm>
struct Def {
  using Cluster = cute::conditional_t<k2Sm, Shape<_2, _1, _1>, Shape<_1, _1, _1>>;
  using MainSched = cute::conditional_t<k2Sm, cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100,
                                        cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100>;
  using EpiSched = cute::conditional_t<k2Sm, cutlass::epilogue::TmaWarpSpecialized2Sm, cutlass::epilogue::TmaWarpSpecialized1Sm>;
  // alpha * acc (+ beta * C + bias, both off) with alpha a scalar or a per-row vector, chosen per launch
  using Fusion = cutlass::epilogue::fusion::PerRowLinCombPerRowBiasEltAct<cutlass::epilogue::thread::Identity, ElementD, ElementAcc,
                                                                          ElementD, ElementD, ElementAcc>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementAcc,
      ElementD, cutlass::layout::RowMajor, 8, ElementD, cutlass::layout::RowMajor, 8, EpiSched, Fusion>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, cutlass::layout::RowMajor, 32, ElementB, cutlass::layout::ColumnMajor, 32,
      ElementAcc, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      MainSched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using G1 = Def<Shape<_128, _128, _256>, false>::Gemm;    // tile 0: 1-SM 128 x 128
using G2 = Def<Shape<_256, _128, _256>, true>::Gemm;     // tile 1: 2-SM 256 x 128, 2 x 1 cluster
using G3 = Def<Shape<_256, _256, _256>, true>::Gemm;     // tile 2: 2-SM 256 x 256, 2 x 1 cluster

template <class Gemm>
typename Gemm::Arguments make_args(void const* A, void const* SFA, void const* B, void const* SFB, void* D, int M, int N, int K,
                                   int ldd, const float* alpha_ptr, bool alpha_per_row) {
  auto stride_A = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, {M, ldd, 1});
  using Cfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout_SFA = Cfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto layout_SFB = Cfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  using EA = typename ElementA::DataType;
  using SA = typename ElementA::ScaleFactorType;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {static_cast<EA const*>(A), stride_A, static_cast<EA const*>(B), stride_B,
       static_cast<SA const*>(SFA), layout_SFA, static_cast<SA const*>(SFB), layout_SFB},
      {{}, static_cast<ElementD const*>(D), stride_D, static_cast<ElementD*>(D), stride_D}};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.thread.alpha_ptr = alpha_ptr;        // the operands' scales, folded by the activation quantizer
  args.epilogue.thread.dAlpha = {alpha_per_row, cute::_0{}, 0};    // a vector along M, or one scalar
  return args;
}

template <class Gemm>
int run(void const* A, void const* SFA, void const* B, void const* SFB, void* D, int M, int N, int K, int ldd, const float* alpha_ptr,
        bool alpha_per_row, void* ws, size_t ws_bytes, cudaStream_t stream) {
  auto args = make_args<Gemm>(A, SFA, B, SFB, D, M, N, K, ldd, alpha_ptr, alpha_per_row);
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 6;
  if (Gemm::get_workspace_size(args) > ws_bytes) return 7;
  if (gemm.initialize(args, ws, stream) != cutlass::Status::kSuccess) return 8;
  return gemm.run(stream) == cutlass::Status::kSuccess ? 0 : 4;
}

template <class Gemm>
size_t ws_size(int M, int N, int K) {
  return Gemm::get_workspace_size(make_args<Gemm>(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, N, nullptr, false));
}

}  // namespace a15_nvfp4

size_t alpamayo15_nvfp4_gemm_workspace(int M, int N, int K, int tile) {
  if (tile == 1) return a15_nvfp4::ws_size<a15_nvfp4::G2>(M, N, K);
  if (tile == 2) return a15_nvfp4::ws_size<a15_nvfp4::G3>(M, N, K);
  return a15_nvfp4::ws_size<a15_nvfp4::G1>(M, N, K);
}

int alpamayo15_nvfp4_gemm_bf16(void const* A, void const* SFA, void const* B, void const* SFB, void* D, int M, int N, int K,
                               int tile, void* workspace, size_t workspace_bytes, cudaStream_t stream, int ldd,
                               const float* alpha_ptr, int alpha_per_row) {
  if (!A || !SFA || !B || !SFB || !D) return 1;
  if (ldd <= 0) ldd = N;
  if (M <= 0 || N <= 0 || K <= 0 || (K % 64) != 0 || (N % 8) != 0 || ldd < N || (ldd % 8) != 0 || tile < 0 || tile > 2) return 2;
  if (alpha_per_row && (!alpha_ptr || (reinterpret_cast<uintptr_t>(alpha_ptr) & 15u))) return 3;
  const bool per_row = alpha_per_row != 0;
  if (tile == 1) return a15_nvfp4::run<a15_nvfp4::G2>(A, SFA, B, SFB, D, M, N, K, ldd, alpha_ptr, per_row, workspace, workspace_bytes, stream);
  if (tile == 2) return a15_nvfp4::run<a15_nvfp4::G3>(A, SFA, B, SFB, D, M, N, K, ldd, alpha_ptr, per_row, workspace, workspace_bytes, stream);
  return a15_nvfp4::run<a15_nvfp4::G1>(A, SFA, B, SFB, D, M, N, K, ldd, alpha_ptr, per_row, workspace, workspace_bytes, stream);
}

}  // namespace fp4
}  // namespace flash_rt
