// SPDX-License-Identifier: Apache-2.0
//
// Alpamayo 1.5 (Thor, sm_110a) BF16 GEMMs on tcgen05 through CUTLASS: D = act(acc + bias[n]) (+ C), FP32
// accumulation, BF16 operands and output. A [M, K] row-major, B = weight [N, K] row-major (nn.Linear layout),
// bias [N] (optional), C [M, N] row-major residual (optional, added after the activation; identity only),
// D [M, N] row-major.
//   act: 0 identity, 1 GELU (tanh approximation, gelu_pytorch_tanh), 2 GELU (erf)
//   tile: 0 = 128 x 128 x 64 per SM, 2 = 256 x 128 x 64 and 3 = 256 x 256 x 64 on a 2-SM UMMA (2 x 1 cluster; the
//         vision tower: every activation / residual form), 1 = 64 x 128 x 64 (few-row operands such as the action
//         expert; identity, no residual)
// The residual is added to the activated value, so proj / fc2 (x + linear(..)) and the patch embedding (+ position
// embedding) are one launch each. 16-byte alignment on every operand; K % 8 == 0, N % 8 == 0.

#include <cuda_runtime.h>

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace {

using ElementAB = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;
using ElementCompute = float;
constexpr int kAlign = 8;

namespace fe = cutlass::epilogue::fusion;
constexpr auto kRN = cutlass::FloatRoundStyle::round_to_nearest;

template <class Tile>
using BiasNode = fe::Sm90RowBroadcast<0, Tile, ElementAB, ElementCompute, Stride<_0, _1, int64_t>>;
template <class Tile>
using AccBias = fe::Sm90EVT<fe::Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, kRN>, fe::Sm90AccFetch, BiasNode<Tile>>;
// D = act(acc + bias)
template <class Tile, template <class> class Act>
using NoResTree = fe::Sm90EVT<fe::Sm90Compute<Act, ElementD, ElementCompute, kRN>, AccBias<Tile>>;
// D = (acc + bias) + C
template <class Tile>
using ResTree = fe::Sm90EVT<fe::Sm90Compute<cutlass::plus, ElementD, ElementCompute, kRN>, AccBias<Tile>, fe::Sm90SrcFetch<ElementAB>>;

template <class Tile, class Fusion, bool k2Sm = false>
struct GemmDef {
  using Cluster = cute::conditional_t<k2Sm, Shape<_2, _1, _1>, Shape<_1, _1, _1>>;
  using MainSched = cute::conditional_t<k2Sm, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                                        cutlass::gemm::collective::KernelScheduleAuto>;
  using EpiSched = cute::conditional_t<k2Sm, cutlass::epilogue::TmaWarpSpecialized2Sm,
                                       cutlass::epilogue::collective::EpilogueScheduleAuto>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementCompute, ElementAB,
      cutlass::layout::RowMajor, kAlign, ElementD, cutlass::layout::RowMajor, kAlign,
      EpiSched, Fusion>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementAB, cutlass::layout::RowMajor, kAlign, ElementAB,
      cutlass::layout::ColumnMajor, kAlign, ElementAcc, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      MainSched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// FP8 E4M3 x FP8 E4M3 -> BF16 with per-row activation scales and per-column weight scales:
// D[m, n] = acc[m, n] * a_s[m] * w_s[n]  (the W8A8 tier of the action expert).
using ElementF8 = cutlass::float_e4m3_t;
template <class Tile>
using RowScaleNode = fe::Sm90ColBroadcast<0, Tile, float, ElementCompute, Stride<_1, _0, int64_t>>;   // per m
template <class Tile>
using ColScaleNode = fe::Sm90RowBroadcast<0, Tile, float, ElementCompute, Stride<_0, _1, int64_t>>;   // per n
template <class Tile>
using Fp8ScaleTree = fe::Sm90EVT<fe::Sm90Compute<cutlass::multiplies, ElementD, ElementCompute, kRN>,
    fe::Sm90EVT<fe::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, kRN>, fe::Sm90AccFetch, RowScaleNode<Tile>>,
    ColScaleNode<Tile>>;

// The same with bias / activation / residual (the W8A8 vision tier): D = act(acc * a_s[m] * w_s[n] + bias[n]) (+ C).
template <class Tile>
using Fp8Scaled = fe::Sm90EVT<fe::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, kRN>,
    fe::Sm90EVT<fe::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, kRN>, fe::Sm90AccFetch, RowScaleNode<Tile>>,
    ColScaleNode<Tile>>;
template <class Tile>
using Fp8ScaledBias = fe::Sm90EVT<fe::Sm90Compute<cutlass::plus, ElementCompute, ElementCompute, kRN>, Fp8Scaled<Tile>, BiasNode<Tile>>;
template <class Tile, template <class> class Act>
using Fp8ActTree = fe::Sm90EVT<fe::Sm90Compute<Act, ElementD, ElementCompute, kRN>, Fp8ScaledBias<Tile>>;
template <class Tile>
using Fp8ResTree = fe::Sm90EVT<fe::Sm90Compute<cutlass::plus, ElementD, ElementCompute, kRN>, Fp8ScaledBias<Tile>, fe::Sm90SrcFetch<ElementAB>>;

template <class Tile, bool k2Sm = false, class Fusion_ = Fp8ScaleTree<Tile>, class ElemC = void>
struct Fp8GemmDef {
  using Cluster = cute::conditional_t<k2Sm, Shape<_2, _1, _1>, Shape<_1, _1, _1>>;
  using MainSched = cute::conditional_t<k2Sm, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                                        cutlass::gemm::collective::KernelScheduleAuto>;
  using EpiSched = cute::conditional_t<k2Sm, cutlass::epilogue::TmaWarpSpecialized2Sm,
                                       cutlass::epilogue::collective::EpilogueScheduleAuto>;
  using Fusion = Fusion_;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementCompute, ElemC,
      cutlass::layout::RowMajor, kAlign, ElementD, cutlass::layout::RowMajor, kAlign,
      EpiSched, Fusion>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementF8, cutlass::layout::RowMajor, 16, ElementF8,
      cutlass::layout::ColumnMajor, 16, ElementAcc, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      MainSched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
using F8G64 = Fp8GemmDef<Shape<_64, _128, _128>>;            // tile 1: few-row operands (action expert)
using F8G128 = Fp8GemmDef<Shape<_128, _128, _128>>;          // tile 0
using F8G256P = Fp8GemmDef<Shape<_256, _128, _128>, true>;   // tile 2: 2-SM UMMA, 2 x 1 cluster (prefill)
using Tile256F8 = Shape<_256, _128, _128>;
template <class T> using IdentityActF = cutlass::epilogue::thread::Identity<T>;
template <class T> using GeluTanhActF = cutlass::epilogue::thread::GELU_taylor<T>;
template <class T> using GeluErfActF = cutlass::epilogue::thread::GELU<T>;
using F8G256Bias = Fp8GemmDef<Tile256F8, true, Fp8ActTree<Tile256F8, IdentityActF>>;
using F8G256Tanh = Fp8GemmDef<Tile256F8, true, Fp8ActTree<Tile256F8, GeluTanhActF>>;
using F8G256Erf = Fp8GemmDef<Tile256F8, true, Fp8ActTree<Tile256F8, GeluErfActF>>;
using F8G256Res = Fp8GemmDef<Tile256F8, true, Fp8ResTree<Tile256F8>, ElementAB>;

// INT8 x INT8 -> FP16 with per-row activation scales and per-column weight scales (per-channel INT8 prefill tier):
// D[m, n] = half(float(acc[m, n]) * a_s[m] * w_s[n]), int32 accumulation.
using ElementI8 = int8_t;
using ElementI32 = int32_t;
using ElementH = cutlass::half_t;
template <class Tile>
using Int8ScaleTree = fe::Sm90EVT<fe::Sm90Compute<cutlass::multiplies, ElementH, ElementCompute, kRN>,
    fe::Sm90EVT<fe::Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, kRN>, fe::Sm90AccFetch, RowScaleNode<Tile>>,
    ColScaleNode<Tile>>;

template <class Tile, bool k2Sm = false>
struct Int8GemmDef {
  using Cluster = cute::conditional_t<k2Sm, Shape<_2, _1, _1>, Shape<_1, _1, _1>>;
  using MainSched = cute::conditional_t<k2Sm, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                                        cutlass::gemm::collective::KernelScheduleAuto>;
  using EpiSched = cute::conditional_t<k2Sm, cutlass::epilogue::TmaWarpSpecialized2Sm,
                                       cutlass::epilogue::collective::EpilogueScheduleAuto>;
  using Fusion = Int8ScaleTree<Tile>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementI32, ElementCompute, void,
      cutlass::layout::RowMajor, kAlign, ElementH, cutlass::layout::RowMajor, kAlign,
      EpiSched, Fusion>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementI8, cutlass::layout::RowMajor, 16, ElementI8,
      cutlass::layout::ColumnMajor, 16, ElementI32, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      MainSched>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
using I8G128 = Int8GemmDef<Shape<_128, _128, _128>>;
using I8G64 = Int8GemmDef<Shape<_64, _128, _128>>;
using I8G256P = Int8GemmDef<Shape<_256, _128, _128>, true>;   // 2-SM UMMA, 2 x 1 cluster

template <class G>
typename G::Gemm::Arguments make_int8_args(const void* A, const void* B, const float* a_s, const float* w_s, void* D,
                                           int M, int N, int K) {
  using Gemm = typename G::Gemm;
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, make_shape(N, K, 1));
  auto sD = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, make_shape(M, N, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                {M, N, K, 1},
                                {static_cast<const ElementI8*>(A), sA, static_cast<const ElementI8*>(B), sB},
                                {{}, nullptr, sD, static_cast<ElementH*>(D), sD}};
  auto& f = args.epilogue.thread;
  f.op_0.op_1.ptr_col = a_s;
  f.op_0.op_1.null_default = 1.0f;
  f.op_0.op_1.dCol = {_1{}, _0{}, int64_t(0)};
  f.op_1.ptr_row = w_s;
  f.op_1.null_default = 1.0f;
  f.op_1.dRow = {_0{}, _1{}, int64_t(0)};
  return args;
}

template <class G>
int run_int8(const void* A, const void* B, const float* a_s, const float* w_s, void* D, int M, int N, int K,
             void* workspace, unsigned long long workspace_bytes, int pdl, cudaStream_t stream) {
  using Gemm = typename G::Gemm;
  auto args = make_int8_args<G>(A, B, a_s, w_s, D, M, N, K);
  Gemm gemm;
  if (Gemm::get_workspace_size(args) > workspace_bytes) return 3;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 5;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return 6;
  if (gemm.run(stream, nullptr, pdl != 0) != cutlass::Status::kSuccess) return 7;
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

template <class G>
typename G::Gemm::Arguments make_fp8_args(const void* A, const void* B, const float* a_s, const float* w_s, void* D,
                                          int M, int N, int K, int ldd) {
  using Gemm = typename G::Gemm;
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, make_shape(N, K, 1));
  typename Gemm::GemmKernel::StrideD sD{int64_t(ldd), _1{}, int64_t(0)};    // D rows ldd elements apart
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                {M, N, K, 1},
                                {static_cast<const ElementF8*>(A), sA, static_cast<const ElementF8*>(B), sB},
                                {{}, nullptr, sD, static_cast<ElementD*>(D), sD}};
  auto& f = args.epilogue.thread;
  f.op_0.op_1.ptr_col = a_s;                  // per-row (M) activation scale
  f.op_0.op_1.null_default = 1.0f;
  f.op_0.op_1.dCol = {_1{}, _0{}, int64_t(0)};
  f.op_1.ptr_row = w_s;                       // per-column (N) weight scale
  f.op_1.null_default = 1.0f;
  f.op_1.dRow = {_0{}, _1{}, int64_t(0)};
  return args;
}

template <class T> using IdentityAct = cutlass::epilogue::thread::Identity<T>;
template <class T> using GeluTanhAct = cutlass::epilogue::thread::GELU_taylor<T>;
template <class T> using GeluErfAct = cutlass::epilogue::thread::GELU<T>;

using Tile128 = Shape<_128, _128, _64>;
using Tile64 = Shape<_64, _128, _64>;
using Tile256 = Shape<_256, _128, _64>;
using Tile256W = Shape<_256, _256, _64>;

using G128Id = GemmDef<Tile128, NoResTree<Tile128, IdentityAct>>;
using G128Tanh = GemmDef<Tile128, NoResTree<Tile128, GeluTanhAct>>;
using G128Erf = GemmDef<Tile128, NoResTree<Tile128, GeluErfAct>>;
using G128Res = GemmDef<Tile128, ResTree<Tile128>>;
using G64Id = GemmDef<Tile64, NoResTree<Tile64, IdentityAct>>;
using G256Id = GemmDef<Tile256, NoResTree<Tile256, IdentityAct>, true>;      // 2-SM UMMA, 2 x 1 cluster
using G256Tanh = GemmDef<Tile256, NoResTree<Tile256, GeluTanhAct>, true>;
using G256Erf = GemmDef<Tile256, NoResTree<Tile256, GeluErfAct>, true>;
using G256Res = GemmDef<Tile256, ResTree<Tile256>, true>;
using G256WId = GemmDef<Tile256W, NoResTree<Tile256W, IdentityAct>, true>;   // tile 3: 2-SM UMMA, 256 x 256
using G256WTanh = GemmDef<Tile256W, NoResTree<Tile256W, GeluTanhAct>, true>;
using G256WErf = GemmDef<Tile256W, NoResTree<Tile256W, GeluErfAct>, true>;
using G256WRes = GemmDef<Tile256W, ResTree<Tile256W>, true>;

template <class Def, bool kRes>
typename Def::Gemm::Arguments make_args(const void* A, const void* B, const void* bias, const void* C, void* D, int M,
                                        int N, int K) {
  using Gemm = typename Def::Gemm;
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, make_shape(N, K, 1));
  auto sC = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, make_shape(M, N, 1));
  auto sD = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{}, make_shape(M, N, 1));
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                {M, N, K, 1},
                                {static_cast<const ElementAB*>(A), sA, static_cast<const ElementAB*>(B), sB},
                                {{}, static_cast<const ElementAB*>(C), sC, static_cast<ElementD*>(D), sD}};
  auto& f = args.epilogue.thread;
  auto& bias_args = f.op_0.op_1;        // (acc + bias) is the first child of both trees; bias its second child
  bias_args.ptr_row = static_cast<const ElementAB*>(bias);
  bias_args.null_default = ElementAB(0);
  bias_args.dRow = {_0{}, _1{}, int64_t(0)};
  return args;
}

template <class Def, bool kRes>
int run(const void* A, const void* B, const void* bias, const void* C, void* D, int M, int N, int K, void* ws,
        size_t ws_bytes, int pdl, cudaStream_t stream) {
  using Gemm = typename Def::Gemm;
  auto args = make_args<Def, kRes>(A, B, bias, C, D, M, N, K);
  Gemm gemm;
  if (Gemm::get_workspace_size(args) > ws_bytes) return 3;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 5;
  if (gemm.initialize(args, ws, stream) != cutlass::Status::kSuccess) return 6;
  if (gemm.run(stream, nullptr, pdl != 0) != cutlass::Status::kSuccess) return 7;
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

template <class Def, bool kRes>
unsigned long long ws_size(int M, int N, int K) {
  return Def::Gemm::get_workspace_size(make_args<Def, kRes>(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K));
}

// bias / activation (/ residual C) forms: the scaled accumulator plus the bias is the first child of both trees.
template <class G, bool kRes>
typename G::Gemm::Arguments make_fp8_act_args(const void* A, const void* B, const float* a_s, const float* w_s,
                                              const void* bias, const void* C, void* D, int M, int N, int K, int ldd) {
  using Gemm = typename G::Gemm;
  auto sA = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, make_shape(M, K, 1));
  auto sB = cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, make_shape(N, K, 1));
  typename Gemm::GemmKernel::StrideD sD{int64_t(ldd), _1{}, int64_t(0)};
  typename Gemm::GemmKernel::StrideC sC{int64_t(ldd), _1{}, int64_t(0)};
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                {M, N, K, 1},
                                {static_cast<const ElementF8*>(A), sA, static_cast<const ElementF8*>(B), sB},
                                {{}, static_cast<const ElementAB*>(C), sC, static_cast<ElementD*>(D), sD}};
  auto& sb = args.epilogue.thread.op_0;        // (acc * a_s * w_s) + bias
  sb.op_0.op_0.op_1.ptr_col = a_s;
  sb.op_0.op_0.op_1.null_default = 1.0f;
  sb.op_0.op_0.op_1.dCol = {_1{}, _0{}, int64_t(0)};
  sb.op_0.op_1.ptr_row = w_s;
  sb.op_0.op_1.null_default = 1.0f;
  sb.op_0.op_1.dRow = {_0{}, _1{}, int64_t(0)};
  sb.op_1.ptr_row = static_cast<const ElementAB*>(bias);
  sb.op_1.null_default = ElementAB(0);
  sb.op_1.dRow = {_0{}, _1{}, int64_t(0)};
  return args;
}

template <class G, bool kRes>
int run_fp8_act(const void* A, const void* B, const float* a_s, const float* w_s, const void* bias, const void* C, void* D,
                int M, int N, int K, int ldd, void* workspace, unsigned long long workspace_bytes, int pdl, cudaStream_t stream) {
  using Gemm = typename G::Gemm;
  auto args = make_fp8_act_args<G, kRes>(A, B, a_s, w_s, bias, C, D, M, N, K, ldd);
  Gemm gemm;
  if (Gemm::get_workspace_size(args) > workspace_bytes) return 3;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 5;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return 6;
  if (gemm.run(stream, nullptr, pdl != 0) != cutlass::Status::kSuccess) return 7;
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

template <class G>
int run_fp8(const void* A, const void* B, const float* a_s, const float* w_s, void* D, int M, int N, int K, int ldd,
            void* workspace, unsigned long long workspace_bytes, int pdl, cudaStream_t stream) {
  using Gemm = typename G::Gemm;
  auto args = make_fp8_args<G>(A, B, a_s, w_s, D, M, N, K, ldd);
  Gemm gemm;
  if (Gemm::get_workspace_size(args) > workspace_bytes) return 3;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 5;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return 6;
  if (gemm.run(stream, nullptr, pdl != 0) != cutlass::Status::kSuccess) return 7;
  return cudaPeekAtLastError() == cudaSuccess ? 0 : 4;
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

int check(int M, int N, int K, int act, int tile, bool res) {
  if (M <= 0 || N <= 0 || K <= 0 || (N % kAlign) || (K % kAlign) || act < 0 || act > 2 || tile < 0 || tile > 3) return 2;
  if (res && act != 0) return 2;
  if (tile == 1 && (act != 0 || res)) return 2;
  return 0;
}

}  // namespace

extern "C" {

int flashrt_a15_bf16_gemm_sm100_workspace(int M, int N, int K, int act, int tile, int residual,
                                          unsigned long long* workspace_bytes) {
  if (int rc = check(M, N, K, act, tile, residual != 0)) return rc;
  if (tile == 3) {
    if (residual) *workspace_bytes = ws_size<G256WRes, true>(M, N, K);
    else if (act == 0) *workspace_bytes = ws_size<G256WId, false>(M, N, K);
    else if (act == 1) *workspace_bytes = ws_size<G256WTanh, false>(M, N, K);
    else *workspace_bytes = ws_size<G256WErf, false>(M, N, K);
    return 0;
  }
  if (tile == 2) {
    if (residual) *workspace_bytes = ws_size<G256Res, true>(M, N, K);
    else if (act == 0) *workspace_bytes = ws_size<G256Id, false>(M, N, K);
    else if (act == 1) *workspace_bytes = ws_size<G256Tanh, false>(M, N, K);
    else *workspace_bytes = ws_size<G256Erf, false>(M, N, K);
    return 0;
  }
  if (residual) *workspace_bytes = ws_size<G128Res, true>(M, N, K);
  else if (tile == 1) *workspace_bytes = ws_size<G64Id, false>(M, N, K);
  else if (act == 0) *workspace_bytes = ws_size<G128Id, false>(M, N, K);
  else if (act == 1) *workspace_bytes = ws_size<G128Tanh, false>(M, N, K);
  else *workspace_bytes = ws_size<G128Erf, false>(M, N, K);
  return 0;
}

// FP8 W8A8 GEMM (action expert tier): A [M, K] e4m3 row-major, B [N, K] e4m3 row-major, a_s [M] fp32, w_s [N] fp32,
// D [M, N] bf16. K % 16 == 0, N % 8 == 0, 16-byte aligned operands.
// tile 0 = 128 x 128 x 128, 1 = 64 x 128 x 128, 2 = 256 x 128 x 128 on a 2-SM UMMA; ldd = leading dimension of D
// (>= N: a column block of a wider output).
int flashrt_a15_fp8_gemm_sm100_workspace(int M, int N, int K, int tile, unsigned long long* workspace_bytes) {
  if (M <= 0 || N <= 0 || K <= 0 || (N % kAlign) || (K % 16) || tile < 0 || tile > 2) return 2;
  *workspace_bytes = tile == 2 ? F8G256P::Gemm::get_workspace_size(make_fp8_args<F8G256P>(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, N))
                   : tile == 1 ? F8G64::Gemm::get_workspace_size(make_fp8_args<F8G64>(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, N))
                   : F8G128::Gemm::get_workspace_size(make_fp8_args<F8G128>(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, N));
  return 0;
}
// FP8 GEMM with bias, activation (0 identity, 1 GELU tanh, 2 GELU erf) and optional residual C (identity only) on the
// 2-SM 256 x 128 x 128 tile: D = act(acc * a_s[m] * w_s[n] + bias[n]) (+ C); C and D rows ldd elements apart.
int flashrt_a15_fp8_gemm_act_sm100_workspace(int M, int N, int K, int act, int residual, unsigned long long* workspace_bytes) {
  if (M <= 0 || N <= 0 || K <= 0 || (N % kAlign) || (K % 16) || act < 0 || act > 2 || (residual && act != 0)) return 2;
  if (residual) *workspace_bytes = F8G256Res::Gemm::get_workspace_size(make_fp8_act_args<F8G256Res, true>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, N));
  else if (act == 0) *workspace_bytes = F8G256Bias::Gemm::get_workspace_size(make_fp8_act_args<F8G256Bias, false>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, N));
  else if (act == 1) *workspace_bytes = F8G256Tanh::Gemm::get_workspace_size(make_fp8_act_args<F8G256Tanh, false>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, N));
  else *workspace_bytes = F8G256Erf::Gemm::get_workspace_size(make_fp8_act_args<F8G256Erf, false>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, N));
  return 0;
}
int flashrt_a15_fp8_gemm_act_sm100(const void* A, const void* B, const float* a_s, const float* w_s, const void* bias,
                                   const void* C, void* D, int M, int N, int K, int act, int ldd, void* workspace,
                                   unsigned long long workspace_bytes, int pdl, cudaStream_t stream) {
  if (!A || !B || !a_s || !w_s || !D || (workspace_bytes && !workspace)) return 1;
  if (M <= 0 || N <= 0 || K <= 0 || (N % kAlign) || (K % 16) || act < 0 || act > 2 || (C && act != 0) || ldd < N || (ldd % kAlign)) return 2;
  if (!aligned16(A) || !aligned16(B) || !aligned16(D) || !aligned16(a_s) || !aligned16(w_s) || (bias && !aligned16(bias)) ||
      (C && !aligned16(C))) return 5;
  if (C) return run_fp8_act<F8G256Res, true>(A, B, a_s, w_s, bias, C, D, M, N, K, ldd, workspace, workspace_bytes, pdl, stream);
  if (act == 0) return run_fp8_act<F8G256Bias, false>(A, B, a_s, w_s, bias, nullptr, D, M, N, K, ldd, workspace, workspace_bytes, pdl, stream);
  if (act == 1) return run_fp8_act<F8G256Tanh, false>(A, B, a_s, w_s, bias, nullptr, D, M, N, K, ldd, workspace, workspace_bytes, pdl, stream);
  return run_fp8_act<F8G256Erf, false>(A, B, a_s, w_s, bias, nullptr, D, M, N, K, ldd, workspace, workspace_bytes, pdl, stream);
}
int flashrt_a15_fp8_gemm_sm100(const void* A, const void* B, const float* a_s, const float* w_s, void* D, int M, int N,
                               int K, int tile, int ldd, void* workspace, unsigned long long workspace_bytes, int pdl,
                               cudaStream_t stream) {
  if (!A || !B || !a_s || !w_s || !D || (workspace_bytes && !workspace)) return 1;
  if (M <= 0 || N <= 0 || K <= 0 || (N % kAlign) || (K % 16) || tile < 0 || tile > 2 || ldd < N || (ldd % kAlign)) return 2;
  if (!aligned16(A) || !aligned16(B) || !aligned16(D) || !aligned16(a_s) || !aligned16(w_s)) return 5;
  if (tile == 2) return run_fp8<F8G256P>(A, B, a_s, w_s, D, M, N, K, ldd, workspace, workspace_bytes, pdl, stream);
  if (tile == 1) return run_fp8<F8G64>(A, B, a_s, w_s, D, M, N, K, ldd, workspace, workspace_bytes, pdl, stream);
  return run_fp8<F8G128>(A, B, a_s, w_s, D, M, N, K, ldd, workspace, workspace_bytes, pdl, stream);
}

// Per-channel INT8 GEMM (prefill tier): A [M, K] int8 row-major, B [N, K] int8 row-major, a_s [M] fp32, w_s [N] fp32,
// D [M, N] fp16; tile 0 = 128 x 128 x 128, 1 = 64 x 128 x 128, 2 = 256 x 128 x 128 on a 2-SM UMMA (2 x 1 cluster).
// K % 16 == 0, N % 8 == 0, 16-byte aligned operands.
int flashrt_a15_int8_gemm_sm100_workspace(int M, int N, int K, int tile, unsigned long long* workspace_bytes) {
  if (M <= 0 || N <= 0 || K <= 0 || (N % kAlign) || (K % 16) || tile < 0 || tile > 2) return 2;
  *workspace_bytes = tile == 2 ? I8G256P::Gemm::get_workspace_size(make_int8_args<I8G256P>(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K))
                   : tile == 1 ? I8G64::Gemm::get_workspace_size(make_int8_args<I8G64>(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K))
                               : I8G128::Gemm::get_workspace_size(make_int8_args<I8G128>(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K));
  return 0;
}

int flashrt_a15_int8_gemm_sm100(const void* A, const void* B, const float* a_s, const float* w_s, void* D, int M, int N,
                                int K, int tile, void* workspace, unsigned long long workspace_bytes, int pdl,
                                cudaStream_t stream) {
  if (!A || !B || !a_s || !w_s || !D || (workspace_bytes && !workspace)) return 1;
  if (M <= 0 || N <= 0 || K <= 0 || (N % kAlign) || (K % 16) || tile < 0 || tile > 2) return 2;
  if (!aligned16(A) || !aligned16(B) || !aligned16(D) || !aligned16(a_s) || !aligned16(w_s)) return 5;
  if (tile == 2) return run_int8<I8G256P>(A, B, a_s, w_s, D, M, N, K, workspace, workspace_bytes, pdl, stream);
  return tile ? run_int8<I8G64>(A, B, a_s, w_s, D, M, N, K, workspace, workspace_bytes, pdl, stream)
              : run_int8<I8G128>(A, B, a_s, w_s, D, M, N, K, workspace, workspace_bytes, pdl, stream);
}

int flashrt_a15_bf16_gemm_sm100(const void* A, const void* B, const void* bias, const void* C, void* D, int M, int N,
                                int K, int act, int tile, void* workspace, unsigned long long workspace_bytes, int pdl,
                                cudaStream_t stream) {
  if (!A || !B || !D || (workspace_bytes && !workspace)) return 1;
  if (int rc = check(M, N, K, act, tile, C != nullptr)) return rc;
  if (!aligned16(A) || !aligned16(B) || !aligned16(D) || (bias && !aligned16(bias)) || (C && !aligned16(C))) return 5;
  if (tile == 3) {
    if (C) return run<G256WRes, true>(A, B, bias, C, D, M, N, K, workspace, workspace_bytes, pdl, stream);
    if (act == 0) return run<G256WId, false>(A, B, bias, nullptr, D, M, N, K, workspace, workspace_bytes, pdl, stream);
    if (act == 1) return run<G256WTanh, false>(A, B, bias, nullptr, D, M, N, K, workspace, workspace_bytes, pdl, stream);
    return run<G256WErf, false>(A, B, bias, nullptr, D, M, N, K, workspace, workspace_bytes, pdl, stream);
  }
  if (tile == 2) {
    if (C) return run<G256Res, true>(A, B, bias, C, D, M, N, K, workspace, workspace_bytes, pdl, stream);
    if (act == 0) return run<G256Id, false>(A, B, bias, nullptr, D, M, N, K, workspace, workspace_bytes, pdl, stream);
    if (act == 1) return run<G256Tanh, false>(A, B, bias, nullptr, D, M, N, K, workspace, workspace_bytes, pdl, stream);
    return run<G256Erf, false>(A, B, bias, nullptr, D, M, N, K, workspace, workspace_bytes, pdl, stream);
  }
  if (C) return run<G128Res, true>(A, B, bias, C, D, M, N, K, workspace, workspace_bytes, pdl, stream);
  if (tile == 1) return run<G64Id, false>(A, B, bias, nullptr, D, M, N, K, workspace, workspace_bytes, pdl, stream);
  if (act == 0) return run<G128Id, false>(A, B, bias, nullptr, D, M, N, K, workspace, workspace_bytes, pdl, stream);
  if (act == 1) return run<G128Tanh, false>(A, B, bias, nullptr, D, M, N, K, workspace, workspace_bytes, pdl, stream);
  return run<G128Erf, false>(A, B, bias, nullptr, D, M, N, K, workspace, workspace_bytes, pdl, stream);
}

}  // extern "C"
