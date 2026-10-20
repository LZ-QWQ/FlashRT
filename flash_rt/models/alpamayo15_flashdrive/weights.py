"""Weight preparation for the FlashRT FlashDrive backend: FlashDrive's ParoQuant (Marlin W4A8) linears unpacked into
the tcgen05 prefill GEMM's operands, the per-channel INT8 requantization tier, the pairwise-rotation tables in the
fused activation kernel's bank order (and as dense matrices), and the fragment-ordered BF16 / FP8 / INT8 copies of
the small linears (LM head, DFlash draft) for the few-row GEMMs, and the fused W4A8 linear groups that hold them and
launch their GEMMs (``W4A8Group``). The preparation runs once when the backend's runtime is built."""
from __future__ import annotations

import torch

from flash_rt.models.alpamayo15.marlin_w4a8 import marlin_w4a8_unpack, pack_w4a8_small
from flash_rt.models.alpamayo15.weights import fp8_fragment_layout, quantize_fp8_rows, quantize_int8_rows

DEV = "cuda"
TILE_MIN_N = 2048           # few-row W4A8 GEMMs with at least this many output channels stream one warp per 16-row tile
TCGEN05_MIN_ROWS = 256      # rows of the construction-time check of the unpacked W4A8 layout against Marlin
INT8_PREFILL_TILE = 3       # per-channel INT8 prefill GEMM: 256 x 256 x 128 on a 2-SM UMMA (278-370 TOPS on Thor; 2: 256 x 128, 200-250; identical integers)
SMALL_MAX_ROWS = 16         # rows served by the few-row mma kernels (verification, trajectory start); more rows: tcgen05


def marlin_w4a8_to_int8(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, K: int, N: int,
                        maps: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(w8, s)``: ``w8 [N, K]`` int8 = ``w - z`` and ``s [K/128, N]`` int32 (the tcgen05 prefill GEMM operands)."""
    w, z, s = marlin_w4a8_unpack(qweight, scales, qzeros, K, N, maps)
    G = K // 128
    w8 = (w.to(torch.int16) - z.to(torch.int16).repeat_interleave(128, 0)).to(torch.int8).t().contiguous()
    return w8, s


def w4a8_to_int8_channel(w8: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel INT8 requantization of Marlin's group-scaled integers: ``w8 [N, K]`` int8 (w - z) and ``s [K/128, N]``
    int32 group scales give the exact integer weight ``w8 * s[group]``; its per-channel symmetric INT8 codes
    (``round(v / c)``, ``c = absmax / 127``) and the fp32 channel scale ``c`` (1 for an all-zero channel) are returned
    for the per-channel INT8 prefill GEMM (``D = half(acc * a_scale[m] * c[n])``). A precision tier: the codes carry the
    group-scale ratios of a channel with at most half a step of error."""
    N, K = w8.shape
    v = w8.to(torch.int32) * s.t().repeat_interleave(128, 1)           # [N, K] exact integers (|w8| <= 15, s <= 4096)
    amax = v.abs().amax(dim=1).to(torch.float32)
    c = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    codes = torch.round(v.to(torch.float32) / c[:, None]).clamp_(-127.0, 127.0).to(torch.int8).contiguous()
    return codes, c.contiguous()


def paro_pair_bank_order(theta: torch.Tensor, pairs: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder the 64 pairs of every (layer, channel group) of a ParoQuant rotation so that the eight pairs a
    quarter-warp of the tile kernel handles together touch eight distinct shared-memory bank groups with their first
    channel (bank group = 16-byte chunk index ``c + c // 8`` mod 8 of the 8-row tile layout). The pairs of a layer are
    disjoint, so their order does not change the rotation; returns (theta, pairs) in the new order."""
    G = K // 128
    pr = pairs.view(8, G, 64, 2).to(torch.int64)
    th = theta.view(8, G, 64)
    i = pr[..., 0]
    res = (i + (i >> 3)) & 7
    ar = torch.arange(64, device=pairs.device)
    order = torch.argsort(res * 64 + ar, dim=-1)                       # pairs sorted by bank residue (stable)
    res_s = torch.gather(res, -1, order)
    counts = torch.nn.functional.one_hot(res, 8).sum(-2)                # [8, G, 8]
    starts = counts.cumsum(-1) - counts
    rank = ar - torch.gather(starts, -1, res_s)                         # rank within the residue class
    main = rank < 8
    pos = torch.where(main, rank * 8 + res_s, torch.zeros_like(rank))   # bucket = rank, lane-in-bucket = residue
    taken = torch.zeros_like(rank).scatter_add(-1, pos, main.long()) > 0
    free_order = torch.argsort(taken.long() * 64 + ar, dim=-1)           # free positions first, ascending
    ovf = (~main).long().cumsum(-1) - 1
    pos = torch.where(main, pos, torch.gather(free_order, -1, ovf.clamp(min=0)))
    new_pr = torch.empty_like(pr).scatter_(-2, pos.unsqueeze(-1).expand(-1, -1, -1, 2),
                                           torch.gather(pr, -2, order.unsqueeze(-1).expand(-1, -1, -1, 2)))
    new_th = torch.empty_like(th).scatter_(-1, pos, torch.gather(th, -1, order))
    return new_th.reshape(8, K // 2).contiguous(), new_pr.reshape(8, K).to(pairs.dtype).contiguous()


def paro_rotation_matrices(theta: torch.Tensor, pairs: torch.Tensor, K: int) -> torch.Tensor:
    """The 128 x 128 matrix of every channel group of a ParoQuant rotation: ``R[g][c][k]`` such that
    ``y[g*128 + c] = sum_k R[g][c][k] x[g*128 + k]`` equals the 8 Givens layers applied in order (FP32 products of
    the layer cosines/sines, returned as FP16 for the tensor-core kernel)."""
    G = K // 128
    R = torch.eye(128, device=theta.device, dtype=torch.float32).repeat(G, 1, 1)         # [G, 128, 128]
    th = theta.float().view(8, G, 64)
    pr = pairs.view(8, G, 64, 2).long()
    gidx = torch.arange(G, device=theta.device).unsqueeze(1)
    for r in range(8):
        c, s_ = torch.cos(th[r]).unsqueeze(-1), torch.sin(th[r]).unsqueeze(-1)       # [G, 64, 1]
        i, j = pr[r, :, :, 0], pr[r, :, :, 1]                                        # [G, 64]
        Ri, Rj = R[gidx, i], R[gidx, j]                                              # rows i, j: [G, 64, 128]
        R[gidx, i] = c * Ri + s_ * Rj
        R[gidx, j] = c * Rj - s_ * Ri
    return R.to(torch.float16).contiguous()


def bf16_fragment_layout(w: torch.Tensor) -> torch.Tensor:
    """Reorder a [N, K] bf16 weight into the ``flashdrive_bf16_gemm_frag`` operand: ``[ceil(N/16)][K/16][32 lanes][16 B]``,
    lane ``4g + c`` holding the four ``mma.m16n8k16`` A registers of one 16-wide k step (rows g, g+8; k 2c and 8+2c).
    Rows are zero-padded to a multiple of 16."""
    N, K = w.shape
    if K % 16:
        raise ValueError(f"K={K} must be a multiple of 16")
    T = (N + 15) // 16
    if T * 16 != N:
        w = torch.cat([w, w.new_zeros(T * 16 - N, K)])
    # n = 16T + 8rh + g ; k = 16kc + 8half + 2c + e  ->  [T, kc, g, c, half, rh, e]  (register a = 2half + rh)
    frag = w.view(T, 2, 8, K // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).contiguous()
    return frag.view(-1)


class Bf16Fragment:
    """Fragment-ordered BF16 copy of a [N, K] linear weight for the streaming few-row GEMM."""

    def __init__(self, w: torch.Tensor) -> None:
        with torch.no_grad():
            self.frag = bf16_fragment_layout(w.detach())
        self.shape = tuple(w.shape)


class Fp8Fragment:
    """Per-row one-byte copy of a [N, K] linear weight (``fmt`` "fp8": E4M3 codes, "int8": symmetric INT8 codes; both
    with per-row FP32 scales) in the ``alpamayo15_w8a16_gemm_bf16`` fragment order."""

    def __init__(self, w: torch.Tensor, fmt: str = "fp8") -> None:
        with torch.no_grad():
            codes, scale = (quantize_int8_rows if fmt == "int8" else quantize_fp8_rows)(w.detach())
            self.frag, self.scale = fp8_fragment_layout(codes, scale)
        self.shape = tuple(w.shape)
        self.fmt = 1 if fmt == "int8" else 0


class W4A8Group:
    """FlashDrive ParoQuant linears that share one input (q|k|v, o, gate|up, down) executed as one fused
    activation pass (``flashdrive_w4a8_act_prep``: FP16 cast, pairwise rotation, per-token INT8 quantization,
    bit-exact with paroquant + vLLM) and the W4A8 GEMM of each linear on FlashRT kernels computing Marlin's integer
    group-scaled sums exactly: the INT8 tcgen05 GEMM on the unpacked ``w - z`` weights for row tiles (streaming-window
    and first-window prefill), the few-row ``mma.sync`` kernel on the packed nibbles for at most ``SMALL_MAX_ROWS``
    rows (verification, trajectory start). The unpacked layout is checked against vLLM's Marlin kernel once per shape
    when the backend is built (``check_against_marlin``). Returns the FP16 GEMM outputs; FlashDrive's cast back to
    BF16 is applied by the consumer kernels on load."""

    def __init__(self, fvk, wrappers, scratch: dict, maps: dict | None = None, act_rot: str = "paro", ktile: int = 256,
                 prefill_weights: str = "w4a8") -> None:
        from flashdrive.quantization import BF16RotateLinearWrapper
        from vllm.scalar_type import scalar_types
        mods = []
        for w in wrappers:
            if not isinstance(w, BF16RotateLinearWrapper):
                raise TypeError(f"expected FlashDrive's BF16RotateLinearWrapper, got {type(w).__name__}")
            mods.append(w.rotate_linear)
        self.K = mods[0].in_features
        for m in mods:
            q, r = m.qlinear, m.rotation
            if m.in_features != self.K or self.K % 128:
                raise ValueError("fused W4A8 group needs one input width, a multiple of 128")
            if (r.theta.shape != (8, self.K // 2) or r.pairs.shape != (8, self.K) or r.channel_scales.numel() != self.K
                    or r.theta.dtype != torch.float16 or r.channel_scales.dtype != torch.float16
                    or r.pairs.dtype != torch.int16):
                raise ValueError("unexpected ParoQuant rotation layout (8 layers, group 128, fp16 angles/scales)")
            if q.quant_type != scalar_types.uint4 or q.bias is not None or q.g_idx.numel() or q.scales.dtype != torch.float16:
                raise ValueError("unexpected Marlin W4A8 layout (uint4 + zero points, fp16 group scales, no act-order/bias)")
        self.mods = mods
        self.N = [m.out_features for m in mods]
        self._scratch = scratch     # shared by every group: consumed by this group's GEMMs before the next group runs
        self.act_rot = act_rot
        self.ktile = ktile
        self.rot = [paro_rotation_matrices(m.rotation.theta, m.rotation.pairs, self.K) for m in mods] if act_rot == "dense" else None
        tabs = [paro_pair_bank_order(m.rotation.theta, m.rotation.pairs, self.K) for m in mods]   # bank-ordered pair tables
        self.theta, self.pairs = [t for t, _ in tabs], [p for _, p in tabs]
        # tcgen05 INT8 ((w - z) int8 [N, K], or per-channel INT8 codes with prefill_weights="int8") for row tiles,
        # mma.sync on packed nibbles + zero points for a few rows
        self.int8, self.small = [], []
        self.int8pc = [] if prefill_weights == "int8" else None
        for m, N in zip(mods, self.N):
            w, z, sg = marlin_w4a8_unpack(m.qlinear.qweight, m.qlinear.scales, m.qlinear.qzeros, self.K, N, maps)
            w8 = (w.to(torch.int16) - z.to(torch.int16).repeat_interleave(128, 0)).to(torch.int8).t().contiguous()
            if self.int8pc is not None:
                self.int8pc.append(w4a8_to_int8_channel(w8, sg))
                del w8
            else:
                self.int8.append((w8, sg))
            self.small.append((pack_w4a8_small(w, self.K, N), z.contiguous(), sg))
            del w, z

    def _buffers(self, M: int) -> dict:
        n = len(self.mods)
        b = self._scratch.get((M, self.K, n))
        if b is None:
            Ma = (M + 31) // 32 * 32             # the GEMMs read whole token tiles (8 rows mma, 32 rows tcgen05): zeros
            b = {"xq": [torch.zeros(Ma, self.K, device=DEV, dtype=torch.int8) for _ in range(n)],
                 "as": [torch.zeros(Ma, 1, device=DEV, dtype=torch.float32) for _ in range(n)]}
            if self.rot is not None or M <= SMALL_MAX_ROWS:   # two-pass preparation: FP16 rows and per-group maxima
                b["xr"] = [torch.empty(M, self.K, device=DEV, dtype=torch.float16) for _ in range(n)]
                b["part"] = [torch.empty(M, self.K // 128, device=DEV, dtype=torch.float32) for _ in range(n)]
            self._scratch[(M, self.K, n)] = b
        return b

    def prep(self, fvk, mode: int, src: torch.Tensor | None, M: int, src2: torch.Tensor | None = None,
             residual: torch.Tensor | None = None, norm_w: torch.Tensor | None = None, eps: float = 0.0) -> None:
        """Fused producer + activation preparation of ``M`` rows (``flashdrive_w4a8_act_prep_tile``): mode 0 plain bf16
        rows, mode 1 residual (+= bf16(delta fp16) when ``src`` is given) RMSNorm, mode 2 SiLU-mul of fp16 gate | up."""
        b = self._buffers(M)
        ptrs = lambda ts: [t.data_ptr() for t in ts]
        fvk.flashdrive_w4a8_act_prep_tile(
            mode, src.data_ptr() if src is not None else 0, src2.data_ptr() if src2 is not None else 0,
            (src if src is not None else residual).stride(0), residual.data_ptr() if residual is not None else 0,
            residual.stride(0) if residual is not None else 0, norm_w.data_ptr() if norm_w is not None else 0, eps, M,
            self.K, ptrs(self.theta), ptrs(self.pairs),
            ptrs(m.rotation.channel_scales for m in self.mods), ptrs(m.qlinear.input_global_scale for m in self.mods),
            ptrs(b["xq"]), ptrs(b["as"]), 0, 0, torch.cuda.current_stream().cuda_stream)

    def gemm(self, fvk, M: int) -> list[torch.Tensor]:
        """The GEMMs of the group on the prepared INT8 activations of ``M`` rows: the few-row kernel up to
        ``SMALL_MAX_ROWS`` rows, the tcgen05 GEMM otherwise."""
        b = self._buffers(M)
        outs = []
        if M <= SMALL_MAX_ROWS:
            for (wp, z, sg), xq, a_s, N in zip(self.small, b["xq"], b["as"], self.N):
                out = torch.empty(M, N, dtype=torch.float16, device=DEV)
                gemm = fvk.flashdrive_w4a8_gemm_tile if N >= TILE_MIN_N else fvk.flashdrive_w4a8_gemm_small
                gemm(xq.data_ptr(), a_s.data_ptr(), wp.data_ptr(), z.data_ptr(), sg.data_ptr(), out.data_ptr(), M, N, self.K,
                     torch.cuda.current_stream().cuda_stream)
                outs.append(out)
            return outs
        if self.int8pc is not None:
            for (codes, c), xq, a_s, N in zip(self.int8pc, b["xq"], b["as"], self.N):
                outs.append(self._gemm_int8pc(fvk, xq, codes, c, a_s, M, N))
            return outs
        for (w8, sg), xq, a_s, N in zip(self.int8, b["xq"], b["as"], self.N):
            outs.append(self._gemm_int8(fvk, xq, w8, sg, a_s, M, N))
        return outs

    def __call__(self, fvk, x: torch.Tensor) -> list[torch.Tensor]:
        M = x.shape[0]
        b = self._buffers(M)
        ptrs = lambda ts: [t.data_ptr() for t in ts]
        stream = torch.cuda.current_stream().cuda_stream
        if self.rot is not None:          # dense rotation on tensor cores, then the shared quantization pass
            fvk.flashdrive_w4a8_rotate_dense(x.data_ptr(), x.stride(0), M, self.K, ptrs(self.rot),
                                             ptrs(m.rotation.channel_scales for m in self.mods), ptrs(b["xr"]),
                                             ptrs(b["part"]), stream)
            fvk.flashdrive_w4a8_quant(M, self.K, ptrs(b["xr"]), ptrs(b["part"]),
                                      ptrs(m.qlinear.input_global_scale for m in self.mods), ptrs(b["xq"]), ptrs(b["as"]),
                                      stream)
        elif M > SMALL_MAX_ROWS:          # row tiles: producer-fused preparation
            self.prep(fvk, 0, x, M)
        else:                             # a few rows: the two-pass kernels spread the groups over the SMs
            fvk.flashdrive_w4a8_act_prep(
                x.data_ptr(), x.stride(0), M, self.K, ptrs(self.theta), ptrs(self.pairs),
                ptrs(m.rotation.channel_scales for m in self.mods),
                ptrs(m.qlinear.input_global_scale for m in self.mods), ptrs(b["xr"]), ptrs(b["part"]), ptrs(b["xq"]),
                ptrs(b["as"]), stream)
        return self.gemm(fvk, M)

    def _gemm_int8(self, fvk, xq, w8, sg, a_s, M, N) -> torch.Tensor:
        key = ("int8", M, self.K, N)
        ws = self._scratch.get(key)
        if ws is None:
            nbytes, n_ones = fvk.flashdrive_w4a8_gemm_sm100_sizes(M, N, self.K)
            ws = (torch.empty(max(int(nbytes), 1), dtype=torch.uint8, device=DEV),     # padded: the last token tile
                  torch.ones(int(n_ones) + 1024, dtype=torch.int32, device=DEV))      # may read past the layout
            self._scratch[key] = ws
        out = torch.empty(M, N, dtype=torch.float16, device=DEV)
        fvk.flashdrive_w4a8_gemm_sm100(xq.data_ptr(), w8.data_ptr(), sg.data_ptr(), ws[1].data_ptr(), a_s.data_ptr(),
                                       out.data_ptr(), M, N, self.K, ws[0].data_ptr(), ws[0].numel(),
                                       torch.cuda.current_stream().cuda_stream, self.ktile)
        return out

    def _gemm_int8pc(self, fvk, xq, codes, c, a_s, M, N) -> torch.Tensor:
        key = ("int8pc", M, self.K, N)
        ws = self._scratch.get(key)
        if ws is None:
            nbytes = fvk.alpamayo15_int8_gemm_sm100_workspace(M, N, self.K, INT8_PREFILL_TILE)
            ws = torch.empty(max(int(nbytes), 16), dtype=torch.uint8, device=DEV)
            self._scratch[key] = ws
        out = torch.empty(M, N, dtype=torch.float16, device=DEV)
        fvk.alpamayo15_int8_gemm_sm100(xq.data_ptr(), codes.data_ptr(), a_s.data_ptr(), c.data_ptr(), out.data_ptr(), M, N,
                                       self.K, INT8_PREFILL_TILE, ws.data_ptr(), ws.numel(),
                                       torch.cuda.current_stream().cuda_stream)
        return out

    def check_against_marlin(self, fvk, rows: int = TCGEN05_MIN_ROWS) -> float:
        """Run Marlin and the FlashRT GEMMs (tcgen05 at ``rows`` rows, the few-row kernel at 8 and 16 rows) of every
        linear on random INT8 activations; returns the smallest fraction of identical FP16 outputs. Raises on a gross
        disagreement (unpacking or layout mismatch): an output off by more than one FP16 ulp plus Marlin's FP32
        slice-sum error bound, or fewer than 99% identical outputs."""
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import should_use_atomic_add_reduce
        g = torch.Generator(device=DEV).manual_seed(0)
        worst = 1.0
        for R in (rows, 8, 16):
            xq = torch.randint(-127, 128, (R, self.K), dtype=torch.int8, device=DEV, generator=g)
            u = torch.rand(R, 1, device=DEV, generator=g) * 2e-2 + 1e-3     # per-token absmax / 127, as act prep
            for m, (w8, sg), (wp, z, _), N in zip(self.mods, self.int8, self.small, self.N):
                q = m.qlinear
                a_s = (u * q.input_global_scale).contiguous()
                ref = ops.marlin_gemm(xq, None, q.qweight, q.bias, q.scales, a_s, None, q.qzeros, q.g_idx,
                                      q.g_idx_sort_indices, q.workspace, q.quant_type, size_m=R, size_n=N, size_k=self.K,
                                      use_atomic_add=should_use_atomic_add_reduce(R, N, self.K, xq.device, torch.float16),
                                      use_fp32_reduce=True, is_zp_float=False)
                if R == rows:
                    out = self._gemm_int8(fvk, xq, w8, sg, a_s, R, N)
                else:
                    out = torch.empty(R, N, dtype=torch.float16, device=DEV)
                    fvk.flashdrive_w4a8_gemm_small(xq.data_ptr(), a_s.data_ptr(), wp.data_ptr(), z.data_ptr(),
                                                   sg.data_ptr(), out.data_ptr(), R, N, self.K,
                                                   torch.cuda.current_stream().cuda_stream)
                r = ref.float()
                tol = (r.abs() * 2.0 ** -10).clamp_min(2.0 ** -24) + r.abs().amax(dim=1, keepdim=True) * 2.0 ** -20
                same = float((out == ref).float().mean())
                if bool(((out.float() - r).abs() > tol).any()) or same < 0.99:
                    raise RuntimeError(f"FlashRT W4A8 GEMM ({R} rows) disagrees with Marlin (weight layout?)")
                worst = min(worst, same)
        return worst
