"""ParoQuant W4A8 linears (the ``z-lab/Alpamayo-1.5-10B-PARO`` checkpoint) for the FlashRT Alpamayo 1.5 engines.

A ParoQuant linear is a learned pairwise rotation of its input (8 layers of Givens rotations within 128-channel
groups, per-channel scales) followed by a Marlin W4A8 GEMM: UINT4 weights with zero points, group 128, FP16 group
scales that Marlin's INT8-activation path turns into integers, and a per-token INT8 activation scale times a global
input scale. The engine unpacks the Marlin buffers once into the integers Marlin multiplies and runs the few-row GEMMs
of the FlashDrive backend on them.

The index maps of the Marlin tile layout follow vLLM's ``marlin_utils`` / ``marlin_utils_test`` (Apache-2.0):
``get_weight_perm(4, is_a_8bit=True)``, ``get_scale_perms()`` and ``marlin_permute_weights`` with a 16-wide tile.
"""
from __future__ import annotations

import os

import torch

DEV = "cuda"
GROUP = 128
TILE = 16


def _weight_perm_a8() -> torch.Tensor:
    """Marlin's 4-bit weight permutation for INT8 activations (``get_weight_perm(4, is_a_8bit=True)``)."""
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in (0, 1):
            for row in (4 * (i % 4), 4 * (i % 4) + 1, 4 * (i % 4) + 2, 4 * (i % 4) + 3,
                        4 * (i % 4 + 4), 4 * (i % 4 + 4) + 1, 4 * (i % 4 + 4) + 2, 4 * (i % 4 + 4) + 3):
                perm1.append(16 * row + col + 8 * block)
        for j in range(2):
            perm.extend(p + 512 * j for p in perm1)
    perm = torch.tensor(perm, dtype=torch.int64)
    interleave = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])
    return perm.view(-1, 8)[:, interleave].reshape(-1)


def _scale_perms() -> tuple[list[int], list[int]]:
    scale_perm = [i + 8 * j for i in range(8) for j in range(8)]
    scale_perm_single = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    return scale_perm, scale_perm_single


def _permute_weights_a8(q_w: torch.Tensor, K: int, N: int, perm: torch.Tensor) -> torch.Tensor:
    """``marlin_permute_weights`` for the INT8-activation layout (32 x 32 tiles)."""
    q_w = q_w.reshape(K // (TILE * 2), TILE * 2, N // TILE, TILE).permute(0, 2, 1, 3).reshape(K // TILE, N * TILE)
    return q_w.reshape(-1, perm.numel())[:, perm].reshape(q_w.shape)


def marlin_index_maps(K: int, N: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Device index maps (weight nibbles, group scales, zero points) that invert the Marlin layout of a [K, N] linear."""
    G = K // GROUP
    P = _permute_weights_a8(torch.arange(K * N, dtype=torch.int32).reshape(K, N), K, N, _weight_perm_a8())
    scale_perm, scale_perm_single = _scale_perms()
    grid = torch.arange(G * N, dtype=torch.int32)
    Ps = grid.reshape(-1, len(scale_perm_single))[:, torch.tensor(scale_perm_single)].reshape(-1)
    Pz = grid.reshape(-1, len(scale_perm))[:, torch.tensor(scale_perm)].reshape(-1)
    return P.reshape(-1).to(DEV, torch.long), Ps.to(DEV, torch.long), Pz.to(DEV, torch.long)


def marlin_w4a8_unpack(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, K: int, N: int,
                       maps: dict | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a Marlin W4A8 linear into the integers Marlin multiplies: ``w [K, N]`` uint8 (0..15), ``z [K/128, N]``
    uint8 and ``s [K/128, N]`` int32 (``marlin_act_int8_process_scales`` integers in [0, 4096])."""
    G = K // GROUP
    key = (K, N)
    m = maps.get(key) if maps is not None else None
    if m is None:
        m = marlin_index_maps(K, N)
        if maps is not None:
            maps[key] = m

    def nibbles(t):                      # [R, C] int32 -> [R * 8C], nibble i of word j at 8j + i
        return torch.stack([(t >> (4 * i)) & 0xF for i in range(8)], dim=-1).reshape(-1)

    w = torch.empty(K * N, dtype=torch.uint8, device=DEV)
    w[m[0]] = nibbles(qweight.to(DEV)).to(torch.uint8)
    z = torch.empty(G * N, dtype=torch.uint8, device=DEV)
    z[m[2]] = nibbles(qzeros.to(DEV)).to(torch.uint8)
    s = torch.empty(G * N, dtype=torch.int32, device=DEV)
    s[m[1]] = scales.to(DEV).view(torch.int16).reshape(-1).to(torch.int32)
    if bool((s < 0).any()) or bool((s > 4096).any()):
        raise ValueError("unexpected Marlin INT8-activation group scales (expected integers in [0, 4096])")
    return w.view(K, N), z.view(G, N), s.view(G, N)


def pack_w4a8_small(w: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """Pack ``w [K, N]`` uint8 (0..15) into the operand layout of the few-row W4A8 GEMMs:
    ``[N/16][K/64][32 lanes][16 bytes]``, each lane's bytes being its two mma.m16n8k32 A fragments (rows g, g+8
    of the tile; k = 4t..4t+3 and 16+4t..16+4t+3 of each 32-wide step) for the chunk's two k steps, first step in
    the low nibbles."""
    if K % 64 or N % 16:
        raise ValueError("pack_w4a8_small needs K % 64 == 0 and N % 16 == 0")
    # n = 16 T + 8 rh + g ; k = 64 C + 32 ks + 16 half + 4 t + j
    v = w.t().contiguous().view(N // 16, 2, 8, K // 64, 2, 2, 4, 4)          # [T, rh, g, C, ks, half, t, j]
    v = v.permute(0, 3, 2, 6, 4, 5, 1, 7).contiguous()                      # [T, C, g, t, ks, half, rh, j]
    v = v.view(N // 16, K // 64, 32, 2, 16)                                 # lane = 4 g + t; byte = 4 (2 half + rh) + j
    return (v[..., 0, :] | (v[..., 1, :] << 4)).contiguous()


W4_META = 64           # bytes of zero points (16 x u8), integer scales (16 x i16) and padding per tile and group
W4_BLOCK = W4_META + 1024


def pack_w4a8_grouped(ws: list[torch.Tensor], zs: list[torch.Tensor], sgs: list[torch.Tensor], K: int) -> torch.Tensor:
    """One operand buffer for the grouped few-row GEMM of linears sharing an input, ``[tiles, K/128, 1088]`` uint8:
    per 16-channel tile and 128-wide group the tile's zero points (u8), its integer group scales (i16, 0..4096),
    16 bytes of padding and its 1 KB of nibbles in the ``pack_w4a8_small`` layout; the linears' tiles in order.
    ``ws[i]`` [K, N_i] uint8 codes, ``zs[i]`` [K/128, N_i] uint8, ``sgs[i]`` [K/128, N_i] int32."""
    G = K // GROUP
    nib = torch.cat([pack_w4a8_small(w, K, w.shape[1]) for w in ws], dim=0)         # [NT, K/64, 32, 16]
    NT = nib.shape[0]
    z = torch.cat(zs, dim=1)
    s = torch.cat(sgs, dim=1)
    if z.shape != (G, 16 * NT) or s.shape != (G, 16 * NT):
        raise ValueError("pack_w4a8_grouped: zero points / scales do not match the codes")
    if bool((s < 0).any()) or bool((s > 4096).any()):
        raise ValueError("pack_w4a8_grouped: integer group scales must lie in [0, 4096]")
    zt = z.t().contiguous().view(NT, 16, G).permute(0, 2, 1).contiguous()                       # [NT, G, 16]
    st = s.t().contiguous().view(NT, 16, G).permute(0, 2, 1).contiguous().to(torch.int16).view(torch.uint8).view(NT, G, 32)
    pad = torch.zeros(NT, G, W4_META - 48, device=nib.device, dtype=torch.uint8)
    return torch.cat([zt, st, pad, nib.view(NT, G, 1024)], dim=2).contiguous()


PARO_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_paro_language_model(path: str, n_layers: int) -> list[dict]:
    """The ParoQuant linears of every language-model layer of a ``Alpamayo-1.5-10B-PARO`` checkpoint directory
    (safetensors), as dicts ``{name: {"qweight", "scales", "qzeros", "igs", "theta", "pairs", "channel_scales", "K", "N"}}``
    on the device. Only the language model's rotated W4A8 linears are read; everything else in the checkpoint
    (vision tower, norms, embeddings, LM head, expert) equals the base model up to an FP16 round trip and is not used."""
    from safetensors import safe_open
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors files in {path!r}")
    prefix = "vlm.model.language_model.layers."
    layers: list[dict] = [{} for _ in range(n_layers)]
    for f in files:
        with safe_open(os.path.join(path, f), "pt", device="cpu") as st:
            for key in st.keys():
                if not key.startswith(prefix) or ".rotate_linear." not in key:
                    continue
                rest = key[len(prefix):]
                li, _, part, _, name = rest.split(".", 4)     # e.g. 0 . self_attn . q_proj . rotate_linear . qlinear.qweight
                lin = layers[int(li)].setdefault(part, {})
                lin[name.split(".")[-1]] = st.get_tensor(key)
    for li, layer in enumerate(layers):
        for name in PARO_LINEARS:
            lin = layer.get(name)
            if lin is None or any(k not in lin for k in ("qweight", "scales", "qzeros", "input_global_scale", "theta", "pairs",
                                                         "channel_scales")):
                raise ValueError(f"layer {li}: incomplete ParoQuant linear {name!r} in {path!r}")
            G, N = lin["scales"].shape
            K = G * GROUP
            if (lin["theta"].shape != (8, K // 2) or lin["pairs"].shape != (8, K) or lin["channel_scales"].numel() != K
                    or lin["theta"].dtype != torch.float16 or lin["pairs"].dtype != torch.int16
                    or lin["channel_scales"].dtype != torch.float16 or lin["scales"].dtype != torch.float16
                    or lin["qweight"].shape != (K // TILE, N * 2) or lin["qzeros"].shape != (G, N // 8)):
                raise ValueError(f"layer {li} {name}: unexpected ParoQuant / Marlin layout")
            lin["K"], lin["N"] = K, N
            lin["igs"] = lin.pop("input_global_scale").to(DEV, torch.float32).reshape(1)
            for k in ("theta", "pairs"):
                lin[k] = lin[k].to(DEV).contiguous()
            lin["channel_scales"] = lin["channel_scales"].to(DEV).reshape(-1).contiguous()
    return layers


class W4A8DecodeGroup:
    """ParoQuant linears sharing one input (q|k|v, o, gate|up, down) for a few rows: the FlashDrive backend's
    activation preparation (FP16 cast, pairwise rotation, per-token INT8 quantization, one set per linear) and one
    grouped few-row W4A8 GEMM launch over the packed nibbles, zero points and integer group scales of every linear,
    written as BF16 column blocks of one row-major output. Up to ``FUSED_ROWS`` rows the quantization runs in the
    GEMM's prologue (two launches per group: rotation, GEMM); beyond, the two-pass preparation precedes the GEMM."""

    FUSED_ROWS = 4

    WIDE_TILES = 1024       # from this many 16-row weight tiles on a group counts as wide
    NARROW = (1, 2, 4)      # (ksplit, prefetch, warps) measured fastest on Thor at one row for the narrow groups
    WIDE = (2, 4, 4)        # ... and for the wide ones (gate|up)

    def __init__(self, fvk, lins: list[dict], max_rows: int, maps: dict, narrow: tuple[int, int, int] | None = None,
                 wide: tuple[int, int, int] | None = None) -> None:
        self._fvk = fvk
        self.K = lins[0]["K"]
        if any(lin["K"] != self.K for lin in lins):
            raise ValueError("a W4A8 group needs one input width")
        self.N = [lin["N"] for lin in lins]
        # the rotation tables of the group in one 128-byte aligned arena: prefetched into the L2 by the preceding
        # group's GEMM (a cold table read right before a streaming kernel slows that kernel by about 30% on Thor)
        tables = [lin[key] for lin in lins for key in ("theta", "pairs", "channel_scales")]
        sizes = [(t.numel() * t.element_size() + 127) // 128 * 128 for t in tables]
        self.tables = torch.zeros(sum(sizes) + 128, device=DEV, dtype=torch.uint8)
        off = (-self.tables.data_ptr()) % 128
        self.tables_region = (self.tables.data_ptr() + off, sum(sizes))
        views = []
        for t, size in zip(tables, sizes):
            v = self.tables[off:off + t.numel() * t.element_size()].view(t.dtype).view(t.shape)
            v.copy_(t)
            views.append(v)
            off += size
        self.theta = views[0::3]
        self.pairs = views[1::3]
        self.scales = views[2::3]
        self.igs = [lin["igs"] for lin in lins]
        ws, zs, sgs = [], [], []
        for lin in lins:
            w, z, s = marlin_w4a8_unpack(lin["qweight"], lin["scales"], lin["qzeros"], self.K, lin["N"], maps)
            ws.append(w)
            zs.append(z)
            sgs.append(s)
        self.w_pack = pack_w4a8_grouped(ws, zs, sgs, self.K)
        del ws, zs, sgs
        wide_group = sum(self.N) // 16 >= self.WIDE_TILES
        self.ksplit, self.prefetch, self.warps = (wide or self.WIDE) if wide_group else (narrow or self.NARROW)
        n = len(lins)
        rows = (max_rows + 7) // 8 * 8          # the GEMM reads whole 8-row tiles of the INT8 activations
        self.xr = [torch.empty(max_rows, self.K, device=DEV, dtype=torch.float16) for _ in range(n)]
        self.part = [torch.empty(max_rows, self.K // GROUP, device=DEV, dtype=torch.float32) for _ in range(n)]
        self.xq = [torch.zeros(rows, self.K, device=DEV, dtype=torch.int8) for _ in range(n)]
        self.a_scale = [torch.zeros(rows, device=DEV, dtype=torch.float32) for _ in range(n)]
        ptrs = lambda ts: [t.data_ptr() for t in ts]
        self._prep_ptrs = (ptrs(self.theta), ptrs(self.pairs), ptrs(self.scales), ptrs(self.igs), ptrs(self.xr),
                           ptrs(self.part), ptrs(self.xq), ptrs(self.a_scale))
        self._rot_ptrs = (ptrs(self.theta), ptrs(self.pairs), ptrs(self.scales), ptrs(self.xr), ptrs(self.part))
        self._gemm_ptrs = (ptrs(self.xq), ptrs(self.a_scale), self.w_pack.data_ptr())
        self._gemm_q_ptrs = (ptrs(self.xr), ptrs(self.part), ptrs(self.igs), self.w_pack.data_ptr())
        # a CTA of the fused form quantizes one problem: every problem's first tile must start a CTA
        self.fused_ok = all(sum(self.N[:i]) // 16 % (self.warps // self.ksplit) == 0 for i in range(1, n))

    def __call__(self, x: torch.Tensor, out: torch.Tensor, stream: int, prefetch: tuple[int, int] | None = None) -> None:
        """``out[:, blocks] = x @ W_i.T`` for the group's linears, column blocks in order (``out`` row-major [M, sum N]);
        ``prefetch`` = (pointer, bytes) of a region the GEMM pulls into the L2 while it streams (the next group's
        ``tables_region``)."""
        M = x.shape[0]
        pf_ptr, pf_bytes = prefetch if prefetch is not None else (0, 0)
        if out.shape[1] != sum(self.N) or out.shape[0] < M or out.stride(1) != 1:
            raise ValueError(f"W4A8 group writes {sum(self.N)} columns into a row-major output of {tuple(out.shape)}")
        fvk = self._fvk
        cols, col = [], 0
        for N in self.N:
            cols.append(out[:, col:col + N].data_ptr())
            col += N
        if M <= self.FUSED_ROWS and M <= self.warps and self.fused_ok:
            fvk.flashdrive_w4a8_rotate(x.data_ptr(), x.stride(0), M, self.K, *self._rot_ptrs, stream)
            xr, part, igs, w_pack = self._gemm_q_ptrs
            fvk.alpamayo15_w4a8_gemm_grouped_q(xr, part, igs, w_pack, cols, M, self.N, self.K, out.stride(0), self.prefetch,
                                               self.warps, stream, self.ksplit, [], [], pf_ptr, pf_bytes)
            return
        fvk.flashdrive_w4a8_act_prep(x.data_ptr(), x.stride(0), M, self.K, *self._prep_ptrs, stream)
        xq, a_scale, w_pack = self._gemm_ptrs
        fvk.alpamayo15_w4a8_gemm_grouped(xq, a_scale, w_pack, cols, M, self.N, self.K, out.stride(0), self.prefetch, self.warps,
                                         stream, self.ksplit, [], [], pf_ptr, pf_bytes)


@torch.no_grad()
def quantize_w4_rows(weight: torch.Tensor, chunk_rows: int = 8192) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Own INT4 quantization of a ``[N, K]`` linear for the grouped few-row GEMM: asymmetric round-to-nearest per
    (output channel, 128-wide input group) with the group scales normalized per output channel to the integers the
    GEMM multiplies. N is padded to a multiple of 16 with zero channels. Returns the packed operand
    (``pack_w4a8_grouped`` layout, ``[N_pad/16, K/128, 1088]``), zero points ``[K/128, N_pad]`` uint8, integer group
    scales ``[K/128, N_pad]`` int32 in [0, 4096] and the per-channel factor ``[N_pad]`` fp32 (``col_scale`` of the
    GEMM); dequantized weight = ``(q - z) * sg * col_scale``."""
    N, K = weight.shape
    if K % GROUP:
        raise ValueError("quantize_w4_rows needs K % 128 == 0")
    if chunk_rows % 16:
        raise ValueError("quantize_w4_rows needs chunk_rows % 16 == 0")
    G = K // GROUP
    N_pad = (N + 15) // 16 * 16
    packs, zs, sgs, css = [], [], [], []
    for n0 in range(0, N_pad, chunk_rows):
        end = min(n0 + chunk_rows, N_pad)
        w = weight[n0:min(end, N)].to(DEV, torch.float32)
        if end > N:                                                                    # zero channels of the padded tail
            w = torch.cat([w, torch.zeros(end - N, K, device=DEV)])
        w = w.view(-1, G, GROUP)                                                       # [n, G, 128]
        mn, mx = w.amin(dim=2), w.amax(dim=2)
        s = (mx - mn) / 15.0                                                           # ideal group scale
        s_max = s.amax(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        sg = torch.ceil(s / s_max * 4096.0).clamp_(0, 4096)                            # integer scales, never below the ideal
        cs = (s_max / 4096.0).squeeze(1)
        s_eff = (sg * cs.unsqueeze(1)).clamp_min(torch.finfo(torch.float32).tiny)     # the scale the GEMM realizes
        z = torch.round(-mn / s_eff).clamp_(0, 15)
        q = torch.round(w / s_eff.unsqueeze(2) + z.unsqueeze(2)).clamp_(0, 15).to(torch.uint8)
        zq, sq = z.t().to(torch.uint8).contiguous(), sg.t().to(torch.int32).contiguous()
        packs.append(pack_w4a8_grouped([q.view(-1, K).t().contiguous()], [zq], [sq], K))
        zs.append(zq)
        sgs.append(sq)
        css.append(cs.contiguous())
        del w, q
    return (torch.cat(packs, dim=0), torch.cat(zs, dim=1).contiguous(), torch.cat(sgs, dim=1).contiguous(),
            torch.cat(css).contiguous())


class W4LmHead:
    """The LM head as an own INT4 linear (``quantize_w4_rows``) with per-token INT8 activations
    (``alpamayo15_quantize_int8_rows_bf16``) on the grouped few-row GEMM, for a few rows."""

    def __init__(self, fvk, weight: torch.Tensor, max_rows: int, cfg: tuple[int, int, int] | None = None) -> None:
        self._fvk = fvk
        self.N, self.K = weight.shape
        self.w_pack, self.zp, self.sg, self.col_scale = quantize_w4_rows(weight)
        self.N_pad = self.w_pack.shape[0] * 16
        self.ksplit, self.prefetch, self.warps = cfg or W4A8DecodeGroup.WIDE
        rows = (max_rows + 7) // 8 * 8
        self.xq = torch.zeros(rows, self.K, device=DEV, dtype=torch.int8)
        self.a_scale = torch.zeros(rows, device=DEV, dtype=torch.float32)

    def __call__(self, x: torch.Tensor, out: torch.Tensor, stream: int) -> None:
        """``out = x @ dequant(W).T`` (bf16 [M, N]) from the INT8-quantized rows of ``x`` [M, K] bf16."""
        M = x.shape[0]
        if out.shape[1] != self.N or out.shape[0] < M or x.shape[1] != self.K or out.stride(1) != 1:
            raise ValueError("W4 LM head needs x [M, K] and a row-major out [M, N]")
        fvk = self._fvk
        fvk.alpamayo15_quantize_int8_rows_bf16(x.data_ptr(), x.stride(0), self.xq.data_ptr(), self.a_scale.data_ptr(), M, self.K,
                                               stream)
        fvk.alpamayo15_w4a8_gemm_grouped([self.xq.data_ptr()], [self.a_scale.data_ptr()], self.w_pack.data_ptr(), [out.data_ptr()],
                                         M, [self.N_pad], self.K, out.stride(0), self.prefetch, self.warps, stream, self.ksplit,
                                         [self.col_scale.data_ptr()], [self.N])
