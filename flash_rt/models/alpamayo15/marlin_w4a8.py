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
    two-pass activation preparation (FP16 cast, pairwise rotation, per-token INT8 quantization, one set per linear)
    and its few-row W4A8 GEMMs on the packed nibbles, zero points and integer group scales; outputs written as BF16
    column blocks of one row-major output."""

    TILE_MIN_N = 2048       # GEMMs with at least this many output channels stream one warp per 16-row weight tile

    def __init__(self, fvk, lins: list[dict], max_rows: int, maps: dict) -> None:
        self._fvk = fvk
        self.K = lins[0]["K"]
        if any(lin["K"] != self.K for lin in lins):
            raise ValueError("a W4A8 group needs one input width")
        self.N = [lin["N"] for lin in lins]
        self.theta = [lin["theta"] for lin in lins]
        self.pairs = [lin["pairs"] for lin in lins]
        self.scales = [lin["channel_scales"] for lin in lins]
        self.igs = [lin["igs"] for lin in lins]
        self.packed = []
        for lin in lins:
            w, z, s = marlin_w4a8_unpack(lin["qweight"], lin["scales"], lin["qzeros"], self.K, lin["N"], maps)
            self.packed.append((pack_w4a8_small(w, self.K, lin["N"]), z.contiguous(), s.contiguous()))
            del w
        n = len(lins)
        rows = (max_rows + 7) // 8 * 8          # the GEMMs read whole 8-row tiles of the INT8 activations
        self.xr = [torch.empty(max_rows, self.K, device=DEV, dtype=torch.float16) for _ in range(n)]
        self.part = [torch.empty(max_rows, self.K // GROUP, device=DEV, dtype=torch.float32) for _ in range(n)]
        self.xq = [torch.zeros(rows, self.K, device=DEV, dtype=torch.int8) for _ in range(n)]
        self.a_scale = [torch.zeros(rows, device=DEV, dtype=torch.float32) for _ in range(n)]

    def __call__(self, x: torch.Tensor, out: torch.Tensor, stream: int) -> None:
        """``out[:, blocks] = x @ W_i.T`` for the group's linears, column blocks in order (``out`` row-major [M, sum N])."""
        M = x.shape[0]
        ptrs = lambda ts: [t.data_ptr() for t in ts]
        fvk = self._fvk
        fvk.flashdrive_w4a8_act_prep(x.data_ptr(), x.stride(0), M, self.K, ptrs(self.theta), ptrs(self.pairs),
                                     ptrs(self.scales), ptrs(self.igs), ptrs(self.xr), ptrs(self.part), ptrs(self.xq),
                                     ptrs(self.a_scale), stream)
        col = 0
        ld = out.stride(0)
        for (wp, z, sg), xq, a_s, N in zip(self.packed, self.xq, self.a_scale, self.N):
            gemm = fvk.flashdrive_w4a8_gemm_tile if N >= self.TILE_MIN_N else fvk.flashdrive_w4a8_gemm_small
            gemm(xq.data_ptr(), a_s.data_ptr(), wp.data_ptr(), z.data_ptr(), sg.data_ptr(), out[:, col:col + N].data_ptr(),
                 M, N, self.K, stream, ld, 1)
            col += N
        if col != out.shape[1]:
            raise ValueError(f"W4A8 group writes {col} columns into an output of {out.shape[1]}")
