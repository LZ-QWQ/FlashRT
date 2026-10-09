"""GROOT N1.7 RTX frontend with an NVFP4 DiT action head (SM120).

``GrootN17TorchFrontendRtxFP4`` keeps the RTX FP8 backbone
(``GrootN17TorchFrontendRtxFP8``) and runs the 32-layer DiT on the skinny
NVFP4 (W4A4) family: every DiT GEMM is block-scaled e2m1 with a per-tensor
weight global scale, and the consumers that sum the K-split partials emit the
next GEMM's NVFP4 operand together with the bias, GELU, residual and
(Ada)LayerNorm, so the hot path has no standalone quantize, elementwise or
copy launch (see ``flash_rt.models.groot_n17.pipeline_rtx_nvfp4``).

The DiT weights are quantized from the checkpoint's BF16 tensors (not from
the FP8-rounded copies the shared loader keeps). Activations use per-16
dynamic block scales with global scale 1, so no activation calibration runs.
At M = 41 (1 state + 40 action tokens) the DiT streams its weights once per
denoising step; NVFP4 halves those bytes against FP8.
"""

from __future__ import annotations

import glob
import os

import torch

from flash_rt.frontends.torch.groot_n17_rtx_fp8 import GrootN17TorchFrontendRtxFP8

_BLOCK = "action_head.model.transformer_blocks.{}.{}"


class GrootN17TorchFrontendRtxFP4(GrootN17TorchFrontendRtxFP8):
    """N1.7 RTX inference frontend: FP8 backbone + NVFP4 DiT."""

    _DIT_QUANT = "nvfp4_skinny"
    _DIT_WEIGHT_MSE_STEPS = int(os.environ.get("FLASHRT_GROOT_N17_DIT_FP4_MSE", "4"))

    def _ckpt_bf16_tensor(self, key: str) -> torch.Tensor:
        from safetensors import safe_open
        if not hasattr(self, "_ckpt_key_index"):
            index = {}
            for shard in sorted(glob.glob(os.path.join(self.checkpoint_path, "model-*.safetensors"))):
                with safe_open(shard, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        index[k] = shard
            self._ckpt_key_index = index
        with safe_open(self._ckpt_key_index[key], framework="pt", device=str(self.device)) as f:
            return f.get_tensor(key).to(torch.bfloat16)

    def _quantize_nvfp4_site(self, w_nk: torch.Tensor, bias: torch.Tensor) -> tuple:
        """BF16 [N, K] weight -> packed NVFP4 + scales + global scale; keeps
        the tensors alive and returns the pointer tuple the DiT consumes."""
        fvk = self._fvk
        N, K = w_nk.shape
        w_nk = w_nk.contiguous()
        gs = (w_nk.float().abs().amax() / (6.0 * 448.0)).clamp_min(1e-12).reshape(1).contiguous()
        q = torch.empty(N, K // 2, dtype=torch.uint8, device=self.device)
        sf = torch.empty(N, K // 16, dtype=torch.uint8, device=self.device)
        packed = torch.empty_like(q)
        sf_packed = torch.empty_like(sf)
        for rc, what in (
                (fvk.nvfp4_skinny_quantize_weight(w_nk.data_ptr(), gs.data_ptr(), q.data_ptr(), sf.data_ptr(),
                                                  N, K, 0, self._DIT_WEIGHT_MSE_STEPS), "quantize"),
                (fvk.nvfp4_skinny_pack_weight(q.data_ptr(), sf.data_ptr(), packed.data_ptr(),
                                              sf_packed.data_ptr(), N, K, 0), "pack")):
            if rc != 0:
                raise RuntimeError(f"N1.7 NVFP4 DiT weight {what} failed for [{N}, {K}]: cudaError {rc}")
        bias = bias.to(torch.bfloat16).contiguous()
        self._dit_nvfp4_keep += [packed, sf_packed, gs, bias]
        return (packed.data_ptr(), sf_packed.data_ptr(), gs.data_ptr(), bias.data_ptr())

    def _prepare_dit_nvfp4_skinny(self, step_weights, bp, dims, num_steps: int, Sa: int) -> None:
        from flash_rt.models.groot_n17 import pipeline_rtx_nvfp4 as P
        fvk = self._fvk
        if not hasattr(fvk, "nvfp4_skinny_gemm"):
            raise RuntimeError("the N1.7 NVFP4 DiT needs a GPU_ARCH=120 kernel build (skinny NVFP4 family)")
        D, FF = P.D, P.FF
        dev = self.device
        if not hasattr(self, "_dit_nvfp4_sites"):
            self._dit_nvfp4_keep = []
            qkv, q, o, up, down = [], [], [], [], []
            with torch.no_grad():
                for li in range(P.LAYERS):
                    t = lambda name: self._ckpt_bf16_tensor(_BLOCK.format(li, name))   # noqa: E731
                    if li % 2 == 1:
                        w = torch.cat([t("attn1.to_q.weight"), t("attn1.to_k.weight"), t("attn1.to_v.weight")], 0)
                        b = torch.cat([self._dit_q_b[li], self._dit_k_b[li], self._dit_v_b[li]])
                        qkv.append(self._quantize_nvfp4_site(w, b))
                    else:
                        q.append(self._quantize_nvfp4_site(t("attn1.to_q.weight"), self._dit_q_b[li]))
                    o.append(self._quantize_nvfp4_site(t("attn1.to_out.0.weight"), self._dit_o_b[li]))
                    up.append(self._quantize_nvfp4_site(t("ff.net.0.proj.weight"), self._dit_ff_proj_b[li]))
                    down.append(self._quantize_nvfp4_site(t("ff.net.2.weight"), self._dit_ff_down_b[li]))
            self._dit_nvfp4_sites = {"qkv": qkv, "q": q, "o": o, "ff_up": up, "ff_down": down}
            self._k_dit_xn_fp4 = torch.empty(Sa, D // 2, dtype=torch.uint8, device=dev)
            self._k_dit_xn_sf = torch.zeros(Sa, D // 16, dtype=torch.uint8, device=dev)
            self._k_dit_hid_fp4 = torch.empty(Sa, FF // 2, dtype=torch.uint8, device=dev)
            self._k_dit_hid_sf = torch.zeros(Sa, FF // 16, dtype=torch.uint8, device=dev)
            self._k_dit_partials = torch.empty(P.partial_floats(fvk, Sa), dtype=torch.float32, device=dev)
            self._k_dit_one = torch.ones(1, dtype=torch.float32, device=dev)
        self._dit_nvfp4_bufs = {
            "h": bp["h"],
            "xn": self._k_dit_xn_fp4.data_ptr(), "xn_sf": self._k_dit_xn_sf.data_ptr(),
            "hid": self._k_dit_hid_fp4.data_ptr(), "hid_sf": self._k_dit_hid_sf.data_ptr(),
            "partials": self._k_dit_partials.data_ptr(), "one": self._k_dit_one.data_ptr(),
            "hmod": self._k_hmod.data_ptr(),
        }
        for step in range(num_steps):
            step_weights[step].update(self._dit_nvfp4_sites)
            step_weights[step]["out_scale"] = self._k_oproj_scale[step].data_ptr()
            step_weights[step]["out_shift"] = self._k_oproj_shift[step].data_ptr()

    def _dit_nvfp4_skinny_step(self, step: int, weights: dict, dims: dict, stream: int) -> None:
        from flash_rt.models.groot_n17 import pipeline_rtx_nvfp4 as P
        P.dit_forward_nvfp4(self._fvk, self._dit_nvfp4_bufs, weights, dims, attn=self._dit_attn,
                            stream=int(stream), pdl=os.environ.get("FLASHRT_GROOT_N17_DIT_PDL", "1") != "0")
