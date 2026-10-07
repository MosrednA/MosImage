"""Experimental official Klein NVFP4 loader for the paired offline benchmark.

Uses Apache-2.0 Comfy Kitchen CUDA kernels directly. No BF16 GEMM fallback.
The app's provider registry is deliberately independent of this experiment.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Literal, Protocol

import torch
from accelerate import init_empty_weights
from diffusers import Flux2Transformer2DModel
from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict
from safetensors import safe_open
from torch import Tensor, nn

BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"
BASE_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
QUANT_REPO = "black-forest-labs/FLUX.2-klein-4b-nvfp4"
QUANT_REVISION = "1db2b2f776c24b76f1122e5f69ab1949fc620068"
QUANT_FILENAME = "flux-2-klein-4b-nvfp4.safetensors"


class KernelApi(Protocol):
    def quantize_nvfp4(self, x: Tensor, per_tensor_scale: Tensor, *, pad_16x: bool) -> tuple[Tensor, Tensor]: ...
    def scaled_mm_nvfp4(self, a: Tensor, b: Tensor, *, tensor_scale_a: Tensor,
        tensor_scale_b: Tensor, block_scale_a: Tensor, block_scale_b: Tensor,
        out_dtype: torch.dtype) -> Tensor: ...


class QuantLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["nvfp4"]


class QuantMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format_version: Literal["1.0"]
    layers: dict[str, QuantLayer]


def kernel_api() -> KernelApi:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".nvfp4-deps"))
    import comfy_kitchen as ck
    from importlib.metadata import version

    if version("comfy-kitchen") != "0.2.37":
        raise RuntimeError("This benchmark is pinned to comfy-kitchen 0.2.37.")
    ck.registry.disable("eager")
    ck.registry.disable("triton")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("The native NVFP4 benchmark requires a Blackwell CUDA GPU.")
    return ck


class Nvfp4Linear(nn.Module):
    def __init__(self, packed: Tensor, blocks: Tensor, scale: Tensor, input_scale: Tensor,
                 kernel: KernelApi) -> None:
        super().__init__()
        if packed.dtype != torch.uint8 or packed.ndim != 2:
            raise ValueError("NVFP4 weights must be packed two-dimensional uint8 tensors.")
        self.out_features, packed_columns = packed.shape
        self.in_features = packed_columns * 2
        if self.out_features % 128 or self.in_features % 64:
            raise ValueError("This checkpoint adapter requires 128-row/64-column alignment.")
        if blocks.dtype != torch.float8_e4m3fn or blocks.shape != (self.out_features, self.in_features // 16):
            raise ValueError("Invalid checkpoint block-scale dtype or shape.")
        for scalar in (scale, input_scale):
            if scalar.dtype != torch.float32 or scalar.numel() != 1 or not torch.isfinite(scalar).all() or scalar.item() <= 0:
                raise ValueError("NVFP4 per-tensor scales must be finite positive float32 scalars.")
        # Buffers move with component offload; they are already loaded here and
        # should not appear among missing parameters of the BF16 remainder.
        self.register_buffer("packed", packed.contiguous(), persistent=False)
        self.register_buffer("blocks", blocks.contiguous(), persistent=False)
        self.register_buffer("scale", scale.reshape(1), persistent=False)
        self.register_buffer("input_scale", input_scale.reshape(1), persistent=False)
        self.kernel = kernel
        self.calls = 0

    def forward(self, x: Tensor) -> Tensor:
        if x.device.type != "cuda" or self.packed.device != x.device:
            raise RuntimeError("NVFP4 inputs and packed weights must share a CUDA device.")
        if self.blocks.dtype != torch.float8_e4m3fn or self.scale.dtype != torch.float32:
            raise RuntimeError("NVFP4 scales were unexpectedly cast by pipeline placement.")
        if x.shape[-1] != self.in_features:
            raise ValueError("NVFP4 linear input dimensions do not match its checkpoint.")
        flat = x.reshape(-1, self.in_features).contiguous()
        qx, bx = self.kernel.quantize_nvfp4(flat, self.input_scale, pad_16x=True)
        output = self.kernel.scaled_mm_nvfp4(qx, self.packed, tensor_scale_a=self.input_scale,
            tensor_scale_b=self.scale, block_scale_a=bx, block_scale_b=self.blocks, out_dtype=x.dtype)
        self.calls += 1
        return output[:flat.shape[0], :self.out_features].reshape(*x.shape[:-1], self.out_features)


def target_modules(name: str) -> tuple[str, ...]:
    parts = name.split(".")
    block, index, suffix = parts[0], parts[1], ".".join(parts[2:])
    if block == "double_blocks":
        prefix = f"transformer_blocks.{index}."
        mapping = {"img_attn.proj": ("attn.to_out.0",), "img_mlp.0": ("ff.linear_in",),
            "img_mlp.2": ("ff.linear_out",), "txt_attn.proj": ("attn.to_add_out",),
            "txt_mlp.0": ("ff_context.linear_in",), "txt_mlp.2": ("ff_context.linear_out",),
            "img_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
            "txt_attn.qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")}
    elif block == "single_blocks":
        prefix = f"single_transformer_blocks.{index}."
        mapping = {"linear1": ("attn.to_qkv_mlp_proj",), "linear2": ("attn.to_out",)}
    else:
        raise ValueError(f"Unexpected quantized checkpoint module: {name}")
    return tuple(prefix + target for target in mapping[suffix])


def load_transformer() -> Flux2Transformer2DModel:
    kernel = kernel_api()
    config_path = hf_hub_download(BASE_REPO, "transformer/config.json", revision=BASE_REVISION,
                                  local_files_only=True, token=False)
    checkpoint = hf_hub_download(QUANT_REPO, QUANT_FILENAME, revision=QUANT_REVISION,
                                  local_files_only=True, token=False)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    with init_empty_weights():
        model = Flux2Transformer2DModel.from_config(config).to(dtype=torch.bfloat16)
    with safe_open(checkpoint, framework="pt", device="cpu") as saved:
        metadata = saved.metadata()
        if metadata is None or "_quantization_metadata" not in metadata:
            raise ValueError("Official quantization metadata is missing.")
        quant = QuantMetadata.model_validate_json(metadata["_quantization_metadata"])
        if len(quant.layers) != 80:
            raise ValueError("The official Klein checkpoint must have 80 quantized source modules.")
        consumed: set[str] = set()
        for name in quant.layers:
            packed_key, block_key = name + ".weight", name + ".weight_scale"
            scale_key, input_key = name + ".weight_scale_2", name + ".input_scale"
            packed, blocks = saved.get_tensor(packed_key), saved.get_tensor(block_key)
            scale, input_scale = saved.get_tensor(scale_key), saved.get_tensor(input_key)
            consumed.update((packed_key, block_key, scale_key, input_key))
            targets = target_modules(name)
            # All QKV splits are multiples of 128 rows. Whole cuBLAS scale
            # tiles therefore remain intact; the CUDA split test checks this.
            weights = packed.chunk(len(targets), dim=0)
            scales = blocks.chunk(len(targets), dim=0)
            for target, weight, block_scales in zip(targets, weights, scales, strict=True):
                previous = model.get_submodule(target)
                if not isinstance(previous, nn.Linear) or previous.bias is not None:
                    raise ValueError(f"Unexpected target linear module: {target}")
                replacement = Nvfp4Linear(weight, block_scales, scale, input_scale, kernel)
                if (replacement.in_features, replacement.out_features) != (previous.in_features, previous.out_features):
                    raise ValueError(f"Checkpoint dimensions mismatch: {target}")
                parent, leaf = target.rsplit(".", 1)
                setattr(model.get_submodule(parent), leaf, replacement)
        remaining = {key: saved.get_tensor(key) for key in saved.keys() if key not in consumed}
    converted = convert_flux2_transformer_checkpoint_to_diffusers(remaining)
    model.load_state_dict(converted, strict=True, assign=True)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("Checkpoint load left uninitialized parameters.")
    return model.eval()


def kernel_calls(model: nn.Module) -> int:
    return sum(module.calls for module in model.modules() if isinstance(module, Nvfp4Linear))


def transformer_storage_bytes(model: nn.Module) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in (*model.parameters(), *model.buffers()))
