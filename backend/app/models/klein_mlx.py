"""FLUX.2 Klein's transformer and text encoder in MLX, quantized, behind torch-shaped modules.

Apple silicon only. The diffusers pipelines keep doing everything around the
two large components (references, masks, the scheduler and the VAE on MPS),
so every edit mode works unchanged. Each wrapper answers the few calls a Klein
pipeline makes, and tensors cross between PyTorch and MLX without a copy.

MLX quantizes the linear layers of the transformer blocks and of the encoder's
decoder layers group-wise (affine, 64 weights per scale and bias). Embedders,
modulation, norms and the output projection stay BF16: they are about 4% of
the weights and steer every token.

The layer names follow diffusers' ``Flux2Transformer2DModel`` exactly, so the
checkpoints load without renaming and ``KleinLora`` finds the same targets.
The Flux2 code mirrors diffusers 0.41; mflux (MIT) showed the way.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
import torch

from app.models.klein_lora import stacked, unstack

# Weight formats: BF16 as stored, or affine 8-bit/4-bit with groups of 64.
Precision = Literal["bf16", "q8", "q4"]
BITS: dict[str, int] = {"q8": 8, "q4": 4}
GROUP_SIZE = 64
# Klein reads these hidden states of its Qwen3 encoder; hidden_states[27] is the output of layer 26.
HIDDEN_STATE_LAYERS = (9, 18, 27)


# ------------------------------------------------------------------ interop

def _to_mx(tensor: torch.Tensor, dtype: mx.Dtype | None = None) -> mx.array:
    """Shares the tensor's memory (MLX 0.32 DLPack); the caller has synchronized MPS."""
    array = mx.asarray(tensor.detach())
    return array if dtype is None else array.astype(dtype)


def _synchronize(tensor: torch.Tensor) -> None:
    """MLX reads the tensor's memory directly, so PyTorch's queued work on it must finish first."""
    if tensor.device.type == "mps":
        torch.mps.synchronize()


def _to_torch(array: mx.array, like: torch.Tensor) -> torch.Tensor:
    mx.eval(array)
    # A copy, so the result no longer depends on MLX's buffer after this call.
    return torch.from_dlpack(array).to(device=like.device, dtype=like.dtype).clone()


# ------------------------------------------------------------------ transformer

def _rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Flux's interleaved rotary embedding: pairs (x0, x1) rotate together. ``cos``/``sin`` are [S, D/2]."""
    batch, heads, length, dim = x.shape
    pairs = x.astype(mx.float32).reshape(batch, heads, length, dim // 2, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = mx.stack([real * cos - imag * sin, real * sin + imag * cos], axis=-1)
    return rotated.reshape(batch, heads, length, dim).astype(x.dtype)


def _heads(x: mx.array, heads: int) -> mx.array:
    batch, length, _ = x.shape
    return x.reshape(batch, length, heads, -1).transpose(0, 2, 1, 3)


def _merge(x: mx.array) -> mx.array:
    batch, _, length, _ = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, length, -1)


def _attention(query: mx.array, key: mx.array, value: mx.array) -> mx.array:
    return mx.fast.scaled_dot_product_attention(query, key, value, scale=query.shape[-1] ** -0.5)


def _swiglu(x: mx.array) -> mx.array:
    gate, value = mx.split(x, 2, axis=-1)
    return nn.silu(gate) * value


class _LayerNorm(nn.Module):
    """LayerNorm without affine parameters, as Flux2 uses it everywhere."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.layer_norm(x, None, None, self.eps)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float) -> None:
        super().__init__()
        inner = int(dim * mult)
        self.linear_in = nn.Linear(dim, inner * 2, bias=False)
        self.linear_out = nn.Linear(inner, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_out(_swiglu(self.linear_in(x)))


class Attention(nn.Module):
    """Joint attention of a double-stream block: text tokens first, then image tokens."""

    def __init__(self, dim: int, heads: int, head_dim: int, eps: float) -> None:
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_out = [nn.Linear(inner, dim, bias=False)]
        self.add_q_proj = nn.Linear(dim, inner, bias=False)
        self.add_k_proj = nn.Linear(dim, inner, bias=False)
        self.add_v_proj = nn.Linear(dim, inner, bias=False)
        self.norm_added_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_add_out = nn.Linear(inner, dim, bias=False)

    def __call__(self, image: mx.array, text: mx.array, rope: tuple[mx.array, mx.array]) -> tuple[mx.array, mx.array]:
        query = mx.concatenate([self.norm_added_q(_heads(self.add_q_proj(text), self.heads)),
                                self.norm_q(_heads(self.to_q(image), self.heads))], axis=2)
        key = mx.concatenate([self.norm_added_k(_heads(self.add_k_proj(text), self.heads)),
                              self.norm_k(_heads(self.to_k(image), self.heads))], axis=2)
        value = mx.concatenate([_heads(self.add_v_proj(text), self.heads), _heads(self.to_v(image), self.heads)], axis=2)
        out = _merge(_attention(_rope(query, *rope), _rope(key, *rope), value))
        length = text.shape[1]
        return self.to_out[0](out[:, length:]), self.to_add_out(out[:, :length])


class ParallelSelfAttention(nn.Module):
    """A single-stream block's attention and MLP, sharing one input and one output projection."""

    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.heads = heads
        self.inner = heads * head_dim
        self.mlp_hidden = int(dim * mlp_ratio)
        self.to_qkv_mlp_proj = nn.Linear(dim, self.inner * 3 + self.mlp_hidden * 2, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)
        self.to_out = nn.Linear(self.inner + self.mlp_hidden, dim, bias=False)

    def __call__(self, x: mx.array, rope: tuple[mx.array, mx.array]) -> mx.array:
        projected = self.to_qkv_mlp_proj(x)
        query, key, value, mlp = mx.split(projected, [self.inner, 2 * self.inner, 3 * self.inner], axis=-1)
        query = _rope(self.norm_q(_heads(query, self.heads)), *rope)
        key = _rope(self.norm_k(_heads(key, self.heads)), *rope)
        out = _merge(_attention(query, key, _heads(value, self.heads)))
        return self.to_out(mx.concatenate([out, _swiglu(mlp)], axis=-1))


def _modulation(mod: mx.array, sets: int) -> list[list[mx.array]]:
    """Splits a modulation vector into ``sets`` (shift, scale, gate) triples, each [B, 1, D]."""
    chunks = mx.split(mod[:, None, :], 3 * sets, axis=-1)
    return [chunks[3 * index:3 * index + 3] for index in range(sets)]


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.norm1 = _LayerNorm(eps)
        self.norm1_context = _LayerNorm(eps)
        self.attn = Attention(dim, heads, head_dim, eps)
        self.norm2 = _LayerNorm(eps)
        self.ff = FeedForward(dim, mlp_ratio)
        self.norm2_context = _LayerNorm(eps)
        self.ff_context = FeedForward(dim, mlp_ratio)

    def __call__(self, image: mx.array, text: mx.array, mod_image: mx.array, mod_text: mx.array,
                 rope: tuple[mx.array, mx.array]) -> tuple[mx.array, mx.array]:
        (shift, scale, gate), (mlp_shift, mlp_scale, mlp_gate) = _modulation(mod_image, 2)
        (c_shift, c_scale, c_gate), (c_mlp_shift, c_mlp_scale, c_mlp_gate) = _modulation(mod_text, 2)
        image_out, text_out = self.attn((1 + scale) * self.norm1(image) + shift,
                                        (1 + c_scale) * self.norm1_context(text) + c_shift, rope)
        image = image + gate * image_out
        image = image + mlp_gate * self.ff(self.norm2(image) * (1 + mlp_scale) + mlp_shift)
        text = text + c_gate * text_out
        text = text + c_mlp_gate * self.ff_context(self.norm2_context(text) * (1 + c_mlp_scale) + c_mlp_shift)
        return image, text


class SingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.norm = _LayerNorm(eps)
        self.attn = ParallelSelfAttention(dim, heads, head_dim, mlp_ratio, eps)

    def __call__(self, x: mx.array, mod: mx.array, rope: tuple[mx.array, mx.array]) -> mx.array:
        ((shift, scale, gate),) = _modulation(mod, 1)
        return x + gate * self.attn((1 + scale) * self.norm(x) + shift, rope)


class _TimestepEmbedding(nn.Module):
    def __init__(self, channels: int, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(channels, dim, bias=False)
        self.linear_2 = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class _TimeGuidanceEmbed(nn.Module):
    def __init__(self, channels: int, dim: int) -> None:
        super().__init__()
        self.channels = channels
        self.timestep_embedder = _TimestepEmbedding(channels, dim)

    def __call__(self, timestep: mx.array) -> mx.array:
        # diffusers' Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0): cos first, then sin.
        half = self.channels // 2
        frequencies = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        angles = timestep.astype(mx.float32)[:, None] * frequencies[None, :]
        projected = mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)
        return self.timestep_embedder(projected.astype(timestep.dtype))


class _Modulation(nn.Module):
    def __init__(self, dim: int, sets: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim * 3 * sets, bias=False)

    def __call__(self, temb: mx.array) -> mx.array:
        return self.linear(nn.silu(temb))


class _AdaNormOut(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.linear = nn.Linear(dim, dim * 2, bias=False)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        scale, shift = mx.split(self.linear(nn.silu(temb)).astype(x.dtype), 2, axis=-1)
        return mx.fast.layer_norm(x, None, None, self.eps) * (1 + scale[:, None, :]) + shift[:, None, :]


class Flux2Transformer(nn.Module):
    """The Klein denoiser; the call signature of diffusers' forward, minus the reference KV cache."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        heads, head_dim = config["num_attention_heads"], config["attention_head_dim"]
        dim, eps, mlp_ratio = heads * head_dim, config.get("eps", 1e-6), config.get("mlp_ratio", 3.0)
        if config.get("guidance_embeds") or config.get("patch_size", 1) != 1:
            raise ValueError("Only the distilled Klein transformers (no guidance embedding, patch size 1) are supported.")
        self.rope_theta = config["rope_theta"]
        self.axes_dims = tuple(config["axes_dims_rope"])
        self.time_guidance_embed = _TimeGuidanceEmbed(config["timestep_guidance_channels"], dim)
        self.double_stream_modulation_img = _Modulation(dim, 2)
        self.double_stream_modulation_txt = _Modulation(dim, 2)
        self.single_stream_modulation = _Modulation(dim, 1)
        self.x_embedder = nn.Linear(config["in_channels"], dim, bias=False)
        self.context_embedder = nn.Linear(config["joint_attention_dim"], dim, bias=False)
        self.transformer_blocks = [TransformerBlock(dim, heads, head_dim, mlp_ratio, eps) for _ in range(config["num_layers"])]
        self.single_transformer_blocks = [SingleTransformerBlock(dim, heads, head_dim, mlp_ratio, eps)
                                          for _ in range(config["num_single_layers"])]
        self.norm_out = _AdaNormOut(dim, eps)
        self.proj_out = nn.Linear(dim, config.get("out_channels") or config["in_channels"], bias=False)

    def _rope(self, ids: mx.array) -> tuple[mx.array, mx.array]:
        """cos and sin per token and frequency, [S, D/2], over the four position axes (T, H, W, L)."""
        angles = []
        for axis, dim in enumerate(self.axes_dims):
            frequencies = 1.0 / (self.rope_theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
            angles.append(ids[:, axis:axis + 1].astype(mx.float32) * frequencies[None, :])
        angle = mx.concatenate(angles, axis=-1)
        return mx.cos(angle), mx.sin(angle)

    def __call__(self, hidden_states: mx.array, encoder_hidden_states: mx.array, timestep: mx.array,
                 img_ids: mx.array, txt_ids: mx.array) -> mx.array:
        dtype = hidden_states.dtype
        temb = self.time_guidance_embed((timestep.astype(dtype) * 1000).astype(dtype))
        mod_image = self.double_stream_modulation_img(temb)
        mod_text = self.double_stream_modulation_txt(temb)
        mod_single = self.single_stream_modulation(temb)
        image = self.x_embedder(hidden_states)
        text = self.context_embedder(encoder_hidden_states)
        rope = self._rope(mx.concatenate([txt_ids, img_ids], axis=0))
        for block in self.transformer_blocks:
            image, text = block(image, text, mod_image, mod_text, rope)
        length = text.shape[1]
        joint = mx.concatenate([text, image], axis=1)
        for block in self.single_transformer_blocks:
            joint = block(joint, mod_single, rope)
        return self.proj_out(self.norm_out(joint[:, length:], temb))


# ------------------------------------------------------------------ text encoder

class _Qwen3Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, head_dim: int, eps: float, theta: float) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.theta = heads, kv_heads, theta
        self.q_proj = nn.Linear(dim, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        query = self.q_norm(_heads(self.q_proj(x), self.heads))
        key = self.k_norm(_heads(self.k_proj(x), self.kv_heads))
        value = _heads(self.v_proj(x), self.kv_heads)
        dims = query.shape[-1]
        query = mx.fast.rope(query, dims, traditional=False, base=self.theta, scale=1.0, offset=0)
        key = mx.fast.rope(key, dims, traditional=False, base=self.theta, scale=1.0, offset=0)
        out = mx.fast.scaled_dot_product_attention(query, key, value, scale=dims ** -0.5, mask=mask)
        return self.o_proj(_merge(out))


class _Qwen3MLP(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Qwen3Layer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        dim, eps = config["hidden_size"], config["rms_norm_eps"]
        theta = config.get("rope_theta") or (config.get("rope_parameters") or {}).get("rope_theta", 1_000_000.0)
        self.input_layernorm = nn.RMSNorm(dim, eps=eps)
        self.self_attn = _Qwen3Attention(dim, config["num_attention_heads"], config["num_key_value_heads"],
                                         config.get("head_dim") or dim // config["num_attention_heads"], eps, theta)
        self.post_attention_layernorm = nn.RMSNorm(dim, eps=eps)
        self.mlp = _Qwen3MLP(dim, config["intermediate_size"])

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3Encoder(nn.Module):
    """The first 27 decoder layers of Qwen3, which is all Klein reads; no final norm, no LM head."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [_Qwen3Layer(config) for _ in range(max(HIDDEN_STATE_LAYERS))]

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> dict[int, mx.array]:
        x = self.embed_tokens(input_ids)
        length = input_ids.shape[1]
        # Causal, and padding keys hidden; padding sits on the right, so no row is empty.
        causal = mx.tril(mx.ones((length, length), dtype=mx.bool_))
        mask = causal[None, None] & attention_mask.astype(mx.bool_)[:, None, None, :]
        states: dict[int, mx.array] = {}
        for index, layer in enumerate(self.layers, 1):
            x = layer(x, mask)
            if index in HIDDEN_STATE_LAYERS:
                states[index] = x
        return states


# ------------------------------------------------------------------ LoRA

class LoraLinear(nn.Module):
    """A low-rank residual on a BF16 or quantized linear, or on another residual; the base weights stay as they are."""

    def __init__(self, base: nn.Module, down: mx.array, up: mx.array) -> None:
        super().__init__()
        self.base = base
        self.down = down
        self.up = up
        self.scale = 0.0

    def __call__(self, x: mx.array) -> mx.array:
        result = self.base(x)
        if self.scale == 0:
            return result
        residual = (x.astype(self.down.dtype) @ self.down.T) @ self.up.T
        return result + (residual * self.scale).astype(result.dtype)


def _linear_dims(module: nn.Module) -> tuple[int, int]:
    """(in, out) features of a BF16 or quantized linear, also under residuals."""
    if isinstance(module, LoraLinear):
        return _linear_dims(module.base)
    if isinstance(module, nn.QuantizedLinear):
        out_features, packed = module.weight.shape
        return packed * 32 // module.bits, out_features
    if isinstance(module, nn.Linear):
        out_features, in_features = module.weight.shape
        return in_features, out_features
    raise ValueError(f"Not a linear layer: {type(module).__name__}")


def _parent(model: nn.Module, name: str) -> tuple[Any, str]:
    """The module holding ``name`` and the leaf key, walking lists by index like diffusers' get_submodule."""
    *path, leaf = name.split(".")
    node: Any = model
    for part in path:
        node = node[int(part)] if isinstance(node, list) else getattr(node, part)
    return node, leaf


def _get(parent: Any, leaf: str) -> Any:
    return parent[int(leaf)] if isinstance(parent, list) else getattr(parent, leaf)


def _set(parent: Any, leaf: str, module: nn.Module) -> None:
    if isinstance(parent, list):
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


@dataclass(frozen=True)
class MlxLoraTarget:
    """The MLX counterpart of ``klein_lora.LoraTarget``."""

    parent: Any
    leaf: str
    layer: LoraLinear

    def current(self) -> nn.Module:
        return _get(self.parent, self.leaf)

    def attach(self) -> None:
        _set(self.parent, self.leaf, self.layer)

    def attached(self) -> bool:
        return stacked(self.current(), self.layer)

    def restore(self) -> None:
        unstack(self.current(), self.layer, lambda module: _set(self.parent, self.leaf, module))

    @property
    def storage_bytes(self) -> int:
        return self.layer.down.nbytes + self.layer.up.nbytes


# ------------------------------------------------------------------ loading

def _quantize_linears(module: nn.Module, precision: Precision) -> None:
    if precision != "bf16":
        nn.quantize(module, group_size=GROUP_SIZE, bits=BITS[precision],
                    class_predicate=lambda _path, layer: isinstance(layer, nn.Linear))


def _load(model: nn.Module, weights: Iterator[tuple[str, mx.array]], quantized: list[nn.Module], precision: Precision,
          check_cancelled: Callable[[], None]) -> None:
    """Loads BF16 weights, then quantizes and evaluates one block at a time, so only one block is ever held twice."""
    model.load_weights(list(weights), strict=True)
    for module in quantized:
        check_cancelled()
        _quantize_linears(module, precision)
        mx.eval(module.parameters())
    check_cancelled()
    mx.eval(model.parameters())
    # The BF16 originals were freed above; MLX would otherwise keep them cached.
    mx.clear_cache()


def _safetensors(folder: Path) -> Iterator[tuple[str, mx.array]]:
    for shard in sorted(folder.glob("*.safetensors")):
        # Lazy: a tensor is read from disk when it is first evaluated.
        yield from mx.load(str(shard)).items()


def load_transformer(folder: Path, precision: Precision, check_cancelled: Callable[[], None]) -> "MlxTransformer":
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = Flux2Transformer(config)
    weights = ((key, value.astype(mx.bfloat16)) for key, value in _safetensors(folder))
    _load(model, weights, [*model.transformer_blocks, *model.single_transformer_blocks], precision, check_cancelled)
    return MlxTransformer(model.eval(), config, precision)


def load_text_encoder(folder: Path, precision: Precision, check_cancelled: Callable[[], None]) -> "MlxTextEncoder":
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    model = Qwen3Encoder(config)
    keep = len(model.layers)

    def weights() -> Iterator[tuple[str, mx.array]]:
        for key, value in _safetensors(folder):
            if not key.startswith("model."):
                continue  # lm_head: logits are never used
            name = key.removeprefix("model.")
            if name == "norm.weight" or (name.startswith("layers.") and int(name.split(".")[1]) >= keep):
                continue
            yield name, value.astype(mx.bfloat16)

    _load(model, weights(), model.layers, precision, check_cancelled)
    return MlxTextEncoder(model.eval(), config, precision)


# ------------------------------------------------------------------ torch-facing wrappers

class _MlxModule(torch.nn.Module):
    """A torch module without parameters: pipeline .to() calls leave the MLX weights alone."""

    def __init__(self, precision: Precision) -> None:
        super().__init__()
        self.precision = precision

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def device(self) -> torch.device:
        return torch.device("mps")

    def _apply(self, fn: Any, recurse: bool = True) -> "_MlxModule":
        return self

    @property
    def storage_bytes(self) -> int:
        from mlx.utils import tree_flatten
        return sum(value.nbytes for _, value in tree_flatten(self.mlx.parameters()))


class MlxTransformer(_MlxModule):
    """Stands in for ``Flux2Transformer2DModel`` inside ``Flux2KleinPipeline`` and ``Flux2KleinInpaintPipeline``."""

    def __init__(self, model: Flux2Transformer, config: dict[str, Any], precision: Precision) -> None:
        super().__init__(precision)
        from diffusers.configuration_utils import FrozenDict

        self.mlx = model
        self.config = FrozenDict(config)
        self._context: tuple[torch.Tensor, mx.array] | None = None

    def cache_context(self, name: str) -> Any:
        return nullcontext()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, timestep: torch.Tensor,
                img_ids: torch.Tensor, txt_ids: torch.Tensor, guidance: torch.Tensor | None = None,
                joint_attention_kwargs: dict[str, Any] | None = None, return_dict: bool = True, **kwargs: Any) -> Any:
        if kwargs.get("kv_cache_mode") is not None:
            raise NotImplementedError("The MLX transformer has no reference KV cache.")
        _synchronize(hidden_states)
        # The prompt embedding is the same tensor at every step; convert it once.
        if self._context is None or self._context[0] is not encoder_hidden_states:
            self._context = (encoder_hidden_states, _to_mx(encoder_hidden_states, mx.bfloat16))
        output = self.mlx(_to_mx(hidden_states, mx.bfloat16), self._context[1], _to_mx(timestep, mx.bfloat16),
                          _to_mx(img_ids[0] if img_ids.ndim == 3 else img_ids),
                          _to_mx(txt_ids[0] if txt_ids.ndim == 3 else txt_ids))
        result = _to_torch(output, hidden_states)
        return SimpleNamespace(sample=result) if return_dict else (result,)

    def lora_targets(self, pairs: dict[str, dict[str, torch.Tensor]],
                     check_cancelled: Callable[[], None]) -> list[MlxLoraTarget]:
        """Validates every LoRA pair against its linear before anything is attached."""
        targets = []
        for name, pair in pairs.items():
            check_cancelled()
            try:
                parent, leaf = _parent(self.mlx, name)
                base = _get(parent, leaf)
            except (AttributeError, IndexError, KeyError, ValueError) as exc:
                raise ValueError(f"Unknown LoRA target: {name}") from exc
            if not isinstance(base, (nn.Linear, nn.QuantizedLinear, LoraLinear)):
                raise ValueError(f"Unsupported LoRA target: {name}")
            down, up = pair["A"], pair["B"]
            in_features, out_features = _linear_dims(base)
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] == 0:
                raise ValueError("LoRA matrices must have a positive two-dimensional rank.")
            if down.shape[1] != in_features or tuple(up.shape) != (out_features, down.shape[0]):
                raise ValueError("LoRA dimensions do not match the target linear.")
            down, up = down.to(torch.bfloat16), up.to(torch.bfloat16)
            if not torch.isfinite(down).all() or not torch.isfinite(up).all():
                raise ValueError("LoRA weights must be finite in BF16.")
            layer = LoraLinear(base, mx.array(down.float().numpy()).astype(mx.bfloat16),
                               mx.array(up.float().numpy()).astype(mx.bfloat16))
            targets.append(MlxLoraTarget(parent, leaf, layer))
        mx.eval([(target.layer.down, target.layer.up) for target in targets])
        return targets


class MlxTextEncoder(_MlxModule):
    """Answers the pipeline's ``text_encoder(input_ids, attention_mask, output_hidden_states=True)``."""

    def __init__(self, model: Qwen3Encoder, config: dict[str, Any], precision: Precision) -> None:
        super().__init__(precision)
        self.mlx = model
        self.config = SimpleNamespace(**config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: Any) -> Any:
        _synchronize(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        states = self.mlx(_to_mx(input_ids, mx.int32), _to_mx(attention_mask, mx.int32))
        anchor = torch.empty(0, dtype=torch.bfloat16, device=input_ids.device)
        hidden: list[torch.Tensor | None] = [None] * (max(HIDDEN_STATE_LAYERS) + 1)
        for index, state in states.items():
            hidden[index] = _to_torch(state, anchor)
        return SimpleNamespace(hidden_states=hidden)
