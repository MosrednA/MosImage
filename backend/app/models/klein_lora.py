"""Reversible transformer-only LoRA residuals, including native NVFP4 linears.

Reads the Flux2 LoRA exports Diffusers reads (ai-toolkit, ComfyUI, kohya, PEFT
and Diffusers itself) through its own converters. Packed base weights are
never merged, cast or dequantized. No PEFT dependency. Adapters stack: each
residual wraps the layer before it and can be taken out in any order. The MLX
transformer of a Mac (``klein_mlx``) takes the same adapters through its own
linear layers.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from diffusers.loaders.lora_conversion_utils import (_convert_kohya_flux2_lora_to_diffusers,
                                                     _convert_non_diffusers_flux2_lora_to_diffusers)
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn import functional as F

from app.models.nvfp4_runtime import Nvfp4Linear

if TYPE_CHECKING:
    from app.jobs import JobControl
    from app.models.klein_mlx import MlxLoraTarget


@dataclass(frozen=True)
class AdapterCheckpoint:
    name: str
    repo: str
    revision: str
    filename: str
    sha256: str


CONSISTENCY_V2 = AdapterCheckpoint("Consistency V2", "dx8152/Flux2-Klein-9B-Consistency",
    "8df0c7338cf68cfcd89ca7e461fe679905634607", "Flux2-Klein-9B-consistency-V2.safetensors",
    "61db2017ce420b97bd5ef11984e5a894c90003a6bbf0dc9473f8d7b9ebb3ff93")
FAL_OUTPAINT = AdapterCheckpoint("Specialized outpaint", "fal/flux-2-klein-4B-outpaint-lora",
    "b11770ac6a3cf9325dcf81742c12b1c4e257880f", "flux-outpaint-lora.safetensors",
    "9623927f2471613e0c9d7810da3c39f0b277a448193b223ea3cee67d5750974a")
# Elusarca's Detail Enhancer v2, the same file as on Civitai (FLUX.1 [dev] non-commercial licence).
DETAIL_ENHANCER = AdapterCheckpoint("Detail enhancer", "reverentelusarca/detail-enhancer-flux-klein-9b",
    "82dd21d7b206572f1a110f31b4b4295e9c57f43b", "klein_9b_enhancer_v2.safetensors",
    "520832b426ef976ba81aeb7ad235ac920642c189b55120c51e540541d27f9072")


def resolve_checkpoint(checkpoint: AdapterCheckpoint, control: "JobControl") -> Path:
    from app.config import settings
    from huggingface_hub import hf_hub_download

    control.set_phase("loading", checkpoint.name)
    control.raise_if_cancelled()
    folder = settings.project_root / "models/adapters" / checkpoint.repo.split("/")[-1]
    path = folder / checkpoint.filename
    if not path.is_file():
        path = Path(hf_hub_download(checkpoint.repo, checkpoint.filename,
            revision=checkpoint.revision, local_dir=folder, token=False))
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            control.raise_if_cancelled()
            digest.update(block)
    if digest.hexdigest() != checkpoint.sha256:
        raise ValueError(f"{checkpoint.name} checkpoint checksum mismatch; remove the cached file and retry.")
    return path


def lora_pairs(state: dict[str, Tensor], metadata: dict[str, str] | None) -> dict[str, dict[str, Tensor]]:
    """Each Diffusers linear a LoRA file changes, as ``{"A": down, "B": up}`` with its alpha applied.

    Mirrors ``Flux2LoraLoaderMixin.lora_state_dict`` but rejects what it would
    drop or only warn about (DoRA, text-encoder layers, unknown keys): a partly
    applied LoRA looks plausible and is wrong. Works on meta tensors too, as
    long as the alphas are real.
    """
    if not state:
        raise ValueError("The file holds no tensors.")
    if any("dora_scale" in key or "lora_magnitude_vector" in key for key in state):
        raise ValueError("DoRA LoRAs are not supported.")
    if any(key.startswith(("lora_te", "text_encoder.")) for key in state):
        raise ValueError("The LoRA also changes the text encoder; only transformer LoRAs are supported.")
    if any(key.startswith("lora_unet_") for key in state):
        remaining = dict(state)
        # Pops what it converts and folds each alpha into the weights.
        converted = _convert_kohya_flux2_lora_to_diffusers(remaining)
        if remaining:
            raise ValueError(f"Unsupported LoRA tensor: {next(iter(remaining))}")
    elif all(key.startswith("transformer.") for key in state) or any(
            key.startswith(("transformer_blocks.", "single_transformer_blocks.")) for key in state):
        # Diffusers' own export, also without its "transformer." prefix as some
        # ComfyUI uploads are; its alpha lives in the adapter metadata.
        scale = _peft_scale(metadata)
        converted = {key: value * scale if ".lora_A." in key else value for key, value in _fold_alpha(state).items()}
    else:
        prefix = next((p for p in ("diffusion_model.", "base_model.model.") if all(k.startswith(p) for k in state)), None)
        if prefix is None:
            raise ValueError("Expected a transformer-only Flux2 LoRA (ai-toolkit, ComfyUI, kohya, PEFT or Diffusers).")
        if any(k.startswith(prefix + 'final_layer.adaLN_modulation.') for k in state):
            # Original BFL stores shift/scale in the opposite order to Diffusers.
            # Neither evaluated export contains this target; the pinned LoRA
            # converter does not swap its B rows as the base-weight converter does.
            raise ValueError("Original final AdaLN LoRA targets require additional output-row conversion.")
        # The pinned vendor converter handles fused QKV and input/modulation layers.
        try:
            converted = _convert_non_diffusers_flux2_lora_to_diffusers(
                _fold_alpha({"diffusion_model." + k[len(prefix):]: v for k, v in state.items()}))
        except ValueError as exc:
            raise ValueError("The LoRA has layers FLUX.2 Klein does not have.") from exc
    pairs: dict[str, dict[str, Tensor]] = {}
    for key, value in converted.items():
        module, separator, suffix = key.removeprefix("transformer.").rpartition(".lora_")
        if not separator or suffix not in ("A.weight", "B.weight"):
            raise ValueError(f"Unsupported LoRA tensor: {key}")
        pairs.setdefault(module, {})[suffix[0]] = value
    for name, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Incomplete LoRA pair: {name}")
    if not pairs:
        raise ValueError("The LoRA contains no linear targets.")
    return pairs


def read_lora(checkpoint: Path) -> dict[str, dict[str, Tensor]]:
    with safe_open(str(checkpoint), framework="pt", device="cpu") as file:
        return lora_pairs({key: file.get_tensor(key) for key in file.keys()}, file.metadata())


def _fold_alpha(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Scales each down matrix by alpha / rank, the way ComfyUI reads ``.alpha`` keys."""
    folded = {key: value for key, value in state.items() if not key.endswith(".alpha")}
    for key in state.keys() - folded.keys():
        module = key.removesuffix(".alpha")
        down = next((module + suffix for suffix in (".lora_A.weight", ".lora_down.weight") if module + suffix in folded), None)
        if down is None:
            raise ValueError(f"LoRA alpha without weights: {module}")
        folded[down] = folded[down] * (float(state[key]) / folded[down].shape[0])
    return folded


def _peft_scale(metadata: dict[str, str] | None) -> float:
    """PEFT's ``lora_alpha / r`` from Diffusers' adapter metadata; without it Diffusers assumes one."""
    raw = (metadata or {}).get("lora_adapter_metadata")
    if not raw:
        return 1.0
    config: dict[str, Any] = json.loads(raw)
    if any(key.startswith("transformer.") for key in config):
        config = {key.removeprefix("transformer."): value for key, value in config.items() if key.startswith("transformer.")}
    if config.get("use_dora") or config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("DoRA and per-layer LoRA ranks are not supported.")
    rank = config.get("r")
    if not isinstance(rank, (int, float)) or rank <= 0:
        return 1.0
    alpha = config.get("lora_alpha", rank)
    return float(alpha) / (sqrt(rank) if config.get("use_rslora") else rank)


def stacked(top: Any, layer: Any) -> bool:
    """Whether ``layer`` is in the chain of residuals that starts at ``top``."""
    node = top
    while isinstance(node, type(layer)):
        if node is layer:
            return True
        node = node.base
    return False


def unstack(top: Any, layer: Any, replace_top: Callable[[Any], None]) -> None:
    """Takes ``layer`` out of its chain; the residual above it then wraps what it wrapped."""
    if top is layer:
        replace_top(layer.base)
        return
    node = top
    while isinstance(node, type(layer)):
        if node.base is layer:
            node.base = layer.base
            return
        node = node.base
    raise RuntimeError("A LoRA target was replaced while the adapter was attached.")


class LoraLinear(nn.Module):
    """A low-rank residual on a linear, or on another residual of the same linear."""

    def __init__(self, base: nn.Linear | Nvfp4Linear | LoraLinear, down: Tensor, up: Tensor) -> None:
        super().__init__()
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] == 0:
            raise ValueError("LoRA matrices must have a positive two-dimensional rank.")
        if down.shape[1] != base.in_features or up.shape != (base.out_features, down.shape[0]):
            raise ValueError("LoRA dimensions do not match the target linear.")
        if not all(t.is_floating_point() and torch.isfinite(t).all() for t in (down, up)):
            raise ValueError("LoRA weights must be finite floating-point matrices.")
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        linear = base
        while isinstance(linear, LoraLinear):
            linear = linear.base
        device = linear.packed.device if isinstance(linear, Nvfp4Linear) else linear.weight.device
        dtype = torch.bfloat16 if isinstance(linear, Nvfp4Linear) else linear.weight.dtype
        if device.type == "meta":
            raise ValueError("Load base weights before attaching a LoRA.")
        down, up = down.to(dtype=dtype), up.to(dtype=dtype)
        if not torch.isfinite(down).all() or not torch.isfinite(up).all():
            raise ValueError("LoRA weights overflow the target runtime dtype.")
        self.register_buffer("down", down.to(device=device))
        self.register_buffer("up", up.to(device=device))
        self.scale = 0.0

    def forward(self, x: Tensor) -> Tensor:
        result = self.base(x)
        if self.scale == 0:
            return result
        residual = F.linear(F.linear(x.to(self.down.dtype), self.down), self.up)
        return result + (residual * self.scale).to(result.dtype)


@dataclass(frozen=True)
class LoraTarget:
    parent: nn.Module
    leaf: str
    layer: LoraLinear

    def current(self) -> nn.Module:
        return getattr(self.parent, self.leaf)

    def attach(self) -> None:
        setattr(self.parent, self.leaf, self.layer)

    def attached(self) -> bool:
        return stacked(self.current(), self.layer)

    def restore(self) -> None:
        unstack(self.current(), self.layer, lambda module: setattr(self.parent, self.leaf, module))

    @property
    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.layer.down, self.layer.up))


class KleinLora:
    """One attached adapter. All targets validate before the model is mutated.

    Attach several to stack them; detach them in any order.
    """

    def __init__(self, targets: tuple[LoraTarget | MlxLoraTarget, ...]) -> None:
        self.targets = targets
        self.attached = True

    @classmethod
    def attach(cls, model: nn.Module, checkpoint: Path,
               check_cancelled: Callable[[], None]) -> KleinLora:
        check_cancelled()
        pairs = read_lora(checkpoint)
        targets: list[LoraTarget | MlxLoraTarget] = []
        if hasattr(model, "lora_targets"):
            # The MLX transformer: the same layer names, its own linear layers.
            targets.extend(model.lora_targets(pairs, check_cancelled))
        else:
            for name, pair in pairs.items():
                check_cancelled()
                try:
                    base = model.get_submodule(name)
                except AttributeError as exc:
                    raise ValueError(f"Unknown LoRA target: {name}") from exc
                if not isinstance(base, (nn.Linear, Nvfp4Linear, LoraLinear)):
                    raise ValueError(f"Unsupported LoRA target: {name}")
                parent, _, leaf = name.rpartition(".")
                targets.append(LoraTarget(model.get_submodule(parent) if parent else model, leaf,
                                          LoraLinear(base, pair["A"], pair["B"])))
        if not targets:
            raise ValueError("The LoRA contains no linear targets.")
        check_cancelled()
        # No cancellation or I/O during publication; a cancelled preparation
        # leaves every original module intact.
        for target in targets:
            target.attach()
        return cls(tuple(targets))

    def set_scale(self, scale: float) -> None:
        if not self.attached:
            raise RuntimeError("This LoRA has been detached.")
        if not isfinite(scale) or not 0 <= scale <= 2:
            raise ValueError("LoRA scale must be finite and between zero and two.")
        for target in self.targets:
            target.layer.scale = scale

    def detach(self) -> None:
        if not self.attached:
            return
        if not all(t.attached() for t in self.targets):
            raise RuntimeError("A LoRA target was replaced while the adapter was attached.")
        for target in self.targets:
            target.restore()
        self.attached = False

    @property
    def storage_bytes(self) -> int:
        return sum(target.storage_bytes for target in self.targets)
