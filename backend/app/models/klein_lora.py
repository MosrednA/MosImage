"""Reversible transformer-only LoRA residuals, including native NVFP4 linears.

This deliberately supports the two evaluated ai-toolkit/PEFT Flux2 exports.
Packed base weights are never merged, cast or dequantized. No PEFT dependency.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_flux2_lora_to_diffusers
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F

from app.models.nvfp4_runtime import Nvfp4Linear

if TYPE_CHECKING:
    from app.jobs import JobControl


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


class LoraLinear(nn.Module):
    def __init__(self, base: nn.Linear | Nvfp4Linear, down: Tensor, up: Tensor) -> None:
        super().__init__()
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] == 0:
            raise ValueError("LoRA matrices must have a positive two-dimensional rank.")
        if down.shape[1] != base.in_features or up.shape != (base.out_features, down.shape[0]):
            raise ValueError("LoRA dimensions do not match the target linear.")
        if not all(t.is_floating_point() and torch.isfinite(t).all() for t in (down, up)):
            raise ValueError("LoRA weights must be finite floating-point matrices.")
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        device = base.packed.device if isinstance(base, Nvfp4Linear) else base.weight.device
        dtype = torch.bfloat16 if isinstance(base, Nvfp4Linear) else base.weight.dtype
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


class KleinLora:
    """One attached adapter. All targets validate before the model is mutated."""

    def __init__(self, targets: tuple[LoraTarget, ...]) -> None:
        self.targets = targets
        self.attached = True

    @classmethod
    def attach(cls, model: nn.Module, checkpoint: Path,
               check_cancelled: Callable[[], None]) -> KleinLora:
        check_cancelled()
        original = load_file(str(checkpoint), device="cpu")
        prefixes = ("diffusion_model.", "base_model.model.")
        prefix = next((p for p in prefixes if original and all(k.startswith(p) for k in original)), None)
        if prefix is None:
            raise ValueError("Expected a transformer-only Flux2 ai-toolkit or PEFT export.")
        if any(k.startswith(prefix + 'final_layer.adaLN_modulation.') for k in original):
            # Original BFL stores shift/scale in the opposite order to Diffusers.
            # Neither evaluated export contains this target; the pinned LoRA
            # converter does not swap its B rows as the base-weight converter does.
            raise ValueError("Original final AdaLN LoRA targets require additional output-row conversion.")
        # The pinned vendor converter handles fused QKV and input/modulation layers.
        converted = _convert_non_diffusers_flux2_lora_to_diffusers(
            {"diffusion_model." + k[len(prefix):]: v for k, v in original.items()})
        pairs: dict[str, dict[str, Tensor]] = {}
        for key, value in converted.items():
            module, separator, suffix = key.removeprefix("transformer.").rpartition(".lora_")
            if not separator or suffix not in ("A.weight", "B.weight"):
                raise ValueError(f"Unsupported LoRA tensor: {key}")
            pairs.setdefault(module, {})[suffix[0]] = value
        targets: list[LoraTarget] = []
        for name, pair in pairs.items():
            check_cancelled()
            if set(pair) != {"A", "B"}:
                raise ValueError(f"Incomplete LoRA pair: {name}")
            base = model.get_submodule(name)
            if not isinstance(base, (nn.Linear, Nvfp4Linear)):
                raise ValueError(f"Unsupported or already adapted LoRA target: {name}")
            parent, _, leaf = name.rpartition(".")
            targets.append(LoraTarget(model.get_submodule(parent) if parent else model, leaf,
                                      LoraLinear(base, pair["A"], pair["B"])))
        if not targets:
            raise ValueError("The LoRA contains no linear targets.")
        check_cancelled()
        # No cancellation or I/O during publication; a cancelled preparation
        # leaves every original module intact.
        for target in targets:
            setattr(target.parent, target.leaf, target.layer)
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
        if any(getattr(t.parent, t.leaf) is not t.layer for t in self.targets):
            raise RuntimeError("A LoRA target was replaced while the adapter was attached.")
        for target in self.targets:
            setattr(target.parent, target.leaf, target.layer.base)
        self.attached = False

    @property
    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for target in self.targets
                   for t in (target.layer.down, target.layer.up))
