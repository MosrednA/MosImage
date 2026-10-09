"""FLUX.2 Klein 9B text encoder held in RAM and copied to the GPU per prompt.

Klein reads three intermediate Qwen3-8B hidden states, so the layers above the
last one it reads never run. Their weights are skipped at load, the projection
weights are stored as per-row scaled FP8, and the token embedding is looked up
on the CPU. That leaves about 5 GiB, pinned in RAM, which takes under half a
second to copy to the GPU for an encode and is released straight afterwards.
The resident transformer therefore never shares the card with a 16 GB encoder.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

HIDDEN_STATE_LAYERS = (9, 18, 27)
# hidden_states[27] is the input of layer 27; the final norm only touches the
# last entry, so one layer past the deepest one read keeps all three exact.
KEEP_LAYERS = max(HIDDEN_STATE_LAYERS) + 1


def load_text_encoder(root: Path, check_cancelled: Callable[[], None]) -> Any:
    """Build the streamed encoder from a snapshot holding ``text_encoder/``."""
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from torch import nn
    from torch.func import functional_call
    from transformers import Qwen3Config, Qwen3Model

    class Fp8Linear(nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            weight = weight.float()
            scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / torch.finfo(torch.float8_e4m3fn).max
            self.register_buffer("weight", (weight / scale).to(torch.float8_e4m3fn))
            self.register_buffer("scale", scale.to(torch.bfloat16))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return nn.functional.linear(x, self.weight.to(x.dtype) * self.scale)

    class CpuEmbedding(nn.Module):
        """A 1.2 GB table of which a prompt reads 512 rows."""

        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("weight", weight)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            rows = nn.functional.embedding(input_ids.cpu(), self.weight)
            return rows.to(input_ids.device, non_blocking=True)

    class StreamedTextEncoder(nn.Module):
        def __init__(self, model: nn.Module, cpu_state: dict[str, torch.Tensor]) -> None:
            super().__init__()
            self.inner = model
            self.config = model.config
            self.dtype = torch.bfloat16
            self._cpu_state = cpu_state

        @property
        def device(self) -> torch.device:
            # Diffusers places the prompt tokens on this device.
            return torch.device("cuda")

        def _apply(self, fn: Any, recurse: bool = True) -> "StreamedTextEncoder":
            # Pipeline .to() calls must not move the RAM copy onto the GPU.
            return self

        def forward(self, **kwargs: Any) -> Any:
            gpu = {name: tensor.to("cuda", non_blocking=True) for name, tensor in self._cpu_state.items()}
            try:
                return functional_call(self.inner, gpu, args=(), kwargs=kwargs)
            finally:
                del gpu

    folder = root / "text_encoder"
    config = Qwen3Config.from_pretrained(folder)
    config.num_hidden_layers = KEEP_LAYERS
    config.layer_types = config.layer_types[:KEEP_LAYERS]
    with init_empty_weights():
        model = Qwen3Model(config)
    index = json.loads((folder / "model.safetensors.index.json").read_text(encoding="utf-8"))
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        check_cancelled()
        with safe_open(str(folder / shard), framework="pt") as weights:
            for key in weights.keys():
                if not key.startswith("model."):
                    continue  # lm_head: logits are never used
                name = key.removeprefix("model.")
                if name.startswith("layers.") and int(name.split(".")[1]) >= KEEP_LAYERS:
                    continue
                # clone() detaches the tensor from the memory-mapped shard,
                # which would otherwise stay resident after loading.
                tensor = weights.get_tensor(key).to(torch.bfloat16).clone()
                if name.startswith("layers.") and name.endswith("proj.weight"):
                    parent, leaf = name.removesuffix(".weight").rsplit(".", 1)
                    setattr(model.get_submodule(parent), leaf, Fp8Linear(tensor))
                elif name == "embed_tokens.weight":
                    model.embed_tokens = CpuEmbedding(tensor)
                else:
                    state[name] = tensor
    model.load_state_dict(state, strict=False, assign=True)
    # The rotary table is a non-persistent buffer, so the checkpoint lacks it.
    model.rotary_emb = type(model.rotary_emb)(config=config)
    model.eval().requires_grad_(False)
    leftover = [name for name, tensor in (*model.named_parameters(), *model.named_buffers()) if tensor.is_meta]
    if leftover:
        raise RuntimeError(f"Text encoder weights missing from the checkpoint: {', '.join(leftover[:5])}")
    check_cancelled()
    cpu_state: dict[str, torch.Tensor] = {}
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        tensor.data = tensor.data.pin_memory()
        if not name.startswith("embed_tokens."):
            cpu_state[name] = tensor
    return StreamedTextEncoder(model, cpu_state)
