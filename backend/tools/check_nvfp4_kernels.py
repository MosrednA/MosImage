"""Explicit CUDA adapter checks, kept out of CPU-only application test runs."""
from __future__ import annotations

import torch
from torch import Tensor

from nvfp4_transformer import Nvfp4Linear, kernel_api


def main() -> None:
    ck = kernel_api()
    generator = torch.Generator("cuda").manual_seed(123)
    weights = torch.randn(768, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    scale = weights.abs().amax().float() / (448 * 6)
    packed, blocks = ck.quantize_nvfp4(weights, scale, pad_16x=False)
    for index in range(3):
        part, part_scales = ck.quantize_nvfp4(weights[index * 256:(index + 1) * 256].contiguous(), scale, pad_16x=False)
        assert torch.equal(packed[index * 256:(index + 1) * 256], part)
        assert torch.equal(blocks[index * 256:(index + 1) * 256].view(torch.uint8), part_scales.view(torch.uint8))
    inputs = torch.randn(1, 17, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
    input_scale = inputs.abs().amax().float() / (448 * 6)
    layer = Nvfp4Linear(packed, blocks, scale, input_scale, ck)
    with torch.inference_mode():
        actual = layer(inputs)
    assert actual.shape == (1, 17, 768) and torch.isfinite(actual).all()
    # Compare the adapter's native GEMM to the same quantized operands after
    # dequantization, not to the original higher-precision random tensors.
    import comfy_kitchen
    qx, bx = ck.quantize_nvfp4(inputs.reshape(-1, 256), input_scale, pad_16x=True)
    decoded_x: Tensor = comfy_kitchen.dequantize_nvfp4(qx, input_scale, bx, torch.bfloat16)
    decoded_w: Tensor = comfy_kitchen.dequantize_nvfp4(packed, scale, blocks, torch.bfloat16)
    reference = torch.nn.functional.linear(decoded_x[:17], decoded_w).reshape_as(actual)
    relative_mean_error = (actual.float() - reference.float()).abs().mean() / reference.float().abs().mean()
    assert relative_mean_error.item() < 0.01, relative_mean_error.item()
    assert layer.calls == 1
    try:
        layer(inputs.cpu())
    except RuntimeError:
        pass
    else:
        raise AssertionError("A CPU input must not silently fall back to BF16.")
    torch.cuda.synchronize()
    print(f"Passed scale-tile splitting, 17-row padding/reshape, native GEMM reference ({relative_mean_error.item():.6f} relative MAE), and no CPU fallback.")


if __name__ == "__main__":
    main()
