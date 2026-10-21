import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from torch import nn

from app.models.klein_lora import KleinLora, LoraLinear, lora_pairs
from app.models.nvfp4_runtime import Nvfp4Linear


class KleinLoraTests(unittest.TestCase):
    def test_residual_matches_independent_linear_oracle_and_zero_is_exact(self):
        torch.manual_seed(7)
        base = nn.Linear(8, 12, bias=True)
        down, up, x = torch.randn(3, 8), torch.randn(12, 3), torch.randn(2, 4, 8)
        layer = LoraLinear(base, down, up)
        original = base(x)
        self.assertTrue(torch.equal(layer(x), original))
        layer.scale = .6
        expected = original + .6 * ((x @ down.T) @ up.T)
        torch.testing.assert_close(layer(x), expected)
        layer.scale = 0
        self.assertTrue(torch.equal(layer(x), original))

    def test_nvfp4_buffers_are_preserved_without_merging_or_casting(self):
        # Construction does not invoke the kernel. Real CUDA coverage is in the benchmark.
        class UnusedKernel:
            pass
        base = Nvfp4Linear(torch.zeros(128, 32, dtype=torch.uint8),
                           torch.ones(128, 4).to(torch.float8_e4m3fn),
                           torch.ones(1), torch.ones(1), UnusedKernel())
        buffers = tuple(base.buffers())
        layer = LoraLinear(base, torch.randn(2, 64), torch.randn(128, 2))
        self.assertIs(layer.base, base)
        self.assertTrue(all(a is b for a, b in zip(buffers, base.buffers(), strict=True)))
        self.assertEqual(base.blocks.dtype, torch.float8_e4m3fn)
        self.assertEqual(base.scale.dtype, torch.float32)
        self.assertEqual(layer.down.dtype, torch.bfloat16)

    def checkpoint(self, directory: str, second_width: int = 4) -> Path:
        path = Path(directory) / "adapter.safetensors"
        save_file({"diffusion_model.transformer_blocks.0.0.lora_A.weight": torch.ones(2, 4),
                   "diffusion_model.transformer_blocks.0.0.lora_B.weight": torch.ones(4, 2),
                   "diffusion_model.transformer_blocks.0.1.lora_A.weight": torch.ones(2, second_width),
                   "diffusion_model.transformer_blocks.0.1.lora_B.weight": torch.ones(4, 2)}, str(path))
        return path

    def model(self) -> nn.Module:
        model = nn.Module()
        model.transformer_blocks = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))])
        return model

    def test_attach_disable_and_detach_restore_original_modules(self):
        model = self.model()
        originals = tuple(model.transformer_blocks[0])
        with tempfile.TemporaryDirectory() as directory:
            adapter = KleinLora.attach(model, self.checkpoint(directory), lambda: None)
        self.assertEqual(len(adapter.targets), 2)
        x = torch.ones(1, 4)
        baseline = originals[1](originals[0](x))
        self.assertTrue(torch.equal(model.transformer_blocks[0](x), baseline))
        adapter.set_scale(.6)
        self.assertFalse(torch.equal(model.transformer_blocks[0](x), baseline))
        adapter.set_scale(0)
        self.assertTrue(torch.equal(model.transformer_blocks[0](x), baseline))
        for invalid in (float("nan"), float("inf"), -1, 2.1):
            with self.assertRaises(ValueError):
                adapter.set_scale(invalid)
        adapter.detach()
        adapter.detach()
        self.assertEqual(tuple(model.transformer_blocks[0]), originals)
        with self.assertRaises(RuntimeError):
            adapter.set_scale(1)

    def test_invalid_later_target_and_cancel_do_not_partially_attach(self):
        model = self.model()
        originals = tuple(model.transformer_blocks[0])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                KleinLora.attach(model, self.checkpoint(directory, 5), lambda: None)
            self.assertEqual(tuple(model.transformer_blocks[0]), originals)
            calls = 0
            def cancel():
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("cancelled")
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                KleinLora.attach(model, self.checkpoint(directory), cancel)
            self.assertEqual(tuple(model.transformer_blocks[0]), originals)

    def test_fused_qkv_export_splits_output_and_shares_input_correctly(self):
        model = nn.Module()
        model.transformer_blocks = nn.ModuleList([nn.Module()])
        model.transformer_blocks[0].attn = nn.Module()
        for name in ("to_q", "to_k", "to_v"):
            setattr(model.transformer_blocks[0].attn, name, nn.Linear(4, 4, bias=False))
        down, up = torch.randn(2, 4), torch.randn(12, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            save_file({"base_model.model.double_blocks.0.img_attn.qkv.lora_A.weight": down,
                       "base_model.model.double_blocks.0.img_attn.qkv.lora_B.weight": up}, str(path))
            adapter = KleinLora.attach(model, path, lambda: None)
        adapter.set_scale(1.1)
        x = torch.randn(2, 4)
        for target, chunk in zip(adapter.targets, up.chunk(3), strict=True):
            expected = target.layer.base(x) + 1.1 * ((x @ down.T) @ chunk.T)
            torch.testing.assert_close(target.layer(x), expected)

    def random_checkpoint(self, directory: str, seed: int) -> tuple[Path, torch.Tensor, torch.Tensor]:
        """A LoRA on the first linear of ``model()`` only."""
        generator = torch.Generator().manual_seed(seed)
        down, up = torch.randn(2, 4, generator=generator), torch.randn(4, 2, generator=generator)
        path = Path(directory) / f"adapter-{seed}.safetensors"
        save_file({"diffusion_model.transformer_blocks.0.0.lora_A.weight": down,
                   "diffusion_model.transformer_blocks.0.0.lora_B.weight": up}, str(path))
        return path, down, up

    def test_stacked_adapters_add_up_and_detach_in_any_order(self):
        x = torch.randn(3, 4)
        scales = (.5, 1.5)
        for first in (0, 1):
            model = self.model()
            original = model.transformer_blocks[0][0]
            with tempfile.TemporaryDirectory() as directory:
                checkpoints = [self.random_checkpoint(directory, seed) for seed in (0, 1)]
                adapters = [KleinLora.attach(model, path, lambda: None) for path, _, _ in checkpoints]
            def residual(index: int) -> torch.Tensor:
                _, down, up = checkpoints[index]
                return scales[index] * ((x @ down.T) @ up.T)
            self.assertTrue(torch.equal(model.transformer_blocks[0][0](x), original(x)))
            for adapter, scale in zip(adapters, scales, strict=True):
                adapter.set_scale(scale)
            torch.testing.assert_close(model.transformer_blocks[0][0](x), original(x) + residual(0) + residual(1))
            adapters[first].detach()
            torch.testing.assert_close(model.transformer_blocks[0][0](x), original(x) + residual(1 - first))
            adapters[1 - first].detach()
            self.assertIs(model.transformer_blocks[0][0], original)

    def test_detach_refuses_a_target_replaced_underneath_it(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            adapter = KleinLora.attach(model, self.random_checkpoint(directory, 0)[0], lambda: None)
        model.transformer_blocks[0][0] = nn.Linear(4, 4)
        with self.assertRaisesRegex(RuntimeError, "replaced"):
            adapter.detach()

    def test_kohya_alpha_is_folded_into_the_down_matrix(self):
        down, up = torch.randn(2, 4), torch.randn(4, 2)
        pairs = lora_pairs({"lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": down,
                            "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": up,
                            "lora_unet_double_blocks_0_img_attn_proj.alpha": torch.tensor(1.0)}, None)
        pair = pairs["transformer_blocks.0.attn.to_out.0"]
        torch.testing.assert_close(pair["A"], down * .5)
        torch.testing.assert_close(pair["B"], up)

    def test_comfyui_alpha_keys_scale_like_kohya(self):
        down, up = torch.randn(2, 4), torch.randn(4, 2)
        pairs = lora_pairs({"diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight": down,
                            "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight": up,
                            "diffusion_model.double_blocks.0.img_attn.proj.alpha": torch.tensor(1.0)}, None)
        torch.testing.assert_close(pairs["transformer_blocks.0.attn.to_out.0"]["A"], down * .5)

    def test_diffusers_export_applies_its_peft_alpha(self):
        down, up = torch.randn(4, 4), torch.randn(4, 4)
        state = {"transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight": down,
                 "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight": up}
        metadata = {"lora_adapter_metadata": json.dumps({"transformer.r": 4, "transformer.lora_alpha": 2})}
        torch.testing.assert_close(lora_pairs(state, metadata)["transformer_blocks.0.attn.to_out.0"]["A"], down * .5)
        torch.testing.assert_close(lora_pairs(state, None)["transformer_blocks.0.attn.to_out.0"]["A"], down)

    def test_diffusers_names_without_their_prefix_and_rslora(self):
        # As in the "comfyui" upload of Limbicnation/pixel-art-lora; its Diffusers twin trained with rsLoRA.
        down, up = torch.randn(4, 4), torch.randn(4, 4)
        bare = {"transformer_blocks.0.attn.to_out.0.lora_A.weight": down, "proj_out.lora_A.weight": down,
                "transformer_blocks.0.attn.to_out.0.lora_B.weight": up, "proj_out.lora_B.weight": up}
        self.assertEqual(set(lora_pairs(bare, None)), {"transformer_blocks.0.attn.to_out.0", "proj_out"})
        rslora = {"lora_adapter_metadata": json.dumps({"transformer.r": 4, "transformer.lora_alpha": 8, "transformer.use_rslora": True})}
        torch.testing.assert_close(lora_pairs(bare, rslora)["proj_out"]["A"], down * 4)

    def test_lora_parts_that_cannot_be_applied_whole_are_rejected(self):
        a, b = torch.ones(2, 4), torch.ones(4, 2)
        cases = {
            "DoRA": {"diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight": a,
                     "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight": b,
                     "diffusion_model.double_blocks.0.img_attn.proj.dora_scale": torch.ones(1, 4)},
            "text encoder": {"lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": a,
                             "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": b,
                             "lora_te1_encoder_layers_0_q_proj.lora_down.weight": a},
            "Unsupported": {"lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": a,
                            "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": b,
                            "lora_unet_unknown_layer.lora_down.weight": a},
            "Klein does not have": {"diffusion_model.unknown.lora_A.weight": a, "diffusion_model.unknown.lora_B.weight": b},
            "transformer-only": {"unet.down.lora_A.weight": a},
        }
        for message, state in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                lora_pairs(state, None)

    def test_nonfinite_and_dtype_overflow_weights_are_rejected(self):
        for value in (float('nan'),float('inf'),1e300):
            with self.assertRaises(ValueError):
                LoraLinear(nn.Linear(4,4),torch.full((2,4),value,dtype=torch.float64),torch.ones(4,2))

    def test_original_adaln_order_is_not_silently_mapped_to_diffusers(self):
        model=nn.Module()
        model.norm_out=nn.Module()
        model.norm_out.linear=nn.Linear(4,8)
        original=model.norm_out.linear
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapter.safetensors'
            save_file({'diffusion_model.final_layer.adaLN_modulation.1.lora_A.weight':torch.ones(2,4),
                       'diffusion_model.final_layer.adaLN_modulation.1.lora_B.weight':torch.ones(8,2)},str(path))
            with self.assertRaisesRegex(ValueError,'output-row conversion'):
                KleinLora.attach(model,path,lambda:None)
        self.assertIs(model.norm_out.linear,original)


if __name__ == "__main__":
    unittest.main()
