"""The MLX Klein runtime of a Mac against diffusers and transformers, on tiny random models."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file

from app import runtime
from app.config import Settings, default_apple_weights, settings
from app.models.flux2_klein_apple import Flux2Klein9bAppleProvider, Flux2KleinAppleProvider
from app.models.flux2_klein_nvfp4 import BASE_REPO_9B, BASE_REVISION_9B
from app.models.klein_lora import KleinLora

MLX = importlib.util.find_spec("mlx") is not None and sys.platform == "darwin"
# Every block linear takes a multiple of 64 inputs, so the tiny model quantizes like the real one.
TINY_TRANSFORMER = {"patch_size": 1, "in_channels": 16, "num_layers": 1, "num_single_layers": 2, "attention_head_dim": 32,
                    "num_attention_heads": 2, "joint_attention_dim": 64, "timestep_guidance_channels": 32, "mlp_ratio": 3.0,
                    "axes_dims_rope": [8, 8, 8, 8], "rope_theta": 2000, "eps": 1e-6, "guidance_embeds": False}


def relative_error(ours: torch.Tensor, reference: torch.Tensor) -> float:
    return float((ours.float() - reference.float()).norm() / reference.float().norm())


@unittest.skipUnless(MLX, "MLX runs on Apple silicon only")
class MlxTransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

        torch.manual_seed(0)
        self.reference = Flux2Transformer2DModel(**TINY_TRANSFORMER).eval()
        self.directory = TemporaryDirectory()
        self.folder = Path(self.directory.name)
        save_file({k: v.contiguous() for k, v in self.reference.state_dict().items()},
                  str(self.folder / "diffusion_pytorch_model.safetensors"))
        (self.folder / "config.json").write_text(json.dumps(dict(self.reference.config)), encoding="utf-8")
        latents = torch.randn(1, 16, 4, 6)
        context = torch.randn(1, 8, 64)
        self.inputs = {"hidden_states": Flux2KleinPipeline._pack_latents(latents), "encoder_hidden_states": context,
                       "timestep": torch.tensor([0.6]), "img_ids": Flux2KleinPipeline._prepare_latent_ids(latents),
                       "txt_ids": Flux2KleinPipeline._prepare_text_ids(context), "guidance": None, "return_dict": False}

    def tearDown(self) -> None:
        self.directory.cleanup()

    def load(self, precision: str):
        from app.models.klein_mlx import load_transformer
        return load_transformer(self.folder, precision, lambda: None)

    def run_model(self, model) -> torch.Tensor:
        with torch.inference_mode():
            return model(**self.inputs)[0]

    def test_bf16_matches_the_diffusers_transformer(self) -> None:
        expected = self.run_model(self.reference)
        output = self.run_model(self.load("bf16"))
        self.assertEqual(output.shape, expected.shape)
        self.assertLess(relative_error(output, expected), 0.02)

    def test_quantized_weights_are_smaller_and_stay_close(self) -> None:
        import mlx.nn as nn

        bf16, q8, q4 = (self.load(precision) for precision in ("bf16", "q8", "q4"))
        self.assertGreater(bf16.storage_bytes, q8.storage_bytes)
        self.assertGreater(q8.storage_bytes, q4.storage_bytes)
        # Only the blocks' linears are quantized; embedders, modulation and the output stay BF16.
        self.assertIsInstance(q8.mlx.transformer_blocks[0].ff.linear_in, nn.QuantizedLinear)
        self.assertIsInstance(q8.mlx.single_transformer_blocks[1].attn.to_qkv_mlp_proj, nn.QuantizedLinear)
        self.assertIsInstance(q8.mlx.context_embedder, nn.Linear)
        self.assertIsInstance(q8.mlx.proj_out, nn.Linear)
        expected = self.run_model(self.reference)
        self.assertLess(relative_error(self.run_model(q8), expected), 0.03)
        self.assertLess(relative_error(self.run_model(q4), expected), 0.3)

    def test_pipeline_facing_attributes(self) -> None:
        model = self.load("q8")
        self.assertEqual(model.config.in_channels, 16)
        self.assertEqual(model.dtype, torch.bfloat16)
        with model.cache_context("uncond"):
            pass
        self.assertIs(model.to("cpu"), model)
        with self.assertRaises(NotImplementedError):
            model(**self.inputs, kv_cache_mode="extract")

    def lora(self, directory: str) -> Path:
        """A BFL-named export: fused image QKV, an output projection and a single-block output."""
        generator = torch.Generator().manual_seed(1)
        tensors = {}
        for name, (inputs, outputs) in {"double_blocks.0.img_attn.qkv": (64, 192), "double_blocks.0.img_attn.proj": (64, 64),
                                        "single_blocks.1.linear2": (256, 64)}.items():
            tensors[f"base_model.model.{name}.lora_A.weight"] = torch.randn(4, inputs, generator=generator)
            tensors[f"base_model.model.{name}.lora_B.weight"] = torch.randn(outputs, 4, generator=generator)
        path = Path(directory) / "adapter.safetensors"
        save_file(tensors, str(path))
        return path

    def test_lora_matches_the_torch_adapter_and_detaches_cleanly(self) -> None:
        from app.models.klein_mlx import LoraLinear, _get

        model = self.load("bf16")
        before = self.run_model(model)
        with TemporaryDirectory() as directory:
            path = self.lora(directory)
            reference_adapter = KleinLora.attach(self.reference, path, lambda: None)
            adapter = KleinLora.attach(model, path, lambda: None)
        self.assertEqual(len(adapter.targets), 5)  # q, k, v, to_out.0 and the single block's to_out
        self.assertTrue(all(isinstance(target.layer, LoraLinear) for target in adapter.targets))
        self.assertGreater(adapter.storage_bytes, 0)
        torch.testing.assert_close(self.run_model(model), before)  # attached at scale zero
        adapter.set_scale(1.0)
        reference_adapter.set_scale(1.0)
        expected = self.run_model(self.reference)
        output = self.run_model(model)
        self.assertGreater(relative_error(output, before), 0.05)
        self.assertLess(relative_error(output, expected), 0.02)
        originals = [target.layer.base for target in adapter.targets]
        adapter.detach()
        self.assertTrue(all(_get(t.parent, t.leaf) is base for t, base in zip(adapter.targets, originals, strict=True)))
        torch.testing.assert_close(self.run_model(model), before)

    def test_stacked_loras_add_up_and_detach_in_any_order(self) -> None:
        from app.models.klein_mlx import _get

        model = self.load("bf16")
        before = self.run_model(model)
        originals = None
        with TemporaryDirectory() as directory:
            path = self.lora(directory)
            single = KleinLora.attach(model, path, lambda: None)
            single.set_scale(1.0)
            expected = self.run_model(model)
            originals = [target.layer.base for target in single.targets]
            single.detach()
            first = KleinLora.attach(model, path, lambda: None)
            second = KleinLora.attach(model, path, lambda: None)
        first.set_scale(0.5)
        second.set_scale(0.5)
        self.assertLess(relative_error(self.run_model(model), expected), 0.02)
        first.detach()
        second.set_scale(1.0)
        self.assertLess(relative_error(self.run_model(model), expected), 0.02)
        second.detach()
        self.assertTrue(all(_get(t.parent, t.leaf) is base for t, base in zip(second.targets, originals, strict=True)))
        torch.testing.assert_close(self.run_model(model), before)

    def test_lora_runs_on_quantized_weights_without_touching_them(self) -> None:
        import mlx.nn as nn

        model = self.load("q4")
        base = model.mlx.transformer_blocks[0].attn.to_q
        before = self.run_model(model)
        with TemporaryDirectory() as directory:
            path = self.lora(directory)
            reference_adapter = KleinLora.attach(self.reference, path, lambda: None)
            adapter = KleinLora.attach(model, path, lambda: None)
        adapter.set_scale(1.0)
        reference_adapter.set_scale(1.0)
        # The adapter wraps the quantized layer itself, without dequantizing it.
        self.assertIsInstance(base, nn.QuantizedLinear)
        self.assertTrue(any(target.layer.base is base for target in adapter.targets))
        delta = self.run_model(model) - before
        expected = self.run_model(self.reference) - self.run_model_without(reference_adapter)
        # The residual itself is computed in BF16, so it follows the torch adapter closely.
        self.assertLess(relative_error(delta, expected), 0.1)

    def run_model_without(self, adapter: KleinLora) -> torch.Tensor:
        adapter.set_scale(0)
        try:
            return self.run_model(self.reference)
        finally:
            adapter.set_scale(1.0)

    def test_invalid_lora_target_attaches_nothing(self) -> None:
        model = self.load("q8")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            save_file({"base_model.model.double_blocks.0.img_attn.proj.lora_A.weight": torch.ones(2, 64),
                       "base_model.model.double_blocks.0.img_attn.proj.lora_B.weight": torch.ones(64, 2),
                       "base_model.model.double_blocks.0.img_mlp.0.lora_A.weight": torch.ones(2, 32),
                       "base_model.model.double_blocks.0.img_mlp.0.lora_B.weight": torch.ones(384, 2)}, str(path))
            original = model.mlx.transformer_blocks[0].attn.to_out[0]
            with self.assertRaisesRegex(ValueError, "dimensions"):
                KleinLora.attach(model, path, lambda: None)
        self.assertIs(model.mlx.transformer_blocks[0].attn.to_out[0], original)


@unittest.skipUnless(MLX, "MLX runs on Apple silicon only")
class MlxTextEncoderTests(unittest.TestCase):
    def test_hidden_states_match_transformers_with_right_padding(self) -> None:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from app.models.klein_mlx import load_text_encoder

        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=28, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16, max_position_embeddings=64, rms_norm_eps=1e-6, rope_theta=1e6)
        reference = Qwen3ForCausalLM(config).eval()
        ids = torch.randint(0, 128, (1, 12))
        mask = torch.tensor([[1] * 9 + [0] * 3])
        with torch.inference_mode():
            expected = reference.model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            save_file({k: v.contiguous() for k, v in reference.state_dict().items()}, str(folder / "model.safetensors"))
            config.to_json_file(str(folder / "config.json"))
            encoders = {precision: load_text_encoder(folder, precision, lambda: None) for precision in ("bf16", "q8")}
        self.assertEqual(len(encoders["bf16"].mlx.layers), 27)  # hidden_states[27] needs 27 layers, no more
        for precision, limit in (("bf16", 0.02), ("q8", 0.03)):
            with torch.inference_mode():
                states = encoders[precision](input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
            for layer in (9, 18, 27):
                self.assertEqual(states[layer].dtype, torch.bfloat16)
                self.assertLess(relative_error(states[layer][0, :9], expected[layer][0, :9]), limit, (precision, layer))


class AppleWeightDefaultTests(unittest.TestCase):
    def test_weights_and_warm_up_follow_unified_memory(self) -> None:
        # Unified memory -> weights, warm-up model: BF16 from 64 GB, 8-bit from 48 GB, then 4-bit;
        # Quality warms up while its measured peak fits with room for macOS beside it.
        expected = {128: ("bf16", "flux2-klein-9b"), 64: ("bf16", "flux2-klein-9b"), 48: ("q8", "flux2-klein-9b"),
                    36: ("q4", "flux2-klein-9b"), 32: ("q4", "flux2-klein-4b"), 16: ("q4", "flux2-klein-4b")}
        for memory, (weights, warm_up) in expected.items():
            with patch("app.config.physical_memory_gb", return_value=memory):
                profile = Settings(_env_file=None, preferred_device="mps")
            self.assertEqual((profile.apple_weights, profile.warm_up_model), (weights, warm_up), memory)
        self.assertEqual(default_apple_weights(None), "bf16")

    def test_an_explicit_choice_wins_and_moves_the_warm_up(self) -> None:
        with patch("app.config.physical_memory_gb", return_value=48):
            forced = Settings(_env_file=None, preferred_device="mps", apple_weights="bf16")
            small = Settings(_env_file=None, preferred_device="mps", apple_weights="q4")
        self.assertEqual((forced.apple_weights, forced.warm_up_model), ("bf16", "flux2-klein-4b"))
        self.assertEqual((small.apple_weights, small.warm_up_model), ("q4", "flux2-klein-9b"))

    def test_the_model_name_says_which_weights_run(self) -> None:
        labels = {"bf16": "FLUX.2 Klein 9B", "q8": "FLUX.2 Klein 9B (8-bit)", "q4": "FLUX.2 Klein 9B (4-bit)"}
        for weights, label in labels.items():
            with patch.object(settings, "apple_weights", weights):
                self.assertEqual(Flux2Klein9bAppleProvider().info().display_name, label)
        with patch.object(settings, "apple_weights", "q4"):
            self.assertEqual(Flux2KleinAppleProvider().display_name, "FLUX.2 Klein 4B (4-bit)")


class MlxProviderTests(unittest.TestCase):
    def test_quantized_weights_load_both_large_components_in_mlx(self) -> None:
        provider = Flux2Klein9bAppleProvider()
        generation, inpaint = MagicMock(), MagicMock()
        generation.from_pretrained.return_value.config = {"is_distilled": True}
        vendor = SimpleNamespace(Flux2KleinPipeline=generation, Flux2KleinInpaintPipeline=inpaint)
        hub = SimpleNamespace(snapshot_download=MagicMock(return_value="snapshot"))
        mlx = SimpleNamespace(load_transformer=MagicMock(return_value="mlx transformer"),
                              load_text_encoder=MagicMock(return_value="mlx encoder"))
        control = MagicMock()
        with patch.dict("sys.modules", {"diffusers": vendor, "huggingface_hub": hub, "app.models.klein_mlx": mlx}), \
                patch.object(settings, "apple_weights", "q4"), patch("app.models.flux2_klein_apple.selected_dtype", return_value="bf16"):
            provider._create_pipeline("text-to-image", control)
        loaded = generation.from_pretrained.call_args
        self.assertEqual((loaded.args, loaded.kwargs["revision"]), ((BASE_REPO_9B,), BASE_REVISION_9B))
        self.assertIsNone(loaded.kwargs["transformer"])
        self.assertIsNone(loaded.kwargs["text_encoder"])
        mlx.load_transformer.assert_called_once_with(Path("snapshot/transformer"), "q4", control.raise_if_cancelled)
        mlx.load_text_encoder.assert_called_once_with(Path("snapshot/text_encoder"), "q4", control.raise_if_cancelled)
        built = generation.call_args.kwargs
        self.assertEqual((built["transformer"], built["text_encoder"], built["is_distilled"]), ("mlx transformer", "mlx encoder", True))

    def test_bf16_keeps_the_pytorch_pipeline(self) -> None:
        provider = Flux2KleinAppleProvider()
        generation = MagicMock()
        with patch.dict("sys.modules", {"diffusers": SimpleNamespace(Flux2KleinPipeline=generation, Flux2KleinInpaintPipeline=MagicMock())}), \
                patch.object(settings, "apple_weights", "bf16"), patch.object(provider, "_create_mlx_pipeline") as mlx, \
                patch("app.models.flux2_klein_apple.selected_dtype", return_value="bf16"):
            provider._create_pipeline("text-to-image", MagicMock())
        mlx.assert_not_called()
        self.assertNotIn("transformer", generation.from_pretrained.call_args.kwargs)

    def test_caches_are_released_after_a_job_on_a_mac_only(self) -> None:
        torch_stub = object()
        with patch.dict("sys.modules", {"torch": torch_stub}), patch("app.runtime.empty_cache") as empty:
            with patch.object(settings, "preferred_device", "cuda"):
                runtime.release_cached_memory()
            empty.assert_not_called()
            with patch.object(settings, "preferred_device", "mps"):
                runtime.release_cached_memory()
            empty.assert_called_once_with(torch_stub)

    def test_freed_mlx_buffers_are_returned_with_the_rest(self) -> None:
        torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                     backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
        mlx = SimpleNamespace(clear_cache=MagicMock())
        with patch.dict("sys.modules", {"mlx.core": mlx}):
            runtime.empty_cache(torch_stub)
        mlx.clear_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
