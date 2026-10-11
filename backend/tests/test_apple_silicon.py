from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import torch

from app import runtime
from app.config import Settings, settings
from app.models.flux2_klein_apple import Flux2Klein9bAppleProvider, Flux2KleinAppleProvider
from app.models.flux2_klein_nvfp4 import BASE_REPO, BASE_REPO_9B, BASE_REVISION, BASE_REVISION_9B
from app.models.klein_text_encoder import HIDDEN_STATE_LAYERS, KEEP_LAYERS, load_resident_text_encoder
from app.models.registry import ModelRegistry
from app.schemas import EditRequest, EnhanceRequest, InpaintForm, OutpaintForm, TextToImageRequest


def apple_profile():
    return patch.object(settings, "preferred_device", "mps")


def fake_torch(*, cuda: bool = False, mps: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, empty_cache=MagicMock()),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        mps=SimpleNamespace(empty_cache=MagicMock()),
        bfloat16="bf16", float16="fp16", float32="fp32",
    )


class AppleProfileTests(unittest.TestCase):
    @patch("app.config.physical_memory_gb", return_value=128)
    def test_each_profile_names_its_quality_and_fast_models(self, _memory: MagicMock) -> None:
        apple = Settings(_env_file=None, preferred_device="mps")
        self.assertEqual((apple.quality_model, apple.fast_model, apple.warm_up_model),
                         ("flux2-klein-9b", "flux2-klein-4b", "flux2-klein-9b"))
        cuda = Settings(_env_file=None, preferred_device="cuda")
        self.assertEqual((cuda.quality_model, cuda.fast_model, cuda.warm_up_model),
                         ("flux2-klein-9b-nvfp4", "flux2-klein-4b-nvfp4", "flux2-klein-9b-nvfp4"))
        self.assertEqual(Settings(_env_file=None, preferred_device="mps", warm_up_model="flux2-klein-4b").warm_up_model,
                         "flux2-klein-4b")

    def test_a_mac_without_room_for_quality_warms_up_fast(self) -> None:
        with patch("app.config.physical_memory_gb", return_value=24):
            self.assertEqual(Settings(_env_file=None, preferred_device="mps").warm_up_model, "flux2-klein-4b")
            self.assertEqual(Settings(_env_file=None, preferred_device="cuda").warm_up_model, "flux2-klein-9b-nvfp4")
        with patch("app.config.physical_memory_gb", return_value=None):
            self.assertEqual(Settings(_env_file=None, preferred_device="mps").warm_up_model, "flux2-klein-9b")

    def test_a_mac_without_room_for_quality_tiles_the_vae(self) -> None:
        # An M4 MacBook Air with 16 GB runs 4-bit Fast; untiled, one decode would need 6.5 GiB more.
        for memory, tiling in ((16, True), (32, True), (36, False), (128, False)):
            with patch("app.config.physical_memory_gb", return_value=memory):
                self.assertEqual(Settings(_env_file=None, preferred_device="mps").apple_vae_tiling, tiling, memory)
        with patch("app.config.physical_memory_gb", return_value=16):
            self.assertFalse(Settings(_env_file=None, preferred_device="mps", apple_vae_tiling=False).apple_vae_tiling)
            self.assertFalse(Settings(_env_file=None, preferred_device="cuda").apple_vae_tiling)

    @unittest.skipUnless(sys.platform == "darwin", "macOS only")
    def test_metal_fallback_is_set_before_torch_loads(self) -> None:
        # torch reads it at import, and SAM can import torch before runtime.torch_module() runs.
        self.assertEqual(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), "1")

    def test_registry_offers_the_bf16_models_on_a_mac(self) -> None:
        with apple_profile():
            models = {info.name: info for info in ModelRegistry().list()}
        self.assertEqual(set(models), {"flux2-klein-9b", "flux2-klein-4b"})
        self.assertEqual((models["flux2-klein-9b"].preset, models["flux2-klein-4b"].preset), ("quality", "fast"))
        self.assertEqual(models["flux2-klein-9b"].model_id, BASE_REPO_9B)

    def test_registry_keeps_nvfp4_on_cuda(self) -> None:
        models = {info.name: info.preset for info in ModelRegistry().list()}
        self.assertEqual(models, {"flux2-klein-9b-nvfp4": "quality", "flux2-klein-4b-nvfp4": "fast", "flux2-klein-4b": None})

    def test_requests_default_to_the_quality_model_and_gate_its_options(self) -> None:
        with apple_profile():
            self.assertEqual(TextToImageRequest(prompt="scene").model_name, "flux2-klein-9b")
            self.assertTrue(EditRequest(prompt="x", model_name="flux2-klein-9b", consistency_lora=True).consistency_lora)
            self.assertEqual(EnhanceRequest(model_name="flux2-klein-9b").detail, 0.4)
            self.assertTrue(InpaintForm(prompt="x", model_name="flux2-klein-9b", consistency_lora=True).consistency_lora)
            self.assertTrue(OutpaintForm(prompt="x", model_name="flux2-klein-4b", left=8, specialized_outpaint=True).specialized_outpaint)
            with self.assertRaisesRegex(ValueError, "Quality"):
                EditRequest(prompt="x", model_name="flux2-klein-4b", consistency_lora=True)
            with self.assertRaisesRegex(ValueError, "Fast model"):
                OutpaintForm(prompt="x", model_name="flux2-klein-9b", left=8, specialized_outpaint=True)


class AppleRuntimeTests(unittest.TestCase):
    def test_metal_is_selected_with_bfloat16(self) -> None:
        torch_stub = fake_torch(mps=True)
        with apple_profile():
            self.assertEqual(runtime.selected_device(torch_stub), "mps")
            self.assertEqual(runtime.selected_dtype(torch_stub), "bf16")
        self.assertEqual(runtime.selected_device(torch_stub), "cpu")

    def test_generation_stops_with_a_mac_hint_without_metal(self) -> None:
        with apple_profile(), patch.object(runtime, "torch_module", return_value=fake_torch()), \
                patch.object(settings, "allow_cpu_generation", False):
            with self.assertRaisesRegex(RuntimeError, "Metal"):
                runtime.require_generation_device()

    def test_freed_metal_memory_is_returned(self) -> None:
        torch_stub = fake_torch(mps=True)
        runtime.empty_cache(torch_stub)
        torch_stub.mps.empty_cache.assert_called_once_with()
        torch_stub.cuda.empty_cache.assert_not_called()

    def test_system_info_names_the_chip(self) -> None:
        with apple_profile(), patch.object(runtime, "torch_module", return_value=SimpleNamespace(**vars(fake_torch(mps=True)), __version__="2")), \
                patch.object(runtime, "_apple_chip", return_value=("Apple M5 Max", 128)):
            info = runtime.system_info()
        self.assertEqual((info["selected_device"], info["device_name"], info["unified_memory_gb"]), ("mps", "Apple M5 Max", 128))
        self.assertTrue(info["mps_available"])
        self.assertNotIn("setup_hint", info)

    def test_system_info_names_the_model_the_editor_starts_on(self) -> None:
        with apple_profile(), patch.object(settings, "warm_up_model", "flux2-klein-4b"), patch.object(settings, "apple_vae_tiling", True), \
                patch.object(runtime, "torch_module", return_value=SimpleNamespace(**vars(fake_torch(mps=True)), __version__="2")), \
                patch.object(runtime, "_apple_chip", return_value=("Apple M4", 16)):
            info = runtime.system_info()
        self.assertEqual((info["warm_up_model"], info["fast_model"], info["apple_vae_tiling"]), ("flux2-klein-4b", "flux2-klein-4b", True))


class AppleProviderTests(unittest.TestCase):
    def test_pipelines_stay_resident_in_unified_memory(self) -> None:
        provider = Flux2KleinAppleProvider()
        pipe = MagicMock()
        with apple_profile(), patch.object(settings, "enable_cpu_offload", True), \
                patch("app.models.flux2_klein.selected_device", return_value="mps"), patch.object(provider, "_load_torch"):
            self.assertEqual({provider._placement_for(mode) for mode in ("safe", "balanced", "performance")}, {"mps"})
            provider._place_pipeline(pipe, "mps")
        pipe.to.assert_called_once_with("mps")
        pipe.enable_model_cpu_offload.assert_not_called()

    def test_a_small_mac_decodes_in_512_px_tiles(self) -> None:
        from diffusers import AutoencoderKLFlux2

        def tiny_vae() -> AutoencoderKLFlux2:
            # Klein's VAE layout (four blocks, 8x scale, 1024 px tiles) with few channels.
            return AutoencoderKLFlux2(block_out_channels=(8, 8, 8, 8), layers_per_block=1, norm_num_groups=4, latent_channels=4).eval()

        latents = torch.zeros(1, 4, 72, 72)  # 576 px: one tile at 1024 px, four at 512 px
        for tiling, sizes, tiled in ((True, (512, 64), True), (False, (1024, 128), False)):
            vae = tiny_vae()
            with patch.object(settings, "apple_vae_tiling", tiling):
                Flux2KleinAppleProvider()._configure_memory(SimpleNamespace(vae=vae))
            self.assertEqual((vae.tile_sample_min_size, vae.tile_latent_min_size), sizes)
            with patch.object(vae, "tiled_decode", wraps=vae.tiled_decode) as tiled_decode, torch.inference_mode():
                self.assertEqual(vae.decode(latents).sample.shape, (1, 3, 576, 576))
            self.assertEqual(tiled_decode.called, tiled)

    def test_9b_loads_pinned_bf16_weights_with_the_trimmed_encoder(self) -> None:
        provider = Flux2Klein9bAppleProvider()
        generation, inpaint, encoder = MagicMock(), MagicMock(), MagicMock()
        inpaint.from_pretrained.return_value.config = {"is_distilled": True}
        vendor = SimpleNamespace(Flux2KleinPipeline=generation, Flux2KleinInpaintPipeline=inpaint)
        hub = SimpleNamespace(snapshot_download=MagicMock(return_value="snapshot"))
        loader = SimpleNamespace(load_resident_text_encoder=MagicMock(return_value=encoder))
        control = MagicMock()
        modules = {"diffusers": vendor, "huggingface_hub": hub, "app.models.klein_text_encoder": loader}
        with patch.dict("sys.modules", modules), patch("app.models.flux2_klein_apple.selected_dtype", return_value="bf16"), \
                patch("app.models.flux2_klein_apple.selected_device", return_value="mps"):
            provider._create_pipeline("inpaint", control)
        loaded = inpaint.from_pretrained.call_args
        self.assertEqual((loaded.args, loaded.kwargs["revision"], loaded.kwargs["torch_dtype"]), ((BASE_REPO_9B,), BASE_REVISION_9B, "bf16"))
        self.assertIsNone(loaded.kwargs["text_encoder"])
        loader.load_resident_text_encoder.assert_called_once_with(Path("snapshot"), "mps", control.raise_if_cancelled)
        built = inpaint.call_args.kwargs
        self.assertIs(built["text_encoder"], encoder)
        self.assertTrue(built["is_distilled"])

    def test_4b_loads_its_pinned_bf16_pipeline(self) -> None:
        provider = Flux2KleinAppleProvider()
        generation = MagicMock()
        with patch.dict("sys.modules", {"diffusers": SimpleNamespace(Flux2KleinPipeline=generation, Flux2KleinInpaintPipeline=MagicMock())}), \
                patch("app.models.flux2_klein_apple.selected_dtype", return_value="bf16"):
            provider._create_pipeline("text-to-image", MagicMock())
        loaded = generation.from_pretrained.call_args
        self.assertEqual((loaded.args, loaded.kwargs["revision"]), ((BASE_REPO,), BASE_REVISION))

    def test_only_9b_takes_the_edit_adapters(self) -> None:
        with self.assertRaisesRegex(ValueError, "Quality 9B"):
            Flux2KleinAppleProvider()._configure_edit_adapters(MagicMock(), True, MagicMock())
        provider = Flux2Klein9bAppleProvider()
        adapter = MagicMock()
        with patch("app.models.klein_lora.resolve_checkpoint", return_value=Path("consistency.safetensors")), \
                patch("app.models.klein_lora.KleinLora.attach", return_value=adapter), patch.object(provider, "_collect"):
            provider._configure_edit_adapters(MagicMock(), True, MagicMock())
        adapter.set_scale.assert_called_once_with(0.6)

    def test_specialized_outpaint_attaches_its_lora_once_and_runs_on_metal(self) -> None:
        provider = Flux2KleinAppleProvider()
        pipe = MagicMock(return_value=SimpleNamespace(images=[Image.new("RGB", (64, 64))]))
        adapter = MagicMock()
        torch_stub = MagicMock()
        with TemporaryDirectory() as directory, apple_profile(), \
                patch("app.models.flux2_klein_apple.require_generation_device"), \
                patch("app.models.flux2_klein_apple.selected_device", return_value="mps"), \
                patch("app.models.klein_lora.resolve_checkpoint", return_value=Path("outpaint.safetensors")), \
                patch("app.models.klein_lora.KleinLora.attach", return_value=adapter) as attach, \
                patch.object(provider, "_load_pipe", return_value=pipe), patch.object(provider, "_load_torch", return_value=torch_stub), \
                patch.object(provider, "_encode_prompt", return_value=MagicMock()):
            root = Path(directory)
            Image.new("RGB", (96, 64)).save(root / "input.png")
            form = OutpaintForm(prompt="continue", left=32, specialized_outpaint=True, model_name="flux2-klein-4b", seed=7)
            provider.outpaint(form, root / "input.png", root / "output.png", MagicMock())
            provider.outpaint(form, root / "input.png", root / "output.png", MagicMock())
            with Image.open(root / "output.png") as output:
                self.assertEqual(output.size, (96, 64))
        attach.assert_called_once()
        adapter.set_scale.assert_called_with(1.1)
        torch_stub.Generator.assert_called_with("mps")
        with apple_profile(), self.assertRaisesRegex(ValueError, "Fast model"):
            Flux2Klein9bAppleProvider().outpaint(form, Path("in.png"), Path("out.png"), MagicMock())


class ParentWatchTests(unittest.TestCase):
    def test_engine_stops_once_the_app_that_started_it_is_gone(self) -> None:
        from app import main

        with patch.object(main.os, "getppid", side_effect=[41, 41, 1]), patch.object(main.os, "getpid", return_value=99), \
                patch.object(main.os, "kill") as kill, patch.object(main.os, "_exit") as hard_exit, patch.object(main.time, "sleep") as sleep:
            main._exit_with_parent(41)
        kill.assert_called_once_with(99, main.signal.SIGINT)
        hard_exit.assert_called_once_with(0)
        self.assertEqual(sleep.call_count, 3)

    def test_engine_stops_once_start_sh_is_gone_behind_its_supervisor(self) -> None:
        from app import main

        with patch.object(main.os, "getppid", return_value=41), patch.object(main, "_running", side_effect=[True, False]) as running, \
                patch.object(main.os, "getpid", return_value=99), patch.object(main.os, "kill") as kill, \
                patch.object(main.os, "_exit") as hard_exit, patch.object(main.time, "sleep"):
            main._exit_with_parent(41, watched=7)
        self.assertEqual(running.call_args_list[-1].args, (7,))
        kill.assert_called_once_with(99, main.signal.SIGINT)
        hard_exit.assert_called_once_with(0)

    def test_a_vanished_process_is_not_running(self) -> None:
        from app import main

        with patch.object(main.os, "kill", side_effect=ProcessLookupError):
            self.assertFalse(main._running(7))
        with patch.object(main.os, "kill", side_effect=PermissionError):
            self.assertTrue(main._running(7))


class ResidentTextEncoderTests(unittest.TestCase):
    def test_trimmed_encoder_returns_the_hidden_states_klein_reads(self) -> None:
        from safetensors.torch import save_file
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=KEEP_LAYERS + 3,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64)
        full = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
        # As loaded for real: BF16 weights, with the rotary table kept in float32.
        full.model.rotary_emb = type(full.model.rotary_emb)(config=config)
        with TemporaryDirectory() as directory:
            folder = Path(directory) / "text_encoder"
            folder.mkdir()
            config.save_pretrained(folder)
            weights = {key: tensor.contiguous() for key, tensor in full.state_dict().items()}
            save_file(weights, str(folder / "model.safetensors"))
            (folder / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {key: "model.safetensors" for key in weights}}), encoding="utf-8")
            trimmed = load_resident_text_encoder(Path(directory), "cpu", lambda: None)
        self.assertEqual(len(trimmed.layers), KEEP_LAYERS)
        tokens = torch.tensor([[1, 5, 9, 3, 7]])
        with torch.inference_mode():
            expected = full.model(input_ids=tokens, output_hidden_states=True).hidden_states
            actual = trimmed(input_ids=tokens, output_hidden_states=True).hidden_states
        for layer in HIDDEN_STATE_LAYERS:
            torch.testing.assert_close(actual[layer], expected[layer])


if __name__ == "__main__":
    unittest.main()
