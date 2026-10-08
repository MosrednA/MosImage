from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline

from app.config import settings
from app.models.z_image import ZImageTurboProvider


class ZImagePlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = ZImageTurboProvider()

    def test_balanced_inpaint_uses_component_offload(self) -> None:
        with patch.object(settings, "enable_cpu_offload", True):
            self.assertEqual(self.provider._inpaint_placement_for("balanced"), "offload")

    def test_performance_inpaint_stays_gpu_resident(self) -> None:
        with patch.object(settings, "enable_cpu_offload", True):
            self.assertEqual(self.provider._inpaint_placement_for("performance"), "cuda")

    def test_disabled_offload_keeps_inpaint_gpu_resident(self) -> None:
        with patch.object(settings, "enable_cpu_offload", False):
            self.assertEqual(self.provider._inpaint_placement_for("safe"), "cuda")

    def test_task_switches_reuse_components_without_changing_bfloat16(self) -> None:
        class TinyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones((1, 1), dtype=torch.bfloat16))
                self.config = SimpleNamespace(block_out_channels=[1])

            @property
            def dtype(self) -> torch.dtype:
                return self.weight.dtype

            @property
            def device(self) -> torch.device:
                return self.weight.device

        original = ZImagePipeline(
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=TinyModule(),
            text_encoder=TinyModule(),
            tokenizer=None,
            transformer=TinyModule(),
        )
        self.provider._text_pipe = original
        self.provider._text_placement = "offload"
        with patch.object(settings, "max_loaded_pipelines", 1), patch.object(
            self.provider, "_place_pipeline"
        ), patch("app.models.z_image.selected_dtype", return_value=torch.bfloat16):
            inpaint = self.provider._load_inpaint_pipe("offload")
            for name in ("vae", "text_encoder", "transformer"):
                with self.subTest(task="inpaint", component=name):
                    self.assertIs(getattr(inpaint, name), getattr(original, name))
                    self.assertEqual(getattr(inpaint, name).dtype, torch.bfloat16)
            self.assertIsNone(self.provider._text_pipe)

            text = self.provider._load_text_pipe("offload")
            for name in ("vae", "text_encoder", "transformer"):
                with self.subTest(task="text-to-image", component=name):
                    self.assertIs(getattr(text, name), getattr(original, name))
                    self.assertEqual(getattr(text, name).dtype, torch.bfloat16)
            self.assertIsNone(self.provider._inpaint_pipe)


if __name__ == "__main__":
    unittest.main()
