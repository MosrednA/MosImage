from __future__ import annotations

import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
