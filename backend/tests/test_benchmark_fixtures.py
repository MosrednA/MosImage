from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from create_benchmark_fixtures import FIXTURES, Suite, digest


class FrozenBenchmarkFixtureTests(unittest.TestCase):
    def test_all_four_masks_match_the_frozen_sources_and_binary_exports(self) -> None:
        suite = Suite.model_validate_json((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual({fixture.name for fixture in suite.fixtures}, {"garden", "kitchen", "portrait", "architecture"})
        self.assertEqual(suite.seeds, (7, 42, 20261007))
        for fixture in suite.fixtures:
            with self.subTest(fixture=fixture.name):
                self.assertEqual(digest(FIXTURES / fixture.image), fixture.image_sha256)
                self.assertEqual(digest(FIXTURES / fixture.mask), fixture.mask_sha256)
                with Image.open(FIXTURES / fixture.image) as source, Image.open(FIXTURES / fixture.mask) as rgba, Image.open(FIXTURES / f"{fixture.name}.mask-binary.png") as binary:
                    self.assertEqual(source.size, (fixture.width, fixture.height))
                    self.assertEqual(rgba.size, source.size)
                    alpha = np.asarray(rgba.getchannel("A"))
                    self.assertTrue(np.array_equal(alpha, np.asarray(binary)))
                    self.assertEqual(set(np.unique(alpha)), {0, 255})
                    self.assertEqual(int(np.count_nonzero(alpha)), fixture.selected_pixels)
                    self.assertEqual(rgba.getchannel("A").getbbox(), fixture.mask_bbox)
                with Image.open(FIXTURES / "prepared" / fixture.name / "input.png") as source, Image.open(FIXTURES / "prepared" / fixture.name / "mask.png") as soft:
                    self.assertEqual(source.size, fixture.model_size)
                    self.assertEqual(soft.size, source.size)
                    self.assertLessEqual(max(source.size), suite.max_dimension)
                    self.assertTrue(all(size % 32 == 0 for size in source.size))
                    self.assertGreater(len(np.unique(np.asarray(soft))), 2)


if __name__ == "__main__":
    unittest.main()
