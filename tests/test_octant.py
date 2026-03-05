import os
import sys
import unittest

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.frame_processing import pre_process_frame
from src.glyph_tables import OCTANT_GLYPHS, OCTANT_GLYPH_SWAP


class OctantTests(unittest.TestCase):
    def test_octant_lookup_tables_cover_all_masks(self) -> None:
        self.assertEqual(len(OCTANT_GLYPHS), 256)
        self.assertEqual(len(OCTANT_GLYPH_SWAP), 256)
        self.assertEqual(OCTANT_GLYPHS[0], " ")
        self.assertEqual(OCTANT_GLYPHS[255], "█")
        self.assertTrue(any(OCTANT_GLYPH_SWAP))

    def test_octant_preprocess_produces_styles(self) -> None:
        frame = torch.tensor(
            [
                [[255, 0, 0], [0, 0, 0]],
                [[255, 0, 0], [0, 0, 0]],
                [[0, 0, 255], [0, 255, 0]],
                [[0, 0, 255], [0, 255, 0]],
            ],
            dtype=torch.uint8,
        )
        cfg = Config(
            width=2,
            height=4,
            device=torch.device("cpu"),
            render_mode="octant",
        )

        xs, ys, styles, updated = pre_process_frame(None, frame, cfg)

        self.assertEqual(xs.numel(), 1)
        self.assertEqual(ys.numel(), 1)
        self.assertEqual(styles.shape, (1, 7))
        self.assertEqual(updated.shape, (1, 1, 7))

    def test_octant_diffing_skips_identical_frames(self) -> None:
        frame = torch.zeros((4, 2, 3), dtype=torch.uint8)
        cfg = Config(
            width=2,
            height=4,
            device=torch.device("cpu"),
            render_mode="octant",
        )

        _, _, _, previous = pre_process_frame(None, frame, cfg)
        xs, ys, styles, updated = pre_process_frame(previous, frame, cfg)

        self.assertEqual(xs.numel(), 0)
        self.assertEqual(ys.numel(), 0)
        self.assertEqual(styles.shape, (0, 7))
        self.assertIs(updated, previous)


if __name__ == "__main__":
    unittest.main()
