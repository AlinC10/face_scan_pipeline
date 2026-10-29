"""
Unit tests for the segmented face region and the spectacle-frame removal (plan P1.8-P1.9).
"""

import unittest
from pathlib import Path

import cv2
import numpy as np

from photogrammetry.face_correspondence import _extrapolate_quadratic
from photogrammetry.face_segmentation import SEGMENTER_PATH, _complete_frames, inpaint_frames, segment_faces

PAIR = Path(__file__).resolve().parent.parent / "input_images" / "rig"


class TestHelpers(unittest.TestCase):
    def test_quadratic_extrapolation_beyond_the_mesh(self):
        h, w = 300, 400
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        true = 500.0 - 0.004 * (xx - 200) ** 2 - 0.002 * (yy - 150) ** 2      # convex, like a face
        known = (xx - 200) ** 2 / 80 ** 2 + (yy - 150) ** 2 / 70 ** 2 < 1.0
        target = ~known & ((xx - 200) ** 2 / 150 ** 2 + (yy - 150) ** 2 / 130 ** 2 < 1.0)
        out = _extrapolate_quadratic(true, known, np.zeros_like(true), np.zeros_like(true), target)
        self.assertLess(np.abs(out[target] - true[target]).max(), 1e-6)

    def test_frames_completed_but_not_the_eyebrows(self):
        img = np.full((800, 800, 3), 180, np.uint8)
        cv2.rectangle(img, (200, 300), (380, 420), (40, 40, 40), 6)      # frame
        cv2.line(img, (200, 220), (380, 230), (40, 40, 40), 8)          # eyebrow: thin and dark too
        face = np.zeros((800, 800), bool)
        face[100:700, 100:700] = True
        seg = np.zeros((800, 800), bool)
        seg[296:304, 200:300] = True                                     # the segmenter found only a piece
        out = _complete_frames(img, seg, face)
        self.assertTrue(out[300:420, 378:383].any())                     # far side of the frame added
        self.assertFalse(out[215:235, 250:330].any())                    # eyebrow kept

    def test_inpaint_removes_a_dark_frame(self):
        img = np.full((400, 400, 3), (150, 170, 200), np.uint8)
        mask = np.zeros((400, 400), bool)
        cv2.rectangle(img, (100, 100), (300, 250), (30, 30, 30), 5)
        cv2.rectangle(mask.view(np.uint8), (100, 100), (300, 250), 1, 11)
        out = inpaint_frames(img, mask.astype(bool))
        self.assertLess(np.abs(out[mask].astype(int) - [150, 170, 200]).max(), 12)


@unittest.skipUnless(SEGMENTER_PATH.is_file() and (PAIR / "0" / "f_150.jpg").is_file(), "segmenter or test pair missing")
class TestSegmenterOnTheTestPair(unittest.TestCase):
    def test_face_reaches_the_hairline_and_finds_the_frames(self):
        imgs = [cv2.imread(str(PAIR / c / "f_150.jpg")) for c in "01"]
        segs = segment_faces(imgs)
        for sg in segs:
            self.assertGreater(sg.face_fraction, 0.10)
            self.assertGreater(sg.frames.mean(), 0.003)                  # the test subject wears glasses
            # The hair touches the face region from above (the hairline is the border)
            border = cv2.dilate(sg.face.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~sg.face
            self.assertGreater((border & sg.hair).sum(), 0.1 * border.sum())


if __name__ == "__main__":
    unittest.main()
