"""
Unit tests for the stereo calibration from the disc tags of the face pair (plan P1.7).
Synthetic projections of the printed layout; the real-photo test runs only when the Turntable Studio
disc sweep is on this machine.
"""

import unittest
from pathlib import Path

import cv2
import numpy as np

from photogrammetry.pair_calibration import (
    MIN_TAGS_PER_CAMERA,
    calibrate_face_pair,
    calibrate_pair_from_tags,
    load_tag_layout_mm,
)
from photogrammetry.rig_calibration import load_rig_calibration

SWEEP = Path("/home/alin/Desktop/template_meshroom/turntable_studio/input_images/rig")


def _project(layout, R, t, K, D, ids, noise_px, rng):
    out = {}
    for i in ids:
        p, _ = cv2.projectPoints(layout[i], cv2.Rodrigues(R)[0], t, K, D)
        out[i] = p.reshape(4, 2) + rng.normal(0, noise_px, (4, 2))
    return out


class TestPairCalibration(unittest.TestCase):
    def setUp(self):
        self.base = load_rig_calibration(None)
        self.layout = load_tag_layout_mm(tag_scale=1.0)
        self.rng = np.random.default_rng(0)
        # Disc ~ 400 mm in front of camera 0, tilted like the rig (cameras look down at the disc)
        self.Rd = cv2.Rodrigues(np.array([np.radians(-110.0), 0.0, 0.0]))[0]
        self.td = np.array([0.0, 60.0, 420.0])

    def views(self, K0, K1, R, T, ids=range(12), noise=0.5):
        d0 = _project(self.layout, self.Rd, self.td, K0, self.base.D0, ids, noise, self.rng)
        d1 = _project(self.layout, R @ self.Rd, R @ self.td + T, K1, self.base.D1, ids, noise, self.rng)
        return d0, d1

    def test_recovers_a_small_change_of_the_rig(self):
        """Focal +0.8 % / -0.5 %, baseline +1.5 mm, rotation 0.3 deg: recovered from one pair."""
        K0, K1 = self.base.K0.copy(), self.base.K1.copy()
        K0[:2, :2] *= 1.008
        K1[:2, :2] *= 0.995
        dR = cv2.Rodrigues(np.radians([0.0, 0.3, 0.0]))[0]
        R = dR @ self.base.R
        T = self.base.T_mm * (self.base.baseline_mm + 1.5) / self.base.baseline_mm
        d0, d1 = self.views(K0, K1, R, T)
        res = calibrate_pair_from_tags(d0, d1, self.base, self.layout)
        self.assertTrue(res.accepted, res.reason)
        c = res.calibration
        self.assertAlmostEqual(c.K0[0, 0] / K0[0, 0], 1.0, delta=0.004)
        self.assertAlmostEqual(c.K1[0, 0] / K1[0, 0], 1.0, delta=0.004)
        self.assertAlmostEqual(c.baseline_mm, np.linalg.norm(T), delta=0.6)
        self.assertLess(np.degrees(np.linalg.norm(cv2.Rodrigues(c.R @ R.T)[0])), 0.1)

    def test_rejects_a_changed_lens(self):
        K0 = self.base.K0.copy()
        K0[:2, :2] *= 1.06                         # lens rotated: focal length +6 %
        d0, d1 = self.views(K0, self.base.K1, self.base.R, self.base.T_mm, noise=0.2)
        res = calibrate_pair_from_tags(d0, d1, self.base, self.layout)
        self.assertFalse(res.accepted)
        self.assertIn("focal", res.reason)

    def test_too_few_tags(self):
        d0, d1 = self.views(self.base.K0, self.base.K1, self.base.R, self.base.T_mm, ids=range(MIN_TAGS_PER_CAMERA - 1))
        res = calibrate_pair_from_tags(d0, d1, self.base, self.layout)
        self.assertFalse(res.accepted)
        self.assertIn("not enough disc tags", res.reason)

    @unittest.skipUnless((SWEEP / "0" / "f_1005.jpg").is_file(), "Turntable Studio disc photos not on this machine")
    def test_real_pair_from_the_disc_sweep(self):
        """One pair of the 40-image calibration set, arbitrary turntable angle: within 1.5 mm / 1.5 % / 0.3 deg."""
        res = calibrate_face_pair(SWEEP / "0" / "f_1005.jpg", SWEEP / "1" / "f_1005.jpg", self.base, tag_scale=1.0)
        self.assertTrue(res.accepted, res.reason)
        self.assertLess(abs(res.stats["baseline_change_mm"]), 1.5)
        self.assertLess(max(abs(v) for v in res.stats["focal_change"]), 0.015)
        self.assertLess(res.stats["rotation_change_deg"], 0.3)


if __name__ == "__main__":
    unittest.main()
