"""
Unit tests for the texture fixes (plan P0.5-P0.7): slope-adaptive visibility, local colour gain,
harmonic fill over the mesh. Synthetic data only.
"""

import unittest

import numpy as np
import trimesh

from photogrammetry.face_texture import _delta_e, _depth_slope, _local_gain, mesh_harmonic_fill


class TestHarmonicFill(unittest.TestCase):
    def test_linear_between_fixed_ends_and_across_uv_seams(self):
        bar = trimesh.creation.box((10.0, 1.0, 1.0)).subdivide().subdivide().subdivide()
        # Split every vertex per face (worst-case UV seams): the fill must still see one connected surface
        split = trimesh.Trimesh(bar.vertices[bar.faces].reshape(-1, 3),
                                np.arange(3 * len(bar.faces)).reshape(-1, 3), process=False)
        x = split.triangles_center[:, 0]
        vals = np.zeros((len(x), 3))
        vals[x < -4.5] = 200.0
        known = (x < -4.5) | (x > 4.5)
        out = mesh_harmonic_fill(split, np.arange(len(x)), np.full((len(x), 3), 1 / 3), vals, known)
        for pos, expect in ((-2.0, 140.0), (0.0, 100.0), (2.0, 60.0)):
            sel = np.abs(x - pos) < 0.8
            self.assertAlmostEqual(out[sel, 0].mean(), expect, delta=12.0)
        self.assertTrue(np.all(np.diff([out[np.abs(x - p) < 0.8, 0].mean() for p in (-3, -1, 1, 3)]) < 0))

    def test_component_without_known_values_gets_the_mean(self):
        a = trimesh.creation.box((1, 1, 1))
        b = trimesh.creation.box((1, 1, 1))
        b.apply_translation([5, 0, 0])
        m = trimesh.util.concatenate([a, b])
        known = m.triangles_center[:, 0] < 2
        vals = np.where(known[:, None], 50.0, 0.0) * np.ones((1, 3))
        out = mesh_harmonic_fill(m, np.arange(len(m.faces)), np.full((len(m.faces), 3), 1 / 3), vals, known)
        np.testing.assert_allclose(out[~known], 50.0, atol=1e-3)


class TestVisibilityAndColour(unittest.TestCase):
    def test_depth_slope(self):
        d = np.tile(np.arange(10, dtype=np.float64) * 2.0, (5, 1))
        d[:, 7:] = np.inf
        s = _depth_slope(d)
        self.assertAlmostEqual(s[2, 3], 2.0)
        self.assertEqual(s[2, 8], 0.0)                     # undefined depth: no slope

    def test_local_gain_follows_a_spatial_colour_drift(self):
        rng = np.random.default_rng(0)
        pos = np.c_[rng.uniform(0, 150, 20000), rng.uniform(0, 150, 20000), np.zeros(20000)]
        true_gain = 1.0 + 0.1 * (pos[:, 0] / 150.0)       # 1.0 on the left, 1.1 on the right
        low1 = rng.uniform(80, 200, (20000, 3))
        low0 = low1 * true_gain[:, None]
        both = np.ones(20000, bool)
        g = _local_gain(pos, low0, low1, both, sigma_mm=15.0, global_ratio=np.median(low0 / low1, axis=0))
        inner = (pos[:, 0] > 30) & (pos[:, 0] < 120)
        self.assertLess(np.abs(g[inner, 0] - true_gain[inner]).max(), 0.01)
        self.assertLess(np.median(_delta_e(low0 / np.sqrt(g), low1 * np.sqrt(g))), 0.5)


if __name__ == "__main__":
    unittest.main()
