"""
Unit tests for the export scale, the tag-disc print scale and the Draco compression (plan P0.1-P0.4).
No network, no GPU, no photographs.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from converters import convert_to_glb_draco
from converters.draco_compressor import DRACO_EXT, read_glb, write_glb
from photogrammetry.face_pipeline import export_face_deliverables, to_print_frame
from photogrammetry.rig_calibration import load_rig_calibration
from tools.check_deliverables import check, glb_vertices


def _uv_grid(nx=30, ny=40, step=5.0):
    """A curved height-field patch in mm (head frame: x, y up, z towards the viewer) with UVs."""
    xs, ys = np.meshgrid(np.arange(nx) * step, np.arange(ny) * step)
    zs = 20.0 * np.exp(-((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2) / (2 * 40.0 ** 2))
    v = np.stack([xs.ravel() - xs.mean(), ys.ravel() - ys.mean(), zs.ravel()], axis=1)
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    f = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3)])
    m = trimesh.Trimesh(v, f, process=False)
    uv = np.stack([xs.ravel() / xs.max(), ys.ravel() / ys.max()], axis=1)
    m.visual = trimesh.visual.TextureVisuals(uv=uv)
    return m


class TestTagScale(unittest.TestCase):
    def test_baseline_rescaled_once(self):
        calib = load_rig_calibration(None)
        self.assertEqual(calib.tag_scale, 1.0)
        c98 = calib.with_tag_scale(0.98)
        self.assertAlmostEqual(c98.baseline_mm, calib.baseline_mm * 0.98, places=6)
        self.assertEqual(c98.tag_scale, 0.98)
        # Already at 0.98: asking for 0.98 again changes nothing (no double correction)
        self.assertAlmostEqual(c98.with_tag_scale(0.98).baseline_mm, c98.baseline_mm, places=9)
        np.testing.assert_allclose(c98.R, calib.R)
        np.testing.assert_allclose(c98.K0, calib.K0)
        self.assertEqual(c98.scaled(0.5).tag_scale, 0.98)

    def test_turntable_calib_records_its_tag_scale(self):
        own = load_rig_calibration(None)
        rvec0, rvec1 = np.zeros(3), np.array([0.0, np.radians(-19.7), 0.0])
        data = {"cams": ["0", "1"], "tag_scale": 0.98, "params": {
            "K_0": [2547.2, 2028.0, 1520.0], "K_1": [2552.4, 2028.0, 1520.0],
            "D_0": [-0.03, 0.0, 0.0], "D_1": [-0.03, 0.0, 0.0],
            "R_0": rvec0.tolist(), "R_1": rvec1.tolist(), "C_0": [0.0, 0.0, 0.0], "C_1": [0.161, 0.0, 0.0]}}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "turntable_calib.json"
            p.write_text(json.dumps(data))
            c = load_rig_calibration(p)
        self.assertEqual(c.tag_scale, 0.98)
        self.assertAlmostEqual(c.with_tag_scale(0.98).baseline_mm, 161.0, places=3)
        self.assertNotEqual(own.tag_scale, c.tag_scale)


class TestDraco(unittest.TestCase):
    def textured(self):
        m = _uv_grid()
        tex = Image.fromarray(np.full((64, 64, 3), 128, np.uint8))
        m.visual = trimesh.visual.TextureVisuals(uv=m.visual.uv, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex))
        return m

    def test_compressed_glb_keeps_geometry_and_is_smaller(self):
        m = self.textured()
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.glb"
            comp = Path(tmp) / "comp.glb"
            convert_to_glb_draco(m, raw, draco=False)
            convert_to_glb_draco(m, comp, draco=True)
            js, _ = read_glb(comp)
            self.assertIn(DRACO_EXT, js["extensionsUsed"])
            self.assertIn(DRACO_EXT, js["extensionsRequired"])
            self.assertLess(comp.stat().st_size, raw.stat().st_size)
            v, info = glb_vertices(comp)
            self.assertTrue(info["draco"])
            np.testing.assert_allclose(np.sort(v.max(0) - v.min(0)), np.sort(m.extents), rtol=1e-3)

    def test_animated_primitives_are_left_uncompressed(self):
        """Morph targets / skins keep their vertex order: Draco (which reorders) must skip them."""
        m = self.textured()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "anim.glb"
            convert_to_glb_draco(m, path, draco=False)
            js, binc = read_glb(path)
            prim = js["meshes"][0]["primitives"][0]
            prim["targets"] = [{"POSITION": prim["attributes"]["POSITION"]}]
            write_glb(path, js, binc)
            from converters.draco_compressor import draco_compress_glb
            self.assertIsNone(draco_compress_glb(path))
            js2, _ = read_glb(path)
            self.assertNotIn(DRACO_EXT, js2.get("extensionsUsed", []))


class TestExportUnits(unittest.TestCase):
    def test_every_format_in_the_same_units(self):
        high = _uv_grid()
        low = _uv_grid(15, 20, 10.0)
        tex = np.full((256, 256, 3), 120, np.uint8)
        nrm = np.full((256, 256, 3), (255, 128, 128), np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            export_face_deliverables(out, "model", high, low, tex, tex, nrm, nrm, export_scale=100.0)
            ext = high.extents
            (out / "object.json").write_text(json.dumps({
                "dimensions_mm": {"width": ext[0], "height": ext[1], "depth": ext[2]},
                "scale": {"export_scale": 100.0}}))
            rep = check(out)
            failed = [c for c in rep["checks"] if not c["ok"]]
            self.assertEqual(failed, [])
            obj = trimesh.load(out / "obj" / "high" / "high_model.obj", force="mesh", process=False)
            np.testing.assert_allclose(obj.extents, high.extents / 10.0, rtol=1e-6)     # mm -> cm
            stl = trimesh.load(out / "stl" / "high_model.stl")
            self.assertAlmostEqual(stl.extents[2], high.extents[1] / 10.0, places=4)    # height along Z

    def test_print_frame(self):
        m = trimesh.creation.box((1.0, 3.0, 2.0))
        p = to_print_frame(m)
        np.testing.assert_allclose(p.extents, [1.0, 2.0, 3.0], atol=1e-9)              # Y (height) -> Z
        self.assertAlmostEqual(p.bounds[0, 2], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
