"""
Unit tests for the GNM Head v3 path (plan P2): evaluator vs the official gnm_numpy reference, fit on a
synthetic head, closed bust for printing, glTF writer with skin + morph targets.
Needs assets/gnm_head.npz (gitignored, fetched by gnm_head.ensure_model); skipped when it is missing.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from converters.draco_compressor import read_glb
from photogrammetry.gltf_writer import GLTFMaterial, GLTFPrimitive, GLTFSkeleton, write_glb
from photogrammetry.gnm_head import GNM_PATH, GNMHead, GNMParams

FIXTURE = Path(__file__).parent / "data" / "gnm_official_reference.npz"
HAVE_MODEL = Path(GNM_PATH).is_file()


@unittest.skipUnless(HAVE_MODEL, "assets/gnm_head.npz missing (python -c 'from photogrammetry.gnm_head import ensure_model; ensure_model()')")
class TestGNM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = GNMHead()

    def test_matches_official_reference(self):
        ref = np.load(FIXTURE)
        vid = ref["vertex_indices"]
        for k in range(3):
            p = GNMParams(ref[f"identity_{k}"], ref[f"expression_{k}"], ref[f"rotations_{k}"], ref[f"translation_{k}"])
            err = np.abs(self.g(p)[vid] - ref[f"vertices_{k}"]).max()
            self.assertLess(err, 1e-6, f"sample {k}: {err} m")

    def test_fit_synthetic_face(self):
        from photogrammetry.gnm_fit import GNMFitter
        rng = np.random.default_rng(1)
        p = self.g.zero_params()
        p.identity[:20] = rng.normal(0, 0.8, 20)
        p.scale = 1000.0
        V = self.g(p)
        F = self.g.triangles[self.g.skin_triangles]
        used = np.unique(F)
        front = used[V[used, 2] > np.percentile(V[used, 2], 60)]                    # the visible face only
        # sparse exact correspondences (as the MediaPipe embedding gives) start the rigid alignment
        lv = rng.choice(front, 40, replace=False)
        lm = (V[lv], np.repeat(lv[:, None], 3, axis=1), np.tile([1.0, 0.0, 0.0], (40, 1)), np.ones(40))
        res = GNMFitter(self.g).fit(V[front], landmarks=lm, iterations=8, expression=False)
        self.assertLess(res.stats["surface_rms_mm"], 2.0)

    def test_head_solid_watertight(self):
        from photogrammetry.gnm_assembly import head_solid
        p = self.g.zero_params()
        p.scale = 1000.0
        mesh = trimesh.Trimesh(self.g(p), self.g.triangles[self.g.skin_triangles], process=False)
        chin_y = np.percentile(mesh.vertices[:, 1], 30)
        solid = head_solid(mesh, chin_y)
        self.assertTrue(solid.is_watertight)
        self.assertGreater(solid.volume, 0)
        self.assertAlmostEqual(solid.bounds[0, 1], chin_y - 35.0, delta=1.0)


class TestGLTFWriter(unittest.TestCase):
    def test_skin_and_morph_targets(self):
        box = trimesh.creation.box((1, 1, 1))
        n = len(box.vertices)
        joints = np.zeros((n, 4), np.uint16)
        joints[:, 1] = 1
        weights = np.tile([0.7, 0.3, 0.0, 0.0], (n, 1))
        prim = GLTFPrimitive(box.vertices, box.faces, 0, normals=box.vertex_normals, uvs=box.vertices[:, :2] + 0.5,
                             joints=joints, weights=weights, morph_deltas=[np.full((n, 3), 0.1), -np.full((n, 3), 0.1)])
        tex = np.full((8, 8, 3), 128, np.uint8)
        skel = GLTFSkeleton(["neck", "head"], [-1, 0], np.array([[0, -1, 0], [0, 0, 0]], float))
        with tempfile.TemporaryDirectory() as d:
            path = write_glb(Path(d) / "a.glb", [prim], [GLTFMaterial("skin", base_color_png=tex)], skel, ["up", "down"])
            js, binc = read_glb(path)
        mesh = js["meshes"][0]
        self.assertEqual(mesh["extras"]["targetNames"], ["up", "down"])
        self.assertEqual(len(mesh["primitives"][0]["targets"]), 2)
        self.assertEqual(len(js["skins"]), 1)
        self.assertEqual(js["skins"][0]["joints"], [0, 1])
        for bv in js["bufferViews"]:
            self.assertLessEqual(bv["byteOffset"] + bv["byteLength"], len(binc))
        json.dumps(js)


if __name__ == "__main__":
    unittest.main()
