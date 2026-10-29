"""
Unit Tests for the Dense Two-View Face Reconstruction
=====================================================
Tests:
  1. Rig calibration: own JSON and turntable_calib.json give the same stereo geometry.
  2. Relative-pose refinement recovers a perturbed rotation from synthetic correspondences.
  3. Barycentric / z-buffered rasterisation.
  4. Height-field solidification is watertight.
  5. Texture baking of a textured plane seen by two synthetic cameras.
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import trimesh

from photogrammetry.face_correspondence import rasterize_triangles
from photogrammetry.face_surface import solidify_height_field
from photogrammetry.face_texture import TextureCamera, bake_albedo, rasterize_mesh, unwrap_uv
from photogrammetry.rig_calibration import (
    RigCalibration,
    epipolar_errors_px,
    load_rig_calibration,
    refine_relative_pose,
)

TURNTABLE_CALIB = {
    "cams": ["0", "1"],
    "params": {
        "R_0": [-1.037193487755534, -0.3441942477973438, 2.8495504971369985],
        "C_0": [0.24516080723993075, 0.13907883381388145, -0.24699474905018348],
        "K_0": [2547.203497719371, 2028.0, 1520.0],
        "D_0": [-0.03444816844687779, 0.0, 0.0],
        "R_1": [-0.5440951796805619, -0.39270285260433346, 3.02263900643061],
        "C_1": [0.10577506849874811, 0.14035518653591397, -0.3339513519083231],
        "K_1": [2552.4415838483274, 2028.0, 1520.0],
        "D_1": [0.027184267193408992, 0.0, 0.0],
    },
    "intrinsics": {"0": {"width": 4056, "height": 3040}},
}


def _synthetic_pairs(calib: RigCalibration, n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.c_[rng.uniform(-80, 80, n), rng.uniform(-90, 90, n), rng.uniform(220, 320, n)]
    p0, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), calib.K0, calib.D0)
    p1, _ = cv2.projectPoints(X, cv2.Rodrigues(calib.R)[0], calib.T_mm, calib.K1, calib.D1)
    return p0.reshape(-1, 2), p1.reshape(-1, 2)


def test_default_calibration_matches_turntable_model():
    own = load_rig_calibration()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "turntable_calib.json"
        path.write_text(json.dumps(TURNTABLE_CALIB))
        tt = load_rig_calibration(path)
    assert abs(own.baseline_mm - tt.baseline_mm) < 1e-6
    assert np.allclose(own.R, tt.R, atol=1e-9)
    assert np.allclose(own.K1, tt.K1)
    assert 150.0 < own.baseline_mm < 180.0
    assert 15.0 < own.relative_rotation_deg < 25.0


def test_refinement_recovers_perturbed_rotation():
    truth = load_rig_calibration()
    p0, p1 = _synthetic_pairs(truth)
    p1 += np.random.default_rng(1).normal(0, 0.3, p1.shape)
    p1[:300] += np.random.default_rng(2).uniform(-40, 40, (300, 2))      # 10 % gross outliers

    dR = cv2.Rodrigues(np.radians([0.5, -0.4, 0.3]))[0]
    perturbed = truth.with_pose(dR @ truth.R, truth.T_mm)
    assert np.median(epipolar_errors_px(perturbed, p0, p1)) > 3.0

    refined = refine_relative_pose(perturbed, p0, p1)
    assert refined.refinement["accepted"]
    assert np.median(epipolar_errors_px(refined, p0[300:], p1[300:])) < 0.4
    assert abs(refined.baseline_mm - truth.baseline_mm) < 1e-6           # metric scale is kept


def test_refinement_rejects_implausible_correction():
    truth = load_rig_calibration()
    p0, p1 = _synthetic_pairs(truth)
    moved = truth.with_pose(cv2.Rodrigues(np.radians([0.0, 8.0, 0.0]))[0] @ truth.R, truth.T_mm)
    out = refine_relative_pose(moved, p0, p1)
    assert out.refinement["accepted"] is False
    assert np.allclose(out.R, moved.R)


def test_rasterize_triangles_zbuffer_keeps_closest():
    pts = np.array([[0, 0], [20, 0], [0, 20], [0, 0], [20, 0], [0, 20]], dtype=float)
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    vals = np.array([[5.0], [5.0], [5.0], [2.0], [2.0], [2.0]])
    out = rasterize_triangles(pts, tris, vals, (25, 25), depth_channel=0)
    assert np.nanmax(out) == 2.0
    assert np.isnan(out[24, 24, 0])

    fid, bc = rasterize_mesh(np.array([[[0, 0], [10, 0], [0, 10]]], dtype=float), (12, 12))
    assert (fid == 0).sum() > 40
    assert np.allclose(bc[fid == 0].sum(axis=1), 1.0, atol=1e-5)


def _height_field_mesh(n: int = 40) -> trimesh.Trimesh:
    xs, ys = np.meshgrid(np.linspace(-50, 50, n), np.linspace(-60, 60, n))
    zs = 20 * np.exp(-(xs ** 2 + ys ** 2) / 1500.0)
    verts = np.c_[xs.ravel(), ys.ravel(), zs.ravel()]
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate([np.c_[a.ravel(), b.ravel(), c.ravel()], np.c_[b.ravel(), d.ravel(), c.ravel()]])
    return trimesh.Trimesh(verts, faces, process=True)


def test_solidify_height_field_is_watertight():
    surface = _height_field_mesh()
    solid = solidify_height_field(surface, back_offset_mm=3.0)
    assert solid.is_watertight
    assert solid.volume > 0
    assert abs(solid.bounds[0, 2] - (surface.bounds[0, 2] - 3.0)) < 1e-6


def test_bake_albedo_two_synthetic_cameras():
    mesh = _height_field_mesh(30)
    if np.mean(mesh.face_normals[:, 2]) < 0:
        mesh.invert()
    uv_mesh = unwrap_uv(mesh, 512)

    # A camera image whose colour encodes the image column; both cameras see the same scene.
    K = np.array([[900.0, 0, 400], [0, 900.0, 300], [0, 0, 1]])
    img = np.zeros((600, 800, 3), np.uint8)
    img[..., 1] = 150
    img[..., 2] = np.linspace(40, 220, 800).astype(np.uint8)[None, :]
    cams = []
    for yaw in (-10.0, 10.0):
        R = cv2.Rodrigues(np.radians([180.0, yaw, 0.0]))[0]        # looks along -z of the mesh frame
        t = np.array([0.0, 0.0, 400.0])
        cams.append(TextureCamera(img, K, np.zeros(5), R, t))
    tex, stats = bake_albedo(uv_mesh, cams, size=512, visibility_scale=0.5)
    assert tex.shape == (512, 512, 3)
    assert stats["texels_seen_fraction"] > 0.95
    seen = tex.sum(axis=2) > 0
    assert abs(float(np.median(tex[..., 1][seen])) - 150.0) < 3.0


from tests._functional import function_tests  # noqa: E402

load_tests = function_tests(globals())
