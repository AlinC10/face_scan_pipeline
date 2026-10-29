"""
exp_raft_stereo.py — Experiment (CPU only, no CUDA): RAFT-Stereo on the prior-warped face pair vs the DIS
measurement of the dense face pipeline. Not part of the pipeline or of the Docker image (tools/ is ignored).

For each reference view the other view is warped by the pipeline's smooth disparity (MediaPipe prior +
robust correction) minus a shift S, so the network only sees a small positive residual r in [0, 2S]:
    d = smooth - S + r
The right reference is handled by mirroring both images (RAFT-Stereo predicts on the left image grid).
Validity: left-right consistency of the two references (< 1 px) and r inside (0.5, 2S - 0.5).

Setup (once, outside the repository; RAFT-Stereo is MIT licensed, weights from its download_models.sh):
    git clone https://github.com/princeton-vl/RAFT-Stereo.git ~/ext/RAFT-Stereo
    cd ~/ext/RAFT-Stereo && bash download_models.sh
Run:
    .venv/bin/python tools/exp_raft_stereo.py --repo ~/ext/RAFT-Stereo \\
        --weights ~/ext/RAFT-Stereo/models/raftstereo-middlebury.pth -o output/exp_raft
Outputs in -o: stats.json (coverage, agreement with DIS in px / mm), dis/ and <tag>/ fused shaded reliefs
(face_surface_fusion.jpg), coverage_dis_vs_net.jpg.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import photogrammetry.face_correspondence as fc
from photogrammetry.rig_calibration import load_rig_calibration, refine_relative_pose
from photogrammetry.face_surface import fuse_face_surface

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="clone of github.com/princeton-vl/RAFT-Stereo")
ap.add_argument("--weights", required=True, help="e.g. models/raftstereo-middlebury.pth")
ap.add_argument("-l", "--left", default="input_images/rig/0/f_150.jpg")
ap.add_argument("-r", "--right", default="input_images/rig/1/f_150.jpg")
ap.add_argument("--calib", default=None, help="rig calibration (default assets/rig_calibration.json)")
ap.add_argument("-o", "--out", default="output/exp_raft")
ap.add_argument("--threads", type=int, default=0, help="CPU threads for PyTorch (0 = all)")
ap.add_argument("--iters", type=int, default=32)
ap.add_argument("--shift", type=float, default=24.0)
ap.add_argument("--tag", default="raft")
a = ap.parse_args()

sys.path.insert(0, str(Path(a.repo) / "core"))
from raft_stereo import RAFTStereo  # noqa: E402
from utils.utils import InputPadder  # noqa: E402

logging.basicConfig(level=logging.WARNING)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
if a.threads:
    torch.set_num_threads(a.threads)

# Same front end as photogrammetry/face_pipeline.py: landmarks, guided matches, pose refinement, rectification.
img0, img1 = cv2.imread(a.left), cv2.imread(a.right)
calib = load_rig_calibration(a.calib)
lm0, lm1 = fc.detect_landmarks([img0, img1])
p0, p1 = fc.guided_raw_matches(img0, img1, lm0, lm1)
calib = refine_relative_pose(calib, p0, p1)
rect = fc.rectify_pair(img0, img1, calib, 0.5)
l0, l1 = rect.rectify_points(lm0, 0), rect.rectify_points(lm1, 1)
z0, z1 = lm0[:, 2], lm1[:, 2]


class Args:
    hidden_dims = [128] * 3
    corr_implementation = "reg"
    shared_backbone = False
    corr_levels = 4
    corr_radius = 4
    n_downsample = 2
    context_norm = "batch"
    slow_fast_gru = False
    n_gru_layers = 3
    mixed_precision = False


model = RAFTStereo(Args())
state = torch.load(a.weights, map_location="cpu", weights_only=True)
model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
model.eval()


def raft(left_rgb, right_rgb):
    t = lambda im: torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float()[None]
    i1, i2 = t(left_rgb), t(right_rgb)
    padder = InputPadder(i1.shape, divis_by=32)
    i1, i2 = padder.pad(i1, i2)
    with torch.no_grad():
        _, flow = model(i1, i2, iters=a.iters, test_mode=True)
    return -padder.unpad(flow).squeeze().numpy()


def crop_box(mask, pad=48):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return max(ys.min() - pad, 0), min(ys.max() + pad, h), max(xs.min() - pad, 0), min(xs.max() + pad, w)


S = a.shift
est = {
    "left": fc.estimate_disparity(rect, l0, l1, z0, reference="left"),
    "right": fc.estimate_disparity(rect, l1, l0, z1, reference="right"),
}
h, w = est["left"].smooth.shape
mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
L = cv2.cvtColor(rect.left, cv2.COLOR_BGR2RGB)
R = cv2.cvtColor(rect.right, cv2.COLOR_BGR2RGB)

d_net, r_net, timing = {}, {}, {}
for ref in ("left", "right"):
    e = est[ref]
    cur = np.where(e.face_mask, e.smooth, 0).astype(np.float32) - S
    y0, y1, x0, x1 = crop_box(e.face_mask)
    if ref == "left":
        warped = cv2.remap(R, mx - cur, my, cv2.INTER_LINEAR)          # warped(x) = right(x - cur)
        A, B = L[y0:y1, x0:x1], warped[y0:y1, x0:x1]
    else:
        warped = cv2.remap(L, mx + cur, my, cv2.INTER_LINEAR)          # warped(x) = left(x + cur)
        A, B = R[y0:y1, x0:x1][:, ::-1], warped[y0:y1, x0:x1][:, ::-1]  # mirrored: right ref becomes "left"
    t0 = time.time()
    r = raft(A, B)
    timing[ref] = round(time.time() - t0, 1)
    if ref == "right":
        r = r[:, ::-1]
    full = np.full((h, w), np.nan, np.float32)
    full[y0:y1, x0:x1] = r
    r_net[ref] = full
    d_net[ref] = (cur + full).astype(np.float32)
    print(f"{ref}: crop {x1 - x0}x{y1 - y0}, RAFT {timing[ref]} s, residual r median {np.nanmedian(full[e.face_mask]):.2f} px",
          flush=True)

# Left-right consistency: d_L(x) vs d_R(x - d_L(x)) (right-ref grid) and the other way round.
valid_net = {}
for ref, other, sgn in (("left", "right", -1.0), ("right", "left", 1.0)):
    d = d_net[ref]
    back = cv2.remap(np.nan_to_num(d_net[other], nan=-1e4), mx + sgn * np.nan_to_num(d), my, cv2.INTER_LINEAR)
    r = r_net[ref]
    valid_net[ref] = (est[ref].face_mask & np.isfinite(d) & (np.abs(back - d) < 1.0)
                      & (r > 0.5) & (r < 2 * S - 0.5))

fB = rect.focal_px * rect.calib.baseline_mm
stats = {"weights": Path(a.weights).name, "iters": a.iters, "shift_px": S, "raft_s": timing}
for ref in ("left", "right"):
    e = est[ref]
    both = e.valid & valid_net[ref]
    diff = np.abs(d_net[ref] - e.disparity)[both]
    Z = fB / np.maximum(e.disparity[both] - float(rect.P1[0, 2] - rect.P2[0, 2]), 1.0)
    mm_per_px = np.median(Z) ** 2 / fB
    stats[ref] = {
        "dis_valid_pct": round(100 * e.valid.sum() / e.face_mask.sum(), 1),
        "net_valid_pct": round(100 * valid_net[ref].sum() / e.face_mask.sum(), 1),
        "agree_median_px": round(float(np.median(diff)), 3),
        "agree_p90_px": round(float(np.percentile(diff, 90)), 3),
        "agree_median_mm": round(float(np.median(diff) * mm_per_px), 3),
        "agree_p90_mm": round(float(np.percentile(diff, 90) * mm_per_px), 3),
    }
print(json.dumps(stats, indent=1))

# Surfaces: same fusion, measurement from DIS vs from the network.
for name, ests in (
    ("dis", [est["left"], est["right"]]),
    (a.tag, [fc.DisparityEstimate(r, d_net[r], valid_net[r], est[r].smooth, est[r].prior, est[r].face_mask,
                                  est[r].shape_fit) for r in ("left", "right")]),
):
    surf = fuse_face_surface(rect, ests, l0, debug_dir=out / name)
    print(name, {k: surf.stats[k] for k in ("measured_coverage", "inlier_coverage", "prior_correction_p95_abs_mm")})
    np.save(out / f"{name}_z.npy", surf.mesh.vertices[:, 2])

# Coverage maps
for ref in ("left",):
    e = est[ref]
    y0, y1, x0, x1 = crop_box(e.face_mask, 0)
    vis = cv2.cvtColor(cv2.cvtColor(rect.left, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    vis[e.valid] = (0.5 * vis[e.valid] + [0, 120, 0]).astype(np.uint8)
    vis2 = cv2.cvtColor(cv2.cvtColor(rect.left, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    vis2[valid_net[ref]] = (0.5 * vis2[valid_net[ref]] + [0, 0, 140]).astype(np.uint8)
    cv2.imwrite(str(out / "coverage_dis_vs_net.jpg"), np.hstack([vis[y0:y1, x0:x1], vis2[y0:y1, x0:x1]]))
(out / "stats.json").write_text(json.dumps(stats, indent=1))
