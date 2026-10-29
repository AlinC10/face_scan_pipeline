"""
GNM Head Fitting (NumPy / SciPy, no PyTorch)
============================================
Fits GNM Head v3 to a measured face, in the head frame of the dense pipeline (mm; x to the
subject's left, y up, z towards the viewer — the same axes as GNM, which is in metres).

Model, with the joints in the bind pose (no pose correctives) and a global rigid motion:
    v(id, ex) = R_g (T + B_id id + B_ex ex) * 1000 + t_g
Linear in (id, ex) for a fixed (R_g, t_g), so every ICP iteration is one regularised linear
least-squares problem (as gnm/shape/fitting_utils/regularized_least_squares.py in the GNM repo):

  1. rigid start: Umeyama (no scale: both are metric) on the inner iBUG-68 landmarks
     (GNM HEAD_SPARSE_68 vs the MediaPipe landmarks mapped to 68);
  2. repeat: closest points of the data on the current GNM skin (dense surface samples with
     known barycentric coordinates, KD-tree) -> point-to-plane rows; landmark rows (point-to-point);
     Tikhonov prior on id / ex (unit-variance components) -> solve; rigid update (Procrustes on the
     same correspondences); robust re-weighting (Geman-McClure) and a shrinking rejection radius.

Expression components: the 150 lower-face components and the eye components are free, the
tongue and the iris stay at zero; identity: the 170 head components (eyeball / teeth stay zero).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from photogrammetry.gnm_head import GNMHead, GNMParams

log = logging.getLogger("Photogrammetry.GNMFit")

# MediaPipe face mesh (468) -> iBUG 68, the mapping commonly used with dlib-68 models
# (jaw 0-16, brows 17-26, nose 27-35, eyes 36-47, mouth 48-67; subject's right first, as iBUG).
MP_TO_IBUG68 = np.array([
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    71, 63, 105, 66, 107, 336, 296, 334, 293, 301,
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87,
])
INNER_68 = np.arange(17, 68)            # brows, nose, eyes, mouth (the jaw contour depends on the view)


@dataclass
class GNMFitResult:
    params: GNMParams                     # identity / expression, rotations = 0, translation = 0, scale 1000 (mm)
    R: np.ndarray                         # head frame <- GNM (scaled) rotation
    t: np.ndarray                         # (mm)
    vertices: np.ndarray                  # (V, 3) fitted GNM in the head frame (mm)
    stats: Dict[str, float] = field(default_factory=dict)

    def transform(self, gnm_vertices_mm: np.ndarray) -> np.ndarray:
        """GNM vertices (mm, model frame) -> head frame."""
        return gnm_vertices_mm @ self.R.T + self.t


def _umeyama_rigid(src: np.ndarray, dst: np.ndarray, w: Optional[np.ndarray] = None):
    w = np.ones(len(src)) if w is None else w
    w = w / w.sum()
    ms, md = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    H = ((src - ms) * w[:, None]).T @ (dst - md)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, md - R @ ms


class GNMFitter:
    def __init__(self, gnm: Optional[GNMHead] = None, n_identity: int = 170, samples_per_edge: int = 3,
                 surface_groups: Sequence[str] = ("skin",)):
        self.g = gnm or GNMHead()
        self.n_id = int(n_identity)
        ex_names = self.g.expression_names
        self.ex_idx = np.array([i for i, n in enumerate(ex_names)
                                if n.startswith(("lower_face", "left_eye", "right_eye"))])
        # Dense samples of the skin surface: barycentric grid on every skin triangle
        tri = self.g.triangles[self.g.skin_triangles]
        bary = [(a / samples_per_edge, b / samples_per_edge, 1 - (a + b) / samples_per_edge)
                for a in range(samples_per_edge + 1) for b in range(samples_per_edge + 1 - a)]
        bary = np.array(bary)
        self.sample_tri = np.repeat(tri, len(bary), axis=0)                    # (S, 3) vertex ids
        self.sample_bary = np.tile(bary, (len(tri), 1))                         # (S, 3)
        self.tri = tri

    # ------------------------------------------------------------------ model pieces (mm)
    def bind_mm(self, idv: np.ndarray, ex: np.ndarray) -> np.ndarray:
        """Bind-pose vertices (mm) from the free identity / expression coefficients."""
        id_full = np.zeros(self.g.identity_dim)
        id_full[:self.n_id] = idv
        ex_full = np.zeros(self.g.expression_dim)
        ex_full[self.ex_idx] = ex
        return self.g.bind_vertices(id_full, ex_full) * 1000.0

    def _params(self, idv_free, ex_free) -> GNMParams:
        p = self.g.zero_params()
        p.identity[:self.n_id] = idv_free
        p.expression[self.ex_idx] = ex_free
        p.scale = 1000.0
        return p

    def _basis_at(self, vid: np.ndarray, bary: np.ndarray):
        """Model-frame (mm) basis rows at barycentric points: identity (P, 3, n_id), expression (P, 3, n_ex)."""
        Bi = np.einsum("pk,ipkc->pci", bary, self.g.id_basis[:self.n_id][:, vid]) * 1000.0
        Be = np.einsum("pk,epkc->pce", bary, self.g.ex_basis[self.ex_idx][:, vid]) * 1000.0
        return Bi, Be

    # ------------------------------------------------------------------ landmark correspondences
    def landmarks68_correspondence(self, data68: np.ndarray, jaw_weight: float = 0.0):
        """(data, vid, bary, weight) for the iBUG-68 landmarks of GNM (the jaw contour off by default)."""
        w = np.ones(68)
        w[:17] = jaw_weight
        return data68, self.g.lm68_idx, self.g.lm68_w, w

    # ------------------------------------------------------------------ fitting
    def fit(
        self,
        points: np.ndarray,
        point_weights: Optional[np.ndarray] = None,
        landmarks: Optional[tuple] = None,
        landmark_weight: float = 0.3,
        lambda_identity: float = 2.0,
        lambda_expression: float = 6.0,
        iterations: int = 10,
        max_points: int = 15000,
        init_radius_mm: float = 12.0,
        final_radius_mm: float = 3.0,
        expression: bool = True,
        seed: int = 0,
    ) -> GNMFitResult:
        """
        points: (N, 3) measured surface points (mm, head frame), with optional weights.
        landmarks: (data (L, 3), vid (L, 3), bary (L, 3), weight (L,)) — data points and the GNM surface points
        (barycentric) they correspond to; they start the rigid alignment and constrain the fit.
        Regularisation: lambda * ||coef||^2 against residuals in mm (unit-variance coefficients).
        """
        rng = np.random.default_rng(seed)
        if len(points) > max_points:
            sel = rng.choice(len(points), max_points, replace=False)
            points = points[sel]
            point_weights = None if point_weights is None else point_weights[sel]
        pw = np.ones(len(points)) if point_weights is None else np.asarray(point_weights, dtype=np.float64)
        n_ex = len(self.ex_idx) if expression else 0
        idv = np.zeros(self.n_id)
        ex = np.zeros(len(self.ex_idx))
        if landmarks is not None:
            l_data, l_vid, l_bary, l_w = (np.asarray(a) for a in landmarks)
            ok = np.isfinite(l_data).all(axis=1) & (l_w > 0)
            l_data, l_vid, l_bary, l_w = l_data[ok], l_vid[ok], l_bary[ok], l_w[ok].astype(np.float64)
        else:
            l_data = np.zeros((0, 3))

        # 1. rigid start on the landmarks (or the centroids when there are none)
        V = self.bind_mm(idv, ex)
        if len(l_data) >= 6:
            model_lm = np.einsum("lk,lkc->lc", l_bary, V[l_vid])
            R, t = _umeyama_rigid(model_lm, l_data, l_w)
        else:
            R, t = np.eye(3), points.mean(0) - V[self.g.group("skin")].mean(0)

        stats: Dict[str, float] = {}
        radii = np.geomspace(init_radius_mm, final_radius_mm, iterations)
        for it, radius in enumerate(radii):
            V = self.bind_mm(idv, ex)
            Vw = V @ R.T + t
            S = np.einsum("sk,skc->sc", self.sample_bary, Vw[self.sample_tri])
            dist, nn = cKDTree(S).query(points, k=1)
            vid, bary = self.sample_tri[nn], self.sample_bary[nn]
            a, b, c = Vw[vid[:, 0]], Vw[vid[:, 1]], Vw[vid[:, 2]]
            nrm = np.cross(b - a, c - a)
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            q = np.einsum("pk,pkc->pc", bary, Vw[vid])
            r_pl = np.sum((points - q) * nrm, axis=1)
            keep = dist < radius
            s = max(1.0, 0.5 * radius)                               # Geman-McClure scale
            w = pw * keep * (s * s / (s * s + r_pl ** 2)) ** 2
            P = np.nonzero(w > 1e-6)[0]
            if len(P) < 200:
                log.warning("GNM fit: only %d correspondences at iteration %d", len(P), it)
                break
            Bi, Be = self._basis_at(vid[P], bary[P])
            nR = nrm[P] @ R                                          # n^T R
            rows = [np.einsum("pc,pci->pi", nR, Bi)] + ([np.einsum("pc,pce->pe", nR, Be)] if n_ex else [])
            A = np.concatenate(rows, axis=1) * np.sqrt(w[P])[:, None]
            AtA = A.T @ A
            Aty = A.T @ (r_pl[P] * np.sqrt(w[P]))
            w_sum = float(w[P].sum())
            if len(l_data):
                Bli, Ble = self._basis_at(l_vid, l_bary)
                cur = np.einsum("lk,lkc->lc", l_bary, Vw[l_vid])
                RB = [np.einsum("mc,lci->lmi", R, Bli)] + ([np.einsum("mc,lce->lme", R, Ble)] if n_ex else [])
                Al = np.concatenate(RB, axis=2).reshape(-1, AtA.shape[0])
                # landmark block weighted as `landmark_weight` of the surface block
                wl = np.repeat(l_w / l_w.sum(), 3) * landmark_weight * w_sum
                AtA += Al.T @ (Al * wl[:, None])
                Aty += Al.T @ ((l_data - cur).reshape(-1) * wl)
            coef = np.concatenate([idv] + ([ex] if n_ex else []))
            lam = np.concatenate([np.full(self.n_id, lambda_identity), np.full(n_ex, lambda_expression)]) * w_sum / 1000.0
            delta = np.linalg.solve(AtA + np.diag(lam), Aty - lam * coef)
            idv = idv + delta[:self.n_id]
            if n_ex:
                ex = ex + delta[self.n_id:]
            # rigid update (weighted point-to-point on the same correspondences + the landmarks)
            V = self.bind_mm(idv, ex)
            src = [np.einsum("pk,pkc->pc", bary[P], V[vid[P]])]
            dst = [points[P]]
            ww = [w[P]]
            if len(l_data):
                src.append(np.einsum("lk,lkc->lc", l_bary, V[l_vid]))
                dst.append(l_data)
                ww.append(l_w / l_w.sum() * landmark_weight * w_sum)
            R, t = _umeyama_rigid(np.concatenate(src), np.concatenate(dst), np.concatenate(ww))
            stats = {"iterations": it + 1, "correspondences": int(len(P)), "radius_mm": round(float(radius), 2),
                     "point_to_plane_rms_mm": round(float(np.sqrt(np.average(r_pl[P] ** 2, weights=w[P]))), 3)}

        V = self.bind_mm(idv, ex)
        result = GNMFitResult(self._params(idv, ex), R, t, V @ R.T + t, stats)
        result.stats.update(self.evaluate(result, points, landmarks=(l_data, l_vid, l_bary) if len(l_data) else None))
        result.stats.update({"identity_norm": round(float(np.linalg.norm(idv) / np.sqrt(self.n_id)), 3),
                             "expression_norm": round(float(np.linalg.norm(ex) / np.sqrt(max(len(ex), 1))), 3)})
        return result

    def surface_correspondence(self, vertices_world: np.ndarray, points: np.ndarray):
        """Closest GNM skin point of each data point: (distance, vid (N, 3), bary (N, 3))."""
        S = np.einsum("sk,skc->sc", self.sample_bary, vertices_world[self.sample_tri])
        dist, nn = cKDTree(S).query(points, k=1)
        return dist, self.sample_tri[nn], self.sample_bary[nn]

    def evaluate(self, result: GNMFitResult, points: np.ndarray, landmarks=None) -> Dict[str, float]:
        """Point-to-plane distance of the data points to the fitted skin, and the landmark error (mm)."""
        Vw = result.vertices
        _, vid, bary = self.surface_correspondence(Vw, points)
        a, b, c = Vw[vid[:, 0]], Vw[vid[:, 1]], Vw[vid[:, 2]]
        nrm = np.cross(b - a, c - a)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        r = np.abs(np.sum((points - np.einsum("pk,pkc->pc", bary, Vw[vid])) * nrm, axis=1))
        out = {"surface_rms_mm": round(float(np.sqrt(np.mean(r ** 2))), 3),
               "surface_median_mm": round(float(np.median(r)), 3),
               "surface_p90_mm": round(float(np.percentile(r, 90)), 3)}
        if landmarks is not None:
            l_data, l_vid, l_bary = landmarks
            e = np.linalg.norm(np.einsum("lk,lkc->lc", l_bary, Vw[l_vid]) - l_data, axis=1)
            out["landmark_median_mm"] = round(float(np.median(e)), 2)
            out["landmark_p90_mm"] = round(float(np.percentile(e, 90)), 2)
        return out
