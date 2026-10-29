"""
Two-View Texture Baking for the Facial Mesh
===========================================
Bakes an albedo atlas for a UV-unwrapped facial mesh from the two original
12.3 MP photographs, using the (refined) calibrated camera poses.

  1. UV atlas: xatlas charts packed into a square texture.
  2. Texel geometry: every texel inside a chart gets its 3-D position and normal
     (barycentric interpolation over its triangle).
  3. Visibility: the surface without the measured detail is rendered into a depth buffer per camera
     (half the photo resolution); a texel is visible if it is not behind it (soft ramp, 3 mm tolerance)
     and faces the camera. The weights are smoothed over neighbouring texels (no flicker, no stripes).
  4. Weights: w_c = visible · cos(θ_c)^3 · border fade, where θ_c is the angle
     between the normal and the ray to camera c.
  5. Colour balance: a smooth per-channel gain map (sigma 15 mm) from texels seen well by both cameras,
     split evenly between them.
  6. Two-band blending: the low frequencies (σ = 6 px) are averaged with w_c,
     the high frequencies come from the most frontal camera (weights w_c^8),
     which keeps the pores sharp and avoids ghosting from residual misalignment.
  7. Unseen / grazing texels: harmonic fill of the low band on the mesh (Laplace); charts padded.

A tangent-space normal map of the high-polygon surface can also be baked onto
the low-polygon atlas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

log = logging.getLogger("Photogrammetry.FaceTexture")


@dataclass
class TextureCamera:
    """A photograph and its pinhole + radial distortion model, posed in the mesh frame."""
    image: np.ndarray           # BGR, full resolution
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray               # mesh frame -> camera rotation
    t: np.ndarray               # mesh frame -> camera translation (mesh units)
    _bands: Optional[Tuple[float, np.ndarray, np.ndarray]] = None

    def bands(self, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """(image, low-pass image) as float32, cached between bakes."""
        if self._bands is None or self._bands[0] != sigma:
            img = self.image.astype(np.float32)
            self._bands = (sigma, img, cv2.GaussianBlur(img, (0, 0), sigma))
        return self._bands[1], self._bands[2]

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates (N, 2) and depth (N,) of mesh-frame points."""
        pc = pts @ self.R.T + self.t
        px, _ = cv2.projectPoints(pts.reshape(-1, 1, 3).astype(np.float64), cv2.Rodrigues(self.R)[0],
                                  self.t.astype(np.float64), self.K, self.D)
        return px.reshape(-1, 2), pc[:, 2]


# ---------------------------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------------------------

def rasterize_mesh(
    tri_px: np.ndarray,
    shape: Tuple[int, int],
    tri_depth: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised triangle rasteriser (pixel centres). `tri_px` is (F, 3, 2). Returns the
    triangle id per pixel (-1 = empty) and the barycentric coordinates (h, w, 3).
    With `tri_depth` (F, 3), overlapping triangles are resolved by the smallest depth.
    """
    h, w = shape
    face_id = -np.ones((h, w), dtype=np.int64)
    bary = np.zeros((h, w, 3), dtype=np.float32)
    best = np.full((h, w), np.inf) if tri_depth is not None else None

    lo = np.floor(tri_px.min(axis=1)).astype(np.int64)
    hi = np.ceil(tri_px.max(axis=1)).astype(np.int64)
    size = np.maximum(hi - lo + 1, 1).max(axis=1)
    keep = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < w) & (lo[:, 1] < h)
    # Buckets of similar bounding-box size are processed as dense (n, k, k) blocks.
    for k in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096):
        sel = np.where(keep & (size <= k))[0]
        keep[sel] = False
        if len(sel) == 0:
            continue
        for chunk in np.array_split(sel, max(1, int(len(sel) * k * k // 4_000_000) + 1)):
            _raster_block(tri_px[chunk], lo[chunk], k, chunk, (h, w), face_id, bary,
                          None if tri_depth is None else tri_depth[chunk], best)
    return face_id, bary


def _raster_block(tp, lo, k, ids, shape, face_id, bary, depth, best):
    h, w = shape
    oy, ox = np.mgrid[0:k, 0:k]
    px = lo[:, None, None, 0] + ox[None]            # (n, k, k)
    py = lo[:, None, None, 1] + oy[None]
    a, b, c = tp[:, 0], tp[:, 1], tp[:, 2]
    v0 = b - a
    v1 = c - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    good = np.abs(den) > 1e-12
    den = np.where(good, den, 1.0)
    qx = px + 0.5 - a[:, None, None, 0]
    qy = py + 0.5 - a[:, None, None, 1]
    l1 = (qx * v1[:, None, None, 1] - qy * v1[:, None, None, 0]) / den[:, None, None]
    l2 = (qy * v0[:, None, None, 0] - qx * v0[:, None, None, 1]) / den[:, None, None]
    l0 = 1.0 - l1 - l2
    eps = -1e-6
    inside = (l0 >= eps) & (l1 >= eps) & (l2 >= eps) & good[:, None, None]
    inside &= (px >= 0) & (px < w) & (py >= 0) & (py < h)
    n_idx, yy, xx = np.nonzero(inside)
    if len(n_idx) == 0:
        return
    pix_x = px[n_idx, yy, xx]
    pix_y = py[n_idx, yy, xx]
    bc = np.stack([l0[n_idx, yy, xx], l1[n_idx, yy, xx], l2[n_idx, yy, xx]], axis=1)
    if depth is not None:
        z = (bc * depth[n_idx]).sum(axis=1)
        order = np.lexsort((z, pix_y * w + pix_x))
        flat = (pix_y * w + pix_x)[order]
        first = np.ones(len(flat), bool)
        first[1:] = flat[1:] != flat[:-1]
        sel = order[first]
        closer = z[sel] < best[pix_y[sel], pix_x[sel]]
        sel = sel[closer]
        best[pix_y[sel], pix_x[sel]] = z[sel]
    else:
        sel = np.arange(len(n_idx))
    face_id[pix_y[sel], pix_x[sel]] = ids[n_idx[sel]]
    bary[pix_y[sel], pix_x[sel]] = bc[sel]


# ---------------------------------------------------------------------------------------------
# UV atlas
# ---------------------------------------------------------------------------------------------

def unwrap_uv(mesh: trimesh.Trimesh, resolution: int, density: Optional[np.ndarray] = None,
              min_density: float = 0.35) -> trimesh.Trimesh:
    """
    xatlas atlas; returns a mesh whose vertices are split along the chart seams, with UVs.
    `density` (per vertex, 0..1): where it is low the surface is shrunk (towards the centroid of the dense part)
    before the unwrap, so it takes less of the atlas: the measured face keeps its texel density on a whole head
    whose back is only filled in. The returned mesh keeps the real positions.
    """
    import xatlas

    pos = mesh.vertices
    if density is not None:
        dens = np.clip(np.asarray(density, dtype=np.float64), 0.0, 1.0)
        c = pos[dens > 0.5].mean(axis=0) if (dens > 0.5).any() else pos.mean(axis=0)
        s = min_density + (1.0 - min_density) * dens
        pos = c + (pos - c) * s[:, None]
    atlas = xatlas.Atlas()
    atlas.add_mesh(pos.astype(np.float32), mesh.faces.astype(np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = resolution
    pack.padding = 4
    pack.bilinear = True
    atlas.generate(pack_options=pack)
    vmapping, indices, uvs = atlas[0]
    out = trimesh.Trimesh(mesh.vertices[vmapping], indices.astype(np.int64), process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=uvs)
    log.info("UV atlas: %d charts, %d -> %d vertices, utilisation %.0f%%",
             atlas.chart_count, len(mesh.vertices), len(vmapping), 100 * atlas.utilization)
    return out


def _texel_geometry(mesh: trimesh.Trimesh, size: int):
    """Texel -> (triangle, 3-D position, normal) for all texels covered by the UV charts."""
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    tri_uv = uv[mesh.faces] * size
    tri_uv[..., 1] = size - tri_uv[..., 1]          # image rows grow downwards, v grows upwards
    fid, bc = rasterize_mesh(tri_uv, (size, size))
    covered = fid >= 0
    f = fid[covered]
    b = bc[covered].astype(np.float64)
    tri = mesh.faces[f]
    pos = (mesh.vertices[tri] * b[..., None]).sum(axis=1)
    vn = np.asarray(mesh.vertex_normals)
    nrm = (vn[tri] * b[..., None]).sum(axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    return covered, f, b, pos, nrm


def _depth_buffer(mesh: trimesh.Trimesh, cam: TextureCamera, scale: float) -> np.ndarray:
    h, w = cam.image.shape[:2]
    shape = (int(h * scale), int(w * scale))
    px, z = cam.project(mesh.vertices)
    tri_px = px[mesh.faces] * scale
    fid, bc = rasterize_mesh(tri_px, shape, tri_depth=z[mesh.faces])
    depth = np.full(shape, np.inf)
    ok = fid >= 0
    depth[ok] = (bc[ok] * z[mesh.faces[fid[ok]]]).sum(axis=1)
    return depth


def _sample(img: np.ndarray, px: np.ndarray, width: int = 4096) -> np.ndarray:
    """Bilinear samples of `img` at pixel coordinates (remap needs maps below 32767 x 32767)."""
    n = len(px)
    rows = -(-n // width)
    pad = np.zeros((rows * width, 2), np.float32)
    pad[:n] = px
    mx = pad[:, 0].reshape(rows, width)
    my = pad[:, 1].reshape(rows, width)
    out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out.reshape(rows * width, -1)[:n].astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Baking
# ---------------------------------------------------------------------------------------------

def bake_albedo(
    mesh: trimesh.Trimesh,
    cameras: List[TextureCamera],
    size: int = 4096,
    visibility_scale: float = 0.5,
    depth_tolerance: float = 3.0,
    low_sigma_px: float = 6.0,
    sharpness: float = 8.0,
    visibility_mesh: Optional[trimesh.Trimesh] = None,
    gain_sigma_mm: Optional[float] = 15.0,
    weak_weight: float = 0.05,
    fill_prior: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    debug_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Two-band, visibility-aware albedo atlas (BGR uint8, size x size) for a UV mesh (mesh frame in mm).

    * Visibility: depth buffer of `visibility_mesh` (the surface without the measured detail; default the
      mesh itself) at `visibility_scale` of the photographs, with a tolerance above the detail amplitude,
      and a soft ramp instead of a hard test, so neighbouring texels do not flicker between cameras.
    * Colour balance: a smooth gain map cam0 / cam1 (sigma `gain_sigma_mm` over the face, in the mesh
      x-y plane) from texels both cameras see well, instead of one global gain (None: global gain only).
      The visibility tolerance grows with the local depth slope of the buffer (grazing surfaces).
    * Unseen or grazing texels (total weight < `weak_weight`): harmonic (Laplace) fill of the low band over
      the mesh surface from the well-seen texels around them, blended in as the weight drops: no edges.
      `fill_prior` = (per-vertex weight of `mesh`, BGR colour): where that weight is high the unseen texels take
      this colour instead (the hair colour on the unseen scalp, plan P3.3).
    """
    covered, f, b, pos, nrm = _texel_geometry(mesh, size)
    n_tex = len(pos)
    vis_mesh = visibility_mesh if visibility_mesh is not None else mesh

    weights, lows, highs, visible_l = [], [], [], []
    for cam in cameras:
        px, z = cam.project(pos)
        depth = _depth_buffer(vis_mesh, cam, visibility_scale)
        slope = _depth_slope(depth)
        q = np.clip((px * visibility_scale).round().astype(int), 0, [depth.shape[1] - 1, depth.shape[0] - 1])
        # Nearest-pixel lookups on a steep (grazing) surface are off by up to half a pixel of depth change,
        # so the tolerance grows with the local slope. depth = inf outside the surface: visible.
        tol = depth_tolerance + 1.5 * slope[q[:, 1], q[:, 0]]
        dz = depth[q[:, 1], q[:, 0]] + tol - z
        vis = np.clip(dz / tol, 0.0, 1.0)
        vis[~np.isfinite(dz)] = 1.0
        ray = cam.centre - pos
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        cos = np.clip(np.sum(nrm * ray, axis=1), 0.0, 1.0)
        h, w = cam.image.shape[:2]
        border = np.minimum.reduce([px[:, 0], px[:, 1], w - 1 - px[:, 0], h - 1 - px[:, 1]])
        fade = np.clip(border / 100.0, 0.0, 1.0)
        wgt = _smooth_texel_values(vis * cos ** 3 * fade, covered, sigma=2.0)
        img, low_img = cam.bands(low_sigma_px)
        low = _sample(img=low_img, px=px)
        high = _sample(img, px) - low
        weights.append(wgt)
        lows.append(low)
        highs.append(high)
        visible_l.append(vis > 0.5)
        log.info("Texture camera: %.0f%% of the texels visible", 100.0 * np.mean(wgt > 0))

    W = np.stack(weights)                                       # (C, N)
    stats: Dict[str, float] = {}
    if len(cameras) == 2:
        both = (W[0] > 0.3) & (W[1] > 0.3)
        if both.sum() > 1000:
            ratio = np.median((lows[0][both] + 1.0) / (lows[1][both] + 1.0), axis=0)
            stats["colour_gain_cam1_to_cam0_bgr"] = [round(float(x), 4) for x in ratio]
            if gain_sigma_mm:
                gain = _local_gain(pos, lows[0], lows[1], both, sigma_mm=gain_sigma_mm, global_ratio=ratio)
            else:
                gain = np.broadcast_to(np.clip(ratio, 0.5, 2.0), lows[0].shape)
            g = np.sqrt(gain)
            lows[0] /= g
            highs[0] /= g
            lows[1] *= g
            highs[1] *= g
            log.info("Colour balance cam1 -> cam0: global (B, G, R) %s, local gain range %.3f-%.3f",
                     np.round(ratio, 3), float(gain.min()), float(gain.max()))
            sel = (W[0] > 0) & (W[1] > 0)
            dE = _delta_e(lows[0][sel], lows[1][sel])
            stats["camera_delta_e_median"] = round(float(np.median(dE)), 2)
            stats["camera_delta_e_p90"] = round(float(np.percentile(dE, 90)), 2)
            log.info("Colour difference between the cameras after balancing: dE median %.1f, p90 %.1f",
                     stats["camera_delta_e_median"], stats["camera_delta_e_p90"])

    wsum = W.sum(axis=0)
    seen = wsum > 1e-6
    low = sum(W[c][:, None] * lows[c] for c in range(len(cameras))) / np.maximum(wsum, 1e-9)[:, None]
    Ws = W ** sharpness
    wssum = Ws.sum(axis=0)
    high = sum(Ws[c][:, None] * highs[c] for c in range(len(cameras))) / np.maximum(wssum, 1e-30)[:, None]
    colour = np.clip(low + high, 0, 255)

    # Texels no camera sees, or sees only at a grazing angle (e.g. under the nose, the cameras look down):
    # smooth low-band colour of the nearest well-seen surface (Gaussian weights in 3-D), blended in as the
    # total weight drops, so there is no edge where the photographs end.
    strong = wsum >= weak_weight
    weak = ~strong
    rows, cols = np.nonzero(covered)
    if weak.any() and strong.any():
        fill = mesh_harmonic_fill(mesh, f, b, low, strong)[weak]
        if fill_prior is not None:
            vw, bgr = fill_prior
            tw = np.clip(np.sum(b[weak] * np.asarray(vw)[mesh.faces[f[weak]]], axis=1), 0.0, 1.0)[:, None]
            fill = (1.0 - tw) * fill + tw * np.asarray(bgr, dtype=np.float64)[None, :]
        alpha = np.clip(wsum[weak] / weak_weight, 0.0, 1.0)[:, None] ** 2
        colour[weak] = alpha * colour[weak] + (1.0 - alpha) * fill

    tex = np.zeros((size, size, 3), np.uint8)
    tex[rows, cols] = colour.astype(np.uint8)
    tex = add_gutter(tex, covered)
    stats.update({
        "texture_size": size,
        "texels": int(n_tex),
        "texels_seen_fraction": round(float(seen.mean()), 4),
        "texels_seen_by_both_fraction": round(float(np.mean((W > 0).all(axis=0))), 4) if len(cameras) > 1 else 0.0,
    })
    if debug_dir is not None and len(cameras) == 2:
        _save_texture_debug(Path(debug_dir), size, covered, W, visible_l, lows)
    return tex, stats


def _depth_slope(depth: np.ndarray) -> np.ndarray:
    """Largest depth change to a 4-neighbour (mm per depth-buffer pixel); 0 where undefined."""
    d = np.where(np.isfinite(depth), depth, np.nan)
    slope = np.zeros_like(d)
    for axis in (0, 1):
        g = np.abs(np.diff(d, axis=axis))
        g = np.nan_to_num(g, nan=0.0)
        pad = [(0, 0), (0, 0)]
        pad[axis] = (0, 1)
        slope = np.maximum(slope, np.pad(g, pad))
        pad[axis] = (1, 0)
        slope = np.maximum(slope, np.pad(g, pad))
    return np.minimum(slope, 20.0)          # occlusion edges: the depth jump itself must not hide them


def _smooth_texel_values(values: np.ndarray, covered: np.ndarray, sigma: float) -> np.ndarray:
    """Mask-aware Gaussian blur of per-texel values in the atlas (removes texel-to-texel flicker)."""
    img = np.zeros(covered.shape, np.float32)
    img[covered] = values
    num = cv2.GaussianBlur(img, (0, 0), sigma)
    den = cv2.GaussianBlur(covered.astype(np.float32), (0, 0), sigma)
    out = num / np.maximum(den, 1e-6)
    return np.where(values > 0, out[covered], 0.0)       # never make an invisible texel visible


def _local_gain(pos, low0, low1, both, sigma_mm: float, global_ratio, cell_mm: float = 2.0) -> np.ndarray:
    """
    Per-texel cam1 -> cam0 colour gain: log-ratio of the low bands where both cameras see well,
    smoothed on a grid over the mesh x-y plane (normalized convolution), global ratio where unsupported.
    """
    xy = pos[:, :2]
    lo = xy.min(axis=0)
    grid = np.floor((xy - lo) / cell_mm).astype(int)
    gh, gw = grid[:, 1].max() + 1, grid[:, 0].max() + 1
    flat = grid[:, 1] * gw + grid[:, 0]
    logr = np.log((low0 + 1.0) / (low1 + 1.0))
    sig = sigma_mm / cell_mm
    out = np.empty_like(logr)
    cnt = np.bincount(flat[both], minlength=gh * gw).reshape(gh, gw).astype(np.float32)
    den = cv2.GaussianBlur(cnt, (0, 0), sig)
    for ch in range(logr.shape[1]):
        acc = np.bincount(flat[both], weights=logr[both, ch], minlength=gh * gw).reshape(gh, gw).astype(np.float32)
        num = cv2.GaussianBlur(acc, (0, 0), sig)
        g_log = np.log(float(global_ratio[ch]))
        support = np.clip(den / max(float(den.max()) * 0.05, 1e-6), 0.0, 1.0)
        field = np.where(den > 1e-6, num / np.maximum(den, 1e-6), g_log)
        field = support * field + (1.0 - support) * g_log
        out[:, ch] = field.ravel()[flat]
    return np.clip(np.exp(out), 0.5, 2.0)


def mesh_harmonic_fill(mesh: trimesh.Trimesh, tex_face: np.ndarray, tex_bary: np.ndarray, values: np.ndarray,
                       known: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Harmonic (Laplace) interpolation of per-texel `values` over the SURFACE, not the atlas: the well-seen
    texels give the colour of the mesh vertices they cover; the other vertices solve L x = 0 on the vertex
    graph (UV seams merged, so charts are connected as on the face); every texel then takes the barycentric
    interpolation. Smooth, continuous across charts, correct even for a chart nobody sees.
    """
    from scipy.sparse import coo_matrix, identity
    from scipy.sparse.linalg import spsolve

    V = np.asarray(mesh.vertices)
    # Seam duplicates (same position, different UV) -> one graph vertex
    _, inv = np.unique(np.round(V / 1e-4).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    n = int(inv.max()) + 1
    F = inv[np.asarray(mesh.faces)]
    tri = F[tex_face]                                            # (T, 3) graph vertices of each texel
    wk = tex_bary * known[:, None]
    acc_w = np.bincount(tri.ravel(), weights=wk.ravel(), minlength=n)
    acc_v = np.stack([np.bincount(tri.ravel(), weights=(wk * values[:, c:c + 1]).ravel(), minlength=n)
                      for c in range(values.shape[1])], axis=1)
    is_known = acc_w > 0.5
    x = np.zeros((n, values.shape[1]))
    x[is_known] = acc_v[is_known] / acc_w[is_known, None]
    unknown = ~is_known
    if unknown.any():
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        e = np.unique(np.sort(e, axis=1), axis=0)
        A = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
        L = (coo_matrix((np.asarray(A.sum(axis=1)).ravel(), (np.arange(n), np.arange(n))), shape=(n, n)) - A).tocsr()
        U = np.nonzero(unknown)[0]
        K = np.nonzero(is_known)[0]
        Luu = L[U][:, U] + eps * identity(len(U), format="csr")   # eps: components without known vertices
        rhs = -(L[U][:, K] @ x[K]) + eps * x[K].mean(axis=0)
        x[U] = np.column_stack([spsolve(Luu.tocsc(), rhs[:, c]) for c in range(values.shape[1])])
    return (x[tri] * tex_bary[..., None]).sum(axis=1)


def _delta_e(bgr0: np.ndarray, bgr1: np.ndarray) -> np.ndarray:
    lab0 = cv2.cvtColor(np.clip(bgr0, 0, 255).astype(np.uint8)[None], cv2.COLOR_BGR2LAB)[0].astype(np.float32)
    lab1 = cv2.cvtColor(np.clip(bgr1, 0, 255).astype(np.uint8)[None], cv2.COLOR_BGR2LAB)[0].astype(np.float32)
    # OpenCV 8-bit Lab: L scaled by 255/100, a/b offset by 128
    lab0[:, 0] *= 100.0 / 255.0
    lab1[:, 0] *= 100.0 / 255.0
    return np.linalg.norm(lab0 - lab1, axis=1)


def _save_texture_debug(folder: Path, size, covered, W, visible_l, lows) -> None:
    """Visibility (red cam0, green cam1), weight share of cam0 and colour difference maps of the atlas."""
    folder.mkdir(parents=True, exist_ok=True)
    rows, cols = np.nonzero(covered)
    small = (1024, 1024)

    def put(values, cmap=None):
        img = np.zeros((size, size), np.float32)
        img[rows, cols] = values
        u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        col = cv2.applyColorMap(u8, cmap) if cmap is not None else cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        col[~covered] = 0
        return cv2.resize(col, small, interpolation=cv2.INTER_AREA)

    vis = np.zeros((size, size, 3), np.uint8)
    vis[rows, cols, 2] = visible_l[0] * 255
    vis[rows, cols, 1] = visible_l[1] * 255
    cv2.imwrite(str(folder / "texture_visibility_r_cam0_g_cam1.jpg"), cv2.resize(vis, small, interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(folder / "texture_weight_share_cam0.jpg"),
                put(W[0] / np.maximum(W.sum(0), 1e-9), cv2.COLORMAP_COOL))
    sel = (W[0] > 0) & (W[1] > 0)
    dE = np.zeros(len(rows), np.float32)
    dE[sel] = _delta_e(lows[0][sel], lows[1][sel])
    cv2.imwrite(str(folder / "texture_colour_diff_dE_over_20.jpg"), put(dE / 20.0, cv2.COLORMAP_INFERNO))


def add_gutter(tex: np.ndarray, covered: np.ndarray, iterations: int = 16) -> np.ndarray:
    """Pads the charts outward so bilinear filtering and mip-maps do not bleed the background."""
    out = tex.copy()
    known = covered.copy()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(iterations):
        grown = cv2.dilate(known.astype(np.uint8), kernel).astype(bool)
        ring = grown & ~known
        if not ring.any():
            break
        blurred = cv2.blur(np.where(known[..., None], out, 0).astype(np.float32), (3, 3))
        count = cv2.blur(known.astype(np.float32), (3, 3))
        out[ring] = (blurred[ring] / np.maximum(count[ring, None], 1e-6)).astype(np.uint8)
        known = grown
    return out


def bake_normal_map(low: trimesh.Trimesh, high: trimesh.Trimesh, size: int = 2048) -> np.ndarray:
    """
    Tangent-space normal map (BGR uint8, OpenGL convention) of the high-polygon surface
    on the UV atlas of the low-polygon mesh.
    """
    covered, f, b, pos, n_low = _texel_geometry(low, size)

    # Per-face tangent frame from the UV parametrisation.
    uv = np.asarray(low.visual.uv, dtype=np.float64)
    tri = low.faces
    p0, p1, p2 = (low.vertices[tri[:, i]] for i in range(3))
    t0, t1, t2 = (uv[tri[:, i]] for i in range(3))
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-12, 1e-12, r)
    T = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]
    T = T[f]
    T -= n_low * np.sum(n_low * T, axis=1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    B = np.cross(n_low, T)
    flip = np.sign(r)[f]
    B *= flip[:, None]

    # High-polygon normal at the texel position (inverse-distance weights over 4 neighbours).
    tree = cKDTree(high.vertices)
    dist, idx = tree.query(pos, k=4)
    wk = 1.0 / np.maximum(dist, 1e-6)
    n_high = (np.asarray(high.vertex_normals)[idx] * wk[..., None]).sum(axis=1)
    n_high /= np.linalg.norm(n_high, axis=1, keepdims=True) + 1e-12

    ts = np.stack([np.sum(n_high * T, 1), np.sum(n_high * B, 1), np.sum(n_high * n_low, 1)], axis=1)
    ts[:, 2] = np.maximum(ts[:, 2], 0.05)
    ts /= np.linalg.norm(ts, axis=1, keepdims=True)
    rgb = ((ts * 0.5 + 0.5) * 255).round().astype(np.uint8)

    nm = np.zeros((size, size, 3), np.uint8)
    nm[...] = (255, 128, 128)                       # flat normal in BGR
    rows, cols = np.nonzero(covered)
    nm[rows, cols] = rgb[:, ::-1]
    return add_gutter(nm, covered)


def flat_normal_map(size: int) -> np.ndarray:
    nm = np.zeros((size, size, 3), np.uint8)
    nm[...] = (255, 128, 128)
    return nm
