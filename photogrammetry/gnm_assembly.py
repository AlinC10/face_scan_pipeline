"""
Whole Head from GNM + the Dense Face (plan P2.3-P2.8)
====================================================
  1. GNM Head is fitted to the measured face (gnm_fit.py): the measured cells of the height field
     and the 468 MediaPipe landmarks through the MediaPipe -> GNM embedding.
  2. Detail: the fitted GNM skin is subdivided twice (~0.6 mm edges on the face) and every vertex
     moves along its normal onto the measured height field, with a 15 mm ramp from the border of
     the measured face and a fade on grazing normals, so there is no step where the data ends.
     The low mesh (GNM topology) takes the same displacement at its vertices.
  3. Expression: "capture" keeps the fitted expression; "neutral" evaluates GNM with the expression
     at zero and applies the same per-vertex displacement (identity and detail kept).
  4. Solid for printing: the neck is cut flat 35 mm below the chin and the section is closed.
  5. Animatable low GLB: GNM topology with its own UV atlas (skin), eyes, teeth, tongue, skeleton
     (neck, head, eyes) and morph targets (semantic expressions + "neutral").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh

from photogrammetry.gnm_fit import GNMFitResult, GNMFitter
from photogrammetry.gnm_head import ASSETS, GNMHead

log = logging.getLogger("Photogrammetry.GNMAssembly")

EMBEDDING_PATH = ASSETS / "gnm_mediapipe_embedding.npz"
EXPRESSIONS_PATH = ASSETS / "gnm_semantic_expressions.npz"
DETAIL_RAMP_MM = 15.0
MAX_DISPLACEMENT_MM = 6.0
NEAR_FULL_MM = 4.0                 # vertices within this distance of the measured surface take the full detail
NEAR_ZERO_MM = 8.0                 # ... and none beyond this one
NECK_CUT_BELOW_CHIN_MM = 35.0
HAIR_THICKNESS_MM = 5.0            # hair cap (P3.3): the bald GNM scalp raised by this much where nothing is measured
GLB_EXPRESSIONS = ("HAPPY", "SMILE_WIDE", "SURPRISE", "WINK_LEFT", "WINK_RIGHT", "SQUINT", "PUCKER", "DISGUST",
                   "CORNERS_DOWN", "BLOW", "MOUTH_LEFT", "MOUTH_RIGHT")
PART_COLOURS = {"eyes": (0.92, 0.91, 0.88, 1.0), "teeth": (0.93, 0.9, 0.82, 1.0), "tongue": (0.78, 0.42, 0.42, 1.0)}


@dataclass
class HeadAssembly:
    fit: GNMFitResult
    high: trimesh.Trimesh                          # detailed skin, head frame (mm)
    low: trimesh.Trimesh                           # GNM skin topology, displaced (mm), process=False
    low_corner_uv: np.ndarray                      # (F, 3, 2) GNM skin UVs per corner
    parts: Dict[str, trimesh.Trimesh]              # eyes / teeth / tongue (mm)
    skin_vertex_ids: np.ndarray                    # GNM vertex id of every low vertex
    expression: str
    stats: Dict[str, float] = field(default_factory=dict)
    hair_low: Optional[np.ndarray] = None          # hair-cap weight (0-1) of every low vertex


def fit_gnm(surface, fitter: Optional[GNMFitter] = None) -> Tuple[GNMFitter, GNMFitResult]:
    """GNM on the measured face: measured height-field cells + the MediaPipe landmarks (embedding)."""
    fitter = fitter or GNMFitter()
    emb = np.load(EMBEDDING_PATH)
    lm = surface.landmarks_head[:468]
    lw = np.where(emb["residual_mm"] < 3.0, 1.0, 0.0)
    res = fitter.fit(surface.measured_points, surface.measured_weights,
                     landmarks=(lm, emb["vid"], emb["bary"], lw), iterations=12,
                     lambda_identity=1.0, lambda_expression=3.0, landmark_weight=0.1)
    log.info("GNM fit: surface RMS %.2f mm (median %.2f), landmarks median %.1f mm, identity %.2f sigma",
             res.stats["surface_rms_mm"], res.stats["surface_median_mm"], res.stats.get("landmark_median_mm", -1),
             res.stats["identity_norm"])
    return fitter, res


def _vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    return np.asarray(trimesh.Trimesh(V, F, process=False).vertex_normals)


def _detail_ramp(surface) -> np.ndarray:
    """Per height-field cell: 0 at the border of the measured face, 1 beyond DETAIL_RAMP_MM inside it."""
    valid = np.isfinite(surface.height_field).astype(np.uint8)
    dist = cv2.distanceTransform(valid, cv2.DIST_L2, 5) * surface.grid_resolution_mm
    t = np.clip(dist / DETAIL_RAMP_MM, 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def normal_displacement(surface, V: np.ndarray, N: np.ndarray, ramp: np.ndarray, faces: Optional[np.ndarray] = None,
                        smooth_iterations: int = 4):
    """
    Displacement along the normals towards the height field, and its weight. The vertical gap dz becomes the
    normal distance dz * n_z (exact for a surface parallel to the model's, stable on steep slopes where dividing
    by n_z amplified the noise), faded on grazing normals, at the border of the face and away from the surface
    (back of the head, inside of the mouth, eye sockets), then smoothed over the mesh neighbours.
    """
    z0 = surface.sample_height(V[:, :2])
    inside = np.isfinite(z0)
    gap = np.where(inside, z0 - V[:, 2], 0.0)
    nz = np.clip(N[:, 2], 0.0, 1.0)
    d = np.clip(gap * nz, -MAX_DISPLACEMENT_MM, MAX_DISPLACEMENT_MM)
    res = surface.grid_resolution_mm
    col = np.clip(((V[:, 0] - surface.grid_lo[0]) / res).round().astype(int), 0, ramp.shape[1] - 1)
    row = np.clip(((surface.grid_hi[1] - V[:, 1]) / res).round().astype(int), 0, ramp.shape[0] - 1)
    facing = np.clip((nz - 0.5) / 0.3, 0.0, 1.0)
    near = np.clip(1.0 - (np.abs(gap) - NEAR_FULL_MM) / (NEAR_ZERO_MM - NEAR_FULL_MM), 0.0, 1.0)
    w = np.where(inside, ramp[row, col], 0.0) * facing * near
    dw = d * w
    if faces is not None and smooth_iterations:
        # neighbour averaging (uniform Laplacian) of the weighted displacement: removes vertex-level noise
        from scipy.sparse import coo_matrix
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        n = len(V)
        A = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel() + 1.0
        for _ in range(smooth_iterations):
            dw = (dw + A @ dw) / deg
    return dw, w


def _subdivide(V: np.ndarray, F: np.ndarray, times: int = 2):
    for _ in range(times):
        V, F = trimesh.remesh.subdivide(V, F)
    return V, F


def _laplacian_smooth(values: np.ndarray, F: np.ndarray, iterations: int) -> np.ndarray:
    from scipy.sparse import coo_matrix
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    n = len(values)
    A = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel() + 1.0
    v = values.astype(np.float64)
    for _ in range(iterations):
        v = (v + A @ v) / deg
    return v


def hair_region(g, identity: np.ndarray, used: np.ndarray, F_low: np.ndarray) -> np.ndarray:
    """
    Scalp weight (0-1) of the skin vertices `used`, from the GNM regions in the model frame (y up, z forward):
    the exterior skin outside the face mask and the ears, above the top of the ears, or behind them down to
    the ear lobes (the nape); smoothed over the mesh so the cap has no step at its border.
    """
    Vm = g.bind_vertices(identity, np.zeros(g.expression_dim))[used] * 1000.0
    ears = g.group("ears")[used]
    ear_top, ear_bottom = Vm[ears, 1].max(), Vm[ears, 1].min()
    ear_z = Vm[ears, 2].mean()
    base = g.group("skin_exterior")[used] & ~g.group("hockey_mask", "ears", "eyes", "eye_sockets")[used]
    scalp = base & ((Vm[:, 1] > ear_top) | ((Vm[:, 2] < ear_z) & (Vm[:, 1] > ear_bottom)))
    w = _laplacian_smooth(scalp.astype(np.float64), F_low, 12)
    return np.clip((w - 0.2) / 0.6, 0.0, 1.0)


def assemble_head(surface, fitter: GNMFitter, fit: GNMFitResult, expression: str = "capture",
                  high_max_faces: int = 200_000, decimate=None, hair_thickness_mm: float = HAIR_THICKNESS_MM) -> HeadAssembly:
    g = fitter.g
    skin_tri = g.triangles[g.skin_triangles]
    used = np.unique(skin_tri)
    remap = -np.ones(g.n_vertices, int)
    remap[used] = np.arange(len(used))
    F_low = remap[skin_tri]

    V_cap = fit.vertices                                         # fitted expression, head frame (mm)
    if expression == "neutral":
        p = fit.params.copy()
        p.expression[:] = 0.0
        V_out = (g.bind_vertices(p.identity, p.expression) * 1000.0) @ fit.R.T + fit.t
    else:
        V_out = V_cap
    ramp = _detail_ramp(surface)

    # Low: GNM topology, detail at the vertices
    N_cap = _vertex_normals(V_cap[used], F_low)
    d_low, w_low = normal_displacement(surface, V_cap[used], N_cap, ramp, F_low, smooth_iterations=1)
    N_out = _vertex_normals(V_out[used], F_low)
    # Hair cap: raised where the scalp is not measured (the stereo already holds the hair it sees)
    hair = hair_region(g, fit.params.identity, used, F_low) if hair_thickness_mm > 0 else np.zeros(len(used))
    hair_low = hair * (1.0 - np.clip(w_low, 0.0, 1.0))
    V_low = V_out[used] + (d_low + hair_thickness_mm * hair_low)[:, None] * N_out
    low = trimesh.Trimesh(V_low, F_low, process=False)

    # High: subdivided twice, detail from the height field (computed on the captured expression)
    S_cap, F_high = _subdivide(V_cap[used], F_low)
    S_out, _ = _subdivide(V_out[used], F_low)
    d_high, w_high = normal_displacement(surface, S_cap, _vertex_normals(S_cap, F_high), ramp, F_high)
    hair_high = _subdivide(np.c_[hair, np.zeros((len(hair), 2))], F_low)[0][:, 0] * (1.0 - np.clip(w_high, 0.0, 1.0))
    S_out = S_out + (d_high + hair_thickness_mm * hair_high)[:, None] * _vertex_normals(S_out, F_high)
    high = trimesh.Trimesh(S_out, F_high, process=True)
    n_sub = len(high.faces)
    if decimate is not None and len(high.faces) > high_max_faces:
        high = decimate(high, high_max_faces)

    # Separate parts (eyes without the corneas, teeth, tongue), same pose as the output skin
    V_parts = (g.bind_vertices(fit.params.identity, fit.params.expression if expression != "neutral"
                               else np.zeros(g.expression_dim)) * 1000.0) @ fit.R.T + fit.t
    parts = {}
    for name, groups in (("eyes", ("left_eye", "right_eye")), ("teeth", ("upper_teeth_and_gums", "lower_teeth_and_gums")),
                         ("tongue", ("tongue",))):
        mask = g.group(*groups) & ~g.group("eye_exteriors")
        tris = g.triangles[mask[g.triangles].all(axis=1)]
        if len(tris):
            ids = np.unique(tris)
            loc = -np.ones(g.n_vertices, int)
            loc[ids] = np.arange(len(ids))
            m = trimesh.Trimesh(V_parts[ids], loc[tris], process=False)
            m.metadata["gnm_vertex_ids"] = ids
            parts[name] = m
    stats = {"expression": expression, "low_vertices": int(len(V_low)), "high_faces_subdivided": int(n_sub),
             "high_faces": int(len(high.faces)), "detail_weighted_fraction_high": round(float(np.mean(w_high > 0.5)), 3),
             "displacement_p95_mm": round(float(np.percentile(np.abs(d_high[w_high > 0.5]), 95)), 2)
             if (w_high > 0.5).any() else 0.0,
             "hair_cap_mm": hair_thickness_mm, "hair_cap_fraction_low": round(float(np.mean(hair_low > 0.5)), 3)}
    log.info("GNM head assembled (%s): low %d vertices, high %d -> %d faces, detail on %.0f%% of the high vertices",
             expression, len(V_low), n_sub, len(high.faces), 100 * stats["detail_weighted_fraction_high"])
    return HeadAssembly(fit, high, low, g.triangle_uvs[g.skin_triangles], parts, used, expression, stats, hair_low)


# ---------------------------------------------------------------------------------------------- solids
def head_solid(mesh: trimesh.Trimesh, chin_y: float, below_mm: float = NECK_CUT_BELOW_CHIN_MM) -> trimesh.Trimesh:
    """
    Closed bust for printing: the faces above y = chin - below_mm are kept, the vertices of the open
    border are moved onto that plane (flat base) and the section is closed by a fan from its centre.
    """
    m = mesh.copy()
    m.merge_vertices()
    y_cut = chin_y - below_mm
    keep = (m.vertices[m.faces][:, :, 1] >= y_cut).all(axis=1)
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: len(p.faces))
    V = m.vertices.copy()
    edges = m.edges[trimesh.grouping.group_rows(m.edges_sorted, require_count=1)]
    if len(edges):
        border = np.unique(edges)
        low = border[V[border, 1] < y_cut + 0.5 * below_mm]          # the neck border (not stray holes)
        V[low, 1] = y_cut
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(edges.tolist())
        faces = [m.faces]
        for comp in nx.connected_components(G):
            centre = V[list(comp)].mean(axis=0)
            V = np.vstack([V, centre])
            c = len(V) - 1
            loop_edges = [e for e in edges if e[0] in comp]
            faces.append(np.array([[b, a, c] for a, b in loop_edges]))   # reversed border edges
        m = trimesh.Trimesh(V, np.vstack(faces), process=True)
    trimesh.repair.fix_normals(m)
    return m


# ---------------------------------------------------------------------------------------------- UVs
def split_by_corner_uv(V: np.ndarray, F: np.ndarray, corner_uv: np.ndarray):
    """Vertices duplicated where the UV differs between the corners (seams): (V', F', UV', source vertex)."""
    keys = np.concatenate([F.reshape(-1, 1).astype(np.float64), np.round(corner_uv.reshape(-1, 2) * 1e6)], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    src = uniq[:, 0].astype(int)
    uv = uniq[:, 1:] / 1e6
    return V[src], inv.reshape(F.shape), uv, src


# ---------------------------------------------------------------------------------------------- animated GLB
def write_animated_glb(path, assembly: HeadAssembly, fitter: GNMFitter, low_uv: trimesh.Trimesh, low_src: np.ndarray,
                       tex_bgr: np.ndarray, normal_bgr: np.ndarray, factor: float,
                       expressions=GLB_EXPRESSIONS) -> Path:
    """
    The low head as an animatable GLB: skin (GNM topology, texture + normal map), eyes, teeth, tongue; skeleton
    neck -> head -> eyes with the GNM skinning weights; morph targets = the semantic expressions (and NEUTRAL when
    the capture expression is kept), as deltas of the GNM evaluated in the bind pose.
    low_src: GNM vertex id of every vertex of low_uv (UV seams duplicate vertices).
    """
    from photogrammetry.gltf_writer import GLTFMaterial, GLTFPrimitive, GLTFSkeleton, write_glb

    g = fitter.g
    fit = assembly.fit
    ex_base = np.zeros(g.expression_dim) if assembly.expression == "neutral" else fit.params.expression
    lib = np.load(EXPRESSIONS_PATH)
    names = [str(n) for n in lib["names"]]
    targets, target_names = [], []
    base = g.bind_vertices(fit.params.identity, ex_base) * 1000.0
    for n in expressions:
        if n in names:
            ex = ex_base + lib["expressions"][names.index(n)].astype(np.float64)
            targets.append(((g.bind_vertices(fit.params.identity, ex) * 1000.0) - base) @ fit.R.T * factor)
            target_names.append(n)
    if assembly.expression != "neutral":
        targets.append(((g.bind_vertices(fit.params.identity, np.zeros(g.expression_dim)) * 1000.0) - base) @ fit.R.T * factor)
        target_names.append("NEUTRAL")
    W = g.skin_weights.T                                            # (V, 4)
    joints = np.tile(np.arange(4, dtype=np.uint16), (g.n_vertices, 1))

    def prim(V, F, ids, material, uvs=None, normals=None):
        return GLTFPrimitive(V * factor, F, material, normals=normals, uvs=uvs, joints=joints[ids], weights=W[ids],
                             morph_deltas=[t[ids] for t in targets])

    skin_ids = assembly.skin_vertex_ids[low_src]
    prims = [prim(low_uv.vertices, low_uv.faces, skin_ids, 0, uvs=np.asarray(low_uv.visual.uv),
                  normals=np.asarray(low_uv.vertex_normals))]
    mats = [GLTFMaterial("skin", base_color_png=cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB),
                         normal_png=cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB))]
    for name, part in assembly.parts.items():
        mats.append(GLTFMaterial(name, base_color=PART_COLOURS.get(name, (0.8, 0.8, 0.8, 1.0)), roughness=0.4))
        prims.append(prim(part.vertices, part.faces, part.metadata["gnm_vertex_ids"], len(mats) - 1,
                          normals=np.asarray(part.vertex_normals)))
    J = (g.bind_joints(fit.params.identity) * 1000.0) @ fit.R.T + fit.t
    skeleton = GLTFSkeleton(g.joint_names, list(g.joint_parents), J * factor)
    return write_glb(Path(path), prims, mats, skeleton, target_names, name="head")
