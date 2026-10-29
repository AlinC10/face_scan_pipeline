"""
GNM Head v3 (Google, Apache 2.0) — NumPy evaluator
==================================================
A parametric head model: identity (253: 170 head, 3 eyeball, 80 teeth), expression
(383: 100 + 100 eyes, 150 lower face, 32 tongue, 1 iris), 4 joints (neck, head, eyes),
linear blend skinning and pose correctives. 17 821 vertices, metres, Y up, the face
looking along +Z. Skin, eyes, teeth / gums and tongue; 46 vertex groups (ears, nose, ...).

The official package (github.com/google/GNM) depends on TensorFlow and Jupyter; this
module evaluates the same model from the released `gnm_head.npz` with NumPy only, following
gnm/shape/gnm_common.py:

    V_bind = T + sum_i id_i B_id_i + sum_e ex_e B_ex_e
    J_bind = T_J + sum_i id_i B_J_i
    V_bind += P (vec(R_j - I))                         pose correctives
    G_j = G_parent(j) [R_j | J_j - J_parent(j)]       (root: [R_0 | J_0 + t])
    v = sum_j w_jv (R_j^world v + t_j^world - R_j^world J_j)

Model file: assets/gnm_head.npz (Hugging Face google/gnm-v3, v3_0), downloaded on first use and
checked against the published SHA-256. Landmarks: assets/gnm_head_sparse_68.txt (iBUG-68
order, 3 vertex indices + barycentric weights per landmark, from the GNM repository).
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

log = logging.getLogger("Photogrammetry.GNM")

ASSETS = Path(__file__).resolve().parent.parent / "assets"
GNM_PATH = ASSETS / "gnm_head.npz"
GNM_URL = "https://huggingface.co/google/gnm-v3/resolve/main/v3_0/gnm_head.npz"
GNM_SHA256 = "61d78bbfb4ad8e0b38495804a4caef3214d3df00f8c3f68761e63b41ce3747eb"
LANDMARKS_68_PATH = ASSETS / "gnm_head_sparse_68.txt"


def ensure_model(path: Union[str, Path] = GNM_PATH) -> Path:
    """The model file, downloaded and verified when missing (the Docker image ships it)."""
    p = Path(path)
    if not p.is_file():
        log.info("Downloading GNM Head v3 (53 MB) from %s", GNM_URL)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(GNM_URL, tmp)
        tmp.replace(p)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != GNM_SHA256:
        raise ValueError(f"{p} has SHA-256 {digest}, expected {GNM_SHA256} (corrupt or different version)")
    return p


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrices (as gnm_common.axis_angle_to_rotation_matrix)."""
    a = np.asarray(axis_angle, dtype=np.float64)
    angle = np.sqrt(np.maximum(np.sum(a * a, axis=-1, keepdims=True), 1e-16))
    k = a / angle
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = np.sin(angle)[..., None], np.cos(angle)[..., None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


@dataclass
class GNMParams:
    identity: np.ndarray            # (253,)
    expression: np.ndarray          # (383,)
    rotations: np.ndarray           # (4, 3) axis-angle per joint
    translation: np.ndarray         # (3,)
    scale: float = 1.0              # metres -> output units (e.g. 1000 for mm), applied last

    def copy(self) -> "GNMParams":
        return GNMParams(self.identity.copy(), self.expression.copy(), self.rotations.copy(),
                         self.translation.copy(), self.scale)


class GNMHead:
    def __init__(self, path: Union[str, Path] = GNM_PATH):
        z = np.load(ensure_model(path), allow_pickle=False)
        self.template = z["template_vertex_positions"].astype(np.float64)          # (V, 3)
        self.id_basis = z["vertex_identity_basis"].astype(np.float64)               # (I, V, 3)
        self.ex_basis = z["expression_basis"].astype(np.float64)                    # (E, V, 3)
        self.joint_template = z["template_joint_positions"].astype(np.float64)      # (J, 3)
        self.joint_id_basis = z["joint_identity_basis"].astype(np.float64)          # (I, J, 3)
        self.joint_parents = z["joint_parent_indices"].astype(int)
        self.joint_names = [str(n) for n in z["joint_names"]]
        self.skin_weights = z["skinning_weights"].astype(np.float64)                # (J, V)
        self.pose_regressor = z["pose_correctives_regressor"].astype(np.float64)    # (J*9, V*3)
        self.triangles = z["triangles"].astype(np.int64)
        self.triangle_uvs = z["triangle_uvs"].astype(np.float64)                    # (T, 3, 2), per region
        self.quads = z["quads"].astype(np.int64)
        self.group_names = [str(n) for n in z["vertex_group_names"]]
        self.groups = z["vertex_groups"].astype(np.float32)
        self.identity_names = [str(n) for n in z["identity_names"]]
        self.expression_names = [str(n) for n in z["expression_names"]]
        self.mirror = z["mirror_indices"].astype(int)
        lm = np.loadtxt(LANDMARKS_68_PATH)
        self.lm68_idx = lm[:, ::2].astype(int)
        self.lm68_w = lm[:, 1::2].astype(np.float64)

    @property
    def n_vertices(self) -> int:
        return len(self.template)

    @property
    def identity_dim(self) -> int:
        return len(self.id_basis)

    @property
    def expression_dim(self) -> int:
        return len(self.ex_basis)

    def zero_params(self) -> GNMParams:
        return GNMParams(np.zeros(self.identity_dim), np.zeros(self.expression_dim),
                         np.zeros((len(self.joint_names), 3)), np.zeros(3))

    # ------------------------------------------------------------------ evaluation
    def bind_vertices(self, identity: np.ndarray, expression: np.ndarray) -> np.ndarray:
        return (self.template + np.einsum("i,ivc->vc", identity, self.id_basis)
                + np.einsum("e,evc->vc", expression, self.ex_basis))

    def bind_joints(self, identity: np.ndarray) -> np.ndarray:
        return self.joint_template + np.einsum("i,ijc->jc", identity, self.joint_id_basis)

    def pose_correctives(self, rotations: np.ndarray) -> np.ndarray:
        feats = (rodrigues(rotations) - np.eye(3)).reshape(-1)
        return (feats @ self.pose_regressor).reshape(-1, 3)

    def joint_transforms(self, joints: np.ndarray, rotations: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """World 4x4 transform of every joint (gnm_common.joint_transforms_world)."""
        R = rodrigues(rotations)
        G = []
        for j in range(len(joints)):
            local = np.eye(4)
            local[:3, :3] = R[j]
            if j == 0:
                local[:3, 3] = joints[0] + translation
            else:
                local[:3, 3] = joints[j] - joints[self.joint_parents[j]]
            G.append(local if j == 0 else G[self.joint_parents[j]] @ local)
        return np.stack(G)

    def skin(self, vertices: np.ndarray, joints: np.ndarray, rotations: np.ndarray, translation: np.ndarray) -> np.ndarray:
        G = self.joint_transforms(joints, rotations, translation)
        Rw, tw = G[:, :3, :3], G[:, :3, 3]
        t_bind = tw - np.einsum("jmn,jn->jm", Rw, joints)
        Rv = np.einsum("jv,jmn->vmn", self.skin_weights, Rw)
        tv = np.einsum("jv,jm->vm", self.skin_weights, t_bind)
        return np.einsum("vmn,vn->vm", Rv, vertices) + tv

    def __call__(self, p: GNMParams) -> np.ndarray:
        """Vertices (V, 3) in metres x p.scale."""
        v = self.bind_vertices(p.identity, p.expression) + self.pose_correctives(p.rotations)
        return self.skin(v, self.bind_joints(p.identity), p.rotations, p.translation) * p.scale

    def landmarks68(self, vertices: np.ndarray) -> np.ndarray:
        return np.einsum("lk,lkc->lc", self.lm68_w, vertices[self.lm68_idx])

    # ------------------------------------------------------------------ groups and meshes
    def group(self, *names: str, threshold: float = 0.5) -> np.ndarray:
        mask = np.zeros(self.n_vertices, bool)
        for n in names:
            mask |= self.groups[self.group_names.index(n)] > threshold
        return mask

    @cached_property
    def skin_triangles(self) -> np.ndarray:
        """Triangles of the skin region (no eyes, teeth, tongue)."""
        skin = self.group("skin")
        return np.nonzero(skin[self.triangles].all(axis=1))[0]
