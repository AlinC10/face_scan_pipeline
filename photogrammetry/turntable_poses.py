#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
turntable_poses.py — calculează pozițiile camerelor pentru un scanner cu turntable,
în loc să le lase pe seama StructureFromMotion din Meshroom.

IDEEA
-----
Camerele stau pe o platformă fixă (urcă doar pe tijă, între niveluri). Platanul se rotește
cu un pas cunoscut. Deci pozițiile tuturor pozelor sunt date de un model mic:

    poziția pozei (cameră c, nivel L, pas k) =
        rotație în jurul axei platanului cu  θ = semn · k · pas + decalaj_nivel[L]
        aplicată camerei c, ridicată cu (L − primul_nivel) · urcare  pe direcția tijei

Necunoscutele modelului (≈ 25 de numere în total):
  * pentru fiecare cameră: orientarea (3) și poziția (3) la nivelul de jos, unghi 0;
  * axa platanului: punctul unde taie discul (2) și înclinarea ei (2);
  * direcția de urcare a platformei (2);
  * pasul real de rotație (1) și câte un mic decalaj de unghi pentru fiecare nivel în plus;
  * parametrii interni ai fiecărei camere: focală, centru optic, distorsiune (6).

Ele se estimează din colțurile AprilTag-urilor de pe disc (poziții cunoscute exact, din PDF),
văzute în toate pozele. Rezultatul: toate cele 80 de poziții sunt consecvente între ele
(pas exact, rig fix), inclusiv pentru pozele în care obiectul acoperă tag-urile.

SISTEMUL DE COORDONATE (același ca în SfMTransform / .mg-ul tău)
  originea = centrul discului cu markeri, Y în sus (normala discului),
  X spre dreapta pe foaie, Z „spre tine” pe foaie, unități în METRI.

CE PRIMEȘTE
  --input       cameraInit.sfm (ieșirea nodului CameraInit, format JSON AliceVision)
  --layout      geometria tag-urilor (JSON), implicit platan_260mm_layout.json de lângă script

CE SCRIE
  --output      cameras_known.sfm : camerele cu parametri calibrați + pozițiile tuturor pozelor
  --calibOut    turntable_calib.json : modelul complet, refolosibil la alte scanări
  --report      raport text cu verificări (eroare în pixeli, unghiuri, distanțe)

DEPENDENȚE: python3, numpy, scipy, opencv-python (>= 4.7, are cv2.aruco.ArucoDetector).

Numele opțiunilor sunt în stilul Meshroom (--input, --stepDeg ...) ca scriptul să poată fi rulat
și direct din nodul Meshroom „TurntablePoses”, și manual din terminal.
"""

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Windows: când Meshroom trimite ieșirea în fișierul de log, consola folosește cp1252 și
# diacriticele (ă, ș, ț) dau UnicodeEncodeError. Forțăm UTF-8 pentru ieșire.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

try:
    import cv2
except ImportError:  # mesaj clar în loc de traceback
    sys.exit("[eroare] lipsește OpenCV: pip install opencv-python-headless")
try:
    from scipy.optimize import least_squares
except ImportError:
    sys.exit("[eroare] lipsește SciPy: pip install scipy")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LAYOUT = os.path.join(SCRIPT_DIR, "platan_260mm_layout.json")


# =============================================================================================
# 1. Geometria discului cu tag-uri
# =============================================================================================

def ring_layout(n=12, radius_m=0.1, size_m=0.03, start_deg=15.0):
    """Generează geometria unui inel de tag-uri, la fel ca PDF-ul tău (verificat cu OpenCV).

    Tag-ul i stă la unghiul α = start + i·360/n (măsurat de la +X spre −Z), rotit cu latura pe rază.
    Colțurile sunt în ordinea OpenCV: 0=stânga-sus, 1=dreapta-sus, 2=dreapta-jos, 3=stânga-jos
    (în sistemul propriu al tag-ului).
    Folosește-o dacă tipărești un disc nou cu alți parametri (în loc de fișierul JSON)."""
    tags = []
    h = size_m / 2.0
    for i in range(n):
        a = math.radians(start_deg + i * 360.0 / n)
        b = a - math.pi / 2.0                      # rotația tag-ului (în planul foii)
        ca, sa = math.cos(b), math.sin(b)
        cx, cy = radius_m * math.cos(a), radius_m * math.sin(a)   # coordonate „pe foaie” (y în sus)
        corners = []
        for lx, ly in ((-h, h), (h, h), (h, -h), (-h, -h)):
            px = ca * lx - sa * ly + cx
            py = sa * lx + ca * ly + cy
            corners.append([px, 0.0, -py])        # foaie → lume: X = x, Z = −y
        tags.append({"id": i, "center": [cx, 0.0, -cy], "corners": corners})
    return {"family": "tag16h5", "units": "m", "tags": tags}


def load_layout(path, scale):
    """Citește geometria tag-urilor și o înmulțește cu `scale` (corecția de print)."""
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            lay = json.load(f)
    else:
        if path:
            print("[atenție] nu găsesc layout-ul %s, folosesc inelul implicit (12 tag-uri, r=100 mm, 30 mm)" % path)
        lay = ring_layout()
    corners = {}
    for t in lay["tags"]:
        corners[int(t["id"])] = np.asarray(t["corners"], dtype=np.float64) * scale
    return corners


# =============================================================================================
# 2. Detecția tag-urilor (rulează în paralel, câte un proces pe imagine)
# =============================================================================================

def detect_tags(path):
    """Întoarce {id: colțuri 4×2 în pixeli} pentru o imagine. Rulează într-un proces separat."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return path, None
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    prm = cv2.aruco.DetectorParameters()
    # tag16h5 are puține combinații → poate da detecții false. Nu acceptăm biți corectați.
    prm.errorCorrectionRate = 0.0
    prm.minMarkerPerimeterRate = 0.01           # acceptă și tag-uri mici (≈ 10 px latură la 4056 px)
    prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    prm.cornerRefinementWinSize = 5
    det = cv2.aruco.ArucoDetector(dic, prm)
    corners, ids, _ = det.detectMarkers(img)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            out[int(i)] = c.reshape(4, 2).astype(np.float64)
    return path, out


# =============================================================================================
# 3. Mici utilitare de geometrie (numpy)
# =============================================================================================

def rodrigues(v):
    """Vector de rotație (axă·unghi, radiani) → matrice 3×3."""
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def rot_to_vec(R):
    v, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return v.reshape(3)


def unit_from_tilts(tx, tz):
    """Direcție aproape verticală (+Y) dată prin două înclinări mici (radiani)."""
    v = np.array([math.tan(tx), 1.0, math.tan(tz)])
    return v / np.linalg.norm(v)


def tilts_from_unit(u):
    u = u / np.linalg.norm(u)
    if u[1] < 0:
        u = -u
    return math.atan2(u[0], u[1]), math.atan2(u[2], u[1])


def project(R, C, K, D, X):
    """Proiectează punctele X (N×3, lume) într-o cameră. Același model ca AliceVision
    pinhole + radialk3:  x_cam = R·(X − C);  p = x/z;  p' = p·(1 + k1 r² + k2 r⁴ + k3 r⁶);
    pixel = f·p' + (cx, cy).   K = (f, cx, cy) în pixeli, D = (k1, k2, k3)."""
    Xc = (X - C) @ R.T
    z = Xc[:, 2]
    x = Xc[:, 0] / z
    y = Xc[:, 1] / z
    r2 = x * x + y * y
    d = 1.0 + D[0] * r2 + D[1] * r2 * r2 + D[2] * r2 * r2 * r2
    return np.stack([K[0] * x * d + K[1], K[0] * y * d + K[2]], axis=1), z


def fit_circle_3d(P):
    """Cerc prin puncte 3D: (centru, normală, rază). Plan prin SVD + cerc Kasa în plan."""
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c)
    n, u, v = Vt[2], Vt[0], Vt[1]
    xy = np.stack([(P - c) @ u, (P - c) @ v], axis=1)
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    b = -(xy ** 2).sum(axis=1)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = -D / 2, -E / 2
    r = math.sqrt(max(cx * cx + cy * cy - F, 0.0))
    return c + cx * u + cy * v, n, r


def mean_rotation(Rs):
    """Media mai multor rotații (proiecția sumei pe SO(3))."""
    U, _, Vt = np.linalg.svd(sum(Rs))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, 2] *= -1
        R = U @ Vt
    return R


# =============================================================================================
# 4. Modelul turntable-ului: parametri ↔ vector de optimizare
# =============================================================================================

class TurntableModel:
    """Ține toți parametrii modelului și știe să calculeze poziția oricărei poze."""

    def __init__(self, cams, levels, first_level, step_deg, lift_m, sign=1.0):
        self.cams = list(cams)              # id-urile camerelor (intrinsicId din Meshroom)
        self.levels = sorted(levels)
        self.first_level = first_level
        self.sign = sign                    # sensul de rotație (+1 / −1), ales automat
        self.lift_m = lift_m                # urcarea nominală pe nivel (metri)
        self.p = {}                         # parametrii, pe nume
        self.free = {}                      # care parametri se optimizează
        self.p["step"] = np.array([step_deg])
        self.p["axis_p"] = np.zeros(2)      # (x, z) unde axa taie planul discului
        self.p["axis_t"] = np.zeros(2)      # înclinarea axei față de normala discului
        self.p["lift_t"] = np.zeros(2)      # înclinarea direcției de urcare
        self.p["lift_s"] = np.array([1.0])  # factor de corecție al urcării (1 = exact cât ai setat)
        self.level_pose = False             # True = platforma poate avea mică rotație/deplasare pe nivel
        self.tag_refine = False             # True = se corectează înălțimea/înclinarea fiecărui tag
        self.tag_ctr = {}                   # centrele tag-urilor (din layout), pentru corecție
        for L in self.levels:
            if L != self.levels[0]:
                self.p["off_%d" % L] = np.zeros(1)   # decalaj de unghi al nivelului (grade)
                self.p["lvlR_%d" % L] = np.zeros(3)  # rotația platformei la nivelul L (față de nivelul de jos)
                self.p["lvlT_%d" % L] = np.zeros(3)  # deplasarea în plus a platformei (pe lângă urcare)
        for c in self.cams:
            self.p["R_%s" % c] = np.zeros(3)
            self.p["C_%s" % c] = np.zeros(3)
            self.p["K_%s" % c] = np.zeros(3)         # f, cx, cy (pixeli)
            self.p["D_%s" % c] = np.zeros(3)         # k1, k2, k3
        for k in self.p:
            self.free[k] = np.ones(len(self.p[k]), dtype=bool)

    # --- împachetare pentru optimizator ---
    def keys(self):
        return sorted(self.p)

    def pack(self):
        return np.concatenate([self.p[k][self.free[k]] for k in self.keys()]) if self.n_free() else np.zeros(0)

    def unpack(self, x):
        i = 0
        for k in self.keys():
            m = self.free[k]
            n = int(m.sum())
            if n:
                self.p[k] = self.p[k].copy()
                self.p[k][m] = x[i:i + n]
                i += n

    def n_free(self):
        return int(sum(m.sum() for m in self.free.values()))

    # --- geometrie ---
    def axis(self):
        a = unit_from_tilts(*self.p["axis_t"])
        p = np.array([self.p["axis_p"][0], 0.0, self.p["axis_p"][1]])
        return p, a

    def theta(self, level, index):
        """Unghiul platanului (radiani) pentru poza (nivel, pas)."""
        deg = self.sign * index * self.p["step"][0]
        if level != self.levels[0]:
            deg += self.p["off_%d" % level][0]
        return math.radians(deg)

    def pose(self, cam, level, index):
        """Poziția pozei în sistemul discului: (R lume→cameră, centrul camerei C)."""
        p, a = self.axis()
        Ra = rodrigues(a * self.theta(level, index))
        u = unit_from_tilts(*self.p["lift_t"])
        Rc = rodrigues(self.p["R_%s" % cam])
        C_lab = self.p["C_%s" % cam] + (level - self.levels[0]) * self.lift_m * self.p["lift_s"][0] * u
        if self.level_pose and level != self.levels[0]:
            # platforma (ambele camere împreună) se rotește puțin în jurul centrului ei și se mai deplasează
            c0 = np.mean([self.p["C_%s" % c] for c in self.cams], axis=0)
            Rl = rodrigues(self.p["lvlR_%d" % level])
            C_lab = c0 + Rl @ (self.p["C_%s" % cam] - c0) + (C_lab - self.p["C_%s" % cam]) + self.p["lvlT_%d" % level]
            Rc = Rc @ Rl.T
        R = Rc @ Ra
        C = p + Ra.T @ (C_lab - p)
        return R, C

    def to_json(self):
        return {"sign": self.sign, "levels": self.levels, "firstLevel": self.first_level, "levelPose": self.level_pose, "tagRefine": self.tag_refine,
                "liftNominal_m": self.lift_m, "cams": self.cams,
                "params": {k: v.tolist() for k, v in self.p.items()}}

    @staticmethod
    def from_json(d):
        m = TurntableModel(d["cams"], d["levels"], d["firstLevel"], d["params"]["step"][0],
                           d["liftNominal_m"], d["sign"])
        for k, v in d["params"].items():
            m.p[k] = np.asarray(v, dtype=np.float64)
            m.free[k] = np.ones(len(v), dtype=bool)
        m.level_pose = bool(d.get("levelPose", False))
        m.tag_refine = bool(d.get("tagRefine", False))
        return m


# =============================================================================================
# 5. Programul principal
# =============================================================================================

def parse_bool(s):
    return str(s).strip().lower() in ("1", "true", "yes", "da", "on")


def build_args():
    ap = argparse.ArgumentParser(description="Poziții de cameră pentru turntable din AprilTag-uri.")
    ap.add_argument("--input", required=True, help="cameraInit.sfm (JSON) de la nodul CameraInit")
    ap.add_argument("--output", required=True, help="fișierul .sfm cu poziții (pentru FeatureMatching/SfMTriangulation)")
    ap.add_argument("--calibOut", default="", help="unde salvez modelul (JSON), ca să-l refolosești")
    ap.add_argument("--report", default="", help="fișier text cu raportul (implicit lângă --output)")
    ap.add_argument("--layout", default=DEFAULT_LAYOUT, help="geometria tag-urilor (JSON)")
    ap.add_argument("--tagScale", type=float, default=1.0,
                    help="corecția de print: (distanța măsurată) / (distanța din PDF). 1.0 = print exact")
    ap.add_argument("--stepDeg", type=float, default=18.0, help="pasul nominal al platanului (grade)")
    ap.add_argument("--liftMm", type=float, default=80.0, help="urcarea nominală între niveluri (mm)")
    ap.add_argument("--nameRegex", default=r"(?P<level>\d)(?P<index>\d{3})$",
                    help="regex pe numele fișierului (fără extensie) cu grupurile level și index. "
                         "Implicit: f_1005 → nivel 1, pas 5")
    ap.add_argument("--firstLevel", type=int, default=1, help="numărul nivelului de jos (de la care se măsoară urcarea)")
    ap.add_argument("--distortion", default="radialk3", choices=["radialk3", "radialk1", "none"],
                    help="modelul de distorsiune (radialk3 = k1,k2,k3 ca în AliceVision)")
    ap.add_argument("--estimateStep", default="True", help="estimează pasul real (True) sau îl fixează la --stepDeg")
    ap.add_argument("--estimateLift", default="False",
                    help="estimează și mărimea urcării (de obicei nu e nevoie: tija T8 e precisă)")
    ap.add_argument("--levelOffsets", default="True",
                    help="permite un mic decalaj de unghi la începutul fiecărui nivel (dacă platanul nu revine exact la 0)")
    ap.add_argument("--intrinsicsMode", default="focal_k1", choices=["focal_k1", "full", "nominal"],
                    help="fără --intrinsicsIn: ce parametri interni se estimează din tag-uri. "
                         "focal_k1 (implicit) = doar focala și k1, centrul optic fix în mijloc — stabil; "
                         "full = focală, centru, k1..k3 — doar dacă tag-urile se văd din unghiuri foarte diferite; "
                         "nominal = focala din CameraInit, fără distorsiune (test)")
    ap.add_argument("--levelPose", default="False",
                    help="True = platforma poate avea o mică rotație/deplasare la fiecare nivel (glisiere cu joc)")
    ap.add_argument("--tagRefine", default="False",
                    help="True = corectează denivelările hârtiei: înălțimea și înclinarea fiecărui tag")
    ap.add_argument("--liftDirection", default="free", choices=["free", "axis"],
                    help="free = direcția de urcare se estimează; axis = urcarea e paralelă cu axa platanului")
    ap.add_argument("--intrinsicsIn", default="",
                    help="JSON de la calibrate_intrinsics.py (tablă de șah, toată imaginea). Recomandat: "
                         "tag-urile de pe disc acoperă doar partea de jos a imaginii, deci distorsiunea sus e ghicită")
    ap.add_argument("--fixIntrinsics", default="True",
                    help="cu --intrinsicsIn: True = folosește parametrii exact; False = doar ca punct de plecare")
    ap.add_argument("--calibIn", default="",
                    help="model salvat anterior (turntable_calib.json). Dacă e dat, se pleacă de la el")
    ap.add_argument("--refine", default="True",
                    help="cu --calibIn: True = rafinez cu tag-urile din pozele astea; False = folosesc modelul exact cum e")
    ap.add_argument("--lockIntrinsics", default="True",
                    help="scrie camerele ca blocate (Meshroom nu le mai modifică)")
    ap.add_argument("--maxReprojErr", type=float, default=3.0,
                    help="colțuri cu eroare mai mare (px) sunt excluse în a doua trecere")
    ap.add_argument("--debugDir", default="", help="dacă e dat: imagini de control cu colțurile detectate/proiectate")
    ap.add_argument("--debugCount", type=int, default=6, help="câte imagini de control")
    ap.add_argument("--jobs", type=int, default=0, help="procese pentru detecție (0 = toate nucleele)")
    ap.add_argument("--verboseLevel", default="info", help="(pentru compatibilitate cu Meshroom, ignorat)")
    return ap.parse_args()


_ASCII = {"ă": "a", "â": "a", "î": "i", "ș": "s", "ş": "s", "ț": "t", "ţ": "t", "Ă": "A", "Â": "A", "Î": "I",
          "Ș": "S", "Ş": "S", "Ț": "T", "Ţ": "T", "°": " grd", "≥": ">=", "≤": "<=", "×": "x", "−": "-", "→": "->",
          "„": '"', "”": '"', "“": '"', "–": "-", "—": "-", "·": "*", "²": "^2", "⁴": "^4", "⁶": "^6", "θ": "theta",
          "λ": "lambda", "μ": "mu"}


def _ascii(msg):
    """Logul din Meshroom (Windows) citește textul ca cp1252; fără diacritice nu mai apar caractere stricate.
    Raportul scris în fișier rămâne cu diacritice."""
    return "".join(_ASCII.get(ch, ch) for ch in str(msg)).encode("ascii", "replace").decode("ascii")


def log(msg):
    print(_ascii(msg), flush=True)


def main():
    t0 = time.time()
    args = build_args()
    estimate_step = parse_bool(args.estimateStep)
    estimate_lift = parse_bool(args.estimateLift)
    level_offsets = parse_bool(args.levelOffsets)
    refine = parse_bool(args.refine)
    lock_intr = parse_bool(args.lockIntrinsics)
    report_path = args.report or os.path.splitext(args.output)[0] + "_report.txt"
    rep = []                                       # liniile raportului

    def R(msg=""):
        rep.append(msg)
        log(msg)

    # ---------------------------------------------------------------- 5.1 citește CameraInit
    with open(args.input, "r", encoding="utf-8") as f:
        sfm = json.load(f)
    intr_by_id = {str(i["intrinsicId"]): i for i in sfm["intrinsics"]}
    rx = re.compile(args.nameRegex)
    views = []
    for v in sfm["views"]:
        stem = os.path.splitext(os.path.basename(v["path"]))[0]
        m = rx.search(stem)
        if not m:
            sys.exit("[eroare] nu pot citi nivelul/pasul din numele '%s' cu regex-ul %s" % (stem, args.nameRegex))
        views.append({"viewId": str(v["viewId"]), "intrinsicId": str(v["intrinsicId"]), "path": v["path"],
                      "level": int(m.group("level")), "index": int(m.group("index")),
                      "w": int(v.get("width") or intr_by_id[str(v["intrinsicId"])]["width"]),
                      "h": int(v.get("height") or intr_by_id[str(v["intrinsicId"])]["height"])})
    cams = sorted({v["intrinsicId"] for v in views})
    levels = sorted({v["level"] for v in views})
    R("=== turntable_poses ===")
    R("Poze: %d   camere (intrinsics): %d   niveluri: %s" % (len(views), len(cams), levels))
    for c in cams:
        it = intr_by_id[c]
        n = sum(1 for v in views if v["intrinsicId"] == c)
        R("  camera %s (serial %s): %d poze, %sx%s px, senzor %.3f mm" %
          (c, it.get("serialNumber", "?"), n, it["width"], it["height"], float(it["sensorWidth"])))

    # ---------------------------------------------------------------- 5.2 tag-uri
    layout = load_layout(args.layout or DEFAULT_LAYOUT, args.tagScale)
    R("Tag-uri în layout: %d (scară print %.4f)" % (len(layout), args.tagScale))
    jobs = args.jobs or os.cpu_count() or 1
    paths = [v["path"] for v in views]
    det = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for p, d in ex.map(detect_tags, paths):
            det[p] = d
    missing_files = [p for p in paths if det[p] is None]
    if missing_files:
        sys.exit("[eroare] nu pot citi imaginile: %s ..." % missing_files[:3])
    for v in views:
        d = det[v["path"]]
        v["tags"] = {i: c for i, c in d.items() if i in layout}      # ignoră ID-uri care nu sunt pe disc
        v["foreign"] = sorted(i for i in d if i not in layout)
    n_tags = [len(v["tags"]) for v in views]
    R("Detecție: %.1f s; tag-uri pe poză: min %d, median %d, max %d; poze fără tag-uri: %d" %
      (time.time() - t0, min(n_tags), int(np.median(n_tags)), max(n_tags), sum(1 for n in n_tags if n == 0)))
    foreign = sorted({i for v in views for i in v["foreign"]})
    if foreign:
        R("  [atenție] ID-uri detectate care nu sunt în layout (ignorate, posibil detecții false): %s" % foreign)

    # ---------------------------------------------------------------- 5.3 inițializare
    model = None
    if args.calibIn:
        with open(args.calibIn, "r", encoding="utf-8") as f:
            model = TurntableModel.from_json(json.load(f))
        if sorted(model.cams) != cams:
            sys.exit("[eroare] camerele din %s (%s) nu corespund cu cele din poze (%s). "
                     "S-au schimbat camerele/focala în CameraInit?" % (args.calibIn, model.cams, cams))
        for L in levels:
            if L != model.levels[0] and "off_%d" % L not in model.p:
                model.p["off_%d" % L] = np.zeros(1)
                model.free["off_%d" % L] = np.ones(1, dtype=bool)
        model.levels = sorted(set(model.levels) | set(levels))
        # pasul și urcarea vin din parametrii acestei rulări (le poți schimba între scanări);
        # restul (camere, axă) vine din modelul salvat
        if abs(model.p["step"][0] - args.stepDeg) > 0.2:
            model.p["step"] = np.array([args.stepDeg])
        model.lift_m = args.liftMm / 1000.0
        R("Pornesc de la modelul salvat: %s (pas %.3f°, urcare %.1f mm)" %
          (args.calibIn, model.p["step"][0], args.liftMm))
    else:
        model = None
    known = load_intrinsics(args.intrinsicsIn, cams, intr_by_id, R) if args.intrinsicsIn else {}
    if model is None:
        model = init_model(views, cams, levels, layout, intr_by_id, args, R, known)
    for c, (Kv, Dv) in known.items():          # calibrarea cu tablă de șah are prioritate
        model.p["K_%s" % c], model.p["D_%s" % c] = Kv.copy(), Dv.copy()

    # ce parametri se optimizează
    model.free["step"][:] = estimate_step
    model.free["lift_s"][:] = estimate_lift
    model.level_pose = parse_bool(args.levelPose)
    for L in model.levels:
        if L != model.levels[0]:
            model.free["lvlR_%d" % L][:] = model.level_pose
            model.free["lvlT_%d" % L][:] = model.level_pose
            if model.level_pose:
                # o rotație a platformei în jurul verticalei = un decalaj de unghi al platanului;
                # păstrăm una singură ca să nu existe două soluții echivalente
                model.free["off_%d" % L][:] = False
                model.p["off_%d" % L] = np.zeros(1)
    if parse_bool(args.tagRefine):
        model.tag_refine = True
        model.tag_ctr = {i: layout[i].mean(axis=0) for i in layout}
        for i in layout:
            if "tag_%d" % i not in model.p:
                model.p["tag_%d" % i] = np.zeros(3)
            model.free["tag_%d" % i] = np.ones(3, dtype=bool)
    if args.liftDirection == "axis" or model.level_pose:
        # cu levelPose, deplasarea pe nivel o preia lvlT; direcția de urcare rămâne pe axă
        model.free["lift_t"][:] = False
        model.p["lift_t"] = model.p["axis_t"].copy()
    for L in model.levels:
        if L != model.levels[0]:
            # cu levelPose, decalajul de unghi e deja inclus în rotația platformei
            model.free["off_%d" % L][:] = level_offsets and not model.level_pose
    for c in cams:
        D = model.free["D_%s" % c]
        D[:] = {"radialk3": [True, True, True], "radialk1": [True, False, False], "none": [False] * 3}[args.distortion]
        if not args.intrinsicsIn:
            if args.intrinsicsMode == "focal_k1":
                model.free["K_%s" % c][:] = [True, False, False]       # centrul optic rămâne în mijloc
                D[:] = [args.distortion != "none", False, False]
                model.p["D_%s" % c] = model.p["D_%s" % c] * D
            elif args.intrinsicsMode == "nominal":
                model.free["K_%s" % c][:] = False
                D[:] = False
        if args.intrinsicsIn and parse_bool(args.fixIntrinsics):
            model.free["K_%s" % c][:] = False                 # parametrii vin din calibrarea cu tablă de șah
            model.free["D_%s" % c][:] = False                 # (modelul lor de distorsiune are prioritate)
        elif args.distortion != "radialk3":
            model.p["D_%s" % c] = model.p["D_%s" % c] * D   # pune la zero termenii nefolosiți

    # ---------------------------------------------------------------- 5.4 optimizare globală
    obs = build_observations(views, layout)
    if len(obs["uv"]) == 0 and not args.calibIn:
        sys.exit("[eroare] niciun tag detectat în nicio poză. Verifică imaginile și --layout.")
    if len(obs["uv"]) and (refine or not args.calibIn):
        for it in range(2):
            res = optimize(model, views, obs)
            err = np.linalg.norm(residuals(model, views, obs).reshape(-1, 2), axis=1)
            thr = max(args.maxReprojErr, 4.0 * float(np.median(err)))
            bad = err > thr
            R("Optimizare %d: %d colțuri, eroare mediană %.3f px, RMS %.3f px, excluse peste %.2f px: %d (%s)" %
              (it + 1, len(err), float(np.median(err)), float(np.sqrt((err ** 2).mean())), thr, int(bad.sum()),
               res.message if hasattr(res, "message") else ""))
            if not bad.any():
                break
            obs = {k: v[~bad] for k, v in obs.items()}

    # ---------------------------------------------------------------- 5.5 raport
    write_report(model, views, obs, intr_by_id, args, R)

    # ---------------------------------------------------------------- 5.6 scrie .sfm și modelul
    dist_out = "radialk3" if (args.intrinsicsIn and parse_bool(args.fixIntrinsics)) else args.distortion
    out = make_sfm(sfm, model, views, intr_by_id, dist_out, lock_intr)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    R("\nScris: %s" % args.output)
    if args.calibOut:
        calib = model.to_json()
        calib["intrinsics"] = {c: {"serialNumber": intr_by_id[c].get("serialNumber", ""),
                                   "width": int(intr_by_id[c]["width"]), "height": int(intr_by_id[c]["height"])}
                               for c in cams}
        with open(args.calibOut, "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)
        R("Scris: %s" % args.calibOut)
    if args.debugDir:
        write_debug(model, views, layout, args.debugDir, args.debugCount)
        R("Imagini de control: %s" % args.debugDir)
    R("Timp total: %.1f s" % (time.time() - t0))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")


# =============================================================================================
# 6. Inițializarea modelului din pozițiile individuale ale pozelor
# =============================================================================================

def load_intrinsics(path, cams, intr_by_id, R):
    """Citește calibrarea cu tablă de șah și o potrivește camerelor după serialNumber
    (același ca Exif:BodySerialNumber din poze). Scalează dacă rezoluția diferă."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    known = {}
    for c in cams:
        it = intr_by_id[c]
        serial = it.get("serialNumber", "")
        if serial not in data:
            sys.exit("[eroare] %s nu are calibrare pentru camera cu serial '%s' (are: %s)" %
                     (path, serial, ", ".join(data)))
        d = data[serial]
        s = int(it["width"]) / float(d["width"])
        if abs(s - int(it["height"]) / float(d["height"])) > 1e-3:
            sys.exit("[eroare] calibrarea pentru '%s' are alt raport de aspect (%sx%s)" % (serial, d["width"], d["height"]))
        known[c] = (np.array([d["f"] * s, d["cx"] * s, d["cy"] * s]), np.array(d["k"], dtype=np.float64))
        R("Parametri interni din %s pentru camera %s (serial %s): f = %.1f px, centru (%+.1f, %+.1f), k = %s" %
          (os.path.basename(path), c, serial, known[c][0][0], known[c][0][1] - int(it["width"]) / 2,
           known[c][0][2] - int(it["height"]) / 2, np.round(known[c][1], 4).tolist()))
    return known


def init_model(views, cams, levels, layout, intr_by_id, args, R, known=None):
    """Calibrare OpenCV pe fiecare cameră → poziția fiecărei poze → axă, sens, pas → model.
    Dacă `known` are parametrii interni (tablă de șah), doar pozițiile se calculează din tag-uri."""
    known = known or {}
    model = TurntableModel(cams, levels, args.firstLevel, args.stepDeg, args.liftMm / 1000.0)
    per_pose = {}       # viewId → (R, C)
    for c in cams:
        it = intr_by_id[c]
        w, h = int(it["width"]), int(it["height"])
        if c in known:
            Kv, Dv = known[c]
            model.p["K_%s" % c], model.p["D_%s" % c] = Kv.copy(), Dv.copy()
            Km = np.array([[Kv[0], 0, Kv[1]], [0, Kv[0], Kv[2]], [0, 0, 1]])
            dist = np.array([Dv[0], Dv[1], 0, 0, Dv[2]])
            for v in views:
                if v["intrinsicId"] != c or len(v["tags"]) < 2:
                    continue
                X = np.concatenate([layout[i] for i in v["tags"]])
                uv = np.concatenate([v["tags"][i] for i in v["tags"]])
                ok, rv, tv = cv2.solvePnP(X, uv, Km, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    Rm, _ = cv2.Rodrigues(rv)
                    per_pose[v["viewId"]] = (Rm, (-Rm.T @ tv).ravel())
            continue
        f_mm = float(it.get("focalLength") or it.get("initialFocalLength") or 0) or 3.9
        f_px = f_mm * w / float(it["sensorWidth"])
        K0 = np.array([[f_px, 0, w / 2.0], [0, f_px, h / 2.0], [0, 0, 1]])
        vs = [v for v in views if v["intrinsicId"] == c and len(v["tags"]) >= 2]
        if len(vs) < 3:
            sys.exit("[eroare] camera %s are doar %d poze cu ≥2 tag-uri vizibile; nu o pot calibra. "
                     "Fă o scanare cu discul liber sau dă --calibIn." % (c, len(vs)))
        objs = [np.concatenate([layout[i] for i in v["tags"]]).astype(np.float32) for v in vs]
        imgs = [np.concatenate([v["tags"][i] for i in v["tags"]]).astype(np.float32) for v in vs]
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST)
        if args.distortion == "radialk1" or args.intrinsicsMode == "focal_k1":
            flags |= cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        if args.distortion == "none" or args.intrinsicsMode == "nominal":
            flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        if args.intrinsicsMode in ("focal_k1", "nominal"):
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        if args.intrinsicsMode == "nominal":
            flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, (w, h), K0.copy(), np.zeros(5), flags=flags)
        R("Calibrare inițială camera %s: RMS %.3f px, f = %.1f px (%.3f mm), centru (%+.1f, %+.1f) px față de mijloc, k = %s" %
          (c, rms, K[0, 0], K[0, 0] * float(it["sensorWidth"]) / w, K[0, 2] - w / 2, K[1, 2] - h / 2,
           np.round(dist.ravel()[[0, 1, 4]], 4).tolist()))
        model.p["K_%s" % c] = np.array([K[0, 0], K[0, 2], K[1, 2]])
        model.p["D_%s" % c] = dist.ravel()[[0, 1, 4]].astype(np.float64)
        for v, rv, tv in zip(vs, rvecs, tvecs):
            Rm, _ = cv2.Rodrigues(rv)
            per_pose[v["viewId"]] = (Rm, (-Rm.T @ tv).ravel())

    # axa: fiecare (cameră, nivel) descrie un cerc în jurul axei
    centers, normals = [], []
    for c in cams:
        for L in levels:
            P = np.array([per_pose[v["viewId"]][1] for v in views
                          if v["intrinsicId"] == c and v["level"] == L and v["viewId"] in per_pose])
            if len(P) >= 4:
                ctr, n, r = fit_circle_3d(P)
                if n[1] < 0:
                    n = -n
                centers.append(ctr)
                normals.append(n)
    if not normals:
        sys.exit("[eroare] prea puține poze cu tag-uri pentru a găsi axa platanului.")
    a = np.mean(normals, axis=0)
    a /= np.linalg.norm(a)
    # punctul axei în planul discului: media centrelor de cerc, proiectată pe y=0 de-a lungul axei
    pts = [ctr - a * (ctr[1] / a[1]) for ctr in centers]
    p = np.mean(pts, axis=0)
    model.p["axis_t"] = np.array(tilts_from_unit(a))
    model.p["axis_p"] = np.array([p[0], p[2]])

    # sensul și pasul: unghiul camerei în jurul axei scade/crește liniar cu indexul pozei
    e1 = np.cross(a, [0, 0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    slopes = []
    for c in cams:
        for L in levels:
            vs = sorted([v for v in views if v["intrinsicId"] == c and v["level"] == L and v["viewId"] in per_pose],
                        key=lambda v: v["index"])
            if len(vs) < 3:
                continue
            ang = np.unwrap([math.atan2((per_pose[v["viewId"]][1] - p) @ e2, (per_pose[v["viewId"]][1] - p) @ e1)
                             for v in vs])
            idx = np.array([v["index"] for v in vs], dtype=float)
            slopes.append(np.polyfit(idx, np.degrees(ang), 1)[0])
    slope = float(np.median(slopes))
    # camera apare rotită cu −θ în sistemul discului ⇒ pantă = −semn·pas
    model.sign = -1.0 if slope > 0 else 1.0
    step_est = abs(slope)
    R("Axa: înclinată %.2f° față de normala discului, trece la (%.1f, %.1f) mm de centrul discului" %
      (math.degrees(math.acos(min(1.0, a[1]))), p[0] * 1000, p[2] * 1000))
    R("Rotație: sens %s, pas estimat din poze %.3f° (nominal %.3f°)" %
      ("+" if model.sign > 0 else "−", step_est, args.stepDeg))
    if abs(step_est - args.stepDeg) > 0.05 * args.stepDeg:
        R("  [atenție] pasul din poze (%.2f°) diferă de --stepDeg (%.2f°). Folosesc valoarea din poze; "
          "verifică setarea motorului și numele fișierelor." % (step_est, args.stepDeg))
        if parse_bool(args.estimateStep):
            model.p["step"] = np.array([step_est])

    # poziția fiecărei camere „la unghi 0, nivelul de jos”: medie peste toate pozele ei
    u = a.copy()                                   # direcția de urcare: inițial paralelă cu axa
    model.p["lift_t"] = np.array(tilts_from_unit(u))
    for L in levels[1:]:
        model.p["off_%d" % L] = np.zeros(1)
    for c in cams:
        Rs, Cs = [], []
        for v in views:
            if v["intrinsicId"] != c or v["viewId"] not in per_pose:
                continue
            Ri, Ci = per_pose[v["viewId"]]
            th = model.theta(v["level"], v["index"])
            Ra = rodrigues(a * th)
            Rs.append(Ri @ Ra.T)
            C_lab = p + Ra @ (Ci - p)
            Cs.append(C_lab - (v["level"] - levels[0]) * model.lift_m * u)
        model.p["R_%s" % c] = rot_to_vec(mean_rotation(Rs))
        model.p["C_%s" % c] = np.median(np.array(Cs), axis=0)
    return model


# =============================================================================================
# 7. Observații, reziduuri, optimizare
# =============================================================================================

def build_observations(views, layout):
    vi, X, uv, tid = [], [], [], []
    for k, v in enumerate(views):
        for i, c in v["tags"].items():
            vi.extend([k] * 4)
            tid.extend([i] * 4)
            X.append(layout[i])
            uv.append(c)
    if not vi:
        return {"vi": np.zeros(0, int), "X": np.zeros((0, 3)), "uv": np.zeros((0, 2)),
                "tid": np.zeros(0, int), "ctr": np.zeros((0, 3))}
    X = np.concatenate(X)
    tid = np.array(tid)
    ctr = np.array([layout[i].mean(axis=0) for i in tid])
    return {"vi": np.array(vi), "X": X, "uv": np.concatenate(uv), "tid": tid, "ctr": ctr}


# Corecția tag-urilor: hârtia de pe platan nu e perfect plată. Fiecare tag poate fi puțin mai sus/jos
# (dy) și puțin înclinat (rx, rz). Camerele privesc discul sub ~15°, unde 1 mm de denivelare
# înseamnă ~7 px în imagine, deci contează mult.
TAG_PRIOR = (0.0005, math.radians(1.0), math.radians(1.0))   # 1 sigma: 0.5 mm, 1°, 1°


def tag_params_adjusted(model):
    """Corecțiile tag-urilor fără componenta „de plan”: o înclinare/ridicare comună a tuturor tag-urilor
    e echivalentă cu o înclinare a axei sau o mutare a camerelor, deci nu poate fi aflată din poze.
    Scădem planul care trece cel mai bine prin înălțimi și media înclinărilor: rămân doar denivelările
    relative (o cută, o margine ridicată), iar planul discului rămâne cel definit de tag-uri."""
    ids = sorted(model.tag_ctr)
    P = np.array([model.p.get("tag_%d" % i, np.zeros(3)) for i in ids])
    C = np.array([model.tag_ctr[i] for i in ids])
    A = np.column_stack([np.ones(len(ids)), C[:, 0], C[:, 2]])
    coef = np.linalg.lstsq(A, P[:, 0], rcond=None)[0]
    adj = P.copy()
    adj[:, 0] = P[:, 0] - A @ coef
    adj[:, 1] = P[:, 1] - P[:, 1].mean()
    adj[:, 2] = P[:, 2] - P[:, 2].mean()
    return {i: adj[k] for k, i in enumerate(ids)}


def tag_offset(model, X, ctr, tid):
    """Colțurile corectate: X' = centru + (0, dy, 0) + v + w × v, cu v = X − centru, w = (rx, 0, rz)."""
    if not model.tag_refine or len(X) == 0:
        return X
    adj = tag_params_adjusted(model)
    P = np.array([adj.get(int(i), np.zeros(3)) for i in tid])
    v = X - ctr
    w = np.column_stack([P[:, 1], np.zeros(len(P)), P[:, 2]])
    return ctr + np.column_stack([np.zeros(len(P)), P[:, 0], np.zeros(len(P))]) + v + np.cross(w, v)


def refined_layout(model, layout):
    out = {}
    for i, c in layout.items():
        ctr = np.repeat(c.mean(axis=0)[None], 4, axis=0)
        out[i] = tag_offset(model, c, ctr, [i] * 4)
    return out


def residuals(model, views, obs):
    out = np.empty_like(obs["uv"])
    Xall = tag_offset(model, obs["X"], obs["ctr"], obs["tid"])
    for k in np.unique(obs["vi"]):
        m = obs["vi"] == k
        v = views[k]
        Rm, C = model.pose(v["intrinsicId"], v["level"], v["index"])
        c = v["intrinsicId"]
        uv, z = project(Rm, C, model.p["K_%s" % c], model.p["D_%s" % c], Xall[m])
        out[m] = np.where(z[:, None] > 0, uv - obs["uv"][m], 1e3)
    return out.ravel()


def optimize(model, views, obs):
    def fun(x):
        model.unpack(x)
        r = residuals(model, views, obs)
        if model.tag_refine:   # prior slab: tag-urile rămân aproape de poziția din PDF
            pri = np.concatenate([model.p[k] / np.array(TAG_PRIOR) for k in model.p if k.startswith("tag_")])
            r = np.concatenate([r, pri])
        return r
    x0 = model.pack()
    res = least_squares(fun, x0, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=200)
    model.unpack(res.x)
    return res


# =============================================================================================
# 8. Raport: verificări pe care le poți citi
# =============================================================================================

def write_report(model, views, obs, intr_by_id, args, R):
    p, a = model.axis()
    R("\n--- Modelul final ---")
    R("Pas: %.4f° (nominal %.3f°), sens %s" % (model.p["step"][0], args.stepDeg, "+" if model.sign > 0 else "−"))
    for L in model.levels[1:]:
        R("Decalaj de unghi nivel %d față de nivelul %d: %+.3f°" % (L, model.levels[0], model.p["off_%d" % L][0]))
        if model.level_pose:
            R("  platforma la nivelul %d: rotită %.3f°, deplasată în plus %.2f mm" %
              (L, math.degrees(np.linalg.norm(model.p["lvlR_%d" % L])), np.linalg.norm(model.p["lvlT_%d" % L]) * 1000))
            R("  (include și decalajul de unghi al platanului la acel nivel, deci nu e neapărat o mișcare reală)")
    R("Axa: înclinare %.3f° față de normala discului; trece la (%+.2f, %+.2f) mm de centrul discului" %
      (math.degrees(math.acos(min(1.0, a[1]))), p[0] * 1000, p[2] * 1000))
    if model.tag_refine:
        adj = tag_params_adjusted(model)
        R("Corecția tag-urilor (denivelări ale hârtiei, față de planul discului): " + ", ".join(
            "%d: %+.2f mm" % (i, adj[i][0] * 1000) for i in sorted(adj)))
    u = unit_from_tilts(*model.p["lift_t"])
    R("Urcare: %.2f mm/nivel, unghi față de axă %.3f°" %
      (model.lift_m * model.p["lift_s"][0] * 1000, math.degrees(math.acos(min(1.0, abs(u @ a))))))
    for c in model.cams:
        it = intr_by_id[c]
        w, h, sw = int(it["width"]), int(it["height"]), float(it["sensorWidth"])
        K, D = model.p["K_%s" % c], model.p["D_%s" % c]
        Rc = rodrigues(model.p["R_%s" % c])
        C = model.p["C_%s" % c]
        d = C - p
        h_ax = d @ a
        rad = np.linalg.norm(d - h_ax * a)
        look = Rc.T @ np.array([0, 0, 1.0])                 # direcția în care privește camera
        tilt = math.degrees(math.asin(max(-1.0, min(1.0, -(look @ a)))))
        R("Camera %s (serial %s):" % (c, it.get("serialNumber", "?")))
        R("   distanța față de axă %.1f mm, înălțimea față de disc %.1f mm, înclinată în jos %.2f°" %
          (rad * 1000, h_ax * 1000, tilt))
        R("   focală %.2f px = %.4f mm, centru optic (%+.2f, %+.2f) px față de mijloc, distorsiune k = %s" %
          (K[0], K[0] * sw / w, K[1] - w / 2, K[2] - h / 2, np.round(D, 5).tolist()))

    # eroarea pe fiecare poză + comparație cu o poziție calculată liber (detectează sărituri de motor)
    err = np.linalg.norm(residuals(model, views, obs).reshape(-1, 2), axis=1) if len(obs["uv"]) else np.zeros(0)
    R("\n--- Pe fiecare poză ---")
    R("poza                         tag  RMS(px)  dif.liberă(mm/°)")
    worst = []
    for k, v in enumerate(views):
        m = obs["vi"] == k
        name = os.path.join(os.path.basename(os.path.dirname(v["path"])), os.path.basename(v["path"]))
        if not m.any():
            R("%-28s %3d     -      doar din model" % (name, len(v["tags"])))
            continue
        rms = float(np.sqrt((err[m] ** 2).mean()))
        extra = ""
        if len(v["tags"]) >= 2:
            c = v["intrinsicId"]
            K, D = model.p["K_%s" % c], model.p["D_%s" % c]
            Km = np.array([[K[0], 0, K[1]], [0, K[0], K[2]], [0, 0, 1]])
            Xr = tag_offset(model, obs["X"][m], obs["ctr"][m], obs["tid"][m])
            ok, rv, tv = cv2.solvePnP(Xr, obs["uv"][m], Km, np.array([D[0], D[1], 0, 0, D[2]]),
                                      flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                Rf, _ = cv2.Rodrigues(rv)
                Cf = (-Rf.T @ tv).ravel()
                Rm, Cm = model.pose(c, v["level"], v["index"])
                dC = np.linalg.norm(Cf - Cm) * 1000
                dR = math.degrees(np.linalg.norm(rot_to_vec(Rf @ Rm.T)))
                extra = "%6.2f / %.2f" % (dC, dR)
                worst.append((dC, name))
        R("%-28s %3d  %7.3f   %s" % (name, len(v["tags"]), rms, extra))
    if len(err):
        R("\nTotal: %d colțuri, RMS %.3f px, maxim %.3f px" % (len(err), float(np.sqrt((err ** 2).mean())), float(err.max())))
        R("Bun: RMS sub ~0.5 px. Peste 1 px: verifică --tagScale, focusul (s-a mișcat în timpul scanării?) sau vibrațiile.")
    if worst:
        worst.sort(reverse=True)
        R("Cele mai mari diferențe model/poziție liberă: %s" %
          ", ".join("%s %.1f mm" % (n, d) for d, n in worst[:5]))
        R("(Poziția liberă e calculată doar din tag-urile acelei poze; câțiva mm sunt normali când se văd puține tag-uri.\n"
          " Zeci de mm la o poză anume = motorul a sărit un pas sau poza e mișcată.)")


# =============================================================================================
# 9. Scrierea fișierului .sfm pentru Meshroom
# =============================================================================================

def fmt(x):
    return repr(float(x))


def make_sfm(sfm, model, views, intr_by_id, dist_model, lock):
    out = copy.deepcopy(sfm)
    for it in out["intrinsics"]:
        c = str(it["intrinsicId"])
        if c not in model.cams:
            continue
        w, h, sw = int(it["width"]), int(it["height"]), float(it["sensorWidth"])
        K, D = model.p["K_%s" % c], model.p["D_%s" % c]
        f_mm = K[0] * sw / w                       # AliceVision: f_px = f_mm · width / sensorWidth
        it["initialFocalLength"] = fmt(f_mm)
        it["focalLength"] = fmt(f_mm)
        it["pixelRatio"] = "1"
        it["principalPoint"] = [fmt(K[1] - w / 2.0), fmt(K[2] - h / 2.0)]   # față de mijlocul imaginii
        it["type"] = "pinhole"
        if dist_model == "radialk3":
            it["distortionType"], it["distortionParams"] = "radialk3", [fmt(d) for d in D]
        elif dist_model == "radialk1":
            it["distortionType"], it["distortionParams"] = "radialk1", [fmt(D[0])]
        else:
            it["distortionType"], it["distortionParams"] = "none", []
        it["undistortionType"] = "none"
        it["undistortionParams"] = ""
        it["undistortionOffset"] = ["0", "0"]
        it["initializationMode"] = "calibrated"
        it["distortionInitializationMode"] = "calibrated"
        flag = "true" if lock else "false"
        for k in ("locked", "scaleLocked", "offsetLocked", "distortionLocked", "pixelRatioLocked"):
            it[k] = flag
    # fiecare poză primește propria poziție; scoatem structura de rig (pozițiile sunt deja exacte)
    by_id = {v["viewId"]: v for v in views}
    poses = []
    for v in out["views"]:
        vid = str(v["viewId"])
        v.pop("rigId", None)
        v.pop("subPoseId", None)
        v["poseId"] = vid
        mv = by_id[vid]
        Rm, C = model.pose(mv["intrinsicId"], mv["level"], mv["index"])
        poses.append({"poseId": vid, "pose": {"transform": {
            "rotation": [fmt(x) for x in Rm.flatten(order="F")],    # AliceVision salvează coloană cu coloană
            "center": [fmt(x) for x in C]},
            "locked": "true", "rotationOnly": "false", "removable": "false"}})
    out["poses"] = poses
    out.pop("rigs", None)
    return out


# =============================================================================================
# 10. Imagini de control
# =============================================================================================

def write_debug(model, views, layout, folder, count):
    os.makedirs(folder, exist_ok=True)
    pick = sorted(views, key=lambda v: (v["intrinsicId"], v["level"], v["index"]))
    sel = sorted(set(np.linspace(0, len(pick) - 1, max(1, count)).round().astype(int)))
    lay = refined_layout(model, layout)
    allX = np.concatenate([lay[i] for i in sorted(lay)])
    for v in [pick[i] for i in sel]:
        img = cv2.imread(v["path"])
        if img is None:
            continue
        c = v["intrinsicId"]
        Rm, C = model.pose(c, v["level"], v["index"])
        uv, z = project(Rm, C, model.p["K_%s" % c], model.p["D_%s" % c], allX)
        for (x, y), zz in zip(uv, z):           # roșu: unde ar trebui să fie colțurile după model
            if zz > 0:
                cv2.drawMarker(img, (int(round(x)), int(round(y))), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
        for i, cs in v["tags"].items():         # verde: unde le-a găsit detectorul
            for x, y in cs:
                cv2.circle(img, (int(round(x)), int(round(y))), 9, (0, 255, 0), 2)
            cv2.putText(img, str(i), tuple(int(t) for t in cs.mean(axis=0)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 255, 0), 3)
        name = "%s_L%d_%03d.jpg" % (c, v["level"], v["index"])
        cv2.imwrite(os.path.join(folder, name), cv2.resize(img, None, fx=0.5, fy=0.5))


if __name__ == "__main__":
    main()
