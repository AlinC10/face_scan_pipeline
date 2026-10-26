# MediaPipe 3D Facial Scanner & Multi-Pipeline 3D Reconstructor

Sistem complet de nivel industrial pentru **scanare facială 3D, calibrare stereo dinamică, deformare anatomică precisă și export automat de modele 3D texturate**.

Proiectul combină **Google MediaPipe Tasks (Vision)**, **OpenCV**, **Trimesh** și algoritmi avansați de fotogrametrie densă (**CREStereo / Screened Poisson Surface Reconstruction**) pentru rig-uri stereo calibrate cu camere Arducam IMX477.

---

## Cuprins

1. [Arhitectura Generală a Sistemului](#1-arhitectura-generală-a-sistemului)
2. [Matricea Scripturilor: Ce Script Folosești și Când?](#2-matricea-scripturilor-ce-script-folosești-și-când)
3. [Ghidul Detaliat al Scripturilor de Reconstrucție 3D](#3-ghidul-detaliat-al-scripturilor-de-reconstrucție-3d)
   - [Pipeline 1: Reconstructor Stereo Canonic cu Urechi Sudate și Subdiviziune](#pipeline-1-reconstructor-stereo-canonic-cu-urechi-sudate-și-subdiviziune)
   - [Pipeline 2: Fotogrametrie Stereo Densă pe Windows / GPU (CREStereo & Poisson)](#pipeline-2-fotogrametrie-stereo-densă-pe-windows--gpu-crestereo--poisson)
   - [Pipeline 3: Extractor de Repere Faciale & Blendshapes ARKit](#pipeline-3-extractor-de-repere-faciale--blendshapes-arkit)
4. [Convenția Standard a Livrabilelor de Ieșire](#4-convenția-standard-a-livrabilelor-de-ieșire)
5. [Structura Directoarelor & Fișierelor](#5-structura-directoarelor--fișierelor)
6. [Instalare & Dependențe](#6-instalare--dependențe)

---

## 1. Arhitectura Generală a Sistemului

Arhitectura proiectului este decuplată modular între **achiziția hardware mecatronică** și **procesarea 3D computațională**:

```
[1. ACHIZIȚIE HARDWARE PE RASPBERRY PI 5] (daemon nativ 3d-scanner-raspberry-pi)
  │  • Motor Z NEMA 17 pe tija T8x8 (100 pași/mm) & Endstops GPIO 22 / GPIO 4
  │  • 2x Arducam IMX477 (12.3MP, convergență 27°, pitch 15° în jos, focus VCM)
  │  • Auto-încadrare continuă în RAM din lores 960x720 (zero disk wear)
  │  • Compensare anti-backlash T8 (-3mm -> +3mm) la cota optimă Z_best
  │  • Captură stereo 12.3MP (4056x3040) salvată metric: f_{Z}.jpg în input_images/rig/0 și 1
  │  • Upload asincron în fundal către Cloudflare R2
  ▼
[2. PROCESARE 3D & FOTOGRAMETRIE] (workspace: mediapipe_script / RunPod GPU)
  │  • Reconstrucție stereo canonică (reconstruct_3d.py):
  │     - Extragere 478 repere MediaPipe din fotografiile stereo f_{Z}.jpg
  │     - Pavilioane auriculare 3D sudate la 18-20° (reconstruction/ear_generator.py)
  │     - Filtru anti-umbră la baza maxilarului și fuziune textură UV
  │     - Subdiviziune Nivel 1 (1.9k vertecși) și Nivel 2 (7.5k vertecși)
  │  • Fotogrametrie densă (run_photogrammetry_windows.py):
  │     - Calibrare epipolară dinamică & disparitate neurală CREStereo
  │     - Reconstrucție organică Poisson via Open3D & Decimare QEM
  │  • Export standardizat în output/: high_model.* și low_model.* (OBJ, GLB, STL, 3MF)
```

---

## 2. Matricea Scripturilor: Ce Script Folosești și Când?

| Obiectivul Tău | Scriptul Executabil | Modul Python | Ce Produce în `output/` |
| :--- | :--- | :--- | :--- |
| **Reconstrucție facială completă din folderul rig-ului sau pereche stereo** | `./run_reconstruct.sh` | [`reconstruct_3d.py`](reconstruct_3d.py) | Pachetul complet de 8 livrabile standard (`high_model.*` și `low_model.*` în `obj`, `glb`, `stl`, `3mf`) |
| **Reconstrucție facială densă metrică din perechea stereo (implicit în Docker)** | `python run_photogrammetry_runpod.py` | [`photogrammetry/face_pipeline.py`](photogrammetry/face_pipeline.py) | Cele 8 livrabile standard (high ~150k fețe, low 20k fețe, texturi 4096/2048) + manifest |
| **Fotogrametrie densă pe Windows cu GPU NVIDIA (CREStereo + Poisson)** | `python run_photogrammetry_windows.py` | [`run_photogrammetry_windows.py`](run_photogrammetry_windows.py) | Master Mesh Poisson dens, hartă disparitate, decimare 15k triunghiuri |
| **Extragere repere 3D metrice & animație facială (52 ARKit blendshapes)** | `./run_scanner.sh` | [`face_scanner.py`](face_scanner.py) | Fișier JSON cu 478 repere metrice, orientare 4x4, blendshapes și imagine adnotată |

> [!NOTE]
> Vechile scripturi de control mecatronic standalone (`rig_capture.py`, `scanner_orchestrator.py`, `run_orchestrator.sh`, `run_capture.sh`) au fost arhivate în [`deprecated/mechatronics_capture/`](deprecated/mechatronics_capture/) deoarece scanerul fizic de pe Raspberry Pi include acum nativ această funcționalitate ([`PLAN_FACE_SCAN.md`](/home/alin/Desktop/3d-scanner-raspberry-pi/PLAN_FACE_SCAN.md)).

---

## 3. Ghidul Detaliat al Scripturilor de Reconstrucție 3D

### Pipeline 1: Reconstructor Stereo Canonic cu Urechi Sudate și Subdiviziune

Scriptul [`reconstruct_3d.py`](reconstruct_3d.py) (apelat prin `./run_reconstruct.sh`) este motorul principal de generare a feței umane 3D texturate.

#### Caracteristici Tehnice Cheie:
1. **Deformare Stereo Metrică:** Preia perechea stereo `f_{Z}.jpg` (Camera 0 profil drept, Camera 1 profil stâng) și triangulează coordonatele 3D pe baza reperelor MediaPipe.
2. **Pavilioane Auriculare Sudate ([`reconstruction/ear_generator.py`](reconstruction/ear_generator.py)):** Integrează urechi 3D solide (carcasă posterioară închisă), calibrate la unghiul anatomic real (18-20° față de craniu), texturate din perspectivele optice optime.
3. **Filtru Anti-Umbră la Baza Maxilarului:** Curăță automat dungile negre cauzate de umbra gâtului sub pitch-ul camerei de 15°.
4. **Nivele Duble de Subdiviziune:**
   * **Low-Poly (`low_model.*`):** Nivel 1 ($1.913$ vertecși / $3.736$ fețe), neted, fără fațetări, cu textură JPG comprimată.
   * **High-Poly (`high_model.*`):** Nivel 2 ($7.561$ vertecși / $14.944$ fețe) cu densitate sporită a curburii și textură PNG necomprimată.
5. **Scalare Biometrică:** Scalat la 1:1 în **metri** pentru OBJ/GLB (Blender) și în **milimetri** pentru STL/3MF (imprimare 3D).

#### Comenzi de Rulare:

```bash
# 1. Rulare pe folderul rig-ului stereo (detectează automat input_images/rig/0 și 1):
./run_reconstruct.sh -i input_images/rig -o output/

# 2. Rulare pe o pereche stereo specifică:
./run_reconstruct.sh -l input_images/rig/0/f_150.jpg -r input_images/rig/1/f_150.jpg -o output/

# 3. Rulare cu rezoluție personalizată de textură (ex: 4096):
./run_reconstruct.sh -i input_images/rig -o output/ --tex-size 4096
```

---

### Pipeline 2: Reconstrucție Facială Densă din Două Vederi (`--mode dense`, implicit)

Scriptul [`run_photogrammetry_runpod.py`](run_photogrammetry_runpod.py) (și containerul Docker) rulează implicit
pipeline-ul dens din [`photogrammetry/face_pipeline.py`](photogrammetry/face_pipeline.py). Rulează pe CPU
(~2.5 min pentru o pereche 12.3 MP) și nu are nevoie de GPU.

```bash
python run_photogrammetry_runpod.py --rig-dir input_images/rig -o output/
# cu altă calibrare (ex. direct din Turntable Studio):
python run_photogrammetry_runpod.py --rig-dir input_images/rig --calib cale/turntable_calib.json -o output/
```

1. **Calibrarea rig-ului** ([`photogrammetry/rig_calibration.py`](photogrammetry/rig_calibration.py)):
   [`assets/rig_calibration.json`](assets/rig_calibration.json) conține intrinsecii și poziția relativă a camerelor,
   măsurate cu `turntable_poses.py` pe discul AprilTag ($B = 164.3$ mm, rotație relativă $19.7^\circ$,
   $f \approx 2550$ px, distorsiune radială). Se acceptă și un `turntable_calib.json` nemodificat.
2. **Potriviri dense ghidate de MediaPipe** ([`photogrammetry/face_correspondence.py`](photogrammetry/face_correspondence.py)):
   reperele MediaPipe au erori de 10–20 px între vederi la 12.3 MP, deci servesc doar ca inițializare. Imaginea
   camerei 1 este deformată pe camera 0 prin teselarea feței, apoi fluxul optic DIS măsoară reziduul sub-pixel
   (verificare înainte-înapoi). Pe aceste ~14k corespondențe se rafinează rotația și direcția bazei
   (lungimea bazei rămâne cea calibrată, deci scara e metrică): eroarea epipolară scade de la ~10 px la ~0.8 px.
3. **Disparitate din ambele vederi**: rectificare Bouguet la 2028×1520, DIS ghidat de un prior de formă (fața 3D
   MediaPipe a fiecărei vederi, adusă la adâncime metrică prin $Z = c_0 + c_1 z_{mp} + c_2 x + c_3 y$ estimat robust
   pe măsurători), cu rame de ochelari mascate (black-hat) și verificări înainte-înapoi / verticale.
4. **Fuziune** ([`photogrammetry/face_surface.py`](photogrammetry/face_surface.py)): punctele ambelor vederi, ponderate
   cu unghiul de vizare, devin un câmp de înălțimi în sistemul capului (aliniere Umeyama cu fața canonică) la 0.5 mm;
   corecția robustă a priorului + detaliul măsurat; în spatele lentilelor doar corecția netedă.
5. **Mesh-uri**: high = grila de 0.5 mm (~150k fețe), low = decimare QEM la 20k fețe; atlase UV xatlas.
6. **Textură** ([`photogrammetry/face_texture.py`](photogrammetry/face_texture.py)): din pozele originale 12.3 MP,
   cu vizibilitate (z-buffer per cameră), ponderi $\cos^3\theta$, echilibrare de culoare între camere și amestec pe
   două benzi (frecvențe joase mediate, detaliul din camera cea mai frontală); hartă de normale tangent-space
   a mesh-ului high coaptă pe atlasul low.
7. **Livrabile**: OBJ/GLB în metri, STL/3MF în mm ca solid închis (fața + placă spate), plus
   `photogrammetry_manifest.json` și imagini de control în `output/debug/`.

Modurile vechi rămân disponibile: `--mode poisson` (SGBM + Poisson) și `--mode hybrid`.

---

### Pipeline 3: Extractor de Repere Faciale & Blendshapes ARKit

Scriptul [`face_scanner.py`](face_scanner.py) (apelat prin `./run_scanner.sh`) extrage datele biometrice și coeficienții de animație:

```bash
./run_scanner.sh -i sample_face.jpg -o output_landmarks/ --save-json --save-annotated --save-obj
```

---

## 4. Convenția Standard a Livrabilelor de Ieșire

Toate fișierele generate sunt organizate în directorul `output/` respectând convenția de nume cerută:

```
output/
├── obj/
│   ├── high/
│   │   ├── high_model.obj             # Mesh High-Poly dens (7.5k vertecși)
│   │   ├── high_model_texture.png     # Textură PNG necomprimată 2K/4K
│   │   ├── high_model_normal.png      # Hartă de normale Tangent-Space
│   │   └── material.mtl               # Referință automată către high_model_texture.png
│   └── low/
│       ├── low_model.obj              # Mesh Low-Poly fin subdivizat (1.9k vertecși)
│       ├── low_model_texture.jpg      # Textură JPG comprimată
│       ├── low_model_normal.png       # Hartă de normale Tangent-Space
│       └── material.mtl               # Referință automată către low_model_texture.jpg
│
├── glb/
│   ├── high_model.glb                 # Format glTF binar cu materiale PBR
│   └── low_model.glb
│
├── stl/
│   ├── high_model.stl                 # Solid 3D scalat în mm pentru printare
│   └── low_model.stl
│
└── 3mf/
    ├── high_model.3mf                 # 3D Manufacturing Format High
    └── low_model.3mf                  # 3D Manufacturing Format Low
```

---

## 5. Structura Directoarelor & Fișierelor

```
mediapipe_script/
├── assets/
│   ├── face_landmarker.task          # Model neuronal Google MediaPipe
│   └── canonical_face_model.obj      # Topologia canonică MediaPipe 3D
├── converters/                       # Convertoare GLB, STL, 3MF, Draco
├── reconstruction/                   # Generatoare geometrie 3D (urechi, relief)
├── deprecated/                       # Arhivă cod mecatronic standalone vechi
│   ├── mechatronics_capture/         # rig_capture, orchestrator, hardware_bridge
│   └── docs/                         # Documente de captură vechi
├── docs/                             # Planuri și ghiduri de dezvoltare
│   ├── implemented/                  # Planuri și soluții tehnice finalizate
│   ├── flame/                        # Arhivă cercetare fitting FLAME monocular
│   ├── GHID_TESTARE_WINDOWS_PHOTOGRAMMETRY.md
│   ├── PLAN_RUNPOD_PHOTOGRAMMETRY.md
│   └── STATUS_PLANURI.md
├── input_images/rig/                 # Imagini de test organizate pe camere (0/ și 1/)
├── requirements.txt                  # Dependențele complete de producție (CUDA/Open3D)
├── reconstruct_3d.py                 # Motorul principal de reconstrucție stereo
├── run_photogrammetry_windows.py     # Pipeline fotogrametrie densă
├── face_scanner.py                   # Extractor 478 repere & ARKit blendshapes
├── run_reconstruct.sh                # Runner Bash pentru reconstruct_3d.py
└── run_scanner.sh                    # Runner Bash pentru face_scanner.py
```

---

## 6. Instalare & Dependențe

Toate pachetele necesare sunt centralizate în [`requirements.txt`](requirements.txt):

```bash
# Creare și activare mediu virtual:
python3 -m venv .venv
source .venv/bin/activate

# Instalare dependențe:
pip install -r requirements.txt
```

---

## 7. Rulare Containerizată cu Docker

Poți rula întregul proiect într-un container Docker izolat, montând folderele locale `input_images` și `output`.

### 1. Construirea Imaginii Docker
```bash
docker build -t 3d-scanner .
```

### 2. Rulare Simplă (Dense Stereo Photogrammetry - Etapele 1-7)
Ia fotografiile din folderul tău local `input_images/` și generează cele 8 livrabile standard direct în folderul tău local `output/`:
```bash
docker run --rm \
  -v $(pwd)/input_images:/app/input_images \
  -v $(pwd)/output:/app/output \
  3d-scanner
```

### 3. Rulare cu Accelerare GPU (NVIDIA)
Dacă ai placă video NVIDIA și NVIDIA Container Toolkit:
```bash
docker run --rm --gpus all \
  -v $(pwd)/input_images:/app/input_images \
  -v $(pwd)/output:/app/output \
  3d-scanner
```

### 4. Rulare Reconstructor Stereo Canonic (`reconstruct_3d.py`)
```bash
docker run --rm \
  -v $(pwd)/input_images:/app/input_images \
  -v $(pwd)/output:/app/output \
  3d-scanner reconstruct
```

### 5. Rulare pe Pereche Stereo Specifică
```bash
docker run --rm \
  -v $(pwd)/input_images:/app/input_images \
  -v $(pwd)/output:/app/output \
  3d-scanner -l input_images/rig/0/f_150.jpg -r input_images/rig/1/f_150.jpg -o output
```

### 6. Shell Interactiv pentru Debugging
```bash
docker run --rm -it \
  -v $(pwd)/input_images:/app/input_images \
  -v $(pwd)/output:/app/output \
  3d-scanner bash
```

