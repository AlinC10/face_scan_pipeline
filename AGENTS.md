# AGENTS.md: Developer & Agent Guide for 3D Facial Scanner

Acest document conține instrucțiunile de context, arhitectură, reguli de siguranță și foaia de parcurs necesare pentru orice agent autonom sau dezvoltator care lucrează în acest codebase.

---

## 1. Regula de Aur (Hardware Read-Only Rule)

> [!CAUTION]
> **Fișierele din directorul `/home/alin/Desktop/3d-scanner-raspberry-pi/` sunt STRICT READ-ONLY.**
> Niciun agent sau script nu are voie să creeze, să modifice sau să șteargă fișiere în acel director.
> Toată logica de viziune computațională, dense stereo și reconstrucție 3D rămâne în workspace-ul `/home/alin/mediapipe_script/`.

---

## 2. Privire de Ansamblu asupra Proiectului & Specificații Hardware

Proiectul construiește un sistem automatizat de scanare facială 3D de înaltă fidelitate, format dintr-un rig mecatronic cu două camere stereo Arducam IMX477 și un motor stepper Z cu tijă T8x8, cu procesare computațională prin Google MediaPipe Vision Tasks, OpenCV și Trimesh.

### Specificații Hardware Permanente ale Rig-ului:
* **Camere Stereo:** 2x **Arducam B0272** (12.3MP, senzor Sony IMX477, 1/2.3", 1.55 µm pixel pitch).
  * **Ansamblu Lentilă:** Arducam **M23390H08** ($f = 3.9\,\text{mm}$, diafragmă F2.8, FOV $75^\circ$ H pe IMX477 1/2.3", montură M12, MOD $0.3\,\text{m}$, distanță focală în pixeli $f_{\text{px}} \approx 2516.13\,\text{px}$).
  * **Geometrie Optică:** distanță interpupilară $B$, **convergență orizontală de 27°** spre interior (Camera 0 profil drept, Camera 1 profil stâng) și **pitch de 15° în jos**.
  * **Dual Stream Nativ:**
    * **`main`**: Rezoluție completă **12.3 MP ($4056 \times 3040$)**, raport 4:3 — captura foto sincronizată finală.
    * **`lores`**: Rezoluție de preview **$960 \times 720$**, raport 4:3 — streaming live și tracking în timp real în RAM.
  * **Focus:** Motorizat VCM prin I2C (`hardware.camera.focuser.Focuser`).
* **Axa Verticală Z & Mecatronică:**
  * Motor **NEMA 17** + Driver TB6600 pe Arduino Uno (`hardware.motor.Motor`).
  * Tija T8x8 are pas de 8 mm, microstepping 1/4 $\implies$ **100 pași / mm** (scală metrică absolută).
  * **Endstop Plafon (SUS):** **GPIO Pin 22** (`pull_up=True`, debounce 20ms). Declanșează oprirea de urgență imediată a motorului.
  * **Endstop Bază (JOS / Home):** **GPIO Pin 4** (`pull_up=True`). Folosit pentru referențierea $Z = 0.0\text{ mm}$.

### Principii de Viziune Computațională & Tracking Optic:
* **Optimizare Edge Computing:** Downscale rapid la **$480 \times 360$** via `cv2.resize` pentru tensorul MediaPipe. MediaPipe redimensionează intern la $256 \times 256$, astfel că reducerea consumă $<1\,\text{ms}$ pe RPi 5 și scade consumul CPU cu **~45%**, menținând stream-ul la 30 FPS fără lag.
* **Fereastra Țintă Optică:** Fereastră centrată pe axa verticală la $Y_{\text{target}} \in [0.45, 0.55]$. Vârful nasului (Landmark `1`) reprezintă ancora centrală pentru compoziția stereo la convergența de 27°.

---

## 3. Foaia de Parcurs (Phased Roadmap)

### FAZA 1 (Prioritate Maximă - Reconstrucție 3D & Fotogrametrie Decuplată):
* **Module principale de procesare 3D:**
  * [`reconstruct_3d.py`](reconstruct_3d.py) & [`reconstruction/`](reconstruction/) (`ear_generator.py`, `relief_generator.py`)
  * [`run_photogrammetry_windows.py`](run_photogrammetry_windows.py) (pipeline dense stereo / fotogrametrie locală pe Windows / Linux)
  * [`model_converter.py`](model_converter.py) & [`converters/`](converters/) (`glb_converter.py`, `draco_compressor.py`, `stl_converter.py`, `thrf_converter.py`)
  * [`face_scanner.py`](face_scanner.py) (extractor 478 repere MediaPipe pentru deformare și calibrare epipolară dinamică)
* **Captura Mecatronică & Hardware (Integrată Nativ în Scanner RPi):**
  * Controlul fizic al rig-ului pe RPi 5 este implementat direct în `/home/alin/Desktop/3d-scanner-raspberry-pi/` conform [`PLAN_FACE_SCAN.md`](/home/alin/Desktop/3d-scanner-raspberry-pi/PLAN_FACE_SCAN.md).
  * Vechile module mecatronice standalone au fost mutate în [`deprecated/mechatronics_capture/`](deprecated/mechatronics_capture/).
* **Documentație & Status Planuri:**
  * [`docs/implemented/PLAN_AUTOFRAMING_SCANNER.md`](docs/implemented/PLAN_AUTOFRAMING_SCANNER.md) — **[DONE & INTEGRAT ÎN SCANNER]**
  * [`docs/implemented/PLAN_FOCUS_AND_EXPORTS.md`](docs/implemented/PLAN_FOCUS_AND_EXPORTS.md) — **[DONE]**
  * [`docs/implemented/PLAN_REFACTORING_SI_DETALII_3D.md`](docs/implemented/PLAN_REFACTORING_SI_DETALII_3D.md) — **[DONE]**
  * [`docs/implemented/IDEI_IMBUNATATIRI_RECONSTRUCT.md`](docs/implemented/IDEI_IMBUNATATIRI_RECONSTRUCT.md) — **[DONE]**
  * [`docs/PLAN_RUNPOD_PHOTOGRAMMETRY.md`](docs/PLAN_RUNPOD_PHOTOGRAMMETRY.md) — **[GATA PENTRU CLOUD DEPLOYMENT]**
* **Obiective Realizate & Validate:**
  1. Auto-încadrare mecatronică pe verticală (`sweep_best_score` sau `direct_stop`) cu Z-stepper și endstop-uri.
  2. Detecție concomitentă dual-camera la 27° (Camera 0 profil drept, Camera 1 profil stâng, ambele nas).
  3. Compensare mecanică a jocului piuliței T8 (backlash take-up 3 mm).
  4. Captură stereo sincronizată la 12.3MP (salvare metrică `f_{round(z_mm)}.jpg` în `input_images/rig/0` și `1`).
  5. Reconstrucție stereo canonică cu pavilioane auriculare sudate (urechi stânga/dreapta texturate din perspectivele optice optime la 18-20°).
  6. Subdiviziune Nivel 1 (1.9k vertecși) și Nivel 2 (7.5k vertecși) netede, plus hărți de normale tangent-space PBR.
  7. Export complet în cele 8 livrabile standard conform convenției: `high_model.*` și `low_model.*` (`obj`, `glb`, `stl`, `3mf`).

### FAZA 2 (Prioritate Medie - Pasul Următor):
* **Fișier de plan:** [`docs/PLAN_HYBRID_STUDIO.md`](docs/PLAN_HYBRID_STUDIO.md).
* **Obiectiv:** Interfață web interactivă de studio pentru streaming live dual-camera, reglaje de expunere/focus, vizualizare 3D în timp real și declanșare scanare.

### FAZA 3 (Deprioritizat - Arhivă Cercetare Viitoare):
* **Folder documente:** [`docs/flame/`](docs/flame/) (`PLAN_MVP_FLAME.md`, `PLAN_FIX_DEFORMATION.md`).
* **Status:** Pus pe pauză din cauza deformărilor anatomice ale fitting-ului monocular. Nu se lucrează pe el în acest stadiu.

---

## 4. Structura Ierarhică a Livrabilelor de Ieșire

Toate modelele 3D sunt salvate în structura standardizată direct în `output/`:
```
output/
├── obj/
│   ├── high/
│   │   ├── high_model.obj             # Mesh High-Poly subdivizat dens (7.5k vertecși)
│   │   ├── high_model_texture.png     # Textură PNG necomprimată fuzionată stereo
│   │   ├── high_model_normal.png      # Hartă de normale tangent-space PBR
│   │   └── material.mtl               # Referință directă către high_model_texture.png
│   └── low/
│       ├── low_model.obj              # Mesh Low-Poly canonic subdivizat (1.9k vertecși)
│       ├── low_model_texture.jpg      # Textură JPG comprimată
│       ├── low_model_normal.png       # Hartă de normale tangent-space PBR
│       └── material.mtl               # Referință directă către low_model_texture.jpg
│
├── glb/
│   ├── high_model.glb                 # Format High-Poly PBR pentru web / Blender
│   └── low_model.glb                  # Format Low-Poly PBR optimizat
│
├── stl/
│   ├── high_model.stl                 # Solid 3D High pentru printare de precizie
│   └── low_model.stl                  # Solid 3D Low pentru printare rapidă
│
└── 3mf/
    ├── high_model.3mf                 # Format 3D Manufacturing High
    └── low_model.3mf                  # Format 3D Manufacturing Low
```

### Spațiile de Completare (Placeholders):
În [`model_converter.py`](model_converter.py) există funcții dedicate cu blocuri marcate clar:
* `convert_to_glb_draco()`: conține `pass` dedicat pentru codul de compresie Draco al utilizatorului.
* `convert_to_stl()`: `# [SPATIU DE COMPLETARE: POST-PROCESARE STL]`
* `convert_to_3mf()`: `# [SPATIU DE COMPLETARE: CONVERSIE 3MF / METADATE PRODUCȚIE]`

---

## 5. Ghid de Rulare și Testare

Toate scripturile de reconstrucție 3D folosesc mediul virtual local `.venv`.

```bash
# 1. Rulare reconstrucție stereo canonică din directorul rig-ului:
./run_reconstruct.sh -i input_images/rig -o output/

# 2. Rulare pe perechi specifice de fotografii stereo:
./run_reconstruct.sh -l input_images/rig/0/f_150.jpg -r input_images/rig/1/f_150.jpg -o output/

# 3. Rulare pipeline fotogrametrie densă (CREStereo / Poisson):
python run_photogrammetry_windows.py --rig-dir input_images/rig -o output/
```

> [!NOTE]
> Captura mecatronică și comanda fizică a motoarelor pe Raspberry Pi 5 se declanșează nativ din daemonul scanerului ([`PLAN_FACE_SCAN.md`](/home/alin/Desktop/3d-scanner-raspberry-pi/PLAN_FACE_SCAN.md)) prin endpoint-ul `POST /scanner/face/capture`.
