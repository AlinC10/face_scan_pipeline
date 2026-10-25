# Ghid Tehnic: Testarea Fotogrametriei Stereo pe PC cu Windows (Înainte de RunPod)

Acest document descrie fluxul de lucru practic și pașii necesari pentru a rula și valida întregul pipeline de fotogrametrie stereo densă (**CREStereo + Screened Poisson + Texturare + Decimare**) local pe un **PC cu Windows echipat cu placă video NVIDIA**, înainte de a implementa worker-ul în cloud pe RunPod.

---

## 1. Arhitectura Fluxului Pragmatic

În loc să facem direct deployment în cloud pe RunPod, împărțim fluxul în doi pași siguri:

```
[1. Raspberry Pi 5 (Scanner Daemon - 3d-scanner-raspberry-pi)]
  │  • Auto-încadrare Z-stepper la Z_best (cu protecție endstops și anti-backlash)
  │  • Captură sincronizată 12.3MP (Cam 0 și Cam 1)
  │  • Salvare metrică: f_{Z}.jpg în input_images/rig/0 și 1
  │  • Upload asincron către Cloudflare R2 (queue dedicat)
  ▼
[2. PC cu Windows (GPU NVIDIA RTX)]  <── [ETAPA CURENTĂ DE VALIDARE]
  │  • Citire din input_images/rig/0 și 1 (sau descărcat din R2)
  │  • Extragere prior metric Z_rig din numele fișierului f_{Z}.jpg
  │  • Auto-calibrare epipolară dinamică (MediaPipe 468 puncte)
  │  • Disparitate densă neurală (CREStereo pe CUDA)
  │  • Triangulare metrică 3D a norului de puncte
  │  • Screened Poisson Surface Reconstruction (Master High-Poly via Open3D)
  │  • Texturare Multi-View & Fuziune UV
  │  • Decimare QEM & Normal Map Tangent-Space
  │  • Export cele 8 livrabile standard în output/
  ▼
[3. Cloud RunPod (Containerizare Docker)]
     • Trecerea codului deja testat și validat pe Windows într-un Dockerfile CUDA.
```

---

## 2. Modul de Input (Identic pe Raspberry Pi, Windows și RunPod)

Scriptul folosește **exact aceeași structură de directoare**, cu convenția metrică de denumire a fotografiilor `f_{round(z_mm)}.jpg`:
```
input_images/rig/
├── 0/
│   └── f_240.jpg      # Camera 0 (Stânga fizic / profil drept la cota Z=240mm)
└── 1/
    └── f_240.jpg      # Camera 1 (Dreapta fizic / profil stâng la cota Z=240mm)
```

Numele `f_240.jpg` encodează înălțimea metrică $Z_{\text{rig}} = 240\,\text{mm}$ a căruciorului pe tija Z, oferind pipeline-ului priorul geometric de scală și boundings pentru disparitate.

---

## 3. Configurarea Mediului pe Windows (și Docker)

Toate bibliotecile necesare sunt centralizate în [`requirements.txt`](../requirements.txt).

### 3.1. Crearea mediului (Conda / Python 3.11):
```powershell
conda create -n photogrammetry python=3.11 -y
conda activate photogrammetry
```

### 3.2. Instalarea pachetelor cu suport NVIDIA CUDA:
```powershell
# 1. PyTorch cu accelerare CUDA 12.1 / 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Instalarea celorlalte biblioteci din requirements.txt
pip install -r requirements.txt
```

> [!NOTE]
> Biblioteca **Open3D** (`open3d>=0.18.0`) inclusă în `requirements.txt` este esențială pe Windows pentru reconstrucția Poisson organică și simplificarea mesh-ului prin Quadric Error Metrics (QEM). Fără ea, algoritmul recurge la o carcasă convexă de test (convex hull).

---

## 4. Rularea Pipeline-ului de Fotogrametrie pe Windows

Scriptul principal de testare este **[`run_photogrammetry_windows.py`](../run_photogrammetry_windows.py)**:

### Modul 1: Rulare automată pe folderul rig-ului:
```powershell
python run_photogrammetry_windows.py --rig-dir input_images/rig -o output/
```

### Modul 2: Rulare pe perechi specifice de fotografii:
```powershell
python run_photogrammetry_windows.py -l input_images/rig/0/f_240.jpg -r input_images/rig/1/f_240.jpg -o output/
```

### Parametri Opționali:
* `--device cuda` : folosește placa video NVIDIA (implicit; dacă nu există GPU, folosește `--device cpu`).
* `--name model` : prefixul modelelor 3D generate (implicit generează `high_model.*` și `low_model.*`).

---

## 5. Structura Livrabilelor Generate

După rulare, scriptul generează ierarhia standard completă direct în directorul `output/`:
```
output/
├── disparity_map.png                   # Harta de disparitate densă colorată
│
├── obj/
│   ├── high/
│   │   ├── high_model.obj              # Master Mesh dens Poisson
│   │   ├── high_model_texture.png      # Textură PNG necomprimată
│   │   ├── high_model_normal.png       # Hartă de normale Tangent-Space
│   │   └── material.mtl
│   └── low/
│       ├── low_model.obj               # Production Low-Poly (neted, fără fațetare)
│       ├── low_model_texture.jpg       # Textură JPG comprimată
│       ├── low_model_normal.png        # Hartă de normale Tangent-Space
│       └── material.mtl
│
├── glb/
│   ├── high_model.glb
│   └── low_model.glb                   # Format PBR optimizat pentru vizualizare web/Blender
│
├── stl/
│   ├── high_model.stl                  # Solid 3D scalat în milimetri pentru printare
│   └── low_model.stl
│
└── 3mf/
    ├── high_model.3mf                  # Format 3D Manufacturing High
    └── low_model.3mf                   # Format 3D Manufacturing Low
```

---

## 6. Pașii de Urmat după Validarea pe Windows

1. **Testarea pe datele reale:** Rularea pe fotografiile `f_{Z}.jpg` capturate cu rig-ul fizic.
2. **Inspectarea modelului 3D:** Deschiderea fișierelor `.obj` sau `.glb` în Blender sau 3D Viewer din Windows pentru a verifica:
   * Curbura anatomică a pomeților și a frunții.
   * Suprafața netedă a feței fără goluri sau zgomot de fond.
   * Urechile reconstruite din perspectivele optice optime.
3. **Deployment RunPod:** Odată ce rezultatul pe Windows este confirmat, pipeline-ul se mută direct în containerul Docker (`docs/PLAN_RUNPOD_PHOTOGRAMMETRY.md`) pentru procesare cloud GPU.
