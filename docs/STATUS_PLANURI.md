# Tablou de Bord: Statusul Tuturor Planurilor de Dezvoltare

Acest document ține evidența centralizată a tuturor planurilor tehnice din proiect, stadiul lor curent de implementare și livrabilele asociate.

Ultima actualizare: **23 Octombrie 2026**

---

## 1. Planuri Implementate și Validate ([x] DONE)

| Nr. | Fișier Plan | Descriere & Obiective | Status | Livrabile & Fișiere Rezultate |
|:---:|---|---|:---:|---|
| **1** | [`docs/implemented/PLAN_AUTOFRAMING_SCANNER.md`](implemented/PLAN_AUTOFRAMING_SCANNER.md) | Auto-încadrare mecatronică pe Z, detecție concomitentă stereo la 27°, eliminare backlash T8, captură 12.3MP | **[x] INTEGRAT ÎN SCANNER** | Integrat nativ în `/home/alin/Desktop/3d-scanner-raspberry-pi/` ([`PLAN_FACE_SCAN.md`](/home/alin/Desktop/3d-scanner-raspberry-pi/PLAN_FACE_SCAN.md)).<br>Scripturile vechi mecatronice au fost arhivate în [`deprecated/mechatronics_capture/`](../deprecated/mechatronics_capture/). |
| **2** | [`PLAN_FOCUS_AND_EXPORTS.md`](../PLAN_FOCUS_AND_EXPORTS.md)<br>([`docs/PLAN_FOCUS_AND_EXPORTS.md`](PLAN_FOCUS_AND_EXPORTS.md)) | Strategie focus în 2 timpi, redefinire nivele Low (Edge) vs High (Cloud), structură ierarhică 8 fișiere | **[x] DONE** | [`converters/`](../converters/) (`glb`, `stl`, `3mf`)<br>[`model_converter.py`](../model_converter.py)<br>Ierarhia `output_scan/` (`obj`, `glb`, `stl`, `3mf`) |
| **3** | [`docs/implemented/PLAN_REFACTORING_SI_DETALII_3D.md`](implemented/PLAN_REFACTORING_SI_DETALII_3D.md) | **Pasul 1:** Modularizare arhitectură & decuplare Draco.<br>**Pasul 2:** Geometrie pavilioane auriculare (urechi) cu fuziune UV din camerele 0 și 1.<br>**Pasul 3:** Subdiviziune Nivel 1 (1.913 v / 3.736 f), relief păr facial (sprâncene, mustață, barbă) & hărți de normale tangent-space PBR. | **[x] DONE** | [`reconstruction/ear_generator.py`](../reconstruction/ear_generator.py)<br>[`reconstruction/relief_generator.py`](../reconstruction/relief_generator.py)<br>[`converters/draco_compressor.py`](../converters/draco_compressor.py)<br>[`output/obj/low/low_model.obj`](../output/obj/low/low_model.obj)<br>`low_model_normal.png` ($2048 \times 2048$) |

---

## 2. Planuri Pregătite pentru Următoarele Etape (În Așteptare / Următorii Pași)

| Nr. | Fișier Plan | Descriere & Obiectiv | Fază | Status |
|:---:|---|---|:---:|:---:|
| **4** | [`docs/PLAN_RUNPOD_PHOTOGRAMMETRY.md`](PLAN_RUNPOD_PHOTOGRAMMETRY.md) | Fotogrametrie densă pe RunPod (GPU RTX 4090): Auto-calibrare epipolară dinamică prin MediaPipe (fără tablă de șah), CREStereo cu warm-start GRU (accelerare 4x), Screened Poisson Master High-Poly (1.5M faces) + Low-Poly decimat pe server (15k faces) cu Tangent Normal Baking. | **Faza 1 (High-Tier)** | **[GATA PENTRU CLOUD DEPLOYMENT]**<br>Specificația matematică și algoritmică este finalizată. Se va implementa când se configurează instanța RunPod. |
| **5** | [`docs/PLAN_HYBRID_STUDIO.md`](PLAN_HYBRID_STUDIO.md) | Sistem Studio-Grade Hibrid Edge-to-Cloud: Interfață Web Studio pe Pi (live stream dual-camera, reglaje lumini/focus, declanșator) + upload automat R2 și procesare asincronă cu streaming de evenimente (SSE). | **Faza 2** | **[URMĂTORUL PAS MARE]** |

---

## 3. Planuri Arhivate / Pe Pauză (Cercetare Viitoare)

| Nr. | Fișier Plan | Descriere & Motiv | Fază | Status |
|:---:|---|---|:---:|:---:|
| **6** | [`docs/flame/PLAN_FIX_DEFORMATION.md`](flame/PLAN_FIX_DEFORMATION.md)<br>[`docs/flame/PLAN_MVP_FLAME.md`](flame/PLAN_MVP_FLAME.md) | Reconstrucție parametrică monoculară cu modelul FLAME (3DMM). | **Faza 3** | **[PE PAUZĂ]**<br>Deprioritizat din cauza deformărilor anatomice ale optimizatorului monocular pe profiluri laterale. Rigul stereo cu 2 camere la 27° oferă acuratețe geometrică superioară. |

---

## 4. Documente de Cercetare Tehnologică și Ghiduri de Calitate

| Fișier | Descriere | Subiecte Cheie |
|---|---|---|
| [`docs/CERCETARE_PIPELINE_FACIAL_AAA_UNREAL.md`](CERCETARE_PIPELINE_FACIAL_AAA_UNREAL.md) | Cercetare comparativă: Industria AAA (Unreal Engine 5 MetaHuman) vs. MediaPipe Scanner | Pipeline MetaHuman, R3DS Wrap3, Cross-Polarization, Strand-Based Grooms vs. Hair Cards, SSS Shading, puntea gratuită *Mesh to MetaHuman*. |
| [`deprecated/docs/GHID_SI_REQUIREMENTS_RPI_CAPTURE.md`](../deprecated/docs/GHID_SI_REQUIREMENTS_RPI_CAPTURE.md) | Ghid de requirements și explicare a funcționării pentru modulul independent `rig_capture.py` pe Raspberry Pi 5 (arhivat) | Cerințe hardware/software, `requirements_rpi.txt`, pinout GPIO/Arduino, flux tehnic pas cu pas (auto-framing, backlash compensation, stereo 12MP capture, R2 upload hook). |
| [`docs/GHID_TESTARE_WINDOWS_PHOTOGRAMMETRY.md`](GHID_TESTARE_WINDOWS_PHOTOGRAMMETRY.md) | Ghid de testare și validare locală a fotogrametriei pe PC cu Windows (GPU NVIDIA) | Pipeline CREStereo + Poisson local, citire `input_images/rig/0` și `1`, integrare Cloudflare R2 hook pe RPi 5. |
| [`docs/implemented/IDEI_IMBUNATATIRI_RECONSTRUCT.md`](implemented/IDEI_IMBUNATATIRI_RECONSTRUCT.md) | Analiza problemelor de reconstrucție pe date reale și soluțiile validate (arhivă tehnică) | Tratarea umbrei maxilarului, recalibrare urechi la 18-20°, subdiviziune High (7.5k vertecși), scalare biometrică. |

