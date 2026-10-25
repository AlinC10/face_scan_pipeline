# Arhitectura Sistemului de Scanare Facială 3D (Stereo-Rig + Stepper)

Document tehnic de arhitectură și specificații pentru scannerul facial bazat pe **Google MediaPipe**, **OpenCV**, **rig stereo cu 2 camere Arducam IMX477** și **deplasare verticală pe tija cu motor stepper**.

---

## 1. Specificațiile Hardware ale Rig-ului (Specificație Permanentă de Proiect)

* **Senzori / Camere**:
  * 2x **Arducam B0272** (Senzor Sony IMX477, 1/2.3", 1.55 µm pixel pitch).
  * **Ansamblu Lentilă:** Arducam **M23390H08** ($f = 3.9\,\text{mm}$, diafragmă F2.8, FOV $75^\circ$ H pe IMX477 1/2.3", montură M12, MOD $0.3\,\text{m}$, distanță focală în pixeli $f_{\text{px}} \approx 2516.13\,\text{px}$).
  * Dual Stream nativ:
    * **`main`**: Rezoluție completă nativă **12.3 MP (4056 × 3040 pixeli)**, 4:3 — captură foto de mare rezoluție.
    * **`lores`**: Rezoluție joasă **960 × 720 pixeli**, 4:3 — streaming live și tracking în timp real în RAM.
  * Lentilă cu **focus motorizat** (I2C control prin `hardware.camera.focuser.Focuser`).
* **Geometrie Rig Stereo**:
  * Unghi de convergență orizontală: **~27°** (Camera 0 / Left $\leftrightarrow$ Camera 1 / Right orientate spre centrul feței).
  * Înclinație verticală (pitch): **~15° în jos**.
* **Mecanică & Cinematică**:
  * Rig montat pe o tijă verticală T8x8, acționat de un **motor stepper NEMA 17** cu driver TB6600 (pas 8 mm, microstepping 1/4 $\implies$ **100 pași / mm**).
  * Endstop de plafon (SUS): **GPIO Pin 22** (`pull_up=True`, debounce 20ms).
  * Endstop de referință (JOS): **GPIO Pin 4** (`pull_up=True`).

---

## 2. Detecția Zonelor în Timp Real & Tracking Optic (Lores 960x720)

### A. Lores direct vs. Downscale suplimentar
* **Optimizare Edge Computing**: Downscale rapid la **480 × 360** via `cv2.resize` pentru tensorul MediaPipe.
  * MediaPipe redimensionează intern la $256 \times 256$.
  * Reducerea consumă $<1\,\text{ms}$ pe RPi 5 și scade consumul CPU cu **~45%**, menținând stream-ul la 30 FPS fără lag.

### B. Algoritmul de Centrare Optică (Closed-Loop Visual Feedback)
* Fereastra țintă optică pe axa verticală este centrată la $Y_{\text{target}} \in [0.45, 0.55]$.
* Nasul (Landmark `1`) reprezintă ancora centrală pentru compoziția stereo la convergența de 27°.

---

## 3. [ISTORIC / ARHIVĂ TEHNICĂ: Faze Inițiale de Dezvoltare]

> [!NOTE]
> Proiectul a evoluat de la ideea inițială a celor 3 opriri separate (frunte, nas, bărbie) la **Auto-încadrarea unică cu funcție de scor continuă (Apex Search)** și captură de înaltă rezoluție integrată nativ în daemonul `3d-scanner-raspberry-pi` ([`PLAN_FACE_SCAN.md`](/home/alin/Desktop/3d-scanner-raspberry-pi/PLAN_FACE_SCAN.md)), urmată de fotogrametrie densă și decuplarea completă a reconstrucției 3D.
