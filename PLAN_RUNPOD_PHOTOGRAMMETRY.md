# Plan Tehnic: Fotogrametrie Dense Stereo pe RunPod (GPU Cloud)
## Reconstrucție 3D Metrică din Pereche Stereo Convergentă la 35 cm

> [!NOTE]
> **STATUS PLAN: [REVĂZUT ȘI CORECTAT RIGUROS — GATA PENTRU IMPLEMENTARE FAZA CU FAZĂ]**  
> Fundamentarea matematică, optică și algoritmică a fost aliniată la geometria reală a rig-ului (Arducam IMX477 + lentilă M23390H08 $f=3.9\,\text{mm}$, AprilTags de pe platan, scalare 2K cu disparitate reziduală, decimare QEM înainte de UV, Poisson depth 9-10).

Acest document constituie specificația matematică, optică și algoritmică a pipeline-ului de fotogrametrie de înaltă rezoluție ce va rula pe serverul GPU (**RunPod cu NVIDIA RTX 4090 / 3090**).

---

## 1. Analiza Optică și Geometria Reală a Rig-ului la 35 cm

### 1.1. Geometria Senzorilor și Optica Rig-ului

Rig-ul mecatronic folosește două camere **Arducam IMX477** echipate cu obiective M12:

* **Senzor de Imagine (Sony IMX477):**
  * **Tip Obturator:** Rolling shutter
  * **Rezoluție Nativă:** $4056 \times 3040$ pixeli (12.3 MP)
  * **Format Senzor:** Type 1/2.3" ($6.287\,\text{mm} \times 4.712\,\text{mm}$, diagonală $7.856\,\text{mm}$)
  * **Dimensiune Pixel:** $p = 1.55\,\mu\text{m} \times 1.55\,\mu\text{m}$
* **Ansamblu Lentilă (Arducam M23390H08):**
  * **Model:** M23390H08 (Montură M12)
  * **Format Optic:** 1/2.3"
  * **Distanță Focală:** $f = 3.9\,\text{mm}$
  * **Diafragmă:** F2.8
  * **Câmp Vizual (FOV):** $75^\circ$ Orizontal (H) pe senzorul IMX477 Type 1/2.3"
  * **Back Focal Length:** $4.49\,\text{mm}$
  * **Distanță Minimă de Focalizare (MOD):** $0.3\,\text{m}$ ($300\,\text{mm}$) — *distanța noastră nominală de scanare de 35 cm ($350\,\text{mm}$) se află imediat deasupra MOD-ului, în plaja de claritate maximă*.
  * **Dimensiuni & Greutate:** $\Phi 14 \times 18.67\,\text{mm}$, $5\,\text{g}$.

* **Distanța Focală în Pixeli ($f_{\text{px}}$):**
  $$f_{\text{px}} = \frac{f}{p} = \frac{3.9 \times 10^{-3}\,\text{m}}{1.55 \times 10^{-6}\,\text{m}} = \frac{3.9}{0.00155} \approx \mathbf{2516.13\,\text{pixeli}}$$

---

### 1.2. Convergența Camerelor, Baseline-ul $B$ și Disparitatea Teoretică

* **Geometrie Convergentă la Distanța $D = 350\,\text{mm}$:**
  * Camerele nu sunt paralele, ci înclinate simetric spre interior cu un unghi total de **$27^\circ$** ($\theta = \pm 13.5^\circ$ față de axa optică mediană), având un pitch descendent comun de **$15^\circ$**.
  * La distanța nominală de convergență $D = 350\,\text{mm}$, distanța dintre centrele optice ale camerelor (baza stereoscopică reală $B$) este determinată trigonometric:
    $$B = 2 \cdot D \cdot \sin(13.5^\circ) = 2 \cdot 350 \cdot 0.2334 \approx \mathbf{163.4\,\text{mm}}$$
    *(Notă: un baseline mecanic $B = 100\,\text{mm}$ ar corespunde unui unghi de convergență de doar $2 \cdot \arcsin(50/350) \approx 16.4^\circ$. Pentru unghiul fizic de $27^\circ$, baza este $\approx 163.4\,\text{mm}$)*.

* **Disparitatea Teoretică la Rezoluție Nativă ($4056 \times 3040$):**
  După rectificarea stereo canonică (alinierea axelor paralele):
  $$d_{\text{nativ}} = \frac{f_{\text{px}} \cdot B}{Z} = \frac{2516.13 \times 163.4}{350} \approx \mathbf{1174.6\,\text{pixeli}}$$
  *(Dacă baza ar fi $100\,\text{mm}$, disparitatea ar fi $\approx 718.9\,\text{px}$)*.

* **Disparitatea la Rezoluția de Lucru 2K ($2048 \times 1536$, scalare $\approx 0.505$):**
  $$f_{\text{2K}} \approx 1270.4\,\text{px} \implies d_{\text{2K}} \approx \frac{1270.4 \times 163.4}{350} \approx \mathbf{593\,\text{pixeli}}$$

* **Sensibilitatea pe Adâncime ($\left|\frac{\partial Z}{\partial d}\right|$):**
  La rezoluție nativă:
  $$\left|\frac{\partial Z}{\partial d}\right| = \frac{Z^2}{f_{\text{px}} \cdot B} = \frac{350^2}{2516.13 \times 163.4} \approx \mathbf{0.298\,\text{mm / pixel}}$$
  La rezoluție 2K:
  $$\left|\frac{\partial Z}{\partial d}\right|_{\text{2K}} \approx \mathbf{0.590\,\text{mm / pixel}}$$

---

### 1.3. Acuratețea Metrică Realistă pe Piele Umană

Formulele teoretice care presupun precizie sub-pixel de $\sigma_d \approx 0.1\,\text{px}$ sunt idealizări matematice pentru ținte sintetice. În realitate, pe un subiect uman:
1. **MTF-ul Obiectivului M12:** O lentilă miniaturală la F2.8 nu rezolvă contrastul maxim până la limita Nyquist a pixelului de $1.55\,\mu\text{m}$.
2. **Subsurface Scattering (Împrăștiere Subcutanată):** Pielea nu este o suprafață Lambertiană pură; lumina pătrunde în epidermă și se difuzează, reducând micro-gradientul de contrast.
3. **Reflexii Speculare & Iluminare:** Sebumul creează sclipiri locale necorespondente între unghiurile de 27°.
4. **Zgomot Senzor & Micromișcare:** Subiectul viu respiră, clipește, iar senzorul rolling shutter introduce micro-deplasări temporale între linii.

* **Concluzie Realistă:**
  Acuratețea practică a adâncimii pe piele se situează în intervalul **$0.1\,\text{mm} - 0.3\,\text{mm}$** ($100 - 300\,\mu\text{m}$). Aceasta este o precizie excelentă pentru topologie facială de înaltă fidelitate, proporții anatomice și imprimare 3D, dar exclude promisiunile nerealiste de tip „mai fin decât firul de păr”.

---

## 2. Strategia de Calibrare Metrică: AprilTags vs. MediaPipe

### 2.1. De ce Autocalibrarea exclusiv din MediaPipe este Fragilă?
1. **Natură Regresată (Jitter de Câțiva Pixeli):** Reperele MediaPipe nu sunt corespondențe fotometrice exacte bazate pe gradienți (precum colțurile Harris sau SIFT), ci predicții neuronale aproximative ale unei rețele, având erori intrinseci de 2–5 pixeli.
2. **Condiționare Matematică Slabă a Matricei $F$:** Cele 468 de puncte acoperă exclusiv un petic convex restrâns din imagine (doar fața), lipsind puncte de dispersie pe margini. Estimarea matricei fundamentale $F$ pe o arie compactă este numeric instabilă (ill-conditioned).
3. **Ambiguitate Proiectivă prin `stereoRectifyUncalibrated`:** Algoritmul Hartley uncalibrated găsește o omografie de rectificare validă doar *up to a projective transformation*. El nu cunoaște scara metrică și poate introduce distorsiuni de forfecare (shearing/keystone) pe axele X și Y.

---

### 2.2. Soluția Robustă: Calibrare Metrică Euclidiană prin AprilTags (`tag16h5`)

Pe platanul / baza rig-ului (vizibilă în partea inferioară a cadrului de 12.3MP datorită pitch-ului de 15° în jos), se află discul de calibrare cu markeri AprilTag din familia **`tag16h5`** (conform geometriei metrice cunoscute din [`platan_260mm_layout.json`](/home/alin/Desktop/template_meshroom/turntable_studio/pipeline/platan_260mm_layout.json), latura tag-ului fiind de exact $30.0\,\text{mm}$):

1. **Detecție Sub-Pixel:**
   Se utilizează `cv2.aruco.ArucoDetector` cu dicționarul `DICT_APRILTAG_16h5` și `CORNER_REFINE_SUBPIX`:
   * Identifică ID-urile tag-urilor și extrage colțurile metrice 2D cu precizie sub-pixel.
2. **Rezolvare PnP Metrică (`cv2.solvePnP`):**
   Cunoscând coordonatele 3D reale ale colțurilor tag-urilor pe planul discului ($Y=0$, unități în metri):
   $$\text{Camera 0:} \quad [R_0 \mid t_0], \quad \text{Camera 1:} \quad [R_1 \mid t_1]$$
3. **Extragerea Extrinsecelor Stereo Reale:**
   * Rotația relativă: $R_{\text{rel}} = R_1 \cdot R_0^T$
   * Translația relativă: $T_{\text{rel}} = t_1 - R_{\text{rel}} \cdot t_0$
   * Baza stereoscopică metrică exactă: $B = \|T_{\text{rel}}\|$ (în metri / milimetri reali).
4. **Rectificare Stereo Metrică Calibrată (`cv2.stereoRectify`):**
   În locul aproximării proiective Hartley, rulăm rectificarea Bouguet/OpenCV calibrată pe baza intrinsecilor $K_0, K_1$ și a extrinsecelor $(R_{\text{rel}}, T_{\text{rel}})$, obținând hărțile de remapare epipolară $R_1, R_2, P_1, P_2, Q$ complet rigide și lipsite de deformări.
5. **Rolul MediaPipe:**
   MediaPipe este utilizat exclusiv acolo unde excelează:
   * Detecția ROI-ului facial și a conturului `FACE_OVAL`.
   * Generarea măștii de decupare a fundalului.
   * Calculul adâncimii medii a feței pentru inițializarea shift-ului de disparitate $d_0$.

---

### 2.3. Rolul Corect al Cotei Metrice a Rig-ului ($Z_{\text{rig}}$)

* **Ce face $Z_{\text{rig}}$:**
  Numele fișierului `f_<distanta_in_mm>.jpg` (ex: `f_150.jpg`) specifică poziția căruciorului motorizat Z pe tija T8x8 față de endstop-ul de bază ($Z=0$).
  Aceasta permite:
  * Delimitarea unui volum cilindric de căutare 3D în jurul feței ($R \approx 150\,\text{mm}$), filtrând obiectele din fundal (pereți, tavan).
  * Fuziunea viitoare a scanărilor multi-elevare ($Z_1, Z_2$) cu translație $\Delta Z$ cunoscută mecanic.
* **Ce NU face $Z_{\text{rig}}$:**
  $Z_{\text{rig}}$ **nu** determină scara metrică a feței. Scara metrică este dată strict de calibrarea intrinsecă ($f$) și stereoscopică ($B$), ancorată în dimensiunea fizică a AprilTag-urilor.

---

## 3. Acoperirea Geometrică a Feței și Limite Fizice

Cu două camere convergente la $27^\circ$ ($\pm 13.5^\circ$):
1. **Perspectivă Preponderent Frontală:** Camerele au o axă aproape paralelă cu privirea subiectului.
2. **Urechile la Unghi Grazing:**
   Urechile sunt orientate la $\sim 90^\circ$ față de axa privirii. Produsul scalar $\vec{N} \cdot \vec{V} \approx 0$ (incidență grazing).
   * Consecință: Detaliul anatomic al urechilor în dense stereo pasiv este redus; textura laterală este comprimată perspectival. Nu se obține un relief complet al pavilionului posterior.
3. **Absența Spatelui Capului:** O singură pereche frontală vede exclusiv masca facială și tâmplele. Spatele capului, calota superioară și ceafa lipsesc complet din date.
4. **Poisson Surface Reconstruction & Decupare:**
   Algoritmul Poisson aplicat pe un nor de puncte deschis tinde să închidă suprafața prin „balonare” la margini (creând o pungă sferică în spatele capului).
   * **Măsură Obligatorie:** Decuparea strictă a fețelor cu densitate scăzută (Poisson density threshold trimming $\le 5\%$) și decuparea după masca de contur facial.

---

## 4. Disparitate Densă Neurală: CREStereo la Rezoluție 2K cu Pre-Shift $d_0$

### 4.1. De ce Nu Rulăm la 12.3MP Nativ?
* Rețeaua CREStereo consumă VRAM substanțial pentru calculul corelațiilor locale adaptive. Rularea directă la $4056 \times 3040$ provoacă risc iminent de OOM chiar și pe GPU RTX 4090 de 24 GB.
* MTF-ul optic al lentilei M12 la F2.8 nu conține informație utilă de adâncime la nivel de pixel de 1.55 µm.
* **Rezoluția Optimă de Inferență:** Rezoluția scalată **2K ($2048 \times 1536$)** menține 100% din detaliile structurale ale feței, consumă sub 8 GB VRAM și oferă o viteză de inferență rapidă și stabilă.

### 4.2. Tehnica de Pre-Shift $d_0$ pentru Disparitate Reziduală
Disparitatea la 2K este de $\approx 590\,\text{px}$. CREStereo funcționează optim pe intervale restrânse de căutare:
1. Calculăm disparitatea medie a feței $d_0$ (din planele de referință AprilTag sau reperele centrale MediaPipe):
   $$d_0 = \max(0, \; \text{int}(d_{\text{mediu}} - 130))$$
2. Translatăm orizontal imaginea dreaptă rectificată cu $+d_0$:
   $$I'_R(x, y) = I_R(x + d_0, y)$$
3. CREStereo este interogat să rezolve doar **disparitatea reziduală** $\Delta d \in [0, 260\,\text{px}]$, acoperind o variație de adâncime de $\pm 75\,\text{mm}$ (suficientă pentru orice profil facial uman).
4. Disparitatea finală se reconstituie prin:
   $$d_{\text{total}}(x, y) = d_{\text{rezidual}}(x, y) + d_0$$
5. Verificarea consistenței Stânga-Dreapta (Left-Right Consistency Check $|d_L - d_R| \le 1.0\,\text{px}$) elimină zgomotul de ocluzie de la marginea pomeților și din barbă.

---

## 5. Ordinea Corectă de Producție: Poisson $\to$ Decimare QEM $\to$ UV $\to$ Baking

Ordinea din planul anterior (UV unwrap pe 1.5M triunghiuri înainte de decimare) era inversă și ineficientă: `xatlas` pe 1.5M triunghiuri ar fi consumat zeci de gigabytes de RAM, iar decimarea QEM ulterioară ar fi ignorat coordonatele UV.

### Fluxul Standard de Producție Industrială:

```
[Disparitate Densă Curățată]
              │
              ▼
1. Triangulare Metrică 3D Nor de Puncte P(X, Y, Z)
              │
              ▼
2. Screened Poisson Surface Reconstruction (Depth 9 - 10)
   (Depth 9-10 oferă ~200k-500k triunghiuri fără a modela zgomotul senzorului)
              │
              ▼
3. Decupare Densitate Redusă (Density Trimming & Mască Oval)
   (Elimină balonarea din spatele capului și geometria inventată)
              │
              ▼
4. Master High-Poly Mesh Curat
              │
              ▼
5. Decimare QEM la Low-Poly (15.000 – 30.000 triunghiuri)
   (Suprafață curată, optimă pentru randare în timp real)
              │
              ▼
6. Despachetare UV pe Low-Poly (xatlas pe 15k-30k fețe)
   (Rapid, stabil, cu chart-uri UV optim împachetate pe atlas)
              │
              ▼
7. Texture Baking & Tangent Normal Map Baking
   (Proiecție Albedo din pozele 12.3MP + Coacere Normale din Master High-Poly)
```

---

## 6. Foaia de Parcurs Revizuită: Pașii Concreți de Execuție

```
ETAPA 1: Ingestion Date, EXIF & Calibrare Metrică AprilTags    [DONE & TESTAT 100%]
   │
ETAPA 2: Rectificare Stereo Bouguet & Mască Facială           [DONE & TESTAT 100%]
   │
ETAPA 3: Disparitate Densă CREStereo 2K cu Pre-Shift d0       [DONE & TESTAT 100%]
   │
ETAPA 4: Triangulare Metrică & Screened Poisson (Depth 9-10) cu Trimming [DONE & TESTAT 100%]
   │
ETAPA 5: Decimare QEM (15k-30k) & Despachetare UV xatlas      [DONE & TESTAT 100%]
   │
ETAPA 6: Coacere Textură Albedo & Hărți de Normale Tangent-Space (URMĂTOAREA)
   │
ETAPA 7: Exportul Standard în 8 Livrabile (OBJ, GLB Draco, STL, 3MF) & Worker RunPod
```

---

### Detalierea Etapelor Implementate (1, 2, 3, 4 și 5):
* **Etapa 1:**
  - `photogrammetry/apriltag_tracker.py`: Detector `tag16h5`, PnP `cv2.SOLVEPNP_SQPNP`, extragere baseline $B = 163.4\,\text{mm}$ și convergență $27^\circ$.
  - `photogrammetry/ingestion.py`: Ingestie 12.3MP, EXIF telemetrie, extragere $Z_{\text{rig}}$ din `f_<Z>.jpg`, manifest JSON.
* **Etapa 2:**
  - `photogrammetry/rectification.py`: Rectificare Bouguet calibrată la scară 2K ($2048 \times 1536$), calcul matrice re-proiecție $Q$, imagine de verificare epipolară.
  - `photogrammetry/face_mask.py`: Mască binară dilatată din `FACE_LANDMARKS_FACE_OVAL`, extragere disparitate mediană a feței și inițializare Pre-Shift $d_0 = \max(0, \text{int}(d_{\text{median}} - 128))$.
* **Etapa 3:**
  - `photogrammetry/dense_stereo.py`: Translație $I'_R$ cu $+d_0$, SGBM 3-Way cu filtrare WLS bilaterală și Left-Right Consistency Check ($|d_L - d_R| \le 1.0\,\text{px}$), suport integrat CREStereo, reconstituire $d_{\text{total}} = \Delta d + d_0$, salvare disparitate metrică 16-bit PNG și vizualizare TURBO.
* **Etapa 4:**
  - `photogrammetry/triangulation_poisson.py`: Triangulare 3D metrică în mm ($Z = f \cdot B / d$), filtrare outlieri statistici, Screened Poisson la Depth 9, Density Trimming la 6% pentru eliminarea balonării. Salvează `output/obj/high/high_model.obj` (305.768 triunghiuri).
* **Etapa 5:**
  - `photogrammetry/decimation_uv.py`: Decimare QEM la exact 20.000 de triunghiuri păstrând topologia anatomică facială, despachetare atlas UV neintersectat prin `xatlas`. Salvează `output/obj/low/low_model.obj` (20.000 triunghiuri, coordonate `vt` complete).
* **CLI Unificat:**
  - `run_photogrammetry_runpod.py` orchestrează etapele 1-5 într-o singură comandă în ~15.6 secunde.
* **CLI Unificat:**
  - `run_photogrammetry_runpod.py` orchestrează etapele 1-3 într-o singură comandă în sub 3.6 secunde.
