# MediaPipe face mesh - bundle for the Raspberry Pi

Face mesh from the 478 MediaPipe landmarks of the stereo pair (no ears), real size from the calibrated rig,
texture baked from both photos, exported as OBJ / GLB / STL / 3MF (high 14k faces, low 3.6k faces).

    pip install -r requirements-rpi.txt          # only what the scanner venv does not have yet
    PYTHON=/path/to/venv/bin/python ./run_reconstruct.sh --rig-dir input_images/rig -o output/   # rig/0/<n>.jpg + rig/1/<n>.jpg
    ./run_reconstruct.sh -l rig/0/f_150.jpg -r rig/1/f_150.jpg -o output/

Options: --tex-size 2048 | --calib file.json (default assets/rig_calibration.json) | --tag-scale 0.98 |
--export-scale 100 (metres -> export units: 100 = centimetres, the same in every format) | --keep-camera-pose.

Output: obj/{high,low}/, glb/, stl/, 3mf/ and object.json (real size in mm, IPD, scale used, head pose removed).
All formats share the same units (cm by default). The STL / 3MF are the open face surface, not a closed solid.

Needs on the Pi: Python 3.9-3.12 (64-bit OS, aarch64), ~0.7 GB RAM, ~10 s for a 12 MP pair on a laptop CPU.
Not needed here: scipy, xatlas, open3d, torch, the photogrammetry/ folder, GNM (assets/gnm_*.npz).
Nothing is written outside the output folder.
