# Cloud: RunPod + Cloudflare R2 (face scans)

Adapted from Turntable Studio (`template_meshroom/turntable_studio/cloud`): one face scan = one pod, started on
demand and terminated as soon as its result is in R2. The face pipeline needs no GPU, so the pod is a **CPU pod**
by default. The pipeline (`run_photogrammetry_runpod.py`, `photogrammetry/`) does not depend on this folder;
`runpod_job.py` (in the Docker image) and `cloud/job.py` (on the Pi / laptop) connect the two.

```
Pi / laptop                              R2 (one bucket)                               RunPod CPU pod
───────────                              ───────────────                               ──────────────
1. job id, preflight
2. upload photos  ─────────────────────▶ face_jobs/<id>/input/rig/0|1/f_150.jpg       (face pair)
                                         face_jobs/<id>/input/calib/0|1/f_1000.jpg     (disc photos, optional)
3. verify every photo is there
4. temporary credentials (TTL = pod runtime cap, 60 min):
   input/ read-only, output/ read-write
5. create the pod ───────────────────────────────────────────────────────────────────▶ 6. download the photos
                                                                                        7. turntable_poses.py on calib/
                                                                                           (if present), face pipeline
                                         face_jobs/<id>/output/ ◀──────────────────────  8. output.zip, stats.json,
                                                                                           run_log.txt, status.json (last)
9. wait for status.json, terminate the pod (always)
10. download + extract output.zip -> results/<id>/
11. delete face_jobs/<id>/ from R2
```

Face jobs live under `face_jobs/`, not `jobs/`: if the object scanner (Turntable Studio) uses the same bucket,
cleaning old face jobs never touches its jobs, and the other way round.

## Calibration in the pod

With `calib/0` and `calib/1` (photos of the free AprilTag disc, named `f_<level><step:03d>.jpg` like an object
scan, e.g. a 20-stop sweep `f_1000.jpg` … `f_1019.jpg`), the pod runs `turntable_poses.py` on them (~5 s on the
CPU) and reconstructs the face with that calibration; `calibration/` in the result holds `turntable_calib.json`
and `turntable_report.txt`. Without them the calibration stored in the image (`assets/rig_calibration.json`) is
used and the run log says so. `status.json` and `stats.json` record which one (`"calibration": "turntable_poses"`
or `"stored"`).

## Files

| File | Role |
|---|---|
| `cloudflare_r2.py` | S3 client for R2, job key layout (`face_jobs/<id>/input|output/`), uploads (never raise), downloads, prefix deletion, preflight, **temporary prefix-scoped credentials** |
| `runpod.py` | RunPod REST client (`POST/GET/DELETE /v1/pods`), CPU or GPU pod request, pod lifecycle with guaranteed termination and bounded retries |
| `job.py` | The whole flow above for one scan (`run_cloud_job`), used by the Pi and by `tools/cloud_scan.py` |
| `../runpod_job.py` | Inside the pod: download, run, upload, `status.json` |
| `../tools/cloud_scan.py` | Runs the flow from the laptop with `input_images/` (not in the Docker image) |

## Keys: who holds what

| Where | Key | Can do |
|---|---|---|
| Pi / laptop (`.env`) | `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`: R2 key limited to the scanner bucket | upload photos, read results, delete jobs |
| | `CLOUDFLARE_R2_API_TOKEN`: Cloudflare API token with **R2 permissions only** | mint temporary credentials |
| | `RUNPOD_API_KEY` | create / delete pods |
| Pod | `R2_IN_*`: temporary, `object-read-only`, prefix `face_jobs/<id>/input/` | read the photos of this job |
| | `R2_OUT_*`: temporary, `object-read-write`, prefix `face_jobs/<id>/output/` | write the result of this job |

The pod never gets the account keys or the RunPod key, and never stops itself; the client terminates it.

## Choosing the pod

`POST https://rest.runpod.io/v1/pods` with:
- **CPU (default, `RUNPOD_COMPUTE_TYPE=CPU`)**: `computeType: "CPU"`, `cpuFlavorIds` in our order
  (`RUNPOD_CPU_FLAVORS`, default `cpu5c, cpu3c, cpu5g, cpu3g`) with `cpuFlavorPriority: "custom"`, and `vcpuCount`
  large enough for the RAM: the face pipeline peaks at 8.9 GB on a 12.3 MP pair, × 1.3 = 12 GB, at ~2 GB per vCPU
  on compute-optimised flavours -> 8 vCPUs (≥ `--min_vcpu`).
- **GPU (`RUNPOD_COMPUTE_TYPE=GPU` or `cloud_scan.py --gpu ...`)**: the Turntable Studio request (`gpuTypeIds`,
  `minRAMPerGPU`, `minVCPUPerGPU`). The pipeline does not use the GPU; only useful when CPU pods have no capacity.
- `containerDiskInGb` = 10 GB + 3 × the photos; no ports, no volume.

Retries are bounded: a create request rejected for lack of capacity is retried `RUNPOD_MAX_CREATE_ATTEMPTS`
times (free); a second pod is started only if the first died without writing a status (likely OOM: 1.5× RAM,
so more vCPUs) or never started its pipeline within `RUNPOD_BOOT_TIMEOUT_S` (stuck image pull).
`pipeline_error`, timeout and cancel are not retried.

## status.json (written last by the pod)

```json
{"job_id": "...", "state": "ok | pipeline_error | input_error | timeout", "error": "...",
 "calibration": "turntable_poses | stored", "resources": {"cpus": 8, "ram_gb": 16.0},
 "timings_s": {"download_input": 1.2, "pipeline": 190.0, "total": 192.5}}
```

## Test from the laptop

```bash
cp .env.example .env       # fill in the keys and FACE_SCAN_IMAGE
python tools/cloud_scan.py --dry_run                 # preflight + the pod request (secrets redacted); no pod
python tools/cloud_scan.py --name alin               # input_images/ -> results/alin-<job_id>/
python tools/cloud_scan.py --name alin --no_spectacles
python tools/cloud_scan.py --keep_remote             # leave face_jobs/<id>/ in R2
python tools/cloud_scan.py --job_id <id> --skip_upload   # retry with the photos already in R2
```

Before the first real run:
1. Build and push the image: `docker build -t <user>/face-scan:v1 . && docker push <user>/face-scan:v1`
   (`requirements-runtime.txt`: no PyTorch). For a private registry, add its credentials in RunPod
   (Settings → Container Registry Auth) and put the id in `RUNPOD_REGISTRY_AUTH_ID`.
2. Check the CPU flavour ids in `--dry_run` against RunPod's list.
3. Optional safety net: an R2 lifecycle rule that expires `face_jobs/` after a few days (abandoned jobs).

## Upload queue on the Pi

`upload_file(path, bucket, key)` never raises (the queue retries; the photo stays on disk). Keys:
`r2.input_key(job_id, "rig/0/f_150.jpg")` and `r2.input_key(job_id, "calib/0/f_1000.jpg")`. At the end of the
scan: `verify_keys(...)` and only then `run_cloud_job(..., skip_upload=True, job_id=job_id)`.
