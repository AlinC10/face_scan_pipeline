"""
runpod_job.py — Entry point inside the RunPod pod (docker-entrypoint.sh starts it when JOB_ID is set).

Environment (set by cloud/job.py):
  JOB_ID, JOB_JSON (run_photogrammetry_pipeline options), R2_ENDPOINT_URL, R2_BUCKET,
  R2_IN_*  temporary read-only credentials on face_jobs/<id>/input/,
  R2_OUT_* temporary read-write credentials on face_jobs/<id>/output/,
  POD_MAX_RUNTIME_S (watchdog; the credentials expire after the same time).

Input (face_jobs/<id>/input/): rig/0 + rig/1 (face pair) and, optionally, calib/0 + calib/1 (free disc photos:
turntable_poses.py calibrates the rig on them; without them the calibration stored in the image is used).

Writes to face_jobs/<id>/output/: output.zip, stats.json, run_log.txt and, LAST, status.json:
  {"state": "ok" | "ram_insufficient" | "pipeline_error" | "input_error" | "timeout", ...}
The pod never stops itself (it has no RunPod key): the client terminates it after reading status.json.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path

from cloud import cloudflare_r2 as r2

logger = logging.getLogger("FaceScan.PodJob")

INPUT_DIR = Path(os.environ.get("POD_INPUT_DIR", "/workspace/input_images"))
OUTPUT_DIR = Path(os.environ.get("POD_OUTPUT_DIR", "/workspace/output"))
WATCHDOG_MARGIN_S = 180   # leave time to upload the log and status.json before the credentials expire

# JOB_JSON keys passed on to run_pipeline(): every JobConfig key except the paths of this machine,
# which are set here. Imported lazily (the module pulls the pipeline); kept in sync by a test.
PIPELINE_OPTIONS = {"name", "spectacles", "tag_scale", "check_apriltags", "calibration_path", "export_scale",
                    "glb_draco", "reconstruction_mode", "stop_after_stage", "rectify_native", "refine_with_features",
                    "texture_size", "segment_face", "remove_frames", "head_model", "expression"}


def describe_resources() -> dict:
    """vCPUs and RAM the container sees (cgroup limit when there is one)."""
    ram_gb = None
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit.isdigit():
            ram_gb = int(limit) / 1024 ** 3
    except OSError:
        pass
    if ram_gb is None:
        try:
            ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
        except (ValueError, OSError, AttributeError):
            pass
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"cpus": cpus, "ram_gb": round(ram_gb, 1) if ram_gb else None}


def zip_outputs(out_dir: Path, zip_path: Path) -> float:
    """Deliverables, debug images, calibration and manifests -> output.zip (run_log.txt as logs/run_log.txt)."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if not p.is_file() or p == zip_path:
                continue
            rel = p.relative_to(out_dir).as_posix()
            zf.write(p, "logs/run_log.txt" if rel == "run_log.txt" else rel)
    return zip_path.stat().st_size / 1024 ** 2


def upload_outputs(out_client, bucket: str, job_id: str, status: dict) -> None:
    """output.zip, stats.json, run_log.txt (whatever exists), then status.json last: the completion signal."""
    for name in ("output.zip", "stats.json", "run_log.txt"):
        path = OUTPUT_DIR / name
        if path.is_file():
            ok = r2.upload_file(str(path), bucket, r2.output_key(job_id, name), client=out_client)
            logger.info(f"Upload {name}: {'ok' if ok else 'FAILED'}")
            if not ok and name == "output.zip" and status.get("state") == "ok":
                status.update(state="pipeline_error", error="output.zip upload failed")
    r2.put_json(status, r2.output_key(job_id, "status.json"), bucket, client=out_client)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")
    job_id = os.environ["JOB_ID"]
    bucket = r2.bucket_from_env()
    job = json.loads(os.environ.get("JOB_JSON") or "{}")
    max_runtime = float(os.environ.get("POD_MAX_RUNTIME_S", "3600"))
    in_client = r2.client_from_env("R2_IN_")
    out_client = r2.client_from_env("R2_OUT_")
    started = time.time()
    timings = {}

    # RunPod may restart the container after it exits: a finished job must not run (and bill) twice
    if r2.get_json(r2.output_key(job_id, "status.json"), bucket, client=out_client):
        logger.info(f"Job {job_id} already has a status.json: nothing to do.")
        return 0

    # First sign of life for the client: without it the pod is replaced after RUNPOD_BOOT_TIMEOUT_S
    r2.put_json({"job_id": job_id, "started_at": started}, r2.output_key(job_id, "started.json"), bucket,
                client=out_client)

    done = threading.Event()

    def watchdog():
        if not done.wait(max(60.0, max_runtime - WATCHDOG_MARGIN_S - (time.time() - started))):
            logger.error("Watchdog: runtime cap reached, reporting timeout.")
            upload_outputs(out_client, bucket, job_id, {"state": "timeout", "job_id": job_id, "started_at": started,
                                                         "elapsed_s": round(time.time() - started, 1)})
            os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_handler = logging.FileHandler(OUTPUT_DIR / "run_log.txt", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(log_handler)

    status = {"job_id": job_id, "state": "pipeline_error", "started_at": started}
    try:
        t0 = time.time()
        keys = r2.download_job_images(job_id, str(INPUT_DIR), bucket, client=in_client)
        timings["download_input"] = round(time.time() - t0, 2)
        logger.info(f"Downloaded {len(keys)} photos in {timings['download_input']} s")
        if not ((INPUT_DIR / "rig" / "0").is_dir() and (INPUT_DIR / "rig" / "1").is_dir()):
            raise r2.R2Error("the job has no face pair in rig/0 and rig/1")
    except Exception as e:
        status.update(state="input_error", error=str(e))
        log_handler.flush()
        upload_outputs(out_client, bucket, job_id, status)
        done.set()
        return 1

    # imported late: the download needs only boto3
    from run_photogrammetry_runpod import run_pipeline
    from photogrammetry.resources import InsufficientRamError

    options = {k: v for k, v in job.items() if k in PIPELINE_OPTIONS}
    unknown = sorted(set(job) - PIPELINE_OPTIONS - {"job_id"})
    if unknown:
        logger.warning(f"Ignored job options: {unknown}")
    status["resources"] = describe_resources()
    status["options"] = options
    try:
        t0 = time.time()
        summary = run_pipeline({"reconstruction_mode": "dense", **options,
                                "rig_dir": str(INPUT_DIR / "rig"), "output_dir": str(OUTPUT_DIR)})
        timings["pipeline"] = round(time.time() - t0, 2)
        if summary.get("status") != "success":
            raise RuntimeError(f"pipeline finished with status {summary.get('status')!r}")
        status.update(state="ok", calibration=summary.get("calibration", {}).get("source"))
    except InsufficientRamError as e:
        status.update(state="ram_insufficient", error=str(e),
                      required_gb=round(e.required_gb, 2), available_gb=round(e.available_gb, 2))
    except Exception as e:
        status.update(state="pipeline_error", error=f"{type(e).__name__}: {e}",
                      traceback=traceback.format_exc()[-4000:])
    finally:
        timings["total"] = round(time.time() - started, 2)
        status["timings_s"] = timings
        # stats.json of the pipeline (telemetry, schema v2) + what only the pod knows
        stats_path = OUTPUT_DIR / "stats.json"
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
        except ValueError:
            stats = {}
        stats["pod"] = {"job_id": job_id, "state": status["state"], "resources": status["resources"],
                        "timings_s": timings}
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        logger.info(f"Job {job_id} finished: {status['state']}")
        log_handler.flush()
        if status["state"] == "ok":
            status["zip_size_mb"] = round(zip_outputs(OUTPUT_DIR, OUTPUT_DIR / "output.zip"), 2)
        upload_outputs(out_client, bucket, job_id, status)
        done.set()
    return 0 if status["state"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
