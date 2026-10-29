"""
job.py — The whole cloud flow for one face scan, run from the Pi or the laptop:

  preflight R2 -> upload photos (rig/ face pair, optional calib/ disc photos) to face_jobs/<id>/input/ -> verify every photo is there
  -> RAM / disk need -> pod (fresh read-only / read-write R2 credentials limited to this job,
     expiring with the pod's runtime cap) -> wait for status.json -> terminate the pod
  -> download + extract output.zip into results/<id>/ -> delete face_jobs/<id>/ from R2.

The R2 job folder is deleted only after output.zip was downloaded AND extracted; after a failure the photos
stay in R2, so the job can be retried with skip_upload=True.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from botocore.exceptions import BotoCoreError, ClientError

from cloud import cloudflare_r2 as r2
from cloud import runpod as rp

logger = logging.getLogger("FaceScan.CloudJob")

DEFAULT_IMAGE_ENV = "FACE_SCAN_IMAGE"
# Peak RAM of the face pipeline on one 12.3 MP pair: 8.9 GB measured (2 x 4056 x 3040, 4096 px atlas), whatever
# the number of disc photos (turntable_poses.py detects their tags one image per process).
FACE_RAM_GB = 9.0
RAM_MARGIN = 1.3
DEFAULT_MIN_VCPU = 8


def _image_size(path: str) -> tuple[int, int]:
    """(width, height) of one photo, read from its header only."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def _merge_cloud_stats(stats_path: Path, report: Dict[str, Any], runs) -> None:
    """
    Adds what only the client knows to the job's stats.json (training data for RAM / time / cost models):
    the pod actually rented, its price, the time from pod creation to the pipeline start (image pull + boot),
    upload / download times and the requested RAM.
    """
    if not stats_path.is_file() or not runs:
        return
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        last = runs[-1]
        pod_started = (report.get("status") or {}).get("started_at")
        stats["cloud"] = {
            "job_id": report["job_id"],
            "plan": report.get("plan"),
            "pods": report.get("pods"),
            "pod_attempts": len(runs),
            "cost_estimate": report.get("cost_estimate"),
            "pull_and_boot_s": round(float(pod_started) - last.started_at, 1) if pod_started else None,
            "client_timings_s": report.get("timings_s"),
            "pod_timings_s": report.get("pod_timings_s"),
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    except (OSError, ValueError, KeyError) as e:
        logger.warning(f"Could not add the cloud data to {stats_path}: {e}")


def plan_job(input_dir: str, min_vcpu: int = DEFAULT_MIN_VCPU) -> Dict[str, Any]:
    """
    Photos of the job and the pod requirements. input_dir must hold the face pair in rig/0 and rig/1;
    calib/0 and calib/1 (disc photos for turntable_poses.py) are optional.
    """
    files = r2.local_image_files(input_dir)
    face = {rel: p for rel, p in files.items() if rel.startswith(("rig/0/", "rig/1/"))}
    if not any(rel.startswith("rig/0/") for rel in face) or not any(rel.startswith("rig/1/") for rel in face):
        raise r2.R2Error(f"No face pair under {input_dir}/rig/0 and {input_dir}/rig/1")
    n_calib = sum(1 for rel in files if rel.startswith(("calib/0/", "calib/1/")))
    width, height = _image_size(next(iter(face.values())))
    input_gb = sum(os.path.getsize(p) for p in files.values()) / 1024 ** 3
    # A 12.3 MP pair needs FACE_RAM_GB; the working images scale with the pixel count.
    estimate = FACE_RAM_GB * max(width * height / (4056 * 3040), 0.25)
    return {
        "n_images": len(files),
        "n_face_photos": len(face),
        "n_calib_photos": n_calib,
        "width": width,
        "height": height,
        "input_gb": round(input_gb, 3),
        "ram_estimate_gb": round(estimate, 2),
        "min_ram_gb": int(math.ceil(estimate * RAM_MARGIN)),
        "min_vcpu": int(min_vcpu),
        "disk_gb": rp.disk_gb_for(input_gb),
        "files": files,
    }


def run_cloud_job(
    input_dir: str,
    results_dir: str = "results",
    job_id: Optional[str] = None,
    pipeline_options: Optional[Dict[str, Any]] = None,
    image: Optional[str] = None,
    dry_run: bool = False,
    skip_upload: bool = False,
    keep_remote: bool = False,
    min_vcpu: int = DEFAULT_MIN_VCPU,
    delete_stale_after_s: Optional[float] = None,
    cancel_event=None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Runs one face scan in the cloud. pipeline_options go to run_photogrammetry_pipeline() in the pod
    (e.g. {"spectacles": False, "name": "model"}). Returns a report with timings, the pod used, the cost estimate and the result folder.
    delete_stale_after_s: first delete the face_jobs/<id>/ folders left in R2 by older jobs untouched for that long
    (r2.delete_stale_jobs; 0 = every older job, None = keep them). This job and anything outside face_jobs/ (e.g. the
    scanner's own folder) are never touched.
    """
    say = on_progress or logger.info
    options = dict(pipeline_options or {})
    image = image or os.environ.get(DEFAULT_IMAGE_ENV)
    bucket = r2.bucket_from_env()
    job_id = job_id or r2.new_job_id()
    report: Dict[str, Any] = {"job_id": job_id, "bucket": bucket, "timings_s": {}, "dry_run": dry_run}
    t = report["timings_s"]

    plan = plan_job(input_dir, min_vcpu)
    files = plan.pop("files")
    report["plan"] = plan
    say(f"Job {job_id}: {plan['n_face_photos']} face + {plan['n_calib_photos']} disc photos ({plan['input_gb']:.2f} GB, "
        f"{plan['width']}x{plan['height']}), RAM {plan['ram_estimate_gb']} GB -> min {plan['min_ram_gb']} GB, "
        f"{plan['min_vcpu']} vCPU, disk {plan['disk_gb']} GB")
    if not plan["n_calib_photos"]:
        say("No disc photos (calib/0, calib/1): the pod uses the stored calibration, turntable_poses.py is not run.")

    client = r2.client_from_env()
    t0 = time.time()
    r2.preflight(bucket, client=client, prefix=r2.job_prefix(job_id) + "_preflight/")
    t["preflight"] = round(time.time() - t0, 2)
    say(f"R2 preflight ok ({t['preflight']} s)")

    if delete_stale_after_s is not None and not dry_run:
        try:
            stale = r2.delete_stale_jobs(delete_stale_after_s, bucket, client=client, keep=[job_id])
            report["deleted_stale_jobs"] = stale
            if stale:
                say(f"Deleted {len(stale)} old job folder(s) from R2: {', '.join(stale)}")
        except r2.R2Error as e:
            say(f"Could not clean old jobs from R2 (continuing): {e}")

    expected = [r2.input_key(job_id, rel) for rel in files]
    if not skip_upload and not dry_run:
        t0 = time.time()
        res = r2.upload_job_images(job_id, input_dir, bucket, client=client)
        t["upload"] = round(time.time() - t0, 2)
        mb = plan["input_gb"] * 1024
        say(f"Uploaded {len(res['uploaded'])} photos ({mb:.0f} MB) in {t['upload']} s "
            f"({mb / max(t['upload'], 1e-3):.1f} MB/s), failed: {len(res['failed'])}")
    if not dry_run:
        missing = r2.verify_keys(expected, bucket, prefix=r2.input_prefix(job_id), client=client)
        if missing:
            raise r2.R2Error(f"{len(missing)} photo(s) missing in R2, not starting a pod: {missing[:5]}")
        say("All photos are in R2.")

    ttl = int(rp.POD_MAX_RUNTIME_S)
    job_spec = dict(options, job_id=job_id)

    def make_env() -> Dict[str, str]:
        env = r2.pod_credentials_env(job_id, ttl) if not dry_run else {
            "R2_ENDPOINT_URL": os.environ.get("R2_ENDPOINT_URL", ""), "R2_BUCKET": bucket,
            "R2_IN_ACCESS_KEY_ID": "<temp>", "R2_IN_SECRET_ACCESS_KEY": "<temp>", "R2_IN_SESSION_TOKEN": "<temp>",
            "R2_OUT_ACCESS_KEY_ID": "<temp>", "R2_OUT_SECRET_ACCESS_KEY": "<temp>", "R2_OUT_SESSION_TOKEN": "<temp>",
        }
        env.update({
            "JOB_ID": job_id,
            "JOB_JSON": json.dumps(job_spec),
            "FACE_SCAN_IMAGE": image or "",
            "POD_MAX_RUNTIME_S": str(ttl),
            "RAM_MARGIN": "1.0",
        })
        return env

    def make_body(min_ram_gb: int) -> Dict[str, Any]:
        if not image:
            raise rp.RunPodError(f"No Docker image: set {DEFAULT_IMAGE_ENV} (e.g. docker.io/<user>/face-scan:v1).")
        return rp.build_pod_request(
            image=image, env=make_env(), min_ram_gb=min_ram_gb, min_vcpu=plan["min_vcpu"],
            disk_gb=plan["disk_gb"], gpu_types=rp.gpu_types_from_env(), name=f"face-{job_id}",
            cloud_type=os.environ.get("RUNPOD_CLOUD_TYPE", "SECURE"),
            registry_auth_id=os.environ.get("RUNPOD_REGISTRY_AUTH_ID") or None,
            cpu_flavors=rp.cpu_flavors_from_env(),
        )

    if dry_run:
        report["pod_request"] = rp.redact(make_body(plan["min_ram_gb"]))
        say("Dry run: pod request that would be sent:\n" + json.dumps(report["pod_request"], indent=2))
        return report

    status_key = r2.output_key(job_id, "status.json")
    started_key = r2.output_key(job_id, "started.json")

    def read_status():
        return r2.get_json(status_key, bucket, client=client)

    def read_started() -> bool:
        return r2.get_json(started_key, bucket, client=client) is not None

    def clear_status():
        for key in (status_key, started_key):
            try:
                client.delete_object(Bucket=bucket, Key=key)
            except (ClientError, BotoCoreError) as e:
                say(f"Could not remove the previous {key.rsplit('/', 1)[-1]}: {e}")

    clear_status()  # a status.json left by an earlier attempt of this job must not end the new pod at once
    t0 = time.time()
    runs = rp.run_job_pods(rp.RunPodClient(), make_body, plan["min_ram_gb"], read_status,
                           cancel_event=cancel_event, on_progress=say, before_retry=clear_status,
                           read_started=read_started)
    t["pods"] = round(time.time() - t0, 2)
    report["pods"] = [{
        "pod_id": r.pod_id, "outcome": r.outcome, "terminated": r.terminated, "elapsed_s": round(r.elapsed_s, 1),
        "cost_estimate": r.cost_estimate, **r.pod_info,
    } for r in runs]
    report["cost_estimate"] = round(sum(r.cost_estimate or 0.0 for r in runs), 4)
    last = runs[-1] if runs else None
    status = (last.status if last else None) or {}
    report["status"] = status
    if status.get("timings_s"):
        report["pod_timings_s"] = status["timings_s"]

    if status.get("state") != "ok":
        report["result"] = "failed"
        say(f"Job failed: {status.get('state') or (last.outcome if last else 'no pod')} {status.get('error', '')}")
        # Keep the photos in R2 for a retry; fetch the log if the pod managed to upload it
        r2.download_file(r2.output_key(job_id, "run_log.txt"),
                         str(Path(results_dir) / job_id / "run_log.txt"), bucket, client=client)
        return report

    t0 = time.time()
    dest = r2.download_job_output(job_id, str(Path(results_dir) / job_id), bucket, client=client)
    t["download"] = round(time.time() - t0, 2)
    report["result"] = "ok"
    report["results_dir"] = str(dest)
    say(f"Result extracted to {dest} ({t['download']} s)")

    _merge_cloud_stats(dest / "stats.json", report, runs)

    if not keep_remote:
        t0 = time.time()
        report["deleted_objects"] = r2.delete_prefix(r2.job_prefix(job_id), bucket, client=client)
        t["cleanup"] = round(time.time() - t0, 2)
        say(f"Deleted {report['deleted_objects']} objects of job {job_id} from R2.")
    (dest / "cloud_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
