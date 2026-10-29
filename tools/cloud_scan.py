#!/usr/bin/env python3
"""
cloud_scan.py — One face scan in the cloud, from this machine:
upload the photos to R2 -> pod on RunPod (face pipeline, CPU) -> output.zip to R2 -> download + extract into
results/<name>-<job_id>/ -> delete the job from R2. Not part of the Docker image (tools/ is in .dockerignore).

The input folder holds the face pair in rig/0 + rig/1 and, optionally, the free disc photos in calib/0 + calib/1
(turntable_poses.py calibrates the rig on them in the pod; without them the stored calibration is used).

  python tools/cloud_scan.py --dry_run                       # R2 check + pod request, no cost
  python tools/cloud_scan.py --name alin                     # input_images/ -> results/alin-<job_id>/
  python tools/cloud_scan.py D:\\scans\\alin --name alin --no_spectacles --set tag_scale=1.002
  python tools/cloud_scan.py --keep_remote                   # leave face_jobs/<id>/ in R2 (debugging)
  python tools/cloud_scan.py --keep_old_jobs                 # do not delete the folders of earlier jobs in R2
  python tools/cloud_scan.py --job_id <id> --skip_upload     # retry a job whose photos are already in R2

Ctrl+C at any time stops the wait and terminates the pod (photos stay in R2 for a retry).
Before uploading, the face_jobs/<id>/ folders of all earlier jobs are deleted (photos and results of old scans; one scan
runs at a time). Only face_jobs/: anything else in the bucket (e.g. the scanner's own folder) is never touched.
With two cloud scans at the same time, start them with --keep_old_jobs, or the second deletes the first.
--set takes any option the pod passes to run_photogrammetry_pipeline() (runpod_job.PIPELINE_OPTIONS: name,
spectacles, tag_scale ...); values are read as JSON when possible (1, 0.5, true), otherwise as text. Negative
flags work too: --set no_spectacles is spectacles=false. Unknown keys are refused (a typo would otherwise be
silently ignored by the pod).

Needs a .env in the repository root (see .env.example): R2_*, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_R2_API_TOKEN,
RUNPOD_API_KEY, FACE_SCAN_IMAGE.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from cloud import cloudflare_r2 as r2  # noqa: E402
from cloud import runpod as rp  # noqa: E402
from cloud.job import plan_job, run_cloud_job  # noqa: E402

from runpod_job import PIPELINE_OPTIONS  # noqa: E402  (what the pod accepts from JOB_JSON)

# Set by the pod itself, not by --set (paths on this machine do not exist in the pod); = PATH_OPTIONS + job_id
RESERVED_OPTIONS = {"job_id", "rig_dir", "left_path", "right_path", "output_dir", "calib_dir", "layout_path"}


def parse_set(items) -> dict:
    """["tag_scale=1.002", "estimate_lift=false", "data_term=area"] -> {"tag_scale": 1.002, ...}."""
    out = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        key = key.strip().lstrip("-")
        if key.startswith("no_") and key[3:] in PIPELINE_OPTIONS and (not sep or raw.strip().lower() in ("", "true", "1")):
            out[key[3:]] = False    # --no_spectacles ...
            continue
        if not sep or not key:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        if key in RESERVED_OPTIONS:
            raise SystemExit(f"--set {key} is set by the cloud flow itself (paths on this machine do not exist in the pod)")
        if key not in PIPELINE_OPTIONS:
            raise SystemExit(f"--set {key}: unknown pipeline option. Known: {', '.join(sorted(PIPELINE_OPTIONS))}")
        try:
            out[key] = json.loads(raw)
        except ValueError:
            out[key] = raw
    return out


def run_interruptible(**kwargs) -> dict:
    """
    run_cloud_job in a worker thread: Ctrl+C in the main thread sets the cancel event, so the pod is
    terminated by run_pod's finally (a blocking wait in the main thread is not always interruptible on Windows).
    """
    cancel = threading.Event()
    box: dict = {}

    def work():
        try:
            box["report"] = run_cloud_job(cancel_event=cancel, **kwargs)
        except BaseException as e:  # reported in the main thread
            box["error"] = e

    worker = threading.Thread(target=work, name="cloud-job", daemon=True)
    worker.start()
    while worker.is_alive():
        try:
            worker.join(0.5)
        except KeyboardInterrupt:
            if cancel.is_set():
                print("\nStill terminating the pod, please wait (closing now could leave it running and billing).")
            else:
                print("\nCancelling: terminating the pod...")
                cancel.set()
    if "error" in box:
        raise box["error"]
    return box["report"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", nargs="?", default=str(ROOT / "input_images"),
                    help="rig/0 + rig/1 (face pair) and optionally calib/0 + calib/1 (disc photos); default: input_images/")
    ap.add_argument("--name", default=None, help="label of the scan: job id and result folder start with it")
    ap.add_argument("--no_spectacles", action="store_true",
                    help="the subject wears no glasses (no spectacle-frame mask, eyebrows and eyelids measured)")
    ap.add_argument("--results_dir", default=str(ROOT / "results"))
    ap.add_argument("--job_id", default=None, help="reuse a job id (with --skip_upload: photos already in R2)")
    ap.add_argument("--image", default=None, help="Docker image (default: FACE_SCAN_IMAGE from .env)")
    ap.add_argument("--dry_run", action="store_true", help="no upload, no pod: print the plan and the pod request")
    ap.add_argument("--skip_upload", action="store_true")
    ap.add_argument("--keep_remote", action="store_true")
    ap.add_argument("--keep_old_jobs", action="store_true",
                    help="keep the face_jobs/<id>/ folders of earlier jobs in R2 (default: all deleted before the upload; "
                         "use this when two scans run in the cloud at the same time)")
    ap.add_argument("--yes", "-y", action="store_true", help="do not ask before starting the pod")
    ap.add_argument("--cpu_flavor", action="append", metavar="FLAVOR",
                    help="only these CPU flavours, e.g. --cpu_flavor cpu5c (repeatable; default: RUNPOD_CPU_FLAVORS "
                         "or the list in cloud/runpod.py)")
    ap.add_argument("--gpu", action="append", metavar="GPU_ID",
                    help='rent a GPU pod instead (RUNPOD_COMPUTE_TYPE=GPU) with these cards, e.g. --gpu "NVIDIA RTX A4000"')
    ap.add_argument("--min_vcpu", type=int, default=8)
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="any pod pipeline option (repeatable)")
    args = ap.parse_args(argv)

    if args.cpu_flavor:
        os.environ["RUNPOD_CPU_FLAVORS"] = ",".join(args.cpu_flavor)   # read by cloud.runpod.cpu_flavors_from_env
    if args.gpu:
        os.environ["RUNPOD_GPU_TYPES"] = ",".join(args.gpu)            # read by cloud.runpod.gpu_types_from_env
        rp.COMPUTE_TYPE = "GPU"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    options = {}
    if args.no_spectacles:
        options["spectacles"] = False
    options.update(parse_set(args.set))

    job_id = args.job_id
    if not job_id and args.name:
        label = re.sub(r"[^A-Za-z0-9_-]+", "-", args.name).strip("-")
        job_id = f"{label}-{r2.new_job_id()}" if label else None

    if not args.dry_run and not args.yes:
        plan = plan_job(args.input_dir, args.min_vcpu, int(options.get("texture_size", 4096)))
        print(f"\n{plan['n_face_photos']} face + {plan['n_calib_photos']} disc photos ({plan['input_gb']:.2f} GB) "
              f"from {args.input_dir}")
        if rp.COMPUTE_TYPE == "GPU":
            print(f"GPU pod: {', '.join(rp.gpu_types_from_env())}, min {plan['min_ram_gb']} GB RAM, "
                  f"{plan['min_vcpu']} vCPU, {plan['disk_gb']} GB disk")
        else:
            print(f"CPU pod: {', '.join(rp.cpu_flavors_from_env())}, "
                  f"{rp.vcpus_for(plan['min_ram_gb'], plan['min_vcpu'])} vCPU (>= {plan['min_ram_gb']} GB RAM), "
                  f"{plan['disk_gb']} GB disk")
        print(f"Options: {json.dumps(options)}")
        if input("Start the pod (billed)? [y/N] ").strip().lower() not in ("y", "yes", "d", "da"):
            print("Nothing started.")
            return 1

    t0 = time.time()
    report = run_interruptible(
        input_dir=args.input_dir, results_dir=args.results_dir, job_id=job_id, pipeline_options=options,
        image=args.image, dry_run=args.dry_run, skip_upload=args.skip_upload, keep_remote=args.keep_remote,
        min_vcpu=args.min_vcpu,
        delete_stale_after_s=None if args.keep_old_jobs else 0.0,
    )
    report["timings_s"]["total"] = round(time.time() - t0, 2)

    print("\n" + "=" * 70)
    print(f" Job {report['job_id']}: {report.get('result', 'dry run' if args.dry_run else '?')}")
    for name, sec in report["timings_s"].items():
        print(f"   {name:<12} {sec:>8.1f} s")
    for pod in report.get("pods", []):
        print(f"   pod {pod['pod_id']}: {pod.get('gpu') or 'CPU'} | {pod.get('memoryInGb')} GB RAM | {pod.get('vcpuCount')} vCPU"
              f" | {pod['outcome']} | {pod['elapsed_s'] / 60:.1f} min | ~${pod.get('cost_estimate') or 0:.3f}")
        if not pod.get("terminated"):
            print(f"   !!! pod {pod['pod_id']} could NOT be confirmed terminated: check https://console.runpod.io/pods")
    if report.get("pod_timings_s"):
        print(f"   in the pod: {json.dumps(report['pod_timings_s'])}")
    if report.get("results_dir"):
        dest = Path(report["results_dir"])
        print(f"   results: {dest}")
        for glb in sorted(dest.glob("glb/*.glb")):
            print(f"     {glb}")
    if report.get("status", {}).get("error"):
        print(f"   error: {report['status']['error']}")
    if report.get("result") == "failed" and not args.dry_run:
        print(f"   retry: python tools/cloud_scan.py --job_id {report['job_id']} --skip_upload")
    print("=" * 70)
    return 0 if report.get("result") in ("ok", None) else 1


if __name__ == "__main__":
    sys.exit(main())
