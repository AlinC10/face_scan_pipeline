"""
runpod.py — One pod per scan job through the RunPod REST API (https://rest.runpod.io/v1).

* The face pipeline runs on the CPU only, so by default the pod is a CPU pod (computeType "CPU",
  cpuFlavorIds + cpuFlavorPriority "custom": RunPod takes the first flavour with capacity, in our order). Its RAM
  follows from the flavour and the vCPU count, so the vCPU count is chosen to cover the RAM the job needs.
  RUNPOD_COMPUTE_TYPE=GPU rents a GPU pod instead (gpuTypeIds, minimum host RAM and vCPUs per GPU).
* Completion is read from R2 (face_jobs/<id>/output/status.json, written last by the pod), not guessed from uptime.
* A pod whose container never starts (e.g. the host's Docker fails to pull the image: "failed to register layer")
  still reports RUNNING. The pod writes face_jobs/<id>/output/started.json first thing; without it after
  BOOT_TIMEOUT_S the pod is terminated and the job retried on a new pod (usually another host).
* Every pod created is terminated (DELETE /pods/{id}) in a `finally`, after success, failure, cancel or timeout.
  When a termination fails no other pod is started, so a lost termination never doubles the bill.
* The pod never receives the RunPod API key: stopping it is the client's job.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger("FaceScan.RunPod")

REST_URL = "https://rest.runpod.io/v1"

# GPU pods (RUNPOD_COMPUTE_TYPE=GPU): the face pipeline does not use the GPU, so the cheapest cards first.
# Exact ids are RunPod's gpuTypeIds enum (some contain double spaces there): check with --dry_run.
DEFAULT_GPU_TYPES = (
    "NVIDIA RTX A4000",
    "NVIDIA RTX A4500",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L4",
    "NVIDIA A40",
)

# CPU pods (default): compute-optimised flavours first (fastest cores, ~2 GB RAM per vCPU), then general purpose
# (~4 GB per vCPU). Ids from RunPod's cpuFlavorIds enum.
DEFAULT_CPU_FLAVORS = ("cpu5c", "cpu3c", "cpu5g", "cpu3g")
CPU_RAM_PER_VCPU_GB = 2.0         # smallest RAM per vCPU among the flavours above: sizes the vCPU count
COMPUTE_TYPE = os.environ.get("RUNPOD_COMPUTE_TYPE", "CPU").strip().upper()

# ── Limits (each one can be overridden by an environment variable) ─────────────────────────────────────────────────
# A pod that runs longer than this is terminated. The pod's R2 credentials expire after the same time.
POD_MAX_RUNTIME_S = float(os.environ.get("RUNPOD_POD_MAX_RUNTIME_S", str(60 * 60)))
# Pods started for one job (a retry happens only for "ram_insufficient" or a pod that died without a status)
MAX_POD_ATTEMPTS = int(os.environ.get("RUNPOD_MAX_POD_ATTEMPTS", "2"))
# Create requests rejected for lack of capacity cost nothing: retried this many times
MAX_CREATE_ATTEMPTS = int(os.environ.get("RUNPOD_MAX_CREATE_ATTEMPTS", "5"))
CREATE_RETRY_S = 30.0
# A pod whose pipeline has not written started.json after this long is replaced (stuck image pull on its host)
BOOT_TIMEOUT_S = float(os.environ.get("RUNPOD_BOOT_TIMEOUT_S", str(10 * 60)))
POLL_S = 5.0                      # pause between two status checks (the pod bills until the one after status.json)
STATUS_MAX_ERRORS = 5             # consecutive failed status queries after which the pod is given up (and terminated)
TERMINATE_ATTEMPTS = 5
TERMINATE_RETRY_S = 5.0
RAM_RETRY_FACTOR = 1.5            # pod died without status (likely OOM): next one asks for this much more RAM


class RunPodError(RuntimeError):
    pass


class RunPodClient:
    """Thin REST client: create / get / delete pod."""

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None,
                 base_url: str = REST_URL):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise RunPodError("RUNPOD_API_KEY is not set.")
        self.http = session or requests.Session()
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        return self.http.request(
            method, f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=60, **kwargs,
        )

    def create_pod(self, body: Dict[str, Any]) -> Dict[str, Any]:
        r = self._request("POST", "/pods", json=body)
        if r.status_code not in (200, 201):
            raise RunPodError(f"Create pod failed ({r.status_code}): {r.text[:500]}")
        return r.json()

    def get_pod(self, pod_id: str) -> Dict[str, Any]:
        r = self._request("GET", f"/pods/{pod_id}")
        if r.status_code != 200:
            raise RunPodError(f"Get pod {pod_id} failed ({r.status_code}): {r.text[:300]}")
        return r.json()

    def delete_pod(self, pod_id: str) -> None:
        r = self._request("DELETE", f"/pods/{pod_id}")
        if r.status_code not in (200, 204, 404):  # 404: already gone
            raise RunPodError(f"Delete pod {pod_id} failed ({r.status_code}): {r.text[:300]}")


def gpu_types_from_env() -> List[str]:
    raw = os.environ.get("RUNPOD_GPU_TYPES", "")
    return [g.strip() for g in raw.split(",") if g.strip()] or list(DEFAULT_GPU_TYPES)


def cpu_flavors_from_env() -> List[str]:
    raw = os.environ.get("RUNPOD_CPU_FLAVORS", "")
    return [f.strip() for f in raw.split(",") if f.strip()] or list(DEFAULT_CPU_FLAVORS)


def disk_gb_for(input_gb: float) -> int:
    """Container disk: the photos, the deliverables (~0.3 GB) and their zip, debug images + 10 GB of headroom."""
    return int(math.ceil(10 + 3 * input_gb))


def vcpus_for(min_ram_gb: int, min_vcpu: int) -> int:
    """vCPUs of a CPU pod: at least min_vcpu, and enough for min_ram_gb on the flavour with the least RAM per vCPU."""
    return max(int(min_vcpu), int(math.ceil(min_ram_gb / CPU_RAM_PER_VCPU_GB)))


def build_pod_request(
    image: str,
    env: Dict[str, str],
    min_ram_gb: int,
    min_vcpu: int,
    disk_gb: int,
    gpu_types: Optional[List[str]] = None,
    name: str = "face-scan-job",
    cloud_type: str = "SECURE",
    registry_auth_id: Optional[str] = None,
    compute_type: Optional[str] = None,
    cpu_flavors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    POST /pods body, no ports, no volume. CPU (default): flavours from `cpu_flavors` in our order, vCPU count
    covering min_ram_gb. GPU: 1 GPU from `gpu_types` in our order, minimum host RAM / vCPUs per GPU.
    """
    compute = (compute_type or COMPUTE_TYPE).upper()
    if compute not in ("CPU", "GPU"):
        raise RunPodError(f"computeType must be CPU or GPU, got {compute!r}")
    body: Dict[str, Any] = {
        "name": name,
        "imageName": image,
        "computeType": compute,
        "cloudType": cloud_type,
        "containerDiskInGb": int(disk_gb),
        "volumeInGb": 0,
        "ports": [],
        "env": dict(env),
    }
    if compute == "CPU":
        body.update({
            "cpuFlavorIds": list(cpu_flavors or DEFAULT_CPU_FLAVORS),
            "cpuFlavorPriority": "custom",
            "vcpuCount": vcpus_for(min_ram_gb, min_vcpu),
        })
    else:
        body.update({
            "gpuCount": 1,
            "gpuTypeIds": list(gpu_types or DEFAULT_GPU_TYPES),
            "gpuTypePriority": "custom",
            "minRAMPerGPU": int(min_ram_gb),
            "minVCPUPerGPU": int(min_vcpu),
        })
    if registry_auth_id:
        body["containerRegistryAuthId"] = registry_auth_id
    return body


def redact(body: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of a request for printing: secrets in env replaced."""
    out = dict(body)
    out["env"] = {k: ("***" if any(s in k for s in ("SECRET", "TOKEN", "KEY")) else v)
                  for k, v in body.get("env", {}).items()}
    return out


def terminate_pod_reliably(client: RunPodClient, pod_id: str) -> bool:
    """
    Terminates a pod and makes sure it is gone: retried a few times (a network blip must not leave a billed pod
    running). Returns False, after saying loudly which pod is still alive, when every attempt failed.
    """
    for attempt in range(1, TERMINATE_ATTEMPTS + 1):
        try:
            client.delete_pod(pod_id)
            return True
        except Exception as e:
            logger.warning(f"Terminating pod {pod_id} failed (attempt {attempt} of {TERMINATE_ATTEMPTS}): {e}")
            if attempt < TERMINATE_ATTEMPTS:
                time.sleep(TERMINATE_RETRY_S)
    logger.error(f"!!! POD {pod_id} COULD NOT BE TERMINATED AND MAY STILL BE BILLED. Terminate it in the RunPod console. !!!")
    return False


@dataclass
class PodRun:
    """What happened to one pod."""
    pod_id: str
    started_at: float
    outcome: str = "failed"          # finished | exited | cancelled | timeout | boot_timeout | status_error
    status: Optional[Dict[str, Any]] = None   # face_jobs/<id>/output/status.json
    terminated: bool = False
    pod_info: Dict[str, Any] = field(default_factory=dict)   # gpu, memoryInGb, vcpuCount, costPerHr
    ended_at: float = 0.0

    @property
    def elapsed_s(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    @property
    def cost_estimate(self) -> Optional[float]:
        rate = self.pod_info.get("costPerHr")
        return round(float(rate) * self.elapsed_s / 3600.0, 4) if rate else None


def _pod_summary(pod: Dict[str, Any]) -> Dict[str, Any]:
    gpu = pod.get("gpu") or {}
    machine = pod.get("machine") or {}
    return {
        "gpu": gpu.get("displayName") or gpu.get("id") or machine.get("gpuTypeId"),
        "memoryInGb": pod.get("memoryInGb"),
        "vcpuCount": pod.get("vcpuCount"),
        "costPerHr": pod.get("adjustedCostPerHr") or pod.get("costPerHr"),
        "dataCenterId": machine.get("dataCenterId"),
    }


def create_pod_with_retries(client: RunPodClient, body: Dict[str, Any],
                            wait: Callable[[float], bool]) -> Optional[Dict[str, Any]]:
    """Create requests rejected (no capacity with these filters) cost nothing: retry a few times, then give up."""
    for attempt in range(1, MAX_CREATE_ATTEMPTS + 1):
        try:
            return client.create_pod(body)
        except Exception as e:
            logger.warning(f"Pod creation failed (attempt {attempt} of {MAX_CREATE_ATTEMPTS}): {e}")
            if attempt < MAX_CREATE_ATTEMPTS and wait(CREATE_RETRY_S):
                return None
    return None


def run_pod(
    client: RunPodClient,
    body: Dict[str, Any],
    read_status: Callable[[], Optional[Dict[str, Any]]],
    cancel_event=None,
    max_runtime_s: float = POD_MAX_RUNTIME_S,
    poll_s: float = POLL_S,
    on_progress: Optional[Callable[[str], None]] = None,
    read_started: Optional[Callable[[], bool]] = None,
    boot_timeout_s: Optional[float] = None,
) -> Optional[PodRun]:
    """
    Creates one pod, waits until its status.json appears in R2 (or the pod exits, is cancelled, runs too long,
    never starts its pipeline within boot_timeout_s or cannot be queried) and ALWAYS terminates it afterwards.
    read_started() says whether the pod's pipeline has started (started.json in R2); None disables the check.
    Returns None when no pod could be created.
    """
    def wait(seconds: float) -> bool:
        if cancel_event is not None:
            return cancel_event.wait(timeout=seconds)
        time.sleep(seconds)
        return False

    say = on_progress or logger.info
    boot_timeout_s = BOOT_TIMEOUT_S if boot_timeout_s is None else boot_timeout_s
    created = create_pod_with_retries(client, body, wait)
    if not created:
        return None
    run = PodRun(pod_id=created["id"], started_at=time.time(), pod_info=_pod_summary(created))
    say(f"Created pod {run.pod_id}: {run.pod_info}")
    try:
        errors = 0
        started = read_started is None
        while True:
            if wait(poll_s):
                say("Job cancelled: terminating the pod.")
                run.outcome = "cancelled"
                break
            if time.time() - run.started_at > max_runtime_s:
                say(f"Pod exceeded {max_runtime_s / 60:g} minutes: terminating it.")
                run.outcome = "timeout"
                break
            try:
                status = read_status()
            except Exception as e:
                status = None
                say(f"Could not read status.json: {e}")
            if status:
                run.status = status
                run.outcome = "finished"
                break
            if not started:
                try:
                    started = bool(read_started())
                except Exception as e:
                    say(f"Could not read started.json: {e}")
                if started:
                    say(f"Pipeline started in the pod after {run.elapsed_s:.0f} s (image pull + boot).")
                elif run.elapsed_s > boot_timeout_s:
                    say(f"Pipeline not started after {boot_timeout_s / 60:g} minutes (image pull stuck on this host?): "
                        "terminating the pod. Its log is on console.runpod.io until then.")
                    run.outcome = "boot_timeout"
                    break
            try:
                pod = client.get_pod(run.pod_id)
                errors = 0
            except Exception as e:
                errors += 1
                say(f"Could not read pod {run.pod_id} ({errors} of {STATUS_MAX_ERRORS}): {e}")
                if errors >= STATUS_MAX_ERRORS:
                    run.outcome = "status_error"
                    break
                continue
            info = _pod_summary(pod)
            run.pod_info.update({k: v for k, v in info.items() if v is not None})
            say(f"Pod {pod.get('desiredStatus')} | {run.elapsed_s / 60:.1f} min | {pod.get('lastStatusChange', '')}")
            if pod.get("desiredStatus") in ("EXITED", "TERMINATED"):
                run.status = read_status()   # the pod may have written it just before exiting
                run.outcome = "finished" if run.status else "exited"
                break
    finally:
        run.ended_at = time.time()
        say(f"Terminating pod {run.pod_id}...")
        run.terminated = terminate_pod_reliably(client, run.pod_id)
    return run


def run_job_pods(
    client: RunPodClient,
    make_body: Callable[[int], Dict[str, Any]],
    min_ram_gb: int,
    read_status: Callable[[], Optional[Dict[str, Any]]],
    cancel_event=None,
    max_runtime_s: float = POD_MAX_RUNTIME_S,
    on_progress: Optional[Callable[[str], None]] = None,
    before_retry: Optional[Callable[[], None]] = None,
    read_started: Optional[Callable[[], bool]] = None,
) -> List[PodRun]:
    """
    Runs the job on at most MAX_POD_ATTEMPTS pods. A new pod is started only when the previous one
    reported "ram_insufficient" (asks for the RAM the pod measured it needs), died without a status
    (likely OOM: asks for RAM_RETRY_FACTOR more) or never started its pipeline (same request, new pod). make_body(min_ram_gb) builds the create request (fresh
    credentials for every pod); before_retry() removes the previous pod's status.json.
    """
    say = on_progress or logger.info
    runs: List[PodRun] = []
    ram = int(min_ram_gb)
    for attempt in range(1, MAX_POD_ATTEMPTS + 1):
        if attempt > 1 and before_retry is not None:
            before_retry()
        say(f"--- Pod attempt {attempt} of {MAX_POD_ATTEMPTS} (min RAM {ram} GB) ---")
        run = run_pod(client, make_body(ram), read_status, cancel_event, max_runtime_s, on_progress=say,
                      read_started=read_started)
        if run is None:
            say("No pod could be created with these filters.")
            break
        runs.append(run)
        if not run.terminated or run.outcome in ("cancelled", "timeout", "status_error"):
            break
        state = (run.status or {}).get("state")
        if run.outcome == "finished" and state == "ram_insufficient":
            ram = max(ram + 8, int(math.ceil(float(run.status.get("required_gb", ram)) * 1.2)))
            continue
        if run.outcome == "exited":
            ram = int(math.ceil(ram * RAM_RETRY_FACTOR))
            continue
        if run.outcome == "boot_timeout":
            continue          # same request: RunPod places the new pod, usually on another host
        break  # ok or pipeline_error: retrying would not help
    return runs
