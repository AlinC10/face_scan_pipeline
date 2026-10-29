"""
telemetry.py — Hardware monitoring, execution timing, peak RAM/VRAM, and model stats
(copied from Turntable Studio, pipeline/telemetry.py).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from .resources import available_cpus, total_ram_gb
from .run_metrics import STATS_SCHEMA_VERSION

logger = logging.getLogger("Photogrammetry.Telemetry")


def get_system_ram_mb() -> float:
    """Returns current system RAM used in Megabytes."""
    try:
        vm = psutil.virtual_memory()
        return vm.used / (1024 * 1024)
    except Exception:
        return 0.0


def get_process_tree_ram_mb() -> float:
    """Returns total RSS of the current process and all child processes in Megabytes."""
    try:
        parent = psutil.Process()
        total_rss = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_rss / (1024 * 1024)
    except Exception:
        return 0.0


def get_process_tree_cpu_seconds() -> float:
    """
    CPU seconds (user + system) of this process, its waited-for children (OpenMVS, texrecon, worker pools
    after they exit) and the children still running.
    """
    try:
        p = psutil.Process()
        t = p.cpu_times()
        total = t.user + t.system + getattr(t, "children_user", 0.0) + getattr(t, "children_system", 0.0)
        for child in p.children(recursive=True):
            try:
                ct = child.cpu_times()
                total += ct.user + ct.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total
    except Exception:
        return 0.0


def get_disk_io_bytes() -> tuple:
    """(read, written) bytes of all disks (host-wide inside a container: an indication, not exact)."""
    try:
        io = psutil.disk_io_counters()
        return (io.read_bytes, io.write_bytes) if io else (0, 0)
    except Exception:
        return (0, 0)


def get_dirs_size_bytes(dirs: List[str]) -> int:
    total = 0
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total


def get_gpu_vram_mb() -> Optional[dict]:
    """Queries nvidia-smi for current VRAM usage across all visible NVIDIA GPUs."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3)
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "used_mb": float(parts[2]),
                    "total_mb": float(parts[3]),
                })
        return gpus if gpus else None
    except Exception:
        return None


@dataclass
class StepRecord:
    name: str
    duration_sec: float
    start_time_iso: str
    end_time_iso: str
    peak_ram_mb: float
    peak_vram_mb: Optional[float] = None
    peak_vram_net_mb: Optional[float] = None    # minus the VRAM already used before the run (desktop, other apps)
    cpu_seconds: Optional[float] = None         # process tree, including OpenMVS / texrecon / worker pools
    cpu_utilization: Optional[float] = None     # cpu_seconds / (duration * CPUs): low = waiting on disk or GPU
    io_read_gb: Optional[float] = None
    io_write_gb: Optional[float] = None
    peak_disk_gb: Optional[float] = None        # size of the watched output/work directories during the step
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MeshMetrics:
    file_path: str
    format: str
    vertex_count: int
    face_count: int
    file_size_mb: float
    is_watertight: Optional[bool] = None
    volume_cm3: Optional[float] = None
    bbox_dimensions_mm: Optional[List[float]] = None


class SystemTelemetry:
    """
    Continuous telemetry monitor capturing peak RAM, VRAM, and per-step benchmarks.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        self.start_wall_time = time.time()
        self.steps: List[StepRecord] = []
        self.mesh_metrics: List[MeshMetrics] = []
        self.global_peak_ram_mb: float = get_process_tree_ram_mb()
        self.global_peak_vram_mb: float = 0.0
        # Run parameters for RAM-model calibration (resources, n_images, resolution, workers, RAM check)
        self.run_info: Dict[str, Any] = {}
        self.run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime(self.start_wall_time))
        self.outcome: Dict[str, Any] = {"state": "running"}
        self.current_step: Optional[str] = None   # step running when an exception left the pipeline
        self.cpus = available_cpus()
        self.watch_dirs: List[str] = []
        self.global_peak_disk_gb: float = 0.0
        self.global_peak_swap_mb: float = 0.0
        self._current_step_peak_disk: float = 0.0
        self._disk_every = 5.0          # directory walks are slower than RAM samples
        self._last_disk_sample = 0.0

        # Background RAM/VRAM sampler thread
        self._sampling = False
        self._sampler_thread: Optional[threading.Thread] = None
        self._current_step_peak_ram: float = 0.0
        self._current_step_peak_vram: float = 0.0
        self._lock = threading.Lock()

        # Initial GPU probe: what is already used before the run (Windows desktop ~700 MB) is the baseline
        initial_gpus = get_gpu_vram_mb()
        self.vram_baseline_mb: float = max(g["used_mb"] for g in initial_gpus) if initial_gpus else 0.0
        if initial_gpus:
            self.global_peak_vram_mb = self.vram_baseline_mb

    def start_sampling(self, interval_sec: float = 0.25):
        """Starts asynchronous background telemetry sampling."""
        if self._sampling:
            return
        self._sampling = True
        self._sampler_thread = threading.Thread(
            target=self._sample_loop, args=(interval_sec,), daemon=True
        )
        self._sampler_thread.start()

    def stop_sampling(self):
        """Stops background telemetry sampling."""
        self._sampling = False
        if self._sampler_thread:
            self._sampler_thread.join(timeout=1.0)
            self._sampler_thread = None

    def _sample_loop(self, interval_sec: float):
        while self._sampling:
            ram = get_process_tree_ram_mb()
            gpus = get_gpu_vram_mb()
            vram = max([g["used_mb"] for g in gpus]) if gpus else 0.0

            try:
                swap = psutil.swap_memory().used / (1024 * 1024)
            except Exception:
                swap = 0.0
            disk = None
            if self.watch_dirs and time.time() - self._last_disk_sample >= self._disk_every:
                self._last_disk_sample = time.time()
                disk = get_dirs_size_bytes(self.watch_dirs) / 1024 ** 3

            with self._lock:
                self.global_peak_swap_mb = max(self.global_peak_swap_mb, swap)
                if disk is not None:
                    self.global_peak_disk_gb = max(self.global_peak_disk_gb, disk)
                    self._current_step_peak_disk = max(self._current_step_peak_disk, disk)
                if ram > self.global_peak_ram_mb:
                    self.global_peak_ram_mb = ram
                if ram > self._current_step_peak_ram:
                    self._current_step_peak_ram = ram

                if vram > self.global_peak_vram_mb:
                    self.global_peak_vram_mb = vram
                if vram > self._current_step_peak_vram:
                    self._current_step_peak_vram = vram

            time.sleep(interval_sec)

    @contextmanager
    def step(self, name: str, **extra_data):
        """Context manager to benchmark a specific pipeline step."""
        t0 = time.time()
        start_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))
        logger.info(f"▶ [STEP START] {name}")
        self.current_step = name

        with self._lock:
            self._current_step_peak_ram = get_process_tree_ram_mb()
            gpus = get_gpu_vram_mb()
            self._current_step_peak_vram = max([g["used_mb"] for g in gpus]) if gpus else 0.0
            self._current_step_peak_disk = 0.0
        cpu0 = get_process_tree_cpu_seconds()
        io0 = get_disk_io_bytes()

        try:
            yield
        finally:
            t1 = time.time()
            elapsed = t1 - t0
            end_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t1))

            cpu_s = max(0.0, get_process_tree_cpu_seconds() - cpu0)
            io1 = get_disk_io_bytes()
            disk_now = get_dirs_size_bytes(self.watch_dirs) / 1024 ** 3 if self.watch_dirs else 0.0
            with self._lock:
                step_peak_ram = max(self._current_step_peak_ram, get_process_tree_ram_mb())
                step_peak_vram = self._current_step_peak_vram if self._current_step_peak_vram > 0 else None
                step_peak_disk = max(self._current_step_peak_disk, disk_now)
                self.global_peak_disk_gb = max(self.global_peak_disk_gb, step_peak_disk)

            rec = StepRecord(
                name=name,
                duration_sec=round(elapsed, 2),
                start_time_iso=start_iso,
                end_time_iso=end_iso,
                peak_ram_mb=round(step_peak_ram, 1),
                peak_vram_mb=round(step_peak_vram, 1) if step_peak_vram else None,
                peak_vram_net_mb=round(max(0.0, step_peak_vram - self.vram_baseline_mb), 1) if step_peak_vram else None,
                cpu_seconds=round(cpu_s, 1),
                cpu_utilization=round(cpu_s / max(elapsed * self.cpus, 1e-6), 3),
                io_read_gb=round(max(0, io1[0] - io0[0]) / 1024 ** 3, 3),
                io_write_gb=round(max(0, io1[1] - io0[1]) / 1024 ** 3, 3),
                peak_disk_gb=round(step_peak_disk, 3) if self.watch_dirs else None,
                extra_data=extra_data,
            )
            self.steps.append(rec)
            if sys.exc_info()[0] is None:
                self.current_step = None
            vram_str = f" | Peak VRAM: {rec.peak_vram_mb:.0f} MB" if rec.peak_vram_mb else ""
            logger.info(f"✔ [STEP FINISH] {name} in {elapsed:.2f}s | Peak RAM: {rec.peak_ram_mb:.0f} MB{vram_str}")

    def watch(self, *dirs: str):
        """Directories whose total size is sampled (peak disk usage per step and for the run)."""
        for d in dirs:
            if d and str(d) not in self.watch_dirs:
                self.watch_dirs.append(str(d))

    def set_outcome(self, state: str, **details):
        """"ok" | "error" (+ error_type, error, failed_step) for the run."""
        self.outcome = {"state": state, **details}

    def set_info(self, **info):
        """Adds run parameters (resources, image count, workers, RAM estimate) to stats.json."""
        self.run_info.update(info)

    def record_mesh(self, file_path: str, is_watertight: Optional[bool] = None,
                    volume_cm3: Optional[float] = None, bbox_mm: Optional[List[float]] = None,
                    v_count: Optional[int] = None, f_count: Optional[int] = None):
        """Analyzes and records geometry metrics for an exported 3D model."""
        p = Path(file_path)
        if not p.is_file():
            return

        size_mb = p.stat().st_size / (1024 * 1024)
        ext = p.suffix.lower().lstrip(".")

        vertex_count = v_count or 0
        face_count = f_count or 0

        # Quick parse if not passed
        if vertex_count == 0 and ext in ["obj", "stl", "ply"]:
            try:
                import trimesh
                mesh = trimesh.load(str(p), process=False)
                if hasattr(mesh, "vertices"):
                    vertex_count = len(mesh.vertices)
                    face_count = len(mesh.faces)
                    if is_watertight is None:
                        is_watertight = bool(mesh.is_watertight)
                    if volume_cm3 is None and mesh.is_watertight:
                        # trimesh volume is in mesh units (meters^3 -> cm^3 is * 1e6)
                        volume_cm3 = round(float(mesh.volume) * 1e6, 2)
                    if bbox_mm is None and hasattr(mesh, "extents"):
                        bbox_mm = [round(float(x) * 1000.0, 1) for x in mesh.extents]
            except Exception as e:
                logger.debug(f"Could not inspect mesh {file_path}: {e}")

        m = MeshMetrics(
            file_path=str(p.name),
            format=ext,
            vertex_count=vertex_count,
            face_count=face_count,
            file_size_mb=round(size_mb, 2),
            is_watertight=is_watertight,
            volume_cm3=volume_cm3,
            bbox_dimensions_mm=bbox_mm,
        )
        self.mesh_metrics.append(m)

    def summary(self) -> Dict[str, Any]:
        """Compiles the full telemetry dataset into a structured dictionary."""
        total_wall_sec = round(time.time() - self.start_wall_time, 2)
        return {
            "schema_version": STATS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.start_wall_time)),
            "outcome": self.outcome,
            "total_execution_time_sec": total_wall_sec,
            "total_execution_time_human": f"{total_wall_sec // 60:.0f}m {total_wall_sec % 60:.1f}s",
            "global_peak_ram_mb": round(self.global_peak_ram_mb, 1),
            "global_peak_ram_gb": round(self.global_peak_ram_mb / 1024, 2),
            "global_peak_vram_mb": round(self.global_peak_vram_mb, 1),
            "global_peak_vram_gb": round(self.global_peak_vram_mb / 1024, 2) if self.global_peak_vram_mb > 0 else 0.0,
            "vram_baseline_mb": round(self.vram_baseline_mb, 1),
            "global_peak_vram_net_mb": round(max(0.0, self.global_peak_vram_mb - self.vram_baseline_mb), 1),
            "global_peak_swap_mb": round(self.global_peak_swap_mb, 1),
            "global_peak_disk_gb": round(self.global_peak_disk_gb, 3),
            # cgroup-aware: inside a container psutil alone reports the host RAM
            "system_ram_total_gb": round(total_ram_gb(), 1),
            "run_info": self.run_info,
            "steps": [asdict(s) for s in self.steps],
            "exported_models": [asdict(m) for m in self.mesh_metrics],
        }

    def save_json(self, out_path: Optional[str] = None) -> str:
        """Saves telemetry summary to stats.json."""
        target = Path(out_path) if out_path else (self.output_dir / "stats.json" if self.output_dir else Path("stats.json"))
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.summary()
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Telemetry report saved to: {target.resolve()}")
        return str(target.resolve())

    def print_table(self):
        """Prints a human-readable summary table to the console."""
        data = self.summary()
        print("\n" + "=" * 78)
        print("                   TURNTABLE STUDIO — EXECUTION TELEMETRY")
        print("=" * 78)
        print(f" Total Time : {data['total_execution_time_human']} ({data['total_execution_time_sec']}s)")
        print(f" Peak RAM   : {data['global_peak_ram_gb']} GB ({data['global_peak_ram_mb']} MB) / {data['system_ram_total_gb']} GB Total")
        if data['global_peak_vram_mb'] > 0:
            print(f" Peak VRAM  : {data['global_peak_vram_gb']} GB ({data['global_peak_vram_mb']} MB)")
        print("-" * 78)
        print(f" {'STEP NAME':<32} | {'TIME (s)':<10} | {'PEAK RAM (MB)':<14} | {'VRAM (MB)'}")
        print("-" * 78)
        for s in data["steps"]:
            vram_txt = f"{s['peak_vram_mb']:>8.0f}" if s.get("peak_vram_mb") else "       -"
            print(f" {s['name']:<32} | {s['duration_sec']:>9.2f}s | {s['peak_ram_mb']:>13.1f} | {vram_txt}")
        print("-" * 78)

        if data["exported_models"]:
            print("\n EXPORTED 3D ASSETS:")
            print(f" {'FILE':<24} | {'VERTICES':<10} | {'FACES':<10} | {'SIZE (MB)':<10} | {'WATERTIGHT'}")
            print("-" * 78)
            for m in data["exported_models"]:
                wt = "YES" if m.get("is_watertight") else ("NO" if m.get("is_watertight") is False else "N/A")
                print(f" {m['file_path']:<24} | {m['vertex_count']:>10,d} | {m['face_count']:>10,d} | {m['file_size_mb']:>9.2f}MB | {wt}")
            print("-" * 78)
        print("=" * 78 + "\n")
