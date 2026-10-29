#!/usr/bin/env python3
"""
collect_stats.py — One CSV row per scan from many stats.json: the training table for predicting
RAM, time, disk, cost and quality of a scan.

Nested fields become dotted columns (run_info.inputs.n_images, run_info.object_images.texture_gradient.mean ...);
every pipeline step becomes step.<slug>.<field> (duration_sec, peak_ram_mb, cpu_utilization, io_write_gb,
peak_disk_gb ...). Runs with different fields still line up: missing values stay empty.

  python tools/collect_stats.py results/*/stats.json output/stats.json -o runs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict


def slug(name: str) -> str:
    """"4.2 OpenMVS DensifyPointCloud" -> "4_2_openmvs_densifypointcloud"."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def flatten(value: Any, prefix: str, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(value, list):
        if all(not isinstance(v, (dict, list)) for v in value):
            out[prefix] = ";".join(str(v) for v in value)
            if value and all(isinstance(v, (int, float)) for v in value):
                out[prefix + ".sum"] = sum(value)
        else:
            for i, v in enumerate(value):
                flatten(v, f"{prefix}.{i}", out)
    else:
        out[prefix] = value


def row_from_stats(data: Dict[str, Any], source: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"source": source}
    steps = data.pop("steps", []) or []
    models = data.pop("exported_models", []) or []
    flatten(data, "", row)
    for s in steps:
        key = "step." + slug(s.get("name", "?"))
        for field, v in s.items():
            if field not in ("name", "start_time_iso", "end_time_iso", "extra_data"):
                row[f"{key}.{field}"] = v
    for m in models:
        key = "model." + slug(m.get("file_path", "?"))
        for field in ("vertex_count", "face_count", "file_size_mb", "is_watertight"):
            row[f"{key}.{field}"] = m.get(field)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stats", nargs="+", help="stats.json files")
    ap.add_argument("-o", "--out", default="runs.csv")
    args = ap.parse_args(argv)

    rows = []
    for name in args.stats:
        try:
            rows.append(row_from_stats(json.loads(Path(name).read_text(encoding="utf-8")), name))
        except (OSError, ValueError) as e:
            print(f"skip {name}: {e}")
    if not rows:
        print("No stats.json could be read.")
        return 1
    columns = ["source"] + sorted({k for r in rows for k in r} - {"source"})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} run(s), {len(columns)} columns -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
