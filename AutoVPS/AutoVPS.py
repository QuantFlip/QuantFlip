

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoVPS – One-Click Pipeline Runner
===================================
This script automates the full AutoVPS discovery pipeline:
  1) get_metadata.py   → {outdir}/{base}.autovps_index.json
  2) get_location.py   → {outdir}/{base}.scales.json
  3) get_VPSList.py    → {outdir}/{base}.vps_rank.json / .csv

Usage
-----
python AutoVPS.py \
  --model /path/to/model.(gguf|safetensors|safetensors.index.json) \
  --outdir runs \
  --depth ends --weights 0.40,0.35,0.25 --topk 100 \
  [--tag EXP1] [--overwrite]

Notes
-----
- This orchestrator shells out to sibling scripts located next to this file.
- It does not install dependencies; ensure `safetensors` and `gguf` are available.
- Paths are resolved absolutely; outputs are grouped under `--outdir`.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable or "python3"

GET_METADATA = os.path.join(SCRIPT_DIR, "get_metadata.py")
GET_LOCATION = os.path.join(SCRIPT_DIR, "get_location.py")
GET_VPSLIST = os.path.join(SCRIPT_DIR, "get_VPSList.py")


@dataclass
class StepResult:
    name: str
    cmd: List[str]
    rc: int
    elapsed: float
    out_path: str


def _run(name: str, cmd: List[str], out_path: str) -> StepResult:
    t0 = time.time()
    print(f"\n[AutoVPS] ▶ {name}:\n$ {shlex.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        rc = 0
    except subprocess.CalledProcessError as e:
        rc = e.returncode
        print(f"[AutoVPS] ✖ {name} failed with code {rc}", file=sys.stderr)
        raise
    finally:
        elapsed = time.time() - t0
        print(f"[AutoVPS] ✔ {name} done in {elapsed:.2f}s. → {out_path}")
    return StepResult(name=name, cmd=cmd, rc=rc, elapsed=elapsed, out_path=out_path)


def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required script/file not found: {path}")


def build_outputs(model_path: str, outdir: str, tag: str | None) -> Tuple[str, str, str, str]:
    base = os.path.basename(model_path)
    # remove multi-extensions like .safetensors.index.json → base = model
    for suf in (".safetensors.index.json", ".safetensors", ".gguf"):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    if tag:
        base = f"{base}.{tag}"
    meta_json = os.path.join(outdir, f"{base}.autovps_index.json")
    scales_json = os.path.join(outdir, f"{base}.scales.json")
    rank_json = os.path.join(outdir, f"{base}.vps_rank.json")
    rank_csv  = os.path.join(outdir, f"{base}.vps_rank.csv")
    return meta_json, scales_json, rank_json, rank_csv


def main():
    ap = argparse.ArgumentParser(description="AutoVPS one-click runner")
    ap.add_argument("--model", required=True, help="Path to .gguf | .safetensors | .safetensors.index.json")
    ap.add_argument("--outdir", default="runs", help="Directory for outputs")
    ap.add_argument("--depth", choices=["ends","front","middle","flat"], default="ends", help="Depth prior mode for VPS ranking")
    ap.add_argument("--weights", default="0.40,0.35,0.25", help="Weights for Structural,Encoding,Impact in VPS ranking")
    ap.add_argument("--topk", type=int, default=0, help="Top-K rows in CSV (0=all)")
    ap.add_argument("--tag", default=None, help="Optional tag suffix for output filenames")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = ap.parse_args()

    # Preflight
    ensure_exists(GET_METADATA)
    ensure_exists(GET_LOCATION)
    ensure_exists(GET_VPSLIST)

    model = os.path.abspath(args.model)
    if not os.path.exists(model):
        ap.error(f"Model file not found: {model}")
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    meta_json, scales_json, rank_json, rank_csv = build_outputs(model, outdir, args.tag)

    # Overwrite guard
    if not args.overwrite:
        for p in (meta_json, scales_json, rank_json, rank_csv):
            if os.path.exists(p):
                print(f"[AutoVPS] Output exists, use --overwrite to replace: {p}")
                pass

    # 1) get_metadata.py
    md_cmd = [PY, GET_METADATA, model, "--out", meta_json]
    _run("get_metadata", md_cmd, meta_json)

    # 2) get_location.py
    loc_cmd = [PY, GET_LOCATION, model, "--out", scales_json]
    _run("get_location", loc_cmd, scales_json)

    # 3) get_VPSList.py
    vps_cmd = [
        PY, GET_VPSLIST, scales_json,
        "--out", rank_json,
        "--csv", rank_csv,
        "--depth", args.depth,
        "--weights", args.weights,
    ]
    if args.topk:
        vps_cmd += ["--topk", str(args.topk)]
    _run("get_VPSList", vps_cmd, f"{rank_json} & {rank_csv}")

    print("\n[AutoVPS] All done.")
    print(f"  metadata : {meta_json}")
    print(f"  scales   : {scales_json}")
    print(f"  rank(json): {rank_json}")
    print(f"  rank(csv) : {rank_csv}")


if __name__ == "__main__":
    main()