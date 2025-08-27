#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoVPS – Locate Scale Offsets
==============================
Given a model file, output where the *scales* live.

Supported inputs:
  • Single-file safetensors:               model.safetensors
  • Multi-shard safetensors (HuggingFace): model.safetensors.index.json
  • GGUF:                                  model.gguf  (scales embedded; we
                                             report quant layout hints only)

Why this script?
  - For safetensors, offsets are in the header JSON (`data_offsets`), not by
    guessing order. We parse the header to get exact byte ranges per tensor and
    match weights to their companion scales (e.g., `*.weight` → `*.weight_scale`).
  - For GGUF, most quant schemes (Qx_K / IQx_*) embed scales inside the tensor
    blocks. We expose quant types & block sizes to support downstream layout math.

Usage
-----
  pip install safetensors gguf
  python get_location.py /path/to/model.safetensors --out scales.json --print
  python get_location.py /path/to/model.safetensors.index.json -o scales.json
  python get_location.py /path/to/model.gguf -o scales.json

Output
------
JSON with two top-level keys:
  - model: {format, notes, ...}
  - locations: [
      {backend, layer_index, weight_name, weight_shape,
       scale_name, scale_shape, scale_dtype, shard, offset_start, offset_end}
    ]
For GGUF, entries have `embedded: true` and no file offsets.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# ---------- Utilities: naming rules ----------
LAYER_INDEX_PATTERNS = (
    re.compile(r"(?:^|\.)layers\.(\d+)\."),  # e.g., model.layers.25.self_attn.o_proj.weight
    re.compile(r"(?:^|\.)blk\.(\d+)\."),     # e.g., blk.25.attn_output.weight (BitNet style)
)


def canonical_layer_index(tensor_name: str) -> Optional[int]:
    for pat in LAYER_INDEX_PATTERNS:
        m = pat.search(tensor_name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


EXCLUDE_TOKENS = (
    ".bias", "bias.",
    "layer_norm", "layernorm", "rms_norm", "norm.", "output_norm",
    "rotary", "rope", "pos_emb", "position", "alibi",
    "embed", "embedding", "tok_embeddings", "token_embd", "lm_head",
)
INCLUDE_HINTS = (
    # Attention
    "attn", "self_attn", "q_proj", "k_proj", "v_proj", "o_proj",
    "out_proj", "dense", "query_key_value", "Wqkv",
    # FFN / MLP
    "ffn", "mlp", "gate_proj", "up_proj", "down_proj",
    "dense_h_to_4h", "dense_4h_to_h",
    # MoE experts
    "experts", "w1", "w2", "w3",
)
WEIGHT_SUFFIXES = (".weight", ".qweight")


def is_quantizable_weight(name: str, shape: Optional[List[int]] = None) -> bool:
    if not name.endswith(WEIGHT_SUFFIXES):
        return False
    lname = name.lower()
    if any(tok in lname for tok in EXCLUDE_TOKENS):
        return False
    if any(h in lname for h in INCLUDE_HINTS):
        if shape is None:
            return True
        return len(shape) == 2 and min(shape) >= 16
    return False


# ---------- SAFETENSORS: header parsing with offsets ----------
@dataclass
class STTensor:
    name: str
    dtype: str
    shape: List[int]
    offset_start: int
    offset_end: int
    shard: str


def read_safetensors_header_with_offsets(path: str) -> Dict[str, STTensor]:
    """Parse a single .safetensors file and return tensors with absolute file offsets.
    Offsets in the header are relative to the beginning of the *data* section.
    The data section starts right after the 8-byte length prefix + header JSON.
    """
    out: Dict[str, STTensor] = {}
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little", signed=False)
        header_bytes = f.read(header_len)
        try:
            header = json.loads(header_bytes)
        except Exception as e:
            raise RuntimeError(f"Invalid safetensors header in {path}: {e}")
        data_base = 8 + header_len
        for name, info in header.items():
            if name in ("__metadata__", "metadata") and not isinstance(info, dict):
                continue
            if not isinstance(info, dict):
                continue
            if "data_offsets" not in info:
                continue
            start, end = info["data_offsets"]
            dtype = str(info.get("dtype", "unknown"))
            shape = list(info.get("shape", []))
            out[name] = STTensor(
                name=name,
                dtype=dtype,
                shape=shape,
                offset_start=data_base + int(start),
                offset_end=data_base + int(end),
                shard=os.path.abspath(path),
            )
    return out


def enumerate_safetensors_from_index(index_json: str) -> Dict[str, STTensor]:
    with open(index_json, "r", encoding="utf-8") as fh:
        idx = json.load(fh)
    weight_map: Dict[str, str] = idx.get("weight_map", {})
    base_dir = os.path.dirname(os.path.abspath(index_json))
    shard_paths = sorted({os.path.join(base_dir, v) for v in weight_map.values()})

    merged: Dict[str, STTensor] = {}
    for shard in shard_paths:
        if not os.path.exists(shard):
            alt = os.path.join(os.getcwd(), os.path.basename(shard))
            shard = alt if os.path.exists(alt) else shard
        if not os.path.exists(shard):
            print(f"[WARN] shard missing: {shard}", file=sys.stderr)
            continue
        merged.update(read_safetensors_header_with_offsets(shard))
    return merged


# ---------- SAFETENSORS: match weights → scales ----------
SCALE_SUFFIX_CANDIDATES = (".weight_scale", ".scales", ".scale")
EXTRA_SUFFIX_CANDIDATES = (".qzeros", ".g_idx")


def match_weight_to_scale_st(tensors: Dict[str, STTensor]) -> List[Tuple[STTensor, STTensor]]:
    names = set(tensors.keys())
    pairs: List[Tuple[STTensor, STTensor]] = []
    for name, t in tensors.items():
        if not is_quantizable_weight(name, t.shape):
            continue
        stem = None
        if name.endswith(".qweight"):
            stem = name[:-8]
            candidates = (stem + ".scales", stem + ".weight_scale", stem + ".scale")
        elif name.endswith(".weight"):
            stem = name[:-7]
            candidates = (stem + ".weight_scale", stem + ".scales", stem + ".scale")
        else:
            continue
        for c in candidates:
            if c in names:
                pairs.append((t, tensors[c]))
                break
    return pairs


# ---------- GGUF: layout hints (no absolute offsets here) ----------
try:
    import gguf  # type: ignore
except Exception:
    gguf = None

BLOCK_SIZE_HINTS = {
    "Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32, "Q8_0": 32,
    "Q2_K": 256, "Q3_K": 256, "Q4_K": 256, "Q5_K": 256, "Q6_K": 256, "Q8_K": 256,
    "IQ1_S": 256, "IQ2_XXS": 256, "IQ2_XS": 256, "IQ2_S": 256,
    "IQ3_XXS": 256, "IQ3_XS": 256, "IQ3_S": 256,
    "IQ4_XS": 256, "IQ4_NL": 256,
    "IQ5_XS": 256, "IQ5_NL": 256,
}


def normalize_qtype_name(qtype_obj) -> str:
    s = str(qtype_obj).upper()
    s = s.replace("GGMLQUANTIZATIONTYPE.", "").replace("GGMLTYPE.", "").replace("GGML_TYPE_", "")
    return s.strip()


# ---------- Output schema ----------
@dataclass
class Location:
    backend: str                     # safetensors | gguf
    layer_index: Optional[int]
    weight_name: str
    weight_shape: List[int]
    scale_name: Optional[str]
    scale_shape: Optional[List[int]]
    scale_dtype: Optional[str]
    shard: Optional[str]             # abs path for safetensors; None for gguf
    offset_start: Optional[int]      # absolute byte offset in shard/file
    offset_end: Optional[int]
    embedded: bool                   # True for gguf
    quant_type: Optional[str]        # gguf only
    block_size_hint: Optional[int]   # gguf only


@dataclass
class Report:
    model: Dict
    locations: List[Location]

    def to_json(self) -> str:
        return json.dumps({
            "model": self.model,
            "locations": [asdict(x) for x in self.locations],
        }, ensure_ascii=False, indent=2)


# ---------- Builders ----------

def build_for_safetensors(path: str) -> Report:
    if path.endswith(".index.json"):
        st_tensors = enumerate_safetensors_from_index(path)
        model_meta = {"format": "safetensors", "sharded": True}
    else:
        st_tensors = read_safetensors_header_with_offsets(path)
        model_meta = {"format": "safetensors", "sharded": False}

    pairs = match_weight_to_scale_st(st_tensors)
    locs: List[Location] = []
    for w, s in pairs:
        locs.append(Location(
            backend="safetensors",
            layer_index=canonical_layer_index(w.name),
            weight_name=w.name,
            weight_shape=w.shape,
            scale_name=s.name,
            scale_shape=s.shape,
            scale_dtype=s.dtype,
            shard=s.shard,
            offset_start=s.offset_start,
            offset_end=s.offset_end,
            embedded=False,
            quant_type=None,
            block_size_hint=None,
        ))

    # Summary counts
    model_meta["n_pairs"] = len(locs)
    model_meta["notes"] = "Offsets derived from safetensors header data_offsets"
    return Report(model=model_meta, locations=locs)


def build_for_gguf(path: str) -> Report:
    if gguf is None:
        raise RuntimeError("gguf not installed: pip install gguf")
    rdr = gguf.GGUFReader(path)
    try:
        names = rdr.get_tensor_names()
    except Exception:
        names = list(getattr(rdr, "tensors", {}).keys())

    locs: List[Location] = []
    for n in names:
        try:
            info = rdr.get_tensor_info(n)
            shape = list(getattr(info, "shape", [])) or list(getattr(info, "ne", []))
            qt = normalize_qtype_name(getattr(info, "type", getattr(info, "tensor_type", "UNKNOWN")))
        except Exception:
            shape, qt = [], "UNKNOWN"
        if not is_quantizable_weight(n, shape):
            continue
        q_upper = qt.upper()
        quantized = ("Q" in q_upper) and not any(x in q_upper for x in ("BF16", "F16", "F32"))
        if not quantized:
            continue
        locs.append(Location(
            backend="gguf",
            layer_index=canonical_layer_index(n),
            weight_name=n,
            weight_shape=shape,
            scale_name=None,
            scale_shape=None,
            scale_dtype=None,
            shard=None,
            offset_start=None,
            offset_end=None,
            embedded=True,
            quant_type=qt,
            block_size_hint=BLOCK_SIZE_HINTS.get(qt),
        ))

    model_meta = {"format": "gguf", "notes": "Scales embedded per quant block; no absolute file offsets."}
    model_meta["n_quantized_tensors"] = len(locs)
    return Report(model=model_meta, locations=locs)


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="AutoVPS: locate scale offsets for safetensors / layout hints for gguf")
    ap.add_argument("model_path", help="Path to .safetensors | .safetensors.index.json | .gguf")
    ap.add_argument("--out", "-o", help="Write JSON report to this path")
    ap.add_argument("--print", dest="do_print", action="store_true", help="Also print report to stdout")
    args = ap.parse_args()

    p = os.path.abspath(args.model_path)
    if not os.path.exists(p):
        print(f"[ERROR] File not found: {p}", file=sys.stderr)
        sys.exit(2)

    lower = p.lower()
    if lower.endswith(".gguf"):
        rep = build_for_gguf(p)
    elif lower.endswith(".safetensors") or lower.endswith(".safetensors.index.json"):
        rep = build_for_safetensors(p)
    else:
        print("[ERROR] Unsupported file type", file=sys.stderr)
        sys.exit(2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(rep.to_json())
        print(f"[OK] wrote: {args.out}")

    if args.do_print or not args.out:
        print(rep.to_json())


if __name__ == "__main__":
    main()
