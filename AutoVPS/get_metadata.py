#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoVPS – Model Metadata Enumerator (safetensors & gguf)
--------------------------------------------------------
Reads a model file (.safetensors, .safetensors.index.json, or .gguf) and
produces a structured summary for AutoVPS:
  1) Architecture hints (if available)
  2) Enumerate *quantizable* linear weight tensors by layer
  3) Count how many layers are actually quantized
  4) Locate the scales that "belong" to each quantized layer

Notes
- safetensors: we do NOT load tensors; we use `safe_open` to read names, shapes,
  dtypes. Scale tensors are usually separate (e.g., *.scales, *.scale, *.qzeros).
- gguf: scales are embedded inside quantized tensors. We report quant type and
  (best-effort) block size derived from known ggml/gguf quant schemes.

You may tailor the predicate `is_quantizable_weight()` to your project rules.
If you have stricter rules, edit that function or add patterns to INCLUDE/EXCLUDE.

Usage
-----
python get_metadata.py /path/to/model.gguf --out autovps_index.json
python get_metadata.py /path/to/model.safetensors --out autovps_index.json
python get_metadata.py /path/to/model.safetensors.index.json --out autovps_index.json

Dependencies
------------
  pip install safetensors gguf

This script avoids loading full tensors; it reads headers/metadata only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# Optional imports — we import lazily in handlers
try:
    from safetensors import safe_open as st_safe_open  # type: ignore
except Exception:
    st_safe_open = None  # Will error only when needed

try:
    import gguf  # type: ignore
except Exception:
    gguf = None  # Will error only when needed


# -----------------------------
# Helpers: naming & predicates
# -----------------------------
EXCLUDE_TOKENS = (
    ".bias", "bias.",
    "layer_norm", "layernorm", "rms_norm", "norm.", "output_norm",
    "rotary", "rope", "pos_emb", "position", "alibi",
    "embed", "embedding", "tok_embeddings", "token_embd", "lm_head"
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

LAYER_INDEX_PATTERNS = (
    re.compile(r"(?:^|\.)layers\.(\d+)\."),  # e.g., model.layers.25.self_attn.o_proj.weight
    re.compile(r"(?:^|\.)blk\.(\d+)\."),     # e.g., blk.25.attn_output.weight
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


def is_quantizable_weight(name: str, shape: Optional[List[int]] = None) -> bool:
    """Heuristic: consider only big linear mats used in Attn/FFN (not norms/bias/embeds).
    You can tighten this predicate for your repo by editing INCLUDE/EXCLUDE.
    """
    # Must look like a weight tensor name
    if not name.endswith(WEIGHT_SUFFIXES):
        return False

    lname = name.lower()
    # Exclusions first
    if any(tok in lname for tok in EXCLUDE_TOKENS):
        return False

    # Positive hints
    if any(h in lname for h in INCLUDE_HINTS):
        # Optional shape check: prefer 2-D
        if shape is None:
            return True
        return len(shape) == 2 and min(shape) >= 16  # ignore tiny aux mats
    return False


# Known block sizes for ggml/gguf quant types (best-effort)
# We normalize type strings before lookup (see normalize_qtype_name)
BLOCK_SIZE_HINTS = {
    # Legacy Q4/Q5/Q8
    "Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32, "Q8_0": 32,
    # K-quant (grouped in 256 elements)
    "Q2_K": 256, "Q3_K": 256, "Q4_K": 256, "Q5_K": 256, "Q6_K": 256, "Q8_K": 256,
    # IQ* family (inference-opt)
    "IQ1_S": 256, "IQ2_XXS": 256, "IQ2_XS": 256, "IQ2_S": 256,
    "IQ3_XXS": 256, "IQ3_XS": 256, "IQ3_S": 256,
    "IQ4_XS": 256, "IQ4_NL": 256,
    "IQ5_XS": 256, "IQ5_NL": 256,
}


def normalize_qtype_name(qtype_obj) -> str:
    s = str(qtype_obj).upper()
    # Examples we may see:
    #  - GGMLQuantizationType.Q4_K
    #  - GGMLType.Q5_0
    #  - Q4_K
    #  - GGML_TYPE_Q6_K
    s = s.replace("GGMLQUANTIZATIONTYPE.", "").replace("GGMLTYPE.", "").replace("GGML_TYPE_", "")
    s = s.strip()
    return s


# -----------------------------
# safetensors reader
# -----------------------------
@dataclass
class STTensor:
    name: str
    dtype: str
    shape: List[int]


def _st_iter_file(st_path: str) -> List[STTensor]:
    if st_safe_open is None:
        raise RuntimeError("safetensors not installed: pip install safetensors")

    tensors: List[STTensor] = []
    with st_safe_open(st_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            # Try best to avoid loading tensor data
            dtype = None
            shape = None
            try:
                dtype = str(f.get_dtype(k))  # type: ignore[attr-defined]
            except Exception:
                try:
                    t = f.get_tensor(k)
                    dtype = str(getattr(t, "dtype", "unknown"))
                except Exception:
                    dtype = "unknown"
            try:
                shape = list(f.get_shape(k))  # type: ignore[attr-defined]
            except Exception:
                try:
                    t = f.get_tensor(k)
                    shape = list(getattr(t, "shape", []))
                except Exception:
                    shape = []
            tensors.append(STTensor(name=k, dtype=dtype, shape=shape))
    return tensors


def _st_from_index(index_json: str) -> List[STTensor]:
    # Load all shards listed in the index and merge their keys
    with open(index_json, "r", encoding="utf-8") as fh:
        idx = json.load(fh)
    weight_map: Dict[str, str] = idx.get("weight_map", {})
    base_dir = os.path.dirname(os.path.abspath(index_json))
    shard_paths = sorted({os.path.join(base_dir, v) for v in weight_map.values()})

    all_tensors: List[STTensor] = []
    for sp in shard_paths:
        if not os.path.exists(sp):
            # Try relative to index dir (already), then current dir
            alt = os.path.join(os.getcwd(), os.path.basename(sp))
            if os.path.exists(alt):
                sp = alt
            else:
                print(f"[WARN] Shard not found: {sp}", file=sys.stderr)
                continue
        all_tensors.extend(_st_iter_file(sp))
    return all_tensors


@dataclass
class STScaleLink:
    weight: str
    scales: List[str]
    extras: List[str]  # e.g., qzeros, g_idx


def _st_match_scales(tensors: List[STTensor]) -> Tuple[Dict[str, STTensor], Dict[str, STScaleLink]]:
    by_name: Dict[str, STTensor] = {t.name: t for t in tensors}
    links: Dict[str, STScaleLink] = {}

    names = set(by_name.keys())
    for name, t in by_name.items():
        if not is_quantizable_weight(name, t.shape):
            continue
        stem = None
        scales: List[str] = []
        extras: List[str] = []

        if name.endswith(".qweight"):
            stem = name[:-8]
            # GPTQ-ish
            for cand in (stem + ".scales", stem + ".qzeros", stem + ".g_idx"):
                if cand in names:
                    (scales if cand.endswith(".scales") else extras).append(cand)
        elif name.endswith(".weight"):
            stem = name[:-7]
            # AWQ/BitNet-ish
            for cand in (stem + ".scales", stem + ".scale"):
                if cand in names:
                    scales.append(cand)
            # Alternate common patterns
            for cand in (stem + ".qzeros", stem + ".g_idx"):
                if cand in names:
                    extras.append(cand)
        else:
            continue

        if scales or extras:
            links[name] = STScaleLink(weight=name, scales=scales, extras=extras)
    return by_name, links


# -----------------------------
# gguf reader
# -----------------------------
@dataclass
class GGUFEntry:
    name: str
    shape: List[int]
    qtype: str


def _gguf_read(path: str) -> Tuple[Dict[str, GGUFEntry], Dict[str, str]]:
    if gguf is None:
        raise RuntimeError("gguf not installed: pip install gguf")

    rdr = gguf.GGUFReader(path)
    # Architecture hint if present (not guaranteed)
    meta = {}
    try:
        # common key in llama.cpp exports
        arch = rdr.get_kv("general.architecture")
        if arch is not None:
            meta["architecture"] = arch
    except Exception:
        pass

    entries: Dict[str, GGUFEntry] = {}
    try:
        names = rdr.get_tensor_names()
    except Exception:
        # older/newer API fallback
        names = list(getattr(rdr, "tensors", {}).keys())

    for n in names:
        try:
            info = rdr.get_tensor_info(n)
            # Flexible attribute access across gguf-py versions
            shape = list(getattr(info, "shape", [])) or list(getattr(info, "ne", []))
            qtype_obj = getattr(info, "type", getattr(info, "tensor_type", "UNKNOWN"))
            qtype = normalize_qtype_name(qtype_obj)
        except Exception:
            shape, qtype = [], "UNKNOWN"
        entries[n] = GGUFEntry(name=n, shape=shape, qtype=qtype)

    return entries, meta


# -----------------------------
# Aggregation & reporting
# -----------------------------
@dataclass
class TensorRecord:
    name: str
    backend: str  # safetensors|gguf
    shape: List[int]
    quantized: bool
    layer_index: Optional[int]
    quant_scheme: Optional[str]
    scales: List[str]  # safetensors: names; gguf: empty (embedded)
    scale_embedded: bool
    block_size_hint: Optional[int]


@dataclass
class Report:
    model: Dict
    tensors: List[TensorRecord]

    def to_json(self) -> str:
        return json.dumps({
            "model": self.model,
            "tensors": [asdict(t) for t in self.tensors],
        }, ensure_ascii=False, indent=2)


def build_report_for_safetensors(in_path: str) -> Report:
    if in_path.endswith(".index.json"):
        st_tensors = _st_from_index(in_path)
        model_meta = {"format": "safetensors", "sharded": True}
    else:
        st_tensors = _st_iter_file(in_path)
        model_meta = {"format": "safetensors", "sharded": False}

    by_name, links = _st_match_scales(st_tensors)

    records: List[TensorRecord] = []
    for name, t in by_name.items():
        if not is_quantizable_weight(name, t.shape):
            continue
        lidx = canonical_layer_index(name)
        # Consider quantized if *.qweight or has matched scales
        quantized = name.endswith(".qweight") or (name in links and bool(links[name].scales))
        scales = links[name].scales if name in links else []
        quant_scheme = "GPTQ-like" if name.endswith(".qweight") else ("AWQ/BitNet-like" if scales else None)
        rec = TensorRecord(
            name=name,
            backend="safetensors",
            shape=t.shape,
            quantized=quantized,
            layer_index=lidx,
            quant_scheme=quant_scheme,
            scales=scales,
            scale_embedded=False,
            block_size_hint=None,
        )
        records.append(rec)

    model_meta["n_quantized_layers"] = len({r.layer_index for r in records if r.quantized and r.layer_index is not None})
    model_meta["n_quantized_tensors"] = sum(1 for r in records if r.quantized)

    return Report(model=model_meta, tensors=records)


def build_report_for_gguf(in_path: str) -> Report:
    entries, meta = _gguf_read(in_path)
    model_meta = {"format": "gguf"}
    model_meta.update(meta)

    records: List[TensorRecord] = []
    for name, e in entries.items():
        if not is_quantizable_weight(name, e.shape):
            continue
        lidx = canonical_layer_index(name)
        qt = normalize_qtype_name(e.qtype)
        # Heuristic: anything with a Q* or IQ* is quantized; else assume not
        q_upper = qt.upper()
        quantized = ("Q" in q_upper) and not any(x in q_upper for x in ("BF16", "F16", "F32"))
        block_hint = BLOCK_SIZE_HINTS.get(qt, None)
        rec = TensorRecord(
            name=name,
            backend="gguf",
            shape=e.shape,
            quantized=quantized,
            layer_index=lidx,
            quant_scheme=qt if quantized else None,
            scales=[],
            scale_embedded=True if quantized else False,
            block_size_hint=block_hint,
        )
        records.append(rec)

    model_meta["n_quantized_layers"] = len({r.layer_index for r in records if r.quantized and r.layer_index is not None})
    model_meta["n_quantized_tensors"] = sum(1 for r in records if r.quantized)

    return Report(model=model_meta, tensors=records)


def main():
    ap = argparse.ArgumentParser(description="AutoVPS: enumerate quantizable layers & scales from safetensors/gguf")
    ap.add_argument("model_path", help="Path to .gguf, .safetensors, or .safetensors.index.json")
    ap.add_argument("--out", "-o", help="Output JSON file (AutoVPS index)")
    ap.add_argument("--print", dest="do_print", action="store_true", help="Print JSON to stdout")
    args = ap.parse_args()

    path = os.path.abspath(args.model_path)
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(2)

    lower = path.lower()
    if lower.endswith(".gguf"):
        rep = build_report_for_gguf(path)
    elif lower.endswith(".safetensors") or lower.endswith(".safetensors.index.json"):
        rep = build_report_for_safetensors(path)
    else:
        print("[ERROR] Unsupported file type. Use .gguf or .safetensors(.index.json)", file=sys.stderr)
        sys.exit(2)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(rep.to_json())
        print(f"[OK] Wrote index: {args.out}")

    if args.do_print or not args.out:
        print(rep.to_json())


if __name__ == "__main__":
    main()
