#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoVPS – Vulnerable Parameter Search (VPS) Ranker
==================================================
Read `scales.json` (produced by get_location.py) and compute a ranked list of
vulnerable scale-bit candidates suitable for fault injection. The ranking
combines:
  • StructuralScore   – layer depth, role, and affected weight size
  • EncodingScore     – BF16 bit-flip impact on the scale value (reads 2 bytes)
  • ImpactScore       – residual path position & gate/attention priors

Outputs:
  - vps_rank.json : detailed ranking with file/byte/bit positions when available
  - vps_rank.csv  : convenient table (subset of fields)

Notes
-----
* For safetensors: we compute exact byte offsets from `scales.json` and
  evaluate all 16 BF16 bit flips on the scalar scale (shape [1]).
* For gguf: per-block scales are embedded; we cannot resolve absolute file
  offsets here. We still produce structural/impact scores and an encoding
  heuristic per quant type, but byte/bit addresses are omitted (embedded=True).
* Bit indexing convention: bit 0 = LSB of the 16-bit BF16 word as stored in
  file (little-endian assumed in safetensors). Global bit offset is
  `byte_offset * 8 + bit_index`.

Usage
-----
  python get_VPSList.py scales.json --out vps_rank.json --csv vps_rank.csv

You may tune weights/priors with CLI flags. No tensors are loaded into RAM;
we only read the 2 bytes for each BF16 scale when available.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# ---------------------
# Helpers: name parsing
# ---------------------
LAYER_INDEX_PATTERNS = (
    re.compile(r"(?:^|\.)layers\.(\d+)\."),
    re.compile(r"(?:^|\.)blk\.(\d+)\."),
)

ROLE_MAP = {
    # attention projections
    "q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_o",
    "query_key_value": "attn_qkv", "Wqkv": "attn_qkv",
    "out_proj": "attn_o", "dense": "attn_o",  # Falcon/NeoX variants
    # ffn / mlp
    "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down",
    "dense_h_to_4h": "ffn_up", "dense_4h_to_h": "ffn_down",
    # MoE
    "w1": "ffn_gate", "w2": "ffn_up", "w3": "ffn_down",
}

ROLE_WEIGHTS_IMPACT = {
    # Higher means more likely to cause catastrophic propagation
    "attn_o": 1.00,
    "attn_q": 0.95,
    "attn_qkv": 0.95,
    "ffn_down": 0.95,
    "ffn_gate": 0.95,
    "ffn_up": 0.90,
    "attn_v": 0.60,
    "attn_k": 0.50,
}

# -------------------------
# BF16 utilities (no numpy)
# -------------------------

def bf16_word_to_float(word: int) -> float:
    """Interpret a 16-bit integer as IEEE-754 bfloat16 (little-endian on disk).
    We up-convert to float32 by shifting into high 16 bits of a 32-bit word.
    """
    # Construct a 32-bit value: bf16 bits in the high 16, low 16 zeroed
    u32 = (word & 0xFFFF) << 16
    return struct.unpack('>f', struct.pack('>I', u32))[0]


def bf16_flip_bit(word: int, bit_index: int) -> int:
    return word ^ (1 << bit_index)


def safe_log2_ratio(a: float, b: float) -> float:
    """Return |log2(a) - log2(b)| with guards for zero/NaN/Inf.
    Interpreted as magnitude change in powers of two. If invalid, return large.
    """
    def _norm(x: float) -> float:
        if not math.isfinite(x) or x == 0.0:
            return float('nan')
        return abs(x)
    a_ = _norm(a)
    b_ = _norm(b)
    if not math.isfinite(a_) or not math.isfinite(b_):
        return 64.0  # effectively catastrophic
    return abs(math.log2(a_) - math.log2(b_))


# ----------------------
# Structural computations
# ----------------------

def extract_layer_index(name: str) -> Optional[int]:
    for pat in LAYER_INDEX_PATTERNS:
        m = pat.search(name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def extract_role(name: str) -> Optional[str]:
    lname = name.lower()
    for k, v in ROLE_MAP.items():
        if k in lname:
            return v
    # try generic substrings
    if ".q_proj" in lname: return "attn_q"
    if ".k_proj" in lname: return "attn_k"
    if ".v_proj" in lname: return "attn_v"
    if ".o_proj" in lname: return "attn_o"
    if "gate_proj" in lname: return "ffn_gate"
    if "up_proj" in lname: return "ffn_up"
    if "down_proj" in lname: return "ffn_down"
    return None


# ----------------------
# Scoring configuration
# ----------------------
@dataclass
class ScoringConfig:
    w_struct: float = 0.40
    w_encoding: float = 0.35
    w_impact: float = 0.25
    # depth shaping: ends>middle by default (two-sided ramp)
    depth_mode: str = "ends"  # choices: ends|front|middle|flat


# ----------------------
# Data structures
# ----------------------
@dataclass
class ScaleLoc:
    backend: str
    layer_index: Optional[int]
    weight_name: str
    weight_shape: List[int]
    scale_name: Optional[str]
    scale_shape: Optional[List[int]]
    scale_dtype: Optional[str]
    shard: Optional[str]
    offset_start: Optional[int]
    offset_end: Optional[int]
    embedded: bool
    quant_type: Optional[str]
    block_size_hint: Optional[int]


@dataclass
class Candidate:
    # identity
    backend: str
    layer: Optional[int]
    role: Optional[str]
    weight_name: str
    scale_name: Optional[str]
    # addressing (safetensors only)
    shard: Optional[str]
    byte_offset: Optional[int]
    bit_index: Optional[int]  # best bit to flip (0..15) for safetensors
    global_bit_offset: Optional[int]
    # sizes & types
    weight_rows: int
    weight_cols: int
    scale_dtype: Optional[str]
    quant_type: Optional[str]
    embedded: bool
    # scores
    structural: float
    encoding: float
    impact: float
    final: float
    # predicted effect
    bf16_word: Optional[int]
    orig_scale: Optional[float]
    flipped_scale: Optional[float]
    delta_log2: Optional[float]


# ----------------------
# Structural / impact scores
# ----------------------

def depth_weight(layer: Optional[int], n_layers: int, mode: str) -> float:
    if layer is None or n_layers <= 1:
        return 1.0
    x = layer / (n_layers - 1)
    if mode == "front":
        return 1.0 - x  # early layers high
    if mode == "middle":
        return 1.0 - abs(x - 0.5) * 2.0  # peak at middle
    if mode == "ends":
        # emphasize both ends: triangle with peaks at 0 and 1, valley at middle
        return abs(x - 0.5) * 2.0
    return 1.0  # flat


def structural_score(weight_shape: List[int], layer: Optional[int], n_layers: int, role: Optional[str], cfg: ScoringConfig) -> float:
    rows = weight_shape[0] if len(weight_shape) > 0 else 0
    cols = weight_shape[1] if len(weight_shape) > 1 else 0
    numel = max(1, rows * cols)
    size_term = min(1.0, math.log10(numel + 1) / 6.0)  # ~1.0 for >=1e6 elems
    depth_term = depth_weight(layer, n_layers, cfg.depth_mode)
    # slight role bias also in structural (o_proj/down slightly higher)
    role_bias = {
        "attn_o": 1.05, "ffn_down": 1.05,
        "ffn_gate": 1.03, "attn_q": 1.02, "ffn_up": 1.01,
    }.get(role or "", 1.0)
    return max(0.0, min(1.0, 0.5 * size_term + 0.5 * depth_term)) * role_bias


def impact_score(role: Optional[str]) -> float:
    return ROLE_WEIGHTS_IMPACT.get(role or "", 0.7)


# ----------------------
# Encoding score (safetensors BF16)
# ----------------------

def read_bf16_word(shard_path: str, byte_offset: int) -> int:
    with open(shard_path, 'rb') as f:
        f.seek(byte_offset)
        b = f.read(2)
        if len(b) != 2:
            raise IOError(f"Failed to read 2 bytes at {byte_offset} from {shard_path}")
        # little-endian 16-bit
        return int.from_bytes(b, 'little', signed=False)


def encoding_score_bf16(word: int) -> Tuple[float, int, float, float]:
    """Evaluate all 16 bit flips; return (score, best_bit, orig_scale, flipped_scale).
    Score is normalized from delta_log2 (capped to 16.0) into [0,1]. Higher is worse.
    """
    s0 = bf16_word_to_float(word)
    best_bit = 0
    best_delta = -1.0
    best_val = s0
    for b in range(16):
        w = bf16_flip_bit(word, b)
        s1 = bf16_word_to_float(w)
        d = safe_log2_ratio(s0, s1)
        if d > best_delta:
            best_delta = d
            best_bit = b
            best_val = s1
    # normalize: treat 16.0 in log2 space as ~max catastrophic
    enc = min(1.0, best_delta / 16.0)
    return enc, best_bit, s0, best_val


def encoding_score_heuristic_for_gguf(quant_type: Optional[str]) -> Tuple[float, int]:
    # Without exact per-block scale value, approximate by quant type family.
    if not quant_type:
        return 0.5, 7  # mid exponent bit as a placeholder
    qt = quant_type.upper()
    # IQ*, Q*_K store per-block scales typically in f16/bf16/int encodings depending on type
    # We assume high vulnerability (0.8) for types with explicit exponent-bearing floats.
    if any(prefix in qt for prefix in ("IQ", "Q*_K", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K")):
        return 0.8, 7
    # Legacy Q4_0/Q5_0 etc: still nontrivial, but treat slightly lower
    if any(prefix in qt for prefix in ("Q4_", "Q5_", "Q8_")):
        return 0.7, 7
    return 0.5, 7


# ----------------------
# Main pipeline
# ----------------------

def load_locations(scales_json: str) -> List[ScaleLoc]:
    with open(scales_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    locs = []
    for entry in data.get('locations', []):
        locs.append(ScaleLoc(
            backend=entry.get('backend'),
            layer_index=entry.get('layer_index'),
            weight_name=entry.get('weight_name'),
            weight_shape=list(entry.get('weight_shape') or []),
            scale_name=entry.get('scale_name'),
            scale_shape=list(entry.get('scale_shape') or []),
            scale_dtype=entry.get('scale_dtype'),
            shard=entry.get('shard'),
            offset_start=entry.get('offset_start'),
            offset_end=entry.get('offset_end'),
            embedded=bool(entry.get('embedded')),
            quant_type=entry.get('quant_type'),
            block_size_hint=entry.get('block_size_hint'),
        ))
    return locs


def infer_layer_count(locs: List[ScaleLoc]) -> int:
    layers = sorted({x.layer_index for x in locs if x.layer_index is not None})
    return (layers[-1] + 1) if layers else 0


def build_candidates(locs: List[ScaleLoc], cfg: ScoringConfig) -> List[Candidate]:
    n_layers = infer_layer_count(locs)
    cands: List[Candidate] = []
    for x in locs:
        role = extract_role(x.weight_name)
        # structural & impact first
        s_struct = structural_score(x.weight_shape, x.layer_index, n_layers, role, cfg)
        s_imp = impact_score(role)

        # encoding
        s_enc = 0.5
        best_bit = None
        word = None
        s0 = None
        s1 = None
        dlog2 = None

        if x.backend == 'safetensors' and (x.scale_dtype or '').upper() in ('BF16', 'BFloat16'.upper()):
            # Require scalar [1] (otherwise offset_end-offset_start > 2)
            if x.offset_start is None or x.shard is None:
                pass
            else:
                try:
                    word = read_bf16_word(x.shard, int(x.offset_start))
                    s_enc, _best_bit, s0, s1 = encoding_score_bf16(word)
                    dlog2 = safe_log2_ratio(s0, s1)
                    best_bit = _best_bit
                except Exception as e:
                    print(f"[WARN] Failed BF16 read/eval for {x.scale_name}: {e}", file=sys.stderr)
        elif x.backend == 'gguf':
            s_enc, best_bit_guess = encoding_score_heuristic_for_gguf(x.quant_type)
            best_bit = best_bit_guess

        # final score
        final = cfg.w_struct * s_struct + cfg.w_encoding * s_enc + cfg.w_impact * s_imp

        rows = x.weight_shape[0] if len(x.weight_shape) > 0 else 0
        cols = x.weight_shape[1] if len(x.weight_shape) > 1 else 0

        cands.append(Candidate(
            backend=x.backend,
            layer=x.layer_index,
            role=role,
            weight_name=x.weight_name,
            scale_name=x.scale_name,
            shard=x.shard,
            byte_offset=x.offset_start,
            bit_index=best_bit,
            global_bit_offset=(x.offset_start * 8 + best_bit) if (x.offset_start is not None and best_bit is not None) else None,
            weight_rows=rows,
            weight_cols=cols,
            scale_dtype=x.scale_dtype,
            quant_type=x.quant_type,
            embedded=x.embedded,
            structural=s_struct,
            encoding=s_enc,
            impact=s_imp,
            final=final,
            bf16_word=word,
            orig_scale=s0,
            flipped_scale=s1,
            delta_log2=dlog2,
        ))
    # sort by final desc
    cands.sort(key=lambda c: c.final, reverse=True)
    return cands


def save_json(cands: List[Candidate], path: str, meta: Dict):
    out = {
        'meta': meta,
        'candidates': [asdict(c) for c in cands],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def save_csv(cands: List[Candidate], path: str, topk: Optional[int] = None):
    fields = [
        'rank', 'backend', 'layer', 'role', 'weight_name', 'scale_name',
        'byte_offset', 'bit_index', 'global_bit_offset',
        'scale_dtype', 'quant_type', 'embedded',
        'structural', 'encoding', 'impact', 'final',
        'orig_scale', 'flipped_scale', 'delta_log2',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i, c in enumerate(cands[:topk] if topk else cands, start=1):
            w.writerow([
                i, c.backend, c.layer, c.role, c.weight_name, c.scale_name,
                c.byte_offset, c.bit_index, c.global_bit_offset,
                c.scale_dtype, c.quant_type, c.embedded,
                f"{c.structural:.4f}", f"{c.encoding:.4f}", f"{c.impact:.4f}", f"{c.final:.4f}",
                f"{c.orig_scale:.6g}" if c.orig_scale is not None else '',
                f"{c.flipped_scale:.6g}" if c.flipped_scale is not None else '',
                f"{c.delta_log2:.6g}" if c.delta_log2 is not None else '',
            ])


def main():
    ap = argparse.ArgumentParser(description='AutoVPS: rank vulnerable scale-bit positions from scales.json')
    ap.add_argument('scales_json', help='Path to scales.json from get_location.py')
    ap.add_argument('--out', '-o', default='vps_rank.json', help='Output JSON path')
    ap.add_argument('--csv', default='vps_rank.csv', help='Output CSV path')
    ap.add_argument('--depth', choices=['ends', 'front', 'middle', 'flat'], default='ends', help='Depth prior mode')
    ap.add_argument('--weights', type=str, default='0.40,0.35,0.25', help='Weights for Structural,Encoding,Impact')
    ap.add_argument('--topk', type=int, default=0, help='Write top-K rows to CSV (0=all)')
    args = ap.parse_args()

    w_struct, w_enc, w_imp = [float(x) for x in args.weights.split(',')]
    cfg = ScoringConfig(w_struct=w_struct, w_encoding=w_enc, w_impact=w_imp, depth_mode=args.depth)

    locs = load_locations(args.scales_json)
    cands = build_candidates(locs, cfg)

    meta = {
        'scales_json': os.path.abspath(args.scales_json),
        'n_candidates': len(cands),
        'weights': asdict(cfg),
        'bit_index_convention': 'bit0 = LSB of BF16 word; little-endian byte order',
    }

    save_json(cands, args.out, meta)
    topk = args.topk if args.topk and args.topk > 0 else None
    save_csv(cands, args.csv, topk=topk)

    print(f"[OK] Wrote {args.out} and {args.csv}. Top-5 preview:")
    for i, c in enumerate(cands[:5], start=1):
        print(f"#{i}: layer={c.layer} role={c.role} backend={c.backend} score={c.final:.4f} addr={c.byte_offset}+b{c.bit_index}")


if __name__ == '__main__':
    main()
