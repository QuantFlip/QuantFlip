

# AutoVPS (Automatic Vulnerable Parameter Search)

> A reproducible pipeline that **enumerates and ranks vulnerable scale bits** in quantized LLMs. It powers the experiments described in our paper (QuantFlip / AutoVPS).

---

## ✨ What it does
- **Format support**: `safetensors` (single file / multi‑shard) and `GGUF`.
- **Zero‑weight loading**: parses **headers/metadata only**; no large tensors are loaded.
- **Precise locating**
  - **safetensors**: reads `data_offsets` from the header to get **exact byte ranges** of scale tensors; evaluates the **most catastrophic BF16 bit** per scale.
  - **gguf**: identifies quantization type and block size. (Scales are embedded inside blocks; we export **layout hints**, not absolute offsets.)
- **Composite scoring (AutoVPS)**: structural priors (layer depth/role/size) + encoding brittleness (bit‑flip Δlog2) + propagation priors (residual/gate/attn).
- **Reproducible outputs**: metadata index, scale locations, and a ranked list of bit‑level fault targets (JSON/CSV).

---

## 📁 Repository layout
```
AutoVPS/
├── AutoVPS.py           # Orchestrator: one‑click pipeline
├── get_metadata.py      # Parse model metadata (container/quant info/tensor index)
├── get_location.py      # Locate scales (file offsets for safetensors; layout hints for gguf)
├── get_VPSList.py       # Score & rank: outputs byte/bit addresses to flip
└── runs/                # Default output directory (configurable)
```

---

## 🔧 Requirements
- Python ≥ 3.8
- Install dependencies:
  ```bash
  pip install safetensors gguf
  ```
> This repo does **not** auto‑install dependencies. For offline setups, prepare local wheels.

---

## 🚀 Quickstart (one command)
Run the full pipeline — **metadata → locations → ranking**:
```bash
python AutoVPS.py \
  --model /path/to/model.(gguf|safetensors|safetensors.index.json) \
  --outdir runs \
  --depth ends --weights 0.40,0.35,0.25 --topk 100 \
  --tag EXP1 --overwrite
```
Outputs in `{outdir}`:
- `{base}.autovps_index.json` – model metadata index
- `{base}.scales.json` – scale locations (with **absolute byte offsets** for safetensors)
- `{base}.vps_rank.json/.csv` – ranked fault targets (with **byte offset + bit index**)

---

## 🧩 Step‑by‑step (modular)
### 1) Metadata parsing
```bash
python get_metadata.py /path/to/model.gguf --out runs/model.autovps_index.json
python get_metadata.py /path/to/model.safetensors --out runs/model.autovps_index.json
python get_metadata.py /path/to/model.safetensors.index.json --out runs/model.autovps_index.json
```
**Purpose**
- Enumerate *quantizable* linear weights in Attn/FFN (excludes Norm/Bias/Embeddings).
- For GGUF, collect quant types; for safetensors, collect names/shapes.

### 2) Locate scales (with file offsets)
```bash
python get_location.py /path/to/model.(gguf|safetensors|safetensors.index.json) \
  --out runs/model.scales.json --print
```
**Highlights**
- **safetensors**: pairs `*.weight|*.qweight` with `*.weight_scale|*.scales|*.scale` and reads header `data_offsets` to get:
  - `offset_start` / `offset_end` (absolute byte offsets)
  - `scale_dtype` (e.g., `BF16`) and `shape` (often `[1]`)
- **gguf**: `embedded=true`, export `quant_type` and `block_size_hint` (K‑family typically `256`).

### 3) Rank vulnerable positions (VPS)
```bash
python get_VPSList.py runs/model.scales.json \
  --out runs/model.vps_rank.json --csv runs/model.vps_rank.csv \
  --depth ends --weights 0.40,0.35,0.25 --topk 50
```
**Scoring**
- **StructuralScore**: depth prior (`front|middle|ends|flat`) + matrix size + light role bias.
- **EncodingScore**: try all 16 **BF16** bit flips on the scalar scale, maximize `|Δ log2(scale)|` → pick the worst bit.
- **ImpactScore**: higher for `o_proj/down/gate/q`, lower for `v/k` (tunable in code).

---

## 📤 Output schema (at a glance)
### `*.scales.json`
```json
{
  "model": {"format": "safetensors", "sharded": false, "n_pairs": 224},
  "locations": [
    {
      "backend": "safetensors",
      "layer_index": 12,
      "weight_name": "model.layers.12.self_attn.o_proj.weight",
      "weight_shape": [4096, 4096],
      "scale_name": "model.layers.12.self_attn.o_proj.weight_scale",
      "scale_shape": [1],
      "scale_dtype": "BF16",
      "shard": "/abs/path/model.safetensors",
      "offset_start": 12345678,
      "offset_end": 12345680,
      "embedded": false
    }
  ]
}
```

### `*.vps_rank.csv`
| rank | backend | layer | role  | weight_name | byte_offset | bit_index | global_bit_offset | structural | encoding | impact | final |
|----:|---------|------:|------|-------------|------------:|----------:|------------------:|-----------:|---------:|-------:|------:|
| 1   | safetensors | 12 | attn_o | …o_proj…   | 12345678    | 7         | 987654321         | 0.92       | 0.98     | 1.00   | 0.96  |

> **Bit convention**: `bit_index=0` is the **LSB** of the 16‑bit BF16 word as stored on disk (**little‑endian**).

---

## ✅ Supported cases & assumptions
- **safetensors**
  - Pairing rules covered: `*.weight ↔ *.weight_scale|*.scales|*.scale`, `*.qweight ↔ *.scales` (with optional `qzeros/g_idx`).
  - Multi‑shard (`*.safetensors.index.json`) is handled by iterating shards.
- **Encoding evaluator**
  - By default, only **BF16 scalar scales** are evaluated bit‑by‑bit. Extend `get_VPSList.py` for FP16/FP32/integer encodings if needed.

---

## 🧪 Reproducibility tips
- Export two leaderboards: **global Top‑K** and **per‑layer Top‑1** for fair coverage.
- Record and fix `--depth/--weights` when reporting results.

---

## ❓ FAQ
- **Why do I see `quantized=false`?**
  - Your naming may differ (e.g., custom `…scale_bf16`). Add the suffix to matchers in `get_location.py`.
- **Why no offsets for GGUF?**
  - By design: scales live inside quant blocks. You’ll need *tensor data start + quant layout* to compute in‑block offsets; this can be added later.
- **Endian/bit numbering?**
  - safetensors values are read little‑endian; `bit_index=0` is the LSB of the BF16 word.

---

## ⚖️ Ethics
AutoVPS is intended for **security research and defensive evaluation**. **Do not** use it to violate laws or service terms. The authors disclaim liability for misuse.

---

## 🤝 Contributing
- If you adjust priors (Structural/Impact), please share evidence or benchmarks.