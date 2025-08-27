# QuantFlip

QuantFlip is a research framework for exploring **bit-flip vulnerabilities in quantized large language models (LLMs)**.  
It provides tools for locating and ranking vulnerable scaling parameters, enabling reproducible experiments on fault injection and resilience evaluation.

---

## 📂 Repository Structure

- **AutoVPS/**  
  Core implementation of *Automatic Vulnerable Parameter Search (AutoVPS)*.  
  This module parses model metadata, locates scaling factors, and generates ranked candidate lists for fault injection.

- **evaluation/**  
  Contains integration with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for standardized benchmarking of LLM performance before and after fault injection.

- **tools/**  
  External tools integrated as submodules.  
  - `blacksmith/`: A Rowhammer-based fault injection framework.  
    You can download and use it directly after initializing submodules (see below).

---

## 🚀 Getting Started

### 1. Clone with submodules
Since this repository uses git submodules, clone with the `--recursive` flag:
```bash
git clone --recursive https://github.com/QuantFlip/QuantFlip.git
cd QuantFlip
```

If you already cloned without `--recursive`, run:
```bash
git submodule update --init --recursive
```

### 2. Setup Python environment
It is recommended to use **conda** or **venv**:
```bash
conda create -n quantflip python=3.10 -y
conda activate quantflip
pip install -r requirements.txt
```

### 3. Run AutoVPS
```bash
python AutoVPS/get_metadata.py
python AutoVPS/get_location.py
python AutoVPS/get_VPSList.py
```

These scripts extract model metadata, locate scaling factors, and generate a ranked list of vulnerable parameter bits.

---

## 🔧 Using Blacksmith (from tools/)

[Blacksmith](https://github.com/comsec-group/blacksmith) is included as a submodule under `tools/`.  
It provides low-level Rowhammer primitives for inducing hardware bit-flips.

### Steps to use:
```bash
cd tools/blacksmith
make
```

Refer to the [Blacksmith documentation](https://github.com/comsec-group/blacksmith) for detailed usage and experimental methodology.

---

## 📊 Evaluation

To evaluate model robustness with injected faults:
```bash
cd evaluation/lm-evaluation-harness
pip install -e .
```
Then run tasks such as:
```bash
lm_eval --model hf --model_args pretrained=<your_model_path> --tasks lambada_openai
```

---

## 📌 Notes
- This repository is for **research purposes only**.  
- Fault injection attacks (e.g., Rowhammer) require privileged hardware access and may cause system instability. **Use with caution.**

---

## 📜 License
This project follows the same license terms as each included submodule:
- QuantFlip core: [MIT License](LICENSE)  
- Blacksmith: [GPL-3.0 License](tools/blacksmith/LICENSE)  
- lm-evaluation-harness: [MIT License](evaluation/lm-evaluation-harness/LICENSE)  
