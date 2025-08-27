# Local LLM Evaluation with `lm-eval-harness`

This repository provides a simple HTTP server that exposes a local HuggingFace model (e.g., BitNet, Llama, Falcon) through an **OpenAI-compatible API** (`/v1/completions`).
 It is designed to work seamlessly with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

------



## 📦 Installation

### Install and configure the server runtime library

```
# Install dependencies
pip install -r requirements.txt
```

Requirements mainly include:

- `transformers`
- `torch`
- `fastapi`
- `uvicorn`
- `pydantic`

### Install and configure the lm-eval test task library

To install the `lm-eval` package from the github repository, run:

```
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

We have provided the documentation for lm-eval in the repository. You can also download it from the official GitHub repository by following the steps provided in the installation.

---



## 🚀 Running the Server

The server loads a model from `./models/<model_name>` and exposes it at `http://127.0.0.1:8000/v1/completions`.

```
python server.py --dst <model_name>
```

Example:

```
python server.py --dst Falcon-E-1B-Base
```

- The model should be placed in: `./models/Falcon-E-1B-Base`
- Default device: **CUDA** if available, otherwise CPU
- Supported data type: `torch.float16` / `torch.bfloat16` / `torch.float32`

------



## 🧪 Testing with `lm-eval-harness`

Once the server is running, you can evaluate it with:

```
lm_eval \
  --model local-completions \
  --model_args base_url=http://127.0.0.1:8000/v1/completions,model=model_name,tokenizer=/home/path_to_model/model_name,api_key=token-abc123 \
  --tasks <task_list> \
  --batch_size <n> \
  --output_path test/<task_name>
```

### Example

```
lm_eval \
  --model local-completions \
  --model_args base_url=http://127.0.0.1:8000/v1/completions,model=Falcon-E-1B-Base,tokenizer=./models/Falcon-E-1B-Base,api_key=token-abc123 \
  --tasks hellaswag,piqa \
  --batch_size 4 \
  --output_path runs/hellaswag_piqa
```

------

## Models Download
You can test it by downloading the QuantFlip-powered model we provide from the repository [test-model](https://huggingface.co/QuantFlip/Falcon3-3B-Base-1.58bit).
------


## 📂 Project Structure

```
.
├── server.py                # FastAPI server exposing /v1/completions
├── requirements.txt         # file to install and configure the server runtime library
├── models/                  # Place downloaded HuggingFace models here
│   └── Falcon-E-1B-Base/  # Example model
└── test/                    # Output folder for lm-eval results
```

------



## 📑 Notes

- The API follows **OpenAI `/v1/completions` format**, so existing OpenAI-compatible clients can also use it.
- `lm-eval-harness` will call the server for loglikelihood (perplexity) and task evaluations.
- If your model has no `pad_token`, the server automatically sets `pad_token = eos_token`.
- Using our code to run AutoVPS and tamper with the model may yield a corrupted model that tests differently than ours.
- This occurs because AutoVPS's list comes from simulation, so some bits may not flip on real hardware.

- This simulation-to-hardware gap causes variation in the final corruption effect.


