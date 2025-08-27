# server.py
import argparse
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import torch
import os
import sys
sys.path.insert(0, os.path.abspath("./models/BitNet-b1.58-2B-4T"))
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--dst", required=True)
args, _ = parser.parse_known_args()
dst = args.dst
MODEL_DIR = f"./models/{dst}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "auto"  # 可改 torch.float16 / torch.bfloat16

app = FastAPI()

# === 加载模型/分词器（BitNet/自定义代码需要 trust_remote_code）===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, trust_remote_code=True)
# 若模型无 pad_token，使用 eos 兜底，避免后面 padding 报错
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=DTYPE, trust_remote_code=True)
model.to(DEVICE).eval()
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = tokenizer.pad_token_id
if getattr(model, "generation_config", None) is not None:
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if model.generation_config.eos_token_id is None and tokenizer.eos_token_id is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id

# ====== OpenAI /v1/completions 输入/输出 ======
class CompletionReq(BaseModel):
    model: str
    prompt: Any  # 支持 str / List[int] / List[List[int]]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    logprobs: Optional[int] = None   # 例如 5
    echo: Optional[bool] = False
    stop: Optional[List[str]] = None


def _normalize_prompts_to_token_ids(prompt: Any) -> Tuple[List[List[int]], List[str]]:
    """
    将 prompt 统一成 List[List[int]]，并返回可回显文本（若可恢复）。
    允许:
      - str
      - List[int]
      - List[List[int]]
    """
    if isinstance(prompt, str):
        enc = tokenizer(prompt, return_tensors="pt")
        return enc["input_ids"].tolist(), [prompt]
    if isinstance(prompt, list):
        if not prompt:
            return [], []
        if isinstance(prompt[0], int):        # List[int]
            return [prompt], [tokenizer.decode(prompt, skip_special_tokens=False)]
        if isinstance(prompt[0], list):       # List[List[int]]
            texts = [tokenizer.decode(x, skip_special_tokens=False) for x in prompt]
            return prompt, texts
    raise ValueError("Unsupported prompt type")


def _top_k_logprobs_row(logits_row: torch.Tensor, k: int) -> Dict[str, float]:
    """
    从单个时间步的 logits 中取 top-k，返回 {token_str: logprob}
    """
    k = max(int(k), 1)
    logp = torch.log_softmax(logits_row, dim=-1)
    topk = torch.topk(logp, k)
    ids = topk.indices.tolist()
    lps = topk.values.tolist()
    return {tokenizer.decode([tid]): float(lp) for tid, lp in zip(ids, lps)}


@torch.no_grad()
@app.post("/v1/completions")
def completions(req: CompletionReq):
    # 统一为 batch token ids
    token_id_batches, echo_texts = _normalize_prompts_to_token_ids(req.prompt)
    if len(token_id_batches) == 0:
        return {"id": "empty", "object": "text_completion", "model": req.model, "choices": []}

    # ========= 关键改动：对 batch 做 padding，并构造 attention_mask =========
    lengths = [len(x) for x in token_id_batches]
    max_len = max(lengths)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # 双保险（极少数模型既无 pad 也无 eos）
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # padded = [x + [pad_id] * (max_len - len(x)) for x in token_id_batches]
    # attn   = [[1] * len(x) + [0] * (max_len - len(x)) for x in token_id_batches]
    padded = [[pad_id] * (max_len - len(x)) + x for x in token_id_batches]
    attn   = [[0] * (max_len - len(x)) + [1] * len(x) for x in token_id_batches]

    input_ids      = torch.tensor(padded, dtype=torch.long, device=DEVICE)       # [B, T_max]
    attention_mask = torch.tensor(attn,   dtype=torch.long, device=DEVICE)       # [B, T_max]
    # =========================================================================

    # ===== 情况 A：lm-eval 的 loglikelihood 路径（echo=True & logprobs>0 & max_tokens∈{0,1}）=====
    if (req.echo is True) and (req.logprobs or 0) > 0 and (req.max_tokens in (0, 1)):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits  # [B, T_max, V]

        # teacher-forced 的下一个 token 似然：对齐到 T-1
        probs_all = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T_max-1, V]
        tgt_all   = input_ids[:, 1:]                              # [B, T_max-1]
        topn = int(req.logprobs)

        choices = []
        B, Tm1, V = probs_all.shape
        for b in range(B):
            L = lengths[b]
            if L <= 1:
                chosen_lp_b = []
                top_logprobs_b = []
            else:
                probs_b = probs_all[b, :L-1, :]                  # 只取有效部分
                tgt_b   = tgt_all[b,   :L-1]                     # [L-1]
                chosen_lp_b = probs_b.gather(1, tgt_b.unsqueeze(-1)).squeeze(-1).tolist()

                # 每个位置的 top-k（dict：{token_str: logprob}）
                top_logprobs_b = []
                for t in range(L-1):
                    top_logprobs_b.append(_top_k_logprobs_row(logits[b, t], topn))

            payload = {
                "tokens": [tokenizer.decode([tid]) for tid in input_ids[b, :L].tolist()],
                "token_logprobs": [None] + [float(x) for x in chosen_lp_b],  # 第一个 token 无条件概率
                "top_logprobs":   [None] + top_logprobs_b,
                "text_offset": None,
            }
            text = echo_texts[b]  # echo 返回原文本（或由 ids 反解）
            choices.append({
                "index": b,
                "text": text,
                "finish_reason": "stop",
                "logprobs": payload,
            })

        return {
            "id": "cmpl-loglikelihood-batch",
            "object": "text_completion",
            "model": req.model,
            "choices": choices,
        }

    # ===== 情况 B：正常生成（HellaSwag 等 loglikelihood 任务不会走这里）=====
    do_sample = bool(req.temperature and req.temperature > 0)
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=req.max_tokens,
        do_sample=do_sample,
        temperature=req.temperature if do_sample else None,
        top_p=req.top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    choices = []
    for b in range(gen.size(0)):
        gen_only = gen[b, lengths[b]:]  # 只取新增部分
        text = tokenizer.decode(gen_only, skip_special_tokens=True)
        choices.append({"index": b, "text": text, "finish_reason": "stop"})
    return {
        "id": "cmpl-generate-batch",
        "object": "text_completion",
        "model": req.model,
        "choices": choices,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)