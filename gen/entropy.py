"""How much choice did each model have? Re-run each generator over its own answers (prompt + answer, teacher forced) and
measure, at every answer token, the entropy of the distribution it actually sampled from (temperature 1.0, top-k 64,
top-p 0.95) and the number of tokens left after top-k/top-p. No key involved.
    python entropy.py MODEL_DIR NAME [--4bit]     reads data/gen_<NAME>_{wm,plain}.jsonl (plain only if present)
Output: data/entropy_<NAME>.json  {mode: {id: {"H": mean nats, "H_full": mean nats, "k_eff": mean kept tokens, "T": tokens}}}"""
import json, os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
mdir, name = sys.argv[1:3]
q4 = "--4bit" in sys.argv
D = "/home/sean/wm/data"
tok = AutoTokenizer.from_pretrained(mdir)
kw = dict(device_map="auto", dtype=torch.bfloat16)
if q4:
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                   bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(mdir, **kw).eval()
dev = model.get_input_embeddings().weight.device
prompts = {r["id"]: r["prompt"] for r in map(json.loads, open(f"{D}/prompts.jsonl"))}


TEMP = float(os.environ.get("WM_TEMP", "1.0"))


def sampled_entropy(logits):
    logits = logits.float() / TEMP
    logp_full = torch.log_softmax(logits.float(), -1)
    H_full = -(logp_full.exp() * logp_full).sum(-1)
    v, _ = logits.float().topk(64, dim=-1)                       # top-k 64
    p = torch.softmax(v, -1)
    c = p.cumsum(-1)
    keep = (c - p) < 0.95                                        # top-p 0.95 (always keeps the first)
    p = p * keep
    p = p / p.sum(-1, keepdim=True)
    H = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
    return H, H_full, keep.sum(-1).float()


out = {}
for mode in ("wm", "plain"):
    path = f"{D}/gen_{name}_{mode}.jsonl"
    if not os.path.exists(path):
        continue
    res = {}
    for r in map(json.loads, open(path)):
        pre = tok.apply_chat_template([{"role": "user", "content": prompts[r["id"]]}], tokenize=False, add_generation_prompt=True)
        a = tok(pre, add_special_tokens=False, return_tensors="pt").input_ids
        b = tok(r["text"], add_special_tokens=False, return_tensors="pt").input_ids
        ids = torch.cat([a, b], 1).to(dev)
        with torch.no_grad():
            lg = model(ids).logits[0, a.shape[1] - 1:-1]
        H, Hf, k = sampled_entropy(lg)
        res[r["id"]] = {"H": round(float(H.mean()), 4), "H_full": round(float(Hf.mean()), 4), "k_eff": round(float(k.mean()), 2), "T": int(b.shape[1])}
    out[mode] = res
    print(name, mode, len(res), "mean H", sum(v["H"] for v in res.values()) / len(res), flush=True)
json.dump(out, open(f"{D}/entropy_{name}.json", "w"))
