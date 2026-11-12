"""Generate answers for every prompt, with or without OUR SynthID key. Same sampling either way (temperature 1.0,
top_k 64, top_p 0.95, the Gemma defaults), 320 new tokens max.
    python generate.py MODEL_DIR NAME {wm|plain} [--4bit]"""
import json, os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, SynthIDTextWatermarkingConfig
mdir, name, mode = sys.argv[1:4]
q4 = "--4bit" in sys.argv
tok = AutoTokenizer.from_pretrained(mdir, padding_side="left")
kw = dict(device_map="auto", dtype=torch.bfloat16)
if q4:
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(mdir, **kw).eval()
wcfg = None
if mode == "wm":
    k = json.load(open(os.environ.get("WM_KEYDIR", "/home/sean/wm/key") + "/synthid.json"))
    wcfg = SynthIDTextWatermarkingConfig(ngram_len=k["ngram_len"], keys=k["keys"], sampling_table_seed=k["sampling_table_seed"],
                                          sampling_table_size=k["sampling_table_size"], context_history_size=k["context_history_size"])
rows = [json.loads(l) for l in open("/home/sean/wm/data/prompts.jsonl")]
path = f"/home/sean/wm/data/gen_{name}_{mode}.jsonl"
import os
done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
rows = [r for r in rows if r["id"] not in done]          # resume after a crash
out = open(path, "a")
dev = model.get_input_embeddings().weight.device
torch.manual_seed(1234)
bs = int(__import__("os").environ.get("BS", "16"))
def run(b):
    texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True) for r in b]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    with torch.no_grad():
        g = model.generate(**enc, do_sample=True, temperature=1.0, top_k=64, top_p=0.95, max_new_tokens=320,
                           watermarking_config=wcfg, pad_token_id=tok.pad_token_id)
    return [tok.decode(seq, skip_special_tokens=True) for seq in g[:, enc.input_ids.shape[1]:]]


i = 0
while i < len(rows):
    b = rows[i:i + bs]
    try:
        texts = run(b)
    except torch.OutOfMemoryError:                       # long passages: halve the batch and retry
        torch.cuda.empty_cache()
        if bs == 1:
            raise
        bs = max(1, bs // 2); print("OOM, batch ->", bs, flush=True); continue
    for r, t in zip(b, texts):
        out.write(json.dumps({"id": r["id"], "kind": r["kind"], "text": t}) + "\n")
    out.flush(); i += len(b); print(f"{i}/{len(rows)} this run", flush=True)
