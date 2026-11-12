"""SynthID-Text scoring with OUR key (mean g-value test). Lives on the key side; attack code never imports this.
Score = mean g-value over the 30 key depths at every position whose 4-token context has not been seen before in
the text. Unmarked text averages 0.5. z is the one-sided z-score of that mean under 'no watermark'
(each g-value a fair coin flip)."""
import json, math, os, torch
KD = os.environ.get("WM_KEYDIR", "/home/sean/wm/key")
from transformers import AutoTokenizer
from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor

K = json.load(open(KD + "/synthid.json"))
PROC = SynthIDTextWatermarkLogitsProcessor(ngram_len=K["ngram_len"], keys=K["keys"], sampling_table_size=K["sampling_table_size"],
                                           sampling_table_seed=K["sampling_table_seed"], context_history_size=K["context_history_size"],
                                           device=torch.device("cpu"))
# generate() built the table on a GPU (CUDA RNG); use that exact table, not a CPU re-draw
PROC.sampling_table = torch.load(KD + "/sampling_table.pt")
TOK = {"g31b": AutoTokenizer.from_pretrained("/home/sean/models/gemma-4-31B-it"),
       "e4b": AutoTokenizer.from_pretrained("/home/sean/models/gemma-4-E4B-it")}
TOK["e4bq4"] = TOK["e4bk2"] = TOK["e4bt05"] = TOK["e4bt07"] = TOK["e4bt13"] = TOK["e4b"]
TOK["g31bt13"] = TOK["g31b"]


def score(text, model="g31b"):
    ids = TOK[model](text, add_special_tokens=False, return_tensors="pt").input_ids
    if ids.shape[1] <= K["ngram_len"]:
        return {"score": 0.5, "z": 0.0, "n": 0}
    with torch.no_grad():
        g = PROC.compute_g_values(ids)[0].float()                    # [positions, depth]
        keep = PROC.compute_context_repetition_mask(ids)[0].bool()   # [positions]
    g = g[keep]
    n = int(g.shape[0])
    if n == 0:
        return {"score": 0.5, "z": 0.0, "n": 0}
    m = float(g.mean())
    z = (m - 0.5) / (0.5 / math.sqrt(n * g.shape[1]))
    return {"score": round(m, 5), "z": round(z, 3), "n": n}
