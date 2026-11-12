"""Key side, second key. Run with WM_KEYDIR=/home/sean/wm/key2.
Scores E4B answers watermarked with key 2, and as checks, unmarked text and key-1 answers under key 2 (should be ~1%).
Thresholds: 1% of the unmarked E4B and human texts under key 2, cross-fitted by prompt-id parity as in analysis.py.
Output: data/key2_summary.json"""
import json, math, os, sys
import numpy as np, torch
assert os.environ.get("WM_KEYDIR", "").endswith("key2")
sys.path.insert(0, "/home/sean/wm/gen")
import scorer
D = "/home/sean/wm/data"
DEPTH = len(scorer.K["keys"]); W = torch.linspace(1.0, 0.1, DEPTH)


def z(text):
    ids = scorer.TOK["e4b"](text, add_special_tokens=False, return_tensors="pt").input_ids
    if ids.shape[1] <= scorer.K["ngram_len"]:
        return 0.0, 0.0
    with torch.no_grad():
        g = scorer.PROC.compute_g_values(ids)[0].float(); k = scorer.PROC.compute_context_repetition_mask(ids)[0].bool()
    g = g[k]; n = g.shape[0]
    if n == 0:
        return 0.0, 0.0
    return (float(g.mean()) - .5) / (.5 / math.sqrt(n * DEPTH)), float(((g - .5) * W).sum()) / (.5 * math.sqrt(n * float((W ** 2).sum())))


neg = [(r["id"], *z(r["text"])) for r in map(json.loads, open(f"{D}/gen_e4b_plain.jsonl"))]
neg += [(p["id"], *z(p["human"])) for p in map(json.loads, open(f"{D}/prompts.jsonl"))]
TH = {par: [float(np.quantile([x[j] for x in neg if x[0] % 2 != par], 0.99)) for j in (1, 2)] for par in (0, 1)}


def rate(rows):
    return {d: round(100 * float(np.mean([r[j] >= TH[r[0] % 2][j - 1] for r in rows])), 1) for d, j in (("mean", 1), ("weighted", 2))}


k2 = [(r["id"], *z(r["text"])) for r in map(json.loads, open(f"{D}/gen_e4bk2_wm.jsonl"))]
k1 = [(r["id"], *z(r["text"])) for r in map(json.loads, open(f"{D}/gen_e4b_wm.jsonl"))]
kinds = {r["id"]: r["kind"] for r in map(json.loads, open(f"{D}/prompts.jsonl"))}
out = {"key2_answers": rate(k2), "key2_open": rate([x for x in k2 if kinds[x[0]] == "open"]),
       "key2_constrained": rate([x for x in k2 if kinds[x[0]] == "constrained"]),
       "unmarked_under_key2": rate(neg), "key1_answers_under_key2": rate(k1)}
json.dump(out, open(f"{D}/key2_summary.json", "w"), indent=1)
print(out)
