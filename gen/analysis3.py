"""Key side. Third-review additions.
1. False positives at each experiment temperature: unwatermarked answers generated at that temperature, judged by the
   same cross-fitted thresholds (set on temperature-1.0 unmarked text) used for the watermarked temperature runs.
2. Paired quality differences against temperature 1.0 on the same prompts, 95% intervals, and an equivalence check
   against a margin of 0.5 points on the 10-point scale.
Output: data/review3_summary.json"""
import json, math, os, sys
import numpy as np, torch
sys.path.insert(0, "/home/sean/wm/gen")
import scorer
D = "/home/sean/wm/data"
DEPTH = len(scorer.K["keys"]); W = torch.linspace(1.0, 0.1, DEPTH)


def zw(text, model):
    ids = scorer.TOK[model](text, add_special_tokens=False, return_tensors="pt").input_ids
    if ids.shape[1] <= scorer.K["ngram_len"]:
        return 0.0
    with torch.no_grad():
        g = scorer.PROC.compute_g_values(ids)[0].float(); k = scorer.PROC.compute_context_repetition_mask(ids)[0].bool()
    g = g[k]
    if g.shape[0] == 0:
        return 0.0
    return float(((g - .5) * W).sum()) / (.5 * math.sqrt(g.shape[0] * float((W ** 2).sum())))


def wilson(k, n):
    p = k / n; z = 1.96; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1)]


# thresholds exactly as analysis.py: 1% of unmarked texts (plain at T1.0 + human, both tokenizers), cross-fitted by id parity
prompts = [json.loads(l) for l in open(f"{D}/prompts.jsonl")]
neg = []
for m in ("e4b", "g31b"):
    neg += [(r["id"], zw(r["text"], m)) for r in map(json.loads, open(f"{D}/gen_{m}_plain.jsonl"))]
    neg += [(p["id"], zw(p["human"], m)) for p in prompts]
TH = {par: float(np.quantile([z for i, z in neg if i % 2 != par], 0.99)) for par in (0, 1)}
out = {"thresholds": TH, "fp_by_temperature": [], "quality_paired": []}
for model, name, T in (("e4b", "e4bt05", 0.5), ("e4b", "e4bt07", 0.7), ("e4b", "e4b", 1.0), ("e4b", "e4bt13", 1.3),
                       ("g31b", "g31b", 1.0), ("g31b", "g31bt13", 1.3)):
    f = f"{D}/gen_{name}_plain.jsonl"
    if not os.path.exists(f):
        continue
    rs = [(r["id"], zw(r["text"], name)) for r in map(json.loads, open(f))]
    k = sum(z >= TH[i % 2] for i, z in rs)
    row = {"model": model, "temp": T, "n": len(rs), "k": int(k), "pct": round(100 * k / len(rs), 1), "ci": wilson(k, len(rs))}
    out["fp_by_temperature"].append(row); print("FP", row, flush=True)

Q = {}
for r in map(json.loads, open(f"{D}/quality.jsonl")):
    if r["score"] is not None:
        Q.setdefault(r["src"], {})[r["id"]] = r["score"]
for model, base, name, T in (("e4b", "gen_e4b_wm", "gen_e4bt05_wm", 0.5), ("e4b", "gen_e4b_wm", "gen_e4bt07_wm", 0.7),
                             ("e4b", "gen_e4b_wm", "gen_e4bt13_wm", 1.3), ("g31b", "gen_g31b_wm", "gen_g31bt13_wm", 1.3)):
    ids = sorted(set(Q[base]) & set(Q[name]))
    d = np.array([Q[name][i] - Q[base][i] for i in ids], dtype=float)
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    lo, hi = m - 1.96 * se, m + 1.96 * se
    row = {"model": model, "temp": T, "vs": 1.0, "n": len(ids), "mean_diff": round(m, 3), "ci": [round(lo, 3), round(hi, 3)],
           "within_0_5": bool(lo > -0.5 and hi < 0.5), "share_changed": round(float((d != 0).mean()), 3)}
    out["quality_paired"].append(row); print("Q", row, flush=True)
json.dump(out, open(f"{D}/review3_summary.json", "w"), indent=1)
print("WROTE review3_summary.json")
