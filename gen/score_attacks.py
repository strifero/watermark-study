"""Key side. Score every attacked text with two detectors and report detection at a 1% false-positive rate.
  mean      the baseline test: mean g-value over all 30 depths, threshold from key/threshold.json
  weighted  weighted mean g-value, depth weights falling linearly from 1.0 to 0.1 (the weighted-mean detector of
            Dathathri et al., 2024), threshold recalibrated at 1% on the same unmarked texts as the baseline
Texts are scored with the tokenizer of the model that wrote the original answer.
    python score_attacks.py data/atk_edits.jsonl data/atk_rewrite_*.jsonl"""
import json, math, sys, glob
import numpy as np, torch
sys.path.insert(0, "/home/sean/wm/gen")
import scorer
D = "/home/sean/wm/data"
DEPTH = len(scorer.K["keys"])
W = torch.linspace(1.0, 0.1, DEPTH)


def both(text, model):
    ids = scorer.TOK[model](text, add_special_tokens=False, return_tensors="pt").input_ids
    if ids.shape[1] <= scorer.K["ngram_len"]:
        return 0.0, 0.0, 0
    with torch.no_grad():
        g = scorer.PROC.compute_g_values(ids)[0].float()
        keep = scorer.PROC.compute_context_repetition_mask(ids)[0].bool()
    g = g[keep]; n = int(g.shape[0])
    if n == 0:
        return 0.0, 0.0, 0
    zm = (float(g.mean()) - 0.5) / (0.5 / math.sqrt(n * DEPTH))
    zw = float(((g - 0.5) * W).sum()) / (0.5 * math.sqrt(n * float((W ** 2).sum())))
    return zm, zw, n


# thresholds
th_mean = json.load(open("/home/sean/wm/key/threshold.json"))["z"]
neg = []
prompts = [json.loads(l) for l in open(f"{D}/prompts.jsonl")]
for m in ("e4b", "g31b"):
    for r in map(json.loads, open(f"{D}/gen_{m}_plain.jsonl")):
        neg.append(both(r["text"], m)[1])
    for p in prompts:
        neg.append(both(p["human"], m)[1])
th_w = float(np.quantile(np.array(neg), 0.99))
json.dump({"z": th_w, "fpr_target": 0.01, "from": "score_attacks.py weighted"}, open("/home/sean/wm/key/threshold_weighted.json", "w"))

files = [f for a in sys.argv[1:] for f in sorted(glob.glob(a))]
rows = []
for f in files:
    for r in map(json.loads, open(f)):
        zm, zw, n = both(r["text"], r["model"])
        rows.append({k: r[k] for k in ("model", "id", "kind", "attacker", "attack", "level")} | {"zm": round(zm, 3), "zw": round(zw, 3), "n": n})
json.dump(rows, open(f"{D}/attack_scores.json", "w"))

clean = {(r["model"], r["id"]): r for r in rows if r["attack"] == "none"}
out = {"threshold_mean": round(th_mean, 3), "threshold_weighted": round(th_w, 3), "results": []}
groups = {}
for r in rows:
    groups.setdefault((r["model"], r["attacker"], r["attack"], r["level"]), []).append(r)
for (m, atk, a, lvl), g in sorted(groups.items(), key=lambda x: tuple(map(str, x[0]))):
    zm = np.array([r["zm"] for r in g]); zw = np.array([r["zw"] for r in g])
    caught = [r for r in g if clean[(r["model"], r["id"])]["zm"] >= th_mean]
    out["results"].append({"model": m, "attacker": atk, "attack": a, "level": lvl, "n": len(g),
                           "flag_mean": round(100 * float((zm >= th_mean).mean()), 1),
                           "flag_weighted": round(100 * float((zw >= th_w).mean()), 1),
                           "flag_mean_of_caught": round(100 * sum(r["zm"] >= th_mean for r in caught) / max(1, len(caught)), 1),
                           "flag_mean_open": round(100 * float(np.mean([r["zm"] >= th_mean for r in g if r["kind"] == "open"])), 1),
                           "median_tokens": int(np.median([r["n"] for r in g]))})
json.dump(out, open(f"{D}/attack_summary.json", "w"), indent=1)
for x in out["results"]:
    print(f'{x["model"]:5s} {x["attacker"]:5s} {x["attack"]:7s} {str(x["level"]):5s} n={x["n"]:3d} mean={x["flag_mean"]:5.1f} w={x["flag_weighted"]:5.1f} caught->{x["flag_mean_of_caught"]:5.1f} open={x["flag_mean_open"]:5.1f} tok={x["median_tokens"]}')
print("thresholds", out["threshold_mean"], out["threshold_weighted"])
