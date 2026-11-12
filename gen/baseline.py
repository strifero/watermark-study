"""Baseline detection: score clean watermarked text, unwatermarked text from the same model, and the human Dolly
answers. Threshold = the z that flags 1% of all unmarked text (plain + human, both models). Report detection rate
at that threshold, split by model and prompt kind (open vs constrained)."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/sean/wm/gen")
from scorer import score
D = "/home/sean/wm/data"
prompts = {r["id"]: r for r in map(json.loads, open(f"{D}/prompts.jsonl"))}
rows = []
for m in ("e4b", "g31b"):
    for mode in ("wm", "plain"):
        for r in map(json.loads, open(f"{D}/gen_{m}_{mode}.jsonl")):
            rows.append({"model": m, "src": mode, "kind": r["kind"], "id": r["id"], **score(r["text"], m)})
    for p in prompts.values():
        rows.append({"model": m, "src": "human", "kind": p["kind"], "id": p["id"], **score(p["human"], m)})
json.dump(rows, open(f"{D}/baseline_scores.json", "w"))
neg = np.array([r["z"] for r in rows if r["src"] != "wm"])
th = float(np.quantile(neg, 0.99))
json.dump({"z": th, "fpr_target": 0.01, "from": "baseline.py"}, open("/home/sean/wm/key/threshold.json", "w"))
out = {"threshold_z": round(th, 3)}
for m in ("e4b", "g31b"):
    for kind in ("open", "constrained", "all"):
        for src in ("wm", "plain", "human"):
            z = np.array([r["z"] for r in rows if r["model"] == m and r["src"] == src and (kind == "all" or r["kind"] == kind)])
            n = [r["n"] for r in rows if r["model"] == m and r["src"] == src and (kind == "all" or r["kind"] == kind)]
            out[f"{m}/{kind}/{src}"] = {"flagged": round(float((z >= th).mean()) * 100, 1), "median_z": round(float(np.median(z)), 2),
                                        "median_scored_tokens": int(np.median(n))}
json.dump(out, open(f"{D}/baseline_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
