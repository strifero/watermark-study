"""Key side. Second-review additions (all from existing data plus the temperature runs when present).
1. Stricter evasion metric: not verified, 80%+ statements kept AND zero contradictions.
2. Entropy groups as proper quintiles (no answer counted twice).
3. 200-token prefixes: the selection rule (only answers that reach 200 tokens), the same answers scored in full,
   and a matched comparison on prompts where both models reached 200 tokens.
4. Temperature experiment: per temperature, sampling entropy, verified rate (cross-fitted thresholds), quality.
Output: data/review2_summary.json"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, "/home/sean/wm/gen")
import analysis as A   # re-runs the cross-fit setup on import; reuses its helpers and thresholds
D = A.D
out = {}

# 1. strict evasion
F = [json.loads(l) for l in open(f"{D}/fidelity.jsonl")]
wm = {}
for f in ("atk_edits.jsonl", "atk_rewrite_e4b.jsonl", "atk_rewrite_g31b.jsonl", "atk_partial_e4b.jsonl"):
    for r in map(json.loads, open(f"{D}/{f}")):
        wm[(r["model"], r["id"], r["attacker"], r["attack"], float(r["level"]))] = r["text"]
det = {}
for k, t in wm.items():
    if any((k[0], k[1], k[2], k[3], float(x["level"])) == k for x in F[:0]):
        pass
need = {(r["model"], r["id"], r["attacker"], r["attack"], float(r["level"])) for r in F}
for k in need:
    zm, zw, n, ge = A.gstats(wm[k], k[0])
    det[k] = zw >= A.TH[k[1] % 2]["zw"]
g = {}
for r in F:
    g.setdefault((r["model"], r["attacker"], r["attack"], r["level"]), []).append(r)
out["strict_evasion"] = []
for key, rs in sorted(g.items(), key=lambda x: tuple(map(str, x[0]))):
    flags = []
    for r in rs:
        k = (r["model"], r["id"], r["attacker"], r["attack"], float(r["level"]))
        flags.append((not det[k]) and r["n"] > 0 and r["present"] / r["n"] >= 0.8 and (r["contradictions"] or 0) == 0)
    out["strict_evasion"].append({"model": key[0], "attacker": key[1], "attack": key[2], "level": key[3], **A.rate(flags)})
    print("STRICT", key, out["strict_evasion"][-1]["pct"], out["strict_evasion"][-1]["ci"])

# 2. entropy quintiles without double counting
pool = []
for m in ("e4b", "g31b", "e4bq4"):
    E = json.load(open(f"{D}/entropy_{m}.json"))["wm"]
    for r in map(json.loads, open(f"{D}/gen_{m}_wm.jsonl")):
        _, _, _, ge = A.gstats(r["text"], m); pool.append((m, E[str(r["id"])]["H"], ge))
pool.sort(key=lambda p: p[1])
qs = np.array_split(np.arange(len(pool)), 5)
out["signal_by_entropy"] = []
for idx in qs:
    sel = [pool[i] for i in idx]
    out["signal_by_entropy"].append({"H_lo": round(sel[0][1], 3), "H_hi": round(sel[-1][1], 3), "n": len(sel),
                                     "share_31b": round(sum(p[0] == "g31b" for p in sel) / len(sel), 2),
                                     "mean_g_excess": round(float(np.mean([p[2] for p in sel])), 4),
                                     "e4b_g_excess": round(float(np.mean([p[2] for p in sel if p[0] != "g31b"])), 4) if any(p[0] != "g31b" for p in sel) else None,
                                     "g31b_g_excess": round(float(np.mean([p[2] for p in sel if p[0] == "g31b"])), 4) if any(p[0] == "g31b" for p in sel) else None})
    print("ENT", out["signal_by_entropy"][-1])

# 3. prefixes
pre = {}
for m in ("e4b", "g31b"):
    for r in map(json.loads, open(f"{D}/gen_{m}_wm.jsonl")):
        T = len(A.scorer.TOK[m](r["text"], add_special_tokens=False).input_ids)
        zf = A.gstats(r["text"], m)[1]; zp = A.gstats(r["text"], m, max_tokens=200)[1]
        pre[(m, r["id"])] = {"T": T, "full": zf >= A.TH[r["id"] % 2]["zw"], "p200": zp >= A.TH[r["id"] % 2]["zw"], "kind": r["kind"]}
P = {}
for m in ("e4b", "g31b"):
    long_ = [v for (mm, _), v in pre.items() if mm == m and v["T"] >= 200]
    short = [v for (mm, _), v in pre.items() if mm == m and v["T"] < 200]
    P[m] = {"n_reach_200": len(long_), "open_share_of_those": round(sum(v["kind"] == "open" for v in long_) / len(long_), 2),
            "full_answer_on_those": A.rate([v["full"] for v in long_]), "prefix200_on_those": A.rate([v["p200"] for v in long_]),
            "full_answer_on_short": A.rate([v["full"] for v in short])}
both = [i for i in range(400) if pre[("e4b", i)]["T"] >= 200 and pre[("g31b", i)]["T"] >= 200]
for m in ("e4b", "g31b"):
    P[m]["matched_prompts_prefix200"] = A.rate([pre[(m, i)]["p200"] for i in both])
P["matched_n"] = len(both)
out["prefix"] = P
print("PREFIX", json.dumps(P))

# 4. temperature experiment
Q = {}
if os.path.exists(f"{D}/quality.jsonl"):
    for r in map(json.loads, open(f"{D}/quality.jsonl")):
        Q.setdefault(r["src"], []).append(r["score"])
TEMPS = [("e4b", "e4bt05", 0.5), ("e4b", "e4bt07", 0.7), ("e4b", "e4b", 1.0), ("e4b", "e4bt13", 1.3), ("g31b", "g31b", 1.0), ("g31b", "g31bt13", 1.3)]
out["temperature"] = []
for model, name, T in TEMPS:
    gf, ef = f"{D}/gen_{name}_wm.jsonl", f"{D}/entropy_{name}.json"
    if not (os.path.exists(gf) and os.path.exists(ef)):
        continue
    E = json.load(open(ef))["wm"]
    rs = []
    for r in map(json.loads, open(gf)):
        zm, zw, n, ge = A.gstats(r["text"], name)
        rs.append({"id": r["id"], "kind": r["kind"], "flag": zw >= A.TH[r["id"] % 2]["zw"], "n": n, "ge": ge, "H": E[str(r["id"])]["H"]})
    q = [s for s in Q.get(f"gen_{name}_wm", []) if s is not None]
    row = {"model": model, "name": name, "temp": T, "entropy": round(float(np.mean([r["H"] for r in rs])), 3),
           "verified": A.rate([r["flag"] for r in rs]), "verified_open": A.rate([r["flag"] for r in rs if r["kind"] == "open"]),
           "verified_constrained": A.rate([r["flag"] for r in rs if r["kind"] == "constrained"]),
           "g_excess": round(float(np.mean([r["ge"] for r in rs])), 4), "median_tokens": int(np.median([r["n"] for r in rs])),
           "quality_mean": round(float(np.mean(q)), 2) if q else None, "quality_n": len(q),
           "quality_ci": [round(float(np.mean(q) - 1.96 * np.std(q, ddof=1) / math.sqrt(len(q))), 2),
                          round(float(np.mean(q) + 1.96 * np.std(q, ddof=1) / math.sqrt(len(q))), 2)] if len(q) > 1 else None}
    out["temperature"].append(row)
    print("TEMP", row)
json.dump(out, open(f"{D}/review2_summary.json", "w"), indent=1)
print("WROTE review2_summary.json")
