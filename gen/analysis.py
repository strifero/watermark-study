"""Key side. Review-round analysis for the write-up.
1. Cross-fitted calibration: prompts split by id parity; each text is judged by the threshold set on the OTHER half's
   unmarked texts (1% of them). False positives are measured the same way, so no text sets its own threshold.
   Wilson 95% intervals on every rate; threshold uncertainty by bootstrapping the calibration negatives.
2. False positives on unmarked text after the same attacks (edits, rewrites, partial rewrites of plain answers).
3. Length and entropy: detection by scored-token count, equal-length 200-token prefixes, per-token signal
   (mean g-value minus 0.5) against the sampling entropy measured by entropy.py.
4. Controls: E4B generated in 4-bit (e4bq4), E4B with a second key (e4bk2, scored by score_key2.py).
5. Survival of the original 5-token windows after each attack, and detection by survival.
6. Paraphrase fidelity from fidelity.jsonl.
Output: data/review_summary.json"""
import glob, json, math, os, sys
import numpy as np, torch
sys.path.insert(0, "/home/sean/wm/gen")
import scorer
D = "/home/sean/wm/data"
DEPTH = len(scorer.K["keys"]); W = torch.linspace(1.0, 0.1, DEPTH); NG = scorer.K["ngram_len"]
rng = np.random.default_rng(0)


def gstats(text, model, max_tokens=None):
    ids = scorer.TOK[model](text, add_special_tokens=False, return_tensors="pt").input_ids
    if max_tokens:
        ids = ids[:, :max_tokens]
    if ids.shape[1] <= NG:
        return 0.0, 0.0, 0, 0.0
    with torch.no_grad():
        g = scorer.PROC.compute_g_values(ids)[0].float()
        keep = scorer.PROC.compute_context_repetition_mask(ids)[0].bool()
    g = g[keep]; n = int(g.shape[0])
    if n == 0:
        return 0.0, 0.0, 0, 0.0
    zm = (float(g.mean()) - 0.5) / (0.5 / math.sqrt(n * DEPTH))
    zw = float(((g - 0.5) * W).sum()) / (0.5 * math.sqrt(n * float((W ** 2).sum())))
    return zm, zw, n, float(g.mean()) - 0.5


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; z = 1.96; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1))


def rate(flags):
    k, n = int(sum(flags)), len(flags)
    return {"pct": round(100 * k / max(1, n), 1), "ci": wilson(k, n), "k": k, "n": n}


prompts = [json.loads(l) for l in open(f"{D}/prompts.jsonl")]
kind = {p["id"]: p["kind"] for p in prompts}

# ---- negatives (unmarked): plain answers + human answers under each tokenizer
neg = []
for m in ("e4b", "g31b"):
    for r in map(json.loads, open(f"{D}/gen_{m}_plain.jsonl")):
        zm, zw, n, _ = gstats(r["text"], m); neg.append({"model": m, "src": "plain", "id": r["id"], "kind": r["kind"], "zm": zm, "zw": zw, "n": n})
    for p in prompts:
        zm, zw, n, _ = gstats(p["human"], m); neg.append({"model": m, "src": "human", "id": p["id"], "kind": p["kind"], "zm": zm, "zw": zw, "n": n})


def thresholds(rows):
    return {"zm": float(np.quantile([r["zm"] for r in rows], 0.99)), "zw": float(np.quantile([r["zw"] for r in rows], 0.99))}


TH = {0: thresholds([r for r in neg if r["id"] % 2 == 1]),   # threshold used to judge even-id texts
      1: thresholds([r for r in neg if r["id"] % 2 == 0])}   # threshold used to judge odd-id texts
th_all = thresholds(neg)


def flag(r, det):
    return r[det] >= TH[r["id"] % 2][det]


boot = {"zm": [], "zw": []}
cal = [r for r in neg if r["id"] % 2 == 0]
for _ in range(1000):
    s = [cal[i] for i in rng.integers(0, len(cal), len(cal))]
    t = thresholds(s); boot["zm"].append(t["zm"]); boot["zw"].append(t["zw"])
out = {"thresholds": {"crossfit_even_judged_by": TH[0], "crossfit_odd_judged_by": TH[1], "all_1600": th_all,
                      "bootstrap_95": {k: [round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)] for k, v in boot.items()}}}
fp = {}
for m in ("e4b", "g31b"):
    for src in ("plain", "human"):
        for kd in ("open", "constrained"):
            g = [r for r in neg if r["model"] == m and r["src"] == src and r["kind"] == kd]
            fp[f"{m}/{src}/{kd}"] = {"mean": rate([flag(r, "zm") for r in g]), "weighted": rate([flag(r, "zw") for r in g])}
fp["all"] = {"mean": rate([flag(r, "zm") for r in neg]), "weighted": rate([flag(r, "zw") for r in neg])}
out["false_positives_clean"] = fp
print("FP clean all", fp["all"], flush=True)


def score_file(path, keep_text=False):
    rows = []
    for r in map(json.loads, open(path)):
        zm, zw, n, ge = gstats(r["text"], r["model"])
        d = {k: r[k] for k in ("model", "id", "kind", "attacker", "attack", "level")} | {"zm": zm, "zw": zw, "n": n, "ge": ge}
        if keep_text:
            d["text"] = r["text"]
        rows.append(d)
    return rows


def table(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["attacker"], r["attack"], r["level"]), []).append(r)
    res = []
    for k, g in sorted(groups.items(), key=lambda x: tuple(map(str, x[0]))):
        res.append({"model": k[0], "attacker": k[1], "attack": k[2], "level": k[3],
                    "mean": rate([flag(r, "zm") for r in g]), "weighted": rate([flag(r, "zw") for r in g]),
                    "median_tokens": int(np.median([r["n"] for r in g]))})
    return res


# ---- 1. attacks on watermarked text, cross-fitted
wm_rows = []
for f in ("atk_edits.jsonl", "atk_rewrite_e4b.jsonl", "atk_rewrite_g31b.jsonl", "atk_partial_e4b.jsonl"):
    wm_rows += score_file(f"{D}/{f}", keep_text=True)
out["attacks_crossfit"] = table(wm_rows)
for x in out["attacks_crossfit"]:
    print("WM", x["model"], x["attacker"], x["attack"], x["level"], x["mean"]["pct"], x["weighted"]["pct"], x["weighted"]["ci"], flush=True)

# ---- 2. false positives after the same attacks on unmarked text
pl_rows = []
for f in ("atk_edits_plain.jsonl", "atk_rewrite_e4b_plain.jsonl", "atk_partial_e4b_plain.jsonl"):
    if os.path.exists(f"{D}/{f}"):
        pl_rows += score_file(f"{D}/{f}")
out["false_positives_attacked"] = table(pl_rows)
for x in out["false_positives_attacked"]:
    print("FP", x["model"], x["attacker"], x["attack"], x["level"], x["mean"]["pct"], x["weighted"]["pct"], flush=True)

# ---- 3. length and entropy
clean = {(r["model"], r["id"]): r for r in wm_rows if r["attack"] == "none"}
L = {}
for m in ("e4b", "g31b"):
    rs = [r for (mm, _), r in clean.items() if mm == m]
    L[m] = {"median_scored_tokens": int(np.median([r["n"] for r in rs])), "mean_g_excess": round(float(np.mean([r["ge"] for r in rs])), 4)}
    for lo, hi in ((0, 150), (150, 250), (250, 400)):
        g = [r for r in rs if lo <= r["n"] < hi]
        L[m][f"tokens_{lo}_{hi}"] = rate([flag(r, "zw") for r in g])
    src = {r["id"]: r["text"] for r in map(json.loads, open(f"{D}/gen_{m}_wm.jsonl"))}
    pre = []
    for i, t in src.items():
        zm, zw, n, ge = gstats(t, m, max_tokens=200)
        if scorer.TOK[m](t, add_special_tokens=False).input_ids.__len__() >= 200:
            pre.append({"id": i, "zm": zm, "zw": zw})
    L[m]["prefix200"] = {"weighted": rate([flag(r, "zw") for r in pre]), "mean": rate([flag(r, "zm") for r in pre])}
    ef = f"{D}/entropy_{m}.json"
    if os.path.exists(ef):
        E = json.load(open(ef))["wm"]
        H = np.array([E[str(r["id"])]["H"] for r in rs]); G = np.array([r["ge"] for r in rs])
        L[m]["sampling_entropy_mean_nats"] = round(float(H.mean()), 3)
        L[m]["kept_tokens_mean"] = round(float(np.mean([E[str(r["id"])]["k_eff"] for r in rs])), 2)
        L[m]["corr_entropy_gexcess"] = round(float(np.corrcoef(H, G)[0, 1]), 3)
        for kd in ("open", "constrained"):
            L[m][f"entropy_{kd}"] = round(float(np.mean([E[str(r["id"])]["H"] for r in rs if r["kind"] == kd])), 3)
out["length_entropy"] = L
# per-token signal vs entropy, pooled across models: bins of sampling entropy
pool = []
for m in ("e4b", "g31b", "e4bq4"):
    ef = f"{D}/entropy_{m}.json"; gf = f"{D}/gen_{m}_wm.jsonl"
    if os.path.exists(ef) and os.path.exists(gf):
        E = json.load(open(ef))["wm"]
        for r in map(json.loads, open(gf)):
            _, _, _, ge = gstats(r["text"], m); pool.append((m, E[str(r["id"])]["H"], ge))
if pool:
    Hs = np.array([p[1] for p in pool]); edges = np.quantile(Hs, [0, .2, .4, .6, .8, 1])
    out["signal_by_entropy"] = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [p for p in pool if a <= p[1] <= b]
        out["signal_by_entropy"].append({"H_lo": round(float(a), 3), "H_hi": round(float(b), 3), "n": len(sel),
                                         "mean_g_excess": round(float(np.mean([p[2] for p in sel])), 4),
                                         "share_31b": round(sum(p[0] == "g31b" for p in sel) / len(sel), 2)})
print("LEN", json.dumps(L), flush=True)

# ---- 4. E4B in 4-bit (key 1)
if os.path.exists(f"{D}/gen_e4bq4_wm.jsonl"):
    rs = []
    for r in map(json.loads, open(f"{D}/gen_e4bq4_wm.jsonl")):
        zm, zw, n, ge = gstats(r["text"], "e4bq4"); rs.append({"id": r["id"], "kind": r["kind"], "zm": zm, "zw": zw, "n": n, "ge": ge})
    out["e4b_4bit"] = {"weighted": rate([flag(r, "zw") for r in rs]), "mean": rate([flag(r, "zm") for r in rs]),
                       "weighted_open": rate([flag(r, "zw") for r in rs if r["kind"] == "open"]),
                       "weighted_constrained": rate([flag(r, "zw") for r in rs if r["kind"] == "constrained"]),
                       "median_scored_tokens": int(np.median([r["n"] for r in rs])), "mean_g_excess": round(float(np.mean([r["ge"] for r in rs])), 4)}
    if os.path.exists(f"{D}/entropy_e4bq4.json"):
        E = json.load(open(f"{D}/entropy_e4bq4.json"))["wm"]
        out["e4b_4bit"]["sampling_entropy_mean_nats"] = round(float(np.mean([E[str(r["id"])]["H"] for r in rs])), 3)
    print("E4B 4bit", out["e4b_4bit"], flush=True)
for kd in ("open", "constrained"):
    for m in ("e4b", "g31b"):
        out["length_entropy"][m][f"weighted_{kd}"] = rate([flag(r, "zw") for (mm, _), r in clean.items() if mm == m and r["kind"] == kd])

# ---- 5. n-gram survival
orig_ng = {}
for m in ("e4b", "g31b"):
    for r in map(json.loads, open(f"{D}/gen_{m}_wm.jsonl")):
        ids = scorer.TOK[m](r["text"], add_special_tokens=False).input_ids
        orig_ng[(m, r["id"])] = {tuple(ids[i:i + NG]) for i in range(len(ids) - NG + 1)}
surv_rows = []
for r in wm_rows:
    ids = scorer.TOK[r["model"]](r["text"], add_special_tokens=False).input_ids
    ng = [tuple(ids[i:i + NG]) for i in range(len(ids) - NG + 1)]
    o = orig_ng[(r["model"], r["id"])]
    r["surv"] = sum(x in o for x in ng) / max(1, len(ng))       # share of the attacked text's windows that are original
    surv_rows.append(r)
sv = {}
for r in surv_rows:
    sv.setdefault((r["model"], r["attacker"], r["attack"], r["level"]), []).append(r["surv"])
out["survival"] = [{"model": k[0], "attacker": k[1], "attack": k[2], "level": k[3], "median_surv": round(float(np.median(v)), 3)} for k, v in sorted(sv.items(), key=lambda x: tuple(map(str, x[0])))]
out["detection_by_survival"] = {}
for m in ("e4b", "g31b"):
    rs = [r for r in surv_rows if r["model"] == m and r["attack"] != "none"]
    out["detection_by_survival"][m] = []
    for lo, hi in ((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
        g = [r for r in rs if lo <= r["surv"] < hi]
        out["detection_by_survival"][m].append({"lo": lo, "hi": min(hi, 1.0), "weighted": rate([flag(r, "zw") for r in g])})
print("SURV", json.dumps(out["detection_by_survival"]), flush=True)

# ---- 6. fidelity
if os.path.exists(f"{D}/fidelity.jsonl"):
    F = [json.loads(l) for l in open(f"{D}/fidelity.jsonl")]
    det = {(r["model"], r["id"], r["attacker"], r["attack"], r["level"]): flag(r, "zw") for r in wm_rows}
    fid = {}
    for r in F:
        fid.setdefault((r["model"], r["attacker"], r["attack"], r["level"]), []).append(r)
    out["fidelity"] = []
    for k, g in sorted(fid.items(), key=lambda x: tuple(map(str, x[0]))):
        tot = sum(r["n"] for r in g)
        keep = [r["present"] / r["n"] for r in g if r["n"]]
        undetected_and_faithful = [(not det.get((r["model"], r["id"], r["attacker"], r["attack"], r["level"]), False)) and r["n"] and r["present"] / r["n"] >= 0.8 for r in g]
        out["fidelity"].append({"model": k[0], "attacker": k[1], "attack": k[2], "level": k[3], "n": len(g),
                                "present_pct": round(100 * sum(r["present"] for r in g) / tot, 1),
                                "changed_pct": round(100 * sum(r["changed"] for r in g) / tot, 1),
                                "missing_pct": round(100 * sum(r["missing"] for r in g) / tot, 1),
                                "unparsed_pct": round(100 * sum(r["unparsed"] for r in g) / tot, 1),
                                "median_answer_keep": round(float(np.median(keep)), 3),
                                "answers_keep_80": rate([x >= 0.8 for x in keep]),
                                "with_contradiction": rate([(r["contradictions"] or 0) > 0 for r in g]),
                                "undetected_and_keep_80": rate(undetected_and_faithful)})
        print("FID", out["fidelity"][-1], flush=True)
json.dump(out, open(f"{D}/review_summary.json", "w"), indent=1)
print("WROTE review_summary.json")
