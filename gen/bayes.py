"""Key side. Train SynthID's learned Bayesian detector (transformers BayesianDetectorModel, Dathathri et al. 2024)
on clean text only, then score every attack on held-out prompts.
Split by prompt: even prompt ids train, odd ids test, so no prompt is seen in both.
  positives: clean watermarked answers (both models)   negatives: unwatermarked answers + human Dolly answers
Threshold: 1% false positives on the held-out (odd id) unmarked texts. The mean and weighted detectors are
re-scored on the same held-out half with their own thresholds recalibrated the same way, so all three compare fairly.
    python bayes.py data/atk_*.jsonl"""
import glob, json, math, sys
import numpy as np, torch
sys.path.insert(0, "/home/sean/wm/gen")
import scorer
from transformers.generation.watermarking import BayesianDetectorConfig, BayesianDetectorModel

D = "/home/sean/wm/data"
DEPTH = len(scorer.K["keys"])
W = torch.linspace(1.0, 0.1, DEPTH)
MAXLEN = 1024
torch.manual_seed(0)


def gvals(text, model):
    ids = scorer.TOK[model](text, add_special_tokens=False, return_tensors="pt").input_ids[:, :MAXLEN + scorer.K["ngram_len"]]
    if ids.shape[1] <= scorer.K["ngram_len"]:
        return torch.zeros(1, DEPTH), torch.zeros(1)
    with torch.no_grad():
        g = scorer.PROC.compute_g_values(ids)[0].float()
        keep = scorer.PROC.compute_context_repetition_mask(ids)[0].float()
    return g, keep


def classic(g, keep):
    k = keep.bool(); gg = g[k]; n = gg.shape[0]
    if n == 0:
        return 0.0, 0.0
    zm = (float(gg.mean()) - 0.5) / (0.5 / math.sqrt(n * DEPTH))
    zw = float(((gg - 0.5) * W).sum()) / (0.5 * math.sqrt(n * float((W ** 2).sum())))
    return zm, zw


def pad(items):
    L = max(g.shape[0] for g, _ in items)
    G = torch.zeros(len(items), L, DEPTH); M = torch.zeros(len(items), L)
    for i, (g, k) in enumerate(items):
        G[i, :g.shape[0]] = g; M[i, :k.shape[0]] = k
    return G, M


prompts = [json.loads(l) for l in open(f"{D}/prompts.jsonl")]
clean = []  # (model, id, label, g, keep)
for m in ("e4b", "g31b"):
    for mode, lab in (("wm", 1), ("plain", 0)):
        for r in map(json.loads, open(f"{D}/gen_{m}_{mode}.jsonl")):
            clean.append((m, r["id"], lab, *gvals(r["text"], m)))
    for p in prompts:
        clean.append((m, p["id"], 0, *gvals(p["human"], m)))
train = [c for c in clean if c[1] % 2 == 0]
test_neg = [c for c in clean if c[1] % 2 == 1 and c[2] == 0]

cfg = BayesianDetectorConfig(watermarking_depth=DEPTH, base_rate=0.5)
det = BayesianDetectorModel(cfg)
opt = torch.optim.Adam(det.parameters(), lr=3e-3)
G, M = pad([(c[3], c[4]) for c in train]); Y = torch.tensor([float(c[2]) for c in train])
val = [c for c in clean if c[1] % 2 == 1]
Gv, Mv = pad([(c[3], c[4]) for c in val]); Yv = torch.tensor([float(c[2]) for c in val])
best, best_state = 1e9, None
for ep in range(400):
    perm = torch.randperm(len(Y))
    for i in range(0, len(Y), 64):
        b = perm[i:i + 64]
        loss = det(G[b], M[b], labels=Y[b], return_dict=True).loss
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % 20 == 19:
        with torch.no_grad():
            vl = float(det(Gv, Mv, labels=Yv, return_dict=True).loss)
        print(f"epoch {ep + 1} train {float(loss):.4f} val {vl:.4f}", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in det.state_dict().items()}
det.load_state_dict(best_state); det.eval()
torch.save(best_state, "/home/sean/wm/key/bayes_detector.pt")


def bscore(items):
    out = []
    for i in range(0, len(items), 64):
        g, k = pad(items[i:i + 64])
        with torch.no_grad():  # log-likelihood ratio, the posterior saturates at 0/1 on long texts
            lw = det.likelihood_model_watermarked(g).clamp_min(1e-30)
            llr = ((torch.log(lw) - math.log(0.5)) * k.unsqueeze(-1)).sum(dim=(1, 2))
        out += [float(x) for x in llr.reshape(-1)]
    return out


neg_b = bscore([(c[3], c[4]) for c in test_neg])
neg_c = [classic(c[3], c[4]) for c in test_neg]
th = {"bayes": float(np.quantile(neg_b, 0.99)), "mean": float(np.quantile([x[0] for x in neg_c], 0.99)),
      "weighted": float(np.quantile([x[1] for x in neg_c], 0.99))}
json.dump(th, open("/home/sean/wm/key/threshold_heldout.json", "w"))
print("held-out thresholds", th, "n_neg", len(test_neg))

rows = []
for f in sorted({f for a in sys.argv[1:] for f in glob.glob(a)}):
    batch = [r for r in map(json.loads, open(f)) if r["id"] % 2 == 1]
    items = [gvals(r["text"], r["model"]) for r in batch]
    bs = bscore(items)
    for r, it, b in zip(batch, items, bs):
        zm, zw = classic(*it)
        rows.append({k: r[k] for k in ("model", "id", "kind", "attacker", "attack", "level")} | {"zm": zm, "zw": zw, "pb": b})
json.dump(rows, open(f"{D}/attack_scores_heldout.json", "w"))
groups = {}
for r in rows:
    groups.setdefault((r["model"], r["attacker"], r["attack"], r["level"]), []).append(r)
res = []
for key, g in sorted(groups.items(), key=lambda x: tuple(map(str, x[0]))):
    d = {"model": key[0], "attacker": key[1], "attack": key[2], "level": key[3], "n": len(g),
         "mean": round(100 * np.mean([r["zm"] >= th["mean"] for r in g]), 1),
         "weighted": round(100 * np.mean([r["zw"] >= th["weighted"] for r in g]), 1),
         "bayes": round(100 * np.mean([r["pb"] >= th["bayes"] for r in g]), 1)}
    res.append(d)
    print(f'{d["model"]:5s} {d["attacker"]:5s} {d["attack"]:7s} {str(d["level"]):5s} n={d["n"]:3d} mean={d["mean"]:5.1f} w={d["weighted"]:5.1f} bayes={d["bayes"]:5.1f}')
json.dump({"thresholds": th, "results": res}, open(f"{D}/attack_summary_heldout.json", "w"), indent=1)
