"""Blind text-edit attacks on the watermarked answers. Attacker side: never reads the key or the scorer.
Each attack is applied at several strengths to every watermarked answer from both models.
  delete  remove a fraction p of words
  swap    swap a fraction p of adjacent word pairs
  typo    one random character edit in a fraction p of words
  crop    keep only the first N words
  dilute  surround a 100-word watermarked excerpt with unrelated human Dolly text so it is a share s of the words
Output: data/atk_edits.jsonl"""
import json, random, sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "wm"   # wm, or plain for the false-positive check
D = "/home/sean/wm/data"
rng = random.Random(20261111)
human = [r["human"] for r in map(json.loads, open(f"{D}/prompts.jsonl"))]
human_words = " ".join(human).split()
ALPHA = "abcdefghijklmnopqrstuvwxyz"


def delete(w, p):
    return [x for x in w if rng.random() >= p]


def swap(w, p):
    w = list(w)
    for i in range(len(w) - 1):
        if rng.random() < p:
            w[i], w[i + 1] = w[i + 1], w[i]
    return w


def typo(w, p):
    out = []
    for x in w:
        if len(x) > 2 and rng.random() < p:
            i = rng.randrange(len(x)); op = rng.random()
            if op < 1 / 3:
                x = x[:i] + x[i + 1:]
            elif op < 2 / 3:
                x = x[:i] + rng.choice(ALPHA) + x[i:]
            else:
                x = x[:i] + rng.choice(ALPHA) + x[i + 1:]
        out.append(x)
    return out


def dilute(w, share):
    core = w[:100]
    need = int(round(len(core) * (1 - share) / share))
    s = rng.randrange(0, len(human_words) - need - 1)
    filler = human_words[s:s + need]
    cut = rng.randrange(0, len(filler) + 1)
    return filler[:cut] + core + filler[cut:]


rows = []
for m in ("e4b", "g31b"):
    for r in map(json.loads, open(f"{D}/gen_{m}_{SRC}.jsonl")):
        w = r["text"].split()
        base = {"model": m, "id": r["id"], "kind": r["kind"], "attacker": "edit"}
        rows.append({**base, "attack": "none", "level": 0, "text": r["text"]})
        rows.append({**base, "attack": "reflow", "level": 0, "text": " ".join(w)})  # whitespace only, the base for the word edits
        for p in (0.1, 0.25, 0.5):
            rows.append({**base, "attack": "delete", "level": p, "text": " ".join(delete(w, p))})
            rows.append({**base, "attack": "swap", "level": p, "text": " ".join(swap(w, p))})
            rows.append({**base, "attack": "typo", "level": p, "text": " ".join(typo(w, p))})
        for n in (25, 50, 100):
            rows.append({**base, "attack": "crop", "level": n, "text": " ".join(w[:n])})
        for s in (0.5, 0.25, 0.1):
            rows.append({**base, "attack": "dilute", "level": s, "text": " ".join(dilute(w, s))})
with open(f"{D}/atk_edits.jsonl" if SRC == "wm" else f"{D}/atk_edits_{SRC}.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows), "rows")
