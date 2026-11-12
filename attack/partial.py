"""Blind partial paraphrase (attacker side: no key, no scorer). Rewrites a fraction f of the sentences of each
watermarked answer one at a time with an unwatermarked model and leaves everything else, including line breaks and
markdown, exactly as it was. Sentences under 5 words (headings, list stubs) are never chosen.
    python partial.py <attacker-model-path> <attacker-name>
Output: data/atk_partial_<attacker>.jsonl with attack "partial" and level f in {0.25, 0.5, 0.75}."""
import json, os, random, re, sys
from vllm import LLM, SamplingParams

D = "/home/sean/wm/data"
PROMPT = ("Rewrite this sentence with different wording and the same meaning. Keep any markdown symbols at the start. "
          "Output only the rewritten sentence.\n\n{t}")
SPLIT = re.compile(r"(?<=[.!?])(\s+)")


def units(text):
    """List of (piece, eligible). Joining all pieces gives back the text exactly."""
    out = []
    for line in text.split("\n"):
        parts = SPLIT.split(line)
        for i, p in enumerate(parts):
            out.append((p, i % 2 == 0 and len(p.split()) >= 5))
        out.append(("\n", False))
    out.pop()
    return out


def main():
    mp, name = sys.argv[1], sys.argv[2]
    src = sys.argv[3] if len(sys.argv) > 3 else "wm"
    out = f"{D}/atk_partial_{name}.jsonl" if src == "wm" else f"{D}/atk_partial_{name}_{src}.jsonl"
    rng = random.Random(7)
    jobs, plans = [], []
    for m in ("e4b", "g31b"):
        for r in map(json.loads, open(f"{D}/gen_{m}_{src}.jsonl")):
            u = units(r["text"])
            elig = [i for i, (_, e) in enumerate(u) if e]
            for f in (0.25, 0.5, 0.75):
                pick = set(rng.sample(elig, round(f * len(elig)))) if elig else set()
                plans.append((r, m, f, u, pick))
                for i in pick:
                    jobs.append(u[i][0])
    uniq = sorted(set(jobs))
    llm = LLM(mp, quantization="fp8", tensor_parallel_size=2, disable_custom_all_reduce=True, max_model_len=2048, gpu_memory_utilization=0.95,
              max_num_batched_tokens=4096, max_num_seqs=64, limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0}, seed=7)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=200, seed=7)
    res = llm.chat([[{"role": "user", "content": PROMPT.format(t=t)}] for t in uniq], sp,
                   chat_template_kwargs={"enable_thinking": False})
    rew = {t: o.outputs[0].text.strip().split("\n")[0].strip().strip('"') or t for t, o in zip(uniq, res)}
    with open(out, "w") as fo:
        for r, m, f, u, pick in plans:
            text = "".join(rew[p] if i in pick else p for i, (p, _) in enumerate(u))
            fo.write(json.dumps({"model": m, "id": r["id"], "kind": r["kind"], "attacker": name, "attack": "partial",
                                 "level": f, "text": text}) + "\n")
    print("DONE", len(plans), "texts,", len(uniq), "sentences")


if __name__ == "__main__":
    main()
