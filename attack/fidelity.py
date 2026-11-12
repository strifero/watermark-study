"""Does the attacked text still say what the original said? Independent judge: Qwen3.5-9B (a different model family
from the Gemma generators and rewriters). No key, no detector.
Step 1: list up to 12 key statements in each original watermarked answer.
Step 2: for each attacked version, mark every statement PRESENT, CHANGED (stated differently in a way that alters the
        meaning or a detail) or MISSING, and count statements in the attacked text that contradict the original.
Attacks judged: paraphrase (small and large rewriter), paraphrase twice, French round trip, half the sentences rewritten,
plus two sanity checks: spacing flattened (should keep everything) and 25% of words deleted (should lose a lot).
Output: data/fidelity.jsonl"""
import json, re, sys
from vllm import LLM, SamplingParams

D = "/home/sean/wm/data"
JUDGE = "/home/sean/models/Qwen3.5-9B"
PICK = [("edit", "reflow", 0), ("edit", "delete", 0.25), ("e4b", "para", 0), ("g31b", "para", 0), ("e4b", "para2", 0),
        ("e4b", "backtr", 0), ("e4b", "partial", 0.5)]
EXTRACT = ("List the key pieces of information in the text below as short, self-contained numbered statements "
           "(1., 2., ...). Include facts, names, numbers, steps and recommendations. At most 12 statements. "
           "Do not add anything that is not in the text. Output only the list.\n\nTEXT:\n{t}")
CHECK = ("Below are statements taken from an ORIGINAL text, and a REWRITE of it. For each statement, decide whether the "
         "REWRITE still conveys it.\nPRESENT: conveyed with the same meaning (wording may differ).\nCHANGED: mentioned, but "
         "a detail, number or meaning is different.\nMISSING: not conveyed.\nAnswer with one line per statement in the form "
         "'<number>: PRESENT|CHANGED|MISSING', then a final line 'CONTRADICTIONS: <n>' giving how many claims in the REWRITE "
         "contradict the ORIGINAL.\n\nSTATEMENTS:\n{s}\n\nREWRITE:\n{r}")


def main():
    orig = {}
    for m in ("e4b", "g31b"):
        for r in map(json.loads, open(f"{D}/gen_{m}_wm.jsonl")):
            orig[(m, r["id"])] = r["text"]
    att = {}
    for f in ("atk_edits.jsonl", "atk_rewrite_e4b.jsonl", "atk_rewrite_g31b.jsonl", "atk_partial_e4b.jsonl"):
        for r in map(json.loads, open(f"{D}/{f}")):
            k = (r["attacker"], r["attack"], r["level"])
            if k in PICK:
                att[(r["model"], r["id"], k)] = r["text"]
    llm = LLM(JUDGE, tensor_parallel_size=2, max_model_len=6144, gpu_memory_utilization=0.92, max_num_seqs=48, seed=3,
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=700)
    kw = {"chat_template_kwargs": {"enable_thinking": False}}
    keys = sorted(orig)
    outs = llm.chat([[{"role": "user", "content": EXTRACT.format(t=orig[k])}] for k in keys], sp, **kw)
    stmts = {}
    for k, o in zip(keys, outs):
        lines = [re.sub(r"^\s*\d+[.)]\s*", "", l).strip() for l in o.outputs[0].text.splitlines() if re.match(r"^\s*\d+[.)]", l)]
        stmts[k] = lines[:12]
    jobs = [(m, i, a) for (m, i, a) in att if stmts.get((m, i))]
    prompts = []
    for m, i, a in jobs:
        s = "\n".join(f"{n + 1}. {x}" for n, x in enumerate(stmts[(m, i)]))
        prompts.append([{"role": "user", "content": CHECK.format(s=s, r=att[(m, i, a)])}])
    outs = llm.chat(prompts, sp, **kw)
    with open(f"{D}/fidelity.jsonl", "w") as fo:
        for (m, i, a), o in zip(jobs, outs):
            t = o.outputs[0].text
            lab = {int(x): y for x, y in re.findall(r"^\s*(\d+)\s*[:.)-]\s*(PRESENT|CHANGED|MISSING)", t, re.M)}
            c = re.search(r"CONTRADICTIONS:\s*(\d+)", t)
            n = len(stmts[(m, i)])
            got = [lab.get(j + 1) for j in range(n)]
            fo.write(json.dumps({"model": m, "id": i, "attacker": a[0], "attack": a[1], "level": a[2], "n": n,
                                 "present": got.count("PRESENT"), "changed": got.count("CHANGED"), "missing": got.count("MISSING"),
                                 "unparsed": got.count(None), "contradictions": int(c.group(1)) if c else None}) + "\n")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in stmts.items()}, open(f"{D}/fidelity_statements.json", "w"))
    print("DONE", len(jobs))


if __name__ == "__main__":
    main()
