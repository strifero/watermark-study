"""Answer quality for the temperature experiment. Independent judge Qwen3.5-9B rates each answer 1 to 10 for
correctness, helpfulness and fluency, given the instruction and the human Dolly answer as a reference. No key.
    python quality.py gen_e4bt05_wm gen_e4b_wm ...      (data file stems)
Output: data/quality.jsonl"""
import json, os, re, sys
from vllm import LLM, SamplingParams

D = "/home/sean/wm/data"
PROMPT = ("You are grading an AI assistant's answer.\n\nINSTRUCTION:\n{p}\n\nREFERENCE ANSWER (written by a person; the AI "
          "answer may differ and still be good):\n{h}\n\nAI ANSWER:\n{a}\n\nRate the AI answer from 1 (useless or wrong) to 10 "
          "(excellent) for correctness, helpfulness and fluency together. Output only the number.")


def main():
    prompts = {r["id"]: r for r in map(json.loads, open(f"{D}/prompts.jsonl"))}
    out = f"{D}/quality.jsonl"
    done = set()
    if os.path.exists(out):
        done = {(r["src"], r["id"]) for r in map(json.loads, open(out))}
    jobs = []
    for stem in sys.argv[1:]:
        for r in map(json.loads, open(f"{D}/{stem}.jsonl")):
            if (stem, r["id"]) not in done:
                jobs.append((stem, r))
    if not jobs:
        print("nothing to do"); return
    llm = LLM("/home/sean/models/Qwen3.5-9B", tensor_parallel_size=2, disable_custom_all_reduce=True, max_model_len=6144,
              gpu_memory_utilization=0.92, max_num_seqs=48, seed=5, limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    msgs = [[{"role": "user", "content": PROMPT.format(p=prompts[r["id"]]["prompt"], h=prompts[r["id"]]["human"], a=r["text"])}] for _, r in jobs]
    res = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    with open(out, "a") as f:
        for (stem, r), o in zip(jobs, res):
            m = re.search(r"\d+", o.outputs[0].text)
            f.write(json.dumps({"src": stem, "id": r["id"], "kind": r["kind"], "score": int(m.group()) if m else None}) + "\n")
    print("DONE", len(jobs))


if __name__ == "__main__":
    main()
