"""Blind rewrite attacks with an unwatermarked LLM (attacker side: no key, no scorer).
    python rewrite.py <attacker-model-path> <attacker-name>
Stages for every watermarked answer from both Gemma models:
  para    paraphrase once
  para2   paraphrase the paraphrase
  backtr  translate to French, then back to English
Sampling: temperature 0.7, top_p 0.95, up to 700 tokens. Output: data/atk_rewrite_<attacker>.jsonl (resumable)."""
import json, os, sys
from vllm import LLM, SamplingParams
MP, NAME = sys.argv[1], sys.argv[2]
SRC = sys.argv[3] if len(sys.argv) > 3 else "wm"   # wm, or plain for the false-positive check
D = "/home/sean/wm/data"
OUT = f"{D}/atk_rewrite_{NAME}.jsonl" if SRC == "wm" else f"{D}/atk_rewrite_{NAME}_{SRC}.jsonl"
PARA = ("Rewrite the following text in your own words. Keep every fact, the structure and roughly the same length, "
        "but change the wording and sentence structure throughout. Output only the rewritten text.\n\n{t}")
TO_FR = "Translate the following text into French. Keep the formatting. Output only the translation.\n\n{t}"
TO_EN = "Translate the following text into English. Keep the formatting. Output only the translation.\n\n{t}"

def main():
    src = []
    for m in ("e4b", "g31b"):
        for r in map(json.loads, open(f"{D}/gen_{m}_{SRC}.jsonl")):
            src.append({"model": m, "id": r["id"], "kind": r["kind"], "text": r["text"]})
    done = set()
    if os.path.exists(OUT):
        for r in map(json.loads, open(OUT)):
            done.add((r["model"], r["id"], r["attack"]))

    llm = LLM(MP, quantization="fp8", tensor_parallel_size=2, disable_custom_all_reduce=True, max_model_len=4096, gpu_memory_utilization=0.95, max_num_batched_tokens=4096,
              max_num_seqs=32, limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0}, seed=11)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=700, seed=11)


    def run(tpl, texts):
        msgs = [[{"role": "user", "content": tpl.format(t=t)}] for t in texts]
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
        return [o.outputs[0].text.strip() for o in outs]


    def write(items, attack, texts):
        with open(OUT, "a") as f:
            for s, t in zip(items, texts):
                f.write(json.dumps({"model": s["model"], "id": s["id"], "kind": s["kind"], "attacker": NAME,
                                    "attack": attack, "level": 0, "text": t}) + "\n")


    todo = [s for s in src if (s["model"], s["id"], "para") not in done]
    p1 = run(PARA, [s["text"] for s in todo]); write(todo, "para", p1)
    para = {(r["model"], r["id"]): r["text"] for r in map(json.loads, open(OUT)) if r["attack"] == "para"}
    todo = [s for s in src if (s["model"], s["id"], "para2") not in done]
    p2 = run(PARA, [para[(s["model"], s["id"])] for s in todo]); write(todo, "para2", p2)
    todo = [s for s in src if (s["model"], s["id"], "backtr") not in done]
    fr = run(TO_FR, [s["text"] for s in todo])
    en = run(TO_EN, fr); write(todo, "backtr", en)
    print("DONE", NAME)


if __name__ == "__main__":
    main()
