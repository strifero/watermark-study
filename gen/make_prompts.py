"""400 prompts from Dolly 15k (CC BY-SA, human-written answers kept as the human-text control).
200 open-ended (creative writing, brainstorming, general QA): many plausible next words, room for the mark.
200 constrained (closed QA, extraction, summarization of a given passage): few plausible words, little room."""
import json, random
from datasets import load_dataset
R = random.Random(7)
ds = list(load_dataset("databricks/databricks-dolly-15k", split="train"))
R.shuffle(ds)
op = [r for r in ds if r["category"] in ("creative_writing", "brainstorming", "general_qa") and 150 < len(r["response"]) < 2500][:200]
cl = [r for r in ds if r["category"] in ("closed_qa", "information_extraction", "summarization") and r["context"] and 150 < len(r["response"]) < 2500 and len(r["context"]) < 3000][:200]
with open("/home/sean/wm/data/prompts.jsonl", "w") as f:
    for i, r in enumerate(op + cl):
        q = r["instruction"] + ("\n\n" + r["context"] if r["context"] else "")
        f.write(json.dumps({"id": i, "kind": "open" if i < 200 else "constrained", "category": r["category"], "prompt": q, "human": r["response"]}) + "\n")
print(len(op), len(cl))
