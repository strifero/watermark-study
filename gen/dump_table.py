# The generator builds its sampling table with a CUDA RNG; a CPU RNG with the same seed gives a different table.
# Build it on the GPU exactly as generate() does, and save it next to the key for the CPU scorer.
import json, torch
from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor
K = json.load(open("" + __import__("os").environ.get("WM_KEYDIR", "/home/sean/wm/key") + "/synthid.json"))
p = SynthIDTextWatermarkLogitsProcessor(ngram_len=K["ngram_len"], keys=K["keys"], sampling_table_size=K["sampling_table_size"],
    sampling_table_seed=K["sampling_table_seed"], context_history_size=K["context_history_size"], device=torch.device("cuda"))
torch.save(p.sampling_table.cpu(), "" + __import__("os").environ.get("WM_KEYDIR", "/home/sean/wm/key") + "/sampling_table.pt")
print("saved", p.sampling_table.float().mean().item())
