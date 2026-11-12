"""Create OUR SynthID-Text key (30 depths, ngram 5). Only the generator and the scoring endpoint read this file."""
import json, secrets, os
k = {"ngram_len": 5, "keys": [secrets.randbelow(2**31) for _ in range(30)], "sampling_table_seed": secrets.randbelow(2**31),
     "sampling_table_size": 2**16, "context_history_size": 1024}
p = os.environ.get("WM_KEYDIR", "/home/sean/wm/key") + "/synthid.json"
assert not os.path.exists(p), "key exists"
json.dump(k, open(p, "w")); os.chmod(p, 0o600); print("key written")
