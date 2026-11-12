# Washed out: SynthID-Text on Gemma 4, attacked blind

Code and per-text results for the trifero.ai lab report *Washed out*. We watermark Gemma 4 E4B and Gemma 4 31B with
SynthID-Text (transformers), keep the key, attack the output with code that never sees the key or the detector, and
measure detection with three detectors.

The watermark keys are **not** included. To reproduce, make your own key (step 1); every number should come out close,
not identical, because the key and the sampling are random.

## Layout

```
gen/      key side: key creation, sampling-table dump, generation, scoring, detectors, analysis
attack/   attacker side: never imports gen/ and never reads a key
data/     prompts, generated answers, attacked texts, per-text scores and summaries (no key material)
```

## Keeping the key sealed

* `gen/make_key.py` writes `key/synthid.json` with mode 0600; keep `key/` (and `key2/`) at mode 0700 on one machine.
* SynthID builds its sampling table with the CUDA random generator. A table rebuilt on the CPU from the same seed is
  different and makes watermarked text score as unmarked. Run `gen/dump_table.py` on a GPU once, right after making the
  key, and keep `key/sampling_table.pt` with the key. It is key material: do not publish it.
* Everything in `attack/` takes only text in and writes text out. The scorer (`gen/scorer.py`) is the only code that
  loads the key, and it runs after the attacks are written.

## Steps (Slurm, two 20 GB GPUs)

1. `python gen/make_key.py` then `python gen/dump_table.py` (GPU). Optional second key: `WM_KEYDIR=key2` for both.
2. `python gen/make_prompts.py` (400 Dolly 15k prompts: 200 open, 200 constrained).
3. Generation: `gen/gen.sbatch` (E4B bf16, 31B 4-bit NF4; temperature 1.0, top-k 64, top-p 0.95, 320 tokens), once
   with the watermark (`wm`) and once without (`plain`).
4. Baseline: `python gen/baseline.py`.
5. Attacks: `sbatch attack/attack.sbatch` (edits, rewrites by E4B and 31B via vLLM), `sbatch attack/round2.sbatch`
   (partial rewrites, Bayesian detector), `sbatch attack/review.sbatch` (second key, E4B 4-bit, entropy, attacks on
   unmarked text, paraphrase fidelity judged by Qwen3.5-9B).
6. Temperature experiment: `sbatch attack/temp.sbatch` (E4B at temperature 0.5, 0.7 and 1.3, 31B at 1.3, entropy for
   each via `gen/entropy.py` with `WM_TEMP`, then answer quality judged by Qwen3.5-9B with `attack/quality.py`).
7. Analysis: `python gen/analysis.py`, `python gen/analysis2.py` and `WM_KEYDIR=key2 python gen/score_key2.py`.

Calibration is two-fold cross-fitted: prompts are split by id parity, a 1% threshold is set on each half's unmarked
texts and used only to judge the other half, and the two held-out halves are pooled.

Software: transformers 5.17.0, vLLM 0.31.0, bitsandbytes, PyTorch with CUDA. Paths in the scripts point at
`/home/sean/...`; change them to your layout.

## Data files

* `prompts.jsonl`: prompt, kind (open or constrained), Dolly category, the human answer.
* `gen_<model>_<wm|plain>.jsonl`: generated answers (`e4bk2` = second key, `e4bq4` = E4B in 4-bit, `e4bt05` /
  `e4bt07` / `e4bt13` / `g31bt13` = temperature 0.5 / 0.7 / 1.3).
* `entropy_<model>.json`: per-answer sampling entropy; `quality.jsonl`: judge scores.
* `atk_*.jsonl`: attacked texts (`*_plain` = the same attacks on unmarked answers).
* `attack_scores*.json`, `review_summary.json`, `review2_summary.json`, `key2_summary.json`, `fidelity*.json*`: per-text scores and summaries.

Dolly 15k is CC BY-SA 3.0; generated and attacked texts derived from it are shared under the same terms.
