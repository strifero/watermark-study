#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=st-ai-1
#SBATCH --gres=gpu:a4500:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=56G
#SBATCH --time=8:00:00
#SBATCH --job-name=wm-review
#SBATCH --output=/home/sean/wm/logs/slurm-%x-%j.log
# Review round: second key, E4B in 4-bit, entropy, attacks on unmarked text, paraphrase fidelity.
set -euo pipefail
export PATH=/home/sean/gemma-card/.venv/bin:$PATH VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E4B=/home/sean/models/gemma-4-E4B-it; G31=/home/sean/models/gemma-4-31B-it
cd /home/sean/wm/gen
echo "== key2 table + E4B key2 $(date)"
[ -f /home/sean/wm/key2/sampling_table.pt ] || WM_KEYDIR=/home/sean/wm/key2 python -u dump_table.py
[ "$(wc -l < /home/sean/wm/data/gen_e4bk2_wm.jsonl 2>/dev/null || echo 0)" -ge 400 ] || WM_KEYDIR=/home/sean/wm/key2 BS=16 python -u generate.py $E4B e4bk2 wm
echo "== E4B 4-bit $(date)"
[ "$(wc -l < /home/sean/wm/data/gen_e4bq4_wm.jsonl 2>/dev/null || echo 0)" -ge 400 ] || BS=16 python -u generate.py $E4B e4bq4 wm --4bit
echo "== entropy $(date)"
[ -f /home/sean/wm/data/entropy_e4b.json ] || python -u entropy.py $E4B e4b
[ -f /home/sean/wm/data/entropy_e4bq4.json ] || python -u entropy.py $E4B e4bq4 --4bit
[ -f /home/sean/wm/data/entropy_g31b.json ] || python -u entropy.py $G31 g31b --4bit
cd /home/sean/wm/attack
echo "== attacks on unmarked text $(date)"
[ "$(wc -l < /home/sean/wm/data/atk_rewrite_e4b_plain.jsonl 2>/dev/null || echo 0)" -ge 2400 ] || python -u rewrite.py $E4B e4b plain
sleep 10
[ -s /home/sean/wm/data/atk_partial_e4b_plain.jsonl ] || python -u partial.py $E4B e4b plain
sleep 10
echo "== fidelity $(date)"
for i in $(seq 1 60); do grep -q DL_DONE /home/sean/wm/logs/dl-qwen.log && break; sleep 30; done
[ -s /home/sean/wm/data/fidelity.jsonl ] || python -u fidelity.py
echo "== DONE $(date)"
