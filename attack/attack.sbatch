#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=st-ai-1
#SBATCH --gres=gpu:a4500:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=56G
#SBATCH --time=10:00:00
#SBATCH --job-name=wm-attack
#SBATCH --output=/home/sean/wm/logs/slurm-%x-%j.log
set -euo pipefail
cd /home/sean/wm/attack
export PATH=/home/sean/gemma-card/.venv/bin:$PATH VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
[ -s /home/sean/wm/data/atk_edits.jsonl ] || python -u edits.py
[ $(wc -l < /home/sean/wm/data/atk_rewrite_e4b.jsonl 2>/dev/null || echo 0) -ge 2400 ] || python -u rewrite.py /home/sean/models/gemma-4-E4B-it e4b
sleep 15
echo "== rewrite g31b $(date)"; python -u rewrite.py /home/sean/models/gemma-4-31B-it g31b
echo "== score $(date)"; cd /home/sean/wm && python -u gen/score_attacks.py "data/atk_edits.jsonl" "data/atk_rewrite_*.jsonl"
echo "== DONE $(date)"
