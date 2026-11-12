#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=st-ai-1
#SBATCH --gres=gpu:a4500:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=56G
#SBATCH --time=6:00:00
#SBATCH --job-name=wm-round2
#SBATCH --output=/home/sean/wm/logs/slurm-%x-%j.log
set -euo pipefail
export PATH=/home/sean/gemma-card/.venv/bin:$PATH VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
cd /home/sean/wm/attack
echo "== partial paraphrase $(date)"
[ -s /home/sean/wm/data/atk_partial_e4b.jsonl ] || python -u partial.py /home/sean/models/gemma-4-E4B-it e4b
cd /home/sean/wm
echo "== full-set scores incl. partial $(date)"
python -u gen/score_attacks.py "data/atk_edits.jsonl" "data/atk_rewrite_*.jsonl" "data/atk_partial_*.jsonl" > logs/score_round2.txt
echo "== bayes $(date)"
python -u gen/bayes.py "data/atk_edits.jsonl" "data/atk_rewrite_*.jsonl" "data/atk_partial_*.jsonl"
echo "== DONE $(date)"
