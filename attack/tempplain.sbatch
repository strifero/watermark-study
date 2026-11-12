#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=st-ai-1
#SBATCH --gres=gpu:a4500:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=56G
#SBATCH --time=6:00:00
#SBATCH --job-name=wm-tempplain
#SBATCH --output=/home/sean/wm/logs/slurm-%x-%j.log
# Unwatermarked answers at the experiment temperatures: false-positive control for the temperature runs.
set -uo pipefail
export PATH=/home/sean/gemma-card/.venv/bin:$PATH TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E4B=/home/sean/models/gemma-4-E4B-it; G31=/home/sean/models/gemma-4-31B-it; D=/home/sean/wm/data
cd /home/sean/wm/gen
n() { wc -l < "$1" 2>/dev/null || echo 0; }
g() { [ "$(n $D/gen_$3_plain.jsonl)" -ge 400 ] || CUDA_VISIBLE_DEVICES=$1 WM_TEMP=$2 BS=8 python -u generate.py $E4B $3 plain; }
echo "== E4B plain $(date)"
( g 0 0.5 e4bt05; g 0 1.3 e4bt13 ) > /home/sean/wm/logs/tp-gpu0.log 2>&1 &
( g 1 0.7 e4bt07 ) > /home/sean/wm/logs/tp-gpu1.log 2>&1 &
wait
echo "== 31B plain T1.3 $(date)"
[ "$(n $D/gen_g31bt13_plain.jsonl)" -ge 400 ] || WM_TEMP=1.3 BS=16 python -u generate.py $G31 g31bt13 plain --4bit
echo "== DONE $(date)"
