#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=st-ai-1
#SBATCH --gres=gpu:a4500:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=56G
#SBATCH --time=8:00:00
#SBATCH --job-name=wm-temp
#SBATCH --output=/home/sean/wm/logs/slurm-%x-%j.log
# Temperature experiment: same model, different sampling freedom. Key 1.
set -uo pipefail
export PATH=/home/sean/gemma-card/.venv/bin:$PATH VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E4B=/home/sean/models/gemma-4-E4B-it; G31=/home/sean/models/gemma-4-31B-it; D=/home/sean/wm/data
cd /home/sean/wm/gen
n() { wc -l < "$1" 2>/dev/null || echo 0; }
gen_e4b() { # $1 gpu $2 temp $3 name
  [ "$(n $D/gen_$3_wm.jsonl)" -ge 400 ] || CUDA_VISIBLE_DEVICES=$1 WM_TEMP=$2 BS=16 python -u generate.py $E4B $3 wm
  [ -f $D/entropy_$3.json ] || CUDA_VISIBLE_DEVICES=$1 WM_TEMP=$2 python -u entropy.py $E4B $3
}
echo "== E4B temps $(date)"
( gen_e4b 0 0.5 e4bt05; gen_e4b 0 1.3 e4bt13 ) > /home/sean/wm/logs/temp-gpu0.log 2>&1 &
( gen_e4b 1 0.7 e4bt07 ) > /home/sean/wm/logs/temp-gpu1.log 2>&1 &
wait
echo "== 31B T1.3 $(date)"
[ "$(n $D/gen_g31bt13_wm.jsonl)" -ge 400 ] || WM_TEMP=1.3 BS=16 python -u generate.py $G31 g31bt13 wm --4bit
[ -f $D/entropy_g31bt13.json ] || WM_TEMP=1.3 python -u entropy.py $G31 g31bt13 --4bit
echo "== quality $(date)"
cd /home/sean/wm/attack && python -u quality.py gen_e4b_wm gen_e4bt05_wm gen_e4bt07_wm gen_e4bt13_wm gen_g31b_wm gen_g31bt13_wm
echo "== DONE $(date)"
