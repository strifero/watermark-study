#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a4500:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=16:00:00
#SBATCH --job-name=wm-gen
#SBATCH --output=/home/sean/wm/logs/slurm-%x-%j.log
set -euo pipefail
cd /home/sean/wm/gen
PY=/home/sean/gemma-card/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# E4B wm + plain done (400 each). 31B resumes where it stopped.
for mode in wm plain; do BS=16 $PY -u generate.py /home/sean/models/gemma-4-31B-it g31b $mode --4bit; done
echo DONE
