#!/bin/bash
# Peak-memory / step-time probe for EBRM at several batch sizes. Submit from the repo root:
#   sbatch slurm/ebrm_memprobe.sbatch
# Read logs/memprobe_<jobid>.out; pick the largest batch whose peak leaves ~10GB headroom on 80GB.
#SBATCH --job-name=memprobe
#SBATCH --partition=gpu_requeue
#SBATCH --constraint="h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%x_%j.out

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source slurm/lib.sh
activate_env             # self-sufficient: does not rely on module/conda from the submitting shell

DISABLE_COMPILE=1 python scripts/ebrm_memprobe.py 8,16,24,32,48,64,96 \
  arch=ebrm epochs=50000 \
  arch.mlp_t=False arch.pos_encodings=rope arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 arch.H_steps=1 \
  arch.alpha_L=0.01 arch.alpha_H=0.1 arch.halt_max_steps=16

echo; echo "--- for reference, TRM baseline at batch 128:"
python scripts/ebrm_memprobe.py 128 arch=trm epochs=50000 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4
