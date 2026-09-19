#!/bin/bash
# Peak-memory / step-time probe for EBRM at several batch sizes. Submit from the repo root:
#   sbatch slurm/ebrm_memprobe.sbatch
# Read logs/memprobe_<jobid>.out; pick the largest batch whose peak leaves ~10GB headroom on 80GB.
#SBATCH --job-name=memprobe
#SBATCH --partition=kempner_requeue   # Kempner cannot be combined with other partitions; fallback: gpu_requeue (account ydu_lab)
#SBATCH --account=kempner_ydu_lab     # Kempner partitions require the lab's Kempner account
#SBATCH --exclude=holygpu8a05402                    # H200 node with a silently unusable GPU (2026-09-16)
#SBATCH --constraint="h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/%x_%j.out

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SELF=slurm/ebrm_memprobe.sbatch   # lets lib.sh resubmit this job (with --exclude) after a node fault
source slurm/lib.sh
activate_env             # self-sufficient: does not rely on module/conda from the submitting shell

DISABLE_COMPILE=1 python scripts/ebrm_memprobe.py 8,16,24,32,48,64,96 \
  arch=ebrm epochs=50000 \
  arch.mlp_t=False arch.pos_encodings=rope arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 arch.H_steps=1 \
  arch.alpha_L=0.01 arch.alpha_H=0.1 arch.halt_max_steps=16

echo; echo "--- for reference, TRM baseline at batch 128:"
python scripts/ebrm_memprobe.py 128 arch=trm epochs=50000 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4
