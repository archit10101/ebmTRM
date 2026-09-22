#!/bin/bash
# Step 1 of the Sudoku experiment: find the largest batch size EBRM fits on this GPU, and what it costs.
# Build the dataset first (login node):
#   python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 \
#     --subsample-size 1000 --num-aug 1000
# Then:  sbatch slurm/sudoku_memprobe.sbatch
# Read logs/sudoku_memprobe_<jobid>.out: use the largest batch size that printed a line (not OOM),
# then put it in BATCH= in slurm/sudoku_trm.sbatch and slurm/sudoku_ebrm.sbatch.
#SBATCH --job-name=sudoku_memprobe
#SBATCH --partition=kempner_requeue   # Kempner cannot be combined with other partitions; fallback: gpu_requeue (account ydu_lab)
#SBATCH --account=kempner_ydu_lab     # Kempner partitions require the lab's Kempner account
#SBATCH --exclude=holygpu8a05402                    # H200 node with a silently unusable GPU (2026-09-16)
#SBATCH --constraint="h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%x_%j.out

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SELF=slurm/sudoku_memprobe.sbatch   # lets lib.sh resubmit this job (with --exclude) after a node fault
source slurm/lib.sh
activate_env             # self-sufficient: does not rely on module/conda from the submitting shell
gpu_preflight            # bad/busy GPU on this node -> exclude node, requeue

DATA=data/sudoku-extreme-1k-aug-1000
# Sudoku is 81 tokens vs Maze's 900, so attention is ~120x cheaper and much larger batches fit.
DISABLE_COMPILE=1 python scripts/ebrm_memprobe.py 128,256,384,512,768,1024,1536 \
  arch=ebrm data_paths="[$DATA]" epochs=50000 \
  arch.mlp_t=False arch.pos_encodings=rope arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 arch.H_steps=1 \
  arch.alpha_L=0.001 arch.alpha_H=0.01 arch.halt_max_steps=16

echo; echo "--- TRM at the same sizes, for comparison:"
python scripts/ebrm_memprobe.py 128,384,768,1536 arch=trm data_paths="[$DATA]" epochs=50000 \
  arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6
