#!/bin/bash
# TRM baseline on Maze-Hard (Adrian's command). Submit from the repo root:  sbatch slurm/trm_maze.sbatch
# Preemptible: the job checkpoints on SIGTERM/SIGUSR1, exits with code 3, and is requeued;
# on restart pretrain.py finds checkpoints/<project>/<run>/latest.pt and resumes.
#SBATCH --job-name=trm_maze
#SBATCH --partition=kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load python/3.10.13-fasrc01 gcc/12.2.0-fasrc01   # gcc>=9 needed by torch.compile / CUDA extensions
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ebrm-trm
export CUDA_HOME="$HOME/.local/share/cuda-12.6.3"
export PATH="$CUDA_HOME/bin:$PATH"

PROJECT=ebrm-maze
RUN_NAME=pretrain_att_maze30x30_1gpu

echo "[sbatch] $(date) job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0} node=$(hostname)"

# Slurm sends USR1 (5 min before the time limit) or TERM (preemption) to this shell; forward to python.
trap 'echo "[sbatch] signal received, forwarding to python"; kill -USR1 "$PID" 2>/dev/null' USR1 TERM

python pretrain.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +project_name=$PROJECT +run_name=$RUN_NAME ema=True &
PID=$!

wait "$PID"; EXIT=$?
if [ "$EXIT" -gt 128 ]; then
  # wait was interrupted by the trapped signal; wait again for python to finish checkpointing
  wait "$PID"; EXIT=$?
fi
echo "[sbatch] $(date) python exited with code $EXIT"

if [ "$EXIT" -eq 3 ]; then
  echo "[sbatch] checkpointed on signal; requeueing"
  scontrol requeue "$SLURM_JOB_ID"
fi
exit "$EXIT"
