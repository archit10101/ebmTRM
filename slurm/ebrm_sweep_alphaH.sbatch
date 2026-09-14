#!/bin/bash
# alpha_H sweep for EBRM on Maze-Hard, as a Slurm job array (one task per value).
#
#   Submit AFTER the TRM baseline job so the lab's 2-GPU cap is respected:
#     sbatch --dependency=afterany:<TRM_JOBID> slurm/ebrm_sweep_alphaH.sbatch
#   Tasks run one at a time (%1). When the first EBRM run also finishes, allow two at once:
#     scontrol update JobId=<ARRAY_JOBID> ArrayTaskThrottle=2
#   Kill a hopeless value early:  scancel <ARRAY_JOBID>_<TASK_INDEX>
#
# Everything else (partition, checkpoint/resume, requeue) matches ebrm_maze.sbatch.
#SBATCH --job-name=ebrm_ah
#SBATCH --partition=gpu_requeue
#SBATCH --constraint="h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --array=0-2%1
#SBATCH --output=logs/%x_%A_%a.out

# ---- the sweep: edit this list; array range above must match its length (0-2 = 3 values) ----
ALPHA_H_VALUES=(0.03 0.3 1.0)
ALPHA_L=0.01
# -----------------------------------------------------------------------------------------

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

ALPHA_H=${ALPHA_H_VALUES[$SLURM_ARRAY_TASK_ID]}
tag() { echo "$1" | sed 's/\./p/'; }   # 0.03 -> 0p03, matching Adrian's run-name convention
PROJECT=ebrm-maze
RUN_NAME="pretrain_ebrm_maze_zy_q_gd_al$(tag $ALPHA_L)_ah$(tag $ALPHA_H)_b400"

module load python/3.10.13-fasrc01 gcc/12.2.0-fasrc01
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ebrm-trm
export CUDA_HOME="$HOME/.local/share/cuda-12.6.3"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[sbatch] $(date) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID alpha_H=$ALPHA_H alpha_L=$ALPHA_L run=$RUN_NAME restarts=${SLURM_RESTART_COUNT:-0} node=$(hostname)"

trap 'echo "[sbatch] signal received, forwarding to python"; kill -USR1 "$PID" 2>/dev/null' USR1 TERM

DISABLE_COMPILE=1 python pretrain.py \
  arch=ebrm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  global_batch_size=400 \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 \
  weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 arch.H_steps=1 \
  arch.alpha_L=$ALPHA_L arch.alpha_H=$ALPHA_H \
  arch.halt_max_steps=16 \
  seed=0 ema=True \
  +project_name=$PROJECT +run_name=$RUN_NAME &
PID=$!

wait "$PID"; EXIT=$?
if [ "$EXIT" -gt 128 ]; then
  wait "$PID"; EXIT=$?
fi
echo "[sbatch] $(date) python exited with code $EXIT"

if [ "$EXIT" -eq 3 ]; then
  echo "[sbatch] checkpointed on signal; requeueing"
  scontrol requeue "$SLURM_JOB_ID"
fi
exit "$EXIT"
