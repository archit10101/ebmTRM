#!/bin/bash
# alpha_H sweep for EBRM on Maze-Hard, as a Slurm job array (one task per value).
#
#   sbatch slurm/ebrm_sweep_alphaH.sbatch
#   Tasks run one at a time (%1): with the TRM baseline that is the lab's 2-GPU cap.
#   When the TRM baseline finishes, allow two tasks at once:
#     scontrol update JobId=<ARRAY_JOBID> ArrayTaskThrottle=2
#   Kill a hopeless value early:  scancel <ARRAY_JOBID>_<TASK_INDEX>
#
# Batch 32: the retained second-order GD cycle needs ~200GB at batch 128 (see scripts/ebrm_memprobe.py).
# EPOCHS: 50000 is the full run (~3-5 days per value at batch 32); 10000 is a 1-day screen with
# 2 eval points that line up with the TRM baseline's first two. Must be a multiple of 5000.
#SBATCH --job-name=ebrm_full
#SBATCH --partition=kempner_requeue   # Kempner cannot be combined with other partitions; fallback: gpu_requeue (account ydu_lab)
#SBATCH --account=ydu_lab_kempner     # Kempner partitions require the lab's Kempner account
#SBATCH --exclude=holygpu8a05402                    # H200 node with a silently unusable GPU (2026-09-16)
#SBATCH --constraint="h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --array=0%2
#SBATCH --output=logs/%x_%A_%a.out

# ---- EDIT: the screen winner(s) to take to 50000 epochs (paired alpha_H / alpha_L).
# One value: --array=0 above. Two values: --array=0-1%2 and two entries in each list. ----
ALPHA_H_VALUES=(0.003)
ALPHA_L_VALUES=(0.0003)
BATCH=32
LR=1e-4             # must equal the screen's (the run name and checkpoint are shared)
PUZZLE_EMB_LR=1e-4
EPOCHS=50000        # full run
EVAL_INTERVAL=2000  # must equal the screen's, so the continuation lines up with its checkpoint
# -----------------------------------------------------------------------------------------

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

ALPHA_H=${ALPHA_H_VALUES[$SLURM_ARRAY_TASK_ID]}
ALPHA_L=${ALPHA_L_VALUES[$SLURM_ARRAY_TASK_ID]}
tag() { echo "$1" | sed 's/\./p/'; }   # 0.03 -> 0p03, matching the run-name convention
PROJECT=ebrm-maze
RUN_NAME="pretrain_ebrm_maze_zy_q_gd_al$(tag $ALPHA_L)_ah$(tag $ALPHA_H)_b${BATCH}_lr$(tag $LR)"

SELF=slurm/ebrm_full.sbatch   # lets lib.sh resubmit this job (with --exclude) after a node fault
source slurm/lib.sh
activate_env             # self-sufficient: does not rely on module/conda from the submitting shell
gpu_preflight            # bad/busy GPU on this node -> exclude node, requeue
START_TS=$(date +%s)

echo "[sbatch] $(date) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID alpha_H=$ALPHA_H alpha_L=$ALPHA_L batch=$BATCH epochs=$EPOCHS run=$RUN_NAME restarts=${SLURM_RESTART_COUNT:-0} node=$(hostname)"

# USR1 = Slurm's warning 5 min before the time limit; TERM = preemption or scancel. Forward both to python
# so it checkpoints, but remember which: only the time limit needs an explicit requeue from us.
STOP_SIGNAL=""
trap 'STOP_SIGNAL=USR1; echo "[sbatch] time-limit warning, forwarding to python"; kill -USR1 "$PID" 2>/dev/null' USR1
trap 'STOP_SIGNAL=TERM; echo "[sbatch] SIGTERM (preemption or scancel), forwarding to python"; kill -USR1 "$PID" 2>/dev/null' TERM

DISABLE_COMPILE=1 python pretrain.py \
  arch=ebrm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  global_batch_size=$BATCH \
  epochs=$EPOCHS eval_interval=$EVAL_INTERVAL \
  lr=$LR puzzle_emb_lr=$PUZZLE_EMB_LR \
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
infra_check_after_run "$EXIT" "$START_TS"   # early CUDA device error -> exclude node, requeue

if [ "$EXIT" -eq 3 ]; then
  if [ "$STOP_SIGNAL" = "USR1" ]; then
    echo "[sbatch] checkpointed before the time limit; requeueing"
    scontrol requeue "$SLURM_JOB_ID"
  else
    # Preempted jobs are requeued by Slurm itself (--requeue); an scancel means stop, so do not fight it.
    echo "[sbatch] checkpointed on SIGTERM; leaving requeue to Slurm"
  fi
fi
exit "$EXIT"
