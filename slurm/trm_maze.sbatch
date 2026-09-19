#!/bin/bash
# TRM baseline on Maze-Hard (README command). Submit from the repo root:  sbatch slurm/trm_maze.sbatch
# Preemptible: the job checkpoints on SIGTERM/SIGUSR1, exits with code 3, and is requeued;
# on restart pretrain.py finds checkpoints/<project>/<run>/latest.pt and resumes.
#SBATCH --job-name=trm_maze
#SBATCH --partition=kempner_requeue   # Kempner cannot be combined with other partitions; fallback: gpu_requeue
#SBATCH --exclude=holygpu8a05402                    # H200 node with a silently unusable GPU (2026-09-16)
#SBATCH --constraint="h100|h200|a100|a40"   # full GPUs only (no MIG slices, no Blackwell RTX 6000)
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

SELF=slurm/trm_maze.sbatch   # lets lib.sh resubmit this job (with --exclude) after a node fault
source slurm/lib.sh
activate_env             # self-sufficient: does not rely on module/conda from the submitting shell
gpu_preflight            # bad/busy GPU on this node -> exclude node, requeue
START_TS=$(date +%s)

PROJECT=ebrm-maze
RUN_NAME=pretrain_att_maze30x30_1gpu

echo "[sbatch] $(date) job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0} node=$(hostname)"

# USR1 = Slurm's warning 5 min before the time limit; TERM = preemption or scancel. Forward both to python
# so it checkpoints, but remember which: only the time limit needs an explicit requeue from us.
STOP_SIGNAL=""
trap 'STOP_SIGNAL=USR1; echo "[sbatch] time-limit warning, forwarding to python"; kill -USR1 "$PID" 2>/dev/null' USR1
trap 'STOP_SIGNAL=TERM; echo "[sbatch] SIGTERM (preemption or scancel), forwarding to python"; kill -USR1 "$PID" 2>/dev/null' TERM

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
