#!/bin/bash
# TRM baseline on Sudoku-Extreme at the batch size EBRM can fit (matched-batch comparison).
# Run slurm/sudoku_memprobe.sbatch first, then set BATCH below to the largest size that fit.
#   sbatch slurm/sudoku_trm.sbatch
# Resumable: preemption/time-limit -> checkpoint -> requeue -> resume (see pretrain.py).
#SBATCH --job-name=sudoku_trm
#SBATCH --partition=kempner_requeue   # Kempner cannot be combined with other partitions; fallback: gpu_requeue (account ydu_lab)
#SBATCH --account=kempner_ydu_lab     # Kempner partitions require the lab's Kempner account
#SBATCH --exclude=holygpu8a05402                    # H200 node with a silently unusable GPU (2026-09-16)
#SBATCH --constraint="h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# ---- EDIT after the probe: the largest batch size EBRM fits on this GPU ----
BATCH=384
# ---------------------------------------------------------------------------
EPOCHS=50000
EVAL_INTERVAL=5000

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
PROJECT=ebrm-sudoku
RUN_NAME=pretrain_trm_sudoku_b${BATCH}
SELF=slurm/sudoku_trm.sbatch   # lets lib.sh resubmit this job (with --exclude) after a node fault
source slurm/lib.sh
activate_env             # self-sufficient: does not rely on module/conda from the submitting shell
gpu_preflight            # bad/busy GPU on this node -> exclude node, requeue
START_TS=$(date +%s)

echo "[sbatch] $(date) job=$SLURM_JOB_ID batch=$BATCH epochs=$EPOCHS run=$RUN_NAME restarts=${SLURM_RESTART_COUNT:-0} node=$(hostname)"

# USR1 = Slurm's warning 5 min before the time limit; TERM = preemption or scancel. Forward both to python
# so it checkpoints, but remember which: only the time limit needs an explicit requeue from us.
STOP_SIGNAL=""
trap 'STOP_SIGNAL=USR1; echo "[sbatch] time-limit warning, forwarding to python"; kill -USR1 "$PID" 2>/dev/null' USR1
trap 'STOP_SIGNAL=TERM; echo "[sbatch] SIGTERM (preemption or scancel), forwarding to python"; kill -USR1 "$PID" 2>/dev/null' TERM

python pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  global_batch_size=$BATCH \
  epochs=$EPOCHS eval_interval=$EVAL_INTERVAL \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
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
