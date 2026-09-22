#!/bin/bash
# EBRM on Sudoku-Extreme, matched to the TRM baseline's batch size.
# Run slurm/sudoku_memprobe.sbatch first, then set BATCH below to the largest size that fit.
#   sbatch slurm/sudoku_ebrm.sbatch
# Resumable: preemption/time-limit -> checkpoint -> requeue -> resume (see pretrain.py).
#SBATCH --job-name=sudoku_ebrm
#SBATCH --partition=kempner_requeue   # Kempner cannot be combined with other partitions; fallback: gpu_requeue (account ydu_lab)
#SBATCH --account=kempner_ydu_lab     # Kempner partitions require the lab's Kempner account
#SBATCH --exclude=holygpu8a05402                    # H200 node with a silently unusable GPU (2026-09-16)
#SBATCH --constraint="h200"   # batch 768 needs >80GB; both runs on the same GPU type for comparable timing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

# ---- Batch size (slurm/sudoku_memprobe.sbatch, 2026-09-22, H200 140GB) ----
# EBRM peak memory is ~0.107 GiB/sample: 512 -> 55 GiB, 768 -> 82.5, 1024 -> 110, 1536 -> OOM.
# 768 is the largest that leaves real headroom AND is TRM's own published Sudoku batch size, so the
# baseline is both matched to EBRM and in its tuned configuration. It needs an H200 (82.5 > H100's 80GB).
# Wall clock is nearly flat in batch size (EBRM ~21h, TRM ~5.4h at 50000 epochs), so a bigger batch buys
# nothing and costs weight updates. For an H100-only pool, use 512.
BATCH=768
# ---------------------------------------------------------------------------
ALPHA_H=0.01        # reference values; the Maze sweep found >=0.03 unstable there
ALPHA_L=0.001
EPOCHS=50000
EVAL_INTERVAL=5000

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
PROJECT=ebrm-maze   # one wandb project for the whole study (name predates the Sudoku runs)
RUN_NAME=pretrain_ebrm_sudoku_al$(echo $ALPHA_L | sed "s/\./p/")_ah$(echo $ALPHA_H | sed "s/\./p/")_b${BATCH}
SELF=slurm/sudoku_ebrm.sbatch   # lets lib.sh resubmit this job (with --exclude) after a node fault
source slurm/lib.sh
activate_env             # self-sufficient: does not rely on module/conda from the submitting shell
gpu_preflight            # bad/busy GPU on this node -> exclude node, requeue
START_TS=$(date +%s)

echo "[sbatch] $(date) job=$SLURM_JOB_ID batch=$BATCH alpha_H=$ALPHA_H alpha_L=$ALPHA_L epochs=$EPOCHS run=$RUN_NAME restarts=${SLURM_RESTART_COUNT:-0} node=$(hostname)"

# USR1 = Slurm's warning 5 min before the time limit; TERM = preemption or scancel. Forward both to python
# so it checkpoints, but remember which: only the time limit needs an explicit requeue from us.
STOP_SIGNAL=""
trap 'STOP_SIGNAL=USR1; echo "[sbatch] time-limit warning, forwarding to python"; kill -USR1 "$PID" 2>/dev/null' USR1
trap 'STOP_SIGNAL=TERM; echo "[sbatch] SIGTERM (preemption or scancel), forwarding to python"; kill -USR1 "$PID" 2>/dev/null' TERM

# DISABLE_COMPILE is required for arch=ebrm (autograd.grad inside forward).
DISABLE_COMPILE=1 python pretrain.py \
  arch=ebrm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  global_batch_size=$BATCH \
  epochs=$EPOCHS eval_interval=$EVAL_INTERVAL \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
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
