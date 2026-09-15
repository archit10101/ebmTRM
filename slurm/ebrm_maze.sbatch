#!/bin/bash
# EBRM on Maze-Hard. Submit from the repo root:  sbatch slurm/ebrm_maze.sbatch
# Preemptible: the job checkpoints on SIGTERM/SIGUSR1, exits with code 3, and is requeued;
# on restart pretrain.py finds checkpoints/<project>/<run>/latest.pt and resumes.
#SBATCH --job-name=ebrm_maze
#SBATCH --partition=gpu_requeue
#SBATCH --constraint="h100|h200"   # 80GB cards. Batch 128 needs ~200GB (OOMs on a 140GB H200); 32 is ~50GB.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source slurm/lib.sh
activate_env             # self-sufficient: does not rely on module/conda from the submitting shell
gpu_preflight            # bad/busy GPU on this node -> exclude node, requeue
START_TS=$(date +%s)

PROJECT=ebrm-maze
RUN_NAME=pretrain_ebrm_maze_zy_q_gd_al0p01_ah0p1_b32

echo "[sbatch] $(date) job=$SLURM_JOB_ID restarts=${SLURM_RESTART_COUNT:-0} node=$(hostname)"

# USR1 = Slurm's warning 5 min before the time limit; TERM = preemption or scancel. Forward both to python
# so it checkpoints, but remember which: only the time limit needs an explicit requeue from us.
STOP_SIGNAL=""
trap 'STOP_SIGNAL=USR1; echo "[sbatch] time-limit warning, forwarding to python"; kill -USR1 "$PID" 2>/dev/null' USR1
trap 'STOP_SIGNAL=TERM; echo "[sbatch] SIGTERM (preemption or scancel), forwarding to python"; kill -USR1 "$PID" 2>/dev/null' TERM

# DISABLE_COMPILE is required for arch=ebrm (autograd.grad inside forward).
# Batch 32. The math attention backend needed for second-order gradients keeps ~4 B*8*916*916 fp32
# matrices per layer per retained GD step alive; batch 128 needs ~200GB. See scripts/ebrm_memprobe.py.
DISABLE_COMPILE=1 python pretrain.py \
  arch=ebrm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  global_batch_size=32 \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 \
  weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=False arch.pos_encodings=rope \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 arch.H_steps=1 \
  arch.alpha_L=0.01 arch.alpha_H=0.1 \
  arch.halt_max_steps=16 \
  seed=0 ema=True \
  +project_name=$PROJECT +run_name=$RUN_NAME &
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
