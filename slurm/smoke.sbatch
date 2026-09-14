#!/bin/bash
# Phase C smoke test as a batch job. Submit from the repo root:  sbatch slurm/smoke.sbatch
# 1) short TRM run -> SIGUSR1 -> checkpoint -> resume to completion   (env + wandb + checkpointing)
# 2) short EBRM run at batch 400 with GPU memory polling              (does it fit?)
# Read the summary at the bottom of logs/smoke_<jobid>.out when it finishes (~15 min).
#SBATCH --job-name=smoke
#SBATCH --partition=gpu_requeue
#SBATCH --constraint="h100|h200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:45:00
#SBATCH --output=logs/%x_%j.out

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load python/3.10.13-fasrc01 gcc/12.2.0-fasrc01
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ebrm-trm
export CUDA_HOME="$HOME/.local/share/cuda-12.6.3"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[smoke] $(date) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import torch, numpy; from adam_atan2 import AdamATan2; print('[smoke] env ok: torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

rm -rf checkpoints/ebrm-maze/smoke_trm checkpoints/ebrm-maze/smoke_ebrm
SMOKE_TRM='arch=trm data_paths=[data/maze-30x30-hard-1k] evaluators=[] epochs=40 eval_interval=20 global_batch_size=128 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 ema=True checkpoint_interval_minutes=0.5 +project_name=ebrm-maze +run_name=smoke_trm'

echo; echo "===== [1/3] TRM: run, signal, checkpoint ====="
python pretrain.py $SMOKE_TRM > logs/smoke_trm.log 2>&1 &
PID=$!
for i in $(seq 1 90); do grep -q "Saved resume checkpoint" logs/smoke_trm.log && break; sleep 5; done
kill -USR1 $PID; wait $PID; TRM_KILL_EXIT=$?
grep -E "Saved resume|Received signal|Exiting|Traceback|Error" logs/smoke_trm.log | tail -4

echo; echo "===== [2/3] TRM: resume to completion ====="
python pretrain.py $SMOKE_TRM > logs/smoke_trm_resume.log 2>&1; TRM_RESUME_EXIT=$?
grep -E "Resuming|^TRAIN|EVALUATE|Traceback|Error" logs/smoke_trm_resume.log | head -6

echo; echo "===== [3/3] EBRM at batch 400 with memory polling ====="
DISABLE_COMPILE=1 python pretrain.py arch=ebrm data_paths="[data/maze-30x30-hard-1k]" evaluators="[]" \
  global_batch_size=400 epochs=20 eval_interval=10 \
  arch.H_cycles=3 arch.L_cycles=6 arch.H_steps=1 arch.alpha_L=0.01 arch.alpha_H=0.1 arch.halt_max_steps=16 \
  ema=True +project_name=ebrm-maze +run_name=smoke_ebrm > logs/smoke_ebrm.log 2>&1 &
PID=$!
# Poll GPU memory in the background (kill -0 would keep seeing the exited python as a zombie until wait reaps it).
PEAK_FILE=logs/smoke_peak_mem.txt; echo 0 > $PEAK_FILE
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 >> $PEAK_FILE; sleep 5; done ) &
POLL=$!
wait $PID; EBRM_EXIT=$?
kill $POLL 2>/dev/null; wait $POLL 2>/dev/null
PEAK=$(sort -n $PEAK_FILE | tail -1)
grep -E "Traceback|Error|OutOfMemory" logs/smoke_ebrm.log | head -3

echo; echo "================= SUMMARY ================="
echo "TRM kill exit code    : $TRM_KILL_EXIT   (want 3)"
echo "TRM resume exit code  : $TRM_RESUME_EXIT   (want 0)"
echo "TRM resumed from      : $(grep -o 'Resuming from step [0-9]*' logs/smoke_trm_resume.log | head -1)   (want 'Resuming from step N', N>0)"
echo "EBRM exit code        : $EBRM_EXIT   (want 0)"
echo "EBRM peak GPU memory  : ${PEAK} MiB of $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MiB"
if [ "$TRM_KILL_EXIT" -eq 3 ] && [ "$TRM_RESUME_EXIT" -eq 0 ] && [ "$EBRM_EXIT" -eq 0 ]; then
  echo "RESULT: ALL GOOD -> delete the smoke_* runs in wandb, then: sbatch slurm/trm_maze.sbatch; sbatch slurm/ebrm_maze.sbatch"
  rm -rf checkpoints/ebrm-maze/smoke_trm checkpoints/ebrm-maze/smoke_ebrm
else
  echo "RESULT: SOMETHING FAILED -> send logs/smoke_trm.log, logs/smoke_trm_resume.log, logs/smoke_ebrm.log"
fi
