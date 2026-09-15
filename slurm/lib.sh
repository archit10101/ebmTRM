# Shared helpers for the sbatch scripts. Source after `cd "$SLURM_SUBMIT_DIR"`.
#  - activate_env: set up the environment without relying on anything inherited from the submitting shell.
#  - gpu_preflight / infra_check_after_run: requeue on infrastructure faults (bad/busy GPU on the assigned
#    node), excluding that node, so a broken node costs a queue wait instead of ending the run. Capped.

EBRM_CONDA_ENV_BIN=${EBRM_CONDA_ENV_BIN:-$HOME/.conda/envs/ebrm-trm/bin}

activate_env() {
  # `module` is a shell function; batch jobs only have it if the submitting shell exported it. Init Lmod if not.
  if ! command -v module > /dev/null 2>&1; then
    local f; for f in /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
      [ -f "$f" ] && { source "$f"; break; }
    done
  fi
  if command -v module > /dev/null 2>&1; then
    module load python/3.10.13-fasrc01 gcc/12.2.0-fasrc01 || echo "[sbatch] warning: module load failed (system gcc 8.5 will be used)"
  else
    echo "[sbatch] warning: 'module' is unavailable in this shell; using the conda env's bin directly"
  fi
  if command -v conda > /dev/null 2>&1; then
    { source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ebrm-trm; } 2>/dev/null || true
  fi
  # Independent of module/conda: the env's own interpreter first on PATH, then the user-space CUDA toolkit.
  export PATH="$EBRM_CONDA_ENV_BIN:$PATH"
  export CUDA_HOME="$HOME/.local/share/cuda-12.6.3"
  export PATH="$CUDA_HOME/bin:$PATH"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  if ! python -c "import torch, adam_atan2" > /dev/null 2>&1; then
    echo "[sbatch] environment check failed: '$(command -v python || echo python-not-found)' cannot import torch/adam_atan2. Not a node fault; not requeueing."
    exit 1
  fi
  echo "[sbatch] python: $(command -v python)"
}

INFRA_MAX_REQUEUES=5
INFRA_PATTERN='CUDA-capable device|busy or unavailable|no CUDA GPUs are available|CUDA driver|cudaErrorDevicesUnavailable|NCCL error|ECC error|Xid'

_job_log() { scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *StdOut=//p'; }

_infra_requeues_so_far() {
  local log; log=$(_job_log)
  if [ -n "$log" ] && [ -f "$log" ]; then
    grep -c "\[sbatch\] infra fault" "$log" || true   # grep -c prints 0 but exits 1 on no match
  else
    echo 0
  fi
}

# requeue_excluding_node <reason>
requeue_excluding_node() {
  local n; n=$(_infra_requeues_so_far)
  if [ "$n" -ge "$INFRA_MAX_REQUEUES" ]; then
    echo "[sbatch] infra fault ($1) but already requeued $n times; giving up. Check the node list and resubmit."
    exit 1
  fi
  local node; node=$(hostname -s)
  local excl; excl=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/.*ExcNodeList=\([^ ]*\).*/\1/p')
  if [ -n "$excl" ] && [ "$excl" != "(null)" ]; then excl="$excl,$node"; else excl="$node"; fi
  echo "[sbatch] infra fault ($1) on $node; excluding it and requeueing (attempt $((n + 1))/$INFRA_MAX_REQUEUES)"
  scontrol update JobId="$SLURM_JOB_ID" ExcNodeList="$excl" 2>/dev/null || true
  sleep 30
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}

# Run before launching python: is the GPU usable at all?
gpu_preflight() {
  if ! nvidia-smi -L > /dev/null 2>&1; then requeue_excluding_node "nvidia-smi failed"; fi
  if ! timeout 120 python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda'); torch.cuda.synchronize()" > /dev/null 2>&1; then
    requeue_excluding_node "torch cannot use the GPU"
  fi
}

# Run after python exits: requeue if it died of a device fault (not a bug, not an OOM) within the first minutes.
# infra_check_after_run <exit_code> <start_epoch_seconds>
infra_check_after_run() {
  local exit_code=$1 started=$2 log; log=$(_job_log)
  [ "$exit_code" -eq 0 ] || [ "$exit_code" -eq 3 ] && return 0
  [ $(( $(date +%s) - started )) -gt 600 ] && return 0   # ran for a while: treat as a real failure
  [ -n "$log" ] && [ -f "$log" ] && tail -n 200 "$log" | grep -qE "$INFRA_PATTERN" && ! tail -n 200 "$log" | grep -q "out of memory" \
    && requeue_excluding_node "python exit $exit_code with a device error"
  return 0
}
