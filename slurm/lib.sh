# Shared helpers for the sbatch scripts. Source after `cd "$SLURM_SUBMIT_DIR"`.
# Requeue on infrastructure faults (bad/busy GPU on the assigned node), excluding that node,
# so a broken node costs a queue wait instead of ending the run. Capped to avoid looping on a real bug.

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
