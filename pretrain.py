from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import sys
import math
import time
import signal
import yaml
import shutil
import copy

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Resumable checkpoint written periodically and on SIGTERM/SIGUSR1 (Slurm preemption / time limit).
RESUME_CHECKPOINT_NAME = "latest.pt"
STOP_EXIT_CODE = 3  # Exit code after a signal-triggered checkpoint, so the sbatch script can requeue.
_STOP_REQUESTED = False


def _request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Received signal {signum}: will checkpoint after the current step and exit.", flush=True)


def _ignore_stop_signals_in_worker(_worker_id):
    # Slurm signals every process in the job. The main process handles the checkpoint;
    # data loader workers must survive until it exits.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

    # Resumable training (preemptible partitions)
    resume: bool = True # resume from <checkpoint_path>/latest.pt if it exists
    checkpoint_interval_minutes: float = 20.0 # how often to write latest.pt during training

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        worker_init_fn=_ignore_stop_signals_in_worker,
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device(DEVICE):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def resume_checkpoint_file(config: PretrainConfig) -> Optional[str]:
    if config.checkpoint_path is None:
        return None
    return os.path.join(config.checkpoint_path, RESUME_CHECKPOINT_NAME)


def save_resume_checkpoint(config: PretrainConfig, train_state: TrainState, ema_helper: Optional[EMAHelper],
                           iter_id: int, batches_done: int, phase: str, wandb_run_id: Optional[str], rank: int):
    """Save everything needed to continue training from exactly this point.

    phase is "train" (batches_done batches of iteration iter_id are complete) or
    "eval" (iteration iter_id finished training; its evaluation has not run yet).
    The ACT carry is deliberately not saved: in-flight puzzles simply restart.
    """
    path = resume_checkpoint_file(config)
    if rank != 0 or path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    state = {
        "model": train_state.model.state_dict(),
        "ema": ema_helper.state_dict() if ema_helper is not None else None,
        "optimizers": [optim.state_dict() for optim in train_state.optimizers],
        "step": train_state.step,
        "iter_id": iter_id,
        "batches_done": batches_done,
        "phase": phase,
        "wandb_run_id": wandb_run_id,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
    }
    # Write atomically so a kill mid-save can never leave a corrupt latest.pt
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)
    print(f"Saved resume checkpoint at step {train_state.step} (iter {iter_id}, {batches_done} batches, phase {phase})", flush=True)


def load_resume_checkpoint(config: PretrainConfig) -> Optional[dict]:
    path = resume_checkpoint_file(config)
    if not config.resume or path is None or not os.path.isfile(path):
        return None

    print(f"Found resume checkpoint {path}")
    # Our own file: contains RNG states (numpy), so full unpickling is required.
    # Load on CPU; load_state_dict moves tensors onto the live parameters.
    return torch.load(path, map_location="cpu", weights_only=False)


def apply_resume_checkpoint(state: dict, train_state: TrainState, ema_helper: Optional[EMAHelper], config: PretrainConfig, rank: int):
    train_state.model.load_state_dict(state["model"])
    if ema_helper is not None:
        assert state["ema"] is not None, "Checkpoint has no EMA state but ema=True"
        ema_helper.load_state_dict({k: v.to(DEVICE) for k, v in state["ema"].items()})
    assert len(state["optimizers"]) == len(train_state.optimizers), "Optimizer count changed since checkpoint"
    for optim, optim_state in zip(train_state.optimizers, state["optimizers"]):
        optim.load_state_dict(optim_state)
        # Optimizer.load_state_dict leaves the "step" entry on whatever device it was saved from
        # (CPU here) unless the optimizer is flagged fused/capturable. AdamATan2's fused CUDA
        # kernel needs every state tensor, step included, on the parameter's device.
        for param, param_state in optim.state.items():
            for k, v in param_state.items():
                if torch.is_tensor(v) and v.device != param.device:
                    param_state[k] = v.to(param.device)
    train_state.step = state["step"]

    if rank == 0:
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state(state["rng"]["cuda"])
        np.random.set_state(state["rng"]["numpy"])
    else:
        # Only rank 0's RNG is checkpointed; keep other ranks decorrelated.
        torch.random.manual_seed(config.seed + rank + train_state.step)

    print(f"Resuming from step {train_state.step} (iter {state['iter_id']}, {state['batches_done']} batches, phase {state['phase']})", flush=True)


_WANDB_LOG_FAILURES = 0


def wandb_log(*args, **kwargs):
    """wandb.log that cannot kill training.

    Slurm preemption signals every process in the job, including wandb's background
    service, so a log call right after the signal raises ConnectionResetError.
    Losing a logged point is acceptable; losing the run is not.
    """
    global _WANDB_LOG_FAILURES
    try:
        wandb.log(*args, **kwargs)
    except Exception as e:
        _WANDB_LOG_FAILURES += 1
        if _WANDB_LOG_FAILURES <= 3 or _WANDB_LOG_FAILURES % 1000 == 0:
            print(f"WARNING: wandb.log failed ({type(e).__name__}: {e}); {_WANDB_LOG_FAILURES} failures so far, continuing.", flush=True)


def wandb_finish():
    try:
        wandb.finish()
    except Exception as e:
        print(f"WARNING: wandb.finish failed ({type(e).__name__}: {e}); continuing.", flush=True)


def stop_requested(world_size: int) -> bool:
    # All ranks must agree, otherwise a rank that keeps training deadlocks on the next all-reduce.
    if world_size <= 1:
        return _STOP_REQUESTED
    flag = torch.tensor(int(_STOP_REQUESTED), device=DEVICE)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location=DEVICE)

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        model.load_state_dict(state_dict, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.to(DEVICE) for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device(DEVICE):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.device(DEVICE):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device=DEVICE
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Resume state (loaded before the data loader so it can start at the right iteration)
    resume_state = load_resume_checkpoint(config)
    start_iter_id = resume_state["iter_id"] if resume_state is not None else 0
    start_batches_done = resume_state["batches_done"] if resume_state is not None else 0
    start_phase = resume_state["phase"] if resume_state is not None else "train"

    # Checkpoint on Slurm preemption (SIGTERM) and pre-time-limit warning (SIGUSR1)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGUSR1, _request_stop)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, resume_iteration=start_iter_id, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # EMA (registered before resume so the checkpointed shadow can be restored into it)
    ema_helper = None
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    if resume_state is not None:
        apply_resume_checkpoint(resume_state, train_state, ema_helper, config, rank=RANK)

    # Progress bar and logger
    progress_bar = None
    wandb_run_id = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, initial=train_state.step)
        wandb_run_id = resume_state["wandb_run_id"] if resume_state is not None else None
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True),
                   id=wandb_run_id, resume="allow" if wandb_run_id is not None else None)  # type: ignore
        wandb_run_id = wandb.run.id
        if resume_state is None:
            wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
    del resume_state

    def checkpoint_and_exit(iter_id: int, batches_done: int, phase: str):
        save_resume_checkpoint(config, train_state, ema_helper, iter_id, batches_done, phase, wandb_run_id, rank=RANK)
        if RANK == 0:
            wandb_finish()
        if dist.is_initialized():
            dist.destroy_process_group()
        print(f"Exiting with code {STOP_EXIT_CODE} for requeue.", flush=True)
        sys.exit(STOP_EXIT_CODE)

    # Training Loop
    last_checkpoint_time = time.time()
    for _iter_id in range(start_iter_id, total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        # A resumed "eval" phase means this iteration's training already finished; go straight to evaluation.
        skip_train = (_iter_id == start_iter_id) and (start_phase == "eval")
        batches_to_skip = start_batches_done if (_iter_id == start_iter_id and start_phase == "train") else 0
        batches_done = 0

        if not skip_train:
            if RANK == 0:
                print("TRAIN" + (f" (skipping {batches_to_skip} already-trained batches)" if batches_to_skip else ""))
            train_state.model.train()
            try:
                for set_name, batch, global_batch_size in train_loader:
                    if batches_done < batches_to_skip:
                        # Same seeded batch order as the interrupted run: skip what was already trained on.
                        batches_done += 1
                        continue

                    metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)
                    batches_done += 1
                    if config.ema:
                        ema_helper.update(train_state.model)

                    # Checkpoint before touching wandb: after a preemption signal its service is already dead.
                    if stop_requested(WORLD_SIZE):
                        checkpoint_and_exit(_iter_id, batches_done, "train")

                    if RANK == 0 and metrics is not None:
                        wandb_log(metrics, step=train_state.step)
                        progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                    if RANK == 0 and (time.time() - last_checkpoint_time) > config.checkpoint_interval_minutes * 60:
                        save_resume_checkpoint(config, train_state, ema_helper, _iter_id, batches_done, "train", wandb_run_id, rank=RANK)
                        last_checkpoint_time = time.time()
            except Exception:
                # Anything that breaks after a stop signal (dead data loader worker, dead wandb service, ...)
                # still leaves the completed steps consistent: checkpoint them rather than lose them.
                if _STOP_REQUESTED:
                    checkpoint_and_exit(_iter_id, batches_done, "train")
                raise

            # Training for this iteration is complete; a kill during evaluation resumes here.
            save_resume_checkpoint(config, train_state, ema_helper, _iter_id, batches_done, "eval", wandb_run_id, rank=RANK)
            last_checkpoint_time = time.time()

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config, 
                train_state_eval, 
                eval_loader, 
                eval_metadata, 
                evaluators,
                rank=RANK, 
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)

            if RANK == 0 and metrics is not None:
                wandb_log(metrics, step=train_state.step)
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

        # Iteration fully complete (train + eval): next start is the following iteration.
        if stop_requested(WORLD_SIZE):
            checkpoint_and_exit(_iter_id + 1, 0, "train")
        save_resume_checkpoint(config, train_state, ema_helper, _iter_id + 1, 0, "train", wandb_run_id, rank=RANK)
        last_checkpoint_time = time.time()

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
