"""Measure peak GPU memory and step time of a training step at several batch sizes.

    DISABLE_COMPILE=1 python scripts/ebrm_memprobe.py 16,32,48,64 arch=ebrm arch.alpha_H=0.1 ...

First arg: comma-separated batch sizes (ascending). Remaining args: hydra overrides exactly as
passed to pretrain.py. Uses random maze-shaped data (vocab 6, seq 900), so no dataset is needed.
Peak memory is reached inside the first training step (the retained GD cycle), so 2 steps suffice.
"""
import os, sys, time, math
sys.path.insert(0, os.getcwd())
import torch
from hydra import compose, initialize_config_dir
from utils.functions import load_model_class

# Maze-30x30-hard-1k metadata (dataset/build_maze_dataset.py): PAD + "# SGo", 30*30 cells, one identifier.
VOCAB, SEQ_LEN, NUM_IDS = 6, 900, 1
EPOCH_EXAMPLES = 1000  # training mazes, for the runtime estimate

batch_sizes = [int(x) for x in sys.argv[1].split(",")]
overrides = sys.argv[2:]
device = "cuda" if torch.cuda.is_available() else "cpu"

with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
    cfg = compose(config_name="cfg_pretrain", overrides=overrides)
arch = dict(cfg.arch)
epochs = int(cfg.epochs)

if device == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GiB   arch={arch['name']}   overrides={overrides}", flush=True)

for B in batch_sizes:
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model_cfg = {k: v for k, v in arch.items() if k not in ("name", "loss")}
    model_cfg.update(batch_size=B, vocab_size=VOCAB, seq_len=SEQ_LEN, num_puzzle_identifiers=NUM_IDS, causal=False)
    Model = load_model_class(arch["name"]); Loss = load_model_class(arch["loss"]["name"])
    with torch.device(device):
        model = Loss(Model(model_cfg), loss_type=arch["loss"]["loss_type"])
    model.train()
    batch = {
        "inputs": torch.randint(1, VOCAB, (B, SEQ_LEN), dtype=torch.int32, device=device),
        "labels": torch.randint(1, VOCAB, (B, SEQ_LEN), dtype=torch.int32, device=device),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=device),
    }
    try:
        with torch.device(device):
            carry = model.initial_carry(batch)
        times = []
        for _ in range(2):
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
            loss.backward(); model.zero_grad(set_to_none=True)
            if device == "cuda": torch.cuda.synchronize()
            times.append(time.time() - t0)
        sec = times[-1]
        steps = math.ceil(epochs * EPOCH_EXAMPLES / B)
        peak = f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, " if device == "cuda" else ""
        print(f"batch {B:4d}: {peak}{sec:.2f} s/step  ->  {epochs} epochs = {steps} steps ~ {steps * sec / 3600:.1f} h (training only)", flush=True)
    except torch.OutOfMemoryError:
        print(f"batch {B:4d}: OOM (held {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB when it failed)", flush=True)
        break
    finally:
        del model, batch
        carry = loss = None
        if device == "cuda": torch.cuda.empty_cache()
