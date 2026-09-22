"""Find the batch size that fits, and what it costs, for any arch on any dataset.

    DISABLE_COMPILE=1 python scripts/ebrm_memprobe.py 64,128,256,384 \
        arch=ebrm data_paths="[data/sudoku-extreme-1k-aug-1000]" epochs=50000 arch.alpha_H=0.01 ...

First arg: comma-separated batch sizes (ascending; they need not be powers of two). The rest are
hydra overrides exactly as passed to pretrain.py. Sequence length, vocabulary and the number of
training examples are read from the dataset named by data_paths, so the numbers match a real run;
with no readable dataset it falls back to maze-shaped random data.

Reports peak allocated memory, seconds per step, and the projected training time for `epochs`.
Peak memory is reached inside the first training step (the retained GD cycle), so 2 steps suffice.
Stops at the first batch size that runs out of memory: the largest one printed is the one to use.
"""
import json, os, sys, time, math
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from utils.functions import load_model_class

batch_sizes = [int(x) for x in sys.argv[1].split(",")]
overrides = sys.argv[2:]
device = "cuda" if torch.cuda.is_available() else "cpu"

with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
    cfg = compose(config_name="cfg_pretrain", overrides=overrides)
arch = dict(cfg.arch)
epochs = int(cfg.epochs)

# Dataset shape: read it from the dataset itself so the probe matches the real run.
VOCAB, SEQ_LEN, NUM_IDS, EPOCH_EXAMPLES = 6, 900, 1, 1000  # maze-shaped fallback
data_path = list(cfg.data_paths)[0] if len(cfg.data_paths) else None
meta_file = os.path.join(data_path, "train", "dataset.json") if data_path else None
if meta_file and os.path.isfile(meta_file):
    meta = json.load(open(meta_file))
    VOCAB, SEQ_LEN, NUM_IDS = meta["vocab_size"], meta["seq_len"], meta["num_puzzle_identifiers"]
    EPOCH_EXAMPLES = int(round(meta["total_groups"] * meta["mean_puzzle_examples"]))
    print(f"dataset: {data_path}  seq_len={SEQ_LEN} vocab={VOCAB} identifiers={NUM_IDS} examples/epoch={EPOCH_EXAMPLES}", flush=True)
else:
    print(f"dataset: {data_path} not found; using maze-shaped random data (seq_len={SEQ_LEN}, vocab={VOCAB})", flush=True)

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
        "puzzle_identifiers": torch.randint(0, max(NUM_IDS, 1), (B,), dtype=torch.int32, device=device),
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
