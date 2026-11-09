#!/usr/bin/env python3
"""
Paper 1 (journal) experiment runner — single file, resumable, shardable.

Covers the IC3AI reviewer gap list:
  R1/R2 more datasets      -> PHASE_3 (EMNIST-balanced, 47 classes)
  R1    longer training    -> ROUNDS = 100 everywhere (IC3AI used 30)
  R1    seed-based stats   -> 3 seeds on every main cell, CIs computed at analysis time
  R2    suspicion ablation -> PHASE_2 (W x tau x score signal)
  R2    comm overhead      -> logged per round in every cell (bytes + wall time)

Usage
-----
    python3 run_journal.py                     # run everything, in phase order
    python3 run_journal.py --shard 0 --nshards 4   # 4 of these in parallel on one H100
    python3 run_journal.py --phases 1          # only phase 1
    python3 run_journal.py --list              # print the cell list and exit

Resume
------
One JSON per cell in results/journal/. A cell with "done": true is skipped on
rerun. Partial cells are rewritten every round, so a killed job loses at most
one cell (~30 min), and you can watch progress with:

    ls results/journal | wc -l
    python3 -c "import json;d=json.load(open('results/journal/<cell>.json'));print(len(d['accuracies']))"

FROZEN HYPERPARAMETERS — read before touching anything below
------------------------------------------------------------
BATCH_SIZE = 64 is a TRAINING hyperparameter, not a speed knob. On 2026-05-19 it
was raised to 256 as an "optimization"; 4x fewer gradient steps per round dropped
every result ~25pp and wasted the whole HPC session. If the run is too slow, use
--nshards (parallel processes), not a bigger batch.

Safe speed knobs: --nshards, cudnn.benchmark, num_workers, pin_memory.

PRECISION (director decision 2026-11-06): full fp32, matching IC3AI (GTX 1650,
no bf16/TF32). bf16 autocast and TF32 change the numerics, so they are OFF. The
workload is CPU/overhead-bound, so fp32 costs no wall time.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data import (dirichlet_split, get_cifar10, get_client_class_distribution,
                  get_emnist, make_client_loaders)
from model import CIFAR10Net, MNISTNet
from attacks import (CombinedStrategic, NoAttack, RandomDropout,
                     StrategicDistribution, StrategicGradient)
from aggregators import aawa, coordinate_median, fedavg, krum, trimmed_mean

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
USE_BF16 = False     # see PRECISION above. Do not flip without the director.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results", "journal")
LOCK_DIR = os.environ.get("AAWA_LOCK_DIR", "/tmp/aawa-locks")

# --- frozen ---------------------------------------------------------------
NUM_CLIENTS = 20
ALPHA = 0.1          # extreme non-IID. Run 1/2 failed at 0.5 — do not raise.
ROUNDS = 100
LOCAL_EPOCHS = 3
LR = 0.01
BATCH_SIZE = 64      # see banner above. DO NOT CHANGE.
SEEDS = [42, 123, 7]
SUPPRESSION = 0.3
# --------------------------------------------------------------------------

DATASETS = {
    # name: (loader_fn, model_cls, num_classes)
    "cifar10": (get_cifar10, CIFAR10Net, 10),
    "emnist": (get_emnist, MNISTNet, 47),
}

AGGREGATORS = {
    "fedavg": fedavg,
    "median": coordinate_median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
    "aawa": aawa,
}

ATTACKS = {
    "no_attack": lambda seed: NoAttack(),
    "random_30": lambda seed: RandomDropout(fraction=SUPPRESSION, seed=seed),
    "strategic_gradient": lambda seed: StrategicGradient(fraction=SUPPRESSION, seed=seed),
    "strategic_distribution": lambda seed: StrategicDistribution(fraction=SUPPRESSION),
    "combined_strategic": lambda seed: CombinedStrategic(fraction=SUPPRESSION, seed=seed),
}

# AAWA defaults (IC3AI Table V best config)
W_DEFAULT, TAU_DEFAULT, SIGNAL_DEFAULT = 10, 0.5, "update_norm"


def build_cells(phases):
    """Flat list of experiment cells, ordered by publication value.

    Phase 1 alone produces the main results table. If the HPC window runs out
    mid-way, you still have a paper.
    """
    cells = []

    if 1 in phases:  # main table: CIFAR-10, all attacks x all aggregators x 3 seeds
        for seed in SEEDS:
            for atk in ATTACKS:
                for agg in AGGREGATORS:
                    cells.append(dict(phase=1, dataset="cifar10", attack=atk,
                                      aggregator=agg, seed=seed, rounds=ROUNDS,
                                      window=W_DEFAULT, tau=TAU_DEFAULT,
                                      signal=SIGNAL_DEFAULT))

    if 2 in phases:  # ablation: AAWA suspicion scoring (R2). Single seed is fine.
        for signal in ("update_norm", "grad_norm"):
            for w in (5, 10, 20):
                for tau in (0.3, 0.5, 0.7):
                    cells.append(dict(phase=2, dataset="cifar10",
                                      attack="combined_strategic", aggregator="aawa",
                                      seed=42, rounds=ROUNDS, window=w, tau=tau,
                                      signal=signal))

    if 3 in phases:  # second dataset (R1/R2). Trimmed attack set to bound runtime.
        for seed in SEEDS:
            for atk in ("no_attack", "random_30", "combined_strategic"):
                for agg in AGGREGATORS:
                    cells.append(dict(phase=3, dataset="emnist", attack=atk,
                                      aggregator=agg, seed=seed, rounds=ROUNDS,
                                      window=W_DEFAULT, tau=TAU_DEFAULT,
                                      signal=SIGNAL_DEFAULT))

    if 4 in phases:  # R1 privacy: AAWA with differentially private class histograms.
        # epsilon = inf is the Phase 1 AAWA cell (same seed/attack), not rerun.
        for seed in SEEDS:
            for atk in ("strategic_gradient", "strategic_distribution", "combined_strategic"):
                for eps in DP_EPSILONS:
                    cells.append(dict(phase=4, dataset="cifar10", attack=atk,
                                      aggregator="aawa", seed=seed, rounds=ROUNDS,
                                      window=W_DEFAULT, tau=TAU_DEFAULT,
                                      signal=SIGNAL_DEFAULT, dp_epsilon=eps))

    return cells


DP_EPSILONS = (10.0, 1.0, 0.1)


def privatize_distributions(class_dists, client_sizes, epsilon, seed):
    """Laplace mechanism on each client's class-count histogram, then renormalise.

    L1 sensitivity 1 (add/remove one sample changes one count by 1), so each
    count gets Laplace(0, 1/epsilon). Negative counts are clipped to 0. Uses its
    own numpy Generator so the torch and global numpy RNG streams that drive
    training and the attack are untouched.
    """
    rng = np.random.default_rng(10_000 + seed)
    counts = class_dists * np.asarray(client_sizes, dtype=float)[:, None]
    noisy = np.clip(counts + rng.laplace(0.0, 1.0 / epsilon, counts.shape), 0.0, None)
    sums = noisy.sum(axis=1, keepdims=True)
    uniform = np.full_like(noisy, 1.0 / noisy.shape[1])
    return np.where(sums > 0, noisy / np.where(sums > 0, sums, 1.0), uniform)


def cell_key(c):
    key = f"p{c['phase']}__{c['dataset']}__{c['attack']}__{c['aggregator']}__seed{c['seed']}"
    if c["phase"] == 2:
        key += f"__W{c['window']}_tau{c['tau']}_{c['signal']}"
    if c["phase"] == 4:
        key += f"__eps{c['dp_epsilon']}"
    return key


def train_client(model, loader, global_sd, epochs, lr):
    """Local training. Returns both suspicion signals so the ablation can switch.

    grad_norm   — last-batch gradient norm. What IC3AI used. Noisy proxy.
    update_norm — ||w_local - w_global||_2, the actual local drift this round.
                  findings.md flags this as the journal-grade replacement.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss, total_samples = 0.0, 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                                enabled=(USE_BF16 and DEVICE.type == "cuda")):
                loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    grad_norm = sum(p.grad.norm().item() ** 2
                    for p in model.parameters() if p.grad is not None) ** 0.5

    local_sd = model.state_dict()
    update_norm = sum((local_sd[k].float() - global_sd[k].float()).norm().item() ** 2
                      for k in local_sd) ** 0.5

    return model, total_loss / max(total_samples, 1), grad_norm, update_norm


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                            enabled=(USE_BF16 and DEVICE.type == "cuda")):
            pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def run_cell(cell, bundle, out_path):
    train_set, test_loader, client_indices, class_dists, model_cls, n_classes = bundle
    seed = cell["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Fresh attack instance per cell. IC3AI reused one instance across all five
    # aggregators, so each aggregator saw a different victim sequence — the
    # comparison was not paired. Fixed here; absolute numbers will differ from
    # the IC3AI tables.
    attack = ATTACKS[cell["attack"]](seed)
    agg_fn = AGGREGATORS[cell["aggregator"]]

    # Defender's view of client class distributions. Phase 4 privatises it; the
    # attacker always keeps the true distributions (worst case for the defender).
    defender_dists = class_dists
    if cell.get("dp_epsilon") is not None:
        defender_dists = privatize_distributions(
            class_dists, [len(ix) for ix in client_indices], cell["dp_epsilon"], seed)

    global_model = model_cls(num_classes=n_classes).to(DEVICE)
    local_model = model_cls(num_classes=n_classes).to(DEVICE)
    client_loaders = make_client_loaders(train_set, client_indices, BATCH_SIZE)
    n_clients = len(client_indices)

    model_bytes = sum(p.numel() * 4 for p in global_model.parameters())

    participation_history = {i: [] for i in range(n_clients)}
    score_history = {i: [] for i in range(n_clients)}   # defender's suspicion signal
    observed_current = np.zeros(n_clients)              # attacker's observable

    # The attacker always targets on gradient norm — that is what it can infer
    # from traffic. Only the defender's internal scoring signal is ablated in
    # phase 2. Feeding the ablated signal to both would make the ablation
    # measure a different attack, not a different defense.

    result = dict(cell, accuracies=[], losses=[], participants_per_round=[],
                  victims_per_round=[], round_seconds=[], model_bytes=model_bytes,
                  aawa_extra_bytes_per_round=0, done=False,
                  config=dict(num_clients=NUM_CLIENTS, alpha=ALPHA,
                              local_epochs=LOCAL_EPOCHS, lr=LR,
                              batch_size=BATCH_SIZE, suppression=SUPPRESSION,
                              precision="bf16" if USE_BF16 else "fp32"))

    for rnd in range(cell["rounds"]):
        t0 = time.time()
        client_info = {"grad_norms": observed_current.copy(),
                       "class_distributions": class_dists}
        victims = attack.select_victims(rnd, n_clients, client_info)
        participants = [i for i in range(n_clients) if i not in victims]

        for i in range(n_clients):
            participation_history[i].append(i in participants)

        state_dicts, weights, ids = [], [], []
        global_sd = {k: v.clone() for k, v in global_model.state_dict().items()}
        round_loss = 0.0

        for cid in participants:
            local_model.load_state_dict(global_sd)
            local_model.zero_grad(set_to_none=True)
            local_model, loss, gnorm, unorm = train_client(
                local_model, client_loaders[cid], global_sd, LOCAL_EPOCHS, LR)
            state_dicts.append({k: v.clone() for k, v in local_model.state_dict().items()})
            weights.append(len(client_indices[cid]))
            ids.append(cid)
            observed_current[cid] = gnorm
            score_history[cid].append(unorm if cell["signal"] == "update_norm" else gnorm)
            round_loss += loss

        if state_dicts:
            if cell["aggregator"] == "aawa":
                global_model = agg_fn(global_model, state_dicts, weights, ids,
                                      participation_history, score_history, defender_dists,
                                      window_size=cell["window"], threshold=cell["tau"])
                # server holds one class-distribution vector per client (static),
                # so AAWA's marginal per-round uplink cost over FedAvg is zero.
                result["aawa_extra_bytes_per_round"] = 0
            else:
                global_model = agg_fn(global_model, state_dicts, weights)

        acc = evaluate(global_model, test_loader)
        result["accuracies"].append(acc)
        result["losses"].append(round_loss / max(len(participants), 1))
        result["participants_per_round"].append(len(participants))
        result["victims_per_round"].append(sorted(int(v) for v in victims))
        result["round_seconds"].append(round(time.time() - t0, 3))

        with open(out_path, "w") as f:
            json.dump(result, f)

        if (rnd + 1) % 10 == 0:
            eta = np.mean(result["round_seconds"]) * (cell["rounds"] - rnd - 1) / 60
            print(f"    round {rnd+1}/{cell['rounds']}  acc={acc:.4f}  eta={eta:.1f}m",
                  flush=True)

    result["done"] = True
    result["final_accuracy"] = result["accuracies"][-1]
    with open(out_path, "w") as f:
        json.dump(result, f)
    return result


def load_dataset(name):
    loader_fn, model_cls, n_classes = DATASETS[name]
    train_set, test_set = loader_fn(DATA_DIR)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    return train_set, test_loader, model_cls, n_classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--claim", action="store_true",
                    help="work-queue mode: any number of workers share the full cell list; "
                         "each cell is claimed with an O_EXCL lock file. After a crash, "
                         "delete $AAWA_LOCK_DIR (default /tmp/aawa-locks) before relaunching.")
    args = ap.parse_args()

    cells = build_cells(set(args.phases))
    if args.list:
        for c in cells:
            print(cell_key(c))
        print(f"\n{len(cells)} cells total")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    mine = cells[args.shard::args.nshards]

    print("=" * 62)
    print(f"Paper 1 journal run | device={DEVICE} | shard {args.shard}/{args.nshards}")
    print(f"clients={NUM_CLIENTS} alpha={ALPHA} rounds={ROUNDS} "
          f"local_epochs={LOCAL_EPOCHS} lr={LR} BATCH_SIZE={BATCH_SIZE}")
    print(f"{len(mine)} of {len(cells)} cells assigned to this shard")
    print("=" * 62, flush=True)

    cache = {}
    for n, cell in enumerate(mine, 1):
        key = cell_key(cell)
        out_path = os.path.join(RESULTS_DIR, key + ".json")

        if os.path.exists(out_path):
            try:
                if json.load(open(out_path)).get("done"):
                    print(f"[{n}/{len(mine)}] SKIP {key}", flush=True)
                    continue
            except json.JSONDecodeError:
                pass  # truncated by a kill mid-write; just redo the cell

        if args.claim:  # ponytail: lock files, not a queue server; stale locks need manual cleanup
            try:
                # locks on local disk: O_EXCL is not reliably atomic on the
                # network filesystem that holds RESULTS_DIR (/workspace).
                os.makedirs(LOCK_DIR, exist_ok=True)
                os.close(os.open(os.path.join(LOCK_DIR, key + ".lock"),
                                 os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue

        ds = cell["dataset"]
        if ds not in cache:
            print(f"  loading {ds}...", flush=True)
            train_set, test_loader, model_cls, n_classes = load_dataset(ds)
            idx = {}
            for seed in SEEDS:
                ci = dirichlet_split(train_set, NUM_CLIENTS, ALPHA, seed)
                idx[seed] = (ci, get_client_class_distribution(train_set, ci, n_classes))
            cache[ds] = (train_set, test_loader, model_cls, n_classes, idx)

        train_set, test_loader, model_cls, n_classes, idx = cache[ds]
        client_indices, class_dists = idx[cell["seed"]]
        bundle = (train_set, test_loader, client_indices, class_dists, model_cls, n_classes)

        print(f"[{n}/{len(mine)}] {key}", flush=True)
        t0 = time.time()
        res = run_cell(cell, bundle, out_path)
        print(f"    done final_acc={res['final_accuracy']:.4f} "
              f"({(time.time()-t0)/60:.1f}m)", flush=True)

    print("\nShard complete.")


if __name__ == "__main__":
    main()
