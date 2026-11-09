#!/usr/bin/env python3
"""Paper A v2 runner (../PREREG-v2.md, with amendment 1). Reuses v1's local training and DP
mechanism from run_journal.py unchanged; adds CIFAR-100, UCI HAR, AAWA v2, memory arms
(MIFA-style, and a 10-round staleness cap), per-class accuracy, aggregation-weight logs and
deterministic GPU mode. Refuses to run the grid unless PREREG-v2.md and every file it freezes
match their approved sha256. v1 results (results/journal/) are read-only and never touched.

    python3 run_v2.py --list        # cells, in run order
    python3 run_v2.py --gate        # G1: twin cell, 3 rounds, prints identical? and s/round
    python3 run_v2.py --preflight   # every dataset x every arm, 2 rounds, real data; exit 1 on failure
    python3 run_v2.py --claim       # work-queue worker (launch many)
    AAWA_DETERMINISTIC=0 ...        # only if G1 or the deterministic preflight fails
"""
import os

DETERMINISTIC = os.environ.get("AAWA_DETERMINISTIC", "1") == "1"
if DETERMINISTIC:  # must be set before the CUDA context exists
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse, hashlib, json, re, sys, tempfile, time, traceback

import numpy as np
import torch

import run_journal as V1  # v1 frozen constants, train_client, privatize_distributions
from run_journal import (ALPHA, BATCH_SIZE, LOCAL_EPOCHS, LR, NUM_CLIENTS, ROUNDS,
                         privatize_distributions, train_client)
from data import (dirichlet_split, get_cifar10, get_cifar100, get_client_class_distribution,
                  get_emnist, get_har, make_client_loaders, subject_split)
from model import CIFAR10Net, HARNet, MNISTNet
from attacks import CombinedStrategic, NoAttack, RandomDropout, StrategicDistribution, StrategicGradient
from aggregators import aawa, aawa_v2, fedavg, memory_avg

torch.backends.cudnn.benchmark = not DETERMINISTIC  # run_journal turned it on at import
torch.backends.cudnn.deterministic = DETERMINISTIC
torch.use_deterministic_algorithms(DETERMINISTIC)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)  # projects/paper-1 (paths in the PREREG hash tables are relative to it)
PREREG = os.path.join(BASE, "PREREG-v2.md")
PREREG_SHA256 = "566c94ac291599359ed0dcc13cbfc3ae12254f710839209e020eec914855f389"  # PREREG-v2.md with amendment 1, approved by Claude 2026-11-06
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results", "v2")
LOCK_DIR = os.environ.get("AAWA_LOCK_DIR", "/tmp/aawa-v2-locks")
DEVICE = V1.DEVICE

SEEDS = (1, 2, 3, 4, 5)
SUPPRESSION, CAP = 0.3, 3.0
W_V1, TAU_V1 = 10, 0.5            # AAWA v1 as published, update_norm signal
MEM_WINDOW = {"mifa": None, "mem10": 10}   # mem10: updates from the last 10 rounds incl. current (AAWA's window)
ARMS = ("fedavg", "aawa", "aawa_v2", "mifa", "mem10")
FULL_PARTICIPATION_EQUALS_FEDAVG = ("aawa_v2", "mifa", "mem10")  # so no no_attack cells for these
# name: (loader, model, classes, split). "dirichlet" = v1's Dirichlet(alpha) over NUM_CLIENTS;
# "subject" = HAR's natural one-client-per-subject federation (21 clients).
DATASETS = {"cifar10": (get_cifar10, CIFAR10Net, 10, "dirichlet"),
            "cifar100": (get_cifar100, CIFAR10Net, 100, "dirichlet"),
            "emnist": (get_emnist, MNISTNet, 47, "dirichlet"),
            "har_dir": (get_har, HARNet, 6, "dirichlet"),
            "har_subject": (get_har, HARNet, 6, "subject")}
ATTACKS = ("no_attack", "random_30", "strategic_gradient", "strategic_distribution", "combined_strategic")
STRAT2 = ("strategic_distribution", "combined_strategic")
MIN_CLIENT_SIZE = 10  # NIID-Bench rule (Li et al., ICDE 2022): redraw a Dirichlet split until every client has >= 10
SPLIT_ATTEMPT = {}    # (dataset, seed) -> redraws needed; 0 means v1's split for that seed, unchanged


def dirichlet_min(train_set, seed):
    """v1's dirichlet_split (attempt 0 = v1's split for this seed); if a client gets < MIN_CLIENT_SIZE
    samples, redraw from SeedSequence([seed, attempt]), a stream disjoint from every other seed in use."""
    for attempt in range(100):
        ci = dirichlet_split(train_set, NUM_CLIENTS, ALPHA, seed if attempt == 0 else np.random.SeedSequence([seed, attempt]))
        if min(len(c) for c in ci) >= MIN_CLIENT_SIZE:
            return ci, attempt
    raise ValueError(f"seed {seed}: no Dirichlet split with every client >= {MIN_CLIENT_SIZE} in 100 draws")


REQUIRED_FROZEN = {"analyze_v2.py", "hpc-package/run_v2.py", "hpc-package/run_journal.py", "hpc-package/src/aggregators.py",
                   "hpc-package/src/attacks.py", "hpc-package/src/data.py", "hpc-package/src/model.py"}


def make_attack(name, frac, seed):
    return {"no_attack": lambda: NoAttack(),
            "random_30": lambda: RandomDropout(fraction=frac, seed=seed),
            "strategic_gradient": lambda: StrategicGradient(fraction=frac, seed=seed),
            "strategic_distribution": lambda: StrategicDistribution(fraction=frac),
            "combined_strategic": lambda: CombinedStrategic(fraction=frac, seed=seed)}[name]()


def build_cells():
    """PREREG-v2 + amendment 1 grid (650 cells), grouped by dataset so workers reload data rarely:
    EMNIST (A, D, B, C), CIFAR-100 (A, C), HAR Dirichlet (H), HAR subjects (H), CIFAR-10 (A)."""
    cells = []

    def add(block, ds, atk, agg, seed, frac=SUPPRESSION, cap=CAP, eps=None):
        cells.append(dict(block=block, dataset=ds, attack=atk, aggregator=agg, seed=seed,
                          suppression=frac, cap=cap, dp_epsilon=eps, rounds=ROUNDS))

    def main_block(block, ds):
        for s in SEEDS:
            for atk in ATTACKS:
                for agg in ARMS:
                    if not (agg in FULL_PARTICIPATION_EQUALS_FEDAVG and atk == "no_attack"):
                        add(block, ds, atk, agg, s)

    main_block("A", "emnist")
    for s in SEEDS:
        for frac in (0.1, 0.5):
            for atk in ("random_30",) + STRAT2:
                for agg in ("fedavg", "aawa_v2"):
                    add("D", "emnist", atk, agg, s, frac=frac)
    for s in SEEDS:
        for atk in STRAT2:
            for cap in (2.0, 5.0):
                add("B", "emnist", atk, "aawa_v2", s, cap=cap)
    for ds in ("emnist", "cifar100"):
        if ds == "cifar100":
            main_block("A", "cifar100")
        for s in SEEDS:
            for atk in STRAT2:
                add("C", ds, atk, "aawa_v2", s, eps=1.0)
    main_block("H", "har_dir")
    main_block("H", "har_subject")
    main_block("A", "cifar10")
    return cells


def cell_key(c):
    k = f"{c['block']}__{c['dataset']}__{c['attack']}__{c['aggregator']}__f{c['suppression']}__seed{c['seed']}"
    if c["aggregator"] == "aawa_v2":
        k += f"__R{c['cap']:g}"
    if c["dp_epsilon"] is not None:
        k += f"__eps{c['dp_epsilon']:g}"
    return k


def dump(obj, path):
    """Atomic JSON write: a kill mid-write leaves the previous complete file, never a torn one."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, loader, n_classes):
    """Overall and per-class test accuracy. Counting on CPU: CUDA bincount is nondeterministic."""
    model.eval()
    correct, total = torch.zeros(n_classes), torch.zeros(n_classes)
    for x, y in loader:
        y = torch.as_tensor(y)
        pred = model(x.to(DEVICE, non_blocking=True)).argmax(1).cpu()
        total += torch.bincount(y, minlength=n_classes)
        correct += torch.bincount(y[pred == y], minlength=n_classes)
    return (correct.sum() / total.sum()).item(), (correct / total.clamp(min=1)).tolist()


def run_cell(cell, bundle, out_path):
    train_set, test_loader, client_indices, class_dists, model_cls, n_classes = bundle
    seed, n = cell["seed"], len(client_indices)
    torch.manual_seed(seed)
    np.random.seed(seed)
    attack = make_attack(cell["attack"], cell["suppression"], seed)
    sizes = [len(ix) for ix in client_indices]
    defender = class_dists if cell["dp_epsilon"] is None else \
        privatize_distributions(class_dists, sizes, cell["dp_epsilon"], seed)  # attacker keeps the truth

    global_model = model_cls(num_classes=n_classes).to(DEVICE)
    local_model = model_cls(num_classes=n_classes).to(DEVICE)
    loaders = make_client_loaders(train_set, client_indices, BATCH_SIZE)
    part_hist = {i: [] for i in range(n)}
    score_hist = {i: [] for i in range(n)}
    observed = np.zeros(n)  # attacker's observable: last-batch gradient norm, as in v1
    memory = {}             # memory arms only: cid -> (latest delta, round made)
    agg = cell["aggregator"]

    result = dict(cell, accuracies=[], class_accuracies=[], weights=[], contributors=[], participants=[],
                  victims_per_round=[], round_seconds=[], client_sizes=sizes,
                  class_counts=np.rint(class_dists * np.array(sizes)[:, None]).astype(int).tolist(),
                  deterministic=DETERMINISTIC, prereg_sha256=PREREG_SHA256, diverged_at=None, done=False,
                  config=dict(num_clients=n, alpha=ALPHA, local_epochs=LOCAL_EPOCHS, lr=LR,
                              batch_size=BATCH_SIZE, precision="fp32", aawa_v1=(W_V1, TAU_V1, "update_norm"),
                              split=DATASETS[cell["dataset"]][3], split_redraws=SPLIT_ATTEMPT.get((cell["dataset"], seed)),
                              torch=torch.__version__, device=str(DEVICE)))

    for rnd in range(cell["rounds"]):
        t0 = time.time()
        victims = attack.select_victims(rnd, n, {"grad_norms": observed.copy(),
                                                 "class_distributions": class_dists})
        participants = [i for i in range(n) if i not in victims]
        for i in range(n):
            part_hist[i].append(i in participants)

        global_sd = {k: v.clone() for k, v in global_model.state_dict().items()}
        sds, weights = [], []
        for cid in participants:
            local_model.load_state_dict(global_sd)
            local_model.zero_grad(set_to_none=True)
            local_model, _, gnorm, unorm = train_client(local_model, loaders[cid], global_sd, LOCAL_EPOCHS, LR)
            sds.append({k: v.clone() for k, v in local_model.state_dict().items()})
            weights.append(sizes[cid])
            observed[cid] = gnorm
            score_hist[cid].append(unorm)

        log = []
        if sds:
            if agg == "fedavg":
                global_model = fedavg(global_model, sds, weights)
            elif agg == "aawa":
                global_model = aawa(global_model, sds, weights, participants, part_hist, score_hist,
                                    defender, window_size=W_V1, threshold=TAU_V1, log=log)
            elif agg == "aawa_v2":
                global_model = aawa_v2(global_model, sds, weights, participants, defender, sizes,
                                       cap=cell["cap"], log=log)
            else:
                global_model = memory_avg(global_model, sds, weights, participants, memory, rnd, sizes,
                                          window=MEM_WINDOW[agg], log=log)
        w, contrib = (log[0] if agg in MEM_WINDOW else (log[0], None)) if log else (None, None)
        if result["diverged_at"] is None and not all(bool(torch.isfinite(p).all()) for p in global_model.parameters()):
            result["diverged_at"] = rnd  # a result, not a crash: the cell runs on and is flagged in the analysis

        acc, per_class = evaluate(global_model, test_loader, n_classes)
        result["accuracies"].append(acc)
        result["class_accuracies"].append([round(a, 5) for a in per_class])
        result["weights"].append(w)
        result["contributors"].append(contrib)
        result["participants"].append(participants)
        result["victims_per_round"].append(sorted(int(v) for v in victims))
        result["round_seconds"].append(round(time.time() - t0, 3))
        dump(result, out_path)
        if (rnd + 1) % 10 == 0:
            print(f"    round {rnd + 1}/{cell['rounds']}  acc={acc:.4f}", flush=True)

    result["done"] = True
    result["final_accuracy"] = result["accuracies"][-1]
    result["last10_accuracy"] = float(np.mean(result["accuracies"][-10:]))
    dump(result, out_path)
    return result


_cache = {}


def bundle_for(cell):
    """Load one dataset at a time (workers move through the grid dataset by dataset; caching all
    five per worker would cost ~0.5 GB RAM each across 30 workers)."""
    ds = cell["dataset"]
    if ds not in _cache:
        _cache.clear()
        loader_fn, model_cls, n_classes, split = DATASETS[ds]
        train_set, test_set = loader_fn(DATA_DIR, download=False)  # setup_v2.sh fetched and verified the data
        # in-process loading: no worker processes, so no fork/forkserver differences between Python
        # versions and no extra processes competing with 30 grid workers for CPU
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=0)
        splits = {}
        for s in SEEDS:
            ci, SPLIT_ATTEMPT[ds, s] = dirichlet_min(train_set, s) if split == "dirichlet" else (subject_split(train_set), 0)
            splits[s] = (ci, get_client_class_distribution(train_set, ci, n_classes))
        _cache[ds] = (train_set, test_loader, model_cls, n_classes, splits)
    train_set, test_loader, model_cls, n_classes, splits = _cache[ds]
    ci, cd = splits[cell["seed"]]
    return train_set, test_loader, ci, cd, model_cls, n_classes


def sha256(path):
    """run_v2.py is hashed with its own PREREG_SHA256 line blanked, so the PREREG can freeze it too."""
    data = open(path, "rb").read()
    if os.path.basename(path) == "run_v2.py":
        data = re.sub(rb"^PREREG_SHA256 = .*$", b"PREREG_SHA256 = <approved>", data, flags=re.M)
    return hashlib.sha256(data).hexdigest()


def prereg_ok():
    """PREREG-v2.md must match the approved hash, and every file in its hash tables must match its
    latest listed hash (an amendment's table supersedes earlier rows for the same file)."""
    h = sha256(PREREG)
    if PREREG_SHA256 is None or h != PREREG_SHA256:
        sys.exit(f"REFUSING: PREREG-v2.md sha256 {h} != approved {PREREG_SHA256}")
    frozen = dict(re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", open(PREREG).read(), re.M))
    missing = REQUIRED_FROZEN - set(frozen)
    if missing:
        sys.exit(f"REFUSING: PREREG-v2.md does not freeze {sorted(missing)}")
    for rel, want in frozen.items():
        got = sha256(os.path.join(BASE, rel))
        if got != want:
            sys.exit(f"REFUSING: {rel} sha256 {got} != frozen {want}")
    return len(frozen)


def gate():
    """G1: the same EMNIST cell twice for 3 rounds."""
    c = dict(build_cells()[0], rounds=3)
    b = bundle_for(c)
    with tempfile.TemporaryDirectory() as tmp:
        r1 = run_cell(c, b, os.path.join(tmp, "1.json"))
        r2 = run_cell(c, b, os.path.join(tmp, "2.json"))
    same = r1["accuracies"] == r2["accuracies"] and r1["class_accuracies"] == r2["class_accuracies"]
    print(f"G1 deterministic={DETERMINISTIC} identical={same} "
          f"s_per_round={np.mean(r1['round_seconds'][1:] + r2['round_seconds'][1:]):.3f} acc={r1['accuracies']}")  # round 1 = warm-up


def preflight():
    """Real data, every dataset x every arm (+ DP and a 0.5 fraction), 2 rounds each. Exit 1 on failure."""
    assert DEVICE.type == "cuda", "no GPU visible: refusing to run the grid on CPU"
    n = prereg_ok()
    print(f"frozen files verified: {n}; deterministic={DETERMINISTIC}; device={torch.cuda.get_device_name(0)}")
    probes = []
    for ds in DATASETS:
        for agg in ARMS:
            probes.append(dict(block="P", dataset=ds, attack="combined_strategic", aggregator=agg, seed=1,
                               suppression=SUPPRESSION, cap=CAP, dp_epsilon=None, rounds=2))
        probes.append(dict(probes[-1], aggregator="aawa_v2", dp_epsilon=1.0, attack="strategic_distribution"))
        probes.append(dict(probes[-1], aggregator="fedavg", dp_epsilon=None, suppression=0.5, attack="random_30"))
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        for c in probes:
            try:
                b = bundle_for(c)
                r = run_cell(c, b, os.path.join(tmp, "p.json"))
                sizes = [len(x) for x in b[2]]
                print(f"  ok  {c['dataset']:11s} {c['aggregator']:8s} eps={c['dp_epsilon']} f={c['suppression']} "
                      f"acc={r['accuracies']} clients={len(sizes)} min/max size={min(sizes)}/{max(sizes)} "
                      f"s/round={np.mean(r['round_seconds']):.2f}", flush=True)
            except Exception:
                fails += 1
                print(f"  FAIL {c['dataset']} {c['aggregator']}\n{traceback.format_exc()}", flush=True)
    print("PREFLIGHT " + ("PASS" if fails == 0 else f"FAIL ({fails})"))
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--claim", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    cells = build_cells()
    if args.list:
        for c in cells:
            print(cell_key(c))
        print(f"\n{len(cells)} cells")
        return
    if args.gate:
        return gate()
    if args.preflight:
        return preflight()
    prereg_ok()
    assert DEVICE.type == "cuda", "no GPU visible: refusing to run the grid on CPU"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    errors = 0
    for c in cells[args.shard::args.nshards]:
        key = cell_key(c)
        out = os.path.join(RESULTS_DIR, key + ".json")
        if os.path.exists(out):
            try:
                if json.load(open(out)).get("done"):
                    open(os.path.join(RESULTS_DIR, key + ".done"), "w").close()  # marker may be missing
                    continue
            except json.JSONDecodeError:
                pass  # cannot happen with atomic writes; redo if it does
        if args.claim:  # ponytail: lock files on local disk; the launcher clears locks of unfinished cells
            os.makedirs(LOCK_DIR, exist_ok=True)
            try:
                os.close(os.open(os.path.join(LOCK_DIR, key + ".lock"), os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue
        print(f"[{key}]", flush=True)
        t0 = time.time()
        try:
            r = run_cell(c, bundle_for(c), out)
            open(os.path.join(RESULTS_DIR, key + ".done"), "w").close()  # cheap progress marker for the launcher
            print(f"    done last10={r['last10_accuracy']:.4f} ({(time.time() - t0) / 60:.1f}m)", flush=True)
        except Exception as e:  # one bad cell must not take the worker (and its queue) down
            errors += 1
            tb = traceback.format_exc()
            print(f"    ERROR {key}\n{tb}", flush=True)
            with open(os.path.join(RESULTS_DIR, key + ".err"), "a") as f:
                f.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}\n{tb}\n")
            msg = f"{type(e).__name__} {e}"
            if isinstance(e, torch.cuda.OutOfMemoryError) or any(t in msg for t in ("CUDA", "CUBLAS", "cuDNN", "NCCL")):
                # this process's CUDA context may be unusable: free the cell for a fresh worker and exit
                if args.claim:
                    try:
                        os.remove(os.path.join(LOCK_DIR, key + ".lock"))
                    except FileNotFoundError:
                        pass
                sys.exit(3)
    print(f"\nWorker complete. errors={errors}")


if __name__ == "__main__":
    main()
