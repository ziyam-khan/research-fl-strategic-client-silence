#!/usr/bin/env python3
"""Smallest check that fails if the runner logic breaks. CPU, seconds, no real data.

    python3 smoke_test.py

Covers: a cell runs and writes every round; the defender's ablated signal does
not leak into the attacker's observable; AAWA reduces exactly to FedAvg under
full participation and does not once a similar client is suppressed; completed
cells are skipped on resume and truncated ones are redone.
"""

import json
import os
import shutil
import tempfile

import numpy as np
import torch

import run_journal as R
from model import CIFAR10Net


class TinyDS:
    def __init__(self, n=160, num_classes=4):
        g = torch.Generator().manual_seed(0)
        self.data = torch.randn(n, 3, 32, 32, generator=g)
        self.targets = [i % num_classes for i in range(n)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]


def make_bundle():
    from data import dirichlet_split, get_client_class_distribution, make_client_loaders
    ds = TinyDS()
    test_loader = torch.utils.data.DataLoader(ds, batch_size=64)
    ci = dirichlet_split(ds, R.NUM_CLIENTS, R.ALPHA, 42)
    ci = [c if c else [0] for c in ci]          # tiny data can leave a client empty
    cd = get_client_class_distribution(ds, ci, 4)
    return (ds, test_loader, ci, cd, CIFAR10Net, 4)


def _agg_inputs(n=4):
    """n synthetic client updates + a global model, for direct aggregator tests."""
    torch.manual_seed(0)
    g = CIFAR10Net(num_classes=4)
    sds = []
    for i in range(n):
        m = CIFAR10Net(num_classes=4)
        sds.append({k: v.clone() for k, v in m.state_dict().items()})
    return g, sds, [100.0 * (i + 1) for i in range(n)]


def _flat(model):
    return torch.cat([v.float().flatten() for _, v in sorted(model.state_dict().items())])


def check_aawa_reduces_to_fedavg():
    from aggregators import aawa, fedavg
    import copy
    n = 4
    g, sds, w = _agg_inputs(n)
    hist = {i: [True] * 10 for i in range(n)}          # nobody ever absent
    scores = {i: [1.0 + i] * 10 for i in range(n)}
    dists = torch.eye(n, 4).numpy()

    a = aawa(copy.deepcopy(g), sds, list(w), list(range(n)), hist, scores, dists)
    f = fedavg(copy.deepcopy(g), sds, list(w))
    assert torch.allclose(_flat(a), _flat(f), atol=1e-6), \
        "AAWA does not reduce to FedAvg under full participation"


def check_aawa_diverges_when_suppressed():
    from aggregators import aawa, fedavg
    import copy
    n = 4
    g, sds, w = _agg_inputs(n)
    present = [0, 1, 2]
    # client 3 absent recently and was a high-signal client => must be flagged
    hist = {0: [True] * 10, 1: [True] * 10, 2: [True] * 10, 3: [False] * 10}
    scores = {0: [1.0] * 10, 1: [1.0] * 10, 2: [1.0] * 10, 3: [50.0] * 10}
    # client 0 shares the suppressed client's distribution, client 1 does not.
    # Compensation is driven by that similarity, so it must reweight them apart.
    # (One-hot orthogonal distributions give similarity 0 for everyone and AAWA
    # is genuinely inert — see the note in the run summary.)
    dists = np.array([[0.5, 0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.5, 0.5],
                      [0.25, 0.25, 0.25, 0.25],
                      [0.5, 0.5, 0.0, 0.0]])

    a = aawa(copy.deepcopy(g), [sds[i] for i in present], [w[i] for i in present],
             present, hist, scores, dists)
    f = fedavg(copy.deepcopy(g), [sds[i] for i in present], [w[i] for i in present])
    assert not torch.allclose(_flat(a), _flat(f), atol=1e-6), \
        "AAWA is inert against a suppressed high-value client — compensation never fired"


def cell(**kw):
    c = dict(phase=1, dataset="cifar10", attack="no_attack", aggregator="fedavg",
             seed=42, rounds=2, window=10, tau=0.5, signal="update_norm")
    c.update(kw)
    return c


def main():
    R.DEVICE = torch.device("cpu")
    R.NUM_CLIENTS, R.LOCAL_EPOCHS = 4, 1
    tmp = tempfile.mkdtemp()
    try:
        bundle = make_bundle()

        # 1. a cell runs, records every round, and marks itself done
        p = os.path.join(tmp, "a.json")
        res = R.run_cell(cell(), bundle, p)
        assert res["done"] is True
        assert len(res["accuracies"]) == 2, res["accuracies"]
        assert len(res["round_seconds"]) == 2
        assert res["model_bytes"] > 0
        assert json.load(open(p))["done"] is True

        # 2. the defender's ablated signal must not leak into the attacker's
        #    observable. fedavg ignores the signal entirely, so switching it must
        #    leave the victim sequence bit-identical. If someone rewires the
        #    ablated score back into client_info, this fails.
        a = R.run_cell(cell(aggregator="fedavg", attack="strategic_gradient",
                            signal="update_norm", rounds=6), bundle, os.path.join(tmp, "b.json"))
        b_ = R.run_cell(cell(aggregator="fedavg", attack="strategic_gradient",
                             signal="grad_norm", rounds=6), bundle, os.path.join(tmp, "c.json"))
        assert a["victims_per_round"] == b_["victims_per_round"], \
            "defender's ablation signal is leaking into the attack"
        assert any(v for v in a["victims_per_round"]), "attack never suppressed anyone"

        # 3. design invariant, checked on the aggregator directly (end-to-end
        #    accuracy is chance-pinned on synthetic data, too coarse to see this):
        #    with full participation AAWA must reduce exactly to FedAvg.
        check_aawa_reduces_to_fedavg()

        # 4. and it must NOT reduce to FedAvg once clients go absent
        check_aawa_diverges_when_suppressed()

        # 4b. Phase 4 privacy: noisy histograms are valid distributions, are
        #     deterministic per seed, actually move at small epsilon, and never
        #     reach the attacker (fedavg ignores them, so victims must match).
        cd = bundle[3]; sizes = [len(ix) for ix in bundle[2]]
        n1 = R.privatize_distributions(cd, sizes, 0.1, 42)
        assert np.allclose(n1.sum(axis=1), 1.0) and (n1 >= 0).all()
        assert np.array_equal(n1, R.privatize_distributions(cd, sizes, 0.1, 42))
        assert not np.allclose(n1, cd), "epsilon=0.1 noise had no effect"
        pa = R.run_cell(cell(aggregator="fedavg", attack="strategic_distribution", rounds=4),
                        bundle, os.path.join(tmp, "d.json"))
        pb = R.run_cell(cell(aggregator="fedavg", attack="strategic_distribution", rounds=4,
                             phase=4, dp_epsilon=0.1), bundle, os.path.join(tmp, "e.json"))
        assert pa["victims_per_round"] == pb["victims_per_round"], "DP noise leaked to attacker"
        assert pa["accuracies"] == pb["accuracies"], "DP path perturbed training RNG"
        q = R.run_cell(cell(aggregator="aawa", attack="combined_strategic", rounds=3,
                            phase=4, dp_epsilon=1.0), bundle, os.path.join(tmp, "f.json"))
        assert q["done"] and q["config"]["precision"] == "fp32"

        # 5. resume: done cells are skipped, truncated ones are redone
        R.RESULTS_DIR = tmp
        R.build_cells = lambda phases: [cell(attack="random_30")]
        R.load_dataset = lambda name: (bundle[0], bundle[1], bundle[4], bundle[5])
        R.dirichlet_split = lambda *a, **k: bundle[2]
        R.get_client_class_distribution = lambda *a, **k: bundle[3]
        R.SEEDS = [42]

        import sys
        sys.argv = ["x"]
        R.main()
        done_path = os.path.join(tmp, R.cell_key(cell(attack="random_30")) + ".json")
        assert os.path.exists(done_path)
        mtime = os.path.getmtime(done_path)
        R.main()                                  # second pass must skip
        assert os.path.getmtime(done_path) == mtime, "completed cell was recomputed"

        open(done_path, "w").write('{"trunc')     # simulate a kill mid-write
        R.main()
        assert json.load(open(done_path))["done"] is True, "truncated cell not redone"

        print("\nOK — all smoke checks passed")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
