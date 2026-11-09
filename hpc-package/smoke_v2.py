#!/usr/bin/env python3
"""Gate G0 of PREREG-v2: the smallest checks that fail if the v2 logic breaks. CPU, no real data.

    .venv/bin/python smoke_v2.py      (then: .venv/bin/python smoke_test.py for v1)
"""
import copy, os, tempfile
import numpy as np
import torch

import run_v2 as V
import smoke_test as S          # TinyDS, make_bundle, _agg_inputs, _flat from the v1 smoke test
from aggregators import aawa_v2, aawa_v2_weights, fedavg, memory_avg

cpu = torch.device("cpu")
V.DEVICE = V.V1.DEVICE = cpu
V.LOCAL_EPOCHS = 1
S.R.NUM_CLIENTS = 4

# 1. full participation is fedavg() bit for bit
g, sds, w = S._agg_inputs(4)
dists = np.array([[.5, .5, 0, 0], [0, 0, .5, .5], [.25, .25, .25, .25], [.5, .5, 0, 0]])
a = aawa_v2(copy.deepcopy(g), sds, list(w), [0, 1, 2, 3], dists, w)
assert torch.equal(S._flat(a), S._flat(fedavg(copy.deepcopy(g), sds, list(w)))), "full participation != FedAvg"

# 2. a client absent since round 0 (no history needed) shifts weight to whoever holds its classes,
#    weights sum to 1, factors stay <= cap
wv, f = aawa_v2_weights([w[i] for i in (0, 1, 2)], [0, 1, 2], dists, w, cap=3.0)
base = np.array(w[:3]) / sum(w[:3])
assert abs(wv.sum() - 1) < 1e-12 and (f <= 3.0 + 1e-12).all(), (wv, f)
assert wv[0] > base[0] and f[0] > 1, "holder of the absent client's classes not up-weighted"
assert wv[1] < base[1] and f[1] < 1, "holder of over-represented classes not down-weighted"
a = aawa_v2(copy.deepcopy(g), sds[:3], w[:3], [0, 1, 2], dists, w)
assert not torch.equal(S._flat(a), S._flat(fedavg(copy.deepcopy(g), sds[:3], w[:3])))

# 3. a returning victim is never down-weighted for its absence: weight ratio w/b >= 1 when it
#    holds nothing over-represented, > 1 when it holds a withheld class
d2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])
wv, _ = aawa_v2_weights([100, 100, 100], [0, 1, 3], d2, [100] * 4)   # client 2 absent, 3 just returned
assert np.allclose(wv * 3, 1), wv                                     # nothing to restore: exactly FedAvg
wv, _ = aawa_v2_weights([100, 100, 100], [0, 2, 3],
                        np.array([[.7, .3, 0, 0], [0, 0, .5, .5], [0, .9, 0, .1], [.25] * 4]), [100] * 4)
assert wv[2] * 3 > 1, wv                                              # client 3 holds class 1, withheld with client 1

# 3b. memory arms (amendment 1): full participation is fedavg() bit for bit; never-seen clients
#     contribute nothing; the staleness cap drops old updates; memory is per run
g, sds, w = S._agg_inputs(4)
mem = {}
a = memory_avg(copy.deepcopy(g), sds, list(w), [0, 1, 2, 3], mem, 0, w)
assert torch.equal(S._flat(a), S._flat(fedavg(copy.deepcopy(g), sds, list(w)))), "memory != FedAvg at full participation"
assert sorted(mem) == [0, 1, 2, 3] and all(r == 0 for _, r in mem.values())
mem, log = {}, []
a = memory_avg(copy.deepcopy(g), sds[:3], w[:3], [0, 1, 2], mem, 0, w, log=log)   # client 3 never seen
assert log[0][1] == [0, 1, 2], log
ref = fedavg(copy.deepcopy(g), sds[:3], w[:3])
assert torch.allclose(S._flat(a), S._flat(ref), atol=1e-6), "never-seen client changed the aggregate"
for window, now, want in ((None, 50, [0, 1, 2, 3]), (10, 9, [0, 1, 2, 3]), (10, 10, [0, 1, 2])):
    mem, log = {}, []                                                                  # boundary: age 9 in, age 10 out
    gg = memory_avg(copy.deepcopy(g), [sds[3]], [w[3]], [3], mem, 0, w)               # client 3 seen at round 0
    memory_avg(copy.deepcopy(gg), sds[:3], w[:3], [0, 1, 2], mem, now, w, window=window, log=log)
    assert sorted(log[0][1]) == want, (window, now, log[0][1])
    assert abs(sum(log[0][0]) - 1) < 1e-12
mem = {}
memory_avg(copy.deepcopy(g), [sds[3]], [w[3]], [3], mem, 0, w)
d_before = S._flat(type("M", (), {"state_dict": lambda self: mem[3][0]})())
gg = memory_avg(copy.deepcopy(g), sds[:3], w[:3], [0, 1, 2], mem, 1, w)          # reuse, then 3 returns
memory_avg(copy.deepcopy(gg), [sds[3]], [w[3]], [3], mem, 2, w)
assert mem[3][1] == 2 and not torch.equal(S._flat(type("M", (), {"state_dict": lambda self: mem[3][0]})()), d_before), \
    "returning client's memory not refreshed"

# 4. grid: 650 cells, unique keys, defences that equal FedAvg at full participation never on no_attack
cells = V.build_cells()
assert len(cells) == 650 and len({V.cell_key(c) for c in cells}) == 650, len(cells)
assert not any(c["aggregator"] in V.FULL_PARTICIPATION_EQUALS_FEDAVG and c["attack"] == "no_attack" for c in cells)
from collections import Counter
assert Counter(c["block"] for c in cells) == {"A": 330, "D": 60, "B": 20, "C": 20, "H": 220}, Counter(c["block"] for c in cells)
assert [c["dataset"] for c in cells][0] == "emnist" and cells[-1]["dataset"] == "cifar10"

# 5. end to end on tiny data: logs present, v1 under a from-start attack == FedAvg weights (D1),
#    twin runs identical, DP never reaches the attacker
bundle = S.make_bundle()
sizes = [len(ix) for ix in bundle[2]]
tmp = tempfile.mkdtemp()
cell = lambda **k: dict(dict(block="A", dataset="emnist", attack="strategic_distribution", aggregator="aawa",
                             seed=1, suppression=0.3, cap=3.0, dp_epsilon=None, rounds=3), **k)
r = V.run_cell(cell(), bundle, os.path.join(tmp, "a.json"))
assert r["done"] and len(r["class_accuracies"]) == 3 and len(r["class_accuracies"][0]) == 4
assert [sum(c) for c in r["class_counts"]] == sizes, "class_counts do not match client sizes"
for ws, part in zip(r["weights"], r["participants"]):
    fa = np.array([sizes[i] for i in part], float)
    assert np.abs(np.array(ws) - fa / fa.sum()).max() < 1e-9, "v1 is not FedAvg under strategic_distribution"
r2 = V.run_cell(cell(aggregator="aawa_v2"), bundle, os.path.join(tmp, "b.json"))
r3 = V.run_cell(cell(aggregator="aawa_v2"), bundle, os.path.join(tmp, "c.json"))
assert r2["accuracies"] == r3["accuracies"] and r2["weights"] == r3["weights"], "twin runs differ"
r4 = V.run_cell(cell(aggregator="aawa_v2", dp_epsilon=1.0, block="C"), bundle, os.path.join(tmp, "d.json"))
assert r4["victims_per_round"] == r2["victims_per_round"], "DP noise reached the attacker"
r5 = V.run_cell(cell(aggregator="fedavg", attack="combined_strategic", suppression=0.5), bundle, os.path.join(tmp, "e.json"))
assert all(len(v) == 2 for v in r5["victims_per_round"]), "suppression fraction not applied"
assert r["deterministic"] is True and r["prereg_sha256"] == V.PREREG_SHA256
for arm in ("mifa", "mem10"):
    m1 = V.run_cell(cell(aggregator=arm, attack="combined_strategic", rounds=4), bundle, os.path.join(tmp, "m1.json"))
    m2 = V.run_cell(cell(aggregator=arm, attack="combined_strategic", rounds=4), bundle, os.path.join(tmp, "m2.json"))
    assert m1["accuracies"] == m2["accuracies"], f"{arm}: memory leaked between runs or nondeterministic"
    assert all(set(p) <= set(c) for p, c in zip(m1["participants"], m1["contributors"])), "participant missing from contributors"
    md = V.run_cell(cell(aggregator=arm, rounds=3), bundle, os.path.join(tmp, "m3.json"))   # strategic_distribution
    assert all(sorted(c) == sorted(p) for p, c in zip(md["participants"], md["contributors"])), "never-seen victim contributed"
assert not os.path.exists(os.path.join(tmp, "a.json.tmp")), "atomic write left a temp file"

# 5b. a diverged (NaN) model is flagged as a result, not raised
real_fedavg = V.fedavg
def nan_fedavg(g, sds_, w_):
    g = real_fedavg(g, sds_, w_)
    with torch.no_grad():
        for p_ in g.parameters():
            p_.fill_(float("nan"))
    return g
V.fedavg = nan_fedavg
rn = V.run_cell(cell(aggregator="fedavg", attack="no_attack", rounds=2), bundle, os.path.join(tmp, "n.json"))
V.fedavg = real_fedavg
assert rn["done"] and rn["diverged_at"] == 0, rn["diverged_at"]

# 5c. worker error handling: CUDA-type error -> .err appended, lock released, worker exits 3;
#     other error -> .err, lock kept, worker continues; success -> .done; done JSON without marker -> marker restored
import sys as _sys
saved = (V.prereg_ok, V.DEVICE, V.build_cells, V.RESULTS_DIR, V.LOCK_DIR, V.run_cell, V.bundle_for)
wk = os.path.join(tmp, "wk"); V.RESULTS_DIR, V.LOCK_DIR = wk, os.path.join(wk, "locks")
V.prereg_ok, V.DEVICE, V.bundle_for = (lambda: 7), torch.device("cuda"), (lambda c: None)
one = cell(attack="random_30")
V.build_cells = lambda: [one]
key = V.cell_key(one); lock = os.path.join(V.LOCK_DIR, key + ".lock"); err = os.path.join(wk, key + ".err")
_sys.argv = ["run_v2.py", "--claim"]
def boom(*a): raise RuntimeError("CUDA error: an illegal memory access was encountered")
V.run_cell = boom
try:
    V.main(); raise AssertionError("worker did not exit on a CUDA error")
except SystemExit as e:
    assert e.code == 3 and not os.path.exists(lock) and open(err).read().count("=== ") == 1
def bad(*a): raise ValueError("bug")
V.run_cell = bad
V.main()
assert os.path.exists(lock) and open(err).read().count("=== ") == 2, "non-CUDA error must keep the lock"
os.remove(lock)
V.run_cell = lambda c, b, out: (V.dump(dict(done=True), out), {"last10_accuracy": 0.5})[1]
V.main()
done = os.path.join(wk, key + ".done")
assert os.path.exists(done); os.remove(done); os.remove(lock)
V.main()
assert os.path.exists(done), "done JSON without marker: marker not restored"
V.prereg_ok, V.DEVICE, V.build_cells, V.RESULTS_DIR, V.LOCK_DIR, V.run_cell, V.bundle_for = saved

# 6. the grid refuses to run unless PREREG-v2.md matches the approved hash
saved = V.PREREG_SHA256
V.PREREG_SHA256 = "0" * 64
try:
    V.prereg_ok()
    raise AssertionError("ran against an unapproved PREREG")
except SystemExit:
    pass
V.PREREG_SHA256 = saved
print("OK — G0 v2 smoke checks passed")
