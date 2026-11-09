#!/usr/bin/env python3
"""Exploratory diagnosis of the journal grid. NOT pre-registered: these numbers explain
the PREREG verdicts in ANALYSIS.md, they do not replace them. Needs numpy (CIFAR split).

Usage: python3 diagnose.py   (prints; the interpretation lives in ANALYSIS.md, below the marker)
"""
import collections, json, os, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "hpc-package")
J = os.path.join(P, "results", "journal")
IC3AI_RUN4 = os.path.join(HERE, "..", "ic3ai-2026", "results", "run4-cifar10-100rounds", "results.json")
SEEDS = (42, 123, 7)
L = lambda k: json.load(open(os.path.join(J, k + ".json")))
pp = lambda x: round(float(x) * 100, 2)

labels = []  # CIFAR-10 python batches: official pickle format, md5-checked at download
for i in range(1, 6):
    with open(os.path.join(P, "data", "cifar-10-batches-py", f"data_batch_{i}"), "rb") as f:
        labels += pickle.load(f, encoding="latin1")["labels"]
labels = np.array(labels)


def split(seed, n=20, alpha=0.1):  # copy of src/data.py dirichlet_split, labels only
    rng, ci = np.random.default_rng(seed), [[] for _ in range(n)]
    for c in range(10):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        cut = (np.cumsum(rng.dirichlet(np.repeat(alpha, n))) * len(idx)).astype(int)[:-1]
        for i, ch in enumerate(np.split(idx, cut)):
            ci[i].extend(ch.tolist())
    return ci


print("== 1. Identical-config twins: P1 AAWA combined seed 42 vs P2 W10/tau0.5/update_norm")
a = L("p1__cifar10__combined_strategic__aawa__seed42")
b = L("p2__cifar10__combined_strategic__aawa__seed42__W10_tau0.5_update_norm")
first = next(i for i, (x, y) in enumerate(zip(a["accuracies"], b["accuracies"])) if x != y)
print(f"   same config {a['config'] == b['config']}; curves split at round {first + 1}; "
      f"final {pp(a['final_accuracy'])} vs {pp(b['final_accuracy'])}")

print("== 2. Round-to-round sd over rounds 91-100 (pp), per seed")
for atk in ("no_attack", "combined_strategic"):
    for agg in ("fedavg", "aawa"):
        print(f"   {atk} {agg}", [round(float(np.std(L(f'p1__cifar10__{atk}__{agg}__seed{s}')['accuracies'][-10:])) * 100, 2) for s in SEEDS])

print("== 3. Last-10-round mean, CIFAR P1 (per seed, mean, AAWA-FedAvg)")
for atk in ("no_attack", "random_30", "strategic_gradient", "strategic_distribution", "combined_strategic"):
    r = {g: [pp(np.mean(L(f"p1__cifar10__{atk}__{g}__seed{s}")["accuracies"][-10:])) for s in SEEDS] for g in ("fedavg", "aawa")}
    print(f"   {atk:23s} fedavg {r['fedavg']} {np.mean(r['fedavg']):.2f} | aawa {r['aawa']} {np.mean(r['aawa']):.2f} | gap {np.mean(r['aawa']) - np.mean(r['fedavg']):+.2f}")
r4 = json.load(open(IC3AI_RUN4))
print("   IC3AI Run 4 (1 seed):", {a: {g: pp(np.mean(v[-10:])) for g, v in d.items()} for a, d in r4.items()})

print("== 4. Victims, CIFAR: persistence and data lost (clients suppressed >= 90 of 100 rounds)")
for atk in ("combined_strategic", "strategic_distribution"):
    for s in SEEDS:
        ci = split(s)
        cnt = np.array([[np.sum(labels[c] == k) for k in range(10)] for c in ci])
        for g in ("fedavg", "aawa"):
            v = L(f"p1__cifar10__{atk}__{g}__seed{s}")["victims_per_round"]
            freq = collections.Counter(x for r in v for x in r)
            heavy = [c for c in range(20) if freq[c] >= 90]
            lost = cnt[heavy].sum(0) / cnt.sum(0)
            print(f"   {atk} s{s} {g}: {len({tuple(r) for r in v})} distinct victim sets; heavy {heavy}; "
                  f"data {cnt[heavy].sum() / cnt.sum():.0%}; worst class lost {lost.max():.0%}")

print("== 5. AAWA exposure (W=10): returning victims (participant, window rate <= 0.5) and rounds with a flaggable absentee")
for ds, ph, atks in (("cifar10", 1, ("combined_strategic", "strategic_gradient", "strategic_distribution", "random_30")),
                     ("emnist", 3, ("combined_strategic", "random_30"))):
    for atk in atks:
        out = []
        for s in SEEDS:
            hist, ever, ret, flag = {i: [] for i in range(20)}, set(), 0, 0
            for r in L(f"p{ph}__{ds}__{atk}__aawa__seed{s}")["victims_per_round"]:
                part = [i for i in range(20) if i not in r]
                for i in range(20):
                    hist[i].append(i in part)
                rate = {i: sum(hist[i][-10:]) / len(hist[i][-10:]) for i in range(20)}
                ret += sum(1 for i in part if rate[i] <= 0.5 and len(hist[i]) >= 5)
                flag += any(i in ever and rate[i] < 0.5 for i in r)  # no score history -> never flagged
                ever.update(part)
            out.append(f"s{s} {ret}/{flag}")
        print(f"   {ds} {atk:23s}", "; ".join(out))
