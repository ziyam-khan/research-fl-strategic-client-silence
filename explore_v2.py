#!/usr/bin/env python3
"""Exploratory analysis of the v2 grid (NOT pre-registered; the confirmatory verdicts are analyze_v2.py's).
Written 2026-11-09 after all 650 cells completed. Everything here is labelled exploratory in the paper.

Usage: python3 explore_v2.py            (stdlib only; prints a report)
Also imported by plot_v2.py for loading and statistics.
"""
import glob, json, os, statistics as st
from math import sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "hpc-package", "results", "v2")
SEEDS = (1, 2, 3, 4, 5)
DS = ("emnist", "cifar100", "cifar10", "har_dir", "har_subject")
DS_LABEL = {"emnist": "EMNIST", "cifar100": "CIFAR-100", "cifar10": "CIFAR-10",
            "har_dir": "HAR (Dirichlet)", "har_subject": "HAR (subjects)"}
ATK = ("random_30", "strategic_gradient", "strategic_distribution", "combined_strategic")
ATK_LABEL = {"no_attack": "No attack", "random_30": "Random", "strategic_gradient": "Gradient",
             "strategic_distribution": "Distribution", "combined_strategic": "Combined"}
METHODS = ("aawa", "aawa_v2", "mifa", "mem10")
M_LABEL = {"fedavg": "FedAvg", "aawa": "AAWA v1", "aawa_v2": "AAWA v2", "mifa": "MIFA", "mem10": "Mem-10"}


def load_all():
    """{(ds, atk, agg, suppression, cap|None, eps|None, seed): result dict} for completed cells."""
    C = {}
    for f in glob.glob(os.path.join(RES, "*.json")):
        d = json.load(open(f))
        if not d.get("done"):
            continue
        cap = d["cap"] if d["aggregator"] == "aawa_v2" else None
        C[(d["dataset"], d["attack"], d["aggregator"], d["suppression"], cap, d["dp_epsilon"], d["seed"])] = d
    return C


def cell(C, ds, atk, agg, seed, frac=0.3, cap=3.0, eps=None):
    if atk == "no_attack" and agg != "aawa":      # v2 / memory arms equal FedAvg with no absence
        agg = "fedavg"
    return C.get((ds, atk, agg, frac, cap if agg == "aawa_v2" else None, eps, seed))


def acc(C, *a, **k):
    """Primary metric per seed: mean test accuracy over rounds 91-100, in %."""
    out = []
    for s in SEEDS:
        d = cell(C, *a, seed=s, **k)
        out.append(100 * d["last10_accuracy"] if d else None)
    return out


def t4_cdf(t):
    """Student-t CDF, 4 degrees of freedom (closed form)."""
    u = t * t / (1 + t * t / 4)
    return 0.5 + 0.375 * (t / sqrt(1 + t * t / 4)) * (1 - u / 12)


def paired(a, b):
    """Mean of a-b over seeds, 95% CI (paired t, df 4) and two-sided p."""
    d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    if len(d) != 5:
        return None
    m, sd = st.mean(d), st.stdev(d)
    se = sd / sqrt(5)
    p = (1.0 if m == 0 else 0.0) if se == 0 else 2 * (1 - t4_cdf(abs(m) / se))  # identical arms -> no difference
    return dict(mean=m, lo=m - 2.776 * se, hi=m + 2.776 * se, p=p, sd=sd)


def gain_matrix(C):
    """G(method) = acc(method) - acc(FedAvg) for every dataset x attack x method, plus FedAvg harm."""
    rows = []
    for ds in DS:
        base = acc(C, ds, "no_attack", "fedavg")
        for atk in ATK:
            fa = acc(C, ds, atk, "fedavg")
            rows.append(dict(ds=ds, atk=atk, what="harm", **paired(base, fa)))
            for m in METHODS:
                rows.append(dict(ds=ds, atk=atk, what=m, **paired(acc(C, ds, atk, m), fa)))
    return rows


def holm(rows, alpha=0.05):
    ps = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    n, ok = len(rows), set()
    for r, i in enumerate(ps):
        if rows[i]["p"] <= alpha / (n - r):
            ok.add(i)
        else:
            break
    for i, row in enumerate(rows):
        row["holm"] = i in ok
    return rows


def staleness(d):
    """Ages (rounds since the client last trained) of memory updates used in each round, and their weight share."""
    last, ages, share = {}, [], []
    for t, (parts, contrib, w) in enumerate(zip(d["participants"], d["contributors"], d["weights"])):
        for c in parts:
            last[c] = t
        stale = [i for i, c in enumerate(contrib) if c not in parts]
        ages += [t - last[contrib[i]] for i in stale if contrib[i] in last]
        if len(w) == len(contrib):
            share.append(sum(w[i] for i in stale) / (sum(w) or 1))
    return ages, share


def absence_runs(d, n=20):
    """Lengths of consecutive-absence runs per client (a client never seen counts its whole absence)."""
    runs, cur = [], [0] * n
    for parts in d["participants"]:
        ps = set(parts)
        for c in range(n):
            if c in ps:
                if cur[c]:
                    runs.append(cur[c])
                cur[c] = 0
            else:
                cur[c] += 1
    return runs + [r for r in cur if r]


def class_points(C, ds, atk):
    """(withheld share w_c, per-class drop) for every class of every FedAvg seed (same rule as analyze_v2 M3)."""
    pts = []
    for s in SEEDS:
        c, base = cell(C, ds, atk, "fedavg", s), cell(C, ds, "no_attack", "fedavg", s)
        cc = c["class_counts"]
        tot = [sum(col) for col in zip(*cc)]
        held = [0.0] * len(tot)
        for vic in c["victims_per_round"]:
            for j in vic:
                for k, n in enumerate(cc[j]):
                    held[k] += n
        R = len(c["victims_per_round"])
        a, b = class_last10(c), class_last10(base)
        pts += [(held[k] / (R * tot[k]), b[k] - a[k]) for k in range(len(tot)) if tot[k]]
    return pts


def class_last10(d):
    ca = d["class_accuracies"][-10:]
    return [100 * st.mean(r[k] for r in ca) for k in range(len(ca[0]))]


def worst_class_harm(C, ds, atk):
    """Per seed: the largest per-class accuracy drop vs no attack (FedAvg), averaged over seeds."""
    v = []
    for s in SEEDS:
        a, b = cell(C, ds, "no_attack", "fedavg", s), cell(C, ds, atk, "fedavg", s)
        v.append(max(x - y for x, y in zip(class_last10(a), class_last10(b))))
    return v


def fmt(r):
    return f"{r['mean']:+6.2f} [{r['lo']:+6.2f},{r['hi']:+6.2f}] p={r['p']:.3f}{' *' if r.get('holm') else ''}"


def main():
    C = load_all()
    print(f"cells loaded: {len(C)}; diverged: {sum(1 for d in C.values() if d.get('diverged_at') is not None)}")

    rows = holm(gain_matrix(C))
    print(f"\n1. Full gain matrix vs FedAvg (pp; 95% CI; * = survives Holm over all {len(rows)} tests)")
    for ds in DS:
        print(f"  {DS_LABEL[ds]}")
        for atk in ATK:
            sel = {r["what"]: r for r in rows if r["ds"] == ds and r["atk"] == atk}
            print(f"    {ATK_LABEL[atk]:<12} harm {fmt(sel['harm'])}")
            for m in METHODS:
                print(f"    {'':<12} {M_LABEL[m]:<7} {fmt(sel[m])}")
    print(f"  Holm survivors: {sum(r['holm'] for r in rows)} of {len(rows)}")

    print("\n2. Convergence: FedAvg, no attack, mean accuracy at rounds 80 / 90 / 100 (5 seeds)")
    for ds in DS:
        cur = [cell(C, ds, "no_attack", "fedavg", s)["accuracies"] for s in SEEDS]
        print(f"  {DS_LABEL[ds]:<16} " + " / ".join(f"{100 * st.mean(c[r - 1] for c in cur):.2f}" for r in (80, 90, 100)))

    print("\n3. Memory mechanism: staleness of reused updates (MIFA / Mem-10), pooled over seeds")
    for m in ("mifa", "mem10"):
        for ds in DS:
            for atk in ATK:
                ages, share = [], []
                for s in SEEDS:
                    a, sh = staleness(cell(C, ds, atk, m, s))
                    ages += a
                    share += sh
                if ages:
                    print(f"  {M_LABEL[m]:<7}{DS_LABEL[ds]:<17}{ATK_LABEL[atk]:<13}median age {st.median(ages):5.1f}  "
                          f"p90 {sorted(ages)[int(0.9 * len(ages))]:4d}  stale weight share {100 * st.mean(share):5.1f}%")

    print("\n4. Absence runs (FedAvg cells): median / mean consecutive rounds absent")
    for ds in ("emnist", "cifar10"):
        for atk in ATK:
            runs = [r for s in SEEDS for r in absence_runs(cell(C, ds, atk, "fedavg", s))]
            print(f"  {DS_LABEL[ds]:<10}{ATK_LABEL[atk]:<13}{st.median(runs):5.1f} / {st.mean(runs):5.1f}  (n={len(runs)})")

    print("\n5. Worst-class harm, FedAvg (largest per-class drop vs no attack; mean ± sd over seeds)")
    for ds in DS:
        print(f"  {DS_LABEL[ds]:<17}" + "  ".join(
            f"{ATK_LABEL[a]} {st.mean(v):5.1f}±{st.stdev(v):4.1f}" for a in ATK for v in [worst_class_harm(C, ds, a)]))

    print("\n6. Power: minimum detectable effect (paired, 5 seeds, ~80% power ≈ (2.776+0.941)·sd/√5)")
    for ds, atk in (("emnist", "combined_strategic"), ("emnist", "strategic_distribution"), ("cifar10", "combined_strategic")):
        r = paired(acc(C, ds, atk, "aawa_v2"), acc(C, ds, atk, "fedavg"))
        print(f"  G(v2) {DS_LABEL[ds]} {ATK_LABEL[atk]}: sd of paired diff {r['sd']:.2f} -> MDE ≈ {3.717 * r['sd'] / sqrt(5):.2f} pp")

    print("\n7. Seed spread of FedAvg under each attack (sd of last-10 accuracy across seeds)")
    for ds in DS:
        print(f"  {DS_LABEL[ds]:<17}" + "  ".join(f"{ATK_LABEL[a]} {st.stdev(acc(C, ds, a, 'fedavg')):.2f}" for a in ("no_attack",) + ATK))


if __name__ == "__main__":
    main()
