#!/usr/bin/env python3
"""Score the v2 grid against PREREG-v2.md. Written before any v2 cell ran; thresholds are copied
from PREREG-v2 and must not be edited after the first v2 cell completes (later analysis goes
below the marker in ANALYSIS-v2.md, labelled exploratory).

Usage: python3 analyze_v2.py [results_dir] [out.md]   (defaults: hpc-package/results/v2, ANALYSIS-v2.md)
Stdlib only. Missing cells -> PENDING. Primary metric: mean test accuracy over rounds 91-100 (pp).
"""
import glob, json, os, statistics as st, sys
from math import sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "hpc-package", "results", "v2")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "ANALYSIS-v2.md")
T975_DF4 = 2.776
SEEDS = (1, 2, 3, 4, 5)
DS = ("emnist", "cifar100", "cifar10")          # the original v2 datasets (claims P1-V9 are defined on these)
HAR = ("har_dir", "har_subject")                # amendment 1
ATK = ("no_attack", "random_30", "strategic_gradient", "strategic_distribution", "combined_strategic")
STRAT = ("strategic_gradient", "strategic_distribution", "combined_strategic")
AGG = ("fedavg", "aawa", "aawa_v2", "mifa", "mem10")

C = {}
for f in glob.glob(os.path.join(RES, "*.json")):
    d = json.load(open(f))
    if d.get("done"):
        C[os.path.basename(f)[:-5]] = d


def key(block, ds, atk, agg, seed, frac=0.3, cap=3.0, eps=None):
    if agg in ("aawa_v2", "mifa", "mem10") and atk == "no_attack":  # identical to FedAvg by construction
        agg = "fedavg"
    k = f"{block}__{ds}__{atk}__{agg}__f{frac}__seed{seed}"
    if agg == "aawa_v2":
        k += f"__R{cap:g}"
    if eps is not None:
        k += f"__eps{eps:g}"
    return k


def cell(ds, atk, agg, seed, frac=0.3, cap=3.0, eps=None):
    block = "H" if ds in HAR else "A" if atk == "no_attack" or (frac == 0.3 and cap == 3.0 and eps is None) else \
        "D" if frac != 0.3 else "B" if cap != 3.0 else "C"
    return C.get(key(block, ds, atk, agg, seed, frac, cap, eps))


def acc(ds, atk, agg, seed, metric="last10", **kw):
    c = cell(ds, atk, agg, seed, **kw)
    if c is None:
        return None
    return 100 * (st.mean(c["accuracies"][-10:]) if metric == "last10" else c["accuracies"][-1])


def seeds(ds, atk, agg, metric="last10", **kw):
    v = [acc(ds, atk, agg, s, metric, **kw) for s in SEEDS]
    return None if None in v else v


def paired(x, y):
    """mean, CI low, CI high of x - y paired by seed."""
    if x is None or y is None:
        return None
    d = [a - b for a, b in zip(x, y)]
    m, h = st.mean(d), T975_DF4 * st.stdev(d) / sqrt(len(d))
    return m, m - h, m + h


def harm(ds, atk, **kw):  # FedAvg harm per seed: acc(no_attack) - acc(atk)
    return paired(seeds(ds, "no_attack", "fedavg"), seeds(ds, atk, "fedavg", **kw))


def gap(ds, atk, agg="aawa_v2", **kw):  # agg - FedAvg, same attack/frac
    fk = {k: v for k, v in kw.items() if k == "frac"}
    return paired(seeds(ds, atk, agg, **kw), seeds(ds, atk, "fedavg", **fk))


rows, surprises = [], []
f2 = lambda p: "—" if p is None else f"{p[0]:+.2f} [{p[1]:+.2f}, {p[2]:+.2f}]"


def verdict(i, claim, ok, ev):
    rows.append((i, claim, "PENDING" if ok is None else "HOLDS" if ok else "FAILS", ev))


def all_ok(*xs):
    """Conjunction of claim parts: a failed part fails the claim even while other parts are pending."""
    if any(x is False for x in xs):
        return False
    return None if any(x is None for x in xs) else True


# ---- primary ----
d = paired(seeds("emnist", "random_30", "fedavg"), seeds("emnist", "combined_strategic", "fedavg"))
hd, hr = harm("emnist", "strategic_distribution"), harm("emnist", "random_30")
verdict("P1", "EMNIST: acc(random) − acc(combined) > 0, CI excl 0; H(dist) > H(random)",
        all_ok(d and d[1] > 0, hd and hr and hd[0] > hr[0]),
        f"random−combined {f2(d)}; H(dist) {f2(hd)} vs H(random) {f2(hr)}")
if d and d[0] < 0:
    surprises.append(f"P1 reverses: combined beats random on EMNIST ({d[0]:+.2f})")

rel = {}
for ds, k in (("cifar100", 100), ("cifar10", 10)):
    no, x = seeds(ds, "no_attack", "fedavg"), seeds(ds, "strategic_distribution", "fedavg")
    rel[ds] = None if no is None or x is None else st.mean((a - b) / (a - 100 / k) for a, b in zip(no, x))
verdict("P2", "Share of above-chance accuracy removed by strategic_distribution: CIFAR-100 > CIFAR-10",
        None if None in rel.values() else rel["cifar100"] > rel["cifar10"],
        "pending" if None in rel.values() else f"CIFAR-100 {rel['cifar100']:.3f} vs CIFAR-10 {rel['cifar10']:.3f}")

g = gap("emnist", "combined_strategic")
verdict("P3", "EMNIST combined: G(v2) ≥ +0.5 pp, CI excl 0", None if g is None else g[0] >= 0.5 and g[1] > 0, f2(g))

# ---- measurement ----
h = {(ds, a): harm(ds, a) for ds in DS for a in ATK[1:]}
m = lambda ds, a: None if h[ds, a] is None else h[ds, a][0]
verdict("M1", "CIFAR-10: H(combined) < H(random) and H(dist) < H(random)",
        all_ok(*(None if m("cifar10", a) is None or m("cifar10", "random_30") is None else m("cifar10", a) < m("cifar10", "random_30")
                 for a in ("combined_strategic", "strategic_distribution"))),
        "; ".join(f"H({a}) {f2(h['cifar10', a])}" for a in ("random_30", "strategic_distribution", "combined_strategic")))
verdict("M2", "CIFAR-100: H(dist) > H(random) and H(combined) > H(random)",
        all_ok(*(None if m("cifar100", a) is None or m("cifar100", "random_30") is None else m("cifar100", a) > m("cifar100", "random_30")
                 for a in ("combined_strategic", "strategic_distribution"))),
        "; ".join(f"H({a}) {f2(h['cifar100', a])}" for a in ("random_30", "strategic_distribution", "combined_strategic")))


def ranks(v):
    o = sorted(range(len(v)), key=v.__getitem__)
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def class_points(ds, atk):
    """(withheld share w_c, accuracy drop_c) for every class of every FedAvg cell of this attack."""
    pts = []
    for s in SEEDS:
        c, base = cell(ds, atk, "fedavg", s), cell(ds, "no_attack", "fedavg", s)
        if c is None or base is None:
            return None
        cc = c["class_counts"]
        tot = [sum(col) for col in zip(*cc)]
        held = [0.0] * len(tot)
        for vic in c["victims_per_round"]:
            for j in vic:
                for k, n in enumerate(cc[j]):
                    held[k] += n
        R = len(c["victims_per_round"])
        for k in range(len(tot)):
            if tot[k] == 0:
                continue
            a_cell = st.mean(r[k] for r in c["class_accuracies"][-10:])
            a_base = st.mean(r[k] for r in base["class_accuracies"][-10:])
            pts.append((held[k] / (R * tot[k]), 100 * (a_base - a_cell)))
    return pts


parts, ok3 = [], []
for ds, thr in (("emnist", 0.3), ("cifar100", 0.3), ("cifar10", 0.0)):
    p = [class_points(ds, a) for a in ("strategic_distribution", "combined_strategic")]
    if None in p:
        ok3.append(None)
        continue
    pts = p[0] + p[1]
    rho = spearman([a for a, _ in pts], [b for _, b in pts])
    ok3.append(bool((rho >= thr if thr else rho > 0)))
    parts.append(f"{ds} ρ={rho:+.2f} (n={len(pts)})")
verdict("M3", "Spearman ρ(withheld share, per-class drop) ≥ 0.3 EMNIST & CIFAR-100, > 0 CIFAR-10", all_ok(*ok3),
        "; ".join(parts) or "pending")

p = class_points("cifar10", "strategic_distribution")
if p is None:
    verdict("M4", "CIFAR-10 dist: classes with w < 0.05 gain accuracy", None, "pending")
else:
    gain = [-b for a, b in p if a < 0.05]
    if gain:
        verdict("M4", "CIFAR-10 dist: classes with w < 0.05 gain accuracy", st.mean(gain) > 0,
                f"mean gain {st.mean(gain):+.2f} pp over {len(gain)} class-seeds")
    else:
        rows.append(("M4", "CIFAR-10 dist: classes with w < 0.05 gain accuracy", "UNTESTABLE", "no class has w < 0.05"))

parts, ok5 = [], []
for a in ("random_30", "strategic_distribution", "combined_strategic"):
    hs = [harm("emnist", a, frac=f) for f in (0.1, 0.3, 0.5)]
    if None in hs:
        ok5.append(None)
        continue
    ok5.append(bool(hs[0][0] < hs[1][0] < hs[2][0]))
    parts.append(f"{a}: " + " / ".join(f"{x[0]:.2f}" for x in hs))
for a in ("strategic_distribution", "combined_strategic"):
    e = [None if harm("emnist", a, frac=f) is None or harm("emnist", "random_30", frac=f) is None
         else harm("emnist", a, frac=f)[0] - harm("emnist", "random_30", frac=f)[0] for f in (0.1, 0.5)]
    if None in e:
        ok5.append(None)
        continue
    ok5.append(bool(e[1] > e[0]))
    parts.append(f"{a}−random excess at 0.1/0.5: {e[0]:+.2f} / {e[1]:+.2f}")
verdict("M5", "EMNIST FedAvg harm rises 0.1<0.3<0.5; strategic excess over random larger at 0.5", all_ok(*ok5),
        "; ".join(parts) or "pending")


# ---- diagnosis replication ----
def ratios(agg, atk, dss):
    """FedAvg-share ratio w_i/b_i for returning victims (window participation <= 0.5, last 10 rounds)."""
    out, maxdev = [], 0.0
    for ds in dss:
        for s in SEEDS:
            c = cell(ds, atk, agg, s)
            if c is None:
                return None, None
            hist = {i: [] for i in range(len(c["client_sizes"]))}
            for vic, part, w in zip(c["victims_per_round"], c["participants"], c["weights"]):
                for i in hist:
                    hist[i].append(i in part)
                if w is None:
                    continue
                tot = sum(c["client_sizes"][i] for i in part)
                for i, wi in zip(part, w):
                    b = c["client_sizes"][i] / tot
                    maxdev = max(maxdev, abs(wi - b))
                    win = hist[i][-10:]
                    if sum(win) / len(win) <= 0.5:
                        out.append(wi / b)
    return out, maxdev


_, dev = ratios("aawa", "strategic_distribution", DS)
verdict("D1", "v1 weights == FedAvg under strategic_distribution (max |Δ| < 1e-9)",
        None if dev is None else dev < 1e-9, "pending" if dev is None else f"max |Δ| {dev:.2e}")
if dev is not None and dev >= 1e-9:
    surprises.append(f"D1 fails: v1 departs from FedAvg under strategic_distribution (max |Δ| {dev:.2e})")
r, _ = ratios("aawa", "combined_strategic", DS)
verdict("D2", "v1 down-weights returning victims under combined (median ratio < 1)",
        None if r is None else bool(r) and st.median(r) < 1,
        "pending" if r is None else f"median {st.median(r):.3f} over {len(r)} events" if r else "no events")
g = gap("emnist", "combined_strategic", "aawa")
verdict("D3", "EMNIST combined: G(v1) ≤ 0", None if g is None else g[0] <= 0, f2(g))

# ---- defence ----
g = gap("emnist", "strategic_distribution")
verdict("V1", "EMNIST dist: G(v2) > 0, CI excl 0", None if g is None else g[1] > 0, f2(g))
gs = [gap("cifar100", a) for a in ("strategic_distribution", "combined_strategic")]
verdict("V2", "CIFAR-100: G(v2) > 0 under dist and combined", all_ok(*(None if x is None else x[0] > 0 for x in gs)),
        "; ".join(f2(x) for x in gs))
gs = [paired(seeds(ds, "combined_strategic", "aawa_v2"), seeds(ds, "combined_strategic", "aawa")) for ds in ("emnist", "cifar100")]
verdict("V3", "acc(v2) − acc(v1) > 0 under combined, EMNIST & CIFAR-100",
        all_ok(*(None if x is None else x[0] > 0 for x in gs)), "; ".join(f2(x) for x in gs))
gs = {ds: gap(ds, "random_30") for ds in DS}
verdict("V4", "random_30: G(v2) ≥ −0.3 on every dataset", all_ok(*(None if x is None else x[0] >= -0.3 for x in gs.values())),
        "; ".join(f"{k} {f2(x)}" for k, x in gs.items()))
gs = {a: gap("cifar10", a) for a in STRAT}
verdict("V5", "CIFAR-10: |G(v2)| ≤ 1 under each strategic attack", all_ok(*(None if x is None else abs(x[0]) <= 1 for x in gs.values())),
        "; ".join(f"{k} {f2(x)}" for k, x in gs.items()))
parts, ok = [], []
for a in ("strategic_distribution", "combined_strategic"):
    ms = [seeds("emnist", a, "aawa_v2", cap=r) for r in (2.0, 3.0, 5.0)]
    if None in ms:
        ok.append(None)
        continue
    mm = [st.mean(x) for x in ms]
    ok.append(bool(max(mm) - min(mm) < 1))
    parts.append(f"{a} R=2/3/5: " + " / ".join(f"{x:.2f}" for x in mm))
verdict("V6", "EMNIST: v2 range across R ∈ {2,3,5} < 1 pp", all_ok(*ok), "; ".join(parts) or "pending")
parts, ok = [], []
for ds in ("emnist", "cifar100"):
    for a in ("strategic_distribution", "combined_strategic"):
        p = paired(seeds(ds, a, "aawa_v2", eps=1.0), seeds(ds, a, "aawa_v2"))
        if p is None:
            ok.append(None)
            continue
        ok.append(bool(abs(p[0]) <= 0.5))
        parts.append(f"{ds}/{a} {f2(p)}")
verdict("V7", "DP ε=1: |v2(ε) − v2| ≤ 0.5 pp", all_ok(*ok), "; ".join(parts) or "pending")
parts, ok = [], []
for a in ("strategic_distribution", "combined_strategic"):
    e = [gap("emnist", a, frac=f) for f in (0.1, 0.5)]
    if None in e:
        ok.append(None)
        continue
    ok.append(bool(e[1][0] > e[0][0]))
    parts.append(f"{a} G at 0.1/0.5: {e[0][0]:+.2f} / {e[1][0]:+.2f}")
verdict("V8", "EMNIST: G(v2) larger at suppression 0.5 than 0.1", all_ok(*ok), "; ".join(parts) or "pending")
r, _ = ratios("aawa_v2", "combined_strategic", ("emnist",))
verdict("V9", "EMNIST combined: v2 median weight ratio of returning victims > 1",
        None if r is None else bool(r) and st.median(r) > 1,
        "pending" if r is None else f"median {st.median(r):.3f} over {len(r)} events" if r else "no events")

# ---- amendment 1: memory arms, v2 inertness, worst class, HAR ----
g = gap("emnist", "combined_strategic", "mifa")
verdict("P4", "EMNIST combined: G(MIFA) ≥ +0.5 pp, CI excl 0", None if g is None else g[0] >= 0.5 and g[1] > 0, f2(g))

parts_ok = [None if (c := cell(ds, "strategic_distribution", "mifa", s)) is None else
            all(sorted(p) == sorted(q) for p, q in zip(c["participants"], c["contributors"]))
            for ds in DS for s in SEEDS]
ok = all_ok(*parts_ok)
verdict("MEM1a", "strategic_distribution: MIFA contributors == participants every round (victims never seen)", ok,
        "pending" if ok is None else "all rounds, all cells" if ok else "a never-seen victim contributed")
gs = {ds: gap(ds, "strategic_distribution", "mifa") for ds in DS}
verdict("MEM1b", "strategic_distribution: |G(MIFA)| < 0.3 pp on each dataset",
        all_ok(*(None if x is None else abs(x[0]) < 0.3 for x in gs.values())), "; ".join(f"{k} {f2(x)}" for k, x in gs.items()))
gs = {ds: gap(ds, "random_30", "mifa") for ds in ("emnist", "cifar100")}
verdict("MEM2", "random_30: G(MIFA) > 0 on EMNIST and CIFAR-100", all_ok(*(None if x is None else x[0] > 0 for x in gs.values())),
        "; ".join(f"{k} {f2(x)}" for k, x in gs.items()))
parts = []
for ds in ("emnist", "cifar100"):
    p = paired(seeds(ds, "combined_strategic", "mifa"), seeds(ds, "combined_strategic", "mem10"))
    parts.append(f"{ds} MIFA − mem10 {f2(p)}")
rows.append(("Q1", "No prediction: staleness trade-off, MIFA vs 10-round cap under combined", "REPORTED", "; ".join(parts)))
g = gap("emnist", "combined_strategic")
verdict("V10", "EMNIST combined: |G(v2)| < 0.3 pp (v2 near-inert, predicted from the v1 replay)",
        None if g is None else abs(g[0]) < 0.3, f2(g))


def worst(ds, atk, s, **kw):
    c = cell(ds, atk, "fedavg", s, **kw)
    if c is None:
        return None
    k = len(c["class_accuracies"][0])
    return 100 * min(st.mean(r[j] for r in c["class_accuracies"][-10:]) for j in range(k))


parts, ok = [], []
for ds in DS:
    hw = {}
    for a in ("random_30", "strategic_distribution"):
        v = [None if worst(ds, "no_attack", s) is None or worst(ds, a, s) is None else worst(ds, "no_attack", s) - worst(ds, a, s)
             for s in SEEDS]
        hw[a] = None if None in v else st.mean(v)
    if None in hw.values():
        ok.append(None)
        continue
    ok.append(bool(hw["strategic_distribution"] > hw["random_30"]))
    parts.append(f"{ds} worst-class harm dist {hw['strategic_distribution']:.2f} vs random {hw['random_30']:.2f}")
verdict("WC1", "Worst-class harm: strategic_distribution > random_30 on EMNIST, CIFAR-100, CIFAR-10", all_ok(*ok),
        "; ".join(parts) or "pending")

hs = {a: harm("har_subject", a) for a in ATK[1:]}
verdict("HAR1", "HAR natural subjects: FedAvg harm < 0.5 pp under every attack",
        all_ok(*(None if x is None else x[0] < 0.5 for x in hs.values())), "; ".join(f"{k} {f2(x)}" for k, x in hs.items()))
a_, b_ = harm("har_dir", "strategic_distribution"), harm("har_subject", "strategic_distribution")
verdict("HAR2", "H(strategic_distribution): HAR Dirichlet α=0.1 > HAR natural subjects",
        None if a_ is None or b_ is None else a_[0] > b_[0], f"Dirichlet {f2(a_)} vs subjects {f2(b_)}")

for ds in DS + HAR:
    for a in ATK[1:]:
        for arm in ("mifa", "mem10"):
            g = gap(ds, a, arm)
            if g and g[0] < -1:
                surprises.append(f"{arm} below FedAvg by > 1 pp: {ds}/{a} ({g[0]:+.2f})")

for ds in ("emnist", "cifar100"):
    for a in STRAT:
        g = gap(ds, a)
        if g and g[0] < -0.5:
            surprises.append(f"v2 below FedAvg by > 0.5 pp: {ds}/{a} ({g[0]:+.2f})")
for ds in DS:
    g = gap(ds, "random_30")
    if g and g[0] < -0.5:
        surprises.append(f"v2 costs > 0.5 pp under random_30: {ds} ({g[0]:+.2f})")
for k_, c in sorted(C.items()):
    if c.get("diverged_at") is not None:
        surprises.append(f"diverged (non-finite weights) at round {c['diverged_at']}: {k_}")
nondet = sum(1 for c in C.values() if not c.get("deterministic"))
if nondet:
    surprises.append(f"G1: {nondet} cells ran nondeterministic")


def table(ds, metric):
    out = ["| Attack | " + " | ".join(AGG) + " |", "|---|" + "---|" * len(AGG)]
    for a in ATK:
        cells_ = []
        for agg in AGG:
            v = seeds(ds, a, agg, metric)
            cells_.append("—" if v is None else f"{st.mean(v):.2f} ± {st.stdev(v):.2f}")
        out.append(f"| {a} | " + " | ".join(cells_) + " |")
    return "\n".join(out)


lines = ["# Paper A v2: results scored against PREREG-v2", "",
         f"Generated by `analyze_v2.py` from {len(C)} of 650 completed cells. Primary metric: mean test accuracy "
         "over rounds 91–100, %. CI: paired t over 5 seeds (df 4). Thresholds as in `PREREG-v2.md`.", "",
         "## Verdicts", "", "| ID | Pre-registered claim | Verdict | Evidence |", "|---|---|---|---|",
         *[f"| {a} | {b.replace('|', chr(92) + '|')} | **{c}** | {e.replace('|', chr(92) + '|')} |" for a, b, c, e in rows], "",
         "## Surprises (report, don't fix)", "", *(["- " + s for s in surprises] or ["- None triggered."]), ""]
for ds in DS + HAR:
    lines += [f"## {ds}, block {'H' if ds in HAR else 'A'} (mean ± sd over 5 seeds; last-10 / round 100)", "", table(ds, "last10"), "",
              table(ds, "final"), ""]
MARK = "<!-- hand-written below: analyze_v2.py keeps everything after this line -->"
prev = open(OUT).read() if os.path.exists(OUT) else ""
open(OUT, "w").write("\n".join(lines) + "\n" + MARK + (prev.split(MARK, 1)[1] if MARK in prev else "\n"))
print("\n".join(lines[: 9 + len(rows)]))
print(f"\n{len(C)}/650 cells; wrote {OUT}")
