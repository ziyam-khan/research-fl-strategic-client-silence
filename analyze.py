#!/usr/bin/env python3
"""Score the journal grid against PREREG.md (approved 2026-11-06). Thresholds are copied
from PREREG and must not be edited after results are seen; amend PREREG instead.

Usage: python3 analyze.py [results_dir]   (default: hpc-package/results/journal)
Writes ANALYSIS.md next to this file. Stdlib only. Missing cells are reported as PENDING.
Metric: test accuracy at the final round (round 100), in percentage points.
"""
import json, sys, glob, os, statistics as st
from math import sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "hpc-package", "results", "journal")
T975_DF2 = 4.303  # two-sided 95% t critical value, 3 seeds -> df 2
SEEDS = (42, 123, 7)
STRAT = ("strategic_gradient", "strategic_distribution", "combined_strategic")
ATTACKS = ("no_attack", "random_30") + STRAT
AGGS = ("fedavg", "median", "trimmed_mean", "krum", "aawa")
KRUM_CHANCE_MAX = 15.0  # "near chance (~10%)": 10 classes, read as <= 15%

cells = {}
for f in glob.glob(os.path.join(RES, "*.json")):
    d = json.load(open(f))
    if not d.get("done"):
        continue
    a = d["final_accuracy"]
    cells[os.path.basename(f)[:-5]] = a * 100 if a <= 1 else a


def acc(phase, ds, atk, agg, seed, extra=""):
    return cells.get(f"p{phase}__{ds}__{atk}__{agg}__seed{seed}{extra}")


def seeds(phase, ds, atk, agg, extra=""):
    v = [acc(phase, ds, atk, agg, s, extra) for s in SEEDS]
    return None if None in v else v


def mean(v):
    return None if v is None else st.mean(v)


def paired(phase, ds, atk, a="aawa", b="fedavg", extra_a="", extra_b=""):
    x, y = seeds(phase, ds, atk, a, extra_a), seeds(phase, ds, atk, b, extra_b)
    if x is None or y is None:
        return None
    d = [p - q for p, q in zip(x, y)]
    m, sd = st.mean(d), st.stdev(d)
    h = T975_DF2 * sd / sqrt(len(d))
    return m, m - h, m + h, d


rows, surprises = [], []


def verdict(eid, claim, ok, detail):
    rows.append((eid, claim, "PENDING" if ok is None else ("HOLDS" if ok else "FAILS"), detail))


f2 = lambda x: "—" if x is None else f"{x:+.2f}"

# ---- Phase 1: CIFAR-10 ----
p = paired(1, "cifar10", "combined_strategic")
verdict("E1.1", "combined_strategic: AAWA−FedAvg ≥ 1.5 pp, 95% CI excludes 0",
        None if p is None else (p[0] >= 1.5 and p[1] > 0),
        "pending" if p is None else f"mean {p[0]:+.2f} pp, CI [{p[1]:+.2f}, {p[2]:+.2f}], per seed {[round(x, 2) for x in p[3]]}")
if p and p[0] < 0:
    surprises.append(f"AAWA loses to FedAvg under combined_strategic on CIFAR-10 ({p[0]:+.2f} pp)")

p = paired(1, "cifar10", "random_30")
verdict("E1.2", "random_30: |AAWA−FedAvg| ≤ 1 pp", None if p is None else abs(p[0]) <= 1,
        "pending" if p is None else f"mean {p[0]:+.2f} pp, CI [{p[1]:+.2f}, {p[2]:+.2f}]")

p = paired(1, "cifar10", "no_attack")
verdict("E1.3", "no_attack: |AAWA−FedAvg| < 0.5 pp", None if p is None else abs(p[0]) < 0.5,
        "pending" if p is None else f"mean {p[0]:+.2f} pp, CI [{p[1]:+.2f}, {p[2]:+.2f}]")

fc, fr = seeds(1, "cifar10", "combined_strategic", "fedavg"), seeds(1, "cifar10", "random_30", "fedavg")
if fc and fr:
    per = [c < r for c, r in zip(fc, fr)]
    ok = st.mean(fc) < st.mean(fr)
    verdict("E1.4", "Duration flip: FedAvg lower under strategic than random_30",
            ok, f"FedAvg combined {st.mean(fc):.2f} vs random {st.mean(fr):.2f}; holds in {sum(per)}/3 seeds")
    if not ok:
        surprises.append("Duration flip disappears across seeds")
else:
    verdict("E1.4", "Duration flip: FedAvg lower under strategic than random_30", None, "pending")

parts, ok5 = [], True
for atk in ATTACKS:
    fa = mean(seeds(1, "cifar10", atk, "fedavg"))
    for agg in ("median", "trimmed_mean"):
        v = mean(seeds(1, "cifar10", atk, agg))
        if fa is None or v is None:
            ok5 = None if ok5 is not False else False
            continue
        if v > fa - 8:
            ok5 = False
            parts.append(f"{agg}/{atk} only {fa - v:.1f} below")
    k = mean(seeds(1, "cifar10", atk, "krum"))
    if k is not None and k > KRUM_CHANCE_MAX:
        ok5 = False
        parts.append(f"krum/{atk} {k:.1f}%")
        surprises.append(f"Krum above chance under {atk} ({k:.1f}%)")
verdict("E1.5", "Median, trimmed mean ≥ 8 pp below FedAvg everywhere; Krum ≤ 15%",
        ok5, "; ".join(parts) or ("all attacks" if ok5 else "pending"))

parts, ok6 = [], True
for atk in STRAT:
    a, b = seeds(1, "cifar10", atk, "aawa"), seeds(1, "cifar10", atk, "fedavg")
    if a is None or b is None:
        ok6 = None if ok6 is not False else False
        continue
    sa, sb = st.stdev(a), st.stdev(b)
    ok6 = ok6 is not False and (ok6 and sa < sb)
    parts.append(f"{atk}: AAWA sd {sa:.2f} vs FedAvg {sb:.2f}")
verdict("E1.6", "AAWA seed std < FedAvg under each strategic attack", ok6, "; ".join(parts) or "pending")

fc_m = mean(fc)
parts, okn = [], True
for atk in ("strategic_gradient", "strategic_distribution"):
    v = mean(seeds(1, "cifar10", atk, "fedavg"))
    if v is None or fc_m is None:
        okn = None
        continue
    okn = okn is not False and okn and v > fc_m
    parts.append(f"{atk} FedAvg {v:.2f} vs combined {fc_m:.2f}")
verdict("E1.new", "Each single strategic attack weaker than combined (FedAvg)", okn, "; ".join(parts) or "pending")

# ---- Phase 2: ablation, combined_strategic, seed 42 ----
abl = {}
for sig in ("update_norm", "grad_norm"):
    abl[sig] = [cells.get(f"p2__cifar10__combined_strategic__aawa__seed42__W{w}_tau{t}_{sig}")
                for w in (5, 10, 20) for t in (0.3, 0.5, 0.7)]
u, g = abl["update_norm"], abl["grad_norm"]
if None not in u:
    rng = max(u) - min(u)
    verdict("E2.1", "update_norm: accuracy range over 9 (W, τ) < 2.5 pp", rng < 2.5,
            f"range {rng:.2f} pp ({min(u):.2f}–{max(u):.2f})" + (f"; grad_norm range {max(g) - min(g):.2f}" if None not in g else ""))
    if rng > 5:
        surprises.append(f"Sensitivity range > 5 pp ({rng:.2f})")
else:
    verdict("E2.1", "update_norm: accuracy range over 9 (W, τ) < 2.5 pp", None, "pending")
verdict("E2.2", "mean(update_norm) ≥ mean(grad_norm) over 9 configs",
        None if None in u + g else st.mean(u) >= st.mean(g),
        "pending" if None in u + g else f"{st.mean(u):.2f} vs {st.mean(g):.2f} — if FAILS, grad_norm becomes the main-table signal")

# ---- Phase 3: EMNIST ----
p = paired(3, "emnist", "combined_strategic")
pc = paired(1, "cifar10", "combined_strategic")
verdict("E3.1", "EMNIST combined_strategic: AAWA ≥ FedAvg (size not predicted)",
        None if p is None else p[0] >= 0,
        "pending" if p is None else f"mean {p[0]:+.2f} pp, CI [{p[1]:+.2f}, {p[2]:+.2f}]; CIFAR gap {f2(pc[0] if pc else None)}")
if p and p[0] < 0:
    surprises.append(f"AAWA loses to FedAvg under combined_strategic on EMNIST ({p[0]:+.2f} pp)")
p = paired(3, "emnist", "no_attack")
verdict("E3.2", "EMNIST no_attack: |AAWA−FedAvg| < 0.5 pp", None if p is None else abs(p[0]) < 0.5,
        "pending" if p is None else f"mean {p[0]:+.2f} pp, CI [{p[1]:+.2f}, {p[2]:+.2f}]")

# ---- Phase 4: DP-noised histograms vs non-private AAWA (Phase 1) ----
dp = {}
for atk in STRAT:
    for eps in (10.0, 1.0, 0.1):
        x, y = seeds(4, "cifar10", atk, "aawa", f"__eps{eps}"), seeds(1, "cifar10", atk, "aawa")
        dp[atk, eps] = None if x is None or y is None else (st.mean(x) - st.mean(y), x, y)

parts, ok = [], True
for atk in STRAT:
    for eps in (10.0, 1.0):
        r = dp[atk, eps]
        if r is None:
            ok = None if ok is not False else False
            continue
        ok = ok is not False and ok and abs(r[0]) <= 0.5
        parts.append(f"{atk} ε={eps:g}: {r[0]:+.2f}")
        if r[0] > 0.5:
            surprises.append(f"DP improves AAWA by > 0.5 pp ({atk}, ε={eps:g}, {r[0]:+.2f})")
verdict("E4.1", "ε=10 and ε=1: AAWA within ±0.5 pp of non-private AAWA", ok, "; ".join(parts) or "pending")

parts, ok = [], True
fed_cs = mean(seeds(1, "cifar10", "combined_strategic", "fedavg"))
for atk in STRAT:
    r = dp[atk, 0.1]
    if r is None:
        ok = None if ok is not False else False
        continue
    ok = ok is not False and ok and r[0] >= -1.5
    parts.append(f"{atk}: {r[0]:+.2f}")
    if r[0] > 0.5:
        surprises.append(f"DP improves AAWA by > 0.5 pp ({atk}, ε=0.1, {r[0]:+.2f})")
r = dp["combined_strategic", 0.1]
if r is not None and fed_cs is not None:
    beat = st.mean(r[1]) > fed_cs
    ok = ok is not False and ok and beat
    parts.append(f"combined AAWA ε=0.1 {st.mean(r[1]):.2f} vs FedAvg {fed_cs:.2f}")
verdict("E4.2", "ε=0.1: loss ≤ 1.5 pp and still beats FedAvg under combined", ok, "; ".join(parts) or "pending")

for atk in STRAT:
    for eps in (10.0, 1.0, 0.1):
        r = dp[atk, eps]
        if r is not None and fed_cs is not None and atk == "combined_strategic" and st.mean(r[1]) < fed_cs:
            surprises.append(f"AAWA with DP falls below FedAvg (combined, ε={eps:g})")

parts, ok = [], True
for atk in STRAT:
    ms = [None if dp[atk, e] is None else st.mean(dp[atk, e][1]) for e in (10.0, 1.0, 0.1)]
    if None in ms:
        ok = None if ok is not False else False
        continue
    # "allowing for seed noise": a rise of accuracy at lower ε counts as a violation only
    # if it exceeds the larger seed sd of the two ε cells being compared.
    sds = [st.stdev(dp[atk, e][1]) for e in (10.0, 1.0, 0.1)]
    viol = [ms[i + 1] - ms[i] for i in range(2) if ms[i + 1] - ms[i] > max(sds[i], sds[i + 1])]
    ok = ok is not False and ok and not viol
    parts.append(f"{atk}: {ms[0]:.2f} / {ms[1]:.2f} / {ms[2]:.2f}" + (f" (rises {max(viol):+.2f}, beyond seed sd)" if viol else ""))
verdict("E4.3", "Accuracy falls as ε falls (10 ≥ 1 ≥ 0.1)", ok, "; ".join(parts) or "pending")

# ---- Tables ----
def table(phase, ds, attacks):
    out = ["| Attack | " + " | ".join(AGGS) + " |", "|---|" + "---|" * len(AGGS)]
    for atk in attacks:
        r = []
        for agg in AGGS:
            v = seeds(phase, ds, atk, agg)
            r.append("—" if v is None else f"{st.mean(v):.2f} ± {st.stdev(v):.2f}")
        out.append(f"| {atk} | " + " | ".join(r) + " |")
    return "\n".join(out)


n_done = len(cells)
lines = [
    "# Paper A — results scored against PREREG",
    "",
    f"Generated by `analyze.py` from {n_done} of 165 completed cells. Metric: test accuracy at round 100, %. "
    "Thresholds are those in `PREREG.md` (approved 2026-11-06). CI: paired t over 3 seeds (df 2).",
    "",
    "## Verdicts",
    "",
    "| ID | Pre-registered claim | Verdict | Evidence |",
    "|---|---|---|---|",
    *[f"| {a} | {b} | **{c}** | {d} |" for a, b, c, d in rows],
    "",
    "## Surprises (report, don't fix)",
    "",
    *(["- " + s for s in surprises] or ["- None triggered."]),
    "",
    "## CIFAR-10, Phase 1 (mean ± sd over 3 seeds)",
    "",
    table(1, "cifar10", ATTACKS),
    "",
    "## EMNIST-balanced, Phase 3",
    "",
    table(3, "emnist", ("no_attack", "random_30", "combined_strategic")),
    "",
]
out = os.path.join(HERE, "ANALYSIS.md")
MARK = "<!-- hand-written below: analyze.py keeps everything after this line -->"
prev = open(out).read() if os.path.exists(out) else ""
open(out, "w").write("\n".join(lines) + "\n" + MARK + (prev.split(MARK, 1)[1] if MARK in prev else "\n"))
print("\n".join(lines[:9 + len(rows)]))
print(f"\n{n_done}/165 cells; wrote {out}")
