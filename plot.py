#!/usr/bin/env python3
"""Figures for the v1 (pre-registered) journal grid. The run failed its PREREG and stays in the
paper as a reported result (director, 2026-11-06). Reads the read-only v1 results; writes
figures/v1/*.pdf (paper) and *.png (preview).

Usage: <venv>/bin/python plot.py   (needs matplotlib + numpy; the ic3ai-2026 .venv has both)
"""
import json, os, statistics as st
from math import sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = os.path.join(HERE, "hpc-package", "results", "journal")
OUT = os.path.join(HERE, "figures", "v1")
SEEDS = (42, 123, 7)
T975_DF2 = 4.303
TWIN_GAP = None  # set below from the two identical-config cells
# Validated categorical slots 1-2 (dataviz reference palette, light): series never colour-only,
# FedAvg solid/circle, AAWA dashed/square, so the figures survive grayscale print.
FED, AAWA, INK, MUTED, BAND = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {"fedavg": dict(color=FED, ls="-", marker="o", label="FedAvg"),
         "aawa": dict(color=AAWA, ls="--", marker="s", label="AAWA")}
ATK_LABEL = {"no_attack": "No attack", "random_30": "Random 30%", "strategic_gradient": "Strategic\n(gradient)",
             "strategic_distribution": "Strategic\n(distribution)", "combined_strategic": "Strategic\n(combined)"}
IC3AI = {"no_attack": -0.26, "random_30": 0.31, "combined_strategic": 2.82}  # Run 4, 1 seed (PREREG)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": BAND, "grid.linewidth": 0.6,
                     "lines.linewidth": 1.2, "pdf.fonttype": 42, "savefig.bbox": "tight"})


def load(key):
    return json.load(open(os.path.join(J, key + ".json")))


def final(ph, ds, atk, agg, s, extra=""):
    return load(f"p{ph}__{ds}__{atk}__{agg}__seed{s}{extra}")["final_accuracy"] * 100


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def fig_curves():
    """Accuracy over rounds, mean of 3 seeds with min-max band."""
    rows = (("cifar10", 1, "CIFAR-10"), ("emnist", 3, "EMNIST-balanced"))
    atks = ("no_attack", "random_30", "combined_strategic")
    fig, ax = plt.subplots(2, 3, figsize=(7.16, 3.6), sharex=True, sharey="row", layout="constrained")
    for r, (ds, ph, dname) in enumerate(rows):
        for c, atk in enumerate(atks):
            a = ax[r, c]
            for agg in ("fedavg", "aawa"):
                m = np.array([load(f"p{ph}__{ds}__{atk}__{agg}__seed{s}")["accuracies"] for s in SEEDS]) * 100
                x = np.arange(1, m.shape[1] + 1)
                a.fill_between(x, m.min(0), m.max(0), color=STYLE[agg]["color"], alpha=0.15, lw=0)
                a.plot(x, m.mean(0), color=STYLE[agg]["color"], ls=STYLE[agg]["ls"], label=STYLE[agg]["label"])
            a.grid(axis="y")
            if r == 0:
                a.set_title(ATK_LABEL[atk].replace(chr(10), " "), color=INK)
            if c == 0:
                a.set_ylabel(f"{dname}\nTest accuracy (%)")
            if r == 1:
                a.set_xlabel("Round")
        ax[r, 0].set_ylim(30 if ds == "cifar10" else 60, None)
    ax[0, 0].legend(frameon=False, loc="lower right")
    fig.suptitle("FedAvg and AAWA track each other under every attack (mean of 3 seeds, band = min–max)",
                 x=0.01, ha="left", fontsize=8, color=INK)
    save(fig, "fig1_curves")


def seed_panel(a, xs, vals, labels, color, title, ylabel, refs=None, band=None):
    """Per-seed dots, mean tick, paired 95% CI; optional IC3AI reference and noise band."""
    if band:
        a.axhspan(-band, band, color=BAND, lw=0, zorder=0)
    a.axhline(0, color=MUTED, lw=0.8, zorder=1)
    for x, v in zip(xs, vals):
        m, h = st.mean(v), T975_DF2 * st.stdev(v) / sqrt(len(v))
        a.errorbar(x, m, yerr=h, color=color, capsize=3, lw=1.2, zorder=3)
        a.plot([x - 0.18, x + 0.18], [m, m], color=color, lw=2, zorder=3)
        a.scatter([x + d for d in (-0.08, 0, 0.08)], v, s=14, color=color, edgecolor="white", lw=0.6, zorder=4)
    if refs:
        rx = [x for x, k in zip(xs, labels) if k in refs]
        a.scatter([x + 0.3 for x in rx], [refs[k] for k in labels if k in refs], marker="D", s=22,
                  facecolor="white", edgecolor=INK, lw=1, zorder=5, label="IC3AI (1 run)")
    a.set_xticks(xs, [ATK_LABEL[k] for k in labels])
    a.set_title(title, color=INK)
    a.set_ylabel(ylabel)
    a.grid(axis="y")


def fig_attack_cost():
    """How much each attack costs FedAvg, per seed, against the no-attack cell of the same seed."""
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.5), sharey=True, gridspec_kw={"width_ratios": [4, 2]}, layout="constrained")
    for a, (ds, ph, atks, name) in zip(ax, (
            ("cifar10", 1, ("random_30", "strategic_gradient", "strategic_distribution", "combined_strategic"), "CIFAR-10"),
            ("emnist", 3, ("random_30", "combined_strategic"), "EMNIST-balanced"))):
        vals = [[final(ph, ds, atk, "fedavg", s) - final(ph, ds, "no_attack", "fedavg", s) for s in SEEDS] for atk in atks]
        seed_panel(a, range(len(atks)), vals, atks, FED, name, "FedAvg change vs no attack (pp)" if ph == 1 else "")
    fig.suptitle("Strategic suppression costs FedAvg <1 pp on CIFAR-10 but 2.2 pp on EMNIST "
                 "(dots = seeds, bar = mean, whisker = 95% CI)", x=0.01, ha="left", fontsize=8, color=INK)
    save(fig, "fig2_attack_cost")


def fig_gap():
    """The pre-registered test: AAWA minus FedAvg at round 100, paired by seed."""
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True, gridspec_kw={"width_ratios": [5, 3]}, layout="constrained")
    for a, (ds, ph, atks, name) in zip(ax, (
            ("cifar10", 1, ("no_attack", "random_30", "strategic_gradient", "strategic_distribution", "combined_strategic"), "CIFAR-10"),
            ("emnist", 3, ("no_attack", "random_30", "combined_strategic"), "EMNIST-balanced"))):
        vals = [[final(ph, ds, atk, "aawa", s) - final(ph, ds, atk, "fedavg", s) for s in SEEDS] for atk in atks]
        seed_panel(a, range(len(atks)), vals, atks, AAWA, name, "AAWA − FedAvg at round 100 (pp)" if ph == 1 else "",
                   refs=IC3AI if ds == "cifar10" else None, band=TWIN_GAP)
    ax[0].axhline(1.5, color=INK, lw=0.8, ls=":")
    ax[0].text(-0.2, 1.6, "PREREG E1.1 target: ≥ +1.5", ha="left", va="bottom", fontsize=6.5, color=INK)
    ax[0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"AAWA does not beat FedAvg (grey band = ±{TWIN_GAP:.2f} pp, the gap between two identical runs)",
                 x=0.01, ha="left", fontsize=8, color=INK)
    save(fig, "fig3_aawa_gap")


def fig_twins():
    a = load("p1__cifar10__combined_strategic__aawa__seed42")["accuracies"]
    b = load("p2__cifar10__combined_strategic__aawa__seed42__W10_tau0.5_update_norm")["accuracies"]
    fig, ax = plt.subplots(figsize=(3.5, 2.2), layout="constrained")
    x = np.arange(1, len(a) + 1)
    ax.plot(x, np.array(a) * 100, color=FED, ls="-", label="Run 1")
    ax.plot(x, np.array(b) * 100, color=AAWA, ls="--", label="Run 2")
    ax.annotate(f"{a[-1]*100:.2f}", (100, a[-1] * 100), xytext=(3, -6), textcoords="offset points", fontsize=6.5, color=INK)
    ax.annotate(f"{b[-1]*100:.2f}", (100, b[-1] * 100), xytext=(3, 2), textcoords="offset points", fontsize=6.5, color=INK)
    ax.set_xlabel("Round"); ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(30, None); ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Same config, same seed, same GPU type:\ncurves split at round 2 (AAWA, combined, seed 42)",
                 loc="left", color=INK)
    save(fig, "fig4_twin_runs")


def fig_sensitivity():
    ws, taus = (5, 10, 20), (0.3, 0.5, 0.7)
    fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.2), sharey=True, layout="constrained")
    grids = {sig: np.array([[final(2, "cifar10", "combined_strategic", "aawa", 42, f"__W{w}_tau{t}_{sig}")
                             for t in taus] for w in ws]) for sig in ("update_norm", "grad_norm")}
    lo, hi = min(g.min() for g in grids.values()), max(g.max() for g in grids.values())
    fed = final(1, "cifar10", "combined_strategic", "fedavg", 42)
    for a, (sig, g) in zip(ax, grids.items()):
        im = a.imshow(g, cmap="Blues", vmin=lo - 2, vmax=hi, aspect="auto")
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{g[i, j]:.1f}", ha="center", va="center", fontsize=7,
                       color="white" if g[i, j] > (lo + hi) / 2 else INK)
        a.set_xticks(range(3), [f"τ={t}" for t in taus]); a.set_yticks(range(3), [f"W={w}" for w in ws])
        a.set_title(f"signal: {sig}", color=INK)
        for s in a.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy (%)")
    fig.suptitle(f"AAWA hyperparameters, combined attack, seed 42 (FedAvg same seed: {fed:.2f})",
                 x=0.01, ha="left", fontsize=8, color=INK)
    save(fig, "fig5_sensitivity")


def fig_dp():
    atks = ("strategic_gradient", "strategic_distribution", "combined_strategic")
    eps = (None, 10.0, 1.0, 0.1)
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.3), sharey=True, layout="constrained")
    for a, atk in zip(ax, atks):
        for i, e in enumerate(eps):
            v = [final(1, "cifar10", atk, "aawa", s) if e is None else final(4, "cifar10", atk, "aawa", s, f"__eps{e}")
                 for s in SEEDS]
            a.plot([i - 0.15, i + 0.15], [st.mean(v)] * 2, color=AAWA, lw=2)
            a.scatter([i - 0.06, i, 0.06 + i], v, s=14, color=AAWA, edgecolor="white", lw=0.6, zorder=3)
        f = [final(1, "cifar10", atk, "fedavg", s) for s in SEEDS]
        a.axhline(st.mean(f), color=FED, lw=1, ls="-")
        a.text(3.4, st.mean(f), "FedAvg", color=INK, fontsize=6.5, va="bottom", ha="right")
        a.set_xticks(range(4), ["no DP", "ε=10", "ε=1", "ε=0.1"]); a.grid(axis="y")
        a.set_title(ATK_LABEL[atk].replace("\n", " "), color=INK)
    ax[0].set_ylabel("AAWA accuracy, round 100 (%)")
    fig.suptitle("Noising AAWA's class histograms changes nothing measurable (dots = seeds, bar = mean)",
                 x=0.01, ha="left", fontsize=8, color=INK)
    save(fig, "fig6_dp")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    TWIN_GAP = abs(final(1, "cifar10", "combined_strategic", "aawa", 42)
                   - final(2, "cifar10", "combined_strategic", "aawa", 42, "__W10_tau0.5_update_norm"))
    assert abs(TWIN_GAP - 2.21) < 0.01, TWIN_GAP  # matches ANALYSIS.md
    for f in (fig_curves, fig_attack_cost, fig_gap, fig_twins, fig_sensitivity, fig_dp):
        f()
