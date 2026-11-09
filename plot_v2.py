#!/usr/bin/env python3
"""Figures for the v2 pre-registered grid (650 cells). Reads the read-only results via explore_v2;
writes figures/v2/*.pdf (paper, vector) and *.png (preview). v1 figures stay in figures/v1 (plot.py).

Usage: ../ic3ai-2026/.venv/bin/python plot_v2.py      (matplotlib + numpy)

Encoding rules (dataviz skill): colour follows the entity in every figure; every series also has a
marker or line style and a direct label, so nothing is colour-only and the figures survive grayscale.
Method slots 1-5 and attack slots 6-8 of the validated reference palette (validator run 2026-11-09).
"""
import os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np
import explore_v2 as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "v2")
INK, MUTED, BAND, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
METHOD = {  # entity -> (colour, marker, line style); AAWA v1 keeps its v1-figure orange
    "fedavg": ("#2a78d6", "o", "-"), "aawa": ("#eb6834", "s", "--"), "mifa": ("#1baf7a", "^", "-"),
    "mem10": ("#eda100", "v", ":"), "aawa_v2": ("#e87ba4", "D", "-."),
}
ATTACK = {  # random is the benign reference: neutral ink, dashed
    "random_30": (MUTED, "o", "--"), "strategic_gradient": ("#008300", "^", "-"),
    "strategic_distribution": ("#6250d6", "s", "-"), "combined_strategic": ("#e34948", "D", "-"),
}
DIVERGE = LinearSegmentedColormap.from_list("rd_gy_bu", ["#a82f2e", "#e34948", "#f0efec", "#2a78d6", "#184f95"])
C = E.load_all()

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": BAND, "grid.linewidth": 0.6,
                     "lines.linewidth": 1.2, "pdf.fonttype": 42, "savefig.bbox": "tight",
                     "figure.facecolor": "white", "axes.titleweight": "bold"})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def fig01_lineage():
    """Flow diagram: how the study got here (conference claim -> failed pre-registration -> v2)."""
    steps = [("IC3AI 2026", "conference paper\nAAWA +2.82 pp\n1 seed, CIFAR-10"),
             ("v1 prereg.", "3 seeds\n10 of 14 claims fail\nAAWA gain: none"),
             ("Diagnosis", "reweighting cannot\nrestore unsent classes\n+ GPU noise 2.2 pp"),
             ("v2 prereg.", "650 cells, 5 seeds\ndeterministic GPU\n+MIFA, Mem-10, HAR"),
             ("Verdicts", "21 hold, 7 fail\nattack effects hold\ndefences inert or flip")]
    fig, ax = plt.subplots(figsize=(7.16, 1.5))
    ax.set_xlim(-0.1, 10.15); ax.set_ylim(0, 1.6); ax.axis("off")
    w, gap = 1.8, 0.25
    for i, (title, body) in enumerate(steps):
        x = 0.05 + i * (w + gap)
        fail = i in (1, 2)
        ax.add_patch(FancyBboxPatch((x, 0.08), w, 1.42, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#f6f5f2" if not fail else "#fbeeee", ec=MUTED if not fail else "#d03b3b", lw=0.8))
        ax.text(x + w / 2, 1.3, title, ha="center", va="center", fontsize=7.5, weight="bold", color=INK)
        ax.text(x + w / 2, 0.66, body, ha="center", va="center", fontsize=6.0, color=MUTED, linespacing=1.35)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.04, 0.79), xytext=(x + w + 0.04, 0.79),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
    save(fig, "fig01_lineage")


def fig02_participation():
    """Raster: who trains each round (EMNIST, seed 1). Three absence regimes."""
    atks = ("random_30", "strategic_gradient", "strategic_distribution", "combined_strategic")
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.75), sharey=True, layout="constrained")
    for ax, atk in zip(axs, atks):
        d = E.cell(C, "emnist", atk, "fedavg", 1)
        M = np.zeros((20, 100))
        for t, parts in enumerate(d["participants"]):
            for c in range(20):
                M[c, t] = 0 if c in parts else 1
        ax.imshow(M, aspect="auto", interpolation="nearest", cmap=LinearSegmentedColormap.from_list("p", [BAND, "#184f95"]),
                  extent=(0.5, 100.5, 19.5, -0.5))
        ax.set_title(E.ATK_LABEL[atk], color=ATTACK[atk][0] if atk != "random_30" else INK)
        ax.set_xlabel("Round"); ax.set_xticks([1, 50, 100])
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("Client"); axs[0].set_yticks([0, 10, 19])
    fig.text(1.0, 0.5, "dark = absent", rotation=90, va="center", ha="left", fontsize=7, color=MUTED)
    save(fig, "fig02_participation")


def fig03_harm_forest():
    """Forest plot: FedAvg accuracy lost to each attack, every dataset (95% CI, 5 seeds)."""
    fig, ax = plt.subplots(figsize=(3.5, 4.3))
    y, ticks, labels = 0, [], []
    base = {ds: E.acc(C, ds, "no_attack", "fedavg") for ds in E.DS}
    for ds in E.DS:
        ax.text(-5.6, y - 0.15, E.DS_LABEL[ds], weight="bold", fontsize=7.5, va="bottom", color=INK)
        y += 1
        for atk in E.ATTACK_ORDER if hasattr(E, "ATTACK_ORDER") else E.ATK:
            r = E.paired(base[ds], E.acc(C, ds, atk, "fedavg"))
            col, mk, _ = ATTACK[atk]
            sig = r["lo"] > 0 or r["hi"] < 0
            ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=1.1, solid_capstyle="round")
            ax.plot(r["mean"], y, mk, ms=5, color=col, mfc=col if sig else "white", mew=1.1)
            ticks.append(y); labels.append(E.ATK_LABEL[atk]); y += 1
        y += 0.6
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(ticks, labels); ax.invert_yaxis(); ax.set_xlim(-5.6, 18)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Accuracy lost vs no attack (pp), FedAvg")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.text(0.99, 1.0, "filled: 95% CI excludes 0", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=MUTED)
    save(fig, "fig03_harm_forest")


def fig04_dose():
    """Dose-response: harm grows with the suppressed fraction, faster for strategic attacks (EMNIST)."""
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    base = E.acc(C, "emnist", "no_attack", "fedavg")
    for atk in ("random_30", "strategic_distribution", "combined_strategic"):
        col, mk, ls = ATTACK[atk]
        xs, ms, lo, hi = [], [], [], []
        for f in (0.1, 0.3, 0.5):
            r = E.paired(base, E.acc(C, "emnist", atk, "fedavg", frac=f))
            xs.append(f); ms.append(r["mean"]); lo.append(r["mean"] - r["lo"]); hi.append(r["hi"] - r["mean"])
        ax.errorbar(xs, ms, yerr=[lo, hi], color=col, marker=mk, ls=ls, ms=5, capsize=2, lw=1.2)
        ax.text(0.515, ms[-1], E.ATK_LABEL[atk], color=INK, va="center", fontsize=7)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xticks([0.1, 0.3, 0.5], ["10%", "30%", "50%"]); ax.set_xlim(0.07, 0.62)
    ax.set_xlabel("Clients suppressed per round"); ax.set_ylabel("Accuracy lost (pp), EMNIST FedAvg")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    save(fig, "fig04_dose")


def fig05_class_scatter():
    """Scatter: the more of a class the absent clients hold, the more that class loses (M3)."""
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.15), sharey=False, layout="constrained")
    for ax, ds in zip(axs, ("emnist", "cifar100", "cifar10")):
        for atk in ("strategic_distribution", "combined_strategic"):
            pts = E.class_points(C, ds, atk)
            col, mk, _ = ATTACK[atk]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=7, marker=mk, color=col, alpha=0.45, lw=0,
                       label=E.ATK_LABEL[atk])
        allp = E.class_points(C, ds, "strategic_distribution") + E.class_points(C, ds, "combined_strategic")
        rho = spearman([p[0] for p in allp], [p[1] for p in allp])
        ax.set_title(E.DS_LABEL[ds]); ax.axhline(0, color=MUTED, lw=0.6)
        ax.text(0.03, 0.95, f"Spearman ρ = {rho:+.2f}\nn = {len(allp)} class-seeds", transform=ax.transAxes,
                va="top", fontsize=6.8, color=INK)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True); ax.set_axisbelow(True)
    axs[0].set_ylabel("Class accuracy lost (pp)")
    fig.supxlabel("Share of the class held by absent clients (strategic attacks, FedAvg)", fontsize=8)
    axs[2].legend(loc="lower right", frameon=False, markerscale=1.6, handletextpad=0.2)
    save(fig, "fig05_class_scatter")


def spearman(x, y):
    rx, ry = (np.argsort(np.argsort(v)).astype(float) for v in (np.array(x), np.array(y)))
    return float(np.corrcoef(rx, ry)[0, 1])


def fig06_gain_heatmap():
    """Annotated diverging heatmap: every defence vs FedAvg on every dataset x attack."""
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.45), sharey=True, layout="constrained")
    lim = 3.0
    for ax, m in zip(axs, ("aawa", "aawa_v2", "mifa", "mem10")):
        G = np.zeros((5, 4)); S = np.zeros((5, 4), bool)
        for i, ds in enumerate(E.DS):
            for j, atk in enumerate(E.ATK):
                r = E.paired(E.acc(C, ds, atk, m), E.acc(C, ds, atk, "fedavg"))
                G[i, j] = r["mean"]; S[i, j] = r["lo"] > 0 or r["hi"] < 0
        im = ax.imshow(np.clip(G, -lim, lim), cmap=DIVERGE, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(5):
            for j in range(4):
                ax.text(j, i, f"{G[i, j]:+.1f}", ha="center", va="center", fontsize=6.6,
                        color="white" if abs(G[i, j]) > 2.0 else INK, weight="bold" if S[i, j] else "normal")
                if S[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.0))
        ax.set_title(E.M_LABEL[m], color=INK)
        ax.set_xticks(range(4), [E.ATK_LABEL[a] for a in E.ATK], rotation=40, ha="right")
        ax.set_yticks(range(5), [E.DS_LABEL[d] for d in E.DS])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.01)
    cb.set_label("Gain over FedAvg (pp)", fontsize=7); cb.outline.set_visible(False)
    fig.text(0.0, -0.02, "Bold and boxed: 95% CI excludes 0 (paired t, 5 seeds). Colour clipped at ±3 pp.", fontsize=6.5, color=MUTED)
    save(fig, "fig06_gain_heatmap")


def fig07_curves():
    """Learning curves under the combined attack, with the no-attack reference."""
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.35), layout="constrained")
    spec = (("emnist", (76, 85.5)), ("cifar100", (8, 28.5)))
    for ax, (ds, yl) in zip(axs, spec):
        ref = np.array([E.cell(C, ds, "no_attack", "fedavg", s)["accuracies"] for s in E.SEEDS]) * 100
        ax.plot(range(1, 101), ref.mean(0), color=MUTED, ls=(0, (1, 1.5)), lw=1.0)
        ax.text(100, ref.mean(0)[-1] + 0.35, "no attack", fontsize=6.5, color=MUTED, va="bottom", ha="right")
        for m in ("fedavg", "aawa_v2", "mifa", "mem10"):
            A = np.array([E.cell(C, ds, "combined_strategic", m, s)["accuracies"] for s in E.SEEDS]) * 100
            col, mk, ls = METHOD[m]
            mu, sd = A.mean(0), A.std(0, ddof=1)
            ax.fill_between(range(1, 101), mu - sd, mu + sd, color=col, alpha=0.10, lw=0)
            ax.plot(range(1, 101), mu, color=col, ls=ls, marker=mk, markevery=[99], ms=4, label=E.M_LABEL[m])
            if m == "mifa":
                ax.text(101.5, mu[-1], "MIFA", fontsize=6.5, color=INK, va="center")
        ax.set_xlim(1, 110); ax.set_ylim(*yl); ax.set_xticks([1, 25, 50, 75, 100])
        ax.set_title(f"{E.DS_LABEL[ds]}, combined attack"); ax.set_xlabel("Round")
        ax.grid(axis="y"); ax.set_axisbelow(True)
    axs[0].set_ylabel("Test accuracy (%)")
    axs[0].legend(loc="lower right", frameon=False, ncol=2, fontsize=6.5)
    save(fig, "fig07_curves")


def fig08_staleness():
    """ECDF: how old the updates are that memory aggregators reuse (image datasets pooled)."""
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.2), sharey=True, layout="constrained")
    for ax, m in zip(axs, ("mifa", "mem10")):
        for atk in ("random_30", "strategic_gradient", "combined_strategic"):
            ages, share = [], []
            for ds in ("emnist", "cifar100", "cifar10"):
                for s in E.SEEDS:
                    a, sh = E.staleness(E.cell(C, ds, atk, m, s))
                    ages += a; share += sh
            ages = np.sort(ages); y = np.arange(1, len(ages) + 1) / len(ages)
            col, mk, ls = ATTACK[atk]
            ax.step(ages, y, where="post", color=col, ls=ls, lw=1.3,
                    label=f"{E.ATK_LABEL[atk]}: median {int(np.median(ages))} rd, {100 * st.mean(share):.0f}% of weight")
        ax.set_title(f"{E.M_LABEL[m]}: age of reused updates"); ax.set_xlabel("Rounds since the client last trained")
        ax.set_xlim(0, 100 if m == "mifa" else 12); ax.grid(True); ax.set_axisbelow(True)
        ax.legend(loc="lower right", frameon=False, fontsize=6.4)
    axs[0].set_ylabel("Share of reused updates (ECDF)")
    save(fig, "fig08_staleness")


def fig09_flip():
    """Dumbbell: memory helps under random absence and loses that advantage under strategic absence."""
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True, layout="constrained")
    for ax, m in zip(axs, ("mifa", "mem10")):
        for i, ds in enumerate(E.DS):
            vals = {}
            for k, atk in enumerate(("random_30", "strategic_gradient", "combined_strategic")):
                r = E.paired(E.acc(C, ds, atk, m), E.acc(C, ds, atk, "fedavg"))
                yy = i + (k - 1) * 0.2
                col, mk, _ = ATTACK[atk]
                sig = r["lo"] > 0 or r["hi"] < 0
                ax.plot([r["lo"], r["hi"]], [yy, yy], color=col, lw=0.8, alpha=0.8)
                ax.plot(r["mean"], yy, mk, color=col, ms=5, mfc=col if sig else "white", mew=1.1,
                        label=E.ATK_LABEL[atk] if i == 0 else None)
                vals[atk] = (r["mean"], yy)
            ax.plot([vals["random_30"][0], vals["combined_strategic"][0]], [i - 0.2, i + 0.2], color=BAND, lw=1.0, zorder=0)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{E.M_LABEL[m]} gain over FedAvg (pp)"); ax.set_xlim(-9, 12.5)
        ax.grid(axis="x"); ax.set_axisbelow(True)
    axs[0].set_yticks(range(5), [E.DS_LABEL[d] for d in E.DS]); axs[0].invert_yaxis()
    axs[0].tick_params(axis="y", length=0)
    axs[1].legend(loc="upper right", frameon=False, title="filled: CI excludes 0", title_fontsize=6.4)
    save(fig, "fig09_flip")


def fig10_class_box():
    """Box + strip: per-class accuracy loss — strategic absence concentrates harm on a few classes."""
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), layout="constrained")
    rng = np.random.default_rng(0)
    for ax, ds in zip(axs, ("emnist", "cifar100", "cifar10")):
        data = []
        for atk in E.ATK:
            v = []
            for s in E.SEEDS:
                a = E.class_last10(E.cell(C, ds, "no_attack", "fedavg", s))
                b = E.class_last10(E.cell(C, ds, atk, "fedavg", s))
                v += [x - y for x, y in zip(a, b)]
            data.append(v)
        bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=MUTED, lw=0.8),
                        capprops=dict(color=MUTED, lw=0.8), boxprops=dict(lw=0.8, ec=MUTED))
        for k, (patch, atk) in enumerate(zip(bp["boxes"], E.ATK)):
            col, mk, _ = ATTACK[atk]
            patch.set_facecolor(col); patch.set_alpha(0.18)
            ax.scatter(k + 1 + rng.uniform(-0.18, 0.18, len(data[k])), data[k], s=2.5, color=col, alpha=0.5, lw=0, marker=mk)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks(range(1, 5), [E.ATK_LABEL[a] for a in E.ATK], rotation=25)
        ax.set_title(E.DS_LABEL[ds]); ax.grid(axis="y"); ax.set_axisbelow(True)
    axs[0].set_ylabel("Per-class accuracy lost (pp)")
    save(fig, "fig10_class_box")


def fig11_replication():
    """Replication funnel: the conference effect of AAWA shrinks to zero as rigour increases."""
    rows = [("CIFAR-10", "IC3AI, 1 seed", 2.82, None, None),
            ("CIFAR-10", "v1, 3 seeds", -0.56, -2.92, 1.80),
            ("CIFAR-10", "v2, 5 seeds", None, None, None),
            ("EMNIST", "v1, 3 seeds", -0.17, -1.01, 0.67),
            ("EMNIST", "v2, 5 seeds", None, None, None)]
    for i, (ds, _, m, lo, hi) in enumerate(rows):
        if m is None:
            key = "cifar10" if ds == "CIFAR-10" else "emnist"
            r = E.paired(E.acc(C, key, "combined_strategic", "aawa"), E.acc(C, key, "combined_strategic", "fedavg"))
            rows[i] = (ds, rows[i][1], r["mean"], r["lo"], r["hi"])
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    col, mk, _ = METHOD["aawa"]
    for i, (ds, lab, m, lo, hi) in enumerate(rows):
        if lo is not None:
            ax.plot([lo, hi], [i, i], color=col, lw=1.1)
        ax.plot(m, i, mk, color=col, ms=5, mfc="white" if lo is None else col, mew=1.1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(1.5, color=MUTED, lw=0.8, ls=":")
    ax.text(1.45, 1.5, "v1 pre-registered\nthreshold +1.5", fontsize=6.2, color=MUTED, va="center", ha="right")
    ax.set_yticks(range(len(rows)), [f"{ds}: {lab}" for ds, lab, *_ in rows]); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("AAWA v1 − FedAvg, combined attack (pp, 95% CI)"); ax.grid(axis="x"); ax.set_axisbelow(True)
    save(fig, "fig11_replication")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig01_lineage, fig02_participation, fig03_harm_forest, fig04_dose, fig05_class_scatter,
              fig06_gain_heatmap, fig07_curves, fig08_staleness, fig09_flip, fig10_class_box, fig11_replication):
        f()


def fig12_horizon_model():
    """Model diagram (FORMAL-MODEL.md, Prop. 3): error bound vs memory horizon k, benign vs strategic."""
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    k = np.linspace(0, 10, 400)
    kb, ks, D = 1.6, 5.0, 1.0
    worst = np.maximum(D, D * k / ks)                      # eps*max(D, L*eta*B*k), scaled
    benign = 0.12 + (D - 0.12) * np.exp(-k / 0.55)         # omission falls away once k covers benign ages
    ax.axvspan(kb, ks, color="#e8f1fc", lw=0)
    ax.plot(k, worst, color=ATTACK["combined_strategic"][0], lw=1.6, label="strategic: worst-case bound")
    ax.plot(k, benign, color=MUTED, ls="--", lw=1.4, label="benign churn: typical error")
    ax.axhline(D, color=MUTED, lw=0.6, ls=":")
    ax.text(9.9, D + 0.04, "omission floor εD (never seen)", ha="right", va="bottom", fontsize=6.3, color=MUTED)
    ax.text((kb + ks) / 2, 0.42, "free\nrobustness", ha="center", fontsize=6.5, color="#184f95")
    for x, lab in ((0, "FedAvg\nk = 0"), (kb, "k_benign"), (ks, "k* = D/(LηB)")):
        ax.plot([x, x], [0, 0.05], color=INK, lw=0.8)
        ax.text(x, -0.09, lab, ha="center", va="top", fontsize=6.3, color=INK)
    ax.annotate("MIFA\nk → ∞", xy=(9.8, D * 9.8 / ks), xytext=(7.7, 1.75), fontsize=6.3, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    ax.set_xlim(-0.3, 10); ax.set_ylim(0, 2.1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Memory horizon k (rounds)", labelpad=18); ax.set_ylabel("Aggregation error")
    ax.legend(loc="upper left", frameon=False, fontsize=6.3)
    save(fig, "fig12_horizon_model")


if __name__ == "__main__":
    fig12_horizon_model()
