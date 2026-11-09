# Paper A (AAWA-journal) — Pre-registered expectations

**Status: APPROVED by director (Ziyam), 2026-11-06. Frozen; changes only as dated amendments below the approval line.**
Written before any journal-grid result was opened. Basis: IC3AI `findings.md` (Run 4, CIFAR-10, 100 rounds, seed 42, grad_norm signal, fp32 on GTX 1650; Run 5 sensitivity, relative only).
Rule: report what comes back. No retuning of AAWA after seeing numbers. Frozen config: 20 clients, α=0.1, 100 rounds, 3 local epochs, LR 0.01, batch 64, suppression 30%, seeds {42, 123, 7}.

Known differences from IC3AI that may move numbers: suspicion signal `update_norm` (IC3AI used `grad_norm`); precision fp32 with TF32 off (director decision 2026-11-06, parity with IC3AI); 3 seeds instead of 1.

## Phase 1 — CIFAR-10, 5 attacks × 5 aggregators × 3 seeds

- **E1.1** Under `combined_strategic`, AAWA beats FedAvg at round 100 by **≥ 1.5 pp** (mean over seeds), with the 95% CI excluding 0. IC3AI single-seed: +2.82 pp.
- **E1.2** Under `random_30`, the AAWA–FedAvg gap is **within ±1 pp** (FedAvg self-corrects over long training). IC3AI: +0.31 pp.
- **E1.3** Under `no_attack`, AAWA ≈ FedAvg (**|gap| < 0.5 pp**): no cost when there is no attack. IC3AI: −0.26 pp.
- **E1.4** Strategic hurts FedAvg more than random at 100 rounds (the duration flip from IC3AI holds across seeds).
- **E1.5** Coordinate median and trimmed mean stay **≥ 8 pp below** FedAvg in every attack setting. Krum stays near chance (~10%) at α=0.1.
- **E1.6** AAWA's seed-to-seed std is lower than FedAvg's under strategic attacks.
- New, no prior: `strategic_gradient` and `strategic_distribution` taken separately. Expect each to be weaker than `combined_strategic` for FedAvg.

## Phase 2 — AAWA ablation, combined_strategic, seed 42

- **E2.1** Across W ∈ {5, 10, 20} × τ ∈ {0.3, 0.5, 0.7}, AAWA accuracy varies by **< 2.5 pp** (robust to hyperparameters, as in IC3AI Run 5).
- **E2.2** `update_norm` ≥ `grad_norm` on average across the 9 configs (the journal's reason for switching signals). If it isn't, report it and keep `grad_norm` as the main-table signal.

## Phase 3 — EMNIST-balanced (47 classes), 3 attacks × 5 aggregators × 3 seeds

- **E3.1** Same ordering as CIFAR: AAWA ≥ FedAvg under `combined_strategic`. Direction predicted, size not: with 47 classes over 20 clients at α=0.1, suppression removes rarer classes, so the gap may be **larger** than on CIFAR.
- **E3.2** No-attack parity holds (|gap| < 0.5 pp).

## Phase 4 — Defender sees DP-noised class histograms (CIFAR-10, AAWA only, 3 strategic attacks × ε ∈ {10, 1, 0.1} × 3 seeds)

Added 2026-11-06, before any Phase 4 cell ran. Mechanism: Laplace noise of scale 1/ε on each client's class-count histogram (L1 sensitivity 1), clipped at 0, renormalised; released once per run, so ε is the total budget. The attacker keeps the true histograms. Comparator: the matching non-private AAWA cell from Phase 1.

Reasoning: each client holds ~2,500 CIFAR images, so even at ε=0.1 the noise (scale 10 counts per class) is small against the classes a client actually holds. It mostly adds small mass to absent classes.

- **E4.1** At ε=10 and ε=1, AAWA is within **±0.5 pp** of non-private AAWA under every strategic attack (seed mean).
- **E4.2** At ε=0.1, AAWA loses **≤ 1.5 pp** against non-private AAWA, and still beats FedAvg under `combined_strategic`.
- **E4.3** The loss grows as ε falls (10 ≥ 1 ≥ 0.1 in accuracy), allowing for seed noise.

## What counts as a surprise (report it, don't fix it)

Any of: AAWA loses to FedAvg under strategic on either dataset; the duration flip disappears across seeds; the sensitivity range is > 5 pp; Krum is above chance; AAWA with DP falls below FedAvg at any ε; or DP *improves* AAWA by more than 0.5 pp.

Director approval: Ziyam, 2026-11-06 (all four phases, E1.1–E4.3)

---

## Amendment 2026-11-06: outcome of v1 and the director's decision

The v1 grid (165 cells) was scored against this file unchanged. **4 of 14 claims hold. E1.1, E3.1 and the core claim, that AAWA beats FedAvg under strategic suppression, fail.** See `ANALYSIS.md`.

Director's decision (Ziyam, 2026-11-06):

- **Option C.** Paper A becomes a measurement paper: when strategic suppression hurts non-IID FL, when it doesn't, and why defences that rely on participation history miss it. AAWA v2 forms the defence section.
- **The failed v1 run stays in the paper** as a reported result, with its figures (`figures/v1/`, from `plot.py`). Nothing is dropped.

Rules that follow:

- v1 results are frozen read-only: `hpc-package/results/journal/`, `logs/` and `journal_results.tgz` (md5 fdce8bc89ae30c4a558da8c7c25cab8c).
- No v2 run writes into the v1 directory.
- AAWA v2 gets its own pre-registration, approved before any v2 cell runs, and uses fresh seeds: never 42/123/7 retuned.
- The predictions above stay as written. They are the record of what was expected.
