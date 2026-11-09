# Paper A v2: pre-registered expectations

**Status: APPROVED by Claude on 2026-11-06 (approval delegated by the director), and frozen.** The file's sha256 is in `hpc-package/run_v2.py`, which refuses to run if this file changes. Nothing here is edited after any v2 cell completes.

**Basis.** This file was written after the v1 results (`ANALYSIS.md`) and before any v2 cell ran. Every mechanism that v1 suggested is retested here on fresh seeds. v1 itself is never re-scored.

**Framing (option C, director 2026-11-06).**
- **Measurement:** when strategic availability suppression actually hurts non-IID FedAvg, and why.
- **Defence:** AAWA v2.
- **Baseline:** AAWA v1, as published at IC3AI, runs alongside unchanged.

## Fixed, as in v1

| Setting | Value |
|---|---|
| Clients | 20 |
| Dirichlet α | 0.1 |
| Rounds | 100 |
| Local epochs | 3 |
| Optimiser | SGD, LR 0.01 |
| Batch size | 64 |
| Precision | fp32, TF32 off |
| Suppression fraction | 0.3 (except Block D) |
| Models | `CIFAR10Net` for CIFAR, `MNISTNet` for EMNIST |

The attack code (`src/attacks.py`) and the attacker's observable (last-batch gradient norm) are unchanged. AAWA v1 runs at W = 10, τ = 0.5 with `update_norm`.

## Changed from v1, and why

1. **Seeds {1, 2, 3, 4, 5}.** They are fresh, and there are 5 because the v1 CIs over 3 seeds were too wide.
2. **Primary metric: mean test accuracy over rounds 91–100.** In v1, accuracy moved 0.5–1.0 pp (sd) from round to round. Round-100 accuracy, v1's metric, is still reported as secondary.
3. **Deterministic GPU mode.** Settings: `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic`, no `cudnn.benchmark`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. In v1, two identical runs ended 2.21 pp apart. Determinism makes every number reproducible. It does not shrink seed variance. Gate G1 decides whether it is kept.
4. **Datasets.**
   - EMNIST-balanced, 47 classes. The main dataset: the only one where the v1 attack hurt.
   - CIFAR-100. New: same image domain and model as CIFAR-10 with 10× the classes, so it isolates class count.
   - CIFAR-10. Replication.
   - CIFAR-100 uses the CIFAR-10 augmentation and normalisation.
5. **Aggregators: FedAvg, AAWA v1, AAWA v2.** Median, trimmed mean and Krum are dropped, because v1 already shows they fail at α = 0.1 with or without an attack. The paper reports them from v1.
6. **New logs.**
   - Per-class test accuracy every round.
   - Each participant's aggregation weight every round, for AAWA v1 and v2. FedAvg's weights follow from client sizes.

## AAWA v2 (frozen specification)

v1 had two diagnosed blind spots:
- It cannot see clients that are absent from round 0, because its suspicion score needs a history.
- It down-weights returning victims.

v2 drops suspicion entirely and corrects the **class coverage** of each round's aggregate. Server-side only; it needs the same per-client class histograms that v1 already assumes the server holds.

**Notation.**
- n_j: client sizes.
- p_j: client class distributions, in the defender's view (DP-noised in Block C).
- π = Σ_j n_j p_j / Σ_j n_j: the population class mixture, over all registered clients.
- P: this round's participants.

**Each round:**

1. If no client is absent, call `fedavg()` unchanged. Full participation is FedAvg exactly, so there are no v2 `no_attack` cells.
2. Base weights b_i = n_i / Σ_{k∈P} n_k for i ∈ P. Participant mixture m = Σ_{i∈P} b_i p_i.
3. Class importance r_c = min(π_c / max(m_c, 1e-12), R) with **R = 3**.
4. Client factor f_i = Σ_c p_ic r_c: the mean importance of client i's samples. Weight w_i = b_i f_i / Σ_k b_k f_k.

**Why R = 3:** it caps any class's importance at 3×. That is enough to restore a class whose participating holders keep a third of it. Block B reports R = 2 and R = 5 whatever they show.

**Properties checked by gate G0:**
- No history is needed, so clients absent from round 0 are corrected (blind spot 1).
- A returning victim holds under-represented classes, so its weight ratio w_i / b_i > 1: it is up-weighted, never down-weighted (blind spot 2).

## Grid (310 cells)

| Block | Cells | Content |
|---|---|---|
| A, main | 210 | 3 datasets × 5 attacks × 5 seeds × {FedAvg, v1, v2}, minus v2 `no_attack` (15, identical to FedAvg) |
| B, cap | 20 | EMNIST × {strategic_distribution, combined_strategic} × R ∈ {2, 5} × 5 seeds, v2 |
| C, DP | 20 | {EMNIST, CIFAR-100} × {distribution, combined} × ε = 1 × 5 seeds, v2. Mechanism: v1's Laplace mechanism (`privatize_distributions`) |
| D, dose | 60 | EMNIST × suppression {0.1, 0.5} × {random, distribution, combined} × {FedAvg, v2} × 5 seeds. "random" is v1's `random_30` attack at the given fraction. `no_attack` is shared with Block A |

**Run order:** A on EMNIST, A on CIFAR-100, D, B, C, then A on CIFAR-10. Results go to `hpc-package/results/v2/`. The v1 directory is read-only.

## Gates (Claude checks before launch)

- **G0, smoke test on CPU.**
  - v2 under full participation reproduces `fedavg()` bit for bit.
  - Weights sum to 1, and every client factor f_i is at most R.
  - A client absent from round 0 still shifts the weights.
  - A returning holder of a withheld class gets weight ratio w_i / b_i > 1.
  - The new logs are present.
  - The v1 smoke test still passes.
- **G1, on the pod.** One EMNIST cell is run twice for 3 rounds.
  - Deterministic mode stays on only if the accuracies match exactly and the round time is at most 1.5× nondeterministic mode.
  - Otherwise the grid runs nondeterministic.
  - Every cell records the mode it ran in.
- **G2.** The CIFAR-100 archive's md5 is verified before use.

## Hypotheses

**Definitions.** All values are on the primary metric and paired by seed.
- FedAvg harm: H(a) = acc(no_attack) − acc(a).
- Gap: G(X, a) = acc(X, a) − acc(FedAvg, a).
- CI: paired t with df 4 (t = 2.776).
- "Mean": the mean over the 5 seeds.

### Primary (the headline rests only on these)

- **P1 (measurement, EMNIST).** acc(random_30) − acc(combined_strategic) > 0, with the CI excluding 0. The mean of H(strategic_distribution) also exceeds H(random_30). v1 prior: combined 81.96 vs random 83.20.
- **P2 (measurement, class count).** The share of learned accuracy that strategic_distribution removes, H / (acc(no_attack) − 100/K) with K classes, is larger on CIFAR-100 than on CIFAR-10 (means). Normalising above chance keeps CIFAR-100's lower accuracy from inflating the ratio by itself.
- **P3 (defence, EMNIST).** G(v2, combined_strategic) ≥ +0.5 pp, with the CI excluding 0. v1 headroom: 2.2 pp.

### Secondary: measurement

- **M1 (CIFAR-10, replicating v1).** H(combined) < H(random_30) and H(distribution) < H(random_30) (means).
- **M2 (CIFAR-100).** H(distribution) > H(random_30) and H(combined) > H(random_30) (means).
- **M3 (coverage mechanism).**
  - Definitions, for each class c in each FedAvg strategic cell (distribution and combined): w_c is the share of class-c training samples held by absent clients, averaged over rounds. The drop is per-class acc(no_attack) − per-class acc(cell), same seed, over rounds 91–100.
  - Prediction, Spearman ρ(w, drop) pooled per dataset: ρ ≥ 0.3 on EMNIST and CIFAR-100, and ρ > 0 on CIFAR-10.
- **M4 (drift offset, CIFAR-10, strategic_distribution).** Classes with w_c < 0.05 gain accuracy against no_attack (mean over those classes and seeds > 0). This is v1's untested reading, now tested.
- **M5 (dose, EMNIST, FedAvg).**
  - H rises with the suppression fraction (0.1 < 0.3 < 0.5) for each attack.
  - H(strategic) − H(random_30) is larger at 0.5 than at 0.1, for distribution and for combined.

### Secondary: replicating the v1 diagnosis on fresh seeds

- **D1 (blind spot 1).** Under strategic_distribution, v1's logged weights equal FedAvg's (max abs difference < 1e-9) in every round of every cell.
- **D2 (blind spot 2).** Under combined_strategic, returning victims get a median v1 weight ratio (v1 weight / FedAvg share) < 1. A returning victim is a participant whose participation over the last 10 rounds, the current round included, is ≤ 0.5. This is the window AAWA v1 itself uses.
- **D3.** G(v1, combined_strategic) ≤ 0 on EMNIST (mean).

### Secondary: defence

- **V1.** EMNIST strategic_distribution: G(v2) > 0, with the CI excluding 0.
- **V2.** CIFAR-100: G(v2) > 0 under distribution and under combined (means).
- **V3.** acc(v2) − acc(v1) > 0 under combined_strategic on EMNIST and on CIFAR-100 (means).
- **V4.** random_30: G(v2) ≥ −0.3 pp on every dataset (no cost under benign dropout).
- **V5.** CIFAR-10: |G(v2)| ≤ 1 pp under each strategic attack. No benefit is predicted there: v1 shows less than 1 pp of headroom.
- **V6 (cap).** On EMNIST, the range of v2 accuracy across R ∈ {2, 3, 5} is < 1 pp for each attack (means).
- **V7 (DP).** |v2(ε = 1) − v2| ≤ 0.5 pp on EMNIST and CIFAR-100, for both attacks. Rationale: noise of scale 1 count per class against thousands of samples per client.
- **V8 (dose).** On EMNIST, G(v2) under distribution and under combined is larger at suppression 0.5 than at 0.1.
- **V9 (mechanism).** Under combined_strategic on EMNIST, returning victims get a median v2 weight ratio > 1.

### Surprises (report them, don't fix them)

- v2 falls below FedAvg by more than 0.5 pp under any strategic attack on EMNIST or CIFAR-100.
- v2 costs more than 0.5 pp under random_30.
- P1 reverses.
- D1 fails, which would mean v1's code path differs from the diagnosis.
- G1 fails.

## Analysis plan and stopping rule

- `analyze_v2.py` is written before any v2 cell runs, with the thresholds copied from this file. It is not edited after the first v2 cell completes; any later analysis is labelled exploratory.
- Every claim is reported. The headline uses P1–P3 only. There are about 25 tests and no multiplicity correction, so the secondary claims are descriptive.
- **If P3 and V1 both fail,** the paper reports v2 as a second negative result next to v1, and the measurement (P1, P2, M1–M5) carries the paper. There is no AAWA v3 before the 31 Jan 2027 submission.

## Approval

Approved by Claude (Opus 5.5), 2026-11-06. The director delegated approval of pre-registrations to Claude that day and looks without approving.

Gate G0 passed before approval: `smoke_v2.py` and the v1 `smoke_test.py` both pass. `analyze_v2.py` ran end to end on synthetic data and on an empty results directory. No v2 cell had run.

Frozen code. A change to any of these files before the run needs a dated amendment here, which changes this file's hash, and the runner then has to be updated on purpose.

| File | sha256 |
|---|---|
| `analyze_v2.py` | `17fa6a26d3402a9a61f0f59f85a0853e1dcd033f160be0b8b78f13256ab19bf9` |
| `hpc-package/src/aggregators.py` | `f61a491786183c5312b4ba42bc71ab058709eac95d287f95595057d5d237feab` |
| `hpc-package/src/attacks.py` | `f4399330883ee279e9ceba4765bd662f7448090874a40502af7f4e1693f5bef7` |
| `hpc-package/src/data.py` | `6b944624132fa1f00ee57a483327b703469868b13979f94a6bb9e7d2376eab15` |
| `hpc-package/src/model.py` | `99a7214d25634f6496c88388c7d4498a19b3464979b4e4f5f75290f3f6d56b39` |

---

## Amendment 1, 2026-11-06 (before any v2 cell ran)

Written and approved by Claude under the director's delegation. The director chose this plan on 2026-11-06 ("I like the plan"). No v2 cell had run, and no v2 number existed. Everything above this section stands unchanged, including P1–P3, M1–M5, D1–D3 and V1–V9, which stay defined on the three image datasets.

### Why

1. **Replay of AAWA v2 on v1 data.** Replayed on v1's actual EMNIST attack rounds (seeds 42/123/7), v2 closes only **3%** of the class-mix gap between participants and the population (total variation 0.197 → 0.191), and never boosts a client above 1.20×. The cause: at α = 0.1 a withheld class sits almost entirely on the absent clients, so no reweighting of present clients can restore it. v2 is therefore predicted near-inert (V10 below), and P3 is expected to fail. This prediction is made now, before any v2 result.
2. **Prior work** (`RELATED-WORK.md`).
   - Server memory of absent clients' latest updates is the published remedy for unavailability: MIFA, Gu et al., NeurIPS 2021.
   - A staleness cutoff on that memory is also published: FedAR, Jiang et al., ECML 2024. Our capped arm is therefore a staleness probe, not a method claim.
   - The closest threat model is Severi et al., CNS 2022, which drops traffic of clients chosen to hurt a target class. Worst-class accuracy is added to bridge to it.
   - IoT-J fit: a smartphone-sensor dataset, UCI HAR, is added.
3. **Staleness on v1 data.** In v1's EMNIST combined-attack rounds, every client was seen at least once. The median absent client's latest update was 37–42 rounds old, and 36–41% were older than 50 rounds.

### Additions

**Memory arms** (`memory_avg` in `src/aggregators.py`)
- The server stores each participant's latest update, Δ_i = w_local − w_global of the round it trained, together with that round.
- Each round the global model moves by Σ_{i∈U} (n_i / Σ_{j∈U} n_j) Δ_i. U is every client seen so far whose stored update was made in the last S rounds, the current round included (age 0 to S−1).
- Clients never seen contribute nothing. Full participation calls `fedavg()` unchanged, so there are no memory `no_attack` cells.
- Two settings:
  - `mifa`: S = ∞. A MIFA-style memory, sample-weighted over the clients seen so far. MIFA itself weights equally and starts every client at zero; ours keeps FedAvg's weighting, so that full participation reduces to FedAvg exactly.
  - `mem10`: S = 10, the same 10-round window AAWA v1 uses. No discount, unlike FedAR, which is credited for the cutoff idea.

**UCI HAR** (smartphone accelerometer and gyroscope; 6 activities; 561 precomputed features in [−1, 1]; 7,352 train and 2,947 test samples; CC BY 4.0)
- Source: UCI repository dataset 240, outer zip md5 `96924e20fe5b8c9ffc2196eec7d5048f`.
- Model `HARNet`: MLP 561–256–128–6. Same frozen training hyperparameters as every other dataset.
- Two federations:
  - `har_dir`: Dirichlet α = 0.1 over 20 clients, as for the other datasets.
  - `har_subject`: the natural federation, one client per subject in the train split (21 clients, 281–409 samples each). The seed changes only training randomness there.

**Split rule** (all Dirichlet splits)
- If a client gets fewer than 10 samples, the split is redrawn from `SeedSequence([seed, k])`, a stream disjoint from every other seed in use. This is NIID-Bench's `min_require_size = 10` rule (Li et al., ICDE 2022).
- Checked before approval:
  - CIFAR-10 and EMNIST seeds 1–5 need no redraw, so their splits are exactly v1's.
  - HAR Dirichlet seeds 1, 2 and 5 need 1, 3 and 4 redraws. Seed 5's first draw left a client with 0 samples.
  - CIFAR-100 is checked by the pod preflight.
- The redraw count is logged in every cell.

**Grid: 650 cells**

| Block | Cells | Change |
|---|---|---|
| A | 330 | Adds `mifa` and `mem10` to each image dataset × {random_30, gradient, distribution, combined} × 5 seeds |
| B, C, D | 20, 20, 60 | Unchanged |
| H | 220 | HAR: {har_dir, har_subject} × 5 attacks × {FedAvg, v1, v2, mifa, mem10} × 5 seeds, without the defence `no_attack` cells that equal FedAvg |

**Run order:** grouped by dataset, so workers reload data rarely: EMNIST (A, D, B, C), CIFAR-100 (A, C), HAR Dirichlet, HAR subjects, CIFAR-10.

**New logs**
- Memory arms record each round's contributors and their weights.

**Operational changes** (no effect on any number)
- Results are written atomically.
- A cell that raises is recorded as `<cell>.err` and does not stop its worker.
- `run_v2.py --preflight` runs every dataset × every arm on real data for 2 rounds before any worker starts.
- The runner refuses to start unless this file and every file in the latest hash table below match. That table now also freezes `run_journal.py` (v1's training loop and frozen hyperparameters) and `run_v2.py`; `run_v2.py` is hashed with its own PREREG-hash line blanked.
- Data loads at run time without download checks: `setup_v2.sh` verifies every archive by md5 first.
- A model whose weights turn non-finite is recorded as diverged (`diverged_at`) and runs on. It is reported as a surprise, not retried.
- A worker that hits a CUDA error frees its cell and exits, and the launcher starts a fresh one. A respawn budget stops the launcher if workers keep dying.
- A relaunch reuses the gate-G1 mode, so the grid never mixes modes.
- G1 timing excludes each twin's first (warm-up) round.
- These operational changes came from a code review (Claude Code `/code-review` at xhigh) of every v2 file before approval. All 14 findings were fixed and re-tested: smoke tests, a launcher simulation with injected crashes, CUDA errors, a persistent bug and lost locks, and real-data runs on the laptop GPU in deterministic mode.

### New hypotheses (same definitions as above; primary metric; paired by seed; df 4)

- **P4 (primary, memory).** EMNIST combined_strategic: G(mifa) ≥ +0.5 pp, with the CI excluding 0.
- **MEM1a (mechanism).** Under strategic_distribution the victims are never seen, so MIFA's contributors equal the participants in every round of every cell, on all three image datasets.
- **MEM1b.** strategic_distribution: |G(mifa)| < 0.3 pp on each image dataset (means).
- **MEM2.** random_30: G(mifa) > 0 on EMNIST and CIFAR-100 (means). This is memory's home ground.
- **Q1 (no prediction; reported).** The staleness trade-off: acc(mifa) − acc(mem10) under combined_strategic on EMNIST and CIFAR-100. Both signs are informative, so no direction is registered.
- **V10.** EMNIST combined_strategic: |G(v2)| < 0.3 pp. v2 is near-inert, predicted from the replay in "Why" (1).
- **WC1 (bridge to Severi).** The worst class is the minimum per-class accuracy over rounds 91–100. Its FedAvg harm against no_attack is larger under strategic_distribution than under random_30 on EMNIST, CIFAR-100 and CIFAR-10 (means).
- **HAR1.** On HAR natural subjects, the FedAvg harm is < 0.5 pp under every attack (means). Each subject does every activity, so classes are not concentrated.
- **HAR2 (concentration, within one dataset).** H(strategic_distribution) on HAR Dirichlet α = 0.1 > H(strategic_distribution) on HAR natural subjects (means).

**New surprise:** a memory arm falls below FedAvg by more than 1 pp under any attack on any dataset.

`analyze_v2.py` scores all of the above. The no-v3 stopping rule extends to the memory arms: no tuning of S or of the memory weighting after any v2 result.

### Approval of amendment 1

Approved by Claude (Opus 5.5) on 2026-11-06, under the director's delegation. No v2 cell had run.

Gates passed before approval:
- G0: `smoke_v2.py`, including the error-path, divergence and mem10-boundary checks, and the v1 `smoke_test.py`.
- `analyze_v2.py` on synthetic results: complete, partial, and partial with an injected violation.
- The launcher simulation.
- Real-data runs of every dataset × arm on the laptop GPU in deterministic mode, with bit-identical twins on HAR (both federations), CIFAR-10 and EMNIST.
- CIFAR-100 is first loaded by `setup_v2.sh` and `--preflight` on the pod.

Frozen files for amendment 1. This table supersedes the earlier one; `run_v2.py` is hashed with its PREREG-hash line blanked.

| File | sha256 |
|---|---|
| `analyze_v2.py` | `7e5c2a471430fe470b2341fd70a1d80872823dff07ff4b08d31aa7dfd231238b` |
| `hpc-package/run_v2.py` | `1c88accd9758a87db49fac897498b31247cd144677f3831d338cfd852333c2fe` |
| `hpc-package/run_journal.py` | `c0d536eca80e63819dde35a8bb981ac4efe8747415b26125d8b8a48f6c00244b` |
| `hpc-package/src/aggregators.py` | `27e9f30b98ac41c330bf0ebca4b000091559d3f456bf0150567a69917c7790a2` |
| `hpc-package/src/attacks.py` | `f4399330883ee279e9ceba4765bd662f7448090874a40502af7f4e1693f5bef7` |
| `hpc-package/src/data.py` | `9557c51485d6ca05e990603dace400c72b97eabe7957095e0b39b7ffa6b05c8b` |
| `hpc-package/src/model.py` | `75b4596a76ff41615286fe50cf51ffc7af8e5a388539291af5c83ec072bd7643` |
