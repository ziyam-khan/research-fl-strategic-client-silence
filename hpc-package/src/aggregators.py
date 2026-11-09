import numpy as np
import torch


def fedavg(global_model, client_state_dicts, client_weights):
    global_dict = global_model.state_dict()
    total_weight = sum(client_weights)

    for key in global_dict:
        global_dict[key] = sum(
            client_state_dicts[i][key].float() * (client_weights[i] / total_weight)
            for i in range(len(client_state_dicts))
        )

    global_model.load_state_dict(global_dict)
    return global_model


def coordinate_median(global_model, client_state_dicts, client_weights):
    global_dict = global_model.state_dict()

    for key in global_dict:
        stacked = torch.stack([sd[key].float() for sd in client_state_dicts])
        global_dict[key] = torch.median(stacked, dim=0).values

    global_model.load_state_dict(global_dict)
    return global_model


def trimmed_mean(global_model, client_state_dicts, client_weights, beta=0.1):
    global_dict = global_model.state_dict()
    n = len(client_state_dicts)
    trim = max(1, int(n * beta))
    if n - 2 * trim < 1:
        trim = max(0, (n - 1) // 2)

    for key in global_dict:
        stacked = torch.stack([sd[key].float() for sd in client_state_dicts])
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[trim:n - trim]
        global_dict[key] = trimmed.mean(dim=0)

    global_model.load_state_dict(global_dict)
    return global_model


def krum(global_model, client_state_dicts, client_weights, f=None):
    """Multi-Krum: select the update closest to its neighbors."""
    n = len(client_state_dicts)
    if f is None:
        f = max(1, int(n * 0.3))

    flat_updates = []
    for sd in client_state_dicts:
        flat = torch.cat([sd[key].float().flatten() for key in sorted(sd.keys())])
        flat_updates.append(flat)
    flat_updates = torch.stack(flat_updates)

    dists = torch.cdist(flat_updates, flat_updates, p=2)

    n_closest = max(1, n - f - 2)
    scores = []
    for i in range(n):
        sorted_dists, _ = torch.sort(dists[i])
        scores.append(sorted_dists[1:n_closest + 1].sum().item())

    best = int(np.argmin(scores))
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = client_state_dicts[best][key].float()

    global_model.load_state_dict(global_dict)
    return global_model


def aawa(global_model, client_state_dicts, client_weights, client_ids,
         participation_history, grad_norm_history, class_distributions,
         window_size=10, threshold=0.5, log=None):
    global_dict = global_model.state_dict()
    n_total_clients = len(participation_history)
    n_participating = len(client_state_dicts)

    avg_grads = np.zeros(n_total_clients)
    for cid in range(n_total_clients):
        gnorms = grad_norm_history[cid][-window_size:]
        if len(gnorms) > 0:
            avg_grads[cid] = np.mean(gnorms)
    max_grad = avg_grads.max() + 1e-8

    reliability_scores = np.ones(n_total_clients)
    for cid in range(n_total_clients):
        history = participation_history[cid][-window_size:]
        if len(history) == 0:
            continue
        participation_rate = sum(history) / len(history)
        normalized_grad = avg_grads[cid] / max_grad

        if (1 - participation_rate) > 0 and normalized_grad > 0:
            suspicion = (1 - participation_rate) * normalized_grad
        else:
            suspicion = 0.0

        reliability_scores[cid] = max(0.1, 1.0 - suspicion)

    absent_ids = set(range(n_total_clients)) - set(client_ids)
    suppressed_dist = np.zeros(class_distributions.shape[1])
    n_suppressed = 0
    for aid in absent_ids:
        if reliability_scores[aid] < threshold:
            suppressed_dist += class_distributions[aid]
            n_suppressed += 1

    adjusted_weights = np.array(client_weights, dtype=float)
    if n_suppressed > 0:
        suppressed_dist /= n_suppressed
        for i, cid in enumerate(client_ids):
            similarity = 1.0 - 0.5 * np.sum(np.abs(
                class_distributions[cid] - suppressed_dist
            ))
            compensation = 1.0 + similarity * (n_suppressed / n_total_clients)
            adjusted_weights[i] *= compensation

    for i, cid in enumerate(client_ids):
        adjusted_weights[i] *= reliability_scores[cid]

    total = adjusted_weights.sum()
    if total > 0:
        adjusted_weights /= total
    if log is not None:  # v2 diagnostics only; does not change the aggregation
        log.append(adjusted_weights.tolist())

    for key in global_dict:
        global_dict[key] = sum(
            client_state_dicts[i][key].float() * adjusted_weights[i]
            for i in range(n_participating)
        )

    global_model.load_state_dict(global_dict)
    return global_model


def aawa_v2_weights(client_weights, client_ids, class_distributions, client_sizes, cap=3.0):
    """Returns (w, f): normalised weights and the uncapped-by-normalisation client factors f_i <= cap."""
    dists = np.asarray(class_distributions, dtype=float)
    sizes = np.asarray(client_sizes, dtype=float)
    pi = (sizes[:, None] * dists).sum(0)
    pi /= pi.sum()
    b = np.array(client_weights, dtype=float)  # a copy: never normalise the caller's array in place
    b = b / b.sum()
    p = dists[list(client_ids)]
    r = np.minimum(pi / np.maximum(b @ p, 1e-12), cap)
    f = p @ r
    w = b * f
    return w / w.sum(), f


def aawa_v2(global_model, client_state_dicts, client_weights, client_ids,
            class_distributions, client_sizes, cap=3.0, log=None):
    """AAWA v2 (PREREG-v2.md): restore the population class mixture of each round's aggregate.

    No suspicion and no history: weight each participant by the mean importance
    pi_c / m_c (capped at `cap`) of its samples, where pi is the class mixture of all
    registered clients and m that of this round's participants under FedAvg weights.
    Full participation calls fedavg() unchanged.
    """
    if len(client_ids) == len(client_sizes):
        if log is not None:
            b = np.asarray(client_weights, dtype=float)
            log.append((b / b.sum()).tolist())
        return fedavg(global_model, client_state_dicts, client_weights)

    w, _ = aawa_v2_weights(client_weights, client_ids, class_distributions, client_sizes, cap)
    if log is not None:
        log.append(w.tolist())
    return fedavg(global_model, client_state_dicts, w.tolist())


def memory_avg(global_model, client_state_dicts, client_weights, client_ids, memory, rnd,
               client_sizes, window=None, log=None):
    """MIFA-style memory (PREREG-v2 amendment 1). Each client's latest update (delta from the
    global model it trained on) is stored with the round it was made; every round the global
    model moves by the sample-weighted mean of the stored updates of all clients seen so far
    whose update was made in the last `window` rounds, the current one included, i.e. age
    rnd - r < window, matching AAWA's window (None = no limit, i.e. MIFA). Clients
    never seen contribute nothing. Full participation calls fedavg() unchanged.
    `memory` (dict cid -> (delta, round)) is owned by the caller, one per run, and mutated here.
    """
    global_sd = global_model.state_dict()
    for sd, cid in zip(client_state_dicts, client_ids):
        memory[cid] = ({k: sd[k].float() - global_sd[k].float() for k in global_sd}, rnd)
    if len(client_ids) == len(client_sizes):
        if log is not None:
            b = np.asarray(client_weights, dtype=float)
            log.append(((b / b.sum()).tolist(), list(client_ids)))
        return fedavg(global_model, client_state_dicts, client_weights)

    use = [c for c, (_, r) in memory.items() if window is None or rnd - r < window]
    n = np.array([client_sizes[c] for c in use], dtype=float)
    w = n / n.sum()
    if log is not None:
        log.append((w.tolist(), use))
    new = {k: global_sd[k].float() + sum(memory[c][0][k] * float(wi) for c, wi in zip(use, w))
           for k in global_sd}
    global_model.load_state_dict(new)
    return global_model
