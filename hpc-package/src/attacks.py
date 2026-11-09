import numpy as np


class NoAttack:
    def select_victims(self, round_num, num_clients, client_info):
        return []


class RandomDropout:
    def __init__(self, fraction=0.3, seed=42):
        self.fraction = fraction
        self.rng = np.random.default_rng(seed)

    def select_victims(self, round_num, num_clients, client_info):
        k = int(num_clients * self.fraction)
        return self.rng.choice(num_clients, k, replace=False).tolist()


class StrategicGradient:
    """Suppress clients with highest gradient norms (most informative updates)."""
    def __init__(self, fraction=0.3, seed=42):
        self.fraction = fraction
        self.rng = np.random.default_rng(seed)

    def select_victims(self, round_num, num_clients, client_info):
        k = int(num_clients * self.fraction)
        grad_norms = client_info.get("grad_norms", None)
        if grad_norms is None or grad_norms.max() == 0:
            return self.rng.choice(num_clients, k, replace=False).tolist()
        ranked = np.argsort(grad_norms)[::-1]
        return ranked[:k].tolist()


class StrategicDistribution:
    """Suppress clients with rarest class distributions (most unique data)."""
    def __init__(self, fraction=0.3):
        self.fraction = fraction

    def select_victims(self, round_num, num_clients, client_info):
        k = int(num_clients * self.fraction)
        class_dists = client_info.get("class_distributions", None)
        if class_dists is None:
            return []
        mean_dist = class_dists.mean(axis=0)
        divergence = np.sum(np.abs(class_dists - mean_dist), axis=1)
        ranked = np.argsort(divergence)[::-1]
        return ranked[:k].tolist()


class CombinedStrategic:
    """Suppress clients scoring high on both gradient norm and distribution rarity."""
    def __init__(self, fraction=0.3, grad_weight=0.5, seed=42):
        self.fraction = fraction
        self.grad_weight = grad_weight
        self.rng = np.random.default_rng(seed)

    def select_victims(self, round_num, num_clients, client_info):
        k = int(num_clients * self.fraction)
        grad_norms = client_info.get("grad_norms", None)
        class_dists = client_info.get("class_distributions", None)
        if grad_norms is None or class_dists is None:
            return self.rng.choice(num_clients, k, replace=False).tolist()
        if grad_norms.max() == 0:
            mean_dist = class_dists.mean(axis=0)
            divergence = np.sum(np.abs(class_dists - mean_dist), axis=1)
            ranked = np.argsort(divergence)[::-1]
            return ranked[:k].tolist()

        grad_ranks = np.argsort(np.argsort(grad_norms)[::-1]).astype(float)
        mean_dist = class_dists.mean(axis=0)
        divergence = np.sum(np.abs(class_dists - mean_dist), axis=1)
        dist_ranks = np.argsort(np.argsort(divergence)[::-1]).astype(float)

        combined = self.grad_weight * grad_ranks + (1 - self.grad_weight) * dist_ranks
        ranked = np.argsort(combined)
        return ranked[:k].tolist()
