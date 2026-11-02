"""One fixed regularized linear read-benefit head, fit only on training sources."""
import numpy as np

def validate_features(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2048 or not np.isfinite(x).all():
        raise ValueError('finite (tasks, 2048) frozen-base read features required')
    return x

class BenefitHead:
    """2048 -> frozen random 32-D -> trained scalar; no prompt/threshold search."""
    def __init__(self):
        self.projection = np.random.default_rng(20261028).normal(size=(2048, 32)) / np.sqrt(2048)
        self.coefficient = None

    def fit(self, features, targets, weights, group_ids):
        x = validate_features(features) @ self.projection
        y, w = np.asarray(targets, dtype=np.float64), np.asarray(weights, dtype=np.float64)
        if len(x) != len(y) or y.shape != w.shape or y.ndim != 1 or len(group_ids) != len(y):
            raise ValueError('aligned training inputs required')
        if not np.isfinite(y).all() or not np.isfinite(w).all() or (abs(y)>1).any() or (w<0).any() or not (w>0).any():
            raise ValueError('finite bounded targets and nonnegative nonempty weights required')
        # Equal total mass per source/question-connected component, not per rotation.
        totals = {}
        for group, weight in zip(group_ids, w):
            totals[group] = totals.get(group, 0.) + weight
        w = np.array([weight / totals[group] if totals[group] else 0. for group, weight in zip(group_ids, w)])
        w /= w.sum()
        self.mean = np.sum(w[:, None]*x, axis=0)
        self.scale = np.sqrt(np.sum(w[:, None]*(x-self.mean)**2, axis=0)).clip(1e-6)
        x = np.column_stack(((x-self.mean)/self.scale, np.ones(len(x))))
        penalty = np.eye(33)  # lambda=1 fixed before features; intercept unpenalized.
        penalty[-1, -1] = 0
        self.coefficient = np.linalg.solve(x.T @ (w[:, None]*x) + penalty, x.T @ (w*y))
        return self

    def predict_gain(self, features):
        if self.coefficient is None:
            raise RuntimeError('head not fitted')
        x = validate_features(features) @ self.projection
        x = np.column_stack(((x-self.mean)/self.scale, np.ones(len(x))))
        return (x @ self.coefficient).clip(-1, 1)

    def routes(self, features):
        # Zero is an accuracy-benefit boundary; not calibrated P(evidence sufficient).
        return np.where(self.predict_gain(features) > 0, 'full', 'recent')
