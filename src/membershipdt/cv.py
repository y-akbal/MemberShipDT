from __future__ import annotations
import itertools
import numpy as np
from .encoding import as_csr, take_rows
from .tree import check_y


def accuracy(y, p): return float(((p > 0.5) == (y > 0.5)).mean())
def neg_log_loss(y, p, eps=np.finfo(float).eps): p = np.clip(p, eps, 1 - eps); return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def neg_brier(y, p): return -float(np.mean((p - y) ** 2))


def auc(y, p):
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    _, first, counts = np.unique(p[order], return_index=True, return_counts=True)
    ranks[order] = np.repeat(first + (counts - 1) / 2 + 1, counts)
    npos, nneg = (y > 0.5).sum(), (y <= 0.5).sum()
    return 0.5 if npos == 0 or nneg == 0 else float((ranks[y > 0.5].sum() - npos * (npos + 1) / 2) / (npos * nneg))


SCORERS = {"accuracy": accuracy, "neg_log_loss": neg_log_loss, "neg_brier": neg_brier, "auc": auc, "roc_auc": auc}


def get_scorer(scoring): return SCORERS[scoring] if isinstance(scoring, str) else scoring


def stratified_folds(y, cv, rng):
    folds = [[] for _ in range(cv)]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, cv)): folds[k].extend(chunk.tolist())
    folds = [np.sort(np.array(f, np.int64)) for f in folds]
    return [(np.concatenate([folds[j] for j in range(cv) if j != k]), folds[k]) for k in range(cv)]


def param_combos(grid):
    grid = [grid] if isinstance(grid, dict) else list(grid)
    return [dict(zip(g, vals)) for g in grid for vals in itertools.product(*g.values())]


def cross_val_score(estimator, X, y, cv=5, scoring="accuracy", sample_weight=None, random_state=None, folds=None, encoded=None):
    indptr, indices, V, _ = as_csr(X, None, True) if encoded is None else encoded
    classes, yb = check_y(y)
    w = None if sample_weight is None else np.asarray(sample_weight, np.float64)
    folds = stratified_folds(yb, cv, np.random.default_rng(random_state)) if folds is None else folds
    score = get_scorer(scoring)
    out = []
    for tr, te in folds:
        est = estimator.clone()
        est.fit(take_rows(indptr, indices, tr), yb[tr], None if w is None else w[tr], n_features=V)
        p = est.predict_proba(take_rows(indptr, indices, te))[:, 1]
        out.append(score(yb[te], p))
    return np.array(out)


class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, scoring="accuracy", n_iter=None, refit=True, random_state=None, verbose=0):
        self.estimator, self.param_grid, self.cv, self.scoring = estimator, param_grid, cv, scoring
        self.n_iter, self.refit, self.random_state, self.verbose = n_iter, refit, random_state, verbose

    def candidates(self, rng):
        combos = param_combos(self.param_grid)
        if self.n_iter is not None and self.n_iter < len(combos): combos = [combos[i] for i in rng.choice(len(combos), self.n_iter, replace=False)]
        return combos

    def fit(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.random_state)
        encoded = as_csr(X, None, True)
        _, yb = check_y(y)
        folds = stratified_folds(yb, self.cv, rng)
        self.cv_results_ = []
        for params in self.candidates(rng):
            scores = cross_val_score(self.estimator.clone(**params), X, y, self.cv, self.scoring, sample_weight, folds=folds, encoded=encoded)
            self.cv_results_.append(dict(params=params, scores=scores, mean=float(scores.mean()), std=float(scores.std())))
            if self.verbose: print(f"{params} -> {scores.mean():.4f} ± {scores.std():.4f}", flush=True)
        best = max(self.cv_results_, key=lambda r: r["mean"])
        self.best_params_, self.best_score_ = best["params"], best["mean"]
        if self.refit: self.best_estimator_ = self.estimator.clone(**self.best_params_).fit(X, y, sample_weight)
        return self

    def predict_proba(self, X): return self.best_estimator_.predict_proba(X)
    def predict(self, X): return self.best_estimator_.predict(X)


class RandomizedSearchCV(GridSearchCV):
    def __init__(self, estimator, param_grid, n_iter=10, **kw): super().__init__(estimator, param_grid, n_iter=n_iter, **kw)
