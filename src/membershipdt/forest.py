from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .encoding import as_csr, frequency_perm, remap
from .tree import MembershipDecisionTree, OOVMixin, check_y, leaf_rules, format_rules
from . import _builder_cy
from . import serialize as S


class MembershipRandomForest(OOVMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=None, matcher=None, match_mode="top1", match_threshold=0.0, match_k=1):
        self.n_estimators, self.max_depth, self.min_samples_leaf, self.min_samples_split = n_estimators, max_depth, min_samples_leaf, min_samples_split
        self.max_features, self.bootstrap, self.n_jobs, self.random_state = max_features, bootstrap, n_jobs, random_state
        self.matcher, self.match_mode, self.match_threshold, self.match_k = matcher, match_mode, match_threshold, match_k

    @property
    def threads(self): return os.cpu_count() or 1 if self.n_jobs in (None, -1, 0) else int(self.n_jobs)

    def make_tree(self, seed): return MembershipDecisionTree(self.max_depth, self.min_samples_leaf, self.min_samples_split, self.max_features, int(seed))

    PARAMS = ("n_estimators", "max_depth", "min_samples_leaf", "min_samples_split", "max_features", "bootstrap", "n_jobs", "random_state", "matcher", "match_mode", "match_threshold", "match_k")

    def get_params(self): return {k: getattr(self, k) for k in self.PARAMS}
    def clone(self, **over): return type(self)(**{**self.get_params(), **over})

    def fit(self, X, y, sample_weight=None, n_features=None):
        self._resolver = None
        indptr, indices, V, vocab = as_csr(X, n_features, True)
        V = max(V, n_features or 0)
        self.vocab_, self.n_features_ = vocab, V
        self.perm_, self.inv_ = frequency_perm(indices, V)
        indptr, indices = remap(indptr, indices, self.inv_)
        self.classes_, yb = check_y(y)
        n = len(yb)
        base = np.ones(n) if sample_weight is None else np.asarray(sample_weight, np.float64)
        seeds = np.random.SeedSequence(self.random_state).spawn(self.n_estimators)

        def fit_one(ss):
            rng = np.random.default_rng(ss)
            w = base * np.bincount(rng.integers(0, n, n), minlength=n) if self.bootstrap else base
            t = self.make_tree(ss.generate_state(1, np.uint64)[0])
            t.n_features_, t.vocab_, t.classes_ = V, None, self.classes_
            return t.fit_encoded(indptr, indices, yb, w)

        with ThreadPoolExecutor(self.threads) as ex: self.estimators_ = list(ex.map(fit_one, seeds))
        self.pack()
        return self

    def pack(self):
        trees = [e.tree_ for e in self.estimators_]
        sizes = np.array([len(t["feature"]) for t in trees], np.int64)
        self.offsets_ = np.concatenate([[0], np.cumsum(sizes)])
        self.feature_ = np.concatenate([t["feature"] for t in trees]).astype(np.int32)
        self.left_ = np.concatenate([np.where(t["left"] >= 0, t["left"] + o, -1) for t, o in zip(trees, self.offsets_)]).astype(np.int32)
        self.right_ = np.concatenate([np.where(t["right"] >= 0, t["right"] + o, -1) for t, o in zip(trees, self.offsets_)]).astype(np.int32)
        self.p1_ = np.concatenate([e.leaf_proba()[:, 1] for e in self.estimators_])
        return self

    def split_tokens(self):
        f = self.feature_
        return {self.token_name(int(t)) for t in np.unique(f[f >= 0])}

    def token_name(self, t): return self.vocab_.id_to_token[int(self.perm_[t])] if self.vocab_ is not None else int(self.perm_[t])

    def to_json(self, **kw): return S.dumps({**S.common_state(self), "estimators": [S.tree_state(e.tree_) for e in self.estimators_]}, **kw)

    @classmethod
    def from_json(cls, s):
        d = S.loads(s)
        f = S.restore_common(cls(**{**d["params"], "matcher": None}), d)
        f.estimators_ = []
        for td in d["estimators"]:
            t = f.make_tree(0)
            t.n_features_, t.vocab_, t.classes_, t.tree_ = f.n_features_, None, f.classes_, S.tree_arrays(td)
            f.estimators_.append(t)
        return f.pack()

    def rules(self, tree=0): return leaf_rules(self.estimators_[tree].tree_, self.token_name)
    def to_rules(self, tree=0, sort_by="n", max_rules=None): return format_rules(self.rules(tree), sort_by, max_rules)

    def feature_importances(self):
        imp = np.zeros(self.n_features_)
        for e in self.estimators_:
            t = e.tree_
            f, N, P = t["feature"], t["n_total"], t["n_pos"]
            inner = f >= 0
            gain = N[inner] * self.gini(N[inner], P[inner]) - N[t["left"][inner]] * self.gini(N[t["left"][inner]], P[t["left"][inner]]) - N[t["right"][inner]] * self.gini(N[t["right"][inner]], P[t["right"][inner]])
            np.add.at(imp, f[inner], gain / max(N[0], 1))
        imp /= max(len(self.estimators_), 1)
        return imp[self.inv_] if imp.sum() == 0 else imp[self.inv_] / imp.sum()

    @staticmethod
    def gini(N, P): return 1 - (P / N) ** 2 - ((N - P) / N) ** 2

    def predict_proba(self, X):
        indptr, indices = self.encode(X)
        p1 = _builder_cy.predict_forest(indptr, indices, self.feature_, self.left_, self.right_, self.p1_, self.offsets_, self.threads)
        return np.stack([1 - p1, p1], 1)

    def predict(self, X): return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]

    @property
    def n_nodes(self): return int(self.offsets_[-1])
