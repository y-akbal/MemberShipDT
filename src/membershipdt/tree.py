from __future__ import annotations
import numpy as np
from .encoding import as_csr, frequency_perm, remap
from . import _builder_cy


def check_y(y):
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) > 2: raise ValueError("binary labels only")
    if len(classes) == 1: classes = np.array([0, 1]) if classes[0] in (0, 1) else np.array([classes[0], classes[0]])
    return classes, (y == classes[1]).astype(np.float64)


def leaf_rules(tree, name):
    feature, left, right, N, P = tree["feature"], tree["left"], tree["right"], tree["n_total"], tree["n_pos"]
    out, stack = [], [(0, ())]
    while stack:
        node, conds = stack.pop()
        if feature[node] < 0: out.append(dict(conditions=conds, n=float(N[node]), p1=float(P[node] / N[node]) if N[node] > 0 else 0.5, leaf=int(node)))
        else:
            t = name(int(feature[node]))
            stack.append((int(right[node]), conds + ((t, True),)))
            stack.append((int(left[node]), conds + ((t, False),)))
    return out


def format_rule(r):
    cond = " and ".join(f"{t!r} {'in' if inside else 'not in'} X" for t, inside in r["conditions"]) or "always"
    return f"if {cond}: p1={r['p1']:.3f} (n={r['n']:g})"


def format_rules(rules, sort_by="n", max_rules=None):
    rules = sorted(rules, key=lambda r: -r[sort_by]) if sort_by else list(rules)
    return "\n".join(format_rule(r) for r in rules[:max_rules]) + "\n"


class MembershipDecisionTree:
    def __init__(self, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features=None, random_state=None):
        self.max_depth, self.min_samples_leaf, self.min_samples_split = max_depth, min_samples_leaf, min_samples_split
        self.max_features, self.random_state = max_features, random_state

    def resolve_max_features(self, V):
        mf = self.max_features
        if mf is None or mf == "all": return 0
        if mf == "sqrt": return max(1, int(np.sqrt(V)))
        if mf == "log2": return max(1, int(np.log2(V))) if V > 1 else 1
        if isinstance(mf, float): return max(1, int(mf * V))
        return int(mf)

    PARAMS = ("max_depth", "min_samples_leaf", "min_samples_split", "max_features", "random_state")

    def get_params(self): return {k: getattr(self, k) for k in self.PARAMS}
    def clone(self, **over): return type(self)(**{**self.get_params(), **over})

    def prepare(self, X, y, sample_weight, n_features):
        indptr, indices, V, vocab = as_csr(X, n_features, True)
        V = max(V, n_features or 0)
        self.vocab_, self.n_features_ = vocab, V
        self.perm_, self.inv_ = frequency_perm(indices, V)
        indptr, indices = remap(indptr, indices, self.inv_)
        self.classes_, yb = check_y(y)
        w = np.ones(len(yb)) if sample_weight is None else np.asarray(sample_weight, np.float64)
        return indptr, indices, yb, w

    def fit(self, X, y, sample_weight=None, n_features=None):
        self.fit_encoded(*self.prepare(X, y, sample_weight, n_features))
        return self

    def fit_encoded(self, indptr, indices, yb, w):
        seed = np.random.SeedSequence(self.random_state).generate_state(1, np.uint64)[0] if self.random_state is not None else np.random.SeedSequence().generate_state(1, np.uint64)[0]
        md = (1 << 30) if self.max_depth is None else int(self.max_depth)
        self.tree_ = _builder_cy.build_tree(indptr, indices, yb, w, self.n_features_, md, int(self.min_samples_leaf), int(self.min_samples_split), self.resolve_max_features(self.n_features_), int(seed))
        return self

    def encode(self, X):
        indptr, indices = as_csr(X, self.vocab_ if self.vocab_ is not None else self.n_features_, False)[:2]
        return remap(indptr, indices, self.inv_) if getattr(self, "inv_", None) is not None else (indptr, indices)

    @property
    def feature_(self):
        f = self.tree_["feature"]
        return np.where(f >= 0, self.perm_[np.maximum(f, 0)], -1) if getattr(self, "perm_", None) is not None else f

    def apply(self, X):
        indptr, indices = self.encode(X)
        return self.apply_encoded(indptr, indices)

    def apply_encoded(self, indptr, indices):
        t = self.tree_
        return _builder_cy.apply(indptr, indices, t["feature"], t["left"], t["right"])

    def leaf_proba(self):
        t = self.tree_
        p1 = np.where(t["n_total"] > 0, t["n_pos"] / np.where(t["n_total"] > 0, t["n_total"], 1), 0.5)
        return np.stack([1 - p1, p1], 1)

    def predict_proba(self, X): return self.leaf_proba()[self.apply(X)]
    def predict(self, X): return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]

    @property
    def n_nodes(self): return len(self.tree_["feature"])

    @property
    def n_leaves(self): return int((self.tree_["feature"] < 0).sum())

    def token_name(self, t):
        t = int(self.perm_[t]) if getattr(self, "perm_", None) is not None else int(t)
        return self.vocab_.id_to_token[t] if self.vocab_ is not None else t

    def rules(self): return leaf_rules(self.tree_, self.token_name)
    def to_rules(self, sort_by="n", max_rules=None): return format_rules(self.rules(), sort_by, max_rules)

    def to_text(self, node=0, depth=0):
        t = self.tree_
        pad = "  " * depth
        if t["feature"][node] < 0: return f"{pad}leaf n={t['n_total'][node]:g} p1={t['n_pos'][node] / max(t['n_total'][node], 1):.3f}\n"
        name = self.token_name(int(t["feature"][node]))
        return f"{pad}{name!r} not in X:\n" + self.to_text(t["left"][node], depth + 1) + f"{pad}{name!r} in X:\n" + self.to_text(t["right"][node], depth + 1)
