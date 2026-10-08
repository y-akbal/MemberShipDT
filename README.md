# MemberShipDT

this repo is for fitting decision trees (and random forests) when your rows are just *sets* of stuff,
not fixed columns. a row is like `["apple", "milk", "eggs"]`, another row is `["beer"]`. splits are
`token in X` vs `token not in X`, gini criterion, binary labels.

the builder is `_builder_cy.pyx` (cython, nogil, openmp predict). a numpy prototype was used to develop it
and the cython tree is tested against a dense brute-force gini reference.

## install

```
pip install -e .
```

needs numpy + cython + a C compiler with openmp.

the extension is compiled with `-march=native` by default (so the .so only runs on cpus like the one that built it).
for a portable build: `MEMBERSHIPDT_MARCH=portable pip install -e .`, or pick an arch: `MEMBERSHIPDT_MARCH=x86-64-v3`.

## use

```python
from membershipdt import MembershipDecisionTree, MembershipRandomForest, GridSearchCV

X = [["a", "b"], ["a"], ["b", "c"], ["c"]]
y = [1, 1, 0, 0]

t = MembershipDecisionTree(max_depth=5).fit(X, y)
print(t.to_text())
t.predict_proba([["a", "zzz"]])
print(t.to_rules())            # one line per leaf / region, e.g. if 'a' in X and 'b' not in X: p1=0.250 (n=4)
t.rules()                      # same thing as dicts: conditions, n, p1, leaf

f = MembershipRandomForest(n_estimators=100, max_features="sqrt", n_jobs=-1, random_state=0).fit(X, y)
f.predict_proba(X)
f.feature_importances()
print(f.to_rules(tree=0, max_rules=10))

s = f.to_json()                               # plain json string, vocab + params + all trees
f2 = MembershipRandomForest.from_json(s)      # same for MembershipDecisionTree

gs = GridSearchCV(MembershipRandomForest(n_estimators=50), {"max_depth": [4, 8, None], "min_samples_leaf": [1, 5]}, cv=5, scoring="auc").fit(X, y)
gs.best_params_, gs.best_estimator_
```

X can be a list of iterables of hashable tokens, a scipy sparse matrix (nonzero = present),
or a pre-encoded `(indptr, indices)` CSR pair (then pass `n_features=`).

## notes

- fit is O(nnz * depth), never O(V * n). per node we do one pass over the node's tokens, count
  `cnt[t]`, `pos[t]`, left child stats fall out as `N - cnt`, `P - pos`.
- token ids get reordered by frequency at fit time so hot counters sit together in cache. side effect: ties between equally good tokens go to the more frequent one.
- `max_features` samples from the tokens actually present in the node, not from all V
  (sampling from V on sparse data mostly hits tokens nobody has).
- bootstrap = multinomial counts as sample weights, the CSR is shared across trees, trees are fit
  in threads with the GIL released.
- forest predict walks all trees per sample in parallel (openmp), proba = mean of leaf fractions.
- unseen tokens at predict time are dropped, they can't matter anyway.
- single tree proba = leaf's weighted positives / weighted total. forest proba = mean of that over trees.
- binary labels only. `min_samples_leaf` / `min_samples_split` count weighted (bootstrapped) samples.
- the slow part for raw python tokens is the dict encoding (pure python, O(nnz)). the forest encodes
  once, not per tree. hand in `(indptr, indices)` or a scipy csr to skip it entirely.
- cv: `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`, scorers accuracy / auc / neg_log_loss / neg_brier.

## bench

`python bench/bench_tree.py` and `python bench/bench_forest.py` (after `pip install -e .`).
