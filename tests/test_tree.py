import numpy as np
import pytest
from membershipdt import MembershipDecisionTree
from reference import ref_tree, ref_apply
from conftest import random_sets, dense


def same_tree(a, b):
    assert len(a["feature"]) == len(b["feature"]), (len(a["feature"]), len(b["feature"]))
    for k in ("feature", "left", "right"): np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    for k in ("n_total", "n_pos"): np.testing.assert_allclose(a[k], b[k], err_msg=k)


@pytest.mark.parametrize("seed", range(40))
def test_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    n, V, L = int(rng.integers(5, 120)), int(rng.integers(2, 25)), int(rng.integers(1, 8))
    X, y = random_sets(rng, n, V, L)
    md, msl = int(rng.integers(1, 8)), int(rng.integers(1, 4))
    w = rng.integers(0, 4, n).astype(float) if seed % 3 == 0 else np.ones(n)
    if w.sum() == 0: w[0] = 1
    ids = (np.arange(n), np.array([int(t) for row in X for t in sorted(set(row))]))
    indptr = np.cumsum([0] + [len(set(r)) for r in X])
    t = MembershipDecisionTree(max_depth=md, min_samples_leaf=msl)
    t.n_features_, t.vocab_, t.classes_ = V, None, np.array([0, 1])
    t.fit_encoded(indptr, ids[1].astype(np.int32), y.astype(float), w)
    D = dense(X, V)
    ref = ref_tree(D, y.astype(float), w, md, msl)
    same_tree(t.tree_, ref)
    np.testing.assert_array_equal(t.apply_encoded(indptr, ids[1].astype(np.int32)), ref_apply(D, ref))


def test_python_api_roundtrip():
    X = [["a", "b"], ["a"], ["b", "c"], ["c"], ["a", "c"], ["d"], ["b", "d"], ["a", "b", "d"]]
    y = [1, 1, 0, 0, 1, 0, 0, 1]
    t = MembershipDecisionTree().fit(X, y)
    assert t.n_nodes == 3 and t.tree_["feature"][0] == 0
    np.testing.assert_array_equal(t.predict(X), y)
    p = t.predict_proba([["a", "unseen"], ["unseen"], []])
    np.testing.assert_allclose(p[:, 1], [1, 0, 0])
    assert "'a' in X" in t.to_text()


def test_string_labels():
    X = [["x"], ["y"], ["x", "y"], ["z"]]
    t = MembershipDecisionTree().fit(X, ["cat", "dog", "cat", "dog"])
    assert list(t.classes_) == ["cat", "dog"]
    assert list(t.predict(X)) == ["cat", "dog", "cat", "dog"]


@pytest.mark.parametrize("X,y", [
    ([[], [], []], [0, 1, 0]),
    ([["a"], ["a"], ["a"]], [0, 1, 1]),
    ([["a"], ["b"]], [1, 1]),
    ([["a", "a", "a"], ["a", "b"]], [0, 1]),
    ([[]], [1]),
])
def test_degenerate(X, y):
    t = MembershipDecisionTree().fit(X, y)
    assert t.n_leaves >= 1
    assert t.predict_proba(X).shape == (len(X), 2)


def test_token_in_every_row_never_split():
    X = [["a", "b"], ["a"], ["a", "c"], ["a", "b", "c"]]
    t = MembershipDecisionTree().fit(X, [1, 0, 1, 0])
    assert 0 not in t.tree_["feature"]


def test_min_samples_leaf():
    rng = np.random.default_rng(0)
    X, y = random_sets(rng, 200, 30, 6)
    t = MembershipDecisionTree(min_samples_leaf=7).fit(X, y)
    leaves = t.tree_["feature"] < 0
    assert t.tree_["n_total"][leaves].min() >= 7


def test_max_features_is_deterministic_and_restricts():
    rng = np.random.default_rng(1)
    X, y = random_sets(rng, 300, 50, 8)
    a = MembershipDecisionTree(max_features=3, random_state=5).fit(X, y)
    b = MembershipDecisionTree(max_features=3, random_state=5).fit(X, y)
    c = MembershipDecisionTree(max_features=3, random_state=6).fit(X, y)
    np.testing.assert_array_equal(a.tree_["feature"], b.tree_["feature"])
    assert not np.array_equal(a.tree_["feature"], c.tree_["feature"]) or a.n_nodes != c.n_nodes


def test_scipy_sparse_input():
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(2)
    X, y = random_sets(rng, 100, 20, 5)
    M = sp.csr_matrix(dense(X, 20).astype(float))
    t1 = MembershipDecisionTree(max_depth=4).fit(M, y)
    idx = np.array([int(t) for row in X for t in sorted(set(row))], np.int32)
    t2 = MembershipDecisionTree(max_depth=4).fit((np.cumsum([0] + [len(set(r)) for r in X]), idx), y, n_features=20)
    np.testing.assert_array_equal(t1.feature_, t2.feature_)
    assert set(t1.feature_[t1.feature_ >= 0]) <= set(range(20))
    np.testing.assert_array_equal(t1.predict(M), t2.predict_proba(M).argmax(1))