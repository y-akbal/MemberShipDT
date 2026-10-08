import numpy as np
import pytest
from membershipdt import MembershipRandomForest, MembershipDecisionTree
from conftest import random_sets


def test_proba_shape_and_sum():
    rng = np.random.default_rng(0)
    X, y = random_sets(rng, 300, 40, 6)
    f = MembershipRandomForest(n_estimators=20, random_state=0).fit(X, y)
    p = f.predict_proba(X)
    assert p.shape == (300, 2)
    np.testing.assert_allclose(p.sum(1), 1)
    assert set(f.predict(X)) <= {0, 1}
    assert f.predict_proba([["unseen"], []]).shape == (2, 2)


def test_deterministic():
    rng = np.random.default_rng(1)
    X, y = random_sets(rng, 200, 30, 5)
    a = MembershipRandomForest(n_estimators=10, random_state=7).fit(X, y)
    b = MembershipRandomForest(n_estimators=10, random_state=7).fit(X, y)
    np.testing.assert_array_equal(a.feature_, b.feature_)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def test_forest_beats_tree_on_noisy_data():
    rng = np.random.default_rng(2)
    X, y = random_sets(rng, 2000, 60, 8)
    Xt, yt = random_sets(np.random.default_rng(2), 2000, 60, 8)
    Xtr, ytr, Xte, yte = X[:1500], y[:1500], X[1500:], y[1500:]
    tree = MembershipDecisionTree().fit(Xtr, ytr)
    forest = MembershipRandomForest(n_estimators=50, random_state=0).fit(Xtr, ytr)
    acc_t, acc_f = (tree.predict(Xte) == yte).mean(), (forest.predict(Xte) == yte).mean()
    assert acc_f >= acc_t - 0.02


def test_single_tree_forest_matches_tree():
    rng = np.random.default_rng(3)
    X, y = random_sets(rng, 300, 30, 5)
    f = MembershipRandomForest(n_estimators=1, bootstrap=False, max_features=None, random_state=0).fit(X, y)
    t = MembershipDecisionTree().fit(X, y)
    np.testing.assert_array_equal(f.feature_, t.tree_["feature"])
    np.testing.assert_allclose(f.predict_proba(X), t.predict_proba(X))


def test_sample_weight_zero_rows_ignored():
    X = [["a"], ["a"], ["b"], ["b"], ["a", "b"]]
    y = [1, 1, 0, 0, 1]
    f = MembershipRandomForest(n_estimators=5, bootstrap=False, max_features=None, random_state=0).fit(X, y, sample_weight=[1, 1, 1, 1, 0])
    for e in f.estimators_: assert e.tree_["n_total"][0] == 4