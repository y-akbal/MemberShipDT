import json
import numpy as np
import pytest
from membershipdt import MembershipDecisionTree, MembershipRandomForest, GridSearchCV
from conftest import random_sets


def roundtrip(est, X):
    s = est.to_json()
    assert isinstance(s, str)
    back = type(est).from_json(s)
    np.testing.assert_array_equal(back.predict_proba(X), est.predict_proba(X))
    np.testing.assert_array_equal(back.predict(X), est.predict(X))
    assert back.get_params() == est.get_params()
    assert type(est).from_json(json.loads(s)).to_json() == s
    return back


def test_tree_roundtrip_strings():
    rng = np.random.default_rng(0)
    X, y = random_sets(rng, 300, 30, 6)
    X = [[f"tok{t}" for t in row] for row in X]
    t = MembershipDecisionTree(max_depth=6, min_samples_leaf=2).fit(X, y)
    back = roundtrip(t, X)
    assert back.to_rules() == t.to_rules() and back.to_text() == t.to_text()
    assert back.predict_proba([["tok1", "never_seen"], []]).shape == (2, 2)


def test_tree_roundtrip_int_tokens_and_string_labels():
    X = [[1, 2], [1], [2, 3], [3], [1, 3]]
    t = MembershipDecisionTree().fit(X, ["yes", "yes", "no", "no", "yes"])
    back = roundtrip(t, X)
    assert list(back.classes_) == ["no", "yes"] and list(back.predict(X)) == list(t.predict(X))


def test_tree_roundtrip_preencoded_no_vocab():
    rng = np.random.default_rng(1)
    X, y = random_sets(rng, 200, 20, 5)
    idx = np.array([int(t) for row in X for t in sorted(set(row))], np.int32)
    csr = (np.cumsum([0] + [len(set(r)) for r in X]), idx)
    t = MembershipDecisionTree(max_depth=4).fit(csr, y, n_features=20)
    back = roundtrip(t, csr)
    assert back.vocab_ is None and back.n_features_ == 20


def test_forest_roundtrip():
    rng = np.random.default_rng(2)
    X, y = random_sets(rng, 400, 40, 6)
    X = [[f"w{t}" for t in row] for row in X]
    f = MembershipRandomForest(n_estimators=12, max_depth=5, random_state=3).fit(X, y)
    back = roundtrip(f, X)
    assert len(back.estimators_) == 12 and back.n_nodes == f.n_nodes
    np.testing.assert_array_equal(back.feature_, f.feature_)
    np.testing.assert_allclose(back.feature_importances(), f.feature_importances())
    assert back.to_rules(3) == f.to_rules(3)


def test_tuned_estimator_roundtrip():
    rng = np.random.default_rng(3)
    X, y = random_sets(rng, 500, 30, 6)
    gs = GridSearchCV(MembershipRandomForest(n_estimators=6, random_state=0), {"max_depth": [2, 5]}, cv=3, random_state=0).fit(X, y)
    back = MembershipRandomForest.from_json(gs.best_estimator_.to_json())
    assert back.max_depth == gs.best_params_["max_depth"]
    np.testing.assert_array_equal(back.predict_proba(X), gs.predict_proba(X))


def test_wrong_type_and_bad_tokens():
    t = MembershipDecisionTree().fit([["a"], ["b"]], [0, 1])
    with pytest.raises(ValueError): MembershipRandomForest.from_json(t.to_json())
    bad = MembershipDecisionTree().fit([[("x", 1)], [("y", 2)]], [0, 1])
    with pytest.raises(TypeError): bad.to_json()
