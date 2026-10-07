import numpy as np
import pytest
from membershipdt import MembershipDecisionTree, MembershipRandomForest, GridSearchCV, RandomizedSearchCV, cross_val_score, stratified_folds, take_rows
from membershipdt.cv import auc, accuracy, neg_log_loss
from conftest import random_sets


def test_take_rows():
    indptr, indices = np.array([0, 2, 2, 5, 6]), np.array([1, 3, 0, 2, 4, 7], np.int32)
    ip, ix = take_rows(indptr, indices, [3, 1, 2])
    np.testing.assert_array_equal(ip, [0, 1, 1, 4])
    np.testing.assert_array_equal(ix, [7, 0, 2, 4])
    ip, ix = take_rows(indptr, indices, [1])
    assert ip.tolist() == [0, 0] and ix.size == 0


def test_stratified_folds_cover_everything():
    y = np.array([0] * 30 + [1] * 10)
    folds = stratified_folds(y, 5, np.random.default_rng(0))
    assert len(folds) == 5
    all_test = np.sort(np.concatenate([te for _, te in folds]))
    np.testing.assert_array_equal(all_test, np.arange(40))
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0
        assert y[te].sum() == 2


def test_auc_against_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y, p = (rng.random(300) < 0.4).astype(float), np.round(rng.random(300), 1)
    assert abs(auc(y, p) - sk.roc_auc_score(y, p)) < 1e-12
    assert abs(neg_log_loss(y, p) + sk.log_loss(y, p)) < 1e-9


def test_cross_val_score_shapes(backend):
    rng = np.random.default_rng(0)
    X, y = random_sets(rng, 400, 40, 6)
    s = cross_val_score(MembershipDecisionTree(max_depth=4, backend=backend), X, y, cv=4, scoring="auc", random_state=0)
    assert s.shape == (4,) and np.all((s >= 0) & (s <= 1))


def test_grid_search_finds_reasonable_depth(backend):
    rng = np.random.default_rng(1)
    X, y = random_sets(rng, 1500, 50, 8)
    gs = GridSearchCV(MembershipDecisionTree(backend=backend), {"max_depth": [1, 3, 6, None], "min_samples_leaf": [1, 10]}, cv=3, scoring="accuracy", random_state=0).fit(X, y)
    assert len(gs.cv_results_) == 8
    assert gs.best_params_["max_depth"] != 1
    assert gs.best_score_ == max(r["mean"] for r in gs.cv_results_)
    assert gs.predict(X).shape == (1500,)
    assert gs.predict_proba([["unseen"]]).shape == (1, 2)


def test_randomized_search_and_forest(backend):
    rng = np.random.default_rng(2)
    X, y = random_sets(rng, 600, 40, 6)
    rs = RandomizedSearchCV(MembershipRandomForest(n_estimators=10, backend=backend, random_state=0), {"max_depth": [2, 4, 8, None], "max_features": ["sqrt", 3, None], "min_samples_leaf": [1, 3, 5]}, n_iter=5, cv=3, scoring="neg_log_loss", random_state=0).fit(X, y)
    assert len(rs.cv_results_) == 5
    assert hasattr(rs, "best_estimator_") and rs.best_estimator_.n_estimators == 10
    assert set(rs.predict(X)) <= {0, 1}


def test_cv_is_deterministic(backend):
    rng = np.random.default_rng(3)
    X, y = random_sets(rng, 300, 30, 5)
    a = GridSearchCV(MembershipDecisionTree(backend=backend), {"max_depth": [2, 4]}, cv=3, random_state=5).fit(X, y)
    b = GridSearchCV(MembershipDecisionTree(backend=backend), {"max_depth": [2, 4]}, cv=3, random_state=5).fit(X, y)
    assert [r["mean"] for r in a.cv_results_] == [r["mean"] for r in b.cv_results_]


def test_remap_keeps_token_names(backend):
    X = [["rare", "common"], ["common"], ["common", "mid"], ["mid"], ["common", "mid", "rare"]]
    y = [1, 0, 1, 1, 1]
    t = MembershipDecisionTree(backend=backend).fit(X, y)
    txt = t.to_text()
    assert "'common'" in txt or "'mid'" in txt or "'rare'" in txt
    assert t.inv_[t.vocab_.get("common")] == 0
    np.testing.assert_array_equal(t.predict(X), y)
    f = MembershipRandomForest(n_estimators=5, bootstrap=False, max_features=None, backend=backend).fit(X, y)
    np.testing.assert_array_equal(f.predict(X), y)
    imp = f.feature_importances()
    assert imp.shape == (3,) and abs(imp.sum() - 1) < 1e-9
