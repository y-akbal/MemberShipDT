import copy
import json
import warnings
import numpy as np
import pytest
from membershipdt import MembershipDecisionTree, MembershipRandomForest, Matcher, RegexMatcher, NGramMatcher, CallableMatcher, register, GridSearchCV
from membershipdt.matching import Resolver, MATCHERS
from conftest import random_sets

X = [["apple", "milk"], ["apple"], ["milk", "bread"], ["bread"], ["apple", "bread"], ["eggs"], ["milk", "eggs"], ["apple", "milk", "eggs"]]
y = [1, 1, 0, 0, 1, 0, 0, 1]


def test_abstract_contract():
    with pytest.raises(TypeError): Matcher()
    class M(Matcher):
        def match(self, token): return []
    assert M().bind(["a"]).candidates == ("a",)


def test_default_drops_unseen():
    t = MembershipDecisionTree().fit(X, y)
    assert t.predict_proba([["aple"]])[0, 1] == 0.0


def test_ngram_fixes_typos_top1():
    t = MembershipDecisionTree(matcher=NGramMatcher(n=2), match_threshold=0.3).fit(X, y)
    assert t.predict_proba([["aple"]])[0, 1] == 1.0
    assert t.predict_proba([["zzzzzz"]])[0, 1] == 0.0


def test_matcher_sees_only_split_tokens_and_is_not_mutated():
    seen = {}
    class Spy(Matcher):
        def build(self): seen["cands"] = self.candidates
        def match(self, token): return [(self.candidates[0], 1.0)]
    m = Spy()
    t = MembershipDecisionTree(matcher=m).fit(X, y)
    t.predict_proba([["unknown"]])
    assert set(seen["cands"]) == t.split_tokens() == {"apple"}
    assert m.candidates == () and not hasattr(m, "cands")


def test_contract_violations_raise():
    class Bad(Matcher):
        def match(self, token): return [("not_a_candidate", 1.0)]
    class Worse(Matcher):
        def match(self, token): return 42
    with pytest.raises(ValueError): MembershipDecisionTree(matcher=Bad()).fit(X, y).predict_proba([["q"]])
    with pytest.raises(TypeError): MembershipDecisionTree(matcher=Worse()).fit(X, y).predict_proba([["q"]])
    with pytest.raises(ValueError): MembershipDecisionTree(matcher=NGramMatcher(), match_mode="soft").fit(X, y).predict_proba([["q"]])


def test_threshold_and_union_mode():
    rng = np.random.default_rng(0)
    Xr, yr = random_sets(rng, 600, 12, 4)
    Xr = [[f"item_{t:02d}" for t in row] for row in Xr]
    f = MembershipRandomForest(n_estimators=10, random_state=0, matcher=NGramMatcher(n=2, k=5), match_mode="union", match_k=3, match_threshold=0.0).fit(Xr, yr)
    r = f.resolver()
    ids = r("item_0x")
    assert 1 < len(ids) <= 3
    f.match_mode = "top1"
    assert len(f.resolver()("item_0x")) == 1
    f.match_threshold = 1.01
    assert f.resolver()("item_0x") == []


def test_regex_matcher_and_cache():
    Xr = [[f"price_{i}", "cat"] if i % 2 else [f"price_{i}"] for i in range(40)]
    yr = [1 if i % 2 else 0 for i in range(40)]
    t = MembershipDecisionTree(matcher=RegexMatcher([(r"^cat_?\w*$", "cat")])).fit(Xr, yr)
    p = t.predict_proba([["cat_v2"], ["cat_v2"], ["catalog"], ["dog"]])[:, 1]
    assert p[0] == 1.0 and p[1] == 1.0 and p[2] == 1.0 and p[3] == 0.0
    assert t.resolver().calls == 3


def test_callable_matcher_and_custom_registered_json():
    cm = CallableMatcher(lambda tok, cands: [(c, 1.0) for c in cands if str(c).startswith(str(tok)[:2])][:1])
    t = MembershipDecisionTree(matcher=cm).fit(X, y)
    assert t.predict_proba([["apricot"]])[0, 1] == 1.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        back = MembershipDecisionTree.from_json(t.to_json())
    assert w and back.matcher is None

    @register
    class Prefix(Matcher):
        def __init__(self, n=2): self.n = n
        def match(self, tok): return [(c, 1.0) for c in self.candidates if str(c)[:self.n] == str(tok)[:self.n]][:1]
        def to_config(self): return dict(n=self.n)
    t2 = MembershipDecisionTree(matcher=Prefix(3)).fit(X, y)
    back = MembershipDecisionTree.from_json(t2.to_json())
    assert isinstance(back.matcher, Prefix) and back.matcher.n == 3
    assert back.predict_proba([["appetite"]])[0, 1] == 1.0
    del MATCHERS["Prefix"]


def test_builtin_matchers_json_roundtrip_forest():
    rng = np.random.default_rng(1)
    Xr, yr = random_sets(rng, 300, 20, 5)
    Xr = [[f"w{t}" for t in row] for row in Xr]
    f = MembershipRandomForest(n_estimators=5, random_state=0, matcher=NGramMatcher(n=2, k=2), match_mode="union", match_k=2, match_threshold=0.2).fit(Xr, yr)
    d = json.loads(f.to_json())
    assert d["params"]["matcher"] == {"type": "NGramMatcher", "config": {"n": 2, "k": 2, "pad": True}}
    back = MembershipRandomForest.from_json(d)
    q = [["w1x", "w7"], ["nope"]]
    np.testing.assert_array_equal(back.predict_proba(q), f.predict_proba(q))
    assert back.get_params()["match_k"] == 2


def test_matcher_survives_cv_clone():
    rng = np.random.default_rng(2)
    Xr, yr = random_sets(rng, 300, 15, 4)
    Xr = [[f"w{t}" for t in row] for row in Xr]
    gs = GridSearchCV(MembershipDecisionTree(matcher=NGramMatcher(n=2)), {"max_depth": [2, 4]}, cv=3, random_state=0).fit(Xr, yr)
    assert isinstance(gs.best_estimator_.matcher, NGramMatcher)
    assert gs.predict_proba([["w1z"]]).shape == (1, 2)


def test_preencoded_input_is_never_matched():
    rng = np.random.default_rng(3)
    Xr, yr = random_sets(rng, 100, 10, 4)
    idx = np.array([int(t) for row in Xr for t in sorted(set(row))], np.int32)
    csr = (np.cumsum([0] + [len(set(r)) for r in Xr]), idx)
    t = MembershipDecisionTree(matcher=NGramMatcher()).fit(csr, yr, n_features=10)
    assert t.resolver() is None and t.predict_proba(csr).shape == (100, 2)
