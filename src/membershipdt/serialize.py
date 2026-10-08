import json
import numpy as np
from .encoding import Vocab

JSONABLE = (str, int, float, bool, type(None))


def tree_state(tree): return {k: np.asarray(v).tolist() for k, v in tree.items()}
def tree_arrays(d): return dict(feature=np.array(d["feature"], np.int32), left=np.array(d["left"], np.int32), right=np.array(d["right"], np.int32), n_total=np.array(d["n_total"], np.float64), n_pos=np.array(d["n_pos"], np.float64))


def plain(t): return t.item() if isinstance(t, np.generic) else t


def vocab_state(vocab):
    if vocab is None: return None
    toks = [plain(t) for t in vocab.id_to_token]
    bad = next((t for t in toks if not isinstance(t, JSONABLE)), None)
    if bad is not None: raise TypeError(f"token {bad!r} of type {type(bad).__name__} is not json serializable")
    return toks


def vocab_from(tokens):
    if tokens is None: return None
    v = Vocab()
    for t in tokens: v.add(t)
    return v


def common_state(est):
    return dict(format=1, type=type(est).__name__, params=est.get_params(), classes=np.asarray(est.classes_).tolist(), n_features=int(est.n_features_), vocab=vocab_state(est.vocab_), perm=np.asarray(est.perm_).tolist())


def restore_common(est, d):
    if d.get("type") != type(est).__name__: raise ValueError(f"json is a {d.get('type')}, not a {type(est).__name__}")
    est.classes_, est.n_features_, est.vocab_ = np.array(d["classes"]), int(d["n_features"]), vocab_from(d["vocab"])
    est.perm_ = np.array(d["perm"], np.int32)
    est.inv_ = np.empty_like(est.perm_)
    est.inv_[est.perm_] = np.arange(len(est.perm_), dtype=np.int32)
    return est


def dumps(d, **kw): return json.dumps(d, **kw)
def loads(s): return s if isinstance(s, dict) else json.loads(s)
