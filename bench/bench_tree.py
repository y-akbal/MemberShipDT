import sys, time
import numpy as np
from membershipdt import MembershipDecisionTree


def make(n, V, L, seed=0):
    rng = np.random.default_rng(seed)
    lens = rng.integers(1, 2 * L, n)
    indptr = np.concatenate([[0], np.cumsum(lens)])
    toks = rng.integers(0, V, lens.sum()).astype(np.int32)
    key = np.repeat(np.arange(n, dtype=np.int64), lens) * V + toks
    key = np.unique(key)
    indptr = np.zeros(n + 1, np.int64); np.add.at(indptr, key // V + 1, 1); indptr = np.cumsum(indptr)
    toks = (key % V).astype(np.int32)
    secret = rng.choice(V, 5, replace=False)
    hit = np.zeros(n, bool)
    for s in secret: hit |= np.add.reduceat(np.concatenate([toks == s, [False]]), indptr[:-1])[:n].astype(bool) & (np.diff(indptr) > 0)
    y = (hit ^ (rng.random(n) < 0.1)).astype(float)
    return indptr, toks, y


def run(backend, n, V, L, depth):
    indptr, toks, y = make(n, V, L)
    t = MembershipDecisionTree(max_depth=depth, backend=backend)
    t.n_features_, t.vocab_, t.classes_ = V, None, np.array([0, 1])
    t0 = time.perf_counter(); t.fit_encoded(indptr, toks, y, np.ones(n)); fit = time.perf_counter() - t0
    t0 = time.perf_counter(); t.apply_encoded(indptr, toks); pred = time.perf_counter() - t0
    print(f"{backend:7s} n={n:>9,} V={V:>8,} nnz={len(toks):>11,} depth={depth:3d} nodes={t.n_nodes:6d}  fit={fit:7.2f}s  predict={pred:6.2f}s", flush=True)


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "auto"
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    for n, V, L, d in [(20_000, 1_000, 10, 8), (100_000, 10_000, 10, 12), (100_000, 100_000, 10, 12), (1_000_000, 100_000, 20, 16), (1_000_000, 1_000_000, 20, 16)]:
        if n * scale < 1: continue
        run(backend, int(n * scale), V, L, d)
