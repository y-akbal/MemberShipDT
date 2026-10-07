import sys, time
import numpy as np
from membershipdt import MembershipRandomForest
from bench_tree import make


def run(backend, n, V, L, trees, depth, zipf=False):
    indptr, toks, y = make(n, V, L, zipf=zipf)
    f = MembershipRandomForest(n_estimators=trees, max_depth=depth, backend=backend, random_state=0)
    t0 = time.perf_counter(); f.fit((indptr, toks), y, n_features=V); fit = time.perf_counter() - t0
    t0 = time.perf_counter(); p = f.predict_proba((indptr, toks)); pred = time.perf_counter() - t0
    acc = ((p[:, 1] > 0.5) == y).mean()
    print(f"{backend:7s} {'zipf' if zipf else 'unif'} n={n:>9,} V={V:>8,} nnz={len(toks):>11,} trees={trees:4d} depth={depth:3d} nodes={f.n_nodes:8,}  fit={fit:7.2f}s  predict={pred:6.2f}s  train_acc={acc:.3f}", flush=True)


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "auto"
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    for n, V, L, T, d, z in [(20_000, 1_000, 10, 100, 12, 0), (100_000, 10_000, 10, 100, 16, 0), (1_000_000, 100_000, 20, 100, 20, 0), (1_000_000, 1_000_000, 20, 50, 20, 0), (1_000_000, 1_000_000, 20, 50, 20, 1)]:
        if n * scale < 1: continue
        run(backend, int(n * scale), V, L, T, d, bool(z))
