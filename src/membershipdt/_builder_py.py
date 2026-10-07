from __future__ import annotations
import numpy as np
from ._rng import XorShift64

TOL = 1e-12


def proxy(n, p): return 0.0 if n <= 0 else (p * p + (n - p) * (n - p)) / n


def gather(indptr, indices, idx):
    starts, lens = indptr[idx], indptr[idx + 1] - indptr[idx]
    total = int(lens.sum())
    if total == 0: return np.empty(0, np.int32), np.empty(0, np.int64)
    local = np.repeat(np.arange(len(idx)), lens)
    off = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(starts, lens)
    return indices[off], local


def choose_features(touched, k, rng):
    m = len(touched)
    if k <= 0 or k >= m: return touched
    t = touched.copy()
    for j in range(k):
        r = j + rng.below(m - j)
        t[j], t[r] = t[r], t[j]
    return t[:k]


def best_split(cand, cnt, pos, N, P, msl, tol):
    nr, pr = cnt[cand], pos[cand]
    nl, pl = N - nr, P - pr
    ok = (nr >= msl) & (nl >= msl) & (nr > 0) & (nl > 0)
    if not ok.any(): return -1, -np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(ok, (pr * pr + (nr - pr) ** 2) / np.where(nr > 0, nr, 1) + (pl * pl + (nl - pl) ** 2) / np.where(nl > 0, nl, 1), -np.inf)
    best = score.max()
    return int(cand[score >= best - tol].min()), float(best)


def build_tree(indptr, indices, y, w, V, max_depth, min_samples_leaf, min_samples_split, max_features, seed):
    samples = np.flatnonzero(w > 0).astype(np.int64)
    rng = XorShift64(seed)
    feature, left, right, n_total, n_pos = [], [], [], [], []
    stack = [(0, len(samples), 0, -1, 0)]
    while stack:
        start, end, depth, parent, side = stack.pop()
        node = len(feature)
        feature.append(-1); left.append(-1); right.append(-1)
        if parent >= 0: (right if side else left)[parent] = node
        idx = samples[start:end]
        wi = w[idx]
        N, P = float(wi.sum()), float((wi * y[idx]).sum())
        n_total.append(N); n_pos.append(P)
        if depth >= max_depth or P <= 0 or P >= N or N < min_samples_split or N < 2 * min_samples_leaf: continue
        toks, local = gather(indptr, indices, idx)
        if toks.size == 0: continue
        touched, first, inv = np.unique(toks, return_index=True, return_inverse=True)
        order = np.argsort(first, kind="stable")
        touched = touched[order]
        cnt = np.bincount(inv, weights=wi[local], minlength=len(touched))
        pos = np.bincount(inv, weights=(wi * y[idx])[local], minlength=len(touched))
        full_cnt, full_pos = np.zeros(V), np.zeros(V)
        full_cnt[touched], full_pos[touched] = cnt[order], pos[order]
        cand = choose_features(touched, max_features, rng)
        tol = TOL * N
        t, score = best_split(cand, full_cnt, full_pos, N, P, min_samples_leaf, tol)
        if t < 0 or score <= proxy(N, P) + tol: continue
        mask = np.zeros(len(idx), bool)
        mask[local[toks == t]] = True
        samples[start:end] = np.concatenate([idx[~mask], idx[mask]])
        mid = start + int((~mask).sum())
        feature[node] = t
        stack.append((mid, end, depth + 1, node, 1))
        stack.append((start, mid, depth + 1, node, 0))
    return dict(feature=np.array(feature, np.int32), left=np.array(left, np.int32), right=np.array(right, np.int32), n_total=np.array(n_total), n_pos=np.array(n_pos))


def apply(indptr, indices, feature, left, right, n_jobs=0):
    out = np.empty(len(indptr) - 1, np.int64)
    for i in range(len(out)):
        row, node = set(indices[indptr[i]:indptr[i + 1]].tolist()), 0
        while feature[node] >= 0: node = right[node] if feature[node] in row else left[node]
        out[i] = node
    return out


def predict_forest(indptr, indices, feature, left, right, p1, offsets, n_jobs=0):
    out = np.zeros(len(indptr) - 1)
    for k in range(len(offsets) - 1):
        f, l, r = feature[offsets[k]:offsets[k + 1]] , left[offsets[k]:offsets[k + 1]] - offsets[k], right[offsets[k]:offsets[k + 1]] - offsets[k]
        f = np.where(f >= 0, f, -1)
        out += p1[offsets[k] + apply(indptr, indices, f, l, r)]
    return out / (len(offsets) - 1)
