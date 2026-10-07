import numpy as np


def gini(n, p): return 0.0 if n == 0 else 1 - (p / n) ** 2 - ((n - p) / n) ** 2


def ref_tree(D, y, w, max_depth=1 << 30, msl=1, mss=2):
    feature, left, right, n_total, n_pos = [], [], [], [], []

    def grow(rows, depth):
        node = len(feature)
        feature.append(-1); left.append(-1); right.append(-1)
        N, P = w[rows].sum(), (w[rows] * y[rows]).sum()
        n_total.append(N); n_pos.append(P)
        if depth >= max_depth or P == 0 or P == N or N < mss or N < 2 * msl: return node
        best, best_t, parent = None, -1, gini(N, P)
        for t in range(D.shape[1]):
            m = D[rows, t] == 1
            nr, nl = w[rows][m].sum(), w[rows][~m].sum()
            if nr < msl or nl < msl or nr == 0 or nl == 0: continue
            pr, pl = (w[rows] * y[rows])[m].sum(), (w[rows] * y[rows])[~m].sum()
            g = (nr * gini(nr, pr) + nl * gini(nl, pl)) / N
            if best is None or g < best - 1e-9: best, best_t = g, t
        if best is None or best >= parent - 1e-9: return node
        feature[node] = best_t
        m = D[rows, best_t] == 1
        left[node] = grow(rows[~m], depth + 1)
        right[node] = grow(rows[m], depth + 1)
        return node

    grow(np.arange(len(y)), 0)
    return dict(feature=np.array(feature), left=np.array(left), right=np.array(right), n_total=np.array(n_total, float), n_pos=np.array(n_pos, float))


def ref_apply(D, tree):
    out = []
    for i in range(D.shape[0]):
        node = 0
        while tree["feature"][node] >= 0: node = tree["right"][node] if D[i, tree["feature"][node]] else tree["left"][node]
        out.append(node)
    return np.array(out)
