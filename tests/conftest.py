import numpy as np
import pytest
from membershipdt.tree import BACKENDS


def random_sets(rng, n, V, max_len, p_pos=None):
    X = [list(rng.choice(V, size=rng.integers(0, max_len + 1), replace=True)) for _ in range(n)]
    if p_pos is None:
        secret = rng.choice(V, min(3, V), replace=False)
        y = np.array([int(any(t in row for t in secret)) ^ int(rng.random() < 0.15) for row in X])
    else:
        y = (rng.random(n) < p_pos).astype(int)
    return X, y


def dense(X, V):
    D = np.zeros((len(X), V), np.int8)
    for i, row in enumerate(X): D[i, list(set(row))] = 1
    return D


@pytest.fixture(params=[k for k, v in BACKENDS.items() if v is not None])
def backend(request): return request.param
