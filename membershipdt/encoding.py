from __future__ import annotations
import numpy as np

__all__ = ["Vocab", "encode", "as_csr"]


class Vocab:
    __slots__ = ("token_to_id", "id_to_token")

    def __init__(self): self.token_to_id, self.id_to_token = {}, []
    def __len__(self): return len(self.id_to_token)
    def get(self, tok, default=-1): return self.token_to_id.get(tok, default)

    def add(self, tok):
        i = self.token_to_id.get(tok)
        if i is None:
            i = len(self.id_to_token)
            self.token_to_id[tok] = i
            self.id_to_token.append(tok)
        return i


def encode(X, vocab, grow=True):
    indptr = np.empty(len(X) + 1, dtype=np.int64)
    indptr[0] = 0
    chunks, nnz, t2i, add = [], 0, vocab.token_to_id, vocab.add
    for r, row in enumerate(X):
        ids = sorted({add(t) for t in row}) if grow else sorted({t2i[t] for t in row if t in t2i})
        nnz += len(ids)
        indptr[r + 1] = nnz
        chunks.append(ids)
    indices = np.fromiter((i for ids in chunks for i in ids), dtype=np.int32, count=nnz)
    return indptr, indices


def normalize_csr(indptr, indices, n_features):
    indptr = np.ascontiguousarray(indptr, dtype=np.int64)
    indices = np.ascontiguousarray(indices, dtype=np.int32)
    n, V = indptr.shape[0] - 1, max(int(n_features), 1)
    if indices.size and (indices.min() < 0 or indices.max() >= V): raise ValueError("token id out of range")
    key = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr)) * V + indices
    if key.size == 0 or np.all(np.diff(key) > 0): return indptr, indices
    key = np.unique(key)
    new_indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(new_indptr, key // V + 1, 1)
    return np.cumsum(new_indptr), (key % V).astype(np.int32)


def as_csr(X, vocab, grow):
    if isinstance(X, tuple) and len(X) == 2 and hasattr(X[0], "__len__") and not isinstance(X[0], (list, set, frozenset)):
        indptr, indices = np.asarray(X[0]), np.asarray(X[1])
        V = (int(indices.max()) + 1 if indices.size else 0) if grow else vocab
        indptr, indices = normalize_csr(indptr, indices, V)
        return indptr, indices, V, None
    if hasattr(X, "tocsr") and hasattr(X, "shape"):
        Xc = X.tocsr()
        Xc.eliminate_zeros()
        V = Xc.shape[1] if grow else vocab
        indptr, indices = normalize_csr(Xc.indptr, Xc.indices, V)
        return indptr, indices, V, None
    vocab = Vocab() if vocab is None else vocab
    indptr, indices = encode(X, vocab, grow)
    return indptr, indices, len(vocab), vocab
