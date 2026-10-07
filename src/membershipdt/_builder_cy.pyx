# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, language_level=3
import numpy as np
from libc.stdlib cimport malloc, realloc, free
from libc.stdint cimport uint64_t, int64_t, int32_t
from libc.math cimport INFINITY
from cython.parallel cimport prange
import os

cdef double TOL = 1e-12

cdef class Stack:
    cdef int64_t* start
    cdef int64_t* end
    cdef int32_t* depth
    cdef int32_t* parent
    cdef int32_t* side
    cdef int64_t n, cap

    def __cinit__(self, int64_t cap=64):
        self.n, self.cap = 0, cap
        self.start = <int64_t*>malloc(cap * sizeof(int64_t)); self.end = <int64_t*>malloc(cap * sizeof(int64_t))
        self.depth = <int32_t*>malloc(cap * sizeof(int32_t)); self.parent = <int32_t*>malloc(cap * sizeof(int32_t)); self.side = <int32_t*>malloc(cap * sizeof(int32_t))

    def __dealloc__(self): free(self.start); free(self.end); free(self.depth); free(self.parent); free(self.side)

    cdef void push(self, int64_t start, int64_t end, int32_t depth, int32_t parent, int32_t side) noexcept nogil:
        if self.n == self.cap:
            self.cap *= 2
            self.start = <int64_t*>realloc(self.start, self.cap * sizeof(int64_t)); self.end = <int64_t*>realloc(self.end, self.cap * sizeof(int64_t))
            self.depth = <int32_t*>realloc(self.depth, self.cap * sizeof(int32_t)); self.parent = <int32_t*>realloc(self.parent, self.cap * sizeof(int32_t)); self.side = <int32_t*>realloc(self.side, self.cap * sizeof(int32_t))
        self.start[self.n] = start; self.end[self.n] = end; self.depth[self.n] = depth; self.parent[self.n] = parent; self.side[self.n] = side
        self.n += 1


cdef class Nodes:
    cdef int32_t* feature
    cdef int32_t* left
    cdef int32_t* right
    cdef double* n_total
    cdef double* n_pos
    cdef int64_t n, cap

    def __cinit__(self, int64_t cap=64):
        self.n, self.cap = 0, cap
        self.feature = <int32_t*>malloc(cap * sizeof(int32_t)); self.left = <int32_t*>malloc(cap * sizeof(int32_t)); self.right = <int32_t*>malloc(cap * sizeof(int32_t))
        self.n_total = <double*>malloc(cap * sizeof(double)); self.n_pos = <double*>malloc(cap * sizeof(double))

    def __dealloc__(self): free(self.feature); free(self.left); free(self.right); free(self.n_total); free(self.n_pos)

    cdef int64_t add(self, int32_t parent, int32_t side, double N, double P) noexcept nogil:
        if self.n == self.cap:
            self.cap *= 2
            self.feature = <int32_t*>realloc(self.feature, self.cap * sizeof(int32_t)); self.left = <int32_t*>realloc(self.left, self.cap * sizeof(int32_t)); self.right = <int32_t*>realloc(self.right, self.cap * sizeof(int32_t))
            self.n_total = <double*>realloc(self.n_total, self.cap * sizeof(double)); self.n_pos = <double*>realloc(self.n_pos, self.cap * sizeof(double))
        cdef int64_t node = self.n
        self.feature[node] = -1; self.left[node] = -1; self.right[node] = -1; self.n_total[node] = N; self.n_pos[node] = P
        if parent >= 0:
            if side: self.right[parent] = <int32_t>node
            else: self.left[parent] = <int32_t>node
        self.n += 1
        return node

    def arrays(self):
        out = dict(feature=np.empty(self.n, np.int32), left=np.empty(self.n, np.int32), right=np.empty(self.n, np.int32), n_total=np.empty(self.n, np.float64), n_pos=np.empty(self.n, np.float64))
        cdef int32_t[::1] of = out["feature"], ol = out["left"], orr = out["right"]
        cdef double[::1] ot = out["n_total"], op = out["n_pos"]
        cdef int64_t i
        for i in range(self.n):
            of[i] = self.feature[i]; ol[i] = self.left[i]; orr[i] = self.right[i]; ot[i] = self.n_total[i]; op[i] = self.n_pos[i]
        return out


cdef class Workspace:
    cdef double* cnt
    cdef double* pos
    cdef double* score
    cdef int32_t* mark
    cdef int32_t* touched
    cdef int64_t* tmp
    cdef int64_t V

    def __cinit__(self, int64_t V, int64_t n):
        self.V = V
        self.cnt = <double*>malloc(V * sizeof(double)); self.pos = <double*>malloc(V * sizeof(double)); self.score = <double*>malloc(V * sizeof(double))
        self.mark = <int32_t*>malloc(V * sizeof(int32_t)); self.touched = <int32_t*>malloc(V * sizeof(int32_t)); self.tmp = <int64_t*>malloc(n * sizeof(int64_t))
        cdef int64_t i
        for i in range(V): self.mark[i] = -1

    def __dealloc__(self): free(self.cnt); free(self.pos); free(self.score); free(self.mark); free(self.touched); free(self.tmp)


cdef inline uint64_t splitmix64(uint64_t x) noexcept nogil:
    x += <uint64_t>0x9E3779B97F4A7C15
    x = (x ^ (x >> 30)) * <uint64_t>0xBF58476D1CE4E5B9
    x = (x ^ (x >> 27)) * <uint64_t>0x94D049BB133111EB
    return x ^ (x >> 31)


cdef inline uint64_t xorshift(uint64_t* s) noexcept nogil:
    cdef uint64_t x = s[0]
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    s[0] = x
    return x



cdef inline bint has_row(const int64_t[::1] indptr, const int32_t[::1] indices, int64_t s, int32_t t) noexcept nogil:
    cdef int64_t lo = indptr[s], hi = indptr[s + 1], mid
    if hi - lo <= 24:
        while lo < hi and indices[lo] < t: lo += 1
        return lo < hi and indices[lo] == t
    hi -= 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if indices[mid] < t: lo = mid + 1
        else: hi = mid
    return indices[lo] == t


cdef inline int64_t walk(const int64_t[::1] indptr, const int32_t[::1] indices, const int32_t[::1] feature, const int32_t[::1] left, const int32_t[::1] right, int64_t s, int64_t node) noexcept nogil:
    while feature[node] >= 0:
        node = right[node] if has_row(indptr, indices, s, feature[node]) else left[node]
    return node


cdef int threads(int n_jobs) noexcept:
    return os.cpu_count() or 1 if n_jobs <= 0 else n_jobs


cdef inline double proxy(double n, double p) noexcept nogil: return 0.0 if n <= 0 else (p * p + (n - p) * (n - p)) / n


def build_tree(const int64_t[::1] indptr, const int32_t[::1] indices, const double[::1] y, const double[::1] w, int64_t V, int64_t max_depth, double min_samples_leaf, double min_samples_split, int64_t max_features, uint64_t seed):
    cdef int64_t[::1] samples = np.flatnonzero(np.asarray(w) > 0).astype(np.int64)
    cdef int64_t n = samples.shape[0]
    cdef Workspace ws = Workspace(V if V > 0 else 1, n if n > 0 else 1)
    cdef Stack stack = Stack()
    cdef Nodes nodes = Nodes()
    cdef uint64_t rs = splitmix64(seed)
    cdef int64_t i, j, s, node, start, end, mid, nt, ncand, r, nleft, nright, depth, jbest
    cdef int32_t t, best_t, tt, parent, side
    cdef double wi, yi, N, P, nr, pr, nl, pl, sc, best, tol, dtmp
    if rs == 0: rs = 1
    with nogil:
        stack.push(0, n, 0, -1, 0)
        while stack.n > 0:
            stack.n -= 1
            start = stack.start[stack.n]; end = stack.end[stack.n]; depth = stack.depth[stack.n]; parent = stack.parent[stack.n]; side = stack.side[stack.n]
            N = 0; P = 0
            for i in range(start, end):
                s = samples[i]
                N += w[s]; P += w[s] * y[s]
            node = nodes.add(parent, side, N, P)
            if depth >= max_depth or P <= 0 or P >= N or N < min_samples_split or N < 2 * min_samples_leaf: continue
            nt = 0
            for i in range(start, end):
                s = samples[i]
                wi = w[s]; yi = wi * y[s]
                for j in range(indptr[s], indptr[s + 1]):
                    t = indices[j]
                    r = ws.mark[t]
                    if r < 0:
                        r = nt; ws.mark[t] = <int32_t>nt; ws.touched[nt] = t; nt += 1
                        ws.cnt[r] = 0; ws.pos[r] = 0
                    ws.cnt[r] += wi; ws.pos[r] += yi
            for j in range(nt): ws.mark[ws.touched[j]] = -1
            if nt == 0: continue
            ncand = nt
            if max_features > 0 and max_features < nt:
                ncand = max_features
                for j in range(ncand):
                    r = j + <int64_t>(xorshift(&rs) % <uint64_t>(nt - j))
                    tt = ws.touched[j]; ws.touched[j] = ws.touched[r]; ws.touched[r] = tt
                    dtmp = ws.cnt[j]; ws.cnt[j] = ws.cnt[r]; ws.cnt[r] = dtmp
                    dtmp = ws.pos[j]; ws.pos[j] = ws.pos[r]; ws.pos[r] = dtmp
            tol = TOL * N
            best = -INFINITY
            for j in range(ncand):
                nr = ws.cnt[j]; pr = ws.pos[j]; nl = N - nr; pl = P - pr
                if nr >= min_samples_leaf and nl >= min_samples_leaf and nr > 0 and nl > 0:
                    sc = (pr * pr + (nr - pr) * (nr - pr)) / nr + (pl * pl + (nl - pl) * (nl - pl)) / nl
                else: sc = -INFINITY
                ws.score[j] = sc
                if sc > best: best = sc
            best_t = -1
            for j in range(ncand):
                t = ws.touched[j]
                if ws.score[j] >= best - tol and (best_t < 0 or t < best_t): best_t = t
            if best_t < 0 or best <= proxy(N, P) + tol: continue
            nleft = start; nright = 0
            for i in range(start, end):
                s = samples[i]
                if has_row(indptr, indices, s, best_t): ws.tmp[nright] = s; nright += 1
                else: samples[nleft] = s; nleft += 1
            for i in range(nright): samples[nleft + i] = ws.tmp[i]
            nodes.feature[node] = best_t
            stack.push(nleft, end, <int32_t>(depth + 1), <int32_t>node, 1)
            stack.push(start, nleft, <int32_t>(depth + 1), <int32_t>node, 0)
    return nodes.arrays()


def apply(const int64_t[::1] indptr, const int32_t[::1] indices, const int32_t[::1] feature, const int32_t[::1] left, const int32_t[::1] right, int n_jobs=0):
    cdef int64_t n = indptr.shape[0] - 1, i
    cdef int nth = threads(n_jobs)
    out = np.empty(n, np.int64)
    cdef int64_t[::1] o = out
    for i in prange(n, nogil=True, schedule="static", num_threads=nth):
        o[i] = walk(indptr, indices, feature, left, right, i, 0)
    return out


def predict_forest(const int64_t[::1] indptr, const int32_t[::1] indices, const int32_t[::1] feature, const int32_t[::1] left, const int32_t[::1] right, const double[::1] p1, const int64_t[::1] offsets, int n_jobs=0):
    cdef int64_t n = indptr.shape[0] - 1, T = offsets.shape[0] - 1, i, k
    cdef int nth = threads(n_jobs)
    cdef double acc
    out = np.empty(n, np.float64)
    cdef double[::1] o = out
    for i in prange(n, nogil=True, schedule="static", num_threads=nth):
        acc = 0
        for k in range(T): acc = acc + p1[walk(indptr, indices, feature, left, right, i, offsets[k])]
        o[i] = acc / T
    return out


cdef extern from "stdlib.h":
    void qsort(void* base, size_t n, size_t size, int (*cmp)(const void*, const void*) noexcept nogil) noexcept nogil


cdef int cmp_i32(const void* a, const void* b) noexcept nogil:
    return (<int32_t*>a)[0] - (<int32_t*>b)[0]


cdef inline void sort_row(int32_t* a, int64_t n) noexcept nogil:
    cdef int64_t i, j
    cdef int32_t v
    if n > 64:
        qsort(a, n, sizeof(int32_t), cmp_i32)
        return
    for i in range(1, n):
        v = a[i]; j = i - 1
        while j >= 0 and a[j] > v:
            a[j + 1] = a[j]; j -= 1
        a[j + 1] = v


def remap(const int64_t[::1] indptr, const int32_t[::1] indices, const int32_t[::1] inv, int n_jobs=0):
    cdef int64_t n = indptr.shape[0] - 1, i, j
    cdef int nth = threads(n_jobs)
    out = np.empty(indices.shape[0], np.int32)
    cdef int32_t[::1] o = out
    for i in prange(n, nogil=True, schedule="static", num_threads=nth):
        for j in range(indptr[i], indptr[i + 1]): o[j] = inv[indices[j]]
        sort_row(&o[indptr[i]], indptr[i + 1] - indptr[i])
    return out
