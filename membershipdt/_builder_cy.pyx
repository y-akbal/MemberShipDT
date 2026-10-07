# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, language_level=3
import numpy as np
from libc.stdlib cimport malloc, realloc, free
from libc.stdint cimport uint64_t, int64_t, int32_t
from libc.math cimport INFINITY

cdef double TOL = 1e-12

ctypedef struct Frame:
    int64_t start, end
    int32_t depth, parent, side


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
    while lo < hi:
        mid = (lo + hi) >> 1
        if indices[mid] < t: lo = mid + 1
        else: hi = mid
    return lo < indptr[s + 1] and indices[lo] == t


cdef inline double proxy(double n, double p) noexcept nogil: return 0.0 if n <= 0 else (p * p + (n - p) * (n - p)) / n


def build_tree(const int64_t[::1] indptr, const int32_t[::1] indices, const double[::1] y, const double[::1] w, int64_t V, int64_t max_depth, double min_samples_leaf, double min_samples_split, int64_t max_features, uint64_t seed):
    cdef int64_t[::1] samples = np.flatnonzero(np.asarray(w) > 0).astype(np.int64)
    cdef int64_t n = samples.shape[0], VV = V if V > 0 else 1
    cdef double* cnt = <double*>malloc(VV * sizeof(double))
    cdef double* pos = <double*>malloc(VV * sizeof(double))
    cdef double* score = <double*>malloc(VV * sizeof(double))
    cdef int64_t* mark = <int64_t*>malloc(VV * sizeof(int64_t))
    cdef int32_t* touched = <int32_t*>malloc(VV * sizeof(int32_t))
    cdef int64_t* tmp = <int64_t*>malloc((n if n > 0 else 1) * sizeof(int64_t))
    cdef int64_t cap = 64, nnodes = 0, scap = 64, sp = 0
    cdef int32_t* feature = <int32_t*>malloc(cap * sizeof(int32_t))
    cdef int32_t* left = <int32_t*>malloc(cap * sizeof(int32_t))
    cdef int32_t* right = <int32_t*>malloc(cap * sizeof(int32_t))
    cdef double* n_total = <double*>malloc(cap * sizeof(double))
    cdef double* n_pos = <double*>malloc(cap * sizeof(double))
    cdef Frame* stack = <Frame*>malloc(scap * sizeof(Frame))
    cdef Frame f
    cdef uint64_t rs = splitmix64(seed)
    cdef int64_t i, j, k, s, node, start, end, mid, nt, ncand, r, nleft, nright
    cdef int32_t t, best_t, tt
    cdef double wi, yi, N, P, nr, pr, nl, pl, sc, best, tol, parent_score
    if rs == 0: rs = 1
    with nogil:
        for i in range(VV): mark[i] = -1
        stack[0].start = 0; stack[0].end = n; stack[0].depth = 0; stack[0].parent = -1; stack[0].side = 0
        sp = 1
        while sp > 0:
            sp -= 1
            f = stack[sp]
            start = f.start; end = f.end
            if nnodes == cap:
                cap *= 2
                feature = <int32_t*>realloc(feature, cap * sizeof(int32_t))
                left = <int32_t*>realloc(left, cap * sizeof(int32_t))
                right = <int32_t*>realloc(right, cap * sizeof(int32_t))
                n_total = <double*>realloc(n_total, cap * sizeof(double))
                n_pos = <double*>realloc(n_pos, cap * sizeof(double))
            node = nnodes
            nnodes += 1
            feature[node] = -1; left[node] = -1; right[node] = -1
            if f.parent >= 0:
                if f.side: right[f.parent] = <int32_t>node
                else: left[f.parent] = <int32_t>node
            N = 0; P = 0
            for i in range(start, end):
                s = samples[i]
                N += w[s]; P += w[s] * y[s]
            n_total[node] = N; n_pos[node] = P
            if f.depth >= max_depth or P <= 0 or P >= N or N < min_samples_split or N < 2 * min_samples_leaf: continue
            nt = 0
            for i in range(start, end):
                s = samples[i]
                wi = w[s]; yi = wi * y[s]
                for j in range(indptr[s], indptr[s + 1]):
                    t = indices[j]
                    if mark[t] < 0:
                        mark[t] = node; touched[nt] = t; nt += 1
                        cnt[t] = 0; pos[t] = 0
                    cnt[t] += wi; pos[t] += yi
            if nt == 0: continue
            ncand = nt
            if max_features > 0 and max_features < nt:
                ncand = max_features
                for j in range(ncand):
                    r = j + <int64_t>(xorshift(&rs) % <uint64_t>(nt - j))
                    tt = touched[j]; touched[j] = touched[r]; touched[r] = tt
            tol = TOL * N
            best = -INFINITY
            for j in range(ncand):
                t = touched[j]
                nr = cnt[t]; pr = pos[t]; nl = N - nr; pl = P - pr
                if nr >= min_samples_leaf and nl >= min_samples_leaf and nr > 0 and nl > 0:
                    sc = (pr * pr + (nr - pr) * (nr - pr)) / nr + (pl * pl + (nl - pl) * (nl - pl)) / nl
                else: sc = -INFINITY
                score[t] = sc
                if sc > best: best = sc
            best_t = -1
            for j in range(ncand):
                t = touched[j]
                if score[t] >= best - tol and (best_t < 0 or t < best_t): best_t = t
            for j in range(nt): mark[touched[j]] = -1
            if best_t < 0 or best <= proxy(N, P) + tol: continue
            nleft = start; nright = 0
            for i in range(start, end):
                s = samples[i]
                if has_row(indptr, indices, s, best_t): tmp[nright] = s; nright += 1
                else: samples[nleft] = s; nleft += 1
            for i in range(nright): samples[nleft + i] = tmp[i]
            mid = nleft
            feature[node] = best_t
            if sp + 2 > scap:
                scap *= 2
                stack = <Frame*>realloc(stack, scap * sizeof(Frame))
            stack[sp].start = mid; stack[sp].end = end; stack[sp].depth = f.depth + 1; stack[sp].parent = <int32_t>node; stack[sp].side = 1
            stack[sp + 1].start = start; stack[sp + 1].end = mid; stack[sp + 1].depth = f.depth + 1; stack[sp + 1].parent = <int32_t>node; stack[sp + 1].side = 0
            sp += 2
    out = dict(feature=np.empty(nnodes, np.int32), left=np.empty(nnodes, np.int32), right=np.empty(nnodes, np.int32), n_total=np.empty(nnodes, np.float64), n_pos=np.empty(nnodes, np.float64))
    cdef int32_t[::1] of = out["feature"], ol = out["left"], orr = out["right"]
    cdef double[::1] ot = out["n_total"], op = out["n_pos"]
    for i in range(nnodes):
        of[i] = feature[i]; ol[i] = left[i]; orr[i] = right[i]; ot[i] = n_total[i]; op[i] = n_pos[i]
    free(cnt); free(pos); free(score); free(mark); free(touched); free(tmp)
    free(feature); free(left); free(right); free(n_total); free(n_pos); free(stack)
    return out


def apply(const int64_t[::1] indptr, const int32_t[::1] indices, const int32_t[::1] feature, const int32_t[::1] left, const int32_t[::1] right):
    cdef int64_t n = indptr.shape[0] - 1, i, node
    out = np.empty(n, np.int64)
    cdef int64_t[::1] o = out
    with nogil:
        for i in range(n):
            node = 0
            while feature[node] >= 0:
                node = right[node] if has_row(indptr, indices, i, feature[node]) else left[node]
            o[i] = node
    return out
