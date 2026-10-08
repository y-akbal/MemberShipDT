from __future__ import annotations
import copy
import re
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict

__all__ = ["Matcher", "RegexMatcher", "NGramMatcher", "CallableMatcher", "Resolver", "MATCHERS", "register"]

MATCHERS = {}


def register(cls): MATCHERS[cls.__name__] = cls; return cls


class Matcher(ABC):
    candidates = ()

    def bind(self, candidates):
        m = copy.copy(self)
        m.candidates = tuple(candidates)
        m.build()
        return m

    def build(self): pass

    @abstractmethod
    def match(self, token) -> list[tuple[object, float]]: ...

    def to_config(self) -> dict | None: return None

    @classmethod
    def from_config(cls, config): return cls(**config)

    def __repr__(self): return f"{type(self).__name__}({self.to_config() or ''})"


@register
class RegexMatcher(Matcher):
    def __init__(self, rules, flags=0):
        self.rules, self.flags = [(p, t) for p, t in rules], int(flags)
        self.compiled = [(re.compile(p, self.flags), t) for p, t in self.rules]

    def match(self, token):
        s = token if isinstance(token, str) else str(token)
        return [(t, 1.0) for rx, t in self.compiled if rx.search(s)][:1]

    def to_config(self): return dict(rules=[list(r) for r in self.rules], flags=self.flags)


@register
class NGramMatcher(Matcher):
    def __init__(self, n=3, k=3, pad=True):
        self.n, self.k, self.pad = int(n), int(k), bool(pad)

    def grams(self, token):
        s = token if isinstance(token, str) else str(token)
        s = f"{'#' * (self.n - 1)}{s}{'#' * (self.n - 1)}" if self.pad else s
        return {s[i:i + self.n] for i in range(max(len(s) - self.n + 1, 1))}

    def build(self):
        self.sizes, self.index = [], defaultdict(list)
        for i, c in enumerate(self.candidates):
            g = self.grams(c)
            self.sizes.append(len(g))
            for x in g: self.index[x].append(i)

    def match(self, token):
        g = self.grams(token)
        hits = defaultdict(int)
        for x in g:
            for i in self.index.get(x, ()): hits[i] += 1
        scored = sorted(((h / (len(g) + self.sizes[i] - h), i) for i, h in hits.items()), key=lambda p: (-p[0], p[1]))
        return [(self.candidates[i], s) for s, i in scored[:self.k]]

    def to_config(self): return dict(n=self.n, k=self.k, pad=self.pad)


class CallableMatcher(Matcher):
    def __init__(self, fn): self.fn = fn
    def match(self, token): return self.fn(token, self.candidates)


class Resolver:
    def __init__(self, matcher, candidates, name_to_id, mode="top1", threshold=0.0, k=1):
        if mode not in ("top1", "union"): raise ValueError("match_mode must be 'top1' or 'union'")
        self.matcher = matcher.bind(candidates)
        self.allowed, self.name_to_id, self.mode, self.threshold, self.k = set(candidates), name_to_id, mode, float(threshold), int(k)
        self.cache, self.calls = {}, 0

    def __call__(self, token):
        if token in self.cache: return self.cache[token]
        self.calls += 1
        ids = self.cache[token] = self.resolve(token)
        return ids

    def resolve(self, token):
        out = self.matcher.match(token)
        if out is None: out = []
        try: pairs = [(t, float(s)) for t, s in out]
        except (TypeError, ValueError) as e: raise TypeError(f"{type(self.matcher).__name__}.match must return an iterable of (token, score) pairs") from e
        bad = next((t for t, _ in pairs if t not in self.allowed), None)
        if bad is not None: raise ValueError(f"{type(self.matcher).__name__}.match returned {bad!r}, which is not one of its bound candidates")
        pairs = [p for p in pairs if p[1] >= self.threshold]
        pairs = pairs[:1] if self.mode == "top1" else pairs[:self.k]
        return [self.name_to_id[t] for t, _ in pairs]


def matcher_state(m):
    if m is None: return None
    cfg = m.to_config()
    if cfg is None or type(m).__name__ not in MATCHERS:
        warnings.warn(f"{type(m).__name__} is not json serializable (register it and implement to_config); dropping it")
        return None
    return dict(type=type(m).__name__, config=cfg)


def matcher_from_state(d): return None if d is None else MATCHERS[d["type"]].from_config(d["config"])
