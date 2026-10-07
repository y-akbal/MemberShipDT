MASK = (1 << 64) - 1


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


class XorShift64:
    __slots__ = ("s",)

    def __init__(self, seed): self.s = splitmix64(int(seed) & MASK) or 1

    def next(self):
        x = self.s
        x ^= (x << 13) & MASK
        x ^= x >> 7
        x ^= (x << 17) & MASK
        self.s = x
        return x

    def below(self, n): return self.next() % n
