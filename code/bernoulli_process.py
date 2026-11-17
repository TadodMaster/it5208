"""Bernoulli process implemented from scratch.

Only the uniform generator `random.random()` and basic math are used.
No library routine that already implements Bernoulli / Binomial / Geometric
sampling or pmf (e.g. numpy.random.binomial, scipy.stats) is used.
"""
import math
import random


class BernoulliProcess:
    """X_1, X_2, ... i.i.d. Bernoulli(p)."""

    def __init__(self, p, seed=None):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = p
        self.rng = random.Random(seed)

    # ---- simulation -------------------------------------------------
    def trial(self):
        """One Bernoulli(p) trial: 1 if U < p else 0 (U ~ Uniform[0,1))."""
        return 1 if self.rng.random() < self.p else 0

    def path(self, n):
        """Sample path X_1..X_n."""
        return [self.trial() for _ in range(n)]

    @staticmethod
    def counting(path):
        """Counting process S_k = X_1 + ... + X_k."""
        s, out = 0, []
        for x in path:
            s += x
            out.append(s)
        return out

    @staticmethod
    def arrival_times(path):
        """Times (1-indexed) of successes T_1 < T_2 < ..."""
        return [i + 1 for i, x in enumerate(path) if x == 1]

    @staticmethod
    def interarrival_times(path):
        """Gaps between successes (first gap measured from time 0)."""
        t = BernoulliProcess.arrival_times(path)
        return [b - a for a, b in zip([0] + t[:-1], t)]

    def waiting_time(self):
        """Trials until first success (Geometric(p)), by direct simulation."""
        k = 1
        while self.trial() == 0:
            k += 1
        return k

    # ---- theory (closed-form, written by hand) -----------------------
    def binomial_pmf(self, n, k):
        return math.comb(n, k) * self.p**k * (1 - self.p) ** (n - k)

    def geometric_pmf(self, k):
        return (1 - self.p) ** (k - 1) * self.p

    def neg_binomial_pmf(self, r, k):
        """P(r-th success occurs at trial k), k >= r."""
        return math.comb(k - 1, r - 1) * self.p**r * (1 - self.p) ** (k - r)


def mean_var(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, v
