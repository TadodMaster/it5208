"""Demo for the Bernoulli process. Run: python demo.py  (figures -> ../figures)"""
import math
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bernoulli_process import BernoulliProcess, mean_var

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()


def demo_paths(p=0.3, n=50):
    bp = BernoulliProcess(p, seed=1)
    path = bp.path(n)
    print(f"[1] Sample path (p={p}, n={n}):\n    {''.join(map(str, path))}")
    print("    arrival times:", bp.arrival_times(path))
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].stem(range(1, n + 1), path, basefmt=" ")
    ax[0].set_ylabel("$X_k$")
    ax[0].set_title(f"Bernoulli process, p={p}")
    for s in range(3):
        c = BernoulliProcess(p, seed=10 + s).counting(BernoulliProcess(p, seed=10 + s).path(n))
        ax[1].step(range(1, n + 1), c, where="post", label=f"path {s + 1}")
    ax[1].plot(range(1, n + 1), [p * k for k in range(1, n + 1)], "k--", label="$np$")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("$S_k$")
    ax[1].legend()
    save("sample_paths.png")


def demo_binomial(p=0.3, n=20, runs=20000):
    bp = BernoulliProcess(p, seed=2)
    counts = [sum(bp.path(n)) for _ in range(runs)]
    m, v = mean_var(counts)
    print(f"[2] S_{n}: mean={m:.3f} (theory {n*p:.3f}), var={v:.3f} (theory {n*p*(1-p):.3f})")
    ks = range(n + 1)
    emp = [counts.count(k) / runs for k in ks]
    plt.bar(ks, emp, alpha=0.6, label="simulation")
    plt.plot(ks, [bp.binomial_pmf(n, k) for k in ks], "ro-", label="Binomial(n,p)")
    plt.xlabel("number of successes"); plt.legend(); plt.title(f"S_n, n={n}, p={p}")
    save("binomial.png")


def demo_geometric(p=0.3, runs=20000):
    bp = BernoulliProcess(p, seed=3)
    w = [bp.waiting_time() for _ in range(runs)]
    m, v = mean_var(w)
    print(f"[3] Waiting time: mean={m:.3f} (theory {1/p:.3f}), var={v:.3f} (theory {(1-p)/p**2:.3f})")
    ks = range(1, 21)
    plt.bar(ks, [w.count(k) / runs for k in ks], alpha=0.6, label="simulation")
    plt.plot(ks, [bp.geometric_pmf(k) for k in ks], "ro-", label="Geometric(p)")
    plt.xlabel("trials until first success"); plt.legend()
    save("geometric.png")


def demo_memoryless(p=0.3, runs=40000):
    bp = BernoulliProcess(p, seed=4)
    w = [bp.waiting_time() for _ in range(runs)]
    s = 5
    tail = [x - s for x in w if x > s]
    print(f"[4] Memoryless: E[T]={sum(w)/runs:.3f}, E[T-{s} | T>{s}]={sum(tail)/len(tail):.3f}")


def demo_poisson_limit(lam=3.0, runs=20000):
    print("[5] Poisson limit: Binomial(n, lam/n) -> Poisson(lam)")
    ks = range(0, 12)
    pois = [math.exp(-lam) * lam**k / math.factorial(k) for k in ks]
    plt.plot(ks, pois, "k*-", label=f"Poisson({lam})")
    for n in (5, 20, 200):
        bp = BernoulliProcess(lam / n, seed=n)
        plt.plot(ks, [bp.binomial_pmf(n, k) if k <= n else 0 for k in ks], "o--", label=f"n={n}")
    plt.xlabel("k"); plt.legend(); plt.title("Binomial -> Poisson")
    save("poisson_limit.png")


def demo_application_quality(p=0.02, n=500):
    """Application: defective items on a production line."""
    bp = BernoulliProcess(p, seed=5)
    defects = bp.path(n)
    print(f"[6] Production line (defect rate {p}): {sum(defects)} defects in {n} items; "
          f"P(>=15) = {sum(bp.binomial_pmf(n, k) for k in range(15, n + 1)):.4f}")


if __name__ == "__main__":
    demo_paths()
    demo_binomial()
    demo_geometric()
    demo_memoryless()
    demo_poisson_limit()
    demo_application_quality()
    print("Figures saved in", os.path.abspath(OUT))
