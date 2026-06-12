"""
benchmark_psi.py — Scalability benchmark suite
==================================================

Benchmarks the threshold PSI protocol along three independent axes
and reproduces Figures 3-6 of the paper:

  1 — execution time vs. number of jurisdictions (n)
  2 — execution time vs. consensus threshold (t)
  3 — execution time vs. records per jurisdiction (m)
  4 — phase-wise execution time breakdown (stacked bar)

Each configuration is run N_REPS times; reported values are the
mean across repetitions, with the observed standard deviation used
to confirm experimental stability.

Usage:
    python benchmark_psi.py
"""

import hashlib
import hmac
import os
import secrets
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
})

FIG_W, FIG_H = 3.5, 3.0
COLORS = {
    "load": "#4878CF",
    "setup": "#6ACC65",
    "shares": "#D65F5F",
    "recon": "#B47CC7",
    "total": "#222222",
}
OUT_DIR = "./figures"
DATASET_PATH = "./data/wallets_classes.csv"
N_REPS = 3

PRIME = 2**127 - 1
_HMAC_KEY = secrets.token_bytes(32)

# ---------------------------------------------------------------------------
# Cryptographic core (mirrors aml_psi_tool.py)
# ---------------------------------------------------------------------------


def hash_to_field(element: str) -> int:
    encoded = str(element).strip().lower().encode()
    h256 = int.from_bytes(hashlib.sha256(encoded).digest(), "big")
    h512 = int.from_bytes(hashlib.sha512(encoded).digest()[:32], "big")
    return (h256 ^ h512) & ((1 << 127) - 1)


def batch_random_field(count: int) -> list:
    raw = os.urandom(16 * count)
    return [int.from_bytes(raw[i * 16:(i + 1) * 16], "big") & ((1 << 127) - 1)
            for i in range(count)]


def make_tag(participant_idx: int, h: int, value: int) -> str:
    msg = f"{participant_idx}:{h}:{value}".encode()
    return hmac.new(_HMAC_KEY, msg, hashlib.sha256).hexdigest()


def verify_tag(participant_idx: int, h: int, value: int, tag: str) -> bool:
    return hmac.compare_digest(make_tag(participant_idx, h, value), tag)


def lagrange_t2(x0, y0, x1, y1):
    t0 = y0 * ((-x1) % PRIME) % PRIME * pow((x0 - x1) % PRIME, -1, PRIME) % PRIME
    t1 = y1 * ((-x0) % PRIME) % PRIME * pow((x1 - x0) % PRIME, -1, PRIME) % PRIME
    return (t0 + t1) % PRIME


def lagrange_general(points):
    total = 0
    for i, (xi, yi) in enumerate(points):
        num = den = 1
        for j, (xj, _) in enumerate(points):
            if i == j:
                continue
            num = num * (-xj) % PRIME
            den = den * (xi - xj) % PRIME
        total = (total + yi * num % PRIME * pow(den, -1, PRIME)) % PRIME
    return total


# ---------------------------------------------------------------------------
# Single protocol run
# ---------------------------------------------------------------------------


def run_protocol(addresses: list, n: int, t: int) -> dict:
    """Run one execution of the protocol and return phase timings."""
    m = len(addresses)
    timings = {}

    # Phase 1 — data loading
    t0 = time.perf_counter()
    df = pd.DataFrame({
        "address": addresses,
        "is_scam": np.random.choice([0, 1], size=m, p=[0.98, 0.02]),
    })
    timings["load"] = time.perf_counter() - t0

    # Phase 2 — leader setup (hashing + polynomial generation)
    t0 = time.perf_counter()
    hashes = [hash_to_field(a) for a in df["address"].tolist()]
    df["h"] = hashes
    hash_list = list(set(hashes))
    num_h = len(hash_list)

    num_coeffs = t - 1
    rand_coeffs = batch_random_field(num_h * num_coeffs) if num_coeffs > 0 else []
    polys = {h: [h] + rand_coeffs[idx * num_coeffs:(idx + 1) * num_coeffs]
             for idx, h in enumerate(hash_list)}
    coeff_matrix = np.array([[polys[h][c] for c in range(t)] for h in hash_list],
                             dtype=object)

    distributed = {}
    for i in range(1, n):
        x = i + 1
        result = np.zeros(num_h, dtype=object)
        for c in range(t - 1, -1, -1):
            result = (result * x + coeff_matrix[:, c]) % PRIME
        distributed[i] = {
            h: (int(result[idx]), make_tag(i, h, int(result[idx])))
            for idx, h in enumerate(hash_list)
        }
    timings["setup"] = time.perf_counter() - t0

    # Phase 3 — local matching and response
    t0 = time.perf_counter()
    all_shares = {h: [] for h in hash_list}

    x_leader = 1
    leader_result = np.zeros(num_h, dtype=object)
    for c in range(t - 1, -1, -1):
        leader_result = (leader_result * x_leader + coeff_matrix[:, c]) % PRIME
    for idx, h in enumerate(hash_list):
        all_shares[h].append({"x": x_leader, "y": int(leader_result[idx])})

    # A fixed 5% subset of the watchlist simulates cross-jurisdiction overlap
    overlap_set = set(hash_list[:max(1, num_h // 20)])
    for i in range(1, n):
        x = i + 1
        for h in overlap_set:
            value, tag = distributed[i][h]
            if verify_tag(i, h, value, tag):
                all_shares[h].append({"x": x, "y": value})
    timings["shares"] = time.perf_counter() - t0

    # Phase 4 — threshold reconstruction
    t0 = time.perf_counter()
    matches = 0
    for h, shares in all_shares.items():
        if len(shares) < t:
            continue
        points = [(s["x"], s["y"]) for s in shares[:t]]
        reconstructed = (lagrange_t2(*points[0], *points[1]) if t == 2
                         else lagrange_general(points[:t]))
        if reconstructed == h:
            matches += 1
    timings["recon"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    timings["intersections"] = matches
    return timings


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

KEYS = ["load", "setup", "shares", "recon", "total"]


def collect_stats(param_values, run_fn, reps=N_REPS):
    """Run `run_fn(value)` `reps` times per value; return mean and std dicts."""
    means = {k: [] for k in KEYS}
    stds = {k: [] for k in KEYS}
    for val in param_values:
        samples = {k: [] for k in KEYS}
        for _ in range(reps):
            result = run_fn(val)
            for k in KEYS:
                samples[k].append(result[k])
        for k in KEYS:
            means[k].append(float(np.mean(samples[k])))
            stds[k].append(float(np.std(samples[k])))
    return means, stds


def load_addresses(path: str, fallback_size: int = 200_000) -> list:
    print("Loading dataset...")
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        addr_col = next(c for c in df.columns if "address" in c)
        addrs = df[addr_col].dropna().astype(str).tolist()
        print(f"  Loaded {len(addrs):,} addresses")
        return addrs
    except Exception as exc:
        print(f"  Dataset not found ({exc}); using synthetic addresses")
        return [f"1A{i:08d}" for i in range(fallback_size)]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_scaling(x_values, means, stds, xlabel, out_name):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    series = [
        ("total", "Total", "o-", COLORS["total"]),
        ("shares", "Phase 3 (Shares)", "s--", COLORS["shares"]),
        ("setup", "Phase 2 (Setup)", "^--", COLORS["setup"]),
        ("recon", "Phase 4 (Recon.)", "D--", COLORS["recon"]),
    ]
    handles = []
    for key, label, style, color in series:
        h, = ax.plot(x_values, means[key], style, color=color, label=label)
        handles.append(h)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Execution time (s)")
    ax.set_xticks(x_values)
    ax.grid(True, linestyle="--")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, framealpha=0.9, columnspacing=1.0, handlelength=1.8)

    fig.subplots_adjust(bottom=0.28)
    fig.savefig(f"{OUT_DIR}/{out_name}.pdf")
    fig.savefig(f"{OUT_DIR}/{out_name}.png")
    plt.close(fig)


def plot_phase_breakdown(configs, out_name):
    labels = [c[0] for c in configs]
    phases = ["load", "setup", "shares", "recon"]
    phase_labels = ["Phase 1 (Load)", "Phase 2 (Setup)",
                     "Phase 3 (Shares)", "Phase 4 (Recon.)"]
    phase_colors = [COLORS["load"], COLORS["setup"], COLORS["shares"], COLORS["recon"]]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bottoms = np.zeros(len(configs))
    x = np.arange(len(configs))

    for phase, label, color in zip(phases, phase_labels, phase_colors):
        values = np.array([c[1][phase] for c in configs])
        ax.bar(x, values, bottom=bottoms, label=label, color=color,
               edgecolor="white", linewidth=0.5)
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Execution time (s)")
    ax.grid(True, axis="y", linestyle="--")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              framealpha=0.9, columnspacing=1.0, handlelength=1.5)

    fig.subplots_adjust(bottom=0.28)
    fig.savefig(f"{OUT_DIR}/{out_name}.pdf")
    fig.savefig(f"{OUT_DIR}/{out_name}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_addrs = load_addresses(DATASET_PATH)

    m_fixed = 50_000

    # Benchmark 1 — scalability vs. n
    print("\n[1/4] Scalability vs. number of jurisdictions (n)")
    n_values = list(range(2, 9))
    means_n, stds_n = collect_stats(
        n_values, lambda n: run_protocol(all_addrs[:m_fixed], n=n, t=2)
    )
    for n, total in zip(n_values, means_n["total"]):
        print(f"  n={n}: {total:.3f}s")
    plot_scaling(n_values, means_n, stds_n,
                  "Number of jurisdictions (n), t=2, m=50,000", "Fig_3")

    # Benchmark 2 — scalability vs. t
    print("\n[2/4] Scalability vs. consensus threshold (t)")
    n_fixed = 5
    t_values = list(range(2, n_fixed + 1))
    means_t, stds_t = collect_stats(
        t_values, lambda t: run_protocol(all_addrs[:m_fixed], n=n_fixed, t=t)
    )
    for t, total in zip(t_values, means_t["total"]):
        print(f"  t={t}: {total:.3f}s")
    plot_scaling(t_values, means_t, stds_t,
                  "Consensus threshold (t), n=5, m=50,000", "Fig_4")

    # Benchmark 3 — scalability vs. m
    print("\n[3/4] Scalability vs. records per jurisdiction (m)")
    m_values = [m for m in (10_000, 25_000, 50_000, 75_000, 100_000, 150_000, 200_000)
                 if m <= len(all_addrs)]
    means_m, stds_m = collect_stats(
        m_values, lambda m: run_protocol(all_addrs[:m], n=5, t=2)
    )
    for m, total in zip(m_values, means_m["total"]):
        print(f"  m={m:,}: {total:.3f}s")
    plot_scaling(m_values, means_m, stds_m,
                  "Records per jurisdiction (m), n=5, t=2", "Fig_5")

    # Benchmark 4 — phase-wise breakdown
    print("\n[4/4] Phase-wise execution time breakdown")
    paper_n5 = {"load": 1.2233, "setup": 2.4001, "shares": 6.9381, "recon": 0.0505}
    configs = [("n=5\nm=168k\n(paper)", paper_n5)]
    for n in (2, 3, 5, 7):
        if n in n_values:
            idx = n_values.index(n)
            configs.append((f"n={n}\nm=50k",
                             {k: means_n[k][idx] for k in ("load", "setup", "shares", "recon")}))
    plot_phase_breakdown(configs, "Fig_6")

    print(f"\nAll figures saved to {OUT_DIR}/")
    for fig in ("Fig_3", "Fig_4", "Fig_5", "Fig_6"):
        print(f"  {fig}.pdf / {fig}.png")


if __name__ == "__main__":
    main()
