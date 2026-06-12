"""
prepare_dataset.py — Elliptic++ jurisdiction partitioning
============================================================

Partitions the Elliptic++ wallet classification dataset into
N_JURISDICTIONS local datasets with a controlled, multi-level
overlap structure, and writes them in the CSV format expected by
aml_psi_tool.py.

Input:
  wallets_classes.csv   (required)
  wallets_features.csv  (optional)

Output:
  <OUTPUT_DIR>/addr_labels_J0.csv ... addr_labels_J{N-1}.csv

Each output file has the columns:
  address      — wallet address (string)
  is_scam      — 1 if illicit, 0 if licit/unknown
  description  — text label ("illicit" / "licit" / "unknown")

Overlap structure
------------------
  N_WAY_COUNT  addresses are placed in ALL jurisdictions
  TRIPLE_COUNT addresses are placed in random triples of jurisdictions
  PAIR_COUNT   addresses are placed in random pairs of jurisdictions

All overlapping addresses are drawn from the illicit subset, to
emulate a watchlist of high-priority cross-border targets.
"""

import itertools
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR = "./data"
OUTPUT_DIR = "./data"
CLASSES_CSV = os.path.join(INPUT_DIR, "wallets_classes.csv")
FEATURES_CSV = os.path.join(INPUT_DIR, "wallets_features.csv")

N_JURISDICTIONS = 5
RANDOM_SEED = 42

N_WAY_COUNT = 2000    # addresses shared by all jurisdictions
TRIPLE_COUNT = 3000   # addresses shared by jurisdiction triples
PAIR_COUNT = 5000     # addresses shared by jurisdiction pairs


def map_label(value: str) -> tuple[int, str]:
    """Map a raw class label to (is_scam, description)."""
    v = str(value).strip().lower()
    if v in ("1", "illicit", "fraud", "scam"):
        return 1, "illicit"
    if v in ("2", "licit", "legitimate", "normal"):
        return 0, "licit"
    return 0, "unknown"


def load_classes(path: str) -> pd.DataFrame:
    print(f"[1/4] Loading {path} ...")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    addr_col = next((c for c in df.columns
                      if "address" in c or "wallet" in c or "id" in c), df.columns[0])
    class_col = next((c for c in df.columns
                       if "class" in c or "label" in c or "category" in c), df.columns[1])

    df = df.rename(columns={addr_col: "address", class_col: "class_label"})
    df["address"] = df["address"].astype(str).str.strip()
    df[["is_scam", "description"]] = df["class_label"].apply(
        lambda x: pd.Series(map_label(x))
    )
    df = df.drop_duplicates(subset=["address"])

    print(f"      {len(df):,} unique addresses")
    print(f"      Illicit: {df['is_scam'].sum():,} "
          f"({df['is_scam'].mean() * 100:.1f}%)")
    return df[["address", "is_scam", "description"]]


def scale_overlap_counts(n_illicit: int) -> tuple[int, int, int]:
    """Scale down overlap counts if not enough illicit addresses are available."""
    n_way, triple, pair = N_WAY_COUNT, TRIPLE_COUNT, PAIR_COUNT
    needed = n_way + triple + pair
    if needed > n_illicit:
        ratio = n_illicit / needed
        n_way = int(n_way * ratio * 0.9)
        triple = int(triple * ratio * 0.9)
        pair = int(pair * ratio * 0.9)
        print("[WARN] Scaling overlap counts to available illicit addresses")
    return n_way, triple, pair


def build_jurisdictions(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Partition df into N_JURISDICTIONS datasets with controlled overlap."""
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    illicit_df = df[df["is_scam"] == 1].reset_index(drop=True)
    licit_df = df[df["is_scam"] == 0].reset_index(drop=True)

    n_way, triple, pair = scale_overlap_counts(len(illicit_df))
    print(f"      Illicit: {len(illicit_df):,}, Licit: {len(licit_df):,}")
    print(f"      Overlap — all-way: {n_way:,}, triple: {triple:,}, pair: {pair:,}")

    ptr = 0
    pool_all = illicit_df.iloc[ptr: ptr + n_way]; ptr += n_way
    pool_triple = illicit_df.iloc[ptr: ptr + triple]; ptr += triple
    pool_pair = illicit_df.iloc[ptr: ptr + pair]; ptr += pair
    illicit_private = illicit_df.iloc[ptr:]

    private_pool = pd.concat([illicit_private, licit_df]).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    juris_frames = [[] for _ in range(N_JURISDICTIONS)]

    # All-jurisdiction overlap
    for i in range(N_JURISDICTIONS):
        juris_frames[i].append(pool_all)

    # Triple overlap: split evenly across all 3-combinations
    triples = list(itertools.combinations(range(N_JURISDICTIONS), 3))
    block = len(pool_triple) // len(triples)
    for k, combo in enumerate(triples):
        subset = pool_triple.iloc[k * block:(k + 1) * block]
        for i in combo:
            juris_frames[i].append(subset)

    # Pair overlap: split evenly across all 2-combinations
    pairs = list(itertools.combinations(range(N_JURISDICTIONS), 2))
    block = len(pool_pair) // len(pairs)
    for k, combo in enumerate(pairs):
        subset = pool_pair.iloc[k * block:(k + 1) * block]
        for i in combo:
            juris_frames[i].append(subset)

    # Remaining private addresses, split evenly
    chunk = len(private_pool) // N_JURISDICTIONS
    for i in range(N_JURISDICTIONS):
        start = i * chunk
        end = start + chunk if i < N_JURISDICTIONS - 1 else len(private_pool)
        juris_frames[i].append(private_pool.iloc[start:end])

    jurisdictions = []
    for i in range(N_JURISDICTIONS):
        juris_df = (
            pd.concat(juris_frames[i])
            .drop_duplicates(subset=["address"])
            .sample(frac=1, random_state=RANDOM_SEED + i)
            .reset_index(drop=True)
        )
        jurisdictions.append(juris_df)
        print(f"      P{i}: {len(juris_df):,} addresses "
              f"(illicit: {juris_df['is_scam'].sum():,}, "
              f"licit: {(juris_df['is_scam'] == 0).sum():,})")

    return jurisdictions


def report_overlap(jurisdictions: list[pd.DataFrame]) -> None:
    sets = [set(j["address"]) for j in jurisdictions]
    print("\nPairwise overlap:")
    for i in range(N_JURISDICTIONS):
        for j in range(i + 1, N_JURISDICTIONS):
            inter = len(sets[i] & sets[j])
            print(f"  P{i} & P{j}: {inter:,} shared addresses")

    all_inter = len(sets[0].intersection(*sets[1:]))
    print(f"  P0 & P1 & ... & P{N_JURISDICTIONS - 1}: "
          f"{all_inter:,} shared by all jurisdictions")


def main() -> None:
    print("=" * 60)
    print("Elliptic++ dataset partitioning")
    print("=" * 60)

    df = load_classes(CLASSES_CSV)

    if os.path.exists(FEATURES_CSV):
        print(f"\n[2/4] Found {FEATURES_CSV} (not used by the PSI tool)")
    else:
        print(f"\n[2/4] {FEATURES_CSV} not found — proceeding with classes only")

    print(f"\n[3/4] Building {N_JURISDICTIONS} jurisdictions...")
    np.random.seed(RANDOM_SEED)
    jurisdictions = build_jurisdictions(df)

    print(f"\n[4/4] Writing output to {OUTPUT_DIR} ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, juris_df in enumerate(jurisdictions):
        out_path = os.path.join(OUTPUT_DIR, f"addr_labels_J{i}.csv")
        juris_df.to_csv(out_path, index=False)
        print(f"      Wrote {out_path} ({len(juris_df):,} rows)")

    print("\n" + "=" * 60)
    report_overlap(jurisdictions)
    print("\nNext: open aml_psi_tool.py, set n=%d, t=2, and run." % N_JURISDICTIONS)
    print("=" * 60)


if __name__ == "__main__":
    main()
