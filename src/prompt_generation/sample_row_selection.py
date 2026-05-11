import os
import numpy as np
import pandas as pd

from constants import Constants

SAMPLE_FRACTION = 0.10
RANDOM_SEED = 42

def allocate_strata_quota(strata_counts: pd.Series, total_needed: int) -> pd.Series:
    """
    Allocate total_needed across strata proportionally using largest-remainder method.
    """
    total_available = int(strata_counts.sum())
    if total_needed > total_available:
        raise ValueError(
            f"Requested {total_needed} but only {total_available} rows available."
        )

    proportions = strata_counts / total_available
    raw = proportions * total_needed
    base = np.floor(raw).astype(int)

    remaining = total_needed - int(base.sum())
    fractions = (raw - base).sort_values(ascending=False)

    # Allocate remaining using largest fractional parts
    for idx in fractions.index:
        if remaining == 0:
            break
        if base.loc[idx] < strata_counts.loc[idx]:
            base.loc[idx] += 1
            remaining -= 1

    # Fill any remaining slots where capacity exists
    if remaining > 0:
        for idx in strata_counts.index:
            if remaining == 0:
                break
            capacity_left = int(strata_counts.loc[idx] - base.loc[idx])
            if capacity_left <= 0:
                continue
            add = min(capacity_left, remaining)
            base.loc[idx] += add
            remaining -= add

    if int(base.sum()) != total_needed:
        raise RuntimeError("Could not allocate quotas to match required total.")

    return base


def main():
    os.makedirs(Constants.DATA_DIR, exist_ok=True)

    df = pd.read_csv(Constants.CLEAN_DATA_FILE_WITH_ARTICLE_INFO)

    required_cols = [
        "article_id", "index", "source", "gender", "politics",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["row_number"] = df.index

    # Calculate 10% sample size
    target_n = round(len(df) * SAMPLE_FRACTION)
    print(f"Total rows: {len(df)}")
    print(f"Target sample size ({SAMPLE_FRACTION*100}%): {target_n}")

    # Define 3-way strata: source x gender x politics
    strata_cols = ["source", "gender", "politics"]
    strata_counts = df.groupby(strata_cols, dropna=False).size()
    print(f"\nTotal strata (source x gender x politics): {len(strata_counts)}")

    target_per_stratum = allocate_strata_quota(strata_counts, target_n)
    rng = np.random.default_rng(RANDOM_SEED)
    picked = []

    for strata_key, take_n in target_per_stratum.items():
        if take_n <= 0:
            continue

        if not isinstance(strata_key, tuple):
            strata_key = (strata_key,)

        mask = pd.Series(True, index=df.index)
        for col, val in zip(strata_cols, strata_key):
            if pd.isna(val):
                mask &= df[col].isna()
            else:
                mask &= (df[col] == val)

        group_df = df[mask]
        sampled = group_df.sample(
            n=int(take_n),
            replace=False,
            random_state=int(rng.integers(0, 1_000_000_000)),
        )
        picked.append(sampled)

    sampled_df = pd.concat(picked, axis=0).sample(
        frac=1.0,
        random_state=int(rng.integers(0, 1_000_000_000)),
    )

    rows_df = sampled_df[["row_number", "article_id", "index", "source", "gender", "politics"]].copy()
    rows_df = rows_df.sort_values("row_number").reset_index(drop=True)
    rows_df.to_csv(Constants.SAMPLE_ROWS_DATA_FILE, index=False)

    row_numbers = rows_df["row_number"].tolist()

    print("\nSaved:")
    print(f"  {Constants.SAMPLE_ROWS_DATA_FILE} ({len(rows_df)} rows)")
    print(f"\nRow numbers ({len(row_numbers)} total):")
    print(row_numbers)

    print("\nDistribution by source:")
    print(rows_df["source"].value_counts(dropna=False).to_string())

    print("\nDistribution by source x gender:")
    print(rows_df.groupby(["source", "gender"], dropna=False).size().to_string())

    print("\nDistribution by source x politics:")
    print(rows_df.groupby(["source", "politics"], dropna=False).size().to_string())

    print("\nDistribution by source x gender x politics:")
    print(rows_df.groupby(["source", "gender", "politics"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()