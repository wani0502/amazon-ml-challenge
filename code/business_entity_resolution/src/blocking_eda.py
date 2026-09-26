import os
import pandas as pd

GROUND_TRUTH_PATH = "dataset/train/train_ground_truth.tsv"
CANDIDATE_PATH = "output/candidate_pairs.tsv"
SOURCE1_PATH = "processed/train/source1_processed.tsv"
OUTPUT_PATH = "eda_reports/blocking_eda.tsv"

gt = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str
)

source1 = pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    dtype=str,
    usecols=["entity_id"]
)

gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

rows = []

for source1_id, matched_ids in zip(
    gt["source1_entity_id"],
    gt["matched_entity_ids"]
):
    for matched_id in matched_ids.split(","):
        matched_id = matched_id.strip()

        if matched_id.startswith("S2-"):
            rows.append((source1_id, matched_id))

ground_truth = pd.DataFrame(
    rows,
    columns=["source1_id", "source2_id"]
).drop_duplicates()

ground_truth_keys = set(
    zip(
        ground_truth["source1_id"],
        ground_truth["source2_id"]
    )
)

candidate_counts = {}
captured = 0
candidate_pairs = 0

for chunk in pd.read_csv(
    CANDIDATE_PATH,
    sep="\t",
    dtype=str,
    chunksize=1_000_000
):
    candidate_pairs += len(chunk)

    for source1_id in chunk["source1_id"]:
        candidate_counts[source1_id] = (
            candidate_counts.get(source1_id, 0) + 1
        )

    keys = zip(
        chunk["source1_id"],
        chunk["source2_id"]
    )

    captured += sum(
        pair in ground_truth_keys
        for pair in keys
    )

candidate_series = pd.Series(candidate_counts)

total_ground_truth = len(ground_truth)

recall = (
    captured / total_ground_truth
    if total_ground_truth
    else 0
)

zero_candidates = len(
    set(source1["entity_id"]) -
    set(candidate_counts.keys())
)

results = {
    "source1_rows": len(source1),
    "source2_rows": pd.read_csv(
        "processed/train/source2_processed.tsv",
        sep="\t",
        usecols=["entity_id"]
    ).shape[0],
    "true_s1_s2_pairs": total_ground_truth,
    "candidate_pairs": candidate_pairs,
    "true_matches_captured": captured,
    "blocking_recall": recall,
    "avg_candidates_per_s1": candidate_series.mean(),
    "median_candidates_per_s1": candidate_series.median(),
    "p95_candidates_per_s1": candidate_series.quantile(0.95),
    "p99_candidates_per_s1": candidate_series.quantile(0.99),
    "zero_candidate_s1": zero_candidates
}

print("\nBLOCKING EDA")
print("-" * 50)

for key, value in results.items():
    if key == "blocking_recall":
        print(f"{key}: {value:.6f} ({value * 100:.2f}%)")
    elif "candidates_per_s1" in key:
        print(f"{key}: {value:,.2f}")
    else:
        print(f"{key}: {value:,}")

os.makedirs("eda_reports", exist_ok=True)

pd.DataFrame([results]).to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)

print(f"\nSaved: {OUTPUT_PATH}")