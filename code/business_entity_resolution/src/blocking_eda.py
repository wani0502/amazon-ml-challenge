import os
import pandas as pd

GROUND_TRUTH_PATH = "dataset/train/train_ground_truth.tsv"
SOURCE1_PATH = "processed/train/source1_processed.tsv"
SOURCE2_PATH = "processed/train/source2_processed.tsv"
OUTPUT_PATH = "eda_reports/blocking_cumulative_recall.tsv"

SOURCE1_COLUMNS = [
    "entity_id",
    "country_clean",
    "business_name_clean",
    "business_name_core",
    "business_name_token_sorted",
    "business_address_clean",
    "business_address_token_sorted",
    "address_numbers"
]

SOURCE2_COLUMNS = SOURCE1_COLUMNS

LEGAL_SUFFIXES = {
    "pvt",
    "private",
    "limited",
    "ltd",
    "llp",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co"
}

CANDIDATE_COUNTS = {
    "B1": 11521672,
    "B2": 33183795,
    "B3": 34927678,
    "B4": 35482337,
    "B5": 35716054,
    "B6": 36053603
}

BLOCK_NAMES = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6"
]

def clean(series):
    return series.fillna("").astype(str).str.strip().str.lower()

def make_name_key(value):
    if not value:
        return ""
    tokens = value.split()
    tokens = [x for x in tokens if x not in LEGAL_SUFFIXES]
    return "_".join(tokens[:2]) if tokens else ""

def prepare(df):
    df["country_key"] = clean(df["country_clean"])
    df["name_clean_key"] = clean(df["business_name_clean"])
    df["name_core_key"] = clean(df["business_name_core"])
    df["name_sorted_key"] = clean(df["business_name_token_sorted"])
    df["address_clean_key"] = clean(df["business_address_clean"])
    df["address_sorted_key"] = clean(df["business_address_token_sorted"])
    df["address_numbers_key"] = clean(df["address_numbers"])

    df["name_key"] = df["name_core_key"].apply(make_name_key)

    df["B1"] = df["country_key"] + "||" + df["name_clean_key"]
    df["B2"] = df["country_key"] + "||" + df["name_core_key"]
    df["B3"] = df["country_key"] + "||" + df["name_sorted_key"]
    df["B4"] = df["country_key"] + "||" + df["address_clean_key"]
    df["B5"] = df["country_key"] + "||" + df["address_sorted_key"]

    df["B6"] = ""

    valid = (
        df["country_key"].ne("")
        & df["address_numbers_key"].ne("")
        & df["name_key"].ne("")
    )

    df.loc[valid, "B6"] = (
        df.loc[valid, "country_key"]
        + "||"
        + df.loc[valid, "address_numbers_key"]
        + "||"
        + df.loc[valid, "name_key"]
    )

    return df

source1 = pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    dtype=str,
    usecols=SOURCE1_COLUMNS
)

source2 = pd.read_csv(
    SOURCE2_PATH,
    sep="\t",
    dtype=str,
    usecols=SOURCE2_COLUMNS
)

source1 = prepare(source1)
source2 = prepare(source2)

source1_lookup = source1[
    ["entity_id"] + BLOCK_NAMES
].copy()

source2_lookup = source2[
    ["entity_id"] + BLOCK_NAMES
].copy()

source1_lookup = source1_lookup.rename(
    columns={"entity_id": "source1_id"}
)

source2_lookup = source2_lookup.rename(
    columns={"entity_id": "source2_id"}
)

captured_counts = {block: 0 for block in BLOCK_NAMES}
cumulative_captured = {block: 0 for block in BLOCK_NAMES}

total_true_pairs = 0

reader = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str,
    chunksize=100000
)

for gt_chunk in reader:
    gt_chunk["matched_entity_ids"] = (
        gt_chunk["matched_entity_ids"]
        .fillna("")
        .str.split(",")
    )

    gt_chunk = gt_chunk.explode("matched_entity_ids")

    gt_chunk["matched_entity_ids"] = (
        gt_chunk["matched_entity_ids"]
        .fillna("")
        .str.strip()
    )

    gt_chunk = gt_chunk[
        gt_chunk["matched_entity_ids"].str.startswith("S2-")
    ]

    gt_pairs = gt_chunk[
        ["source1_entity_id", "matched_entity_ids"]
    ].copy()

    gt_pairs.columns = ["source1_id", "source2_id"]

    if gt_pairs.empty:
        continue

    gt_pairs = gt_pairs.drop_duplicates()

    gt_pairs = gt_pairs.merge(
        source1_lookup,
        on="source1_id",
        how="left"
    )

    gt_pairs = gt_pairs.merge(
        source2_lookup,
        on="source2_id",
        how="left",
        suffixes=("_s1", "_s2")
    )

    total_true_pairs += len(gt_pairs)

    block_hits = pd.DataFrame(index=gt_pairs.index)

    for block in BLOCK_NAMES:
        block_hits[block] = (
            gt_pairs[f"{block}_s1"].fillna("")
            == gt_pairs[f"{block}_s2"].fillna("")
        ) & gt_pairs[f"{block}_s1"].fillna("").ne("")

        captured_counts[block] += int(block_hits[block].sum())

    cumulative = pd.Series(False, index=gt_pairs.index)

    for block in BLOCK_NAMES:
        cumulative = cumulative | block_hits[block]
        cumulative_captured[block] += int(cumulative.sum())

results = []

for block in BLOCK_NAMES:
    results.append({
        "block": block,
        "candidate_pairs": CANDIDATE_COUNTS[block],
        "true_pairs_captured_this_block": captured_counts[block],
        "true_pairs_captured_cumulative": cumulative_captured[block],
        "cumulative_recall": cumulative_captured[block] / total_true_pairs
    })

results_df = pd.DataFrame(results)

os.makedirs("eda_reports", exist_ok=True)

results_df.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False
)

print()
print("BLOCKING CUMULATIVE RECALL")
print("=" * 70)
print(f"Total true S1-S2 pairs: {total_true_pairs:,}")
print()

for _, row in results_df.iterrows():
    print(
        f"{row['block']}: "
        f"{row['candidate_pairs']:,} candidates | "
        f"{row['true_pairs_captured_this_block']:,} new true pairs | "
        f"{row['true_pairs_captured_cumulative']:,} cumulative true pairs | "
        f"{row['cumulative_recall'] * 100:.2f}% recall"
    )

print()
print(f"Saved: {OUTPUT_PATH}")