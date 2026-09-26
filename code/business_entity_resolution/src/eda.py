from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import preprocess_dataframe


def find_project_root():
    current_file = Path(__file__).resolve()

    for parent in [current_file.parent, *current_file.parents]:
        if (
            (parent / "dataset" / "train").is_dir()
            and (parent / "dataset" / "test").is_dir()
        ):
            return parent

    raise FileNotFoundError(
        "Could not find project root containing dataset/train and dataset/test."
    )


ROOT_DIR = find_project_root()

TRAIN_DIR = ROOT_DIR / "dataset" / "train"
TEST_DIR = ROOT_DIR / "dataset" / "test"
PROCESSED_DIR = ROOT_DIR / "processed"
EDA_DIR = ROOT_DIR / "eda_reports"

EDA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SOURCE_FILES = {
    "train_source1": TRAIN_DIR / "train_source1.tsv",
    "train_source2": TRAIN_DIR / "train_source2.tsv",
    "train_source3": TRAIN_DIR / "train_source3.tsv",
    "test_source1": TEST_DIR / "test_source1.tsv",
    "test_source2": TEST_DIR / "test_source2.tsv",
    "test_source3": TEST_DIR / "test_source3.tsv",
}

PROCESSED_FILES = {
    "train_source1": PROCESSED_DIR / "train" / "source1_processed.tsv",
    "train_source2": PROCESSED_DIR / "train" / "source2_processed.tsv",
    "train_source3": PROCESSED_DIR / "train" / "source3_processed.tsv",
    "test_source1": PROCESSED_DIR / "test" / "source1_processed.tsv",
    "test_source2": PROCESSED_DIR / "test" / "source2_processed.tsv",
    "test_source3": PROCESSED_DIR / "test" / "source3_processed.tsv",
}

RAW_FIELDS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

PROCESSED_FIELDS = [
    "business_name_clean",
    "business_name_core",
    "business_name_compact",
    "business_name_ascii",
    "business_name_core_ascii",
    "business_name_token_sorted",
    "business_address_clean",
    "address_compact",
    "business_address_ascii",
    "business_address_token_sorted",
    "address_numbers",
    "country_clean",
]


def load_tsv(path, usecols=None):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        usecols=usecols
    )


def collision_stats(df, column):
    series = (
        df[column]
        .fillna("")
        .astype("string")
        .str.strip()
    )

    nonempty = series[series != ""]

    if len(nonempty) == 0:
        return {
            "nonempty_rows": 0,
            "unique_values": 0,
            "repeated_values": 0,
            "rows_in_collisions": 0,
            "collision_row_percentage": 0.0,
            "max_frequency": 0
        }

    counts = nonempty.value_counts()

    repeated = counts[counts > 1]

    return {
        "nonempty_rows": int(len(nonempty)),
        "unique_values": int(len(counts)),
        "repeated_values": int(len(repeated)),
        "rows_in_collisions": int(repeated.sum()),
        "collision_row_percentage": round(
            repeated.sum() / len(nonempty) * 100,
            4
        ),
        "max_frequency": int(counts.max())
    }


def build_dataset_summary():
    rows = []

    for dataset_name, path in SOURCE_FILES.items():
        df = load_tsv(
            path,
            usecols=RAW_FIELDS
        )

        rows.append({
            "dataset": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
            "duplicate_rows": int(
                df.duplicated().sum()
            ),
            "duplicate_entity_ids": int(
                df["entity_id"].duplicated().sum()
            ),
            "unique_entity_ids": int(
                df["entity_id"].nunique()
            )
        })

        del df

    return pd.DataFrame(rows)


def build_field_quality():
    rows = []

    for dataset_name, path in SOURCE_FILES.items():
        df = load_tsv(
            path,
            usecols=RAW_FIELDS
        )

        for field in RAW_FIELDS:
            series = (
                df[field]
                .fillna("")
                .astype("string")
                .str.strip()
            )

            missing_mask = series.eq("")

            rows.append({
                "dataset": dataset_name,
                "field": field,
                "rows": len(df),
                "missing_count": int(
                    missing_mask.sum()
                ),
                "missing_percentage": round(
                    missing_mask.mean() * 100,
                    4
                ),
                "unique_values": int(
                    series.nunique(
                        dropna=False
                    )
                ),
                "unique_percentage": round(
                    series.nunique(
                        dropna=False
                    ) / len(df) * 100,
                    4
                )
            })

        del df

    return pd.DataFrame(rows)


def build_raw_collision_report():
    rows = []

    for dataset_name, path in SOURCE_FILES.items():
        df = load_tsv(
            path,
            usecols=[
                "business_name",
                "business_address",
                "country"
            ]
        )

        for field in [
            "business_name",
            "business_address",
            "country"
        ]:
            stats = collision_stats(
                df,
                field
            )

            rows.append({
                "dataset": dataset_name,
                "field": field,
                **stats
            })

        del df

    return pd.DataFrame(rows)


def build_country_report():
    rows = []

    for dataset_name, path in SOURCE_FILES.items():
        df = load_tsv(
            path,
            usecols=["country"]
        )

        counts = (
            df["country"]
            .fillna("")
            .astype("string")
            .str.strip()
            .value_counts(
                dropna=False
            )
        )

        for country, count in counts.items():
            rows.append({
                "dataset": dataset_name,
                "country": country,
                "count": int(count),
                "percentage": round(
                    count / len(df) * 100,
                    4
                )
            })

        del df

    return pd.DataFrame(rows)


def build_length_report():
    rows = []

    for dataset_name, path in SOURCE_FILES.items():
        df = load_tsv(
            path,
            usecols=[
                "business_name",
                "business_address"
            ]
        )

        name_length = (
            df["business_name"]
            .fillna("")
            .astype("string")
            .str.len()
        )

        address_length = (
            df["business_address"]
            .fillna("")
            .astype("string")
            .str.len()
        )

        name_words = (
            df["business_name"]
            .fillna("")
            .astype("string")
            .str.split()
            .str.len()
        )

        address_words = (
            df["business_address"]
            .fillna("")
            .astype("string")
            .str.split()
            .str.len()
        )

        rows.extend([
            {
                "dataset": dataset_name,
                "field": "business_name",
                "metric": "characters_mean",
                "value": float(
                    name_length.mean()
                )
            },
            {
                "dataset": dataset_name,
                "field": "business_name",
                "metric": "characters_median",
                "value": float(
                    name_length.median()
                )
            },
            {
                "dataset": dataset_name,
                "field": "business_name",
                "metric": "characters_p99",
                "value": float(
                    name_length.quantile(0.99)
                )
            },
            {
                "dataset": dataset_name,
                "field": "business_name",
                "metric": "words_mean",
                "value": float(
                    name_words.mean()
                )
            },
            {
                "dataset": dataset_name,
                "field": "business_address",
                "metric": "characters_mean",
                "value": float(
                    address_length.mean()
                )
            },
            {
                "dataset": dataset_name,
                "field": "business_address",
                "metric": "characters_median",
                "value": float(
                    address_length.median()
                )
            },
            {
                "dataset": dataset_name,
                "field": "business_address",
                "metric": "characters_p99",
                "value": float(
                    address_length.quantile(0.99)
                )
            },
            {
                "dataset": dataset_name,
                "field": "business_address",
                "metric": "words_mean",
                "value": float(
                    address_words.mean()
                )
            }
        ])

        del df

    return pd.DataFrame(rows)


def parse_match_ids(value):
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def load_ground_truth():
    return load_tsv(
        TRAIN_DIR / "train_ground_truth.tsv"
    )


def build_ground_truth_analysis(ground_truth):
    gt = ground_truth.copy()

    gt["matched_ids_list"] = (
        gt["matched_entity_ids"]
        .apply(parse_match_ids)
    )

    gt["num_matches"] = (
        gt["matched_ids_list"]
        .str.len()
    )

    match_distribution = (
        gt["num_matches"]
        .value_counts()
        .sort_index()
        .rename_axis("num_matches")
        .reset_index(
            name="source1_entities"
        )
    )

    match_distribution["percentage"] = (
        match_distribution["source1_entities"]
        / len(gt)
        * 100
    ).round(4)

    return gt, match_distribution


def build_processed_collision_report():
    rows = []

    for dataset_name, path in PROCESSED_FILES.items():
        df = load_tsv(
            path,
            usecols=PROCESSED_FIELDS
        )

        for field in PROCESSED_FIELDS:
            stats = collision_stats(
                df,
                field
            )

            rows.append({
                "dataset": dataset_name,
                "field": field,
                **stats
            })

        del df

    return pd.DataFrame(rows)


def build_transformation_examples(
    sample_size=1000
):
    rows = []

    for dataset_name, path in SOURCE_FILES.items():
        df = load_tsv(
            path,
            usecols=RAW_FIELDS
        ).head(sample_size)

        processed = preprocess_dataframe(
            df
        )

        for index in range(len(df)):
            raw_name = str(
                df.iloc[index]["business_name"]
            )

            clean_name = str(
                processed.iloc[index][
                    "business_name_clean"
                ]
            )

            core_name = str(
                processed.iloc[index][
                    "business_name_core"
                ]
            )

            raw_address = str(
                df.iloc[index]["business_address"]
            )

            clean_address = str(
                processed.iloc[index][
                    "business_address_clean"
                ]
            )

            if (
                raw_name != clean_name
                or clean_name != core_name
                or raw_address != clean_address
            ):
                rows.append({
                    "dataset": dataset_name,
                    "entity_id": str(
                        df.iloc[index]["entity_id"]
                    ),
                    "raw_name": raw_name,
                    "clean_name": clean_name,
                    "core_name": core_name,
                    "raw_address": raw_address,
                    "clean_address": clean_address
                })

        del df
        del processed

    return pd.DataFrame(rows)


def build_true_match_pairs(
    ground_truth
):
    pairs = ground_truth.copy()

    pairs["matched_ids_list"] = (
        pairs["matched_entity_ids"]
        .apply(parse_match_ids)
    )

    pairs = pairs[
        [
            "source1_entity_id",
            "matched_ids_list"
        ]
    ]

    pairs = pairs.explode(
        "matched_ids_list"
    )

    pairs = pairs.rename(
        columns={
            "matched_ids_list":
                "matched_entity_id"
        }
    )

    pairs["matched_entity_id"] = (
        pairs["matched_entity_id"]
        .fillna("")
        .astype("string")
        .str.strip()
    )

    pairs = pairs[
        pairs["matched_entity_id"] != ""
    ]

    return pairs


def build_true_match_preprocessing_evaluation(
    ground_truth
):
    pairs = build_true_match_pairs(
        ground_truth
    )

    source1 = load_tsv(
        PROCESSED_FILES["train_source1"],
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "business_name_clean",
            "business_name_core",
            "business_name_compact",
            "business_name_ascii",
            "business_name_core_ascii",
            "business_name_token_sorted",
            "business_address_clean",
            "address_compact",
            "business_address_ascii",
            "business_address_token_sorted",
            "address_numbers",
            "country_clean"
        ]
    )

    source1 = source1.rename(
        columns={
            column: f"s1_{column}"
            for column in source1.columns
        }
    )

    pairs_eval = pairs.merge(
        source1,
        left_on="source1_entity_id",
        right_on="s1_entity_id",
        how="left"
    )

    del source1

    results = []

    target_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "business_name_clean",
        "business_name_core",
        "business_name_compact",
        "business_name_ascii",
        "business_name_core_ascii",
        "business_name_token_sorted",
        "business_address_clean",
        "address_compact",
        "business_address_ascii",
        "business_address_token_sorted",
        "address_numbers",
        "country_clean"
    ]

    for source_name in [
        "train_source2",
        "train_source3"
    ]:
        target = load_tsv(
            PROCESSED_FILES[source_name],
            usecols=target_columns
        )

        target = target.rename(
            columns={
                column: f"m_{column}"
                for column in target.columns
            }
        )

        relevant_ids = set(
            pairs_eval[
                "matched_entity_id"
            ]
            .dropna()
            .astype(str)
        )

        target = target[
            target["m_entity_id"]
            .astype(str)
            .isin(relevant_ids)
        ]

        prefix = (
            "S2-"
            if source_name.endswith("2")
            else "S3-"
        )

        pairs_source = pairs_eval[
            pairs_eval["matched_entity_id"]
            .astype(str)
            .str.startswith(prefix)
        ]

        merged = pairs_source.merge(
            target,
            left_on="matched_entity_id",
            right_on="m_entity_id",
            how="left"
        )

        results.append(
            merged
        )

        del target
        del pairs_source

    pairs_eval = pd.concat(
        results,
        ignore_index=True
    )

    del results

    def exact_nonempty(left, right):
        left = (
            left.fillna("")
            .astype(str)
            .str.strip()
        )

        right = (
            right.fillna("")
            .astype(str)
            .str.strip()
        )

        return (
            (left != "")
            & (right != "")
            & (left == right)
        )

    pairs_eval["raw_name_exact"] = exact_nonempty(
        pairs_eval["s1_business_name"],
        pairs_eval["m_business_name"]
    )

    pairs_eval["clean_name_exact"] = exact_nonempty(
        pairs_eval["s1_business_name_clean"],
        pairs_eval["m_business_name_clean"]
    )

    pairs_eval["core_name_exact"] = exact_nonempty(
        pairs_eval["s1_business_name_core"],
        pairs_eval["m_business_name_core"]
    )

    pairs_eval["ascii_name_exact"] = exact_nonempty(
        pairs_eval["s1_business_name_ascii"],
        pairs_eval["m_business_name_ascii"]
    )

    pairs_eval["token_sorted_name_exact"] = exact_nonempty(
        pairs_eval["s1_business_name_token_sorted"],
        pairs_eval["m_business_name_token_sorted"]
    )

    pairs_eval["raw_address_exact"] = exact_nonempty(
        pairs_eval["s1_business_address"],
        pairs_eval["m_business_address"]
    )

    pairs_eval["clean_address_exact"] = exact_nonempty(
        pairs_eval["s1_business_address_clean"],
        pairs_eval["m_business_address_clean"]
    )

    pairs_eval["ascii_address_exact"] = exact_nonempty(
        pairs_eval["s1_business_address_ascii"],
        pairs_eval["m_business_address_ascii"]
    )

    pairs_eval["token_sorted_address_exact"] = exact_nonempty(
        pairs_eval["s1_business_address_token_sorted"],
        pairs_eval["m_business_address_token_sorted"]
    )

    pairs_eval["numbers_exact"] = exact_nonempty(
        pairs_eval["s1_address_numbers"],
        pairs_eval["m_address_numbers"]
    )

    pairs_eval["country_exact"] = exact_nonempty(
        pairs_eval["s1_country_clean"],
        pairs_eval["m_country_clean"]
    )

    agreement_fields = [
        "raw_name_exact",
        "clean_name_exact",
        "core_name_exact",
        "ascii_name_exact",
        "token_sorted_name_exact",
        "raw_address_exact",
        "clean_address_exact",
        "ascii_address_exact",
        "token_sorted_address_exact",
        "numbers_exact",
        "country_exact"
    ]

    rows = []

    for field in agreement_fields:
        exact_count = int(
            pairs_eval[field].sum()
        )

        rows.append({
            "measure": field,
            "exact_pairs": exact_count,
            "total_true_pairs": len(
                pairs_eval
            ),
            "exact_percentage": round(
                exact_count
                / len(pairs_eval)
                * 100,
                4
            )
        })

    return pd.DataFrame(rows)


def create_visualizations(
    ground_truth
):
    train_sources = {
        "Source 1": SOURCE_FILES[
            "train_source1"
        ],
        "Source 2": SOURCE_FILES[
            "train_source2"
        ],
        "Source 3": SOURCE_FILES[
            "train_source3"
        ]
    }

    plt.figure(
        figsize=(10, 6)
    )

    for dataset_name, path in train_sources.items():
        df = load_tsv(
            path,
            usecols=["business_name"]
        )

        values = (
            df["business_name"]
            .fillna("")
            .astype("string")
            .str.len()
        )

        upper = values.quantile(
            0.99
        )

        values = values[
            values <= upper
        ]

        plt.hist(
            values,
            bins=40,
            histtype="step",
            density=True,
            label=dataset_name
        )

        del df

    plt.title(
        "Business Name Length Distribution"
    )
    plt.xlabel("Characters")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "business_name_length.png",
        dpi=150
    )

    plt.close()

    plt.figure(
        figsize=(10, 6)
    )

    for dataset_name, path in train_sources.items():
        df = load_tsv(
            path,
            usecols=["business_address"]
        )

        values = (
            df["business_address"]
            .fillna("")
            .astype("string")
            .str.len()
        )

        upper = values.quantile(
            0.99
        )

        values = values[
            values <= upper
        ]

        plt.hist(
            values,
            bins=40,
            histtype="step",
            density=True,
            label=dataset_name
        )

        del df

    plt.title(
        "Business Address Length Distribution"
    )
    plt.xlabel("Characters")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "business_address_length.png",
        dpi=150
    )

    plt.close()

    match_counts = (
        ground_truth["num_matches"]
        .value_counts()
        .sort_index()
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        match_counts.index.astype(str),
        match_counts.to_numpy()
    )

    plt.title(
        "Number of Matches per Source 1 Entity"
    )
    plt.xlabel("Number of matches")
    plt.ylabel(
        "Number of Source 1 entities"
    )
    plt.tight_layout()

    plt.savefig(
        EDA_DIR / "ground_truth_match_counts.png",
        dpi=150
    )

    plt.close()


def main():
    print("Running dataset analysis...")

    dataset_summary = build_dataset_summary()

    dataset_summary.to_csv(
        EDA_DIR / "dataset_summary.tsv",
        sep="\t",
        index=False
    )

    print("Dataset summary saved.")

    field_quality = build_field_quality()

    field_quality.to_csv(
        EDA_DIR / "field_quality.tsv",
        sep="\t",
        index=False
    )

    print("Field quality report saved.")

    raw_collision_report = (
        build_raw_collision_report()
    )

    raw_collision_report.to_csv(
        EDA_DIR / "raw_collision_report.tsv",
        sep="\t",
        index=False
    )

    print("Raw collision report saved.")

    country_report = build_country_report()

    country_report.to_csv(
        EDA_DIR / "country_distribution.tsv",
        sep="\t",
        index=False
    )

    print("Country distribution saved.")

    length_report = build_length_report()

    length_report.to_csv(
        EDA_DIR / "length_report.tsv",
        sep="\t",
        index=False
    )

    print("Length report saved.")

    print("Analyzing ground truth...")

    ground_truth = load_ground_truth()

    gt, ground_truth_summary = (
        build_ground_truth_analysis(
            ground_truth
        )
    )

    ground_truth_summary.to_csv(
        EDA_DIR / "ground_truth_match_distribution.tsv",
        sep="\t",
        index=False
    )

    gt[
        [
            "source1_entity_id",
            "matched_entity_ids",
            "num_matches"
        ]
    ].to_csv(
        EDA_DIR / "ground_truth_parsed.tsv",
        sep="\t",
        index=False
    )

    print("Ground truth reports saved.")

    print("Analyzing processed representations...")

    processed_collision_report = (
        build_processed_collision_report()
    )

    processed_collision_report.to_csv(
        EDA_DIR / "processed_collision_report.tsv",
        sep="\t",
        index=False
    )

    print("Processed collision report saved.")

    print("Auditing preprocessing transformations...")

    transformation_audit = (
        build_transformation_examples(
            sample_size=1000
        )
    )

    transformation_audit.to_csv(
        EDA_DIR / "normalization_transformation_audit.tsv",
        sep="\t",
        index=False
    )

    print("Transformation audit saved.")

    print(
        "Evaluating preprocessing on known true matches..."
    )

    agreement_report = (
        build_true_match_preprocessing_evaluation(
            ground_truth
        )
    )

    agreement_report.to_csv(
        EDA_DIR / "true_match_preprocessing_evaluation.tsv",
        sep="\t",
        index=False
    )

    print(
        "True-match preprocessing evaluation saved."
    )

    create_visualizations(
        gt
    )

    print("Visualizations saved.")
    print(f"EDA reports saved in: {EDA_DIR}")


if __name__ == "__main__":
    main()