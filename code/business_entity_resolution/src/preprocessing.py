from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]

TRAIN_DIR = ROOT_DIR / "dataset" / "train"
TEST_DIR = ROOT_DIR / "dataset" / "test"
PROCESSED_DIR = ROOT_DIR / "processed"

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "llc",
    "llp",
    "plc",
    "pvt",
    "private",
    "proprietary",
}

OUTPUT_COLUMNS = [
    "entity_id",
    "business_name_clean",
    "business_name_core",
    "business_address_clean",
    "address_numbers",
    "country_clean",
]

def normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def normalize_name(value):
    value = normalize_text(value)
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def normalize_name_without_suffix(value):
    value = normalize_name(value)
    tokens = value.split()

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)

def normalize_address(value):
    value = normalize_text(value)
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def extract_numbers(value):
    value = normalize_text(value)
    return " ".join(re.findall(r"\d+", value))

def normalize_country(value):
    return normalize_text(value)

def preprocess_dataframe(df):
    df["business_name_clean"] = (
        df["business_name"]
        .apply(normalize_name)
    )

    df["business_name_core"] = (
        df["business_name"]
        .apply(normalize_name_without_suffix)
    )

    df["business_address_clean"] = (
        df["business_address"]
        .apply(normalize_address)
    )

    df["address_numbers"] = (
        df["business_address"]
        .apply(extract_numbers)
    )

    df["country_clean"] = (
        df["country"]
        .apply(normalize_country)
    )

    return df[OUTPUT_COLUMNS]

def process_source(input_path, output_path):
    print(f"Loading {input_path.name}...")

    df = pd.read_csv(
        input_path,
        sep="\t"
    )

    print(f"Loaded: {df.shape}")

    processed_df = preprocess_dataframe(df)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print(f"Saved: {output_path}")

    del df
    del processed_df

def main():
    train_output_dir = PROCESSED_DIR / "train"
    test_output_dir = PROCESSED_DIR / "test"

    train_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    test_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    process_source(
        TRAIN_DIR / "train_source1.tsv",
        train_output_dir / "source1_processed.tsv"
    )

    process_source(
        TRAIN_DIR / "train_source2.tsv",
        train_output_dir / "source2_processed.tsv"
    )

    process_source(
        TRAIN_DIR / "train_source3.tsv",
        train_output_dir / "source3_processed.tsv"
    )

    process_source(
        TEST_DIR / "test_source1.tsv",
        test_output_dir / "source1_processed.tsv"
    )

    process_source(
        TEST_DIR / "test_source2.tsv",
        test_output_dir / "source2_processed.tsv"
    )

    process_source(
        TEST_DIR / "test_source3.tsv",
        test_output_dir / "source3_processed.tsv"
    )

if __name__ == "__main__":
    main()