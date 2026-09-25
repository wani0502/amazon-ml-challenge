from pathlib import Path
import re
import unicodedata
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]

TRAIN_DIR = ROOT_DIR / "dataset" / "train"
TEST_DIR = ROOT_DIR / "dataset" / "test"

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
    df = df.copy()

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

    return df

def main():
   

    print("\nLoading training data...")

    train_source1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t"
    )

    train_source2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t"
    )

    train_source3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t"
    )

    print("✓ Training data loaded")

    print("\nPreprocessing Source 1...")
    train_source1 = preprocess_dataframe(train_source1)
    print("✓ Source 1 complete")

    print("\nPreprocessing Source 2...")
    train_source2 = preprocess_dataframe(train_source2)
    print("✓ Source 2 complete")

    print("\nPreprocessing Source 3...")
    train_source3 = preprocess_dataframe(train_source3)
    print("✓ Source 3 complete")

    print("\nGenerated preprocessing columns:")

    print(
        train_source1.columns.tolist()
    )

    print("\nSample preprocessing results:")

    print(
        train_source1[
            [
                "business_name",
                "business_name_clean",
                "business_name_core",
                "business_address",
                "business_address_clean",
                "address_numbers",
                "country",
                "country_clean",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nChecking missing values in generated columns:")

    generated_columns = [
        "business_name_clean",
        "business_name_core",
        "business_address_clean",
        "address_numbers",
        "country_clean",
    ]

    print(
        train_source1[generated_columns]
        .isna()
        .sum()
        .to_string()
    )

 

if __name__ == "__main__":
    main()