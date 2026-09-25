from pathlib import Path
import re
import unicodedata
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]

TRAIN_DIR = ROOT_DIR / "dataset" / "train"
TEST_DIR = ROOT_DIR / "dataset" / "test"


def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize("NFKC", value)

    value = value.lower()

    value = re.sub(r"\s+", " ", value)

    value = value.strip()

    return value


def normalize_name(value):

    value = normalize_text(value)

    value = re.sub(r"[^\w\s]", " ", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_address(value):

    value = normalize_text(value)

    value = re.sub(r"[^\w\s]", " ", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_country(value):

    return normalize_text(value)


def preprocess_dataframe(df):

    df = df.copy()

    df["business_name_clean"] = (
        df["business_name"]
        .apply(normalize_name)
    )

    df["business_address_clean"] = (
        df["business_address"]
        .apply(normalize_address)
    )

    df["country_clean"] = (
        df["country"]
        .apply(normalize_country)
    )

    return df


def main():

    print("=" * 60)
    print("       BUSINESS ENTITY RESOLUTION")
    print("             PREPROCESSING")
    print("=" * 60)

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

    print("\nSample preprocessing results:")

    print(
        train_source1[
            [
                "business_name",
                "business_name_clean",
                "business_address",
                "business_address_clean",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()