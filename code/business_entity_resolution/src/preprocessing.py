from pathlib import Path
import re
import unicodedata
import pandas as pd


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


NAME_LEGAL_ALIASES = {
    "inc": "incorporated",
    "incorporated": "incorporated",
    "corp": "corporation",
    "corporation": "corporation",
    "co": "company",
    "company": "company",
    "ltd": "limited",
    "limited": "limited",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "pvt": "private",
    "private": "private",
    "proprietary": "proprietary",
}


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


ADDRESS_ALIASES = {
    "st": "street",
    "str": "street",
    "street": "street",
    "rd": "road",
    "road": "road",
    "ave": "avenue",
    "av": "avenue",
    "avenue": "avenue",
    "blvd": "boulevard",
    "boulevard": "boulevard",
    "dr": "drive",
    "drive": "drive",
    "ln": "lane",
    "lane": "lane",
    "hwy": "highway",
    "highway": "highway",
    "pkwy": "parkway",
    "parkway": "parkway",
    "ctr": "center",
    "center": "center",
    "ct": "court",
    "court": "court",
    "pl": "place",
    "place": "place",
    "apt": "apartment",
    "apartment": "apartment",
    "ste": "suite",
    "suite": "suite",
    "fl": "floor",
    "floor": "floor",
}


OUTPUT_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
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


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKC",
        value
    )

    value = value.casefold()

    value = value.replace(
        "&",
        " and "
    )

    value = re.sub(
        r"[/\-]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def clean_tokens(value):
    value = normalize_text(value)

    value = re.sub(
        r"[^\w\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def normalize_name(value):
    value = clean_tokens(value)

    tokens = value.split()

    normalized_tokens = []

    for token in tokens:
        token = NAME_LEGAL_ALIASES.get(
            token,
            token
        )

        normalized_tokens.append(token)

    return " ".join(normalized_tokens)


def strip_legal_suffixes(value):
    value = normalize_name(value)

    tokens = value.split()

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def token_sorted_text(value):
    value = clean_tokens(value)

    tokens = value.split()

    return " ".join(
        sorted(tokens)
    )


def normalize_address(value):
    value = clean_tokens(value)

    tokens = value.split()

    normalized_tokens = []

    for token in tokens:
        token = ADDRESS_ALIASES.get(
            token,
            token
        )

        normalized_tokens.append(token)

    return " ".join(normalized_tokens)


def token_sorted_address(value):
    value = normalize_address(value)

    tokens = value.split()

    return " ".join(
        sorted(tokens)
    )


def strip_diacritics(value):
    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKD",
        value
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    return value


def make_compact(value):
    if pd.isna(value):
        return ""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).casefold()
    )


def extract_numbers(value):
    if pd.isna(value):
        return ""

    value = str(value)

    numbers = re.findall(
        r"\d+",
        value
    )

    return " ".join(numbers)


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
        .apply(strip_legal_suffixes)
    )

    df["business_name_compact"] = (
        df["business_name_clean"]
        .apply(make_compact)
    )

    df["business_name_ascii"] = (
        df["business_name_clean"]
        .apply(strip_diacritics)
    )

    df["business_name_core_ascii"] = (
        df["business_name_core"]
        .apply(strip_diacritics)
    )

    df["business_name_token_sorted"] = (
        df["business_name_clean"]
        .apply(token_sorted_text)
    )

    df["business_address_clean"] = (
        df["business_address"]
        .apply(normalize_address)
    )

    df["address_compact"] = (
        df["business_address_clean"]
        .apply(make_compact)
    )

    df["business_address_ascii"] = (
        df["business_address_clean"]
        .apply(strip_diacritics)
    )

    df["business_address_token_sorted"] = (
        df["business_address_clean"]
        .apply(token_sorted_address)
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
        sep="\t",
        dtype="string",
        keep_default_na=False
    )

    print(f"Loaded: {df.shape}")

    processed_df = preprocess_dataframe(
        df
    )

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
    train_output_dir = (
        PROCESSED_DIR / "train"
    )

    test_output_dir = (
        PROCESSED_DIR / "test"
    )

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