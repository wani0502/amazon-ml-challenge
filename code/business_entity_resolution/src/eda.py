from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



ROOT_DIR = Path(__file__).resolve().parents[3]

TRAIN_DIR = ROOT_DIR / "dataset" / "train"
TEST_DIR = ROOT_DIR / "dataset" / "test"



print("Loading Train Source 1...")
train_source1 = pd.read_csv(
    TRAIN_DIR / "train_source1.tsv",
    sep="\t"
)
print(f"✓ Train Source 1: {train_source1.shape}")

print("Loading Train Source 2...")
train_source2 = pd.read_csv(
    TRAIN_DIR / "train_source2.tsv",
    sep="\t"
)
print(f"✓ Train Source 2: {train_source2.shape}")

print("Loading Train Source 3...")
train_source3 = pd.read_csv(
    TRAIN_DIR / "train_source3.tsv",
    sep="\t"
)
print(f"✓ Train Source 3: {train_source3.shape}")

print("Loading Ground Truth...")
ground_truth = pd.read_csv(
    TRAIN_DIR / "train_ground_truth.tsv",
    sep="\t"
)
print(f"✓ Ground Truth: {ground_truth.shape}")

print("Loading Test Source 1...")
test_source1 = pd.read_csv(
    TEST_DIR / "test_source1.tsv",
    sep="\t"
)
print(f"✓ Test Source 1: {test_source1.shape}")

print("Loading Test Source 2...")
test_source2 = pd.read_csv(
    TEST_DIR / "test_source2.tsv",
    sep="\t"
)
print(f"✓ Test Source 2: {test_source2.shape}")

print("Loading Test Source 3...")
test_source3 = pd.read_csv(
    TEST_DIR / "test_source3.tsv",
    sep="\t"
)
print(f"✓ Test Source 3: {test_source3.shape}")


datasets = {
    "Train Source 1": train_source1,
    "Train Source 2": train_source2,
    "Train Source 3": train_source3,
    "Test Source 1": test_source1,
    "Test Source 2": test_source2,
    "Test Source 3": test_source3,
}

for name, df in datasets.items():
    print("\n" + "-" * 60)
    print(name)
    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())
    print("Data types:")
    print(df.dtypes)

print("\nGround Truth")
print("Shape:", ground_truth.shape)
print("Columns:", ground_truth.columns.tolist())
print(ground_truth.dtypes)



for name, df in {
    "Train Source 1": train_source1,
    "Train Source 2": train_source2,
    "Train Source 3": train_source3,
}.items():

    print(name)
    print(
        df[
            [
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ]
        ].head(5).to_string(index=False)
    )

print("\nGround Truth samples:")
print(ground_truth.head(5).to_string(index=False))



for name, df in datasets.items():
    missing_count = df.isna().sum()
    missing_percentage = df.isna().mean() * 100

    missing_table = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage.round(4)
    })

    
    print(name)
    print(missing_table)



for name, df in datasets.items():
    
    print(name)

    print(
        "Exact duplicate rows:",
        df.duplicated().sum()
    )

    print(
        "Duplicate entity IDs:",
        df["entity_id"].duplicated().sum()
    )

    print(
        "Duplicate business names:",
        df["business_name"].duplicated().sum()
    )

    print(
        "Duplicate business addresses:",
        df["business_address"].duplicated().sum()
    )



for name, df in datasets.items():
    print("\n" + "-" * 60)
    print(name)

    print(
        df["country"]
        .value_counts(dropna=False)
        .to_string()
    )



for name, df in datasets.items():
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


    print(name)

    print("\nBusiness name length:")
    print(name_length.describe())

    print("\nBusiness address length:")
    print(address_length.describe())

    print("\nBusiness name word count:")
    print(name_words.describe())

    print("\nBusiness address word count:")
    print(address_words.describe())



gt = ground_truth.copy()

gt["num_matches"] = (
    gt["matched_entity_ids"]
    .fillna("")
    .astype(str)
    .apply(
        lambda x:
        0 if x.strip() == ""
        else len(x.split(","))
    )
)

print("\nNumber of matches per Source 1 entity:")
print(
    gt["num_matches"]
    .value_counts()
    .sort_index()
)

print("\nMatch count statistics:")
print(gt["num_matches"].describe())

match_distribution = (
    gt["num_matches"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nMatch distribution (%):")
print(match_distribution.round(2))

print("\nCreating visualizations...")

plt.figure(figsize=(10, 6))

for name, df in {
    "Source 1": train_source1,
    "Source 2": train_source2,
    "Source 3": train_source3,
}.items():

    name_length = (
        df["business_name"]
        .fillna("")
        .astype("string")
        .str.len()
    )

    upper = name_length.quantile(0.99)

    sns.histplot(
        name_length[name_length <= upper],
        bins=50,
        stat="density",
        element="step",
        fill=False,
        label=name
    )

plt.title("Business Name Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

for name, df in {
    "Source 1": train_source1,
    "Source 2": train_source2,
    "Source 3": train_source3,
}.items():

    address_length = (
        df["business_address"]
        .fillna("")
        .astype("string")
        .str.len()
    )

    upper = address_length.quantile(0.99)

    sns.histplot(
        address_length[address_length <= upper],
        bins=50,
        stat="density",
        element="step",
        fill=False,
        label=name
    )

plt.title("Business Address Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

match_counts = (
    gt["num_matches"]
    .value_counts()
    .sort_index()
)

sns.barplot(
    x=match_counts.index,
    y=match_counts.values
)

plt.title("Number of Matches per Source 1 Entity")
plt.xlabel("Number of Matches")
plt.ylabel("Number of Source 1 Entities")
plt.tight_layout()
plt.show()

