from pathlib import Path
import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directory
DATA_DIR = PROJECT_ROOT / "data" / "cleaned_data"


def inspect_file(file_name):
    file_path = DATA_DIR / file_name

    print("\n" + "=" * 70)
    print(f"FILE: {file_name}")
    print("=" * 70)

    # Load parquet
    df = pd.read_parquet(file_path)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nDuplicate entity IDs:")
    if "entity_id" in df.columns:
        print(df["entity_id"].duplicated().sum())

    if "country" in df.columns:
        print("\nCountry distribution:")
        print(df["country"].value_counts(dropna=False).head(20))

    return df


def main():

    files = [
        "train_source1.parquet",
        "train_source2.parquet",
        "train_source3.parquet",
        "train_ground_truth.parquet",
        "test_source1.parquet",
        "test_source2.parquet",
        "test_source3.parquet",
    ]

    for file_name in files:
        inspect_file(file_name)


if __name__ == "__main__":
    main()