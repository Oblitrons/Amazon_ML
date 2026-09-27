from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "cleaned_data"


def analyze_file(file_name):

    path = DATA_DIR / file_name

    print("\n" + "=" * 70)
    print(file_name)
    print("=" * 70)

    df = pd.read_parquet(
        path,
        columns=[
            "entity_id",
            "clean_name",
            "clean_address",
            "clean_country"
        ]
    )

    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # Empty values
    # --------------------------------------------------------

    empty_name = (
        df["clean_name"].fillna("").str.strip().eq("")
    ).sum()

    empty_address = (
        df["clean_address"].fillna("").str.strip().eq("")
    ).sum()

    print(f"\nEmpty clean_name: {empty_name:,}")
    print(f"Empty clean_address: {empty_address:,}")

    # --------------------------------------------------------
    # Name frequency
    # --------------------------------------------------------

    name_counts = (
        df.loc[
            df["clean_name"].str.strip().ne(""),
            "clean_name"
        ]
        .value_counts()
    )

    print("\nUnique non-empty clean names:")
    print(f"{len(name_counts):,}")

    print("\nMost frequent names:")
    print(name_counts.head(20))

    print("\nName frequency statistics:")
    print(name_counts.describe())

    # --------------------------------------------------------
    # Address frequency
    # --------------------------------------------------------

    address_counts = (
        df.loc[
            df["clean_address"].str.strip().ne(""),
            "clean_address"
        ]
        .value_counts()
    )

    print("\nUnique non-empty clean addresses:")
    print(f"{len(address_counts):,}")

    print("\nMost frequent addresses:")
    print(address_counts.head(20))

    print("\nAddress frequency statistics:")
    print(address_counts.describe())

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    print("\nCountry distribution:")
    print(df["clean_country"].value_counts())

    return df


def main():

    files = [
        "train_source1.parquet",
        "train_source2.parquet",
        "train_source3.parquet"
    ]

    for file_name in files:
        analyze_file(file_name)


if __name__ == "__main__":
    main()