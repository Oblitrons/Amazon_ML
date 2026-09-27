from pathlib import Path
import argparse
import gc

import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from rapidfuzz import fuzz


FUZZY_FEATURES = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "address_ratio",
    "address_partial_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
]

RAW_COLUMNS = {
    "name1",
    "name2",
    "address1",
    "address2",
    "country1",
    "country2",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print("=" * 70)
    print("FUZZY FEATURE GENERATION")
    print("=" * 70)
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Batch : {args.batch_size:,}")

    pf = pq.ParquetFile(input_path)

    all_columns = pf.schema_arrow.names

    required_raw = [
        "name1",
        "name2",
        "address1",
        "address2",
    ]

    for col in required_raw:
        if col not in all_columns:
            raise ValueError(f"Required column missing: {col}")

    # Keep IDs + all deterministic/model columns.
    # Drop only raw text columns after fuzzy features are calculated.
    output_base_columns = [
        col for col in all_columns
        if col not in RAW_COLUMNS
    ]

    read_columns = all_columns

    print(f"Input columns : {len(all_columns)}")
    print(f"Output base columns: {len(output_base_columns)}")
    print(f"Rows: {pf.metadata.num_rows:,}")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    writer = None
    rows_written = 0
    batch_number = 0

    try:
        for batch in pf.iter_batches(
            batch_size=args.batch_size,
            columns=read_columns,
            use_threads=True,
        ):
            batch_number += 1

            df = batch.to_pandas()

            # Convert missing values to empty strings for RapidFuzz.
            name1 = df["name1"].fillna("").astype(str).tolist()
            name2 = df["name2"].fillna("").astype(str).tolist()
            address1 = df["address1"].fillna("").astype(str).tolist()
            address2 = df["address2"].fillna("").astype(str).tolist()

            df["name_ratio"] = [
                fuzz.ratio(a, b)
                for a, b in zip(name1, name2)
            ]

            df["name_partial_ratio"] = [
                fuzz.partial_ratio(a, b)
                for a, b in zip(name1, name2)
            ]

            df["name_token_sort_ratio"] = [
                fuzz.token_sort_ratio(a, b)
                for a, b in zip(name1, name2)
            ]

            df["name_token_set_ratio"] = [
                fuzz.token_set_ratio(a, b)
                for a, b in zip(name1, name2)
            ]

            df["address_ratio"] = [
                fuzz.ratio(a, b)
                for a, b in zip(address1, address2)
            ]

            df["address_partial_ratio"] = [
                fuzz.partial_ratio(a, b)
                for a, b in zip(address1, address2)
            ]

            df["address_token_sort_ratio"] = [
                fuzz.token_sort_ratio(a, b)
                for a, b in zip(address1, address2)
            ]

            df["address_token_set_ratio"] = [
                fuzz.token_set_ratio(a, b)
                for a, b in zip(address1, address2)
            ]

            # Final schema:
            # IDs + deterministic numeric features + fuzzy features.
            output_columns = output_base_columns + FUZZY_FEATURES

            # Remove accidental duplicates while preserving order.
            output_columns = list(dict.fromkeys(output_columns))

            out_df = df[output_columns]

            table = pyarrow.Table.from_pandas(
                out_df,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(table)

            rows_written += len(out_df)

            if batch_number == 1 or batch_number % 10 == 0:
                print(
                    f"Batch {batch_number:,} | "
                    f"Rows written: {rows_written:,} / {pf.metadata.num_rows:,}"
                )

            del batch
            del df
            del out_df
            del table
            gc.collect()

    finally:
        if writer is not None:
            writer.close()

    print()
    print("=" * 70)
    print("FUZZY FEATURE GENERATION COMPLETE")
    print("=" * 70)
    print(f"Rows written: {rows_written:,}")
    print()
    print(f"Output file:")
    print(output_path)
    print()
    print("Fuzzy features:")
    for feature in FUZZY_FEATURES:
        print(f"  - {feature}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
