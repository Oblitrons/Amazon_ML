"""
Member 3 - Predict business entity matches.

Workflow
--------
1. Load trained LightGBM matcher.
2. Load selected model features.
3. Load optimized F0.5 threshold.
4. Read fuzzy feature parquet in batches.
5. Predict match probabilities.
6. Write all candidate predictions.
7. Write positive predictions as submission TSV.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_FEATURES = "output/fuzzy_features.parquet"
DEFAULT_MODEL_DIR = "output/matcher"
DEFAULT_OUTPUT = "output/predicted_matches.parquet"
DEFAULT_SUBMISSION = "output/submission.tsv"

DEFAULT_BATCH_SIZE = 50_000


def load_model(model_dir: Path):
    model_path = model_dir / "lightgbm_matcher.pkl"
    features_path = model_dir / "model_features.json"
    threshold_path = model_dir / "threshold.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not features_path.exists():
        raise FileNotFoundError(f"Model feature file not found: {features_path}")

    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

    with model_path.open("rb") as f:
        model = pickle.load(f)

    with features_path.open("r", encoding="utf-8") as f:
        model_features = json.load(f)

    with threshold_path.open("r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    threshold = float(threshold_data["threshold"])

    return model, model_features, threshold


def validate_features(df: pd.DataFrame, model_features: list[str]) -> None:
    missing = [
        column
        for column in model_features
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Prediction input is missing required model features: "
            + ", ".join(missing)
        )


def predict_batches(
    feature_path: Path,
    model,
    model_features: list[str],
    threshold: float,
    output_path: Path,
    submission_path: Path,
    batch_size: int,
    limit: int | None = None,
) -> None:
    parquet_file = pq.ParquetFile(feature_path)

    required_ids = {
        "source1_entity_id",
        "candidate_entity_id",
    }

    schema_columns = set(parquet_file.schema_arrow.names)
    missing_ids = required_ids - schema_columns

    if missing_ids:
        raise ValueError(
            "Feature parquet is missing required ID columns: "
            + ", ".join(sorted(missing_ids))
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    if submission_path.exists():
        submission_path.unlink()

    writer = None
    total_rows = 0
    total_positive = 0

    submission_initialized = False

    try:
        for record_batch in parquet_file.iter_batches(
            batch_size=batch_size
        ):
            df = record_batch.to_pandas()

            if limit is not None:
                remaining = limit - total_rows

                if remaining <= 0:
                    break

                if len(df) > remaining:
                    df = df.iloc[:remaining].copy()

            if df.empty:
                continue

            validate_features(df, model_features)

            X = df[model_features].copy()

            for column in X.columns:
                if X[column].dtype == bool:
                    X[column] = X[column].astype(np.int8)

            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)

            probabilities = model.predict_proba(X)[:, 1]

            predictions = (
                probabilities >= threshold
            ).astype(np.int8)

            result = pd.DataFrame(
                {
                    "source1_entity_id": df["source1_entity_id"].values,
                    "candidate_entity_id": df[
                        "candidate_entity_id"
                    ].values,
                    "match_probability": probabilities,
                    "prediction": predictions,
                }
            )

            table = pa.Table.from_pandas(
                result,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(table)

            positive_rows = result[
                result["prediction"] == 1
            ][
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            ]

            if not positive_rows.empty:
                positive_rows.to_csv(
                    submission_path,
                    sep="\t",
                    index=False,
                    header=not submission_initialized,
                    mode="a",
                )

                submission_initialized = True
                total_positive += len(positive_rows)

            total_rows += len(result)

            if total_rows % 500_000 < len(result):
                print(
                    f"Processed {total_rows:,} rows | "
                    f"predicted matches: {total_positive:,}"
                )

    finally:
        if writer is not None:
            writer.close()

    print()
    print("Prediction complete.")
    print(f"Rows processed:       {total_rows:,}")
    print(f"Predicted matches:    {total_positive:,}")
    print(f"Threshold:            {threshold:.6f}")
    print(f"Prediction parquet:   {output_path}")
    print(f"Submission TSV:       {submission_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict business entity matches."
    )

    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        help="Fuzzy feature parquet.",
    )

    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory containing trained model artifacts.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output prediction parquet.",
    )

    parser.add_argument(
        "--submission",
        default=DEFAULT_SUBMISSION,
        help="Output submission TSV.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Prediction batch size.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for testing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_path = Path(args.features)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    submission_path = Path(args.submission)

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature parquet not found: {feature_path}"
        )

    model, model_features, threshold = load_model(
        model_dir
    )

    print(f"Feature file: {feature_path}")
    print(f"Model directory: {model_dir}")
    print(f"Model features: {len(model_features)}")
    print(f"Threshold: {threshold:.6f}")
    print()

    predict_batches(
        feature_path=feature_path,
        model=model,
        model_features=model_features,
        threshold=threshold,
        output_path=output_path,
        submission_path=submission_path,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()