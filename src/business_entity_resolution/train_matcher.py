"""
Member 3 - Production-scale business entity matcher.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


RANDOM_SEED = 42
VALIDATION_SIZE = 0.20

DEFAULT_FEATURES = "output/fuzzy_features.parquet"
DEFAULT_GROUND_TRUTH = "data/cleaned_data/train_ground_truth.parquet"
DEFAULT_OUTPUT_DIR = "output/matcher"

DEFAULT_MAX_NEGATIVES = 2_000_000
DEFAULT_NEGATIVE_RATIO = 5

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": 4,
    "verbosity": -1,
}

ID_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train production-scale business entity matcher."
    )

    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
    )

    parser.add_argument(
        "--ground-truth",
        default=DEFAULT_GROUND_TRUTH,
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--max-negatives",
        type=int,
        default=DEFAULT_MAX_NEGATIVES,
    )

    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=DEFAULT_NEGATIVE_RATIO,
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=VALIDATION_SIZE,
    )

    return parser.parse_args()


def detect_ground_truth_columns(con, ground_truth_path):
    columns = con.execute(
        f"""
        DESCRIBE SELECT *
        FROM read_parquet('{ground_truth_path.as_posix()}')
        """
    ).fetchdf()["column_name"].tolist()

    if "source1_entity_id" not in columns:
        raise ValueError(
            f"Ground truth missing source1_entity_id. Columns: {columns}"
        )

    if "matched_entity_ids" not in columns:
        raise ValueError(
            f"Ground truth missing matched_entity_ids. Columns: {columns}"
        )

    return "source1_entity_id", "matched_entity_ids"


def inspect_feature_columns(con, feature_path):
    columns = con.execute(
        f"""
        DESCRIBE SELECT *
        FROM read_parquet('{feature_path.as_posix()}')
        """
    ).fetchdf()["column_name"].tolist()

    required = {
        "source1_entity_id",
        "candidate_entity_id",
    }

    missing = required - set(columns)

    if missing:
        raise ValueError(
            "Feature parquet missing IDs: "
            + ", ".join(sorted(missing))
        )

    return columns


def select_model_features(columns):
    excluded = ID_COLUMNS | {"label"}

    features = [
        column
        for column in columns
        if column not in excluded
    ]

    if not features:
        raise ValueError("No model features found.")

    return features


def create_training_sample(
    con,
    feature_path,
    ground_truth_path,
    model_features,
    max_negatives,
    negative_ratio,
):
    feature_sql_path = feature_path.as_posix()
    gt_sql_path = ground_truth_path.as_posix()

    selected_columns = ",\n            ".join(
        f'f."{column}"'
        for column in model_features
    )

    print()
    print("Building bounded training sample with DuckDB...")

    # The ground truth stores multiple matched candidate IDs in one
    # comma-separated VARCHAR:
    #
    # source1_entity_id | matched_entity_ids
    # S1-xxx             | S2-xxx,S2-yyy,S3-zzz
    #
    # Split the list into one positive pair per candidate ID.

    positive_query = f"""
        SELECT
            f.source1_entity_id,
            f.candidate_entity_id,
            {selected_columns},
            CAST(1 AS INTEGER) AS label
        FROM read_parquet('{feature_sql_path}') f
        INNER JOIN (
            SELECT DISTINCT
                CAST(source1_entity_id AS VARCHAR)
                    AS source1_entity_id,
                TRIM(candidate_entity_id) AS candidate_entity_id
            FROM read_parquet('{gt_sql_path}'),
            UNNEST(
                string_split(
                    COALESCE(matched_entity_ids, ''),
                    ','
                )
            ) AS t(candidate_entity_id)
            WHERE TRIM(candidate_entity_id) <> ''
        ) g
        ON f.source1_entity_id = g.source1_entity_id
        AND f.candidate_entity_id = g.candidate_entity_id
    """

    positives = con.execute(
        positive_query
    ).fetchdf()

    if positives.empty:
        raise ValueError(
            "No ground-truth positive pairs exist in "
            "the candidate set."
        )

    positive_count = len(positives)

    print(
        f"Candidate positives available: {positive_count:,}"
    )

    desired_negatives = min(
        max_negatives,
        positive_count * negative_ratio,
    )

    print(
        f"Requested negatives: {desired_negatives:,}"
    )

    # A candidate is negative when its pair does not occur in any
    # source1_entity_id + matched_entity_ids combination in GT.

    negative_query = f"""
        SELECT
            f.source1_entity_id,
            f.candidate_entity_id,
            {selected_columns},
            CAST(0 AS INTEGER) AS label
        FROM read_parquet('{feature_sql_path}') f
        WHERE NOT EXISTS (
            SELECT 1
            FROM read_parquet('{gt_sql_path}') g,
            UNNEST(
                string_split(
                    COALESCE(g.matched_entity_ids, ''),
                    ','
                )
            ) AS t(candidate_entity_id)
            WHERE CAST(g.source1_entity_id AS VARCHAR)
                    = f.source1_entity_id
              AND TRIM(t.candidate_entity_id)
                    = f.candidate_entity_id
        )
        ORDER BY
            hash(
                f.source1_entity_id,
                f.candidate_entity_id,
                {RANDOM_SEED}
            )
        LIMIT {desired_negatives}
    """

    negatives = con.execute(
        negative_query
    ).fetchdf()

    print(
        f"Selected negatives: {len(negatives):,}"
    )

    if negatives.empty:
        raise ValueError(
            "No negative candidate pairs were selected."
        )

    sample = pd.concat(
        [positives, negatives],
        ignore_index=True,
    )

    sample = sample.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    print(
        f"Training sample: {len(sample):,} rows"
    )

    return sample


def grouped_split(df, validation_size):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=RANDOM_SEED,
    )

    train_idx, valid_idx = next(
        splitter.split(
            df,
            df["label"],
            groups=df["source1_entity_id"],
        )
    )

    train_df = df.iloc[train_idx].copy()
    valid_df = df.iloc[valid_idx].copy()

    print()
    print("Grouped train/validation split:")
    print(f"Training rows:   {len(train_df):,}")
    print(f"Validation rows: {len(valid_df):,}")
    print(
        f"Training positives: "
        f"{int(train_df['label'].sum()):,}"
    )
    print(
        f"Validation positives: "
        f"{int(valid_df['label'].sum()):,}"
    )

    return train_df, valid_df


def clean_matrix(df, model_features):
    X = df[model_features].copy()

    for column in X.columns:
        if X[column].dtype == bool:
            X[column] = X[column].astype(np.int8)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    return X


def f_beta_score(precision, recall, beta=0.5):
    beta_squared = beta * beta

    denominator = (
        beta_squared * precision + recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + beta_squared)
        * precision
        * recall
        / denominator
    )


def search_best_threshold(y_true, probabilities):
    rows = []

    thresholds = np.arange(
        0.01,
        1.001,
        0.005,
    )

    best_threshold = 0.5
    best_f05 = -1.0

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(np.int8)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f05 = f_beta_score(
            precision,
            recall,
            beta=0.5,
        )

        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f0_5": float(f05),
            }
        )

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = float(threshold)

    return (
        best_threshold,
        best_f05,
        pd.DataFrame(rows),
    )


def main():
    args = parse_args()

    feature_path = Path(args.features)
    ground_truth_path = Path(args.ground_truth)
    output_dir = Path(args.output_dir)

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}"
        )

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {ground_truth_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("MEMBER 3 - PRODUCTION-SCALE BUSINESS ENTITY MATCHER")
    print("=" * 70)

    print(f"Features:      {feature_path}")
    print(f"Ground truth:  {ground_truth_path}")
    print(f"Output:        {output_dir}")
    print(
        f"Max negatives: {args.max_negatives:,}"
    )
    print(
        f"Negative ratio: {args.negative_ratio}:1"
    )

    con = duckdb.connect()

    con.execute(
        "PRAGMA threads=4"
    )

    con.execute(
        "PRAGMA memory_limit='8GB'"
    )

    print()
    print("Inspecting feature schema...")

    feature_columns = inspect_feature_columns(
        con,
        feature_path,
    )

    model_features = select_model_features(
        feature_columns
    )

    print(
        f"Model features: {len(model_features)}"
    )

    for feature in model_features:
        print(f"  {feature}")

    source_column, candidate_column = (
        detect_ground_truth_columns(
            con,
            ground_truth_path,
        )
    )

    print()
    print("Ground-truth columns:")
    print(f"  source:    {source_column}")
    print(f"  candidate: {candidate_column}")

    sample = create_training_sample(
        con=con,
        feature_path=feature_path,
        ground_truth_path=ground_truth_path,
        model_features=model_features,
        max_negatives=args.max_negatives,
        negative_ratio=args.negative_ratio,
    )

    con.close()

    train_df, valid_df = grouped_split(
        sample,
        args.validation_size,
    )

    X_train = clean_matrix(
        train_df,
        model_features,
    )

    y_train = train_df["label"].astype(
        np.int8
    ).to_numpy()

    X_valid = clean_matrix(
        valid_df,
        model_features,
    )

    y_valid = valid_df["label"].astype(
        np.int8
    ).to_numpy()

    print()
    print("Training LightGBM...")

    model = lgb.LGBMClassifier(
        **LGB_PARAMS,
        n_estimators=1000,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_valid, y_valid),
        ],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=75,
                verbose=True,
            ),
            lgb.log_evaluation(
                period=50,
            ),
        ],
    )

    print()
    print("Generating validation probabilities...")

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    print()
    print("Searching F0.5 threshold...")

    (
        best_threshold,
        best_f05,
        threshold_df,
    ) = search_best_threshold(
        y_valid,
        probabilities,
    )

    predictions = (
        probabilities >= best_threshold
    ).astype(np.int8)

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_valid,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_valid,
        probabilities,
    )

    matrix = confusion_matrix(
        y_valid,
        predictions,
    )

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Threshold: {best_threshold:.4f}"
    )
    print(
        f"Precision: {precision:.6f}"
    )
    print(
        f"Recall:    {recall:.6f}"
    )
    print(
        f"F0.5:      {best_f05:.6f}"
    )
    print(
        f"ROC-AUC:   {roc_auc:.6f}"
    )
    print(
        f"PR-AUC:    {pr_auc:.6f}"
    )

    print()
    print("Confusion matrix:")
    print(matrix)

    importance = pd.DataFrame(
        {
            "feature": model_features,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print()
    print("Top feature importance:")
    print(
        importance.head(20).to_string(
            index=False
        )
    )

    model_path = (
        output_dir / "lightgbm_matcher.pkl"
    )

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    features_path = (
        output_dir / "model_features.json"
    )

    with features_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            model_features,
            f,
            indent=2,
        )

    threshold_path = (
        output_dir / "threshold.json"
    )

    with threshold_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "threshold": best_threshold,
                "metric": "F0.5",
            },
            f,
            indent=2,
        )

    metrics_path = (
        output_dir / "validation_metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "threshold": best_threshold,
                "precision": precision,
                "recall": recall,
                "f0_5": best_f05,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "validation_rows": len(valid_df),
                "validation_positives": int(
                    y_valid.sum()
                ),
                "validation_negatives": int(
                    len(y_valid) - y_valid.sum()
                ),
                "best_iteration": getattr(
                    model,
                    "best_iteration_",
                    None,
                ),
            },
            f,
            indent=2,
        )

    threshold_df.to_csv(
        output_dir / "threshold_search.csv",
        index=False,
    )

    importance.to_csv(
        output_dir / "feature_importance.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best threshold:       {best_threshold:.4f}"
    )
    print(
        f"Validation F0.5:      {best_f05:.6f}"
    )
    print(
        f"Validation precision: {precision:.6f}"
    )
    print(
        f"Validation recall:    {recall:.6f}"
    )

    print()
    print("Saved:")
    print(model_path)
    print(features_path)
    print(threshold_path)
    print(metrics_path)
    print(
        output_dir / "threshold_search.csv"
    )
    print(
        output_dir / "feature_importance.csv"
    )


if __name__ == "__main__":
    main()
