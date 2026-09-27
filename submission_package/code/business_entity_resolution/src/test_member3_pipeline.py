"""
Member 3 - End-to-end pipeline smoke test.

This test creates a tiny synthetic entity-resolution dataset and verifies
the complete Member 3 workflow without requiring the full production data.

Tested stages
--------------
1. Candidate pair features
2. Fuzzy features
3. Ground-truth labeling
4. LightGBM training
5. F0.5 threshold selection
6. Prediction
7. Submission generation
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = ROOT / "output" / "member3_test"
DATA_DIR = TEST_DIR / "data"
SOURCE_DIR = DATA_DIR / "cleaned_data"

SOURCE1_PATH = SOURCE_DIR / "train_source1.parquet"
SOURCE2_PATH = SOURCE_DIR / "train_source2.parquet"
SOURCE3_PATH = SOURCE_DIR / "train_source3.parquet"
GROUND_TRUTH_PATH = SOURCE_DIR / "train_ground_truth.parquet"

CANDIDATE_PATH = TEST_DIR / "candidate_pairs.parquet"

PAIR_FEATURES_PATH = TEST_DIR / "pair_features.parquet"
FUZZY_FEATURES_PATH = TEST_DIR / "fuzzy_features.parquet"

MATCHER_DIR = TEST_DIR / "matcher"
PREDICTIONS_PATH = TEST_DIR / "predicted_matches.parquet"
SUBMISSION_PATH = TEST_DIR / "submission.tsv"


def run_command(command: list[str]) -> None:
    print()
    print("=" * 80)
    print("RUNNING:")
    print(" ".join(str(x) for x in command))
    print("=" * 80)

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def create_synthetic_data() -> None:
    print("Creating synthetic test data...")

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    source1 = pd.DataFrame(
        [
            {
                "entity_id": "s1_001",
                "clean_name": "acme technologies private limited",
                "clean_address": "123 main street delhi",
                "clean_country": "IN",
            },
            {
                "entity_id": "s1_002",
                "clean_name": "global trading company",
                "clean_address": "45 park road mumbai",
                "clean_country": "IN",
            },
            {
                "entity_id": "s1_003",
                "clean_name": "alpha software limited",
                "clean_address": "10 lake road bangalore",
                "clean_country": "IN",
            },
            {
                "entity_id": "s1_004",
                "clean_name": "north star industries",
                "clean_address": "77 market street delhi",
                "clean_country": "IN",
            },
            {
                "entity_id": "s1_005",
                "clean_name": "sunrise exports private limited",
                "clean_address": "9 station road pune",
                "clean_country": "IN",
            },
            {
                "entity_id": "s1_006",
                "clean_name": "delta manufacturing",
                "clean_address": "88 industrial area chennai",
                "clean_country": "IN",
            },
        ]
    )

    source2 = pd.DataFrame(
        [
            {
                "entity_id": "s2_001",
                "clean_name": "acme technologies pvt ltd",
                "clean_address": "123 main st delhi",
                "clean_country": "IN",
            },
            {
                "entity_id": "s2_002",
                "clean_name": "global trading company",
                "clean_address": "45 park road mumbai",
                "clean_country": "IN",
            },
            {
                "entity_id": "s2_003",
                "clean_name": "alpha software ltd",
                "clean_address": "10 lake rd bangalore",
                "clean_country": "IN",
            },
            {
                "entity_id": "s2_004",
                "clean_name": "north star industries",
                "clean_address": "77 market street delhi",
                "clean_country": "IN",
            },
            {
                "entity_id": "s2_005",
                "clean_name": "sunrise exports pvt ltd",
                "clean_address": "9 station road pune",
                "clean_country": "IN",
            },
            {
                "entity_id": "s2_006",
                "clean_name": "delta manufacturing",
                "clean_address": "88 industrial area chennai",
                "clean_country": "IN",
            },
            {
                "entity_id": "s2_007",
                "clean_name": "completely different company",
                "clean_address": "999 unknown road kolkata",
                "clean_country": "IN",
            },
            {
                "entity_id": "s2_008",
                "clean_name": "random business services",
                "clean_address": "12 random avenue hyderabad",
                "clean_country": "IN",
            },
        ]
    )

    source3 = pd.DataFrame(
        [
            {
                "entity_id": "s3_001",
                "clean_name": "Acme Tech",
                "clean_address": "123 Main Street Delhi",
                "clean_country": "IN",
            },
            {
                "entity_id": "s3_002",
                "clean_name": "Global Trading Co",
                "clean_address": "45 Park Rd Mumbai",
                "clean_country": "IN",
            },
            {
                "entity_id": "s3_003",
                "clean_name": "Unrelated Business",
                "clean_address": "500 Unknown Road Jaipur",
                "clean_country": "IN",
            },
        ]
    )

    # Candidate pairs intentionally contain both positive and negative pairs.
    candidates = pd.DataFrame(
        [
            ("s1_001", "s2_001"),
            ("s1_001", "s2_007"),
            ("s1_001", "s3_001"),
            ("s1_002", "s2_002"),
            ("s1_002", "s2_008"),
            ("s1_002", "s3_002"),
            ("s1_003", "s2_003"),
            ("s1_003", "s2_007"),
            ("s1_004", "s2_004"),
            ("s1_004", "s2_008"),
            ("s1_005", "s2_005"),
            ("s1_005", "s2_007"),
            ("s1_006", "s2_006"),
            ("s1_006", "s3_003"),
        ],
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
    )

    # True matches.
    # Keep the same ground-truth representation as the real dataset:
    # one source1 entity with comma-separated matched candidate IDs.
    ground_truth = pd.DataFrame(
        [
            ("s1_001", "s2_001,s3_001"),
            ("s1_002", "s2_002,s3_002"),
            ("s1_003", "s2_003"),
            ("s1_004", "s2_004"),
            ("s1_005", "s2_005"),
            ("s1_006", "s2_006"),
        ],
        columns=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    source1.to_parquet(
        SOURCE1_PATH,
        index=False,
    )

    source2.to_parquet(
        SOURCE2_PATH,
        index=False,
    )

    source3.to_parquet(
        SOURCE3_PATH,
        index=False,
    )

    ground_truth.to_parquet(
        GROUND_TRUTH_PATH,
        index=False,
    )

    candidates.to_parquet(
        CANDIDATE_PATH,
        index=False,
    )

    print(f"Created source1:     {len(source1):,}")
    print(f"Created source2:     {len(source2):,}")
    print(f"Created source3:     {len(source3):,}")
    print(f"Created candidates:  {len(candidates):,}")
    print(f"Created true pairs:  {len(ground_truth):,}")


def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected output was not created: {path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"Output file is empty: {path}"
        )

    print(f"OK: {path}")


def validate_pair_features() -> None:
    df = pd.read_parquet(PAIR_FEATURES_PATH)

    required = {
        "source1_entity_id",
        "candidate_entity_id",
        "name_exact",
        "address_exact",
        "country_exact",
        "name1_length",
        "name2_length",
        "address1_length",
        "address2_length",
    }

    missing = required - set(df.columns)

    if missing:
        raise AssertionError(
            "pair_features is missing columns: "
            + ", ".join(sorted(missing))
        )

    if len(df) != 14:
        raise AssertionError(
            f"Expected 14 pair-feature rows, got {len(df)}"
        )

    print(
        f"Pair feature validation passed: "
        f"{len(df):,} rows"
    )


def validate_fuzzy_features() -> None:
    df = pd.read_parquet(FUZZY_FEATURES_PATH)

    required = {
        "source1_entity_id",
        "candidate_entity_id",
        "name_ratio",
        "name_partial_ratio",
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "address_ratio",
        "address_partial_ratio",
        "address_token_sort_ratio",
        "address_token_set_ratio",
    }

    missing = required - set(df.columns)

    if missing:
        raise AssertionError(
            "fuzzy_features is missing columns: "
            + ", ".join(sorted(missing))
        )

    if len(df) != 14:
        raise AssertionError(
            f"Expected 14 fuzzy-feature rows, got {len(df)}"
        )

    numeric_columns = [
        column
        for column in required
        if column not in {
            "source1_entity_id",
            "candidate_entity_id",
        }
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise AssertionError(
                f"{column} is not numeric"
            )

    print(
        f"Fuzzy feature validation passed: "
        f"{len(df):,} rows"
    )


def validate_matcher() -> None:
    required = [
        MATCHER_DIR / "lightgbm_matcher.pkl",
        MATCHER_DIR / "model_features.json",
        MATCHER_DIR / "threshold.json",
        MATCHER_DIR / "validation_metrics.json",
        MATCHER_DIR / "threshold_search.csv",
        MATCHER_DIR / "feature_importance.csv",
    ]

    for path in required:
        check_file(path)

    with (MATCHER_DIR / "model_features.json").open(
        "r",
        encoding="utf-8",
    ) as f:
        model_features = json.load(f)

    if not model_features:
        raise AssertionError(
            "No model features were saved."
        )

    with (MATCHER_DIR / "threshold.json").open(
        "r",
        encoding="utf-8",
    ) as f:
        threshold_data = json.load(f)

    threshold = float(threshold_data["threshold"])

    if not 0.0 <= threshold <= 1.0:
        raise AssertionError(
            f"Invalid threshold: {threshold}"
        )

    with (MATCHER_DIR / "validation_metrics.json").open(
        "r",
        encoding="utf-8",
    ) as f:
        metrics = json.load(f)

    required_metrics = {
        "precision",
        "recall",
        "f0_5",
    }

    missing = required_metrics - set(metrics)

    if missing:
        raise AssertionError(
            "Validation metrics missing: "
            + ", ".join(sorted(missing))
        )

    print(
        f"Matcher validation passed. "
        f"Features: {len(model_features)}, "
        f"threshold: {threshold:.6f}"
    )


def validate_predictions() -> None:
    predictions = pd.read_parquet(
        PREDICTIONS_PATH
    )

    required = {
        "source1_entity_id",
        "candidate_entity_id",
        "match_probability",
        "prediction",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise AssertionError(
            "Prediction output is missing: "
            + ", ".join(sorted(missing))
        )

    if len(predictions) != 14:
        raise AssertionError(
            f"Expected 14 predictions, got {len(predictions)}"
        )

    if not predictions["match_probability"].between(
        0,
        1,
    ).all():
        raise AssertionError(
            "Prediction probabilities outside [0, 1]."
        )

    if not predictions["prediction"].isin(
        [0, 1]
    ).all():
        raise AssertionError(
            "Prediction column contains values other than 0/1."
        )

    check_file(SUBMISSION_PATH)

    submission = pd.read_csv(
        SUBMISSION_PATH,
        sep="\t",
    )

    expected_columns = [
        "source1_entity_id",
        "candidate_entity_id",
    ]

    if list(submission.columns) != expected_columns:
        raise AssertionError(
            "Unexpected submission columns: "
            + str(list(submission.columns))
        )

    print(
        f"Prediction validation passed: "
        f"{len(predictions):,} rows, "
        f"{len(submission):,} predicted matches."
    )


def main() -> None:
    print("=" * 80)
    print("MEMBER 3 END-TO-END PIPELINE TEST")
    print("=" * 80)

    create_synthetic_data()

    python = sys.executable

    run_command(
        [
            python,
            "-m",
            "business_entity_resolution.matching_features",
            "--candidates",
            str(CANDIDATE_PATH),
            "--source-dir",
            str(SOURCE_DIR),
            "--output",
            str(PAIR_FEATURES_PATH),
        ]
    )

    check_file(PAIR_FEATURES_PATH)
    validate_pair_features()

    run_command(
        [
            python,
            "-m",
            "business_entity_resolution.fuzzy_features",
            "--input",
            str(PAIR_FEATURES_PATH),
            "--output",
            str(FUZZY_FEATURES_PATH),
            "--batch-size",
            "100",
        ]
    )

    check_file(FUZZY_FEATURES_PATH)
    validate_fuzzy_features()

    run_command(
        [
            python,
            "-m",
            "business_entity_resolution.train_matcher",
            "--features",
            str(FUZZY_FEATURES_PATH),
            "--ground-truth",
            str(GROUND_TRUTH_PATH),
            "--output-dir",
            str(MATCHER_DIR),
            "--max-negatives",
            "100",
            "--negative-ratio",
            "5",
        ]
    )

    validate_matcher()

    run_command(
        [
            python,
            "-m",
            "business_entity_resolution.predict_matches",
            "--features",
            str(FUZZY_FEATURES_PATH),
            "--model-dir",
            str(MATCHER_DIR),
            "--output",
            str(PREDICTIONS_PATH),
            "--submission",
            str(SUBMISSION_PATH),
            "--batch-size",
            "100",
        ]
    )

    validate_predictions()

    print()
    print("=" * 80)
    print("ALL MEMBER 3 TESTS PASSED")
    print("=" * 80)
    print()
    print("The production pipeline has NOT been run yet.")
    print("No Git commit or push has been performed.")


if __name__ == "__main__":
    main()