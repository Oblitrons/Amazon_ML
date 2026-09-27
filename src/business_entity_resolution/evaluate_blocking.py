from pathlib import Path
import sys
import duckdb


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "cleaned_data"
GT_PATH = DATA_DIR / "train_ground_truth.parquet"


# ============================================================
# MAIN
# ============================================================

def evaluate_blocking(candidate_path):

    candidate_path = Path(candidate_path)

    if not candidate_path.exists():
        print(f"ERROR: Candidate file not found:")
        print(candidate_path)
        return

    print("=" * 70)
    print("BLOCKING EVALUATION")
    print("=" * 70)

    print(f"\nCandidate file:")
    print(candidate_path)

    con = duckdb.connect()

    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='8GB'")

    # --------------------------------------------------------
    # Total true pairs
    # --------------------------------------------------------

    print("\n[1/3] Counting ground-truth pairs...")

    total_true_pairs = con.execute(
        f"""
        SELECT
            COUNT(*)
        FROM (
            SELECT
                source1_entity_id,
                UNNEST(
                    STRING_SPLIT(matched_entity_ids, ',')
                ) AS candidate_entity_id
            FROM read_parquet('{GT_PATH}')
            WHERE
                matched_entity_ids IS NOT NULL
                AND TRIM(matched_entity_ids) <> ''
        )
        """
    ).fetchone()[0]

    print(f"Total true match pairs: {total_true_pairs:,}")

    # --------------------------------------------------------
    # Captured true pairs
    # --------------------------------------------------------

    print("\n[2/3] Checking captured true pairs...")

    captured_pairs = con.execute(
        f"""
        SELECT COUNT(*)

        FROM (

            SELECT
                gt.source1_entity_id,
                TRIM(
                    UNNEST(
                        STRING_SPLIT(gt.matched_entity_ids, ',')
                    )
                ) AS candidate_entity_id

            FROM read_parquet('{GT_PATH}') AS gt

            WHERE
                gt.matched_entity_ids IS NOT NULL
                AND TRIM(gt.matched_entity_ids) <> ''

        ) AS truth

        INNER JOIN read_parquet('{candidate_path}') AS candidates

            ON truth.source1_entity_id =
               candidates.source1_entity_id

            AND truth.candidate_entity_id =
                candidates.candidate_entity_id
        """
    ).fetchone()[0]

    missed_pairs = total_true_pairs - captured_pairs

    recall = (
        captured_pairs / total_true_pairs
        if total_true_pairs > 0
        else 0
    )

    print(f"True pairs captured: {captured_pairs:,}")
    print(f"True pairs missed:   {missed_pairs:,}")

    print(
        f"\nPair-level blocking recall: "
        f"{recall:.6%}"
    )

    # --------------------------------------------------------
    # Candidate reduction
    # --------------------------------------------------------

    print("\n[3/3] Calculating candidate reduction...")

    candidate_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{candidate_path}')
        """
    ).fetchone()[0]

    s1_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{DATA_DIR / "train_source1.parquet"}'
        )
        """
    ).fetchone()[0]

    s2_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{DATA_DIR / "train_source2.parquet"}'
        )
        """
    ).fetchone()[0]

    s3_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{DATA_DIR / "train_source3.parquet"}'
        )
        """
    ).fetchone()[0]

    theoretical_all_pairs = s1_count * (s2_count + s3_count)

    candidate_ratio = (
        candidate_count / theoretical_all_pairs
        if theoretical_all_pairs > 0
        else 0
    )

    reduction = 1 - candidate_ratio

    print(f"\nCandidate pairs: {candidate_count:,}")
    print(
        f"Theoretical all-pairs: "
        f"{theoretical_all_pairs:,}"
    )

    print(
        f"\nCandidate pair ratio: "
        f"{candidate_ratio:.10%}"
    )

    print(
        f"Candidate reduction: "
        f"{reduction:.6%}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Candidate pairs:       {candidate_count:,}")
    print(f"True pairs:            {total_true_pairs:,}")
    print(f"Captured true pairs:   {captured_pairs:,}")
    print(f"Missed true pairs:      {missed_pairs:,}")
    print(f"Blocking recall:       {recall:.6%}")
    print(f"Candidate reduction:   {reduction:.6%}")

    con.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:\n"
            "python evaluate_blocking.py "
            "<candidate_parquet>\n"
        )

        sys.exit(1)

    generate_path = sys.argv[1]

    evaluate_blocking(generate_path)