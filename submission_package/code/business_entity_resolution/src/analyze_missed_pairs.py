from pathlib import Path
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "cleaned_data"
OUTPUT_DIR = PROJECT_ROOT / "output"

S1_PATH = DATA_DIR / "train_source1.parquet"
S2_PATH = DATA_DIR / "train_source2.parquet"
S3_PATH = DATA_DIR / "train_source3.parquet"
GT_PATH = DATA_DIR / "train_ground_truth.parquet"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs_v1.parquet"


def analyze_missed_pairs():

    print("=" * 70)
    print("ANALYZING MISSED BLOCKING PAIRS")
    print("=" * 70)

    con = duckdb.connect()

    # ------------------------------------------------------------
    # Ground truth pairs
    # ------------------------------------------------------------

    print("\n[1] Loading ground truth pairs...")

    gt_query = f"""
        SELECT
            source1_entity_id,
            TRIM(
                UNNEST(
                    STRING_SPLIT(matched_entity_ids, ',')
                )
            ) AS candidate_entity_id

        FROM read_parquet(
            '{GT_PATH.as_posix()}'
        )

        WHERE
            matched_entity_ids IS NOT NULL
            AND TRIM(matched_entity_ids) <> ''
    """

    # ------------------------------------------------------------
    # Find pairs missed by V1
    # ------------------------------------------------------------

    print("[2] Finding missed pairs...")

    missed_query = f"""
        SELECT
            gt.source1_entity_id,
            gt.candidate_entity_id
        FROM ({gt_query}) AS gt

        LEFT JOIN read_parquet(
            '{CANDIDATE_PATH.as_posix()}'
        ) AS candidates

        ON gt.source1_entity_id =
           candidates.source1_entity_id

        AND gt.candidate_entity_id =
            candidates.candidate_entity_id

        WHERE candidates.source1_entity_id IS NULL
    """

    missed_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ({missed_query})
        """
    ).fetchone()[0]

    print(
        f"\nMissed pairs: {missed_count:,}"
    )

    # ------------------------------------------------------------
    # Sample missed pairs
    # ------------------------------------------------------------

    print("\n[3] Creating sample of missed pairs...")

    sample_query = f"""
        SELECT
            missed.source1_entity_id,

            s1.business_name AS s1_name,
            s1.clean_name AS s1_clean_name,

            s1.business_address AS s1_address,
            s1.clean_address AS s1_clean_address,

            s1.country AS s1_country,
            s1.clean_country AS s1_clean_country,

            missed.candidate_entity_id,

            candidate.business_name AS candidate_name,
            candidate.clean_name AS candidate_clean_name,

            candidate.business_address AS candidate_address,
            candidate.clean_address AS candidate_clean_address,

            candidate.country AS candidate_country,
            candidate.clean_country AS candidate_clean_country

        FROM ({missed_query}) AS missed

        INNER JOIN read_parquet(
            '{S1_PATH.as_posix()}'
        ) AS s1

        ON missed.source1_entity_id =
           s1.entity_id

        INNER JOIN (

            SELECT *
            FROM read_parquet(
                '{S2_PATH.as_posix()}'
            )

            UNION ALL

            SELECT *
            FROM read_parquet(
                '{S3_PATH.as_posix()}'
            )

        ) AS candidate

        ON missed.candidate_entity_id =
           candidate.entity_id

        USING SAMPLE 1000 ROWS
    """

    sample_path = OUTPUT_DIR / "missed_pairs_sample.parquet"

    con.execute(
        f"""
        COPY ({sample_query})
        TO '{sample_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    print(
        f"Sample saved to:\n{sample_path}"
    )

    # ------------------------------------------------------------
    # Display examples
    # ------------------------------------------------------------

    print("\n[4] Showing first 20 missed pairs...\n")

    rows = con.execute(
        f"""
        SELECT
            source1_entity_id,
            s1_name,
            candidate_name,
            s1_address,
            candidate_address,
            s1_clean_name,
            candidate_clean_name,
            s1_clean_address,
            candidate_clean_address
        FROM read_parquet(
            '{sample_path.as_posix()}'
        )
        LIMIT 20
        """
    ).fetchall()

    for row in rows:

        print("-" * 70)

        print(f"S1 ID:              {row[0]}")
        print(f"S1 Name:            {row[1]}")
        print(f"Candidate Name:     {row[2]}")

        print(f"S1 Address:         {row[3]}")
        print(f"Candidate Address:  {row[4]}")

        print(f"S1 Clean Name:      {row[5]}")
        print(f"Candidate Clean:    {row[6]}")

        print(f"S1 Clean Address:   {row[7]}")
        print(f"Candidate Clean:    {row[8]}")

    print("\n" + "=" * 70)

    con.close()


if __name__ == "__main__":
    analyze_missed_pairs()