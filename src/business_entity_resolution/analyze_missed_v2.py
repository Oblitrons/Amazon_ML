from pathlib import Path
import duckdb


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "cleaned_data"
OUTPUT_DIR = ROOT_DIR / "output"

GT_PATH = DATA_DIR / "train_ground_truth.parquet"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs_v2_combined.parquet"

S1_PATH = DATA_DIR / "train_source1.parquet"
S2_PATH = DATA_DIR / "train_source2.parquet"
S3_PATH = DATA_DIR / "train_source3.parquet"

SAMPLE_PATH = OUTPUT_DIR / "missed_v2_sample.parquet"


def analyze_missed_pairs():

    print("=" * 70)
    print("V2-C MISSED PAIR ANALYSIS")
    print("=" * 70)

    con = duckdb.connect()

    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='8GB'")

    # --------------------------------------------------------
    # 1. Build ground-truth pairs
    # --------------------------------------------------------

    print("\n[1/4] Building ground-truth pairs...")

    con.execute(
        f"""
        CREATE TEMP TABLE truth_pairs AS

        SELECT
            source1_entity_id,
            TRIM(
                UNNEST(
                    STRING_SPLIT(matched_entity_ids, ',')
                )
            ) AS candidate_entity_id

        FROM read_parquet('{GT_PATH}')

        WHERE
            matched_entity_ids IS NOT NULL
            AND TRIM(matched_entity_ids) <> ''
        """
    )

    truth_count = con.execute(
        """
        SELECT COUNT(*)
        FROM truth_pairs
        """
    ).fetchone()[0]

    print(f"Total true pairs: {truth_count:,}")

    # --------------------------------------------------------
    # 2. Find missed pairs
    # --------------------------------------------------------

    print("\n[2/4] Finding missed V2-C pairs...")

    con.execute(
        f"""
        CREATE TEMP TABLE missed_pairs AS

        SELECT
            t.source1_entity_id,
            t.candidate_entity_id

        FROM truth_pairs AS t

        LEFT JOIN read_parquet('{CANDIDATE_PATH}') AS c

            ON t.source1_entity_id =
               c.source1_entity_id

            AND t.candidate_entity_id =
                c.candidate_entity_id

        WHERE c.source1_entity_id IS NULL
        """
    )

    missed_count = con.execute(
        """
        SELECT COUNT(*)
        FROM missed_pairs
        """
    ).fetchone()[0]

    print(f"Missed true pairs: {missed_count:,}")

    # --------------------------------------------------------
    # 3. Attach actual records
    # --------------------------------------------------------

    print("\n[3/4] Attaching S1/S2/S3 records...")

    con.execute(
        f"""
        CREATE TEMP TABLE all_candidates AS

        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            clean_name,
            clean_address,
            clean_country

        FROM read_parquet('{S2_PATH}')

        UNION ALL

        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            clean_name,
            clean_address,
            clean_country

        FROM read_parquet('{S3_PATH}')
        """
    )

    con.execute(
        f"""
        CREATE TEMP TABLE missed_details AS

        SELECT

            m.source1_entity_id,

            s1.business_name AS s1_name,
            s1.business_address AS s1_address,
            s1.country AS s1_country,
            s1.clean_name AS s1_clean_name,
            s1.clean_address AS s1_clean_address,

            m.candidate_entity_id,

            c.business_name AS candidate_name,
            c.business_address AS candidate_address,
            c.country AS candidate_country,
            c.clean_name AS candidate_clean_name,
            c.clean_address AS candidate_clean_address

        FROM missed_pairs AS m

        INNER JOIN read_parquet('{S1_PATH}') AS s1
            ON m.source1_entity_id = s1.entity_id

        INNER JOIN all_candidates AS c
            ON m.candidate_entity_id = c.entity_id
        """
    )

    # --------------------------------------------------------
    # 4. Save sample
    # --------------------------------------------------------

    print("\n[4/4] Creating sample...")

    con.execute(
        f"""
        COPY (

            SELECT *

            FROM missed_details

            USING SAMPLE 1000 ROWS

        )

        TO '{SAMPLE_PATH}'

        (FORMAT PARQUET)
        """
    )

    print("\nSample saved to:")
    print(SAMPLE_PATH)

    # --------------------------------------------------------
    # Print examples
    # --------------------------------------------------------

    rows = con.execute(
        """
        SELECT
            s1_clean_name,
            candidate_clean_name,
            s1_clean_address,
            candidate_clean_address,
            s1_country,
            candidate_country

        FROM missed_details

        USING SAMPLE 20 ROWS
        """
    ).fetchall()

    print("\n" + "=" * 70)
    print("20 MISSED TRUE-PAIR EXAMPLES")
    print("=" * 70)

    for i, row in enumerate(rows, 1):

        (
            s1_name,
            candidate_name,
            s1_address,
            candidate_address,
            s1_country,
            candidate_country
        ) = row

        print(f"\n--- Example {i} ---")

        print(f"S1 name:        {s1_name}")
        print(f"Candidate name: {candidate_name}")

        print(f"S1 address:     {s1_address}")
        print(f"Candidate addr: {candidate_address}")

        print(f"Country:        {s1_country} / {candidate_country}")

    con.close()


if __name__ == "__main__":
    analyze_missed_pairs()