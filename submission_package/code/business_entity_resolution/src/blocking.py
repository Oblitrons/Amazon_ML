"""
Member 2: Blocking and Candidate Generation

Blocking V1
-----------
1. Country + exact cleaned business name
2. Country + exact cleaned address
3. Union the candidates
4. Remove duplicate candidate pairs

The goal is high recall while keeping
the number of candidate pairs manageable.
"""

from pathlib import Path
import duckdb


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "cleaned_data"
OUTPUT_DIR = PROJECT_ROOT / "output"

S1_PATH = DATA_DIR / "train_source1.parquet"
S2_PATH = DATA_DIR / "train_source2.parquet"
S3_PATH = DATA_DIR / "train_source3.parquet"


# ============================================================
# Generate candidate pairs
# ============================================================

def generate_candidates():

    print("=" * 70)
    print("BLOCKING V1")
    print("=" * 70)

    print("\nOpening DuckDB...")

    con = duckdb.connect()

    # --------------------------------------------------------
    # Exact-name blocking
    # --------------------------------------------------------

    print("\n[1/2] Running exact-name blocking...")

    name_query = f"""
        SELECT DISTINCT
            s1.entity_id AS source1_entity_id,
            candidate.entity_id AS candidate_entity_id
        FROM read_parquet('{S1_PATH.as_posix()}') AS s1

        INNER JOIN (

            SELECT
                entity_id,
                clean_country,
                clean_name
            FROM read_parquet('{S2_PATH.as_posix()}')

            UNION ALL

            SELECT
                entity_id,
                clean_country,
                clean_name
            FROM read_parquet('{S3_PATH.as_posix()}')

        ) AS candidate

        ON s1.clean_country = candidate.clean_country
        AND s1.clean_name = candidate.clean_name

        WHERE
            s1.clean_name IS NOT NULL
            AND candidate.clean_name IS NOT NULL
            AND TRIM(s1.clean_name) <> ''
            AND TRIM(candidate.clean_name) <> ''
    """

    name_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ({name_query})
        """
    ).fetchone()[0]

    print(
        f"Exact-name candidate pairs: "
        f"{name_count:,}"
    )

    # --------------------------------------------------------
    # Exact-address blocking
    # --------------------------------------------------------

    print("\n[2/2] Running exact-address blocking...")

    address_query = f"""
        SELECT DISTINCT
            s1.entity_id AS source1_entity_id,
            candidate.entity_id AS candidate_entity_id
        FROM read_parquet('{S1_PATH.as_posix()}') AS s1

        INNER JOIN (

            SELECT
                entity_id,
                clean_country,
                clean_address
            FROM read_parquet('{S2_PATH.as_posix()}')

            UNION ALL

            SELECT
                entity_id,
                clean_country,
                clean_address
            FROM read_parquet('{S3_PATH.as_posix()}')

        ) AS candidate

        ON s1.clean_country = candidate.clean_country
        AND s1.clean_address = candidate.clean_address

        WHERE
            s1.clean_address IS NOT NULL
            AND candidate.clean_address IS NOT NULL
            AND TRIM(s1.clean_address) <> ''
            AND TRIM(candidate.clean_address) <> ''
    """

    address_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ({address_query})
        """
    ).fetchone()[0]

    print(
        f"Exact-address candidate pairs: "
        f"{address_count:,}"
    )

    # --------------------------------------------------------
    # Combine both blocking strategies
    # --------------------------------------------------------

    print("\nCombining name + address candidates...")

    combined_query = f"""
        SELECT DISTINCT
            source1_entity_id,
            candidate_entity_id

        FROM (

            {name_query}

            UNION ALL

            {address_query}

        )
    """

    # --------------------------------------------------------
    # Save candidate pairs as Parquet
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / "candidate_pairs_v1.parquet"

    print("\nWriting candidate pairs...")

    con.execute(
        f"""
        COPY ({combined_query})
        TO '{output_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    # --------------------------------------------------------
    # Count final candidates
    # --------------------------------------------------------

    final_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_path.as_posix()}')
        """
    ).fetchone()[0]

    print(
        f"\nTotal unique candidate pairs: "
        f"{final_count:,}"
    )

    print(
        f"\nSaved to:\n{output_path}"
    )

    con.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    generate_candidates()