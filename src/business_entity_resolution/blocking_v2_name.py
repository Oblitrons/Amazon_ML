from pathlib import Path
import duckdb


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "cleaned_data"
OUTPUT_DIR = ROOT_DIR / "output"

S1_PATH = DATA_DIR / "train_source1.parquet"
S2_PATH = DATA_DIR / "train_source2.parquet"
S3_PATH = DATA_DIR / "train_source3.parquet"

V1_PATH = OUTPUT_DIR / "candidate_pairs_v1.parquet"

V2_PATH = OUTPUT_DIR / "candidate_pairs_v2_name.parquet"


# ============================================================
# SETTINGS
# ============================================================

MAX_TOKEN_FREQUENCY = 50
MIN_TOKEN_LENGTH = 3


# ============================================================
# MAIN
# ============================================================

def generate_v2_name_candidates():

    print("=" * 70)
    print("BLOCKING V2-A: RARE NAME TOKEN BLOCKING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # DuckDB setup
    # --------------------------------------------------------

    con = duckdb.connect()

    # Prevent DuckDB from using excessive memory
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='8GB'")

    # --------------------------------------------------------
    # Step 1: Load V1 candidates
    # --------------------------------------------------------

    print("\n[1/5] Loading V1 candidate pairs...")

    v1_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{V1_PATH}')
        """
    ).fetchone()[0]

    print(f"V1 candidate pairs: {v1_count:,}")

    # --------------------------------------------------------
    # Step 2: Build rare token list
    # --------------------------------------------------------

    print("\n[2/5] Finding rare name tokens...")

    con.execute(
        f"""
        CREATE TEMP TABLE rare_name_tokens AS

        SELECT
            token,
            COUNT(*) AS frequency

        FROM (

            SELECT
                UNNEST(
                    LIST_DISTINCT(
                        STRING_SPLIT(clean_name, ' ')
                    )
                ) AS token

            FROM read_parquet('{S2_PATH}')

            UNION ALL

            SELECT
                UNNEST(
                    LIST_DISTINCT(
                        STRING_SPLIT(clean_name, ' ')
                    )
                ) AS token

            FROM read_parquet('{S3_PATH}')
        )

        WHERE
            LENGTH(token) >= {MIN_TOKEN_LENGTH}
            AND token <> ''

        GROUP BY token

        HAVING COUNT(*) <= {MAX_TOKEN_FREQUENCY}
        """
    )

    rare_count = con.execute(
        """
        SELECT COUNT(*)
        FROM rare_name_tokens
        """
    ).fetchone()[0]

    print(f"Rare name tokens: {rare_count:,}")

    # --------------------------------------------------------
    # Step 3: Create token index for S1
    # --------------------------------------------------------

    print("\n[3/5] Building S1 rare-token index...")

    con.execute(
        f"""
        CREATE TEMP TABLE s1_name_tokens AS

        SELECT DISTINCT

            s1.entity_id,
            s1.clean_country,
            tokens.token

        FROM read_parquet('{S1_PATH}') AS s1

        CROSS JOIN LATERAL (

            SELECT
                UNNEST(
                    LIST_DISTINCT(
                        STRING_SPLIT(s1.clean_name, ' ')
                    )
                ) AS token

        ) AS tokens

        INNER JOIN rare_name_tokens AS r

            ON tokens.token = r.token

        WHERE
            LENGTH(tokens.token) >= {MIN_TOKEN_LENGTH}
            AND tokens.token <> ''
        """
    )

    s1_token_rows = con.execute(
        """
        SELECT COUNT(*)
        FROM s1_name_tokens
        """
    ).fetchone()[0]

    print(f"S1 rare-token rows: {s1_token_rows:,}")

    # --------------------------------------------------------
    # Step 4: Create token index for S2 + S3
    # --------------------------------------------------------

    print("\n[4/5] Building S2/S3 rare-token index...")

    con.execute(
        f"""
        CREATE TEMP TABLE candidate_name_tokens AS

        SELECT DISTINCT

            s.entity_id,
            s.clean_country,
            tokens.token

        FROM (

            SELECT
                entity_id,
                clean_country,
                clean_name

            FROM read_parquet('{S2_PATH}')

            UNION ALL

            SELECT
                entity_id,
                clean_country,
                clean_name

            FROM read_parquet('{S3_PATH}')

        ) AS s

        CROSS JOIN LATERAL (

            SELECT
                UNNEST(
                    LIST_DISTINCT(
                        STRING_SPLIT(s.clean_name, ' ')
                    )
                ) AS token

        ) AS tokens

        INNER JOIN rare_name_tokens AS r

            ON tokens.token = r.token

        WHERE
            LENGTH(tokens.token) >= {MIN_TOKEN_LENGTH}
            AND tokens.token <> ''
        """
    )

    candidate_token_rows = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_name_tokens
        """
    ).fetchone()[0]

    print(
        f"S2/S3 rare-token rows: "
        f"{candidate_token_rows:,}"
    )

    # --------------------------------------------------------
    # Generate candidates using token equality
    # --------------------------------------------------------

    print("\nGenerating rare name-token candidates...")

    con.execute(
        """
        CREATE TEMP TABLE v2_name_only AS

        SELECT DISTINCT

            s1.entity_id AS source1_entity_id,
            c.entity_id AS candidate_entity_id

        FROM s1_name_tokens AS s1

        INNER JOIN candidate_name_tokens AS c

            ON s1.clean_country = c.clean_country
            AND s1.token = c.token
        """
    )

    rare_candidate_count = con.execute(
        """
        SELECT COUNT(*)
        FROM v2_name_only
        """
    ).fetchone()[0]

    print(
        f"Rare name-token candidate pairs: "
        f"{rare_candidate_count:,}"
    )

    # --------------------------------------------------------
    # Step 5: Combine with V1
    # --------------------------------------------------------

    print("\n[5/5] Combining V1 + rare-token candidates...")

    con.execute(
        f"""
        COPY (

            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM read_parquet('{V1_PATH}')

            UNION

            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM v2_name_only

        )

        TO '{V2_PATH}'

        (FORMAT PARQUET)
        """
    )

    v2_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{V2_PATH}')
        """
    ).fetchone()[0]

    new_candidates = v2_count - v1_count

    print("\n" + "=" * 70)
    print("V2-A RESULTS")
    print("=" * 70)

    print(f"V1 candidate pairs:       {v1_count:,}")
    print(f"Rare-token candidates:     {rare_candidate_count:,}")
    print(f"V2 total candidates:       {v2_count:,}")
    print(f"New candidates added:      {new_candidates:,}")

    print("\nSaved to:")
    print(V2_PATH)

    con.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    generate_v2_name_candidates()