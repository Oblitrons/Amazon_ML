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

V2_PATH = OUTPUT_DIR / "candidate_pairs_v2_address.parquet"


# ============================================================
# SETTINGS
# ============================================================

MAX_TOKEN_FREQUENCY = 50
MIN_TOKEN_LENGTH = 3


# ============================================================
# MAIN
# ============================================================

def generate_v2_address_candidates():

    print("=" * 70)
    print("BLOCKING V2-B: RARE ADDRESS TOKEN BLOCKING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Keep memory usage controlled
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='8GB'")

    # --------------------------------------------------------
    # Step 1: Load V1
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
    # Step 2: Find rare address tokens
    # --------------------------------------------------------

    print("\n[2/5] Finding rare address tokens...")

    con.execute(
        f"""
        CREATE TEMP TABLE rare_address_tokens AS

        SELECT
            token,
            COUNT(*) AS frequency

        FROM (

            SELECT
                UNNEST(
                    LIST_DISTINCT(
                        STRING_SPLIT(clean_address, ' ')
                    )
                ) AS token

            FROM read_parquet('{S2_PATH}')

            WHERE
                clean_address IS NOT NULL
                AND TRIM(clean_address) <> ''

            UNION ALL

            SELECT
                UNNEST(
                    LIST_DISTINCT(
                        STRING_SPLIT(clean_address, ' ')
                    )
                ) AS token

            FROM read_parquet('{S3_PATH}')

            WHERE
                clean_address IS NOT NULL
                AND TRIM(clean_address) <> ''

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
        FROM rare_address_tokens
        """
    ).fetchone()[0]

    print(f"Rare address tokens: {rare_count:,}")

    # --------------------------------------------------------
    # Step 3: Build S1 address-token index
    # --------------------------------------------------------

    print("\n[3/5] Building S1 rare-address-token index...")

    con.execute(
        f"""
        CREATE TEMP TABLE s1_address_tokens AS

        SELECT DISTINCT

            s1.entity_id,
            s1.clean_country,
            tokens.token

        FROM read_parquet('{S1_PATH}') AS s1

        CROSS JOIN LATERAL (

            SELECT
                UNNEST(
                    LIST_DISTINCT(
                        STRING_SPLIT(s1.clean_address, ' ')
                    )
                ) AS token

        ) AS tokens

        INNER JOIN rare_address_tokens AS r

            ON tokens.token = r.token

        WHERE
            s1.clean_address IS NOT NULL
            AND TRIM(s1.clean_address) <> ''
            AND LENGTH(tokens.token) >= {MIN_TOKEN_LENGTH}
            AND tokens.token <> ''
        """
    )

    s1_token_rows = con.execute(
        """
        SELECT COUNT(*)
        FROM s1_address_tokens
        """
    ).fetchone()[0]

    print(
        f"S1 rare-address-token rows: "
        f"{s1_token_rows:,}"
    )

    # --------------------------------------------------------
    # Step 4: Build S2/S3 address-token index
    # --------------------------------------------------------

    print("\n[4/5] Building S2/S3 rare-address-token index...")

    con.execute(
        f"""
        CREATE TEMP TABLE candidate_address_tokens AS

        SELECT DISTINCT

            s.entity_id,
            s.clean_country,
            tokens.token

        FROM (

            SELECT
                entity_id,
                clean_country,
                clean_address

            FROM read_parquet('{S2_PATH}')

            WHERE
                clean_address IS NOT NULL
                AND TRIM(clean_address) <> ''

            UNION ALL

            SELECT
                entity_id,
                clean_country,
                clean_address

            FROM read_parquet('{S3_PATH}')

            WHERE
                clean_address IS NOT NULL
                AND TRIM(clean_address) <> ''

        ) AS s

        CROSS JOIN LATERAL (

            SELECT
                UNNEST(
                    LIST_DISTINCT(
                        STRING_SPLIT(s.clean_address, ' ')
                    )
                ) AS token

        ) AS tokens

        INNER JOIN rare_address_tokens AS r

            ON tokens.token = r.token

        WHERE
            LENGTH(tokens.token) >= {MIN_TOKEN_LENGTH}
            AND tokens.token <> ''
        """
    )

    candidate_token_rows = con.execute(
        """
        SELECT COUNT(*)
        FROM candidate_address_tokens
        """
    ).fetchone()[0]

    print(
        f"S2/S3 rare-address-token rows: "
        f"{candidate_token_rows:,}"
    )

    # --------------------------------------------------------
    # Generate address-token candidates
    # --------------------------------------------------------

    print("\nGenerating rare address-token candidates...")

    con.execute(
        """
        CREATE TEMP TABLE v2_address_only AS

        SELECT DISTINCT

            s1.entity_id AS source1_entity_id,
            c.entity_id AS candidate_entity_id

        FROM s1_address_tokens AS s1

        INNER JOIN candidate_address_tokens AS c

            ON s1.clean_country = c.clean_country
            AND s1.token = c.token
        """
    )

    address_candidate_count = con.execute(
        """
        SELECT COUNT(*)
        FROM v2_address_only
        """
    ).fetchone()[0]

    print(
        f"Rare address-token candidate pairs: "
        f"{address_candidate_count:,}"
    )

    # --------------------------------------------------------
    # Combine V1 + V2-B
    # --------------------------------------------------------

    print("\n[5/5] Combining V1 + rare-address candidates...")

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

            FROM v2_address_only

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

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("V2-B RESULTS")
    print("=" * 70)

    print(f"V1 candidate pairs:          {v1_count:,}")
    print(
        f"Rare address candidates:      "
        f"{address_candidate_count:,}"
    )
    print(f"V2-B total candidates:        {v2_count:,}")
    print(f"New candidates added:         {new_candidates:,}")

    print("\nSaved to:")
    print(V2_PATH)

    con.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    generate_v2_address_candidates()