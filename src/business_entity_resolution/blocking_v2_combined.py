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

V2_PATH = OUTPUT_DIR / "candidate_pairs_v2_combined.parquet"


# ============================================================
# SETTINGS
# ============================================================

MAX_TOKEN_FREQUENCY = 50
MIN_TOKEN_LENGTH = 3


# ============================================================
# MAIN
# ============================================================

def generate_v2_combined_candidates():

    print("=" * 70)
    print("BLOCKING V2-C: RARE NAME + ADDRESS TOKEN BLOCKING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    # Control memory usage
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='8GB'")

    # --------------------------------------------------------
    # Step 1: Load V1
    # --------------------------------------------------------

    print("\n[1/7] Loading V1 candidate pairs...")

    v1_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{V1_PATH}')
        """
    ).fetchone()[0]

    print(f"V1 candidate pairs: {v1_count:,}")

    # ========================================================
    # NAME BLOCK
    # ========================================================

    # --------------------------------------------------------
    # Step 2: Find rare NAME tokens
    # --------------------------------------------------------

    print("\n[2/7] Finding rare name tokens...")

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

    rare_name_count = con.execute(
        """
        SELECT COUNT(*)
        FROM rare_name_tokens
        """
    ).fetchone()[0]

    print(f"Rare name tokens: {rare_name_count:,}")

    # --------------------------------------------------------
    # Step 3: Build NAME candidate index
    # --------------------------------------------------------

    print("\n[3/7] Generating rare name-token candidates...")

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

    con.execute(
        """
        CREATE TEMP TABLE name_candidates AS

        SELECT DISTINCT

            s1.entity_id AS source1_entity_id,
            c.entity_id AS candidate_entity_id

        FROM s1_name_tokens AS s1

        INNER JOIN candidate_name_tokens AS c

            ON s1.clean_country = c.clean_country
            AND s1.token = c.token
        """
    )

    name_candidate_count = con.execute(
        """
        SELECT COUNT(*)
        FROM name_candidates
        """
    ).fetchone()[0]

    print(
        f"Rare name-token candidates: "
        f"{name_candidate_count:,}"
    )

    # ========================================================
    # ADDRESS BLOCK
    # ========================================================

    # --------------------------------------------------------
    # Step 4: Find rare ADDRESS tokens
    # --------------------------------------------------------

    print("\n[4/7] Finding rare address tokens...")

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

    rare_address_count = con.execute(
        """
        SELECT COUNT(*)
        FROM rare_address_tokens
        """
    ).fetchone()[0]

    print(
        f"Rare address tokens: "
        f"{rare_address_count:,}"
    )

    # --------------------------------------------------------
    # Step 5: Build ADDRESS candidate index
    # --------------------------------------------------------

    print("\n[5/7] Generating rare address-token candidates...")

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

    con.execute(
        """
        CREATE TEMP TABLE address_candidates AS

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
        FROM address_candidates
        """
    ).fetchone()[0]

    print(
        f"Rare address-token candidates: "
        f"{address_candidate_count:,}"
    )

    # --------------------------------------------------------
    # Step 6: Combine V1 + NAME + ADDRESS
    # --------------------------------------------------------

    print("\n[6/7] Combining V1 + name + address candidates...")

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

            FROM name_candidates

            UNION

            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM address_candidates

        )

        TO '{V2_PATH}'

        (FORMAT PARQUET)
        """
    )

    # --------------------------------------------------------
    # Step 7: Final statistics
    # --------------------------------------------------------

    print("\n[7/7] Calculating final candidate count...")

    v2_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{V2_PATH}')
        """
    ).fetchone()[0]

    new_candidates = v2_count - v1_count

    print("\n" + "=" * 70)
    print("V2-C RESULTS")
    print("=" * 70)

    print(f"V1 candidate pairs:            {v1_count:,}")
    print(
        f"Rare name candidates:          "
        f"{name_candidate_count:,}"
    )
    print(
        f"Rare address candidates:       "
        f"{address_candidate_count:,}"
    )
    print(f"V2-C total candidates:          {v2_count:,}")
    print(f"New candidates added:           {new_candidates:,}")

    print("\nSaved to:")
    print(V2_PATH)

    con.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    generate_v2_combined_candidates()