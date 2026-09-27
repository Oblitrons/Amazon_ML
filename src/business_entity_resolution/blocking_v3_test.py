from pathlib import Path
import duckdb


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "cleaned_data"
OUTPUT_DIR = ROOT_DIR / "output"

S1_PATH = ROOT_DIR / "output" / "test_parquet" / "test_source1.parquet"
S2_PATH = ROOT_DIR / "output" / "test_parquet" / "test_source2.parquet"
S3_PATH = ROOT_DIR / "output" / "test_parquet" / "test_source3.parquet"

V2_C_PATH = OUTPUT_DIR / "candidate_pairs_v2_empty_test.parquet"

OUTPUT_PATH = (
    OUTPUT_DIR /
    "candidate_pairs_v3_test.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Character n-gram length.
#
# Example:
# "digital"
#
# becomes:
# digi
# igit
# gita
# ital
#
NGRAM_SIZE = 4


# Ignore character n-grams that occur too frequently.
#
# Generic fragments such as:
# " ltd"
# " inc"
# "pvt "
# etc.
# can occur millions of times.
#
# Starting with 500 keeps the blocking selective.
MAX_NGRAM_FREQUENCY = 500


# Ignore very rare grams that occur only once.
#
# A typo can create a unique gram, but a gram occurring only
# once cannot generate a useful cross-record block unless the
# same typo happens to occur in both records.
MIN_NGRAM_FREQUENCY = 2


# Maximum number of fuzzy-name candidates retained per S1.
TOP_K = 20


# Only use a limited number of informative character grams
# from each S1 name.
#
# This prevents long names from generating huge candidate sets.
MAX_S1_NGRAMS = 20


# DuckDB memory settings.
DUCKDB_MEMORY = "8GB"
DUCKDB_THREADS = 4


# ============================================================
# CONNECTION
# ============================================================

def create_connection():

    con = duckdb.connect()

    con.execute(
        f"SET memory_limit='{DUCKDB_MEMORY}'"
    )

    con.execute(
        f"SET threads={DUCKDB_THREADS}"
    )

    con.execute(
        "SET preserve_insertion_order=false"
    )

    return con


# ============================================================
# BUILD SOURCE TABLES
# ============================================================

def build_source_tables(con):

    print("\n[1/6] Loading source tables...")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE source1 AS
        SELECT
            entity_id,
            clean_name,
            clean_country
        FROM read_parquet('{S1_PATH}')
        WHERE clean_name IS NOT NULL
          AND TRIM(clean_name) <> ''
          AND clean_country IS NOT NULL
          AND TRIM(clean_country) <> ''
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE candidates AS

        SELECT
            entity_id,
            clean_name,
            clean_country
        FROM read_parquet('{S2_PATH}')
        WHERE clean_name IS NOT NULL
          AND TRIM(clean_name) <> ''
          AND clean_country IS NOT NULL
          AND TRIM(clean_country) <> ''

        UNION ALL

        SELECT
            entity_id,
            clean_name,
            clean_country
        FROM read_parquet('{S3_PATH}')
        WHERE clean_name IS NOT NULL
          AND TRIM(clean_name) <> ''
          AND clean_country IS NOT NULL
          AND TRIM(clean_country) <> ''
    """)

    s1_count = con.execute("""
        SELECT COUNT(*)
        FROM source1
    """).fetchone()[0]

    candidate_count = con.execute("""
        SELECT COUNT(*)
        FROM candidates
    """).fetchone()[0]

    print(
        f"S1 records:       {s1_count:,}"
    )

    print(
        f"S2 + S3 records:  {candidate_count:,}"
    )


# ============================================================
# CREATE CHARACTER N-GRAMS
# ============================================================

def build_candidate_ngrams(con):

    print(
        "\n[2/6] Building candidate character n-grams..."
    )

    print(
        f"N-gram size: {NGRAM_SIZE}"
    )

    print(
        f"Frequency range: "
        f"{MIN_NGRAM_FREQUENCY} - "
        f"{MAX_NGRAM_FREQUENCY}"
    )

    # We generate n-grams using DuckDB's range().
    #
    # For a string:
    #
    #   abcdef
    #
    # with NGRAM_SIZE = 4:
    #
    #   abcd
    #   bcde
    #   cdef
    #
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE candidate_ngrams_raw AS

        SELECT
            entity_id,
            clean_country,
            LOWER(
                SUBSTR(
                    clean_name,
                    r.i,
                    {NGRAM_SIZE}
                )
            ) AS ngram

        FROM candidates

        CROSS JOIN LATERAL range(
            1,
            GREATEST(
                LENGTH(clean_name) - {NGRAM_SIZE} + 2,
                1
            )
        ) AS r(i)

        WHERE LENGTH(clean_name) >= {NGRAM_SIZE}
    """)

    raw_count = con.execute("""
        SELECT COUNT(*)
        FROM candidate_ngrams_raw
    """).fetchone()[0]

    print(
        f"Raw candidate n-gram rows: "
        f"{raw_count:,}"
    )

    print(
        "\nCalculating n-gram frequencies..."
    )

    con.execute("""
        CREATE OR REPLACE TEMP TABLE candidate_ngram_frequency AS

        SELECT
            clean_country,
            ngram,
            COUNT(*) AS frequency

        FROM candidate_ngrams_raw

        GROUP BY
            clean_country,
            ngram
    """)

    total_unique = con.execute("""
        SELECT COUNT(*)
        FROM candidate_ngram_frequency
    """).fetchone()[0]

    print(
        f"Unique country-specific n-grams: "
        f"{total_unique:,}"
    )

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE useful_ngrams AS

        SELECT
            clean_country,
            ngram,
            frequency

        FROM candidate_ngram_frequency

        WHERE frequency >= {MIN_NGRAM_FREQUENCY}
          AND frequency <= {MAX_NGRAM_FREQUENCY}
    """)

    useful_count = con.execute("""
        SELECT COUNT(*)
        FROM useful_ngrams
    """).fetchone()[0]

    print(
        f"Useful n-grams: "
        f"{useful_count:,}"
    )


# ============================================================
# BUILD CANDIDATE INVERTED INDEX
# ============================================================

def build_candidate_index(con):

    print(
        "\n[3/6] Building candidate inverted index..."
    )

    # Keep only useful n-grams.
    #
    # DISTINCT prevents repeated character n-grams inside the
    # same business name from artificially increasing overlap.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE candidate_index AS

        SELECT DISTINCT
            n.entity_id,
            n.clean_country,
            n.ngram,
            f.frequency

        FROM candidate_ngrams_raw AS n

        INNER JOIN useful_ngrams AS f
            ON n.clean_country = f.clean_country
           AND n.ngram = f.ngram
    """)

    index_rows = con.execute("""
        SELECT COUNT(*)
        FROM candidate_index
    """).fetchone()[0]

    print(
        f"Candidate index rows: "
        f"{index_rows:,}"
    )

    # Raw n-grams are no longer required.
    con.execute(
        "DROP TABLE candidate_ngrams_raw"
    )

    con.execute(
        "DROP TABLE candidate_ngram_frequency"
    )


# ============================================================
# BUILD S1 N-GRAMS
# ============================================================

def build_s1_ngrams(con):

    print(
        "\n[4/6] Building S1 character n-grams..."
    )

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1_ngrams_raw AS

        SELECT
            entity_id,
            clean_country,
            LOWER(
                SUBSTR(
                    clean_name,
                    r.i,
                    {NGRAM_SIZE}
                )
            ) AS ngram

        FROM source1

        CROSS JOIN LATERAL range(
            1,
            GREATEST(
                LENGTH(clean_name) - {NGRAM_SIZE} + 2,
                1
            )
        ) AS r(i)

        WHERE LENGTH(clean_name) >= {NGRAM_SIZE}
    """)

    # Only retain useful grams.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE s1_useful_ngrams AS

        SELECT DISTINCT
            s.entity_id,
            s.clean_country,
            s.ngram,
            u.frequency

        FROM s1_ngrams_raw AS s

        INNER JOIN useful_ngrams AS u
            ON s.clean_country = u.clean_country
           AND s.ngram = u.ngram
    """)

    useful_rows = con.execute("""
        SELECT COUNT(*)
        FROM s1_useful_ngrams
    """).fetchone()[0]

    print(
        f"S1 useful n-gram rows: "
        f"{useful_rows:,}"
    )

    con.execute(
        "DROP TABLE s1_ngrams_raw"
    )

    # --------------------------------------------------------
    # Keep only the most informative grams per S1.
    #
    # Lower frequency = more selective = more useful.
    # --------------------------------------------------------

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE s1_block_ngrams AS

        SELECT
            entity_id,
            clean_country,
            ngram,
            frequency

        FROM (

            SELECT
                entity_id,
                clean_country,
                ngram,
                frequency,

                ROW_NUMBER() OVER (
                    PARTITION BY entity_id
                    ORDER BY
                        frequency ASC,
                        ngram
                ) AS rn

            FROM s1_useful_ngrams

        )

        WHERE rn <= {MAX_S1_NGRAMS}
    """)

    selected_rows = con.execute("""
        SELECT COUNT(*)
        FROM s1_block_ngrams
    """).fetchone()[0]

    print(
        f"S1 selected blocking grams: "
        f"{selected_rows:,}"
    )

    con.execute(
        "DROP TABLE s1_useful_ngrams"
    )


# ============================================================
# GENERATE FUZZY-NAME CANDIDATES
# ============================================================

def generate_candidates(con):

    print(
        "\n[5/6] Generating fuzzy-name candidates..."
    )

    print(
        "\nJoining S1 and candidate records "
        "through shared character n-grams..."
    )

    # --------------------------------------------------------
    # Step 1:
    #
    # Find records sharing at least one useful character gram.
    #
    # Score them by:
    #
    #   shared_ngrams
    #
    # and give more importance to rare grams using:
    #
    #   1 / frequency
    #
    # This means a rare shared fragment contributes more than
    # a very common fragment.
    # --------------------------------------------------------

    con.execute("""
        CREATE OR REPLACE TEMP TABLE fuzzy_pair_scores AS

        SELECT
            s.entity_id AS source1_entity_id,
            c.entity_id AS candidate_entity_id,

            COUNT(*) AS shared_ngrams,

            SUM(
                1.0 / GREATEST(c.frequency, 1)
            ) AS rarity_score

        FROM s1_block_ngrams AS s

        INNER JOIN candidate_index AS c

            ON s.clean_country = c.clean_country
           AND s.ngram = c.ngram

        GROUP BY
            s.entity_id,
            c.entity_id
    """)

    raw_pairs = con.execute("""
        SELECT COUNT(*)
        FROM fuzzy_pair_scores
    """).fetchone()[0]

    print(
        f"Raw fuzzy-name pairs: "
        f"{raw_pairs:,}"
    )

    # --------------------------------------------------------
    # Step 2:
    #
    # Rank candidates separately for every S1 entity.
    #
    # Ranking:
    #
    # 1. shared n-grams
    # 2. rarity score
    #
    # This is blocking, not final matching.
    # --------------------------------------------------------

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fuzzy_topk AS

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM (

            SELECT
                source1_entity_id,
                candidate_entity_id,

                shared_ngrams,
                rarity_score,

                ROW_NUMBER() OVER (

                    PARTITION BY source1_entity_id

                    ORDER BY
                        shared_ngrams DESC,
                        rarity_score DESC,
                        candidate_entity_id

                ) AS rn

            FROM fuzzy_pair_scores

        )

        WHERE rn <= {TOP_K}
    """)

    topk_count = con.execute("""
        SELECT COUNT(*)
        FROM fuzzy_topk
    """).fetchone()[0]

    covered_s1 = con.execute("""
        SELECT COUNT(DISTINCT source1_entity_id)
        FROM fuzzy_topk
    """).fetchone()[0]

    print(
        f"Top-K fuzzy candidates: "
        f"{topk_count:,}"
    )

    print(
        f"S1 entities receiving fuzzy candidates: "
        f"{covered_s1:,}"
    )

    # --------------------------------------------------------
    # Step 3:
    #
    # Combine V2-C and V3 fuzzy candidates.
    # --------------------------------------------------------

    print(
        "\nCombining V2-C + fuzzy-name candidates..."
    )

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE final_candidates AS

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM read_parquet(
            '{V2_C_PATH}'
        )

        UNION

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM fuzzy_topk
    """)

    final_count = con.execute("""
        SELECT COUNT(*)
        FROM final_candidates
    """).fetchone()[0]

    v2_count = con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet('{V2_C_PATH}')
    """).fetchone()[0]

    new_candidates = con.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM fuzzy_topk

            EXCEPT

            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM read_parquet('{V2_C_PATH}')
        )
    """).fetchone()[0]

    print(
        f"\nV2-C candidates:       {v2_count:,}"
    )

    print(
        f"New fuzzy candidates:   {new_candidates:,}"
    )

    print(
        f"Final V3 candidates:    {final_count:,}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    con.execute(f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id

            FROM final_candidates
        )

        TO '{OUTPUT_PATH}'
        (FORMAT PARQUET)
    """)

    print(
        f"\nSaved V3 candidate file:"
        f"\n{OUTPUT_PATH}"
    )

    return final_count


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V3 FUZZY NAME BLOCKING")
    print("=" * 70)

    print("\nConfiguration:")
    print(
        f"  NGRAM_SIZE:            {NGRAM_SIZE}"
    )
    print(
        f"  MIN_NGRAM_FREQUENCY:   {MIN_NGRAM_FREQUENCY}"
    )
    print(
        f"  MAX_NGRAM_FREQUENCY:   {MAX_NGRAM_FREQUENCY}"
    )
    print(
        f"  MAX_S1_NGRAMS:         {MAX_S1_NGRAMS}"
    )
    print(
        f"  TOP_K:                 {TOP_K}"
    )
    print(
        f"  DuckDB memory:         {DUCKDB_MEMORY}"
    )
    print(
        f"  DuckDB threads:        {DUCKDB_THREADS}"
    )

    con = create_connection()

    try:

        # 1
        build_source_tables(con)

        # 2
        build_candidate_ngrams(con)

        # 3
        build_candidate_index(con)

        # 4
        build_s1_ngrams(con)

        # 5
        final_count = generate_candidates(con)

        print("\n[6/6] DONE")
        print("=" * 70)

        print(
            f"\nFinal V3 candidate pairs: "
            f"{final_count:,}"
        )

        print(
            "\nNow evaluate V3 with:"
        )

        print(
            "python "
            "src/business_entity_resolution/"
            "evaluate_blocking.py "
            "output/candidate_pairs_v3_fuzzy_name.parquet"
        )

        print(
            "\nV2-C baseline:"
        )

        print(
            "  Recall:      56.660764%"
        )

        print(
            "  Candidates:  46,856,081"
        )

        print(
            "\nSend me the evaluator output before changing"
            " any parameters."
        )

    finally:

        con.close()


if __name__ == "__main__":
    main()


