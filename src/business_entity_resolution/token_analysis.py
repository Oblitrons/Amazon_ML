"""
Analyze token frequencies for Blocking V2.

We will use this analysis to decide which name/address
tokens are useful for token-based blocking.

Important:
- Do NOT use empty tokens.
- Analyze Source 2 + Source 3 together.
- Keep country in the analysis.
- Do not generate candidate pairs yet.
"""

from pathlib import Path
import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "cleaned_data"

S2_PATH = DATA_DIR / "train_source2.parquet"
S3_PATH = DATA_DIR / "train_source3.parquet"


def analyze_tokens():

    print("=" * 70)
    print("TOKEN FREQUENCY ANALYSIS")
    print("=" * 70)

    con = duckdb.connect()

    # ------------------------------------------------------------
    # Create combined candidate-source view
    # ------------------------------------------------------------

    print("\nLoading Source 2 + Source 3...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW candidates AS

        SELECT
            entity_id,
            clean_country,
            clean_name,
            clean_address
        FROM read_parquet(
            '{S2_PATH.as_posix()}'
        )

        UNION ALL

        SELECT
            entity_id,
            clean_country,
            clean_name,
            clean_address
        FROM read_parquet(
            '{S3_PATH.as_posix()}'
        )
        """
    )

    total_records = con.execute(
        """
        SELECT COUNT(*)
        FROM candidates
        """
    ).fetchone()[0]

    print(
        f"Total candidate records: {total_records:,}"
    )

    # ============================================================
    # NAME TOKEN ANALYSIS
    # ============================================================

    print("\n" + "=" * 70)
    print("NAME TOKEN FREQUENCY")
    print("=" * 70)

    name_tokens_query = """
        SELECT
            token,
            COUNT(*) AS frequency
        FROM candidates,
        UNNEST(
            STRING_SPLIT(clean_name, ' ')
        ) AS t(token)
        WHERE
            clean_name IS NOT NULL
            AND TRIM(clean_name) <> ''
            AND TRIM(token) <> ''
        GROUP BY token
    """

    name_token_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ({name_tokens_query})
        """
    ).fetchone()[0]

    print(
        f"\nUnique name tokens: "
        f"{name_token_count:,}"
    )

    print("\nMost frequent name tokens:")

    rows = con.execute(
        f"""
        SELECT
            token,
            frequency
        FROM ({name_tokens_query})
        ORDER BY frequency DESC
        LIMIT 30
        """
    ).fetchall()

    for token, frequency in rows:
        print(
            f"{token:<30} {frequency:>12,}"
        )

    # ------------------------------------------------------------
    # Name token frequency buckets
    # ------------------------------------------------------------

    print("\nName token frequency distribution:")

    buckets = [
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-50", 11, 50),
        ("51-100", 51, 100),
        ("101-500", 101, 500),
        ("501-1,000", 501, 1000),
        ("1,001-5,000", 1001, 5000),
        ("5,001-10,000", 5001, 10000),
        ("10,001+", 10001, None),
    ]

    for label, lower, upper in buckets:

        if upper is None:

            query = f"""
                SELECT COUNT(*)
                FROM ({name_tokens_query})
                WHERE frequency >= {lower}
            """

        else:

            query = f"""
                SELECT COUNT(*)
                FROM ({name_tokens_query})
                WHERE frequency BETWEEN {lower} AND {upper}
            """

        count = con.execute(query).fetchone()[0]

        print(
            f"{label:<15} {count:>12,} unique tokens"
        )

    # ============================================================
    # ADDRESS TOKEN ANALYSIS
    # ============================================================

    print("\n" + "=" * 70)
    print("ADDRESS TOKEN FREQUENCY")
    print("=" * 70)

    address_tokens_query = """
        SELECT
            token,
            COUNT(*) AS frequency
        FROM candidates,
        UNNEST(
            STRING_SPLIT(clean_address, ' ')
        ) AS t(token)
        WHERE
            clean_address IS NOT NULL
            AND TRIM(clean_address) <> ''
            AND TRIM(token) <> ''
        GROUP BY token
    """

    address_token_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ({address_tokens_query})
        """
    ).fetchone()[0]

    print(
        f"\nUnique address tokens: "
        f"{address_token_count:,}"
    )

    print("\nMost frequent address tokens:")

    rows = con.execute(
        f"""
        SELECT
            token,
            frequency
        FROM ({address_tokens_query})
        ORDER BY frequency DESC
        LIMIT 30
        """
    ).fetchall()

    for token, frequency in rows:
        print(
            f"{token:<30} {frequency:>12,}"
        )

    # ------------------------------------------------------------
    # Address token frequency buckets
    # ------------------------------------------------------------

    print("\nAddress token frequency distribution:")

    for label, lower, upper in buckets:

        if upper is None:

            query = f"""
                SELECT COUNT(*)
                FROM ({address_tokens_query})
                WHERE frequency >= {lower}
            """

        else:

            query = f"""
                SELECT COUNT(*)
                FROM ({address_tokens_query})
                WHERE frequency BETWEEN {lower} AND {upper}
            """

        count = con.execute(query).fetchone()[0]

        print(
            f"{label:<15} {count:>12,} unique tokens"
        )

    # ============================================================
    # COUNTRY-SPECIFIC TOKEN FREQUENCY
    # ============================================================

    print("\n" + "=" * 70)
    print("COUNTRY-SPECIFIC NAME TOKEN FREQUENCY")
    print("=" * 70)

    country_name_tokens_query = """
        SELECT
            clean_country,
            token,
            COUNT(*) AS frequency
        FROM candidates,
        UNNEST(
            STRING_SPLIT(clean_name, ' ')
        ) AS t(token)
        WHERE
            clean_name IS NOT NULL
            AND TRIM(clean_name) <> ''
            AND TRIM(token) <> ''
        GROUP BY
            clean_country,
            token
    """

    countries = con.execute(
        """
        SELECT DISTINCT clean_country
        FROM candidates
        ORDER BY clean_country
        """
    ).fetchall()

    for (country,) in countries:

        print(f"\nCountry: {country}")

        rows = con.execute(
            f"""
            SELECT
                token,
                frequency
            FROM ({country_name_tokens_query})
            WHERE clean_country = ?
            ORDER BY frequency DESC
            LIMIT 15
            """,
            [country]
        ).fetchall()

        for token, frequency in rows:

            print(
                f"  {token:<27} {frequency:>10,}"
            )

    print("\n" + "=" * 70)
    print("TOKEN ANALYSIS COMPLETE")
    print("=" * 70)

    con.close()


if __name__ == "__main__":
    analyze_tokens()