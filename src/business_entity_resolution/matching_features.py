from pathlib import Path
import argparse
import duckdb


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = ROOT / "data" / "cleaned_data"
DEFAULT_OUTPUT = ROOT / "output" / "pair_features.parquet"


def main():
    parser = argparse.ArgumentParser(
        description="Generate pairwise matching features."
    )

    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Candidate pairs parquet."
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory containing train_source1/2/3.parquet."
    )

    parser.add_argument("--source1", type=Path)
    parser.add_argument("--source2", type=Path)
    parser.add_argument("--source3", type=Path)

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    args.candidates = args.candidates.resolve()
    args.output = args.output.resolve()

    if args.source1 and args.source2 and args.source3:
        source1 = args.source1.resolve()
        source2 = args.source2.resolve()
        source3 = args.source3.resolve()
    else:
        source_dir = (
            args.source_dir.resolve()
            if args.source_dir
            else DEFAULT_SOURCE_DIR.resolve()
        )

        source1 = source_dir / "train_source1.parquet"
        source2 = source_dir / "train_source2.parquet"
        source3 = source_dir / "train_source3.parquet"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PAIR FEATURE GENERATION")
    print("=" * 70)

    print(f"\nCandidates: {args.candidates}")
    print(f"Source 1:   {source1}")
    print(f"Source 2:   {source2}")
    print(f"Source 3:   {source3}")
    print(f"Output:     {args.output}")

    if not args.candidates.exists():
        raise FileNotFoundError(
            f"\nCandidate file not found:\n{args.candidates}"
        )

    for path in [source1, source2, source3]:
        if not path.exists():
            raise FileNotFoundError(
                f"\nSource file not found:\n{path}"
            )

    con = duckdb.connect()

    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")

    # ------------------------------------------------------------
    # Candidate pairs
    # ------------------------------------------------------------

    print("\nCreating candidate view...")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW candidates AS
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
            CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id
        FROM read_parquet('{args.candidates.as_posix()}')
        """
    )

    candidate_count = con.execute(
        """
        SELECT COUNT(*)
        FROM candidates
        """
    ).fetchone()[0]

    print(f"Candidate pairs: {candidate_count:,}")

    # ------------------------------------------------------------
    # Source 1
    # ------------------------------------------------------------

    print("\nCreating source1 view...")

    def source_relation(path):
        p = path.as_posix()

        if path.suffix.lower() == ".parquet":
            return f"read_parquet('{p}')"

        return f"""
            read_csv_auto(
                '{p}',
                delim='\\t',
                header=true,
                ignore_errors=false
            )
        """

    con.execute(
        f"""
        CREATE OR REPLACE VIEW source1 AS
        SELECT *
        FROM {source_relation(source1)}
        """
    )

    # ------------------------------------------------------------
    # S2 + S3
    # ------------------------------------------------------------

    print("Creating source2/source3 combined view...")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW candidates_source AS

        SELECT
            CAST(entity_id AS VARCHAR) AS entity_id,
            CAST(clean_name AS VARCHAR) AS clean_name,
            CAST(clean_address AS VARCHAR) AS clean_address,
            CAST(clean_country AS VARCHAR) AS clean_country
        FROM {source_relation(source2)}

        UNION ALL

        SELECT
            CAST(entity_id AS VARCHAR) AS entity_id,
            CAST(clean_name AS VARCHAR) AS clean_name,
            CAST(clean_address AS VARCHAR) AS clean_address,
            CAST(clean_country AS VARCHAR) AS clean_country
        FROM {source_relation(source3)}
        """
    )

    # ------------------------------------------------------------
    # Normalize Source 1 column names
    # ------------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE VIEW source1_clean AS
        SELECT
            CAST(entity_id AS VARCHAR) AS entity_id,
            CAST(clean_name AS VARCHAR) AS clean_name,
            CAST(clean_address AS VARCHAR) AS clean_address,
            CAST(clean_country AS VARCHAR) AS clean_country
        FROM {source_relation(source1)}
        """
    )

    # ------------------------------------------------------------
    # Build joined pair table
    # ------------------------------------------------------------

    print("\nJoining candidate pairs to source records...")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE pair_data AS

        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,

            s1.clean_name AS name1,
            s1.clean_address AS address1,
            s1.clean_country AS country1,

            s2.clean_name AS name2,
            s2.clean_address AS address2,
            s2.clean_country AS country2

        FROM candidates c

        INNER JOIN source1_clean s1
            ON c.source1_entity_id = s1.entity_id

        INNER JOIN candidates_source s2
            ON c.candidate_entity_id = s2.entity_id
        """
    )

    joined_count = con.execute(
        """
        SELECT COUNT(*)
        FROM pair_data
        """
    ).fetchone()[0]

    print(f"Successfully joined pairs: {joined_count:,}")

    # ------------------------------------------------------------
    # Deterministic features
    # ------------------------------------------------------------

    print("\nGenerating deterministic features...")

    con.execute(
        """
        CREATE OR REPLACE TABLE pair_features AS

        SELECT

            source1_entity_id,
            candidate_entity_id,

            CAST(
                LOWER(TRIM(COALESCE(name1, ''))) =
                LOWER(TRIM(COALESCE(name2, '')))
                AS INTEGER
            ) AS name_exact,

            CAST(
                LOWER(TRIM(COALESCE(address1, ''))) =
                LOWER(TRIM(COALESCE(address2, '')))
                AS INTEGER
            ) AS address_exact,

            CAST(
                LOWER(TRIM(COALESCE(country1, ''))) =
                LOWER(TRIM(COALESCE(country2, '')))
                AS INTEGER
            ) AS country_exact,

            CAST(
                name1 IS NULL OR TRIM(name1) = ''
                AS INTEGER
            ) AS name1_missing,

            CAST(
                name2 IS NULL OR TRIM(name2) = ''
                AS INTEGER
            ) AS name2_missing,

            CAST(
                address1 IS NULL OR TRIM(address1) = ''
                AS INTEGER
            ) AS address1_missing,

            CAST(
                address2 IS NULL OR TRIM(address2) = ''
                AS INTEGER
            ) AS address2_missing,

            LENGTH(COALESCE(name1, '')) AS name1_length,
            LENGTH(COALESCE(name2, '')) AS name2_length,

            ABS(
                LENGTH(COALESCE(name1, '')) -
                LENGTH(COALESCE(name2, ''))
            ) AS name_length_diff,

            LENGTH(COALESCE(address1, '')) AS address1_length,
            LENGTH(COALESCE(address2, '')) AS address2_length,

            ABS(
                LENGTH(COALESCE(address1, '')) -
                LENGTH(COALESCE(address2, ''))
            ) AS address_length_diff,

            name1,
            name2,
            address1,
            address2,
            country1,
            country2

        FROM pair_data
        """
    )

    feature_count = con.execute(
        """
        SELECT COUNT(*)
        FROM pair_features
        """
    ).fetchone()[0]

    print(f"Feature rows: {feature_count:,}")

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    print(f"\nSaving to:\n{args.output}")

    con.execute(
        f"""
        COPY pair_features
        TO '{args.output.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    print("\nDONE")
    print("=" * 70)

    con.close()


if __name__ == "__main__":
    main()

