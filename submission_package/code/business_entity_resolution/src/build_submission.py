from pathlib import Path
import argparse
import duckdb


def main():
    parser = argparse.ArgumentParser(
        description="Build final Amazon ML challenge submission files."
    )

    parser.add_argument(
        "--source1",
        required=True,
        help="Test Source-1 parquet."
    )

    parser.add_argument(
        "--candidates",
        required=True,
        help="Final candidate-pair parquet."
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help="Pair-level prediction parquet."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for matching_results.tsv and candidate_pairs.tsv."
    )

    args = parser.parse_args()

    source1 = Path(args.source1).resolve()
    candidates = Path(args.candidates).resolve()
    predictions = Path(args.predictions).resolve()
    output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    matching_output = output_dir / "matching_results.tsv"
    candidate_output = output_dir / "candidate_pairs.tsv"

    for path in [source1, candidates, predictions]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    con = duckdb.connect()

    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")

    print("=" * 70)
    print("FINAL CHALLENGE SUBMISSION AGGREGATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Source 1: authoritative list of entities.
    # ------------------------------------------------------------

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW source1 AS
        SELECT DISTINCT
            CAST(entity_id AS VARCHAR) AS source1_entity_id
        FROM read_parquet('{source1.as_posix()}')
    """)

    s1_count = con.execute("""
        SELECT COUNT(*)
        FROM source1
    """).fetchone()[0]

    print(f"Source-1 entities: {s1_count:,}")

    # ------------------------------------------------------------
    # Final candidates.
    #
    # DISTINCT prevents duplicate candidate IDs per S1.
    # ------------------------------------------------------------

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW candidate_pairs AS
        SELECT DISTINCT
            CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
            CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id
        FROM read_parquet('{candidates.as_posix()}')
        WHERE source1_entity_id IS NOT NULL
          AND candidate_entity_id IS NOT NULL
    """)

    candidate_count = con.execute("""
        SELECT COUNT(*)
        FROM candidate_pairs
    """).fetchone()[0]

    print(f"Candidate pairs: {candidate_count:,}")

    # ------------------------------------------------------------
    # Model-positive predictions.
    # ------------------------------------------------------------

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW positive_pairs AS
        SELECT DISTINCT
            CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
            CAST(candidate_entity_id AS VARCHAR) AS candidate_entity_id
        FROM read_parquet('{predictions.as_posix()}')
        WHERE prediction = 1
          AND source1_entity_id IS NOT NULL
          AND candidate_entity_id IS NOT NULL
    """)

    positive_count = con.execute("""
        SELECT COUNT(*)
        FROM positive_pairs
    """).fetchone()[0]

    print(f"Positive predicted pairs: {positive_count:,}")

    # ------------------------------------------------------------
    # Safety invariant:
    #
    # Every predicted match MUST exist in the final candidate set.
    # ------------------------------------------------------------

    outside_candidates = con.execute("""
        SELECT COUNT(*)
        FROM positive_pairs p
        LEFT JOIN candidate_pairs c
          ON p.source1_entity_id = c.source1_entity_id
         AND p.candidate_entity_id = c.candidate_entity_id
        WHERE c.candidate_entity_id IS NULL
    """).fetchone()[0]

    if outside_candidates != 0:
        raise RuntimeError(
            "INVALID SUBMISSION: predicted matches exist outside "
            f"the candidate set ({outside_candidates:,} pairs)."
        )

    print("Candidate-set invariant: PASS")

    # ------------------------------------------------------------
    # Candidate output.
    #
    # Exactly one row per S1.
    # Empty string for entities with no candidates.
    # ------------------------------------------------------------

    con.execute(f"""
        COPY (
            SELECT
                s.source1_entity_id,

                COALESCE(
                    STRING_AGG(
                        c.candidate_entity_id,
                        ',' ORDER BY c.candidate_entity_id
                    ),
                    ''
                ) AS candidate_entity_ids

            FROM source1 s

            LEFT JOIN candidate_pairs c
              ON s.source1_entity_id = c.source1_entity_id

            GROUP BY
                s.source1_entity_id

            ORDER BY
                s.source1_entity_id
        )
        TO '{candidate_output.as_posix()}'
        (
            FORMAT CSV,
            DELIMITER '\\t',
            HEADER TRUE
        )
    """)

    # ------------------------------------------------------------
    # Matching output.
    #
    # Exactly one row per S1.
    # Empty string = singleton/no predicted match.
    # ------------------------------------------------------------

    con.execute(f"""
        COPY (
            SELECT
                s.source1_entity_id,

                COALESCE(
                    STRING_AGG(
                        p.candidate_entity_id,
                        ',' ORDER BY p.candidate_entity_id
                    ),
                    ''
                ) AS matched_entity_ids

            FROM source1 s

            LEFT JOIN positive_pairs p
              ON s.source1_entity_id = p.source1_entity_id

            GROUP BY
                s.source1_entity_id

            ORDER BY
                s.source1_entity_id
        )
        TO '{matching_output.as_posix()}'
        (
            FORMAT CSV,
            DELIMITER '\\t',
            HEADER TRUE
        )
    """)

    # ------------------------------------------------------------
    # Validate exact row counts.
    # ------------------------------------------------------------

    candidate_rows = con.execute(f"""
        SELECT COUNT(*)
        FROM read_csv(
            '{candidate_output.as_posix()}',
            delim='\\t',
            header=true
        )
    """).fetchone()[0]

    matching_rows = con.execute(f"""
        SELECT COUNT(*)
        FROM read_csv(
            '{matching_output.as_posix()}',
            delim='\\t',
            header=true
        )
    """).fetchone()[0]

    print()
    print(f"candidate_pairs.tsv rows:  {candidate_rows:,}")
    print(f"matching_results.tsv rows: {matching_rows:,}")

    if candidate_rows != s1_count:
        raise RuntimeError(
            "candidate_pairs.tsv does not contain exactly one row "
            "per Source-1 entity."
        )

    if matching_rows != s1_count:
        raise RuntimeError(
            "matching_results.tsv does not contain exactly one row "
            "per Source-1 entity."
        )

    print("One-row-per-S1 invariant: PASS")

    print()
    print("FINAL OUTPUTS")
    print(f"  {matching_output}")
    print(f"  {candidate_output}")
    print()
    print("AGGREGATION COMPLETE")
    print("=" * 70)

    con.close()


if __name__ == "__main__":
    main()
