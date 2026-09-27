from pathlib import Path
import duckdb
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"

MATCHING = OUTPUT / "matching_results.tsv"
CANDIDATES = OUTPUT / "candidate_pairs.tsv"
S1 = OUTPUT / "test_parquet" / "test_source1.parquet"
S2 = OUTPUT / "test_parquet" / "test_source2.parquet"
S3 = OUTPUT / "test_parquet" / "test_source3.parquet"


def main():
    required = [MATCHING, CANDIDATES, S1, S2, S3]

    for p in required:
        if not p.exists():
            print(f"MISSING: {p}")
            sys.exit(1)

    con = duckdb.connect()

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s1 AS
        SELECT DISTINCT CAST(entity_id AS VARCHAR) AS id
        FROM read_parquet('{S1.as_posix()}')
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s2 AS
        SELECT DISTINCT CAST(entity_id AS VARCHAR) AS id
        FROM read_parquet('{S2.as_posix()}')
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s3 AS
        SELECT DISTINCT CAST(entity_id AS VARCHAR) AS id
        FROM read_parquet('{S3.as_posix()}')
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW matches AS
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
            matched_entity_ids
        FROM read_csv_auto(
            '{MATCHING.as_posix()}',
            delim='\\t',
            header=true
        )
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW candidates AS
        SELECT
            CAST(source1_entity_id AS VARCHAR) AS source1_entity_id,
            candidate_entity_ids
        FROM read_csv_auto(
            '{CANDIDATES.as_posix()}',
            delim='\\t',
            header=true
        )
    """)

    s1_count = con.execute("SELECT COUNT(*) FROM s1").fetchone()[0]
    match_count = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    candidate_count = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    print(f"Source-1 entities:       {s1_count:,}")
    print(f"matching_results rows:   {match_count:,}")
    print(f"candidate_pairs rows:    {candidate_count:,}")

    if match_count != s1_count:
        raise RuntimeError("matching_results.tsv row count mismatch")

    if candidate_count != s1_count:
        raise RuntimeError("candidate_pairs.tsv row count mismatch")

    duplicate_s1_matches = con.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT source1_entity_id, COUNT(*) n
            FROM matches
            GROUP BY source1_entity_id
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    duplicate_s1_candidates = con.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT source1_entity_id, COUNT(*) n
            FROM candidates
            GROUP BY source1_entity_id
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    if duplicate_s1_matches or duplicate_s1_candidates:
        raise RuntimeError("Duplicate Source-1 rows detected")

    # Every prediction must be contained in the candidate set.
    outside = con.execute("""
        WITH predicted AS (
            SELECT
                m.source1_entity_id,
                TRIM(x.id) AS candidate_id
            FROM matches m,
            UNNEST(string_split(
                COALESCE(m.matched_entity_ids, ''),
                ','
            )) AS x(id)
            WHERE TRIM(x.id) <> ''
        ),
        allowed AS (
            SELECT
                c.source1_entity_id,
                TRIM(x.id) AS candidate_id
            FROM candidates c,
            UNNEST(string_split(
                COALESCE(c.candidate_entity_ids, ''),
                ','
            )) AS x(id)
            WHERE TRIM(x.id) <> ''
        )
        SELECT COUNT(*)
        FROM predicted p
        LEFT JOIN allowed a
          ON p.source1_entity_id = a.source1_entity_id
         AND p.candidate_id = a.candidate_id
        WHERE a.candidate_id IS NULL
    """).fetchone()[0]

    if outside:
        raise RuntimeError(
            f"{outside:,} predicted IDs are outside the candidate set"
        )

    # Predicted IDs must exist in S2 or S3.
    invalid_ids = con.execute("""
        WITH predicted AS (
            SELECT DISTINCT TRIM(x.id) AS candidate_id
            FROM matches m,
            UNNEST(string_split(
                COALESCE(m.matched_entity_ids, ''),
                ','
            )) AS x(id)
            WHERE TRIM(x.id) <> ''
        ),
        valid_ids AS (
            SELECT id FROM s2
            UNION
            SELECT id FROM s3
        )
        SELECT COUNT(*)
        FROM predicted p
        LEFT JOIN valid_ids v
          ON p.candidate_id = v.id
        WHERE v.id IS NULL
    """).fetchone()[0]

    if invalid_ids:
        raise RuntimeError(
            f"{invalid_ids:,} predicted IDs do not exist in test S2/S3"
        )

    print()
    print("ALL FINAL SUBMISSION CHECKS PASSED")


if __name__ == "__main__":
    main()
