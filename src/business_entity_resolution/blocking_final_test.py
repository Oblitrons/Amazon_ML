from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"

S1 = OUT / "test_parquet" / "test_source1.parquet"
S2 = OUT / "test_parquet" / "test_source2.parquet"
S3 = OUT / "test_parquet" / "test_source3.parquet"

OUTPUT = OUT / "candidate_pairs_final_test.parquet"

NAME_CAP = 200
ADDRESS_CAP = 100


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    
    temp_dir = ROOT / ".tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{temp_dir.as_posix()}'")

    print("=" * 70)
    print("FINAL TEST BLOCKER")
    print("=" * 70)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW s1 AS
        SELECT
            CAST(entity_id AS VARCHAR) AS source1_entity_id,
            CAST(clean_name AS VARCHAR) AS clean_name,
            CAST(clean_address AS VARCHAR) AS clean_address,
            CAST(clean_country AS VARCHAR) AS clean_country
        FROM read_parquet('{S1.as_posix()}')
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW candidates AS
        SELECT
            CAST(entity_id AS VARCHAR) AS candidate_entity_id,
            CAST(clean_name AS VARCHAR) AS clean_name,
            CAST(clean_address AS VARCHAR) AS clean_address,
            CAST(clean_country AS VARCHAR) AS clean_country
        FROM read_parquet('{S2.as_posix()}')

        UNION ALL

        SELECT
            CAST(entity_id AS VARCHAR) AS candidate_entity_id,
            CAST(clean_name AS VARCHAR) AS clean_name,
            CAST(clean_address AS VARCHAR) AS clean_address,
            CAST(clean_country AS VARCHAR) AS clean_country
        FROM read_parquet('{S3.as_posix()}')
    """)

    print("\n[1/3] Exact-name candidates...")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE name_candidates AS
        SELECT
            s.source1_entity_id,
            c.candidate_entity_id
        FROM s1 s
        INNER JOIN candidates c
          ON s.clean_country = c.clean_country
         AND s.clean_name = c.clean_name
        WHERE s.clean_name IS NOT NULL
          AND TRIM(s.clean_name) <> ''
          AND c.clean_name IS NOT NULL
          AND TRIM(c.clean_name) <> ''
          AND (
              SELECT COUNT(*)
              FROM candidates c2
              WHERE c2.clean_country = c.clean_country
                AND c2.clean_name = c.clean_name
          ) <= {NAME_CAP}
    """)

    name_count = con.execute(
        "SELECT COUNT(*) FROM name_candidates"
    ).fetchone()[0]

    print(f"Exact-name candidates: {name_count:,}")

    print("\n[2/3] Exact-address candidates...")

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE address_candidates AS
        SELECT
            s.source1_entity_id,
            c.candidate_entity_id
        FROM s1 s
        INNER JOIN candidates c
          ON s.clean_country = c.clean_country
         AND s.clean_address = c.clean_address
        WHERE s.clean_address IS NOT NULL
          AND TRIM(s.clean_address) <> ''
          AND c.clean_address IS NOT NULL
          AND TRIM(c.clean_address) <> ''
          AND (
              SELECT COUNT(*)
              FROM candidates c2
              WHERE c2.clean_country = c.clean_country
                AND c2.clean_address = c.clean_address
          ) <= {ADDRESS_CAP}
    """)

    address_count = con.execute(
        "SELECT COUNT(*) FROM address_candidates"
    ).fetchone()[0]

    print(f"Exact-address candidates: {address_count:,}")

    print("\n[3/3] Combining candidates...")

    con.execute("""
        CREATE OR REPLACE TEMP TABLE final_candidates AS
        SELECT source1_entity_id, candidate_entity_id
        FROM name_candidates

        UNION

        SELECT source1_entity_id, candidate_entity_id
        FROM address_candidates
    """)

    final_count = con.execute(
        "SELECT COUNT(*) FROM final_candidates"
    ).fetchone()[0]

    covered = con.execute("""
        SELECT COUNT(DISTINCT source1_entity_id)
        FROM final_candidates
    """).fetchone()[0]

    s1_count = con.execute(
        "SELECT COUNT(*) FROM s1"
    ).fetchone()[0]

    con.execute(f"""
        COPY (
            SELECT
                source1_entity_id,
                candidate_entity_id
            FROM final_candidates
        )
        TO '{OUTPUT.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    print()
    print(f"S1 records:              {s1_count:,}")
    print(f"S1 with candidates:      {covered:,}")
    print(f"Final candidate pairs:   {final_count:,}")
    print(f"Saved: {OUTPUT}")

    con.close()


if __name__ == "__main__":
    main()

