#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
SEED_FILE = ROOT / "rag" / "data" / "acts_seed.json"
SCHEMA_FILE = ROOT / "scripts" / "legal_acts_schema.sql"


def main() -> None:
    load_dotenv(ENV_PATH)
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN not set in backend/.env")

    if not SEED_FILE.exists():
        raise RuntimeError(f"Seed file missing: {SEED_FILE}")

    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    # Ensure schema exists
    cur.execute(SCHEMA_FILE.read_text(encoding="utf-8"))

    acts = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    for row in acts:
        canonical = row.get("canonical_name")
        aliases = row.get("aliases") or []
        if not canonical:
            continue

        cur.execute(
            """
            INSERT INTO legal_acts (canonical_name, is_active)
            VALUES (%s, TRUE)
            ON CONFLICT (canonical_name)
            DO UPDATE SET updated_at = NOW()
            RETURNING id
            """,
            (canonical,),
        )
        act_id = cur.fetchone()[0]

        merged_aliases = set(aliases)
        merged_aliases.add(canonical)
        for alias in merged_aliases:
            cur.execute(
                """
                INSERT INTO legal_act_aliases (act_id, alias)
                VALUES (%s, %s)
                ON CONFLICT (act_id, alias) DO NOTHING
                """,
                (act_id, alias),
            )

    conn.commit()
    conn.close()
    print("Acts catalog seeded successfully.")


if __name__ == "__main__":
    main()
