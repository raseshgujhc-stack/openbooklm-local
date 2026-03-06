#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


@dataclass
class ActRow:
    act_id: int
    canonical_name: str


def _strip_the(name: str) -> str:
    return re.sub(r"^\s*the\s+", "", name, flags=re.IGNORECASE).strip()


def _strip_year(name: str) -> str:
    return re.sub(r",?\s*\d{4}\s*$", "", name).strip()


def _title_aliases(name: str) -> set[str]:
    # Keep only safe human-readable alias (yearless title).
    base = _strip_year(_strip_the(name))
    return {base} if len(base) >= 2 else set()


def _find_best_act_id(
    rows: list[ActRow],
    pattern: str,
    prefer: list[str] | None = None,
) -> int | None:
    rx = re.compile(pattern, re.IGNORECASE)
    candidates = [r for r in rows if rx.search(r.canonical_name)]
    if not candidates:
        return None

    def score(name: str) -> int:
        s = 0
        ln = name.lower()
        if "amendment" in ln:
            s -= 20
        if "rules" in ln or "regulations" in ln or "order" in ln:
            s -= 12
        if prefer:
            for p in prefer:
                if re.search(p, name, re.IGNORECASE):
                    s += 15
        if re.search(r",\s*\d{4}$", name):
            s += 2
        return s

    candidates.sort(key=lambda r: score(r.canonical_name), reverse=True)
    return candidates[0].act_id


def main() -> None:
    load_dotenv(ENV_PATH)
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN not set in backend/.env")

    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    cur.execute("SELECT id, canonical_name FROM legal_acts WHERE is_active = TRUE")
    rows = [ActRow(act_id=r[0], canonical_name=r[1]) for r in cur.fetchall()]
    if not rows:
        raise RuntimeError("No active acts found in legal_acts")

    inserted_aliases = 0

    # 1) Safe generic aliases (yearless canonical title only).
    for act in rows:
        for alias in _title_aliases(act.canonical_name):
            cur.execute(
                """
                INSERT INTO legal_act_aliases (act_id, alias)
                VALUES (%s, %s)
                ON CONFLICT (act_id, alias) DO NOTHING
                """,
                (act.act_id, alias),
            )
            inserted_aliases += cur.rowcount

    # 2) Curated judiciary aliases with preferred principal statutes.
    curated_patterns = [
        (r"indian\s+penal\s+code", [r"1860"], ["IPC", "I.P.C.", "Penal Code", "Indian Penal Code"]),
        (r"bharatiya\s+nyaya\s+sanhita", [r"2023"], ["BNS", "B.N.S."]),
        (
            r"code\s+of\s+criminal\s+procedure",
            [r"1973"],
            ["CrPC", "Cr.P.C.", "Code of Criminal Procedure"],
        ),
        (r"bharatiya\s+nagarik\s+suraksha\s+sanhita", [r"2023"], ["BNSS", "B.N.S.S."]),
        (r"indian\s+evidence\s+act", [r"1872"], ["Evidence Act", "Indian Evidence Act"]),
        (r"bharatiya\s+sakshya\s+adhiniyam", [r"2023"], ["BSA", "B.S.A."]),
        (r"information\s+technology\s+act", [r"2000"], ["IT Act", "I.T. Act"]),
        (r"constitution\s+of\s+india", [r"1950"], ["Constitution", "COI"]),
    ]

    # Remove curated aliases globally first to avoid wrong pre-existing mappings.
    curated_aliases = sorted({a for _, _, aliases in curated_patterns for a in aliases})
    cur.execute("DELETE FROM legal_act_aliases WHERE alias = ANY(%s)", (curated_aliases,))

    for pattern, prefer, aliases in curated_patterns:
        act_id = _find_best_act_id(rows, pattern, prefer)
        if not act_id:
            continue
        for alias in aliases:
            cur.execute(
                """
                INSERT INTO legal_act_aliases (act_id, alias)
                VALUES (%s, %s)
                ON CONFLICT (act_id, alias) DO NOTHING
                """,
                (act_id, alias),
            )
            inserted_aliases += cur.rowcount

    # 3) Act successions (legacy -> new criminal law stack).
    cur.execute("DELETE FROM legal_act_successions")
    successions = [
        (
            r"indian\s+penal\s+code",
            [r"1860"],
            r"bharatiya\s+nyaya\s+sanhita",
            [r"2023"],
            "2024-07-01",
            "Legacy IPC replaced by BNS.",
        ),
        (
            r"code\s+of\s+criminal\s+procedure",
            [r"1973"],
            r"bharatiya\s+nagarik\s+suraksha\s+sanhita",
            [r"2023"],
            "2024-07-01",
            "Legacy CrPC replaced by BNSS.",
        ),
        (
            r"indian\s+evidence\s+act",
            [r"1872"],
            r"bharatiya\s+sakshya\s+adhiniyam",
            [r"2023"],
            "2024-07-01",
            "Legacy Indian Evidence Act replaced by BSA.",
        ),
    ]

    inserted_successions = 0
    for from_pat, from_pref, to_pat, to_pref, effective_on, notes in successions:
        from_id = _find_best_act_id(rows, from_pat, from_pref)
        to_id = _find_best_act_id(rows, to_pat, to_pref)
        if not from_id or not to_id:
            continue
        cur.execute(
            """
            INSERT INTO legal_act_successions (from_act_id, to_act_id, effective_on, notes)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (from_act_id, to_act_id) DO NOTHING
            """,
            (from_id, to_id, effective_on, notes),
        )
        inserted_successions += cur.rowcount

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM legal_act_aliases")
    total_aliases = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM legal_act_successions")
    total_successions = cur.fetchone()[0]
    conn.close()

    print("Alias / succession population completed.")
    print(f"Inserted aliases this run: {inserted_aliases}")
    print(f"Inserted successions this run: {inserted_successions}")
    print(f"Total aliases in DB: {total_aliases}")
    print(f"Total successions in DB: {total_successions}")


if __name__ == "__main__":
    main()
