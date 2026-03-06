from __future__ import annotations

import json
import re
import difflib
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple


CATALOG_JSON = Path(__file__).resolve().parent / "data" / "acts_seed.json"


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _norm_alias(s: str) -> str:
    s = s.lower()
    s = s.replace(".", " ")
    s = s.replace(",", " ")
    s = re.sub(r"[()]", " ", s)
    s = _norm_spaces(s)
    return s


def _strip_year(s: str) -> str:
    return re.sub(r",?\s*\d{4}$", "", s).strip()


def _canonicalize_section(sec: str) -> str:
    sec = _norm_spaces(sec)
    sec = sec.replace(" ", "")
    return f"Section {sec}"


@lru_cache(maxsize=1)
def _load_alias_map() -> Tuple[Dict[str, str], str]:
    alias_to_canonical: Dict[str, str] = {}
    source = "json_seed"

    # DB-first path (if tables exist and DB is reachable)
    try:
        from db import get_repo

        repo = get_repo()
        cur = repo.conn.cursor()
        cur.execute(
            """
            SELECT a.canonical_name, aa.alias
            FROM legal_acts a
            JOIN legal_act_aliases aa ON aa.act_id = a.id
            WHERE a.is_active = TRUE
            """
        )
        rows = cur.fetchall()
        if rows:
            for canonical, alias in rows:
                if not canonical or not alias:
                    continue
                alias_to_canonical[_norm_alias(alias)] = canonical
                alias_to_canonical[_norm_alias(_strip_year(alias))] = canonical
                alias_to_canonical[_norm_alias(canonical)] = canonical
                alias_to_canonical[_norm_alias(_strip_year(canonical))] = canonical
            source = "database"
            return alias_to_canonical, source
    except Exception:
        pass

    # JSON fallback
    if CATALOG_JSON.exists():
        data = json.loads(CATALOG_JSON.read_text(encoding="utf-8"))
        for row in data:
            canonical = row.get("canonical_name")
            aliases = row.get("aliases") or []
            if not canonical:
                continue
            alias_to_canonical[_norm_alias(canonical)] = canonical
            alias_to_canonical[_norm_alias(_strip_year(canonical))] = canonical
            for a in aliases:
                alias_to_canonical[_norm_alias(a)] = canonical
                alias_to_canonical[_norm_alias(_strip_year(a))] = canonical

    return alias_to_canonical, source


def get_catalog_source() -> str:
    return _load_alias_map()[1]


def get_alias_map() -> Dict[str, str]:
    """
    Public alias map accessor used by retrieval modules.
    """
    return _load_alias_map()[0]


@lru_cache(maxsize=1)
def _load_acronym_map() -> Dict[str, str]:
    """
    Build acronym -> canonical map from loaded aliases/canonical names.
    Example:
    - "negotiable instruments act" -> "nia", "ni"
    - "bharatiya nagarik suraksha sanhita" -> "bnss"
    """
    alias_map, _ = _load_alias_map()
    out: Dict[str, str] = {}
    stop = {"the", "of", "and"}
    tail_words = {"act", "code", "sanhita", "adhiniyam", "rules", "regulation", "regulations"}

    for alias, canonical in alias_map.items():
        tokens = [t for t in re.findall(r"[a-z0-9]+", alias.lower()) if t]
        if len(tokens) < 2:
            continue
        core = [t for t in tokens if t not in stop]
        if len(core) < 2:
            continue

        full_acr = "".join(t[0] for t in core if t and t[0].isalpha())
        if len(full_acr) >= 2 and full_acr not in out:
            out[full_acr] = canonical

        if core[-1] in tail_words and len(core) >= 3:
            stem = core[:-1]
            stem_acr = "".join(t[0] for t in stem if t and t[0].isalpha())
            if len(stem_acr) >= 2 and stem_acr not in out:
                out[stem_acr] = canonical

    return out


def normalize_act_name(name: str | None) -> str | None:
    if not name:
        return None
    alias_map, _ = _load_alias_map()
    key = _norm_alias(name)
    if key in alias_map:
        return alias_map[key]
    key2 = _norm_alias(_strip_year(name))
    if key2 in alias_map:
        return alias_map[key2]

    # Acronym fallback for shorthand mentions like "NI Act", "MV Act", "BNSS".
    acronym_map = _load_acronym_map()
    words = [w for w in re.findall(r"[a-z0-9]+", key2) if w]
    tail_words = {"act", "code", "sanhita", "adhiniyam", "rules", "regulation", "regulations"}
    if words:
        if len(words) == 1 and words[0] in acronym_map:
            return acronym_map[words[0]]
        stem = [w for w in words if w not in tail_words]
        if len(stem) == 1 and stem[0] in acronym_map:
            return acronym_map[stem[0]]

    # Fuzzy fallback for small ASR/text variations:
    # e.g., "negotiable instrument act" vs "negotiable instruments act"
    candidates = difflib.get_close_matches(key2, alias_map.keys(), n=1, cutoff=0.92)
    if candidates:
        return alias_map.get(candidates[0])
    return None


def _find_act_mentions(text: str) -> List[Tuple[int, int, str]]:
    alias_map, _ = _load_alias_map()
    lowered = text.lower()
    mentions: List[Tuple[int, int, str]] = []

    # Longest aliases first to reduce overlap conflicts.
    aliases = sorted(alias_map.keys(), key=len, reverse=True)
    for alias in aliases:
        if len(alias) < 4:
            continue
        parts = [p for p in alias.split(" ") if p]
        if not parts:
            continue
        # Allow punctuation/spacing variants, e.g. "crpc", "cr.p.c.", "cr p c".
        separator = r"[\s\.\-()/,]*"
        pattern = rf"(?<!\w){separator.join(re.escape(p) for p in parts)}(?!\w)"
        for m in re.finditer(pattern, lowered):
            mentions.append((m.start(), m.end(), alias_map[alias]))

    # De-overlap spans: keep first (longest alias due to sort)
    mentions.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept: List[Tuple[int, int, str]] = []
    for st, en, canon in mentions:
        if kept and st < kept[-1][1]:
            continue
        kept.append((st, en, canon))
    return kept


def extract_acts_with_sections(text: str) -> List[dict]:
    text = text or ""
    acts_sections: Dict[str, List[str]] = {}
    act_order: List[str] = []

    def ensure_act(act: str):
        if act not in acts_sections:
            acts_sections[act] = []
            act_order.append(act)

    # 1) Strong pattern: "Section X of the <Act Name>"
    explicit = re.finditer(
        r"\b(?:Section|Sec\.?|S\.)\s*(\d+[A-Za-z\-]*)\s+of\s+the\s+([A-Za-z][A-Za-z\s,().&-]{3,120})",
        text,
        flags=re.IGNORECASE,
    )
    for m in explicit:
        sec = _canonicalize_section(m.group(1))
        canon = normalize_act_name(m.group(2))
        if canon:
            ensure_act(canon)
            if sec not in acts_sections[canon]:
                acts_sections[canon].append(sec)

    # 2) Detect all act mentions
    mentions = _find_act_mentions(text)
    for _, _, canon in mentions:
        ensure_act(canon)

    # 3) Generic section references, mapped to nearest act mention.
    section_hits = list(
        re.finditer(r"\b(?:Section|Sec\.?|S\.)\s*(\d+[A-Za-z\-]*)\b", text, flags=re.IGNORECASE)
    )
    if section_hits and mentions:
        for sm in section_hits:
            sec = _canonicalize_section(sm.group(1))
            spos = sm.start()

            nearest = None
            nearest_dist = 10**9
            for st, en, canon in mentions:
                # Prefer nearby act mention in same local window.
                dist = min(abs(spos - st), abs(spos - en))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = canon
            if nearest and nearest_dist <= 240:
                if sec not in acts_sections[nearest]:
                    acts_sections[nearest].append(sec)

    # 4) If only one act is present, attach unmatched sections to it.
    if len(act_order) == 1:
        only_act = act_order[0]
        for sm in section_hits:
            sec = _canonicalize_section(sm.group(1))
            if sec not in acts_sections[only_act]:
                acts_sections[only_act].append(sec)

    # 5) Domain fallback for MACT/Motor Accident matters where acts are often implicit.
    # This avoids missing act metadata in motor accident judgments with sparse formatting.
    if not act_order:
        lower = text.lower()
        mact_patterns = [
            r"motor\s+accident\s+claims?\s+tribunal",
            r"motor\s+accident\s+claim\s+petition",
            r"\bmacp\b",
            r"\bm\.?\s*a\.?\s*c\.?\s*p\.?\b",
        ]
        if any(re.search(p, lower) for p in mact_patterns):
            canon = normalize_act_name("Motor Vehicles Act")
            if canon:
                ensure_act(canon)

    return [{"act": act, "sections": acts_sections[act]} for act in act_order]
