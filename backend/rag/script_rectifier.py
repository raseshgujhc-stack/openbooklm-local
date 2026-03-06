import re
from typing import Dict, List, Tuple


SPEAKER_NAMES = ["Rahul", "Priya", "Vikas", "Anita"]


def apply_spoken_formatting_commands(text: str) -> str:
    """
    Convert spoken editing commands to symbols/structure.
    Supports punctuation, brackets, paragraph breaks, and table directives.
    """
    out = text or ""
    if not out.strip():
        return ""

    command_map = {
        r"\bfull\s+stop\b": ".",
        r"\bperiod\b": ".",
        r"\bcomma\b": ",",
        r"\bquestion\s+mark\b": "?",
        r"\bexclamation\s+mark\b": "!",
        r"\bcolon\b": ":",
        r"\bsemi\s*colon\b": ";",
        r"\bopen\s+bracket\b": "(",
        r"\bin\s+bracket\b": "(",
        r"\bclose\s+bracket\b": ")",
        r"\bnew\s+paragraph\b": "\n\n",
        r"\bnext\s+paragraph\b": "\n\n",
        r"\bnext\s+para\b": "\n\n",
        r"\bnew\s+line\b": "\n",
        r"\bnext\s+line\b": "\n",
        r"\bstart\s+table\b": " [[TABLE_START]] ",
        r"\bnext\s+column\b": " [[TABLE_COL]] ",
        r"\bnext\s+row\b": " [[TABLE_ROW]] ",
        r"\bend\s+table\b": " [[TABLE_END]] ",
    }
    for pattern, replacement in command_map.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    out = _render_spoken_tables(out)

    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([(\[])\s+", r"\1", out)
    out = re.sub(r"\s+([)\]])", r"\1", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _render_spoken_tables(text: str) -> str:
    """
    Convert [[TABLE_*]] tokens to a markdown table.
    """
    token_re = re.compile(r"\[\[TABLE_START\]\](.*?)\[\[TABLE_END\]\]", flags=re.DOTALL)

    def _table_block_to_md(block: str) -> str:
        rows = []
        for row_blob in block.split("[[TABLE_ROW]]"):
            cells = [c.strip(" .,\n\t") for c in row_blob.split("[[TABLE_COL]]")]
            cells = [c for c in cells if c]
            if cells:
                rows.append(cells)
        if not rows:
            return ""

        width = max(len(r) for r in rows)
        norm_rows = [r + [""] * (width - len(r)) for r in rows]
        header = norm_rows[0]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        for row in norm_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n" + "\n".join(lines) + "\n"

    return token_re.sub(lambda m: _table_block_to_md(m.group(1)), text)


def _split_into_sentences(text: str) -> List[str]:
    raw = re.sub(r"\s+", " ", (text or "")).strip()
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
    if not parts:
        return [raw]
    return parts


def _extract_turns(normalized: str, allowed_list: List[str]) -> List[Tuple[str, str]]:
    turns: List[Tuple[str, str]] = []
    allowed_set = set(allowed_list)
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("|") or ":" not in line:
            continue
        speaker, content = line.split(":", 1)
        speaker = speaker.strip()
        content = content.strip()
        if speaker in allowed_set and content:
            turns.append((speaker, content))
    return turns


def _rebalance_single_speaker_script(turns: List[Tuple[str, str]], allowed_list: List[str]) -> str:
    """
    If model emits only one speaker while multi-speaker mode is requested,
    distribute content across selected speakers in alternating turns.
    """
    # Prefer preserving original turn boundaries if model already returned many turns
    # but with the same speaker label.
    original_units = [text.strip() for _, text in turns if text.strip()]
    if len(original_units) >= 2:
        lines: List[str] = []
        for idx, unit in enumerate(original_units):
            speaker = allowed_list[idx % len(allowed_list)]
            lines.append(f"{speaker}: {unit}")
        return "\n".join(lines).strip()

    merged_text = " ".join(original_units).strip()
    if not merged_text:
        return ""

    sentences = _split_into_sentences(merged_text)
    # If model output is one very long sentence, split it once near midpoint.
    if len(sentences) == 1 and len(allowed_list) > 1:
        s = sentences[0]
        mid = len(s) // 2
        split_at = s.rfind(", ", 0, mid)
        if split_at < 0:
            split_at = s.rfind(" ", 0, mid)
        if split_at > 20:
            sentences = [s[:split_at].strip(), s[split_at + 1 :].strip()]

    lines: List[str] = []
    for idx, sentence in enumerate(sentences):
        if not sentence:
            continue
        speaker = allowed_list[idx % len(allowed_list)]
        lines.append(f"{speaker}: {sentence}")
    return "\n".join(lines).strip()


def normalize_and_validate_podcast_script(
    script: str,
    *,
    speakers: int,
) -> Tuple[str, Dict]:
    """
    Normalize dictation commands and validate that every spoken turn uses
    allowed speaker names.
    """
    allowed_list = SPEAKER_NAMES[: max(1, min(int(speakers or 2), 4))]
    allowed = set(allowed_list)
    normalized = apply_spoken_formatting_commands(script or "")
    if not normalized.strip():
        raise ValueError("Script is empty after normalization.")

    bad_lines: List[int] = []
    has_turn = False
    for idx, raw_line in enumerate(normalized.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("|"):
            continue
        if ":" not in line:
            bad_lines.append(idx)
            continue
        speaker, _ = line.split(":", 1)
        if speaker.strip() not in allowed:
            bad_lines.append(idx)
        else:
            has_turn = True

    if bad_lines:
        raise ValueError(
            "Invalid script format. Every non-table line must be "
            f"`SpeakerName: text` using allowed speakers. Bad lines: {bad_lines[:8]}"
        )
    if not has_turn:
        raise ValueError("Script must include at least one valid speaker line.")

    turns = _extract_turns(normalized, allowed_list)
    used_speakers = {speaker for speaker, _ in turns}
    rebalanced = False
    if len(allowed_list) > 1 and len(used_speakers) == 1:
        repaired = _rebalance_single_speaker_script(turns, allowed_list)
        if repaired:
            normalized = repaired
            turns = _extract_turns(normalized, allowed_list)
            used_speakers = {speaker for speaker, _ in turns}
            rebalanced = True

    return normalized, {
        "allowed_speakers": sorted(list(allowed)),
        "used_speakers": sorted(list(used_speakers)),
        "rebalanced_for_missing_speakers": rebalanced,
    }
