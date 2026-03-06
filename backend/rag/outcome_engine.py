import json
from pathlib import Path
from rag.vector_store import BASE_DIR


OUTCOME_KEYWORDS = [
    "appeal is allowed",
    "appeal is dismissed",
    "petition is allowed",
    "petition is dismissed",
    "conviction is set aside",
    "conviction is confirmed",
    "compensation",
    "enhanced",
    "reduced",
    "modified",
    "award to be drawn",
    "ordered accordingly",
    "in view of the above",
    "the respondent shall",
    "is directed to",
]


def extract_outcome_section(notebook_id: str):
    """
    Extract likely operative portion from final chunks.
    """

    meta_path = BASE_DIR / f"{notebook_id}.json"

    if not meta_path.exists():
        return None

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    if not metadata:
        return None

    # Take last 30% of document
    cutoff = int(len(metadata) * 0.7)
    final_chunks = metadata[cutoff:]

    outcome_chunks = []

    for chunk in final_chunks:
        text_lower = chunk["text"].lower()

        if any(keyword in text_lower for keyword in OUTCOME_KEYWORDS):
            outcome_chunks.append(chunk["text"])

    if not outcome_chunks:
        return None

    return "\n\n".join(outcome_chunks)

