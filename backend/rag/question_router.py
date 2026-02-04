# rag/question_router.py

def classify_question(question: str, available_entities: dict = None) -> dict:
    """
    HARD router for obvious metadata questions.
    Runs BEFORE LLM intent classification.

    Args:
        question: User question string.
        available_entities: Dict showing what metadata exists, e.g.,
            {"document": True, "case": False, "order_date": True}

    Returns:
        {
            "route": "metadata" | "semantic",
            "operation": "list" | "count" | None,
            "entities": dict,
            "fallback_to_semantic": bool
        }
    """
    if available_entities is None:
        available_entities = {}

    q = question.lower().strip()

    result = {
        "route": "semantic",
        "operation": None,
        "entities": {},
        "fallback_to_semantic": False
    }

    # ----------------------------------
    # COUNT / QUANTITY QUESTIONS
    # ----------------------------------
    if any(x in q for x in ["how many", "count", "number of"]):
        result.update({
            "route": "metadata",
            "operation": "count",
            "entities": {
                "document": any(x in q for x in ["document", "documents", "pdf", "pdfs", "file", "files"]),
                "case": "case" in q,
            }
        })

    # ----------------------------------
    # LISTING QUESTIONS
    # ----------------------------------
    elif any(x in q for x in ["list", "show", "which"]):
        result.update({
            "route": "metadata",
            "operation": "list",
            "entities": {
                "case": "case" in q,
                "document": any(x in q for x in ["document", "documents", "pdf", "pdfs", "file", "files"]),
                "order_date": "order date" in q,
                "document_type": "type" in q,
            }
        })

    # ----------------------------------
    # FALLBACK CHECK
    # ----------------------------------
    if result["route"] == "metadata":
        entities_required = [k for k, v in result["entities"].items() if v]
        if not any(available_entities.get(e, False) for e in entities_required):
            # No data available, fallback to semantic
            result["route"] = "semantic"
            result["operation"] = None
            result["fallback_to_semantic"] = True

    return result

