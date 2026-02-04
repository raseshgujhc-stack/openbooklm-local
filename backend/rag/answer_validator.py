# file: answer_validator.py
def validate_answer(answer: str, context: str, question: str) -> bool:
    """
    Validate if answer is properly grounded in context
    """
    if not answer or answer.strip() == "":
        return False
    
    answer_lower = answer.lower()
    
    # Check for vague phrases
    vague_phrases = [
        "based on the information",
        "generally speaking",
        "typically",
        "usually",
        "in many cases",
        "it seems",
        "it appears",
        "might be",
        "could be",
        "possibly",
        "probably",
    ]
    
    for phrase in vague_phrases:
        if phrase in answer_lower:
            return False
    
    # Check if answer contains exact phrases from context
    context_words = set(context.lower().split())
    answer_words = set(answer_lower.split())
    overlap = len(context_words.intersection(answer_words))
    
    if overlap < 3:  # Too few overlapping words
        return False
    
    return True
