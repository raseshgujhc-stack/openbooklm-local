#rag/intent_classifier.py
from rag.llm import llm
import json

def classify_intent(question: str) -> dict:
    prompt = f"""
You are an intent classifier for a document system.

Your job:
- DO NOT answer the question
- DO NOT explain
- ONLY classify the intent

Return VALID JSON exactly in this schema:
{{
  "intent_type": "metadata | semantic | hybrid",
  "operation": "list | count | filter | summarize | compare | explain",
  "entities": {{
    "case": false,
    "order_date": false,
    "document_type": false,
    "act": false
  }},
  "filters": {{
    "document_type": null,
    "case_stage": null
  }}
}}

Question:
"{question}"
"""

    response = llm(
        prompt,
        temperature=0.0,
        max_tokens=200
    )

    import re
    raw_text = response["choices"][0]["text"].strip()

    # Extract JSON block if LLM added extra text
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)

    if not json_match:
        print("⚠ Intent classifier returned invalid JSON:")
        print(raw_text)
        return {
            "intent_type": "semantic",
            "operation": None,
            "entities": {},
            "filters": {}
        }

    try:
        return json.loads(json_match.group())
    except Exception as e:
        print("⚠ JSON parsing failed:", e)
        print("Raw response:", raw_text)
        return {
            "intent_type": "semantic",
            "operation": None,
            "entities": {},
            "filters": {}
        }

