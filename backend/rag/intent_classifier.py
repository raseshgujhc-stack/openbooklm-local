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

    return json.loads(response["choices"][0]["text"])

