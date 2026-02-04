from rag.ask_pdf import load_pdf
from rag.rag_pipeline import answer_question_from_pdf

# Load PDF once
load_pdf("sample.pdf")

# Ask question
answer = answer_question_from_pdf(
    "What is this document about?"
)

print("\nANSWER:\n", answer)
