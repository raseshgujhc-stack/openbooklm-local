# rag/podcast.py

from rag.llm import llm_generate_text

SPEAKER_NAMES = ["Rahul", "Priya", "Vikas", "Anita"]

def generate_podcast_script(
    context: str,
    speakers: int = 2,
) -> str:
    speakers = max(1, min(speakers, 4))
    active_speakers = SPEAKER_NAMES[:speakers]

    speaker_rules = "\n".join(
        [f"- {name}: speaks naturally" for name in active_speakers]
    )

    prompt = f"""
You are a professional podcast script writer.

STRICT RULES (VERY IMPORTANT):
- Output ONLY dialogue lines
- EACH line MUST start with one of these speaker names EXACTLY:
  {", ".join(active_speakers)}
- Format MUST be:
  SpeakerName: sentence
- DO NOT write "Speaker 1", "Speaker 2"
- DO NOT add headings, greetings, or narration
- DO NOT leave blank lines
- English only

Allowed speakers:
{speaker_rules}

Content:
{context}

Begin the podcast now.
"""

    return llm_generate_text(prompt)
