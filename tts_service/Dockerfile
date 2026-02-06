FROM python:3.11-slim

WORKDIR /app

# ✅ Required to avoid Coqui interactive license prompt
ENV COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# ✅ Keep exact versions (DO NOT change)
RUN pip install \
    torch==2.1.2 \
    TTS==0.22.0 \
    transformers==4.37.2 \
    fastapi \
    uvicorn

# ✅ Existing app
COPY app.py .

# ✅ ADD: speaker creation script (no removal)
COPY create_speakers.py .

# ✅ ADD: persistent locations
# - models: downloaded XTTS model
# - speakers: speaker embeddings / wavs
VOLUME ["/data/models", "/data/speakers"]

EXPOSE 9000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
