# OpenBookLM Local (Judiciary RAG + STT + TTS)

This repository runs a local-first legal intelligence stack:
- `backend/rag` for ingestion, metadata, RAG, chat, collection management, and admin APIs.
- `openbooklm-ui` for user/admin UI.
- `stt_service` for microphone/file transcription with legal formatting.
- `tts_service` for podcast/audio generation.

## Architecture
- UI (Next.js): user workflows (upload, chat, collections, transcribe, admin)
- Backend API (FastAPI, `:8002`): core logic + Postgres-backed persistence
- STT Service (FastAPI WS/HTTP, `:8003`): real-time speech-to-text
- TTS Service (`:9000`): podcast generation/audio serving
- Postgres (`notebooklm`): users, notebooks, collections, chat jobs, metadata
- FAISS artifacts (`data/faiss`): per-notebook and global retrieval indexes

## End-to-End Flow
1. User uploads PDF
2. Backend queues ingestion job
3. Worker extracts text + metadata, chunks, embeds, stores FAISS + metadata
4. Notebook status moves `queued -> processing -> ready|failed`
5. User asks question (notebook/collection/global) via async `submit -> status` polling
6. Backend applies ingestion-aware gating:
   - Single notebook chat is blocked until notebook is `ready`
   - Collection chat is allowed if at least one document is `ready`
7. Backend routes metadata intent or semantic RAG retrieval
8. Answer cache is checked for repeated questions (recheck-style prompts bypass cache)
9. LLM generates answer (or cached answer is returned)
10. UI renders response with citations and runtime (`seconds`, `cached` badge)

## Latest Chat/Ingress Rules
- Single document:
  - `queued/processing`: chat disabled with “ingestion in progress”
  - `failed`: chat disabled with retry/re-upload message
  - `ready`: chat enabled
- Collection:
  - if `ready_count == 0` and ingesting docs exist: backend returns wait message
  - if `ready_count > 0` and some docs still ingesting: chat continues on ready docs and shows notice

## New/Updated Runtime APIs
- `GET /notebook/{notebook_id}/status`: notebook readiness for UI gating
- `GET /collection/{collection_id}/contents`: now includes per-notebook `status` and collection counts:
  - `ready_count`, `ingesting_count`, `failed_count`
- `POST /chat/submit`: may return `ingestion_notice` for partially-ready collections
- `GET /chat/status/{job_id}`: includes `result_payload.runtime`:
  - `cached`, `cache_lookup_ms`, `generation_ms`, `total_ms`
- Chat specialization/filter payload (submit + sync chat):
  - `global_sub_collection_ids: string[]` to restrict global retrieval to selected global collections
  - `specialization: string` to steer answer style (e.g., `criminal`, `civil`, `constitutional`, `evidence`, `procedural`, `tax`)
- STT rectification:
  - `POST /stt/api/v1/rectify`: submit edited transcript + remarks after verification
  - `GET /stt/api/v1/rectify/guide`: ready-to-use dictation/editing guide (for one-click UI help)
  - `POST /stt/api/v1/export/docx`: export transcript as Word `.docx` with proper table structures

## STT Review Flow
1. Record and transcribe audio from `Live Transcribe`.
2. Review/edit transcript in **Review And Rectify Transcript** panel.
3. Use spoken commands in edits, such as:
   - `full stop -> .`, `comma -> ,`
   - `open bracket -> (`, `close bracket -> )`
   - `next para -> paragraph break`
   - `start table`, `next column`, `next row`, `end table`
4. Save rectification with optional remarks.
5. Corrections are stored and reused as dynamic phrase fixes for future transcriptions.

## Global Collection + Specialization Flow
1. In notebook chat or collection chat, enable **Include Global Knowledge**.
2. Select one or more global sub-collections (example: `NI_ACT`, `CRPC`, `EVIDENCE_ACT`, `IPC`, `CPC`).
3. Select AI specialization for answer framing.
4. Submit question.
5. Backend filters global chunks to the selected sub-collections only, then synthesizes the answer with specialization guidance.
6. Citations and answer cache remain scope-aware and include the same filter/specialization context.

## Podcast (TTS) Review Flow
1. Generate podcast draft script (`/podcast/generate` with `auto_generate_audio=false`).
2. Review and edit script in UI.
3. Use spoken commands in script edits:
   - `full stop`, `comma`, `open bracket`, `close bracket`
   - `next para`, `next line`
   - `start table`, `next column`, `next row`, `end table`
4. Save edited script with optional remarks (`/podcast/commit/{job_id}`).
5. Generate final audio from approved script (`regenerate_audio=true`).

Podcast helper APIs:
- `GET /podcast/guide`: ready-to-use editing guide for UI
- `POST /podcast/commit/{job_id}`: save rectified script and optionally regenerate audio

## Services and Ports
- UI dev: `3000`
- Backend RAG: `8002`
- STT: `8003`
- TTS: `9000`
- HTTPS LAN gateway: `443`

## Quick Start (Development)
### 1) Backend
```bash
cd backend
source venv/bin/activate
uvicorn rag.app:app --host 0.0.0.0 --port 8002 --reload
```

### 2) UI
```bash
cd openbooklm-ui
npm install
npm run dev
```

### 3) STT
```bash
cd stt_service
docker compose up -d --build stt-service
```

### 4) TTS
Ensure your TTS service is running on `:9000`.

## HTTPS / LAN Mode (for microphone access)
Browsers block microphone on insecure LAN origins (`http://<ip>`). Use HTTPS:
```bash
/home/ubuntu/openbooklm-local/scripts/start_lan_https.sh 10.225.18.120
```
Open:
- `https://10.225.18.120`

Trust certificate on clients:
- `infra/local-https/certs/lan.crt`

## Authentication
Login API is in `backend/rag/app.py` (`POST /login`).
Default admin credential may be reset during ops; verify with DB/admin if invalid.

## Key Code Areas
- Backend API: `backend/rag/app.py`
- Ingestion: `backend/rag/ingest.py`
- Metadata QA: `backend/rag/metadata_engine.py`
- RAG orchestration: `backend/rag/rag_pipeline.py`
- Vector store: `backend/rag/vector_store.py`
- STT entrypoint: `stt_service/main.py`
- Legal formatting: `stt_service/core_models/legal_formatter.py`
- UI shell: `openbooklm-ui/src/app/page.tsx`

## Operations
### Health checks
- Backend: `GET /health` (or root depending deployment)
- STT: `GET http://<host>:8003/health`
- TTS: `GET http://<host>:9000/health`

### Common issues
- Mic blocked: run via HTTPS and trust cert.
- STT online but WS fails: check `stt_service` logs and `/ws/transcribe` path.
- Invalid credentials: verify `users` table and password hash reset path.
- Global/collection counts mismatch: verify `document_metadata` and collection scope query.

## Developer Notes
- Keep API base as same-origin proxy in UI for LAN HTTPS compatibility.
- Prefer adding migrations/schema guards in startup for safety.
- For long-running user tasks, use submit+poll pattern (chat/podcast) instead of blocking requests.

For detailed runbook, see `docs/DEVELOPMENT_MANUAL.md`.
