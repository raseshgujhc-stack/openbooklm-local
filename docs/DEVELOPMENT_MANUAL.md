# Development Manual

## 1. Repository Layout
- `backend/rag`: primary API, ingestion, retrieval, metadata, workers
- `backend/db`: Postgres repository wiring
- `stt_service`: judicial speech-to-text service
- `openbooklm-ui`: frontend app
- `infra/local-https`: HTTPS reverse proxy for LAN usage
- `scripts`: operational utility scripts

## 2. Runtime Components
### Backend API (`backend/rag/app.py`)
Responsibilities:
- Login/auth header enforcement
- Notebook/collection CRUD
- Upload and ingestion queueing
- Async chat jobs (submit/status)
- Ingestion-aware chat gating (notebook strict, collection partial)
- Repeated-question answer cache + runtime metrics
- Admin panel APIs (users/workers/audit)

### Ingestion (`backend/rag/ingest.py` + worker)
Pipeline:
1. Parse PDF text
2. Chunk content
3. Embed chunks
4. Save FAISS vectors
5. Extract metadata (deterministic + model-assisted)
6. Store metadata in Postgres
7. Mark notebook `ready` (or `failed`) for chat gating

### RAG (`backend/rag/rag_pipeline.py`)
Decision order:
1. Fast metadata route (count/pages/list)
2. Router classification
3. Retrieval scope (notebook / collection / global)
4. LLM synthesis
5. Return structured payload (`answer`, `citations`, `runtime`)

### Chat Submission and Gating (`POST /chat/submit`)
1. Validate scope ownership (notebook or collection)
2. Notebook mode:
   - block if notebook status is `queued`, `processing`, or `failed`
3. Collection mode:
   - block only when `ready_count == 0`
   - allow partial chat when some docs are ready and some are still ingesting
4. Enqueue async `chat_jobs` worker run
5. Optional `ingestion_notice` returned for partial collection readiness

### Answer Cache + Runtime Metrics
- Cache table: `chat_answer_cache`
- Key dimensions:
  - `scope_type` (`notebook` / `collection`)
  - `scope_id`
  - `user_scope` (null for shared global collection cache)
  - `include_global`
  - normalized question hash
- Recheck words (`recheck`, `verify again`, `regenerate`, etc.) bypass cache.
- Runtime is attached to chat result payload:
  - `cached`, `cache_lookup_ms`, `generation_ms`, `total_ms`

### STT (`stt_service/main.py`)
- WebSocket stream endpoint: `/ws/transcribe`
- Health endpoint: `/health`
- File API routes: `/api/v1/transcribe`, `/api/v1/formats`
- Uses `JudicialTranscriber` + `LegalFormatter`

### UI (`openbooklm-ui/src/app/page.tsx`)
View modes:
- Notebooks
- Collections
- Transcribe
- Admin

Includes session timeout tracking and role-aware navigation.

## 3. Data Stores
### Postgres
Core tables used by runtime:
- `users`
- `notebooks`
- `collections`
- `ingest_jobs`
- `chat_jobs`
- `chat_answer_cache`
- `chat_history`
- `collection_chat_history`
- `document_metadata`
- `metadata_retry_jobs`
- `podcast_jobs`
- admin runtime/audit tables

### FAISS Files
Stored under `data/faiss`:
- `<notebook_id>.index`
- `<notebook_id>.json`
- global index files (depending build path)

## 4. Local Development Workflow
1. Start Postgres and dependencies
2. Start backend (`uvicorn rag.app:app`)
3. Start UI (`npm run dev`)
4. Start STT (`docker compose up -d --build`)
5. Start TTS
6. Use HTTPS gateway for LAN mic scenarios

## 5. HTTPS Gateway Workflow
Run:
```bash
/home/ubuntu/openbooklm-local/scripts/start_lan_https.sh <LAN_IP>
```

Routes:
- `/backend/*` -> backend:8002
- `/stt/*` -> stt:8003
- `/tts/*` -> tts:9000
- `/` -> ui:3000

Install certificate on each client:
- `infra/local-https/certs/lan.crt`

## 6. Testing and Verification
### UI
```bash
npm --prefix openbooklm-ui run typecheck
```

### Backend health
```bash
curl http://127.0.0.1:8002/
```

### Notebook readiness
```bash
curl -H "X-User-Id: <user-id>" \
  http://127.0.0.1:8002/notebook/<notebook-id>/status
```

### Collection readiness
```bash
curl -H "X-User-Id: <user-id>" \
  http://127.0.0.1:8002/collection/<collection-id>/contents
```

### STT health + WS
```bash
curl http://127.0.0.1:8003/health
wscat -c ws://127.0.0.1:8003/ws/transcribe
```

### Login validation
```bash
curl -X POST http://127.0.0.1:8002/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"..."}'
```

## 7. Troubleshooting Runbook
### Invalid credentials
- Confirm user exists in `users`
- Reset password hash via bcrypt if needed
- Ensure UI points to correct backend base URL

### STT shown offline
- Check `/health` and WS handshake separately
- Verify gateway `/stt/ws/transcribe` upgrade path
- Inspect container logs: `docker logs stt_service-stt-service-1`

### Microphone blocked
- Must use HTTPS origin on LAN
- Certificate must be trusted on client machine

### Chat timeout/failed response
- Verify chat job table updates (`chat_jobs` status)
- Check backend worker logs for LLM/retrieval errors
- If notebook chat is blocked, confirm notebook status endpoint shows `ready`
- If collection has mixed statuses, verify `ready_count` and `ingesting_count`

## 8. Coding Guidelines
- Keep comments concise and explanatory for non-obvious logic.
- Favor submit+poll APIs for heavy tasks.
- Avoid hardcoded host URLs in UI; prefer env or same-origin proxy routes.
- Add schema guards for new runtime tables/indexes.

## 9. Suggested Next Enhancements
- Add migration framework for DB schema versioning.
- Add integration tests for upload -> ingest -> ask flow.
- Add structured observability (request id, job id tracing).
