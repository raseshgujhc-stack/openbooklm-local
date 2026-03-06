# File Flow Map (Core Runtime)

## Backend API Layer
- `backend/rag/app.py`: HTTP API surface and startup workers
  - async chat submit/status
  - notebook/collection ingestion-readiness checks
  - answer cache + runtime metrics in payload
  - validates and persists `global_sub_collection_ids` + `specialization` in chat jobs/cache keys
- `backend/rag/ingest_worker.py`: queued ingestion execution
- `backend/rag/metadata_retry_worker.py`: retries metadata extraction failures

## Ingestion + Metadata
- `backend/rag/pdf_reader.py`: PDF text extraction
- `backend/rag/ingest.py`: chunk/embed/metadata persistence pipeline
- `backend/rag/metadata_engine.py`: metadata-only answer logic
- `backend/rag/act_catalog.py`: legal acts/sections normalization and lookup

## Retrieval + Generation
- `backend/rag/vector_store.py`: FAISS save/load/index utilities
- `backend/rag/rag_pipeline.py`: retrieval strategy and final answer synthesis
  - global chunk filtering by selected sub-collections
  - specialization-aware LLM question steering
- `backend/rag/model_router.py`: model selection logic

## STT
- `stt_service/main.py`: websocket + health + app startup
- `stt_service/core_models/transcription.py`: audio decode + whisper inference
- `stt_service/core_models/legal_formatter.py`: realtime/final legal text formatting
- `stt_service/api/routes.py`: file transcription REST endpoints

## TTS
- `tts_service/app.py`: podcast text-to-speech HTTP API using XTTS
- `tts_service/create_speakers.py`: speaker folder/bootstrap and smoke test helper
- `tts_service/Dockerfile`: reproducible TTS runtime image build

## UI
- `openbooklm-ui/src/app/page.tsx`: main shell and navigation
- `openbooklm-ui/src/components/PdfChatBox.tsx`: document chat
  - notebook status polling and chat lock while ingesting
  - global sub-collection filters + specialization selector
- `openbooklm-ui/src/components/CollectionChat.tsx`: collection chat
  - collection readiness banner and partial-ingestion notice
  - global sub-collection filters + specialization selector
- `openbooklm-ui/src/components/CollectionManager.tsx`: collection CRUD + modals
- `openbooklm-ui/src/components/LiveTranscribe.tsx`: microphone transcription UI
- `openbooklm-ui/src/components/AdminPanel.tsx`: admin operations
- `openbooklm-ui/src/components/ServiceStatus.tsx`: service health widget
- `openbooklm-ui/src/components/NotebookList.tsx`: notebook status badges
- `openbooklm-ui/src/lib/api.ts`: frontend API contracts/helpers

## Infra
- `infra/local-https/nginx.conf.template`: LAN HTTPS reverse proxy config
- `scripts/start_lan_https.sh`: cert generation + gateway startup
