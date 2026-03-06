# Local Network HTTPS Gateway (Mic-safe)

This gateway exposes OpenBookLM on HTTPS so browser microphone APIs are allowed.

## 1) Start core services on host
- UI: `http://127.0.0.1:3000`
- Backend RAG: `http://127.0.0.1:8002`
- STT: `http://127.0.0.1:8003`
- TTS: `http://127.0.0.1:9000`

## 2) Start HTTPS gateway

```bash
/home/ubuntu/openbooklm-local/scripts/start_lan_https.sh 10.225.18.120
```

Open:
- `https://10.225.18.120`

Proxy routes:
- `/backend/*` -> `:8002`
- `/stt/*` -> `:8003`
- `/tts/*` -> `:9000`
- all other paths -> `:3000`

UI features that rely on this gateway:
- Notebook readiness polling: `/backend/notebook/{id}/status`
- Collection readiness polling: `/backend/collection/{id}/contents`
- Async chat polling: `/backend/chat/submit` + `/backend/chat/status/{job_id}`
- STT browser mic: secure origin + trusted cert required

## 3) Trust certificate on each client

Install this cert in OS/browser trusted roots:
- `/home/ubuntu/openbooklm-local/infra/local-https/certs/lan.crt`

Without trust, browser may still block microphone on some setups.

## 4) Stop gateway

```bash
docker compose -f /home/ubuntu/openbooklm-local/infra/local-https/docker-compose.yml down
```
