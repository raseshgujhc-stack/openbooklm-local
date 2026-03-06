#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <LAN_HOST_OR_IP>"
  echo "Example: $0 10.225.18.120"
  exit 1
fi

LAN_HOST="$1"
ROOT_DIR="/home/ubuntu/openbooklm-local"
HTTPS_DIR="$ROOT_DIR/infra/local-https"
COMPOSE_FILE="$HTTPS_DIR/docker-compose.yml"
CERT_DIR="$HTTPS_DIR/certs"

mkdir -p "$CERT_DIR"

cat > "$HTTPS_DIR/openssl.cnf" <<CNF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_req

[dn]
CN = $LAN_HOST
O = OpenBookLM Local

[v3_req]
subjectAltName = @alt_names
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth

[alt_names]
DNS.1 = $LAN_HOST
IP.1 = $LAN_HOST
CNF

openssl req -x509 -nodes -newkey rsa:2048 \
  -days 825 \
  -keyout "$CERT_DIR/lan.key" \
  -out "$CERT_DIR/lan.crt" \
  -config "$HTTPS_DIR/openssl.cnf" >/dev/null 2>&1

sed "s/__LAN_HOST__/$LAN_HOST/g" "$HTTPS_DIR/nginx.conf.template" > "$HTTPS_DIR/nginx.conf"

# Stop any previous gateway instance
(docker rm -f openbooklm-lan-gateway >/dev/null 2>&1 || true)

docker compose -f "$COMPOSE_FILE" up -d

echo
cat <<MSG
HTTPS gateway started for: https://$LAN_HOST

Routes:
- /backend/* -> :8002
- /stt/*     -> :8003 (WebSocket supported)
- /tts/*     -> :9000
- /          -> :3000

For browser microphone access, trust this certificate on each client machine:
$CERT_DIR/lan.crt
MSG
