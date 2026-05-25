#!/usr/bin/env bash
# Start MLflow + Langfuse alongside the existing OTel stack.
# The OTel stack (Jaeger, Prometheus, Grafana, OTel Collector) runs on otel_default network.
#
# Usage:
#   bash deploy/local/start-extras.sh          # start
#   bash deploy/local/start-extras.sh stop     # stop

set -euo pipefail

NETWORK=otel_default

stop() {
  echo "→ Removing MLflow + Langfuse..."
  podman rm -f obs-mlflow obs-langfuse obs-langfuse-db 2>/dev/null || true
  echo "✓ Done."
}

start() {
  # ── MLflow ────────────────────────────────────────────────────────────────
  echo "→ Creating MLflow volume..."
  podman volume create obs-mlflow-data 2>/dev/null || true

  echo "→ Starting MLflow..."
  podman rm -f obs-mlflow 2>/dev/null || true
  podman run -d \
    --name obs-mlflow \
    --network "$NETWORK" \
    -p 5001:5000 \
    -v obs-mlflow-data:/mlflow \
    ghcr.io/mlflow/mlflow:v3.12.0 \
    mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --backend-store-uri "sqlite:////mlflow/mlflow.db" \
      --artifacts-destination /mlflow/artifacts \
      --default-artifact-root mlflow-artifacts:/

  # ── Langfuse Postgres ─────────────────────────────────────────────────────
  echo "→ Creating Langfuse DB volume..."
  podman volume create obs-langfuse-db 2>/dev/null || true

  echo "→ Starting Langfuse Postgres..."
  podman rm -f obs-langfuse-db 2>/dev/null || true
  podman run -d \
    --name obs-langfuse-db \
    --network "$NETWORK" \
    -e POSTGRES_USER=langfuse \
    -e POSTGRES_PASSWORD=langfuse \
    -e POSTGRES_DB=langfuse \
    -v obs-langfuse-db:/var/lib/postgresql/data \
    postgres:15-alpine

  echo "→ Waiting for Postgres (up to 30s)..."
  for i in $(seq 1 30); do
    if podman exec obs-langfuse-db pg_isready -U langfuse -q 2>/dev/null; then
      echo "  ✓ Postgres ready (${i}s)"
      break
    fi
    sleep 1
  done

  # ── Langfuse ──────────────────────────────────────────────────────────────
  echo "→ Starting Langfuse..."
  podman rm -f obs-langfuse 2>/dev/null || true
  podman run -d \
    --name obs-langfuse \
    --network "$NETWORK" \
    -p 3100:3000 \
    -e "DATABASE_URL=postgresql://langfuse:langfuse@obs-langfuse-db:5432/langfuse" \
    -e "NEXTAUTH_URL=http://localhost:3100" \
    -e "NEXTAUTH_SECRET=local-dev-secret-not-for-prod" \
    -e "SALT=local-dev-salt-not-for-prod" \
    -e "TELEMETRY_ENABLED=false" \
    -e "LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=true" \
    langfuse/langfuse:2

  echo ""
  echo "✅  Full observability stack is up:"
  echo ""
  echo "   Langfuse   →  http://localhost:3100   ← sign up → get API keys"
   echo "   MLflow     →  http://localhost:5001"
  echo "   Jaeger     →  http://localhost:16686"
  echo "   Prometheus →  http://localhost:9090"
  echo "   Grafana    →  http://localhost:3000"
  echo "   OTel HTTP  →  http://localhost:4318"
  echo ""
  echo "After getting Langfuse keys, fill in deploy/local/.env.observability,"
  echo "then run the agent:"
  echo ""
  echo "  source deploy/local/.env.observability"
  echo "  uv run kubeflow-mcp agent --model openai/<model> --base-url <url> --langfuse"
}

case "${1:-start}" in
  stop)  stop  ;;
  start) start ;;
  *)     echo "Usage: $0 [start|stop]"; exit 1 ;;
esac
