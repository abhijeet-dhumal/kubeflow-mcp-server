#!/usr/bin/env bash
# Spin up OTel + MLflow + Langfuse local observability stack via podman run.
# No podman-compose needed.
#
# Usage:
#   ./deploy/local/start.sh          # start
#   ./deploy/local/start.sh stop     # stop + remove containers
#   ./deploy/local/start.sh logs     # tail all logs

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NETWORK=obs-net

stop() {
  echo "→ Stopping observability stack..."
  podman rm -f \
    obs-jaeger obs-otel obs-prometheus obs-grafana \
    obs-mlflow obs-langfuse-db obs-langfuse 2>/dev/null || true
  podman network rm "$NETWORK" 2>/dev/null || true
  echo "✓ Done."
}

logs() {
  podman logs -f obs-jaeger obs-otel obs-prometheus obs-mlflow obs-langfuse 2>&1
}

start() {
  # ── Idempotent: tear down if already running ──────────────────────────────
  stop 2>/dev/null || true

  echo "→ Creating network: $NETWORK"
  podman network create "$NETWORK" 2>/dev/null || true

  # ── Jaeger ────────────────────────────────────────────────────────────────
  echo "→ Starting Jaeger..."
  podman run -d --name obs-jaeger --network "$NETWORK" \
    -e COLLECTOR_OTLP_ENABLED=true \
    -p 16686:16686 \
    jaegertracing/all-in-one:latest

  # ── MLflow ────────────────────────────────────────────────────────────────
  echo "→ Starting MLflow..."
  podman volume create obs-mlflow-data 2>/dev/null || true
  podman run -d --name obs-mlflow --network "$NETWORK" \
    -p 5000:5000 \
    -v obs-mlflow-data:/mlflow \
    ghcr.io/mlflow/mlflow:v3.12.0 \
    mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --backend-store-uri sqlite:////mlflow/mlflow.db \
      --artifacts-destination /mlflow/artifacts \
      --default-artifact-root mlflow-artifacts:/

  # ── OTel Collector ────────────────────────────────────────────────────────
  echo "→ Starting OTel Collector..."
  podman run -d --name obs-otel --network "$NETWORK" \
    -p 4318:4318 \
    -p 8889:8889 \
    -v "$SCRIPT_DIR/otel-collector-config.yaml:/etc/otel/config.yaml:ro,Z" \
    otel/opentelemetry-collector-contrib:latest \
    --config /etc/otel/config.yaml

  # ── Prometheus ────────────────────────────────────────────────────────────
  echo "→ Starting Prometheus..."
  podman run -d --name obs-prometheus --network "$NETWORK" \
    -p 9090:9090 \
    -v "$SCRIPT_DIR/prometheus.yml:/etc/prometheus/prometheus.yml:ro,Z" \
    prom/prometheus:latest

  # ── Grafana ───────────────────────────────────────────────────────────────
  echo "→ Starting Grafana..."
  podman run -d --name obs-grafana --network "$NETWORK" \
    -e GF_AUTH_ANONYMOUS_ENABLED=true \
    -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin \
    -p 3001:3000 \
    grafana/grafana:latest

  # ── Langfuse Postgres ─────────────────────────────────────────────────────
  echo "→ Starting Langfuse Postgres..."
  podman volume create obs-langfuse-db 2>/dev/null || true
  podman run -d --name obs-langfuse-db --network "$NETWORK" \
    -e POSTGRES_USER=langfuse \
    -e POSTGRES_PASSWORD=langfuse \
    -e POSTGRES_DB=langfuse \
    -v obs-langfuse-db:/var/lib/postgresql/data \
    postgres:15-alpine

  echo "→ Waiting for Postgres to be ready..."
  for i in $(seq 1 20); do
    if podman exec obs-langfuse-db pg_isready -U langfuse -q 2>/dev/null; then
      echo "  ✓ Postgres ready (${i}s)"
      break
    fi
    sleep 1
  done

  # ── Langfuse ──────────────────────────────────────────────────────────────
  echo "→ Starting Langfuse..."
  podman run -d --name obs-langfuse --network "$NETWORK" \
    -p 3000:3000 \
    -e DATABASE_URL=postgresql://langfuse:langfuse@obs-langfuse-db:5432/langfuse \
    -e NEXTAUTH_URL=http://localhost:3000 \
    -e NEXTAUTH_SECRET=local-dev-secret-not-for-prod \
    -e SALT=local-dev-salt-not-for-prod \
    -e TELEMETRY_ENABLED=false \
    -e LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=true \
    langfuse/langfuse:2

  echo ""
  echo "✅  Stack is up. Services:"
  echo "   Langfuse   →  http://localhost:3000   (create project → get API keys)"
  echo "   Jaeger     →  http://localhost:16686"
  echo "   MLflow     →  http://localhost:5000"
  echo "   Prometheus →  http://localhost:9090"
  echo "   Grafana    →  http://localhost:3001"
  echo ""
  echo "Next: create a Langfuse account at http://localhost:3000, copy the API keys"
  echo "      into deploy/local/.env.observability, then:"
  echo ""
  echo "  source deploy/local/.env.observability"
  echo "  uv run kubeflow-mcp agent --model openai/<model> --base-url <url> --langfuse"
}

case "${1:-start}" in
  stop)  stop  ;;
  logs)  logs  ;;
  start) start ;;
  *)     echo "Usage: $0 [start|stop|logs]"; exit 1 ;;
esac
