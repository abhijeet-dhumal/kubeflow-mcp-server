# Local OTel Stack

Two options — pick whichever fits your setup.

---

## Option A — Homebrew (no Docker, simplest)

Jaeger all-in-one (≥ 1.35) accepts OTLP/HTTP directly on `:4318`, so no
collector is needed for traces. Install once, `start` on boot.

```bash
brew install jaeger prometheus grafana
brew services start jaeger
brew services start prometheus
brew services start grafana      # optional — Jaeger UI is usually enough
```

Default ports after `brew services start`:

| Service | URL | Purpose |
|---|---|---|
| Jaeger UI | http://localhost:16686 | Trace explorer |
| Jaeger OTLP/HTTP | http://localhost:4318 | Where agents send spans |
| Prometheus | http://localhost:9090 | Metrics |
| Grafana | http://localhost:3000 | Dashboards |

> **Prometheus scrape config** — brew installs Prometheus with
> `/usr/local/etc/prometheus.yml` (Intel) or
> `/opt/homebrew/etc/prometheus.yml` (Apple Silicon).
> Add a scrape job for your app there if needed.

Stop services:

```bash
brew services stop jaeger prometheus grafana
```

---

## Option B — Docker Compose (full collector pipeline)

Runs OTel Collector → Jaeger + Prometheus + Grafana.  Use this when you want
the collector's batch/retry/memory-limiter processors, or plan to forward to a
remote backend.

```bash
docker compose -f deploy/otel/docker-compose.yml up -d
docker compose -f deploy/otel/docker-compose.yml down
```

| Service | URL |
|---|---|
| Jaeger UI | http://localhost:16686 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| OTLP HTTP (collector) | http://localhost:4318 |

---

## Agent OTel (`kubeflow-mcp agent`)

Same command regardless of which option you chose — both listen on `:4318`:

```bash
# via flag
uv run kubeflow-mcp agent \
  --provider litellm \
  --model ollama/qwen3:8b \
  --otel-endpoint http://localhost:4318

# via env var (picked up automatically)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
uv run kubeflow-mcp agent --provider litellm --model ollama/qwen3:8b
```

Spans appear in Jaeger under **service = `kubeflow-mcp-agent`**.

---

## FastMCP OTel (`kubeflow-mcp serve`) — zero code

FastMCP activates OTel via env vars only:

```bash
OTEL_SERVICE_NAME=kubeflow-mcp-server \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
uv run kubeflow-mcp serve
```

For stdio (Cursor / Claude Code), add to your MCP server JSON config:

```json
{
  "env": {
    "OTEL_SERVICE_NAME": "kubeflow-mcp-server",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4318"
  }
}
```

Spans appear under **service = `kubeflow-mcp-server`** — same Jaeger UI,
different service filter.
