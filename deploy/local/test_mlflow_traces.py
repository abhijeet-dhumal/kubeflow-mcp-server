"""Diagnostic for MLflow tracing: verifies flush + server export."""
import logging
import os
import sys
import time

MLFLOW_URI = "http://localhost:5001"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
# Clear the OTel endpoint so MLflow's MlflowV3SpanProcessor does NOT try to export
# metrics via OTLP (which requires opentelemetry-exporter-otlp-proto-* packages that
# may not be installed). Agent OTel tracing is handled separately via setup_otel_tracer().
os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
os.environ.pop("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", None)
# Force MLflow to use its own REST API rather than hijacking OTEL_EXPORTER_OTLP_ENDPOINT.
# Without this, when OTEL_EXPORTER_OTLP_ENDPOINT is set (for Jaeger), MLflow routes
# all GenAI traces through the OTel Collector instead of the MLflow tracking server.
os.environ["MLFLOW_ENABLE_OTLP_EXPORTER"] = "false"
# Force synchronous export so spans are sent before the script exits
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"

# Show MLflow internal warnings/errors including silent failures
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
logging.getLogger("mlflow").addHandler(_handler)
logging.getLogger("mlflow").setLevel(logging.DEBUG)

import mlflow  # noqa: E402 — import after env var so URI is picked up
from mlflow.entities import SpanType  # noqa: E402

mlflow.set_tracking_uri(MLFLOW_URI)

# ── versions ─────────────────────────────────────────────────────────────────
try:
    import langfuse

    try:
        lf_v = langfuse.version.__version__
    except AttributeError:
        lf_v = langfuse._version.__version__
    print(f"langfuse:          {lf_v}")
except Exception as e:
    print(f"langfuse error:    {e}")

print(f"mlflow:            {mlflow.__version__}")
print(f"tracking uri:      {mlflow.get_tracking_uri()}")
print(f"has flush:         {hasattr(mlflow, 'flush_trace_async_logging')}")
print()

# ── intercept the raw JSON sent by the client to diagnose 400 ────────────────
print("--- Capturing raw payload sent to POST /api/3.0/mlflow/traces ---")
try:
    import urllib.request as _ur
    _orig_urlopen = _ur.urlopen

    def _intercepting_urlopen(req, *args, **kwargs):
        if hasattr(req, "full_url") and "/traces" in req.full_url and req.method == "POST":
            body = req.data.decode() if req.data else ""
            print(f"  → {req.full_url}")
            print(f"  body (first 500 chars): {body[:500]}")
        return _orig_urlopen(req, *args, **kwargs)

    _ur.urlopen = _intercepting_urlopen
except Exception as e:
    print(f"  interceptor setup failed: {e}")

# ── create a root span ────────────────────────────────────────────────────────
print("--- Creating root span ---")
mlflow.set_experiment("trace-smoke-test")
try:
    with mlflow.start_span(
        name="smoke.test.trace",
        span_type=SpanType.CHAT_MODEL,
        attributes={
            "mlflow.traceSessionId": "smoke-session-001",
            "agent.framework": "test",
        },
    ) as span:
        span.set_inputs({"user": "hello"})
        span.set_outputs({"assistant": "world"})
    print("✅ start_span() succeeded")
except Exception as e:
    print(f"❌ start_span() failed: {type(e).__name__}: {e}")
    sys.exit(1)

# ── flush async export ────────────────────────────────────────────────────────
print("--- Flushing async trace export ---")
try:
    mlflow.flush_trace_async_logging(terminate=True)
    print("✅ flush succeeded")
except AttributeError:
    print("⚠️  flush_trace_async_logging not found — trying sleep(3)")
    time.sleep(3)
except Exception as e:
    print(f"❌ flush error: {type(e).__name__}: {e}")

# extra safety sleep
time.sleep(2)

# ── query server ──────────────────────────────────────────────────────────────
print("--- Querying MLflow REST API for traces ---")
try:
    import json
    import urllib.request

    # get experiment id
    req = urllib.request.Request(
        f"{MLFLOW_URI}/api/2.0/mlflow/experiments/get-by-name?experiment_name=trace-smoke-test"
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        d = json.loads(r.read())
    exp_id = d["experiment"]["experiment_id"]
    print(f"  experiment_id:   {exp_id}")

    # MLflow 3.x requires "locations" (TraceLocation objects), not "experiment_ids".
    # Use mlflow.search_traces() which handles the conversion automatically.
    from mlflow.entities import TraceLocation

    loc = TraceLocation.from_experiment_id(exp_id)
    print(f"  → using mlflow.search_traces(locations=[experiment_id={exp_id!r}])")
    traces_df = mlflow.search_traces(locations=[loc], max_results=10)
    print(f"  traces found:    {len(traces_df)}")
    for _, row in traces_df.iterrows():
        print(f"  - {row.get('trace_id', '?')[:16]}  session={row.get('mlflow.traceSessionId', '-')}")
except Exception as e:
    print(f"  error: {e}")

# ── also try direct MlflowClient.log_trace ───────────────────────────────────
print()
print("--- Testing MlflowClient direct trace logging ---")
try:
    from mlflow.entities import LiveSpan, Trace, TraceData, TraceInfo

    client = mlflow.MlflowClient(tracking_uri=MLFLOW_URI)
    # Check if log_trace exists
    if not hasattr(client, "log_trace"):
        print("⚠️  MlflowClient.log_trace not available — skip")
    else:
        print("✅ MlflowClient.log_trace exists")
except Exception as e:
    print(f"❌ {type(e).__name__}: {e}")
