"""Quick MLflow API inspection — runs, traces, sessions."""
import json
import os
import urllib.request

BASE = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")


def get(path, params=None):
    url = f"{BASE}{path}"
    if params:
        from urllib.parse import urlencode
        url += "?" + urlencode(params, doseq=True)
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


# ── Experiments ──────────────────────────────────────────────────────────────
print("=== EXPERIMENTS ===")
try:
    d = post("/api/2.0/mlflow/experiments/search", {"max_results": 10})
    for e in d.get("experiments", []):
        print(f"  [{e['experiment_id']}] {e['name']}")
except Exception as ex:
    print(f"  error: {ex}")

# ── Runs ─────────────────────────────────────────────────────────────────────
print("\n=== RUNS (last 5) ===")
try:
    d = post(
        "/api/2.0/mlflow/runs/search",
        {"experiment_ids": ["0"], "max_results": 5, "order_by": ["start_time DESC"]},
    )
    for r in d.get("runs", []):
        info = r["info"]
        metrics = {m["key"]: m["value"] for m in r.get("data", {}).get("metrics", [])}
        relevant = {k: v for k, v in metrics.items() if "session" in k or "turn" in k}
        print(f"  {info['run_id'][:8]}  {info['status']}  \"{info.get('run_name', '-')}\"")
        if relevant:
            print(f"    metrics: {relevant}")
except Exception as ex:
    print(f"  error: {ex}")

# ── Traces ───────────────────────────────────────────────────────────────────
print("\n=== TRACES ===")
try:
    d = get("/api/2.0/mlflow/traces", {"experiment_ids": "0", "max_results": 30})
    traces = d.get("traces", [])
    print(f"  Total: {len(traces)}")
    sessions: dict = {}
    for t in traces:
        info = t.get("trace_info", {})
        tags = {x["key"]: x["value"] for x in info.get("tags", [])}
        name = tags.get("mlflow.traceName", info.get("request_id", "?")[:12])
        sid = tags.get("mlflow.traceSessionId", "untagged")
        status = info.get("status", "-")
        print(f"  {info.get('request_id','?')[:14]}  name={name}  status={status}  session={sid[:24]}")
        sessions.setdefault(sid, []).append(info.get("request_id", "?")[:8])

    print("\n=== SESSIONS ===")
    if not sessions:
        print("  (no session-tagged traces found — run the agent first with new code)")
    for sid, ids in sessions.items():
        print(f"  {sid[:36]}  ({len(ids)} traces)  ids={ids[:4]}")
except Exception as ex:
    print(f"  error: {ex}")
