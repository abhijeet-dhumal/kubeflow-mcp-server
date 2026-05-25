"""
MLflow data-lineage verifier.

Checks that every logged agent session has a consistent chain of:
  Experiment → Run (tags / params / metrics) → Artifacts → Traces

Usage:
    python deploy/local/check_lineage.py [--uri http://localhost:5001]
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from collections import defaultdict
from typing import Any

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")

# ── helpers ───────────────────────────────────────────────────────────────────

def _get(path: str) -> Any:
    with urllib.request.urlopen(f"{MLFLOW_URI}{path}", timeout=8) as r:
        return json.loads(r.read())


def _post(path: str, body: dict) -> Any:
    req = urllib.request.Request(
        f"{MLFLOW_URI}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read().decode("utf-8", errors="replace")}


def _ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


def _fail(msg: str) -> None:
    print(f"  ❌ {msg}")


# ── 1. list experiments ───────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"MLflow lineage check  →  {MLFLOW_URI}")
print(f"{'='*60}\n")

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_URI)
except ImportError:
    print("❌ mlflow not installed"); sys.exit(1)

data = _get("/api/2.0/mlflow/experiments/search?max_results=20")
experiments = data.get("experiments", [])
print(f"[1] Experiments found: {len(experiments)}")
for exp in experiments:
    print(f"    {exp['experiment_id']:>4}  {exp['name']:<35}  artifact_location={exp.get('artifact_location','?')}")

# Check artifact_location is mlflow-artifacts:/ (not a local path)
local_artifact_exps = [e for e in experiments if e.get("artifact_location", "").startswith("/")]
if local_artifact_exps:
    _warn(f"{len(local_artifact_exps)} experiment(s) still use a local artifact root "
          f"(leftover from before the mlflow-artifacts:/ fix — restart with a fresh volume to clear)")
else:
    _ok("All experiments use mlflow-artifacts:/ (HTTP artifact upload)")

# ── 2. runs per experiment ────────────────────────────────────────────────────

print(f"\n[2] Runs per experiment")
all_runs: list[dict] = []
for exp in experiments:
    exp_id = exp["experiment_id"]
    runs_data = _post("/api/2.0/mlflow/runs/search", {
        "experiment_ids": [exp_id],
        "max_results": 50,
    })
    runs = runs_data.get("runs", [])
    all_runs.extend(runs)
    if not runs:
        continue
    print(f"\n  Experiment '{exp['name']}' ({exp_id})  — {len(runs)} run(s)")
    for run in runs:
        info = run["info"]
        tags = {t["key"]: t["value"] for t in run.get("data", {}).get("tags", [])}
        params = {p["key"]: p["value"] for p in run.get("data", {}).get("params", [])}
        metrics = {m["key"]: m["value"] for m in run.get("data", {}).get("metrics", [])}
        run_id = info["run_id"][:8]
        model = tags.get("agent.model", params.get("agent.model", "?"))
        framework = tags.get("agent.framework", params.get("agent.framework", "?"))
        session_id = tags.get("agent.session_id", params.get("agent.session_id", "?"))
        turns = int(metrics.get("agent.session.turns", 0))
        tool_calls = int(metrics.get("agent.session.tool_calls", 0))
        llm_calls = int(metrics.get("agent.session.llm_calls", 0))
        status = info.get("status", "?")

        print(f"    run={run_id}  [{status}]  model={model}  framework={framework}")
        print(f"           session={session_id}")
        print(f"           turns={turns}  tool_calls={tool_calls}  llm_calls={llm_calls}")

        # validate required tags exist
        missing_tags = [k for k in ("agent.framework", "agent.model", "agent.session_id") if k not in tags]
        if missing_tags:
            _warn(f"run {run_id}: missing tags: {missing_tags}")
        else:
            _ok(f"run {run_id}: required tags present")

        # validate metrics
        if turns == 0 and status == "FINISHED":
            _warn(f"run {run_id}: FINISHED but agent.session.turns=0 — session may have ended before any turn")
        elif turns > 0:
            _ok(f"run {run_id}: {turns} turn(s) recorded")

# ── 3. traces per experiment ──────────────────────────────────────────────────

print(f"\n[3] Traces per experiment")

session_to_traces: dict[str, list] = defaultdict(list)
run_to_traces: dict[str, list] = defaultdict(list)

for exp in experiments:
    exp_id = exp["experiment_id"]
    try:
        # search_traces(experiment_ids=[...]) converts to locations internally.
        # TraceLocation objects fail against the local file store fallback — use
        # the experiment_ids kwarg which has a stable code path.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            traces_df = mlflow.search_traces(experiment_ids=[exp_id], max_results=100)
    except Exception as e:
        _warn(f"experiment {exp_id}: search_traces failed: {e}")
        continue

    if len(traces_df) == 0:
        print(f"  Experiment '{exp['name']}' ({exp_id})  — 0 traces")
        continue

    print(f"\n  Experiment '{exp['name']}' ({exp_id})  — {len(traces_df)} trace(s)")
    for _, row in traces_df.iterrows():
        trace_id = str(row.get("trace_id", "?"))[:20]
        session_id = str(row.get("mlflow.traceSessionId", "-"))
        run_id = str(row.get("mlflow.runId", "-"))
        span_name = str(row.get("span_name", "-"))
        status = str(row.get("status", "-"))
        print(f"    {trace_id}  span={span_name:<30}  session={session_id[:20]}  run={run_id[:8]}")
        session_to_traces[session_id].append(trace_id)
        run_to_traces[run_id].append(trace_id)

# ── 4. cross-reference: run ↔ traces ─────────────────────────────────────────

print(f"\n[4] Cross-reference: run ↔ traces")
runs_with_traces = set(run_to_traces.keys()) - {"-", "nan", ""}
runs_all = {r["info"]["run_id"] for r in all_runs}

for run_id_full in runs_all:
    run_id_short = run_id_full[:8]
    trace_count = len(run_to_traces.get(run_id_full, []))
    if trace_count > 0:
        _ok(f"run {run_id_short}: linked to {trace_count} trace(s)")
    else:
        _warn(f"run {run_id_short}: no traces linked via mlflow.runId tag")

# ── 5. cross-reference: session_id ↔ run tags ────────────────────────────────

print(f"\n[5] Cross-reference: session_id consistency (run tags ↔ trace attributes)")
run_sessions = {}
for run in all_runs:
    tags = {t["key"]: t["value"] for t in run.get("data", {}).get("tags", [])}
    sid = tags.get("agent.session_id", "")
    if sid:
        run_sessions[run["info"]["run_id"]] = sid

for session_id, trace_ids in session_to_traces.items():
    if session_id in ("-", "nan", ""):
        continue
    run_match = [rid for rid, sid in run_sessions.items() if sid == session_id]
    if run_match:
        _ok(f"session {session_id[:20]}: matched to run {run_match[0][:8]}, {len(trace_ids)} trace(s)")
    else:
        _warn(f"session {session_id[:20]}: {len(trace_ids)} trace(s) found but NO run with matching session_id")

# ── 6. artifact spot-check ────────────────────────────────────────────────────

print(f"\n[6] Artifact spot-check (last FINISHED run)")
finished_runs = [r for r in all_runs if r["info"].get("status") == "FINISHED"]
if finished_runs:
    last = sorted(finished_runs, key=lambda r: r["info"]["end_time"], reverse=True)[0]
    run_id = last["info"]["run_id"]
    exp_id = last["info"]["experiment_id"]
    try:
        arts = _get(f"/api/2.0/mlflow/artifacts/list?run_id={run_id}")
        files = [f["path"] for f in arts.get("files", [])]
        print(f"  run {run_id[:8]}  →  top-level artifacts: {files}")
        expected = {"agent_session_summary.json", "chat_session.json", "agent_turns"}
        found = set(files)
        missing = expected - found
        if missing:
            _warn(f"expected artifacts missing: {missing}")
        else:
            _ok("agent_session_summary.json + chat_session.json + agent_turns/ all present")
    except Exception as e:
        _warn(f"artifact list failed: {e}")
else:
    _warn("no FINISHED runs found — run the agent and exit cleanly (not Ctrl+C) to generate a closed run")

print(f"\n{'='*60}\n")
