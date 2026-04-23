# Manage Training Jobs

Use the kubeflow-mcp MCP server tools to list, inspect, suspend, resume, and clean up training jobs.

## Common Workflows

### List and filter jobs

- `list_training_jobs()` — all jobs in current namespace
- `list_training_jobs(status="Running")` — only active jobs
- `list_training_jobs(runtime="torch-tune", limit=10)` — filter by runtime

### Inspect a specific job

- `get_training_job(name="<job-name>")` — full status, config, and conditions
- `get_training_logs(name="<job-name>")` — pod logs with `failure_hint`
- `get_training_events(name="<job-name>")` — K8s events (scheduling, image pull, etc.)

### Pause and resume

- `suspend_training_job(name="<job-name>", confirmed=true)` — frees GPU resources without deleting the job
- `resume_training_job(name="<job-name>", confirmed=true)` — resumes from where it was suspended

Use suspend/resume to temporarily free GPUs for higher-priority work.

### Wait for completion

- `wait_for_training(name="<job-name>", timeout_seconds=3600)` — blocks until job reaches `Complete` or `Failed`
- Specify `target_statuses=["Complete","Failed"]` to catch either outcome

### Clean up

- `delete_training_job(name="<job-name>", confirmed=true)` — permanent, irreversible
- To bulk clean: list jobs with a status filter, then delete each one

## Key Details

- All write operations (`suspend`, `resume`, `delete`) require `confirmed=true` — call with `confirmed=false` first to preview
- Job names follow K8s naming rules: lowercase, alphanumeric, hyphens only, max 63 chars
- `suspend` only works on `Running` jobs; `resume` only works on `Suspended` jobs
