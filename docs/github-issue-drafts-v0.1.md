# GitHub issue drafts — v0.1 concrete release

**Repo:** `kubeflow/mcp-server` · **Milestone:** `v0.1`  
**Rule:** Draft here → you post. Do not auto-create.

Labels to use (adjust if missing): `area/release`, `area/engprod`, `area/testing`, `area/docs`, `area/security`, `area/core`, `kind/bug`, `kind/feature`, `good first issue`

---

## Already open — do not recreate (attach to `v0.1`)

| Issue | Role for 0.1 | Your action |
|-------|--------------|-------------|
| [#56](https://github.com/kubeflow/mcp-server/issues/56) | Epic tracker | Post **Draft A** comment below |
| [#12](https://github.com/kubeflow/mcp-server/issues/12) | SECURITY.md | Milestone + drive #14 |
| [#16](https://github.com/kubeflow/mcp-server/issues/16) | Release workflow | Milestone + drive #28 |
| [#9](https://github.com/kubeflow/mcp-server/issues/9) | Test epic | Comment: scaffold ≠ close; children #67/#68 |
| [#67](https://github.com/kubeflow/mcp-server/issues/67) / [#68](https://github.com/kubeflow/mcp-server/issues/68) | Coverage work | Milestone |
| [#62](https://github.com/kubeflow/mcp-server/issues/62) | Conformance (Should) | Milestone |
| [#33](https://github.com/kubeflow/mcp-server/issues/33) | HF_HOME Must | Milestone; one PR of #50/#83 |
| [#42](https://github.com/kubeflow/mcp-server/issues/42) | Crash-loop logs Should | Milestone |
| [#79](https://github.com/kubeflow/mcp-server/issues/79) | Init bug Must-triage | Milestone |

**Out of v0.1 milestone:** #15 agent, #5 Spark, #34 Optimizer, #10 benchmarks, #63 full Kind e2e

---

## Draft A — Comment on #56 (post first)

**Title:** _(comment, not new issue)_

```markdown
### Refreshing RC1 / 0.1 scope (2026-07)

**0.1.0rc1 = serve release** (Trainer + core + CI/security/tests).  
Not blockers: agent (#15), Spark (#5), Optimizer (#34), benchmarks (#10), full Kind e2e (#63).

#### Must
- [ ] Test scaffold (#6) — Relates to #9; coverage via #67/#68
- [ ] SECURITY.md (#12 / #14)
- [ ] Release workflow + RELEASE.md (#16 / #28)
- [ ] HTTP `/health` + `/ready` + fix Dockerfile HEALTHCHECK (new issue)
- [ ] Coverage CI floor (new issue)
- [ ] Docs: ARCHITECTURE/ROADMAP match main (new issue)
- [ ] OpenShift HF_HOME (#33 — one PR)
- [ ] Triage/fix #79
- [ ] Version bump + TestPyPI RC + announce

#### Should
- [ ] MCP conformance (#62 / #81)
- [ ] Crash-loop logs (#42 / #47)
- [ ] HF model suggest follow-ups (#65 / #66) if capacity

#### Won’t block 0.1
- #10, #15, #5, #34, #63

Please attach Must (+ Should as capacity) to milestone **v0.1**.
Tracking checklist also in-repo: `docs/ACTION-ITEMS.md` Waves 0–2.
```

---

## New issues to create (in order)

Post **one at a time**. After each lands, link it from #56.

### Issue 1 — HTTP probes (P0) ← **start here**

**Title:** `fix(core): add HTTP /health and /ready; fix Dockerfile HEALTHCHECK`

**Labels:** `kind/bug`, `area/core`, `area/engprod`, `area/release`

**Body:**

```markdown
### Description

Container and K8s-style deploys need HTTP liveness/readiness probes. Today:

- MCP tool `health_check` exists, but there is **no** HTTP `/health` or `/ready`.
- `Dockerfile` `HEALTHCHECK` hits `/health`, so the image health check is **broken**.
- `ARCHITECTURE.md` claims `/health` `/ready` `/metrics` as Available — docs ahead of code.

This is a **v0.1 Must** (see #56).

### Proposal

1. Add a minimal HTTP edge (or FastMCP/Starlette routes) exposing:
   - `GET /health` — process up (liveness)
   - `GET /ready` — config/auth/policy loaded; optional light dependency check (readiness)
2. Fix `Dockerfile` `HEALTHCHECK` to use the real path/port.
3. Document probe usage in README or deploy notes.
4. Update `ARCHITECTURE.md` component table to match reality (`/metrics` stays out of scope unless implemented in the same PR — prefer **not** to block on Prometheus for 0.1).

### Out of scope

- Full Prometheus `/metrics` / OTel metrics (follow-up; see industry gap analysis)
- OIDC, Helm chart
- Changing MCP `health_check` tool semantics

### Done when

- [ ] `GET /health` and `GET /ready` return 200 on a running HTTP server
- [ ] Dockerfile HEALTHCHECK passes against a locally built image
- [ ] Unit or smoke test covers the routes
- [ ] ARCHITECTURE.md no longer claims probes that do not exist
- [ ] Linked from #56

### Relates to

- #56 (0.1 release)
- ROADMAP Phase 2 health endpoints
- `docs/ACTION-ITEMS.md` items 1.7, 1.8
```

---

### Issue 2 — Coverage gate

**Title:** `ci: add coverage threshold to test-python workflow`

**Labels:** `area/engprod`, `area/testing`, `good first issue`

**Body:**

```markdown
### Description

ROADMAP Phase 2 calls for a coverage gate (~75%). CI runs coverage/Coveralls today but does **not** fail PRs that drop coverage. We need a floor for v0.1 credibility.

### Proposal

- Add `pytest-cov` fail-under (or equivalent) to `.github/workflows/test-python.yaml` / `make test-python`
- Start with a **floor we already meet on main**, then ratchet toward 75%
- Document the floor in CONTRIBUTING.md

### Out of scope

- 75% on day one if baseline is lower
- e2e / Kind coverage (#63)

### Done when

- [ ] PRs that drop below the configured floor fail CI
- [ ] Floor documented in CONTRIBUTING.md
- [ ] Linked from #56 and #9

### Relates to

- #9, #67, #68, #56
- `docs/ACTION-ITEMS.md` item 1.11
```

---

### Issue 3 — Docs honesty

**Title:** `docs: sync ARCHITECTURE.md and ROADMAP.md with shipped main`

**Labels:** `area/docs`, `good first issue`

**Body:**

```markdown
### Description

Several docs overclaim or underclaim vs `main`:

- ARCHITECTURE marks OTel as “in review” though `core/telemetry.py` ships
- ARCHITECTURE claims `/health` `/ready` `/metrics` and `http_edge.py` Available — not accurate
- ROADMAP says Langfuse is “wired at server level now” — no Langfuse code on main
- README links SECURITY.md which is still missing (#12)

### Proposal

1. Refresh ARCHITECTURE component status table to match main.
2. Fix ROADMAP Langfuse / observability wording (planned vs shipped).
3. Cross-link SECURITY.md only after #12 lands (or note “pending #12”).
4. Point readers at `docs/design/` for strategy (optional one-liner in ARCHITECTURE or README).

### Out of scope

- Implementing missing features (probes = separate issue)
- Full strategy doc rewrite

### Done when

- [ ] No false “Available” rows for missing modules/endpoints
- [ ] OTel marked shipped
- [ ] ROADMAP Langfuse line corrected
- [ ] Linked from #56

### Relates to

- #56, #12
- `docs/ACTION-ITEMS.md` item 0.5 / 0.7
```

---

### Issue 4 — (optional v0.1 Should) Metrics deferral tracker

Only if you want a placeholder so `/metrics` is not forgotten:

**Title:** `feat(observability): add OTel metrics and optional Prometheus /metrics`

**Labels:** `area/observability`, `area/core`

**Body:** Short: post-0.1 / Incubating; tool call counters + latency; not a 0.1 Must. Relates to industry gap O1. **Milestone:** none or post-0.1.

---

## Posting order (one by one)

1. Comment **Draft A** on #56  
2. Create **Issue 1** (probes) → paste URL into #56  
3. Create **Issue 2** (coverage) → link #56  
4. Create **Issue 3** (docs sync) → link #56  
5. Milestone existing Must issues  
6. Continue existing PRs: #14, #28, #6, #33/#83, #79  

---

## After Issue 1 is filed

Reply in chat with the issue number; we draft the next PR-sized implementation plan or move to Issue 2.
