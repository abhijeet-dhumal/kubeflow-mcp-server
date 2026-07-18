# Master Action Items — Kubeflow MCP Server

**Purpose:** Single checklist to address strategy, graduation, industry gaps (security, observability, …), and ecosystem composition — efficiently, in order.  
**Updated:** 2026-07-18 · **Status:** Working tracker (not a KEP)

| Detail lives in… | Link |
|------------------|------|
| Strategy primer | [design/e2e-agent-surface-strategy.md](design/e2e-agent-surface-strategy.md) |
| RC1 ops + initiative IDs | [maintainer-strategy-2026.md](maintainer-strategy-2026.md) |
| Industry gaps | [design/industry-standards-gap-analysis.md](design/industry-standards-gap-analysis.md) |
| Ecosystem / WG | [design/kubeflow-ecosystem-ai-toolkit-map.md](design/kubeflow-ecosystem-ai-toolkit-map.md) |
| Skills / marketplace | [design/ai-toolkit-ecosystem-vision.md](design/ai-toolkit-ecosystem-vision.md) |

**How to use:** Work **Wave 0 → 1 → 2** before opening Wave 3+ epics. Check boxes in PRs/issues; keep this file honest.

---

## Wave 0 — Tracking & honesty (this week)

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 0.1 | Rewrite [#56](https://github.com/kubeflow/mcp-server/issues/56) Must/Should/Won’t (serve RC1 only) | maintainer-strategy §3 | ☐ |
| 0.2 | Attach Must issues to milestone `v0.1` | #9 #12 #16 #33 #42 #56 #62 #67 #68 #79 | ☐ |
| 0.3 | Confirm `/hold` on Spark #51, Optimizer #48, benchmarks #26 | — | ☐ |
| 0.4 | Triage [#79](https://github.com/kubeflow/mcp-server/issues/79) (repro or close) | — | ☐ |
| 0.5 | Fix doc lies: ARCHITECTURE (OTel shipped; no fake `/metrics`/`http_edge`); ROADMAP (Langfuse not wired) | gap §7 | ☐ |
| 0.6 | Open issue: coverage gate on `test-python` | A1 | ☐ |
| 0.7 | Open issue: ARCHITECTURE status table refresh | A2 | ☐ |
| 0.8 | WG ML Experience agenda: e2e strategy one-pager (S0) | e2e §8 | ☐ |

---

## Wave 1 — RC1 musts (credibility)

### Security (P0)

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 1.1 | Land **SECURITY.md** + threat model | #12 #14 · E7 · S1 | ☐ |
| 1.2 | Document residual risks (script AST, `UNSAFE_SCRIPTS`, static API keys) | SECURITY.md | ☐ |
| 1.3 | Align `update_training_job` with confirm-gate (or document exception) | S4 | ☐ |

### Release & supply chain (P0)

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 1.4 | Merge release workflow + RELEASE.md | #16 #28 · G1 · C5 | ☐ |
| 1.5 | Version bump → TestPyPI RC → validate `kubeflow-mcp serve` | #56 | ☐ |
| 1.6 | GitHub Release → PyPI + GHCR image | G1 | ☐ |

### Reliability / probes (P0)

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 1.7 | Implement HTTP **`/health`** and **`/ready`** (minimal edge) | O2 R1 R2 | ☐ |
| 1.8 | Fix **Dockerfile HEALTHCHECK** to match real endpoints | R2 | ☐ |

### Tests & quality

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 1.9 | Land test scaffold [#6](https://github.com/kubeflow/mcp-server/pull/6) (*Relates to* #9) | — | ☐ |
| 1.10 | Drive [#67](https://github.com/kubeflow/mcp-server/issues/67) / [#68](https://github.com/kubeflow/mcp-server/issues/68); land [#82](https://github.com/kubeflow/mcp-server/pull/82) discovery slice | — | ☐ |
| 1.11 | Add **coverage floor** CI (ratchet toward 75%) | A1 · gap coverage | ☐ |
| 1.12 | Review/merge MCP conformance [#81](https://github.com/kubeflow/mcp-server/pull/81) (or minimal subset) | #62 · M3 | ☐ |

### Trainer P0 bugs

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 1.13 | One PR for OpenShift **HF_HOME** | #33 #50/#83 · P1 | ☐ |
| 1.14 | Crash-loop log fallback (rebase #47) | #42 · P1 | ☐ |

---

## Wave 2 — 0.1.0 ship + discovery

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 2.1 | Tag **0.1.0**; announce `#kubeflow-ml-experience` | G1 | ☐ |
| 2.2 | Publish to **MCP Registry** | G6 · MCP09 | ☐ |
| 2.3 | List on Smithery/Glama (optional) | G6 | ☐ |
| 2.4 | Website / docs: “AI agents” install page | G4 | ☐ |
| 2.5 | [#54](https://github.com/kubeflow/mcp-server/issues/54) subprojects page | G5 | ☐ |
| 2.6 | Changelog (git-cliff) if ready | #59 #72 | ☐ |
| 2.7 | Nightly OSV / dependency audit on main | #29 #30 · C2 | ☐ |
| 2.8 | IDE one-file profiles (Cursor / Claude / VS Code) in-repo | A3 · C1 | ☐ |
| 2.9 | Open community **discussion issue** (e2e strategy S1) | e2e §8.7 | ☐ |
| 2.10 | Start **Adopters.md** drive (target ≥3) | G7 | ☐ |

---

## Wave 3 — Observability & audit (industry P1)

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 3.1 | OTel **metrics**: tool calls, latency, errors (MeterProvider) | O1 | ☐ |
| 3.2 | Optional Prometheus **`/metrics`** or document OTel-only path | O1 | ☐ |
| 3.3 | Harden spans to **OTel MCP/GenAI semconv** (name format, `error.type`) | O3 | ☐ |
| 3.4 | Trace **context propagation** from MCP `_meta` / HTTP | O4 | ☐ |
| 3.5 | Map JWT/`sub` → audit identity + span attrs | O6 · MCP08 | ☐ |
| 3.6 | Document durable audit export (OTLP logs → SIEM) | O5 | ☐ |
| 3.7 | Reference `deploy/otel` (collector + Jaeger/Grafana) on main | O7 | ☐ |
| 3.8 | Pin **tool description checksums** in CI | S5 · MCP03 · #67 | ☐ |
| 3.9 | Eval Tier-1 **rule-based safety judges** on PR | A6 · M5 | ☐ |

---

## Wave 4 — Agent + suite composition

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 4.1 | #15 design ACK; agent contracts (D1) | A1 | ☐ |
| 4.2 | `kubeflow-mcp agent` MVP — **one** provider | A1 | ☐ |
| 4.3 | Agent OTel turn/tool spans (correlate with serve) | A / O9 | ☐ |
| 4.4 | Optional MLflow/Langfuse — **KEP-897 URI story only** | A7 | ☐ |
| 4.5 | Install profile **trainer-dev** | C1 | ☐ |
| 4.6 | Profile **kubeflow-ai-suite** (mcp + docs-agent + spark-history-mcp) | C2 · P7 | ☐ |
| 4.7 | Profile **platform-admin** (+ kubernetes-mcp) | C3 | ☐ |
| 4.8 | Joint blog/demo with spark-history-mcp + docs-agent maintainers | C5 | ☐ |

---

## Wave 5 — Product depth (LLMOps tool plane)

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 5.1 | **Hub MCP** tools (register/list/get/catalog discover) | P2 | ☐ |
| 5.2 | **Optimizer MCP** (OptimizationJob path) | P3 · unhold #34 | ☐ |
| 5.3 | TrainJob **progress** tools when KEP-2779 ready | P4 | ☐ |
| 5.4 | Plan extensible BuiltinTrainer tools (#2839) | P5 | ☐ |
| 5.5 | HF suggest / ranking follow-ups | #65 #66 · P1 | ☐ |
| 5.6 | Pipelines MCP — only after SDK KEP-125 beta | P6 | ☐ |
| 5.7 | SparkClient MCP — unhold #5 only after design freeze | P7 | ☐ |

---

## Wave 6 — Enterprise (industry P2)

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 6.1 | **Helm + Kustomize** + probe/values docs | E1 · C4 | ☐ |
| 6.2 | NetworkPolicy / PodSecurity hardening section | S6 | ☐ |
| 6.3 | OIDC / OAuth 2.1 HTTP (or mandate gateway-only) | E2 · MCP07 | ☐ |
| 6.4 | Audience-bound tokens / protected-resource metadata (or gateway) | MCP07 | ☐ |
| 6.5 | **SubjectAccessReview** tool authz | E3 · S3 | ☐ |
| 6.6 | Per-user / per-`sub` rate limits | R3 · E3 | ☐ |
| 6.7 | MCP Server Card `/.well-known` | E4 · M1 | ☐ |
| 6.8 | **agentgateway** reference compose + docs | E5 | ☐ |
| 6.9 | A2A endpoint (if orchestrator demand) | E6 | ☐ |
| 6.10 | SBOM on image/release + **cosign**/provenance | C1 C2 · MCP04 | ☐ |
| 6.11 | OpenSSF Scorecard workflow (optional) | C3 | ☐ |

---

## Wave 7 — Skills, community KEP, graduation

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 7.1 | Short community **KEP** after S1 ACK (or KEP-936 follow-on) | e2e §8 · S2 | ☐ |
| 7.2 | Propose / create **`kubeflow/ai-toolkit`** | C4 · vision | ☐ |
| 7.3 | Seed 3–5 official Agent Skills + `catalog.yaml` | vision AT1 | ☐ |
| 7.4 | Skill: fine-tune → Hub → KServe path | P8 | ☐ |
| 7.5 | Package in **community-distribution** (optional→recommended) | G3 | ☐ |
| 7.6 | Nominate **Incubating** on PROJECTS.md | G2 | ☐ |
| 7.7 | GFI ladder + GSoC themes published | M1 M2 | ☐ |
| 7.8 | Shared review cadence with SDK/Trainer OWNERS | M3 | ☐ |
| 7.9 | Quarterly “state of Kubeflow MCP” blog | M4 | ☐ |
| 7.10 | CNCF/AAIF narrative: agentic on Kubeflow | C6 | ☐ |

---

## Wave 8 — Later differentiators (do not start early)

| # | Action | Refs | Done |
|---|--------|------|:----:|
| 8.1 | Native `ctx.elicit()` when clients ready | M2 · ROADMAP | ☐ |
| 8.2 | AGNTCY tool-call signatures | E8 | ☐ |
| 8.3 | GenAI token/cost metrics | OTel GenAI | ☐ |
| 8.4 | Feast MCP | sdk#239 | ☐ |
| 8.5 | Notebooks WorkspaceKind + mcp sidecar | A4 | ☐ |
| 8.6 | Multi-framework agent adapters (beyond one provider) | demo carve | ☐ |

---

## Efficiency rules (how not to thrash)

1. **Never** open Wave 5–7 epics that block Wave 1 Musts.  
2. Prefer **gateway compose (6.8)** over building a full IdP in-tree for 6.3–6.4.  
3. **One** OpenShift HF_HOME PR; close duplicates.  
4. Skills/`ai-toolkit` **after** 0.1rc1 tag.  
5. Hub before Pipelines/Spark tools unless a dedicated owner appears.  
6. Fix **doc claims** in the same PR as features (or before).  
7. Every new issue needs Done-when + Out-of-scope; link a row `#` from this file.

---

## Success bar (when to stop calling it “experiment”)

- [ ] Waves 0–2 complete (0.1 + Registry + honest docs + SECURITY + probes)  
- [ ] Wave 3 enough for MELT story (metrics + semconv + audit export docs)  
- [ ] Wave 4 suite profile live **or** Wave 5 Hub tools live  
- [ ] ≥3 adopters · community-distribution path · Incubating nomination in flight  

---

## Quick count

| Wave | Focus | Approx. items |
|------|-------|---------------|
| 0 | Tracking / honesty | 8 |
| 1 | RC1 musts | 14 |
| 2 | Ship + discover | 10 |
| 3 | Observability / audit | 9 |
| 4 | Agent + suite | 8 |
| 5 | Hub / Optimizer / … | 7 |
| 6 | Enterprise | 11 |
| 7 | Skills / KEP / graduate | 10 |
| 8 | Later | 6 |

**Total tracked:** ~83 · **Do first:** 0.* + 1.* (~22 items) unlock everything else.
