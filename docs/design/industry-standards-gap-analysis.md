# Industry Standards Gap Analysis — kubeflow/mcp-server

**Audience:** Maintainers, security reviewers, WG ML Experience  
**Status:** Research snapshot vs `main` @ ~2026-07 (`70aeb1f` class) · **Updated:** 2026-07-18  
**Not a KEP.** Code wins over docs when they disagree.

| Companion | Role |
|-----------|------|
| [e2e-agent-surface-strategy.md](e2e-agent-surface-strategy.md) | What we intend to become |
| [maintainer-strategy-2026.md](../maintainer-strategy-2026.md) §7 | When we close gaps (initiative IDs) |
| [ROADMAP.md](../../ROADMAP.md) | Phased commitments |

**Baselines used**

| Domain | Standard / reference |
|--------|----------------------|
| Protocol auth | MCP authorization → OAuth 2.1, PKCE, audience-bound tokens (RFC 8707), protected-resource metadata (RFC 9728) |
| MCP risks | **OWASP MCP Top 10** (2026) |
| Observability | OpenTelemetry **GenAI / MCP semantic conventions** (Development stability) |
| Cloud-native agents | CNCF AI TCG / “cloud native agentic standards” (MELT, least-privilege tools, gateways) |
| Enterprise control plane | MCP gateway pattern (agentgateway, CSA agentic MCP guidance) |
| Supply chain | SBOM, signed images/provenance, continuous vuln scanning (OSV/pip-audit/Trivy) |
| Kubeflow maturity | [PROJECTS.md](https://github.com/kubeflow/community/blob/master/subprojects/PROJECTS.md) Experimental + Development disclaimer |

---

## 1. Executive scorecard

| Domain | Industry expectation | mcp-server today | Grade | Priority to close |
|--------|---------------------|------------------|-------|-------------------|
| **Observability — traces** | OTel MCP/GenAI spans + context propagation | Partial: tool spans + some MCP/GenAI attrs | **B−** | Medium (harden to semconv) |
| **Observability — metrics** | RED/USE + tool latency/error counters; `/metrics` or OTel metrics | **Missing** (docs claim `/metrics`) | **D** | High for K8s deploy |
| **Observability — logs/audit** | Immutable, identity-rich, SIEM-exportable audit | Structured audit logs; in-process only | **C** | High for enterprise |
| **Security — authn** | OAuth 2.1 resource server + audience binding | Bearer API key + JWKS JWT; **no** OIDC/OAuth 2.1 AS flow | **C** | High (HTTP); stdio OK for local |
| **Security — authz** | Per-caller least privilege; K8s RBAC / scopes | Server-wide persona + optional ns allowlist; **no** SAR | **C−** | High (Phase 4) |
| **Security — tool integrity** | Pin/hash tools; detect description rug-pulls | Soft integrity tests / TODOs | **D+** | Medium |
| **Security — docs/threat model** | SECURITY.md + threat model | **SECURITY.md missing** (#12) | **F** | **P0 for RC1** |
| **Reliability / probes** | `/health` `/ready`; sane HEALTHCHECK | MCP `health_check` only; **Dockerfile HEALTHCHECK broken** | **D** | **P0** |
| **MCP protocol completeness** | Server card, elicitation, conformance | Transports + progressive tools; no card/elicit/conformance suite | **C** | Medium |
| **Enterprise packaging** | Helm, NetworkPolicy, gateway compose | Image CI only; no Helm/NP | **D** | Phase 4 |
| **Supply chain** | SBOM, cosign, provenance, Scorecard | pip-audit + Trivy CRITICAL; no SBOM/sign | **C−** | Medium |
| **Eval / safety CI** | Rule judges + protocol conformance on PR | Designed, **not on main** | **D** | Post-RC |
| **Doc honesty** | Docs match code | Several overclaims (see §8) | **D** | **P0** |

**Overall:** Strong **local/dev MCP server** foundations (confirm gates, personas, resilience wrappers, OTel traces). Below **2026 industry bar** for HTTP multi-tenant production, full MELT observability, OAuth-resource-server auth, and supply-chain attestations — consistent with Experimental / “not for production” maturity.

---

## 2. Observability gaps

### 2.1 Industry bar (OTel + CNCF agentic)

| Expectation | Detail |
|-------------|--------|
| Traces | Span name `{mcp.method.name} {gen_ai.tool.name}`; attrs `mcp.method.name`, `mcp.session.id`, `mcp.protocol.version`, `gen_ai.tool.name`, `gen_ai.operation.name=execute_tool`, `error.type` on failure |
| Propagation | Parent from MCP `params._meta` / W3C `traceparent`; agent ↔ server one trace |
| Metrics | Tool call count, latency, errors; optional GenAI token/cost metrics |
| Logs | Correlated with `trace_id` / session; exportable |
| Audit | Who (identity) called what tool with what (redacted) args → durable sink |
| Content capture | Privacy-aware modes for args/results (OTel GenAI opt-in) |

### 2.2 What we have

- Optional OTLP HTTP traces (`core/telemetry.py`, `--otel-endpoint` / env).  
- Audit wrap sets MCP/GenAI-ish attributes + correlation UUID (`server.py`, `logging.py`, `middleware.py`).  
- Param masking on audit/spans (`security.py`).

### 2.3 Gaps (precise)

| Gap ID | Gap | Industry ref | Severity | Close via |
|--------|-----|--------------|----------|-----------|
| **O1** | No OTel **metrics** / Prometheus `/metrics` | MELT; K8s ops | High | MeterProvider + counters/histograms; or scrape endpoint |
| **O2** | HTTP `/health` `/ready` **absent**; Dockerfile probes `/health` | K8s probes | **P0** | HTTP edge + fix HEALTHCHECK |
| **O3** | Span naming / attribute set may not fully match current OTel MCP semconv (incl. `error.type`, span name format) | [OTel MCP semconv](https://github.com/open-telemetry/semantic-conventions-genai) | Medium | Audit against latest Development spec; pin version |
| **O4** | Weak **trace context propagation** from client `_meta` / HTTP | OTel MCP server guidance | Medium | Extract/inject context per semconv |
| **O5** | Audit not **immutable / exportable** (no SIEM hook, no signed log stream) | OWASP MCP08; CSA | High (enterprise) | OTLP logs or sidecar → Loki/Splunk; document retention |
| **O6** | JWT/`sub` not consistently mapped into audit identity | MCP08 | Medium | Bind AuthContext → audit + spans |
| **O7** | No reference **collector + Grafana** dashboards on `main` | CNCF agentic ops | Low–Med | `deploy/otel` example (demo has patterns) |
| **O8** | ROADMAP claims Langfuse “wired now” — **no code** | Honesty | Medium | Fix ROADMAP or implement |
| **O9** | No agent-level OTel on `main` (demo only) | End-to-end agent traces | Post-0.1 | Phase 3 / A3 |

---

## 3. Security gaps

### 3.1 OWASP MCP Top 10 mapping

| ID | Risk | Industry defense | mcp-server | Gap |
|----|------|------------------|------------|-----|
| **MCP01** | Token mismanagement / secret exposure | Short-lived scoped tokens; no passthrough | Static API keys + JWT verify; kubeconfig/cluster creds are the real power | Keys often long-lived; no token exchange to K8s |
| **MCP02** | Privilege escalation / scope creep | Fine-grained scopes; expiry | Server-wide persona | No per-caller / per-tool OAuth scopes |
| **MCP03** | Tool poisoning / rug pull | Pin tool definitions; drift alerts | Integrity tests incomplete | **No** pinned checksum gate in CI |
| **MCP04** | Supply chain | SBOM, signed artifacts, OSV | pip-audit + Trivy CRITICAL | No SBOM/cosign/provenance |
| **MCP05** | Command injection / unsafe exec | Validate + sandbox | AST script checks; `UNSAFE_SCRIPTS` escape hatch | Best-effort; document residual risk |
| **MCP06** | Intent / prompt injection | Isolate untrusted content | Confirm gates reduce agency | No response scanning / quarantine |
| **MCP07** | Insufficient authn/z | OAuth 2.1 + audience + MFA for sensitive ops | Bearer/JWT; open HTTP warned | **No** OAuth 2.1 AS, RFC 9728 metadata, audience binding |
| **MCP08** | Lack of audit/telemetry | Immutable identity-rich audit | Structured logs | Not durable; identity weak |
| **MCP09** | Shadow MCP servers | Registry / allowlist | N/A (we are a server) | Publish to MCP Registry; document known-good |
| **MCP10** | Context oversharing | Scoped context | Resources + progressive tools help | No formal context ACL |

### 3.2 Authn / authz vs MCP 2025–2026 spec direction

| Expectation | Status |
|-------------|--------|
| OAuth 2.1 resource server | ❌ Not implemented (Phase 4 / E2) |
| PKCE / AS delegation | ❌ |
| Audience-bound tokens (RFC 8707) | ❌ |
| `/.well-known/oauth-protected-resource` (RFC 9728) | ❌ |
| 401 + `WWW-Authenticate` metadata | ❌ (partial FastMCP behavior TBD) |
| Prefer **gateway** for enterprise auth | Documented as Phase 4 compose with agentgateway — **correct pattern**, not yet shipped |
| stdio local without OAuth | ✅ Acceptable for desktop IDEs |

### 3.3 Other security gaps

| Gap ID | Gap | Severity | Close via |
|--------|-----|----------|-----------|
| **S1** | **SECURITY.md + threat model missing** | **P0** | #12 / #14 |
| **S2** | Persona not bound to caller identity | High | JWT claims → persona/scopes; SAR (E2/E3) |
| **S3** | No SubjectAccessReview | High | Phase 4 |
| **S4** | `update_training_job` skips confirm pattern | Medium | Align with confirm-gate |
| **S5** | Tool description checksum not enforced | Medium | Pin baseline (#67 / integrity) |
| **S6** | No NetworkPolicy / PodSecurity docs | Medium | Helm + hardening guide |
| **S7** | Secret redaction not comprehensive scanner | Low–Med | Expand patterns; never return secrets in tools |

---

## 4. Reliability & operability gaps

| Gap ID | Gap | Industry bar | Severity |
|--------|-----|--------------|----------|
| **R1** | No HTTP liveness/readiness | Every K8s service | **P0** |
| **R2** | Dockerfile HEALTHCHECK points at missing `/health` | Image runnable in cluster | **P0** |
| **R3** | Rate limit not per-identity | Multi-tenant MCP | High (w/ HTTP) |
| **R4** | Retry helpers unused by tools | Resilient K8s clients | Medium |
| **R5** | No unified tool deadline / bulkhead | Gateway + server SLOs | Medium |
| **R6** | Process-local limiter/breaker (won’t share across replicas) | Document sticky/singleton or external RL | Medium when scaling HTTP |

---

## 5. MCP protocol & product gaps

| Gap ID | Gap | Severity |
|--------|-----|----------|
| **M1** | No MCP Server Card / `/.well-known/mcp.json` (or current SEP) | Medium (Phase 4) |
| **M2** | No native `ctx.elicit()` (confirm-gate is pragmatic substitute) | Low–Med |
| **M3** | No protocol conformance suite in CI (#62/#81) | High for credibility |
| **M4** | Progressive/semantic modes under-tested | Medium |
| **M5** | No eval Tier-1 safety judges on PR | Medium (post-RC) |

---

## 6. Supply chain & packaging gaps

| Gap ID | Gap | Severity |
|--------|-----|----------|
| **C1** | No SBOM attached to releases/images | Medium |
| **C2** | No cosign / provenance (attestations perm unused) | Medium |
| **C3** | No OpenSSF Scorecard workflow | Low–Med |
| **C4** | No Helm/Kustomize | High for platform install |
| **C5** | PyPI trusted publishing still in flight (#16/#28) | **P0** for 0.1 |

---

## 7. Doc vs code mismatches (trust debt)

| Claim | Reality | Action |
|-------|---------|--------|
| ARCHITECTURE: `/health` `/ready` `/metrics` Available | Not implemented | Fix docs **or** implement (prefer both: probes first) |
| ARCHITECTURE: `http_edge.py` Available | File absent | Remove claim |
| ARCHITECTURE: OTel “in review” | Shipped | Update table |
| ROADMAP: Langfuse wired at server | No code | Fix ROADMAP |
| README → SECURITY.md | 404 | Land #12 |

Dishonest docs are an industry-standards failure mode (adopters assume controls that don’t exist).

---

## 8. Prioritized closure plan

### P0 — before calling 0.1 “real”

| Item | Gap IDs | Initiative |
|------|---------|------------|
| SECURITY.md + threat model | S1 | G / #12 |
| Fix probes or Dockerfile HEALTHCHECK | O2, R1, R2 | E1 precursor |
| Honest ARCHITECTURE/ROADMAP | §7 | A2-style docs |
| Release path (PyPI) | C5 | G1 |

### P1 — Incubating / “usable in cluster”

| Item | Gap IDs | Initiative |
|------|---------|------------|
| HTTP `/health` `/ready` + optional `/metrics` | O1, O2, R1 | E1 |
| OTel metrics + semconv hardening | O1, O3, O4 | Observability |
| Conformance + coverage gate | M3 | A1 coverage, #62 |
| Durable audit export story | O5, O6, MCP08 | Docs + OTLP logs |
| Tool integrity pin | S5, MCP03 | #67 |

### P2 — enterprise / industry peer

| Item | Gap IDs | Initiative |
|------|---------|------------|
| OAuth 2.1 / OIDC + audience (or **mandate gateway**) | MCP07, S2 | E2, E5 |
| SAR + per-user rate limit | S3, R3 | E3 |
| Helm + NetworkPolicy + hardening | C4, S6 | E1 |
| SBOM + cosign | C1, C2, MCP04 | engprod |
| Gateway reference (agentgateway) | MCP enterprise pattern | E5 |
| Eval Tier-1 judges | M5 | post-RC |

### P3 — differentiators (not table stakes)

Agent OTel continuity, Langfuse optional, A2A, AGNTCY signing, full GenAI token metrics — after table stakes.

---

## 9. What we already do *better* than many MCP servers

Do not undersell these when talking to industry peers:

| Strength | Why it matters |
|----------|----------------|
| **Confirm / preview gates** on mutating tools | Mitigates excessive agency (OWASP LLM / MCP06-adjacent) |
| **Personas + policy YAML** | Least-privilege tool sets (CNCF: narrowly scoped tools) |
| **Namespace allowlists** | Multi-tenant footgun reduction |
| **Script AST safety** (default on) | MCP05 mitigation (imperfect but real) |
| **Circuit breaker + rate limit** on tool path | Basic abuse resistance |
| **Secret masking** in audit/spans | MCP01 hygiene |
| **Progressive / semantic tool modes** | Token/context safety |
| **OTel traces optional but present** | Ahead of uninstrumented tutorial servers |

Positioning: **best-in-class for Kubeflow-domain safety defaults**; **catch-up required** on OAuth-resource-server, MELT metrics/probes, and supply-chain attestations.

---

## 10. Recommended positioning statement

> kubeflow-mcp implements strong **domain safety** (confirm gates, personas, validation, optional OTel traces) suitable for local and trusted-cluster use. Against **2026 MCP industry standards**, it is not yet a full OAuth 2.1 resource server, does not expose Kubernetes-grade HTTP probes/metrics, and lacks SECURITY.md, Helm, and signed-SBOMs. Closing P0/P1 gaps is required before Incubating and before any “production-ready” messaging; enterprise authz should prefer **gateway + Phase 4 OIDC/SAR** rather than reinventing an IdP inside the server.

---

## 11. Sources

- OWASP MCP Top 10 (2026 industry summaries)  
- MCP authorization / OAuth 2.1 + RFC 8707 / 9728 guidance (Microsoft MCP security 2026; MCP-for-beginners)  
- OpenTelemetry GenAI/MCP semantic conventions (Development)  
- CNCF cloud native agentic standards / AI TCG checklist direction  
- CSA / gateway guidance for MCP control planes  
- In-repo audit of `kubeflow_mcp/core/*`, workflows, Dockerfile, ROADMAP, ARCHITECTURE (2026-07)

**Update when:** SECURITY.md lands, HTTP probes ship, OAuth/gateway reference lands, or OTel MCP semconv stabilize (re-audit attributes).
