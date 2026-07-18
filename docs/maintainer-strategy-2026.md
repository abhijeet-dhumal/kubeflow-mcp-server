# Maintainer Strategy — Kubeflow MCP Server (2026)

**Audience:** Approvers / active maintainers (`OWNERS`)  
**Status:** Working doc (not a KEP) · **Updated:** 2026-07-18  
**Doc map:** [ACTION-ITEMS.md](ACTION-ITEMS.md) (master checklist) · [ROADMAP](../ROADMAP.md) · [ARCHITECTURE](../ARCHITECTURE.md) · [design/README](design/README.md) · [#56 release](https://github.com/kubeflow/mcp-server/issues/56)

| Read this for… | Document |
|----------------|----------|
| **Strategy primer** (layers, five moves, community KEP path) | [design/e2e-agent-surface-strategy.md](design/e2e-agent-surface-strategy.md) |
| Weekly merge / RC1 ops | **This file** (§1–§6) |
| Experimental → Incubating → “platform default” | **This file** (§7) |
| Skills / marketplace / multi-repo shape | [design/ai-toolkit-ecosystem-vision.md](design/ai-toolkit-ecosystem-vision.md) |
| Per-project WG linkages | [design/kubeflow-ecosystem-ai-toolkit-map.md](design/kubeflow-ecosystem-ai-toolkit-map.md) |
| Industry gaps (security, observability, …) | [design/industry-standards-gap-analysis.md](design/industry-standards-gap-analysis.md) |

---

## 1. North star

**Make `kubeflow-mcp` the default agent interface to Kubeflow — not an optional experiment.**  
Do that by graduating maturity (PROJECTS.md), owning the LLMOps tool plane, and composing siblings (docs-agent, spark-history-mcp) — without turning this repo into a marketplace monorepo.

| Priority | Plane | When |
|----------|-------|------|
| **P0** | `kubeflow-mcp serve` (Trainer + platform) | **0.1 / RC1** |
| **P1** | Credibility: tests, release, SECURITY, adopters | **Through Incubating** |
| **P2** | `kubeflow-mcp agent` + Hub/Optimizer clients | **Post-0.1** |
| **P3** | Enterprise (OIDC, Helm, SAR) + Pipelines/Spark MCP | **Phase 4–6** |
| **P4** | Skills catalog / install profiles | Sibling `ai-toolkit` (propose post-RC1) |

### Principles

1. Serve before agent — never block RC1 on agent/MLflow/eval judges.  
2. Ship thin, track deep — epic + children; reject XXL “fixes #9” PRs.  
3. Prototype is a quarry (`demo/oss-summit-2026`) — carve, don’t merge.  
4. Issues only when startable (~2 weeks).  
5. Hold Phase 6 impl through RC1; docs under `/hold` OK.  
6. No self-`/approve` on own PRs.  
7. Wrap **SDK** only; compose sibling MCPs; align MLflow with **KEP-897** (don’t invent a tracker).

---

## 2. Baseline (main @ mid-2026)

| Area | State |
|------|-------|
| Trainer tools + core (auth, policy, resilience, health, security) | ✅ |
| Server OTel | ✅ (`core/telemetry.py`) |
| Optimizer / Hub | 🔲 stubs |
| Tests / e2e / benchmarks | ⚠️ thin / empty shells |
| Agent / eval / MLflow sinks | ❌ demo-only |
| PyPI release + SECURITY.md | 🔄 in flight (#16/#28, #12/#14) |
| PROJECTS.md maturity | **Experimental** |

---

## 3. RC1 = serve release

**In:** Trainer MCP, core platform, OTel, image CI, SECURITY, release path, P0 OpenShift/Trainer fixes, credible unit tests.  
**Out:** Agent (#15), Hub/Optimizer/Spark impl, Kind e2e as gate, benchmarks as gate, marketplace/skills repo.

### [#56](https://github.com/kubeflow/mcp-server/issues/56) blockers

| Must | Should | Won’t block |
|------|--------|-------------|
| #6 scaffold (*Relates to* #9) + #67/#68 started | #62/#81 conformance | #10/#26 benchmarks |
| #12/#14 SECURITY.md | #42/#47 crash-loop logs | #15 agent |
| #16/#28 release workflow | Coverage CI gate (A1) | #5/#34 Spark/Optimizer |
| #33 HF_HOME (one of #50/#83) | | #63 full Kind e2e |
| #79 triage | | #29 OSV (prefer post-RC) |
| Version + TestPyPI RC + announce | | |

**Milestone `v0.1`:** `#9 #12 #16 #33 #42 #56 #62 #67 #68 #79` (+ `#65` if capacity).

### Waves

| Wave | Focus | Exit |
|------|-------|------|
| **W0** Now | Fix #56 tracking; holds; triage #79 | Tracking = reality |
| **W1** ~2–4w | RC1 musts | Checklist green |
| **W2** +~2w | `0.1.0` tag | PyPI + image + announce |
| **W3** post-0.1 | Agent carve + first extra client | Agent MVP or Hub/Optimizer tools |
| **W4+** | Enterprise + composition + graduation | Incubating / distribution default |

---

## 4. Merge queue

**Order:** P0 bugs (#33/#42/#79) → release/security (#28/#14) → tests (#6/#82/#81) → Trainer UX (#66) → engprod (#72/#30) → held Phase 6 (leave held).

| Size | Policy |
|------|--------|
| ≤300 | Normal |
| 301–500 | Split or justify |
| >500 | Design / epic child required |

**Cadence:** Mon review queue · Wed holds · Fri update #56 (no new epics mid-week).

---

## 5. Issue hygiene (existing)

| Action | Issues |
|--------|--------|
| Rewrite tracking | #56 |
| Epic only (#6 = scaffold) | #9 → children #67/#68; #82 = discovery slice |
| RC1 must | #12/#14, #16/#28, #33, #79 |
| RC1 should | #62/#81, #42/#47 |
| Defer / hold | #10/#26, #15 (until W3), #5/#51, #34/#48, #63 |
| Post-RC | #29/#30, #59/#72 |

**Open now (W0):**  
- **A1** `ci: coverage threshold on test-python`  
- **A2** `docs: refresh ARCHITECTURE status table (OTel shipped)`  

**After #6:** bite-size #68/#67 children if needed (confirmed train paths, checksum baseline).  
**At RC tag:** GA checklist, nightly OSV, eval Tier-1 (MLflow sink only after Tier-1).  
**Post-0.1 agent (#15 children):** contracts → one provider → agent OTel → optional MLflow/Langfuse → compose docs.  
**Unhold later:** #34 Optimizer, Hub tools, #5 Spark — when SDK ready.

Draft bodies / WG comment templates: [Appendix A](#appendix-a--templates).

---

## 6. Demo carve + observability

```text
RC1 first → then agent contracts + 1 provider + OTel → optional MLflow/Langfuse → frameworks later
```

| Layer | Mechanism | MLflow |
|-------|-----------|--------|
| Serve | `core/telemetry` + OTEL_* | Optional collector sink only |
| Agent | Turn/tool spans + middleware | Optional `MLFLOW_TRACKING_URI` |
| Platform | **KEP-897** | First-class Kubeflow tracker — align, don’t compete |

No TrainJob experiment-tracking MCP tools in 0.1 (Hub + KEP-897 own lineage/tracking).

---

## 7. Initiatives: Experimental → next big thing

Graduation target (community [maturity](https://github.com/kubeflow/community/blob/master/subprojects/maturity_requirements.md)): **Experimental → Incubating → Graduated**, plus *de facto* “install Kubeflow → agents just work.”

### 7.1 Maturity & distribution (make it official)

| ID | Initiative | Why it graduates us | When |
|----|------------|---------------------|------|
| **G1** | Ship **0.1.0** (PyPI + GHCR) with RELEASE.md | Installable product, not a clone-and-pray experiment | W1–W2 |
| **G2** | Meet **Incubating** criteria (OWNERS depth, docs, CI, adopters) | PROJECTS.md promotion | 0.1 → +1–2 quarters |
| **G3** | Land in **community-distribution** as optional→default package | Users get MCP with the platform | Post-0.1 |
| **G4** | Website + install guide (“AI agents” first-class nav) | Discovery = legitimacy | With G1 |
| **G5** | [#54](https://github.com/kubeflow/mcp-server/issues/54) subprojects page + blog cadence | Community visibility | Continuous |
| **G6** | Publish to **MCP Registry** (+ list Smithery/Glama) | Global agent discovery | At 0.1 tag |
| **G7** | Adopters.md with **3+ named orgs/users** | Maturity evidence | Drive intentionally |

### 7.2 Product depth (own the LLMOps tool plane)

| ID | Initiative | Link upstream | When |
|----|------------|---------------|------|
| **P1** | Harden Trainer tools (OpenShift, logs, HF suggest) | #33 #42 #65/#66 | RC1 → 0.1.x |
| **P2** | **Hub MCP** (register / version / catalog discover) | Hub + SDK `ModelRegistryClient` | First extra client post-0.1 |
| **P3** | **Optimizer MCP** (OptimizationJob path) | Katib + Trainer OptimizationJob | After Hub or parallel if owner |
| **P4** | TrainJob **progress/metrics** tools | Trainer [KEP-2779](https://github.com/kubeflow/trainer/tree/main/proposals/2779-trainjob-progress) | When CRD exposes data |
| **P5** | Extensible BuiltinTrainer tools (TRL/Unsloth/…) | Trainer [#2839](https://github.com/kubeflow/trainer/issues/2839) | Don’t hardcode torchtune forever |
| **P6** | Pipelines MCP | SDK [KEP-125](https://github.com/kubeflow/sdk/tree/main/proposals/125-kfp-client) | After client beta |
| **P7** | Spark: compose History MCP now; SparkClient MCP later | [spark-history-mcp](https://github.com/kubeflow/mcp-apache-spark-history-server), #5 | Profile now / tools later |
| **P8** | End-to-end skill path: **fine-tune → Hub → KServe** | Hub + KServe | Skills + thin tools |

### 7.3 Agent & IDE experience (daily driver)

| ID | Initiative | Notes | When |
|----|------------|-------|------|
| **A1** | `kubeflow-mcp agent` MVP (one provider) | Carve from demo; #15 | W3 |
| **A2** | Progressive/semantic tool modes default for small models | Token story = adoption | With agent / serve |
| **A3** | Cursor / Claude / VS Code **one-file** profiles in-repo | Friction &lt; 5 minutes | 0.1 |
| **A4** | Notebooks **WorkspaceKind** recipe (mcp sidecar) | WG Notebooks v2 | Post-Workspaces |
| **A5** | Compose **docs-agent** MCP (ask docs + do Trainer) | KEP-867 | Install profile |
| **A6** | Eval Tier-1 (safety judges) on PR | ROADMAP Phase 2 | Post-RC |
| **A7** | Optional agent MLflow/Langfuse — **same URI story as KEP-897** | No competing tracker | W3+ |

### 7.4 Enterprise & platform (why platform teams care)

| ID | Initiative | Notes | When |
|----|------------|-------|------|
| **E1** | Helm + Kustomize + probes docs | In-cluster default | Phase 4 |
| **E2** | OIDC / OAuth 2.1 for HTTP serve | Align SDK auth [#281](https://github.com/kubeflow/sdk/issues/281) | Phase 4 |
| **E3** | SubjectAccessReview-bound tools | Real multi-tenant | Phase 4 |
| **E4** | MCP Server Card `/.well-known` | Discovery | Phase 4 |
| **E5** | **agentgateway** reference compose | MCP federation — don’t rebuild | Phase 4 |
| **E6** | A2A endpoint | Orchestrators | Phase 4 |
| **E7** | Threat model + SECURITY.md living doc | #12 | RC1 then continuous |
| **E8** | Audit / tool-call signing (AGNTCY) | ROADMAP | Later |

### 7.5 Ecosystem composition (suite, not silo)

| ID | Initiative | Partners |
|----|------------|----------|
| **C1** | Install profile **trainer-dev** (mcp-server only) | — |
| **C2** | Profile **kubeflow-ai-suite** = mcp-server + docs-agent + spark-history-mcp | Sibling repos |
| **C3** | Profile **platform-admin** = + kubernetes-mcp-server | containers/kubernetes-mcp-server |
| **C4** | Propose **`kubeflow/ai-toolkit`** (skills + catalog + profiles) | See vision doc — **after RC1** |
| **C5** | Joint WG demos (Training + ML Experience + Data) | Trainer ROADMAP MCP line |
| **C6** | CNCF / AAIF narrative: “agentic on Kubeflow” | OTel GenAI, MCP Registry, A2A |

### 7.6 Community & GSoC (force multiplier)

| ID | Initiative |
|----|------------|
| **M1** | GFI ladder: docs → tests (#67/#68) → one tool → one skill |
| **M2** | GSoC / outreach: Hub tools, eval judges, Notebooks recipe, skills catalog |
| **M3** | Shared OWNERS/reviewers with SDK / Trainer for API changes |
| **M4** | Quarterly “state of Kubeflow MCP” blog + demo video refresh |

### 7.7 What “next big thing” looks like (success bar)

| Signal | Target |
|--------|--------|
| Maturity | **Incubating** on PROJECTS.md within ~2 quarters of 0.1 |
| Distribution | In community-distribution (optional, then recommended) |
| Discovery | MCP Registry listing + website AI agents page |
| Coverage | Trainer + **Hub** (or Optimizer) tools in production use |
| Composition | Documented suite profile with ≥1 sibling MCP |
| Adopters | ≥3 public adopters / case studies |
| Reliability | Coverage gate + conformance + no P0 OpenShift footguns |
| Narrative | “Talk to Kubeflow” is a WG talking point, not a side project |

### 7.8 Explicit non-goals (stay sharp)

- Replacing SDK, kubectl, or KServe  
- Building a global MCP/skills marketplace SaaS  
- Merging docs-agent / spark-history-mcp into this monorepo  
- Blocking releases on multi-framework agent zoo  
- First-party TrainJob MLflow CRUD tools (platform KEP-897 + Hub)

---

## 8. Your checklist

**W0:** Rewrite #56 · milestone · holds · #79 · open A1/A2 · #6 · one HF_HOME PR  
**W1:** #14 #28 · #67/#68/#82 · #81 · coverage gate · no agent impl  
**W2:** TestPyPI → PyPI · announce · MCP Registry · Batch C issues  
**W3+:** #15 carve · Hub or Optimizer · suite profiles · ai-toolkit proposal · Incubating push  

Cross-WG outreach list: [ecosystem map §8](design/kubeflow-ecosystem-ai-toolkit-map.md#8-cross-project-collaboration-checklist-for-you).

---

## 9. Success metrics (ops)

| Horizon | Signal |
|---------|--------|
| RC1 | Musts green; TestPyPI install works |
| 0.1.0 | PyPI + image; #56 closed or GA follow-up |
| Incubating | G1–G7 + P2 or P3 underway |
| Health | Fewer XXL PRs; milestones honest; ROADMAP matches main |

---

## Appendix A — Templates

### #56 comment

```text
Refreshing RC1 scope so tracking matches main + ROADMAP.

**0.1.0rc1 = serve release** (trainer + core + CI/security/tests). Agent (#15),
Spark (#5), Optimizer (#34), and benchmarks (#10) are **not** RC1 blockers.

### Must
- Test scaffold (#6) + coverage path (#67/#68)
- SECURITY.md (#12/#14)
- Release workflow (#16/#28)
- OpenShift HF_HOME (#33 — one PR)
- #79 triage/fix
- Version bump + TestPyPI RC

### Should
- MCP conformance (#62/#81)
- Crash-loop logs (#42/#47)
- Coverage CI gate

### Won’t block RC1
- #10, #15, #5/#34, #63 full Kind e2e

Attaching Must list to milestone v0.1. Objections welcome before rc1.
```

### #9 comment

```text
#6 lands scaffold/harness only (Relates to #9). Coverage stays in #67/#68.
#82 = discovery slice of #68. Benchmarks = #10 (not RC1 critical path).
```

### A1 / A2 issue titles

- `ci: add coverage threshold to test-python workflow` — floor we already meet, ratchet to 75%.  
- `docs: refresh ARCHITECTURE.md component table (OTel shipped; drop stale in-review rows)`.

### #15 kickoff (post-RC1)

```text
Phase 3 after 0.1rc1. Epic stays #15; land demo slices:
contracts → one provider → agent OTel → optional MLflow/Langfuse → compose docs.
Serve stays OTel-first; MLflow env-gated and aligned with KEP-897.
```

---

**Not here:** LLDs ([design/](design/)), SECURITY prose (#12), KEP text (community/).  
**Update when:** RC1 ships, Incubating push starts, or ROADMAP phases change.
