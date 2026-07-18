# Kubeflow Ecosystem Map — Linkages for an AI Toolkit / MCP Strategy

**Audience:** Cross-WG planning · mcp-server maintainers  
**Status:** Research snapshot · **Updated:** 2026-07-18 · not a KEP  
**Index:** [design/README](README.md) · **Strategy primer:** [e2e-agent-surface-strategy.md](e2e-agent-surface-strategy.md) · **Vision (skills/marketplace):** [ai-toolkit-ecosystem-vision.md](ai-toolkit-ecosystem-vision.md) · **Graduation initiative IDs:** [maintainer-strategy §7](../maintainer-strategy-2026.md#7-initiatives-experimental--next-big-thing)

**Sources:** [PROJECTS.md](https://github.com/kubeflow/community/blob/master/subprojects/PROJECTS.md) · [wg-list](https://github.com/kubeflow/community/blob/master/wg-list.md) · component ROADMAPs/KEPs · AAIF/CNCF landscape

| Section | Content |
|---------|---------|
| §1–2 | Findings + WGs |
| §3–4 | Maturity map + per-project initiatives |
| §5–6 | Industry links + opportunity timing |
| §7–11 | Org shape, checklist, risks, contacts |

---

## 1. Executive findings

1. **Kubeflow already has three Experimental AI-agent surfaces:** `mcp-server`, `mcp-apache-spark-history-server`, and `docs-agent` (RAG/MCP for docs). A “one-stop toolkit” should **compose** these, not merge them.
2. **SDK is the contract layer** (Incubating): Trainer / Optimizer / Hub / Spark(Connect) / Pipelines(alpha) are wrap-ready at different maturity — MCP tools should track SDK, not invent parallel APIs.
3. **KEP-897 (MLflow-first experiment tracking)** is the platform answer for GenAI tracing/experiments — aligns with optional MLflow on agent/eval paths; do **not** invent a competing tracker inside mcp-server.
4. **KEP-936 already anticipates skills + multi-MCP** (see `assets/skills-mcp.png`, Multi-MCP table). Community narrative exists; execution needs an `ai-toolkit` (or equivalent) home.
5. **Industry stack to align with (not rebuild):** AAIF (MCP, AGENTS.md, goose), A2A (LF), agentgateway (LF/AAIF), MCP Registry, Agent Skills (`agentskills.io`), OTel GenAI semconv, CNCF AI TCG agentic checklist.

---

## 2. Working groups — who owns what

| WG | Slack / call | Subprojects most relevant to AI toolkit | Linkage play |
|----|--------------|----------------------------------------|--------------|
| **[ML Experience](https://github.com/kubeflow/community/tree/master/wg-ml-experience)** | `#kubeflow-ml-experience` · bi-weekly Wed 08:00 PT | SDK, Kale, mcp-server (KEP-936 ownership), UX | **Primary home** for AI toolkit / skills catalog / agent UX |
| **[Training](https://github.com/kubeflow/community/tree/master/wg-training)** | `#kubeflow-trainer` · bi-weekly Wed 08:00 PT | Trainer, Katib | TrainJob progress (#2779), OptimizationJob, LLM BuiltinTrainers (#2839) → MCP tools + skills |
| **[Data](https://github.com/kubeflow/community/tree/master/wg-data)** | Hub + Spark calls | Hub (Model Registry + Catalog), Spark Operator | Hub MCP tools; compose with Spark History MCP; Catalog discovery skills |
| **[Pipelines](https://github.com/kubeflow/community/tree/master/wg-pipelines)** | `#kubeflow-pipelines` | Pipelines, KFP | PipelinesClient MCP after KEP-125 stabilizes; MLflow KEP-12862 |
| **[Notebooks](https://github.com/kubeflow/community/tree/master/wg-notebooks)** | `#kubeflow-notebooks` | Notebooks v1/v2 Workspaces | Host IDE agents / mcp-server sidecars; Kale GSoC composable notebooks |

**Implication:** Cross-WG “AI Toolkit” initiative needs ML Experience as sponsor, with Training/Data/Pipelines/Notebooks as **dependency WGs**, not silent scope grabs.

---

## 3. Maturity map (official subprojects)

From [PROJECTS.md](https://github.com/kubeflow/community/blob/master/subprojects/PROJECTS.md):

### Graduated (platform pillars)

| Subproject | Repo | AI-toolkit role |
|------------|------|-----------------|
| **Trainer** | [kubeflow/trainer](https://github.com/kubeflow/trainer) | Primary MCP client today; LLM FT + progress metrics |
| **Katib** | [kubeflow/katib](https://github.com/kubeflow/katib) | Engine behind OptimizerClient / OptimizationJob |
| **Hub** | [kubeflow/hub](https://github.com/kubeflow/hub) | Model Registry + **Model Catalog** (federated HF etc.) — next MCP + skills |
| **Pipelines** | [kubeflow/pipelines](https://github.com/kubeflow/pipelines) | Orchestration MCP via SDK PipelinesClient |
| **Spark Operator** | [kubeflow/spark-operator](https://github.com/kubeflow/spark-operator) | Job lifecycle; agent debug via sibling MCP |
| **Notebooks** | [kubeflow/notebooks](https://github.com/kubeflow/notebooks) | Workspace host for agent IDEs |

### Incubating

| Subproject | Repo | AI-toolkit role |
|------------|------|-----------------|
| **SDK** | [kubeflow/sdk](https://github.com/kubeflow/sdk) | **Single Python façade** MCP must wrap ([KEP-819](https://github.com/kubeflow/community/tree/master/proposals/819-kubeflow-sdk)) |
| **Kale** | [kubeflow/kale](https://github.com/kubeflow/kale) | Notebook→pipeline; GSoC 2026 composable notebooks — skill goldmine |

### Experimental (agent-native already)

| Subproject | Repo | AI-toolkit role |
|------------|------|-----------------|
| **MCP Server** | [kubeflow/mcp-server](https://github.com/kubeflow/mcp-server) | Capability runtime (Trainer first) |
| **MCP Spark History Server** | [kubeflow/mcp-apache-spark-history-server](https://github.com/kubeflow/mcp-apache-spark-history-server) | **Peer pattern** — MCP + CLI + skills export |
| **MLflow Integration** | [kubeflow/mlflow-integration](https://github.com/kubeflow/mlflow-integration) | Platform experiment tracking ([KEP-897](https://github.com/kubeflow/community/tree/master/proposals/897-experiment-tracking)) |

### Adjacent / not in PROJECTS.md but critical

| Project | Org | Role |
|---------|-----|------|
| **KServe** | [kserve/kserve](https://github.com/kserve/kserve) | GenAI inference (LLMInferenceService, InferenceGraph) — deploy skills after Hub |
| **docs-agent** | [kubeflow/docs-agent](https://github.com/kubeflow/docs-agent) | Docs RAG MCP ([KEP-867](https://github.com/kubeflow/community/tree/master/proposals/867-kubeflow-documentation-ai)); GSoC agentic RAG |
| **Arena** | [kubeflow/arena](https://github.com/kubeflow/arena) | Legacy CLI — avoid duplicating; SDK/MCP is the modern path |
| **Dashboard** | [kubeflow/dashboard](https://github.com/kubeflow/dashboard) | UI shell — deep-links to Hub/MLflow/MCP docs |
| **community-distribution** | [kubeflow/community-distribution](https://github.com/kubeflow/community-distribution) | Package optional mcp-server + MLflow later |
| **website / blog** | kubeflow/website, blog | Narrative + install profiles |

---

## 4. Per-project deep dive — initiatives to link

### 4.1 Trainer (`wg-training`) — highest priority MCP dependency

| Initiative | Link | Toolkit action |
|------------|------|----------------|
| ROADMAP MCP integration | [trainer ROADMAP](https://github.com/kubeflow/trainer/blob/main/ROADMAP.md) | Keep mcp-server in sync; joint demos |
| LLM Trainer v2 / torchtune | [KEP-2401](https://github.com/kubeflow/trainer/tree/main/proposals/2401-llm-trainer-v2) | Skills: fine-tune-lora, HF gated models |
| Dynamic BuiltinTrainers (TRL/Unsloth/…) | [#2839](https://github.com/kubeflow/trainer/issues/2839), [#2752](https://github.com/kubeflow/trainer/issues/2752) | Extensible MCP trainer tools — **don’t hardcode only torchtune forever** |
| TrainJob progress / live metrics | [KEP-2779](https://github.com/kubeflow/trainer/tree/main/proposals/2779-trainjob-progress) | Agent monitoring tools get real signal (not only logs) |
| OptimizationJob (Katib trials over TrainJob) | [#3749](https://github.com/kubeflow/trainer/issues/3749) etc. | Skill: HPO loop → best trial → retrain |
| Data cache (platform) | [KEP-2655](https://github.com/kubeflow/community/tree/master/proposals/2655-kubeflow-data-cache) | Future skill: “warm cache then train” |
| Checkpointing | trainer [#2777](https://github.com/kubeflow/trainer/issues/2777) | Lifecycle tools later |

### 4.2 Katib + SDK Optimizer

| Initiative | Link | Toolkit action |
|------------|------|----------------|
| OptimizerClient shipped | [sdk optimize docs](https://github.com/kubeflow/sdk/blob/main/docs/source/optimize/index.rst) | Unhold Optimizer MCP **after** 0.1 + API stable |
| SDK HPO KEP | [sdk KEP-46](https://github.com/kubeflow/sdk/tree/main/proposals/46-hyperparameter-optimization) | Tool schemas follow SDK, not raw Katib REST |
| LLM HPO (historical Katib) | [katib KEP-2339](https://github.com/kubeflow/katib/tree/master/docs/proposals/2339-hpo-for-llm-fine-tuning) | Context for skills; prefer OptimizationJob path |

### 4.3 Hub — Model Registry + Model Catalog (`wg-data`)

| Initiative | Link | Toolkit action |
|------------|------|----------------|
| Hub rename / umbrella | [KEP-907](https://github.com/kubeflow/community/tree/master/proposals/907-model-registry-renaming), website Hub docs | Brand skills as “Hub”, not “model-registry” only |
| Registry OpenAPI | [hub OpenAPI](https://github.com/kubeflow/hub/blob/main/api/openapi/model-registry.yaml) | MCP hub client wraps SDK `ModelRegistryClient` |
| **Model Catalog** (federated discovery, HF etc.) | [Hub overview](https://www.kubeflow.org/docs/components/hub/overview/) | **High-value skills:** discover → register → deploy; complements HF MCP |
| KServe ↔ Registry controller | model-registry [#577](https://github.com/kubeflow/model-registry/issues/577) (completed lineage) | Skill: promote registered model → InferenceService |
| API v1 graduation | hub KEP-0004 | Gate “stable” Hub MCP tools on API maturity |

### 4.4 Pipelines (`wg-pipelines`)

| Initiative | Link | Toolkit action |
|------------|------|----------------|
| PipelinesClient in SDK | [sdk KEP-125](https://github.com/kubeflow/sdk/tree/main/proposals/125-kfp-client), [#125](https://github.com/kubeflow/sdk/issues/125) | MCP pipelines module **after** alpha→beta |
| MLflow integration (KFP) | pipelines proposal KEP-12862 (cited from KEP-897) | Skills correlate pipeline runs ↔ MLflow experiments |
| Kale composable notebooks (GSoC 2026) | [kale roadmap](https://github.com/kubeflow/kale/blob/main/docs/source/roadmap.md) | Skills: notebook→pipeline; multi-notebook workflows |

### 4.5 Spark (`wg-data`)

| Initiative | Link | Toolkit action |
|------------|------|----------------|
| Spark Operator | [spark-operator](https://github.com/kubeflow/spark-operator) | Runtime for jobs |
| SparkClient (Connect shipped; batch WIP) | [sdk KEP-107](https://github.com/kubeflow/sdk/tree/main/proposals/107-spark-client), [#520](https://github.com/kubeflow/sdk/issues/520) | mcp-server #5 stays held until batch story clear |
| **MCP Spark History Server** | [mcp-apache-spark-history-server](https://github.com/kubeflow/mcp-apache-spark-history-server) | **Compose in install profiles now** — already MCP + CLI + skill export |
| SDK ROADMAP “AI-Assisted Spark via MCP” | sdk ROADMAP → mcp-server #5 | Joint profile: `spark-debug` = History MCP (+ optional SparkClient later) |

### 4.6 Notebooks (`wg-notebooks`)

| Initiative | Link | Toolkit action |
|------------|------|----------------|
| Notebooks 2.0 Workspaces | [notebooks ROADMAP](https://github.com/kubeflow/notebooks/blob/main/ROADMAP.md), [#85](https://github.com/kubeflow/notebooks/issues/85) | Long-term: WorkspaceKind template with mcp-server + Cursor/code-server |
| VS Code / Jupyter images | notebooks v1 | Short-term docs: “run mcp-server in notebook pod” recipe skill |

### 4.7 KServe (external but ecosystem-critical)

| Initiative | Link | Toolkit action |
|------------|------|----------------|
| LLMInferenceService | kserve ROADMAP / llmisvc issues | Post-Hub skill: deploy LLM endpoint |
| InferenceGraph / RAG agents | [#3829](https://github.com/kserve/kserve/issues/3829) | Compose docs-agent RAG **or** user RAG graphs — don’t reinvent serving agents as MCP |
| Open Inference / OpenAI-compatible APIs | kserve ROADMAP | Agent tools call inference HTTP; optional thin MCP later |

### 4.8 docs-agent + Documentation AI

| Initiative | Link | Toolkit action |
|------------|------|----------------|
| KEP-867 Documentation AI | [community#867](https://github.com/kubeflow/community/tree/master/proposals/867-kubeflow-documentation-ai) | Knowledge MCP — compose with action MCP |
| Agentic RAG GSoC 2026 | docs-agent [#176](https://github.com/kubeflow/docs-agent/issues/176), [#200](https://github.com/kubeflow/docs-agent/issues/200) | Shared patterns (eval, MCP tools); **don’t fork RAG into mcp-server** |
| PR triage / CI agents | docs-agent issues | Internal maintainer agents — separate from user-facing toolkit |

### 4.9 MLflow Integration + experiment tracking

| Initiative | Link | Toolkit action |
|------------|------|----------------|
| **KEP-897** MLflow-first platform | [proposal](https://github.com/kubeflow/community/tree/master/proposals/897-experiment-tracking) | **Strategic align:** agent/eval OTel→MLflow optional sinks match platform direction |
| `kubeflow/mlflow-integration` | Experimental subproject | Prefer platform MLflow over embedding MLflow server in mcp-server |
| SDK MLflow issue | [sdk#63](https://github.com/kubeflow/sdk/issues/63) | Future: auto-log TrainJob metrics into MLflow — agent-visible |
| GenAI tracing in MLflow | Called out in KEP-897 | Agent session middleware (demo branch) should target **same** tracking URI story |

### 4.10 SDK cross-cutting (Incubating)

| Initiative | Link | Toolkit action |
|------------|------|----------------|
| Unified SDK KEP | [KEP-819](https://github.com/kubeflow/community/tree/master/proposals/819-kubeflow-sdk) | MCP always wraps SDK clients |
| kube-authkit / unified auth | [sdk#281](https://github.com/kubeflow/sdk/issues/281) | mcp-server Phase 4 OIDC/SAR must align |
| OTel in SDK / TrainJobs | [sdk#164](https://github.com/kubeflow/sdk/issues/164), [#399](https://github.com/kubeflow/sdk/issues/399) | End-to-end traces: agent → mcp → TrainJob |
| Feast client | [sdk#239](https://github.com/kubeflow/sdk/issues/239) | Far-future MCP; feature-store skills later |
| Workspace snapshot → TrainJob | [sdk#48](https://github.com/kubeflow/sdk/issues/48) | Killer agent UX skill when unfrozen |
| MCP companion | KEP-936, mcp-server | This repo |

### 4.11 Distribution, UI, examples

| Project | Linkage |
|---------|---------|
| **community-distribution** | Optional packages: mcp-server, mlflow-integration, spark-history-mcp |
| **dashboard** | Links to Hub UI, MLflow UI, “AI tools” docs |
| **examples** | Seed recipe content for ai-toolkit skills |
| **website / blog** | Install-profile landing; announce multi-MCP composition |

---

## 5. Industry / foundation initiatives worth linking

### 5.1 Protocol & identity (align, don’t fork)

| Initiative | Org | Value for Kubeflow |
|------------|-----|-------------------|
| **MCP** + **MCP Registry** | AAIF / Anthropic origins | Publish mcp-server; discovery API |
| **AGENTS.md** | OpenAI → widely adopted | Already in mcp-server / sdk / hub — keep consistent |
| **Agent Skills** ([agentskills.io](https://agentskills.io)) | Open standard | Canonical format for ai-toolkit skills |
| **A2A** (Agent2Agent) | Linux Foundation | mcp-server Phase 4 `/a2a`; orchestrator federation |
| **AGNTCY Identity** | Linux Foundation (cited in KEP-936 / ROADMAP) | Tool-call signatures / supply-chain trust |
| **AAIF** (Agentic AI Foundation) | LF — MCP, goose, AGENTS.md, agentgateway | Narrative: Kubeflow is CNCF *infra* for AAIF *protocols* |
| **CNCF AI TCG** agentic checklist | CNCF | Conformance language for “agentic on Kubernetes” |

### 5.2 Gateways & composition

| Project | Value |
|---------|-------|
| **[agentgateway](https://github.com/agentgateway/agentgateway)** | MCP + A2A + LLM proxy; OTel; enterprise federation — already named in mcp-server ROADMAP Phase 4 |
| **LiteLLM Proxy** | Org LLM budgets — stay external (ROADMAP) |
| Microsoft MCP Gateway / IBM Context Forge | Alternatives mentioned in KEP DESIGN — document compatibility, don’t pick exclusive |

### 5.3 Observability & eval

| Project | Value |
|---------|-------|
| **OpenTelemetry GenAI / MCP semconv** | Server + agent spans; cost/GPU metrics story with CNCF |
| **Langfuse** | LLM session UX — optional (already in demo/ROADMAP) |
| **MLflow GenAI** | Align with KEP-897 — preferred platform tracker |
| **DeepEval / LLM-as-judge** | Eval Tier-2 in mcp-server ROADMAP |

### 5.4 Model / data ecosystems (compose)

| Project | Value |
|---------|-------|
| **Hugging Face MCP / Hub** | Model/dataset search — Multi-MCP with Hub Catalog |
| **Feast** | Feature store — SDK #239 → later MCP |
| **Kueue / KAI** (Trainer ROADMAP scheduling) | Quota-aware skills (“why won’t my job schedule?”) |

### 5.5 Peer MCP servers (composition profiles)

| Server | Compose when |
|--------|--------------|
| [containers/kubernetes-mcp-server](https://github.com/containers/kubernetes-mcp-server) | Platform-admin profile (pods, events) — KEP non-goal to replace |
| [kubeflow/mcp-apache-spark-history-server](https://github.com/kubeflow/mcp-apache-spark-history-server) | Spark debug profile **today** |
| [kubeflow/docs-agent](https://github.com/kubeflow/docs-agent) MCP | “Ask Kubeflow docs” + “do Trainer actions” |
| HF MCP | Model ID validation / download guidance alongside `pre_flight` |

---

## 6. Opportunity matrix — what to link when

Executable initiative IDs (G*/P*/A*/E*/C*/M*): **[maintainer-strategy §7](../maintainer-strategy-2026.md#7-initiatives-experimental--next-big-thing)**.

| Priority | Opportunity | Depends on | Maps to |
|----------|-------------|------------|---------|
| **P0** | Finish mcp-server 0.1 (Trainer) | RC1 gates | G1, P1 |
| **P0** | Align MLflow with KEP-897 | Community ACK | A7 |
| **P1** | Suite profile: mcp-server + docs-agent + spark-history-mcp | Sibling maintainers | C2 |
| **P1** | Seed Agent Skills + MCP Registry publish | 0.1 tag | C4, G6 |
| **P2** | Hub MCP + Catalog skills | SDK Hub | P2, P8 |
| **P2** | Optimizer MCP + OptimizationJob skills | Trainer+Katib | P3 |
| **P2** | TrainJob progress tools | KEP-2779 | P4 |
| **P3** | Pipelines MCP / Notebooks sidecar / KServe deploy skill | KEP-125, Workspaces | P6, A4, P8 |
| **P4** | A2A + agentgateway / Feast | Phase 4 / sdk#239 | E5–E6 |

---

## 7. Recommended org structure (refined)

```text
                        WG ML Experience (sponsor)
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
   kubeflow/mcp-server     kubeflow/ai-toolkit*     kubeflow/sdk
   (capability MCP)        (skills + profiles +     (API contract)
                            thin catalog)                  │
          │                       │              ┌─────────┴─────────┐
          │                       │              ▼                   ▼
          │                       │         Graduated ops      Experimental
          │                       │         Trainer Katib      MLflow integ.
          │                       │         Hub Pipelines      docs-agent
          │                       │         Spark Notebooks    spark-hist MCP
          ▼                       ▼
   Compose profiles ────────► kubernetes-mcp · HF-mcp · KServe HTTP · agentgateway
```

\*Propose after mcp-server 0.1rc1 — see [ai-toolkit-ecosystem-vision.md](ai-toolkit-ecosystem-vision.md).

**Do not create:** a fourth MCP that re-wraps Trainer; a Kubeflow-owned global MCP marketplace; MLflow fork.

---

## 8. Cross-project collaboration checklist (for you)

### This quarter (with RC1)

- [ ] Brief **ML Experience WG**: AI toolkit vision + ecosystem map (this doc)
- [ ] Sync with **Training WG** on #2839 / #2779 tool impact
- [ ] Sync with **Data WG** (Hub Catalog + Spark History MCP composition)
- [ ] Explicit **KEP-897 alignment** note in mcp-server ARCHITECTURE (MLflow = platform tracker)
- [ ] Reach out to **spark-history-mcp** + **docs-agent** maintainers for joint “Kubeflow MCP suite” blog/profile

### After 0.1

- [ ] Community issue/KEP for `ai-toolkit`
- [ ] Joint milestone with Hub for first Hub MCP tools
- [ ] Pipelines WG: watch KEP-125 / MLflow-12862 before promising Pipelines MCP
- [ ] Notebooks WG: WorkspaceKind prototype for agent IDE

### Industry liaison (lightweight)

- [ ] MCP Registry publish checklist
- [ ] agentgateway reference compose (Phase 4)
- [ ] Track AAIF / CNCF AI TCG docs for “agentic on Kubeflow” messaging
- [ ] Agent Skills validator in ai-toolkit CI

---

## 9. Gaps / risks found in research

| Gap | Risk | Mitigation |
|-----|------|------------|
| Three Experimental MCP-ish projects, no suite story | User confusion | Shared install profiles + website page |
| Hub Catalog vs HF MCP overlap | Duplicate discovery UX | Catalog = Kubeflow-governed federation; HF MCP = raw Hub; skills explain when to use which |
| KServe “agent” means sidecar, not MCP | Naming collision in docs | Prefer “inference agent sidecar” vs “AI agent / MCP” |
| Kale + Notebooks + Pipelines all moving | Skills rot | Version `requires` on skills; CI against SDK tags |
| Arena still exists | Contributors invent parallel CLIs | Point to SDK/MCP in docs; don’t MCP-wrap Arena |
| PROJECTS.md Experimental bar is low | Over-promise | Keep maturity labels honest in toolkit marketing |

---

## 10. One-page “who to talk to”

| Topic | People / channel |
|-------|------------------|
| mcp-server / AI toolkit sponsorship | WG ML Experience chairs · `#kubeflow-ml-experience` |
| Trainer tool schema / LLM runtimes | WG Training · `#kubeflow-trainer` |
| Hub Catalog + Registry MCP | WG Data · Hub call |
| Spark History MCP compose | Spark on K8s call · spark-history-mcp maintainers |
| MLflow platform | KEP-897 authors · `#kubeflow-discuss` / ML Experience |
| Docs RAG MCP | docs-agent maintainers · KEP-867 |
| Notebooks host for agents | WG Notebooks |
| Pipelines MCP timing | WG Pipelines · KEP-125 |
| Inference deploy skills | KServe community (external) |

---

## 11. Summary for strategy

**Kubeflow’s path to a one-stop AI toolkit is not “grow mcp-server until it contains the universe.”**  
It is:

1. **Capability MCPs** wrapping the **SDK** (Trainer → Hub → Optimizer → Pipelines → Spark),  
2. **Sibling MCPs** already in-org (Spark History, docs-agent) composed via profiles,  
3. **Skills + catalog** (propose `ai-toolkit`) for playbooks,  
4. **Platform MLflow** (KEP-897) for experiments/traces,  
5. **Industry protocols** (MCP Registry, Agent Skills, A2A, agentgateway, OTel) for discovery and enterprise.

Use this map when prioritizing issues, WG agenda items, and GSoC/community projects so every skill or MCP module has a **named upstream initiative** to ride — not a parallel invention.

---

## 12. References (entry points)

- [Kubeflow Subprojects / maturity](https://github.com/kubeflow/community/blob/master/subprojects/PROJECTS.md)  
- [WG list](https://github.com/kubeflow/community/blob/master/wg-list.md)  
- [KEP-936 MCP Server](https://github.com/kubeflow/community/tree/master/proposals/936-kubeflow-mcp-server)  
- [KEP-819 Unified SDK](https://github.com/kubeflow/community/tree/master/proposals/819-kubeflow-sdk)  
- [KEP-897 MLflow experiment tracking](https://github.com/kubeflow/community/tree/master/proposals/897-experiment-tracking)  
- [KEP-867 Documentation AI](https://github.com/kubeflow/community/tree/master/proposals/867-kubeflow-documentation-ai)  
- [SDK ROADMAP](https://github.com/kubeflow/sdk/blob/main/ROADMAP.md) · [Trainer ROADMAP](https://github.com/kubeflow/trainer/blob/main/ROADMAP.md)  
- [agentskills.io](https://agentskills.io) · [MCP Registry](https://github.com/modelcontextprotocol/registry) · [agentgateway](https://github.com/agentgateway/agentgateway)
