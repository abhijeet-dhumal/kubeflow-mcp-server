# End-to-End Strategy — Kubeflow Agent Surface

**Audience:** Maintainers, WG ML Experience chairs, anyone pitching or reviewing the AI/MCP direction  
**Status:** Canonical plain-language strategy (not a KEP) · **Updated:** 2026-07-18  
**Index:** [design/README](README.md)

| Need detail on… | Go to |
|-----------------|-------|
| RC1 merge queue, issue batches, initiative IDs (G/P/A/E/C/M) | [maintainer-strategy-2026.md](../maintainer-strategy-2026.md) |
| Skills vs MCP vs marketplace; `ai-toolkit` phases | [ai-toolkit-ecosystem-vision.md](ai-toolkit-ecosystem-vision.md) |
| Per-project WGs, KEPs, industry links | [kubeflow-ecosystem-ai-toolkit-map.md](kubeflow-ecosystem-ai-toolkit-map.md) |
| Runtime phases (serve/agent/clients) | [ROADMAP.md](../../ROADMAP.md) · [KEP-936](https://github.com/kubeflow/community/tree/master/proposals/936-kubeflow-mcp-server) |
| Gaps vs industry (OTel, OWASP MCP, OAuth) | [industry-standards-gap-analysis.md](industry-standards-gap-analysis.md) |

This document is the **strategy primer**: what we are doing, why, in what order, and how (if) to raise a community proposal. It does not replace ops checklists or research tables.

---

## 1. Problem

Kubeflow’s graduated components (Trainer, Hub, Katib, Pipelines, Spark, Notebooks) are strong, but **AI agents cannot use them as one product**. Users must learn SDK + Kubernetes + several UIs. `kubeflow/mcp-server` exists (KEP-936) but is **Experimental** on [PROJECTS.md](https://github.com/kubeflow/community/blob/master/subprojects/PROJECTS.md) — easy to treat as an optional side demo.

**Goal:** Make **“talk to Kubeflow”** the default agent interface to the platform — credible, distributed, composed with siblings — without turning `mcp-server` into a marketplace monorepo.

---

## 2. One-sentence strategy

**0.1 proves the runtime → composition + Hub proves the platform → skills + Incubating prove agents’ default path into Kubeflow.**

---

## 3. Layered architecture (one-stop for users, multi-repo for maintainers)

```text
User / IDE / Agent
        │
        ├─ Skills + install profiles     →  proposed kubeflow/ai-toolkit (playbooks)
        ├─ MCP tools (actions)           →  mcp-server (+ spark-history-mcp, docs-agent)
        ├─ Python APIs                   →  kubeflow/sdk
        └─ Cluster reality               →  Trainer, Hub, Katib, Pipelines, KServe, …
```

| Layer | Job | Home |
|-------|-----|------|
| **Platform APIs** | Train, tune, register, pipeline, serve | Operators + SDK |
| **Tool plane (MCP)** | Machine-callable actions; auth, confirm, personas | `kubeflow/mcp-server` (+ sibling MCPs) |
| **Skill plane** | Procedural expertise (`SKILL.md`) | Proposed `kubeflow/ai-toolkit` |
| **Discovery** | Find/install/trust | Official [MCP Registry](https://github.com/modelcontextprotocol/registry) + thin Kubeflow catalog |
| **Composition / enterprise** | Multi-server, tenancy, budgets | Gateways ([agentgateway](https://github.com/agentgateway/agentgateway), LiteLLM) — mcp-server Phase 4 |

**Rule:** One-stop **product story**. Multi-repo **ownership**. Compose; do not absorb kubernetes-mcp, HF-mcp, docs-agent, or spark-history-mcp into this monorepo.

---

## 4. Three working docs — three jobs

| Document | Job | Answers |
|----------|-----|---------|
| [maintainer-strategy-2026.md](../maintainer-strategy-2026.md) | **Ops** — how maintainers run mcp-server this quarter | What merges for 0.1? Which initiative IDs graduate us? |
| [ai-toolkit-ecosystem-vision.md](ai-toolkit-ecosystem-vision.md) | **Architecture** — product shape | Skills vs MCP vs marketplace; who owns what? |
| [kubeflow-ecosystem-ai-toolkit-map.md](kubeflow-ecosystem-ai-toolkit-map.md) | **Alliances** — WG/KEP inventory | Which upstream initiative do we ride? What must we not duplicate? |

Mnemonic: **ops → architecture → alliances**. This file sits above them as the narrative glue.

---

## 5. Five moves (execution order)

### Move 1 — Ship a real serve product (0.1 / RC1)

Deliver `kubeflow-mcp serve` with Trainer tools, core platform (auth, policy, resilience, health, security), server OTel, SECURITY.md, release path (PyPI + image), and credible unit tests.

**Out of 0.1:** agent CLI, Hub/Optimizer/Spark implementations, Kind e2e as a hard gate, benchmarks as a hard gate, skills repo, marketplace SaaS.

Until Move 1 lands, nothing else is credible. Details: [maintainer-strategy §3](../maintainer-strategy-2026.md#3-rc1--serve-release).

### Move 2 — Graduate maturity (Experimental → Incubating)

Promotion is evidence-based ([maturity requirements](https://github.com/kubeflow/community/blob/master/subprojects/maturity_requirements.md)):

- Installable release (PyPI + GHCR)
- CI/coverage/conformance credibility
- SECURITY + OWNERS depth
- Website / subprojects visibility
- MCP Registry listing
- Named adopters
- Path into **community-distribution**

Initiative IDs: **G1–G7** in [maintainer-strategy §7.1](../maintainer-strategy-2026.md#71-maturity--distribution-make-it-official).

### Move 3 — Own the LLMOps tool plane (still mcp-server)

After Trainer: add **Hub** (and/or **Optimizer**) MCP tools wrapping SDK clients; later Pipelines when [KEP-125](https://github.com/kubeflow/sdk/tree/main/proposals/125-kfp-client) matures. Prefer composing [spark-history-mcp](https://github.com/kubeflow/mcp-apache-spark-history-server) over blocking on SparkClient MCP (#5).

Ride Trainer [#2839](https://github.com/kubeflow/trainer/issues/2839) (extensible trainers) and [KEP-2779](https://github.com/kubeflow/trainer/tree/main/proposals/2779-trainjob-progress) (live progress). Target story: **fine-tune → register (Hub) → deploy (KServe)**.

Initiative IDs: **P1–P8** in [maintainer-strategy §7.2](../maintainer-strategy-2026.md#72-product-depth-own-the-llmops-tool-plane).

### Move 4 — Compose a suite (don’t monopolize)

Document install profiles so Experimental siblings become one user-facing suite:

| Profile | Contents |
|---------|----------|
| `trainer-dev` | mcp-server (trainer) |
| `kubeflow-ai-suite` | mcp-server + docs-agent + spark-history-mcp |
| `platform-admin` | suite + [kubernetes-mcp-server](https://github.com/containers/kubernetes-mcp-server) |

Initiative IDs: **C1–C6** in [maintainer-strategy §7.5](../maintainer-strategy-2026.md#75-ecosystem-composition-suite-not-silo).

### Move 5 — Skills + thin catalog (`ai-toolkit`, after RC1)

- **MCP tools** = what the agent *can* do (live RPC).  
- **Agent Skills** ([agentskills.io](https://agentskills.io)) = *how* to do it well (progressive disclosure).  
- **Marketplace** = curated Kubeflow catalog + publish to MCP Registry — **not** a second Smithery.

Propose `kubeflow/ai-toolkit` only after 0.1rc1 is tagged. Phases: [ai-toolkit vision §8](ai-toolkit-ecosystem-vision.md#8-phased-delivery-aligned-with-mcp-server-roadmap).

---

## 6. Cross-cutting decisions (precise)

### 6.1 MLflow

Platform experiment tracking is **[KEP-897](https://github.com/kubeflow/community/tree/master/proposals/897-experiment-tracking)** (`kubeflow/mlflow-integration`).  

- Serve: OpenTelemetry first; optional collector → MLflow.  
- Agent/eval: optional `MLFLOW_TRACKING_URI` sinks, same platform story.  
- **Do not** invent TrainJob experiment-tracking MCP tools in 0.1 or a competing tracker inside mcp-server.

### 6.2 KEP-936 boundaries (respect)

- MCP **wraps SDK**; does not replace it.  
- Does **not** replace kubectl → use kubernetes-mcp.  
- **Multi-MCP** composition (kubeflow × k8s × HF).  
- Gateway mode for enterprise.  
- Smithery/Glama = discovery UX listings, not owned infrastructure.

Any plan that contradicts these needs a **new community KEP**, not silent scope creep.

### 6.3 Demo branches

`demo/oss-summit-2026` (agent, MLflow middleware, eval, compose) is a **quarry**: carve into Phase 3+ PRs. It is not an upstream merge target and must not block RC1.

### 6.4 Explicit non-goals

- Replacing SDK, kubectl, or KServe  
- Global MCP/skills marketplace SaaS  
- Merging sibling MCP repos into mcp-server  
- Blocking releases on multi-framework agent adapters  
- Marketing 0.1 as Graduated / production-Stable (ROADMAP: Stable needs Phases 2 **and** 3)

---

## 7. What “next big thing” vs “optional experiment” looks like

| Optional experiment | Strategy success |
|---------------------|------------------|
| Cool demo on a fork/branch | PyPI + image + MCP Registry + website page |
| Trainer-only forever | Trainer → Hub (or Optimizer) → train→register→deploy story |
| Isolated Experimental repo | Suite profiles + joint WG demos |
| “Try if you like agents” | In community-distribution; **Incubating** on PROJECTS.md |
| Scope-creep monorepo | Clear layer ownership + community ACK |
| No adopters | ≥3 named adopters / case studies |

Full success bar and initiative tables: [maintainer-strategy §7.7](../maintainer-strategy-2026.md#77-what-next-big-thing-looks-like-success-bar).

---

## 8. Community proposal — whether, what, when

### 8.1 Recommendation

**Yes — raise a community proposal for the end-to-end agent-surface strategy**, after (or tightly coupled with) a credible 0.1rc1 path.

| Already covered | Still missing at community level |
|-----------------|----------------------------------|
| KEP-936: mcp-server as SDK-wrapping MCP runtime | Suite composition, skills/`ai-toolkit`, graduation path, KEP-897 alignment, multi-WG dependency map |

Do **not** paste all working markdown into a mega-KEP on day one. Do **not** block 0.1 on the new KEP merging.

### 8.2 Recommended sequence

| Step | Artifact | When | Owner venue |
|------|----------|------|-------------|
| **S0** | WG ML Experience agenda item / one-pager (this doc §1–§5 + §7) | Now / during RC1 | `#kubeflow-ml-experience` |
| **S1** | Community **discussion issue** | After 0.1rc1 tagged (or when Must checklist is clearly on track) | `kubeflow/community` |
| **S2** | Short **KEP** (design + non-goals + phases) | After chairs ACK on S1 | `kubeflow/community/proposals/` |
| **S3** | Optional: create `kubeflow/ai-toolkit` | After S2 ACK + seed skills ready | New repo under WG ML Experience |
| **S4** | mcp-server **Incubating** nomination | When G1–G7 evidence exists | PROJECTS.md + maturity process |

⚠️ Chairs may prefer a **KEP-936 follow-on / amendment** instead of a new number. Either is fine if ownership (WG ML Experience) and non-goals are explicit.

### 8.3 What the community KEP must claim

1. **mcp-server** remains the Kubeflow **capability MCP runtime** (continues KEP-936).  
2. **Sibling MCPs** (docs-agent, spark-history-mcp, and external k8s/HF MCPs) are composed via **install profiles**, not merged.  
3. Optional sibling **`ai-toolkit`** (or equivalent) owns **Agent Skills + catalog + profiles**.  
4. **Graduation path** for mcp-server (Experimental → Incubating) with measurable criteria.  
5. **MLflow** aligns with KEP-897; mcp-server does not own platform experiment tracking.  
6. **Non-goals:** global marketplace SaaS; replace SDK/kubectl; absorb all MCP servers; block 0.1 on skills.  
7. **Dependencies:** Trainer (progress/extensible trainers), Hub Catalog, SDK clients, Pipelines KEP-125 timing, Notebooks Workspaces (later).

### 8.4 What the KEP must not claim

- Delivery of Hub + Optimizer + Pipelines + agent + marketplace in one release  
- Commitment to a specific marketplace product or SaaS  
- Replacement of KServe “agent” sidecars with MCP  
- That 0.1 is Graduated / production-Stable  

### 8.5 Suggested titles

**Discussion issue:**  
`Proposal: End-to-end Kubeflow agent surface (MCP suite, skills catalog, graduation path)`

**KEP (working title):**  
`KEP: Kubeflow AI Agent Surface — MCP runtime, suite composition, and skills catalog`

### 8.6 Pitch paragraph (reusable)

> KEP-936 delivers the Kubeflow MCP **runtime**. To make agents a first-class way to use Kubeflow—not an Experimental side project—we need an explicit end-to-end strategy: graduate mcp-server with a real 0.1 release and distribution path; extend the tool plane across SDK clients (Trainer → Hub/Optimizer → …); **compose** sibling MCPs (docs-agent, spark-history) via install profiles; align experiment tracking with KEP-897; and optionally add an `ai-toolkit` skills/catalog repo using the Agent Skills standard. Global MCP discovery stays the official MCP Registry. We are asking WG ML Experience to ACK this layering and graduation path so multi-WG work stays coordinated.

### 8.7 Draft discussion-issue body (for copy-paste when ready)

```markdown
### Summary

Propose an end-to-end strategy for Kubeflow’s AI agent surface so mcp-server
graduates from Experimental “optional demo” to the default agent interface,
without turning one repo into a marketplace monorepo.

### Motivation

- KEP-936 covers the MCP runtime wrapping the SDK.
- Users need a coherent path: discover → install → train → register → deploy,
  plus docs/Spark debug via sibling MCPs.
- Skills (agentskills.io) and install profiles are not specified in KEP-936.
- Maturity/distribution (Incubating, community-distribution, MCP Registry)
  need an explicit community ACK.

### Proposal (layers)

1. **Capability runtime** — kubeflow/mcp-server (KEP-936 continues).
2. **Suite composition** — install profiles with docs-agent, spark-history-mcp,
   optional kubernetes-mcp / HF-mcp.
3. **Skills + catalog** — proposed kubeflow/ai-toolkit (post-0.1).
4. **Discovery** — publish to MCP Registry; thin Kubeflow catalog; no new global registry.
5. **Tracking** — align with KEP-897 MLflow; OTel-first in mcp-server.
6. **Graduation** — Evidence-based Experimental → Incubating path.

### Non-goals

- Global marketplace SaaS
- Absorb sibling MCP repos
- Replace SDK / kubectl / KServe
- Block mcp-server 0.1 on skills or Hub/Optimizer completion

### Phasing

- **Now:** mcp-server 0.1rc1 (serve + Trainer).
- **Next:** Registry publish, suite profiles, Hub or Optimizer tools, agent MVP.
- **Then:** ai-toolkit KEP/repo, enterprise gateway (agentgateway), Incubating nomination.

### References

- Working docs in kubeflow/mcp-server:
  - docs/design/e2e-agent-surface-strategy.md
  - docs/maintainer-strategy-2026.md
  - docs/design/ai-toolkit-ecosystem-vision.md
  - docs/design/kubeflow-ecosystem-ai-toolkit-map.md
- KEP-936, KEP-897, KEP-867, SDK KEP-125, Trainer KEP-2779 / #2839

### Ask

ACK from WG ML Experience chairs on layering + non-goals, and guidance on
whether this should be a new KEP or a KEP-936 follow-on.
```

---

## 9. Timeline (strategy view)

```text
W0–W2   Move 1: RC1 / 0.1 serve release + G1/G6 groundwork
        S0: WG brief (optional in parallel)

W2–W3   Move 2 start: adopters, distribution, website, Registry
        S1: community discussion issue

W3+     Move 3–4: Hub/Optimizer tools, suite profiles, agent MVP
        S2: short KEP after ACK

Later   Move 5: ai-toolkit repo (S3)
        Move 2 complete: Incubating nomination (S4)
        Phase 4 enterprise (OIDC, Helm, agentgateway, A2A)
```

Ops waves: [maintainer-strategy §3](../maintainer-strategy-2026.md#3-rc1--serve-release).

---

## 10. Who decides what

| Decision | Forum |
|----------|-------|
| mcp-server RC1 scope / merges | mcp-server OWNERS (this repo) |
| Agent-surface layering / ai-toolkit / suite story | **WG ML Experience** |
| Trainer tool schema / progress CRD | WG Training |
| Hub Catalog / Registry MCP timing | WG Data (Hub) |
| Pipelines MCP timing | WG Pipelines + SDK KEP-125 |
| MLflow platform contract | KEP-897 / ML Experience |
| PROJECTS.md maturity change | Community maturity process + sponsoring WG |

Contact table: [ecosystem map §10](kubeflow-ecosystem-ai-toolkit-map.md#10-one-page-who-to-talk-to).

---

## 11. Maintainer reading order

1. This file (§1–§5, §8) — strategy  
2. [maintainer-strategy](../maintainer-strategy-2026.md) §3 + §7 + §8 — execute  
3. [ecosystem map](kubeflow-ecosystem-ai-toolkit-map.md) — WG outreach  
4. [ai-toolkit vision](ai-toolkit-ecosystem-vision.md) — when proposing skills repo / KEP  

Community proposal checklist: complete **S0** (WG brief) anytime; open **S1** issue only when 0.1rc1 is tagged or Must checklist is clearly on track — draft body in [§8.7](#87-draft-discussion-issue-body-for-copy-paste-when-ready).

---

## 12. Document control

| Role | Responsibility |
|------|----------------|
| mcp-server approvers | Keep this aligned with ROADMAP and RC1 reality |
| WG ML Experience | ACK community proposal path (§8) |
| Authors | Prefer links over duplicating initiative tables |

**Update when:** 0.1 ships, community issue/KEP opens, Incubating nomination starts, or layering decisions change.

**Not in this doc:** PR-level checklists, full WG research tables, skill YAML schemas (see linked docs).
