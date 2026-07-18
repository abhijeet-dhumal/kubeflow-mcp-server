# Vision: Kubeflow AI Toolkit — One-Stop Agent Surface for the Ecosystem

**Audience:** WG ML Experience · mcp-server / SDK maintainers · potential `ai-toolkit` owners  
**Status:** Directional plan (not a KEP; **not** mcp-server 0.1 scope) · **Updated:** 2026-07-18  
**Index:** [design/README](README.md) · **Strategy primer:** [e2e-agent-surface-strategy.md](e2e-agent-surface-strategy.md) · **Ops / graduation IDs:** [maintainer-strategy §7](../maintainer-strategy-2026.md#7-initiatives-experimental--next-big-thing) · **Per-project map:** [kubeflow-ecosystem-ai-toolkit-map.md](kubeflow-ecosystem-ai-toolkit-map.md)

| This doc owns | Defer to |
|---------------|----------|
| Multi-repo layers, skills, thin marketplace, `ai-toolkit` phases | Narrative/KEP path → e2e strategy · inventory → map · RC1/IDs → maintainer-strategy |

---

## 1. Executive answer

**Yes — Kubeflow should become the one-stop *AI-native* surface for its own ecosystem.**  
**No — that surface must not all live inside `kubeflow/mcp-server`.**

| Layer | Job | Where it lives |
|-------|-----|----------------|
| **Platform APIs** | Train, tune, register, pipeline, serve | Operators + `kubeflow/sdk` |
| **Tool plane (MCP)** | Machine-callable tools with auth, confirm, personas | `kubeflow/mcp-server` (+ sibling MCP servers) |
| **Skill plane** | Procedural expertise (`SKILL.md`) — *how* to run workflows | Proposed `kubeflow/ai-toolkit` (skills + recipes) |
| **Discovery plane** | Find / install / trust packages | Official MCP Registry + curated Kubeflow catalog (not a second Smithery) |
| **Composition plane** | Multi-server, multi-tenant, budgets | Gateways (`agentgateway`, LiteLLM) — Phase 4 of mcp-server |

ROADMAP already foreshadows this:

> *higher-level toolkits and marketplaces (for example, a `kubeflow/ai-toolkit` repo) are considered under future scope.*

KEP-936 already draws a **Multi-MCP** boundary: kubeflow-mcp owns Kubeflow domain tools; `kubernetes-mcp-server` owns generic K8s; Hugging Face MCP owns HF Hub — compose, don’t monopolize.

---

## 2. Problem framing

### What “one-stop” means to users

A data scientist or platform engineer using Cursor / Claude Code / a custom agent should be able to:

1. **Discover** the right Kubeflow capabilities (train, HPO, registry, pipelines, Spark, …)
2. **Install** them in one flow (MCP config + optional skills)
3. **Execute** safely (preview/confirm, personas, namespace policy, audit)
4. **Follow recipes** (fine-tune → register → deploy) without inventing the workflow each time
5. **Compose** with non-Kubeflow tools (K8s, HF, observability) without forking Kubeflow

Today they get (1)–(3) partially for **Trainer only**, via `mcp-server`. Recipes (#4) and discovery (#2 at ecosystem scale) are ad hoc. Marketplace (#2 at org scale) does not exist under Kubeflow branding.

### What “one-stop” must *not* mean

- Replacing the SDK or operators  
- Absorbing every MCP in the universe into one process  
- Building a general-purpose agent app store that competes with Smithery / MCP Registry / Hugging Face Agents Hub  
- Shipping a mega-monorepo that blocks 0.1 release

---

## 3. Research synthesis (2026 landscape)

### 3.1 Two complementary primitives

| Primitive | What it is | Token / trust model | Kubeflow fit |
|-----------|------------|---------------------|--------------|
| **MCP tools** | Live RPC to a server (list jobs, `fine_tune`, …) | Schema always or progressive load; needs auth to cluster | **mcp-server** |
| **Agent Skills** ([agentskills.io](https://agentskills.io)) | Folder + `SKILL.md` (+ scripts/refs) teaching *procedures* | Progressive disclosure: name/description always; body on demand | **ai-toolkit skills** |

Industry consensus (2025–2026): skills and MCP are complementary — skills encode *playbooks*; MCP encodes *capabilities*. Registries that mix them without this distinction create noise (duplicate skills, 15+ incompatible directories).

### 3.2 Discovery / marketplace reality

| System | Role | Implication for Kubeflow |
|--------|------|---------------------------|
| **[MCP Registry](https://github.com/modelcontextprotocol/registry)** (official, API freeze v0.1) | App-store API for MCP servers | **Publish** `kubeflow-mcp` here; do not reinvent |
| Smithery / Glama / PulseMCP | UX catalogs, OAuth installers | KEP DESIGN already lists them as marketplace UX — **list**, don’t own |
| Agent Skills marketplaces (SkillsMP, HF Agents Hub, …) | Skill discovery; fragmented | Publish **canonical** Kubeflow skills; mirror to 1–2 hubs max |
| Org gateways (agentgateway, MCP Gateway) | Enterprise control plane | Align with mcp-server Phase 4 — federation stays external |

**Strategic takeaway:** Kubeflow should own **canonical content + trust metadata**, not another global registry protocol.

### 3.3 Kubeflow inventory

Full per-project / WG table: **[ecosystem map §3–§4](kubeflow-ecosystem-ai-toolkit-map.md)**.  
Short form: Trainer MCP ✅ → Hub/Optimizer stubs → Pipelines/Spark later; compose spark-history-mcp + docs-agent + k8s-mcp + HF-mcp. Product story = LLMOps (train → HPO → register → deploy); MCP modules = API.

### 3.4 What KEP-936 already decided (respect it)

- MCP **wraps SDK**, does not replace it  
- **Non-goal:** replace kubectl — use kubernetes-mcp  
- **Multi-MCP ecosystem** table: kubeflow-mcp × kubernetes-mcp × hf-mcp  
- Gateway mode for enterprise  
- Marketplace row: Smithery/Glama for discovery UX  

Any “one-stop” plan that contradicts these needs a **new community KEP**, not a silent scope creep in mcp-server.

---

## 4. Target architecture — three repos, one story

```text
                    ┌─────────────────────────────────────────┐
                    │  Kubeflow AI Toolkit (product story)    │
                    │  website / docs / install meta-CLI      │
                    └─────────────────┬───────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
 ┌─────────────────┐      ┌─────────────────────┐      ┌──────────────────┐
 │ kubeflow/       │      │ kubeflow/ai-toolkit │      │ External MCPs    │
 │ mcp-server      │      │ (NEW — propose)     │      │ k8s / HF / …     │
 │                 │      │                     │      │                  │
 │ • serve tools   │◄────►│ • skills/ (SKILL.md)│      │ composed via     │
 │ • agent (P3)    │      │ • recipes / evals   │      │ client mcp.json  │
 │ • personas      │      │ • skill-registry    │      │ or agentgateway  │
 │ • OTel          │      │   catalog.yaml      │      │                  │
 └────────┬────────┘      │ • IDE bundles       │      └──────────────────┘
          │               │ • marketplace UI    │
          ▼               │   (thin, optional)  │
 ┌─────────────────┐      └─────────────────────┘
 │ kubeflow/sdk    │
 │ + operators     │
 └─────────────────┘
```

### 4.1 `kubeflow/mcp-server` — **capability runtime** (keep focused)

**Owns:** FastMCP serve, client plugins (`trainer`, later `optimizer`, `hub`, …), auth/policy/resilience, OTel, confirm gates, optional local agent.

**Does not own:** Global marketplace UI, third-party skill hosting, generic K8s tools, HF Hub CRUD.

### 4.2 `kubeflow/ai-toolkit` — **skills, recipes, catalog** (propose)

**Owns:**

| Package | Description |
|---------|-------------|
| **Skills** | `agentskills.io`-compliant folders: e.g. `fine-tune-lora`, `openshift-trainer-debug`, `register-model-after-train` |
| **Skill registry (catalog)** | Versioned `catalog.yaml` (or OCI/Git tags) listing official + community skills with provenance |
| **Recipes** | End-to-end workflows that bind skills ↔ MCP tools ↔ SDK examples |
| **Install profiles** | Ready `mcp.json` / Cursor / Claude bundles: “Trainer only”, “LLMOps full”, “Platform admin” |
| **Eval cases** | Golden agent scenarios (can absorb demo `eval/` later) |
| **Thin marketplace** | Static site or Hub page that *indexes* catalog + deep-links to MCP Registry — not a new protocol |

**Does not own:** Cluster credentials, TrainJob CRDs, production MCP process.

### 4.3 Gateways & official registries — **compose, don’t fork**

- Publish mcp-server to **MCP Registry**  
- Optionally list on Smithery/Glama  
- Enterprise: document agentgateway + LiteLLM (already Phase 4)  
- Skills: publish to GitHub + optional HF Agents Hub mirror  

---

## 5. Skill registry — concrete design

### 5.1 What a Kubeflow skill looks like

```text
skills/fine-tune-lora/
├── SKILL.md                 # agentskills.io frontmatter + playbook
├── references/
│   ├── openshift-pip.md     # link/copy of platform-fixes guidance
│   └── confirm-gate.md
├── scripts/                 # optional validators (no cluster secrets)
└── assets/
    └── mcp.profile.json     # suggested tools / persona for this skill
```

Example frontmatter:

```yaml
---
name: fine-tune-lora
description: >
  Run LoRA/QLoRA fine-tuning on Kubeflow Trainer via MCP: pre_flight,
  preview fine_tune, confirm, then monitor logs. Use when the user wants
  to fine-tune an LLM on a Kubeflow cluster.
license: Apache-2.0
compatibility: >
  Requires kubeflow-mcp serve with trainer client; cluster with Trainer v2;
  HF token secret if pulling gated models.
metadata:
  kubeflow.domain: trainer
  kubeflow.min_mcp: "0.1.0"
  mcp.tools: "pre_flight fine_tune get_training_logs wait_for_training"
allowed-tools: pre_flight fine_tune get_training_logs wait_for_training
---
```

### 5.2 Catalog (`skill-registry`)

Start **Git-native** (no new service):

```yaml
# catalog/v1/catalog.yaml
apiVersion: kubeflow.org/ai-toolkit/v1alpha1
kind: SkillCatalog
skills:
  - name: fine-tune-lora
    version: 0.1.0
    path: skills/fine-tune-lora
    domains: [trainer]
    trust: official          # official | community | experimental
    requires:
      mcp_server: ">=0.1.0"
      clients: [trainer]
  - name: openshift-trainer-debug
    version: 0.1.0
    path: skills/openshift-trainer-debug
    domains: [trainer, platform]
    trust: official
```

Later (Phase B): optional OCI artifacts (`oci://…/skills/fine-tune-lora:0.1.0`) and a tiny API that serves the same YAML — still not a global multi-tenant marketplace.

### 5.3 Trust model (avoid HF-style skill spam)

| Tier | Who can publish | Review |
|------|-----------------|--------|
| **official** | Kubeflow OWNERS / WG | Required |
| **community** | Anyone via PR | Maintainer ACK + basic security (no secret exfil scripts) |
| **experimental** | Forks / demos | Clearly labeled; not in default install profile |

CI: validate `SKILL.md` with agentskills reference validator; fail on missing `requires.mcp_server`.

---

## 6. Marketplace — what to build vs buy

| Capability | Build under Kubeflow? | Notes |
|------------|----------------------|-------|
| MCP server discovery (global) | **No** — publish to MCP Registry | Official protocol |
| Skill discovery (Kubeflow-branded) | **Yes — thin catalog** | Git + website page |
| One-click IDE install | **Yes — profiles** in ai-toolkit | `mcp.json` templates |
| OAuth / multi-tenant SaaS store | **No** (initially) | Use enterprise gateway products |
| Community contributions | **Yes — PRs to ai-toolkit** | Same as Kubeflow examples |
| Ratings / social feed | **Defer** | High moderation cost, low early value |

**Positioning line for WG:**

> *Kubeflow AI Toolkit is the curated skill + install catalog for Kubeflow’s MCP servers. Global MCP discovery remains the MCP Registry; we are a first-class publisher and a domain-specific skill authority.*

---

## 7. Product packaging — “one-stop” UX

### Install profiles (user-facing)

| Profile | MCP servers | Skills | Persona default |
|---------|-------------|--------|-----------------|
| **trainer-dev** | kubeflow-mcp (trainer) | fine-tune-lora, debug-oom | `ml-engineer` |
| **llmops** | kubeflow-mcp (trainer+hub+optimizer when ready) | train→register→promote | `ml-engineer` |
| **platform** | kubeflow-mcp + kubernetes-mcp | inspect-runtime, quota-check | `platform-admin` |
| **readonly-observer** | kubeflow-mcp | list-and-monitor | `readonly` |

Meta-CLI (lives in ai-toolkit or thin wrapper):

```bash
# illustrative — not implemented
kf-ai install trainer-dev          # writes ~/.config/Cursor/mcp.json + skills
kf-ai skills list
kf-ai skills add fine-tune-lora
kf-ai doctor                       # checks kubeconfig, trainer CRDs, mcp version
```

### Website / docs story

Single landing (website or ai-toolkit README):

1. What is Kubeflow for agents?  
2. Install profile picker  
3. Skill catalog  
4. Deep links to mcp-server + SDK + operators  
5. Security / personas  

---

## 8. Phased delivery (aligned with mcp-server roadmap)

Do **not** start marketplace work before mcp-server 0.1rc1. Sequence:

### Phase AT0 — Now (mcp-server focused)

- Finish serve RC1 (see [maintainer-strategy-2026.md](../maintainer-strategy-2026.md))  
- Publish intent: short design discussion in WG ML Experience  
- **Do not** create `ai-toolkit` repo until RC1 is tagged  

### Phase AT1 — Foundation (post-0.1, ~1–2 months)

| Deliverable | Owner repo |
|-------------|------------|
| Community discussion / lightweight KEP: “Kubeflow AI Toolkit” | `kubeflow/community` |
| Create `kubeflow/ai-toolkit` (Apache-2.0, OWNERS under WG ML Experience) | community |
| Seed 3–5 **official** skills from existing docs (`platform-fixes`, confirm-gate, pre_flight→fine_tune) | ai-toolkit |
| `catalog.yaml` + CI validator | ai-toolkit |
| Install profile `trainer-dev` (mcp.json + skills) | ai-toolkit |
| Publish mcp-server to **MCP Registry** | mcp-server |

### Phase AT2 — LLMOps widen (with SDK clients)

| Deliverable | Depends on |
|-------------|------------|
| Skills for Hub register/promote | Hub MCP tools (mcp-server Phase 6) |
| Skills for Katib HPO loop | Optimizer MCP tools |
| Profile `llmops` | Above |
| Eval golden cases moved from demo → ai-toolkit | Agent MVP optional |

### Phase AT3 — Composition & enterprise

| Deliverable | Notes |
|-------------|-------|
| Document multi-MCP profiles (kubeflow + k8s + hf) | Matches KEP Multi-MCP |
| Optional thin catalog website | Static; feeds from catalog.yaml |
| Gateway reference (agentgateway) | mcp-server Phase 4 |
| Community skill tier + review SLA | Avoid spam |

### Phase AT4 — Optional “marketplace” productization

Only if AT1–AT3 prove demand:

- Signed skill bundles (cosign)  
- Org-private catalog overlay (GitOps)  
- In-cluster skill cache ConfigMap/CRD — **only if** enterprises ask  

---

## 9. Governance & scope control

| Decision | Recommendation |
|----------|----------------|
| Who owns ai-toolkit? | WG ML Experience (same as mcp-server) |
| Relationship to mcp-server | Sibling; mcp-server remains CNCF-adjacent runtime |
| New KEP needed? | **Yes** for ai-toolkit repo + catalog schema (short); mcp-server stays KEP-936 |
| GSoC / contributors | Skills + catalog = excellent GFI surface; runtime stays harder |
| Branding | “Kubeflow AI Toolkit” = umbrella; “Kubeflow MCP Server” = runtime |

### Anti-goals (write into the KEP)

1. Not a general agent marketplace  
2. Not a replacement for MCP Registry  
3. Not a place to reimplement SDK clients  
4. Not blocking mcp-server releases  
5. Not hosting unreviewed executable skills that exfiltrate kubeconfigs  

---

## 10. Risks

| Risk | Mitigation |
|------|------------|
| Scope explosion kills 0.1 | Hard gate: AT0 = no ai-toolkit until RC1 |
| Skill spam / low quality | Trust tiers + CI + OWNERS for `official/` |
| Duplicate of HF Agents Hub | We are **domain authority** for Kubeflow; mirror optionally |
| mcp-server becomes a kitchen sink | Keep plugin stubs; skills carry workflows |
| Two TracerProviders / obs chaos | Skills document OTel env; runtime owns telemetry |
| Contributor confusion (“where do I PR?”) | CONTRIBUTING matrices in both repos |

---

## 11. Success metrics

| Horizon | Metric |
|---------|--------|
| AT1 | ≥5 official skills; `trainer-dev` profile works on Cursor in &lt;10 minutes |
| AT2 | One documented train→register path using skills + MCP |
| AT3 | Multi-MCP profile used in a public demo / blog |
| Adoption | External PRs to community skills; MCP Registry listing live |
| Maintainer health | mcp-server release cadence unaffected by toolkit PRs |

---

## 12. Maintainer action plan (you)

### Immediate (this month) — **research → consensus, no repo yet**

1. Share this doc in `#kubeflow-ml-experience` + ping WG chairs  
2. Keep mcp-server on [maintainer-strategy-2026.md](../maintainer-strategy-2026.md) RC1 path  
3. Inventory 5 skill candidates from existing docs/eval (demo `improved-skill-v1` is a clue)  
4. Draft **community issue**: “Proposal: kubeflow/ai-toolkit for skills + install profiles”  

### After 0.1rc1

5. Open community KEP (short) for ai-toolkit  
6. Create repo; seed skills + catalog  
7. Publish mcp-server to MCP Registry  
8. Add ROADMAP pointer from mcp-server → ai-toolkit  

### Explicitly defer

- Marketplace SaaS, ratings, payments  
- In-cluster skill operator  
- Absorbing Spark/HF MCP into mcp-server monorepo  

---

## 13. Draft GitHub / community artifacts (when ready)

| When | Artifact | Venue |
|------|----------|-------|
| After RC1 discussion | Issue: “Proposal: Kubeflow AI Toolkit (skills + catalog)” | `kubeflow/community` |
| After ACK | KEP: ai-toolkit scope, catalog schema, trust tiers | `kubeflow/community` |
| Repo exists | Issues: seed skills, catalog CI, trainer-dev profile, MCP Registry publish | `kubeflow/ai-toolkit` + mcp-server |
| Ongoing | Sync ROADMAP “future scope” → link this doc + KEP | mcp-server |

### Suggested community issue title

`Proposal: kubeflow/ai-toolkit — Agent Skills catalog and install profiles for Kubeflow MCP`

### One-paragraph pitch

> KEP-936 delivers the Kubeflow MCP **runtime**. To become a one-stop AI surface for the Kubeflow ecosystem we also need **skills** (procedural playbooks), **install profiles**, and a **curated catalog** — without building a global MCP marketplace. We propose a sibling repo, `kubeflow/ai-toolkit`, that publishes agentskills.io-compatible skills and Git-native catalog metadata, while mcp-server remains the capability plane and the official MCP Registry remains the global discovery API.

---

## 14. Summary diagram — decision cheat sheet

```text
User asks: “I want Kubeflow as my AI toolkit”

  Need live cluster actions?     → mcp-server tools (+ SDK)
  Need a repeatable playbook?    → ai-toolkit skill
  Need to find the server?       → MCP Registry (+ our catalog links)
  Need K8s primitives?           → kubernetes-mcp-server (compose)
  Need HF model search?          → hf-mcp (compose)
  Need org SSO / budgets?        → gateway (Phase 4)
  Need everything in one git PR? → ❌ wrong — split by layer
```

---

## 15. References

- KEP-936 README + DESIGN (Multi-MCP, marketplace row, non-goals)  
- ROADMAP.md — mcp-server phases; ai-toolkit called out as future  
- [agentskills.io](https://agentskills.io) / [agentskills/agentskills](https://github.com/agentskills/agentskills)  
- [modelcontextprotocol/registry](https://github.com/modelcontextprotocol/registry)  
- Landscape notes: fragmented skill registries; skills ≠ MCP; compose gateways  
- Sibling: [kubeflow/mcp-apache-spark-history-server](https://github.com/kubeflow/mcp-apache-spark-history-server) — proof multi-MCP already exists in-org  

---

**Bottom line:** Aim for **one-stop product story**, **multi-repo architecture**. Skills registry + thin marketplace catalog belong in **`kubeflow/ai-toolkit`**. Keep **`mcp-server`** as the trusted capability runtime. Publish to the **official MCP Registry**; don’t rebuild it. Sequence: **RC1 → community KEP → ai-toolkit seed → LLMOps skills as clients land.**
