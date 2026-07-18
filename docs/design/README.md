# Design docs — Kubeflow MCP Server

Working designs and research. **Not KEPs** unless promoted to `kubeflow/community/proposals/`.

## Start here

| Doc | Purpose | Audience |
|-----|---------|----------|
| **[../ACTION-ITEMS.md](../ACTION-ITEMS.md)** | **Master checklist** — all waves, security/obs/graduation | Execute from here |
| **[e2e-agent-surface-strategy.md](e2e-agent-surface-strategy.md)** | **Canonical strategy primer** — layers, five moves, community proposal path | Everyone (read first) |
| [../maintainer-strategy-2026.md](../maintainer-strategy-2026.md) | RC1 ops + graduation initiative IDs (G/P/A/E/C/M) | Maintainers |
| [ai-toolkit-ecosystem-vision.md](ai-toolkit-ecosystem-vision.md) | Multi-repo shape: MCP vs skills vs marketplace | WG ML Experience |
| [kubeflow-ecosystem-ai-toolkit-map.md](kubeflow-ecosystem-ai-toolkit-map.md) | Per-project / WG / industry linkages | Cross-WG planning |
| [industry-standards-gap-analysis.md](industry-standards-gap-analysis.md) | Gaps vs OTel, OWASP MCP Top 10, OAuth 2.1, CNCF agentic | Security / observability owners |

## Agent runtime (Phase 3)

| Doc | Purpose |
|-----|---------|
| [agent-provider-architecture.md](agent-provider-architecture.md) | Pluggable providers |
| [litellm-agent-redesign.md](litellm-agent-redesign.md) | LiteLLM agent path |
| [production-gaps-lld.md](production-gaps-lld.md) | Production gap LLD (demo-era) |

## Architecture assets

| Asset | Purpose |
|-------|---------|
| [architecture.mmd](architecture.mmd) / [architecture-diagram.html](architecture-diagram.html) | Diagram sources |
| [../assets/architecture.svg](../assets/architecture.svg) | Rendered target architecture (see also root [ARCHITECTURE.md](../../ARCHITECTURE.md)) |

## How to use

1. **Understand the strategy** → [e2e-agent-surface-strategy.md](e2e-agent-surface-strategy.md).  
2. **Shipping 0.1** → [maintainer-strategy](../maintainer-strategy-2026.md).  
3. **WG / community proposal** → e2e strategy §8 + vision + ecosystem map.  
4. **Agent implementation** → agent-provider + litellm redesign; carve from demo per maintainer-strategy.  
5. **Promote to KEP** → graduate content into `kubeflow/community` (do not paste all files raw).

Keep docs short: link rather than duplicate. Update `Last updated` when materially changing recommendations.
