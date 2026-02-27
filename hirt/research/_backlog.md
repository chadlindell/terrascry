# Research Backlog

This file tracks research needs and priorities for the HIRT project.

## Active Research

### High Priority

*All high priority research has been reviewed and integrated. See Completed Research below.*

### Medium Priority

| Topic | Question | Status | Research File(s) | Target Section |
|-------|----------|--------|-------------------|----------------|
| Inversion software | Best open-source options for HIRT data? | Partial | (needs dedicated research) | inversion, roadmap |
| Field validation | Published crosshole case studies? | Needs work | (needs dedicated research) | scenarios |
| Alternative probes | Micro-probe designs? | Partial | (needs dedicated research) | mechanical, roadmap |

### Low Priority / Future

*No active low priority items. All previous items resolved -- see Completed Research and Reviewed (No Integration Needed) below.*

---

## Completed Research

Research that has been reviewed and integrated into the Technical Manual:

| Research File | Integrated Into | Date |
|---------------|-----------------|------|
| `electronics/2026-01-29-electronics-modernization.md` | `docs/build-guide/electronics.qmd` (modernization roadmap section) | 2026-02-12 |
| `electronics/2026-01-29-manufacturing-cost-report.md` | `docs/build-guide/bill-of-materials.qmd` (manufacturing cost analysis) | 2026-02-12 |
| `deployment/probe-insertion-methods-summary.md` | `docs/field-guide/progressive-deployment.qmd` (insertion methods table) | 2026-02-12 |
| `regulatory/2026-01-29-regulatory-compliance-uxo.md` | `docs/getting-started/safety.qmd` (HERO analysis, regulatory framework, permit-to-dig) | 2026-02-12 |
| `deployment/probe-insertion-methods-research-B.md` | `docs/field-guide/deployment.qmd` (bentonite prohibition, polymer fluid guidance, borehole stability data) | 2026-02-12 |
| `deployment/portable-hydraulic-push-consolidated-research.md` | `docs/field-guide/deployment.qmd` (push force data, soil-specific penetration rates, UXO detection thresholds) | 2026-02-12 |
| `deployment/borehole-creation-methods-catalog.md` | `docs/field-guide/deployment.qmd` (borehole methods comparison table) | 2026-02-12 |
| `deployment/probe-insertion-methods-research-A.md` | `docs/field-guide/deployment.qmd` (depth range data incorporated into methods table) | 2026-02-12 |
| `deployment/uxo-detection-during-push-research-A.md` | `docs/getting-started/safety.qmd` (UXO standoff distances, abort criteria, magnetometer detection ranges) | 2026-02-12 |
| `deployment/uxo-detection-during-push-research-B.md` | `docs/field-guide/deployment.qmd` (pre-push UXO assessment checklist, real-time monitoring guidance) | 2026-02-12 |
| `deployment/feasibility-robotic-deployment.md` | `docs/developer/roadmap.qmd` (autonomous deployment section, TRL 3-4 assessment) | 2026-02-12 |

---

## Reviewed -- No Integration Needed

Research documents reviewed during Phase B that contain background/speculative content with no additional actionable data beyond what is already integrated:

| Research File | Rationale | Date |
|---------------|-----------|------|
| `deployment/feasibility-biomimetic-root-growth.md` | Speculative concept (TRL 1); no actionable data for current manual | 2026-02-12 |
| `deployment/feasibility-hydraulic-push.md` | Superseded by consolidated hydraulic push research (already integrated) | 2026-02-12 |
| `deployment/feasibility-water-jet.md` | Core findings already captured in Tier 1 insertion method and borehole methods table | 2026-02-12 |
| `deployment/portable-reaction-force-research-A.md` | Anchor system data consolidated into portable-hydraulic-push-consolidated-research.md (already integrated) | 2026-02-12 |
| `deployment/portable-reaction-force-research-B.md` | Duplicate scope with research-A; key findings already in consolidated document | 2026-02-12 |
| `literature/comparable-projects-catalog.md` | Background context only; no specific data points needed in manual sections | 2026-02-12 |
| `literature/documentation-best-practices.md` | Internal process guidance; informed CLAUDE.md style guide, not manual content | 2026-02-12 |

---

## Research Directory Structure

```
research/
├── _backlog.md          # This file
├── deployment/          # Probe insertion methods (13 documents)
│   ├── borehole-creation-methods-catalog.md
│   ├── feasibility-biomimetic-root-growth.md
│   ├── feasibility-hydraulic-push.md
│   ├── feasibility-robotic-deployment.md
│   ├── feasibility-water-jet.md
│   ├── portable-hydraulic-push-consolidated-research.md
│   ├── portable-reaction-force-research-A.md
│   ├── portable-reaction-force-research-B.md
│   ├── probe-insertion-methods-research-A.md
│   ├── probe-insertion-methods-research-B.md
│   ├── probe-insertion-methods-summary.md
│   ├── uxo-detection-during-push-research-A.md
│   └── uxo-detection-during-push-research-B.md
├── electronics/         # Circuit modernization (2 documents)
│   ├── 2026-01-29-electronics-modernization.md
│   └── 2026-01-29-manufacturing-cost-report.md
├── regulatory/          # Legal/compliance (1 document)
│   └── 2026-01-29-regulatory-compliance-uxo.md
└── literature/          # Academic papers, prior art (2 documents)
    ├── comparable-projects-catalog.md
    └── documentation-best-practices.md
```

---

## How to Add Research

1. Create a markdown file in the appropriate subdirectory
2. Use naming convention: `YYYY-MM-DD-topic-name.md` or `topic-name.md`
3. Include:
   - Summary of findings
   - Sources/references
   - Relevance to HIRT
   - Recommended actions
4. Update this backlog with the new research item

---

## Research -> Technical Manual Pipeline

```
research/topic.md
    |
Review for accuracy and relevance
    |
Extract key findings
    |
Integrate into Technical Manual section
    |
Mark as "Completed" in this backlog
    |
Optionally archive original research
```
