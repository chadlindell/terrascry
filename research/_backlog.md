# Research Backlog

This file tracks research needs and priorities for the HIRT project.

## Active Research

### High Priority

| Topic | Question | Status | Research File(s) | Target Section |
|-------|----------|--------|-------------------|----------------|
| Probe insertion | Best method for different soil types? | Has research | `deployment/probe-insertion-methods-summary.md` + 12 related files | deployment, roadmap |
| UXO safety | Regulatory requirements by jurisdiction? | Has research | `regulatory/2026-01-29-regulatory-compliance-uxo.md` | safety, regulations |
| Electronics | Modern component alternatives? | Has research | `electronics/2026-01-29-electronics-modernization.md` | electronics, roadmap |

### Medium Priority

| Topic | Question | Status | Research File(s) | Target Section |
|-------|----------|--------|-------------------|----------------|
| Inversion software | Best open-source options for HIRT data? | Partial | (needs dedicated research) | inversion, roadmap |
| Field validation | Published crosshole case studies? | Needs work | (needs dedicated research) | scenarios |
| Cost optimization | Where can we reduce BOM cost? | Has research | `electronics/2026-01-29-manufacturing-cost-report.md` | bill-of-materials |

### Low Priority / Future

| Topic | Question | Status | Research File(s) | Target Section |
|-------|----------|--------|-------------------|----------------|
| Automation | Robotic probe deployment? | Has research | `deployment/feasibility-robotic-deployment.md` | roadmap |
| Alternative probes | Micro-probe designs? | Partial | (needs dedicated research) | mechanical, roadmap |
| Comparable projects | What similar systems exist? | Has research | `literature/comparable-projects-catalog.md` | overview |

---

## Completed Research

Research that has been reviewed and integrated into the Technical Manual:

| Research File | Integrated Into | Date |
|---------------|-----------------|------|
| (none yet) | | |

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
