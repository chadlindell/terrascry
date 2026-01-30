# Research Backlog

This file tracks research needs and priorities for the HIRT project.

## Active Research

### High Priority

| Topic | Question | Status | Target Section |
|-------|----------|--------|----------------|
| Probe insertion | Best method for different soil types? | Has research | 10, 18 |
| UXO safety | Regulatory requirements by jurisdiction? | Has research | 19 |
| Electronics | Modern component alternatives? | Has research | 06, 18 |

### Medium Priority

| Topic | Question | Status | Target Section |
|-------|----------|--------|----------------|
| Inversion software | Best open-source options for HIRT data? | Partial | 12, 18 |
| Field validation | Published crosshole case studies? | Needs work | 17 |
| Cost optimization | Where can we reduce BOM cost? | Has research | 04 |

### Low Priority / Future

| Topic | Question | Status | Target Section |
|-------|----------|--------|----------------|
| Automation | Robotic probe deployment? | Has research | 18 |
| Alternative probes | Micro-probe designs? | Partial | 05, 18 |

---

## Completed Research

Research that has been reviewed and integrated into the whitepaper:

| Research File | Integrated Into | Date |
|---------------|-----------------|------|
| (none yet) | | |

---

## Research Directory Structure

```
research/
├── _backlog.md          # This file
├── deployment/          # Probe insertion methods
│   ├── borehole-creation-methods-catalog.md
│   ├── feasibility-*.md
│   ├── portable-*.md
│   ├── probe-insertion-methods-*.md
│   └── uxo-detection-during-push-*.md
├── electronics/         # Circuit modernization
│   ├── electronics-modernization.md
│   └── manufacturing-cost-report.md
├── regulatory/          # Legal/compliance
│   └── regulatory-compliance-uxo.md
└── literature/          # Academic papers, prior art
    └── (empty - add papers here)
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

## Research → Whitepaper Pipeline

```
research/topic.md
    ↓
Review for accuracy and relevance
    ↓
Extract key findings
    ↓
Integrate into whitepaper section
    ↓
Mark as "Completed" in this backlog
    ↓
Optionally archive original research
```
