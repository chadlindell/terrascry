# HIRT Technical Manual Outline

This document tracks the status and requirements for each documentation section.

## Status Legend

| Status | Meaning |
|--------|---------|
| `NEW` | Newly created, needs review |
| `DRAFT` | Written but needs review |
| `REVIEW` | Under review, may need revisions |
| `COMPLETE` | Reviewed and finalized |
| `MIGRATED` | Migrated from old whitepaper structure |

---

## Getting Started

### index.qmd - Task Map
- **Status:** NEW
- **Purpose:** Cross-cutting navigation, user-type routing
- **Notes:** Links field ops to build troubleshooting, builders to acceptance tests

### overview.qmd
- **Status:** MIGRATED
- **Purpose:** What is HIRT, dual-channel approach, use cases
- **Source:** 01-executive-summary.qmd
- **Notes:** Covers both MIT-3D and ERT-Lite, applications, cost summary

### quick-start.qmd
- **Status:** NEW
- **Purpose:** Rapid deployment guide for experienced users
- **Notes:** Minimal setup checklist, first survey in 2-3 hours

### safety.qmd
- **Status:** NEW (split from 19-ethics-legal-safety)
- **Purpose:** UXO safety protocols, HERO compliance
- **Notes:** Critical for field deployment

---

## Field Guide

### deployment.qmd
- **Status:** MIGRATED
- **Purpose:** Standard deployment procedures
- **Source:** 10-field-operations.qmd
- **Notes:** Comprehensive field guidance, cable management

### progressive-deployment.qmd
- **Status:** NEW
- **Purpose:** Advanced four-phase deployment workflow
- **Notes:** Shallow to deep, data-driven decisions, UXO protocols

### data-acquisition.qmd
- **Status:** NEW (split from 11-data-recording)
- **Purpose:** Practical acquisition procedures
- **Notes:** TX-RX matrix, coordinate systems

### interpretation.qmd
- **Status:** NEW (split from 12-data-interpretation)
- **Purpose:** Practical interpretation guidance
- **Notes:** Anomaly identification, quality control

### scenarios.qmd
- **Status:** MIGRATED
- **Purpose:** Detailed application scenarios
- **Source:** 17-application-scenarios.qmd
- **Notes:** Bomb crater, woodland burial, swamp recovery

### troubleshooting.qmd
- **Status:** MIGRATED (needs hybrid rewrite)
- **Purpose:** Common problems and solutions
- **Source:** 13-troubleshooting.qmd
- **Notes:** Needs hybrid format (tables + narrative)

---

## Build Guide

### bill-of-materials.qmd
- **Status:** MIGRATED
- **Purpose:** Complete parts list with costs
- **Source:** 04-bill-of-materials.qmd
- **Notes:** Links to detailed BOMs in /hardware/bom/

### mechanical.qmd
- **Status:** MIGRATED
- **Purpose:** Physical construction details
- **Source:** 05-mechanical-design.qmd
- **Notes:** Extensive diagram coverage

### electronics.qmd
- **Status:** MIGRATED
- **Purpose:** Circuit designs for MIT and ERT
- **Source:** 06-electronics-circuits.qmd
- **Notes:** Datasheet URLs in references

### assembly.qmd
- **Status:** MIGRATED
- **Purpose:** Step-by-step assembly procedures
- **Source:** 07-assembly-wiring.qmd
- **Notes:** Links to hardware/drawings/

### testing.qmd
- **Status:** MIGRATED
- **Purpose:** QC procedures, acceptance criteria
- **Source:** 08-testing-verification.qmd
- **Notes:** Specs aligned with calibration

### calibration.qmd
- **Status:** MIGRATED
- **Purpose:** Calibration for field readiness
- **Source:** 09-calibration.qmd
- **Notes:** Frequency-dependent Q factor specs

### validation.qmd
- **Status:** NEW
- **Purpose:** HardwareX compliance validation
- **Notes:** Placeholder with (Target) specs

---

## Theory

### physics.qmd
- **Status:** MIGRATED
- **Purpose:** Theoretical foundation
- **Source:** 02-physics-theory.qmd
- **Notes:** DOIs on all citations

### architecture.qmd
- **Status:** MIGRATED
- **Purpose:** System design, component relationships
- **Source:** 03-system-architecture.qmd
- **Notes:** Block diagrams, signal flow

### sensor-modalities.qmd
- **Status:** NEW
- **Purpose:** Sensor maturity framework
- **Notes:** Supported/Recommended/Future labeling

### inversion.qmd
- **Status:** NEW (split from 12-data-interpretation)
- **Purpose:** Mathematical basis, algorithms
- **Notes:** pyGIMLi, SimPEG references

### uncertainty.qmd
- **Status:** NEW
- **Purpose:** Limitations, non-uniqueness
- **Notes:** "What No Anomaly Actually Means"

---

## Developer Guide

### contributing.qmd
- **Status:** NEW
- **Purpose:** Contribution guidelines
- **Notes:** Development workflow, code standards

### firmware.qmd
- **Status:** NEW
- **Purpose:** Firmware architecture
- **Notes:** Placeholder pending firmware maturity

### data-formats.qmd
- **Status:** NEW (split from 11-data-recording)
- **Purpose:** CSV format specifications
- **Notes:** Compatible with pyGIMLi/SimPEG

### roadmap.qmd
- **Status:** MIGRATED
- **Purpose:** Future development plans
- **Source:** 18-future-development.qmd
- **Notes:** Software citations with DOIs

---

## Appendices

### glossary.qmd
- **Status:** MIGRATED
- **Purpose:** Technical term definitions
- **Source:** 14-glossary.qmd
- **Notes:** Added DDS, Delta-Sigma ADC, Howland Current Source

### quick-reference.qmd
- **Status:** MIGRATED
- **Purpose:** Field reference card
- **Source:** 15-quick-reference.qmd
- **Notes:** Compact format for field use

### checklists.qmd
- **Status:** MIGRATED
- **Purpose:** Pre/post-deployment checklists
- **Source:** 16-field-checklists.qmd
- **Notes:** Printable format

### regulations.qmd
- **Status:** NEW (split from 19-ethics-legal-safety)
- **Purpose:** Legal requirements by jurisdiction
- **Notes:** Permits, chain of custody

---

## Quality Checklist

- [x] All sections render to HTML without warnings
- [x] All sections render to PDF without errors
- [x] Cross-references resolve correctly
- [x] Terminology consistent (MIT-3D, ERT-Lite, probe, crosshole)
- [x] Specifications consistent across sections
- [x] Academic citations have DOIs where available
- [x] Glossary covers key technical terms
- [x] Documentation restructured to onion model (v3.0)
- [x] GitHub Actions CI workflow created
- [ ] Troubleshooting hybrid rewrite (keep tables + add narrative)
- [ ] Performance claim qualifications throughout
- [ ] Progressive deployment diagrams (5-6 new)
- [ ] Research backlog reviewed and integrated

---

## Pending Tasks

1. **Troubleshooting hybrid rewrite** - Apply hybrid format (keep diagnostic tables, add narrative context sections)
2. **Performance qualifications** - Add (Measured)/(Modeled)/(Target) labels to all specs
3. **Progressive deployment diagrams** - Create workflow flowchart, depth extension decision tree, etc.
4. **Sensor modality diagrams** - Multi-modal fusion illustration
