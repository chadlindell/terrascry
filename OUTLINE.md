# HIRT Whitepaper Outline

This document tracks the status and requirements for each whitepaper section.

## Status Legend

| Status | Meaning |
|--------|---------|
| `RESEARCH` | Needs research before writing |
| `DRAFT` | Written but needs review |
| `REVIEW` | Under review, may need revisions |
| `COMPLETE` | Reviewed and finalized |

---

## Part I: Foundations

### 00 - Index
- **Status:** COMPLETE
- **Purpose:** Document navigation, reader paths, structure overview
- **Lines:** ~250
- **Notes:** Updated with consistent terminology

### 01 - Executive Summary
- **Status:** COMPLETE
- **Purpose:** High-level overview for decision-makers
- **Lines:** ~200
- **Notes:** Covers both MIT-3D and ERT-Lite, applications, cost summary

### 02 - Physics Theory
- **Status:** COMPLETE
- **Purpose:** Theoretical foundation for both sensing modalities
- **Lines:** ~400
- **Notes:** DOIs added to all citations, added "Crosshole Advantage" expansion

### 03 - System Architecture
- **Status:** COMPLETE
- **Purpose:** Overall system design, component relationships
- **Lines:** ~350
- **Notes:** Block diagrams, signal flow, added "Site Suitability Decision Guide"

---

## Part II: Building

### 04 - Bill of Materials
- **Status:** COMPLETE
- **Purpose:** Complete parts list with costs and suppliers
- **Lines:** ~550
- **Notes:** Links to detailed BOMs in /hardware/bom/

### 05 - Mechanical Design
- **Status:** COMPLETE
- **Purpose:** Physical construction - probes, rods, housings
- **Lines:** ~750
- **Notes:** Extensive diagram coverage

### 06 - Electronics and Circuits
- **Status:** COMPLETE
- **Purpose:** Circuit designs for MIT and ERT channels
- **Lines:** ~550
- **Notes:** Datasheet URLs added to references

### 07 - Assembly and Wiring
- **Status:** COMPLETE
- **Purpose:** Step-by-step assembly procedures
- **Lines:** ~750
- **Notes:** Links to expanded hardware/drawings/

### 08 - Testing and Verification
- **Status:** COMPLETE
- **Purpose:** QC procedures, acceptance criteria
- **Lines:** ~350
- **Notes:** Specs aligned with calibration section

### 09 - Calibration
- **Status:** COMPLETE
- **Purpose:** Calibration procedures for field readiness
- **Lines:** ~400
- **Notes:** Frequency-dependent Q factor specs

---

## Part III: Field Operations

### 10 - Field Operations
- **Status:** COMPLETE
- **Purpose:** Deployment procedures, survey patterns
- **Lines:** ~1000
- **Notes:** Comprehensive field guidance, added "Cable Management" section

### 11 - Data Recording
- **Status:** COMPLETE
- **Purpose:** Data formats, acquisition procedures
- **Lines:** ~1300
- **Notes:** Longest section, extensive diagrams

### 12 - Data Interpretation
- **Status:** COMPLETE
- **Purpose:** Analysis methods, visualization
- **Lines:** ~900
- **Notes:** Software tool references with DOIs, added "Signature Catalogue" and "Uncertainty & Limitations"

### 13 - Troubleshooting
- **Status:** COMPLETE
- **Purpose:** Common problems and solutions
- **Lines:** ~550
- **Notes:** EMI terminology clarified

---

## Part IV: Analysis & Applications

### 17 - Application Scenarios
- **Status:** COMPLETE
- **Purpose:** Detailed scenarios for primary applications
- **Lines:** ~750
- **Notes:** Bomb crater, woodland burial, swamp recovery

### 18 - Future Development
- **Status:** COMPLETE
- **Purpose:** Roadmap, potential improvements
- **Lines:** ~700
- **Notes:** Software citations with DOIs added

### 19 - Ethics, Legal, and Safety
- **Status:** COMPLETE
- **Purpose:** UXO protocols, legal considerations
- **Lines:** ~350
- **Notes:** Critical for field deployment, added "Chain of Custody" section

---

## Part V: Appendices

### 14 - Glossary
- **Status:** COMPLETE
- **Purpose:** Technical term definitions
- **Lines:** ~100
- **Notes:** Added DDS, Delta-Sigma ADC, Howland Current Source

### 15 - Quick Reference
- **Status:** COMPLETE
- **Purpose:** Field reference card
- **Lines:** ~120
- **Notes:** Compact format for field use

### 16 - Field Checklists
- **Status:** COMPLETE
- **Purpose:** Pre-deployment, post-deployment checklists
- **Lines:** ~250
- **Notes:** Printable format

---

## Research Integration Backlog

Research documents in `/research/` that may inform whitepaper updates:

| Research File | Potential Section | Status |
|---------------|-------------------|--------|
| `deployment/*.md` | 10-Field Operations, 18-Future Development | Review needed |
| `electronics/electronics-modernization.md` | 06-Electronics, 18-Future Development | Review needed |
| `electronics/manufacturing-cost-report.md` | 04-BOM | Review needed |
| `regulatory/regulatory-compliance-uxo.md` | 19-Ethics-Legal-Safety | Review needed |

---

## Quality Checklist

- [x] All sections render to HTML without warnings
- [x] All sections render to PDF without errors
- [x] Cross-references resolve correctly
- [x] Terminology consistent (MIT-3D, ERT-Lite, probe, crosshole)
- [x] Specifications consistent across sections
- [x] Academic citations have DOIs where available
- [x] Glossary covers key technical terms
- [x] Whitepaper reorganized into five parts (v2.0)
- [ ] Research backlog reviewed and integrated
- [ ] LaTeX export tested for journal submission
