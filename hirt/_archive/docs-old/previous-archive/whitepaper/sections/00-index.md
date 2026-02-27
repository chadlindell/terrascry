# HIRT Whitepaper - Documentation Index

## Hybrid Impedance-Resistivity Tomography System

**Version:** 2.0
**Date:** January 2026
**Status:** Consolidated Documentation (19 sections in 5 parts)

---

## About This Document

This whitepaper provides complete documentation for the HIRT system, a hybrid geophysical survey tool combining Magnetic Induction Tomography (MIT) and Electrical Resistivity Tomography (ERT) for subsurface imaging in forensic, archaeological, and environmental applications.

### Key Applications

- **WWII bomb crater investigation** - UXO assessment and burial detection
- **Woodland burial search** - Clandestine grave detection under tree cover
- **Wetland/swamp surveys** - Subsurface mapping in challenging terrain

---

## Document Structure

The whitepaper is organized into **5 parts** containing **19 sections**:

### PART I: FOUNDATIONS

| # | Section | Description |
|---|---------|-------------|
| 01 | [Executive Summary](01-executive-summary.md) | What is HIRT, use cases, capabilities, target audience |
| 02 | [Physics & Theory](02-physics-theory.md) | MIT and ERT principles, frequency selection, measurement geometry |
| 03 | [System Architecture](03-system-architecture.md) | Micro-probe design, centralized hub, array configurations |

### PART II: BUILDING

| # | Section | Description |
|---|---------|-------------|
| 04 | [Bill of Materials](04-bill-of-materials.md) | Complete BOM with costs ($1,800-3,900), part numbers, suppliers |
| 05 | [Mechanical Design](05-mechanical-design.md) | Rod segments, coils, ERT rings, junction box, 3D prints, CAD |
| 06 | [Electronics & Circuits](06-electronics-circuits.md) | MIT circuit, ERT circuit, power, complete schematics |
| 07 | [Assembly & Wiring](07-assembly-wiring.md) | Step-by-step assembly, wiring diagrams, waterproofing |
| 08 | [Testing & Verification](08-testing-verification.md) | Test procedures, QC checklist, pass/fail criteria |
| 09 | [Calibration](09-calibration.md) | Coil, TX, RX, ERT calibration; field quick-check; schedule |

### PART III: FIELD OPERATIONS

| # | Section | Description |
|---|---------|-------------|
| 10 | [Field Operations](10-field-operations.md) | Planning, grid design, installation, measurement protocols |
| 11 | [Data Recording](11-data-recording.md) | MIT/ERT file formats, probe registry, metadata |
| 12 | [Data Interpretation](12-data-interpretation.md) | Resolution, detection limits, combined analysis |
| 13 | [Troubleshooting](13-troubleshooting.md) | Diagnostics, repairs, when to abort |

### PART IV: REFERENCE

| # | Section | Description |
|---|---------|-------------|
| 14 | [Glossary](14-glossary.md) | Acronyms and terminology |
| 15 | [Quick Reference](15-quick-reference.md) | Printable field card |
| 16 | [Field Checklists](16-field-checklists.md) | Pre/on-site/post deployment checklists |
| 17 | [Application Scenarios](17-application-scenarios.md) | Detailed playbooks for crater, woods, swamp |
| 18 | [Future Development](18-future-development.md) | Software roadmap, hardware improvements, manufacturing status |
| 19 | [Ethics, Legal & Safety](19-ethics-legal-safety.md) | Permits, UXO protocols, conductivity monitoring |

---

## Reader Paths

### Path A: System Builder

Building a HIRT system from scratch:

```
01 Executive Summary
     ↓
03 System Architecture → understand the design
     ↓
04 Bill of Materials → order parts
     ↓
05 Mechanical Design → manufacture/print components
     ↓
06 Electronics & Circuits → assemble PCBs
     ↓
07 Assembly & Wiring → put it all together
     ↓
08 Testing & Verification → verify function
     ↓
09 Calibration → calibrate before deployment
```

### Path B: Field Operator

Operating an existing HIRT system in the field:

```
15 Quick Reference → keep on hand
     ↓
16 Field Checklists → pre-deployment
     ↓
10 Field Operations → detailed procedures
     ↓
11 Data Recording → understand data formats
     ↓
13 Troubleshooting → when issues arise
     ↓
19 Ethics, Legal & Safety → UXO sites especially
```

### Path C: Data Analyst

Processing and interpreting HIRT data:

```
02 Physics & Theory → understand measurements
     ↓
11 Data Recording → data format specs
     ↓
12 Data Interpretation → analysis methods
     ↓
17 Application Scenarios → context for interpretation
```

### Path D: Quick Start

Minimal reading for experienced users:

```
01 Executive Summary → 5 min overview
     ↓
15 Quick Reference → field card
     ↓
10 Field Operations → detailed if needed
```

---

## Visual Workflow

```
                    ┌─────────────────────────┐
                    │   01. Executive Summary │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ 02. Physics     │   │ 03. System Arch │   │ 19. Ethics/     │
│     & Theory    │   │                 │   │     Safety      │
└────────┬────────┘   └────────┬────────┘   └─────────────────┘
         │                     │
         │     ┌───────────────┴───────────────┐
         │     │         PART II: BUILD        │
         │     │ ┌────┐ ┌────┐ ┌────┐ ┌────┐  │
         │     │ │ 04 │→│ 05 │→│ 06 │→│ 07 │  │
         │     │ │BOM │ │Mech│ │Elec│ │Assy│  │
         │     │ └────┘ └────┘ └────┘ └──┬─┘  │
         │     │            ┌────┐ ┌────┐│    │
         │     │            │ 08 │→│ 09 ││    │
         │     │            │Test│ │ Cal│◄┘   │
         │     │            └────┘ └──┬─┘     │
         │     └──────────────────────┼───────┘
         │                            │
         │     ┌──────────────────────┴───────┐
         │     │     PART III: FIELD OPS      │
         │     │ ┌────┐ ┌────┐ ┌────┐ ┌────┐ │
         │     │ │ 10 │→│ 11 │→│ 12 │ │ 13 │ │
         └─────┼►│Field│ │Data│ │Intr│ │Trbl│ │
               │ └────┘ └────┘ └──┬─┘ └────┘ │
               └──────────────────┼──────────┘
                                  │
               ┌──────────────────┴──────────┐
               │      PART IV: REFERENCE     │
               │ ┌────┐ ┌────┐ ┌────┐ ┌────┐│
               │ │ 14 │ │ 15 │ │ 16 │ │ 17 ││
               │ │Glos│ │QRef│ │Chck│ │Scen││
               │ └────┘ └────┘ └────┘ └────┘│
               │        ┌────┐              │
               │        │ 18 │ Future Dev   │
               │        └────┘              │
               └─────────────────────────────┘
```

---

## Quick Topic Lookup

| Topic | Primary Section | Related Sections |
|-------|-----------------|------------------|
| **Coil winding** | 05 Mechanical Design | 09 Calibration |
| **Current source (ERT)** | 06 Electronics | 08 Testing |
| **Data formats** | 11 Data Recording | 12 Interpretation |
| **DDS/TX circuit** | 06 Electronics | 09 Calibration |
| **Frequency selection** | 02 Physics | 10 Field Ops |
| **Grid layout** | 10 Field Operations | 15 Quick Reference |
| **Lock-in detection** | 06 Electronics | 02 Physics |
| **Part numbers** | 04 Bill of Materials | - |
| **PCB layout** | 06 Electronics | 07 Assembly |
| **Probe insertion** | 10 Field Operations | 05 Mechanical |
| **QC checklist** | 08 Testing | 16 Checklists |
| **Reciprocity** | 09 Calibration | 12 Interpretation |
| **Ring electrodes** | 05 Mechanical | 06 Electronics |
| **Schematics** | 06 Electronics | 04 BOM |
| **Skin depth** | 02 Physics | 12 Interpretation |
| **STL files** | 05 Mechanical | 18 Future Dev |
| **Time-lapse** | 19 Ethics/Safety | 10 Field Ops |
| **UXO safety** | 19 Ethics/Safety | 10 Field Ops |

---

## Document Conventions

### File Naming
- Format: `XX-topic-name.md` where XX = section number (01-19)
- Lowercase with hyphens

### Cross-References
- Internal links: `[Section Title](XX-filename.md)`
- Section references in text: "See Section 10: Field Operations"

### Measurement Units
- Length: meters (m), millimeters (mm)
- Frequency: kilohertz (kHz), hertz (Hz)
- Current: milliamps (mA), microamps (uA)
- Resistance: ohms, kilohms (k-ohm), megohms (M-ohm)
- Conductivity: microsiemens/cm (uS/cm)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-01 | Consolidated from 33 to 19 sections |
| 1.0 | 2025-01 | Complete whitepaper package (33 sections) |
| 0.9 | 2024-12 | Manufacturing release (16mm modular design) |
| 0.5 | 2024-11 | Initial documentation structure |

---

## Total Documentation Size

| Part | Sections | Approx. Size |
|------|----------|--------------|
| I. Foundations | 3 | 26 KB |
| II. Building | 6 | 95 KB |
| III. Field Operations | 4 | 37 KB |
| IV. Reference | 6 | 37 KB |
| **Total** | **19** | **~195 KB** |

*Consolidated from 33 sections (~236 KB), eliminating ~40 KB of redundant content.*

---

*This is the master index for the HIRT whitepaper v2.0. All 19 sections are contained in this directory.*
