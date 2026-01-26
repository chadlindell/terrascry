# HIRT Project Documentation

**Hybrid Inductive-Resistive Tomography (HIRT)** - A modular, in-ground probe array integrating magneto-inductive (low-frequency EM) and electrical resistivity measurements for high-resolution 3-D subsurface imaging in archaeological and forensic contexts.

## Documentation Structure

### Technical Whitepaper (Primary)

The complete technical whitepaper is built with [Quarto](https://quarto.org/) and produces both PDF and HTML output.

- **Location:** [`hirt-whitepaper/`](hirt-whitepaper/)
- **Build:** `cd hirt-whitepaper && quarto render`
- **Output:** `_output/HIRT_Whitepaper.pdf` and `_output/index.html`

The whitepaper covers:
- Physics theory (MIT-3D and ERT-Lite)
- System architecture
- Bill of materials and build instructions
- Assembly, testing, and calibration
- Field operations and data recording
- Troubleshooting and reference materials

### Field Guide

Quick reference materials for field deployment:

- **[Quick Reference](field-guide/quick-reference.md)** - One-page field reference
- **[Coil Winding Recipe](field-guide/coil-winding-recipe.md)** - Detailed coil specifications
- **[ERT Source Schematic](field-guide/ert-source-schematic.md)** - ERT current source design

### Research

Background research and feasibility studies:

- **[research/](research/)** - Probe insertion methods, UXO detection research

### Hardware Documentation

Authoritative hardware specifications are in [`/hardware/`](../hardware/):

- **[BOM](../hardware/bom/)** - Bills of materials with order sheets
- **[CAD](../hardware/cad/)** - OpenSCAD source, STLs, slicer profiles
- **[Schematics](../hardware/schematics/)** - Electronics and mechanical schematics

### Build Documentation

Assembly and testing procedures are in [`/build/`](../build/):

- Assembly guides
- Calibration procedures
- QC checklists

## Quick Start

1. **Build the Whitepaper:** `cd hirt-whitepaper && quarto render`
2. **Review BOM:** [`/hardware/bom/`](../hardware/bom/) for components and costs
3. **Order Components:** Use [order sheets](../hardware/bom/order-sheets/)
4. **Build System:** Follow [`/build/`](../build/) instructions
5. **Deploy:** Use [Field Guide](field-guide/) for deployment

## System Specifications

| Parameter | Value |
|-----------|-------|
| Probe Count | 20–24 probes |
| Probe Depth | 1.5–3.0 m |
| Probe Spacing | 1.0–2.0 m |
| MIT Frequencies | 2–50 kHz |
| ERT Current | 0.5–2 mA |
| Investigation Depth | 2–6 m |
| Cost | $1,800–3,900 |

## Archived Documentation

Legacy documentation preserved for reference:

- **[_archive/pdf-generator/](_archive/pdf-generator/)** - Original ReportLab PDF generation system
- **[_archive/whitepaper/](_archive/whitepaper/)** - Original markdown whitepaper sections

These have been superseded by the Quarto whitepaper.

## Project Status

- **Documentation:** v2.0 (Quarto whitepaper)
- **Hardware:** Design phase
- **Software:** Future development

## License

[To be determined]

