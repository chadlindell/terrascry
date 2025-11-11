# HIRT Project Documentation

**Hybrid Inductive-Resistive Tomography (HIRT)** - A modular, in-ground probe array integrating magneto-inductive (low-frequency EM) and electrical resistivity measurements for high-resolution 3-D subsurface imaging in archaeological and forensic contexts.

## Project Overview

This project documents the design, construction, and deployment of a DIY probe-array subsurface imaging system for WWII aircraft crash sites and possible graves in sandy/loamy soils. The system integrates:

- **MIT-3D (Magneto-Inductive Tomography)**: Low-frequency EM measurements using TX/RX coils
- **ERT-Lite (Electrical Resistivity Tomography)**: Current injection and voltage measurement

## Documentation Structure

### White Paper
- **[Main Document](whitepaper/main.md)** - Complete white paper with table of contents
- **[Sections](whitepaper/sections/)** - Individual sections covering all aspects:
  - Scope & Use Cases
  - Ethics, Legal & Safety
  - Concept and Physics
  - System Architecture
  - BOM and Build Instructions
  - Calibration and Deployment
  - Troubleshooting and Playbooks

### Field Guide
- **[Quick Reference](field-guide/quick-reference.md)** - One-page field reference
- **[Coil Winding Recipe](field-guide/coil-winding-recipe.md)** - Detailed coil specifications
- **[ERT Source Schematic](field-guide/ert-source-schematic.md)** - ERT current source design

## Hardware Documentation

### Bill of Materials (BOM)
- **[Probe BOM](hardware/bom/probe-bom.md)** - Per-probe components and costs
- **[Base Hub BOM](hardware/bom/base-hub-bom.md)** - Shared/base components
- **[Shared Components BOM](hardware/bom/shared-components-bom.md)** - Tools and supplies
- **[Order Sheets](hardware/bom/order-sheets/)** - CSV files ready for procurement

### Schematics
- **[Electronics](hardware/schematics/electronics/)** - Circuit diagrams (placeholders)
  - Probe electronics block diagram
  - MIT circuit schematic
  - ERT circuit schematic
  - Base hub circuit schematic
- **[Mechanical](hardware/schematics/mechanical/)** - Assembly drawings (placeholders)
  - Probe assembly
  - Rod specifications
  - ERT ring mounting

### Drawings
- **[Probe Head Drawing](hardware/drawings/probe-head-drawing.md)** - Technical drawings (placeholder)
- **[Assembly Drawings](hardware/drawings/assembly-drawings.md)** - Complete assembly (placeholder)

## Build Documentation

- **[Assembly Guide](build/assembly-guide.md)** - Step-by-step build instructions
- **[Calibration Procedures](build/calibration-procedures.md)** - Pre-field calibration
- **[QC Checklist](build/qc-checklist.md)** - Quality control procedures

## Quick Start

1. **Read the [White Paper](whitepaper/main.md)** for complete system overview
2. **Review [BOM files](hardware/bom/)** to understand components and costs
3. **Check [Order Sheets](hardware/bom/order-sheets/)** for procurement
4. **Follow [Build Documentation](build/)** for assembly
5. **Use [Field Guide](field-guide/)** for deployment

## System Specifications

- **Probe Count:** 20–24 probes (standard section)
- **Probe Depth:** 1.5–3.0 m (depending on deployment)
- **Probe Spacing:** 1.0–2.0 m (adjustable)
- **MIT Frequencies:** 2–50 kHz (selectable)
- **ERT Current:** 0.5–2 mA
- **Depth Range:** 2–6 m (depending on configuration)
- **Cost:** $1,800–3,900 (complete starter kit)

## Use Cases

- **Filled Bomb Craters:** 10–15 m diameter, ~3 m deep
- **Woods Burials:** Single/multiple interments, 0.6–1.5 m depth
- **Swamp/Margin Impacts:** Deeper targets (>5 m) from accessible margins

## Important Notes

- **Ethics & Legal:** Obtain all permits and UXO clearance before deployment
- **Safety:** Follow EOD protocols, maintain safe practices
- **Minimal Intrusion:** Use rim/perimeter probing when possible
- **Software:** Data processing and inversion are separate (future development)

## Project Status

- **Documentation:** Complete (v0.9)
- **Hardware:** Design phase (schematics/drawings to be completed)
- **Software:** Future development (see [Section 19](whitepaper/sections/19-next-steps.md))

## Contributing

This is a DIY/open-source project. Contributions, improvements, and field testing results are welcome.

## License

[To be determined]

## Contact

[To be added]

