# HIRT - Hybrid Inductive-Resistive Tomography

A modular, in-ground probe array integrating magneto-inductive (low-frequency EM) and electrical resistivity measurements for high-resolution 3-D subsurface imaging in archaeological and forensic contexts.

## Project Description

**Hybrid Inductive-Resistive Tomography (HIRT)** is a DIY probe-array subsurface imaging system developed for archaeological recovery of WWII aircraft crash sites. The system integrates magneto-inductive and electrical resistivity measurements using modular dual-role probes to generate volumetric conductivity and resistivity models to depths of 3–6 m.

> "We employed a Hybrid Inductive-Resistive Tomography (HIRT) probe array, a novel low-intrusion subsurface imaging method developed for archaeological recovery of WWII aircraft crash sites. The system integrates magneto-inductive and electrical resistivity measurements using modular dual-role probes to generate volumetric conductivity and resistivity models to depths of 3–6 m."

## Quick Links

- **[Documentation](docs/README.md)** - Complete project documentation
- **[Technical Whitepaper](docs/hirt-whitepaper/)** - Full technical documentation (Quarto)
- **[BOM Files](hardware/bom/)** - Bill of Materials and order sheets
- **[Field Guide](docs/field-guide/)** - Field deployment guides

## Project Structure

```
HIRT/
├── docs/                    # Documentation
│   ├── hirt-whitepaper/    # Technical whitepaper (Quarto - PDF & HTML)
│   ├── field-guide/        # Field-ready guides
│   ├── research/           # Research documents
│   └── _archive/           # Deprecated documentation
├── hardware/               # Hardware documentation
│   ├── bom/               # Bill of Materials
│   ├── cad/               # OpenSCAD source and STLs
│   ├── schematics/        # Circuit diagrams
│   └── drawings/          # Technical drawings
├── images/                 # Generated images and photos
├── build/                  # Build instructions
└── README.md              # This file
```

## Key Features

- **Dual-Method Imaging:** MIT (magneto-inductive) + ERT (electrical resistivity)
- **Micro-Probe Design:** 12mm OD passive probes (~10× less disturbance)
- **Modular Design:** Identical passive probes + central electronics hub
- **Low Cost:** $1,800–3,900 for complete starter kit
- **Field-Ready:** Designed for archaeological/forensic field teams
- **DIY-Friendly:** Detailed documentation for construction
- **Archaeology-Friendly:** Minimal intrusion, acceptable for sensitive contexts

## System Overview

- **Probes:** 20–24 identical passive micro-probes (12mm OD)
- **Depth:** 2–6 m investigation depth (depending on configuration)
- **Resolution:** 0.5–1.5 m lateral resolution
- **Applications:** Aircraft wreckage, disturbed fill, voids, potential burials
- **Design:** Minimal intrusion (~10× less disturbance than 2" probes)

## Getting Started

1. **Read the Documentation:** Start with [docs/README.md](docs/README.md)
2. **Review the Whitepaper:** [docs/hirt-whitepaper/](docs/hirt-whitepaper/) (build with `quarto render`)
3. **Check BOM:** [hardware/bom/](hardware/bom/) for components
4. **Order Components:** Use [order sheets](hardware/bom/order-sheets/)
5. **Build System:** Follow [build/](build/) instructions

## Important Warnings

⚠️ **UXO Risk:** WWII sites can contain live ordnance. Do not deploy until cleared by qualified EOD/UXO professionals.

⚠️ **Ethics & Legal:** Obtain all permits and permissions. Follow jurisdictional requirements for archaeological/forensic work.

⚠️ **Human Remains:** Treat as forensic/archaeological context with proper protocols.

## Status

- **Documentation:** v2.0 (Quarto whitepaper complete)
- **Hardware Design:** In progress (schematics/drawings pending)
- **Software:** Future development (see whitepaper Section 18)

## Cost Estimate

- **Complete Starter Kit:** $1,800–3,900
  - 20 probes: $1,400–3,000
  - Base hub: $200–500
  - Tools and supplies: $200–400

## Use Cases

- Filled bomb craters (~10–15 m diameter, ~3 m deep)
- Woods burials (0.6–1.5 m depth)
- Swamp/perched water impacts (>5 m targets)

## Documentation

See [docs/README.md](docs/README.md) for complete documentation index.

## License

[To be determined]

## Patent & IP Status

- The micro-probe mechanics, CAD, and assembly notes published in this repo act as a defensive publication so the community retains DIY access to the design.
- A limited-scope provisional filing (covering the hybrid MIT/ERT deployment workflow and base hub control chain) is being prepared; reach out via the project issue tracker for coordination before filing derivative patents.
- By contributing, you acknowledge that your submissions may be referenced in any future patent filings (defensive or otherwise) and that you retain the right to practice your contributions.
- Please document any prior-art references inside pull requests so we can maintain a clear freedom-to-operate record.
- See `docs/IP-LOG.md` for a running log of publications, prior art, and filing status.

## Contributing

Contributions, improvements, and field testing results are welcome.
