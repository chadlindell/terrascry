# HIRT - Hybrid Inductive-Resistivity Tomography

A dual-channel subsurface imaging system combining electromagnetic induction (MIT-3D) and electrical resistivity (ERT-Lite) for true 3D tomographic imaging in archaeological and forensic investigations.

## What is HIRT?

HIRT uses insertable probes in crosshole geometry to image the subsurface with two complementary methods:

- **MIT-3D** (Magneto-Inductive Tomography): Detects metallic objects including aluminum at depths to 3m
- **ERT-Lite** (Electrical Resistivity): Maps soil disturbance patterns indicating burial or excavation

## Key Features

| Feature | Specification |
|---------|---------------|
| Depth | 2-6 m investigation depth |
| Resolution | 0.5-1.5 m lateral resolution |
| Cost | $1,800-3,900 complete system |
| Probes | 16-24 modular 16mm OD probes |
| Output | 3D conductivity and resistivity models |

## Primary Applications

- Filled bomb crater investigation (UXO, human remains)
- Woodland burial detection (clandestine graves)
- Swamp/marsh crash site recovery

## Getting Started

### Read the Technical Manual
```bash
cd docs && quarto preview
```
The Technical Manual is the single source of truth for all technical documentation.

### Quick Links

| Resource | Location |
|----------|----------|
| Technical Manual | `docs/` |
| Bill of Materials | `hardware/bom/` |
| CAD Files (OpenSCAD) | `hardware/cad/` |
| Circuit Schematics | `hardware/schematics/` |
| Research Documents | `research/` |

## Project Structure

```
HIRT/
├── VISION.md          # Project goals and constraints
├── STATUS.md          # Current state and active work
├── OUTLINE.md         # Documentation section status
├── CLAUDE.md          # Style guide for documentation
│
├── docs/              # Technical Manual (Quarto)
│   ├── index.qmd              # Task map landing page
│   ├── getting-started/       # Onboarding (overview, quick-start, safety)
│   ├── field-guide/           # Operations (deployment, data, troubleshooting)
│   ├── build-guide/           # Construction (BOM, mechanical, electronics)
│   ├── theory/                # Technical depth (physics, inversion, sensors)
│   ├── developer/             # Contributors (firmware, data-formats, roadmap)
│   ├── appendices/            # Reference (glossary, checklists, regulations)
│   └── diagrams/              # Python diagram generators
│
├── research/          # Research by topic
│   ├── deployment/    # Probe insertion methods
│   ├── electronics/   # Modernization research
│   ├── regulatory/    # Legal/compliance
│   └── literature/    # Prior art and comparable projects
│
├── hardware/          # Hardware documentation
│   ├── bom/          # Bills of materials
│   ├── cad/          # OpenSCAD sources and STLs
│   ├── drawings/     # Assembly drawings
│   └── schematics/   # Circuit documentation
│
└── _archive/          # Legacy files (preserved)
```

## Building the Documentation

Requires [Quarto](https://quarto.org/) and Python 3.

```bash
# Install Python dependencies
pip install matplotlib numpy pillow

# Render Technical Manual to HTML and PDF
cd docs && quarto render

# Preview with live reload
cd docs && quarto preview

# Render HTML only
cd docs && quarto render --to html

# Render PDF only
cd docs && quarto render --to pdf
```

## Warnings

**UXO Risk:** WWII sites may contain live ordnance. Do not deploy until cleared by qualified EOD/UXO professionals.

**Legal Requirements:** Obtain all permits and permissions. Follow jurisdictional requirements for archaeological and forensic work.

**Human Remains:** Follow proper forensic/archaeological protocols when remains may be present.

## Project Status

| Component | Status |
|-----------|--------|
| Technical Manual | Complete (30 sections, HTML + PDF) |
| Hardware Design | Complete (schematics, CAD, BOM) |
| Research | Ongoing (see `research/`) |
| Software | Future development |

## License

Open source - license to be determined.

## Contributing

Contributions, improvements, and field testing results are welcome. See `docs/developer/contributing.qmd` for contribution guidelines and `CLAUDE.md` for documentation style guide.
