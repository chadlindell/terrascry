# HIRT Whitepaper - Quarto Project

This directory contains the unified HIRT whitepaper built with Quarto, producing both print-quality PDF and web-ready HTML from a single source.

## Prerequisites

### Install Quarto

Download and install from [quarto.org/docs/get-started](https://quarto.org/docs/get-started/)

```bash
# Verify installation
quarto --version
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

This installs matplotlib, numpy, pillow, jupyter, and ipykernel for diagram generation.

## Building the Whitepaper

### Full Build (PDF + HTML)

```bash
cd docs/hirt-whitepaper
quarto render
```

Outputs:
- `_output/HIRT_Whitepaper.pdf` - Combined PDF book
- `_output/index.html` - Web version with navigation

### PDF Only

```bash
quarto render --to pdf
```

### HTML Only

```bash
quarto render --to html
```

### Single Section (for testing)

```bash
quarto render sections/01-executive-summary.qmd --to pdf
```

## Project Structure

```
docs/hirt-whitepaper/
├── _quarto.yml              # Project configuration
├── index.qmd                # Cover page and preface
├── sections/
│   ├── 00-index.qmd         # Document index and reader paths
│   ├── 01-executive-summary.qmd
│   ├── 02-physics-theory.qmd
│   ├── ...                  # 20 total sections
│   └── 19-ethics-legal-safety.qmd
├── diagrams/                # Python diagram generation library
│   ├── __init__.py
│   ├── circuits.py
│   ├── physics.py
│   └── ...
├── assets/
│   └── styles/
│       └── custom.css       # HTML styling
├── _extensions/
│   └── hirt-journal/        # Custom Typst template (optional)
└── _output/                 # Generated outputs (gitignored)
```

## Document Organization

The whitepaper is organized into 4 parts:

| Part | Sections | Content |
|------|----------|---------|
| **I: Foundations** | 01-03 | Theory, architecture, overview |
| **II: Building** | 04-09 | BOM, design, assembly, testing |
| **III: Field Operations** | 10-13 | Deployment, data, troubleshooting |
| **IV: Reference** | 14-19 | Glossary, checklists, appendices |

## Reader Paths

- **System Builder:** 01 > 03 > 04-09 (build from scratch)
- **Field Operator:** 15 > 16 > 10 > 11 (operate existing system)
- **Data Analyst:** 02 > 11 > 12 > 17 (interpret results)
- **Quick Start:** 01 > 15 > 10 (minimal reading)

## Hardware Documentation Integration

This whitepaper references authoritative hardware documentation in `/hardware/`:

- `/hardware/bom/` - Bills of materials with part numbers
- `/hardware/schematics/` - Circuit and mechanical schematics
- `/hardware/cad/` - OpenSCAD source and STL files

The whitepaper provides conceptual overviews; hardware docs contain exact specifications.

## Diagram Generation

Diagrams are generated using matplotlib code embedded in `.qmd` files. The diagram library in `diagrams/` provides reusable functions. Some sections import from the legacy `pdf-generator/` during migration.

To test diagrams independently:

```bash
cd docs/hirt-whitepaper
python -c "from diagrams.circuits import *; print('OK')"
```

## Customization

### PDF Styling

Edit `_quarto.yml` format settings:
- `documentclass`, `papersize`, `margin-*` for layout
- `mainfont`, `fontsize` for typography

### HTML Styling

Edit `assets/styles/custom.css`:
- Color variables match HIRT branding
- Responsive layout adjustments

### Adding Sections

1. Create `sections/XX-topic-name.qmd`
2. Add to `_quarto.yml` in appropriate part
3. Include YAML frontmatter with title/subtitle

## Troubleshooting

**"ModuleNotFoundError" for diagrams:**
- Ensure Python path includes diagram library
- Run from `docs/hirt-whitepaper/` directory

**PDF rendering fails:**
- Install Typst or LaTeX backend
- Check `_quarto.yml` format settings

**Missing figures:**
- Verify matplotlib and pillow installed
- Check Jupyter kernel available

## Migration Notes

This Quarto project replaces:
- `/docs/pdf-generator/` - Legacy ReportLab PDF generation
- `/docs/whitepaper/` - Stale markdown files

These directories are retained for reference during migration but are no longer the authoritative source.
