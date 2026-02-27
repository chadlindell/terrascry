# Archived Documentation

This directory contains deprecated documentation that has been superseded by the Quarto whitepaper in [`../hirt-whitepaper/`](../hirt-whitepaper/).

## Contents

### `pdf-generator/`

Original ReportLab-based PDF generation system. Contains:
- Python scripts for generating PDFs (`create_hirt_intro.py`, etc.)
- Diagram generation library (`lib/diagrams/`)
- Section content (`sections/`)

**Note:** Virtual environments and output directories were removed before archiving.

### `whitepaper/`

Original markdown-based whitepaper sections:
- 20 sections (`00-index.md` through `19-ethics-legal-safety.md`)
- Scientific narrative documents

## Why Archived?

The Quarto-based system (`hirt-whitepaper/`) provides:
- Single-source publishing to PDF and HTML
- Embedded diagram generation with Python code blocks
- Native cross-references and table of contents
- Dark mode and responsive HTML output
- Better maintainability

## Reference Use

These files may be useful for:
- Diagram function reference (some Quarto sections still import from here)
- Historical comparison
- Content that may not have been fully migrated

---

*Archived: January 2026*
