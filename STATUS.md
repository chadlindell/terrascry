# HIRT Project Status

*Last updated: 2026-02-01*

## Current Focus

**Documentation restructure complete - Technical Manual ready for review**

The documentation has been completely restructured from "whitepaper" to "Technical Manual" format following the onion model architecture (Field Ops -> Build Guide -> Theory -> Developer).

## Recent Accomplishments

### 2026-02-01: Major Documentation Restructure
- Renamed from "whitepaper" to "docs" (HIRT Technical Manual)
- Restructured to "onion model" (Getting Started -> Field Guide -> Build Guide -> Theory -> Developer -> Appendices)
- Created new sections:
  - `progressive-deployment.qmd` - Advanced four-phase deployment workflow
  - `sensor-modalities.qmd` - Sensor maturity framework (Supported/Recommended/Future)
  - `validation.qmd` - HardwareX compliance validation procedures
  - `uncertainty.qmd` - Limitations and "what no anomaly means"
  - `contributing.qmd` - Contribution guidelines
  - `firmware.qmd` - Firmware architecture documentation
  - `quick-start.qmd` - New quick start guide
  - `index.qmd` - Task map landing page with cross-cutting navigation
- Split files:
  - `data-recording` -> `data-acquisition.qmd` + `data-formats.qmd`
  - `data-interpretation` -> `interpretation.qmd` + `inversion.qmd`
  - `ethics-legal-safety` -> `safety.qmd` + `regulations.qmd`
- Created GitHub Actions CI workflow for Quarto rendering
- Fixed all Unicode character issues for PDF generation
- Fixed cross-reference issues
- Both HTML and PDF render successfully

### 2026-01-30: Project Reorganization
- Cleaned up root directory structure
- Archived legacy files to `_archive/`
- Consolidated research into `research/` with topic subdirectories

## Active Work

| Task | Status | Notes |
|------|--------|-------|
| Documentation restructure | Complete | 30 sections render cleanly |
| HTML generation | Complete | Full site at `docs/_output/` |
| PDF generation | Complete | 2.5 MB Technical Manual |
| CI/CD workflow | Complete | `.github/workflows/quarto-render.yml` |
| Troubleshooting hybrid rewrite | Pending | Plan Phase 3 |
| Performance claim qualifications | Pending | Plan Phase 3 |
| Progressive deployment diagrams | Pending | Plan Phase 4 |

## Blockers

None currently.

## Next Steps

1. **Troubleshooting rewrite** - Apply hybrid approach (keep tables, add narrative context)
2. **Performance qualifications** - Add (Measured)/(Modeled)/(Target) labels throughout
3. **Progressive deployment diagrams** - Create 5-6 new diagrams
4. **Update CLAUDE.md** - Update paths and structure references
5. **Field validation** - Plan first prototype tests

## Quick Reference

```bash
# Render documentation
cd docs && quarto render

# Preview documentation (live reload)
cd docs && quarto preview

# Render PDF only
cd docs && quarto render --to pdf

# Render HTML only
cd docs && quarto render --to html
```

## Directory Structure

```
docs/
├── index.qmd                    # Task map landing page
├── getting-started/             # Onboarding
│   ├── overview.qmd
│   ├── quick-start.qmd
│   └── safety.qmd
├── field-guide/                 # Operations
│   ├── deployment.qmd
│   ├── progressive-deployment.qmd
│   ├── data-acquisition.qmd
│   ├── interpretation.qmd
│   ├── scenarios.qmd
│   └── troubleshooting.qmd
├── build-guide/                 # Construction
│   ├── bill-of-materials.qmd
│   ├── mechanical.qmd
│   ├── electronics.qmd
│   ├── assembly.qmd
│   ├── testing.qmd
│   ├── calibration.qmd
│   └── validation.qmd
├── theory/                      # Technical depth
│   ├── physics.qmd
│   ├── architecture.qmd
│   ├── sensor-modalities.qmd
│   ├── inversion.qmd
│   └── uncertainty.qmd
├── developer/                   # Contributors
│   ├── contributing.qmd
│   ├── firmware.qmd
│   ├── data-formats.qmd
│   └── roadmap.qmd
└── appendices/                  # Reference
    ├── glossary.qmd
    ├── quick-reference.qmd
    ├── checklists.qmd
    └── regulations.qmd
```

## Session Notes

Documentation restructure complete. Old whitepaper archived to `_archive/whitepaper-pre-rewrite-20260201/`.

---

*Next session: Hybrid troubleshooting rewrite, performance claim qualifications*
