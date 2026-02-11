# HIRT Project Status

*Last updated: 2026-02-11*

## Current Focus

**Repository cleanup complete - ready for Phase 2 (prototype validation)**

The documentation has been restructured, build artifacts excluded from tracking, and the repository is in a clean, maintainable state.

## Recent Accomplishments

### 2026-02-11: Repository Cleanup
- Updated `.gitignore` to exclude Quarto build output, Jupyter caches, and site libraries
- Removed ~103 build artifact files from git tracking (still on disk)
- Rewrote README.md for current Technical Manual structure
- Updated research backlog with accurate file references (18 research documents across 4 directories)
- Cleaned up stale branches (dev, feature/research-2026-02-01, claude/review-last-task-t0wom)

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
| Repository cleanup | Complete | Build artifacts untracked, README updated |
| Troubleshooting hybrid rewrite | Pending | Keep tables, add narrative context |
| Performance claim qualifications | Pending | Add (Measured)/(Modeled)/(Target) labels |
| Progressive deployment diagrams | Pending | 5-6 new diagrams needed |
| Research backlog integration | Pending | 18 documents to review and integrate |

## Blockers

None currently.

## Next Steps

1. **Integrate research findings** - Review 18 research documents and integrate key findings into relevant Technical Manual sections
2. **Troubleshooting rewrite** - Apply hybrid approach (keep tables, add narrative context)
3. **Performance qualifications** - Add (Measured)/(Modeled)/(Target) labels throughout
4. **Progressive deployment diagrams** - Create 5-6 new diagrams
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

Repository cleanup complete. Ready for Phase 2: research integration and remaining documentation polish.

---

*Next session: Integrate research findings, troubleshooting hybrid rewrite*
