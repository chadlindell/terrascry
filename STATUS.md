# HIRT Project Status

*Last updated: 2026-02-12*

## Current Focus

**Phase B quality pass complete -- all research integrated, full qualification sweep done**

Phase B addressed the remaining quality debt: 14 research documents reviewed (7 integrated, 7 closed as no-integration-needed), ~190 numeric claims qualified across 14 .qmd files, and a critical bentonite correctness bug fixed in deployment.qmd.

## Recent Accomplishments

### 2026-02-12: Phase B Quality Pass (Research Integration & Extended Qualification)
- **Critical fix:** Replaced bentonite slurry recommendation with polymer drilling fluid (xanthan gum) in deployment.qmd -- bentonite causes 80% resistivity reduction and is PROHIBITED for ERT measurements
- **HIGH priority research (4 docs -> deployment.qmd):**
  - Added Borehole Creation Methods comparison table (8 methods)
  - Added Push Force Requirements by Soil Type table (6 soil types)
  - Added Pre-Push UXO Assessment checklist and Real-Time Monitoring guidance
- **MEDIUM priority research (3 docs -> 3 files):**
  - UXO safe standoff distances and abort criteria -> safety.qmd
  - Pre-push UXO assessment checklist -> deployment.qmd
  - Autonomous probe deployment (TRL 3-4, ROGO reference) -> roadmap.qmd
- **LOW priority research (7 docs):** Reviewed and closed -- speculative/duplicate content with no actionable data beyond existing integrations
- **Extended qualification sweep:** Added (Measured)/(Modeled)/(Target)/(Manufacturer spec) labels to ~190 numeric claims across 14 remaining .qmd files
- **Research backlog:** All 14 documents accounted for in _backlog.md with full traceability

### 2026-02-12: Phase A Quality Pass (Iteration 2)
- **Research integration:** 4 research documents integrated into manual sections
  - Electronics modernization -> `electronics.qmd` (AD8421, OPA2186, AD7124-8 upgrade path)
  - Manufacturing costs -> `bill-of-materials.qmd` (PCBA and CNC cost analysis)
  - Deployment feasibility -> `progressive-deployment.qmd` (insertion method comparison table)
  - HERO safety -> `safety.qmd` (regulatory framework, MNFS thresholds, permit-to-dig, mathematical basis)
- **Performance qualification sweep:** Added (Measured)/(Modeled)/(Target) labels to ~25 numeric claims across 4 files (physics.qmd, sensor-modalities.qmd, bill-of-materials.qmd, progressive-deployment.qmd)
- **New diagrams created:**
  - `docs/diagrams/sensor_modalities.py` (new module): Sensor maturity framework, tilt error propagation, multi-modal fusion architecture
  - `docs/diagrams/progressive_deployment.py` (extended): 4-panel progressive cross-section replacing visual placeholder
  - Embedded `create_depth_extension_procedure()` diagram in progressive-deployment.qmd
- **Troubleshooting polish:** Added field repairs narrative lead-in, expanded "When to Abort" with risk escalation guidance and HERO safety cross-references

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
| Troubleshooting hybrid rewrite | Complete | Narrative context added to field repairs and abort sections |
| Performance claim qualifications | Complete | All 18 .qmd files now have qualification labels |
| Progressive deployment diagrams | Complete | Cross-section, depth extension, workflow diagrams done |
| Sensor modality diagrams | Complete | Maturity framework, tilt error, fusion architecture |
| Research integration (Phase A) | Complete | 4 key research docs integrated inline |
| Research integration (Phase B) | Complete | 7 more docs integrated, 7 reviewed and closed |
| Extended qualification sweep | Complete | ~190 claims labeled across 14 files |
| Bentonite correctness fix | Complete | Polymer drilling fluid replaces prohibited bentonite |

## Blockers

None currently.

## Next Steps

1. **Phase C: Deep theory pass** - Expand HERO analysis in physics, modernized signal chain in architecture
2. **Phase C: Extended diagram coverage** - Architecture block diagram, uncertainty visualization
3. **Field validation** - Plan first prototype tests

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

Phase B quality pass complete. All research documents reviewed and integrated or closed. Full qualification sweep applied to all .qmd files. Critical bentonite bug fixed. Ready for Phase C: deep theory pass and extended diagram coverage.

---

*Next session: Phase C -- deep theory pass, extended diagrams, field validation planning*
