# HIRT - Project Information

## Project Overview

HIRT (Hybrid Inductive-Resistivity Tomography) is a dual-channel subsurface imaging system for archaeological and forensic investigations. Uses crosshole geometry with probes inserted into the ground to achieve true 3D tomographic imaging.

### Sensing Modalities
- **MIT-3D** (Magneto-Inductive Tomography): Low-frequency TX/RX coils (2-50 kHz) for metal detection including aluminum
- **ERT-Lite** (Electrical Resistivity): Ring electrodes with small currents (0.5-2 mA) for soil disturbance mapping

### Primary Applications
- Filled bomb crater investigation (UXO, remains)
- Woods/forest burial detection
- Swamp/marsh crash site recovery

## Essential Commands

### Documentation (Quarto)
```bash
# Render documentation (HTML + PDF)
cd docs && quarto render

# Preview documentation (live reload)
cd docs && quarto preview

# Render PDF only
cd docs && quarto render --to pdf

# Render HTML only
cd docs && quarto render --to html

# Render specific section
cd docs && quarto render getting-started/overview.qmd
```

### CAD (OpenSCAD)
```bash
# Render STL from OpenSCAD
openscad -o output.stl hardware/cad/openscad/modular_flush_connector.scad
```

## Directory Structure
```
HIRT/
├── VISION.md              # Project goals and constraints
├── STATUS.md              # Current state, active work
├── OUTLINE.md             # Section-by-section status
├── CLAUDE.md              # Style guide and conventions
│
├── docs/                  # Main deliverable (Quarto Technical Manual)
│   ├── index.qmd              # Task map landing page
│   ├── getting-started/       # Onboarding
│   │   ├── overview.qmd
│   │   ├── quick-start.qmd
│   │   └── safety.qmd
│   ├── field-guide/           # Operations
│   │   ├── deployment.qmd
│   │   ├── progressive-deployment.qmd
│   │   ├── data-acquisition.qmd
│   │   ├── interpretation.qmd
│   │   ├── scenarios.qmd
│   │   └── troubleshooting.qmd
│   ├── build-guide/           # Construction
│   │   ├── bill-of-materials.qmd
│   │   ├── mechanical.qmd
│   │   ├── electronics.qmd
│   │   ├── assembly.qmd
│   │   ├── testing.qmd
│   │   ├── calibration.qmd
│   │   └── validation.qmd
│   ├── theory/                # Technical depth
│   │   ├── physics.qmd
│   │   ├── architecture.qmd
│   │   ├── sensor-modalities.qmd
│   │   ├── inversion.qmd
│   │   └── uncertainty.qmd
│   ├── developer/             # Contributors
│   │   ├── contributing.qmd
│   │   ├── firmware.qmd
│   │   ├── data-formats.qmd
│   │   └── roadmap.qmd
│   ├── appendices/            # Reference
│   │   ├── glossary.qmd
│   │   ├── quick-reference.qmd
│   │   ├── checklists.qmd
│   │   └── regulations.qmd
│   ├── diagrams/              # Python diagram generators
│   └── _quarto.yml            # Quarto config
│
├── research/              # Research documents by topic
│   ├── _backlog.md        # Research tracking
│   ├── deployment/        # Probe insertion methods
│   ├── electronics/       # Circuit modernization
│   ├── regulatory/        # Legal/compliance
│   └── literature/        # Academic papers
│
├── hardware/              # Hardware documentation
│   ├── bom/              # Bills of materials
│   ├── cad/              # OpenSCAD sources, STLs
│   ├── drawings/         # Assembly drawings
│   └── schematics/       # Circuit documentation
│
├── .claude/               # RIPER workflow, memory bank
└── _archive/              # Inactive files (includes old whitepaper/)
```

## Technology Stack
- **Documentation**: Quarto, Python (matplotlib, PIL for diagrams)
- **CAD**: OpenSCAD (parametric 3D modeling)
- **Physics**: MIT-3D coil design, ERT electrode theory
- **Electronics**: Analog signal conditioning, microcontroller firmware

## RIPER Workflow

This project uses the RIPER development process for structured, context-efficient work.

### Available Commands
- `/riper:strict` - Enable strict RIPER protocol enforcement
- `/riper:research` - Research mode for information gathering
- `/riper:innovate` - Innovation mode for brainstorming (optional)
- `/riper:plan` - Planning mode for specifications
- `/riper:execute` - Execution mode for implementation
- `/riper:execute <substep>` - Execute a specific substep from the plan
- `/riper:review` - Review mode for validation
- `/memory:save` - Save context to memory bank
- `/memory:recall` - Retrieve from memory bank
- `/memory:list` - List all memories

### Workflow Phases
1. **Research & Innovate** - Understand physics, existing implementations, prior art
2. **Plan** - Create detailed technical specifications saved to memory bank
3. **Execute** - Implement exactly what was specified in the approved plan
4. **Review** - Validate implementation against the plan

### Using the Workflow
1. Start with `/riper:strict` to enable strict mode enforcement
2. Use `/riper:research` to investigate topics (physics, papers, components)
3. Optionally use `/riper:innovate` to brainstorm approaches
4. Create a plan with `/riper:plan`
5. Execute with `/riper:execute` (or `/riper:execute 1.2` for specific steps)
6. Validate with `/riper:review`

## Memory Bank Policy

### CRITICAL: Repository-Level Memory Bank
- Memory-bank location: `.claude/memory-bank/`
- All memories are branch-aware and date-organized
- Memories persist across sessions

### Memory Bank Structure
```
.claude/memory-bank/
+-- [branch-name]/
    +-- plans/      # Technical specifications
    +-- reviews/    # Code review reports
    +-- sessions/   # Session context
```

## Research Guidelines for HIRT

### Physics & Theory
- Electromagnetic induction principles
- Electrical resistivity tomography theory
- Crosshole/borehole geophysics methods
- Signal processing for subsurface imaging

### Hardware Research
- Coil design for MIT (ferrite cores, winding patterns)
- Electrode materials and contact resistance
- Analog front-end design (instrumentation amplifiers, filters)
- PCB design for mixed-signal applications

### Prior Art & References
- Academic papers on MIT and ERT
- Commercial geophysical equipment specifications
- Archaeological geophysics case studies
- UXO detection literature
