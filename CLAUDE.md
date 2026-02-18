# HIRT Project Context

HIRT (Hybrid Inductive-Resistivity Tomography) is a dual-channel subsurface imaging system for archaeology, forensics, and environmental applications.

## Quick Reference

### RIPER Workflow
Use `/riper:strict` to enable mode tracking. Available modes:
- `/riper:research` - Information gathering (read-only)
- `/riper:innovate` - Brainstorming approaches
- `/riper:plan` - Create specifications (saved to memory bank)
- `/riper:execute` - Implement approved plans
- `/riper:review` - Validate implementation

### Memory Bank
- `/memory:save` - Save context to `.claude/memory-bank/`
- `/memory:recall` - Retrieve saved context
- `/memory:list` - List all memories

### Visualization
- `/visualize:scan` - Find all visual placeholders (read-only)
- `/visualize:expand` - Convert placeholders to diagram code

### Key Commands
- Render documentation: `cd docs && quarto render`
- Preview documentation: `cd docs && quarto preview`
- Render PDF only: `cd docs && quarto render --to pdf`
- Render HTML only: `cd docs && quarto render --to html`
- Export STL: `openscad -o output.stl input.scad`

## Sensor Pod Integration

HIRT shares a sensor pod with Pathfinder (ZED-F9P RTK GPS + BNO055 IMU + BMP390 barometer + DS3231 RTC in IP67 enclosure). The pod connects via M8 4-pin connector and PCA9615 differential I2C over Cat5 STP cable.

**HIRT-specific usage:**
- GPS records probe insertion point positions (one-shot per probe, cm-accuracy RTK)
- Probe positions stored in survey metadata CSV for inversion geometry
- Pod IMU provides surface orientation during position recording
- Pod is physically moved between Pathfinder and HIRT as needed

**Pathfinder surface data as inversion boundary condition:**
- Pathfinder's magnetic gradient map constrains the top layer of HIRT's 3D model
- Pathfinder's EMI conductivity constrains upper resistivity layers
- LiDAR DEM corrects HIRT's inversion mesh for actual terrain
- Cross-gradient regularization couples multi-physics models in SimPEG

See: `research/electronics/sensor-pod-integration.md`, `../Pathfinder/research/multi-sensor-architecture/sensor-pod-design.md`

## Project Structure
```
HIRT/
├── VISION.md          # Project goals and constraints
├── STATUS.md          # Current state, active work
├── OUTLINE.md         # Section-by-section status
├── CLAUDE.md          # This file - style guide
├── docs/              # Main deliverable (Quarto Technical Manual)
│   ├── index.qmd              # Task map landing page
│   ├── getting-started/       # Onboarding (overview, quick-start, safety)
│   ├── field-guide/           # Operations (deployment, data, troubleshooting)
│   ├── build-guide/           # Construction (BOM, mechanical, electronics)
│   ├── theory/                # Technical depth (physics, inversion, sensors)
│   ├── developer/             # Contributors (firmware, data-formats, roadmap)
│   ├── appendices/            # Reference (glossary, checklists, regulations)
│   └── diagrams/              # Python diagram generators
├── research/          # Research by topic
│   ├── deployment/    # Probe insertion methods
│   ├── electronics/   # Circuit modernization
│   ├── regulatory/    # Legal/compliance
│   └── literature/    # Academic papers
├── hardware/          # Hardware documentation
│   ├── bom/          # Bills of materials
│   ├── cad/          # OpenSCAD sources, STLs
│   ├── drawings/     # Assembly drawings
│   └── schematics/   # Circuit documentation
└── _archive/          # Inactive files (includes old whitepaper/)
```

---

## Writing Style Guide

### Philosophy
The documentation follows a **hybrid approach**:
- **Narrative prose** for theory, physics, and design sections - builds understanding progressively
- **Lists and tables** for troubleshooting, checklists, and field procedures - quick scanning under stress

Prioritize **narrative flow over enumeration** in explanatory sections. Lists are appropriate for:
- Bill of materials and specifications
- Step-by-step procedures where sequence matters
- Quick reference tables
- Troubleshooting diagnostics

### Performance Claims Framework
All specifications must be qualified with their validation status:

| Qualifier | Usage | Example |
|-----------|-------|---------|
| **(Measured)** | Bench/field tested | "Bench measurements show 100 nV RMS noise" |
| **(Modeled)** | Theoretical analysis | "Simulations predict 2-5x resolution improvement" |
| **(Target)** | Design goal, not yet validated | "Design targets <$4000 total cost" |

### Audience
Primary readers are technically sophisticated academics, forensic investigators, and archaeological researchers. Assume familiarity with basic physics and engineering concepts but not specialized geophysics. The writing should be:
- **Rigorous**: Claims supported by equations, measurements, or citations
- **Accessible**: Complex concepts introduced with context before diving into detail
- **Practical**: Theory always connected to implementation decisions

### Voice and Tone
- Active voice preferred ("The coil generates..." not "A field is generated by...")
- First person plural for design decisions ("We selected..." not "It was decided...")
- Present tense for describing how the system works
- Past tense for describing the development process or experiments

### Terminology
Maintain consistency throughout:
- **MIT-3D**: The inductive channel (not "EMI", "metal detector", or "inductive sensor")
- **ERT-Lite**: The resistivity channel (not "galvanic", "DC resistivity")
- **Crosshole**: The measurement geometry (not "borehole-to-borehole")

#### Component Terminology
- **Probe**: The complete insertable sensor assembly (sensors + electronics + housing)
- **Rod segment**: Individual fiberglass tube sections that stack to form probe structure
- **Pilot rod**: Metal tool used to create insertion hole (distinct from sensor probe)
- **Survey stake**: Temporary marker for grid positioning (not the sensor probe)

#### Unit Formatting
- **Resistance**: Use "M-ohm" (megohm), "k-ohm" (kilohm), or "ohms" (not special symbols in prose)
- **Inductance**: Use "mH" (millihenries) consistently
- **Spacing**: Follow SI convention with space between value and unit (e.g., "16 mm", "2 kHz")
- **Unicode**: Avoid special Unicode characters (arrows, Greek letters) in source - use ASCII alternatives for PDF compatibility

### Sensor Maturity Framework
Document sensors by maturity level:
- **S (Supported Now)**: Fully validated, documented in build guide (MIT-3D, ERT-Lite)
- **R (Recommended Extension)**: Tested principle, straightforward integration (Accelerometer, Temperature)
- **F (Future Exploration)**: Promising concept, requires development (Moisture, pH, Gas)

### Visual-First Writing
Complex concepts deserve rich visuals. When writing about something that would benefit from a diagram, schematic, or illustration, insert a visual placeholder block:

```markdown
::: {.visual-placeholder}
**Figure Concept**: [short title]

[Detailed description of what the visual should show, including:
- Key elements and their relationships
- Annotations or callouts needed
- Scale or proportion requirements
- Color coding or visual hierarchy
- What understanding this should convey]

**Type**: [diagram | schematic | photo | chart | cross-section | exploded-view]
:::
```

These placeholders serve as specifications for a visualization agent to expand into actual Python diagram code. Write the description richly enough that someone could draw it without additional context.

### Section Structure
Each major section should:
1. Open with context (why this matters, how it connects to previous sections)
2. Develop the core content with narrative progression
3. Include visuals at decision points or complex concepts
4. Close with implications for the next topic or practical takeaways

Avoid the pattern of: heading, bullet list, heading, bullet list. Instead: heading, introductory paragraph, developed content with embedded specifics, transitional conclusion.

---

@.claude/project-info.md
