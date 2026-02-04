# Pathfinder Project Context

Pathfinder is a handheld multi-sensor fluxgate gradiometer for rapid geophysical reconnaissance.

## Quick Reference

### Key Commands
- Build firmware: `cd firmware && pio run`
- Upload firmware: `cd firmware && pio run -t upload`
- Run tests: `cd firmware && pio test`
- Visualize data: `python firmware/tools/visualize_data.py <file.csv>`
- Spatial map: `python firmware/tools/visualize_data.py <file.csv> --map`

### Memory Bank
- `/memory:save` - Save context to `.claude/memory-bank/`
- `/memory:recall` - Retrieve saved context
- `/memory:list` - List all memories

### RIPER Workflow
Use `/riper:strict` to enable mode tracking. Available modes:
- `/riper:research` - Information gathering (read-only)
- `/riper:innovate` - Brainstorming approaches
- `/riper:plan` - Create specifications (saved to memory bank)
- `/riper:execute` - Implement approved plans
- `/riper:review` - Validate implementation

## Project Structure
```
Pathfinder/
├── README.md              # Project overview
├── VISION.md              # Goals and constraints
├── CLAUDE.md              # This file - style guide
├── docs/
│   └── design-concept.md  # Technical design
├── hardware/
│   ├── schematics/        # Circuit diagrams
│   ├── cad/               # Frame design (harness mounts, etc.)
│   └── bom/               # Bills of materials
├── firmware/              # Arduino/PlatformIO code
│   ├── src/main.cpp       # Main firmware
│   ├── include/config.h   # Configuration
│   └── tools/             # Python utilities
├── research/              # Background research
└── .claude/               # Memory bank
```

---

## Writing Style Guide

### Performance Claims Framework
All specifications must be qualified with their validation status:

| Qualifier | Usage | Example |
|-----------|-------|---------|
| **(Measured)** | Bench/field tested | "Bench tests show 50 nT noise floor" |
| **(Modeled)** | Theoretical estimate | "Detection depth estimated at 1.5 m" |
| **(Target)** | Design goal, not yet validated | "Target weight <1.5 kg" |

### Terminology
Maintain consistency throughout:
- **Gradiometer**: Pair of vertically-separated magnetometers measuring field gradient
- **Fluxgate**: Type of magnetometer sensor used in this design
- **Trapeze**: The horizontal bar carrying sensors, suspended from harness
- **Swath**: Width of ground covered in single pass
- **Top sensor**: Reference sensor (~50 cm above ground)
- **Bottom sensor**: Signal sensor (~15-20 cm above ground)

### Voice and Tone
- Active voice preferred ("The sensor measures..." not "The field is measured by...")
- First person plural for design decisions ("We selected..." not "It was decided...")
- Present tense for describing how the system works
- Past tense for describing the development process or experiments

### Design Principles
1. **Harness-first**: Arms never bear weight
2. **Speed over resolution**: Good enough fast beats perfect slow
3. **Field-rugged**: No carts, no flat-ground assumptions
4. **DIY-accessible**: Under $1000, globally-available components

---

## Related Projects
- **HIRT**: Crosshole tomography for detailed 3D imaging (complementary tool)
- **OhmPi**: Open-source resistivity meter
- **FG Sensors**: DIY fluxgate kits

@.claude/project-info.md
