# Pathfinder Project Context

Pathfinder is a handheld multi-sensor fluxgate gradiometer for rapid geophysical reconnaissance. It serves as a companion tool to HIRT, providing rapid pre-screening before detailed tomographic investigation.

## Quick Reference

### Key Commands
- Build firmware: `cd firmware && pio run`
- Upload firmware: `cd firmware && pio run -t upload`
- Run tests: `cd firmware && pio test`

## Project Structure
```
Pathfinder/
├── README.md          # Project overview
├── VISION.md          # Goals and constraints
├── CLAUDE.md          # This file
├── docs/              # Documentation
│   └── design-concept.md  # Current design
├── hardware/          # Hardware documentation
│   ├── schematics/    # Circuit diagrams
│   ├── cad/           # 3D models (harness mounts, etc.)
│   └── bom/           # Bills of materials
├── firmware/          # Arduino/PlatformIO code
└── research/          # Background research
```

## Terminology

- **Gradiometer**: Pair of vertically-separated magnetometers measuring field gradient
- **Fluxgate**: Type of magnetometer sensor used in this design
- **Trapeze**: The horizontal bar carrying sensors, suspended from harness
- **Swath**: Width of ground covered in single pass

## Design Principles

1. **Harness-first**: Arms never bear weight
2. **Speed over resolution**: Good enough fast beats perfect slow
3. **Field-rugged**: No carts, no flat-ground assumptions
4. **DIY-accessible**: Under $1000, globally-available components

## Relationship to HIRT

Pathfinder is a **screening tool** that identifies where to deploy HIRT:

```
Pathfinder (5-10 min)     HIRT (60-90 min)
    Rapid scan       -->   Detailed tomography
    Find anomalies   -->   Characterize targets
    Wide area        -->   Focused grid
```

## Writing Style

Follow HIRT conventions:
- Specifications qualified as (Measured), (Modeled), or (Target)
- Active voice preferred
- Present tense for describing how things work
- First person plural for design decisions
