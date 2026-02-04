# Pathfinder - Project Information

## Project Overview

Pathfinder is a handheld multi-sensor fluxgate gradiometer for rapid geophysical reconnaissance. It identifies magnetic anomalies in 5-10 minutes so detailed investigation systems can be deployed efficiently.

### Key Specifications
- Swath width: 1.5-2.0 m (4 gradiometer pairs) (Target)
- Detection depth: 0.5-1.5 m (Target)
- System weight: <1.5 kg (Target)
- Build cost: <$900 (Target)
- Coverage rate: >3,000 m^2/hour (Target)

### Physical Configuration
The design follows the proven "trapeze" configuration used by commercial systems like the Bartington Grad601, extended to 4 sensor pairs for wider swath coverage:
- Carbon fiber or aluminum crossbar (1.5-2.0 m)
- 4 vertically-separated sensor pairs at 50 cm spacing
- Harness-based suspension (backpack straps + bungee isolation)
- Bottom sensors ~15-20 cm above ground

## Essential Commands

### Firmware (PlatformIO)
```bash
cd firmware && pio run           # Build
cd firmware && pio run -t upload # Upload to Arduino Nano
cd firmware && pio test          # Run tests
```

### Data Processing
```bash
python firmware/tools/visualize_data.py PATH0001.CSV        # Time series
python firmware/tools/visualize_data.py PATH0001.CSV --map  # Spatial map
```

## Directory Structure
```
Pathfinder/
├── README.md              # Project overview
├── VISION.md              # Goals and constraints
├── CLAUDE.md              # Style guide
├── docs/
│   └── design-concept.md  # Technical design
├── hardware/
│   ├── schematics/        # Circuit diagrams
│   ├── cad/               # Frame design
│   └── bom/               # Bills of materials
├── firmware/              # Arduino/PlatformIO code
│   ├── src/main.cpp       # Main firmware
│   ├── include/config.h   # Configuration
│   └── tools/             # Python utilities
├── research/              # Background research
└── .claude/               # Memory bank
```

## Technology Stack
- **Firmware**: Arduino Nano, PlatformIO
- **Sensors**: FG-3+ fluxgate magnetometers (8 sensors, 4 pairs)
- **ADC**: 2x ADS1115 16-bit I2C
- **GPS**: NEO-6M module
- **Data**: SD card CSV logging
- **Visualization**: Python (matplotlib, pandas)

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

## Memory Bank Policy

### Repository-Level Memory Bank
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

## Related Projects

Pathfinder complements but does not depend on:
- **HIRT**: Crosshole tomography for detailed 3D imaging
- **OhmPi**: Open-source resistivity meter
- **FG Sensors**: DIY fluxgate kits
