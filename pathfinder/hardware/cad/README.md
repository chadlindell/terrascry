# Pathfinder CAD Files

3D models and rendering tools for the Pathfinder gradiometer frame and harness system.

## Directory Structure

```
cad/
├── openscad/
│   └── pathfinder_frame.scad    # Parametric 3D model
├── scripts/
│   └── render_views.sh          # Batch PNG rendering
├── stl/                         # Generated STL files
└── frame-design.md              # Design specifications
```

## OpenSCAD Model

The parametric model in `openscad/pathfinder_frame.scad` includes:

- Carbon fiber crossbar (2000mm x 25mm)
- 4 PVC drop tubes (500mm x 21.3mm)
- 8 FG-3+ fluxgate sensors (4 pairs)
- Center mount D-ring for harness attachment

### Render Modes

Set the `part` variable to select a view:

| Mode | Description |
|------|-------------|
| `assembled` | Complete trapeze system |
| `exploded` | Components separated with gaps |
| `cross_section` | Half-section showing tube interiors |
| `deployment` | With operator silhouette for scale |
| `single_tube` | Single sensor pair detail |

### Parametric Variables

Key dimensions can be adjusted:

```scad
CROSSBAR_LENGTH = 2000;        // mm
CROSSBAR_OD = 25;              // mm
DROP_TUBE_LENGTH = 500;        // mm
BOTTOM_SENSOR_HEIGHT = 175;    // mm from ground
SENSOR_POSITIONS = [250, 750, 1250, 1750];  // mm from left
```

## Rendering

### Prerequisites

- OpenSCAD 2021.01+
- `xvfb-run` for headless rendering (Linux)

### Generate PNG Images

```bash
# Render all views
./scripts/render_views.sh all

# Render single view
./scripts/render_views.sh assembled
./scripts/render_views.sh exploded

# Export STL file
./scripts/render_views.sh --stl
```

Output files are saved to `docs/assets/images/assembly/`.

### Manual Rendering

```bash
# Open in OpenSCAD GUI
openscad openscad/pathfinder_frame.scad

# Command-line rendering
openscad -D 'part="assembled"' \
  --imgsize=1200,900 \
  --camera=0,0,0,55,25,25,4000 \
  -o pathfinder_assembled.png \
  openscad/pathfinder_frame.scad
```

## Python Diagrams

Annotated assembly diagrams using matplotlib are in `docs/diagrams/`:

```python
from docs.diagrams.assembly import create_exploded_view

fig = create_exploded_view()
fig.savefig('exploded.png', dpi=150)
```

Or render all diagrams:

```bash
python -m docs.diagrams.assembly
```

## Design Specifications

See `frame-design.md` for complete specifications including:

- Component selection (crossbar, tubes, harness)
- Parts list with costs
- Assembly instructions
- Safety notes
