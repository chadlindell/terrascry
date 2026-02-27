# Image Generation Prompts for HIRT Project

This file contains detailed prompts for generating images using nanobanana or other AI image generation tools. Each prompt is designed to create a specific image needed for the HIRT project documentation.

## How to Use

1. Copy the prompt text for the image you need
2. Paste into nanobanana or your preferred image generator
3. Adjust style/parameters as needed (technical illustration, diagram, photo-realistic, etc.)
4. Save generated images to appropriate directories:
   - System diagrams → `images/system-diagrams/`
   - Field deployment → `images/field-deployment/`
   - Assembly photos → `images/assembly-photos/`
   - Calibration/testing → `images/calibration-testing/`

---

## System Architecture Images

### Prompt 1: Complete HIRT System Architecture Diagram
**File:** `images/system-diagrams/hirt-system-architecture.png`
**Style:** Technical diagram, clean lines, professional

```
Create a technical diagram showing the complete HIRT (Hybrid Inductive-Resistive Tomography) system architecture. The image should show:

- A central base hub/control unit in a weatherproof enclosure
- 20 probe units arranged in a grid pattern (4x5 grid, 2 meter spacing)
- Each probe should show: cylindrical rod inserted into ground, electronics pod at top, cable running to base hub
- Base hub should show: battery, current source, data logger/tablet, sync distribution
- Cables connecting probes to base hub (power and data)
- Field tablet/computer connected to base hub
- Style: Clean technical illustration, side view showing probes in ground, top view inset showing grid layout
- Color scheme: Professional blue/gray tones, probes in orange/red for visibility
- Labels: "Base Hub", "Probe Array", "Field Tablet", "Power/Data Cables"
- Background: Light gray or white, minimal
```

### Prompt 2: Single Probe Cross-Section Detail
**File:** `images/system-diagrams/probe-cross-section.png`
**Style:** Technical cutaway diagram, detailed

```
Create a detailed technical cross-section diagram of a single HIRT probe showing:

- Cylindrical fiberglass rod (25mm diameter, 1-3 meters long)
- Probe head/nose capsule at bottom (30mm diameter, 100mm long) containing:
  - TX coil (transmit coil) wound on ferrite core
  - RX coil (receive coil) wound on ferrite core, orthogonal to TX
  - Electronics PCB with components (DDS, amplifiers, ADC, MCU)
  - Potting material filling capsule
- Two ERT ring electrodes mounted on rod (at 0.5m and 1.5m from tip)
- Cable running up center of rod with strain reliefs
- Waterproof seals and cable glands
- Labels showing: "TX Coil", "RX Coil", "Electronics Pod", "ERT Rings", "Cable", "Rod"
- Style: Technical cutaway illustration, half-section view
- Color: Gray rod, colored coils (blue TX, green RX), brown electronics pod, silver rings
- Background: White, minimal
```

### Prompt 3: Probe Electronics Block Diagram
**File:** `images/system-diagrams/probe-electronics-block.png`
**Style:** Electronic block diagram, schematic style

```
Create an electronic block diagram showing the internal electronics of a HIRT probe:

- MCU (ESP32) at center
- MIT subsystem: DDS generator → TX driver → TX coil, RX coil → preamp → instrumentation amp → ADC → MCU
- ERT subsystem: Ring electrodes → multiplexer → differential amp → ADC → MCU
- Power: Input → regulators → 3.3V and 5V rails
- Communications: MCU → RS485/Ethernet interface → cable
- Sync: Clock input → MCU
- Style: Clean electronic block diagram, boxes for components, arrows showing signal flow
- Color: Blue for digital, red for analog, green for power, yellow for communications
- Labels: Component names (DDS, ADC, MCU, etc.)
- Background: White
```

---

## Field Deployment Images

### Prompt 4: Field Deployment Grid Layout (Top View)
**File:** `images/field-deployment/grid-layout-top-view.png`
**Style:** Technical diagram, aerial view

```
Create a top-down view diagram showing a 10x10 meter field deployment grid:

- 20 probes arranged in 4x5 grid (2 meter spacing)
- Each probe shown as small circle with ID number (P01-P20)
- Grid lines showing 2-meter spacing
- Coordinate system (0-10m on both axes)
- Base hub shown at edge of grid
- Cables running from probes to base hub (organized routing)
- Measurement paths shown as dashed lines (TX→RX pairs)
- Labels: "10m", "2m spacing", "Base Hub", probe IDs
- Style: Clean technical diagram, aerial/satellite view style
- Color: Green/brown ground, blue probes, red base hub, gray cables
- Background: Light earth tones
```

### Prompt 5: Field Deployment Side View (Cross-Section)
**File:** `images/field-deployment/deployment-side-view.png`
**Style:** Technical cross-section, educational

```
Create a side-view cross-section showing probes deployed in ground:

- Ground surface at top (sandy/loamy soil texture)
- 5 probes inserted vertically into ground (3 meters deep)
- Each probe showing: rod, electronics pod at bottom, ERT rings visible
- Measurement paths shown as curved lines between probes (magnetic field paths)
- Depth markers: 0m, 1m, 2m, 3m, 4m, 5m
- Target objects shown as darker shapes in ground (metallic objects, disturbed soil)
- Labels: "Surface", "Probe Depth 3m", "Measurement Path", "Target Depth"
- Style: Technical cross-section illustration
- Color: Brown/tan soil, gray probes, blue measurement paths, dark targets
- Background: Light sky blue above ground, darker below
```

### Prompt 6: Field Team at Work
**File:** `images/field-deployment/field-team.png`
**Style:** Realistic illustration or photo-style

```
Create an illustration showing a field team deploying HIRT probes:

- 2-3 people in field setting (archaeological/forensic context)
- One person inserting probe into ground using pilot rod/driver
- Another person marking probe positions with flags
- Third person operating tablet/computer at base hub
- Probes visible in background (some inserted, some ready)
- Grid markers/flags showing layout
- Professional, respectful atmosphere (archaeological work)
- Style: Realistic illustration or photo-realistic
- Setting: Outdoor, sandy/loamy soil, clear day
- Color: Natural earth tones, professional field gear
- Mood: Professional, methodical, scientific
```

---

## Measurement Geometry Images

### Prompt 7: MIT Measurement Pattern Visualization
**File:** `images/system-diagrams/mit-measurement-pattern.png`
**Style:** Technical diagram, 3D visualization

```
Create a 3D-style technical diagram showing MIT (Magneto-Inductive Tomography) measurement patterns:

- Grid of 9 probes (3x3) shown from isometric view
- One probe highlighted as TX (transmitter) with magnetic field lines radiating outward
- Other probes shown as RX (receivers) with signal strength indicated by color/intensity
- Magnetic field lines shown as curved paths between TX and RX probes
- Conductive target object shown as dark shape affecting field lines (distortion visible)
- Depth shown with probes extending into ground
- Labels: "TX Probe", "RX Probes", "Magnetic Field", "Conductive Target"
- Style: 3D technical visualization, clean lines
- Color: Blue for field lines, red for TX, green for RX, dark gray for target
- Background: Light gradient (sky to ground)
```

### Prompt 8: ERT Current Injection Pattern
**File:** `images/system-diagrams/ert-current-pattern.png`
**Style:** Technical diagram, electrical visualization

```
Create a technical diagram showing ERT (Electrical Resistivity Tomography) current injection patterns:

- Grid of probes (4x5) viewed from top
- Two probes highlighted as current injection points (+ and -)
- Current flow paths shown as colored lines/contours between injection points
- Voltage measurement points shown at other probes
- Equipotential lines shown as curved contours
- Resistivity variations shown as color gradients (high = red, low = blue)
- Labels: "Current Injection +", "Current Injection -", "Voltage Measurement", "Equipotential Lines"
- Style: Technical electrical diagram, contour map style
- Color: Red/blue for resistivity, yellow for current paths, green for measurement points
- Background: White or light gray
```

### Prompt 9: Depth Sensitivity Visualization
**File:** `images/system-diagrams/depth-sensitivity.png`
**Style:** Technical 3D diagram, gradient visualization

```
Create a 3D technical diagram showing depth sensitivity of HIRT system:

- Side view showing 3 probes inserted into ground (3 meters deep)
- Sensitivity volume shown as colored gradient (high sensitivity = bright, low = dim)
- High sensitivity zone near probes (0-2m depth)
- Medium sensitivity zone (2-4m depth)
- Lower sensitivity zone (4-6m depth)
- Sensitivity decreases with distance from probes
- Depth markers: 0m, 1m, 2m, 3m, 4m, 5m, 6m
- Labels: "High Sensitivity", "Medium Sensitivity", "Low Sensitivity", "Probe Depth"
- Style: 3D gradient visualization, technical
- Color: Bright yellow/orange for high sensitivity, fading to blue/purple for low
- Background: Dark below ground, light above
```

---

## Assembly and Construction Images

### Prompt 10: Probe Assembly Exploded View
**File:** `images/assembly-photos/probe-exploded-view.png`
**Style:** Technical exploded diagram, IKEA-style

```
Create an exploded view technical diagram of HIRT probe assembly:

- All components separated and labeled:
  - Nose capsule (electronics pod) at bottom
  - TX coil and RX coil (separate, showing windings)
  - Electronics PCB with components visible
  - Rod sections (1m each, 2-3 sections)
  - Threaded couplers between sections
  - ERT ring electrodes (2-3 rings)
  - Cable with connectors
  - Seals, O-rings, cable glands
- Arrows showing assembly order
- Labels for each component
- Style: Technical exploded view, clean lines
- Color: Different colors for different component types
- Background: White
```

### Prompt 11: Coil Winding Detail
**File:** `images/assembly-photos/coil-winding-detail.png`
**Style:** Technical detail drawing

```
Create a detailed technical drawing showing coil winding for HIRT probe:

- Ferrite rod core (10mm diameter, 100mm long)
- Magnet wire (34 AWG) wound around core in neat layers
- Close-up showing wire turns and spacing
- Two coils shown: TX coil and RX coil (orthogonal orientation)
- Dimensions labeled: wire gauge, number of turns, inductance
- Winding pattern clearly visible
- Labels: "Ferrite Core", "Magnet Wire", "TX Coil", "RX Coil", "200-300 Turns"
- Style: Technical detail drawing, precise
- Color: Gray core, copper-colored wire, clear separation
- Background: White
```

### Prompt 12: Electronics PCB Layout
**File:** `images/assembly-photos/pcb-layout.png`
**Style:** PCB design view, technical

```
Create a PCB layout diagram showing probe electronics board:

- Small circular or rectangular board (fits in 30mm capsule)
- Components arranged and labeled:
  - MCU (ESP32) at center
  - DDS chip (AD9833)
  - Op-amps (TX driver, RX preamp, instrumentation amp)
  - ADC (ADS1256)
  - Multiplexer (CD4051)
  - Voltage regulators
  - Connectors for coils, ERT rings, power, communications
- Traces shown (power in red, ground in blue, signals in green)
- Component outlines and pin numbers visible
- Labels: Component designators (U1, U2, etc.) and values
- Style: PCB layout view, technical
- Color: Green PCB, colored traces, component outlines
- Background: White or dark (as PCB design software)
```

---

## Use Case Scenario Images

### Prompt 13: Bomb Crater Deployment
**File:** `images/field-deployment/bomb-crater-scenario.png`
**Style:** Realistic illustration, scenario-specific

```
Create an illustration showing HIRT probe deployment at a WWII bomb crater site:

- Large bomb crater (10-15m diameter, 3m deep) with filled/settled appearance
- Probes arranged around crater rim and inside crater
- Some probes inserted deeper (3m) near crater center
- Base hub positioned at safe distance
- Team members visible (small scale) for context
- UXO clearance markers/flags visible (safety context)
- Archaeological grid markers
- Labels: "Crater Rim", "Crater Center", "Probe Array", "Base Hub"
- Style: Realistic illustration, professional
- Color: Earth tones, crater darker, probes visible
- Background: Field setting, clear day
- Mood: Professional, careful, archaeological
```

### Prompt 14: Woods Burial Site Deployment
**File:** `images/field-deployment/woods-burial-scenario.png`
**Style:** Realistic illustration, respectful

```
Create a respectful illustration showing HIRT probe deployment at a woods burial site:

- Forest setting with trees in background
- Smaller grid (8x8m) with tighter spacing (1-1.5m)
- Probes inserted to 1.5m depth (shorter than crater scenario)
- Minimal ground disturbance visible
- Archaeological markers/flags
- Base hub at edge of grid
- Professional, respectful atmosphere
- Labels: "8x8m Grid", "1.5m Probe Depth", "Minimal Intrusion"
- Style: Realistic illustration, respectful tone
- Color: Natural forest colors, earth tones
- Background: Forest setting, dappled light
- Mood: Respectful, careful, archaeological
```

### Prompt 15: Swamp/Margin Deployment
**File:** `images/field-deployment/swamp-margin-scenario.png`
**Style:** Technical illustration, challenging terrain

```
Create an illustration showing HIRT probe deployment at swamp/margin site:

- Water/marsh in background
- Probes deployed from accessible dry land/margin
- Some probes angled toward water
- Longer baselines shown (2-3m spacing)
- Base hub on dry ground
- Cables routed carefully to avoid water
- Labels: "Water Margin", "Accessible Ground", "Extended Baselines"
- Style: Technical illustration, challenging terrain
- Color: Water blues/greens, earth tones for dry ground
- Background: Swamp/marsh setting
- Mood: Challenging but methodical
```

---

## Calibration and Testing Images

### Prompt 16: Calibration Test Setup
**File:** `images/calibration-testing/calibration-setup.png`
**Style:** Technical photo-style illustration

```
Create an illustration showing probe calibration test setup:

- Two probes positioned in test area (lab or outdoor)
- Aluminum plate or steel bar positioned between probes (test target)
- Measurement equipment visible (oscilloscope, multimeter, signal generator)
- Cables connecting probes to test equipment
- Calibration sheet/notebook visible
- Test target clearly labeled
- Labels: "Test Target", "Probe 1", "Probe 2", "Test Equipment"
- Style: Technical photo-style illustration
- Color: Lab equipment colors, clear test setup
- Background: Lab or outdoor test area
- Mood: Precise, scientific, methodical
```

### Prompt 17: Field Quality Control Check
**File:** `images/calibration-testing/field-qc-check.png`
**Style:** Realistic field illustration

```
Create an illustration showing field quality control procedures:

- Field technician checking probe connections
- Tablet/computer showing data quality plots
- Probes in background (some deployed)
- QC checklist visible
- Measurement verification in progress
- Labels: "QC Check", "Data Verification", "Reciprocity Test"
- Style: Realistic field illustration
- Color: Natural field colors, equipment visible
- Background: Field deployment site
- Mood: Careful, methodical, quality-focused
```

---

## Conceptual and Overview Images

### Prompt 18: HIRT System Concept Overview
**File:** `images/system-diagrams/hirt-concept-overview.png`
**Style:** Conceptual diagram, educational

```
Create a conceptual overview diagram explaining HIRT system:

- Three-panel illustration:
  - Left: Surface methods (GPR, magnetometry) with limitations shown
  - Center: HIRT probe array showing in-ground sensors
  - Right: 3D subsurface model/results
- Arrows showing: Problem → Solution → Results
- Key advantages highlighted: "True Tomography", "Deeper Penetration", "Dual Method"
- Labels: "Surface Methods", "HIRT Probe Array", "3D Subsurface Model"
- Style: Conceptual, educational, clean
- Color: Blue for surface methods, orange/red for HIRT, green for results
- Background: White or light gradient
- Mood: Educational, innovative, solution-oriented
```

### Prompt 19: Measurement Comparison (MIT vs ERT)
**File:** `images/system-diagrams/mit-vs-ert-comparison.png`
**Style:** Technical comparison diagram

```
Create a side-by-side comparison diagram showing MIT and ERT measurement methods:

- Left side: MIT (Magneto-Inductive)
  - TX coil generating magnetic field
  - RX coil receiving attenuated signal
  - Conductive target affecting field
  - Labels: "Magnetic Field", "Eddy Currents", "Conductive Targets"
  
- Right side: ERT (Electrical Resistivity)
  - Current injection between electrodes
  - Voltage measurement at other electrodes
  - Resistivity variations shown
  - Labels: "Current Injection", "Voltage Measurement", "Resistivity"
  
- Bottom: Combined results showing complementary information
- Style: Technical comparison, clear separation
- Color: Blue for MIT, red for ERT, combined colors for results
- Background: White
- Mood: Technical, comparative, educational
```

### Prompt 20: Data Processing Pipeline Concept
**File:** `images/system-diagrams/data-pipeline-concept.png`
**Style:** Flow diagram, conceptual

```
Create a conceptual flow diagram showing HIRT data processing pipeline:

- Left: Field data collection (probes, measurements)
- Center: Data processing steps:
  - Raw data → QC/validation
  - MIT inversion → conductivity volume
  - ERT inversion → resistivity volume
  - Data fusion → combined model
- Right: Visualization (3D models, depth slices, isosurfaces)
- Arrows showing data flow
- Labels: "Field Data", "Processing", "Visualization", "3D Model"
- Style: Flow diagram, conceptual, clean
- Color: Blue for data, green for processing, orange for results
- Background: White or light gradient
- Mood: Technical, process-oriented, results-focused
```

---

## Component Library Images

### Prompt 21: Micro Probe Tip Manufacturing Detail
**File:** `images/assembly-photos/micro-probe-tip-print.png`  
**Style:** Technical macro illustration, orthographic + inset detail

```
Create a detailed illustration of the 12 mm OD micro probe tip prepared for home 3D printing:

- Show the tip oriented vertically on a PEI plate with an 8 mm brim and a slim stabilization tower connected by a single-layer tab.
- Annotate printer settings: PETG 245 °C nozzle / 80 °C bed, 0.2 mm layers, 100% infill, 5 perimeters, first-layer speed 15 mm/s.
- Include callouts for threaded bore (M12×1.5), tapered exterior, and epoxy soak step (thin epoxy being wicked into threads after tapping).
- Inset diagram showing the M12×1.5 tap with cutting oil, plus another inset showing flood-coating threads with Loctite Marine Epoxy syringe.
- Style: clean orthographic render with dimension arrows and labels (“8 mm brim”, “stabilizer tower”, “solid infill”, “tap after print”).
- Color palette: neutral grays for part, blue callouts, orange epoxy highlights; white background.
```

### Prompt 22: Rod Coupler Stabilized Print
**File:** `images/assembly-photos/micro-rod-coupler-print.png`  
**Style:** Exploded + callout diagram

```
Generate a diagram showing the M12×1.5 rod coupler printed on its side:

- Depict the coupler lying horizontally with dual 10 mm brims and tree supports only under the chamfered ends.
- Show cross-section revealing 100% infill and 6 perimeters, highlighting O-ring grooves and dual threaded bores.
- Include step-by-step callouts: “Print PETG 245 °C / 80 °C”, “Remove supports warm”, “Run M12×1.5 bottoming tap from each side”, “Brush thin epoxy into threads + grooves”.
- Add small inset of QC check: two fiberglass rods threaded into both ends, confirming alignment.
- Style: semi-realistic CAD render with annotations and sequential numbering, white/gray background with teal callout boxes.
```

### Prompt 23: Surface Junction Box Assembly
**File:** `images/assembly-photos/micro-junction-box.png`  
**Style:** Cutaway technical illustration

```
Illustrate the 25 mm × 35 mm PETG surface junction box printed upright:

- Show body with 10 mm brim, stabilizer tower opposite the cable gland, and threaded interior (M12×1.5) highlighted.
- Exploded view of cap, O-ring, cable gland, terminal block, and strain relief hardware.
- Annotate print specs (100% infill, 5 perimeters), tapping steps (M12×1.5 tap, gland hole ream), epoxy sealing of threads, and water dunk test (vacuum syringe, no bubbles).
- Include labels for cable routing, terminal block positions, and gasket groove.
- Style: technical cutaway with subtle shading, color accents for gaskets/epoxy, background grid.
```

### Prompt 24: ERT Ring Collar Production Batch
**File:** `images/assembly-photos/ert-ring-collar-batch.png`  
**Style:** Process storyboard, top-down perspective

```
Create a storyboard showing multiple ERT ring collars being produced:

- Panel 1: Slicer preview with collars nested flat, 5 mm brim, 0.16 mm layers, 100% infill, monotonic top layers highlighted.
- Panel 2: Finished print on bed after cooling, with notes “allow to cool to room temp to preserve 12.00 mm ID”.
- Panel 3: Deburring and bore check using a 12.00 mm gauge rod and 1/16" drill for wire channel cleanup.
- Panel 4: Installation sequence—scuffed fiberglass rod, epoxy applied, collar rotated into position, stainless strip wrapped and overlapped 3 mm, heat-shrink covering the joint.
- Style: clean vector panels with numbered captions, teal/orange highlight colors, white background.
```

---

## Notes for Image Generation

- **Style Consistency:** Maintain consistent style across related images (technical diagrams vs. realistic illustrations)
- **Color Scheme:** Use professional color schemes (blues, grays, earth tones) with highlights for important elements
- **Labels:** Ensure all labels are clear and readable
- **Resolution:** Generate at high resolution (at least 1920x1080, preferably 4K for detailed technical diagrams)
- **Format:** PNG with transparency for diagrams, JPEG for photo-style images
- **File Naming:** Use descriptive names matching the prompts above

---

## Adding New Prompts

When adding new image prompts:
1. Number sequentially (Prompt 21, 22, etc.)
2. Include: File path, Style description, Detailed prompt
3. Specify: Intended use, target audience, style requirements
4. Update this file with the new prompt

---

*Last Updated: 2025-11-16*
*Total Prompts: 24*

