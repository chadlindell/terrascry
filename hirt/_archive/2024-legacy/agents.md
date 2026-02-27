# HIRT Project - Agent Documentation

## Project Overview

**Hybrid Inductive-Resistive Tomography (HIRT)** is a modular, in-ground probe array system for high-resolution 3D subsurface imaging in archaeological and forensic contexts. This project documents the complete design, construction, and deployment of a DIY probe-array system for WWII aircraft crash sites and potential graves.

**Key Characteristics:**
- Dual-method: MIT (Magneto-Inductive Tomography) + ERT (Electrical Resistivity Tomography)
- Modular: 20-24 identical dual-role probes
- Low-cost: $1,800-3,900 complete starter kit
- Field-ready: Designed for archaeological/forensic teams
- DIY-friendly: Complete documentation for construction

---

## Project Structure

```
HIRT/
├── README.md                          # Main project overview
├── agents.md                          # This file - agent documentation
├── IMAGE_GENERATION_PROMPTS.md        # Prompts for AI image generation (nanobanana)
├── Makefile                           # PDF generation from Markdown (pandoc)
├── .gitignore                         # Git ignore patterns
│
├── docs/                              # Main documentation
│   ├── README.md                      # Documentation index
│   ├── whitepaper/                   # White paper (v0.9)
│   │   ├── main.md                   # Main document with TOC
│   │   ├── sections/                 # 19 individual sections
│   │   │   ├── 01-scope.md
│   │   │   ├── 02-ethics-legal-safety.md
│   │   │   ├── 03-concept.md
│   │   │   ├── 04-physics.md
│   │   │   ├── 05-system-architecture.md
│   │   │   ├── 06-bom.md
│   │   │   ├── 07-mechanical-build.md
│   │   │   ├── 08-electronics.md
│   │   │   ├── 09-calibration.md
│   │   │   ├── 10-field-deployment.md
│   │   │   ├── 11-data-spec.md
│   │   │   ├── 12-interpretation.md
│   │   │   ├── 13-troubleshooting.md
│   │   │   ├── 14-cost-build-plan.md
│   │   │   ├── 15-scenario-playbooks.md
│   │   │   ├── 16-optional-addons.md
│   │   │   ├── 17-field-checklists.md
│   │   │   ├── 18-glossary.md
│   │   │   └── 19-next-steps.md      # Software development (future)
│   │   ├── assets/                    # Images for whitepaper (empty, ready for images)
│   │   └── pdf/                      # Generated PDFs (gitignored)
│   │
│   └── field-guide/                  # Field-ready documentation
│       ├── quick-reference.md         # One-page field reference
│       ├── coil-winding-recipe.md     # Detailed coil specifications
│       ├── ert-source-schematic.md    # ERT current source design
│       └── field-operation-manual.md  # Complete field deployment guide
│
├── hardware/                          # Hardware documentation
│   ├── bom/                          # Bill of Materials
│   │   ├── probe-bom.md              # Per-probe BOM (with part numbers)
│   │   ├── base-hub-bom.md           # Base hub BOM (with part numbers)
│   │   ├── shared-components-bom.md  # Tools and supplies BOM
│   │   ├── PROCUREMENT.md            # Procurement workflow guide
│   │   └── order-sheets/             # CSV files for ordering
│   │       ├── probe-order-sheet.csv
│   │       ├── base-hub-order-sheet.csv
│   │       └── complete-kit-order-sheet.csv
│   │
│   ├── schematics/                   # Circuit and mechanical schematics
│   │   ├── electronics/              # Electronic circuit designs
│   │   │   ├── probe-electronics-block.md    # Complete system block diagram
│   │   │   ├── mit-circuit.md                # MIT circuit (DETAILED - complete)
│   │   │   ├── ert-circuit.md                # ERT circuit (DETAILED - complete)
│   │   │   └── base-hub-circuit.md           # Base hub circuit (DETAILED - complete)
│   │   └── mechanical/               # Mechanical specifications
│   │       ├── probe-assembly.md     # Assembly instructions (placeholder)
│   │       ├── rod-specifications.md # Rod specs (DETAILED - complete)
│   │       └── er-ring-mounting.md   # ERT ring mounting (placeholder)
│   │
│   ├── drawings/                     # Technical drawings (placeholders)
│   │   ├── probe-head-drawing.md     # Probe head CAD (placeholder)
│   │   └── assembly-drawings.md      # Assembly drawings (placeholder)
│   └── cad/                          # CAD files for manufacturing
│       ├── openscad/                 # OpenSCAD source files (.scad)
│       ├── stl/                      # 3D printable STL files (generated)
│       ├── step/                     # CNC-ready STEP files (generated)
│       └── docs/                     # Manufacturing documentation
│
├── build/                            # Build and testing documentation
│   ├── assembly-guide.md             # Basic assembly guide (placeholder)
│   ├── assembly-guide-detailed.md    # DETAILED assembly guide (complete)
│   ├── calibration-procedures.md    # Calibration procedures (basic)
│   ├── qc-checklist.md              # Quality control checklist
│   └── testing-procedures.md         # Comprehensive testing procedures (complete)
│
└── images/                           # Image directories (ready for images)
    ├── system-diagrams/
    │   ├── README.md
    │   └── measurement-geometry.md   # Measurement geometry diagrams (ASCII)
    ├── field-deployment/
    │   └── README.md
    ├── assembly-photos/
    │   └── README.md
    └── calibration-testing/
        └── README.md
```

---

## Document Status

### ✅ Complete and Detailed

**Documentation:**
- ✅ White paper (19 sections) - Complete, comprehensive
- ✅ Field operation manual - Complete, detailed procedures
- ✅ Testing procedures - Complete, comprehensive
- ✅ Assembly guide (detailed) - Complete, step-by-step
- ✅ Procurement guide - Complete workflow
- ✅ Quick reference - One-page field guide
- ✅ Coil winding recipe - Detailed specifications
- ✅ ERT source schematic - Design details

**Hardware Design:**
- ✅ MIT circuit schematic - Complete with component values, calculations
- ✅ ERT circuit schematic - Complete with component values, calculations
- ✅ Base hub circuit schematic - Complete with component values, calculations
- ✅ Probe electronics block diagram - Complete system architecture
- ✅ Rod specifications - Complete dimensions, threads, materials
- ✅ Measurement geometry diagrams - ASCII diagrams complete

**BOM and Procurement:**
- ✅ Probe BOM - Complete with specific part numbers
- ✅ Base hub BOM - Complete with specific part numbers
- ✅ Shared components BOM - Complete
- ✅ CSV order sheets - Ready for procurement (with part numbers)

### ⚠️ Placeholder/Incomplete

**Mechanical Drawings:**
- ⚠️ Probe head drawing - Placeholder (needs CAD drawings)
- ⚠️ Assembly drawings - Placeholder (needs technical drawings)
- ⚠️ ERT ring mounting - Placeholder (needs detailed drawings)

**Assembly Documentation:**
- ⚠️ Basic assembly guide - Placeholder (detailed version exists)
- ⚠️ Calibration procedures - Basic (could be expanded)

**Images:**
- ⚠️ All image directories - Empty (ready for generated images)
- ⚠️ See IMAGE_GENERATION_PROMPTS.md for 20 detailed prompts

**Software:**
- ⚠️ Software development - Explicitly deferred (see Section 19)

---

## Key Design Decisions

### System Architecture

1. **Dual-Role Probes:** Each probe performs both MIT (TX/RX) and ERT functions
   - Simplifies logistics
   - Reduces cost
   - Improves data quality (consistent calibration)

2. **Digital Lock-in (Option A):** Chosen over analog lock-in
   - More flexible
   - Software-configurable
   - Better performance
   - Requires 24-bit ADC (ADS1256)

3. **Wired Communication:** RS485 over CAT5 recommended
   - Reliable
   - Low latency
   - No power for radios
   - Wireless option available but optional

4. **Modular Design:** Identical probes, modular base hub
   - Easy to repair
   - Scalable (add more probes)
   - Consistent performance

### Component Selections

**MCU:** ESP32 (WiFi/Bluetooth capable)
- Low cost
- Good performance
- Built-in wireless (optional)
- Large community support

**ADC:** ADS1256 (24-bit delta-sigma)
- High resolution
- Good for digital lock-in
- SPI interface
- Shared between MIT and ERT

**DDS:** AD9833
- Low cost
- Good frequency resolution
- SPI interface
- Adequate for 2-50 kHz range

**Op-Amps:** OPA2277, AD620, INA128
- Low noise
- Good performance
- Reasonable cost
- Industry standard parts

### Cost Optimization

- Bulk ordering (20+ probes) reduces cost 10-20%
- Use standard components (not custom)
- DIY-friendly (no specialized tools required)
- Modular (build incrementally)

---

## Navigation Guide

### For Understanding the System

1. **Start Here:** `README.md` - Project overview
2. **System Overview:** `docs/whitepaper/main.md` - Complete white paper
3. **Architecture:** `hardware/schematics/electronics/probe-electronics-block.md` - System block diagram
4. **Physics:** `docs/whitepaper/sections/04-physics.md` - How it works
5. **Measurement Geometry:** `images/system-diagrams/measurement-geometry.md` - Visual diagrams

### For Building the System

1. **BOM Review:** `hardware/bom/probe-bom.md` and `hardware/bom/base-hub-bom.md`
2. **Order Components:** `hardware/bom/PROCUREMENT.md` and CSV files in `hardware/bom/order-sheets/`
3. **Assembly:** `build/assembly-guide-detailed.md` - Step-by-step instructions
4. **Circuit Design:** 
   - `hardware/schematics/electronics/mit-circuit.md`
   - `hardware/schematics/electronics/ert-circuit.md`
   - `hardware/schematics/electronics/base-hub-circuit.md`
5. **Mechanical:** `hardware/schematics/mechanical/rod-specifications.md`
6. **Testing:** `build/testing-procedures.md` - Comprehensive testing

### For Field Deployment

1. **Field Manual:** `docs/field-guide/field-operation-manual.md` - Complete procedures
2. **Quick Reference:** `docs/field-guide/quick-reference.md` - One-page reference
3. **Checklists:** `docs/whitepaper/sections/17-field-checklists.md`
4. **Troubleshooting:** `docs/whitepaper/sections/13-troubleshooting.md`
5. **Scenarios:** `docs/whitepaper/sections/15-scenario-playbooks.md`

### For Image Generation

1. **Prompts:** `IMAGE_GENERATION_PROMPTS.md` - 20 detailed prompts for nanobanana
2. **Image Directories:** `images/` - Organized by category
3. **Place Images:** In appropriate subdirectories based on prompt file paths

---

## Critical Files Reference

### Must-Read Files

1. **`docs/whitepaper/main.md`** - Complete white paper (start here for overview)
2. **`hardware/schematics/electronics/probe-electronics-block.md`** - System architecture
3. **`hardware/bom/probe-bom.md`** - Component list with part numbers
4. **`build/assembly-guide-detailed.md`** - How to build
5. **`build/testing-procedures.md`** - How to test
6. **`docs/field-guide/field-operation-manual.md`** - How to deploy

### Reference Files

- **`docs/whitepaper/sections/18-glossary.md`** - Terminology
- **`docs/whitepaper/sections/13-troubleshooting.md`** - Problem solving
- **`hardware/bom/PROCUREMENT.md`** - Ordering guide
- **`IMAGE_GENERATION_PROMPTS.md`** - Image generation

---

## Design Specifications Summary

### System Specifications

- **Probe Count:** 20-24 probes (standard section)
- **Probe Depth:** 1.5-3.0 m (depending on deployment)
- **Probe Spacing:** 1.0-2.0 m (adjustable)
- **Section Size:** 10×10 m (standard)
- **MIT Frequencies:** 2, 5, 10, 20, 50 kHz (selectable)
- **ERT Current:** 0.5-2 mA (adjustable)
- **Depth Range:** 2-6 m (depending on configuration)
- **Lateral Resolution:** 0.5-1.5 × spacing

### Component Specifications

**Probe Electronics:**
- MCU: ESP32 DevKit
- DDS: AD9833 (25 MHz clock)
- TX Driver: OPA2277 (gain 2-5x)
- RX Preamp: AD620 (gain 10-1000x)
- Inst. Amp: INA128 (gain 10-100x)
- ADC: ADS1256 (24-bit, 30 kS/s)
- ERT Mux: CD4051 (8-channel)

**Base Hub:**
- Current Source: OPA177 + REF5025 (0.5-2 mA)
- Clock: ECS-100-10-30B-TR (10 MHz)
- Clock Buffer: SN74HC244N (octal buffer)
- Communication: MAX485 (RS485) or Ethernet/WiFi
- Power: 12V battery, LM2596 (5V), AMS1117 (3.3V)

**Mechanical:**
- Rod: Fiberglass, 25mm OD, 1m sections
- Couplers: Glass-filled nylon or 3D-printed PETG
- Capsule: 3D-printed, Ø30mm × 100mm
- ERT Rings: Stainless steel, 12mm wide, at 0.5m, 1.5m, 2.5m

---

## Future Work Needed

### High Priority

1. ✅ **CAD Drawings:**
   - ✅ Probe head 3D model (OpenSCAD + STL export)
   - ✅ Rod coupler 3D model (OpenSCAD + STL export)
   - ✅ ERT ring collar 3D model (OpenSCAD + STL export)
   - ⚠️ Assembly drawings with dimensions (still needed)
   - ⚠️ Base hub enclosure design (future)

2. **PCB Design:**
   - Probe electronics PCB layout
   - Base hub PCB layout
   - Gerber files for manufacturing
   - Or detailed perfboard layouts

3. **Firmware Development:**
   - ESP32 firmware for probes
   - Base hub control firmware
   - Communication protocol
   - Data logging software

4. **Image Generation:**
   - Use prompts in `IMAGE_GENERATION_PROMPTS.md`
   - Generate all 20 images
   - Place in appropriate directories
   - Update documentation to reference images

### Medium Priority

5. **Software Development (Section 19):**
   - Data processing pipeline
   - MIT inversion algorithms
   - ERT inversion algorithms
   - Data fusion
   - Visualization tools

6. **Expanded Testing:**
   - Field validation on known targets
   - Environmental testing results
   - Performance characterization
   - Calibration refinement

7. **Documentation Enhancement:**
   - Add photos from actual builds
   - Add field deployment photos
   - Add measurement result examples
   - Expand troubleshooting based on experience

### Low Priority

8. **Optional Add-ons (Section 16):**
   - Borehole radar integration
   - Seismic crosshole
   - Soil ion tests
   - Magnetometer pre-scan

---

## Important Notes for Agents

### File Naming Conventions

- **Markdown files:** Use kebab-case (e.g., `field-operation-manual.md`)
- **CSV files:** Use kebab-case (e.g., `probe-order-sheet.csv`)
- **Image files:** Use descriptive names matching prompts (e.g., `hirt-system-architecture.png`)
- **Section files:** Numbered with zero-padding (e.g., `01-scope.md`)

### Documentation Standards

- **Markdown format:** All documentation in Markdown
- **Code references:** Use `startLine:endLine:filepath` format for existing code
- **New code:** Use standard markdown code blocks with language tags
- **Links:** Use relative paths within project
- **Images:** Reference from `images/` directory or `docs/whitepaper/assets/`

### Component Part Numbers

- **Always include:** Part number, supplier, and notes
- **Format:** Part numbers in BOM files match CSV order sheets
- **Updates:** When updating part numbers, update both BOM and CSV files
- **Alternatives:** Document alternatives in notes

### Circuit Design Notes

- **Component values:** Always specify tolerance (e.g., 0.1%, 1%, 5%)
- **Calculations:** Include design equations and example calculations
- **Power:** Specify power requirements and current draw
- **Interfaces:** Document all interfaces (SPI, I2C, UART, GPIO)

### Testing and Calibration

- **Test procedures:** Document expected values and tolerances
- **Calibration:** Record calibration data in calibration sheets
- **QC:** Use QC checklist before field deployment
- **Documentation:** Keep test results and calibration records

### Field Procedures

- **Safety first:** Always emphasize UXO clearance and permits
- **Ethics:** Maintain professional, respectful approach
- **Documentation:** Record all field conditions and measurements
- **Data backup:** Emphasize frequent data backup

---

## Common Tasks

### Adding New Documentation

1. **Determine location:** Follow existing structure
2. **Create file:** Use appropriate naming convention
3. **Add links:** Update relevant index files (README.md, main.md)
4. **Cross-reference:** Link to related documents
5. **Update:** Update this agents.md if structure changes

### Updating BOM

1. **Update Markdown BOM:** `hardware/bom/probe-bom.md` or `base-hub-bom.md`
2. **Update CSV:** Corresponding file in `hardware/bom/order-sheets/`
3. **Verify consistency:** Part numbers match between files
4. **Update costs:** Recalculate totals if needed
5. **Document changes:** Note reason for changes

### Adding Circuit Details

1. **Component values:** Specify exact values and tolerances
2. **Calculations:** Include design equations
3. **Interfaces:** Document all connections
4. **Power:** Specify power requirements
5. **Cross-reference:** Link to related circuits

### Generating Images

1. **Use prompts:** Copy from `IMAGE_GENERATION_PROMPTS.md`
2. **Generate:** Use nanobanana or other AI image generator
3. **Save:** Place in appropriate `images/` subdirectory
4. **Reference:** Update documentation to reference images
5. **Add prompt:** If creating new image, add prompt to `IMAGE_GENERATION_PROMPTS.md`

### PDF Generation

1. **Requirements:** pandoc and XeLaTeX installed
2. **Generate:** Run `make whitepaper` or `make all`
3. **Output:** PDFs in `docs/whitepaper/pdf/`
4. **Check:** Verify formatting and links work

---

## Project Status Summary

### Completed ✅

- Complete documentation structure
- White paper (19 sections, comprehensive)
- Field operation manual
- Testing procedures
- Detailed assembly guide
- Circuit schematics (MIT, ERT, Base Hub) with component values
- BOM files with specific part numbers
- CSV order sheets ready for procurement
- Procurement workflow guide
- Measurement geometry diagrams
- System block diagrams
- Rod specifications
- Image generation prompts (20 prompts ready)

### In Progress ⚠️

- Mechanical drawings (placeholders exist, need CAD)
- Image generation (prompts ready, images needed)
- Basic assembly guide (detailed version complete)

### Future 🔮

- Software development (explicitly deferred, see Section 19)
- PCB design files
- Firmware development
- Field testing and validation
- Optional add-ons

---

## Key Contacts and Resources

### Documentation References

- **Main entry:** `README.md`
- **Documentation index:** `docs/README.md`
- **White paper:** `docs/whitepaper/main.md`
- **This file:** `agents.md` (you are here)

### External Resources

- **Digi-Key:** www.digikey.com (primary component supplier)
- **Mouser:** www.mouser.com (alternative supplier)
- **McMaster-Carr:** www.mcmaster.com (mechanical components)
- **Component datasheets:** Available from suppliers

### Design Tools

- **PCB Design:** KiCad (free), Eagle, Altium
- **3D CAD:** FreeCAD, OpenSCAD, Fusion 360
- **Circuit Simulation:** LTspice (free), CircuitLab
- **PDF Generation:** pandoc + XeLaTeX (see Makefile)

---

## Version History

- **v0.9** (Current): Complete documentation, detailed circuits, BOMs with part numbers
- **v0.1** (Initial): Basic structure, placeholder files

---

## Notes for Future Development

### When Adding Software

- Create `software/` directory
- Follow structure outlined in Section 19
- Keep hardware and software documentation separate
- Document APIs and interfaces

### When Adding Images

- Use prompts from `IMAGE_GENERATION_PROMPTS.md`
- Place in appropriate `images/` subdirectory
- Reference in documentation
- Update image prompts file if creating new images

### When Updating Costs

- Update BOM files (Markdown)
- Update CSV order sheets
- Update cost summaries in documentation
- Note date of cost update

### When Field Testing

- Document all results
- Update procedures based on experience
- Add photos to `images/field-deployment/`
- Update troubleshooting guide
- Refine calibration procedures

---

*Last Updated: 2024-03-15*
*Project Status: Documentation Complete (v0.9), Hardware Design In Progress*
*Next Steps: CAD drawings, PCB design, firmware development, image generation*




