# 18. Future Development

## Overview

This document has focused on **hardware and field methods**. The next phase involves **software development** for data processing, inversion, and visualization, as well as continued hardware improvements.

---

## Software Development Roadmap

### Data Processing Pipeline

#### 1. Data QA/QC (Quality Assurance/Quality Control)

**Tasks:**
- Import CSV data files
- Check for missing or corrupted data
- Verify reciprocity (A->B approx B->A)
- Remove outliers and bad measurements
- Flag suspicious data for review

**Requirements:**
- Data validation routines
- Statistical analysis tools
- Visualization of data quality metrics

#### 2. MIT Inversion

**Tasks:**
- Convert amplitude/phase measurements to conductivity/induction volume
- Implement forward modeling (predict measurements from model)
- Implement inverse modeling (reconstruct model from measurements)
- Handle multiple frequencies
- Account for measurement geometry

**Requirements:**
- Forward modeling code (EM field solver)
- Inversion algorithm (e.g., Gauss-Newton, Occam's inversion)
- Regularization for stable solutions
- Uncertainty quantification

#### 3. ERT Inversion

**Tasks:**
- Convert voltage/current measurements to resistivity volume
- Implement forward modeling (DC resistivity solver)
- Implement inverse modeling
- Handle multiple injection geometries
- Account for electrode positions and depths

**Requirements:**
- Forward modeling code (finite element or finite difference)
- Inversion algorithm
- Regularization
- Topography handling (if applicable)

#### 4. Data Fusion

**Tasks:**
- Co-register MIT and ERT volumes
- Combine complementary information
- Generate unified 3D models
- Overlay with other data (GPR, magnetometry, photogrammetry)

**Requirements:**
- Coordinate transformation
- Data fusion algorithms
- Multi-parameter visualization

#### 5. Visualization

**Tasks:**
- Generate depth slices (horizontal cross-sections)
- Create 3D isosurfaces
- Produce volume renderings
- Overlay with site maps/photos
- Export for GIS or CAD software

**Requirements:**
- 3D visualization libraries
- Image generation tools
- GIS integration capabilities

### Software Architecture Considerations

#### Language Options
- **Python:** Good for scientific computing, many libraries available
- **MATLAB:** Powerful but requires license
- **C++:** High performance, good for computationally intensive inversion
- **Hybrid:** Python for high-level, C++ for core algorithms

#### Key Libraries/Tools
- **NumPy/SciPy:** Numerical computing
- **VTK/ParaView:** 3D visualization
- **PyVista:** Python 3D visualization
- **ResIPy/pyGIMLi:** ERT inversion (existing tools)
- **SimPEG:** EM modeling and inversion framework

### Development Timeline Estimate

#### Phase 1: Basic Processing (Months 1-3)
- Data import/export
- Basic QC tools
- Simple visualization

#### Phase 2: Inversion (Months 4-6)
- Forward modeling
- Inversion algorithms
- Basic fusion

#### Phase 3: Advanced Features (Months 7-12)
- Advanced visualization
- Machine learning integration
- User interface
- Documentation

---

## Hardware Improvements

### Wireless Probes
- Reduce cable management complexity
- Enable faster deployment
- Consider LoRa or BLE communication
- Balance power consumption vs. convenience

### Higher Channel Count
- Increase probe density
- Enable finer resolution
- Requires expanded multiplexing
- Consider modular expansion

### Real-time Processing
- Process data during collection
- Enable adaptive survey strategies
- Immediate quality feedback
- Requires field computing capability

### Automated Deployment
- Future consideration for larger surveys
- Robotic probe insertion
- Significant engineering challenge
- May reduce deployment time substantially

---

## Forward Modeling and Validation

### Recommended Forward Modeling Tools

| Tool | Focus | Strengths | URL |
|------|-------|-----------|-----|
| **SimPEG** | MIT + ERT | Full forward/inverse modeling, Python | simpeg.xyz |
| **pyGIMLi** | ERT-focused | Excellent visualization, mature | pygimli.org |
| **Empymod** | 1D layered EM | Fast, accurate for stratified media | empymod.emsig.xyz |

### Standard Validation Scenarios

Before field deployment, validate HIRT system response using synthetic models:

#### Scenario 1: Aluminum Bomb in Sandy Loam
- **Target:** 1m diameter aluminum sphere
- **Depth:** 3m below surface
- **Soil:** 0.1 S/m (moist sand)
- **Purpose:** Validate MIT response to non-ferrous UXO

#### Scenario 2: Grave Shaft (Disturbed Fill)
- **Target:** 0.8x0.5x1.5m rectangular prism
- **Fill:** 0.05 S/m (disturbed, drier)
- **Background:** 0.2 S/m (native clay)
- **Depth:** 1.5-3m
- **Purpose:** Validate ERT contrast detection

#### Scenario 3: Scattered Aircraft Debris
- **Targets:** Multiple conductive fragments (0.2-1m diameter)
- **Depths:** 1-4m, scattered distribution
- **Soil:** 0.15 S/m (loamy soil)
- **Purpose:** Validate MIT multi-target discrimination

#### Scenario 4: Bomb Crater with Heterogeneous Fill
- **Geometry:** 8m diameter, 5m deep crater
- **Fill:** Variable conductivity (0.05-0.5 S/m)
- **Metal:** Fragments at base
- **Purpose:** Combined MIT+ERT response to complex geometry

### Forward Modeling Workflow

1. **Define geometry:** Probe positions, electrode locations
2. **Set material properties:** Conductivity distribution
3. **Compute synthetic data:** Forward model MIT and ERT
4. **Add realistic noise:** Based on HIRT specifications (+/-5 degree phase, ~100 nV noise floor)
5. **Invert synthetic data:** Verify reconstruction accuracy
6. **Compare to resolution limits:** Validate depth/resolution claims

### Validation Requirements

**Before field deployment, document:**
- Detection threshold vs. target size curves
- Depth sensitivity functions for MIT and ERT
- Resolution limits for standard probe configurations
- False positive rates under realistic noise conditions

---

## Known Limitations

### Technical Limitations
- Smaller coil area results in ~19 dB SNR loss (compensated by longer integration times)
- Survey time increases 5-10x compared to commercial systems
- Requires post-processing software for 3D reconstruction
- Limited depth in highly conductive soils

### Current System Constraints
- Electronics SNR adequate for detection but compromised for precise material characterization
- Phase accuracy (+/-5 degrees) vs. commercial (+/-0.5 degrees)
- Noise floor (~100 nV) vs. commercial (~10 nV)

### Operational Considerations
- Field deployment requires trained operators
- Data interpretation requires geophysical expertise
- Results depend on soil conditions and target properties
- Weather and access may limit operations

---

## Manufacturing Status and Releases

### Current Release: 16mm Modular Probe

**Date:** 2024-12-19
**Design Status:** RELEASED FOR PRINTING

### Key Specifications

| Specification | Value |
|---------------|-------|
| Rod Standard | 16mm OD / 12mm ID Fiberglass Tube |
| Connector Type | Flush Modular Inserts (Epoxied) |
| Thread Standard | M12x1.75 ISO (Modified "Chunky" Profile) |
| Wiring | Central 6mm hollow conduit (Confirmed Clear) |

### Design Philosophy

- **Archaeologist brain first, engineer brain second**
- Minimal intrusion: ~10x less disturbance than 25mm design
- Modular: Sensors integrated into 3D-printed couplers
- Passive probes: Electronics stay at surface
- Field-serviceable: Replace segments without rebuilding

### Components Ready for Production

| Component | Status | Notes |
|-----------|--------|-------|
| Male Insert Plug | Ready | Tested, verified |
| Female Sensor Module | Ready | Tested, verified |
| Probe Tip | Ready | Tested, verified |
| Top Cap | Ready | Tested, verified |
| ERT Ring Collar | Ready | Tested, verified |
| Rod Coupler | Ready | Tested, verified |

### Pending Development

| Component | Status | Notes |
|-----------|--------|-------|
| Base Hub Enclosure | In Progress | Backplane PCB design |
| Front Panel | In Progress | DXF file pending |
| Cable Harness | Specified | Ready for fabrication |

### Design Change History

#### v1.0 - Original Design (Deprecated)
- 25mm OD rod
- 50mm insertion hole
- Active electronics in probe head
- Large, heavy design

#### v2.0 - Micro-Probe Design
- 12mm OD rod
- Passive probes
- Electronics at surface
- ~10x less intrusion

#### v3.0 - 16mm Modular Design (Current)
- 16mm OD rod (increased for strength)
- Modular connectors (screw-together)
- M12x1.75 threads (printable)
- Flush-mount design (no snags)
- Integrated sensor modules

### Manufacturing Settings (Bambu A1 Mini Recommended)

| Setting | Value | Notes |
|---------|-------|-------|
| Material | PETG or ASA | Required for impact/UV |
| Layer Height | 0.12mm (Fine) | Critical for threads |
| Infill | 100% (Solid) | Critical for strength |
| Walls | 6 Loops | Solid threaded regions |
| Supports | DISABLED | Use built-in scaffolding |
| Brim | DISABLED | Use built-in Super Brim |
| Speed | 50mm/s Outer Wall | Quality over speed |

### Project Phase

**Current Phase:** Manufacturing
**Next Phase:** Field Testing

---

## Machine Learning Opportunities

### Potential Applications
- **Anomaly detection:** Automatically identify targets
- **Classification:** Distinguish metal vs. void vs. disturbed soil
- **Quality assessment:** Predict data quality
- **Parameter estimation:** Estimate target properties

### Data Requirements
- Labeled training data (synthetic and/or real)
- Sufficient examples for learning
- Validation datasets

---

## Integration with Existing Tools

### Compatibility
- **GIS software:** Export to shapefiles, GeoTIFF
- **CAD software:** Export 3D models
- **Archaeological software:** Integrate with standard tools
- **Data formats:** Standard formats (HDF5, NetCDF)

---

## Documentation Needs

### User Documentation
- Processing workflow guide
- Parameter selection guidelines
- Interpretation guide
- Troubleshooting

### Technical Documentation
- Algorithm descriptions
- Code documentation
- Validation studies
- Performance benchmarks

---

## Note

This software development is **explicitly separate** from the hardware/field guide presented in this document. The hardware system is designed to collect high-quality data that can be processed with standard or custom software tools.

