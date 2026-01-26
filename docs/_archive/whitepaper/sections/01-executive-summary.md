# 1. Executive Summary

## What is HIRT?

HIRT (Hybrid Inductive-Resistivity Tomography) is a **dual-channel subsurface imaging system** designed for archaeological and forensic investigations. By placing sensors inside the ground and measuring through the volume using **crosshole geometry**, HIRT obtains true tomographic coverage for 3D reconstruction of subsurface features.

### Dual-Channel Approach

HIRT uses two complementary sensing methods:

**1. MIT-3D (Magneto-Inductive Tomography)**
- Low-frequency TX/RX coils on rods (2-50 kHz)
- Measures amplitude and phase changes caused by eddy currents
- Maps conductive metal masses (including aluminum)
- Detects bulk conductivity structure

**2. ERT-Lite (Electrical Resistivity)**
- Small ring electrodes on rods
- Injects tiny currents (0.5-2 mA) and reads voltages
- Maps soil resistivity (moisture, disturbance, voids)
- Detects grave shafts and disturbed fill

### Unique Value Proposition

> **Design rule:** Make each probe dual-role (TX & RX for MIT) plus ERT pickup. Identical probes simplify logistics and improve data quality.

**Why HIRT?**
- **Democratized Tech:** brings $100k+ capabilities to the <$4k budget range.
- **Archaeology-First:** 16mm probes cause ~10x less site disturbance than standard 50mm geophysics probes.
- **True 3D:** Measures *through* the volume, not just reflections from the surface.
- **Modular:** "Zone" architecture scales from small test plots to large crash sites.

## Primary Use Cases

### 1. Filled Bomb Crater
- **Size:** ~10-15 m diameter, ~3 m deep
- **Soil:** Sandy/loamy fill
- **Targets:** Potential aluminum/steel parts + remains near base
- **Depth Target:** 0-4+ m
- **Key Method:** MIT + ERT combined

### 2. Woods Burials
- **Type:** Single/multiple interments
- **Depth:** 0.6-1.5 m
- **Context:** Forested archaeological sites
- **Targets:** Small metallic items (buckles, dog tags), grave shaft detection
- **Key Method:** ERT patterns for grave shaft, MIT for artifacts

### 3. Swamp/Perched Water Impacts
- **Environment:** Deeper/softer ground
- **Target Depth:** >5 m targets from margins
- **Challenges:** Water-saturated soils, access limitations
- **Key Method:** Low-frequency MIT from accessible margins

## System Capabilities

### Advantages of In-Ground Probes

Surface GPR/magnetometry are excellent screeners but can yield ambiguous results in complex conditions. HIRT's crosshole geometry provides:

- **True tomographic coverage** through the target volume
- **2-5x better resolution** than surface methods at depths >2m
- **Superior depth discrimination** (targets at 3m vs 4m clearly distinguishable)
- **Non-ferrous detection** (aluminum, which magnetometry cannot detect)
- **3D localization** rather than 2D maps with estimated pseudo-depth

### Depth of Investigation

| Configuration | MIT Depth | ERT Depth | Combined Claim |
|---------------|-----------|-----------|----------------|
| 1.5m probes, 2m spacing | 2-3m | 2-3m | **2-3m (HIGH confidence)** |
| 3.0m probes, 2m spacing | 3-4m | 3-5m | **3-5m (MEDIUM confidence)** |
| Edge cases (conductive soil) | 2-3m | 4-6m | **Up to 6m (LOW confidence)** |

## Limitations

### Non-Goals

- **Producing final 3D visuals** in this step (software is step 2)
- **Replacing standard ethical/excavation practice** or permits
- **Replacing professional archaeological/forensic protocols**

### When Surface Methods Remain Superior

- Rapid large-area screening (10x faster)
- Shallow targets (<1m) where GPR resolution excels
- Purely ferrous targets (magnetometry)
- Initial site characterization before targeted investigation

### Technical Limitations

- Smaller coil area results in ~19 dB SNR loss (compensated by longer integration times)
- Survey time increases 5-10x compared to commercial systems
- Requires post-processing software for 3D reconstruction
- Limited depth in highly conductive soils

## Target Audience

This document is intended for:

- **Archaeologists** investigating WWII sites, burial locations, or crash sites
- **Forensic investigators** requiring non-destructive subsurface imaging
- **Geophysicists** interested in low-cost tomographic methods
- **DIY builders** seeking to construct field-deployable systems
- **Researchers** exploring crosshole electromagnetic/resistivity techniques

## Document Structure

### PART I: Foundations
- **Section 1:** Executive Summary (this section)
- **Section 2:** Physics Theory
- **Section 3:** System Architecture

### PART II: Build Guide
- Hardware specifications and bill of materials
- Coil winding procedures
- Assembly instructions
- Testing and calibration procedures

### PART III: Field Operations
- Field operation manual
- Deployment procedures
- Data collection protocols
- Quality control procedures

### PART IV: Reference
- Glossary of terms
- Quick reference card
- Field checklists
- Application scenarios
- Future development roadmap
- Ethics, legal, and safety guidelines

## Optimal Workflow

The physics supports a **two-stage approach**:

1. **Surface screening** (magnetometry, GPR, EM31): Identify anomalies quickly over large areas
2. **HIRT crosshole follow-up**: Characterize identified anomalies with superior 3D resolution

This leverages the strengths of both approaches while minimizing deployment time.

## Cost Overview

- **Complete starter kit:** $1,800-3,900
- **Compared to commercial systems:** 95%+ savings ($50,000-200,000+ commercial)
- **Probe cost:** $70-150 per probe
- **Standard deployment:** 20-24 probes

