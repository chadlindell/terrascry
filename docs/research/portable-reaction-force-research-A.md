# Portable Reaction Force and Anchoring Solutions for Lightweight Hydraulic Push Systems

**Research Document for HIRT Project**
**Date:** 2026-01-19
**Status:** Initial Research Complete

---

## Executive Summary

This document presents comprehensive research on portable reaction force and anchoring solutions for the HIRT (Hydraulic Impedance Response Tomography) system, which requires pushing 16mm diameter probes to depths of 2-3 meters in remote areas without vehicle access. The estimated push force requirement of 5-15 kN presents significant challenges for lightweight, portable equipment.

**Key Findings:**
- Actual force requirements for a 16mm probe range from approximately **0.4 kN (soft clay) to 8 kN (dense sand)**, with rare cases exceeding 10 kN in very dense/gravelly soils
- Multiple viable anchoring strategies exist, each with trade-offs between weight, reliability, and soil dependency
- A hybrid approach combining **helical anchors + water ballast + mechanical advantage** offers the best balance for remote site operations
- Realistic portable weight budget for 2-3 person carry: **60-80 kg total system weight**

---

## Table of Contents

1. [Force Requirements Analysis](#1-force-requirements-analysis)
2. [Portable Anchoring Methods](#2-portable-anchoring-methods)
3. [Portable Ballast Options](#3-portable-ballast-options)
4. [Novel and Alternative Approaches](#4-novel-and-alternative-approaches)
5. [Weight Budget Analysis](#5-weight-budget-analysis)
6. [Recommended Solutions](#6-recommended-solutions)
7. [Product Examples and Specifications](#7-product-examples-and-specifications)
8. [References](#8-references)

---

## 1. Force Requirements Analysis

### 1.1 Penetration Force Calculation Methodology

The fundamental relationship for cone penetration resistance is:

```
Force (kN) = Cone Resistance (qc) x Cone Area (m^2)
```

For a **16mm diameter probe**:
- Cross-sectional area = pi x (0.008m)^2 = **2.01 cm^2 = 0.000201 m^2**

This is significantly smaller than standard CPT cones (10-15 cm^2), which provides a major advantage for reducing penetration force requirements.

### 1.2 Typical Cone Resistance Values by Soil Type

Based on extensive CPT literature and field data:

| Soil Type | Typical qc Range (MPa) | Notes |
|-----------|------------------------|-------|
| Very soft clay | 0.2 - 1.0 | Minimal resistance |
| Soft to firm clay | 1.0 - 2.0 | Easy penetration |
| Stiff clay | 2.0 - 5.0 | Moderate resistance |
| Very stiff/hard clay | 5.0 - 10.0 | Significant resistance |
| Loose sand | 5.0 - 15.0 | Variable with moisture |
| Medium dense sand | 15.0 - 30.0 | Common field condition |
| Dense sand | 30.0 - 50.0 | High resistance |
| Very dense sand/gravel | 50.0 - 100.0+ | May limit penetration |
| Gravel | Not penetrable | Standard CPT limitation |

*Sources: [Gregg Drilling CPT Guide](https://www.novotechsoftware.com/downloads/PDF/en/Ref/CPT-Guide-5ed-Nov2012.pdf), [ScienceDirect CPT Topics](https://www.sciencedirect.com/topics/engineering/cone-penetration-test)*

### 1.3 Calculated Force Requirements for 16mm Probe

Using the 2.01 cm^2 probe area:

| Soil Condition | qc (MPa) | Calculated Force (kN) | Force (kg-equivalent) |
|----------------|----------|----------------------|----------------------|
| Very soft clay | 0.5 | 0.10 | 10 |
| Soft clay | 1.5 | 0.30 | 31 |
| Stiff clay | 4.0 | 0.80 | 82 |
| Very stiff clay | 8.0 | 1.61 | 164 |
| Loose sand | 10.0 | 2.01 | 205 |
| Medium dense sand | 20.0 | 4.02 | 410 |
| Dense sand | 40.0 | 8.04 | 820 |
| Very dense sand | 60.0 | 12.06 | 1,230 |
| Extreme (theoretical) | 100.0 | 20.10 | 2,050 |

### 1.4 Friction/Adhesion Along Probe Shaft

Additional force is required to overcome friction along the probe shaft as depth increases:

**Sleeve friction (fs) typical values:**
- Clay soils: 10-150 kPa
- Sandy soils: 5-100 kPa

For a 16mm diameter probe at 2m depth:
- Shaft surface area = pi x 0.016m x 2m = 0.100 m^2
- Additional friction force (medium clay, 50 kPa) = 0.100 x 50 = **5.0 kN**
- Additional friction force (sandy soil, 30 kPa) = 0.100 x 30 = **3.0 kN**

### 1.5 Force vs. Depth Relationship

Force requirements increase approximately linearly with depth due to:
1. Increasing overburden pressure (soil becomes denser)
2. Cumulative shaft friction
3. Potential for encountering harder layers

**Estimated total force at various depths (medium dense sand, qc=20 MPa):**
| Depth (m) | Tip Force (kN) | Shaft Friction (kN) | Total Force (kN) |
|-----------|---------------|---------------------|------------------|
| 0.5 | 4.0 | 0.8 | 4.8 |
| 1.0 | 4.0 | 1.5 | 5.5 |
| 1.5 | 4.0 | 2.3 | 6.3 |
| 2.0 | 4.0 | 3.0 | 7.0 |
| 2.5 | 4.0 | 3.8 | 7.8 |
| 3.0 | 4.0 | 4.5 | 8.5 |

### 1.6 Design Force Recommendations

Based on the analysis above:

| Design Scenario | Recommended Force Capacity |
|-----------------|---------------------------|
| **Minimum (soft soils only)** | 3 kN (306 kg) |
| **Standard (typical mixed soils)** | 8 kN (816 kg) |
| **Heavy-duty (dense sand/stiff clay)** | 12 kN (1,223 kg) |
| **Maximum (very difficult conditions)** | 15 kN (1,530 kg) |

**Recommendation:** Design for **10-12 kN** reaction force capacity with a safety factor of 1.25, giving a target anchoring capacity of **12-15 kN**.

---

## 2. Portable Anchoring Methods

### 2.1 Helical/Screw Anchors

Helical anchors are steel shafts with helical plates that are screwed into the ground, providing excellent holding capacity with minimal soil disturbance.

#### Advantages:
- High holding capacity-to-weight ratio
- Can be hand-installed with bar or portable power driver
- Immediate load capacity after installation
- Reusable (can be removed and reinstalled)
- Works in most soil types
- Minimal environmental impact

#### Disadvantages:
- Require longer installation time
- Cannot penetrate gravel or rock
- Capacity varies significantly with soil type
- Require torque measurement for capacity verification

#### Holding Capacity by Soil Class:

Based on CHANCE/A.B. Chance Company soil classification system:

| Model | Soil Class 5 | Soil Class 6 | Soil Class 7 |
|-------|-------------|--------------|--------------|
| 1" x 8" x 66" | 49 kN (11,000 lbs) | 40 kN (9,000 lbs) | 27 kN (6,000 lbs) |
| 1-1/4" x 10" x 66" | 58 kN (13,000 lbs) | 44 kN (10,000 lbs) | 31 kN (7,000 lbs) |

**Soil Class Descriptions:**
- Class 5: Medium dense coarse sand, sandy gravels, stiff to very stiff silts/clays
- Class 6: Loose to medium dense sand, firm clays, compacted fill
- Class 7: Loose fine sand, flood plain soils, fill

*Source: [Lifting.com Helical Anchors](https://lifting.com/earth-anchor-helix-114x10x66.html)*

#### Weight Considerations:
- Typical 66" helical anchor: 4-6 kg each
- Installation bar/lever: 2-3 kg
- For 15 kN capacity in Class 6 soil: 1 anchor sufficient
- **Recommended: 2-3 anchors for redundancy = 12-18 kg total**

### 2.2 Percussion Driven Earth Anchors (PDEA) - Duckbill Type

Duckbill anchors are driven into the ground and then "locked" by pulling to rotate the anchor plate perpendicular to the load direction.

#### Advantages:
- Fast installation (can be driven with sledgehammer or portable jackhammer)
- Compact and lightweight
- Good holding capacity in soft to medium soils
- No rotation required during installation

#### Disadvantages:
- Lower capacity than helical anchors in dense soils
- Require adequate depth for deployment
- Single-use design (cannot be easily removed)

#### Holding Capacity by Model:

| Model | Typical Capacity | Weight |
|-------|------------------|--------|
| Duckbill 40-DB1 | 1.3 kN (300 lbs) | ~0.2 kg |
| Duckbill 68-DB1 | 4.9 kN (1,100 lbs) | ~0.5 kg |
| Duckbill 88-DB1 | 13.3 kN (3,000 lbs) | ~1.0 kg |
| Duckbill 138-DB1 | 22.2 kN (5,000 lbs) | ~1.5 kg |

*Note: Capacities rated in Class 5 (average) soil conditions*

*Source: [MacLean Civil Products Duckbill](https://www.macleancivilproducts.com/product/duckbill)*

**For 15 kN capacity: 2 x Model 88 or 1 x Model 138 = 2-3 kg total anchor weight**

### 2.3 Platipus PDEA System

Professional-grade percussion driven earth anchors with verified load-locking capability.

#### Specifications:
- Available capacities up to 200 kN
- Lightweight, corrosion-resistant
- Can be installed with portable jackhammer/breaker
- Immediate proof testing via load-locking

#### Models for HIRT Application:

| Model | Ultimate Capacity | Min. Depth | Ideal Soil Type |
|-------|------------------|------------|-----------------|
| S2 | 2.5 kN (250 kg) | 450mm | General |
| S4 ARGS | 10 kN (1,000 kg) | 750mm | Non-cohesive |
| S6 ARGS | 10 kN (1,000 kg) | 750mm | Cohesive |
| BAT Series | Higher loads | Variable | Soft cohesive |

*Source: [Platipus Anchors](https://platipus-anchors.com/)*

#### Installation Equipment:
- Portable breaker/jackhammer: 8-15 kg
- Drive steel and accessories: 3-5 kg

### 2.4 Driven Stakes/Pins

Simple steel stakes driven into ground at angles.

#### Pull-out Resistance Factors:
- Stake length
- Stake diameter
- Soil density and cohesion
- Installation angle (optimal 45 degrees from vertical, angled toward load)
- Number of stakes

#### Typical Capacities:
- 600mm x 16mm steel pin in firm clay: 2-4 kN per stake
- 900mm x 20mm rebar in dense sand: 4-8 kN per stake

**Limitations:**
- Highly variable capacity
- Requires proof testing
- May require multiple stakes
- Difficult in rocky or very hard soils

**For 15 kN: Approximately 4-6 heavy-duty stakes = 8-12 kg**

### 2.5 Comparison Table: Anchor Systems

| System | 15 kN Capacity Weight | Installation Time | Reusable | Soil Limitations |
|--------|----------------------|-------------------|----------|------------------|
| Helical (2x) | 12-18 kg | 15-30 min | Yes | Cannot penetrate gravel/rock |
| Duckbill (2x 88) | 2-3 kg | 5-10 min | No | Soft soils preferred |
| Platipus S4/S6 (2x) | 3-5 kg | 10-15 min | Limited | Most soils |
| Driven Stakes (6x) | 8-12 kg | 10-20 min | Yes | Very hard soils difficult |

---

## 3. Portable Ballast Options

### 3.1 Water Bladders/Bags (Fill on Site)

Water-filled bladders provide ballast that can be transported empty and filled on-site if water is available.

#### Specifications:
- Water density: 1 kg/L (8.34 lbs/gallon)
- 100L water = 100 kg (220 lbs) = ~1 kN holding force

#### Products:
| Product | Capacity | Empty Weight | Dimensions |
|---------|----------|--------------|------------|
| 25 gal collapsible tank | 95 kg (210 lbs) | <2.5 kg | Foldable |
| 100 gal bladder | 380 kg (837 lbs) | ~9 kg | 64" x 51" x 12" |
| 55 gal water barrel | 170 kg (375 lbs) | ~8 kg | Standard barrel |

*Source: [Texas Boom Company](https://texasboom.com/news/using-collapsible-bladder-tanks-as-ballast/)*

#### Advantages:
- Zero transport weight for ballast mass
- Can achieve high weights on-site
- Environmentally benign

#### Disadvantages:
- Requires water source on-site
- Time to fill (pump or gravity)
- Potential for leaks
- Must be placed above reaction point

**For 15 kN (1,530 kg) ballast-only: Would require ~1,500L water - NOT practical as sole solution**

### 3.2 Sand Bags (Fill on Site)

#### Specifications:
- Dry sand density: ~1,600 kg/m^3
- 27 lb (12 kg) capacity standard sandbag
- Heavy-duty construction bags: up to 45 kg (100 lbs)

#### Products:
| Type | Capacity | Empty Weight | Notes |
|------|----------|--------------|-------|
| Standard polypropylene | 12 kg | <0.5 kg | Quick tie cords |
| Heavy-duty construction | 45 kg | <1 kg | 22" x 36" |

*Source: [Sandbag Store](https://www.sandbagstore.com/shot-filled-sandbags.html)*

#### Advantages:
- Very low transport weight
- Sand available at most sites
- Durable and reusable

#### Disadvantages:
- Requires sand source
- Labor-intensive to fill
- Still requires significant volume for 15 kN

### 3.3 Portable Weight Plates (Carry-In)

Pre-filled, solid weights for immediate use.

#### Typical Options:
- Steel shot-filled sandbags: 11-25 kg per bag
- Cast iron weights: 10-25 kg plates
- Lead weights: Higher density but expensive/toxic

**For 15 kN: Would require ~150 kg carried in - NOT practical for remote sites**

### 3.4 Human Body Weight Utilization

Using operator weight as part of the reaction system.

#### Typical Operator Weights:
- Average adult: 70-90 kg
- 2-3 operators: 140-270 kg = 1.4-2.7 kN

#### Implementation:
- Standing platform on reaction frame
- Multiple operators can multiply effect
- Most ergonomic when combined with other methods

**Practical contribution: 1.5-2.5 kN from operator standing platform**

### 3.5 Ballast Comparison

| Method | Weight to Transport | Achievable Force | On-Site Requirements |
|--------|---------------------|------------------|---------------------|
| Water bladder (100L) | 9 kg | 1 kN | Water source |
| Sand bags (10x) | 5 kg | 1.2 kN (filled) | Sand source |
| Carry-in plates | 100 kg | 1 kN | None |
| Human operators (2) | 0 kg | 1.5-1.8 kN | Standing platform |

---

## 4. Novel and Alternative Approaches

### 4.1 Mechanical Advantage Systems

Lever and pulley systems can multiply input force, reducing the required anchor/ballast capacity.

#### Lever Systems:
Using a 1st or 2nd class lever with mechanical advantage:
- 3:1 MA lever: 5 kN anchor provides 15 kN push force
- Requires rigid frame and fulcrum

#### Pulley/Block Systems:
- 3:1 mechanical advantage readily achievable with portable blocks
- 5:1 complex systems used in rescue operations
- Requires anchor point above or behind push point

*Source: [Rigging Lab Academy](https://rigginglabacademy.com/fundamentals-of-force-multiplier-mechanics/)*

**Design Concept:**
- Push cylinder connected to pulley system
- 3:1 MA reduces anchor requirement from 15 kN to 5 kN
- Trade-off: 3x stroke required for same penetration distance

### 4.2 Reaction Against Trees/Structures

If available, trees and structures provide excellent anchor points.

#### Tree Anchoring Systems:
- Tree strap anchoring systems: 22-45 kN capacity typical
- Portable Winch tree mount: fits trees 12-36" diameter
- Arborist friction savers: 22-44 kN ratings

*Source: [Portable Winch USA](https://www.portablewinch.com/products/anchoring-system-for-trees-and-posts-with-rubber-pads)*

#### Specifications:
| Product | Capacity | Weight | Tree Diameter |
|---------|----------|--------|---------------|
| PCA-1269 Tree Anchor | 45 kN (10,116 lbs) | ~7 kg | 12-36" |
| PCA-1263 with pads | 45 kN | ~7 kg | 12-36" |

**Advantages:**
- Excellent capacity if trees available
- Minimal equipment weight
- Non-damaging with proper pads

**Limitations:**
- Not always available at test sites
- Requires assessment of tree suitability
- Angle considerations for force direction

### 4.3 Vacuum/Suction Anchors

Vacuum anchors create holding force through atmospheric pressure differential.

#### Industrial Examples:
- 3M DBI-SALA Vacuum Anchor: 22 kN (5,000 lbs) on smooth surfaces
- MSA WinGrip: <6 kg, wet/dry surfaces

*Source: [3M Vacuum Anchor](https://www.3m.com/3M/en_US/p/d/b00040632/)*

**Limitations for HIRT:**
- Require smooth, non-porous surfaces (not natural ground)
- Primarily designed for fall protection on aircraft/industrial surfaces
- NOT suitable for field soil anchoring

### 4.4 Counterweight Lever Systems

Using lever principle with available materials as counterweight.

**Concept:**
- A-frame or boom structure
- Counterweight on long arm (rocks, water, equipment)
- Push cylinder on short arm
- Mechanical advantage reduces counterweight requirement

**Example Calculation:**
- 3:1 lever ratio (counterweight arm 3x push arm length)
- 15 kN push force requires only 5 kN counterweight
- 500 kg of rocks/water/equipment on long arm provides 15 kN push capacity

### 4.5 Portable Winch Integration

Gas or battery-powered portable winches can provide continuous pulling force.

#### Products:
| Model | Pull Capacity | Weight | Power |
|-------|--------------|--------|-------|
| PCW5000 | 10 kN (2,200 lbs) | 16 kg | Gas |
| PCW3000-LI | 10 kN (2,200 lbs) | 9 kg | Battery |
| PCW4000 | 10 kN (2,200 lbs) | 12 kg | Gas |

*Source: [American Arborist Supplies](https://www.arborist.com/category/10330/Portable-Winch.html)*

**Application:**
- Winch anchored to tree/stake system
- Pulls on reaction frame to provide downward force
- Could provide 5-10 kN continuous reaction

### 4.6 Military/Expedition Solutions

Military anchoring systems designed for rapid deployment:

#### Arctic Anchoring:
- All-Terrain Arctic Anchor: Uses MOLLE webbing, spikes from mending plates
- Designed for -60F, 55 mph steady winds, 65 mph gusts
- Emphasis on quiet installation (no hammering for tactical reasons)

*Source: [US Army USMA Project](https://www.army.mil/article/284888)*

#### Expeditionary Equipment:
- Lightweight shelters use combination of stakes and ballast
- Gabion systems (rock-filled cages) for semi-permanent installations

---

## 5. Weight Budget Analysis

### 5.1 Human Carrying Capacity

#### Guidelines for Backpack/Equipment Carry:
- Day hike standard: 10% of body weight
- Multi-day standard: 20% of body weight (~16 kg for 80 kg person)
- Maximum recommended individual load: 25-30 kg over rough terrain

*Source: [REI Expert Advice](https://www.rei.com/learn/expert-advice/backpacking-weight.html)*

#### Two-Person Lift/Carry:
- OSHA recommends <23 kg (51 lbs) per person for extended carrying
- MIL-STD-1472G two-person limit: 79 kg (174 lbs) total
- Practical field limit: 30-40 kg per person for 1-2 km over terrain

*Source: [OSHA Guidelines](https://www.osha.gov/laws-regs/standardinterpretations/2013-06-04-0)*

### 5.2 Team Carrying Capacity (2-3 People)

| Team Size | Max Individual | Total Capacity | Sustained Distance |
|-----------|----------------|----------------|-------------------|
| 2 people | 30 kg | 60 kg | 1-2 km rough terrain |
| 3 people | 25 kg | 75 kg | 1-2 km rough terrain |
| 2 people | 20 kg | 40 kg | 3-5 km rough terrain |
| 3 people | 20 kg | 60 kg | 3-5 km rough terrain |

### 5.3 System Component Weight Breakdown

**Essential Components:**

| Component | Weight Range | Notes |
|-----------|--------------|-------|
| **Hydraulic Cylinder** | 8-15 kg | 10-20 kN capacity double-acting |
| **Hydraulic Pump** | 5-10 kg | Manual or battery electric |
| **Hoses/Fittings** | 2-4 kg | High pressure rated |
| **Reaction Frame** | 10-20 kg | Aluminum or steel construction |
| **Push Rods (2m set)** | 8-12 kg | Threaded connections |
| **Probe Tips (5x)** | 1-2 kg | Consumables |
| **Electronics** | 2-3 kg | Data acquisition |
| **Tools/Accessories** | 3-5 kg | Wrenches, etc. |
| **Subtotal (Core)** | **39-71 kg** | Before anchoring |

**Anchoring Options (add to core):**

| Option | Weight | Provides |
|--------|--------|----------|
| Helical anchors (2x) + bar | 15-20 kg | 15+ kN |
| Duckbill (4x) + hammer | 6-10 kg | 12-15 kN |
| Water bladder (empty) | 9-12 kg | Variable (needs water) |
| Sand bags (20x empty) | 2-3 kg | Variable (needs sand) |
| Tree straps + hardware | 5-8 kg | 15+ kN (if trees available) |

### 5.4 Recommended System Configurations

#### Configuration A: Minimum Weight (Favorable Sites)
*For sites with trees or easy soil conditions*
| Component | Weight |
|-----------|--------|
| Core system | 45 kg |
| Tree straps | 6 kg |
| Backup duckbills (2x) | 3 kg |
| **Total** | **54 kg** |

#### Configuration B: Standard (Most Sites)
*Balanced approach for typical conditions*
| Component | Weight |
|-----------|--------|
| Core system | 50 kg |
| Helical anchors (2x) + bar | 18 kg |
| Water bladder (100L) | 10 kg |
| **Total** | **78 kg** |

#### Configuration C: Heavy Duty (Difficult Sites)
*Maximum capability, requires team of 3*
| Component | Weight |
|-----------|--------|
| Core system | 55 kg |
| Helical anchors (3x) + bar | 25 kg |
| Duckbills (4x) + driver | 12 kg |
| Water bladder (100L) | 10 kg |
| **Total** | **102 kg** |

---

## 6. Recommended Solutions

### 6.1 Primary Recommendation: Hybrid Anchor System

**Design Philosophy:** Combine multiple lightweight anchor types for redundancy and adaptability to varying site conditions.

#### System Components:
1. **Primary Anchors:** 2x helical screw anchors (1" x 8" x 48")
   - Capacity: 27-49 kN total in Class 5-7 soils
   - Weight: 10-12 kg total

2. **Quick-Deploy Backup:** 4x Duckbill 88-DB1
   - Capacity: 40-53 kN total in average soils
   - Weight: 4 kg total

3. **Opportunistic:** Tree strap kit
   - For use when suitable trees available
   - Weight: 6 kg

4. **Supplemental Ballast:** Collapsible water bladder (100L)
   - When water available, adds 1 kN
   - Weight: 9 kg empty

5. **Operator Platform:** Standing platform on frame
   - Adds 0.7-1.8 kN per person
   - Weight: Integrated into frame

### 6.2 Reaction Frame Design Requirements

The reaction frame must:
- Accept 12-15 kN downward force from hydraulic cylinder
- Transfer load to 2-4 anchor points
- Include standing platform for operator weight contribution
- Be collapsible/modular for transport
- Weigh <15 kg total

**Recommended Material:** 6061-T6 Aluminum tube construction with quick-connect joints

### 6.3 Hydraulic System Recommendations

| Parameter | Specification |
|-----------|--------------|
| Push Force | 15 kN minimum, 20 kN preferred |
| Pull Force | 25 kN minimum (for extraction) |
| Stroke | 100-150mm per extension |
| Cylinder Type | Double-acting |
| Pump | Manual or battery-electric |
| Operating Pressure | 700 bar (10,000 psi) |

**Product Reference:** Prolinemax 10/20 Ton cylinder
- 10 ton (89 kN) pull
- 20 ton (178 kN) push
- 4" or 6" stroke options
- Weight: ~16 kg

### 6.4 Operating Procedure

1. **Site Assessment:**
   - Check for trees >12" diameter within 3m
   - Assess soil type (probe with screwdriver, observe excavations)
   - Identify water sources if available

2. **Anchor Selection:**
   - Trees available: Deploy tree straps
   - Soft soil: Deploy Duckbill anchors
   - Firm soil: Deploy helical anchors
   - Mixed: Combination approach

3. **Setup Sequence:**
   - Install anchors at 45 degrees, toward push location
   - Connect cables/straps to reaction frame
   - Position hydraulic system
   - Fill water bladder if available
   - Operator stands on platform

4. **Proof Test:**
   - Apply 50% design load before full operation
   - Check for anchor movement
   - Adjust as needed

5. **Operation:**
   - Push at controlled rate (~2 cm/s)
   - Monitor force readings
   - Stop if force exceeds 12 kN (reassess)

---

## 7. Product Examples and Specifications

### 7.1 Helical Anchors

**Hubbell/CHANCE Helical Anchors**
- Website: [hubbell.com](https://www.hubbell.com/hubbellpowersystems/en/products/power-utilities/anchoring-foundations/helical-anchors-piles/cl/548080)
- 1" x 8" x 66" model: 49 kN in Class 5 soil
- Installation: Manual with 4' steel bar or portable hydraulic driver

**Generic Helical Auger Anchors**
- Website: [lifting.com](https://lifting.com/earth-anchor-helix-1x8x66.html)
- Various sizes available
- Lower cost alternative

### 7.2 Percussion Driven Anchors

**Duckbill Earth Anchors (MacLean Civil)**
- Website: [macleancivilproducts.com](https://www.macleancivilproducts.com/product/duckbill)
- Model 88: 13.3 kN working load
- Model 138: 22.2 kN working load

**Platipus PDEA**
- Website: [platipus-anchors.com](https://platipus-anchors.com/)
- Professional grade
- Load-lock verification included

### 7.3 Tree Anchoring Equipment

**Portable Winch Tree Mount (PCA-1269)**
- Website: [portablewinch.com](https://www.portablewinch.com/products/tree-mount-winch-anchoring-system)
- 45 kN capacity
- 7 kg weight
- Fits 12-36" diameter trees

### 7.4 Water Bladders

**Collapsible Pillow Tanks**
- Website: [tank-depot.com](https://www.tank-depot.com/storage-tanks/storage-tank-types/bladder-storage-tanks/)
- 25-100 gallon options
- Folds flat for transport

### 7.5 Hydraulic Equipment

**Prolinemax Push-Pull Cylinders**
- Website: [prolinemax.com](https://www.prolinemax.com/products/10-20-ton-push-pull-hydraulic-cylinder-4-stroke-double-acting)
- 10/20 ton models
- 4" and 6" stroke options

**Geoprobe Manual Sampling Equipment**
- Website: [geoprobe.com](https://geoprobe.com/tooling/manual-sampling-125-based)
- Probe rod jack for extraction
- Manual slide hammers

---

## 8. References

### 8.1 CPT and Soil Penetration

1. Gregg Drilling (2012). "Guide to Cone Penetration Testing." [PDF](https://www.novotechsoftware.com/downloads/PDF/en/Ref/CPT-Guide-5ed-Nov2012.pdf)

2. Robertson, P.K. (2009). "Interpretation of cone penetration tests - a unified approach." Canadian Geotechnical Journal. [PDF](https://www.cpt-robertson.com/PublicationsPDF/Robertson%202009%20CGJ.pdf)

3. ScienceDirect. "Cone Penetration Test - Overview." [Link](https://www.sciencedirect.com/topics/engineering/cone-penetration-test)

4. Geoprobe Systems. "CPT Cone Penetration Testing." [Link](https://geoprobe.com/direct-image/cpt-cone-penetration-testing)

### 8.2 Earth Anchors

5. US Forest Service. "An Earth Anchor System: Installation and Design Guide." [Link](https://www.fs.usda.gov/t-d/pubs/html/93241804/93241804.html)

6. NRCS. "Technical Supplement 14E: Use and Design of Soil Anchors." [PDF](https://www.ncagr.gov/soil-water/swcstrap-nrcs-ts14e-soil-anchors-guide/open)

7. MacLean Civil Products. "Duckbill Earth Anchors." [Link](https://www.macleancivilproducts.com/product/duckbill)

8. Platipus Anchors. "Earth Anchoring Solutions." [Link](https://platipus-anchors.com/)

9. Hubbell Power Systems. "The Science of Soil Mechanics & Holding Capacity." [Link](https://blog.hubbell.com/en/hubbellpowersystems/the-science-of-soil-mechanics-holding-capacity)

### 8.3 Portable Equipment

10. A.P. van den Berg. "Mini CPT Crawler." [Link](https://www.apvandenberg.com/onshore-cone-penetration-testing/mini-cpt-crawler)

11. Portable Winch USA. "Tree Mount Anchoring System." [Link](https://www.portablewinch.com/products/tree-mount-winch-anchoring-system)

12. American Arborist Supplies. "Portable Winch Equipment." [Link](https://www.arborist.com/category/10330/Portable-Winch.html)

### 8.4 Weight Limits and Ergonomics

13. REI. "How Much Should Your Pack Weigh." [Link](https://www.rei.com/learn/expert-advice/backpacking-weight.html)

14. OSHA. "Safe Weight Limits for Manual Lifting." [Link](https://www.osha.gov/laws-regs/standardinterpretations/2013-06-04-0)

15. CPD Online. "Manual Handling Weight Limits." [Link](https://cpdonline.co.uk/knowledge-base/health-and-safety/manual-handling-weight-limits/)

### 8.5 Mechanical Advantage

16. Rigging Lab Academy. "Force Multiplier Mechanics." [Link](https://rigginglabacademy.com/fundamentals-of-force-multiplier-mechanics/)

17. Wikipedia. "Mechanical Advantage." [Link](https://en.wikipedia.org/wiki/Mechanical_advantage)

---

## Appendix A: Force Calculation Quick Reference

### 16mm Probe Force Calculator

```
Tip Force (kN) = qc (MPa) x 0.000201 (m^2) x 1000
              = qc (MPa) x 0.201

Shaft Friction (kN) = fs (kPa) x pi x 0.016 (m) x Depth (m)
                    = fs (kPa) x 0.0503 x Depth (m) / 1000

Total Force = Tip Force + Shaft Friction
```

### Quick Lookup Table

| Soil Description | Estimated qc | 16mm Tip Force |
|------------------|--------------|----------------|
| Very soft organic | 0.5 MPa | 0.1 kN |
| Soft clay | 1.5 MPa | 0.3 kN |
| Firm clay | 3.0 MPa | 0.6 kN |
| Stiff clay | 5.0 MPa | 1.0 kN |
| Loose sand | 8.0 MPa | 1.6 kN |
| Medium sand | 20 MPa | 4.0 kN |
| Dense sand | 40 MPa | 8.0 kN |
| Very dense | 60 MPa | 12.1 kN |

---

## Appendix B: Soil Classification Quick Guide

### Field Identification

| Class | Description | Field Test |
|-------|-------------|------------|
| Class 5 | Dense sand/gravel, very stiff clay | Difficult to drive stake by hand |
| Class 6 | Medium sand, firm clay, compacted fill | Stake drives with effort |
| Class 7 | Loose sand, soft clay, organic soil | Stake drives easily |
| Class 8 | Very soft organic, saturated | Stake falls under own weight |

### Anchor Selection by Soil Class

| Class | Recommended Primary | Backup |
|-------|---------------------|--------|
| Class 5 | Helical (hand torque) | Driven stakes |
| Class 6 | Helical or Duckbill | Multiple anchors |
| Class 7 | Duckbill (deeper) | Water ballast supplement |
| Class 8 | Not recommended | Seek better location |

---

**Document Version:** 1.0
**Last Updated:** 2026-01-19
**Author:** Claude (AI Research Assistant)
**Project:** HIRT - Hydraulic Impedance Response Tomography
