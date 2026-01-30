# Portable Hydraulic Push System for HIRT: Consolidated Research

**Document Type:** Consolidated Research Summary
**Date:** 2026-01-19
**Sources:** 6 parallel research documents (A/B variants for 3 topics)

---

## Executive Summary

This document consolidates findings from parallel research efforts investigating a portable hydraulic static push (CPT-style) system for HIRT probe deployment in remote areas. The research addressed three critical challenges:

1. **Reaction Force:** How to generate 5-15 kN push force without trucks or heavy anchors
2. **UXO Detection:** How to detect unexploded ordnance during push advancement
3. **Probe Insertion:** How to install probes after hole creation in collapsing soils

### Key Findings

| Topic | Primary Recommendation | Key Specification |
|-------|----------------------|-------------------|
| Reaction Force | Helical screw anchors + ballast | 72-102 kg system, 15-25 kN capacity |
| UXO Detection | Fluxgate magnetometer (two-pass or integrated) | 1 nT sensitivity, 2-4m detection range |
| Probe Insertion | Direct push with expendable tip OR dual-tube casing | Avoid bentonite (80% resistivity reduction) |

**Total Portable System Budget:** 72-102 kg for 2-3 person team
**Estimated Equipment Cost:** $15,000-35,000
**Development Timeline:** 10-18 months to field-ready system

---

## Part 1: Reaction Force Requirements

### 1.1 Push Force Analysis

Both research documents converged on similar force requirements for the 16mm HIRT probe:

**Probe Specifications:**
- Diameter: 16mm
- Cross-sectional area: 2.01 cm²
- Target depth: 2-3 meters

**Force Requirements by Soil Type:**

| Soil Type | Tip Resistance | Friction | Total Force | Source |
|-----------|---------------|----------|-------------|--------|
| Very soft clay | 0.3-0.6 kN | 0.1-0.2 kN | 0.4-0.8 kN | Research A |
| Soft clay | 0.6-1.0 kN | 0.2-0.5 kN | 0.8-1.5 kN | Research A |
| Medium clay | 1.0-2.0 kN | 0.5-1.0 kN | 1.5-3.0 kN | Research A |
| Loose sand | 1.0-3.0 kN | 0.2-0.5 kN | 1.2-3.5 kN | Research B |
| Medium sand | 3.0-6.0 kN | 0.5-1.5 kN | 3.5-7.5 kN | Research B |
| Dense sand/gravel | 6.0-12.0 kN | 1.0-3.0 kN | 7.0-15.0 kN | Research B |

**Consensus Design Target:** 15 kN push capacity (covers 95% of expected conditions)

### 1.2 Anchoring System Comparison

**Helical Screw Anchors (RECOMMENDED)**

Both research documents independently identified helical screw anchors as the optimal portable solution:

| Anchor Size | Helix Diameter | Capacity | Weight | Installation |
|-------------|---------------|----------|--------|--------------|
| Small | 6" (150mm) | 5-15 kN | 4-6 kg | Hand tool |
| Medium | 8" (200mm) | 15-30 kN | 8-12 kg | Hand tool with cheater bar |
| Large | 10" (250mm) | 27-58 kN | 12-18 kg | Power drive recommended |

**Torque-to-Capacity Relationship (Research B):**
```
Ultimate capacity (Q_ult) = K_t × T
Where:
  K_t = 33 m⁻¹ (typical installation factor)
  T = installation torque in kN·m
```

Example: 300 N·m torque → 10 kN capacity

**Alternative Anchoring Methods Evaluated:**

| Method | Capacity | Weight | Portability | Verdict |
|--------|----------|--------|-------------|---------|
| Helical anchors (3x) | 15-45 kN | 12-18 kg | Excellent | **RECOMMENDED** |
| Duckbill anchors | 13-22 kN | 2-4 kg | Good | Backup option |
| Ground screws | 5-15 kN | 3-8 kg | Good | Backup option |
| Deadman anchors | 20-50 kN | 50+ kg | Poor | Not recommended |
| Vehicle ballast | Unlimited | N/A | Requires road | Site-dependent |

### 1.3 Portable System Configurations

**Configuration A: Minimum Weight (Research A)**
- 2 helical anchors (8" helix): 16 kg
- Compact hydraulic cylinder (10 kN): 12 kg
- Hand pump: 8 kg
- Reaction frame: 15 kg
- Accessories: 5 kg
- **Total: 56 kg** (2 person carry)

**Configuration B: Enhanced Capacity (Research B)**
- 3 helical anchors (10" helix): 45 kg
- Hydraulic system: 25 kg
- Battery/power: 10 kg
- **Total: 80 kg** (3 person carry)

**Configuration C: Full Capability (Research A)**
- Anchor system: 25 kg
- Hydraulic system: 30 kg
- Operator ballast platform: 25 kg (allows standing on frame)
- Power system: 15 kg
- Tools and accessories: 10 kg
- **Total: 105 kg** (3-4 person carry)

### 1.4 Human Portability Constraints

Both documents cited military carrying standards:

| Load Type | Sustainable Distance | Weight Limit |
|-----------|---------------------|--------------|
| Fighting load | Unlimited | 22 kg |
| Approach march | 12+ km | 33 kg |
| Emergency road march | 1 km | 35-40 kg |
| Team carry | 1 km | 60-80 kg total |

**Practical Constraint:** Maximum ~100 kg total system for 3-person team over rough terrain

### 1.5 Mechanical Advantage Systems

Research A identified lever/pulley systems to reduce required hydraulic force:

| System | Mechanical Advantage | Trade-off |
|--------|---------------------|-----------|
| Simple lever | 2-5:1 | Increased stroke |
| Block and tackle | 4-10:1 | Slower operation |
| Hydraulic intensifier | 5-20:1 | Additional equipment |

**Recommendation:** 4:1 mechanical advantage allows 4 kN hand pump to generate 16 kN push force

### 1.6 Cost Estimates

**Research A Estimate:**
- Hydraulic system: $2,500-4,500
- Anchor system: $800-1,500
- Frame/structure: $1,500-3,000
- Tools/accessories: $500-1,000
- **Total: $5,300-10,000**

**Research B Estimate:**
- Complete portable system: $3,850-6,820
- With power pack upgrade: $6,850-9,820

**Consensus:** $5,000-10,000 for complete portable push system

---

## Part 2: UXO Detection During Push

### 2.1 Detection Technology Assessment

Both research documents unanimously identified **magnetometry** as the proven, dominant technology:

| Technology | Detection Range | UXO Detection | Miniaturization | HIRT Compatible |
|------------|-----------------|---------------|-----------------|-----------------|
| Fluxgate Magnetometer | 2-4m (250kg bomb) | Excellent | Challenging (18mm available) | **Yes** |
| MEMS Magnetometer | 0.5-1m | Poor | Easy (1mm) | No (50x higher noise) |
| Eddy Current | 5mm only | Poor | Easy | No |
| Metal Detector | 1-2m | Moderate | Difficult | No |
| ERT (existing HIRT) | Contact only | Poor | Already integrated | Supplementary only |
| GPR | Variable | Good | Not feasible | No |

### 2.2 Magnetometer Detection Physics

**Detection follows inverse-cube law:**

```
Detection range ∝ (magnetic moment)^(1/3) / sensitivity^(1/3)
```

**Practical Detection Distances (1 nT sensitivity):**

| Object | Weight | Detection Range |
|--------|--------|-----------------|
| Small UXO | 50 kg | 1.5-2.0 m |
| Medium UXO | 250 kg | 2.5-4.0 m |
| Large UXO | 500 kg | 3.5-5.0 m |
| Very large UXO | 1000 kg | 4.5-6.0 m |

**WWII Bomb Context:**
- SC 250 (common German bomb): 250 kg, 117cm long, 36.8cm diameter
- Typical penetration depth: 4-9m in clay
- 99% of bombs >50kg found above 9m depth
- HIRT target depth (2-3m) is within high-risk zone

### 2.3 Commercial CPT Magnetometer Systems

**Established Manufacturers:**
- Fugro (integrated abort system)
- Gouda Geo MagCone
- Royal Eijkelkamp Geomagnetic Module
- A.P. van den Berg
- Lankelma

**Standard Specifications:**
- Cone diameter: 36-44mm (vs HIRT 16mm)
- Sensitivity: <1 nT
- Sample rate: 10-20 Hz
- Push rate: 20mm/sec

### 2.4 Miniaturization Challenge

**The Problem:** Standard magnetometer cones are 36-44mm; HIRT probes are 16mm.

**Available Small-Diameter Sensors:**

| Sensor | Diameter | Type | Sensitivity | Notes |
|--------|----------|------|-------------|-------|
| WUNTRONIC WFG-110 | 18.4mm | Tri-axial fluxgate | <1 nT | Commercial, needs probe enlargement |
| Meldor small mag | 18mm | Fluxgate | ~1 nT | Commercial |
| Custom development | 12-14mm | Fluxgate | ~1 nT | 12-18 month development |

### 2.5 Recommended Detection Approaches

**Option A: Two-Pass System (Highest Safety)** ★ RECOMMENDED FOR INITIAL DEPLOYMENT

1. First push: Standard 44mm magnetometer cone to clear column
2. Second push: 16mm HIRT probe in pre-cleared position

**Pros:** Uses proven technology, maximum safety
**Cons:** Doubles insertion time, requires two probe types

**Option B: Enlarged Sensor Head**

```
     Probe Tip     Magnetometer      Main Probe Body
         |          Housing              |
    <==>----<===========O===========>--------------------->
    16mm    20-25mm     |              16mm
            (50-80mm length)
                   Fluxgate sensor
```

**Pros:** Single-pass operation, true look-ahead capability
**Cons:** Requires custom development, probe head larger than body

**Option C: Push-Rod Mounted Magnetometer**

Mount magnetometer on push rod above soil surface, monitor as probe descends.

**Pros:** No miniaturization needed, standard sensors
**Cons:** Detection only useful for shallow depths (~1m look-ahead)

### 2.6 Automatic Abort System

Both documents emphasized the need for automatic abort capability:

**Required Features:**
1. Real-time magnetic gradient monitoring
2. Push force/tip resistance monitoring
3. Configurable threshold settings
4. Automatic push-stop on threshold breach (<500ms response)
5. Visual and audible operator alerts
6. Manual override capability

**Recommended Thresholds:**
- Hard abort: >50 nT anomaly (automatic stop)
- Soft alarm: >10 nT anomaly (operator warning)
- Gradient threshold: >5 nT/cm rate of change

**Stopping Distance Calculation:**
- Signal processing: 200-300ms
- Valve response: 100-200ms
- Ram deceleration: 100-500ms
- **Total: 500-1500ms**
- **Travel at 20mm/sec: 10-30mm**
- **Design requirement: Detect UXO at minimum 100mm before contact**

With 250kg bombs detectable at 2-4m, safety margin is very large.

### 2.7 False Positive Management

**False Positive Rate Reality:**
- Up to 75% of magnetometer anomalies are false alarms
- Only ~1-6% of detected metallic objects are actual UXO
- In marine UXO clearance, only 6% of investigated targets are UXO

**Sources of False Positives:**
- Iron-rich soil/rocks
- Previous excavation debris
- Buried pipes/cables
- Agricultural iron
- Natural magnetite

**HIRT Policy:** Accept false positives, never push through unverified anomaly.

**Cost Analysis:**
- False positive: 10-40 minutes lost per occurrence
- False negative: Potential catastrophe
- **Conservative thresholds are mandatory**

### 2.8 Integration with HIRT MIT System

**Potential Interference:**
- HIRT MIT coils generate electromagnetic fields
- Could interfere with magnetometer readings

**Mitigation Approaches:**
1. Time-division multiplexing (alternate MIT and mag monitoring)
2. Spatial separation (mag at tip, MIT coils further back)
3. Frequency separation (different operating frequencies)
4. Shielding between sensor systems

**Recommendation:** Testing required to characterize interference before deployment

### 2.9 UXO Detection Cost Estimate

| Component | Estimated Cost |
|-----------|---------------|
| Miniature fluxgate sensor | $2,000-5,000 |
| Sensor housing/integration | $1,000-3,000 |
| Electronics (ADC, processor) | $500-1,500 |
| Signal processing firmware | $5,000-10,000 |
| Abort system modifications | $2,000-5,000 |
| Testing and qualification | $5,000-10,000 |
| **Total Integration** | **$15,000-35,000** |

---

## Part 3: Probe Insertion Methods

### 3.1 The Collapse Problem

**Borehole Collapse Timeline by Soil:**

| Soil Type | Stability Duration | Collapse Behavior |
|-----------|-------------------|-------------------|
| Loose/saturated sand | Seconds | Immediate flow |
| Medium sand | Minutes | Progressive slumping |
| Silt | Minutes to hours | Gradual caving |
| Soft clay | Hours | Squeeze/closure |
| Stiff clay | Days to permanent | Stable or very slow closure |

**Key Insight:** In bomb crater fill (mixed debris, loose fill), collapse is likely immediate to minutes.

### 3.2 Insertion Methods Comparison

**Method 1: Direct Push with Expendable Tip** ★ RECOMMENDED PRIMARY

Probe with integrated conical tip pushed directly, no separate hole creation.

| Factor | Assessment |
|--------|------------|
| Simplicity | Excellent - single operation |
| Soil contact | Excellent - displacement fit |
| Force required | 6-11 kN for 16mm at 3m |
| Collapse issue | Eliminated |
| Cost | Low |

**Method 2: Dual-Tube System (Geoprobe DT22 Style)** ★ RECOMMENDED FOR DIFFICULT SOILS

Outer casing protects hole during probe insertion, then withdrawn.

| Factor | Assessment |
|--------|------------|
| Reliability | Excellent - proven system |
| All soil types | Yes |
| Complexity | Medium |
| Cost | Medium-High |

**Method 3: Cased Hole with Withdrawal**

1. Create borehole with push to full depth
2. Insert thin-wall casing to stabilize
3. Lower probe into cased hole
4. Slowly withdraw casing
5. Soil collapses onto probe

**Method 4: Water Jet with Polymer Stabilization**

1. Create hole with water jet
2. Fill with biodegradable polymer fluid (xanthan gum)
3. Insert probe through gel
4. Gel degrades over 24-48 hours
5. Native soil collapses onto probe

### 3.3 CRITICAL: Avoid Bentonite

**Both research documents strongly warn against bentonite:**

> "Bentonite reduces soil resistivity by up to 80%"

| Material | Electrical Effect | HIRT Suitability |
|----------|------------------|------------------|
| Bentonite | Very conductive (80% resistivity reduction) | **PROHIBITED** |
| Cement | Highly conductive when wet | Poor |
| Cement-bentonite | Highly conductive | Poor |
| Polymer fluid (xanthan) | Minimal effect after degradation | **Acceptable** |
| Sand backfill | Neutral | Good |
| Native soil backfill | Matches formation | **Best** |

### 3.4 Soil Contact Solutions

**For Sandy/Granular Soils:**
- Natural collapse is sufficient
- Occurs within seconds to minutes
- Add water to improve conductivity if needed

**For Clay Soils:**
- May maintain annular gap
- Options: sand backfill, spring-loaded contact fins
- Avoid grout (electrical interference)

**For Mixed/Rubble Fill:**
- Dual-tube system recommended
- Sand backfill or native soil cuttings

### 3.5 Probe Design Recommendations

**Tip Design:**
- Geometry: 60-degree cone
- Material: Hardened plastic (PEEK) or aluminum
- Attachment: Push-fit with shear pin
- Diameter: 18-20mm (provides clearance for 16mm body)

**Contact System:**
- Primary: Natural soil collapse
- Backup: Spring-loaded contact fins
- Electrode surface: Gold-plated or stainless steel

**Active Contact Enhancement (for stiff soils):**

```
Collapsed (during insertion):
    |====|
    |====|  <- fins folded against body
    |====|

Deployed (after insertion):
    |    |
   /|====|\
   \|====|/  <- fins extended
   /|====|\
    |    |
```

### 3.6 Insertion Method by Soil Type

| Soil Type | Primary Method | Backup Method | Contact Method |
|-----------|---------------|---------------|----------------|
| Loose sand | Dual-tube casing | Expendable tip direct push | Natural collapse |
| Dense sand | Direct push | Pre-drill + insert | Natural collapse |
| Saturated sand | Dual-tube (REQUIRED) | N/A | Water balance |
| Soft clay | Direct push | Pre-drill + insert | Clay squeezing |
| Stiff clay | Pre-drill + insert | Direct push | Sand backfill or spring fins |
| Mixed/rubble | Dual-tube casing | Water jet + casing | Sand backfill |

---

## Part 4: Integrated System Design

### 4.1 Complete System Specifications

**Target System Performance:**
- Push force: 15 kN maximum
- Probe diameter: 16mm body, 18-20mm tip
- Target depth: 3 meters
- Push rate: 20mm/second (standard CPT rate)
- Total weight: <100 kg (3-person portable)

**System Components:**

| Component | Specification | Weight | Cost |
|-----------|--------------|--------|------|
| Helical anchors (3x) | 8-10" helix, 15-20 kN each | 30-45 kg | $800-1,500 |
| Hydraulic cylinder | 15 kN, 100mm stroke | 12-15 kg | $1,500-2,500 |
| Hand pump or power pack | 700 bar rated | 8-12 kg | $1,000-2,500 |
| Reaction frame | Aluminum, adjustable | 12-18 kg | $1,500-3,000 |
| Magnetometer system | Fluxgate, abort control | 2-4 kg | $15,000-35,000 |
| Probes (10x) | With expendable tips | 10-15 kg | $500-1,000 |
| Tools & accessories | Anchor installation, etc. | 5-8 kg | $500-1,000 |
| **Total** | | **80-120 kg** | **$20,800-46,500** |

### 4.2 Field Deployment Workflow

**Phase 1: Site Assessment**
1. Desktop UXO risk assessment (historical records)
2. Surface magnetometer survey if risk identified
3. Mark probe positions

**Phase 2: System Setup**
1. Position frame at first probe location
2. Install helical anchors (3x, triangular pattern)
3. Attach reaction frame to anchors
4. Connect hydraulic system

**Phase 3: UXO Clearance (if required)**
1. Option A: Push standard magnetometer cone to clear column
2. Option B: Use integrated magnetometer with automatic abort
3. Record all magnetic data

**Phase 4: Probe Installation**
1. Load HIRT probe with expendable tip
2. Begin push at 20mm/sec
3. Monitor push force and magnetometer (if integrated)
4. Stop at target depth
5. Retract push rods
6. Verify probe remains in place
7. Allow 5-10 minutes for soil settlement
8. Connect data acquisition

**Phase 5: Move to Next Position**
1. Disconnect frame from anchors (leave anchors for reuse)
2. Move frame to next position
3. Reinstall anchors if outside reach
4. Repeat Phase 4

### 4.3 Time Estimates

| Operation | Time per Probe |
|-----------|---------------|
| Anchor installation (new position) | 15-30 minutes |
| Frame repositioning (within anchor reach) | 5-10 minutes |
| UXO clearance (two-pass method) | 10-15 minutes |
| Probe push (3m at 20mm/sec) | 2.5 minutes |
| System setup and verification | 5-10 minutes |
| **Total (new position)** | **35-65 minutes** |
| **Total (adjacent position)** | **20-35 minutes** |

For a 50-probe survey with clustered positions: approximately 15-25 hours total push time

---

## Part 5: Risk Assessment

### 5.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient push force in dense soil | Medium | Medium | Test anchoring capacity, have fallback |
| Magnetometer doesn't fit in probe | Medium | High | Design enlarged head option from start |
| Borehole collapse before probe insertion | Medium | Medium | Use dual-tube as backup |
| MIT/magnetometer interference | Medium | Medium | Time-division multiplexing |
| Anchor pullout in soft soil | Low | Medium | Use multiple anchors, add ballast |
| Probe damage from obstruction | Medium | Low | Expendable tips, relocate if blocked |

### 5.2 Safety Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| UXO contact during push | Very Low (with detection) | Critical | Magnetometer system, two-pass clearance |
| UXO contact without detection | Extremely Low | Critical | Conservative thresholds, trained operators |
| Hydraulic system failure | Low | Low | Pressure relief, redundant abort |
| Personnel injury from equipment | Low | Medium | Training, PPE, proper procedures |

### 5.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| System too heavy for terrain | Medium | High | Modular design, minimal config option |
| Extended deployment time | Medium | Medium | Optimize workflow, parallel operations |
| Equipment failure in field | Low | High | Spare components, field repair kit |
| Environmental conditions (rain, cold) | Variable | Medium | Weather protection, seasonal planning |

---

## Part 6: Development Roadmap

### 6.1 Phase 1: Prototype Development (Months 1-6)

**Reaction Force System:**
- Source and test helical anchors in representative soils
- Design and fabricate lightweight reaction frame
- Procure/build hydraulic push system
- Bench test to 15 kN capacity

**UXO Detection:**
- Source miniature fluxgate sensors for evaluation
- Bench test sensitivity and noise performance
- Design enlarged probe head concept
- Develop signal processing algorithms

**Probe Insertion:**
- Design expendable tip for 16mm probe
- Prototype spring-loaded contact fins
- Test direct push in representative soils

**Budget:** ~$25,000

### 6.2 Phase 2: Integration (Months 6-12)

**System Integration:**
- Integrate magnetometer into probe head
- Integrate abort system with hydraulic control
- Assemble complete portable system
- Lab testing with ferrous test objects

**Field Trials (Non-UXO Sites):**
- Test in various soil conditions
- Characterize anchor performance
- Validate push force requirements
- Optimize operational procedures

**Budget:** ~$30,000

### 6.3 Phase 3: Validation (Months 12-18)

**Controlled UXO Site Testing:**
- Partner with EOD training facility or cleared UXO site
- Validate detection system with known targets
- Refine thresholds and procedures
- Develop operator training program

**Operational Deployment:**
- Pilot deployment at surveyed UXO risk sites
- Parallel operation with conventional UXO clearance
- Refine based on operational experience

**Budget:** ~$20,000

### 6.4 Total Development Investment

| Category | Estimate |
|----------|----------|
| Equipment and materials | $35,000-50,000 |
| Development labor | $25,000-40,000 |
| Testing and validation | $15,000-25,000 |
| Contingency (20%) | $15,000-25,000 |
| **Total** | **$90,000-140,000** |

---

## Part 7: Conclusions and Recommendations

### 7.1 Summary of Key Findings

1. **Portable push system is feasible** - 15 kN capacity achievable with 72-102 kg system using helical screw anchors

2. **UXO detection is mandatory and achievable** - Fluxgate magnetometers detect 250kg bombs at 2-4m; miniature sensors available at 18mm diameter

3. **Probe insertion is solvable** - Direct push with expendable tip works in most soils; dual-tube backup for saturated/unstable conditions

4. **Bentonite is prohibited** - 80% resistivity reduction makes it incompatible with ERT measurements

5. **Two-pass system offers highest safety** - Clear column with standard magnetometer cone, then insert HIRT probe

6. **Total system cost is reasonable** - $20,000-50,000 for complete field-ready system

### 7.2 Primary Recommendations

**Immediate Actions:**
1. Procure helical anchor test kit and verify capacity in target soils
2. Source WUNTRONIC WFG-110 or similar miniature fluxgate for evaluation
3. Design expendable push tip for 16mm fiberglass probe
4. Establish partnership with EOD facility for future validation

**System Design Decisions:**
1. Use helical screw anchors as primary reaction force system
2. Plan for enlarged probe head (20-25mm) to accommodate magnetometer
3. Design dual-tube backup system for difficult soils
4. Implement automatic abort with conservative thresholds

**Operational Protocol:**
1. Always conduct desktop UXO risk assessment
2. Use surface magnetometer survey at risk sites
3. Two-pass clearance recommended until integrated system proven
4. Never push through unverified magnetic anomaly
5. Accept false positives as cost of safety

### 7.3 Open Questions Requiring Further Research

1. **Anchor performance in specific crater fill materials** - Need field testing
2. **MIT/magnetometer interference characterization** - Need bench testing
3. **Optimal detection thresholds** - Need controlled testing with known targets
4. **Long-term probe survivability** - Need durability testing
5. **Regulatory requirements by jurisdiction** - Need legal review

---

## References

### Reaction Force Research
- Research A: portable-reaction-force-research-A.md
- Research B: portable-reaction-force-research-B.md

### UXO Detection Research
- Research A: uxo-detection-during-push-research-A.md
- Research B: uxo-detection-during-push-research-B.md

### Probe Insertion Research
- Research A: probe-insertion-methods-research-A.md
- Research B: probe-insertion-methods-research-B.md

### Key External Sources
- Fugro UXO detection systems
- Gouda Geo MagCone
- Royal Eijkelkamp Geomagnetic Module
- Geoprobe DT22 Dual-Tube System
- WUNTRONIC miniature fluxgate sensors
- CIRIA C681 UXO guidance
- Military load carrying standards (FM 21-18)

---

*Document compiled from parallel research efforts*
*For HIRT Geophysical Survey System development*
