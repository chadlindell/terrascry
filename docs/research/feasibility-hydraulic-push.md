# Feasibility Assessment: Hydraulic Static Push (CPT-Style) Systems for HIRT Probe Insertion

**Document Version:** 1.0
**Date:** January 2026
**Author:** HIRT Development Team

---

## Executive Summary

This document evaluates the feasibility of using hydraulic static push systems, similar to Cone Penetration Testing (CPT) equipment, for inserting HIRT probes at potential UXO sites. The analysis concludes that **hydraulic static push is highly feasible** for HIRT applications, with several viable equipment options ranging from commercial mini-CPT rigs to purpose-built systems.

### Key Findings

| Criterion | Assessment | Notes |
|-----------|------------|-------|
| Technical Feasibility | **Excellent** | Standard CPT can handle 12-16mm probes easily |
| Equipment Availability | **Good** | Multiple commercial options exist |
| UXO Safety | **Excellent** | Industry already uses CPT+magnetometer for UXO |
| Cost | **Moderate** | $50K-200K for equipment, <$10/hole operational |
| Portability | **Good** | Mini-rigs and anchor-based systems available |
| Remote Operation | **Feasible** | Automation exists (Geomil Hornet) |

**Recommended Approach:** A portable anchor-based hydraulic push system (similar to Gouda Geo 200kN Stand-Alone) modified for HIRT's smaller probe diameter, or a purpose-built lightweight rig.

---

## 1. Technical Feasibility

### 1.1 Can Standard CPT Equipment Handle HIRT Probes?

**YES - with significant margin.**

| Parameter | Standard CPT | HIRT Requirement | Assessment |
|-----------|-------------|------------------|------------|
| Probe diameter | 35.7mm (10cm^2) | 16mm | HIRT is **55% smaller** |
| Hole diameter | 36-44mm | 18-20mm | HIRT is **50% smaller** |
| Target depth | 20-60m typical | 2-3m | HIRT needs only **5-10%** of CPT depth |
| Push force available | 100-200 kN | ~5-15 kN estimated | **10-20x margin** |

**Conclusion:** Standard CPT equipment is vastly overpowered for HIRT applications. This is advantageous - it means lightweight, reduced-capacity systems will work perfectly.

### 1.2 Push Force Requirements for HIRT Probes

**Estimated push force for 16mm diameter probe:**

Using CPT resistance data and scaling for probe area:

| Soil Type | CPT Resistance (qc) | Force for 10cm^2 cone | Force for 2cm^2 HIRT (16mm) |
|-----------|--------------------|-----------------------|----------------------------|
| Soft clay | 0.5-2 MPa | 5-20 kN | **1-4 kN** |
| Stiff clay | 2-5 MPa | 20-50 kN | **4-10 kN** |
| Loose sand | 5-15 MPa | 50-150 kN | **10-30 kN** |
| Dense sand | 15-40 MPa | 150-400 kN | **30-80 kN** |

**Note:** These are conservative estimates. Actual forces may be lower due to:
- Tapered probe tip design
- Smooth fiberglass surface (lower friction)
- No friction sleeve like CPT cones

**Design target:** 50 kN (5 tons) push capacity covers all but the densest sands.

### 1.3 Mini-CPT and Small Diameter Options

Research-grade miniature CPT systems already exist:

| System | Diameter | Cross-Section | Application |
|--------|----------|---------------|-------------|
| Standard CPT | 35.7mm | 10 cm^2 | Field standard |
| Mini CPT | 26mm | 5 cm^2 | Research, centrifuge |
| Microcone | 15mm | 1.76 cm^2 | Lab calibration |
| Research mini-cone | 10mm | 0.79 cm^2 | Laboratory tests |
| **HIRT probe** | **16mm** | **2 cm^2** | **Well within range** |

Sources indicate that 10mm and 15mm diameter penetrometers have been developed and successfully used. HIRT's 16mm probe is slightly larger than these proven systems.

### 1.4 Depth Capability

| System Type | Typical Depth | HIRT Requirement | Assessment |
|-------------|---------------|------------------|------------|
| Handheld CPT | 2-3m | 2-3m | **Exact match** |
| Portable 420M | Up to 6m (20 ft) | 2-3m | **Adequate** |
| Mini CPT crawler | 10-30m | 2-3m | **Far exceeds** |
| Full CPT truck | 30-60m | 2-3m | **Overkill** |

**Conclusion:** Even the smallest CPT systems exceed HIRT depth requirements.

### 1.5 Soil Type Limitations

| Soil Type | CPT Suitability | HIRT Impact |
|-----------|-----------------|-------------|
| Soft clay | Excellent | Easy insertion |
| Stiff clay | Good | Moderate force needed |
| Sandy soils | Good | Most common at bomb sites |
| Gravelly soils | Challenging | May need pilot hole |
| Boulders/rock | Not suitable | Requires alternative method |

**For WWII bomb crater sites:** Typically disturbed, mixed fill over natural soil. Expected to be moderate difficulty - not pristine dense sand or intact clay. CPT approach well-suited.

---

## 2. Equipment Options

### 2.1 Commercial Mini-CPT Equipment

#### A.P. van den Berg Mini CPT Crawler
- **Weight:** < 1,600 kg
- **Width:** 780mm (fits through doorways)
- **Push force:** 100 kN (10 tons)
- **Anchoring:** Soil anchors for reaction force
- **Transport:** Mini van or small trailer
- **Estimated cost:** EUR 80,000-150,000

[Source: A.P. van den Berg](https://www.apvandenberg.com/onshore-cone-penetration-testing/mini-cpt-crawler)

#### In Situ Site Investigation Mini Rig CPT003
- **Weight:** 3.5 tonnes
- **Anchoring:** Automatic self-anchoring system (100mm spiral screw anchors)
- **Reaction force:** Up to 20 tonnes from anchors
- **Depth:** 10-30m depending on ground
- **Production:** 50-100m of testing per day
- **Transport:** Transit van or light truck

[Source: In Situ Site Investigation](https://insitusi.com/services/restricted-access-sites/)

#### Gouda Geo Griffin 200kN Crawler
- **Push force:** 200 kN (20 tons)
- **Anchoring:** Self-anchoring screw system
- **Features:** Auto-leveling, tracked base
- **Best for:** Larger surveys, difficult access

[Source: Gouda Geo-Equipment](https://gouda-geo.com/product/griffin-200-kn-crawler-based-cpt-penetrometer-rig)

### 2.2 Portable/Lightweight Systems (Non-Truck)

#### Gouda Geo 200kN Lightweight Stand-Alone Set
- **Push force:** 200 kN
- **Components:**
  - Penetrometer pusher (2 hydraulic cylinders)
  - Diesel-hydraulic power pack (Hatz 7.5 kW)
  - Ground anchors (4x, providing 10+ tons reaction)
  - Hydraulic wrench for anchor installation
  - Transport trolley
- **Transport:** Standard car + 2-axle trailer
- **Power:** Self-contained diesel unit
- **Estimated cost:** EUR 40,000-80,000

[Source: Gouda Geo-Equipment](https://gouda-geo.com/product/200-kn-lightweight-cpt-penetrometer-set)

#### Gouda Geo 100kN Stand-Alone Set
- **Push force:** 100 kN (may be adequate for HIRT)
- **Smaller, lighter** than 200kN version
- **Lower cost option**

[Source: Gouda Geo-Equipment](https://www.gouda-geo.com/products/cpt-equipment/cpt-penetrometer-rigs-other/100-kn-stand-alone-cpt-penetrometer-set)

#### Geoprobe 420M Portable Drilling Rig
- **Weight:** 425 lbs (193 kg) - manually liftable
- **Dimensions:** 23" wide x 62" tall (folded)
- **Force:** 12,000 lb (53 kN) percussion
- **Stroke:** 42 inches
- **Power:** Remote hydraulic power pack or auxiliary hydraulics
- **Depth:** Designed for < 20 ft (6m)
- **Transport:** Fits through standard doorways

[Source: Geoprobe Systems](https://geoprobe.com/drilling-rigs/420m-drill-rig)

**Note:** The 420M uses percussion (hammer) rather than pure static push, but could be adapted.

### 2.3 Hand-Portable Hydraulic Options

#### Gouda Geo Handheld CPT Penetrometer
- **Weight:** 11 kg complete set
- **Dimensions:** 59 x 19 x 28 cm
- **Depth:** Up to 1 meter
- **Reading:** Hydraulic dial (up to 10,000 kPa)
- **Limitations:** Human-powered, shallow depth only

[Source: Gouda Geo-Equipment](https://gouda-geo.com/product/handheld-cpt-penetrometer)

**Assessment:** Insufficient for 3m HIRT depth, but demonstrates concept viability.

### 2.4 Reaction Force Solutions

For any lightweight rig, reaction force is critical. Options:

#### Ground Anchors (Screw Type)
| Specification | Value |
|---------------|-------|
| Blade diameters | 220mm, 300mm, 400mm |
| Capacity (4 anchors) | 10+ tons (no extensions) |
| Material | High-tension steel, hardened tip |
| Installation | Hydraulic wrench or rig rotary drive |

[Source: Gouda Geo-Equipment](https://gouda-geo.com/product/ground-anchors)

#### Helical Anchor Capacity (Field Data)
| Soil Class | Capacity per Anchor |
|------------|---------------------|
| Dense sand/gravel (Class 5) | ~2,900 kg (6,500 lbs) |
| Medium sand/stiff clay (Class 6) | ~2,270 kg (5,000 lbs) |
| Loose sand/soft clay (Class 7) | ~1,135 kg (2,500 lbs) |

[Source: American Earth Anchors](https://americanearthanchors.com/load-capacity/)

**For HIRT (5-ton requirement):** 2-4 anchors in typical soil provides adequate reaction.

#### Vehicle Ballast
- Use machine's own weight
- Mini-rigs: 1.5-3.5 tonnes base
- Can be supplemented with water tanks or weights

#### Kentledge (Dead Weight)
- Concrete blocks, water tanks
- Simple but requires transport
- 500-2000 kg typical

---

## 3. Remote Operation Adaptation

### 3.1 Existing Automation in CPT Industry

#### Geomil Hornet-100 (State of the Art)
- **Automation:** Fully automated and remotely controlled
- **Features:**
  - 20-meter rod tower for automatic rod handling
  - Predefined maximum depth operation
  - Remote control of entire process
  - Automatic leveling (corrects slopes up to 10 degrees)
- **Weight:** 3 tons coupled to excavator (8.5 tons ballasted)
- **Power:** Excavator hydraulics
- **Applications:** CPT and UXO detection projects

**Quote:** "The Hornet can be configured as a highly automated and remotely controlled system which boosts efficiency and reduces manual handling and associated risk... removing the operator from harm's way when undertaking CPT or UXO detection projects."

[Source: Geomil Equipment](https://www.geomil.com/products/hornet)

### 3.2 Tethered Control Feasibility

**Current Industry Practice:**
- Real-time data acquisition already standard in CPT
- Magnetometer readings transmitted in real-time
- Operator monitors from safe distance
- "Abort" capability when anomalies detected

**For HIRT Adaptation:**
- Add tethered control of:
  - Push/retract hydraulics
  - Anchor deployment
  - Probe angle adjustment
- Control distance: 50-100m via cable or wireless
- Emergency stop capability essential

### 3.3 Real-Time Magnetometer Integration

**Already industry standard for UXO work:**

- **S-Magnetometer cones** (Fugro) - simultaneous UXO detection + geotechnical measurement
- **3D magnetic field measurement** for UXO, sheet piling, ground anchors
- **Detection radius:** ~2m for 50kg bomb
- **Real-time display** to operator for abort decisions

[Source: Fugro](https://www.fugro.com/news/business-news/2020/fugro-uxo-geotechnical-surveys-wismar-port-biomass-power-plant)

**For HIRT:** Could integrate HIRT's own magnetometer data as a "look-ahead" safety system during insertion.

### 3.4 Remote Operation Development Path

| Phase | Capability | Complexity |
|-------|-----------|------------|
| 1 | Tethered push control (wired) | Low |
| 2 | Wireless control (radio/WiFi) | Moderate |
| 3 | Semi-autonomous positioning | Moderate |
| 4 | Fully robotic operation | High |

**Recommendation:** Start with Phase 1 (tethered), which is proven technology.

---

## 4. Cost Analysis

### 4.1 Equipment Costs

#### New Equipment
| Equipment | Estimated Cost | Notes |
|-----------|---------------|-------|
| Gouda 200kN Stand-Alone | $50,000-100,000 | Best value for HIRT |
| A.P. van den Berg Mini Crawler | $100,000-180,000 | Premium option |
| In Situ Mini Rig CPT003 | $150,000-250,000 | Full capability |
| Geomil Hornet-100 | $200,000-350,000 | Includes automation |
| Geoprobe 420M | $25,000-40,000 | Percussion, not static push |

#### Used Equipment
| Equipment | Estimated Cost | Notes |
|-----------|---------------|-------|
| Used Geoprobe 7822DT | $130,000-215,000 | Full truck unit |
| Older Geoprobe direct push | $65,000-135,000 | 2010-2015 vintage |
| Basic used system | $25,000-70,000 | Older models |

[Source: Sun Machinery, Geoprobe Used Listings](https://www.sunmachinery.com/directpush%20and%20probe.html)

#### Rental Costs
| Equipment | Daily Rate | Monthly Rate |
|-----------|------------|--------------|
| Geoprobe 7822DT | ~$480/day | ~$10,000/month |
| CPT truck with operator | $3,000-4,000/day | N/A (includes operator) |
| Portable rig rental | $200-400/day | ~$4,000/month |

[Sources: Heavy Equipment Appraisal, Geoprobe Rentals](https://heavyequipmentappraisal.com/drilling-rig-cost/)

### 4.2 Per-Hole Operational Costs

**Assumptions for 50-hole survey:**
- Equipment amortized over 5 years, 100 surveys
- 2-person crew
- 1 day setup + 2 days operation

| Cost Category | Per Survey | Per Hole |
|---------------|------------|----------|
| Equipment amortization | $500-1,000 | $10-20 |
| Consumables (rod, tips) | $200-400 | $4-8 |
| Labor (3 crew-days) | $1,500-2,500 | $30-50 |
| Fuel/power | $100-200 | $2-4 |
| Transport | $300-500 | $6-10 |
| **Total** | **$2,600-4,600** | **$52-92** |

**Note:** Costs decrease significantly with larger surveys or dedicated equipment.

### 4.3 Comparison to Alternatives

| Method | Equipment Cost | Per-Hole Cost | Speed | UXO Safety |
|--------|----------------|---------------|-------|------------|
| **Hydraulic Push** | $50-200K | $50-90 | Fast | Excellent |
| Hand Auger | $50-200 | $5-10 | Slow | Poor |
| Pilot Rod + Hammer | $200-500 | $10-20 | Moderate | Fair |
| Water Jet | $2,000-5,000 | $20-40 | Moderate | Good |
| Rotary Drill | $50-150K | $100-200 | Slow | Poor |

**Conclusion:** Hydraulic push offers the best combination of speed and safety, with moderate equipment cost.

---

## 5. Practical Deployment

### 5.1 Setup Time for 20-50 Hole Grid

#### Anchor-Based System (Gouda 200kN Stand-Alone)
| Activity | Time per Position | Total for 50 Holes |
|----------|-------------------|-------------------|
| Position equipment | 2-3 min | 100-150 min |
| Install anchors (4x) | 5-10 min | 250-500 min |
| Push probe (3m) | 3-5 min | 150-250 min |
| Extract probe | 2-3 min | 100-150 min |
| Remove anchors | 3-5 min | 150-250 min |
| **Total per hole** | **15-26 min** | **12.5-22 hours** |

**Realistic production:** 20-30 holes/day with experienced crew.

#### Self-Anchoring Crawler (CPT003-style)
| Activity | Time per Position | Total for 50 Holes |
|----------|-------------------|-------------------|
| Drive to position | 1-2 min | 50-100 min |
| Auto-anchor | 2-3 min | 100-150 min |
| Push probe | 3-5 min | 150-250 min |
| Extract probe | 2-3 min | 100-150 min |
| Release anchors | 1-2 min | 50-100 min |
| **Total per hole** | **9-15 min** | **7.5-12.5 hours** |

**Realistic production:** 30-50 holes/day.

### 5.2 Crew Requirements

| System Type | Minimum Crew | Recommended |
|-------------|--------------|-------------|
| Handheld/manual | 2 | 2-3 |
| Stand-alone with anchors | 2 | 3 |
| Mini crawler | 1-2 | 2 |
| Truck-mounted | 2 | 2-3 |

**Required skills:**
- Equipment operation (training available from manufacturers)
- Basic hydraulics understanding
- Soil/ground assessment
- For UXO sites: UXO awareness certification

### 5.3 Transportation to Remote Sites

| System | Transport Vehicle | Access Requirements |
|--------|-------------------|---------------------|
| Gouda Stand-Alone | Car + trailer | Firm track, 2.5m width |
| Mini Crawler | Van or pickup | 1m path, can traverse soft ground |
| 420M Portable | SUV, manual carry | Doorways, stairs, any access |
| Full CPT Truck | 20-ton truck | Road access only |

**For UXO sites (bomb craters):** Mini crawler or portable systems preferred due to:
- Potentially damaged/unstable ground
- Limited road access
- Need for maneuvering around obstacles

### 5.4 Power Requirements

| System | Power Source | Power Required |
|--------|--------------|----------------|
| Stand-alone set | Diesel power pack | 7.5 kW (Hatz engine) |
| Mini crawler | Onboard diesel | 10-20 kW |
| Portable 420M | External hydraulics or power pack | 5-10 kW |
| Electric option | Generator or grid | 10-15 kW |

**Self-sufficiency:** Diesel-hydraulic power packs provide complete independence from grid power - essential for remote sites.

---

## 6. Purpose-Built HIRT Hydraulic Push Rig Concept

### 6.1 Design Rationale

Commercial CPT equipment is optimized for:
- 35mm+ diameter probes
- 20-60m depth capability
- Multiple soil testing modes

HIRT needs:
- 16mm diameter probes only
- 3m depth maximum
- Rapid deployment for many holes
- Maximum portability
- UXO-safe remote operation

**A purpose-built rig could be 50-70% lighter and simpler than commercial CPT equipment.**

### 6.2 Concept Specifications

#### HIRT Micro-Push Rig (Concept)

| Specification | Target Value | Notes |
|---------------|--------------|-------|
| **Push force** | 50 kN (5 tons) | Covers most soils to 3m |
| **Stroke** | 1.5m | Single-stroke to depth |
| **Probe diameter** | 16mm | HIRT standard |
| **Maximum depth** | 3.5m | Slight margin over requirement |
| **Weight** | 150-250 kg | 2-person portable |
| **Dimensions** | 0.6m x 0.6m x 1.8m (collapsed) | Fits in SUV/pickup |
| **Power** | 5 kW electric or gas | Battery-powered option |
| **Anchoring** | 2-4 quick-deploy screw anchors | 60-90 second deployment |
| **Control** | Tethered remote (50m) | Phase 1 UXO safety |

#### System Components

```
HIRT Micro-Push Rig - Component Diagram
========================================

     [Tether Control]----50m cable----[Operator Station]
            |
            v
    +---------------+
    | Control Box   |  <-- Hydraulic valves, safety systems
    | (Waterproof)  |
    +-------+-------+
            |
    +-------v-------+
    |   Push Frame  |  <-- Aluminum/steel frame
    |   +-------+   |
    |   |Cylinder|  |  <-- 50kN hydraulic cylinder
    |   |  ||   |   |      1.5m stroke
    |   |  ||   |   |
    |   +--||---+   |
    |      ||       |
    |   [Probe]     |  <-- 16mm HIRT probe
    |      ||       |
    +------||-------+
           ||
    =======||=======  <-- Ground level
    /    Anchor    \  <-- 2-4 screw anchors
   /                \
```

#### Anchoring System Detail

| Component | Specification |
|-----------|---------------|
| Anchor type | 200mm helical screw |
| Quantity | 4 (provides 8-10 ton reaction) |
| Installation | Cordless drill adapter or hand crank |
| Install time | 15 seconds each |
| Total anchor time | < 90 seconds |

#### Power Options

| Option | Weight | Runtime | Notes |
|--------|--------|---------|-------|
| Battery (48V LiFePO4) | 15 kg | 30-50 holes | Silent, no emissions |
| Gas generator | 25 kg | All day | Refuelable |
| Electric (grid) | Cable | Unlimited | Where available |

### 6.3 Estimated Development Costs

| Phase | Activities | Estimated Cost | Timeline |
|-------|------------|----------------|----------|
| **Design** | CAD, FEA analysis, specifications | $15,000-25,000 | 2-3 months |
| **Prototype** | Frame fabrication, hydraulics, control | $30,000-50,000 | 3-4 months |
| **Testing** | Field trials, iteration | $10,000-20,000 | 2-3 months |
| **Documentation** | Manuals, training materials | $5,000-10,000 | 1 month |
| **Total Prototype** | | **$60,000-105,000** | **8-11 months** |

#### Production Unit Cost Estimate
| Quantity | Unit Cost |
|----------|-----------|
| 1 (prototype) | $60,000-80,000 |
| 5 units | $35,000-50,000 each |
| 10+ units | $25,000-40,000 each |

### 6.4 Alternative: Modify Commercial Equipment

Instead of building from scratch, modify existing equipment:

| Base System | Modifications | Estimated Cost |
|-------------|---------------|----------------|
| Gouda 100kN Stand-Alone | Smaller rod adapters, tether control | $60,000-90,000 |
| Geoprobe 420M | Static push conversion, remote control | $40,000-70,000 |
| Chinese CPT systems | Quality upgrades, safety systems | $30,000-60,000 |

**Recommendation:** For immediate deployment, modify commercial system. For long-term or high-volume use, develop purpose-built rig.

---

## 7. Recommendations

### 7.1 Recommended Equipment Approach

#### Immediate/Prototype Phase
**Purchase:** Gouda Geo 100kN or 200kN Stand-Alone System
- Proven technology
- Adequate for HIRT requirements
- Transportable by car + trailer
- Available now
- **Budget:** $50,000-80,000

**Modifications needed:**
1. Rod adapter for 16mm HIRT probes
2. Extended tether control (50m cable)
3. Quick-deploy anchor kit
4. Battery-powered hydraulic option (future)

#### Production Phase
**Develop:** Purpose-built HIRT Micro-Push Rig
- Optimized for HIRT specifications
- 50-70% lighter than commercial
- Battery-powered option
- Integrated remote control
- **Development budget:** $80,000-120,000

### 7.2 Phased Development Path

| Phase | Timeline | Deliverable | Budget |
|-------|----------|-------------|--------|
| **1. Validation** | 0-6 months | Test with rental CPT equipment | $10,000 |
| **2. Procurement** | 6-12 months | Acquire modified commercial system | $60,000-90,000 |
| **3. Operation** | 12-24 months | Deploy on pilot surveys | Operational costs |
| **4. Optimization** | 24-36 months | Design purpose-built rig | $80,000-120,000 |
| **5. Production** | 36+ months | Manufacture HIRT-specific rigs | $25,000-40,000/unit |

### 7.3 Key Modifications Needed

Regardless of equipment choice, these modifications are essential:

| Modification | Purpose | Complexity |
|--------------|---------|------------|
| 16mm rod adapter | Interface with HIRT probes | Low |
| Extended remote control | UXO safety distance | Moderate |
| Quick-release anchors | Faster setup | Low |
| Probe guide system | Ensure verticality | Low |
| Integrated data logging | Record push force, depth | Moderate |
| Emergency stop (remote) | Safety | Low |

### 7.4 Safety Considerations for UXO Sites

| Requirement | Solution |
|-------------|----------|
| Minimum standoff distance | 50m tethered control |
| Real-time abort capability | Instant hydraulic release |
| Magnetometer pre-scan | Integrate with HIRT sensors |
| Controlled push rate | 20mm/s standard (CPT spec) |
| Emergency procedures | Developed from UXO industry protocols |

### 7.5 Next Steps

1. **Rent commercial CPT equipment** for proof-of-concept testing
2. **Validate push force requirements** in representative soils
3. **Test HIRT probe insertion/extraction** cycles
4. **Develop rod adapter** for standard CPT equipment
5. **Specify tethered control requirements**
6. **Engage with equipment suppliers** for custom options

---

## References and Sources

### Commercial Equipment Manufacturers
- [Gouda Geo-Equipment](https://gouda-geo.com/) - Stand-alone and lightweight CPT systems
- [A.P. van den Berg](https://www.apvandenberg.com/) - Mini CPT crawler, premium systems
- [Geomil Equipment](https://www.geomil.com/) - Hornet automated system
- [Geoprobe Systems](https://geoprobe.com/) - Direct push and portable systems
- [In Situ Site Investigation](https://insitusi.com/) - Mini Rig CPT003

### Technical References
- [Cone Penetration Testing Guide](https://www.novotechsoftware.com/downloads/PDF/en/Ref/CPT-Guide-5ed-Nov2012.pdf) - Robertson & Cabal
- [CPT Interpretation](https://www.geoengineer.org/education/site-characterization-in-situ-testing-general/cone-penetration-testing-cpt) - Geoengineer.org
- [Direct Push Technology](https://clu-in.org/characterization/technologies/dpp.cfm) - EPA CLU-IN

### UXO Detection with CPT
- [Fugro UXO+Geotechnical Surveys](https://www.fugro.com/news/business-news/2020/fugro-uxo-geotechnical-surveys-wismar-port-biomass-power-plant)
- [Intrusive UXO Survey Methods](https://www.brimstoneuxo.com/survey-uxo/intrusive-uxo-survey/)
- [Magnetometer Probes for UXO](https://insitusi.com/case-studies/unexploded-ordnance-uxo-intrusive-magnetometer-system-in-action/)

### Cost and Rental Information
- [Heavy Equipment Appraisal - Drilling Rig Costs](https://heavyequipmentappraisal.com/drilling-rig-cost/)
- [Geoprobe Rentals](https://geoproberentals.com/)
- [Used Geoprobe Listings](https://www.sunmachinery.com/directpush%20and%20probe.html)

---

*Document prepared for HIRT Development Team - January 2026*
