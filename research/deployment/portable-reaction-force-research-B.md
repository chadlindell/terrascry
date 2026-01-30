# Portable Reaction Force Solutions for Lightweight Soil Penetration Equipment

**Document Version:** 1.0
**Date:** January 2026
**Author:** HIRT Research Team

---

## Executive Summary

This document presents comprehensive research on lightweight anchoring and reaction force solutions for portable soil penetration equipment. The target application is pushing 16mm probes to 2-3m depth using hydraulic static push in remote locations where heavy vehicles cannot access.

### Key Requirements

| Parameter | Requirement |
|-----------|-------------|
| Probe diameter | 16mm |
| Target depth | 2-3m |
| Required reaction force | 5-15 kN |
| Portability | Human-portable over 500m-1km rough terrain |
| Crew size | 2-3 people |

### Summary of Findings

| Solution Category | Feasibility | Weight Range | Capacity | Recommended |
|------------------|-------------|--------------|----------|-------------|
| Helical screw anchors | **Excellent** | 5-15 kg each | 5-25 kN each | Yes |
| Lightweight A-frame with anchors | **Good** | 30-80 kg system | 15-50 kN | Yes |
| Partial ballast + anchor hybrid | **Good** | 50-100 kg ballast | 10-20 kN | Yes |
| Deadman/plate anchors | **Moderate** | Variable | 10-30 kN | Situational |
| Vehicle cable anchor | **Good** | Minimal | 20+ kN | When vehicle nearby |

**Recommended Solution:** Modular aluminum A-frame system with 2-4 helical screw anchors, total portable weight of 60-100 kg distributed among 2-3 people.

---

## 1. Soil Mechanics and Force Calculations

### 1.1 Cone Penetration Resistance by Soil Type

Based on Cone Penetration Test (CPT) data, typical cone resistance (qc) values by soil type:

| Soil Type | Cone Resistance (qc) | Typical Range | Notes |
|-----------|---------------------|---------------|-------|
| **Soft clay** | 0.5-2 MPa | 500-2,000 kPa | Easy penetration |
| **Firm clay** | 2-4 MPa | 2,000-4,000 kPa | Moderate resistance |
| **Stiff clay** | 4-8 MPa | 4,000-8,000 kPa | Higher forces needed |
| **Very stiff clay** | 8-15 MPa | 8,000-15,000 kPa | Challenging |
| **Loose sand** | 2-5 MPa | 2,000-5,000 kPa | Variable |
| **Medium dense sand** | 5-15 MPa | 5,000-15,000 kPa | Common condition |
| **Dense sand** | 15-30 MPa | 15,000-30,000 kPa | High resistance |
| **Very dense sand** | 30-50+ MPa | 30,000-50,000+ kPa | Maximum resistance |
| **Silt** | 1-10 MPa | 1,000-10,000 kPa | Highly variable |

Sources: [Geoengineer.org CPT Interpretation](https://www.geoengineer.org/education/site-characterization-in-situ-testing-general/cone-penetration-testing-cpt), [Robertson CPT Guide](https://www.cpt-robertson.com/PublicationsPDF/CPT-Guide-7th-Final-SMALL.pdf)

### 1.2 Skin Friction (Shaft Friction) Values

Friction acting on the rod/probe shaft during penetration:

| Soil Type | Typical Sleeve Friction (fs) | Limiting Values |
|-----------|------------------------------|-----------------|
| Soft clay | 5-25 kPa | - |
| Stiff clay | 25-75 kPa | Max 380 kPa |
| Loose sand | 10-30 kPa | - |
| Dense sand | 30-100 kPa | Max 200 kPa |
| Gravel | 50-150 kPa | - |

Sources: [ScienceDirect - Shaft Friction](https://www.sciencedirect.com/topics/engineering/shaft-friction), [Piles Capacity Reference Manual](https://hetge.com/PCref/granular-soils/)

### 1.3 Total Force Calculation for 16mm Probe

**Formula:**
```
Total Force (F_total) = Tip Resistance (F_tip) + Shaft Friction (F_shaft)

Where:
F_tip = qc × A_tip
F_shaft = fs × A_shaft × depth

For 16mm probe:
A_tip = π × (0.008m)² = 2.01 × 10⁻⁴ m² ≈ 2 cm²
A_shaft per meter = π × 0.016m × 1m = 0.0503 m²/m
```

**Calculated Forces for 16mm Probe at 3m Depth:**

| Soil Type | qc (MPa) | F_tip (kN) | fs (kPa) | F_shaft (kN) | **F_total (kN)** |
|-----------|----------|------------|----------|--------------|------------------|
| Soft clay | 1 | 0.2 | 15 | 2.3 | **2.5** |
| Firm clay | 3 | 0.6 | 40 | 6.0 | **6.6** |
| Stiff clay | 6 | 1.2 | 60 | 9.0 | **10.2** |
| Loose sand | 4 | 0.8 | 20 | 3.0 | **3.8** |
| Medium sand | 10 | 2.0 | 50 | 7.5 | **9.5** |
| Dense sand | 20 | 4.0 | 80 | 12.0 | **16.0** |
| Very dense sand | 35 | 7.0 | 120 | 18.1 | **25.1** |

**Design Recommendation:** Target 15 kN reaction force capacity for most soils, with 25 kN for dense sand conditions.

### 1.4 Force vs. Depth Relationship

The total pushing force increases approximately linearly with depth after initial penetration:

| Depth | Soft Clay | Medium Sand | Dense Sand |
|-------|-----------|-------------|------------|
| 0.5m | 0.6 kN | 2.3 kN | 5.0 kN |
| 1.0m | 1.1 kN | 4.0 kN | 8.5 kN |
| 1.5m | 1.6 kN | 5.8 kN | 12.0 kN |
| 2.0m | 2.0 kN | 7.5 kN | 15.5 kN |
| 2.5m | 2.5 kN | 9.3 kN | 19.0 kN |
| 3.0m | 2.9 kN | 11.0 kN | 22.5 kN |

**Note:** Shaft friction dominates total force at depths beyond 1m for small-diameter probes.

---

## 2. Ground Anchor Technologies

### 2.1 Helical Screw Anchors

Helical (screw-in) anchors are the most practical solution for portable applications.

#### Capacity by Size and Soil Type

| Anchor Size | Shaft Dia | Helix Dia | Class 5 Soil | Class 6 Soil | Class 7 Soil |
|-------------|-----------|-----------|--------------|--------------|--------------|
| 3/4" x 6" x 66" | 19mm | 150mm | 29 kN (6,500 lbs) | 22 kN (5,000 lbs) | 11 kN (2,500 lbs) |
| 1" x 8" x 66" | 25mm | 200mm | 49 kN (11,000 lbs) | 40 kN (9,000 lbs) | 27 kN (6,000 lbs) |
| 1-1/4" x 10" x 66" | 32mm | 250mm | 58 kN (13,000 lbs) | 44 kN (10,000 lbs) | 31 kN (7,000 lbs) |

**Soil Class Definitions:**
- **Class 5:** Medium dense coarse sand, sandy gravels, stiff to very stiff silts and clays
- **Class 6:** Loose to medium dense fine to coarse sand, firm to stiff clays, compacted fill
- **Class 7:** Loose fine sand, alluvium, loess, soft-firm clays, flood plain soils

Sources: [Lifting.com Earth Anchors](https://lifting.com/earth-anchor-helix-034x6x66.html), [American Earth Anchors](https://americanearthanchors.com/load-capacity/)

#### Helical Anchor Torque-Capacity Correlation

The relationship between installation torque and ultimate capacity:

```
Q_ult = K_t × T

Where:
Q_ult = Ultimate capacity (kN)
K_t = Torque factor (typically 10 ft⁻¹ or 33 m⁻¹ for square shaft)
T = Installation torque (ft-lbs or kN-m)
```

| Installation Torque | Estimated Capacity (Kt=10) |
|--------------------|---------------------------|
| 500 ft-lbs (0.68 kN-m) | 5,000 lbs (22 kN) |
| 1,000 ft-lbs (1.36 kN-m) | 10,000 lbs (44 kN) |
| 2,000 ft-lbs (2.71 kN-m) | 20,000 lbs (89 kN) |

**Minimum Depth Requirements:**
- Each helix should be embedded at least 3 feet (0.9m) vertically
- Minimum depth = 6× helix diameter along anchor shaft

Sources: [Hubbell Helical Pile Capacity](https://blog.hubbell.com/en/chancefoundationsolutions/3-methods-to-determine-helical-pile-capacity), [Helical Anchors Inc. Manual](https://helicalanchorsinc.com/wp-content/uploads/2020/05/HAI-Engineering-Manual-min.pdf)

#### Installation Methods

| Method | Tool | Installation Time | Best For |
|--------|------|-------------------|----------|
| Hand crank (T-bar) | Steel bar through anchor eye | 5-10 min/anchor | Soft soils, small anchors |
| Cordless impact wrench | 1/2" or 3/4" drive adapter | 1-3 min/anchor | Most conditions |
| Electric drill with adapter | Speed Staker or similar | 30-90 sec/anchor | Fast deployment |
| Hydraulic wrench | Powered hydraulic tool | 30-60 sec/anchor | Difficult soils |

Sources: [American Earth Anchors Installation](https://americanearthanchors.com/installation/), [Amazon Speed Staker](https://www.amazon.com/Keyfit-Tools-Trampoline-Seconds-Functional/dp/B087SS5R4C)

### 2.2 American Earth Anchors Penetrator Series

Lightweight aluminum screw anchors designed for portable applications:

| Model | Length | Capacity | Weight | Material |
|-------|--------|----------|--------|----------|
| PE9 | 9" (23cm) | ~1,000 lbs (4.4 kN) | Light | Aluminum |
| PE18 | 18" (46cm) | ~2,500 lbs (11 kN) | Light | Aluminum |
| PE26 | 26" (66cm) | ~4,000 lbs (18 kN) | ~2 lbs | Aluminum |
| PE36 | 36" (91cm) | ~6,000 lbs (27 kN) | ~3 lbs | Aluminum |
| PE46 | 46" (117cm) | ~8,000 lbs (36 kN) | ~4 lbs | Aluminum |

**Installation Notes:**
- Install at same angle as load direction
- Minimum spacing = anchor length (reduces capacity by 15% at half spacing)
- Heat-treated 356 aluminum construction
- Reusable with proper care

Sources: [American Earth Anchors Official](https://americanearthanchors.com/), [American Earth Anchors Technical Specs](https://americanearthanchors.com/technical-specs/)

### 2.3 Percussion-Driven Anchors (Duckbill Type)

**WARNING: UXO CONCERN**

Percussion-driven anchors use hammer impacts to drive stakes into ground. This method is **NOT RECOMMENDED** for potential UXO sites due to:
- Impact shock could initiate detonation
- Vibration transmitted through soil
- No controlled penetration rate

If used in non-UXO environments:

| Anchor Type | Capacity | Installation Method |
|-------------|----------|---------------------|
| Small duckbill | 1,000-3,000 lbs | Sledgehammer |
| Medium duckbill | 3,000-6,000 lbs | Pneumatic hammer |
| Large duckbill | 6,000-12,000 lbs | Power driver |

**UXO Site Protocol:**
- Magnetometer survey required before any intrusive work
- Avoid percussion methods entirely
- Use screw-in or hydraulic installation only
- Maintain controlled penetration rates

Sources: [AGS UXO Safety Guidance](https://www.ags.org.uk/item/safety-guidance-unexploded-ordnance/), [Wikipedia - Unexploded Ordnance](https://en.wikipedia.org/wiki/Unexploded_ordnance)

### 2.4 Deadman and Plate Anchors

Buried horizontal anchors using soil passive resistance:

**Capacity Formula:**
```
Capacity depends on:
- Burial depth (h)
- Plate width (B)
- Soil type (φ, γ)
- Pull angle (α)

General relationship: Capacity increases non-linearly with depth
```

**Practical Capacity Estimates (Deadman Off-Road Data):**

| Burial Depth | Soft Sand | Medium Soil | Hard Pack |
|--------------|-----------|-------------|-----------|
| 20" (50cm) | ~1,100 lbs | ~2,000 lbs | ~3,000 lbs |
| 24" (60cm) | ~1,500 lbs | ~2,500 lbs | ~4,000 lbs |
| 30" (75cm) | ~2,500 lbs | ~4,000 lbs | ~6,000 lbs |
| 36" (90cm) | ~4,000 lbs | ~6,000 lbs | ~8,000+ lbs |

**Advantages:**
- No specialized equipment needed
- Uses local materials (logs, plates)
- Good capacity in suitable soils

**Disadvantages:**
- Requires digging (labor intensive)
- Disturbs site
- Slow installation/removal
- Not practical for multiple relocations

Sources: [Deadman Off-Road Capacity](https://www.deadmanoffroad.com/pages/capacity), [Oregon OSHA Anchoring Guidelines](https://osha.oregon.gov/edu/grants/train/documents/osu-guidelines-for-safe-anchoring.pdf)

### 2.5 Anchor Depth vs. Holding Capacity Summary

| Anchor Type | Min Depth for Full Capacity | Typical Install Time |
|-------------|----------------------------|---------------------|
| Helical 6" helix | 3 ft (0.9m) | 1-5 min |
| Helical 10" helix | 4 ft (1.2m) | 2-8 min |
| Penetrator PE26 | 26" (0.66m) | 1-2 min |
| Penetrator PE46 | 46" (1.17m) | 2-4 min |
| Deadman plate | 2-3 ft (0.6-0.9m) | 15-30 min |

---

## 3. Lightweight Frame Designs

### 3.1 A-Frame Configuration

An A-frame provides vertical reaction force using angled legs that transfer load to ground anchors.

**Basic Design Principles:**
```
                    [Hydraulic Cylinder]
                          ||
                    +-----||-----+
                   /      ||      \
                  /       ||       \
                 /     [Probe]      \
                /         |          \
               /          |           \
              /           |            \
         [Anchor]    [Ground]     [Anchor]
```

**Design Parameters:**

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| Apex angle | 45-60 degrees | Balance stability vs. footprint |
| Leg length | 1.5-2.0m | Allows 3m probe insertion |
| Base spread | 1.5-2.5m | Stability vs. transport |
| Material | 6061-T6 Aluminum | Strength-to-weight optimized |
| Tube diameter | 50-75mm | Withstand bending loads |
| Wall thickness | 3-5mm | Structural integrity |

**Weight Estimates by Material:**

| Frame Material | Estimated Weight | Strength/Weight | Cost |
|----------------|------------------|-----------------|------|
| Aluminum 6061-T6 | 15-25 kg | Excellent | Medium |
| Steel (mild) | 35-55 kg | Good | Low |
| Carbon fiber | 8-15 kg | Superior | High |
| Fiberglass | 12-20 kg | Good | Medium |

Sources: [Aluminum Frame Design](https://anglelock.com/blog/aluminum-frame-design-vs-lightweight-structural-steel), [MiniTec Aluminum Systems](https://www.minitecsolutions.com/aluminum-framing/)

### 3.2 Tripod Configuration

**Advantages over A-frame:**
- Self-standing stability
- 360-degree work access
- Better on uneven ground
- Can incorporate pulley system

**Commercial Example - TMG SPT Tripod:**
- Aluminum construction
- Two-section design for transport
- Compatible with capstan motor system
- Used for Standard Penetration Testing

Sources: [TMG SPT Tripod](https://tmgmfg.com/spt-tripod-soil-sampling-equiptment-spt-drop-hammer-aluminum-structure-spt-testing)

### 3.3 Reaction Force Transfer to Anchors

**Direct Anchor Connection:**
```
Frame leg → Cable/strap → Anchor eye

Load path: Vertical push force → Frame → Tension in legs → Anchor pullout resistance
```

**Calculation for Anchor Load:**

For a symmetric A-frame with apex angle θ:
```
Anchor tension (T) = F_push / (2 × sin(θ/2))

Example for 15 kN push, 60° apex angle:
T = 15 kN / (2 × sin(30°)) = 15 kN / (2 × 0.5) = 15 kN per anchor
```

| Apex Angle | Push Force | Load per Anchor (2 anchors) |
|------------|------------|----------------------------|
| 45° | 15 kN | 19.6 kN |
| 60° | 15 kN | 15.0 kN |
| 90° | 15 kN | 10.6 kN |
| 120° | 15 kN | 8.7 kN |

**Note:** Wider apex angles reduce anchor loads but increase frame footprint.

### 3.4 Collapsible/Packable Design Concepts

**Folding A-Frame Design:**

| Feature | Implementation |
|---------|---------------|
| Hinged apex | Quick-release pin joint |
| Telescoping legs | Nested tubes with locking collars |
| Folded dimensions | 0.4m × 0.4m × 1.5m |
| Setup time | 3-5 minutes |

**Modular Assembly Design:**

| Feature | Implementation |
|---------|---------------|
| Tube sections | 0.5-0.75m lengths |
| Connections | Quick-connect sleeve joints |
| Packed dimensions | 0.6m × 0.3m × 0.3m |
| Setup time | 5-10 minutes |

### 3.5 Commercial Lightweight Frame Examples

| Product | Weight | Application | Price Range |
|---------|--------|-------------|-------------|
| Little Beaver Tripod | ~40 kg | Auger lifting, 6-30 ft | $1,500-2,500 |
| Vestil Aluminum Gantry | 50-150 kg | 1/2 to 2 ton capacity | $1,000-5,000 |
| Custom fabrication | 20-50 kg | Purpose-built | $2,000-8,000 |

Sources: [Little Beaver Tripod](https://littlebeaverstore.com/store/little-beaver-tripod-auger-lifting-frame-for-depths-6-to-30-tripod.html), [Vestil Aluminum Gantry](https://www.vestil.com/product.php?FID=522)

---

## 4. Human-Portable Weight Limits

### 4.1 Military Load Carrying Standards

Extensive military research on human load carrying:

| Standard | Recommended Max Load | Source |
|----------|---------------------|--------|
| US Army FM 21-18 Fighting Load | 48 lbs (22 kg) | Army doctrine |
| US Army Approach March Load | 72 lbs (33 kg) | Army doctrine |
| Naval Research Advisory Committee | 50 lbs (23 kg) | 2007 report |
| General Shinseki Goal | 50 lbs (23 kg) | 2010 target |
| Historical research (1800s) | 48 lbs (22 kg) | Long-term data |

**Body Weight Ratio Guidelines:**

| Condition | Max Load (% body weight) | For 80kg person |
|-----------|-------------------------|-----------------|
| Combat/sustained | 30% | 24 kg |
| Approach march | 40% | 32 kg |
| Short distance (<1km) | 45% | 36 kg |
| Very short burst | 50%+ | 40 kg+ |

Sources: [CNAS Soldier's Heavy Load](https://www.cnas.org/publications/reports/the-soldiers-heavy-load-1), [Backpacks Global Military Guide](https://backpacks.global/how-heavy-are-military-backpacks/)

### 4.2 Expedition and Field Equipment Standards

| Activity Type | Recommended Pack Weight | Notes |
|---------------|------------------------|-------|
| Day hiking | 10-15% body weight | Minimal gear |
| Backpacking (multi-day) | 20-25% body weight | Self-sufficient |
| Technical terrain | 25-30% body weight | Slower pace acceptable |
| Maximum sustainable | 35% body weight | Injury risk increases sharply |
| Equipment portage | 40-45% body weight | Short distances only |

Sources: [Rab Backpack Calculator](https://rab.equipment/us/rab-lab/backpack-weight-calculator), [Outdoor Adventure Training](https://outdooradventuretraining.com/2025/05/04/packs-pounds-percentages/)

### 4.3 Realistic Carry Limits for 2-3 Person Team

**Assumptions:**
- Average person: 75-85 kg
- Trained, fit field workers
- Distance: 500m-1km over rough terrain
- Multiple trips acceptable

| Crew Size | Total Portable Weight | Per Person | Notes |
|-----------|----------------------|------------|-------|
| 2 people | 45-70 kg | 22-35 kg | Sustainable for 1km |
| 3 people | 70-100 kg | 23-33 kg | Comfortable for 1km |
| 3 people (short) | 100-130 kg | 33-43 kg | <500m only |

**Recommended Equipment Weight Budget:**

| Component | Target Weight | Justification |
|-----------|---------------|---------------|
| Frame structure | 15-25 kg | Aluminum construction |
| Hydraulic cylinder | 10-15 kg | Compact design |
| Hand pump or power pack | 5-15 kg | Manual or battery |
| Anchors (4x) | 8-20 kg | Depending on type |
| Rods/probe | 5-10 kg | 3m worth |
| Tools/accessories | 5-10 kg | Wrenches, cables |
| **Total system** | **48-95 kg** | **Fits 2-3 person carry** |

### 4.4 Alternative Transport Methods

| Method | Capacity | Terrain Capability | Notes |
|--------|----------|-------------------|-------|
| Backpack carry | 20-35 kg/person | Any walkable terrain | Most versatile |
| Two-person stretcher carry | 50-80 kg | Moderate terrain | Awkward on slopes |
| Wheeled cart | 100-200 kg | Firm paths, mild slopes | Limited terrain |
| Game cart/deer cart | 80-150 kg | Trails, moderate terrain | Good compromise |
| Sled/travois | 50-100 kg | Snow, grass, sand | Seasonal/terrain specific |
| ATV trailer | 200-400 kg | ATV-accessible areas | Requires vehicle |

**Wheeled Cart Options:**

| Product | Capacity | Weight | Features | Price |
|---------|----------|--------|----------|-------|
| Rhino Cart | 900 kg | 15 kg | All-terrain wheels, expands 6-46" | $200-400 |
| Proaim Vanguard | 200 kg | ~25 kg | Collapsible, 8 configurations | $500-800 |
| Game cart (various) | 100-200 kg | 10-20 kg | Designed for rough terrain | $100-300 |

Sources: [Rhino Cart](https://rhinocart.com/), [Proaim Vanguard Cart](https://www.proaim.com/products/proaim-vanguard-collapsible-utility-production-cart-for-film-television-photo-industry)

---

## 5. Hybrid Solutions

### 5.1 Partial Anchor + Partial Ballast Combination

Combining anchors with added weight reduces demands on each system:

**Design Concept:**
```
Total Reaction = Anchor Capacity + Ballast Weight

Example for 15 kN requirement:
- 2 anchors @ 5 kN each = 10 kN
- 50 kg ballast = 0.5 kN
- Frame + equipment weight = 40 kg = 0.4 kN
- Operator weight on frame = 80 kg = 0.8 kN
- Total = 11.7 kN (marginal)

Better: Add water ballast on-site:
- 2 anchors @ 5 kN each = 10 kN
- 50L water (filled on-site) = 0.5 kN
- Equipment = 0.4 kN
- Two operators standing on frame = 1.6 kN
- Total = 12.5 kN
```

**Advantages:**
- Smaller anchors needed
- Less critical anchor installation
- Uses operator weight productively
- Water ballast: no transport weight (if water available)

**Ballast Options:**

| Ballast Type | Weight/Volume | Transportable | On-Site Sourcing |
|--------------|---------------|---------------|------------------|
| Water bags | 1 kg/L | Yes (empty) | Rivers, wells |
| Sand bags | 1.5 kg/L | Yes (empty) | Dig on-site |
| Concrete blocks | 2.4 kg/L | No | Pre-position |
| Steel plates | 7.8 kg/L | Difficult | Pre-position |
| Personnel | ~80 kg/person | N/A | Always available |

### 5.2 Vehicle Cable Anchor (When Vehicle Nearby)

When a vehicle is within 50-100m, it can serve as the primary reaction anchor:

**Setup:**
```
[Vehicle] -------- 50-100m cable -------- [Push Frame] -------- [Probe]
   |                                            |
   Parked with                              Pulley/redirect
   brakes engaged                           at frame apex
```

**Cable/Winch Requirements:**

| Parameter | Specification |
|-----------|---------------|
| Cable type | Steel wire rope or synthetic (Dyneema) |
| Diameter | 8-12mm |
| Breaking strength | >30 kN (2x safety factor) |
| Length | 50-100m |
| Weight (100m Dyneema) | ~5 kg |
| Weight (100m steel) | ~30 kg |

**Vehicle Anchor Capacity:**

| Vehicle Type | Approximate Anchor Capacity |
|--------------|----------------------------|
| Passenger car (1,500 kg) | ~15 kN (on level ground) |
| SUV/Pickup (2,500 kg) | ~25 kN (on level ground) |
| Heavy truck (5,000+ kg) | ~50 kN (on level ground) |

**Notes:**
- Vehicle must be on firm ground
- Engage parking brake + wheel chocks
- Consider vehicle movement under load
- Works best with slight uphill position

### 5.3 Progressive Anchoring Strategy

Start with minimal anchoring, add more as needed based on measured resistance:

**Protocol:**
1. Install 2 small anchors (quick setup)
2. Begin push, monitor force
3. If force exceeds anchor capacity:
   - Stop push
   - Install additional anchors
   - Resume push
4. Extract probe
5. Move to next location

**Advantages:**
- Faster setup in easy soils
- Only uses full anchor capacity when needed
- Reduces overall cycle time

**Equipment for Progressive Anchoring:**

| Item | Purpose | Weight |
|------|---------|--------|
| 2x small anchors (PE26) | Initial setup | 2 kg |
| 2x large anchors (PE46) | Backup for dense soils | 4 kg |
| Force gauge on cylinder | Monitor push force | 1 kg |
| Impact wrench + batteries | Quick anchor installation | 5 kg |

### 5.4 Anchor Sharing Between Test Points

For grid surveys with closely spaced test points:

**Layout Strategy:**
```
Test point spacing: 5m grid

    A1 -------- A2 -------- A3
    |           |           |
    T1 -------- T2 -------- T3
    |           |           |
    A4 -------- A5 -------- A6

Where:
A = Anchor point
T = Test point

Anchors A2, A5 can serve both T1-T2 and T2-T3
```

**Benefits:**
- Reduces total anchor installations by 30-50%
- Frame moves between shared anchors
- Faster survey completion

---

## 6. Commercial Products Research

### 6.1 Portable Penetrometer Systems

#### PANDA Lightweight Dynamic Penetrometer

| Specification | Value |
|---------------|-------|
| Total weight | 18.5-20 kg |
| Depth capability | Up to 6m |
| Cone area | 4 cm² |
| Maximum resistance | 20 MPa |
| Data recording | Digital, GPS-equipped |
| Power | Manual (hammer driven) |
| Portability | Airline hold baggage compatible |

**Pricing:** Approximately $8,000-15,000 USD

Sources: [Sol Solution PANDA](https://www.sol-solution.com/en/products/equipment/panda/), [Insitutek PANDA](https://www.insitutek.com/products/panda-instrumented-dcp/)

**Note:** PANDA uses dynamic (impact) penetration, NOT static push. Not suitable for UXO sites.

#### Gouda Geo Handheld CPT Penetrometer

| Specification | Value |
|---------------|-------|
| Total weight | 11 kg |
| Dimensions | 59 × 19 × 28 cm |
| Depth capability | Up to 1m (body weight powered) |
| Reading range | Up to 10,000 kPa |
| Accuracy | ±8% of full scale |

**Limitations:** Only suitable for shallow depths with body-weight push.

Sources: [Gouda Geo Handheld Penetrometer](https://gouda-geo.com/product/handheld-cpt-penetrometer)

#### Geoprobe 420M Portable Drilling Rig

| Specification | Value |
|---------------|-------|
| Weight | 425 lbs (193 kg) |
| Width | 23" (58 cm) |
| Height (folded) | 62" (157 cm) |
| Stroke | 42" (107 cm) |
| Push force | 12,000 lb (53 kN) percussion |
| Power | Remote hydraulic or auxiliary |

**Notes:**
- Manually liftable mast (<450 lbs)
- Fits through standard doorways
- Requires external hydraulic power
- Percussion-based (not ideal for UXO)

**Pricing:** $25,000-40,000 USD (new)

Sources: [Geoprobe 420M](https://geoprobe.com/420m-direct-push-machine)

### 6.2 Ground Anchor Products

#### American Earth Anchors Penetrator Series

| Model | Length | Capacity | Approx. Price |
|-------|--------|----------|---------------|
| PE18-SQ | 18" | 2,500 lbs | $15-25 |
| PE26-SQ | 26" | 4,000 lbs | $25-40 |
| PE36-HEX | 36" | 6,000 lbs | $40-60 |
| PE46-HEX | 46" | 8,000 lbs | $60-80 |

Sources: [American Earth Anchors Shop](https://americanearthanchors.com/shop-by-product/penetrators/)

#### Helical Anchors (Various Manufacturers)

| Size | Shaft | Helix | Capacity (Class 6) | Approx. Price |
|------|-------|-------|-------------------|---------------|
| Small | 3/4" | 6" | 5,000 lbs | $30-50 |
| Medium | 1" | 8" | 9,000 lbs | $50-80 |
| Large | 1-1/4" | 10" | 10,000 lbs | $80-120 |

Sources: [Lifting.com Earth Anchors](https://lifting.com/earth-anchor-helix-034x6x66.html)

### 6.3 Installation Tools

#### Cordless Impact Wrenches

| Model | Torque | Weight | Price |
|-------|--------|--------|-------|
| Makita TW1000 | 738 ft-lbs | 15 lbs | $400-500 |
| Milwaukee M18 FUEL | 1,000 ft-lbs | 7 lbs | $350-450 |
| DeWalt 20V MAX | 700 ft-lbs | 6 lbs | $300-400 |

#### Anchor Installation Adapters

| Product | Compatibility | Price |
|---------|---------------|-------|
| Keyfit Speed Staker | 1/2" drill, various anchors | $30-50 |
| Vortex Installation Tool | Vortex spiral anchors | $25-40 |
| American Earth Anchors T-handle | AEA Penetrators | $40-60 |

Sources: [Amazon Speed Staker](https://www.amazon.com/Keyfit-Tools-Trampoline-Seconds-Functional/dp/B087SS5R4C), [American Earth Anchors Tools](https://americanearthanchors.com/shop-by-product/installation-accessories/)

### 6.4 Portable Hydraulic Equipment

#### Enerpac Lightweight Hand Pumps

| Model | Oil Capacity | Max Pressure | Weight | Price |
|-------|--------------|--------------|--------|-------|
| P142 | 20 in³ | 10,000 psi | ~5 kg | $400-600 |
| P392 | 55 in³ | 10,000 psi | ~7 kg | $500-800 |

**Features:**
- Glass-filled nylon reservoir
- Non-conductive fiberglass handle
- Two-speed operation (78% fewer strokes)
- Internal pressure relief valve

Sources: [Enerpac P392](https://www.enerpac.com/en-us/lightweight-hand-pumps/hand-pump-two-speed/P392), [Enerpac P142](https://www.enerpac.com/en-us/lightweight-hand-pumps/hand-pump-two-speed/P142)

#### Compact Hydraulic Cylinders

For 15 kN (3,300 lbs) force at 10,000 psi:

```
Required bore area = Force / Pressure
A = 15,000 N / 69 MPa = 217 mm² = 2.17 cm²
Bore diameter = √(4A/π) = 16.6 mm

Minimum bore: ~20mm for 15 kN at 10,000 psi
With 1.5m stroke: cylinder length ~1.7m retracted
```

| Cylinder Type | Bore | Stroke | Capacity | Weight | Price |
|---------------|------|--------|----------|--------|-------|
| Enerpac RC-series | 25mm | Various | 5 ton | 2-10 kg | $300-800 |
| Holmatro Single-acting | 25-40mm | Various | 5-15 ton | 3-15 kg | $500-1,500 |

Sources: [Enerpac Cylinder Sets](https://www.enerpac.com/en-us/cylinders-and-jacks/USCylinderandPumpSets), [Holmatro Cylinders](https://www.holmatro.com/en/industrial/lifting/hydraulic-cylinders-0)

### 6.5 Portable Geotechnical Equipment (General)

#### Hans Backpack Drill Rig

| Specification | Value |
|---------------|-------|
| Power options | Gasoline or electric |
| Weight | Backpack-portable |
| Applications | Core sampling, soil sampling |
| Depth | Variable (shallow) |

Sources: [Hans Backpack Drills](https://www.backpackdrills.com/)

#### Shaw Backpack Portable Core Drill (AMS)

| Specification | Value |
|---------------|-------|
| Core diameter | 41mm |
| Applications | Rock coring, geological exploration |
| Transport | Backpack frame |
| Locations | Remote, forests, mountains |

Sources: [AMS Shaw Backpack Drill](https://www.ams-samplers.com/shaw-portable-core-drill/)

---

## 7. Recommended System Configuration

### 7.1 Minimum Viable Portable System

**Target:** 15 kN reaction force, 3m depth, 2-person portable

| Component | Specification | Weight | Est. Cost |
|-----------|---------------|--------|-----------|
| A-frame (aluminum, collapsible) | 1.8m legs, 60° apex | 20 kg | $2,000-4,000 |
| Hydraulic cylinder | 25mm bore, 1.5m stroke | 12 kg | $500-800 |
| Hand pump (Enerpac P392) | 10,000 psi, 55 in³ | 7 kg | $500-700 |
| Helical anchors (4x) | 1" × 8" × 66" | 12 kg | $200-320 |
| Anchor installation tool | Impact wrench + adapter | 8 kg | $400-500 |
| Probe rods (3m) | 16mm diameter sections | 5 kg | $100-200 |
| Cables, straps, hardware | Anchor connections | 5 kg | $100-200 |
| Tool bag | Wrenches, spares | 3 kg | $50-100 |
| **TOTAL** | | **72 kg** | **$3,850-6,820** |

**Distribution for 2-person carry:**
- Person 1: Frame (20 kg) + pump (7 kg) + tools (3 kg) = 30 kg
- Person 2: Cylinder (12 kg) + anchors (12 kg) + rods (5 kg) + hardware (5 kg) + wrench (8 kg) = 42 kg

**Note:** Second person carries more but can use wheeled cart for portion.

### 7.2 Enhanced System with Power Pack

**Target:** 15 kN reaction force, faster operation, battery powered

| Component | Specification | Weight | Est. Cost |
|-----------|---------------|--------|-----------|
| Modular frame (aluminum) | Tripod or A-frame | 25 kg | $3,000-6,000 |
| Hydraulic cylinder | 30mm bore, 1.5m stroke | 15 kg | $600-1,000 |
| Battery hydraulic pump | 48V LiFePO4 | 18 kg | $1,500-2,500 |
| Helical anchors (4x) | 1-1/4" × 10" × 66" | 16 kg | $320-480 |
| Impact wrench + batteries | Milwaukee M18 | 8 kg | $400-500 |
| Control unit | Remote valve control | 3 kg | $300-500 |
| Probe rods (4m worth) | 16mm sections | 7 kg | $150-250 |
| Cables, hardware | Anchor system | 6 kg | $150-250 |
| Case/bags | Transport protection | 5 kg | $200-300 |
| **TOTAL** | | **103 kg** | **$6,620-11,780** |

**Distribution for 3-person carry:**
- Person 1: Frame (25 kg) + wrench (8 kg) = 33 kg
- Person 2: Cylinder (15 kg) + anchors (16 kg) = 31 kg
- Person 3: Pump (18 kg) + rods (7 kg) + control (3 kg) + hardware (6 kg) = 34 kg
- Cart: Cases (5 kg) for protected transport

### 7.3 Integration with HIRT System

The portable reaction force system must integrate with HIRT probe insertion:

| Integration Point | Requirement | Solution |
|-------------------|-------------|----------|
| Probe connection | 16mm rod to cylinder | Custom adapter sleeve |
| Insertion angle | Vertical ±2° | Spirit level on frame |
| Push rate control | 20mm/s standard | Flow control valve |
| Force monitoring | Real-time display | Pressure gauge → force calc |
| Depth measurement | ±10mm accuracy | Marked rod sections |
| Remote operation | 50m standoff | Tether control panel |
| Emergency stop | Instant release | Solenoid valve + dump |

---

## 8. Conclusions and Recommendations

### 8.1 Primary Findings

1. **Force requirements are manageable:** 15 kN covers most soils; 25 kN provides margin for dense sand.

2. **Helical screw anchors are the best solution:** Portable, reusable, adequate capacity, quick installation.

3. **Total system weight of 70-100 kg is achievable:** Distributable among 2-3 people for 1km carry.

4. **Commercial components are available:** No need for fully custom fabrication.

5. **UXO safety requires static push only:** Avoid percussion/impact methods.

### 8.2 Recommended Development Path

| Phase | Activity | Timeline | Budget |
|-------|----------|----------|--------|
| 1 | Procure anchor samples, test capacity | 1 month | $500 |
| 2 | Design frame (CAD), FEA analysis | 2 months | $5,000 |
| 3 | Fabricate prototype frame | 2 months | $8,000 |
| 4 | Integrate hydraulics, test system | 2 months | $5,000 |
| 5 | Field trials, iterate | 3 months | $5,000 |
| **Total** | | **10 months** | **$23,500** |

### 8.3 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Anchors fail in soft soil | Use more/larger anchors or add ballast |
| Frame too heavy | Use composites or accept wheeled transport |
| Dense sand exceeds capacity | Pre-pilot with smaller rod, progressive depth |
| Hydraulic failure | Carry manual backup pump |
| UXO encounter | Maintain 50m standoff, real-time force monitoring |

### 8.4 Next Steps

1. **Purchase test anchors:** 4x American Earth Anchors PE26 and PE46 for field testing
2. **Rent/borrow hand pump and cylinder:** Validate force requirements in target soils
3. **Design aluminum A-frame:** Optimize for minimum weight while meeting strength requirements
4. **Develop tether control system:** For UXO-safe remote operation
5. **Integrate with HIRT probe system:** Adapter design, depth marking, data logging

---

## References and Sources

### Soil Mechanics
- [Geoengineer.org - CPT Interpretation](https://www.geoengineer.org/education/site-characterization-in-situ-testing-general/cone-penetration-testing-cpt/cpt-interpretation-soil-parameters-e-vs-m)
- [Robertson CPT Guide (7th Edition)](https://www.cpt-robertson.com/PublicationsPDF/CPT-Guide-7th-Final-SMALL.pdf)
- [USGS Cone Penetration Testing](https://earthquake.usgs.gov/research/cpt/)
- [ScienceDirect - Shaft Friction](https://www.sciencedirect.com/topics/engineering/shaft-friction)

### Ground Anchors
- [American Earth Anchors](https://americanearthanchors.com/)
- [Helical Anchors Inc. Engineering Manual](https://helicalanchorsinc.com/wp-content/uploads/2020/05/HAI-Engineering-Manual-min.pdf)
- [Hubbell - Helical Pile Capacity Methods](https://blog.hubbell.com/en/chancefoundationsolutions/3-methods-to-determine-helical-pile-capacity)
- [Lifting.com Earth Anchors](https://lifting.com/earth-anchor-helix-034x6x66.html)
- [Deadman Off-Road Anchor Capacity](https://www.deadmanoffroad.com/pages/capacity)

### Load Carrying Standards
- [CNAS - The Soldier's Heavy Load](https://www.cnas.org/publications/reports/the-soldiers-heavy-load-1)
- [Backpacks Global - Military Backpack Weight](https://backpacks.global/how-heavy-are-military-backpacks/)
- [Rab Backpack Weight Calculator](https://rab.equipment/us/rab-lab/backpack-weight-calculator)
- [Outdoor Adventure Training - Pack Weight](https://outdooradventuretraining.com/2025/05/04/packs-pounds-percentages/)

### Portable Geotechnical Equipment
- [Sol Solution - PANDA Penetrometer](https://www.sol-solution.com/en/products/equipment/panda/)
- [Gouda Geo - Handheld Penetrometer](https://gouda-geo.com/product/handheld-cpt-penetrometer)
- [Geoprobe 420M](https://geoprobe.com/420m-direct-push-machine)
- [TMG SPT Tripod](https://tmgmfg.com/spt-tripod-soil-sampling-equiptment-spt-drop-hammer-aluminum-structure-spt-testing)
- [Hans Backpack Drills](https://www.backpackdrills.com/)

### Hydraulic Equipment
- [Enerpac Lightweight Hand Pumps](https://www.enerpac.com/en-us/manual-pumps/USPumpsManualLightweight)
- [Holmatro Industrial Cylinders](https://www.holmatro.com/en/industrial/lifting/hydraulic-cylinders-0)

### Transport Equipment
- [Rhino Cart All-Terrain](https://rhinocart.com/)
- [Little Beaver Tripod System](https://littlebeaverstore.com/store/little-beaver-tripod-auger-lifting-frame-for-depths-6-to-30-tripod.html)

### UXO Safety
- [AGS UXO Safety Guidance](https://www.ags.org.uk/item/safety-guidance-unexploded-ordnance/)
- [Wikipedia - Unexploded Ordnance](https://en.wikipedia.org/wiki/Unexploded_ordnance)

---

*Document prepared for HIRT Development Team - January 2026*
