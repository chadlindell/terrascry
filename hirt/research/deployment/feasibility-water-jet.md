# Water Jet Drilling / Jetting Feasibility Assessment for HIRT Probe Insertion

**Document Purpose:** Detailed feasibility assessment of water jet drilling/jetting systems for creating 12-16mm diameter holes to 2-3m depth for HIRT probe insertion at potential UXO (unexploded ordnance) sites.

**Key Requirements:**
- Hole diameter: 12-16mm
- Depth: 2-3 meters
- Holes per survey: 20-50 per grid
- Site context: WWII bomb craters, potential UXO presence
- Safety: NO percussion, NO vibration (UXO-safe)

**Assessment Date:** 2026-01-19

---

## 1. Technical Feasibility

### 1.1 Can Water Jetting Achieve 12-16mm Diameter Holes at 2-3m Depth?

**Answer: YES, with appropriate equipment and technique.**

Water jetting is well-established for creating small-diameter holes in unconsolidated soils. The FAO and PRACTICA Foundation document techniques for manual well drilling using jetting that achieve depths of 25-50 meters in appropriate conditions.

**Key evidence:**
- Self-jetting well points routinely achieve 20+ feet (6m+) in sandy soils
- DIY practitioners report "5 minutes to jet a well 20 feet" in favorable conditions
- Drill pipe typically 50mm diameter, creating 100-150mm holes; scaling down for 12-16mm is feasible
- Holland practitioners report jetting "up to 10 meters with little pressure" in sandy soil

**Limitations:**
- Gravel layers cause refusal (water cannot displace heavy stones)
- Dry, compacted clay significantly slows progress
- Very fine clay can clog jetting nozzles

### 1.2 Pressure Requirements

**Soil-Specific Pressure Requirements:**

| Soil Type | Minimum Pressure | Recommended Pressure | Notes |
|-----------|-----------------|---------------------|-------|
| Sand | 40 psi (2.8 bar) | 50-60 psi (3.5-4 bar) | Easiest; high water volume, low pressure |
| Silt | 50-70 psi (3.5-5 bar) | 70-100 psi (5-7 bar) | Good jetting performance |
| Clay | 100-150 psi (7-10 bar) | 150-200 psi (10-14 bar) | Requires small cutting stream at high pressure |
| Gravel | 100-150 psi (7-10 bar) | Not recommended | Water cannot lift heavy stones |
| Mixed soils | 70-100 psi (5-7 bar) | 100-150 psi (7-10 bar) | Variable performance |

**What's Sufficient vs. Overkill:**

- **Sufficient:** 5-8 bar (72-116 psi) - This covers most unconsolidated soils encountered at WWII sites
- **Optimal:** 6-10 bar (87-145 psi) - Provides margin for clay layers
- **Overkill:** >15 bar (>217 psi) - Industrial jet grouting pressures (60 MPa / 600 bar) are completely unnecessary for soil displacement

**Key Insight:** The Honda WH20 high-pressure pump at 5 bar is sufficient for most HIRT applications. Higher pressures may be needed only for stiff clay.

### 1.3 Nozzle Design for Small Diameter Holes

For 12-16mm holes, nozzle design is critical:

**Recommended Nozzle Specifications:**
- **Orifice diameter:** 2-4mm (creates hole ~4-5x orifice diameter)
- **Nozzle geometry:** Conical with 13-15 degree converging angle
- **Throat length:** 2-4x the nozzle diameter (optimal flow characteristics)
- **Configuration:** Single forward-facing orifice for drilling; optional 30-degree angled secondary orifice for wall cutting

**Design Principles:**
- Smaller orifice = higher velocity jet = more cutting power per unit water
- Flow rate varies with square root of pressure difference
- At 5 bar with 3mm orifice: approximately 10-15 L/min flow rate

**Commercial Options:**
- Pressure washer jet nozzles ($2-10 at hardware stores)
- Custom machined brass nozzles
- Self-jetting wellpoint tips (scaled down)

### 1.4 Soil Type Performance Matrix

| Soil Type | Penetration Rate | Water Consumption | Hole Quality | Overall Suitability |
|-----------|-----------------|-------------------|--------------|---------------------|
| **Sand** | EXCELLENT (0.5-2 m/min) | HIGH (30-50 L/m) | GOOD (stable below water table) | EXCELLENT |
| **Silty sand** | GOOD (0.3-1 m/min) | MEDIUM (20-40 L/m) | GOOD | EXCELLENT |
| **Silt** | GOOD (0.2-0.5 m/min) | MEDIUM (20-30 L/m) | MODERATE | GOOD |
| **Soft clay** | MODERATE (0.1-0.3 m/min) | LOW (10-20 L/m) | VARIABLE | GOOD |
| **Stiff clay** | SLOW (0.05-0.1 m/min) | LOW (10-20 L/m) | POOR (may collapse) | MODERATE |
| **Gravel** | POOR (may refuse) | HIGH | N/A | NOT RECOMMENDED |
| **Mixed fill (bomb crater)** | VARIABLE | VARIABLE | VARIABLE | MODERATE-GOOD |

**WWII Bomb Crater Context:**
Bomb crater fill is typically disturbed, loosened soil mixed with debris. This is generally FAVORABLE for water jetting because:
- Fill is not compacted to original density
- Mixed soil types don't form continuous clay layers
- Water infiltration history may have pre-softened material

### 1.5 Hole Stability After Jetting

**Critical Issue:** Water-jetted holes in unconsolidated soil WILL collapse without support.

**Stabilization Options:**

1. **Keep Hole Flooded**
   - Maintain water column during entire operation
   - Hydrostatic pressure supports walls
   - Simple but requires continuous water supply

2. **Bentonite Drilling Mud**
   - Mix bentonite powder with water (gel-like consistency)
   - Forms filter cake on hole walls
   - Provides stability for hours to days
   - Mixing ratio: ~30-50g bentonite per liter water
   - Cost: ~$20-30 per 20kg bag (enough for 100+ holes)

3. **Immediate Probe Insertion**
   - Insert probe immediately after jetting
   - Probe itself supports the hole
   - Requires rapid transition from jetting to insertion

4. **Temporary Casing**
   - PVC or metal tube inserted during jetting
   - Withdrawn as probe is inserted
   - Adds complexity but ensures hole integrity

**Recommendation for HIRT:**
Use bentonite slurry combined with immediate probe insertion. This provides:
- Hole stability during transition
- Lubrication for probe insertion
- Low cost and simple implementation

---

## 2. Equipment Options

### 2.1 Portable Water Jet Systems

**Option A: Modified Pressure Washer System**

| Component | Specification | Cost |
|-----------|--------------|------|
| Pressure washer | 2000-3500 PSI, 2-4 GPM | $200-500 |
| Jetting lance | 3m steel or PVC tube | $20-50 |
| Nozzle | Custom 2-4mm orifice | $10-30 |
| Hose | High-pressure 10-15m | $50-100 |
| **Total** | | **$280-680** |

**Advantages:**
- Widely available, familiar technology
- Sufficient pressure for all soil types
- Electric or gas powered options

**Disadvantages:**
- Overkill pressure for most soils (wastes energy)
- Lower flow rate than dedicated jetting pumps
- May need pressure reduction for sandy soils

**Option B: Honda High-Pressure Water Pump (WH20)**

| Specification | Value |
|--------------|-------|
| Pressure | 5 bar (72 psi) |
| Flow rate | 500 L/min max |
| Engine | Honda GX160 (163cc) |
| Weight | 27 kg |
| Fuel consumption | ~1.2 L/hour |
| Run time | 2.5 hours per tank |
| Price | $800-1,200 |

**Advantages:**
- Purpose-built for water transfer/jetting
- Optimal pressure range for soil jetting
- High flow rate for rapid hole advancement
- Robust, field-serviceable design

**Disadvantages:**
- Higher initial cost
- Heavier than pressure washer

**Option C: Centrifugal Trash Pump**

| Component | Specification | Cost |
|-----------|--------------|------|
| 2" trash pump | 150-200 GPM, 2-3 bar | $300-600 |
| Jetting pipe | 2" galvanized with sharpened tip | $30-50 |
| Reducer nozzle | 2" to 10mm | $20-40 |
| **Total** | | **$350-690** |

**Advantages:**
- Very high flow rate
- Can handle dirty water (recirculation possible)
- Lower pressure = less energy waste in sand

**Disadvantages:**
- May lack pressure for clay layers
- Larger, heavier equipment

### 2.2 Self-Jetting Well Point Systems

Self-jetting well points are specifically designed for soil penetration by water jetting.

**Characteristics:**
- Built-in check valve prevents backflow
- Internal thread for pipe connection
- Nozzle openings at tip for forward jetting
- Typical sizes: 32mm (1.25"), 50mm (2") diameter

**Adaptation for HIRT (12-16mm):**
Scaling down standard well point design:
- Fabricate from 16mm OD stainless steel tube
- Machine small orifices (2-3mm) at tip
- Internal water passage connects to supply hose

**FAO/PRACTICA Documentation:**
The FAO and PRACTICA Foundation provide detailed manuals for low-cost well drilling in developing countries:
- Depths to 25-50m achievable
- Pump specifications: 20 L/s at 6-8 bar
- 63mm fire hose delivery typical
- Drilling fluid additives for depths >25 feet

### 2.3 Pressure Washer-Based Approaches

**DIY Community Experience:**

Numerous documented cases of successful soil boring using pressure washers:

- Boring under sidewalks: "20+ feet in soft backfill" with standard equipment
- Ground rod installation: "8-foot rod in 5 minutes with no sledgehammer"
- Post hole creation: Successful in sandy/loamy soils, challenging in clay

**Pressure Washer Specifications for Soil Jetting:**

| Class | Pressure | Flow | Best For |
|-------|----------|------|----------|
| Light duty | 1300-1800 PSI | 1.5 GPM | Sand only |
| Medium duty | 2000-2800 PSI | 2-2.5 GPM | Sand, silt |
| Heavy duty | 2800-3500 PSI | 2.5-4 GPM | Sand, silt, soft clay |
| Commercial | 3500-4000 PSI | 4+ GPM | All soils except rock |

**Key Insight:** Medium duty pressure washer (2000+ PSI) with turbo nozzle is adequate for most HIRT applications.

### 2.4 Hand-Held Jetting Lances

**Design Concept:**
- 3m steel or aluminum tube (16mm OD)
- High-pressure hose connection at top
- Nozzle tip at bottom
- Hand grips for operator control
- Up-and-down motion assists penetration

**Construction:**
1. Stainless steel tube: 16mm OD, 12mm ID, 3.5m length
2. Welded/threaded nozzle tip with 2-3mm orifice
3. NPT fitting at top for hose connection
4. Rubber hand grips at comfortable intervals

**Operation:**
- Apply water pressure while pushing down
- Use up-and-down "pumping" motion
- Water jet softens soil ahead of lance
- Slurry rises around lance to surface
- Advance rate: 10-30 cm per stroke in sand

### 2.5 Combination Jetting + Static Push Systems

**Hybrid Approach (RECOMMENDED FOR HIRT):**

This combines water jetting to create a pilot hole with static push for final probe insertion.

**Phase 1: Water Jetting**
- Create 20mm pilot hole to full depth
- Fill with bentonite slurry for stability
- Time: 5-15 minutes per hole

**Phase 2: Static Push**
- Insert 12-16mm probe into pre-formed hole
- Minimal force required (slurry-lubricated)
- Probe displaces slurry, not soil
- Time: 1-2 minutes per hole

**Advantages:**
- Best of both methods
- Jetting handles soil variability
- Push ensures precise probe placement
- Eliminates hole collapse issues

---

## 3. Water Requirements

### 3.1 Water Volume Per Hole

**Theoretical Calculation:**

For a 16mm diameter hole, 3m deep:
- Hole volume: π × (0.008m)^2 × 3m = 0.0006 m^3 = 0.6 liters

However, actual water consumption is 100-500x the hole volume because:
- Water carries cuttings to surface
- Water dissipates into surrounding soil
- Continuous flow required during operation

**Empirical Water Consumption Estimates:**

| Soil Type | Water per Meter Depth | Total for 3m Hole |
|-----------|----------------------|-------------------|
| Sand | 30-50 L/m | 90-150 L |
| Silt | 20-40 L/m | 60-120 L |
| Clay | 10-30 L/m | 30-90 L |
| Mixed | 20-40 L/m | 60-120 L |

**Conservative Estimate:** 100 liters per hole average

**With Recirculation:** 30-50 liters per hole (67-50% reduction)

### 3.2 Total Water Needed for Survey Grid

| Grid Size | Without Recirculation | With Recirculation |
|-----------|----------------------|-------------------|
| 20 holes | 2,000 L (2 m^3) | 600-1,000 L |
| 35 holes | 3,500 L (3.5 m^3) | 1,050-1,750 L |
| 50 holes | 5,000 L (5 m^3) | 1,500-2,500 L |

**Practical Implications:**
- 20-hole survey: Standard pickup truck water tank sufficient
- 50-hole survey: May require water resupply or large tank

### 3.3 Water Source Options in Field

**Option 1: Transported Water**
- IBC tank (1,000 L) on trailer: $100-200 for used tank
- Pickup-mounted tank (200-500 L): $150-400
- Multiple 200L drums: $20-40 each

**Option 2: On-Site Water Sources**
- Stream, pond, or lake: Free, requires pump and filtration
- Municipal water supply: May be available near sites
- Tanker delivery: $50-200 per 1,000-5,000 L load

**Option 3: Borehole/Well on Site**
- Jet a preliminary water supply well
- Self-sustaining once established
- Ideal for extended surveys

**Recommendation:** Start with 1,000L IBC tank, implement recirculation to extend capacity.

### 3.4 Recirculation Possibilities

**Simple Settling System:**
1. Jet slurry flows into settling pit/container
2. Heavy particles settle (sand, silt)
3. Clear water drawn from top
4. Reused for subsequent jetting

**Components:**
- Settling tank (200-500 L container)
- Baffle plates to slow flow
- Intake screen/filter
- Return pump (can be main jetting pump)

**Efficiency:**
- 50-70% water recovery typical
- Reduces water consumption by 2-3x
- Minimal additional equipment cost

**Advanced Option: Cyclone Separator**
- Removes fines more efficiently
- Higher equipment cost ($500-1,500)
- Better for clay soils with fine particles

### 3.5 Dealing with Slurry/Spoils

**Challenge:** Water jetting produces significant slurry (water + soil mixture) at surface.

**Management Options:**

1. **Natural Infiltration**
   - Allow slurry to infiltrate adjacent soil
   - Works well in sandy/permeable soils
   - Minimal effort required

2. **Settling and Disposal**
   - Collect in containers
   - Allow solids to settle
   - Decant water for reuse
   - Dispose of solids appropriately

3. **Vacuum Extraction**
   - Shop vacuum or small hydrovac
   - Removes slurry continuously
   - Cleaner operation
   - Added equipment complexity

4. **Containment and Drying**
   - Create lined pit near work area
   - Allow water to evaporate/infiltrate
   - Remove dried solids after survey

**Recommendation for HIRT:**
Combine settling/recirculation with natural infiltration. Create a simple settling pit, recover clean water, let remaining slurry infiltrate or dry.

---

## 4. Remote Operation Potential

### 4.1 Can Jetting Be Fully Remote/Robotic?

**Answer: YES, jetting is highly suitable for remote operation.**

**Favorable Characteristics:**
- No mechanical feedback required (unlike drilling)
- Simple linear motion (push down, pull up)
- Progress monitored by depth marker
- No rotation or complex tool changes
- Failure mode is simple (stop flow, retract)

**Existing Precedents:**
- USA DeBusk HX+ automated hydro excavation system
- Robotic sewer jetting crawlers
- Industrial robotic waterjet cutting systems
- NASA autonomous drilling prototypes

### 4.2 Automated Lance Positioning

**System Concept:**

| Component | Function | Complexity |
|-----------|----------|------------|
| XY Gantry | Position lance over hole location | MEDIUM |
| Z-axis (vertical) | Lower/raise lance | LOW |
| Lance holder | Secure lance during operation | LOW |
| Limit switches | Detect bottom, surface positions | LOW |
| Load cell | Monitor insertion resistance | LOW |

**Control System:**
- Pre-programmed hole locations (GPS or local coordinates)
- Automated sequence: position, lower, jet, retract, move to next
- Simple PLC or Arduino/Raspberry Pi control
- Manual override for anomalies

**Estimated Development:**
- Basic automated positioner: $2,000-5,000
- Full robotic system with navigation: $10,000-30,000

### 4.3 Safety Zone Requirements

**UXO Safety Considerations:**

Water jetting has been identified as UXO-SAFE because:
- No percussion (no impact forces)
- No vibration transmission through soil
- Water pressure dissipates rapidly in soil matrix
- Hydraulic erosion, not mechanical cutting

**Recommended Safety Zones:**

| Risk Level | Minimum Distance | Notes |
|------------|-----------------|-------|
| Active jetting | 5m | Operator protection from spray |
| UXO precaution | 20m | Conservative standoff during operation |
| Emergency | 100m | UXO detonation exclusion (standard) |

**Remote Operation Benefit:**
With automated system, operator can be 20-50m away during jetting, behind suitable protection.

### 4.4 Monitoring Hole Progress

**Depth Monitoring Options:**

1. **Mechanical Depth Marker**
   - Marks on lance tube (every 25cm)
   - Visual observation
   - Simple, reliable

2. **Encoder/Potentiometer**
   - Measures lance travel
   - Real-time depth display
   - Required for automation

3. **Pressure Monitoring**
   - Pressure drop indicates breakthrough to softer layer
   - Pressure rise may indicate obstruction
   - Useful diagnostic information

4. **Flow Monitoring**
   - Sudden flow increase may indicate void or fracture
   - Consistent flow indicates normal operation

**Recommended Monitoring Suite:**
- Depth encoder (primary)
- Pressure gauge (diagnostic)
- Visual observation camera (remote operation)

---

## 5. Cost Analysis

### 5.1 Equipment Costs

**Option A: Basic Manual System**

| Item | Cost |
|------|------|
| Medium-duty pressure washer (2500 PSI) | $350 |
| Custom jetting lance (3m) | $50 |
| High-pressure hose (15m) | $80 |
| Nozzles and fittings | $30 |
| **Total** | **$510** |

**Option B: Dedicated Jetting Pump System**

| Item | Cost |
|------|------|
| Honda WH20 high-pressure pump | $1,000 |
| Jetting lance (3m stainless) | $100 |
| Fire hose (25m) | $150 |
| Settling tank (200L) | $50 |
| Fittings, nozzles, accessories | $100 |
| **Total** | **$1,400** |

**Option C: Semi-Automated System**

| Item | Cost |
|------|------|
| Honda WH20 pump | $1,000 |
| Jetting lance assembly | $150 |
| Simple XYZ positioner frame | $1,500 |
| Control electronics | $300 |
| Sensors and feedback | $200 |
| Water tank (1000L IBC) | $150 |
| Miscellaneous | $200 |
| **Total** | **$3,500** |

### 5.2 Operational Costs

**Per-Hole Operational Costs:**

| Item | Cost per Hole |
|------|--------------|
| Water (100L at $0.01/L) | $1.00 |
| Fuel (0.5L at $2/L) | $1.00 |
| Bentonite (0.5kg at $1.50/kg) | $0.75 |
| Nozzle wear allowance | $0.25 |
| **Total per hole** | **$3.00** |

**Per-Survey Operational Costs:**

| Survey Size | Direct Costs | Labor (8hr at $30/hr) | Total |
|-------------|-------------|----------------------|-------|
| 20 holes | $60 | $240 | $300 |
| 35 holes | $105 | $420 | $525 |
| 50 holes | $150 | $600 | $750 |

### 5.3 Comparison to Hydraulic Push

**Cost Comparison:**

| Method | Equipment Cost | Per-Hole Cost | 20-Hole Survey Total |
|--------|---------------|---------------|---------------------|
| Manual jetting | $500-700 | $3 + labor | $300-400 |
| Dedicated jetting | $1,400 | $3 + labor | $350-450 |
| Semi-auto jetting | $3,500 | $3 + reduced labor | $250-350 |
| CPT rig rental | $2,000/day | Included | $4,000-6,000 |
| Mini-CPT purchase | $30,000-80,000 | $5 + labor | $400-500 |
| Geoprobe contract | N/A | $150-300/hole | $3,000-6,000 |

**Key Insight:** Water jetting offers 10-20x cost savings compared to professional CPT services, with comparable UXO safety.

---

## 6. Practical Deployment

### 6.1 Setup Complexity

**Basic Manual System Setup:**

| Step | Time |
|------|------|
| Unload equipment | 10 min |
| Connect hoses and fittings | 10 min |
| Fill water tank | 20 min |
| Test system | 5 min |
| **Total setup** | **45 min** |

**Semi-Automated System Setup:**

| Step | Time |
|------|------|
| Unload and position equipment | 20 min |
| Assemble positioner frame | 30 min |
| Connect hydraulics/electronics | 20 min |
| Fill water tank | 20 min |
| System test and calibration | 20 min |
| **Total setup** | **110 min** |

### 6.2 Crew Requirements

| System Type | Minimum Crew | Optimal Crew |
|-------------|--------------|--------------|
| Manual jetting | 1 (2 recommended) | 2-3 |
| Semi-automated | 1 | 1-2 |
| Full automation | 0 (monitoring only) | 1 |

**Roles:**
- Operator: Controls jetting, monitors progress
- Assistant: Manages water supply, handles probes
- Surveyor (optional): Records locations, manages data

### 6.3 Power/Water Logistics

**Power Requirements:**

| Equipment | Power Source | Consumption |
|-----------|-------------|-------------|
| Gas pressure washer | Gasoline | 1-2 L/hour |
| Honda WH20 pump | Gasoline | 1.2 L/hour |
| Electric pressure washer | 240V/15A | 1.5-3 kW |
| Control electronics | 12V/240V | 50-100W |

**Logistics Planning:**
- 8-hour day with gas pump: 10L fuel
- Water for 20 holes: 2,000L (without recirculation), 700L (with)
- Bentonite: 10kg

**Transport Requirements:**
- Minimum: Pickup truck with 500L tank
- Recommended: Trailer with 1,000L tank + equipment

### 6.4 Speed (Holes Per Hour)

**Production Rate Estimates:**

| Condition | Time per Hole | Holes per Hour | Holes per 8hr Day |
|-----------|--------------|----------------|-------------------|
| Sand, good conditions | 5-10 min | 6-12 | 40-80 |
| Silt, typical | 10-15 min | 4-6 | 25-40 |
| Clay, challenging | 20-40 min | 1.5-3 | 10-20 |
| Mixed, WWII crater fill | 10-20 min | 3-6 | 20-40 |

**Realistic Production for HIRT Survey:**
- 20-hole grid: 4-8 hours (1 day)
- 35-hole grid: 6-12 hours (1-1.5 days)
- 50-hole grid: 10-18 hours (1.5-2 days)

**Bottlenecks:**
1. Probe insertion after jetting (can parallelize)
2. Water resupply (mitigated by recirculation)
3. Clay layers (unpredictable delays)

---

## 7. Purpose-Built HIRT Jetting System Concept

### 7.1 Design Concept: HIRT Water Jet Probe Insertion System

**Integrated System Components:**

```
                    WATER SUPPLY
                         |
                   [IBC Tank 1000L]
                         |
                    [Pump Unit]
                     Honda WH20
                         |
              [Pressure Regulator]
                  0-8 bar adjustable
                         |
                  [Fire Hose 25m]
                         |
                   [Jetting Lance]
                 3.5m SS tube, 16mm OD
                         |
                  [Jetting Nozzle]
                   3mm orifice
                         |
                     [GROUND]
                         |
              [Settling Pit/Tank]
                         |
                [Return Pump] -----> [Filter] --> Back to tank
```

### 7.2 Integrated Jetting Lance + Probe Insertion

**Dual-Mode Lance Design:**

The lance serves two functions:
1. **Jetting Mode:** Water flows through, creates pilot hole
2. **Guide Mode:** Lance remains in hole, guides probe insertion

**Operation Sequence:**
1. Position lance at hole location
2. Start water flow, push lance down to target depth
3. Stop water flow, lance remains in place
4. Slide HIRT probe down inside the lance (lance acts as guide)
5. Withdraw lance, probe remains in place
6. Move to next hole location

**Design Specifications:**

| Component | Specification |
|-----------|--------------|
| Lance tube OD | 20mm (to accommodate 16mm probe) |
| Lance tube ID | 17mm |
| Wall thickness | 1.5mm |
| Material | 316 stainless steel |
| Length | 3.5m (three 1.2m sections + 0.9m tip) |
| Connections | Quick-lock threaded joints |
| Tip | Hardened steel, 3mm forward jet orifice |
| Weight | ~4 kg complete |

### 7.3 Portable Pump Specifications

**Recommended Pump: Honda WH20 or Equivalent**

| Specification | Value | Rationale |
|--------------|-------|-----------|
| Type | Centrifugal, self-priming | Handles dirty water |
| Pressure | 5 bar (72 psi) | Optimal for soil jetting |
| Flow rate | 500 L/min max | Rapid hole advancement |
| Engine | Honda GX160 (163cc) | Reliable, serviceable |
| Suction lift | 8m | Flexibility in setup |
| Weight | 27 kg | Man-portable |
| Dimensions | 520 x 400 x 450mm | Compact |
| Fuel tank | 3.1L | 2.5 hours runtime |

**Alternative: Two Pumps in Series**
For clay-heavy sites requiring higher pressure:
- Two centrifugal pumps in series = doubled pressure
- Achieves 8-10 bar with standard pumps
- More complexity but adaptable to conditions

### 7.4 Water Tank Sizing

**Calculation Basis:**
- 20 holes minimum per survey
- 100L average water consumption per hole (conservative)
- 50% recirculation efficiency
- 1 day operation target

**Minimum Tank Size:**
- Without recirculation: 2,000L
- With recirculation: 1,000L

**Recommended Configuration:**
- Primary tank: 1,000L IBC (standard, inexpensive, transportable)
- Settling/recovery tank: 200L drum
- Reserve: 200L drum for emergency

**Transport Options:**
1. **Trailer-mounted:** IBC on single-axle trailer ($500-1,000)
2. **Pickup-mounted:** 500L tank in truck bed ($200-400)
3. **Ground-based:** IBC on pallet, forklift positioning ($100)

---

## 8. Recommendations

### 8.1 Best Jetting Approach for HIRT

**Primary Recommendation: Hybrid Jetting + Static Push with Bentonite**

**Rationale:**
- UXO-safe (zero percussion, zero vibration)
- Works in variable soil conditions (bomb crater fill)
- Cost-effective ($1,500-2,000 complete system)
- Field-serviceable with common components
- Achievable production rate (20-40 holes/day)
- Scalable from manual to semi-automated

**System Configuration:**
1. Honda WH20 or equivalent high-pressure pump
2. Custom 20mm OD jetting lance (3.5m, sectional)
3. 1,000L water supply with settling/recirculation
4. Bentonite slurry for hole stabilization
5. Static push mechanism for probe insertion

### 8.2 Equipment Specifications Summary

**Core Equipment List:**

| Item | Specification | Est. Cost |
|------|--------------|-----------|
| High-pressure pump | Honda WH20 or equivalent, 5 bar, 500 L/min | $1,000 |
| Jetting lance | 20mm OD SS tube, 3.5m, sectional | $150 |
| Fire hose | 25m, 63mm, with fittings | $150 |
| Water tank | 1,000L IBC | $150 |
| Settling tank | 200L drum with baffles | $50 |
| Bentonite | 20kg supply | $30 |
| Fittings/accessories | Nozzles, adapters, clamps | $100 |
| Pressure gauge | 0-10 bar | $30 |
| Safety equipment | Goggles, gloves, rain gear | $50 |
| **Total Core System** | | **$1,710** |

**Optional Enhancements:**

| Item | Specification | Est. Cost |
|------|--------------|-----------|
| Pressure regulator | 0-8 bar adjustable | $80 |
| Return pump | Small submersible | $100 |
| Extra lance sections | For deeper holes | $50/section |
| Transport trailer | Single-axle with tank mount | $500-1,000 |
| Semi-auto positioner | Manual XY, powered Z | $1,500-2,500 |
| **Enhanced System** | | **$3,500-5,000** |

### 8.3 Development Path

**Phase 1: Manual Prototype (2-4 weeks)**
- Build basic jetting lance from off-the-shelf components
- Test with standard pressure washer
- Validate hole creation in representative soils
- Measure water consumption, penetration rates
- Cost: $500-700

**Phase 2: Optimized Manual System (4-8 weeks)**
- Procure dedicated pump (Honda WH20)
- Design custom lance with integrated nozzle
- Implement bentonite slurry system
- Build settling/recirculation system
- Field test at representative site
- Document procedures, refine technique
- Cost: $1,500-2,000 total

**Phase 3: Semi-Automated System (2-4 months)**
- Design simple XYZ positioning frame
- Integrate depth encoder and pressure monitoring
- Develop control system (Arduino/Raspberry Pi)
- Test automated sequence
- Evaluate production rates and reliability
- Cost: $3,500-5,000 total

**Phase 4: Integration with HIRT (ongoing)**
- Develop probe insertion sequence after jetting
- Optimize lance design for dual-mode operation
- Integrate with survey positioning system
- Create field operation procedures
- Train operators

### 8.4 Risk Assessment and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Gravel refusal | MEDIUM | HIGH | Pre-survey auger probe, alternative hole locations |
| Clay clogging | MEDIUM | MEDIUM | Bentonite additives, pressure increase |
| Hole collapse | HIGH | MEDIUM | Immediate bentonite, rapid probe insertion |
| Water supply exhaustion | LOW | MEDIUM | Recirculation, reserve supply |
| Equipment failure | LOW | MEDIUM | Spare parts, simple design, field repair |
| UXO encounter | LOW | CRITICAL | Remote operation, magnetometer pre-scan, abort protocol |

### 8.5 Comparison with Alternatives

| Method | UXO Safety | Cost | Speed | Soil Versatility | Remote Potential |
|--------|-----------|------|-------|-----------------|------------------|
| **Water Jetting** | EXCELLENT | LOW | GOOD | GOOD | HIGH |
| Hydraulic Push | EXCELLENT | MEDIUM | GOOD | MODERATE | HIGH |
| Hand Auger | EXCELLENT | VERY LOW | SLOW | MODERATE | LOW |
| Powered Auger | MODERATE | LOW | FAST | GOOD | MEDIUM |
| CPT Rig | EXCELLENT | HIGH | FAST | GOOD | MEDIUM |
| Sonic Drill | POOR | HIGH | VERY FAST | EXCELLENT | MEDIUM |

**Conclusion:**
Water jetting represents an optimal balance of UXO safety, cost-effectiveness, and operational flexibility for HIRT probe insertion. The technology is well-proven for soil penetration, readily adaptable to the 12-16mm hole diameter requirement, and inherently compatible with remote/automated operation.

---

## 9. Sources and References

### Primary Technical Sources

- [FAO - Small Diameter Wells](https://www.fao.org/4/x5567e/x5567e05.htm)
- [PRACTICA Foundation - Jetting Manual Drilling](https://www.practica.org/wp-content/uploads/Manual-jetting.pdf)
- [US Army FM 5-484 - Alternative Well Construction](https://www.globalsecurity.org/military/library/policy/army/fm/5-484/Ch9.htm)
- [ScienceDirect - Jet Drilling Overview](https://www.sciencedirect.com/topics/engineering/jet-drilling)
- [TRUVAC - Ultimate Guide to Hydro Excavation](https://www.truvac.com/news-press/what-is-hydro-excavation)

### Equipment Sources

- [Honda Power Equipment - WH20 Pump](https://powerequipment.honda.com/pumps/models/wh20)
- [NLB Corporation - Water Jetting Equipment](https://www.nlbcorp.com/products/pumps-units/)
- [NorthStar - High-Pressure Water Pump](https://www.amazon.com/NorthStar-High-Pressure-Water-Pump-Engine/dp/B00381T8J2)

### DIY and Practical Experience

- [eHam - Hydro-Ground Rod Installation](https://www.eham.net/article/23198)
- [Mike Holt Forum - Boring Under Sidewalk](https://forums.mikeholt.com/threads/boring-under-6-sidewalk-with-pressure-washer.111116/)
- [TractorByNet - Well Point Jetting](https://www.tractorbynet.com/forums/threads/driving-a-well-point.172792/page-3)
- [DIY Solar Forum - Post Holes with Pressure Washer](https://diysolarforum.com/threads/making-post-holes-with-a-pressure-washer-vs-auger-vs.103726/)

### Drilling Fluid and Stability

- [Drill Your Own Well - Bentonite Drilling](https://drillyourownwell.com/drilling-deeper-with-bentonite/)
- [Trenchlesspedia - Bentonite Drilling Fluid](https://trenchlesspedia.com/bentonite-and-the-use-of-drilling-fluid-in-trenchless-projects/2/3607)
- [Lone Star Drills - Prevent Borehole Collapse](https://www.lonestardrills.com/prevent-borehole-collapse/)

### Nozzle Design

- [TechniWaterjet - Waterjet Nozzle](https://www.techniwaterjet.com/waterjet-nozzle/)
- [LORRIC - Nozzle Pressure and Orifice](https://www.lorric.com/en/Articles/nozzle/all/nozzle-orifice-size)
- [ResearchGate - Effect of Nozzle Geometry on Small Water Jets](https://www.researchgate.net/publication/48352583_The_effect_of_nozzle_geometry_on_the_flow_characteristics_of_small_water_jets)

### Robotic/Remote Systems

- [USA DeBusk - HX+ Automated Hydro Excavation](https://usadebusk.com/service/hx/)
- [Sewer Robotics - Water Jet Cutting](https://www.sewerrobotics.com/water-jet-cutting/)
- [NLB Corporation - Automated Water Jet Systems](https://www.nlbcorp.com/products/accessories/automated-systems/)

### Cost and Comparison

- [CLU-IN - Direct Push Platforms](https://clu-in.org/characterization/technologies/dpp.cfm)
- [Geoprobe - Direct Push Technology](https://geoprobe.com/direct-image/cpt-cone-penetration-testing)
- [Wellpoint Installation Techniques](https://ebrary.net/199729/engineering/wellpoint_installation_techniques)

---

*Document compiled: 2026-01-19*
*For HIRT Geophysical Survey System development*
*Part of borehole creation methods research series*
