# HIRT Probe Insertion Research: Borehole Stability and Sensor-Soil Contact

**Document Type:** Technical Research Report
**Application:** HIRT Geophysical Survey System - Probe Deployment
**Date:** 2026-01-19
**Status:** Complete

---

## Executive Summary

This document addresses a critical challenge in deploying the HIRT instrumented probe: **how to successfully insert a 16mm diameter probe into an 18-20mm borehole at 2-3m depth while maintaining soil contact for electrical/magnetic measurements**.

The challenge is multi-faceted:
- Unconsolidated soil may collapse the hole before or during insertion
- The probe must achieve intimate contact with surrounding soil
- No permanent casing can remain (interferes with measurements)
- Methods must work across sand, clay, and mixed soils
- UXO safety requirements prohibit percussion/vibration

### Key Findings

| Solution Category | Most Promising Approach | Applicability |
|-------------------|------------------------|---------------|
| **Simultaneous Push-Insert** | Geoprobe DT22-style dual tube system | All soils - RECOMMENDED |
| **Fluid-Assisted** | Water column with polymer mud | Sandy/silty soils |
| **Temporary Stabilization** | Biodegradable polymer drilling fluid | Variable soils |
| **Soil Contact** | Natural collapse + time | Sandy/loose soils |
| **Self-Installing** | Integrated expendable tip probe | All soils - RECOMMENDED |

### Primary Recommendation

**Adopt a dual-tube approach with integrated expendable tip:**
1. Push outer casing (2.25" / 57mm OD) to full depth using hydraulic static push
2. Insert probe with expendable tip inside casing
3. Withdraw outer casing while probe remains
4. Allow natural soil collapse to establish contact
5. For difficult soils: use sand backfill or grout around probe

---

## Part 1: Borehole Collapse Mechanics

### 1.1 What Causes Collapse in Different Soil Types

Understanding collapse mechanisms is essential for selecting prevention methods.

#### Sandy/Granular Soils

**Collapse Mechanism:** Lack of cohesion between particles. Sand grains collapse toward the hole immediately after removing drill rods, creating a larger, weaker disturbed zone.

**Characteristics:**
- Collapse is **immediate to seconds** after support removal
- Complete infilling can occur within minutes
- Collapse rate increases with saturation
- Loose sand collapses "almost as fast as it can be bailed"

**Risk Factors:**
- Saturated conditions increase collapse severity
- Fine sand more prone than coarse
- Increasing depth = increasing overburden pressure = faster collapse

#### Silty Soils

**Collapse Mechanism:** Intermediate behavior - some cohesion when moist, but loses structure when disturbed or saturated.

**Characteristics:**
- Collapse occurs over minutes to hours
- "Formation heave" in saturated conditions - material flows upward into void
- Water table effects are pronounced

**Risk Factors:**
- Below water table: very problematic
- Disturbance during insertion triggers collapse

#### Clay Soils

**Collapse Mechanism:** Plastic deformation ("squeezing") rather than particle collapse. Clay slowly flows into the void.

**Characteristics:**
- Collapse is **gradual** - hours to days
- More time available for probe insertion
- Hole may remain open long enough for simple insertion
- Swelling clays can close around probe (beneficial for contact)

**Risk Factors:**
- Soft clays squeeze faster
- May grip/bind on probe during withdrawal operations
- Stiff clays may maintain gap around probe (poor contact)

#### Mixed/Crater Fill Soils

**HIRT Application Note:** WWII bomb crater fill is typically highly disturbed, poorly sorted mixture of native soil, rubble, and organic material. Expect variable and unpredictable behavior.

**Characteristics:**
- Heterogeneous - collapse behavior varies along depth
- May encounter layers of different stability
- Rubble can cause localized voids or bridging

**Sources:**
- [Lone Star Drills - Borehole Collapse Prevention](https://www.lonestardrills.com/prevent-borehole-collapse/)
- [Jeffrey Machine - Pile Drilling in Loose Soils](https://www.jeffreymachine.com/blog/a-complete-guide-to-pile-drilling-loose-soils)
- [ScienceDirect - Borehole Collapse](https://www.sciencedirect.com/topics/engineering/borehole-collapse)

---

### 1.2 Collapse Timing Summary

| Soil Type | Typical Collapse Time | HIRT Insertion Window |
|-----------|----------------------|----------------------|
| Loose dry sand | Seconds | Insufficient |
| Loose saturated sand | Immediate | None - requires casing |
| Dense sand | Seconds to minutes | Marginal |
| Silt (dry) | Minutes to hours | Adequate |
| Silt (saturated) | Seconds to minutes | Marginal |
| Soft clay | Hours to days | Adequate |
| Stiff clay | Days+ | Excellent |
| Mixed crater fill | Variable | Unpredictable |

**Key Insight:** In sandy soils, the insertion method must either:
1. Maintain support throughout (casing, fluid column), OR
2. Install probe simultaneously with advance (no open hole), OR
3. Accept that collapse provides the soil contact

---

### 1.3 Water Table Effects

The water table significantly affects collapse behavior:

**Above Water Table:**
- Capillary suction provides temporary apparent cohesion
- Moist sand may stand briefly; dry sand may flow
- Clay soils generally stable

**Below Water Table:**
- Hydrostatic pressure accelerates collapse
- Saturated sand behaves nearly like fluid
- "Formation heave" - saturated sand flows upward into void when sampler removed
- Water must be added to casing to balance pressure and prevent heave

**HIRT Implication:** If probe deployment extends below water table, dual-tube or fluid-filled methods become essential.

---

### 1.4 Collapse Risk Assessment by Soil Type

| Factor | Low Risk | Medium Risk | High Risk |
|--------|----------|-------------|-----------|
| Soil type | Stiff clay | Soft clay, dense sand | Loose sand, silt |
| Saturation | Dry | Moist | Saturated |
| Depth | <1m | 1-2m | >2m |
| Disturbance | Minimal | Some | Significant |
| Overburden | Light | Medium | Heavy |

**Practical Assessment:** Before survey, use pilot hole or soil probe to assess collapse risk. If hole stays open for >30 seconds, simple insertion may work. If hole collapses immediately, use dual-tube system.

---

## Part 2: Simultaneous Push-Insert Methods (RECOMMENDED)

### 2.1 Geoprobe DT22 Dual Tube System

**The industry gold standard for sampling in collapsing soils.**

**Principle:** Two concentric rod systems - outer casing advances and remains in place while inner sampler is cycled.

**Specifications:**
- Outer casing: 2.25" (57mm) OD probe rods
- Inner sampler: 1.25" (32mm) OD rods with liner
- Sample/probe diameter: 1.125" (28.6mm)
- Cutting shoe threads into leading end of outer casing

**Operation:**
1. Outer casing with cutting shoe is driven to depth
2. Inner rod string with liner is inserted through casing
3. Liner held against cutting shoe while casing advances
4. Inner rods removed, sample retrieved
5. **For HIRT:** Instead of removing sample, leave probe in place
6. Withdraw outer casing while probe remains

**Key Advantages for HIRT:**
- Outer casing **prevents borehole collapse** at all times
- Probe hole remains sealed and aligned during insertion
- Eliminates cross-contamination between depths
- Works in loose sand, saturated sand, all problematic soils
- Built-in grouting capability through tool string

**Adaptation for HIRT:**
- Use 2.25" OD outer casing (existing Geoprobe tooling)
- Design 16mm HIRT probe with attachment to 1.25" center rods
- Push outer casing to 3m depth
- Insert probe through casing
- Retract outer casing, allowing soil to collapse onto probe
- Optional: pump sand or grout as casing withdraws

**Equipment Cost:** ~$5,000-8,000 for DT22 tooling (requires Geoprobe rig)

**Sources:**
- [Geoprobe - DT22 Soil Sampling System](https://geoprobe.com/tooling/dt22-soil-sampling-system)
- [Geoprobe - Soil Sampling Systems Overview](https://geoprobe.com/tooling/soil-sampling-systems-direct-push)
- [Enviro Wiki - Direct Push Sampling](https://www.enviro.wiki/index.php?title=Direct_Push_Sampling)

---

### 2.2 Single-Tube Methods with Expendable Tip

**Principle:** Probe is inserted inside hollow push rods with an expendable (sacrificial) tip. Rods are pushed to depth, then withdrawn, leaving probe and tip in place.

**Commercial Example:** Geoprobe soil gas samplers with expendable tips

**Operation:**
1. Probe pre-loaded inside hollow push rod
2. Expendable tip attached to leading end
3. Rod string pushed to full depth (soil displaced, not removed)
4. Rods retracted slightly to separate from tip
5. Rods fully withdrawn
6. Probe and expendable tip remain in soil
7. Displaced soil naturally collapses around probe

**Key Advantages:**
- Simpler than dual-tube (one rod string)
- Probe protected during entire insertion
- No open hole at any time
- Expendable tip can be designed for HIRT

**HIRT Adaptation - Expendable Tip Design:**

| Component | Specification |
|-----------|---------------|
| Material | Biodegradable plastic or aluminum |
| Diameter | 18-20mm (to match borehole) |
| Geometry | Tapered/conical for displacement |
| Attachment | Break-away connection to probe |
| Cost per unit | $0.50-2.00 (mass production) |

**Critical Design Consideration:** Tip material must not interfere with HIRT measurements. Options:
- Plastic tip (electrically inert)
- Aluminum tip (conductive but non-magnetic)
- Tip designed to separate and fall away during retraction

**Sources:**
- [ENVCO - Expendable Tip Soil Gas Sampler](https://envcoglobal.com/catalog/civil-and-geotechnical/drill-rigs/direct-push-tooling/8-heavy-duty-expendable/)
- [Geoprobe - Direct Push Installation](https://geoprobe.com/sites/default/files/storage/pdfs/soil_gas_sampling_and_monitoring_mk3098_0_0.pdf)

---

### 2.3 Hollow Push Rod with Integrated Sensors

**Novel Concept: The push rod IS the probe**

**Principle:** Instead of inserting a separate probe, integrate sensors directly into a hollow push rod that remains in place.

**Design Concept:**

```
Cross-Section (not to scale):

   +------------------+
   |  Cable conduit   | 3-4mm
   +------------------+
   |  Sensor array    | 10mm
   +------------------+
   |  Steel wall      | 2mm
   +------------------+

   Total OD: ~18mm
   Total Length: 3m
```

**Advantages:**
- No insertion step - probe advances with push
- Full soil contact guaranteed (displacement fit)
- Rod provides structural rigidity
- Multiple sensors along length possible

**Challenges:**
- Higher per-probe cost (steel tube + sensors)
- Probe becomes consumable if not retrievable
- Limited flexibility in sensor configuration

**Application:** Best suited if probe design is finalized and high-volume production justifies tooling.

---

### 2.4 Self-Boring Pressuremeter Concept

**Industry Analog:** Self-boring pressuremeter (SBP)

**Principle:** Probe creates its own borehole as it advances, minimizing disturbance and ensuring perfect fit.

**How SBP Works:**
- Hollow tube with internal rotating cutter
- Sharp tapered cutting shoe at leading edge
- Drilling fluid flushes cuttings through center
- Creates hole that is "exact fit to the pressuremeter"

**Adaptation for HIRT:**
- Miniaturized self-boring head (~18mm diameter)
- Internal cutter removes soil cores
- Probe body follows immediately behind
- Zero time for collapse - continuous support

**Complexity Assessment:** HIGH - requires precision miniaturized cutting mechanism, fluid handling, cuttings disposal. Likely overkill for HIRT application.

**Sources:**
- [Cambridge Insitu - Self-Boring Pressuremeter](https://www.cambridge-insitu.com/self-boring-pressuremeter)
- [Roctest - BOREMAC Self-Boring Pressuremeter](https://roctest.com/en/product/boremac-self-boring-pressuremeter/)

---

## Part 3: Temporary Stabilization Methods

### 3.1 Fluid Column (Hydrostatic Support)

**Principle:** Water or drilling fluid in borehole provides positive pressure against borehole walls, preventing collapse.

**How It Works:**
- Fluid density creates hydrostatic pressure proportional to depth
- Positive pressure against walls resists inward collapse
- Works best with filter cake formation (bentonite/polymer)

**Pressure Calculation:**
```
Hydrostatic Pressure = density x gravity x depth
For water at 3m: ~0.3 bar (4.3 psi)
For bentonite mud at 3m: ~0.35 bar (5 psi)
```

**Key Requirement:** Fluid level must remain **above water table** to maintain positive pressure. Even small positive pressure is sufficient.

**Practical Implementation for HIRT:**

1. Create borehole using water jetting
2. Maintain water column in hole during entire operation
3. Top up water as it seeps into formation
4. Insert probe through water column
5. Allow probe to displace fluid as it descends
6. After probe in place, water drains/seeps away

**Limitations:**
- Requires continuous water supply
- Does not work well in highly permeable gravels
- Fluid loss in coarse soils may exceed replenishment rate

**Sources:**
- [The Driller - Hydrostatic Pressure for Borehole Stability](https://www.thedriller.com/articles/90515-how-does-hydrostatic-head-pressure-ensure-borehole-stability)
- [The Driller - Hydrostatic Pressure](https://www.thedriller.com/articles/92314-hydrostatic-pressure-both-a-friend-and-foe-for-drillers)

---

### 3.2 Bentonite and Polymer Drilling Fluids

**Principle:** Specialized fluids create a semi-permeable barrier (filter cake) on borehole walls, providing structural support beyond simple hydrostatic pressure.

#### Bentonite Mud

**How It Works:**
- Bentonite clay platelets form filter cake on borehole wall
- Filter cake is nearly impermeable
- Hydrostatic pressure acts against filter cake
- Cake thickness: 1-3mm typically

**CRITICAL WARNING FOR HIRT:**

Bentonite significantly affects soil electrical resistivity:
- Resistivity decreases by up to 80% with bentonite
- Creates electrically conductive layer between probe and native soil
- **May severely compromise ERT measurements**

**Bentonite Resistivity Values:**
- Wet bentonite: ~2.5 ohm-m (very conductive)
- Native soil: 10-1000 ohm-m (variable)
- **Contrast creates measurement artifact**

**Recommendation:** Avoid bentonite mud for HIRT applications where electrical measurements are critical. Use polymers instead.

**Sources:**
- [ResearchGate - Bentonite Effects on Soil Resistivity](https://www.researchgate.net/publication/268872902_Effects_of_Bentonite_Content_on_Electrical_Resistivity_of_Soils)
- [Power Quality Blog - Soil Resistivity](https://powerquality.blog/2023/07/31/how-to-reduce-the-soils-resistivity/)

#### Biodegradable Polymer Fluids

**Preferred Alternative for HIRT**

**How It Works:**
- Water-soluble polymers increase fluid viscosity
- Forms gel membrane on borehole walls
- Degrades over time (hours to days)
- Does not permanently alter soil properties

**Common Polymers:**
- **Xanthan gum** - Most common, biodegradable, non-toxic
- **Guar gum** - Biodegradable, short working life
- **PAC (polyanionic cellulose)** - Synthetic but degrades

**Commercial Products:**
- BLACK-BEAR (Matrix Construction Products) - biodegradable, enzyme-breakable
- Bio-Bore (Baroid) - clay-free, biodegradable
- Pure-Bore - multifunctional, low environmental impact

**HIRT Application:**
1. Mix polymer fluid (xanthan gum at 0.5-1% concentration)
2. Fill borehole with polymer fluid during jetting
3. Polymer gel maintains hole stability
4. Insert probe through gel
5. Gel breaks down over 24-48 hours
6. Native soil collapses onto probe

**Electrical Properties:** Polymer fluids have minimal effect on soil electrical properties after degradation. Initial gel is water-based with slight conductivity increase, but degrades to negligible effect.

**Sources:**
- [Matrix CP - BlackBear Biodegradable Drilling Fluid](https://www.matrixcp.com/blackbear-biodegradable-drilling-fluid)
- [The Driller - Polymer Selection for Drilling Fluids](https://www.thedriller.com/articles/92834-contractor-tips-polymer-selection-for-drilling-fluids)

---

### 3.3 Artificial Ground Freezing

**Principle:** Freeze pore water in soil to create temporary solid/stable borehole walls.

**How It Works:**
- Refrigerant circulates through freeze pipes
- Pore water converts to ice
- Ice bonds soil particles together
- Frozen soil becomes impermeable and strong

**Typical Parameters:**
- Brine temperature: -30 to -38 degrees C
- Freeze pipe spacing: 0.8-2m
- Freeze time: Hours to days depending on soil

**Assessment for HIRT:**

| Factor | Rating | Notes |
|--------|--------|-------|
| Technical feasibility | Possible | Proven technology |
| Cost | Very High | Requires refrigeration equipment |
| Practicality | Very Low | Massive overkill for small holes |
| Field deployment | Impractical | Heavy equipment, power requirements |

**Verdict:** **NOT RECOMMENDED** - Ground freezing is for large civil works (tunnels, shafts). Far too complex and expensive for small probe holes.

**Potential Niche Use:** If HIRT surveys are conducted in permafrost regions, natural frozen ground eliminates collapse concerns.

**Sources:**
- [GE+ - Artificial Ground Freezing](https://www.geplus.co.uk/features/how-artificial-ground-freezing-works-as-a-ground-improvement-technique-29-11-2022/)
- [The Constructor - Ground Freezing Technique](https://theconstructor.org/geotechnical/ground-freezing-technique/16944/)

---

### 3.4 Compressed Air Support

**Concept:** Use air pressure to prevent collapse similar to caisson work.

**Assessment:** Impractical for small-diameter open holes. Air would escape through permeable soil immediately. Only works with sealed chambers (pneumatic caissons). **NOT RECOMMENDED.**

---

### 3.5 Inflatable Bladders/Packers

**Principle:** Expandable rubber elements seal against borehole wall.

**Industry Application:** Borehole packers for groundwater sampling, pressure testing, grouting.

**How It Works:**
- Deflated packer lowered into borehole
- Packer inflated with air/water/hydraulic fluid
- Rubber expands to seal against borehole wall
- Creates isolated zone between packers

**Potential HIRT Application:**

**Concept A: Packer-Assisted Insertion**
1. Insert deflated packer to depth
2. Inflate to stabilize upper borehole section
3. Drill/push through packer to lower depth
4. Progressive advancement with staged packers

**Assessment:** Adds significant complexity. Better suited for deep wells than 3m probe holes.

**Concept B: Inflatable Probe Section**
- Probe with expandable rubber section
- After insertion, inflate to ensure soil contact
- Solves the "contact" problem, not the "collapse" problem

**Assessment:** Potentially useful for ensuring contact in stiff clays or boreholes that don't collapse naturally. Adds mechanical complexity to probe.

**Sources:**
- [RST Instruments - Borehole Packers](https://rstinstruments.com/product-category/instruments/borehole-packers/)
- [Roctest - Inflatable Packers](https://roctest.com/en/product/inflatable-packers-bch/)

---

## Part 4: Soil Contact Solutions

### 4.1 Natural Collapse (Recommended Primary Method)

**Principle:** After probe insertion, surrounding soil naturally collapses to fill annular gap.

**When It Works:**
- Sandy soils: Excellent - immediate collapse
- Silty soils: Good - collapse within minutes to hours
- Soft clay: Fair - gradual squeezing over hours
- Stiff clay: Poor - may maintain gap

**Procedure:**
1. Install probe to depth (via casing, direct push, etc.)
2. Withdraw casing or support
3. Wait for natural collapse (time depends on soil)
4. Verify contact via sensor readings or TDR check

**Advantages:**
- Simplest method
- No additional materials
- Guaranteed to work in loose soils
- No interference with measurements

**Verification Method:**
- Electrical continuity check between probe electrodes and surface reference
- TDR (Time Domain Reflectometry) signature change
- Initial measurement comparison to expected values

---

### 4.2 Sand Backfill

**Principle:** Pour clean sand into annulus between probe and borehole wall.

**Standard Practice:** Vibrating wire piezometer installation uses sand pack around sensor.

**Procedure:**
1. Install probe in borehole
2. Prepare clean, fine sand (saturated for below water table)
3. Pour sand around probe, filling annulus
4. Tamp lightly if needed (avoid probe damage)
5. Sand provides continuous contact path

**Sand Specifications:**
- Grain size: Fine to medium (0.1-2mm)
- Material: Clean silica sand, no fines
- Saturate before placement if below water table

**HIRT Consideration:** Sand has different electrical properties than native soil. For ERT measurements, this creates a "skin" effect. However, sand is relatively neutral (high resistivity when dry, moderate when saturated) and may be acceptable.

**Alternative:** Use native soil from borehole cuttings as backfill - matches formation properties.

**Sources:**
- [Encardio Rite - VW Piezometer Installation](https://www.encardio.com/blog/vibrating-wire-piezometer-installation-procedure-in-a-borehole)
- [Geokon - Piezometer Manual](https://www.geokon.com/content/manuals/4500/topics/04_installation.htm)

---

### 4.3 Grout Injection

**Principle:** Pump grout slurry into annulus to fill voids and ensure contact.

**CRITICAL HIRT CONSIDERATION:**

Grout selection is crucial for electrical/magnetic measurements:

| Grout Type | Electrical Effect | Magnetic Effect | HIRT Suitability |
|------------|-------------------|-----------------|------------------|
| Portland cement | Highly conductive when wet | Neutral | POOR |
| Bentonite | Very conductive | Neutral | POOR |
| Cement-bentonite | Highly conductive | Neutral | POOR |
| Gypsum | Moderate conductivity | Neutral | FAIR |
| Silica sand slurry | Low conductivity | Neutral | GOOD |

**Recommendation:** If grout is necessary, use sand slurry (sand + water) or native soil slurry rather than cement or bentonite.

**Fully Grouted Method:**
Modern piezometer practice increasingly uses "fully grouted" installation - entire borehole filled with low-permeability grout. This is faster but may not be suitable for HIRT due to electrical interference.

**Sources:**
- [Geosense - VW Piezometers](https://www.geosense.com/tech-notes/vibrating-wire-piezometers/)
- [Soilinstruments - VW Piezometer Manual](https://www.soilinstruments.com/wp-content/uploads/2020/12/Man106-Vibrating-Wire-Piezometer-Standard-MN1114-Rev1.4.1.pdf)

---

### 4.4 Expandable Probe Sections

**Concept:** Probe with mechanically expandable sections to ensure contact.

**Design Options:**

**Option A: Spring-Loaded Contact Fins**
- Small spring-loaded metal fins along probe body
- Compressed during insertion
- Spring out against borehole wall after deployment
- Provide multiple contact points

**Design Sketch:**
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

**Option B: Inflatable Bladder Sections**
- Rubber sections that inflate after insertion
- Pneumatic or hydraulic inflation
- Provides 360-degree contact

**Option C: Swelling Materials**
- Hydrogel or water-swelling polymer coating
- Absorbs soil moisture and expands
- Passive - no mechanical action required
- Swelling of 10-1000x original volume possible

**Assessment:**

| Option | Complexity | Reliability | Contact Quality | HIRT Compatibility |
|--------|------------|-------------|-----------------|-------------------|
| Spring fins | Medium | Good | Point contact | Good (metal contact) |
| Inflatable | High | Fair | Full circumference | Good |
| Swelling material | Low | Variable | Variable | Uncertain (electrical?) |

**Recommendation:** Spring-loaded contact fins are the most practical active solution. Simple, reliable, and provide direct metal-to-soil contact for electrical measurements.

---

### 4.5 Tapered Probe Design

**Principle:** Slightly tapered probe creates interference fit with borehole.

**Design:**
- Tip diameter: Matches or slightly smaller than borehole
- Body diameter: 0.5-1mm larger than borehole
- Taper angle: Very gradual (1-2 degrees)

**How It Works:**
- Probe slides easily into hole initially
- As pushed deeper, larger diameter compresses against walls
- Final position creates interference fit
- Friction prevents upward migration

**Advantages:**
- Simple design modification
- No moving parts
- Guaranteed contact along entire length
- Insertion force indicates contact quality

**Disadvantages:**
- May require significant insertion force
- Difficult to remove if needed
- Borehole diameter must be consistent

---

## Part 5: Self-Installing Probe Concepts

### 5.1 Integrated Push Tip Design (RECOMMENDED)

**Concept:** Probe has integrated conical tip that remains part of the probe.

**Design:**
```
           /\
          /  \
         /    \     <- Conical tip (expendable or integrated)
        /______\
       |        |
       | PROBE  |   <- Instrumented body
       | BODY   |
       |________|
          ||
         Cable
```

**Operation:**
1. Probe with tip inserted into hollow push rod
2. Entire assembly pushed to depth (soil displaced)
3. Push rod retracted
4. Probe with tip remains in soil
5. Displaced soil collapses around probe body

**Tip Options:**

| Tip Type | Material | Cost | Recovery |
|----------|----------|------|----------|
| Integrated steel | Hardened steel | High | Stays with probe |
| Expendable plastic | HDPE, nylon | $0.50-1 | Left in ground |
| Expendable aluminum | Alloy | $1-2 | Left in ground |
| Breakaway | Steel with shear pin | Medium | Separates on withdrawal |

**HIRT Recommendation:** Expendable plastic tip - electrically inert, low cost, acceptable for single-use deployments.

---

### 5.2 Direct-Push Probe (No Pre-Drilling)

**Concept:** Skip borehole creation entirely. Push probe directly into ground.

**Requirements:**
- Robust conical tip
- Sufficient push force
- Probe body withstands lateral soil pressure

**Force Calculation (Approximate):**

For 16mm probe in medium-density soil:
```
Tip resistance: ~50-100 kPa per mm penetration
Sleeve friction: ~20-50 kPa along shaft
At 3m depth, 16mm probe:
  Tip force: 3-6 kN
  Friction force: 3-5 kN
  Total: 6-11 kN
```

This is within capability of lightweight hydraulic systems (10 kN design target).

**Advantages:**
- Simplest possible approach
- Maximum soil contact (interference fit)
- No hole stability concerns
- Single operation

**Disadvantages:**
- May not penetrate gravel/rubble
- Higher insertion force than pre-drilled hole
- Potential for probe damage if obstruction hit

**Recommendation:** Test direct-push as primary method. Fall back to pre-drilling only if obstructions encountered.

---

### 5.3 Vibration-Assisted Insertion

**CRITICAL UXO WARNING:**

**Vibration is PROHIBITED at potential UXO sites.**

Vibration-assisted methods (vibratory hammer, sonic drilling, vibrating probe insertion) are standard techniques for easier penetration but represent unacceptable risk at WWII bomb crater sites.

**Do NOT use:**
- Vibratory pile drivers
- Sonic drilling
- Vibrating probe insertion tools
- Hammer drills or percussion tools

**Safe alternatives that achieve similar benefits:**
- Water jet lubrication (hydraulic, not mechanical)
- Polymer mud lubrication
- Slow, steady static push

---

### 5.4 Two-Stage Pilot Hole Method

**Concept:** Create small pilot hole first, then enlarge with probe.

**Procedure:**
1. Water jet or push 12mm pilot hole to depth
2. Pilot hole verifies path is clear of obstructions
3. Insert 16mm probe into 12mm hole
4. Probe enlarges hole as it descends (displacement)
5. Tight fit ensures excellent soil contact

**Advantages:**
- Lower force for pilot hole
- Verifies path before committing full-size probe
- Excellent soil contact from interference fit

**Disadvantages:**
- Two operations instead of one
- Pilot hole may collapse before probe insertion

**Hybrid Approach:**
- Pilot rod pushed to depth
- Larger probe pushed down alongside pilot rod
- Pilot rod withdrawn after probe in place
- Combines verification with single-pass insertion

---

## Part 6: Industry Analog Summary

### 6.1 CPT Cone Penetrometer Testing

**Relevance:** Gold standard for in-situ geotechnical testing.

**Key Lessons for HIRT:**
- Push rate: 2 cm/sec (slow, steady)
- Rods flush-threaded for continuous advance
- Sensors in probe tip with cable through hollow rods
- Built-in grouting through rod string
- Multiple sensors possible (resistivity, seismic, etc.)

**CPT with Sensors:** CPT probes already incorporate electrical resistivity sensors for soil characterization. HIRT can adopt similar sensor integration and data acquisition approaches.

**Sources:**
- [Geoprobe - CPT Testing](https://geoprobe.com/direct-image/cpt-cone-penetration-testing)
- [Wikipedia - Cone Penetration Test](https://en.wikipedia.org/wiki/Cone_penetration_test)

---

### 6.2 Monitoring Well Installation

**Relevance:** Installing sensor packages in boreholes with soil/gravel pack.

**Key Lessons:**
- Filter pack (sand) ensures hydraulic connection
- Bentonite seal isolates monitoring zone
- Hollow-stem auger provides temporary casing
- Pre-packed screens simplify installation

**Adaptation for HIRT:**
- Use pre-packed probe design with built-in sand filter
- Adopt tremie pipe technique for placing materials
- Consider formation collapse as beneficial (provides contact)

**Sources:**
- [EPA - Monitoring Well Design and Installation](https://www.epa.gov/sites/default/files/2016-01/documents/design_and_installation_of_monitoring_wells.pdf)
- [Florida DEP - Monitoring Well Manual](https://floridadep.gov/sites/default/files/monitoring-well-manual-formatted-final_2.pdf)

---

### 6.3 Ground Anchor / Soil Nail Installation

**Relevance:** Installing steel rods in collapsing soil.

**Key Lessons:**
- Self-drilling anchors combine drilling and grouting
- Hollow bar allows grout injection during advance
- Pressure grouting fills voids and bonds to soil
- Works in loose, collapsing soils

**Self-Drilling System Analogy:**
The self-drilling soil nail is directly analogous to HIRT probe installation:
- Hollow bar with drill bit at end
- Grout pumped through hollow center as drill advances
- No need for pre-drilling or casing
- Works specifically in "loose or collapsing soils where conventional methods may fail"

**Potential HIRT Adaptation:**
- Hollow probe with jetting tip
- Water pumped through probe during insertion
- Water jets create path, probe follows immediately
- Single operation, no collapse opportunity

**Sources:**
- [Sinorock - Soil Nailing in Loose Soil](https://www.sinorockco.com/news/industry-news/slope-stabilzation-in-loose-soil.html)
- [Geostabilization - Soil Nails Guide](https://www.geostabilization.com/blog-posts/soil-nails-a-guide-to-strengthening-ground-stability/)

---

### 6.4 Vibrating Wire Piezometer Installation

**Relevance:** Installing electronic sensors in boreholes with guaranteed soil contact.

**Key Lessons:**
- Sensor in geotextile bag with saturated sand
- Filter cloth prevents fines migration
- Sand pack creates "response zone" around sensor
- Minimal flow required (VW piezometer is no-flow device)
- Fully grouted method gaining popularity (simpler)

**HIRT Adaptation:**
- Consider probe sleeve made of geotextile
- Sleeve filled with native soil or sand
- Provides buffer for minor annular gaps
- Geotextile degrades over time (if biodegradable)

**Sources:**
- [Encardio Rite - VW Piezometer Installation in Borehole](https://www.encardio.com/blog/vibrating-wire-piezometer-installation-procedure-in-a-borehole)
- [RST Instruments - VW Piezometer Manual](https://rstinstruments.com/wp-content/uploads/ELM005U-VW2100-Vibrating-Wire-Piezometer-Instruction-Manual.pdf)

---

## Part 7: Novel Concepts Assessment

### 7.1 Hollow Push Rod as Probe

**Concept Rating: HIGH POTENTIAL**

**Description:** Instead of inserting probe into pre-formed hole, the push rod itself contains all sensors and remains in place.

**Advantages:**
- Eliminates insertion step entirely
- Guaranteed soil contact (displacement fit)
- Rod provides structural support
- Data cable runs through center

**Implementation:**
- Design 18mm OD hollow steel tube
- Integrate sensors into tube wall or on external surface
- Internal cable routing
- Expendable tip attached for pushing
- After survey, rod remains in place (or retrieved with hydraulic extraction)

**Cost Consideration:** Higher per-probe cost, but eliminates separate rod system.

---

### 7.2 Swelling Probe Coating

**Concept Rating: EXPERIMENTAL**

**Description:** Probe coated with hydrogel that swells when contacted by soil moisture.

**How It Works:**
- Hydrogel coating applied to probe body
- Initially thin/compact
- Absorbs soil moisture over 30-60 minutes
- Swells to fill annular gap
- Provides intimate contact with soil

**Concerns:**
- Electrical properties of hydrogel unknown for ERT
- Swelling time (30-60 min) may be too slow
- Reliability in dry soils questionable
- Long-term stability unknown

**Assessment:** Intriguing concept but requires testing before adoption.

**Sources:**
- [PMC - Proximal Soil Moisture Sensors Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7730258/)
- [MDPI - Soil Moisture Sensing Technologies](https://www.mdpi.com/2073-4395/15/12/2788)

---

### 7.3 Pneumatic Expansion Section

**Concept Rating: MODERATE POTENTIAL**

**Description:** Probe with inflatable section that expands against borehole wall.

**Design:**
- Rigid probe body with rubber bladder section
- Small air line runs through data cable
- After insertion, pump air to inflate bladder
- Bladder presses probe against borehole wall

**Advantages:**
- Active contact control
- Works in various hole sizes
- Can verify contact via pressure feedback

**Disadvantages:**
- Additional pneumatic system needed
- Rubber may degrade over time
- Puncture risk in rocky soil

---

### 7.4 Probe with Multiple Sensor Pods

**Concept Rating: HIGH POTENTIAL**

**Description:** Instead of single continuous probe, use string of discrete sensor pods.

**Design:**
```
Surface
   |
   |--- Cable
   |
  [POD 1] @ 0.5m - Spring contacts
   |
  [POD 2] @ 1.0m - Spring contacts
   |
  [POD 3] @ 2.0m - Spring contacts
   |
  [POD 4] @ 3.0m - Spring contacts
   |
  Tip
```

**Advantages:**
- Each pod has independent spring-loaded contacts
- Flexible - pods can be added/removed
- Easier to achieve contact at multiple depths
- Modular manufacturing

**Implementation:**
- Push carrier tube to depth
- Drop pod string into tube
- Retract tube, pods remain
- Spring contacts engage soil at each depth

---

## Part 8: Recommended Solutions Matrix

### 8.1 By Soil Type

| Soil Type | Primary Method | Backup Method | Contact Method |
|-----------|---------------|---------------|----------------|
| Loose sand | Dual-tube casing | Expendable tip direct push | Natural collapse |
| Dense sand | Direct push | Pre-drill + insert | Natural collapse |
| Saturated sand | Dual-tube casing (REQUIRED) | N/A | Natural collapse + water balance |
| Silt | Polymer fluid + insert | Direct push | Natural collapse |
| Soft clay | Direct push | Pre-drill + insert | Clay squeezing |
| Stiff clay | Pre-drill + insert | Direct push | Sand backfill |
| Mixed/rubble | Dual-tube casing | Water jet + casing | Sand backfill |

---

### 8.2 By Equipment Availability

| Equipment Available | Recommended Method |
|--------------------|-------------------|
| Geoprobe rig | DT22 dual-tube with probe adapter |
| CPT rig | Modify rod string for HIRT probe |
| Water jet system only | Pre-jet hole, polymer stabilize, insert |
| Basic hydraulic push only | Direct push with expendable tip probe |
| No powered equipment | Hand auger + polymer + manual insert |

---

### 8.3 Implementation Priority

**Immediate (Test First):**
1. Direct push with expendable tip - simplest, test if soil allows
2. Water jet + polymer fluid + insert - for sandy soils

**Near-Term Development:**
1. Dual-tube adapter for Geoprobe DT22 system
2. Integrated push-tip probe design
3. Spring-loaded contact fin prototype

**Long-Term (If Needed):**
1. Hollow push rod with integrated sensors
2. Inflatable contact section
3. Swelling coating evaluation

---

## Part 9: Design Recommendations for HIRT Probe

### 9.1 Probe Body

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Diameter | 16mm body, 18mm tip | Allows use in 18-20mm borehole |
| Length | 3m standard | Match target depth |
| Material | Fiberglass composite | Non-magnetic, non-conductive structure |
| Surface | Smooth or micro-textured | Reduce insertion friction |

### 9.2 Tip Design

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Geometry | 60-degree cone | Industry standard, proven penetration |
| Material | Hardened plastic (PEEK) or aluminum | Expendable, non-interfering |
| Attachment | Push-fit with shear pin | Easy assembly, breaks away if stuck |
| Diameter | 18-20mm | Matches borehole, provides clearance for body |

### 9.3 Contact System

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Primary method | Natural soil collapse | Simplest, works in most soils |
| Backup method | Spring-loaded contact fins | Active contact for stiff soils |
| Electrode surface | Gold-plated or stainless | Corrosion resistance, low contact resistance |
| Electrode spacing | Per ERT requirements | Design for measurement geometry |

### 9.4 Cable and Retrieval

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Cable type | Kevlar-reinforced, waterproof | Strength for retrieval, environmental protection |
| Cable exit | Through tip (central) or side port | Protects during insertion |
| Retrieval attachment | Threaded or locking collar | Allows hydraulic extraction |

---

## Part 10: Field Procedure Recommendations

### 10.1 Pre-Survey Assessment

1. **Soil characterization:** Obtain soil type information for site
2. **Pilot test:** Push single probe to assess collapse behavior
3. **Method selection:** Choose primary and backup methods based on pilot
4. **Equipment check:** Verify all systems operational

### 10.2 Standard Installation Procedure

**For Direct Push Method:**

1. Position push rig over survey point
2. Load probe with expendable tip into push assembly
3. Align push rod vertical (within 2 degrees)
4. Begin push at 2 cm/sec rate
5. Monitor push force for obstructions
6. If obstruction: stop, assess, relocate if needed
7. Continue to target depth
8. Retract push rods slowly
9. Verify probe remains in place (cable tension)
10. Allow 5-10 minutes for soil settlement
11. Connect data acquisition
12. Verify sensor readings

**For Dual-Tube Method:**

1. Position rig, advance outer casing to depth
2. Insert probe through casing
3. Begin casing retraction
4. Optional: pump sand/native soil as casing withdraws
5. Complete casing retraction
6. Verify probe position
7. Connect and verify sensors

### 10.3 Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| Hole collapses before insertion | Sandy soil, high water table | Switch to dual-tube or direct push |
| Probe won't reach depth | Obstruction, tight borehole | Re-drill, enlarge hole |
| Poor sensor readings | Insufficient soil contact | Wait longer, add sand backfill |
| Probe stuck during retrieval | Soil grip, swelling clay | Hydraulic extraction, leave if necessary |
| Water in borehole | High water table | Use water-balanced installation |

---

## Conclusions

### Key Findings

1. **Borehole collapse is manageable** - Multiple proven methods exist from geotechnical industry

2. **Dual-tube systems are the gold standard** - Geoprobe DT22 and similar provide reliable installation in all soil types

3. **Direct push with expendable tip is simplest** - Works in many soils, should be tested as primary method

4. **Bentonite must be avoided** - Severely impacts electrical measurements

5. **Natural collapse provides best contact** - In sandy/silty soils, letting soil collapse onto probe is ideal

6. **No exotic technology required** - All recommended methods use proven equipment

### Recommended Development Path

**Phase 1: Prototype and Test**
- Design expendable tip for 16mm probe
- Test direct push in representative soils
- Evaluate collapse behavior and contact quality

**Phase 2: Dual-Tube Adaptation**
- Design adapter for Geoprobe DT22 system
- Test in difficult soils (saturated sand)
- Validate grouting/backfill procedures

**Phase 3: Contact Enhancement**
- Prototype spring-loaded contact fins
- Test in stiff clay conditions
- Optimize for manufacturing

### Final Recommendation

**Primary approach:** Direct push with integrated expendable tip, relying on natural soil collapse for contact.

**Backup for difficult soils:** Dual-tube system (DT22-style) with controlled withdrawal and optional sand backfill.

**Avoid:** Bentonite mud, vibration-assisted methods, complex mechanisms.

---

## References

### Borehole Collapse and Stabilization
- [Lone Star Drills - Prevent Borehole Collapse](https://www.lonestardrills.com/prevent-borehole-collapse/)
- [Jeffrey Machine - Pile Drilling in Loose Soils](https://www.jeffreymachine.com/blog/a-complete-guide-to-pile-drilling-loose-soils)
- [Pile Buck - Tackling Challenging Soils](https://pilebuck.com/tackling-challenging-soils-proven-techniques-pile-driving-foundation-drilling/)
- [Western Equipment Solutions - Temporary Casing](https://westernequipmentsolutions.com/best-practices-for-using-temporary-casing-during-drilling-operations/)

### Direct Push and Dual-Tube Systems
- [Geoprobe - DT22 Soil Sampling System](https://geoprobe.com/tooling/dt22-soil-sampling-system)
- [Geoprobe - Soil Sampling Systems](https://geoprobe.com/tooling/soil-sampling-systems-direct-push)
- [Enviro Wiki - Direct Push Sampling](https://www.enviro.wiki/index.php?title=Direct_Push_Sampling)
- [Ohio EPA - Direct Push Technologies](https://dam.assets.ohio.gov/image/upload/epa.ohio.gov/Portals/30/remedial/docs/groundwater/TGM%20Chap15%20Final,%202-2005Arch.pdf)

### CPT and Penetrometer Testing
- [Geoprobe - CPT Testing](https://geoprobe.com/direct-image/cpt-cone-penetration-testing)
- [Wikipedia - Cone Penetration Test](https://en.wikipedia.org/wiki/Cone_penetration_test)
- [Gregg Drilling - CPT Guide](https://www.novotechsoftware.com/downloads/PDF/en/Ref/CPT-Guide-5ed-Nov2012.pdf)
- [ITRC - Cone Penetrometer Testing](https://asct-1.itrcweb.org/3-5-cone-penetrometer-testing/)

### Drilling Fluids
- [Matrix CP - BlackBear Biodegradable Fluid](https://www.matrixcp.com/blackbear-biodegradable-drilling-fluid)
- [The Driller - Polymer Selection](https://www.thedriller.com/articles/92834-contractor-tips-polymer-selection-for-drilling-fluids)
- [The Driller - Hydrostatic Pressure](https://www.thedriller.com/articles/90515-how-does-hydrostatic-head-pressure-ensure-borehole-stability)

### Bentonite and Electrical Properties
- [ResearchGate - Bentonite Effects on Resistivity](https://www.researchgate.net/publication/268872902_Effects_of_Bentonite_Content_on_Electrical_Resistivity_of_Soils)
- [Power Quality Blog - Soil Resistivity](https://powerquality.blog/2023/07/31/how-to-reduce-the-soils-resistivity/)

### Monitoring Well and Piezometer Installation
- [EPA - Monitoring Well Design](https://www.epa.gov/sites/default/files/2016-01/documents/design_and_installation_of_monitoring_wells.pdf)
- [Encardio Rite - VW Piezometer Installation](https://www.encardio.com/blog/vibrating-wire-piezometer-installation-procedure-in-a-borehole)
- [RST Instruments - VW Piezometer Manual](https://rstinstruments.com/wp-content/uploads/ELM005U-VW2100-Vibrating-Wire-Piezometer-Instruction-Manual.pdf)

### Ground Anchors and Soil Nails
- [Sinorock - Soil Nailing Loose Soil](https://www.sinorockco.com/news/industry-news/slope-stabilzation-in-loose-soil.html)
- [Geostabilization - Soil Nails Guide](https://www.geostabilization.com/blog-posts/soil-nails-a-guide-to-strengthening-ground-stability/)
- [Caltrans - Ground Anchors and Soil Nails](https://dot.ca.gov/-/media/dot-media/programs/engineering/documents/structureconstruction/foundation/sc-foundation-chapt11-a11y.pdf)

### Self-Boring Pressuremeter
- [Cambridge Insitu - Self-Boring Pressuremeter](https://www.cambridge-insitu.com/self-boring-pressuremeter)
- [Roctest - BOREMAC](https://roctest.com/en/product/boremac-self-boring-pressuremeter/)

### Borehole Packers
- [RST Instruments - Borehole Packers](https://rstinstruments.com/product-category/instruments/borehole-packers/)
- [Roctest - Inflatable Packers](https://roctest.com/en/product/inflatable-packers-bch/)

### Ground Freezing
- [GE+ - Artificial Ground Freezing](https://www.geplus.co.uk/features/how-artificial-ground-freezing-works-as-a-ground-improvement-technique-29-11-2022/)
- [The Constructor - Ground Freezing](https://theconstructor.org/geotechnical/ground-freezing-technique/16944/)

### Soil Moisture Sensors and Hydrogels
- [PMC - Proximal Soil Moisture Sensors](https://pmc.ncbi.nlm.nih.gov/articles/PMC7730258/)
- [METER Group - Soil Moisture Sensor Installation](https://metergroup.com/expertise-library/how-to-install-soil-moisture-sensors-faster-better-and-for-higher-accuracy/)

### Expendable Tips and Accessories
- [ENVCO - Expendable Tip Samplers](https://envcoglobal.com/catalog/civil-and-geotechnical/drill-rigs/direct-push-tooling/8-heavy-duty-expendable/)
- [ECT Manufacturing - Expendable Points](https://ectmfg.com/product-category/direct-push-accessories/expendable-points/)

---

*Document compiled: 2026-01-19*
*For HIRT Geophysical Survey System development*
*Companion document to: probe-insertion-methods-summary.md*
