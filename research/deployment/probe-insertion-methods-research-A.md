# HIRT Probe Insertion Methods Research

## Research Context

**HIRT Probe Specifications:**
- Diameter: 16mm fiberglass rods with sensors
- Borehole: Hydraulic push created, ~18-20mm diameter
- Target Depth: 2-3 meters
- Soil Types: Sand, clay, mixed fill (bomb crater sites)
- Key Challenge: Holes may collapse before probe insertion
- Requirement: Good soil-to-probe contact for ERT measurements

---

## 1. Borehole Stability and Collapse Mechanisms

### 1.1 Why Boreholes Collapse

Borehole collapse occurs when the surrounding soil cannot support the vertical walls of the open excavation. The primary factors are:

**Soil Type Factors:**
- **Loose sands/gravels**: Lack cohesion, high internal friction angle but no binding forces. Particles immediately begin filling the void. In permeable sand and gravel strata, drilling fluids may infiltrate directly into the soil rather than forming a stabilizing filter cake.
- **Soft clays/silts**: Prone to "hole closure" - a narrowing time-dependent process where plastic flow under overburden pressure causes the hole to shrink gradually.
- **Mixed/disturbed fill**: Unpredictable behavior; may contain voids, debris, or unstable zones.

**Environmental Factors:**
- **Water table**: Saturated conditions increase instability, particularly in sands where flowing sand conditions can occur.
- **Overburden pressure**: Deeper holes experience greater collapse forces.
- **Time**: Creep behavior in clay can cause delayed failure even if initial stability appears good.

### 1.2 Time Window Before Collapse

Time-dependent stability varies significantly by soil type:

| Soil Type | Stability Characteristics | Time Window |
|-----------|--------------------------|-------------|
| Loose sand | Immediate to rapid collapse | Seconds to minutes |
| Saturated sand | Very rapid, flowing conditions | Seconds |
| Soft clay | Gradual closure/creep | Minutes to hours |
| Stiff clay | More stable, eventual shrinkage | Hours to days |
| Mixed fill | Highly variable | Unpredictable |

**Key Finding**: In unconsolidated materials typical of bomb crater sites, the collapse can be "gradual or sudden depending on soil conditions." Sandy soils are more prone to rapid collapse due to lack of cohesion.

### 1.3 Collapse Prevention Methods

**Drilling Fluid/Mud:**
- Bentonite slurry creates hydrostatic pressure against borehole walls
- Fluid must be denser than surrounding soil
- Forms "filter cake" of overlapping platelets that seal the formation

**Casing:**
- Temporary steel or plastic casing prevents collapse
- Can be advanced ahead of excavation or installed after drilling

**Hydrostatic Head:**
- Maintaining water level in hole above water table provides positive pressure
- Minimum 6 feet (2m) of head above water table recommended in geotechnical practice

---

## 2. Casing and Liner Options

### 2.1 Temporary Steel Casing

**Description**: Thick-walled steel tubes installed to support borehole walls, then withdrawn after probe/concrete placement.

**Typical Specifications:**
- Wall thickness: 0.325-0.500 inches (8-13mm) minimum
- Can be installed by vibratory hammer, oscillator, or push
- Must be cleaned between uses to reduce friction

**Advantages:**
- Very strong, can support deep holes
- Reusable
- Well-established method

**Disadvantages:**
- Heavy equipment required
- Casing must be extracted before setting
- Risk of probe displacement during extraction
- Overkill for 2-3m shallow installations

### 2.2 Thin-Wall PVC Tubes

**Description**: Lightweight plastic tubes providing temporary support.

**Properties:**
- Corrosion resistant
- Resistant to acids/alkalis in soil
- Available in flush-jointed designs
- Can be slotted or solid

**For HIRT Application:**
- Could use ~20-22mm OD PVC tube as temporary sleeve
- Insert probe, then withdraw tube
- Issue: friction during tube withdrawal may disturb probe position

### 2.3 Split Casing (Removable Seam)

**Description**: Casing fabricated with a longitudinal split seam joined by mechanical pins.

**Application Example**: I-95 Fuller Warren Bridge project used split-seam casing that could be expanded to facilitate removal after concrete placement.

**For HIRT Application:**
- Could design a split sleeve that opens laterally
- Allows probe to remain in place while sleeve is removed
- **Promising for HIRT**: Custom split sleeve in 20mm ID could work well

### 2.4 Biodegradable/Dissolvable Options

**Oil & Gas Industry**: Uses dissolvable alloys, plastics, and elastomers that dissolve after placement, eliminating retrieval step.

**Potential HIRT Applications:**
- Biodegradable HDPE sleeving exists (e.g., FieldTech sonic sleeving)
- Could use thin biodegradable sleeve that degrades over time
- **Consideration**: Degradation time must not interfere with measurements

### 2.5 Prepacked Screen Systems

**Description**: From monitoring well industry - screens with sand pre-packed between inner pipe and outer mesh.

**Components:**
- Inner slotted PVC pipe
- Stainless steel mesh exterior
- Sand packed in annular space
- Foam bridges and bentonite quick-seals available

**For HIRT**: Could adapt concept for pre-assembled probe-and-sleeve units.

---

## 3. Drilling Fluid / Mud Options

### 3.1 Bentonite Slurry

**Function**: Creates hydrostatic pressure and forms impermeable filter cake on borehole walls.

**Preparation:**
1. Pre-treat mix water with soda ash to pH 8.5-9 (neutralizes calcium)
2. Mix bentonite with water (20-30% solids by weight typical)
3. Can add PAC (poly anionic cellulose) polymer to reduce fluid loss

**Properties:**
- Thixotropic behavior - gels when stationary, flows when agitated
- Swells to form impermeable barrier
- Self-healing if disturbed

**Borehole Stability Mechanism:**
- Hydrostatic pressure counteracts soil pressure
- Filter cake prevents fluid loss into formation
- Overlapping bentonite platelets "shingle off" against formation

### 3.2 Polymer Drilling Fluids

**Function**: Alternative to bentonite, especially useful for specific conditions.

**Types:**
- PAC (poly anionic cellulose) - reduces fluid loss
- Synthetic polymers - various formulations

**Advantages over bentonite:**
- Cleaner
- Easier disposal
- May be preferred for loose sands

### 3.3 Impact on HIRT Measurements

**Critical Question**: Does drilling fluid interfere with electrical resistivity measurements?

**Considerations:**
- Bentonite has its own electrical properties (clay minerals are conductive)
- Residual mud on probe surface could affect contact resistance
- Mud cake between probe and soil creates measurement artifact

**Mitigation Options:**
1. Displace fluid during probe insertion (tremie method)
2. Use biodegradable polymer that breaks down
3. Design probe tip to scrape/displace mud as it enters
4. Accept some contamination - may be negligible for bulk soil measurements

### 3.4 Fluid Displacement During Insertion

**Tremie Method**: Insert probe with tremie pipe, pump fluid from bottom up as probe descends. The rising fluid level maintains stability while probe advances.

**Positive Displacement**: Probe designed to displace fluid as it enters, pushing mud up and out of the hole.

---

## 4. Push-In-Place vs. Pre-Drill Methods

### 4.1 Direct Push (No Pre-formed Hole)

**Concept**: Push the sensor probe directly into ground without drilling.

**Applicable Conditions:**
- Soft soils (SPT blow counts under 10)
- Clay, silt, loose sand
- Shallow depths (typically < 10m)

**Advantages:**
- No borehole collapse issue - soil remains in contact with probe
- Faster installation
- Minimal soil disturbance
- Natural soil-probe contact achieved

**Limitations:**
- Cannot penetrate dense/hard soils
- Fiberglass may not withstand push forces
- Probe electronics could be damaged by push stresses
- Limited to unconsolidated materials

**Push-In Piezometer Analogy**: These instruments have conical tips and are pushed directly into soft soils. The soil naturally collapses and seals around the instrument. "The ground conditions need to be relatively soft for push-in piezometers to be effective."

### 4.2 Sacrificial Tip Systems

**Concept**: Disposable cone tip leads the probe, stays in ground when probe is retrieved or remains permanently.

**Disposable Cone Penetrometer Tips:**
- Same angles/dimensions as standard cones
- Attach via friction adapter
- Abandoned in place after test
- Used when soils squeeze the cone and trap it

**For HIRT Application:**
- Could design sacrificial conical tip for probe
- Tip remains in ground, probe stays in place
- Useful if probe will be permanent installation

### 4.3 Hybrid Approach: Pre-Drill + Push

**Recommended Method from Piezometer Practice:**

1. Drill borehole to within 1.5-2m (5-6 feet) of target depth
2. Push instrument through final 1.5m into undisturbed soil
3. Soil naturally collapses and seals the boring

**Advantages:**
- Reduces push force required
- Gets probe into undisturbed soil below drilled zone
- Natural seal forms around instrument
- Well-proven in geotechnical practice

**For HIRT (2-3m depth):**
- Pre-drill to 1-1.5m
- Push probe through final 1-2m
- OR push entire depth if soil is soft enough

### 4.4 Dual-Tube Systems

**Description**: Outer casing advances with inner sampling tube. Outer tube prevents collapse while inner probe operates.

**Advantages:**
- Outer casing prevents borehole collapse
- Maintains hole alignment for multiple insertions
- "Dual tube soil sampling eliminates slough, casing the hole off as you proceed"

**For HIRT:**
- Could adapt dual-tube concept
- Outer tube (~22-24mm) advances with probe
- Withdraw outer tube after probe at depth
- Soil collapses onto probe

---

## 5. Tremie/Displacement Methods

### 5.1 Tremie Grouting Principle

**Definition**: A tremie pipe carries material (grout, sand, etc.) to designated depth in borehole, then is slowly withdrawn as material is placed.

**Key Principle**: Material is placed from bottom up, displacing any fluid and preventing voids or bridging.

### 5.2 Sensor Installation with Casing Withdrawal

**Standard Practice for Piezometers:**

1. Install instrument inside casing or hollow stem auger
2. Attach instrument to rigid tremie pipe
3. Place filter sand around instrument tip (if using sand pocket method)
4. Begin withdrawing casing/auger slowly
5. Place bentonite seal above instrument
6. Continue grout/backfill as casing withdraws
7. "Install piezometers one by one from the bottom up inside casing or hollow stem auger. Attach piezometers to rigid tremie pipe and leave in place while pulling casing or auger."

### 5.3 Two-Stage Grouting

**Process:**
1. First stage: Grout bottom 2-3m, attach tremie to casing at planned second stage level
2. Allow first stage to set (12+ hours for bentonite-cement)
3. Second stage: Continue grouting from first stage top
4. Repeat as needed

**For HIRT (2-3m depth)**: Single stage likely sufficient.

### 5.4 Concrete Pile Analogy

**Process:**
1. Install temporary casing to support borehole
2. Pour concrete using tremie pipe from bottom up
3. Slowly withdraw tremie as concrete level rises
4. Extract temporary casing before concrete sets
5. Casing must be removed smoothly to avoid dragging concrete

**Relevance to HIRT**: Same principle applies - insert probe, withdraw casing slowly while allowing soil/grout to fill annular space.

---

## 6. Hole Conditioning Methods

### 6.1 Water Injection for Stability

**Concept**: Maintain positive hydrostatic pressure by keeping borehole filled with water.

**Key Requirements:**
- Water level must be above groundwater table
- Minimum 6 feet (2m) head above water table recommended
- Continuous replenishment may be needed in permeable soils

**Limitations:**
- In very permeable sands, water may drain faster than can be supplied
- Does not provide same stability as drilling mud
- May cause formation softening in clays

### 6.2 Hole Conditioning Before Insertion

**Flushing:**
- Flush borehole with water to remove loose debris
- Verify hole is open to full depth
- Use weighted tape or probe to check for bridging

**For ERT Probes:**
- Water introduced to electrode holes improves electrical contact
- Common practice: wet the hole before inserting electrodes

### 6.3 Reaming for Clearance

**Purpose**: Enlarge hole slightly to ensure probe can pass obstructions.

**Considerations:**
- May disturb soil structure
- Creates larger annular gap to fill
- Only needed if probe encounters resistance during test insertion

---

## 7. Contact Assurance Methods

### 7.1 The Contact Problem

**Challenge**: For HIRT ERT measurements, probe electrodes must make good electrical contact with surrounding soil. An air gap or mud layer between probe and soil will degrade measurements.

**Factors Affecting Contact:**
- Annular space between probe and borehole wall
- Drilling fluid residue
- Soil moisture content
- Electrode surface condition

### 7.2 Natural Soil Collapse

**Concept**: Allow formation materials to naturally collapse around probe after casing/support is removed.

**Advantages:**
- Simple, no additional materials
- Achieves direct soil contact
- Standard practice for driven well points

**Limitations:**
- Not all soils will collapse uniformly
- May leave voids in cohesive soils
- "Some states prohibit allowing the formation to collapse around a well screen"

### 7.3 Bentonite Grout Backfill

**Standard Practice**: Fill annular space with cement-bentonite grout.

**Grout Properties:**
- Should match stiffness of surrounding soil (difficult in practice)
- Provides uniform contact around probe
- Seals against water migration between zones
- "Grout is preferred over sand or gravel backfill. Grout is dimensionally stable and also prevents unwanted migration of water between soil zones."

**Mix Design Considerations:**
- Water:cement ratio controls strength
- More water = weaker grout (closer to soft soil properties)
- Add bentonite for plasticity and dimensional stability
- "Mix water and cement first. Then add bentonite."

**For ERT Measurements:**
- Grout electrical properties may affect readings
- May need to calibrate for grout resistivity
- Could use conductive grout formulation

### 7.4 Sand Pack Method

**Traditional Piezometer Installation:**
1. Place clean fine sand 150mm below sensor tip position
2. Lower sensor into position
3. Add sand to 150mm above sensor
4. Place bentonite seal above sand
5. Continue backfill to surface

**For HIRT:**
- Sand would allow water/electrical conduction
- May not provide as intimate contact as grout or soil collapse
- Risk of sand bridging in narrow annulus

### 7.5 Fully Grouted Installation

**Modern Preferred Method**: Install sensor directly in borehole and completely backfill with cement-bentonite grout.

**Advantages:**
- Simpler and faster than sand pocket method
- Eliminates failures during sand pack placement
- Good for vibrating wire piezometers (essentially no-flow instruments)
- "Researchers firmly suggested to use piezometers in fully grouted boreholes"

**For HIRT**: ERT probes are also essentially no-flow instruments - may benefit from fully grouted installation.

### 7.6 Expandable/Inflatable Elements

**Inflatable Packers:**
- Rubber bladder expands when inflated with air/water/gas
- Can seal against irregular borehole walls
- Standard sizes from 2" to 30" diameter
- Can incorporate sensor feedthroughs

**For HIRT Application:**
- Could design probe with small expandable element
- Inflate after placement to press electrodes against borehole wall
- Complexity and cost may be prohibitive

### 7.7 Pre-Saturation of Sensors

**Important Step**: For piezometers, air must be removed from porous stone filter before installation.

**Method:**
1. Remove porous filter from sensor
2. Submerge sensor completely in clean water
3. Reattach filter underwater
4. Keep submerged until installation

**For HIRT**: Similar principle - ensure electrodes are wetted before insertion to improve initial contact.

---

## 8. Lessons from Similar Applications

### 8.1 Piezometer Installation

**Key Takeaways:**
- Push-in piezometers work in soft soils (SPT < 10)
- Fully grouted method is now preferred over sand pocket
- Hybrid approach: drill near target, push final distance
- Natural soil collapse can provide adequate seal

### 8.2 Inclinometer Casing Installation

**Key Takeaways:**
- Grout backfill preferred for good soil-casing coupling
- Grout stiffness should approximate soil stiffness
- Buoyancy forces act on casing during grouting (8-18 kg per 3m)
- In soft soils, instrument disturbance factor is significant
- Keep hole vertical (within 1 degree)

### 8.3 Temperature Sensor Strings (Thermistor Cables)

**Key Takeaways:**
- Multiple sensors along single cable
- Can be installed in wet holes below water table
- Important to seal head of borehole to prevent air circulation
- Grouting around sensors is standard
- "It is important to properly seal the head of the borehole and the space between multiple sensors to avoid air circulation"

### 8.4 ERT Electrode Installation

**Key Takeaways:**
- Water introduced to electrode holes improves electrical contact
- Electrodes often grouted permanently in boreholes
- Stainless steel electrodes can remain in ground without environmental harm
- Borehole ERT uses electrode arrays with isolation packers
- Smart casing designs use capacitively coupled electrodes (not prone to corrosion)

### 8.5 Direct Push Soil Sampling

**Key Takeaways:**
- Dual-tube systems prevent collapse during sampling
- Single-tube systems faster but prone to hole collapse
- Sealed piston prevents soil intrusion during advancement
- In saturated sands, add water to outer casing to prevent heave

### 8.6 Monitoring Well Installation

**Key Takeaways:**
- Prepacked screens guarantee filter sand placement
- Foam bridges and bentonite quick-seals available as pre-made components
- In flowing sand conditions, fill hollow stem with water to balance pressure

---

## 9. Recommendations for HIRT Probe Installation

### 9.1 Recommended Primary Method: Cased Hole with Withdrawal

**Procedure:**
1. Create borehole using hydraulic push to full depth (2-3m)
2. Immediately insert thin-wall casing/sleeve (20-22mm OD) to stabilize hole
3. Lower HIRT probe into cased hole
4. Slowly withdraw casing while maintaining probe position
5. Allow natural soil collapse onto probe (for sand/loose fill)
6. OR pour thin bentonite-cement grout as casing withdraws (for clay/mixed soils)

**Why This Works:**
- Addresses collapse problem with temporary support
- Achieves good soil contact through collapse or grout
- Proven method from geotechnical industry
- Suitable for 2-3m shallow depth

### 9.2 Alternative Method: Direct Push (Soft Soils Only)

**When to Use:**
- Very soft clay or silt
- Loose, saturated sand
- When minimal soil disturbance is critical

**Procedure:**
1. Design probe with conical tip
2. Push probe directly to depth using hydraulic press
3. Soil naturally seals around probe

**Limitations:**
- May not work in all bomb crater fill materials
- Risk of probe damage from push forces
- Fiberglass strength may be limiting factor

### 9.3 Alternative Method: Hybrid Pre-Drill + Push

**When to Use:**
- Mixed or variable soil conditions
- When top layers are harder than deeper soil

**Procedure:**
1. Pre-drill to 1-1.5m depth
2. Install casing to prevent upper collapse
3. Push probe through final 1-1.5m into undisturbed soil
4. Soil collapses around probe

### 9.4 Contact Assurance Recommendations

**For Sand/Granular Soils:**
- Natural collapse usually sufficient
- May add water to improve contact conductivity
- Develop probe surface area for electrode contact

**For Clay Soils:**
- Fully grouted installation recommended
- Use weak cement-bentonite grout (high water content)
- Grout provides uniform contact

**For Variable/Mixed Fill:**
- Consider thin bentonite slurry backfill
- Allows some soil collapse while filling voids
- Provides consistent contact in variable conditions

### 9.5 Equipment Recommendations

**Temporary Casing Options:**
1. **Thin-wall PVC pipe** (~20mm ID, 22mm OD) - inexpensive, disposable
2. **Split sleeve** (custom fabricated) - opens for removal without disturbing probe
3. **Metal tube with extraction tool** - reusable, requires smooth interior

**Grout Materials:**
- Cement-bentonite mix (1:1 cement:bentonite by weight, high water ratio)
- Or pure bentonite pellets poured and hydrated in place

**Support Tools:**
- Probe guide to center probe in casing
- Casing extractor/withdrawal tool
- Tremie pipe for grout placement

---

## 10. Summary Comparison Table

| Method | Soil Types | Complexity | Contact Quality | Cost | Notes |
|--------|------------|------------|-----------------|------|-------|
| Direct push (no pre-hole) | Soft only | Low | Excellent | Low | Requires probe tip design |
| Open hole (no support) | Stable clay only | Low | Variable | Low | High collapse risk |
| Cased + withdrawal | All | Medium | Good-Excellent | Medium | Recommended primary method |
| Cased + grouted | All | Medium-High | Excellent | Medium | Best for clay/mixed |
| Drilling fluid + displacement | All | High | Variable | Medium | May affect ERT |
| Split casing | All | Medium | Good | Medium | Custom fabrication needed |
| Dual-tube | All | High | Good | High | Overkill for shallow holes |

---

## Sources

### Borehole Stability and Casing
- [Temporary Casing in Drilling: Best Practices](https://westernequipmentsolutions.com/best-practices-for-using-temporary-casing-during-drilling-operations/)
- [Drilled Shafts Guide: Casings & Liners](https://pilebuck.com/drilled-shafts-guide-casings-liners/)
- [A Complete Guide To Pile Drilling in Loose Soils](https://www.jeffreymachine.com/blog/a-complete-guide-to-pile-drilling-loose-soils)
- [Borehole Collapse - How to Fix It Effectively](https://bonvicdrilling.com/borehole-collapse-heres-how-to-fix-it-effectively/)
- [Chapter 6 - Casings and Liners (FHWA)](https://pilebuck.com/drilled-shafts-construction-procedures-fhwa/chapter-6-casings-liners/)
- [How to Prevent Borehole Collapse When Drilling](https://www.lonestardrills.com/prevent-borehole-collapse/)

### Piezometer Installation
- [Piezometers in Fully Grouted Boreholes (Mikkelsen & Green, 2003)](https://www.geosense.com/wp-content/uploads/2021/04/Piezometers-in-fully-grouted-boreholes-Mikkelsen-Green-2003..pdf)
- [Vibrating Wire Piezometer Installation Procedure](https://www.encardio.com/blog/vibrating-wire-piezometer-installation-procedure-in-a-borehole)
- [What is Piezometers: Types, Functions, & How it Works](https://www.encardio.com/blog/piezometers-types-functions-how-it-works)
- [Geokon Model 4500 Piezometer Installation](https://www.geokon.com/content/manuals/4500/topics/04_installation.htm)
- [Solinst Standpipe Piezometers](https://www.solinst.com/products/direct-push-equipment/601-standpipe-piezometers/)

### Drilling Fluids and Bentonite
- [How to Create Borehole Stability Using Drilling Fluid](https://www.thedriller.com/articles/89844-how-to-create-borehole-stability-using-drilling-fluid-or-slurry)
- [Geotechnical Drilling - Bentonite and Cement Grout](https://www.thedriller.com/articles/89600-geotechnical-drilling-can-benefit-from-bentonite-cement-grout)
- [Bentonite in Drilling: Enhancing Efficiency](https://iranbentoniteco.com/bentonite-in-drilling/)
- [Grouting with Bentonite - Water Well Journal](https://waterwelljournal.com/grouting-with-bentonite/)

### Inclinometer Installation
- [Geokon Model 6650 Inclinometer Casing Manual](https://www.geokon.com/content/manuals/6650-Inclinometer-Casing-Manual.pdf)
- [Digital Inclinometer Installation in a Borehole](https://www.encardio.com/blog/digital-inclinometer-installation-in-a-borehole)
- [Guide to Instrumentation - Durham Geo](https://durhamgeo.com/pdf/documents/course%20material/guide-to-instrumentation.pdf)

### Direct Push Technology
- [Direct Push vs Hollow Stem Auger](https://geoprobe.com/articles/building-business-success-direct-push-vs-hollow-stem-auger)
- [Direct Push Sampling - Enviro Wiki](https://www.enviro.wiki/index.php?title=Direct_Push_Sampling)
- [Direct-Push Platforms - CLU-IN](https://clu-in.org/characterization/technologies/dpp.cfm)
- [Soil Sampling with Direct-push Technology](https://www.thedriller.com/articles/87499-soil-sampling-with-direct-push-technology)

### Monitoring Wells
- [Direct-push Monitoring Well Installation](https://www.thedriller.com/articles/88164-direct-push-monitoring-well-installation)
- [Design and Installation of Monitoring Wells - EPA](https://www.epa.gov/sites/default/files/2016-01/documents/design_and_installation_of_monitoring_wells.pdf)
- [Geoprobe Prepacked Screen Monitoring Wells](https://geoprobe.com/tooling/prepacked-screen-monitoring-wells)

### Temperature Sensors and Thermistor Strings
- [A Guide to Thermistor Strings - BeadedStream](https://www.beadedstream.com/a-guide-to-thermistor-strings/)
- [Thermocouple Temperature Sensor - Soil Instruments](https://www.soilinstruments.com/products/temperature-sensors/thermocouple-temperature-sensor/)

### ERT/Resistivity
- [Electrical Resistivity Tomography - CLU-IN](https://clu-in.org/characterization/technologies/default2.focus/sec/Geophysical_Methods/cat/Electrical_Resistivity_Tomography/)
- [Electrical Resistivity Tomography - Wikipedia](https://en.wikipedia.org/wiki/Electrical_resistivity_tomography)

### Inflatable Packers
- [Baski Inflatable Packers](https://www.baski.com/Packers.aspx)
- [The Use of Inflatable Packers - Water Well Journal](https://waterwelljournal.com/the-use-of-inflatable-packers/)
- [Solinst Low Pressure Packers](https://www.solinst.com/products/groundwater-samplers/800-low-pressure-packers/)

### Fiberglass Probes
- [Fiberglass Probe Rod - Humboldt](https://www.humboldtmfg.com/probe-rod-fiberglass.html)
- [Non-Conductive Fiberglass Soil Probe Rod](https://www.ganglongfiberglass.com/fiberglass-soil-probe-rod/)

---

*Research compiled: January 2026*
*For HIRT Project: Humanitarian Impact Resonance Technology*
