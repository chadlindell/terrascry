# Real-Time UXO Detection During Soil Probe Insertion

## Research Summary

**Date:** 2026-01-19
**Context:** HIRT system pushes 16mm probes to 2-3m depth at WWII bomb crater sites
**Requirement:** Detect ferrous/metallic objects WHILE pushing to avoid striking UXO
**Push Speed:** ~20mm/second (standard CPT rate)

---

## Executive Summary

Real-time UXO detection during probe insertion is an established practice in the geotechnical industry, primarily using magnetometer-equipped CPT cones. Commercial systems exist from multiple manufacturers (Fugro, Gouda Geo, Royal Eijkelkamp, A.P. van den Berg) that can detect a 50kg WWII bomb at 2 meters distance. The primary challenge for HIRT integration is the miniaturization required to fit detection capabilities into a 16mm probe system, compared to the industry-standard 44mm diameter magnetometer cones.

**Key Findings:**
- Magnetometry is the dominant technology for real-time UXO detection during pushing
- Commercial CPT magnetometer cones detect 50kg bombs at 2m radius, clearing 4m diameter cylinders
- Detection range follows inverse-cube law: 250kg bomb detectable at ~3-4m, 500kg at ~5-6m
- Automatic abort systems exist (Fugro) that stop on magnetic gradient or tip resistance changes
- Miniaturization to 16mm is challenging but potentially feasible with modern fluxgate sensors
- HIRT's MIT coils may create interference requiring careful shielding/sequencing

---

## 1. Magnetometer-Based Detection

### 1.1 Fluxgate Magnetometers for UXO Detection

Fluxgate magnetometers are the primary sensor technology used in CPT-based UXO detection systems. They measure variations in the Earth's magnetic field caused by ferrous objects.

**Operating Principle:**
- A fluxgate sensor measures the intensity and variations of the magnetic field in three directions (XYZ)
- Ferrous materials create detectable distortions in the local magnetic field
- The magnetic anomaly signal is relayed to a geophysicist in real-time
- Built-in inclinometers (typically +/-25 degrees, accuracy <0.1 degrees) allow determination of object orientation

**Key Specifications (Industry Standard - Gouda Geo MagCone):**
- Probe diameter: 44mm (15 cm2 cone)
- Module length: 350mm (excluding cone tip)
- Sensitivity: 1 nT (nanotesla)
- Measurement output: nT in three axes
- Maximum penetration speed: 20mm/sec (with CPT cone), 50mm/sec (standalone)
- Lateral detection range: up to 2m (clears 4m diameter cylinder)

**Sources:**
- [Gouda Geo Magnetometer Cone](https://gouda-geo.com/product/magnetometer-cone-magcone)
- [Lankelma Magnetometer Cone](https://www.lankelma.com/cpt-services/testing-services/magnetometer-cone/)

### 1.2 Detection Range vs. Object Size

The magnetic field strength diminishes with the **cube of distance**. This fundamental physics limits detection range but allows prediction of detection capabilities.

**General Rule of Thumb:**
- 1 ton (1000 kg) of steel gives 1 nT anomaly at 30m (100 ft)
- Every time distance halves, detectable mass reduces to 1/8th
- 250 lbs (100kg) detectable at 15m (50 ft)
- 30 lbs (15kg) detectable at 8m (25 feet)
- 4 lbs (2kg) detectable at 4m (12 feet)

**Practical Detection Distances for WWII Bombs:**

| Ordnance Type | Weight | Estimated Detection Range |
|---------------|--------|---------------------------|
| 50kg WWII bomb | 50 kg | 2m (confirmed by multiple commercial systems) |
| SC 250 German bomb | 250 kg | 3-4m (estimated) |
| SC 500 German bomb | 500 kg | 5-6m (estimated) |
| 500lb MC aerial bomb | ~227 kg | >7m (UAV survey at 2.5nT noise floor) |
| SC 1000 "Hermann" | >1000 kg | >7m |
| Hand grenade (F1) | ~0.6 kg | 0.5m |

**Historical Detection (1940s Development):**
- 500 kg bomb detectable at 30 feet (9m) depth from surface
- 250 kg bomb detectable at 25 feet (7.6m) depth
- Anti-aircraft shell detectable at 10 feet (3m)

**Sources:**
- [Geometrics - How Far Can a Magnetometer See](https://www.geometrics.com/support/how-far-can-a-magnetometer-see/)
- [SPH Engineering - UXO Detection Distances](https://www.sphengineering.com/news/estimates-of-various-uxo-unexploded-ordnance-maximum-detection-distances-using-magnetometers)

### 1.3 Commercial Magnetometer CPT Probes

**Fugro S-Magnetometer Cone:**
- Tri-axial magnetometer behind standard CPT cone
- Measures geotechnical parameters simultaneously
- Features **automated stop switch** that identifies:
  - Sudden predefined increase in magnetic gradient
  - Sudden increase in tip resistance
- Automatically stops CPT before contact with unsafe objects
- Push speed: 2 cm/s (20mm/s)
- Real-time data relay to geophysicist

**A.P. van den Berg Icone Magneto Module:**
- Digital CPT system with click-on magnetometer module
- Works with Icone data acquisition system
- Non-magnetic housing with standard tapered thread
- Can be housed behind dummy tip or 15 cm2 CPT cone
- High-grade non-magnetic steel construction

**Royal Eijkelkamp Geomagnetic Module:**
- Fluxgate sensor in XYZ direction
- Can determine position and orientation of detected objects
- Detection range: 2m lateral (4m diameter clearance)
- Depths exceeding 20m in soft sediments
- Real-time monitoring capabilities

**Gouda Geo MagCone:**
- 44mm diameter, 350mm module length
- 1 nT sensitivity
- Tri-axial measurement transmitted separately to surface
- Compatible with CPT Explorer data acquisition software
- Can combine with piezocone for simultaneous CPTu data

**Sources:**
- [Fugro UXO Survey Case Study](https://www.fugro.com/expertise/case-studies/superior-data-uxo-survey-important-paris-site-fugro)
- [Royal Eijkelkamp Magnetometer Module](https://www.royaleijkelkamp.com/products/drilling-cpt/cpt/modules/magnetometer-module/)
- [A.P. van den Berg CPT Penetrometers](https://www.apvandenberg.com/onshore-cone-penetration-testing/cpt-penetrometers)

### 1.4 Miniaturization Possibilities

**Current Smallest Commercial Fluxgate Sensors:**

| Sensor | Dimensions | Noise Floor | Notes |
|--------|------------|-------------|-------|
| Low-cost race-track probe | 10mm dia x 30mm | <0.1 nT RMS | Research prototype |
| Bartington Spacemag-Lite | 20x20x20mm | N/A | Spacecraft applications |
| CubeSat fluxgate | 36x32x28mm | 150-200 pT/√Hz at 1Hz | Space-qualified |
| CMOS microfluxgate | 2.5x2.5mm chip | 2.6 nT/√Hz at 1Hz | Silicon-based |

**Challenges for 16mm HIRT Probe Integration:**
1. **Diameter constraint:** Standard magnetometer modules are 44mm; 16mm is 2.75x smaller
2. **Sensor miniaturization:** Smallest commercial tri-axial fluxgates are ~10-20mm
3. **Electronics:** Data acquisition and transmission circuitry needs space
4. **Cable routing:** Signal cables must fit within or alongside probe rod

**Potential Solutions:**
1. **External magnetometer on push rod:** Mount magnetometer module above probe tip, outside soil
2. **Microfluxgate MEMS:** Use silicon-based microfluxgate (2.5mm chip) with external electronics
3. **Single-axis sensor:** Use single fluxgate aligned with push direction (reduced capability)
4. **Tandem operation:** Push separate magnetometer probe ahead of HIRT probe

**Sources:**
- [MDPI - Low-Cost Small-Size Fluxgate Sensor](https://www.mdpi.com/1424-8220/21/19/6598)
- [AGU - Miniature Fluxgate for CubeSat](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JA023147)

---

## 2. Electromagnetic Induction (EMI)

### 2.1 EMI Sensors for Metal Detection

EMI (Electromagnetic Induction) sensors, commonly known as metal detectors, detect both ferrous and non-ferrous metals through eddy current effects.

**Operating Principle:**
- Transmitter coil generates primary alternating magnetic field
- Conductive objects generate eddy currents
- Secondary magnetic field from eddy currents detected by receiver coil
- Detection depends on conductivity, permeability, size, and shape of target

**Advantages over Magnetometry:**
- Detects non-ferrous metals (aluminum, brass, copper)
- Can potentially discriminate object type via spectral analysis
- Less affected by permanent magnetization variations

**Disadvantages:**
- Detection range falls off as **inverse 6th power** (vs cube for magnetometry)
- Shorter detection range for same-sized objects
- More sensitive to soil conductivity variations

### 2.2 Integration with HIRT Probe Tip

**Size Considerations:**
- Commercial EMI coils (DUALEM, EM38) are designed for surface surveys
- Miniaturization to 16mm diameter is theoretically possible
- Would require custom coil winding and electronics

**Interference with HIRT MIT Coils:**
- HIRT's Magnetic Induction Tomography uses similar electromagnetic principles
- Potential for cross-talk and interference
- Solutions:
  - Time-division multiplexing (alternate MIT and UXO detection)
  - Frequency separation (different operating frequencies)
  - Spatial separation (UXO coils at tip, MIT coils further back)

**Technical Challenges:**
- Primary field magnitude much larger than secondary field from targets
- Inter-coil capacitive coupling requires shielding
- Calibration errors and drift need field compensation

**Sources:**
- [MDPI - Modular EMI Sensor Design](https://pmc.ncbi.nlm.nih.gov/articles/PMC11244430/)
- [IEEE - EMI for Landmine Detection](https://ieeexplore.ieee.org/document/4422720)

### 2.3 Time-Domain Electromagnetic (TDEM)

TDEM is an advanced EMI technique with improved discrimination capabilities.

**Operating Principle:**
- Transmitter coil pulses primary field
- After field turns off, eddy currents in targets decay
- Decay characteristics reveal object size, depth, and composition
- Ferrous objects (high permeability) have distinctive signatures

**UXO Detection Performance:**
- >90% detection probability at controlled test sites
- Can discriminate UXO from scrap metal (shape information)
- MetalMapper 2x2 system identifies UXO among debris

**Limitations for HIRT:**
- Systems designed for surface/near-surface surveys
- Miniaturization to probe-mounted system not commercially available
- Processing algorithms optimized for static measurements

**Sources:**
- [OSTI - Time Domain EM Metal Detectors](https://www.osti.gov/biblio/329487)
- [Geometrics UXO Detection](https://www.geometrics.com/solutions/uxo-detection/)

---

## 3. Electrical Resistivity Sensing

### 3.1 Can ERT Detect Metal Objects?

HIRT already incorporates ERT capability. The question is whether this can be leveraged for UXO detection.

**Metal Detection with ERT:**
- Metallic objects show very low resistivity (<0.1 Ohm-m)
- Detectable contrast against most soil types (>3000 Ohm-m for non-metallic)
- Research has demonstrated ERT detection of buried metallic targets

**Significant Limitations:**
1. **Polarization effects:** Metallic pipes/objects block charge carriers at boundaries
2. **Poor resolution:** Shapes, sizes, and geometries are "smeared" in images
3. **Electrode spacing requirements:** 0.25-0.50m spacing needed for reasonable resolution
4. **Interference susceptibility:** Groundwater, contamination, nearby utilities affect results
5. **Not designed for metal detection:** ERT optimized for geological characterization

**Assessment for HIRT UXO Detection:**
- **NOT RECOMMENDED** as primary UXO detection method
- ERT data could provide supplementary information
- Sudden conductivity changes might indicate metal, but unreliable
- Magnetometry remains far superior for ferrous UXO

**Alternative Use:**
- If goal is trench/contamination characterization, ERT useful
- For metal detection, magnetometers or TDEM recommended

**Sources:**
- [Springer - ERT for Shallow Underground Targets](https://link.springer.com/article/10.1007/s44288-024-00058-6)
- [ResearchGate - ERT for Metallic Pipe Mapping](https://www.researchgate.net/post/What-is-the-effectiveness-of-using-Electrical-Resistivity-protocol-in-mapping-burried-metallic-pipe)

---

## 4. Mechanical Sensing

### 4.1 Force Feedback - Tip Resistance Monitoring

CPT systems routinely monitor tip resistance (qc) as a geotechnical parameter. This can also serve as a safety indicator.

**Standard CPT Tip Resistance:**
- Measured by force on cone tip divided by projected area
- Values range from <1 MPa (soft clay) to >50 MPa (dense sand/rock)
- Metal objects would show sudden, extreme resistance increase

**Fugro Automatic Abort System:**
- Monitors both magnetic gradient AND tip resistance
- Automatically stops CPT when predefined thresholds exceeded
- "Additional system can identify a sudden predefined increase in either the magnetic gradient or tip resistance and automatically stops the CPT before coming into contact with a potentially unsafe object"

**Recommended Abort Strategy for HIRT:**
1. Monitor push force continuously
2. Calculate rate of force change (dF/dt)
3. Set thresholds for:
   - Absolute force (e.g., >500N sudden increase)
   - Rate of change (e.g., >100N/s)
4. Automatic stop + operator alert when thresholds exceeded

### 4.2 Distinguishing Rock from Metal by Force Signature

**Research Findings:**
- Vibration and acoustic signals during drilling vary between rock types
- Frequency analysis can identify lithology (limestone, sandstone, coal, mudstone)
- However, rock vs. metal discrimination through force alone is **unreliable**

**Challenges:**
- Rocks and metal can have overlapping penetration resistance
- Natural boulders may have similar hardness to ordnance casings
- Speed of response insufficient for safety-critical detection

**Recommendation:**
- Force monitoring useful as **backup/confirmation** system
- Should NOT be primary UXO detection method
- Combine with magnetometry for comprehensive protection

### 4.3 Acoustic/Vibration Signatures

**Research on Drilling Signatures:**
- Acoustic emission (AE) and vibration analysis used in rock drilling
- Frequency bands 0-25,000 Hz can characterize lithology
- Dominant frequencies 4,000-9,000 Hz correlate with rock properties

**Application to UXO Detection:**
- Metal impact would produce distinctive acoustic signature
- However, detection only occurs AT contact (too late)
- Useful for post-contact identification, not pre-contact warning

**Sources:**
- [ScienceDirect - Rock Drilling Vibration Signals](https://www.sciencedirect.com/science/article/abs/pii/S0003682X1731174X)
- [Springer - Drilling Noise for Rock Layer ID](https://link.springer.com/article/10.1007/s10064-025-04139-9)

---

## 5. Look-Ahead Technologies

### 5.1 Ground Penetrating Radar (GPR)

**Surface GPR:**
- Effective for shallow subsurface imaging
- Detection of metal objects possible
- Resolution limited by frequency/wavelength trade-off

**Borehole GPR:**
- Exists for mining and tunnel applications
- Antenna sizes typically too large for small-diameter probes
- Range in soft ground limited to ~40m ahead (tunnel-scale systems)

**Miniaturization Challenges:**
- GPR antennas need to be proportional to wavelength
- High-frequency (short wavelength) = short range, small antenna
- Low-frequency (long wavelength) = longer range, large antenna
- 16mm diameter severely constrains antenna design

**Assessment:**
- **NOT FEASIBLE** for 16mm HIRT probe integration
- Could be used as pre-survey technique (surface GPR before probing)

### 5.2 Seismic/Acoustic Look-Ahead

**Tunnel Boring Machine Systems:**
- Sonic Soft Ground Probing (SSP): 40m look-ahead, 0.5-1m resolution
- Tunnel Seismic While Drilling (TSWD): Uses TBM vibration as source
- Detection ranges of 10-42m demonstrated in EPB tunneling

**Application to Small Probes:**
- Technology designed for large-scale tunnel boring (meters diameter)
- Requires geophone arrays and sophisticated signal processing
- Not practical for 16mm probe at 20mm/s push rate

**Assessment:**
- **NOT FEASIBLE** for real-time HIRT integration
- Principles could inform future research

**Sources:**
- [Springer - Seismic Imaging Ahead of TBM](https://link.springer.com/chapter/10.1007/978-1-4020-2402-3_49)
- [MDPI - Tunnel Seismic Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC6427705/)

---

## 6. Integration Considerations

### 6.1 Size Constraints (16mm Probe)

**Challenge:**
- Industry-standard magnetometer cones are 44mm diameter
- HIRT probes are 16mm diameter (2.75x smaller)
- Available cross-sectional area: ~200 mm2 vs ~1500 mm2

**Options:**

| Approach | Feasibility | Detection Capability | Notes |
|----------|-------------|---------------------|-------|
| Magnetometer in push rod (above ground) | High | Limited (only near-surface) | Simple, but loses look-ahead at depth |
| Miniature fluxgate at tip | Medium | Good if achievable | Requires custom 10mm sensor package |
| Magnetometer in larger adapter sleeve | High | Standard | 44mm sleeve around 16mm rod in upper section |
| External parallel magnetometer probe | High | Standard | Push two probes: one mag, one HIRT |
| Pre-push magnetometer clearing | High | Standard | Clear column with mag probe before HIRT |

**Recommended Approach:**
For maximum safety, consider a **two-pass system**:
1. First pass: Standard 44mm magnetometer cone to clear column to target depth
2. Second pass: 16mm HIRT probe in pre-cleared position

### 6.2 Real-Time Data Transmission

**Requirements:**
- Sub-second latency for abort decision
- Continuous magnetic field data (3 axes)
- Force/resistance data from push system
- Clear operator display and alerts

**Commercial Systems:**
- CPT data acquisition runs at 10+ samples/second
- Wireless systems available (A.P. van den Berg)
- Real-time display with alarm thresholds

**HIRT Integration:**
- Add magnetometer data channel to existing MIT/ERT data stream
- Implement automated threshold monitoring
- Provide clear visual/audible abort alerts
- Consider automatic push-stop on threshold breach

### 6.3 Automatic Abort Systems

**Fugro System Features:**
- Monitors magnetic gradient (rate of change)
- Monitors tip resistance
- Predefined thresholds for automatic stop
- Stops BEFORE contact with hazardous object

**Recommended HIRT Abort Criteria:**
1. **Magnetic gradient threshold:** >X nT/cm rate of change
2. **Tip resistance threshold:** >Y MPa sudden increase
3. **Operator override:** Manual abort button
4. **Depth threshold:** Stop at calculated maximum bomb penetration depth

### 6.4 False Positive Management

**The False Positive Problem:**
- Up to 75% of magnetometer anomalies are false alarms (scrap, debris, natural features)
- Only ~1% of detected metallic objects are actual UXO
- In marine UXO clearance, only 6% of investigated targets are UXO

**Mitigation Strategies:**
1. **Magnetic signature analysis:** Use tri-axial data to model source characteristics
2. **Signal amplitude:** Larger anomalies more likely to be significant UXO
3. **Site-specific calibration:** Understand local ferrous debris background
4. **Conservative thresholds:** Accept false positives to avoid false negatives
5. **Investigation protocol:** Develop procedure for anomaly verification

**Recommended HIRT Approach:**
- Set conservative thresholds (stop on any significant anomaly)
- Accept that false positives will cause work stoppages
- Safety priority: never push through an unverified anomaly

**Sources:**
- [Geometrics - UXO Detection](https://www.geometrics.com/solutions/uxo-detection/)
- [Zetica UXO - Anomaly Analysis](https://zeticauxo.com/news/is-that-anomaly-really-a-uxb/)

---

## 7. Industry Practice

### 7.1 Current UXO Industry Methods

**Pre-Survey (Non-Intrusive):**
- Desktop risk assessment (historical bombing records, military history)
- Surface magnetometer survey (limited depth, <3m practical)
- Surface GPR survey
- Metal detector survey

**Intrusive Survey Methods:**

| Method | Description | Ground Conditions | Depth |
|--------|-------------|-------------------|-------|
| MagCone (BXP) | Magnetometer pushed via CPT rig | Soft, cohesive | >20m |
| MagDrill (TFG) | Magnetometer lowered into drilled borehole | Any (including hard) | 30m |
| Down-hole mag | Probe lowered at intervals during drilling | Any | >30m |

**MagCone Protocol:**
1. Push magnetometer cone at controlled rate (20mm/s)
2. Monitor magnetic readings in real-time
3. Stop immediately if anomaly detected
4. Clear vertical cylinder (4m diameter) per push
5. Continue to maximum bomb penetration depth (typically 12m)

**MagDrill Protocol:**
1. Drill 1m advance
2. Lower magnetometer probe to bottom of hole
3. Test for magnetic signatures within 1m radius
4. If clear, advance next 1m
5. Repeat to maximum bomb penetration depth

### 7.2 CPT Contractor Practices at UXO Sites

**Crossrail Project (UK) - Best Practice Example:**
- All intrusive ground investigations preceded by mag-cone surveys
- UXO risk assessment procedure developed jointly by client and engineers
- External UXO consultancy review of procedures
- Following CIRIA C681 guidance

**Standard Protocol:**
1. Preliminary UXO Risk Assessment (desktop study)
2. Detailed UXO Risk Assessment (if risk identified)
3. Intrusive mag survey to clear probe/borehole positions
4. Geotechnical investigation only after clearance
5. UXO engineer on-site during intrusive works

**Regulatory Framework (UK):**
- No specific UXO legislation, but covered by:
  - Health and Safety at Work Act 1974
  - Construction Design and Management Regulations 2015
- CIRIA C681 provides industry guidance

### 7.3 Pre-Survey vs. Real-Time Detection Trade-offs

| Aspect | Pre-Survey (Surface) | Real-Time (Intrusive) |
|--------|---------------------|----------------------|
| Depth capability | Limited (<3m typically) | Unlimited (to refusal) |
| Coverage | Wide area | Point-specific (4m cylinder) |
| Speed | Fast | Slower (requires pushing) |
| Accuracy at depth | Poor | Excellent |
| Equipment | Standard survey gear | Specialized CPT rig |
| Cost | Lower | Higher |

**Recommendation for HIRT at WWII Sites:**
1. **Always** conduct pre-survey desktop risk assessment
2. If UXO risk identified, conduct surface magnetometer survey
3. Use intrusive magnetometer clearing for all probe positions
4. Consider MagCone pre-clearing before HIRT probe insertion
5. Maintain real-time monitoring during HIRT push as backup

**Sources:**
- [Brimstone UXO - Intrusive Survey](https://www.brimstoneuxo.com/survey-uxo/intrusive-uxo-survey/)
- [Crossrail Learning Legacy - UXO Best Practice](https://learninglegacy.crossrail.co.uk/documents/crossrail-uxo-risk-assessment-pre-empting-best-practice/)
- [CIRIA C681 Guidance](https://www.brimstoneuxo.com/uxo-news/the-science-of-magnetometry/)

---

## 8. WWII Bomb Penetration Context

### 8.1 Bomb Types and Weights

**German SC Series (Sprengbombe Cylindrisch):**
- SC 50: 50 kg
- SC 250: 250 kg (117cm long, 36.8cm diameter)
- SC 500: 500 kg
- SC 1000 "Hermann": >1000 kg

**Allied Bombs:**
- 250 lb GP: ~113 kg
- 500 lb MC: ~227 kg
- 1000 lb GP: ~454 kg

### 8.2 Penetration Depths

**Theoretical Maximum (1000kg in clay):**
- Vertical: 25m
- Horizontal drift: 8m (J-curve trajectory)

**Observed Maximums:**
- 1000kg bomb: 12.5m actual observed maximum
- Only 1% of bombs >50kg penetrated more than 9m

**Practical Depths by Soil Type:**
- 500 lb (227kg) in wet clay: average 4.5m depth, 1.5m drift
- UXB can create crater >3m diameter, 1.5m deep

**Long Drift Phenomenon:**
- In Oranienburg: average bomb location 12m from fall site
- Bombs can travel horizontally through soil

### 8.3 Implications for HIRT (2-3m Target Depth)

**Risk Assessment:**
- HIRT target depth (2-3m) is within typical bomb penetration zone
- Most bombs found at depths <9m
- 2-3m depth is well within high-risk zone

**Detection Considerations:**
- At 2m ahead look-ahead, can detect 50kg bomb before reaching it
- Larger bombs (250kg+) detectable at 3-5m ahead
- Adequate warning distance at 20mm/s push rate (2m = 100 seconds warning)

**Sources:**
- [Wikipedia - SC250 Bomb](https://en.wikipedia.org/wiki/SC250_bomb)
- [Wikipedia - SC 500 Bomb](https://en.wikipedia.org/wiki/SC_500_bomb)
- [Smithsonian - Unexploded Bombs in Germany](https://www.smithsonianmag.com/history/seventy-years-world-war-two-thousands-tons-unexploded-bombs-germany-180957680/)

---

## 9. Recommendations for HIRT System

### 9.1 Primary Detection Method: Magnetometry

**Recommendation:** Integrate fluxgate magnetometer capability using one of these approaches:

**Option A: Two-Pass System (Highest Safety)**
1. First push: Standard 44mm magnetometer cone to clear column
2. Second push: 16mm HIRT probe in pre-cleared position
- Pros: Uses proven technology, maximum safety
- Cons: Doubles insertion time, requires two probe types

**Option B: Push-Rod Mounted Magnetometer**
1. Mount miniature tri-axial fluxgate on push rod above soil surface
2. Monitor as probe descends (detection range extends below probe)
- Pros: No miniaturization needed, standard sensors
- Cons: Detection only useful for shallow depths

**Option C: Custom Miniature Magnetometer Tip**
1. Develop custom 10-15mm diameter magnetometer package
2. Integrate at or near probe tip
- Pros: True look-ahead capability
- Cons: Requires significant R&D, custom electronics

### 9.2 Automatic Abort System

**Required Features:**
1. Real-time magnetic gradient monitoring
2. Tip resistance / push force monitoring
3. Configurable threshold settings
4. Automatic push-stop on threshold breach
5. Visual and audible operator alerts
6. Manual override capability

**Suggested Thresholds (to be calibrated in field):**
- Magnetic gradient: >5 nT/cm rate of change
- Push force increase: >200N sudden increase
- Depth limit: Site-specific maximum bomb penetration depth

### 9.3 Operational Protocol

**Before Any HIRT Deployment at UXO Risk Sites:**

1. **Desktop Assessment**
   - Research site history (bombing records, military use)
   - Review existing surveys and investigations
   - Determine UXO risk category

2. **Pre-Survey (if risk identified)**
   - Surface magnetometer survey
   - Mark any anomalies for avoidance/investigation

3. **Intrusive Clearing (each probe position)**
   - Option A: Full-depth magnetometer clearing with standard cone
   - Option B: Progressive clearing with down-hole probe

4. **HIRT Insertion (after clearance)**
   - Continuous monitoring throughout push
   - Immediate stop on any anomaly
   - Post-push review of all data

5. **Anomaly Protocol**
   - Do not push through unverified anomalies
   - Engage UXO specialist for investigation
   - Relocate probe position if necessary

### 9.4 Integration with HIRT MIT System

**Potential Interference Issues:**
- HIRT MIT coils generate electromagnetic fields
- Could interfere with magnetometer readings
- Could receive interference from magnetometer drive signals

**Mitigation Approaches:**
1. **Time-division multiplexing:** Alternate MIT acquisition and mag monitoring
2. **Spatial separation:** Magnetometer at tip, MIT coils further back
3. **Frequency separation:** Use different operating frequencies
4. **Shielding:** Electromagnetic screens between sensor systems

**Testing Required:**
- Characterize interference in controlled conditions
- Develop optimal operating sequences
- Validate detection capability with MIT active

---

## 10. Technology Comparison Matrix

| Technology | Detection Range | UXO Detection | Miniaturization | Real-Time | HIRT Compatible |
|------------|-----------------|---------------|-----------------|-----------|-----------------|
| Fluxgate Magnetometer | 2m (50kg bomb) | Excellent (ferrous) | Challenging | Yes | Possible |
| TDEM/EMI | 1-2m | Good (all metals) | Very Difficult | Yes | Unlikely |
| ERT | Poor | Poor | Already integrated | Yes | Supplementary only |
| Force Sensing | Contact only | Backup only | Integrated | Yes | Yes (backup) |
| GPR | Variable | Good | Not feasible | No | No |
| Seismic | Meters | Research only | Not feasible | Limited | No |

---

## 11. Conclusions

1. **Magnetometry is the proven, dominant technology** for real-time UXO detection during soil probing. Commercial systems from Fugro, Gouda Geo, and others successfully detect 50kg bombs at 2m radius.

2. **Miniaturization is the key challenge** for HIRT integration. Standard magnetometer cones are 44mm diameter; HIRT probes are 16mm. Custom development or alternative deployment approaches are needed.

3. **A two-pass system offers the highest safety** with proven technology: clear each position with a standard magnetometer cone before HIRT probe insertion.

4. **Automatic abort systems exist** and should be implemented regardless of detection method. Fugro's system monitors both magnetic gradient and tip resistance.

5. **ERT cannot replace magnetometry** for UXO detection. HIRT's existing ERT capability is valuable for soil characterization but unreliable for metal detection.

6. **False positives are inevitable** and should be accepted. Setting conservative thresholds and investigating all anomalies is safer than risking a missed detection.

7. **Industry protocols provide a proven framework**. Following established practices (CIRIA C681, Crossrail approach) ensures comprehensive risk management.

8. **MIT interference requires investigation**. Testing is needed to characterize and mitigate potential electromagnetic interference between HIRT's MIT coils and any magnetometer system.

---

## 12. Sources and References

### Commercial Equipment Manufacturers
- [Fugro - UXO and Geotechnical Services](https://www.fugro.com)
- [Gouda Geo - Magnetometer Cone](https://gouda-geo.com/product/magnetometer-cone-magcone)
- [Royal Eijkelkamp - Geomagnetic Module](https://www.royaleijkelkamp.com/products/drilling-cpt/cpt/modules/magnetometer-module/)
- [A.P. van den Berg - CPT Equipment](https://www.apvandenberg.com)
- [Lankelma - Magnetometer Cone](https://www.lankelma.com/cpt-services/testing-services/magnetometer-cone/)

### UXO Industry Resources
- [Brimstone UXO - Intrusive Survey Services](https://www.brimstoneuxo.com/survey-uxo/intrusive-uxo-survey/)
- [Zetica UXO - Detection Services](https://zeticauxo.com/investigation/uxo-detection/)
- [Geometrics - UXO Detection Solutions](https://www.geometrics.com/solutions/uxo-detection/)

### Technical References
- [IEEE - Detection of Deeply Buried UXO Using CPT Magnetometers](https://ieeexplore.ieee.org/document/4069126/)
- [SPH Engineering - UXO Detection Distances](https://www.sphengineering.com/news/estimates-of-various-uxo-unexploded-ordnance-maximum-detection-distances-using-magnetometers)
- [Geometrics - How Far Can a Magnetometer See](https://www.geometrics.com/support/how-far-can-a-magnetometer-see/)

### Industry Guidance
- [Crossrail Learning Legacy - UXO Risk Assessment](https://learninglegacy.crossrail.co.uk/documents/crossrail-uxo-risk-assessment-pre-empting-best-practice/)
- CIRIA C681 - Unexploded Ordnance (UXO): A Guide for the Construction Industry

### Sensor Technology
- [MDPI - Low-Cost Small-Size Fluxgate Sensor](https://www.mdpi.com/1424-8220/21/19/6598)
- [PMC - Modular EMI Sensor Design](https://pmc.ncbi.nlm.nih.gov/articles/PMC11244430/)
- [Wikipedia - Magnetic Induction Tomography](https://en.wikipedia.org/wiki/Magnetic_induction_tomography)

---

*Document prepared for HIRT Project - Hydraulic Injection for Resistivity Tomography*
