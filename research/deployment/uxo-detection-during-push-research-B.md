# UXO Detection During Probe Push: Research Report

**Document Version:** 1.0
**Date:** January 2026
**Author:** HIRT Development Team
**Purpose:** Evaluate real-time metal/UXO detection technologies for integration with 16mm diameter soil penetration system

---

## Executive Summary

This research report investigates technologies for detecting buried unexploded ordnance (UXO) during probe insertion at potential WWII bomb crater sites. The goal is to detect buried ordnance **before** physical contact occurs, enabling abort procedures.

### Key Findings

| Detection Method | Feasibility | Detection Range | Integration Complexity |
|-----------------|-------------|-----------------|----------------------|
| CPT Magnetometer Cone | **Proven** | 1.5-2m lateral | Low (industry standard) |
| Miniature Fluxgate | **High** | 1-2m | Moderate |
| MEMS Magnetometer | **Moderate** | <1m (lower sensitivity) | Low |
| Eddy Current/Metal Detector | **Limited** | <0.5m typical | Moderate |
| ERT (Resistivity) Changes | **Limited** | Indirect detection only | N/A (already integrated) |
| Acoustic/Ultrasonic | **Experimental** | Variable | High |

### Recommendation

**Primary approach:** Integrate a tri-axial fluxgate magnetometer system (similar to commercial CPT magnetometer cones) ahead of the HIRT probe tip, operating at the standard 20mm/sec push rate with real-time data processing and automatic abort capability.

**Secondary approach:** Conduct surface magnetometer survey before each probe location using handheld or drone-mounted systems for pre-screening to 3-4m depth.

---

## 1. Existing UXO-CPT Systems

### 1.1 Industry Overview

The combination of Cone Penetration Testing (CPT) with magnetometer sensors for UXO detection is a mature, proven technology used extensively in Britain, Netherlands, Germany, Belgium, and France. These systems are specifically designed to ensure that tests, boreholes, and piles do not encounter unexploded ordnance.

**Key applications:**
- Pre-construction surveys at potential UXO sites
- Pile installation clearance
- Deep excavation surveys (up to 25m depth)
- Sites where surface detection methods are ineffective

### 1.2 Commercial Magnetometer Cone Systems

#### Royal Eijkelkamp Geomagnetic Module
**Specifications:**
- Tri-axial fluxgate magnetometer
- Sensitivity: 1 nT
- Detection radius: ~2m lateral (for 50kg+ ferrous objects)
- Module length: 350mm (excluding cone or dummy tip)
- Housing: Non-magnetic steel, 44mm diameter
- Built-in bidirectional inclinometer (+/-25 degrees, accuracy <0.1 degree)
- Maximum penetration speed: 20mm/sec (with CPT cone), 50mm/sec (standalone)

**Integration:** Can be housed behind dummy tip or 15cm2 CPT cone. Standard tapered thread connection to sounding tubes. "North" marking on exterior for orientation.

**Source:** [Royal Eijkelkamp - Geomagnetic Module](https://www.royaleijkelkamp.com/products/cone-penetration-testing/cpt/modules/geomagnetic-module/)

#### Gouda Geo MagCone
**Specifications:**
- Tri-axial magnetometer for magnetic anomaly modeling
- Sensitivity: 1 nT
- Detection radius: Up to 2m lateral (4m diameter clearance cylinder)
- Module length: 350mm
- Probe diameter: 44mm (with high-grade non-magnetic steel cone)
- Push speed: 20mm/sec (with CPT), 50mm/sec (standalone)

**Capabilities:**
- Real-time magnetic field measurement in three orthogonal directions
- Calculated resultant field
- Can simultaneously collect CPTu data (tip resistance, sleeve friction, pore pressure)

**Source:** [Gouda Geo - MagCone](https://gouda-geo.com/products/cpt-equipment/special-modules-for-cpt/magnetometer-cone-magcone)

#### Lankelma Magnetometer Cone
**Key features:**
- 1.5m detection radius (real-time readings)
- Detects ferrous objects: tanks, drums, pipes, bombs
- Combination cones available with standard CPT sensing
- Effectively halves site investigation time

**Source:** [Lankelma - Magnetometer Cone](https://www.lankelma.com/cpt-services/testing-services/magnetometer-cone/)

#### Foerster FEREX Borehole Detection
**Specifications:**
- Fluxgate gradiometer with 650mm sensor separation
- Probe diameter: 35mm
- Probe length: 850mm
- Measuring range: +/-10,000 nT
- Stability: <1 nT
- Noise: <1 nT peak-to-peak
- Watertight to 100m depth

**Applications:** Deep object detection where surface interference prevents surface scanning.

**Source:** [Institut Dr. Foerster - FEREX Borehole Detection](https://www.foerstergroup.com/en/usa/products/ferex-borehole-detection/unexploded-ordnance-devices-uxo/)

### 1.3 How CPT Magnetometer Systems Work

**Operating principle:** UXO detectors are differential magnetometers with multiple sensors arranged in geometrically true alignment. They measure deviations from zero in a homogeneous magnetic field. Each sensor passing a ferrous object is affected differently, with the gauge deflection in nT depending on object orientation.

**Detection mechanism:**
1. As probe advances, magnetometer continuously samples Earth's magnetic field
2. Ferrous objects create local distortions (anomalies) in the field
3. Tri-axial measurement allows 3D anomaly characterization
4. Real-time display shows field intensity and gradient
5. Operator monitors for threshold exceedance
6. Abort triggered when anomaly indicates proximity to metal

**Source:** [IEEE - Detection of Deeply Buried UXO Using CPT Magnetometers](https://ieeexplore.ieee.org/document/4069126/)

### 1.4 Costs and Availability

**Equipment costs** (estimated, contact manufacturers for quotes):
- Magnetometer module only: EUR 15,000-30,000
- Complete CPT magnetometer system: EUR 40,000-80,000
- Integration with existing CPT rig: EUR 20,000-40,000

**Service costs:**
- CPT magnetometer survey with operator: $3,000-4,000/day
- Typical production: 50-100m of penetration testing per day

**Manufacturers:**
- Royal Eijkelkamp (Netherlands): sales@eijkelkamp.com
- Gouda Geo-Equipment (Netherlands): www.gouda-geo.com
- Institut Dr. Foerster (Germany): www.foerstergroup.com
- Lankelma (UK): www.lankelma.com
- Fugro (Global): www.fugro.com

---

## 2. Miniature Magnetometer Technology

### 2.1 Fluxgate Magnetometers

Fluxgate magnetometers offer the best sensitivity for UXO detection applications. Key characteristics:

**Advantages:**
- High sensitivity (sub-nT resolution)
- Excellent accuracy and linearity
- Low noise (<0.1 nT RMS achievable)
- Proven technology for UXO applications
- Vector (directional) measurement capability

**Disadvantages:**
- Larger than MEMS sensors
- Higher power consumption
- More complex electronics

#### Small-Size Fluxgate Options

**WUNTRONIC WFG-110/WFG-120 Series:**
- Complete 3-axis system
- Diameter: 0.725" (18.4mm) - **compatible with 16mm probe integration**
- Length: 1.5" (38mm)
- Noise: <3x10^-6 G RMS/Hz (~0.3 nT RMS/Hz)
- Range: Up to 1 Gauss (100,000 nT)
- Power: +/-5VDC at +/-30mA (200mW total)
- Rectangular version (WFG-120): 0.75" x 0.75" x 2.75", fits in 1.0" cylinder

**Source:** [WUNTRONIC - Miniatur Fluxgate Magnetometer](https://www.wuntronic.de/en/miniatur-fluxgate-magnetometer-wfg-110.html)

**Research Race-Track Probe Design:**
- Single-component probe size: 10mm diameter x 30mm length
- Complete sensor with data acquisition: 70mm diameter x 100mm length
- Co-based amorphous magnetic core
- Noise: <0.1 nT RMS
- Excellent linearity

**Source:** [MDPI - Design of a Low-Cost Small-Size Fluxgate Sensor](https://www.mdpi.com/1424-8220/21/19/6598)

**SMILE (Small Magnetometer In Low mass Experiment):**
- CubeSat-class digital fluxgate
- Sensor cube: 20mm x 20mm x 20mm (Macor ceramic housing)
- Uses FPGA digital processing
- Very low power consumption
- Customizable operation modes

**Source:** [ResearchGate - Miniaturized digital fluxgate magnetometer](https://www.researchgate.net/publication/231059433_Miniaturized_digital_fluxgate_magnetometer_for_small_spacecraft_applications)

**Meldor High-Temperature 3-Axis Magnetometer:**
- Diameter: 18mm - **potentially compatible with 16mm probe housing**
- Length: 110mm
- Temperature range: up to 145 degrees C
- Low power, single-polarity supply
- Designed for autonomous magnetometric systems

**Source:** [Meldor - Small size high-temperature magnetometer](https://meldor.ru/en/small-size-magnetometer/)

### 2.2 MEMS Magnetometers

MEMS (Micro-Electro-Mechanical Systems) magnetometers offer smaller size and lower power but with reduced sensitivity.

**Comparison to Fluxgate:**

| Parameter | Fluxgate | MEMS (AMR/GMR) |
|-----------|----------|----------------|
| Noise floor | 0.1-1 nT RMS | 10-50 nT RMS |
| Size | 10-20mm diameter | 3-5mm chip |
| Power | 100-400mW | 10-50mW |
| Cost | $100-1,000 | $5-50 |
| Sensitivity | High | Moderate |
| UXO suitability | **Excellent** | **Marginal** |

**Key finding:** MEMS magnetometers (like RM3100) have noise levels of approximately 15 nT RMS, compared to fluxgate systems (like FG3+) at 0.311 nT RMS. This 50x difference in noise floor significantly impacts detection range for small or distant targets.

**Source:** [PMC - Precision Magnetometers for Aerospace Applications](https://pmc.ncbi.nlm.nih.gov/articles/PMC8402258/)

**MEMS Micro-Fluxgate (Research):**
- Hybrid MEMS/fluxgate designs achieving 500 pT/sqrt(Hz) noise at 1 Hz
- Can detect DC magnetic fields of 6 nT
- Sensitivity: 1.27 V/Oe
- Represents best-case MEMS performance

**Source:** [IEEE - Highly-Sensitive MEMS Micro-Fluxgate Magnetometer](https://ieeexplore.ieee.org/document/9812940/)

### 2.3 Fitting in 16mm Diameter Probe

**Challenge:** Standard CPT magnetometer cones are 44mm diameter. HIRT probe is 16mm diameter.

**Assessment:**

| Sensor Type | Smallest Available | Fits in 16mm? | Notes |
|-------------|-------------------|---------------|-------|
| WUNTRONIC WFG-110 | 18.4mm diameter | **Close** | May need custom housing |
| Meldor 3-axis | 18mm diameter | **Close** | Needs probe redesign |
| Research mini-fluxgate | 10mm diameter | **Yes** | Custom development |
| MEMS chips | 3-5mm | **Yes** | Lower sensitivity |

**Options for 16mm integration:**

1. **Enlarge probe head locally:** Create a 20-25mm diameter magnetometer housing just behind the tip, tapering back to 16mm for the electrode array. This is feasible if housing length is <100mm.

2. **Axial sensor arrangement:** Use axially-oriented single-axis fluxgate sensors (like Bartington Mag-B probes) which have smaller cross-sections.

3. **Custom miniature sensor:** Commission development of a ~12mm diameter tri-axial fluxgate specifically for this application.

4. **MEMS with signal processing:** Use high-quality MEMS sensor with advanced signal processing (filtering, averaging) to improve effective sensitivity.

### 2.4 Power Requirements

| Sensor Type | Power | Runtime on 2000mAh Battery |
|-------------|-------|---------------------------|
| Fluxgate (analog) | 200-400mW | 10-20 hours |
| Fluxgate (digital) | 100-200mW | 20-40 hours |
| MEMS magnetometer | 10-50mW | 80-400 hours |

**Conclusion:** Power is not a limiting factor. Even fluxgate sensors can operate for a full survey day on small batteries.

### 2.5 Data Rates for Real-Time Detection

**Industry standard:** CPT magnetometer systems operate at 1 sample/second at 20mm/sec push rate (one reading per 20mm depth increment).

**Higher-rate systems:**
- Geometrics G-882: Up to 20 Hz
- MagArrow II (UAV): 1000 Hz
- General fluxgate capability: 1-100 Hz typical

**For HIRT at 20mm/sec:**
- At 1 Hz: One sample per 20mm of travel
- At 10 Hz: One sample per 2mm of travel
- At 100 Hz: One sample per 0.2mm of travel

**Recommendation:** 10-20 Hz sampling provides excellent spatial resolution while keeping data processing manageable. At 20 Hz with 20mm/sec push, readings are spaced 1mm apart vertically.

**Source:** [Geometrics - How Does Magnetometer Noise Vary with Sample Rate](https://www.geometrics.com/support/how-does-magnetometer-noise-vary-with-sample-rate/)

---

## 3. Detection Physics

### 3.1 Magnetic Anomaly from Buried Bombs

**Fundamental rule of thumb (Geometrics):**
> "One ton (1000 kg) of steel or iron will give a 1 nT anomaly at 100 ft (30m). Since the amount of distortion falls off as the cube with distance and is linear with mass, every time we cut the distance in half, we can see 1/8th the mass."

**Scaling for bomb sizes:**

| Mass | Distance for 1 nT anomaly | Distance for 5 nT anomaly |
|------|---------------------------|---------------------------|
| 1000 kg | 30m | ~17m |
| 250 kg (250kg bomb) | ~19m | ~11m |
| 100 kg | ~14m | ~8m |
| 50 kg | ~11m | ~6m |
| 15 kg | ~8m | ~4.5m |
| 2 kg | ~4m | ~2.2m |

**Source:** [Geometrics - How Far Can a Magnetometer "See"?](https://www.geometrics.com/support/how-far-can-a-magnetometer-see/)

**WWII 250kg (SC250) Bomb Characteristics:**
- Total weight: ~250 kg
- Explosive fill: ~130 kg (Grade II/III designs)
- Steel casing: ~120 kg ferrous material
- Typical dimensions: ~1.6m length x 0.37m diameter

**Magnetic moment calculation:**
The magnetic dipole moment of a ferrous object is approximately: **m = M x V**

Where M is magnetization and V is volume. For steel:
- Magnetic susceptibility: chi approximately 200,000 x 10^-8 m^3/kg
- Saturation magnetization: ~80 A.m^2/kg

For a 120kg steel bomb casing, the magnetic moment would be on the order of 100-500 A.m^2, depending on remanent magnetization.

**Source:** [ScienceDirect - Magnetic Dipole Moment](https://www.sciencedirect.com/topics/engineering/magnetic-dipole-moment)

### 3.2 Detection Distance Calculations

**For a 250kg bomb (120kg steel) with 1 nT detection threshold:**

Using the inverse cube law: B = (mu_0/4*pi) * (2m/r^3) for axial field

At various distances:

| Distance | Expected Anomaly | Detectable? (1 nT threshold) |
|----------|-----------------|------------------------------|
| 0.5m | 500-2000 nT | **Yes** (very strong) |
| 1.0m | 60-250 nT | **Yes** (strong) |
| 1.5m | 18-75 nT | **Yes** (clear) |
| 2.0m | 8-30 nT | **Yes** (detectable) |
| 2.5m | 4-15 nT | **Yes** (detectable) |
| 3.0m | 2-8 nT | **Marginal** |
| 4.0m | 1-3 nT | **At threshold** |
| 5.0m | 0.5-1.5 nT | **Difficult** |

**Critical note:** These calculations assume idealized dipole behavior. Real bombs have:
- Non-uniform magnetization
- Variable remanent magnetization (depends on manufacturing history)
- Orientation effects (transverse vs. axial approach)
- Higher-order multipole contributions

**Practical detection range for 250kg bomb:** 2-4m with 1 nT sensitivity magnetometer.

**Source:** [SPH Engineering - UXO Maximum Detection Distances](https://www.sphengineering.com/news/estimates-of-various-uxo-unexploded-ordnance-maximum-detection-distances-using-magnetometers)

### 3.3 Minimum Detection Distance for Safe Abort

**Critical factors:**

1. **Detection distance:** Distance at which anomaly exceeds threshold
2. **Processing time:** Time to recognize anomaly and trigger abort
3. **Mechanical response:** Time for hydraulic system to stop push
4. **Safety margin:** Additional distance for uncertainties

**Calculation for HIRT system at 20mm/sec:**

| Factor | Time/Distance | Cumulative |
|--------|---------------|------------|
| Signal above noise (2-3 samples) | 100-150ms / 2-3mm | 3mm |
| Processing and decision | 50-100ms / 1-2mm | 5mm |
| Hydraulic valve response | 50-200ms / 1-4mm | 9mm |
| Ram deceleration to stop | 100-300ms / 2-6mm | 15mm |
| **Safety margin** | **50mm minimum** | **65mm** |

**Minimum detection distance required:** Approximately **100mm (10cm)** from probe tip to bomb surface.

**Achieved by 250kg bomb:** 2-4m detection range >> 0.1m required. **Large safety margin.**

**Concern:** Smaller fragments or debris (1-10kg) may only be detectable at 0.5-1m. At 20mm/sec with 100mm stopping distance, this still provides adequate warning, but margin is reduced.

### 3.4 Effect of Push Speed on Detection Reliability

**Standard CPT magnetometer speed:** 20mm/sec (mandated when combined with CPT measurements)

**Maximum standalone speed:** Up to 50mm/sec allowed by manufacturers

**Speed vs. detection reliability:**

| Speed | Samples in 1m approach | Spatial Resolution | Assessment |
|-------|------------------------|-------------------|------------|
| 10mm/sec | 50 (at 1 Hz) | Excellent | Slow but very reliable |
| 20mm/sec | 25 (at 1 Hz) | Good | **Industry standard** |
| 20mm/sec | 500 (at 10 Hz) | Excellent | Better SNR through averaging |
| 50mm/sec | 10 (at 1 Hz) | Marginal | May miss weak anomalies |

**Recommendation:** Maintain 20mm/sec push rate with 10-20 Hz sampling for optimal detection reliability.

### 3.5 Signal Processing for Moving Sensor

**Key challenges:**
- Sensor vibration during push
- Changing background field with depth (geological variations)
- Motion-induced noise
- Electromagnetic interference from hydraulic pump

**Processing approaches:**

1. **High-pass filtering:** Remove DC drift and slow geological variations
2. **Moving average:** Reduce random noise while preserving anomaly signals
3. **Gradient calculation:** First derivative emphasizes anomalies, reduces background
4. **Kalman filtering:** Optimal state estimation with known motion model
5. **Motion compensation:** Remove systematic errors from platform motion

**Industry practice:** Modern CPT systems achieve improvement ratios of 30:1 or better using digital compensation algorithms.

**Source:** [Geometrics - MagComp Software](https://www.geometrics.com/software/magcomp/)

**Recommended filter parameters for HIRT:**
- Low-pass cutoff: 5-10 Hz (removes high-frequency vibration)
- High-pass cutoff: 0.01-0.1 Hz (removes DC drift)
- Detection threshold: 3-5 sigma above local noise floor
- Alarm delay: 2-3 consecutive samples exceeding threshold

---

## 4. Alternative Detection Methods

### 4.1 Eddy Current Sensors

**Operating principle:** AC magnetic field induces eddy currents in conductive metals. Secondary magnetic field from eddy currents is detected.

**Characteristics:**
- Detects both ferrous and non-ferrous metals
- Works through non-conductive materials
- Detection range limited by coil size and frequency

**Detection range relationship:**
> "For inductive technology, sensing range of 50% of the coil diameter can be expected. For example, a sensor with outer diameter of 10mm can be expected to have a sensing range of 5mm."

**For HIRT application:**
- 16mm probe could accommodate ~10mm coil
- Expected sensing range: ~5mm **only**
- **Not adequate for look-ahead UXO detection**

**Best use:** Near-surface metal detection at ground level, not subsurface

**Source:** [Texas Instruments - Inductive Sensing for Metal Detection](https://www.ti.com/lit/SNOAA76)

### 4.2 Miniaturized Metal Detectors

**Pulse Induction (PI) systems:**
- Send short current bursts, measure decay response
- Good for mineralized soils
- Typical coil sizes: 20-200mm diameter
- Detection depth: Approximately 1-5x coil diameter for coin-sized objects

**Very Low Frequency (VLF) systems:**
- Continuous wave transmission/reception
- Better discrimination than PI
- More sensitive to ground mineralization

**Limitations for probe integration:**
- Coil must be larger than target detection depth
- 16mm coil provides only ~50-80mm detection depth
- Insufficient for look-ahead detection

**Conclusion:** Conventional metal detector technology is not suitable for integration into 16mm probe for UXO detection.

**Source:** [MetalDetector.com - How Deep Can a Metal Detector Search?](https://www.metaldetector.com/pages/learnbuying-guide-articlesgetting-startedhow-deep-can-a-metal-detector-search)

### 4.3 Changes in Soil Electrical Properties Near Metal

**ERT (Electrical Resistivity Tomography) considerations:**

HIRT already uses electrical resistivity measurements. Can these detect nearby metal?

**Physical effects:**
- Metal objects create resistivity anomalies (typically low resistivity)
- Large metal objects cause current channeling
- Effects are localized to near vicinity of metal

**Research findings:**
> "If the goal is to characterize contamination from buried drums, the resistivity survey could delineate the trench and plume pathways, but metal detectors or magnetometers can better locate the metal drums."

**For HIRT ERT array:**
- Current electrode spacing determines depth sensitivity
- HIRT's small electrode spacing (mm-cm) optimized for near-probe measurement
- May detect anomaly when very close to metal (within centimeters)
- Not adequate as primary look-ahead detection

**Potential use:** Secondary confirmation of magnetometer anomaly through resistivity signature change.

**Source:** [CLU-IN - Electrical Resistivity Tomography](https://clu-in.org/characterization/technologies/default2.focus/sec/Geophysical_Methods/cat/Electrical_Resistivity_Tomography/)

### 4.4 Acoustic Impedance Changes

**Concept:** Buried metal objects create acoustic impedance discontinuities detectable by ultrasonic sensing.

**Research status:**
> "Researchers demonstrated a high-resolution acoustic system capable of detecting and imaging small buried objects."

**Challenges:**
- High soil attenuation at ultrasonic frequencies
- Impedance matching between sensor and soil is critical
- Variable soil properties affect signal propagation
- Moisture content strongly influences effectiveness

**Current state:**
- Experimental for landmine detection
- 50 kHz acoustic signal used in research systems
- Depth penetration highly variable
- Not yet proven for deep subsurface UXO

**For HIRT:** Acoustic methods are currently **too experimental** for reliable UXO detection. May be viable for future development.

**Source:** [ScienceDaily - High-Resolution Acoustic System Detects Objects Buried In Soil](https://www.sciencedaily.com/releases/2000/10/001003072717.htm)

### 4.5 Ground Penetrating Radar (GPR)

**Relevant only for surface pre-scanning, not probe integration.**

**Depth limitations:**
- Dry sandy soil: Up to 15m penetration
- Moist clay: As little as 0.3-1m penetration
- High conductivity materials greatly attenuate signal

**Metal detection:**
- GPR **cannot see through metal** (waves reflect)
- Can locate metal objects to identify boundaries
- Cannot characterize what is behind/below metal

**For HIRT pre-survey:**
- Useful in dry, sandy soils
- Limited value in clay or wet conditions
- Complements magnetometer surveys

**Source:** [Wikipedia - Ground-penetrating radar](https://en.wikipedia.org/wiki/Ground-penetrating_radar)

---

## 5. Abort System Design

### 5.1 Hydraulic System Response Times

**Typical hydraulic valve response times:**
- Servo valves: 5-50 milliseconds
- Proportional valves: 50-200 milliseconds
- Simple on/off valves: 100-500 milliseconds

**For safety-critical applications:**
> "In safety applications, delayed emergency stops or pressure relief can turn incidents into disasters."

**CPT standard push rate:** 20mm/sec = 0.02m/sec

**Travel during valve response:**

| Valve Type | Response Time | Travel at 20mm/sec |
|------------|--------------|-------------------|
| Servo | 50ms | 1mm |
| Proportional | 200ms | 4mm |
| On/Off | 500ms | 10mm |

**Source:** [Global Electronic Services - Impact of Valve Response Time](https://gesrepair.com/the-impact-of-valve-response-time-in-hydraulic-systems/)

### 5.2 Complete Stopping Distance Calculation

**Components of total stopping distance:**

1. **Signal acquisition:** 2-3 samples at 10 Hz = 200-300ms
2. **Signal processing:** DSP/filtering = 10-50ms
3. **Decision threshold:** Confirmation logic = 50-100ms
4. **Communication delay:** Wired = <10ms, wireless = 50-200ms
5. **Valve response:** Proportional valve = 100-200ms
6. **Ram deceleration:** Depends on load, friction = 100-500ms

**Total estimated time:** 500-1500ms

**Travel at 20mm/sec:** 10-30mm

**Safety factor (2x):** 20-60mm

**Design requirement:** System must detect UXO at minimum **100mm** before contact. With 250kg bomb detectable at 2-4m, margin is very large.

### 5.3 Automatic vs. Operator-Triggered Abort

**Automatic abort:**

Advantages:
- Faster response (no human reaction time)
- Consistent decision criteria
- No operator fatigue/distraction factor
- Can operate 24/7

Disadvantages:
- False positive handling (may stop unnecessarily)
- Less ability to interpret complex signals
- Requires reliable threshold setting

**Operator-triggered abort:**

Advantages:
- Human judgment for ambiguous signals
- Can assess context (known metal in area, etc.)
- Can adjust thresholds in real-time
- Lower false positive rate

Disadvantages:
- Human reaction time: 200-500ms additional
- Fatigue and attention factors
- Requires trained operator

**Recommendation:** Hybrid system:
1. **Automatic abort** for strong signals (>100 nT anomaly)
2. **Operator warning** for moderate signals (10-100 nT anomaly)
3. **Continuous display** of real-time magnetic field data
4. **Manual override** capability

### 5.4 Extraction Procedure After Abort

**When magnetometer detects anomaly and push is stopped:**

**Immediate actions:**
1. Stop hydraulic push (automatic or manual)
2. Record exact position (GPS + depth)
3. Record magnetic field data (save last 30 seconds)
4. Do NOT advance further

**Extraction procedure:**
1. Reverse hydraulic cylinder (extract at controlled rate)
2. Monitor magnetic field during extraction (verify signal decreases)
3. Extract probe completely from ground
4. Mark location with warning marker
5. Relocate to safe distance (minimum 20m for assessment)

**Post-extraction assessment:**
1. Review magnetic data profile
2. Estimate anomaly depth and size from signal shape
3. Assess probability of UXO vs. other metal
4. Decision tree:
   - **High probability UXO:** Evacuate, contact EOD authorities
   - **Moderate probability:** Surface magnetometer survey, may resume nearby
   - **Low probability (debris):** May attempt adjacent location

**Time for extraction:** At 20mm/sec, 3m extraction requires ~150 seconds (2.5 minutes)

**Critical rule:** NEVER attempt to push through or past a detected anomaly. Always extract and assess.

---

## 6. Pre-Push Scanning Options

### 6.1 Surface Magnetometer Surveys

**Handheld Magnetometers:**

**Bartington Grad601-2:**
- Fluxgate gradiometer
- Sensitivity: 0.1 nT
- Detection depth: 2-3m for large ferrous objects
- Walking survey capability

**Geometrics G-858/G-859:**
- Cesium vapor magnetometer
- Sensitivity: <0.02 nT/sqrt(Hz)
- High data rate (10+ Hz)

**Foerster FEREX 4.032/4.034:**
- Fluxgate gradiometer
- Specifically designed for UXO detection
- Array configurations available (up to 4 sensors)

**Surface survey depth limitations:**
> "Non-intrusive (surface) survey methods have depth limitations. For projects with deep intrusive groundworks (more than 4 meters below ground level), an Intrusive Magnetometer Survey is needed."

**Effective depth for surface magnetometer:** 3-4m maximum for 250kg bomb class

**Source:** [Brimstone UXO - Intrusive UXO Survey](https://www.brimstoneuxo.com/survey-uxo/intrusive-uxo-survey/)

### 6.2 Drone-Mounted Magnetometer Surveys

**UAV Magnetometer Systems:**

**Geometrics MagArrow II:**
- Cesium vapor sensor
- Sample rate: 1000 Hz
- Noise: <0.02 nT/sqrt(Hz)
- Survey speed: up to 10 m/s
- Flight altitude: 2-5m typical

**GEM Systems DRONEmag GSMP-35U:**
- Sample rate: up to 20 Hz
- Towed below drone on 5m cable
- Reduces UAV interference

**UAV survey limitations:**
> "The depths at which UXO can be detected is far shallower than other, more traditional methods."

Detection distance = sensor altitude + target depth, so low flying increases effective depth.

**Practical detection depth:** 2-4m for bomb-sized objects (depending on altitude)

**Source:** [MDPI - High-Speed Magnetic Surveying for UXO Using UAV Systems](https://www.mdpi.com/2072-4292/14/5/1134)

### 6.3 Handheld Metal Detector Surveys

**Consumer/Professional Metal Detectors:**

**Detection depth (typical):**
- Small coil (4"): 10-15cm for coin-sized objects
- Medium coil (8"): 20-30cm for coin-sized objects
- Large coil (15"): 30-45cm for coin-sized objects
- Specialized deep-seeking: 1-2m for large objects

**Rule of thumb:**
> "Typically, one can expect depth range of about 1.5 times the diameter of the coil for coin-sized targets."

**For UXO-sized objects (bomb = 37cm diameter):**
- Could be detected at 2-4x coil depth (larger target = deeper detection)
- Realistic detection: 1-2m with large coil

**Limitations:**
- Much shallower than magnetometers
- Affected by ground mineralization
- Not quantitative (distance estimation difficult)

**Source:** [Garrett - How Deep Do Metal Detectors Detect?](https://garrett.com/how-deep-do-metal-detectors-detect/)

### 6.4 Cost/Benefit Analysis

| Method | Equipment Cost | Survey Time (50 holes) | Detection Depth | Reliability |
|--------|---------------|------------------------|-----------------|-------------|
| Surface magnetometer | $5,000-20,000 | 2-4 hours | 3-4m | High |
| Drone magnetometer | $20,000-50,000 | 1-2 hours | 2-4m | High |
| Handheld metal detector | $500-5,000 | 4-8 hours | 1-2m | Moderate |
| Integrated mag (in probe) | $10,000-30,000 | Included in push | Real-time | Highest |

**Recommendation:**
1. **Primary:** Integrated magnetometer in probe (real-time protection)
2. **Secondary:** Surface magnetometer pre-survey (identifies high-risk areas)
3. **Optional:** Metal detector spot-check of specific locations

### 6.5 Combined Approach Protocol

**Recommended pre-push protocol:**

**Phase 1: Desktop assessment**
- Historical records (bomb target area?)
- Previous surveys in area
- Known UXO finds nearby

**Phase 2: Surface magnetometer survey**
- Grid survey of entire work area
- Identify anomaly locations
- Map background field variations
- Flag any anomalies >5 nT above background

**Phase 3: Anomaly investigation**
- For flagged anomalies: additional measurements
- Determine anomaly depth and size estimates
- Decision: Avoid area vs. proceed with caution

**Phase 4: Intrusive survey (HIRT probe)**
- Integrated magnetometer active
- Real-time monitoring during push
- Automatic abort on threshold exceedance
- Continuous data recording

---

## 7. False Positive Management

### 7.1 Sources of False Positives

**In UXO surveys, false positives are very common:**
> "Approximately 4 percent of potential UXO targets picked from offshore projects result in a positive UXO identification."
> "Metal debris accounted for approximately half the false positives."

**Sources of magnetic anomalies that are NOT UXO:**

| Source | Typical Signature | Distinguishing Features |
|--------|------------------|------------------------|
| Iron-rich soil/rocks | Broad, diffuse | Gradual change, no sharp peak |
| Previous excavation debris | Variable, clustered | Often multiple small anomalies |
| Buried pipes/cables | Linear, continuous | Extended along line |
| Building foundations | Large, regular | Geometric pattern |
| Agricultural iron | Small, scattered | Random distribution |
| Natural magnetite | Regional trend | Correlates with geology |
| Vehicle/equipment | Very strong, localized | Known source location |

**Source:** [Oxford Academic - Inverse modelling and classification of magnetic responses](https://academic.oup.com/gji/article/237/1/123/7601866)

### 7.2 Distinguishing UXO from Harmless Metal

**Magnetic signature characteristics of UXO:**
- Compact dipole anomaly
- Characteristic aspect ratio
- Depth consistent with bomb penetration physics
- Strong remanent magnetization component

**Classification approaches:**

**1. Dipole modeling:**
> "Multi-magnetometer profiling allows direct inversion of raw magnetic data to locate and characterize dipoles typically generated by UXO."

**2. Higher-order moment analysis:**
> "Modeling results show the presence of significant higher order moments for more asymmetric objects. The contribution from higher order moments may provide a practical tool for improved UXO discrimination."

**3. Machine learning classification:**
> "Convolutional neural networks are now being used for characterization of magnetic anomalies, including counting dipoles, their position, and prediction of parameters."

**Discrimination success rates:**
> "All UXO could be separated from 68 percent of the false positives using classification methods."

**Source:** [MDPI - Magnetic mapping for detection and characterization of UXO](https://www.sciencedirect.com/science/article/abs/pii/S0926985106000929)

### 7.3 Soil Background Interference

**Magnetic susceptibility of soils:**
> "The magnetic properties of soils are largely determined by the presence of iron oxides. Magnetite, maghemite, and hematite can be formed depending on parent material and pedogenetic processes."

**Background field variations:**
- Earth's field: 20,000-80,000 nT (location dependent)
- Diurnal variation: up to 100 nT
- Geological variations: 1-1000 nT
- Urban interference: highly variable

**For HIRT magnetometer:**
- Use gradiometer configuration (measures field difference, not absolute)
- High-pass filter removes slow variations
- Local background subtraction before each hole

**Source:** [Canadian Science Publishing - Characterization of soil magnetic susceptibility](https://cdnsciencepub.com/doi/10.1139/cjss-2021-0040)

### 7.4 Acceptable False Positive Rate

**For HIRT UXO detection:**

**Cost of false positive:**
- Stop push (1-2 minutes)
- Extract probe (3 minutes)
- Investigate anomaly (5-30 minutes)
- May relocate to adjacent position
- **Total time cost:** 10-40 minutes per false positive

**Cost of false negative (miss):**
- Potential probe contact with UXO
- Risk of detonation (catastrophic)
- **Unacceptable**

**Decision criteria:**
- **Conservative threshold:** Maximize sensitivity, accept higher false positives
- Target: >99.9% detection probability for >50kg ferrous objects
- Accept: 1-5 false positives per 50 pushes (2-10% false positive rate)

**Recommended thresholds:**
- **Hard abort:** Anomaly >50 nT (automatic stop)
- **Soft alarm:** Anomaly >10 nT (operator warning, optional stop)
- **Background variation filter:** <5 nT ignored if gradual

---

## 8. Integration Design Recommendations

### 8.1 Recommended Sensor System

**Primary recommendation:** Miniature tri-axial fluxgate magnetometer

**Specifications:**
- Sensor type: Fluxgate (not MEMS)
- Configuration: Tri-axial
- Sensitivity: <1 nT resolution
- Noise: <0.5 nT RMS at 1 Hz
- Size: <20mm diameter (for integration with enlarged probe head)
- Sample rate: 10-20 Hz
- Power: <500mW

**Candidate sensors:**
1. WUNTRONIC WFG-110 (18.4mm diameter) - needs slight probe enlargement
2. Custom miniature fluxgate (development required)
3. Meldor small magnetometer (18mm diameter)

### 8.2 Mechanical Integration

**Option A: Enlarged sensor head**
```
     Probe Tip     Magnetometer      Main Probe Body
         |          Housing              |
    <==>----<===========O===========>--------------------->
    16mm    20-25mm     |              16mm
            (50-80mm length)
                   Fluxgate sensor
```

**Option B: In-line with reduced electrode array**
```
    Tip    Mag   Electrodes
    |       |       |
    <==>---[M]---o--o--o--o--o-->
    16mm   16mm
           (sensor must be <14mm)
```

**Recommendation:** Option A provides adequate space for proven sensors.

### 8.3 Electronics Integration

**Data acquisition:**
- ADC: 16-bit minimum, 24-bit preferred
- Sample rate: 20 Hz minimum
- Buffer: 30 seconds rolling storage
- Interface: Digital (SPI/I2C) to main controller

**Signal processing:**
- Real-time digital filtering (10 Hz low-pass, 0.1 Hz high-pass)
- Moving average (5-10 samples)
- Threshold detection with hysteresis
- Alarm flag generation

**Communication:**
- Add magnetic field channels to existing HIRT data stream
- Real-time transmission to operator display
- Audio alarm for threshold exceedance

### 8.4 Abort System Integration

**Requirements:**
1. Magnetic field data processed in <50ms
2. Threshold exceedance triggers abort signal
3. Abort signal stops hydraulic push within 200ms
4. Automatic extraction initiated (optional)
5. All data preserved for post-analysis

**Implementation:**
- Dedicated microcontroller for magnetometer processing
- Hardwired abort signal to hydraulic valve
- Redundant operator abort switch (always available)

### 8.5 Estimated Integration Costs

| Component | Estimated Cost |
|-----------|---------------|
| Miniature fluxgate sensor | $2,000-5,000 |
| Sensor housing/integration | $1,000-3,000 |
| Electronics (ADC, processor) | $500-1,500 |
| Signal processing firmware | $5,000-10,000 (development) |
| Abort system modifications | $2,000-5,000 |
| Testing and qualification | $5,000-10,000 |
| **Total integration** | **$15,000-35,000** |

**Recurring cost per probe:** $3,000-6,000 (sensor + housing)

---

## 9. Conclusions and Recommendations

### 9.1 Summary of Key Findings

1. **Proven technology exists:** CPT magnetometer cones are mature, commercially available systems used routinely for UXO detection during ground intrusion.

2. **Detection physics are favorable:** 250kg WWII bombs can be detected at 2-4m distance with 1 nT sensitivity magnetometers. HIRT's push speed of 20mm/sec provides ample stopping distance.

3. **Miniaturization is achievable:** Fluxgate magnetometers as small as 18mm diameter are commercially available. Custom development could achieve 12-14mm.

4. **Push-rate is ideal:** The standard CPT magnetometer push rate of 20mm/sec matches HIRT's operational parameters exactly.

5. **Alternative methods are inferior:** Eddy current, metal detector, acoustic, and ERT methods cannot provide the look-ahead range needed for UXO detection.

6. **False positives are manageable:** Classification algorithms and threshold tuning can reduce false positive rates while maintaining high detection probability.

### 9.2 Recommended Development Path

**Phase 1: Immediate (0-6 months)**
- Source miniature fluxgate sensors for evaluation
- Bench test sensitivity and noise performance
- Design enlarged probe head concept
- Develop signal processing algorithms

**Phase 2: Prototype (6-12 months)**
- Fabricate integrated magnetometer/probe head
- Integrate with HIRT data acquisition
- Develop abort system interface
- Lab testing with ferrous test objects

**Phase 3: Field Validation (12-18 months)**
- Field trials at non-UXO sites
- Characterize soil background effects
- Optimize detection thresholds
- Develop operational procedures

**Phase 4: Deployment (18-24 months)**
- Pilot deployment at surveyed UXO sites
- Parallel operation with conventional UXO clearance
- Refine based on operational experience

### 9.3 Critical Design Requirements

| Requirement | Specification | Rationale |
|-------------|---------------|-----------|
| Sensitivity | <1 nT | Detect 50kg objects at 2m |
| Sample rate | >10 Hz | 2mm spatial resolution |
| Processing latency | <100ms | Adequate stopping distance |
| Abort response | <500ms total | 10mm travel maximum |
| False negative rate | <0.1% | Safety critical |
| False positive rate | <10% | Operational efficiency |

### 9.4 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Sensor doesn't fit in probe | Medium | High | Design enlarged head option |
| Insufficient sensitivity | Low | High | Use fluxgate, not MEMS |
| High false positive rate | Medium | Medium | Threshold tuning, classification |
| Soil interference | Medium | Medium | Gradiometer configuration |
| Mechanical integration issues | Medium | Medium | Prototype testing |
| Abort system failure | Low | Critical | Redundant abort paths |

### 9.5 Final Recommendation

**Proceed with integrated magnetometer development.** The technology is proven, the physics are favorable, and the safety benefit is significant for operations at potential UXO sites.

**Primary system:** Tri-axial fluxgate magnetometer integrated into enlarged probe head, with automatic abort capability.

**Secondary system:** Pre-push surface magnetometer survey protocol for all UXO-risk sites.

---

## References

### Commercial Equipment Sources
- [Royal Eijkelkamp - Geomagnetic Module](https://www.royaleijkelkamp.com/products/cone-penetration-testing/cpt/modules/geomagnetic-module/)
- [Gouda Geo - MagCone](https://gouda-geo.com/products/cpt-equipment/special-modules-for-cpt/magnetometer-cone-magcone)
- [Lankelma - Magnetometer Cone](https://www.lankelma.com/cpt-services/testing-services/magnetometer-cone/)
- [Institut Dr. Foerster - FEREX](https://www.foerstergroup.com/en/usa/products/ferex-borehole-detection/)
- [Geometrics - G-882 Marine Magnetometer](https://www.geometrics.com/product/g-882-marine-magnetometer/)
- [WUNTRONIC - Miniature Fluxgate](https://www.wuntronic.de/en/miniatur-fluxgate-magnetometer-wfg-110.html)

### Technical Publications
- [IEEE - Detection of Deeply Buried UXO Using CPT Magnetometers](https://ieeexplore.ieee.org/document/4069126/)
- [MDPI - Design of a Low-Cost Small-Size Fluxgate Sensor](https://www.mdpi.com/1424-8220/21/19/6598)
- [PMC - Precision Magnetometers for Aerospace Applications](https://pmc.ncbi.nlm.nih.gov/articles/PMC8402258/)
- [ScienceDirect - Magnetic mapping for UXO detection](https://www.sciencedirect.com/science/article/abs/pii/S0926985106000929)
- [ResearchGate - UXO Discrimination and Identification Using Magnetometry](https://www.researchgate.net/publication/266409139_UXO_Discrimination_and_Identification_Using_Magnetometry)

### Industry Guidelines
- [SPH Engineering - UXO Maximum Detection Distances](https://www.sphengineering.com/news/estimates-of-various-uxo-unexploded-ordnance-maximum-detection-distances-using-magnetometers)
- [Geometrics - How Far Can a Magnetometer See?](https://www.geometrics.com/support/how-far-can-a-magnetometer-see/)
- [Brimstone UXO - Intrusive UXO Survey](https://www.brimstoneuxo.com/survey-uxo/intrusive-uxo-survey/)
- [SOCOTEC - Specialist Magnetometer Testing](https://www.socotec.co.uk/our-services/site-investigation/specialist-magnetometer-testing)

### UXO Safety and Fuses
- [Wikipedia - Unexploded Ordnance](https://en.wikipedia.org/wiki/Unexploded_ordnance)
- [Wikipedia - Artillery Fuze](https://en.wikipedia.org/wiki/Artillery_fuze)
- [CAT-UXO - Fuzes](https://cat-uxo.com/explosive-hazards/fuzes)
- [BombsAway UXO - WW2 German Bombs](https://bombsawayuxo.com/ww2-german-bombs)

### Soil and Background Effects
- [Canadian Science Publishing - Soil Magnetic Susceptibility](https://cdnsciencepub.com/doi/10.1139/cjss-2021-0040)
- [CLU-IN - Electrical Resistivity Tomography](https://clu-in.org/characterization/technologies/default2.focus/sec/Geophysical_Methods/cat/Electrical_Resistivity_Tomography/)
- [Wikipedia - Ground-penetrating Radar](https://en.wikipedia.org/wiki/Ground-penetrating_radar)

---

*Document prepared for HIRT Development Team - January 2026*
