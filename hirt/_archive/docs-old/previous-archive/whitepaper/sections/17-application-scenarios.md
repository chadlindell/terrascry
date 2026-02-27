# 17. Application Scenarios

## Scenario 1: Bomb Crater (10-15 m x ~3 m deep)

### Configuration
- **Rods:** 3.0 m length
- **ERT rings:** At 0.5 m, 1.5 m, 2.5 m from tip
- **Probe spacing:** 1.5-2 m
- **Section size:** Covers full crater + rim (may need multiple sections)
- **Probe count:** 20-36 probes depending on crater size

### Measurement Strategy
- **Emphasis:** MIT at 2-20 kHz for aluminum/steel masses near base
- **ERT focus:** Fill bowl geometry and wet pockets
- **Depth target:** 0-4+ m
- **Frequencies:** Lower frequencies (2-5 kHz) for deeper penetration

### Deployment Considerations
- **Rim deployment:** Consider rim-only approach initially
- **Full coverage:** May require multiple overlapping sections
- **Deep insertion:** 3 m probes for full depth coverage
- **Long baselines:** ERT corner-to-corner for deep investigation

### Expected Results
- **MIT:** Large metal objects (engine, landing gear) show strong response
- **ERT:** Crater walls show as resistivity boundaries
- **Fill detection:** Disturbed fill shows different resistivity than native soil
- **Wet zones:** Water accumulation shows as low resistivity

### Field Notes
- **UXO risk:** Ensure EOD clearance before deployment
- Document crater dimensions and depth
- Note any visible wreckage before probing
- Coordinate with excavation team

---

## Scenario 2: Woods Burials

### Configuration
- **Rods:** 1.6 m length
- **ERT rings:** At 0.4 m & 1.2 m from tip
- **Probe spacing:** 1-1.5 m
- **Section size:** 8x8 m
- **Probe count:** 12-16 probes

### Measurement Strategy
- **Emphasis:** ERT patterns for grave shaft detection
- **MIT frequencies:** 5-20 kHz (focus on metallic artifacts)
- **Targets:** Small metallic items (buckles, dog tags may be small--look for clusters)
- **Depth target:** 0-2 m

### Deployment Considerations
- **Minimal intrusion:** Use shallow insertion when possible
- **Tight spacing:** 1 m spacing for small targets
- **Multiple frequencies:** Sweep 5, 10, 20 kHz for different sensitivities
- **ERT focus:** Long baselines to detect disturbed zones

### Expected Results
- **ERT:** Grave shaft shows as resistivity contrast (disturbed, often moister soil)
- **MIT:** Metallic artifacts show as amplitude/phase anomalies
- **Combined:** More complete picture of burial location and contents

### Field Notes
- Document tree roots and other natural features
- Note any surface disturbances
- Coordinate with ground-penetrating radar if available
- Maintain respectful approach to potential burial sites

---

## Scenario 3: Swamp/Margins (>5 m targets)

### Configuration
- **Rods:** As deep as feasible at margins; start 1.5-2 m
- **Probe spacing:** 2-3 m (wider for access challenges)
- **Section size:** Adapt to accessible areas
- **Probe count:** Varies with access

### Measurement Strategy
- **Lower MIT frequencies:** 2-5 kHz for deeper penetration
- **Longer offsets:** Extended TX->RX baselines
- **ERT:** Wider injection pairs across water if possible
- **Shore probes:** Deploy from accessible margins
- **Consider:** Seismic add-on later for complementary data

### Deployment Considerations
- **Access limitations:** Work from shore/margins
- **Water depth:** May limit probe insertion depth
- **Extended baselines:** Use longest practical TX->RX distances
- **Safety:** Ensure safe access, proper equipment

### Expected Results
- **Deep targets:** Lower frequencies and longer offsets improve deep sensitivity
- **Water effects:** High conductivity of water affects both MIT and ERT
- **Marginal detection:** Targets near detection limits may be challenging

### Field Notes
- Document water levels and conditions
- Note access limitations
- Consider complementary methods (seismic, GPR from surface)
- Safety first: ensure safe working conditions

---

## Scenario Comparison

| Scenario | Rod Length | Spacing | Frequencies | Depth Target | Key Method |
|----------|------------|---------|-------------|--------------|------------|
| Woods Burials | 1.6 m | 1-1.5 m | 5-20 kHz | 0-2 m | ERT + MIT |
| Bomb Crater | 3.0 m | 1.5-2 m | 2-20 kHz | 0-4+ m | MIT + ERT |
| Swamp/Margins | 1.5-2 m | 2-3 m | 2-5 kHz | >5 m | Low-freq MIT |

---

## Cost and Timeline Planning

### Starter Kit Cost Breakdown (Standard Section)

#### Probes
- **Quantity:** Build **20 identical** + **2-4 spares**
- **Cost per probe:** $70-150 (see BOM)
- **Total probe cost:** **$1,400-3,000**
  - 20 probes: $1,400-3,000
  - 2-4 spares: $140-600

#### Base Gear
- Current source: $40-80
- Logger/tablet: $100-300
- Batteries: $30-120
- Cables: $20-60
- Sync/clock: $15-40
- **Total base gear:** **$200-500**

#### Tools and Supplies
- Pilot rods: $50-100
- Driver/extraction tools: $50-100
- 3D printing/machining: $50-150
- Miscellaneous tools: $50-100
- **Total tools:** **$200-400**

#### Total Indicative Cost

**Complete starter kit:** **$1,800-3,900**

*Note: Costs are indicative and vary by supplier, quantity discounts, and component choices.*

### Build Timeline (Hands-on)

#### Week 1-2: Prototype Phase
- **Order parts** for 2 prototype probes
- **Print/machine** capsules
- **Wind coils** (TX and RX)
- **Assemble** 2 prototype probes
- **Test** basic functionality

**Deliverables:**
- 2 working prototype probes
- Coil winding procedure documented
- Assembly procedure documented

#### Week 3: Calibration and Testing
- **Bench calibration** of prototypes
- **Aluminum/steel test target** trials
- **Verify** MIT and ERT operation
- **Refine** design based on results
- **Document** calibration procedures

**Deliverables:**
- Calibrated prototypes
- Test results
- Refined design specifications

#### Week 4-5: Scale-Up
- **Order parts** for full set (20-24 probes)
- **Scale assembly** to production quantity
- **Build** 12-20 probes
- **Field shakedown** on sandbox/test lot
- **Identify** and fix any issues

**Deliverables:**
- 12-20 completed probes
- Base hub assembled
- Field test results

#### Week 6+: Field Deployment
- **First real section** scans on site
- **Refine** procedures based on field experience
- **Build** remaining probes if needed
- **Document** lessons learned

**Deliverables:**
- Field-ready system
- Field deployment procedures
- Initial field data

### Budget Considerations

#### Essential (Minimum Viable System)
- 12 probes: $840-1,800
- Base hub: $200-500
- Tools: $200-400
- **Total:** ~$1,200-2,700

#### Recommended (Standard Section)
- 20 probes: $1,400-3,000
- Base hub: $200-500
- Tools: $200-400
- **Total:** ~$1,800-3,900

#### Complete (With Spares)
- 24 probes (20 + 4 spares): $1,680-3,600
- Base hub: $200-500
- Tools: $200-400
- **Total:** ~$2,100-4,500

---

## Optional System Enhancements

### Borehole Radar
- **Description:** External antenna mounted on probe rod
- **Implementation:** Board up-rod; coax-fed dipole/sleeve at tip
- **Benefits:** High-resolution near-probe imaging
- **Use case:** Detailed characterization of anomalies
- **Complexity:** Moderate (requires RF expertise)

### Seismic Crosshole
- **Description:** Seismic source and receivers on probe array
- **Implementation:** Hammer strikes on rod heads; geophones clamped to rods
- **Benefits:** Complementary to EM methods, good for voids
- **Use case:** Detect air-filled voids, compaction changes
- **Complexity:** Moderate (requires seismic processing)

### Soil Ion Spot Tests
- **Description:** Chemical tests for bone diagenesis indicators
- **Implementation:** Phosphate/calcium spot tests at probe locations
- **Benefits:** Direct indicators of organic remains
- **Use case:** Confirm potential burial locations
- **Complexity:** Low (field chemistry tests)

### Magnetometer Sweep
- **Description:** Pre-scan with magnetometer before probe deployment
- **Implementation:** Surface magnetometer survey
- **Benefits:** Identify ferrous metal concentrations
- **Use case:** Prioritize sections for detailed probing
- **Complexity:** Low (standard archaeological tool)

### Integration Considerations

#### Data Fusion
- Co-register multiple data types
- Combine MIT, ERT, seismic, GPR
- Generate multi-parameter 3D models
- Improve interpretation confidence

#### System Complexity
- Each add-on increases system complexity
- Consider power requirements
- Additional data processing needed
- Field deployment time increases

---

## Deployment Considerations Per Scenario

### Adapt for Conditions
- **Tight targets:** Reduce spacing, increase frequency
- **Deep targets:** Increase spacing, decrease frequency, longer rods
- **Wet conditions:** Account for high conductivity in interpretation
- **Dry conditions:** May need to improve ERT contact

### Combine Methods
- **MIT + ERT:** Standard dual-method approach
- **With GPR:** Surface GPR for initial screening
- **With magnetometry:** Pre-scan to prioritize areas
- **With excavation:** Coordinate with dig teams

### General Field Procedures

#### Pre-Deployment
1. **Site assessment:** Evaluate access, safety, target depth
2. **Configuration selection:** Choose rod length, spacing, frequencies
3. **Grid planning:** Lay out probe positions
4. **Equipment check:** Verify all probes, base hub, tools

#### During Deployment
1. **Systematic approach:** Follow "set once, measure many" workflow
2. **Quality control:** Check reciprocity, repeat measurements
3. **Documentation:** Record all conditions, anomalies, issues
4. **Adaptation:** Adjust strategy based on initial results

#### Post-Deployment
1. **Data backup:** Secure all data immediately
2. **Quick analysis:** Generate preliminary plots if possible
3. **Equipment care:** Clean, inspect, repair as needed
4. **Documentation:** Complete field notes and logs

---

## Recommendations

### Start Simple
- Begin with core MIT+ERT system
- Prove methodology on test sites
- Gain field experience
- Identify specific needs

### Add Selectively
- Add enhancements based on demonstrated need
- Consider cost and complexity
- Evaluate field utility
- Document lessons learned

### Maintain Modularity
- Keep system modular
- Ensure add-ons don't compromise core functionality
- Design for easy integration
- Maintain backward compatibility

