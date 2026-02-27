# 33. Robotic/Remote Deployment Systems - Feasibility Assessment

## Executive Summary

This document assesses the feasibility of robotic and remote-controlled systems for HIRT probe insertion at UXO (Unexploded Ordnance) sites, specifically WWII bomb craters. The primary objective is **personnel safety through standoff distance** during the probe insertion phase.

**Key Findings:**
- Robotic deployment is technically feasible but requires custom development
- Existing agricultural soil sampling robots (ROGO) provide a valuable reference design
- Force requirements (5-20 kN) are achievable with proper platform anchoring
- Near-term solution: Remote-controlled mini excavator with custom attachment
- Long-term solution: Purpose-built robotic probe insertion platform
- Estimated development cost: $150,000-$400,000 for custom platform

---

## 1. Context and Requirements

### 1.1 HIRT Probe Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| Probe diameter | 16 mm OD | Fiberglass rod |
| Hole size | 18-20 mm | For 16mm probe |
| Insertion depth | 2-3 m | Crater scenario |
| Probes per grid | 20-50 | 10x10m survey grid |
| Probe spacing | 1.5-2.0 m | Standard grid |
| Target soil | Disturbed crater fill | Mixed, likely compacted |

### 1.2 UXO Safety Context

**Typical WWII Bomb Crater Parameters:**
- Crater diameter: 10-15 m
- Crater depth: ~3 m
- Potential ordnance: 250-1000 kg bombs
- Maximum penetration depth of buried UXO: up to 12 m below surface

**Evacuation Distances (German precedent):**
- 500 kg bomb: 300 m evacuation radius (11,000 people evacuated in Osnabruck)
- 1000 kg bomb: 450+ m evacuation radius (20,000 people evacuated)

**Safe Standoff for Robotic Operations:**
- Minimum recommended: **100 m** for remote operators
- Conservative: **300 m** (consistent with evacuation protocols)
- Control station should be behind blast barriers or terrain feature

### 1.3 Primary Objective

**Keep all personnel outside the UXO hazard zone during probe insertion operations.**

This means:
- No human presence within 100-300 m of potential UXO during insertion
- All probe insertion performed robotically or via remote control
- Only approach site after probes are installed and surveyed

---

## 2. Existing Robotic Systems Review

### 2.1 ROGO Agricultural Soil Sampling Robot

**Source:** [ROGO Soil Sampling Robot](https://rogoag.com/soil-sampling-robot)

The ROGO SmartCore is the most directly relevant existing system:

| Feature | ROGO Specification | HIRT Requirement |
|---------|-------------------|------------------|
| Platform | Kubota UTV-based | Tracked preferred |
| Navigation | RTK GPS (centimeter accuracy) | 2 m grid spacing |
| Positioning | Within inches year-over-year | 10 cm accuracy adequate |
| Insertion | Hydraulic auger | Push or auger |
| Depth control | 1/8 inch accuracy | 10 cm adequate |
| Force adaptation | Learns soil conditions | Variable force needed |
| Coverage | 100,000+ acres proven | 100 m^2 per session |
| Operation | Manned with autonomous override | Fully remote required |

**ROGO Adaptability Assessment:**
- **Pros:** Proven soil penetration, RTK navigation, variable force control
- **Cons:** UTV platform not suitable for UXO sites, requires operator presence
- **Adaptation potential:** Medium - core insertion mechanism valuable, platform needs replacement

**Key insight from ROGO:** Computer-controlled hydraulics that adapt force based on soil resistance is essential for reliable insertion.

### 2.2 PERSEPHONE/Stinger Mining Robots

**Source:** [PERSEPHONE Project](https://www.persephone-mining.eu/), [Stinger Robot](https://arxiv.org/html/2508.06521v1)

The Stinger robot demonstrates critical self-anchoring technology:

| Feature | Stinger Specification | HIRT Relevance |
|---------|----------------------|----------------|
| Anchoring | Tri-leg self-bracing | Reaction force essential |
| Drilling force | 1000 N central leg load | Similar to HIRT needs |
| Control | ROS 2 state machine | Modern, adaptable |
| Environment | Underground tunnels | Surface deployment easier |
| Autonomy | Force-aware closed-loop | Valuable for insertion |

**Key Innovation:** Self-locking tri-leg bracing system that adapts to irregular surfaces.

**Force Distribution During Operation:**
- Soft bracing: ~20 N on legs
- Hard bracing: ~140-280 N on legs
- During drilling: Central leg ~1000 N, side legs ~270 N each

**HIRT Application:** The anchoring concept is directly applicable - any probe insertion robot needs similar reaction force management.

### 2.3 EOD Robots with Manipulation

**Sources:** [L3Harris T7](https://www.l3harris.com/all-capabilities/t7-multi-mission-robotic-system), [T4 Robot](https://www.l3harris.com/all-capabilities/t4-robotic-system)

| Robot | Arm Reach | Lift Capacity | Control Range | Weight |
|-------|-----------|---------------|---------------|--------|
| T7 | 2+ m | 100 kg (near) | 1 km | 342 kg |
| T4 | 2 m horiz / 2.5 m vert | 55 kg (near) / 20 kg (extended) | 1 km | ~200 kg |
| tEODor | 2.86 m | 100 kg | 500 m | ~375 kg |

**Assessment for Probe Insertion:**
- **Reach:** Adequate for surface operations
- **Payload:** Marginal for sustained push force (20-55 kg = 200-540 N)
- **Control:** Excellent remote operation (1 km with haptic feedback)
- **Limitation:** Not designed for sustained push operations; force capacity insufficient

**Conclusion:** EOD robots excel at manipulation but lack the sustained push force and anchoring systems needed for 2-3 m probe insertion. Could potentially hold lightweight probes while another mechanism pushes.

### 2.4 UXO-Safe Robotic Drilling Systems

**Finding:** No purpose-built robotic drilling system exists specifically for UXO environments. Current UXO site characterization typically involves:
- Manual probe/core insertion by EOD-cleared personnel
- Magnetometer surveys (increasingly drone-based)
- Ground-penetrating radar from surface

**Gap identified:** This represents an unmet need in the UXO clearance industry.

---

## 3. Platform Options Analysis

### 3.1 Tracked UGV Platforms

**Evaluated Platforms:**

| Platform | Payload | Size | Power | Price | Suitability |
|----------|---------|------|-------|-------|-------------|
| Clearpath Warthog | 272 kg | Large | Electric | ~$40,000 | Good |
| TEC800 | 800 kg | Large | Diesel/Electric | ~$60,000 | Excellent |
| Bunker Pro 2.0 | 50 kg | Medium | Electric | ~$15,000 | Marginal |
| AgileX Scout | 50 kg | Medium | Electric | ~$10,000 | Marginal |

**Source:** [Clearpath Warthog](https://clearpathrobotics.com/warthog-unmanned-ground-vehicle-robot/), [TEC800](https://www.generationrobots.com/en/404086-tec800-mobile-tracked-robot-ugv.html)

**Recommendation:** TEC800 or similar 500-800 kg payload platform provides adequate capacity for insertion mechanism plus probe storage. Clearpath Warthog is a strong research/development platform with excellent ROS support.

### 3.2 Wheeled Robots

| Platform | Terrain Capability | Stability | Cost |
|----------|-------------------|-----------|------|
| Robotnik RB-SUMMIT | Moderate off-road | Good | ~$30,000 |
| Husarion Panther | Rugged outdoor | Good | ~$20,000 |

**Assessment:** Wheeled platforms offer faster movement but:
- Less stable anchor base on uneven crater terrain
- More susceptible to slippage during push operations
- Tracks preferred for bomb crater terrain (disturbed fill, uneven surface)

### 3.3 Drone-Based Approaches

**Current State:**
- Drones with magnetometers used for UXO surface surveys ([MagArrow II](https://www.geometrics.com/product/magarrow/))
- Operating altitude: 1-3 m for UXO detection
- Speeds up to 10 m/s with 1 cm sampling

**For Probe Insertion:**
- **Fundamental limitation:** Cannot generate 5-20 kN push force
- Maximum practical payload: 10-30 kg (heavy-lift drones)
- No reaction force anchoring possible

**Conclusion:** Drones are excellent for magnetometer pre-survey but **not suitable for probe insertion**.

**Potential hybrid approach:**
1. Drone magnetometer survey identifies anomaly locations
2. Ground robot performs probe insertion
3. Drone monitors operation from safe distance

### 3.4 Rail/Gantry Systems

**Concept:** Fixed gantry spanning survey grid

| Aspect | Assessment |
|--------|------------|
| Positioning accuracy | Excellent (<1 cm) |
| Force capacity | Excellent (unlimited with proper anchoring) |
| Setup time | High (hours to install) |
| Flexibility | Poor (fixed geometry) |
| Cost | Moderate ($20,000-$50,000) |
| Personnel exposure | Requires setup in hazard zone |

**Critical Issue:** Installing the gantry requires personnel presence in the UXO zone, defeating the primary safety objective.

**Modified approach:** Pre-fabricated modular gantry sections that could be robot-deployed. Adds significant complexity.

**Conclusion:** Not recommended as primary approach due to setup requirements.

---

## 4. Manipulation Requirements Analysis

### 4.1 Force Requirements for Probe Insertion

**Reference Data:**

| Source | Force | Soil Type | Probe Size |
|--------|-------|-----------|------------|
| CPT standard | 100-200 kN | All types | 36 mm cone |
| CPT soft soil | 10-25 kN | Clay/silt | 36 mm cone |
| Hand probe | 90-110 N | Soft | 12.7 mm |
| Post driver | 14-142 kN (impact) | All | 50-100 mm posts |
| ROGO auger | Variable (hydraulic) | Agricultural | ~25 mm |

**Estimated HIRT Requirements (16mm probe, 18-20mm hole):**

| Soil Condition | Estimated Push Force | Notes |
|----------------|---------------------|-------|
| Loose fill (disturbed crater) | 2-5 kN | Most likely condition |
| Firm clay | 5-15 kN | May need pilot hole |
| Compacted fill | 10-20 kN | Pilot hole recommended |
| Sandy soil | 1-3 kN | May allow direct push |

**Design Target:** 10 kN sustained push force (with 20 kN peak capability)

**Comparison:** Standard CPT rigs use 15-20 ton (150-200 kN) capacity. HIRT needs approximately **5-10% of CPT force** due to smaller probe diameter.

### 4.2 Reaction Force Handling

**The fundamental challenge:** Newton's third law - pushing 10 kN into the ground requires 10 kN reaction force.

**Anchoring Options:**

| Method | Reaction Capacity | Setup Time | Applicability |
|--------|-------------------|------------|---------------|
| Platform weight | 1,000 kg = ~10 kN | None | Marginal for firm soil |
| Screw anchors | 20-50 kN each | 1-2 min each | Excellent |
| Outrigger legs | Variable | Seconds | Good for soft soil |
| Suction pads | Limited | Minutes | Not suitable for soil |
| Driven stakes | 5-10 kN each | 30 sec | Good |

**Recommended Approach:** Combination of:
1. Heavy platform (>500 kg) for baseline stability
2. Deployable screw anchors (2-4 units) for high-force insertions
3. Outrigger legs for additional stability

**Screw Anchor Specifications:**
- Helical ground anchors can provide 20-50 kN pullout resistance
- Installation: Hydraulic drive, 1-2 minutes per anchor
- Can be robotically deployed and retrieved

### 4.3 Precision Positioning

**Requirements:**
- Grid spacing: 1.5-2.0 m
- Positioning accuracy needed: +/- 10 cm (5% of spacing)
- Absolute position (GPS): +/- 50 cm adequate for grid documentation

**Technical Solutions:**

| Technology | Accuracy | Cost | Notes |
|------------|----------|------|-------|
| RTK-GPS | 2-5 cm | $5,000-$15,000 | Proven in ROGO |
| Visual SLAM | 5-20 cm | $2,000-$5,000 | Requires markers |
| Pre-surveyed markers | 1-2 cm (relative) | $500 | Most reliable |
| LiDAR SLAM | 5-10 cm | $10,000-$30,000 | Good in open terrain |

**Recommendation:** RTK-GPS as primary, with visual confirmation of grid markers.

### 4.4 Multiple Probe Handling

**Requirement:** 20-50 probes per survey session

**Options:**

| Approach | Capacity | Complexity | Efficiency |
|----------|----------|------------|------------|
| Carry all probes | 20-50 | High | Single trip |
| Resupply trips | 5-10 per trip | Low | Multiple trips |
| Staged depot | Unlimited | Medium | Robot retrieves as needed |

**Probe Storage Calculations:**
- Probe length: 3.0 m (assembled)
- Probe weight: ~500 g (fiberglass + components)
- 25 probes: 12.5 kg, ~0.15 m^3 volume (bundled)

**Recommendation:** Robot carries 10-15 probes in magazine; return to edge of exclusion zone for resupply if needed.

---

## 5. Control Architecture

### 5.1 Tethered vs. Wireless Control

| Aspect | Tethered | Wireless |
|--------|----------|----------|
| Range | Limited by cable (typically <500 m) | 1+ km achievable |
| Reliability | High (immune to RF interference) | Moderate (can be interrupted) |
| Data bandwidth | High (gigabit possible) | Moderate (100 Mbps typical) |
| Setup complexity | Moderate (cable management) | Low |
| Snagging risk | High in cluttered terrain | None |
| Power delivery | Possible | Requires onboard battery |

**Recommendation:** Wireless primary with tethered backup option
- Wireless for normal operations (1 km range adequate for 300 m standoff)
- Fiber optic tether available for critical operations or RF-denied environments

### 5.2 Real-Time Video/Sensor Feedback

**Essential Sensors:**

| Sensor | Purpose | Bandwidth | Priority |
|--------|---------|-----------|----------|
| HD camera (front) | Navigation, obstacle avoidance | 5-20 Mbps | Critical |
| HD camera (insertion point) | Probe alignment, insertion monitoring | 5-20 Mbps | Critical |
| Force sensor | Insertion force feedback | <1 kbps | Critical |
| Depth encoder | Insertion depth tracking | <1 kbps | Critical |
| GPS/INS | Position logging | <10 kbps | High |
| Tilt sensor | Platform level | <1 kbps | High |
| Anchor force | Anchor engagement verification | <1 kbps | Medium |

**Video Latency Requirements:**
- Teleoperation: <200 ms round-trip
- Autonomous monitoring: <1 s acceptable

### 5.3 Autonomy Levels

**Recommended Graduated Autonomy:**

| Level | Operation | Human Role |
|-------|-----------|------------|
| 0 | Full teleoperation | Direct joystick control |
| 1 | Point-to-point navigation | Designate waypoints |
| 2 | Grid execution | Approve each insertion |
| 3 | Full grid autonomy | Monitor and intervene if needed |

**Initial Deployment:** Level 1-2 (human approves each insertion)
**Mature System:** Level 3 (autonomous grid execution with monitoring)

**Critical Autonomous Functions:**
- Emergency stop if force exceeds limits
- Automatic withdrawal if probe jams
- Return-to-base if communication lost
- Avoid excessive force that could trigger ordnance

### 5.4 Magnetometer Integration

**Concept:** Integrate magnetometer to detect/avoid ferrous masses during insertion

| Integration Level | Description | Benefit |
|-------------------|-------------|---------|
| Pre-survey | Separate mag survey before probing | Standard practice |
| Robot-mounted | Real-time mag monitoring | Avoid inserting into metal |
| Probe-integrated | Mag sensor in probe tip | Detailed subsurface data |

**Robot-Mounted Magnetometer:**
- Provides real-time anomaly detection
- Can halt insertion if unexpected ferrous target detected
- Helps verify probe locations relative to known anomalies

**Recommended:** Robot-mounted magnetometer array for real-time safety monitoring, with option for probe-integrated sensor in future.

---

## 6. Purpose-Built Robot Concept Design

### 6.1 Overview

**Name:** HIRT Autonomous Probe Insertion Robot (HAPIR)

**Design Philosophy:**
- Minimum viable system for safe probe insertion
- Emphasize reliability over sophistication
- Modular for future capability growth
- Field-serviceable with common tools

### 6.2 Platform Specifications

```
                    HAPIR Concept Design
                    ====================

        +-----------------------------------------+
        |         PROBE MAGAZINE (25 probes)      |
        |    [|||||||||||||||||||||||||||||||]    |
        +-----------------------------------------+
        |                                         |
        |   +--------+              +--------+    |
        |   | ANCHOR |   INSERTION  | ANCHOR |    |
        |   | SCREW  |    TOWER     | SCREW  |    |
        |   |   L    |      |       |   R    |    |
        |   +--------+      |       +--------+    |
        |                   |                     |
        |              +----+----+                |
        |              | FORCE   |                |
        |              | HEAD    |                |
        |              +----+----+                |
        |                   |                     |
        |   +===========+   |   +===========+     |
        |   || TRACK L ||   |   || TRACK R ||     |
        |   +===========+   v   +===========+     |
        |                   |                     |
        +-------------------+---------------------+

                   Front View

        +-------------------------------------+
        |                                     |
        |  CAM                          CAM   |
        |   o                            o    |
        |                                     |
        |  +-----------------------------+    |
        |  |     ELECTRONICS BAY         |    |
        |  |  - Computer (ROS 2)         |    |
        |  |  - Radio + antenna          |    |
        |  |  - Battery (48V, 20 Ah)     |    |
        |  |  - Hydraulic pump           |    |
        |  +-----------------------------+    |
        |                                     |
        |    TRACK       PROBE       TRACK    |
        |   ========    WINDOW      ========  |
        |                                     |
        +-------------------------------------+

                   Top View
```

### 6.3 Detailed Specifications

**Platform:**

| Parameter | Specification | Notes |
|-----------|---------------|-------|
| Type | Tracked UGV | Rubber tracks for terrain |
| Dimensions | 1.5 m L x 1.0 m W x 1.2 m H | Fits in cargo van |
| Weight | 400-600 kg (empty) | Provides reaction mass |
| Ground clearance | 15-20 cm | Traverses crater terrain |
| Maximum grade | 30 degrees | Handles crater slopes |
| Speed | 0.5-2 m/s | Slow for precision |

**Insertion Mechanism:**

| Parameter | Specification | Notes |
|-----------|---------------|-------|
| Type | Hydraulic linear actuator | Simple, powerful |
| Stroke | 1.0 m | Multiple strokes for 3 m depth |
| Force capacity | 20 kN | With safety margin |
| Speed | 2-5 cm/s | Controlled insertion rate |
| Feedback | Force sensor + encoder | Closed-loop control |
| Rotation | Optional auger drive | For difficult soils |

**Anchoring System:**

| Parameter | Specification | Notes |
|-----------|---------------|-------|
| Type | Hydraulic screw anchors | Automatic deployment |
| Quantity | 4 (one per corner) | Redundancy |
| Capacity | 15 kN each | 60 kN total reaction |
| Deployment time | 30 seconds per anchor | Parallel deployment |
| Depth | 30-50 cm | Adequate for most soils |

**Probe Handling:**

| Parameter | Specification | Notes |
|-----------|---------------|-------|
| Magazine capacity | 25 probes (assembled) | Sufficient for one grid |
| Feeding | Gravity-fed carousel | Simple mechanism |
| Assembly | Pre-assembled probes | No field assembly |
| Positioning | 2-axis gantry over insertion point | +/- 2 cm accuracy |

**Power System:**

| Parameter | Specification | Notes |
|-----------|---------------|-------|
| Primary | Li-ion battery pack | 48V, 20 Ah (960 Wh) |
| Backup | Small generator option | For extended operations |
| Endurance | 4-6 hours | Full survey grid |
| Charging | 2-3 hours to full | Swap battery option |

**Control System:**

| Parameter | Specification | Notes |
|-----------|---------------|-------|
| Computer | Industrial PC (ROS 2) | Nvidia Jetson or similar |
| Communication | 900 MHz + 2.4 GHz | Dual-band redundancy |
| Range | 2+ km | Exceeds safety standoff |
| Video | 2x HD cameras + recording | Front + insertion point |
| Sensors | GPS/RTK, IMU, magnetometer | Full situational awareness |

### 6.4 Operation Sequence

```
HAPIR Operational Workflow
==========================

1. DEPLOYMENT
   - Transport robot to grid edge (exclusion zone boundary)
   - Power on, establish radio link
   - Verify all systems nominal
   - Load probe magazine (if not pre-loaded)

2. NAVIGATION TO FIRST POINT
   - Robot navigates autonomously to first grid position
   - Operator monitors via video feed
   - Robot confirms arrival at waypoint

3. ANCHORING
   - Robot deploys anchor screws (4 corners)
   - Force feedback confirms anchor engagement
   - Verify platform is level and stable

4. PROBE INSERTION
   - Retrieve probe from magazine
   - Position probe over insertion point
   - Begin insertion (2-5 cm/s)
   - Monitor force feedback
   - If force exceeds limit: pause, assess, continue or abort
   - Advance full stroke (1 m), retract actuator
   - Advance probe magazine to expose more probe
   - Repeat until target depth (3 m)

5. PROBE RELEASE
   - Disconnect probe from insertion mechanism
   - Verify probe is stable
   - Retract anchors
   - Log position and status

6. MOVE TO NEXT POINT
   - Navigate to next grid position
   - Repeat steps 3-5

7. COMPLETION
   - Return to grid edge
   - Operator verifies all probes installed
   - Begin HIRT measurement sequence (separate operation)
```

### 6.5 Estimated Cost

| Component | Estimated Cost | Notes |
|-----------|----------------|-------|
| Tracked platform | $30,000-$50,000 | Commercial or custom |
| Insertion mechanism | $15,000-$25,000 | Hydraulic actuator + frame |
| Anchoring system | $10,000-$15,000 | 4x screw anchors + drives |
| Probe magazine | $5,000-$10,000 | Carousel + feed mechanism |
| Control electronics | $15,000-$25,000 | Computer, sensors, radios |
| Power system | $5,000-$10,000 | Batteries + management |
| Operator station | $10,000-$20,000 | Displays, controls, shelter |
| Integration + testing | $30,000-$50,000 | Engineering labor |
| Contingency (25%) | $30,000-$50,000 | Unknowns |
| **TOTAL** | **$150,000-$255,000** | Single prototype |

**Production unit (after development):** $80,000-$120,000

---

## 7. Alternative: Remote-Controlled Rig

### 7.1 Concept Overview

**Simpler near-term approach:** Adapt existing construction equipment for remote operation.

### 7.2 Option A: Remote-Controlled Mini Excavator

**Reference:** [Stanley ROC System](https://www.equipmentworld.com/technology/article/14970837/new-stanley-retrofit-kit-turns-compact-excavators-into-remote-control-diggers-with-xbox-like-controller)

| Aspect | Specification |
|--------|---------------|
| Base machine | Mini excavator (1-5 ton class) |
| Remote kit | Stanley ROC or equivalent |
| Installation time | ~5 hours |
| Control range | 100+ m (line of sight) |
| Attachment | Custom probe insertion attachment |

**Custom Attachment Requirements:**
- Mount on excavator arm end
- Probe magazine (10-15 probes)
- Hydraulic push cylinder (from excavator hydraulics)
- Force feedback to operator

**Advantages:**
- Uses proven, commercially available platform
- Remote kits already exist and are proven
- Lower development cost than purpose-built robot
- Excavator provides excellent anchor mass (1-5 tons)
- Can be used for other tasks (hole preparation, site access)

**Disadvantages:**
- Less precise positioning than robotic system
- Requires skilled operator
- Larger footprint
- Not autonomous

**Estimated Cost:**
- Used mini excavator (1-2 ton): $15,000-$30,000
- Remote control kit: $15,000-$25,000
- Custom insertion attachment: $10,000-$20,000
- Integration: $10,000-$15,000
- **TOTAL:** $50,000-$90,000

### 7.3 Option B: Long-Reach Boom from Safe Distance

**Concept:** Position a vehicle with long-reach boom at exclusion zone boundary, reach into hazard zone.

| Parameter | Requirement |
|-----------|-------------|
| Reach needed | 100-300 m |
| Typical excavator reach | 6-12 m (standard), 20-25 m (long-reach) |
| Gap | 75-290 m |

**Assessment:** Standard long-reach excavators reach ~25 m maximum. This is insufficient for 100-300 m standoff. Would require purpose-built crane-scale equipment, making this impractical.

**Alternative:** Use multiple intermediate robot "waypoints" to relay probes to a simpler insertion robot. Adds significant complexity.

**Conclusion:** Long-reach boom approach is not practical for required standoff distances.

### 7.4 Option C: Remote-Controlled Geoprobe-Style Rig

**Reference:** [Geoprobe 420M Portable](https://geoprobe.com/drilling-rigs/420m-drill-rig)

| Geoprobe Model | Push Capacity | Weight | Remote Capability |
|----------------|---------------|--------|-------------------|
| 420M (portable) | ~2 ton | 450 lb | Could be added |
| 6011DT (track) | ~6 ton | 4,800 lb | Could be added |
| 7822DT | 15 ton | ~12,000 lb | Could be added |

**Concept:** Add remote control capability to small Geoprobe rig.

**Advantages:**
- Purpose-built for probe insertion
- Proven reliability
- Excellent force capacity
- Already has anchoring systems

**Disadvantages:**
- No standard remote option (custom development needed)
- Larger/heavier than necessary for HIRT probes
- Expensive base platforms

**Estimated Cost:**
- Used Geoprobe 6011DT: $40,000-$80,000
- Remote control conversion: $30,000-$50,000
- Integration: $20,000-$30,000
- **TOTAL:** $90,000-$160,000

---

## 8. Safety Considerations

### 8.1 Safe Standoff Distance

| Risk Level | Distance | Rationale |
|------------|----------|-----------|
| Minimum operational | 100 m | Reduced lethal blast radius |
| Standard | 300 m | Matches German evacuation protocols |
| Conservative | 500 m | Large bomb (1000 kg) with margin |

**Recommendation:** Design for 300 m standoff minimum, with 500 m capability.

### 8.2 Emergency Stop Mechanisms

**Multi-Level E-Stop:**

| Level | Mechanism | Effect |
|-------|-----------|--------|
| 1 | Operator button | Halt current operation, hold position |
| 2 | Radio command | All motion stops, brakes engage |
| 3 | Watchdog timeout | Auto-stop if communication lost >5 s |
| 4 | Hardware interlock | Physical relay cuts power to actuators |

**Critical Design Rule:** No single failure should allow uncontrolled robot motion.

### 8.3 Recovery Procedures

**If Robot Gets Stuck:**

| Scenario | Recovery Approach |
|----------|-------------------|
| Mobility stuck (tracks) | Wait for EOD clearance, manual recovery |
| Probe stuck in ground | Release probe, leave in place |
| Anchor stuck | Detach anchor, leave in place |
| Total failure | Wait for EOD clearance, manual recovery |

**Key Principle:** Robot is expendable; never risk personnel to recover stuck robot.

**Design for Abandonment:**
- Robot should be designed to be safely abandoned
- No hazardous materials that could complicate UXO clearance
- Easily cut cables/tethers from safe distance

### 8.4 Partial vs. Full Autonomy Trade-offs

| Autonomy Level | Advantages | Risks |
|----------------|------------|-------|
| Full teleoperation | Operator in loop for all decisions | Fatigue, slow, requires skill |
| Supervised autonomy | Fast execution, operator monitors | Must trust system, intervention delay |
| Full autonomy | No operator fatigue, consistent | System failures may go unnoticed |

**Recommendation:** Supervised autonomy (Level 2-3)
- Robot executes grid pattern autonomously
- Operator monitors video and telemetry
- Operator approves each probe insertion (optional, can be disabled)
- Automatic stop on anomalies

### 8.5 Robot-Specific UXO Risks

| Concern | Mitigation |
|---------|------------|
| Robot triggers UXO | Minimal ground pressure, no impact |
| Insertion triggers UXO | Force limiting, slow insertion rate |
| Vibration triggers UXO | Smooth hydraulic actuation, no percussion |
| Static discharge | Grounding system, anti-static materials |
| Radio triggers UXO | Low probability, but use tested frequencies |

**Design Principle:** Minimize all mechanical inputs to ground. HIRT probes are pushed gently (2-5 cm/s), not driven/hammered.

---

## 9. Cost and Development Path

### 9.1 Development Options Comparison

| Option | Development Cost | Timeline | Risk | Capability |
|--------|-----------------|----------|------|------------|
| A: Purpose-built HAPIR | $150,000-$255,000 | 12-18 months | Medium | Excellent |
| B: Remote mini excavator | $50,000-$90,000 | 3-6 months | Low | Good |
| C: Remote Geoprobe | $90,000-$160,000 | 6-12 months | Medium | Excellent |
| D: Adapt ROGO platform | $100,000-$150,000 | 6-12 months | Medium | Good |

### 9.2 Phased Development Approach

**Phase 1: Proof of Concept (3-6 months, $50,000-$90,000)**
- Remote-controlled mini excavator with custom attachment
- Demonstrate safe probe insertion from standoff distance
- Validate force requirements and insertion procedure
- Gather operational data

**Phase 2: Enhanced Capability (6-12 months, $50,000-$100,000)**
- Add precision positioning (RTK-GPS)
- Develop semi-autonomous operation mode
- Integrate magnetometer monitoring
- Improve probe handling (larger magazine)

**Phase 3: Purpose-Built System (12-24 months, $100,000-$200,000)**
- Design and build optimized platform
- Full autonomous capability
- Integrated probe storage and handling
- Production-ready design

### 9.3 Funding Sources

| Source | Likelihood | Amount | Notes |
|--------|------------|--------|-------|
| UXO remediation contracts | High | Variable | Per-site cost recovery |
| Defense research grants | Medium | $100k-$500k | SBIR, BAA programs |
| Academic partnerships | Medium | In-kind | Student projects, equipment |
| Commercial licensing | Future | Royalties | After proof of concept |

---

## 10. Recommendations

### 10.1 Near-Term Solution (0-12 months)

**Recommended: Remote-Controlled Mini Excavator with Custom Attachment**

| Action | Timeline | Cost |
|--------|----------|------|
| Procure used mini excavator (1-2 ton) | Month 1-2 | $20,000-$30,000 |
| Install Stanley ROC remote kit | Month 2-3 | $15,000-$25,000 |
| Design/build custom insertion attachment | Month 2-4 | $15,000-$25,000 |
| Integration and testing | Month 4-6 | $10,000-$15,000 |
| Field trials | Month 6+ | Operational costs |

**Total Near-Term Investment:** $60,000-$95,000

**Advantages:**
- Fastest path to operational capability
- Lowest risk (uses proven components)
- Provides real-world operational experience
- Can be upgraded incrementally

### 10.2 Long-Term Solution (1-3 years)

**Recommended: Purpose-Built HAPIR Platform**

After gaining experience with the near-term solution, develop a purpose-built robotic platform optimized for HIRT probe insertion:

| Feature | Benefit |
|---------|---------|
| Smaller, lighter | Easier transport, less ground pressure |
| Full autonomy | Reduced operator workload |
| Integrated systems | Optimized for probe insertion |
| Higher reliability | Designed for UXO environments |

**Development Path:**
1. Document lessons learned from mini excavator operations
2. Define detailed requirements based on field experience
3. Seek development funding (grants, contracts)
4. Partner with robotics company or university
5. Iterative design, build, test cycle

### 10.3 Key Success Factors

1. **Start simple:** Use remote-controlled existing equipment first
2. **Prioritize safety:** Never compromise on standoff distance
3. **Gather data:** Document all operational experience
4. **Iterate:** Improve based on real-world feedback
5. **Partner:** Collaborate with EOD community for requirements and testing

### 10.4 Not Recommended

| Approach | Reason |
|----------|--------|
| Drone-based insertion | Insufficient force capacity |
| Long-reach boom | Insufficient reach for safe standoff |
| Full autonomy from start | Too risky without operational experience |
| Rail/gantry system | Requires setup in hazard zone |

---

## 11. Conclusion

Robotic deployment of HIRT probes at UXO sites is **technically feasible** and addresses a genuine unmet need for personnel safety in UXO site characterization.

**Key conclusions:**

1. **Force requirements** (5-20 kN) are achievable with proper platform design and anchoring systems.

2. **Existing technology** (agricultural robots, mining robots, EOD robots) provides valuable reference designs and components.

3. **Near-term solution:** A remote-controlled mini excavator with custom attachment can be deployed within 6 months for ~$60,000-$95,000.

4. **Long-term solution:** A purpose-built robotic platform (HAPIR) would cost ~$150,000-$255,000 to develop but would provide superior performance and reliability.

5. **The primary barrier** is not technical capability but rather development resources. The technology exists; it needs to be integrated and optimized for this specific application.

**Recommendation:** Pursue the phased development approach, starting with the remote-controlled mini excavator to build operational experience while planning for the purpose-built platform.

---

## Sources

### Agricultural Robotics
- [ROGO Soil Sampling Robot](https://rogoag.com/soil-sampling-robot)
- [Purdue University - Autonomous Soil Sampling](https://www.purdue.edu/newsroom/archive/releases/2019/Q3/autonomous-robots-enter-fields-to-collect-precise-soil-samples,-help-farmers-improve-yields,-reduce-environmental-impact,-save-money.html)

### Mining Robotics
- [PERSEPHONE Project](https://www.persephone-mining.eu/)
- [Stinger Robot - Self-Bracing Platform](https://arxiv.org/html/2508.06521v1)
- [MDPI - Autonomous Drilling](https://www.mdpi.com/1424-8220/25/13/3953)

### EOD Robotics
- [L3Harris T7 Robotic System](https://www.l3harris.com/all-capabilities/t7-multi-mission-robotic-system)
- [L3Harris T4 Robotic System](https://www.l3harris.com/all-capabilities/t4-robotic-system)
- [Defense Advancement - EOD Robots](https://www.defenseadvancement.com/suppliers/eod-bomb-disposal-robots/)

### Geotechnical Equipment
- [Geoprobe Systems](https://geoprobe.com/)
- [CPT Testing - Wikipedia](https://en.wikipedia.org/wiki/Cone_penetration_test)
- [USGS - Cone Penetration Testing](https://www.usgs.gov/programs/earthquake-hazards/science/cone-penetration-testing-cpt)

### UGV Platforms
- [Clearpath Warthog](https://clearpathrobotics.com/warthog-unmanned-ground-vehicle-robot/)
- [Robotnik RB-SUMMIT](https://robotnik.eu/products/mobile-robots/rb-summit/)
- [Generation Robots - TEC800](https://www.generationrobots.com/en/404086-tec800-mobile-tracked-robot-ugv.html)

### Remote Control Systems
- [Stanley ROC System](https://www.equipmentworld.com/technology/article/14970837/new-stanley-retrofit-kit-turns-compact-excavators-into-remote-control-diggers-with-xbox-like-controller)

### UXO Safety
- [Wikipedia - Unexploded Ordnance](https://en.wikipedia.org/wiki/Unexploded_ordnance)
- [1st Line Defence - UXO](https://www.1stlinedefence.co.uk/unexploded-ordnance-uxo/)
- [NCBI - Bomb Threat Standoff](https://www.dni.gov/nctc/jcat/references.html)

### Magnetometer Surveys
- [Geometrics - MagArrow II](https://www.geometrics.com/product/magarrow/)
- [SPH Engineering - Drone Magnetometers](https://www.sphengineering.com/integrated-systems/technologies/magnetometer)

### Anchoring Systems
- [Jeffrey Machine - Ground Anchors](https://www.jeffreymachine.com/blog/4-different-kinds-of-ground-anchors-for-drilling-projects)
- [FHWA - Ground Anchors and Anchored Systems](https://www.fhwa.dot.gov/engineering/geotech/pubs/if99015.pdf)

### Cost Estimation
- [Standard Bots - Robot Costs 2026](https://standardbots.com/blog/how-much-do-robots-cost)
- [Design1st - Prototype Costs](https://design1st.com/how-much-do-hw-prototypes-cost/)

---

*Document prepared: 2026-01-19*
*Version: 1.0*
*Classification: HIRT Project - Technical Feasibility Assessment*
