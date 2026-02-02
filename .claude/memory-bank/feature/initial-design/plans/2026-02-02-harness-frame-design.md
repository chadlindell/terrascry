# Pathfinder Harness and Frame System - Technical Specification

## Metadata
- **Date**: 2026-02-02
- **Branch**: feature/initial-design
- **Status**: APPROVED FOR EXECUTION
- **Author**: Claude Code (PLAN mode)
- **Implementation Target**: /development/projects/active/Pathfinder/hardware/cad/frame-design.md

## Overview

This specification defines the physical harness and frame system for the Pathfinder 4-channel fluxgate gradiometer. The design follows the proven "trapeze" configuration used by commercial systems (Bartington Grad601, SENSYS MagWalk) but extends it to 4 sensor pairs for wider swath coverage.

### Design Goals
- **Weight**: Total frame/harness system under 500g (sensors + electronics add ~650g for 1.15kg total)
- **Swath**: 1.5m coverage with 4 sensor pairs at 50cm spacing
- **Ergonomics**: Hands-free operation, weight on shoulders/hips, vibration-isolated sensors
- **Cost**: Frame + harness under $100 total
- **Field-rugged**: IP65 electronics, weather-resistant materials, no fragile components

## 1. Crossbar Assembly

### 1.1 Material Selection

**Primary Choice: Carbon Fiber Tube**

| Material | Weight | Rigidity | Cost | Pros | Cons |
|----------|--------|----------|------|------|------|
| Carbon fiber | 150g | Excellent | $40 | Lightweight, rigid, low thermal expansion | Brittle, conductive (EMI concern) |
| Aluminum 6061 | 280g | Good | $25 | Durable, non-magnetic, machinable | Heavier, thermal expansion |
| Fiberglass | 200g | Good | $30 | Non-conductive, tough, weather-resistant | Less rigid, harder to find |

**Recommendation: Carbon Fiber with EMI Mitigation**

Rationale:
- Weight savings (130g vs aluminum) critical for all-day comfort
- Rigidity prevents sensor motion artifacts
- EMI concern addressed by grounding sheath or running cables inside tube
- Commercial systems (Bartington, SENSYS) successfully use carbon fiber

EMI Mitigation:
- Run sensor cables INSIDE carbon tube (shielding effect)
- Ground carbon fiber to electronics common at center point
- Keep ADC and digital electronics at center (equidistant from sensors)

### 1.2 Dimensions

**Crossbar Specification:**
- **Length**: 2.0 m (allows 1.5m sensor spacing + 25cm overhang each end)
- **Outer Diameter**: 25 mm (comfortable hand grip, adequate stiffness)
- **Wall Thickness**: 2 mm (balance between weight and strength)
- **Weight**: ~150g (Target)

**Specific Product Recommendation:**
- DragonPlate Carbon Fiber Tube: 2m x 25mm OD x 2mm wall
- Alternative: RockWest Composites CF tube (USA supplier)
- Alternative: Easy Composites (EU supplier)
- Budget alternative: 1.25" aluminum EMT conduit from hardware store ($15)

### 1.3 End Protection

**End Caps:**
- 3D-printed PLA or PETG caps with:
  - Press-fit attachment (friction hold)
  - Rounded exterior to prevent snagging on vegetation
  - Optional lanyard attachment point
- Print 4x spares (field-replaceable)

**Alternative:** Heat-shrink tubing + rubber cap from hardware store

### 1.4 Center Attachment Point

**Harness Suspension Mount:**
- Location: Exact center of crossbar (1.0m from each end)
- Method: Aluminum collar with D-ring or carabiner attachment
- Fabrication:
  - Option A: 3D-printed collar with embedded D-ring
  - Option B: Commercial aluminum pipe clamp + D-ring
  - Option C: Hose clamp + shackle (field-expedient)

**Specification:**
- D-ring load rating: 50 kg minimum (system weight <2kg, safety factor >25x)
- Collar width: 40mm (distributes load, prevents tube crushing)
- Attachment method: Quick-release carabiner (not sewn/permanent)

## 2. Sensor Drop Tubes

### 2.1 Material and Dimensions

**Material: PVC Electrical Conduit**

Rationale:
- Non-magnetic (critical for magnetometer accuracy)
- Lightweight (~30g per 50cm tube)
- Weather-resistant
- Globally available
- Easy to cut and drill

**Specification:**
- **Material**: Schedule 40 PVC conduit, 20mm (3/4") diameter
- **Length**: 50cm each (sensor vertical separation)
- **Quantity**: 4 tubes
- **Weight**: ~120g total (Target)

**Source**: Hardware store electrical section

### 2.2 Attachment to Crossbar

**Method: Through-Bolts with Spacers**

Design:
1. Drill perpendicular hole through crossbar at 0.25m, 0.75m, 1.25m, 1.75m from left end
2. Insert M5 or 1/4" bolt through:
   - Crossbar
   - PVC tube (drilled hole at top)
   - Nylon spacer (prevents crushing PVC)
3. Secure with locknut or castle nut + cotter pin

**Alignment:** Tubes hang vertically when crossbar is level (check with plumb bob during assembly)

**Alternative (Tool-Free):** Zip-tie mounting
- Drill pairs of holes in PVC tube
- Thread heavy-duty zip ties through crossbar + PVC
- Advantage: No tools needed for field adjustment
- Disadvantage: Less rigid (may rotate)

### 2.3 Sensor Mounting

**Top Sensors (Crossbar Level):**
- Mount directly to crossbar using:
  - 3D-printed sensor clips (custom-fit to specific fluxgate model)
  - Foam padding to dampen vibration
  - Zip-tie or Velcro strap retention
- Position: Adjacent to drop tube attachment point

**Bottom Sensors (50cm Below):**
- Mount inside PVC tube bottom using:
  - 3D-printed insert plug with sensor cavity
  - Compression fit (tight friction hold)
  - Foam dampening layer
- Height above ground: 15-20cm when harness adjusted correctly

**Sensor Orientation:**
- All sensors aligned with same axis (typically vertical Z-axis)
- Verify with compass or alignment jig during assembly
- Mark "FORWARD" direction on crossbar

### 2.4 Cable Routing

**Path:**
1. Bottom sensor cable runs UP inside PVC tube
2. Exits PVC at crossbar junction
3. Runs INSIDE carbon fiber crossbar to center electronics housing
4. Top sensor cable runs directly along crossbar to center

**Cable Management:**
- Use spiral cable wrap or cable ties every 20cm
- Leave 10cm service loop at each sensor (allows removal without desoldering)
- Label cables at both ends (G1-Top, G1-Bottom, G2-Top, etc.)

**Connector Recommendation:**
- JST-XH or JST-PH connectors (standard RC hobby connectors)
- 3-4 pin: GND, VCC, Signal, (Shield)
- Advantage: Polarized, locking, field-replaceable

## 3. Harness System

### 3.1 Shoulder Strap Configuration

**Design: Backpack-Style Dual Straps**

Based on Bartington Grad601 proven design:

**Components:**
- 2x padded shoulder straps (50mm wide, 10mm foam padding)
- Adjustable length (accommodate users 5'4" to 6'4")
- Sternum strap (prevents straps sliding off shoulders)
- Quick-release buckles (emergency doffing)

**Sourcing Options:**
1. **Salvage Approach (Recommended)**: Remove straps from old/damaged backpack
   - Cost: $0-10 (thrift store)
   - Pros: Proven comfort, already assembled
   - Cons: May need modification

2. **Purchase Complete Harness**: Climbing harness or tool belt harness
   - Cost: $30-50
   - Pros: Professional quality, adjustable
   - Cons: May have unnecessary features

3. **DIY Sewing**: 50mm nylon webbing + foam padding + buckles
   - Cost: $15-25
   - Pros: Custom-fit
   - Cons: Requires sewing machine or hand-stitching

**Recommended Product:** Condor H-Harness or similar tactical vest harness (available military surplus)

### 3.2 Waist Belt

**Purpose:** Stabilize load, prevent crossbar swinging during walking

**Specification:**
- Width: 50mm minimum (distributes load on hips)
- Padding: 5-10mm foam
- Quick-release buckle (same emergency doffing requirement)
- Adjustable size: 28"-48" waist

**Integration:** Waist belt connects to shoulder straps via:
- Fixed connection (sewn or riveted), OR
- Detachable (carabiner or buckle) for separate storage

**Alternative:** Use separate tool belt + attach suspension system to it

### 3.3 Elastic Suspension System

**Purpose:** Isolate sensor vibration from operator walking motion

**Critical Design Element:** This is what separates a usable gradiometer from a noisy mess.

**Specification:**
- Material: Latex rubber bungee cord or silicone shock cord
- Stretch: 50-100% extension under load (2kg system = 4-6kg tension)
- Length: 30-50cm unstretched (allows 15-25cm working motion)
- Attachment: Carabiner or S-hook at both ends

**Configuration Options:**

**Option A: Single-Point Suspension (Simplest)**
```
     Shoulder straps
           |
      [D-ring back]
           |
       [Bungee 40cm]
           |
       [Carabiner]
           |
      [Crossbar center]
```
Pros: Simple, minimal parts
Cons: Crossbar may rotate (sensors tilt)

**Option B: Two-Point Suspension (Recommended)**
```
  Shoulder straps
      |        |
  [D-ring] [D-ring]
      |        |
   [Bungee] [Bungee]
      |        |
       \      /
        \    /
      [Spreader bar]
            |
        [Carabiner]
            |
      [Crossbar center]
```
Pros: Prevents rotation, more stable
Cons: Slightly more complex

**Spreader Bar:** 20cm aluminum or wood dowel, prevents bungee cords from tangling

**Bungee Specification:**
- 6mm diameter shock cord
- 40cm length (unstretched)
- Loop or carabiner at each end
- Purchase: Hardware store, marine supply, or Amazon
- Cost: $5 for 2m length

### 3.4 Quick-Release System

**Requirements:**
- Operator can don/doff harness in under 30 seconds
- Emergency release (entanglement, stumble) under 2 seconds
- No tools required

**Implementation:**
- Crossbar-to-bungee: Climbing carabiner (twist-lock or snap-gate)
- Shoulder straps: Side-release buckles (same as backpack buckles)
- Waist belt: Side-release or cobra buckle

**Carabiner Specification:**
- Type: Aluminum snap-gate or twist-lock
- Load rating: 5 kN minimum (1100 lb)
- Gate opening: 15mm minimum (fits over D-ring and bungee loop)
- Cost: $3-5 each, purchase 4x (spares)

**Recommended Product:** Black Diamond Positron or similar climbing carabiner

### 3.5 Height Adjustment

**Goal:** Set bottom sensors 15-20cm above ground for typical operator (5'6" to 6'2")

**Adjustment Method:**
1. **Shoulder Strap Length:** Primary adjustment
   - Slide buckles or ladder-lock adjusters
   - Shorten straps = sensors higher off ground
   - Lengthen straps = sensors lower

2. **Bungee Attachment Point:** Secondary adjustment
   - Move D-ring position on harness back panel
   - Sew multiple D-ring positions (S/M/L settings)

**Setup Procedure:**
1. Don harness
2. Attach crossbar via carabiner
3. Stand upright on level ground
4. Measure bottom sensor height with ruler
5. Adjust shoulder straps until 15-20cm clearance
6. Mark strap position with permanent marker
7. Test walking motion (sensors should not touch ground on normal stride)

**Terrain Adaptation:**
- Tall grass/brush: Shorten straps (raise sensors 25-30cm)
- Smooth ground: Lengthen straps (lower sensors to 10-15cm for better sensitivity)

## 4. Electronics Housing

### 4.1 Enclosure Selection

**Requirements:**
- IP65 rating minimum (dust-tight, water-resistant)
- Size: Accommodate Arduino Nano + 2x ADS1115 + GPS + SD card + battery
- Internal volume: ~10 x 8 x 5 cm minimum
- Weight: <200g
- Cost: <$20

**Recommended Products:**
1. **Outdoor junction box** (electrical aisle at hardware store)
   - Example: Carlon E989N junction box
   - Cost: $8-12
   - Pros: Cheap, available everywhere, gasket-sealed
   - Cons: Utilitarian appearance

2. **Plastic project box with gasket**
   - Example: Hammond 1554 series
   - Cost: $12-18
   - Pros: Clean appearance, multiple sizes
   - Cons: May need to add gasket

3. **Waterproof case**
   - Example: Pelican 1020 Micro Case
   - Cost: $20-25
   - Pros: Premium protection, pressure equalization valve
   - Cons: Expensive for this application

**Selected Recommendation:** Hammond 1554K or equivalent (~$15)

### 4.2 Internal Layout

**Component Arrangement (Top View):**
```
+---------------------------+
|  [GPS Module]             |
|                 [Buzzer]  |
|  [Arduino Nano]           |
|  [ADS1115] [ADS1115]      |
|  [SD Card Module]         |
|                           |
|  [Power Switch] [LED]     |
+---------------------------+
```

**Mounting:**
- Standoffs or adhesive foam pads
- Keep GPS module near top surface (antenna performance)
- Route cables to minimize crosstalk (power away from signal)

**Ventilation:**
- Gore-Tex vent (pressure equalization, prevents condensation)
- Alternative: Small drill hole + adhesive vent membrane
- Cost: $2-5 for vent

### 4.3 Mounting Location Options

**Option A: Belt-Mount (Recommended)**

Attach enclosure to waist belt using:
- Velcro straps
- MOLLE clips (if using tactical harness)
- Belt loop + carabiner

Pros:
- Easy access to power switch and SD card
- Weight on hips (better than shoulders)
- Good GPS antenna position (unobstructed sky view)
- Heat from electronics away from body

Cons:
- Longer cable runs to sensors (~1.5m)
- May interfere with sitting/kneeling

**Option B: Crossbar-Mount**

Attach enclosure to crossbar center using:
- Hose clamps
- 3D-printed bracket
- Velcro straps

Pros:
- Shorter cable runs to sensors
- Clean integration

Cons:
- Harder to access controls
- Weight at worst location (shoulder level)
- GPS antenna may have partial sky obstruction

**Recommendation:** Belt-mount for field usability

### 4.4 Cable Entry

**Method:**
- Drill cable entry holes in enclosure bottom
- Use strain relief grommets or cord grips
  - PG7 cable glands (common size for 6mm cable)
  - Cost: $1-2 each, need 2x (sensor cable bundle, power cable)
- Seal with silicone after installation (field-removable seal)

**Cable Organization:**
- All 8 sensor cables bundle together from crossbar to enclosure
- Spiral wrap or braided sleeve
- Drip loop at enclosure entry (prevents water ingress)

## 5. Detailed Parts List

### 5.1 Crossbar Assembly

| Part | Specification | Qty | Source | Est. Cost |
|------|---------------|-----|--------|-----------|
| Carbon fiber tube | 2.0m x 25mm OD x 2mm wall | 1 | DragonPlate, RockWest Composites | $40 |
| Aluminum tube (alt) | 1.25" EMT conduit, 2m | 1 | Hardware store | $15 |
| End caps | 3D-printed PLA, 25mm ID | 4 | Print yourself or Shapeways | $5 |
| Center mount D-ring | Aluminum, 50kg rating | 1 | Amazon, hardware store | $3 |
| Pipe clamp collar | Aluminum, 25mm ID | 1 | Hardware store | $4 |

**Crossbar Subtotal: $52 (carbon) or $27 (aluminum)**

### 5.2 Sensor Drop Tubes

| Part | Specification | Qty | Source | Est. Cost |
|------|---------------|-----|--------|-----------|
| PVC conduit | 3/4" Schedule 40, 50cm lengths | 4 | Hardware store electrical section | $8 |
| Mounting bolts | M5 x 60mm stainless, with locknuts | 4 | Hardware store | $4 |
| Nylon spacers | 5mm ID x 10mm OD x 25mm long | 4 | Hardware store or Amazon | $3 |
| Sensor clips | 3D-printed, custom fit to sensor | 8 | Print yourself | $8 |
| Foam padding | 5mm neoprene or EVA foam | 1 sheet | Craft store | $5 |

**Drop Tube Subtotal: $28**

### 5.3 Harness System

| Part | Specification | Qty | Source | Est. Cost |
|------|---------------|-----|--------|-----------|
| Shoulder straps | Salvaged from backpack, 50mm padded | 1 set | Thrift store, old backpack | $0-10 |
| Waist belt | 50mm webbing with buckle | 1 | Same as straps, or separate | Included |
| Bungee cord | 6mm shock cord, 1m length | 1 | Hardware store | $3 |
| Carabiners | Aluminum snap-gate, 5kN rating | 4 | REI, Amazon, climbing shop | $12 |
| D-rings | 25mm steel or aluminum | 2 | Fabric/craft store | $2 |
| Spreader bar | 20cm aluminum rod or wood dowel | 1 | Hardware store | $2 |

**Harness Subtotal: $19-29**

### 5.4 Electronics Housing

| Part | Specification | Qty | Source | Est. Cost |
|------|---------------|-----|--------|-----------|
| Enclosure | Hammond 1554K or equivalent, IP65 | 1 | DigiKey, Mouser, Amazon | $15 |
| Cable glands | PG7 nylon, IP68 rated | 2 | Amazon | $4 |
| Gore-Tex vent | Pressure equalization membrane | 1 | DigiKey | $3 |
| Velcro straps | 25mm hook-and-loop, 30cm | 2 | Hardware store | $4 |
| Standoffs | M3 nylon, 10mm | 8 | Amazon | $2 |

**Housing Subtotal: $28**

### 5.5 Cables and Connectors

| Part | Specification | Qty | Source | Est. Cost |
|------|---------------|-----|--------|-----------|
| Sensor cable | 22 AWG 4-conductor shielded | 15m | Amazon, Monoprice | $15 |
| JST-XH connectors | 4-pin, male+female pairs | 10 sets | Amazon | $8 |
| Spiral cable wrap | 10mm diameter, 2m | 1 | Amazon | $5 |
| Heat shrink tubing | Assorted sizes | 1 kit | Hardware store | $5 |
| Cable ties | 100mm, UV-resistant black | 50 | Hardware store | $3 |
| Label tape | Brother P-Touch or equivalent | 1 roll | Office supply | $8 |

**Cable Subtotal: $44**

### 5.6 Tools Required (One-Time Purchase)

| Tool | Purpose | Est. Cost |
|------|---------|-----------|
| Drill + bits | Hole drilling (carbon, PVC, enclosure) | $30-60 |
| Hacksaw or pipe cutter | Cutting PVC, aluminum | $10-15 |
| Soldering iron + solder | Connector assembly | $15-30 |
| Crimping tool (optional) | JST connector crimping | $15-25 |
| Heat gun (optional) | Heat shrink, forming plastic | $15-20 |
| 3D printer access (optional) | Mounts and clips | $0-300 |

**Estimated tool cost if starting from scratch: $85-150**
**Most users already have drill and saw, reducing to $15-30**

### 5.7 Complete Build Cost Summary

| Category | Cost Range |
|----------|------------|
| Crossbar assembly | $27-52 |
| Sensor drop tubes | $28 |
| Harness system | $19-29 |
| Electronics housing | $28 |
| Cables and connectors | $44 |
| **Frame/Harness Total** | **$146-181** |
| | |
| Sensors (8x fluxgate) | $480-640 |
| Electronics (Arduino, ADC, GPS, etc.) | $90-120 |
| **Complete System Total** | **$716-941** |

**Target met:** Under $1000 for complete system
**Margin:** $59-284 remaining for contingency/shipping

## 6. Assembly Procedure

### 6.1 Crossbar Preparation

1. Cut carbon fiber tube to 2.0m length (if not pre-cut)
   - Use carbide blade or dremel with cutoff wheel
   - Wear dust mask (carbon fiber dust is irritant)
   - Smooth cut edges with fine sandpaper

2. Mark sensor mounting positions:
   - Left end = 0cm
   - Sensor 1: 25cm from left end
   - Sensor 2: 75cm from left end
   - Center: 100cm (harness attachment)
   - Sensor 3: 125cm from left end
   - Sensor 4: 175cm from left end
   - Right end = 200cm

3. Drill mounting holes:
   - 5mm diameter at each sensor position
   - Perpendicular to tube axis
   - Use drill press or drilling guide for accuracy
   - Deburr holes

4. Install center mount:
   - Position pipe clamp at 100cm mark
   - Tighten screws (do not overtighten - carbon can crack)
   - Attach D-ring to clamp
   - Verify D-ring can rotate freely

5. Install end caps:
   - Press-fit or glue 3D-printed caps
   - Verify smooth edges (no snagging hazard)

### 6.2 Sensor Drop Tube Assembly

1. Cut PVC conduit:
   - 4x pieces at 50cm length
   - Use pipe cutter or hacksaw
   - Deburr edges

2. Drill mounting holes:
   - 5.5mm hole at 2cm from top of each tube
   - Align hole perpendicular to tube axis

3. Create sensor mounts:
   - 3D-print sensor clips (4x top, 4x bottom)
   - Test-fit sensors in clips
   - Add foam padding if needed

4. Install bottom sensor mounts:
   - Insert bottom sensor mount into PVC tube bottom
   - Should friction-fit snugly
   - Test by inverting tube (mount should not fall out)

### 6.3 Final Assembly

1. Attach drop tubes to crossbar:
   - Insert M5 bolt through crossbar hole
   - Pass through PVC tube hole
   - Add nylon spacer on opposite side
   - Tighten locknut (firm but not crushing PVC)
   - Verify tube hangs vertically (use plumb bob)

2. Install top sensor mounts:
   - Position clip on crossbar adjacent to drop tube
   - Secure with zip tie or Velcro strap
   - Verify alignment (all sensors parallel)

3. Route sensor cables:
   - Bottom sensor cable UP through PVC tube
   - Exit at crossbar junction
   - Run along/inside crossbar to center
   - Top sensor cable runs along crossbar
   - Bundle all cables with spiral wrap
   - Leave 10cm service loop at each sensor

4. Connect to electronics:
   - Run cable bundle from crossbar center to enclosure
   - Install cable gland in enclosure
   - Thread cables through gland
   - Connect to ADC inputs (label connections)
   - Tighten cable gland

### 6.4 Harness Integration

1. Prepare harness:
   - If salvaging backpack straps, remove from pack
   - Sew or rivet D-rings to back panel (two points, 15cm apart)
   - Install waist belt if separate

2. Attach suspension system:
   - Cut bungee cord: 2x 40cm lengths
   - Tie loop in each end, OR install carabiners
   - Attach spreader bar between bungee cords
   - Attach spreader to crossbar center D-ring via carabiner
   - Attach bungee tops to harness D-rings

3. Test suspension:
   - Hang harness from sturdy hook
   - Attach crossbar (without sensors yet)
   - Verify crossbar hangs level
   - Add weight (2kg) to simulate full system
   - Check bungee extension (should stretch 15-25cm)

### 6.5 Electronics Installation

1. Mount components in enclosure:
   - Use standoffs or foam adhesive pads
   - GPS module near top
   - Arduino + ADC boards in center
   - Battery at bottom (lowest center of gravity)

2. Wire connections:
   - Solder sensor connections to ADC inputs
   - GPS to Arduino serial pins
   - SD card to Arduino SPI pins
   - Power distribution from battery

3. Install enclosure on harness:
   - Belt-mount: Velcro straps to waist belt
   - Crossbar-mount: Hose clamps to center section

4. Final cable dressing:
   - Secure all cables with zip ties
   - Create drip loop at enclosure entry
   - Verify no cables interfere with suspension motion

### 6.6 Quality Checks

Before first field use:

- [ ] All bolts tight (check with wrench)
- [ ] No sharp edges or snagging hazards
- [ ] Crossbar hangs level when suspended
- [ ] All sensors aligned in same orientation
- [ ] Cables labeled at both ends
- [ ] Enclosure seals intact (no gaps)
- [ ] Power switch accessible
- [ ] SD card removable without tools
- [ ] Carabiners close securely
- [ ] Bungee cords show no wear/fraying
- [ ] Emergency release functions (test doffing)

## 7. Testing and Validation

### 7.1 Structural Tests

**Static Load Test:**
1. Suspend crossbar from center mount
2. Hang 5kg weight from each end (10kg total)
3. Measure deflection at ends (should be <5cm)
4. Hold for 10 minutes
5. Inspect for cracks, bending, or joint slippage

**Dynamic Test:**
1. Wear harness with full sensor load
2. Walk 100m at normal survey pace
3. Monitor for:
   - Crossbar rotation or swinging
   - Sensor collision with ground
   - Strap slippage
   - Excessive vibration

**Pass Criteria:**
- No structural failure
- Comfortable for 15-minute continuous walk
- Sensors maintain alignment

### 7.2 Sensor Alignment Verification

**Alignment Test:**
1. Place assembled system in open area
2. Orient crossbar north-south
3. Read all 8 sensors simultaneously
4. All sensors should read ~50,000 nT (Earth's field, varies by location)
5. Rotate system 180 degrees
6. Verify readings reverse sign (gradiometer should show minimal change)

**Pass Criteria:**
- All sensors within 500 nT of each other (without targets present)
- Gradient readings <50 nT/m (background noise floor)

### 7.3 Vibration Isolation Test

**Walking Motion Test:**
1. Wear system, stand still
2. Record 10 seconds of sensor data (baseline noise)
3. Walk at normal pace
4. Record 60 seconds of sensor data
5. Compare noise levels (walking vs. stationary)

**Pass Criteria:**
- Walking noise <2x stationary noise
- No aliasing or motion artifacts
- Gradiometer cancellation still effective

### 7.4 Field Trial

**Operational Test:**
1. Survey known test area (buried calibration targets)
2. Record data for 30 minutes continuous operation
3. Monitor for:
   - Operator fatigue
   - Component failures
   - Data quality issues
   - Ergonomic problems

**Pass Criteria:**
- Detect known targets
- No component failures
- Operator reports acceptable comfort level
- Data recoverable from SD card

## 8. Maintenance and Field Repairs

### 8.1 Pre-Survey Inspection

Before each survey:
- [ ] Check all bolts and clamps (hand-tight)
- [ ] Inspect bungee cords for wear
- [ ] Verify carabiners close and lock
- [ ] Test power switch and LED indicator
- [ ] Check SD card presence and free space
- [ ] Inspect cable connections (no loose connectors)
- [ ] Verify sensor alignment (visual check)

### 8.2 Common Failure Modes

| Problem | Likely Cause | Field Fix |
|---------|--------------|-----------|
| Crossbar bends/sags | Overloaded, material fatigue | Support with hand, complete survey, replace tube |
| Drop tube rotates | Loose mounting bolt | Tighten bolt with multi-tool |
| Sensor reads zero/max | Loose cable connection | Reseat connector, check continuity |
| GPS no fix | Antenna obstruction | Reposition enclosure, check antenna cable |
| Excessive noise | Vibration coupling | Check bungee tension, add foam damping |
| Harness strap breaks | Wear, overload | Use backup strap, safety pin, or duct tape |

### 8.3 Field Repair Kit

Recommended items to carry:
- Multi-tool with screwdriver bits
- Spare carabiners (2x)
- Zip ties (10x)
- Duct tape roll
- Bungee cord (1m spare)
- JST connectors (2 sets)
- Wire strippers
- Electrical tape
- Spare batteries
- Spare SD card

Total weight: ~500g
Storage: Small belt pouch or backpack pocket

### 8.4 End-of-Season Maintenance

After field season or every 50 hours operation:
1. Disassemble harness from frame
2. Inspect carbon fiber tube for cracks (visual + tap test)
3. Check all bolt holes for elongation or cracking
4. Wash harness straps (remove sweat salts)
5. Test bungee cords (replace if >25% loss of elasticity)
6. Open enclosure, check for condensation or corrosion
7. Re-seal cable glands with fresh silicone
8. Re-calibrate sensors (see firmware documentation)
9. Update firmware if available
10. Store in dry location, crossbar supported at both ends

## 9. Design Variations

### 9.1 Two-Sensor Configuration (Budget Version)

For <$500 build:
- Crossbar: 1.0m aluminum EMT conduit
- Sensors: 2 pairs only (50cm spacing)
- Harness: Single-shoulder strap (game bag style)
- Performance: 0.5m swath, 50% coverage rate

### 9.2 Extended Configuration (Research Grade)

For premium performance:
- Crossbar: 2.5m carbon fiber
- Sensors: 5-6 pairs (40cm spacing)
- Harness: Load-bearing frame (similar to backpacking pack)
- GPS: RTK-corrected (cm-level positioning)
- Performance: 2.0m swath, comparable to commercial systems

### 9.3 Modular Configuration

Crossbar in sections:
- 3x 0.75m tubes with ferrule connectors
- Advantage: Fits in vehicle, airline-checkable luggage
- Disadvantage: Joints may introduce flex/vibration

### 9.4 Cart-Mounted (Non-Harness)

For paved/flat terrain:
- Mount crossbar on wheeled cart (garden cart, hand truck)
- No harness needed
- Advantage: Zero operator fatigue, higher sensor count possible
- Disadvantage: Limited to smooth terrain, slower setup

## 10. Safety Considerations

### 10.1 Operator Safety

**Physical Hazards:**
- Tripping hazard: Suspended crossbar may catch on obstacles
  - Mitigation: Quick-release carabiner, operator training
- Falling hazard: Loss of balance while harnessed
  - Mitigation: Waist belt stabilization, practice in safe area
- Entanglement: Cables or bungee cords snag on vegetation
  - Mitigation: Cable management, avoid dense brush

**Ergonomic Hazards:**
- Repetitive motion: Walking surveys for hours
  - Mitigation: 10-minute breaks every hour, hydration
- Load distribution: Poor harness adjustment causes back/shoulder pain
  - Mitigation: Proper fitting procedure, adjustable straps

**Environmental Hazards:**
- Heat stress: Electronics + body heat in enclosure
  - Mitigation: Belt-mount configuration, ventilation
- Cold stress: Metal/carbon conduct heat away from body
  - Mitigation: Insulated gloves if hand-guiding crossbar
- Lightning: Carbon fiber crossbar may attract strikes
  - Mitigation: Do not survey during thunderstorms

### 10.2 Equipment Safety

**Electrical:**
- Low voltage system (7.4V LiPo)
- Risk of short circuit if enclosure floods
  - Mitigation: IP65 enclosure, regular seal inspection

**Mechanical:**
- Carbon fiber splinters if tube breaks
  - Mitigation: Wear gloves during assembly, inspect for cracks
- Falling crossbar if harness fails
  - Mitigation: Regular inspection, safety factor in component selection

### 10.3 Environmental Safety

**Contamination:**
- Do not survey in contaminated areas (chemical, biological, radiological)
- If surveying UXO areas, follow EOD safety protocols
- Do not disturb protected archaeological sites without permits

**Impact:**
- Sensors do not contact ground (non-invasive)
- No digging or disturbance
- Follow local regulations for site access

## 11. Success Criteria

This design is successful when:

1. **Performance:**
   - [ ] Total system weight under 1.5 kg
   - [ ] 1.5m swath coverage achieved
   - [ ] Comfortable for 30+ minute continuous surveys
   - [ ] Detects known calibration targets
   - [ ] Data quality comparable to commercial systems

2. **Cost:**
   - [ ] Frame + harness under $200
   - [ ] Complete system under $950

3. **Usability:**
   - [ ] Assembly time under 8 hours
   - [ ] Don/doff time under 60 seconds
   - [ ] Field-repairable with basic tools
   - [ ] No specialized fabrication required

4. **Reliability:**
   - [ ] Survives 10-hour field day without failure
   - [ ] Weather-resistant (light rain, dust)
   - [ ] 50+ hour service life before major maintenance

## 12. Implementation Steps

### Step 1: Procurement (Execute First)
- Purchase all parts from Section 5
- Verify dimensions match specification
- Organize by subsystem (crossbar, tubes, harness, electronics)

### Step 2: Crossbar Assembly
- Cut and prepare carbon fiber tube
- Install end caps and center mount
- Mark and drill sensor mounting holes

### Step 3: Drop Tube Fabrication
- Cut PVC conduit to length
- Drill mounting holes
- 3D-print or fabricate sensor mounts

### Step 4: Integration
- Attach drop tubes to crossbar
- Install sensors in mounts
- Route and secure cables

### Step 5: Harness Construction
- Prepare shoulder straps and waist belt
- Install suspension system
- Attach to crossbar

### Step 6: Electronics Integration
- Assemble electronics in enclosure
- Mount enclosure to harness
- Connect sensor cables

### Step 7: Testing
- Structural load test
- Sensor alignment verification
- Walking motion test
- Field trial with known targets

### Step 8: Documentation
- Photograph assembly process
- Create as-built drawings
- Write user manual section
- Document any deviations from spec

## 13. Documentation Deliverable

Create `/development/projects/active/Pathfinder/hardware/cad/frame-design.md` with:

1. **Overview** (Summary of this specification)
2. **Component Selection** (Sections 1-4 in user-friendly format)
3. **Parts List** (Section 5 as formatted tables)
4. **Assembly Instructions** (Section 6 with photos/diagrams if available)
5. **Testing Procedures** (Section 7)
6. **Maintenance Guide** (Section 8)
7. **Design Variations** (Section 9)
8. **Safety** (Section 10)

Format: Markdown, following Pathfinder documentation style (see CLAUDE.md)

## 14. Open Questions for Future Resolution

1. **3D-Printed Parts:** Should we provide STL files, or just dimensional specifications?
   - Recommendation: Provide both STL + dimensions for hand fabrication

2. **Harness Sizing:** Create S/M/L/XL variants, or fully adjustable?
   - Recommendation: Fully adjustable with sizing guide

3. **Cable Connectors:** JST vs. military-spec vs. terminal blocks?
   - Recommendation: JST for prototype, upgrade path documented

4. **Crossbar Material:** Final decision carbon vs. aluminum?
   - Recommendation: Specify carbon as primary, aluminum as budget alternative

5. **Electronics Enclosure:** Custom 3D-printed vs. commercial?
   - Recommendation: Commercial for waterproofing, custom internal mounts

## 15. References

- Bartington Grad601 Manual: [https://www.bartingtondownloads.com/wp-content/uploads/OM1800.pdf](https://www.bartingtondownloads.com/wp-content/uploads/OM1800.pdf)
- Berkeley Grad601 Field Guide: [https://arf.berkeley.edu/files/attachments/books/Bartington_Grad601_2_Setup_And_Operation_1.pdf](https://arf.berkeley.edu/files/attachments/books/Bartington_Grad601_2_Setup_And_Operation_1.pdf)
- SENSYS MagWalk: [https://sensysmagnetometer.com/products/magwalk-magnetometer-survey-kit/](https://sensysmagnetometer.com/products/magwalk-magnetometer-survey-kit/)
- Commercial harness examples: Climbing harnesses, tool belts, tactical vests

---

**Plan Status:** READY FOR EXECUTION
**Next Phase:** `/riper:execute` to create frame-design.md document
