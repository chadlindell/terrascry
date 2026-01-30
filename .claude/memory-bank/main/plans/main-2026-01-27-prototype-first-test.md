# HIRT Rapid Prototyping and First-Test Plan

**Date:** 2026-01-27
**Branch:** main
**Status:** APPROVED - DOCUMENTATION COMPLETE
**RIPER Phase:** EXECUTE
**Approved:** 2026-01-27

---

## 1. Executive Summary

This plan defines a streamlined workflow for building and testing a **Minimum Viable Prototype (MVP)** of the HIRT sensor probe. The goal is to validate the design's ability to collect meaningful data with minimal time investment, enabling rapid iteration if changes are needed.

**Approach:** "Fail-Fast Prototype" - prioritize speed to first data over production quality.

**Target:** First functional data within 1-2 days of starting, not weeks.

---

## 2. Scope Definition

### 2.1 In Scope
- Single-segment MIT-only prototype probe
- Simplified bench wiring (no production harness)
- Quick functional tests using available equipment
- First detection of metal target
- Documentation of results for iteration decisions

### 2.2 Out of Scope (Deferred to Later)
- ERT electrode integration (Phase 2)
- Multi-segment assembly (Phase 2)
- Production wiring harness (Phase 3)
- Waterproofing/field deployment (Phase 3)
- Full calibration procedures (Phase 3)
- Zone box architecture (Phase 4)

### 2.3 Success Criteria
1. Coil measures 1.5 mH ±20% (relaxed tolerance for prototype)
2. Q factor ≥15 at 10 kHz (relaxed from production spec of ≥30)
3. Visible signal change when metal object approaches coil
4. Threaded assembly screws together without breaking
5. Design decision made: proceed, modify, or redesign

---

## 3. Bill of Materials (Prototype)

### 3.1 3D Printed Parts (Minimum Set)

| Part | Quantity | STL File | Notes |
|------|----------|----------|-------|
| Male rod cap | 1 | `male_rod_cap_threads_test.stl` or single from 4x | Epoxy into rod |
| Sensor body (dual-female) | 1 | `sensor_body_single_test.stl` or single from 4x | Houses coil |
| Probe tip | 1 | Single from `probe_tip_4x.stl` | Bottom terminator |

**Alternative:** Use `_tapready` versions if using tap/die approach.

### 3.2 Mechanical Components

| Component | Quantity | Specification | Source |
|-----------|----------|---------------|--------|
| Fiberglass tube | 1 | 16mm OD × 12mm ID × 300mm | McMaster or existing stock |
| Ferrite rod | 1 | NiZn Ø8mm × 100mm | Fair-Rite or Amazon |
| Magnet wire | 15m | 34 AWG enameled | Existing stock |
| Epoxy | Small amount | 2-part, 5-min or 24-hr | Hardware store |
| O-ring | 1 | AS568-014 (12mm) | McMaster or Amazon |

### 3.3 Wiring (Simplified)

| Component | Quantity | Specification | Notes |
|-----------|----------|---------------|-------|
| Test leads | 2 | Alligator clip or DuPont female | To connect coil to test equipment |
| Hookup wire | 0.5m | 22-26 AWG stranded | Solder to coil tails |

### 3.4 Test Equipment Required

| Equipment | Purpose | Substitute if Unavailable |
|-----------|---------|---------------------------|
| LCR meter | Measure L, Q | Multimeter (inductance mode) or calculate from resonance |
| Function generator | Drive coil | Arduino + DDS module (~$10) |
| Oscilloscope | View response | Soundcard oscilloscope software |
| Multimeter | Continuity, resistance | Required - no substitute |

### 3.5 Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Soldering iron | Wire connections | Fine tip preferred |
| Flush cutters | Remove scaffolding | Or side cutters |
| Sandpaper (400 grit) | Deburr rod ends | |
| M12×1.75 tap | Thread female parts | If using blanks |
| M12×1.75 die | Thread male parts | If using blanks |

---

## 4. Implementation Steps

### Phase 0: Preparation (Parallel Tasks)

#### Step 0.1: Inventory Check
**Action:** Verify all materials are on hand before starting.
- [ ] Ferrite rod (NiZn preferred, MnZn acceptable)
- [ ] 34 AWG magnet wire (minimum 15m)
- [ ] Fiberglass tube segment
- [ ] 3D printer operational with PETG/ASA loaded
- [ ] Test equipment available

**If missing:** Order immediately or identify substitutes.
**Duration:** 15 minutes

#### Step 0.2: Equipment Setup
**Action:** Prepare test equipment for immediate use after assembly.
- [ ] LCR meter: verify battery, test with known inductor if available
- [ ] Function generator: set to 10 kHz sine, 1-3V amplitude
- [ ] Oscilloscope: set to AC coupling, 1V/div, 10μs/div timebase
- [ ] Multimeter: set to resistance mode, verify leads

**Duration:** 15 minutes

---

### Phase 1: 3D Printing (Steps 1.1-1.3)

#### Step 1.1: Select Print Strategy
**Decision Point:** Choose threading approach.

| Option | When to Use | Print Time |
|--------|-------------|------------|
| **A: Direct threads** | First attempt, no tap/die available | ~4-5 hrs |
| **B: Blanks + tap/die** | Have tap/die, want stronger threads | ~3-4 hrs |
| **C: Thread test first** | Uncertain about threads, want to validate | +1 hr |

**Recommended for first prototype:** Option A (direct threads) unless tap/die already in hand.

#### Step 1.2: Print Parts
**Action:** Print minimum part set.

**Settings (Bambu A1 Mini or equivalent):**
- Material: PETG (preferred) or ASA
- Layer height: 0.12mm
- Infill: 100% solid
- Walls: 6 loops
- Supports: OFF (use built-in scaffolding)
- Speed: 50mm/s outer walls

**Parts to print:**
1. 1× sensor body (single or cut from 4x array)
2. 1× male rod cap (single or cut from 4x array)
3. 1× probe tip

**If printing singles:** Modify OpenSCAD to render single parts, or print full 4x and use one.

**Duration:** 4-6 hours (can run overnight)

#### Step 1.3: Post-Process Printed Parts
**Action:** Prepare parts for assembly.
1. Remove from build plate (let cool first)
2. Snip scaffolding with flush cutters
3. If using tap/die approach:
   - Run M12×1.75 tap through female threads
   - Run M12×1.75 die over male threads
4. Test-fit threads by hand (should screw together smoothly)
5. If threads bind: chase with tap/die, or sand lightly

**Pass Criteria:** Parts thread together by hand with moderate resistance.

**Duration:** 30-60 minutes

---

### Phase 2: Coil Winding (Steps 2.1-2.4)

#### Step 2.1: Prepare Ferrite Core
**Action:** Ready core for winding.
1. Inspect for cracks or chips (reject if damaged)
2. Clean with isopropyl alcohol
3. Mark winding zone: center 50mm of core
4. Optionally wrap with single layer of Kapton tape (insulation)

**Duration:** 10 minutes

#### Step 2.2: Wind Coil
**Action:** Wind 300 turns of 34 AWG wire.

**Procedure:**
1. Leave 150mm tail at start, secure with tape
2. Wind single-layer, close-wound (turns touching)
3. Maintain consistent light tension
4. Count turns (use tally counter or marks every 50)
5. Target: 300 turns in ~50mm winding length
6. Leave 150mm tail at end, secure with tape

**Tips:**
- Wind slowly and carefully - this is the critical component
- If wire breaks, splice by twisting ends + solder, or start over
- Winding should be neat but perfection not required for prototype

**Duration:** 45-90 minutes (depends on experience)

#### Step 2.3: Measure Coil (Smoke Test)
**Action:** Verify basic coil parameters before proceeding.

| Test | Method | Pass Criteria |
|------|--------|---------------|
| Continuity | Multimeter, resistance mode | <50 Ω |
| DC Resistance | Multimeter | 5-15 Ω typical |
| Inductance | LCR meter @ 1 kHz | 1.0-2.0 mH |
| Q Factor | LCR meter @ 10 kHz | ≥15 (prototype) |

**If coil fails:**
- Continuity fail: Check for broken wire, re-wind
- Inductance low: Add turns (20-30 more)
- Inductance high: Remove turns
- Q low: Check for shorted turns, loose windings

**Duration:** 10 minutes

#### Step 2.4: Secure Coil
**Action:** Fix windings in place.
1. Apply thin layer of clear nail polish, super glue, or epoxy over windings
2. Let cure (5 min for CA glue, overnight for epoxy)
3. Trim wire tails to ~100mm
4. Strip 5mm insulation from ends (careful - wire is fragile)
5. Tin stripped ends with solder

**Duration:** 15 minutes + cure time

---

### Phase 3: Mechanical Assembly (Steps 3.1-3.3)

#### Step 3.1: Prepare Rod Segment
**Action:** Ready fiberglass tube for assembly.
1. Cut tube to 300mm length (if not pre-cut)
2. Deburr both ends with sandpaper
3. Clean inside and out with IPA
4. Dry completely

**Duration:** 15 minutes

#### Step 3.2: Install Male Cap
**Action:** Epoxy male rod cap into one end of tube.
1. Mix small amount of 2-part epoxy
2. Apply epoxy to outer surface of male cap's insertion shank
3. Insert into rod end, ensuring flush fit
4. Wipe excess epoxy
5. Allow to cure (5-min epoxy: 1 hour minimum; 24-hr epoxy: overnight)

**Note:** For quick prototype, can skip epoxy and friction-fit. Mark "NOT WATERPROOF."

**Duration:** 10 minutes + cure time

#### Step 3.3: Dry-Fit Assembly
**Action:** Test-assemble without permanent bonding.
1. Insert coil (on ferrite rod) into sensor body cavity
2. Route coil wires through sensor body wire channel
3. Thread sensor body onto probe tip (hand tight)
4. Thread male cap (on rod) into sensor body (hand tight)
5. Verify assembly is straight and secure
6. Verify coil wires exit cleanly without pinching

**Pass Criteria:**
- Assembly screws together without cracking
- Coil wires not pinched or damaged
- Reasonable alignment (no major wobble)

**Duration:** 15 minutes

---

### Phase 4: Electrical Integration (Steps 4.1-4.2)

#### Step 4.1: Attach Test Leads
**Action:** Connect coil to test equipment.

**Option A: Direct solder (simplest)**
1. Solder 22 AWG hookup wire to each coil tail
2. Attach alligator clips or DuPont connectors to other end

**Option B: Quick-disconnect (better for iteration)**
1. Crimp or solder small screw terminals to coil tails
2. Use screw terminals for test lead attachment

**Duration:** 15 minutes

#### Step 4.2: Verify Electrical Integrity Post-Assembly
**Action:** Re-test coil after mechanical assembly.

| Test | Expected | Action if Fail |
|------|----------|----------------|
| Continuity | <50 Ω | Check for broken wire, connector issue |
| Inductance | Within 10% of pre-assembly | Check coil position, shielding |

**Note:** Inductance may change slightly due to proximity to sensor body walls. This is expected.

**Duration:** 5 minutes

---

### Phase 5: Functional Testing (Steps 5.1-5.4)

#### Step 5.1: Bench Setup
**Action:** Prepare test configuration.

```
[Function Generator] ──→ [Probe Coil] ──→ [Oscilloscope]
     10 kHz sine              ↑              CH1: across coil
     1-3V p-p                 |
                         [Metal target on stick]
```

**Alternative without function generator:**
- Use Arduino + AD9833 DDS module
- Or: tap into existing electronics if available

**Duration:** 15 minutes

#### Step 5.2: Baseline Measurement (Air)
**Action:** Record coil response with no target present.

1. Connect function generator to coil (series with 100Ω resistor for current limiting)
2. Set frequency to 10 kHz, amplitude to 2V p-p
3. Connect oscilloscope across coil
4. Record: voltage amplitude, phase (if measurable)
5. Note any noise or instability

**Expected:** Stable sine wave, amplitude depends on coil impedance.

**Duration:** 10 minutes

#### Step 5.3: Metal Target Response Test
**Action:** Verify coil detects conductive target.

**Procedure:**
1. Attach metal target (steel bolt, aluminum can, copper pipe) to non-metallic stick
2. Hold probe horizontally on non-metallic surface
3. Slowly bring target toward coil from 50cm away
4. Watch oscilloscope for amplitude/phase change
5. Note distance at which change becomes visible
6. Repeat 3 times for consistency

**Pass Criteria:**
- Observable amplitude or phase change when target within 10cm
- Response repeatable
- Response increases as target approaches

**If no response:**
- Increase drive voltage
- Try different frequency (2 kHz, 20 kHz, 50 kHz)
- Try larger metal target
- Verify coil is actually being driven (current flowing)

**Duration:** 20 minutes

#### Step 5.4: Frequency Sweep (Optional)
**Action:** Characterize response across operating range.

| Frequency | Amplitude (no target) | Amplitude (target at 5cm) | Change |
|-----------|----------------------|---------------------------|--------|
| 2 kHz | _____ | _____ | _____ |
| 5 kHz | _____ | _____ | _____ |
| 10 kHz | _____ | _____ | _____ |
| 20 kHz | _____ | _____ | _____ |
| 50 kHz | _____ | _____ | _____ |

**Duration:** 30 minutes

---

### Phase 6: Documentation and Decision (Steps 6.1-6.2)

#### Step 6.1: Record Results
**Action:** Document all measurements and observations.

**Prototype Test Log:**
```
Date: __________
Prototype ID: MVP-001

COIL PARAMETERS:
- Turns: _____
- Wire gauge: _____
- Core: _____ (material, dimensions)
- Inductance @ 1kHz: _____ mH
- Q @ 10kHz: _____
- DC Resistance: _____ Ω

MECHANICAL:
- Thread fit: Good / Acceptable / Poor
- Assembly alignment: Good / Acceptable / Poor
- Issues noted: _____________________

FUNCTIONAL TEST:
- Drive frequency: _____ kHz
- Drive amplitude: _____ V
- Baseline amplitude: _____ V
- Target response: YES / NO
- Detection distance: ~_____ cm
- Repeatability: Good / Variable / Poor

OBSERVATIONS:
_________________________________
_________________________________

DECISION: [ ] Proceed to Phase 2  [ ] Modify design  [ ] Major redesign
```

**Duration:** 15 minutes

#### Step 6.2: Make Go/No-Go Decision
**Action:** Evaluate results against success criteria.

| Criterion | Result | Pass/Fail |
|-----------|--------|-----------|
| Coil inductance 1.5 mH ±20% | _____ mH | [ ] |
| Q factor ≥15 @ 10 kHz | _____ | [ ] |
| Metal target detected | YES/NO | [ ] |
| Threads functional | YES/NO | [ ] |

**Decision Matrix:**

| Outcome | Action |
|---------|--------|
| All pass | Proceed to Phase 2 (add ERT, multi-segment) |
| Coil fail only | Re-wind coil, retest |
| Mechanical fail only | Adjust print settings, reprint |
| Detection fail | Investigate: coil issue? test setup? physics? |
| Multiple fail | Root cause analysis before proceeding |

---

## 5. Timeline Summary

### Optimistic Path (Everything Works)

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 0: Prep | 30 min | 30 min |
| Phase 1: Printing | 4-6 hrs (overnight OK) | ~6 hrs |
| Phase 2: Coil winding | 1.5 hrs | 7.5 hrs |
| Phase 3: Assembly | 45 min + cure | 8.5 hrs + cure |
| Phase 4: Wiring | 20 min | 9 hrs |
| Phase 5: Testing | 1 hr | 10 hrs |
| Phase 6: Documentation | 30 min | 10.5 hrs |

**Total: ~10-12 hours of active work, potentially 2 days with cure times.**

### Realistic Path (Includes Iteration)

- Add 2-4 hours for coil re-wind if first attempt fails
- Add 4-6 hours for reprint if thread issues
- Add 1-2 hours for test setup debugging

**Budget: 2-3 days to first successful data.**

---

## 6. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Threads break during assembly | Medium | Reprint (4-6 hrs) | Use blanks + tap/die approach; don't over-torque |
| Coil inductance wrong | Medium | Re-wind (1-2 hrs) | Measure during winding; adjust turns |
| No target detection | Low | Debug (2-4 hrs) | Verify coil driven; try multiple frequencies |
| Test equipment unavailable | Medium | Cannot test | Order Arduino + DDS module ($15, 2-day ship) |
| Ferrite core breaks | Low | Wait for replacement | Handle gently; have spares |

---

## 7. Files to Create/Modify

### Files to Create (During EXECUTE)

| File | Purpose |
|------|---------|
| `build/prototype-test-log-template.md` | Blank form for recording results |
| `build/rapid-prototype-guide.md` | Condensed build instructions |

### Files to Reference (No Changes)

| File | Purpose |
|------|---------|
| `hardware/cad/openscad/modular_flush_connector.scad` | CAD source |
| `build/manufacturing-release-notes.md` | Print settings |
| `docs/manufacturing/coil-manufacturing-spec.md` | Coil specs |

---

## 8. Next Phases (After MVP Success)

### Phase 2: Add ERT Capability
- Add electrode ring to sensor body
- Test ERT measurement with multimeter
- Verify coil and electrode coexistence

### Phase 3: Multi-Segment Assembly
- Build 2-segment probe
- Test mechanical connection under load
- Verify wire routing through junction

### Phase 4: Production Wiring
- Implement Phoenix connector system
- Build cable harness
- Test with production electronics

### Phase 5: Waterproofing and Field Test
- Epoxy all joints
- Pressure test sealing
- Deploy in actual soil

---

## 9. Approval Checklist

Before proceeding to EXECUTE mode, confirm:

- [ ] Materials and equipment available (or ordered)
- [ ] Timeline acceptable (2-3 days)
- [ ] Success criteria appropriate for prototype
- [ ] Willing to accept "good enough" for first test
- [ ] Understand this is throwaway/iteration prototype

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-27 | Claude (RIPER PLAN) | Initial plan |
| 1.1 | 2026-01-27 | Claude (RIPER EXECUTE) | Documentation created |

---

## 11. Execution Log

### Documentation Phase (Completed 2026-01-27)

| File | Purpose | Status |
|------|---------|--------|
| `build/prototype-test-log-template.md` | Blank form for recording test results | CREATED |
| `build/rapid-prototype-guide.md` | Condensed quick-reference build guide | CREATED |

### Whitepaper Integration (Completed 2026-01-27)

| File | Changes | Status |
|------|---------|--------|
| `docs/hirt-whitepaper/sections/05-mechanical-design.qmd` | Added "Rapid Prototyping Guide" section with 4 diagrams | UPDATED |
| `docs/hirt-whitepaper/diagrams/mechanical.py` | Added 4 new diagram functions | UPDATED |

**New diagrams added:**
- `create_prototyping_workflow()` - 4-phase MVP build process
- `create_quick_test_setup()` - Bench test equipment diagram
- `create_coil_winding_steps()` - Step-by-step winding guide
- `create_prototype_decision_tree()` - Test result decision flow

### Physical Build Phase (User Action Required)

The following steps require hands-on work:

- [ ] **Phase 0:** Inventory check and equipment setup
- [ ] **Phase 1:** 3D print parts (4-6 hours)
- [ ] **Phase 2:** Wind and test coil (1.5 hours)
- [ ] **Phase 3:** Mechanical assembly (45 min)
- [ ] **Phase 4:** Electrical wiring (20 min)
- [ ] **Phase 5:** Functional testing (1 hour)
- [ ] **Phase 6:** Document results and decide

---

**END OF PLAN**
