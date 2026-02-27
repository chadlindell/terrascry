# HIRT MIT-3D Coil Design Optimization Plan

**Date:** 2026-01-27
**Branch:** main
**Status:** APPROVED - EXECUTING
**RIPER Phase:** EXECUTE
**Approved:** 2026-01-27

---

## 1. Executive Summary

This plan addresses optimization of MIT (Magnetic Induction Tomography) coils for the HIRT probe system operating across the 2-50 kHz frequency range. Based on RESEARCH and INNOVATE phases, the recommended approach is a **phased optimization strategy** that:

1. First characterizes existing coil performance to establish baseline
2. Implements targeted improvements based on measured deficiencies
3. Validates improvements through systematic testing

**Primary Goal:** Achieve Q ≥ 25 across 2-50 kHz while maintaining compatibility with existing 16mm probe body and electronics.

---

## 2. Scope Definition

### 2.1 In Scope
- TX and RX coil electrical optimization
- Wire gauge and construction evaluation
- Core material selection for broadband performance
- Winding configuration optimization
- Self-resonant frequency analysis
- Documentation of design parameters

### 2.2 Out of Scope
- Electronics/amplifier modifications (separate project)
- Probe body mechanical redesign
- Multi-coil switchable systems (future work)
- Active compensation circuits (future work)
- Gradiometer configurations (future work)

### 2.3 Constraints
- Must fit within existing 16mm OD probe body
- Must interface with existing DDS/amplifier electronics
- Target inductance: 1-2 mH (maintain compatibility)
- Budget: < $50 per coil pair (TX+RX)
- Manufacturing: Hand-wound acceptable for prototypes

---

## 3. Technical Specifications

### 3.1 Performance Targets

| Parameter | Current Spec | Optimized Target | Verification Method |
|-----------|-------------|------------------|---------------------|
| Inductance | 1-2 mH | 1.5 mH ± 10% | LCR meter at 1 kHz |
| Q Factor @ 2 kHz | >20 | ≥ 25 | LCR meter |
| Q Factor @ 10 kHz | >20 | ≥ 30 | LCR meter |
| Q Factor @ 50 kHz | Unmeasured | ≥ 20 | LCR meter |
| DC Resistance | <10 Ω | < 8 Ω | DMM |
| Self-Resonant Freq | Unknown | > 200 kHz | VNA sweep |
| TX-RX Coupling | Unmeasured | < -40 dB | VNA S21 |

### 3.2 Selected Design Approach

Based on INNOVATE phase trade-off analysis:

**Wire:** Solid 34 AWG enameled magnet wire (baseline), with 36 AWG evaluation
- Rationale: Skin depth at 50 kHz (0.29mm) still exceeds 34 AWG diameter (0.16mm)
- Litz wire deferred: Cost/complexity not justified until baseline Q measured

**Core:** NiZn ferrite, Ø8mm × 100mm
- Rationale: Better high-frequency performance than MnZn
- Permeability: μᵣ = 125-250 typical (Fair-Rite 61 or equivalent)
- Lower μ requires ~300 turns vs 250 for MnZn

**Winding:** Single-layer bank winding (primary), multi-layer progressive (fallback)
- Rationale: Minimizes self-capacitance, maximizes self-resonant frequency
- Constraint: Must verify 300 turns fits within probe body length

**Configuration:** Orthogonal TX/RX maintained
- No change to physical arrangement
- Focus on electrical optimization only

---

## 4. Implementation Steps

### Phase A: Baseline Characterization (Steps 1-5)

#### Step 1: Acquire Test Equipment
**Action:** Verify/acquire measurement capability
- [ ] LCR meter with 2-50 kHz range (or access to one)
- [ ] Vector Network Analyzer (VNA) for self-resonance and coupling
- [ ] Calibration standards for LCR meter

**Deliverable:** Equipment availability confirmed
**Duration:** 1-2 days

#### Step 2: Fabricate Reference Coils
**Action:** Wind 3 identical coils using current specification for statistical baseline
- [ ] Ferrite core: MnZn Ø8mm × 100mm (current spec)
- [ ] Wire: 34 AWG enameled
- [ ] Turns: 250, single-layer
- [ ] Document winding tension, layer spacing, termination method

**Deliverable:** 3 reference coils with winding log
**Duration:** 1 day

#### Step 3: Measure Baseline Performance
**Action:** Full characterization of reference coils
- [ ] Inductance at 1 kHz, 10 kHz, 50 kHz
- [ ] Q factor at 2 kHz, 5 kHz, 10 kHz, 20 kHz, 50 kHz
- [ ] DC resistance
- [ ] Self-resonant frequency (VNA sweep 100 kHz - 10 MHz)
- [ ] Record ambient temperature during measurements

**Deliverable:** Baseline data spreadsheet with statistical analysis (mean, std dev)
**Duration:** 1 day

#### Step 4: Measure TX-RX Coupling
**Action:** Install reference coils in probe body, measure coupling
- [ ] Mount TX and RX coils in orthogonal configuration
- [ ] Measure S21 coupling at 2, 10, 20, 50 kHz
- [ ] Rotate RX coil in 5° increments, find minimum coupling angle
- [ ] Document optimal orientation

**Deliverable:** Coupling vs angle data, optimal configuration identified
**Duration:** 0.5 day

#### Step 5: Analyze Baseline Results
**Action:** Compare measured performance to targets
- [ ] Identify which parameters meet/exceed targets
- [ ] Identify which parameters fall short
- [ ] Determine primary optimization focus (Q, SRF, coupling, or all)

**Deliverable:** Gap analysis report
**Duration:** 0.5 day

---

### Phase B: Core Material Optimization (Steps 6-9)

#### Step 6: Acquire NiZn Ferrite Cores
**Action:** Source NiZn ferrite rods for comparison
- [ ] Target: Fair-Rite 61 material or equivalent (μᵣ ≈ 125)
- [ ] Dimensions: Ø8mm × 100mm (match baseline geometry)
- [ ] Quantity: 6 cores (for 3 test coils + spares)

**Deliverable:** NiZn cores received and inspected
**Duration:** 1-2 weeks (ordering lead time)

#### Step 7: Wind NiZn Test Coils
**Action:** Fabricate 3 coils on NiZn cores
- [ ] Match winding parameters to baseline (34 AWG, single-layer)
- [ ] Adjust turns to achieve 1.5 mH target (expect ~300-350 turns)
- [ ] Document actual turns count and winding length

**Deliverable:** 3 NiZn coils with winding log
**Duration:** 1 day

#### Step 8: Measure NiZn Performance
**Action:** Full characterization identical to Step 3
- [ ] Inductance at 1 kHz, 10 kHz, 50 kHz
- [ ] Q factor at 2 kHz, 5 kHz, 10 kHz, 20 kHz, 50 kHz
- [ ] DC resistance
- [ ] Self-resonant frequency

**Deliverable:** NiZn data spreadsheet
**Duration:** 1 day

#### Step 9: Compare Core Materials
**Action:** Statistical comparison of MnZn vs NiZn
- [ ] Plot Q vs frequency for both materials
- [ ] Calculate inductance stability across frequency
- [ ] Evaluate self-resonant frequency improvement
- [ ] Decision: Select core material for Phase C

**Deliverable:** Core material comparison report with selection decision
**Duration:** 0.5 day

---

### Phase C: Winding Optimization (Steps 10-14)

#### Step 10: Design Winding Variants
**Action:** Define 3 winding configurations for testing
- [ ] Variant A: Single-layer bank, 34 AWG (control - same as baseline)
- [ ] Variant B: Single-layer bank, 36 AWG (finer wire, more turns possible)
- [ ] Variant C: Two-layer progressive, 34 AWG (compact, evaluate capacitance impact)

**Deliverable:** Winding specification sheets for each variant
**Duration:** 0.5 day

#### Step 11: Fabricate Winding Variants
**Action:** Wind 2 coils of each variant on selected core material
- [ ] Use consistent tension and technique
- [ ] Document any manufacturing difficulties
- [ ] Measure physical dimensions (winding length, OD)

**Deliverable:** 6 coils (2 each of 3 variants)
**Duration:** 2 days

#### Step 12: Measure Winding Variants
**Action:** Full characterization of all variants
- [ ] Same measurement protocol as Steps 3 and 8
- [ ] Additional: Measure self-capacitance if possible (from SRF and L)

**Deliverable:** Variant comparison data
**Duration:** 1 day

#### Step 13: Thermal Stability Test
**Action:** Evaluate temperature effects on best-performing variant
- [ ] Measure L and Q at room temperature (25°C)
- [ ] Heat coil to 40°C (hair dryer or heat gun, controlled)
- [ ] Measure L and Q at 40°C
- [ ] Cool coil to 10°C (refrigerator)
- [ ] Measure L and Q at 10°C
- [ ] Calculate temperature coefficients

**Deliverable:** Temperature stability data
**Duration:** 0.5 day

#### Step 14: Select Optimized Design
**Action:** Analyze all data, select final design
- [ ] Weight factors: Q (40%), SRF (20%), manufacturability (20%), cost (10%), thermal stability (10%)
- [ ] Score each variant
- [ ] Document selection rationale

**Deliverable:** Final design selection report
**Duration:** 0.5 day

---

### Phase D: Validation and Documentation (Steps 15-19)

#### Step 15: Fabricate Validation Set
**Action:** Build 5 coil pairs using optimized design
- [ ] TX coil × 5
- [ ] RX coil × 5
- [ ] Full measurement of each coil
- [ ] Calculate manufacturing repeatability (Cpk if possible)

**Deliverable:** 5 validated coil pairs with measurement certificates
**Duration:** 3 days

#### Step 16: System Integration Test
**Action:** Install optimized coils in complete probe assembly
- [ ] Verify fit within 16mm body
- [ ] Connect to existing electronics
- [ ] Functional test: Generate TX signal, measure RX response
- [ ] Compare SNR to baseline coils (if available)

**Deliverable:** Integration test report
**Duration:** 1 day

#### Step 17: Field Validation (Optional)
**Action:** Limited field test with optimized probe
- [ ] Detect known buried target at multiple frequencies
- [ ] Compare signal strength to baseline (if available)
- [ ] Note any operational issues

**Deliverable:** Field test notes
**Duration:** 1 day

#### Step 18: Update Design Documentation
**Action:** Revise HIRT documentation with optimized specifications
- [ ] Update `docs/field-guide/coil-winding-recipe.md`
- [ ] Update `hardware/schematics/electronics/mit-circuit.md` (if impedance changed)
- [ ] Update `docs/hirt-whitepaper/sections/05-mechanical-design.qmd`
- [ ] Update `docs/hirt-whitepaper/sections/09-calibration.qmd`
- [ ] Add new characterization data to appropriate sections

**Deliverable:** Updated documentation (PR ready)
**Duration:** 1 day

#### Step 19: Create Manufacturing Specification
**Action:** Write formal manufacturing spec for optimized coil
- [ ] Materials list with part numbers/sources
- [ ] Step-by-step winding procedure
- [ ] Quality control acceptance criteria
- [ ] Measurement protocol for incoming inspection

**Deliverable:** `docs/manufacturing/coil-manufacturing-spec.md`
**Duration:** 0.5 day

---

## 5. File Changes Summary

### Files to be Modified
1. `docs/field-guide/coil-winding-recipe.md` - Updated winding parameters
2. `hardware/schematics/electronics/mit-circuit.md` - Updated coil specifications
3. `docs/hirt-whitepaper/sections/05-mechanical-design.qmd` - Updated physical specs
4. `docs/hirt-whitepaper/sections/09-calibration.qmd` - Updated tolerances
5. `hardware/bom/probe-bom.md` - Updated component specifications

### Files to be Created
1. `docs/manufacturing/coil-manufacturing-spec.md` - New manufacturing document
2. `docs/testing/coil-characterization-data.csv` - Measurement data (optional)

### Files Unchanged
- CAD files (no mechanical changes)
- Electronics schematics (circuit unchanged, only component specs)
- Firmware (no software changes)

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| NiZn cores unavailable in required size | Medium | High | Identify 2-3 alternative suppliers; accept Ø6mm if Ø8mm unavailable |
| Single-layer winding exceeds body length | Medium | Medium | Fallback to 2-layer progressive; accept slightly lower SRF |
| Q target not achievable with solid wire | Low | Medium | Evaluate Litz wire for high-frequency variant only |
| Measurement equipment unavailable | Low | High | Identify university/makerspace with LCR meter access |
| Temperature stability unacceptable | Low | Low | Add temperature compensation note to field operations |

---

## 7. Success Criteria

Plan is considered successful if:

1. **Q Factor:** Achieved Q ≥ 25 at 10 kHz (primary metric)
2. **Broadband Performance:** Q ≥ 20 maintained from 2-50 kHz
3. **Self-Resonant Frequency:** SRF > 200 kHz confirmed
4. **Manufacturability:** Coil-to-coil variation < 15% on key parameters
5. **Compatibility:** Coils fit existing probe body and electronics
6. **Documentation:** All design files updated and version-controlled

---

## 8. Dependencies and Prerequisites

### Required Before EXECUTE Phase
- [ ] Access to LCR meter (2-50 kHz capability)
- [ ] Ferrite cores in stock or on order
- [ ] Magnet wire in stock (34 AWG minimum, 36 AWG preferred)
- [ ] Probe body available for fit testing
- [ ] User approval of this plan

### Optional but Recommended
- [ ] Vector Network Analyzer access (for SRF measurement)
- [ ] Temperature-controlled environment (for thermal testing)
- [ ] Baseline coil from current production (for comparison)

---

## 9. Approval Checklist

Before proceeding to EXECUTE mode, confirm:

- [ ] Performance targets are acceptable
- [ ] Scope boundaries are appropriate
- [ ] Implementation steps are complete and correct
- [ ] Risk mitigations are adequate
- [ ] File change list is accurate
- [ ] Dependencies can be satisfied

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-27 | Claude (RIPER PLAN) | Initial plan |
| 1.1 | 2026-01-27 | Claude (RIPER EXECUTE) | Plan approved, documentation phase executed |

---

## 11. Execution Log

### Documentation Phase (Completed 2026-01-27)

The following documentation updates were completed per Steps 18-19:

| File | Action | Status |
|------|--------|--------|
| `docs/field-guide/coil-winding-recipe.md` | Updated with optimized specs | DONE |
| `hardware/schematics/electronics/mit-circuit.md` | Updated coil interface specs | DONE |
| `hardware/bom/probe-bom.md` | Updated component specs and costs | DONE |
| `docs/hirt-whitepaper/sections/09-calibration.qmd` | Updated tolerances and procedures | DONE |
| `docs/manufacturing/coil-manufacturing-spec.md` | Created new manufacturing spec | DONE |

### Physical Implementation Phases (Pending)

The following phases require physical fabrication and measurement:

- **Phase A (Steps 1-5):** Baseline Characterization - PENDING
- **Phase B (Steps 6-9):** Core Material Optimization - PENDING
- **Phase C (Steps 10-14):** Winding Optimization - PENDING
- **Phase D (Steps 15-17):** Validation - PENDING

These steps require:
1. Acquisition of test equipment (LCR meter, VNA)
2. Procurement of NiZn ferrite cores
3. Physical coil fabrication
4. Measurement and characterization

---

**END OF PLAN**
