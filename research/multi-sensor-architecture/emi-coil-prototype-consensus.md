# EMI Coil Prototype Build -- PAL Consensus Validation

**Task:** I8 -- EMI Coil Prototype Build
**Classification:** CRITICAL ENGINEERING
**Date:** 2026-02-18
**Reference:** `emi-coil-design.md`

## Consensus Configuration

| Parameter | Value |
|-----------|-------|
| Models requested | openai/gpt-5.2-pro (neutral), gemini-3-pro-preview (neutral) |
| Models successfully consulted | openai/gpt-5.2-pro (neutral) |
| Models failed | gemini-3-pro-preview (429 RESOURCE_EXHAUSTED -- API quota) |
| Supplementary analysis | Claude Opus 4.6 independent analysis |
| Overall confidence | 8/10 (High) |

---

## Executive Summary

The EMI coil prototype build plan is **technically feasible as a learning prototype** but requires **several critical design changes** before it can achieve its stated performance targets. The fundamental physics (LIN approximation, quadrature extraction), signal chain architecture (AD9833 -> OPA549 -> TX, RX -> AD8421 -> AD630 -> ADS1115), and testing methodology are sound and well-aligned with proven commercial instruments (Geonics EM38).

However, the plan has **one potential showstopper** (TX coil geometry) and **multiple significant risks** that will prevent the prototype from reaching the <1% primary cancellation and <1 uV noise floor targets without modification.

**Overall verdict:** PROCEED WITH MODIFICATIONS. The required changes are material substitutions and additions, not architectural redesigns. Timeline impact is minimal.

---

## Critical Findings

### FINDING 1: TX Coil Geometry -- POTENTIAL SHOWSTOPPER

**Severity:** CRITICAL
**Consensus:** Unanimous (GPT-5.2-pro + independent analysis)

**Issue:** The build plan specifies "30 turns of 24 AWG enameled copper wire wound on 12cm diameter ferrite ring core." If "ferrite ring core" means a **closed toroid**, this is a fundamental problem. A closed toroidal ferrite confines most magnetic flux within the core material -- the opposite of what an EMI transmitter requires. The TX coil must radiate a dipole-like external field into the ground to induce eddy currents.

**Impact:** A closed toroid could reduce external field radiation by an order of magnitude or more, making the instrument non-functional for ground conductivity measurement.

**Recommendation:**
- **PRIMARY:** Use an **air-core loop** (standard in EM38-class instruments). This is the simplest and most proven approach.
- **ALTERNATIVE:** If ferrite is desired for increased inductance, use an **open magnetic path** such as ferrite rod(s) inside a solenoidal coil. Never a closed ring.
- At 15 kHz, ferrite is not necessary -- air-core inductance with 30 turns on a 12cm+ form is adequate for 100mA drive.

**Action required:** Clarify whether "ferrite ring core" means a closed toroid. If so, redesign TX coil as air-core.

---

### FINDING 2: Primary Cancellation Target Too Loose

**Severity:** HIGH
**Consensus:** Unanimous

**Issue:** The build plan targets **<1% of uncompensated primary** at the RX coil. However, the design document (`emi-coil-design.md`, lines 40-51) shows that soil conductivity signals are only **100-3000 ppm** (0.01-0.3%) of the primary field:

| Soil conductivity | Hs/Hp | Relative to 1% residual |
|-------------------|-------|--------------------------|
| 10 mS/m (dry sand) | ~100 ppm | 100x smaller than residual |
| 50 mS/m (loam) | ~500 ppm | 20x smaller than residual |
| 100 mS/m (wet clay) | ~1000 ppm | 10x smaller than residual |

A 1% (10,000 ppm) residual primary is **10-100x larger** than the desired soil signal. While the AD630 lock-in amplifier provides phase discrimination, **phase errors leak in-phase primary into the quadrature (conductivity) channel**:

```
Quadrature leakage = residual_primary x sin(phase_error)
Example: 1% residual x 1 deg phase error = 0.01 x 0.0175 = 175 ppm
This artifact is equivalent to ~10-20 mS/m false conductivity signal.
```

**Recommendation:**
- Tighten primary cancellation target to **<=0.1% (<=1000 ppm)**
- Add **phase trim capability** -- either a precision potentiometer on the AD630 reference phase, or a DAC-controlled phase adjustment
- Include a **periodic air-null calibration routine** in the field testing protocol
- Consider **electronic nulling** (DAC-driven compensation coil) in addition to the passive bucking coil

---

### FINDING 3: Capacitive Coupling Not Addressed

**Severity:** HIGH
**Consensus:** Unanimous

**Issue:** The bucking coil cancels **magnetic (H-field) coupling** between TX and RX. However, **electric field (E-field) capacitive feedthrough** between the TX and RX coils, wiring, and circuit boards is not addressed by a bucking coil and can dominate at these frequencies, especially in a point-to-point prototype with exposed wiring.

**Recommendation:**
- Add **electrostatic (Faraday) shields** to both TX and RX coils
- Implementation: Wrap each coil with a single layer of **slotted copper foil tape**
  - The shield must have a **gap (slot)** to prevent forming a shorted turn (which would create eddy current losses)
  - Connect the shield to analog ground **at one point only**
  - This is standard practice in commercial FDEM instruments
- Shield the cable from RX coil to AD8421 as well

---

### FINDING 4: Bucking Coil Mechanical Stability

**Severity:** HIGH
**Consensus:** Unanimous

**Issue:** The bucking coil position is the most mechanically sensitive element in the design. Coupling between coils scales approximately as 1/r^3 in the far field. A **1mm position shift** in a bucking coil operating at 99% cancellation can cause the residual to **double or triple**, introducing hundreds of ppm of error.

**Recommendations:**
1. **Fine-thread mechanical adjustment:** Use a micrometer-type slide or fine-thread screw mechanism for position tuning
2. **Locking mechanism:** Positive lock (set screw + jam nut) after adjustment
3. **Epoxy potting:** After final tuning, pot the bucking coil mount with rigid epoxy (e.g., West System 105/205)
4. **Material selection:** Use low-creep, thermally stable materials for the crossbar and coil mounts:
   - GOOD: Fiberglass, G10/FR4, carbon fiber, aluminum
   - AVOID: 3D-printed PLA (creeps under load and heat), wood (hygroscopic)
5. **Vibration isolation:** Consider damping material between coil mounts and crossbar to reduce microphonic effects during walking

---

### FINDING 5: Missing Reference Signal Path

**Severity:** HIGH
**Consensus:** Identified in independent analysis, confirmed by GPT-5.2-pro

**Issue:** The build plan specifies the AD630 balanced modulator/demodulator for phase-sensitive detection but does not specify the **reference signal path** from the AD9833 DDS to the AD630. The AD630 requires a reference input at the TX frequency to perform synchronous demodulation. Without this, the AD630 cannot extract I (in-phase) and Q (quadrature) components.

**Recommendation:**
- Route a **reference signal** from the AD9833 square-wave output (or buffered sine output) directly to the AD630 reference input
- For Q-channel extraction, provide a **90-degree phase-shifted reference** (e.g., RC phase-shift network at 15 kHz, or use two AD630s with quadrature references)
- Document the reference signal path explicitly in the build plan
- Consider using the AD9833's MSB output as a digital reference signal

---

## Important Findings

### FINDING 6: Coil Diameters Likely Too Small

**Severity:** MEDIUM-HIGH
**Source:** GPT-5.2-pro (strong recommendation), partially supported by independent analysis

**Issue:** The proposed coil diameters (12cm TX, 8cm RX) are significantly smaller than those used in commercial portable FDEM instruments (typically 20-40cm). The absolute RX voltage is proportional to N * A * omega * B, where A is the coil area. Smaller diameters reduce the absolute signal voltage, pushing the electronics noise requirements into a very demanding regime.

**Recommendation:**
- For the prototype, consider increasing TX coil diameter to **20-30cm** and RX to **12-15cm** (or larger)
- More turns can partially compensate for smaller area, but watch for self-resonance (keep self-resonant frequency well above 15 kHz)
- Alternatively, proceed with the current sizes but accept that the noise floor target may need to be relaxed for initial testing

**Trade-off note:** Larger coils increase weight and wind sensitivity. For a walking-speed field instrument, 20-25cm is a reasonable compromise.

---

### FINDING 7: TX Should Be Current-Driven

**Severity:** MEDIUM
**Consensus:** Unanimous

**Issue:** Copper wire resistance changes at approximately +0.39%/C. If the TX coil is voltage-driven (constant voltage applied), the TX current will vary with temperature, changing the primary field amplitude and introducing measurement drift.

**Recommendation:**
- Implement **current-mode drive**: Add a sense resistor (0.1-1.0 ohm, low-TCR) in series with the TX coil and use feedback to the OPA549 to maintain constant current
- Log TX current amplitude continuously for drift correction
- This also reduces sensitivity to coil resistance changes from mechanical deformation

---

### FINDING 8: ADS1115 May Limit Noise Floor

**Severity:** MEDIUM
**Consensus:** Unanimous

**Issue:** The <1 uV RMS end-to-end noise target is aggressive. While the AD8421 front-end is excellent (3 nV/sqrt(Hz)), the ADS1115 is a general-purpose 16-bit ADC. At post-demodulation DC levels, the ADS1115's quantization noise and thermal noise may dominate the error budget.

**Analysis:**
- ADS1115 at gain=1, 128 SPS: ~7.8 uV LSB, noise ~4 uV RMS
- AD8421 at gain=100 with 10 Hz bandwidth: input-referred noise ~3 nV * sqrt(10) = ~9.5 nV, output ~0.95 uV
- The ADS1115 noise is approximately 4x the signal chain noise, becoming the bottleneck

**Recommendation:**
- For initial prototype: Proceed with ADS1115 but use the highest gain setting (16x, 7.8 uV/LSB) and slowest sample rate (8 SPS, lowest noise)
- For performance upgrade path: Consider a **24-bit delta-sigma ADC** designed for bridge/sensor applications (e.g., ADS1256, ADS1262, or MCP3561) which can achieve sub-uV noise
- Alternatively, increase the AD8421 gain from 100 to 1000 (verify AD8421 gain-bandwidth at 15 kHz) to present a larger signal to the ADC

---

### FINDING 9: AD630 Procurement Risk

**Severity:** MEDIUM
**Consensus:** Unanimous

**Issue:** The AD630 is an older Analog Devices part. While still in production, it is expensive ($25-40 per unit) and availability through common hobbyist distributors (DigiKey, Mouser) can be inconsistent.

**Recommendations:**
1. **Verify supply immediately:** Check current stock and lead times before committing to the build
2. **Order spares:** Buy 2-3 units in case of prototype iteration
3. **Plan a fallback path:**
   - **CMOS switch-based synchronous detector:** 74HC4053 analog switch + op-amp integrator. Simpler, cheaper, but potentially higher offset drift
   - **Digital lock-in:** Sample the RX signal at high rate (100-200 ksps) with a faster ADC and perform I/Q demodulation in software (ESP32 or external DSP). This eliminates the AD630 entirely and provides flexible phase correction. Downside: requires higher-speed ADC and anti-alias filter design
   - **AD630 on breakout board:** Some vendors offer AD630 evaluation/breakout boards that can be used directly in the point-to-point prototype

---

### FINDING 10: Point-to-Point Wiring Best Practices

**Severity:** MEDIUM
**Consensus:** Unanimous

**Issue:** Point-to-point wiring at 15 kHz is manageable but requires discipline to achieve the noise targets.

**Recommendations:**
1. **Place AD8421 physically at the RX coil** -- use short leads from coil to preamp, then route the amplified signal (lower impedance, less susceptible to pickup) back to the main electronics via shielded cable
2. **Star grounding:** Define a single ground reference point. Route all ground returns to this point. Do not daisy-chain grounds
3. **Cable separation:** Keep TX drive wiring (high current, 15 kHz) physically separated from RX signal wiring by at least 10cm. Route on opposite sides of the crossbar if possible
4. **Shielded cable** for RX coil to AD8421 and for AD8421 output to AD630
5. **Twisted pair** for TX drive to reduce radiated magnetic field from the drive cable
6. **Bypass capacitors:** 100nF + 10uF at every IC power pin, placed as close as possible

---

## Responses to Specific Concerns

### Concern 1: Ferrite Core at 15 kHz

**Answer:** Ferrite is **not necessary** at 15 kHz and is **potentially harmful** if used as a closed toroid.

| Configuration | Pros | Cons |
|--------------|------|------|
| Air-core loop | Simple construction; no temperature-dependent permeability; proven in commercial instruments; predictable field pattern | Lower inductance (higher drive voltage needed); larger coil may be needed for adequate TX current |
| Ferrite rod (open path) | Higher inductance; lower drive requirements | Temperature-dependent permeability (-0.1 to -0.3%/C for MnZn); adds weight; field pattern depends on rod geometry |
| Ferrite toroid (closed path) | Highest inductance | **DOES NOT WORK** -- confines flux internally, no external field for ground induction |

**Recommendation:** Use air-core for the prototype. Simple, predictable, proven.

### Concern 2: OPA549 Thermal

**Answer:** Manageable with modest heatsinking.

**Analysis:**
- TX coil resistance: ~5 ohm (30 turns, 24 AWG, 12cm diameter)
- Coil power: I^2 * R = (0.1)^2 * 5 = 50 mW (negligible)
- OPA549 internal dissipation: depends on supply voltage and reactive current. With +/-15V supply and 100mA into a largely inductive load, worst-case dissipation is approximately Vsupply * Iout = 15V * 100mA = 1.5W
- OPA549 thermal resistance (junction to case): 3.7 C/W
- With a TO-220 heatsink (Rth < 10 C/W): Tjunction rise = 1.5W * (3.7 + 10) = ~20 C above ambient

**Recommendations:**
- Use a standard TO-220 clip-on heatsink (Aavid/Boyd 7021 or similar, ~10 C/W)
- Add a **snubber network** (series R-C, e.g., 10 ohm + 100nF) across the TX coil to ensure OPA549 stability with inductive load
- Consider reducing supply voltage if possible (e.g., +/-5V or +/-10V) to reduce dissipation
- OPA549 is significantly over-specified for 100mA (it can deliver 8A). A lower-power op-amp (e.g., OPA548, or even OPA552) could reduce cost and thermal burden

### Concern 3: AD630 Availability

**Answer:** Still in production by Analog Devices as of 2026, but supply is inconsistent and price is high ($25-40). See Finding 9 above for detailed recommendations and fallback strategies.

### Concern 4: Bucking Coil 1mm Shift

**Answer:** Significant degradation is expected. See Finding 4 above for detailed analysis and mechanical design recommendations.

**Quantitative estimate:** The mutual inductance between two coaxial circular loops varies approximately as (1/d^3) for d >> coil radius. For a bucking coil at ~0.3-0.5m from RX:
- 1mm shift from 300mm = 0.33% position change
- Coupling change ~ 3 * 0.33% = ~1% change in compensation
- If compensating 99%, a 1% relative change means residual goes from 1% to ~2% -- a 2x degradation
- This adds ~10,000 ppm of primary leakage, which is 10-100x larger than the soil signal

**Conclusion:** Mechanical stability of the bucking coil mount is the single most important construction quality factor.

### Concern 5: Environmental Sensitivity

**Answer:** Significant impact on calibration. Multiple temperature-dependent effects compound.

| Effect | Mechanism | Magnitude | Mitigation |
|--------|-----------|-----------|------------|
| TX current drift | Copper R increases +0.39%/C | 0.4%/C if voltage-driven | Current-mode drive |
| Ferrite permeability | MnZn ferrite: -0.1 to -0.3%/C | 0.1-0.3%/C | Use air-core (eliminates) |
| RX coil sensitivity | Copper R changes RX coil loading | <0.1%/C (minor) | Negligible |
| AD630/AD8421 drift | Op-amp offset/gain drift | Depends on part specs | Temperature logging + periodic cal |
| Mechanical expansion | Crossbar length changes with temperature | ~23 ppm/C (aluminum) | Use carbon fiber or Invar |

**Recommended mitigation strategy:**
1. Current-mode TX drive (eliminates largest single drift source)
2. Periodic "air null" calibration (lift instrument to 2m height, record zero offset)
3. Temperature sensor on crossbar (log for post-processing correction)
4. Avoid direct sun exposure on the instrument during survey

### Concern 6: Safety at 15 kHz / 100mA

**Answer:** **No RF exposure concern for the operator.**

**Analysis:**
- Magnetic field at 1m from a 30-turn, 12cm diameter, 100mA coil:
  - B = (mu0 * N * I * a^2) / (2 * (a^2 + d^2)^(3/2))
  - B ~ 2.1 uT at 1m on axis
- ICNIRP 2010 reference level for general public at 15 kHz: 27 uT
- Our field is **13x below the exposure limit** at 1m distance, and the operator's body is typically 0.5-1.5m from the coils

**Greater practical risk:** Self-interference with Pathfinder's fluxgate magnetometers. The TDM (time-division multiplexing) scheme in the firmware (50ms fluxgate, 30ms EMI, 20ms settling) is designed to address this, but verify that the EMI TX is fully silenced during the fluxgate measurement window.

---

## Recommended Build Modifications Summary

### Must-Do (Before First Power-Up)

| # | Modification | Effort | Impact |
|---|-------------|--------|--------|
| 1 | Replace ferrite ring TX with air-core loop | Low | Eliminates showstopper risk |
| 2 | Add Faraday shields to TX and RX coils | Low | Eliminates capacitive coupling |
| 3 | Specify AD9833-to-AD630 reference signal path | Low | Enables synchronous detection |
| 4 | Implement star grounding + shielded RX cable | Low | Achieves noise targets |
| 5 | Place AD8421 at RX coil location | Low | Reduces pickup on high-impedance lines |

### Should-Do (For Target Performance)

| # | Modification | Effort | Impact |
|---|-------------|--------|--------|
| 6 | Tighten primary cancellation to <=0.1% | Medium | Enables low-conductivity measurement |
| 7 | Add phase trim capability | Medium | Prevents quadrature leakage |
| 8 | Current-mode TX drive (sense resistor feedback) | Medium | Temperature stability |
| 9 | Rigid bucking coil mount with fine adjust + lock | Medium | Measurement stability |
| 10 | Increase TX coil diameter to 20-25cm | Low | Improved SNR |

### Consider (Upgrade Path)

| # | Modification | Effort | Impact |
|---|-------------|--------|--------|
| 11 | Replace ADS1115 with 24-bit delta-sigma ADC | Medium | Sub-uV noise floor |
| 12 | Digital lock-in (replace AD630) | High | Flexibility, phase correction |
| 13 | Electronic nulling (DAC-driven compensation) | High | Auto-calibration, drift rejection |
| 14 | Carbon fiber crossbar | Low | Thermal stability |

---

## Revised Build Plan (Incorporating Consensus Recommendations)

### TX Coil (Revised)
- **Air-core loop**, 20-25cm diameter
- 30 turns of 24 AWG enameled copper wire
- Single-layer winding on rigid form (fiberglass tube or 3D-printed PETG)
- Faraday shield: slotted copper foil tape, grounded at one point
- Secure to crossbar with rigid fiberglass clamp

### RX Coil (Revised)
- Air-core, 12-15cm diameter
- 30 turns of 28 AWG enameled wire
- Faraday shield: slotted copper foil tape, grounded at one point
- AD8421 breakout mounted directly at coil location inside a small shielded enclosure

### Bucking Coil (Revised)
- Calculated turns at calculated position (per R3 analysis)
- **Fine-thread adjustment mechanism** (M3 or M4 all-thread with jam nuts)
- Rigid mount (G10/FR4 bracket, not PLA)
- Epoxy-pot after final tuning

### Signal Chain (Revised)
```
AD9833 DDS ──► buffer ──► OPA549 (on heatsink with snubber) ──► sense resistor ──► TX coil
  (15 kHz)                 (current feedback from sense resistor)

AD9833 REF OUT ──► [phase trim network] ──► AD630 reference input

TX coil              RX coil ──► AD8421 (AT COIL, gain=100)
                                    │
                              shielded cable
                                    │
                              AD630 (I and Q) ──► RC LPF (fc=10 Hz) ──► ADS1115
```

### Crossbar
- Aluminum or carbon fiber, 1.0m
- Coil mounts: fiberglass clamps with positive locking
- Route TX and RX cabling on **opposite sides** of crossbar

---

## Testing Protocol Additions

### Additional Bench Tests (from consensus)
1. **Phase error measurement:** With only the primary present (no ground), measure the Q-channel output vs. I-channel. The ratio reveals the system phase error. Target: <0.5 degrees.
2. **Temperature sweep:** Operate in a cold/warm environment (e.g., 5C to 35C). Log all channels. Quantify drift in ppm/C.
3. **Microphonics test:** Tap the crossbar and coil mounts while monitoring the AD8421 output on an oscilloscope. Identify and dampen any mechanical resonances.
4. **Self-resonance check:** Sweep the TX frequency and monitor impedance to find coil self-resonant frequencies. Ensure they are well above the operating frequency.

### Additional Field Tests (from consensus)
1. **Height sensitivity:** Measure at multiple heights above ground (10cm, 20cm, 50cm, 1m). Verify the expected depth-sensitivity function.
2. **Orientation sensitivity:** Rotate the instrument 90 degrees (HCP to VCP) and verify the expected change in depth sensitivity.
3. **Repeatability:** Walk the same line 5 times. Quantify measurement repeatability (standard deviation at each point).

---

## Model Availability Note

The gemini-3-pro-preview model could not be consulted due to API quota exhaustion (429 RESOURCE_EXHAUSTED error, repeated across multiple retry attempts). The consensus is therefore based on:
1. openai/gpt-5.2-pro (successful, detailed analysis, confidence 8/10)
2. Claude Opus 4.6 independent engineering analysis

Both analyses independently converged on the same critical findings, providing high confidence in the consensus despite the missing second model perspective. A re-run with gemini-3-pro-preview could be attempted when the API quota resets to obtain a third independent viewpoint.

---

*Generated by PAL Consensus Validation -- 2026-02-18*
*Models: openai/gpt-5.2-pro (neutral), gemini-3-pro-preview (unavailable)*
*Continuation ID: ca7c3571-23fa-412b-953e-222c4967f7de*
