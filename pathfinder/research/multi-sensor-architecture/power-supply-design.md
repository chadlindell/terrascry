# R5: Power Supply Filter Chain Design — Consensus Validation

**Task**: R5 — Power Supply Filter Chain Design (CRITICAL ENGINEERING)
**Date**: 2026-02-18
**Status**: Consensus Complete
**Models consulted**: Claude Opus 4.6 (independent analysis) + openai/gpt-5.2-pro (neutral, 8/10 confidence)
**Note**: gemini-3-pro-preview was requested but unavailable (API quota exhausted). Synthesis proceeds with two-analysis consensus showing strong convergence on all issues.

---

## Executive Summary

The 3-stage power supply architecture (buck converter -> LC filter -> LDO) is **sound in concept** but contains **three critical design errors** that must be corrected before prototyping:

1. **LM78L05 cannot regulate at 5.5V input** (hard failure — 1.7V dropout vs 0.5V headroom)
2. **Undamped LC filter will ring catastrophically** during transients (Q > 100 possible)
3. **120 dB PSRR claim is overstated** (realistic: 95-110 dB — still excellent, but must not be assumed)

All three errors are solvable with component substitutions. No architectural redesign is needed.

---

## Issue-by-Issue Consensus Results

### Issue 1: LM78L05 Dropout Voltage (CRITICAL — DESIGN ERROR)

**Consensus**: UNANIMOUS — The LM78L05 will NOT regulate at 5.5V input.

**Analysis**:
- LM78L05 typical dropout: 1.7V (worst case at temperature: >2.0V)
- Available headroom: 5.5V - 5.0V = 0.5V
- Required headroom: >1.7V (minimum), meaning input must be >6.7V
- At 5.5V input, the LM78L05 operates as an unregulated pass-through (~5.3-5.4V output)
- Regulation impedance rises sharply out of dropout, **increasing** conducted crosstalk between sensors — the exact opposite of the design intent
- Failure worsens at elevated temperature (field conditions, 40C+) and during load transients

**Verdict**: This is a hard design error. The LM78L05 is the wrong part family for a post-5V-rail per-sensor isolation topology. It forces an upstream rail of 7V+ which conflicts with the 5.5V intermediate rail.

**Resolution**: See Issue 2.

### Issue 2: Replacement Regulator Selection (CRITICAL — CORRECTIVE ACTION)

**Consensus**: Replace all 8x LM78L05 with modern low-dropout regulators, fed from the post-LC-filter 5.5V rail (bypassing TPS7A49).

**Recommended Options (priority order)**:

| Priority | Part | Dropout | Iout Max | PSRR | Package | Unit Cost | Notes |
|----------|------|---------|----------|------|---------|-----------|-------|
| **PRIMARY** | AP2112K-5.0 | 250 mV | 600 mA | 70 dB @ 1 kHz | SOT-23-5 | ~$0.35 | Best availability, generous current margin, good PSRR |
| ALTERNATIVE | MCP1700-5002E/TO | 178 mV | 250 mA | ~46 dB @ 1 kHz | TO-92/SOT-23 | ~$0.40 | Lowest dropout; modest PSRR acceptable with upstream LC filter |
| PREMIUM | TPS7A20-5.0 | 300 mV | 300 mA | >60 dB @ 150 kHz | SOT-23 | ~$1.50 | Highest PSRR; best noise isolation per sensor |
| PREMIUM | TPS7A02-5.0 | 200 mV | 200 mA | Ultra-low noise | DSBGA | ~$1.80 | Ultimate quiet per-sensor supply; exotic package |

**Recommended**: AP2112K-5.0 offers the best balance of availability, headroom margin (500mV available >> 250mV dropout), current capability (600mA >> 12mA per FG-3+), and cost ($2.80 total for 8 units).

**Revised Topology**:
```
LiPo 7.4V --> LM2596 (->5.5V) --> Ferrite+LC --+--> TPS7A49 (->5.0V) --> ADS1115, LM2917 ref (precision analog)
                                                 |
                                                 +--> AP2112K-5.0 x8 --> FG-3+ sensors (per-sensor isolation)
                                                 |
                                                 +--> 3.3V LDO --> ESP32, digital sensors

LiPo 7.4V --> Direct --> OPA549 (EMI TX coil driver)
```

**Key Change**: The per-sensor regulators are fed from the 5.5V post-LC-filter rail, NOT from the TPS7A49 5.0V output. This:
- Provides adequate headroom (5.5V - 5.0V = 0.5V >> 250mV dropout for AP2112K)
- Offloads 96mA from the TPS7A49 (which only needs to supply ~25mA for precision analog)
- Still benefits from the LC filter stage attenuation of buck switching noise

### Issue 3: Shielded Inductor for LM2596 (CONSENSUS)

**Recommended Parts**:

| Part Number | Manufacturer | Inductance | Isat | DCR | Size | Notes |
|-------------|-------------|------------|------|-----|------|-------|
| **MSS1260-333ML** | Coilcraft | 33 uH | 2.3 A | 115 mOhm | 12.5x12.5x6mm | Primary recommendation |
| 744773033 | Wurth | 33 uH | 2.35 A | ~120 mOhm | 12x12x6mm | Alternative |
| WE-PD 744778933 | Wurth | 33 uH | 2.35 A | ~120 mOhm | 12x12x8mm | Alternative |

**Design Verification**:
- Peak inductor current: I_peak = I_out + (Vin - Vout) / (2 x L x f)
- I_peak = 1.0A + (7.4V - 5.5V) / (2 x 33uH x 150kHz) = 1.0 + 0.19 = 1.19A
- Both parts provide Isat > 2.0A, giving >60% margin over peak current
- GPT-5.2-pro recommends sizing for Isat >= 2.5A for additional transient margin

**IMPORTANT NOTE**: The LM2596 is a voltage-mode controller that requires ESR in its output capacitor for loop stability. Do NOT use only ceramic output caps — use an electrolytic or aluminum polymer with adequate ESR (per datasheet requirements, typically 20-100 mOhm). This is a known stability requirement for this older controller topology.

### Issue 4: Ferrite Bead Selection (CONSENSUS — CHANGE REQUIRED)

**Problem**: The BLM18PG601SN1 (0603, 600 Ohm @ 100 MHz) is inadequate for two reasons:
1. **Current rating**: 0603 beads are typically rated for <500mA; the 5V rail may carry >200mA continuous
2. **Frequency targeting**: 600 Ohm @ 100 MHz provides only ~10-20 Ohm impedance at 150 kHz — negligible attenuation at the buck switching frequency

**Recommended Replacement**:

| Part Number | Manufacturer | Impedance @ 100 MHz | DC Rating | Size | Notes |
|-------------|-------------|---------------------|-----------|------|-------|
| **BLM31PG601SN1L** | Murata | 600 Ohm | 2A | 1206 | Primary — same impedance profile, higher current |
| MPZ2012S601AT000 | TDK | 600 Ohm | 3A | 0805 | Alternative — compact, high current |

**Design Note**: The ferrite bead serves as a broadband high-frequency attenuator, complementing the LC filter which handles the fundamental switching frequency. The bead is most effective against harmonics (300 kHz, 450 kHz, etc.) and broadband switching noise. Do not rely on the ferrite bead for 150 kHz fundamental attenuation — that is the LC filter's job.

### Issue 5: LC Filter Resonance and Damping (CRITICAL — DESIGN ERROR)

**Problem**: The LC filter (10 uH + 100 uF) has a resonant frequency of ~5.03 kHz. With low-ESR components, the Q factor can be extremely high:

```
Q = (1/R_total) x sqrt(L/C)
  = (1/0.15) x sqrt(10e-6 / 100e-6)
  = 6.67 x 0.316
  = ~2.1 (if ESR is 0.15 Ohm total)
```

However, with modern low-ESR electrolytics (ESR as low as 20-50 mOhm) and low DCR inductors:
```
Q = (1/0.07) x sqrt(10e-6 / 100e-6)
  = 14.3 x 0.316
  = ~4.5
```

And at the resonant frequency, transient voltage can overshoot by a factor of Q. With the 10 uF ceramic in parallel (very low ESR), the effective Q seen during fast transients can be much higher.

**ADDITIONAL RISK (from GPT-5.2-pro)**: The LC filter placed after the buck converter can interact with the LM2596 control loop. The buck regulator expects to "see" its output capacitor ESR for loop stability. A strong LC filter between the buck output cap and the load creates an impedance peak that can cause the buck to oscillate or ring at the LC resonant frequency.

**Resolution — Add Damping**:
- **Option A (recommended)**: Series damping resistor of 0.3-0.5 Ohm in the capacitor branch of the LC filter. This reduces Q while adding minimal voltage drop at load current (0.5 Ohm x 0.2A = 100mV)
- **Option B**: Parallel damping resistor of 100-220 Ohm across the inductor. Provides frequency-dependent damping. Less power loss but less effective at low frequencies
- **Option C**: RC snubber across the filter output (e.g., 10 Ohm + 1 uF in series). Provides targeted damping without DC power loss

**Recommended Implementation**: Use Option A (0.3 Ohm series R) combined with ensuring the LM2596 output capacitor (before the LC filter) meets the controller's ESR requirements independently. The LC filter should be treated as a separate stage that does not affect the buck regulator's feedback loop stability.

### Issue 6: TPS7A49 PSRR and Total Attenuation (CONSENSUS — OVERSTATED)

**Original Claim**: 120 dB total attenuation at 150 kHz (LC filter 60 dB + LDO PSRR 60 dB)

**Corrected Analysis**:

| Stage | Attenuation at 150 kHz | Notes |
|-------|----------------------|-------|
| LC filter (10 uH + 100 uF) | ~59 dB | 20 x log10((150/5)^2) = 20 x log10(900) = 59.1 dB |
| TPS7A49 LDO PSRR | ~40-50 dB | Datasheet 60 dB figure is at lower frequencies; PSRR degrades with frequency and near dropout |
| **Revised Total** | **~95-110 dB** | Still excellent, but not 120 dB |

**Important Caveats** (from GPT-5.2-pro):
- PSRR degrades near dropout voltage — with only 500mV headroom on TPS7A49 (5.5V -> 5.0V), PSRR may be lower than typical datasheet curves
- PSRR varies with load current
- Layout coupling (capacitive, inductive, ground impedance) dominates beyond ~80-90 dB in practical PCB implementations
- The dB calculation assumes both attenuations apply linearly and independently — not guaranteed in practice

**Practical Ripple Estimate**:
- Buck converter output ripple: 30-50 mV p-p
- After 95 dB attenuation: 30mV / 10^(95/20) = 30mV / 56,234 = ~530 pV
- After 110 dB attenuation: 30mV / 10^(110/20) = 30mV / 316,228 = ~95 pV
- Both figures are well below measurement significance for fluxgate frequency-to-voltage conversion

**Recommendation**: Claim "~100 dB combined attenuation (Modeled)" in documentation. Validate with bench measurement using spectrum analyzer at ADC/reference pins.

### Issue 7: Star Grounding on 4-Layer PCB (PARTIAL DISAGREEMENT — RESOLVED)

**My initial analysis**: Traditional split ground planes with 0-Ohm bridge
**GPT-5.2-pro recommendation**: Continuous ground plane with net-tie separation and current-return management by placement/routing

**Consensus Resolution**: GPT-5.2-pro's approach aligns better with modern mixed-signal PCB practice.

**Recommended Implementation**:
```
4-Layer Stack:
  Layer 1: Signal (top) — components, short traces
  Layer 2: Ground (continuous) — solid copper pour, minimal splits
  Layer 3: Power — power distribution
  Layer 4: Signal (bottom) — longer traces, auxiliary routing
```

**Ground Strategy**:
1. Use a **continuous Layer 2 ground plane** (do not split into analog/digital zones)
2. Manage current return paths through **component placement**: group analog circuits (fluxgates, ADC, LM2917) together; group digital circuits (ESP32, SD, GPS) together
3. Place a **net-tie / 0 Ohm resistor** between analog and digital power return paths near the ADC reference point (the "quiet point" of the system)
4. Ensure **no high-current digital return paths** flow through the analog ground area
5. The buck converter and OPA549 high-current returns must route directly to the battery negative without passing through the analog sensing area
6. Keep sensitive analog signals (ADS1115 inputs, LM2917 inputs) routed exclusively over the analog-side ground area

**Rationale**: Hard ground splits often create worse EMI problems than they solve — signals that must cross the split have no return path, creating large loop antennas. A continuous plane with disciplined placement achieves the same noise isolation without these risks.

### Issue 8: Decoupling Strategy (CONSENSUS — ADEQUATE WITH ADDITIONS)

**Current Specification**: 100 nF X7R + 10 uF at each LM78L05 (now AP2112K) output

**Consensus Assessment**: Baseline adequate. Additions recommended:

| Location | Current | Add | Purpose |
|----------|---------|-----|---------|
| Each AP2112K input | (none specified) | **1 uF X7R** | LDO input stability per datasheet |
| Each AP2112K output | 100 nF + 10 uF | Adequate as-is | Local bypass + bulk |
| Each FG-3+ sensor | (none specified) | **22-47 uF** | Local bulk if sensor has pulsed current draw |
| TPS7A49 output | 10 uF + 1 uF | Adequate as-is | LDO stability |
| LM2596 input | (none specified) | **100 uF electrolytic** | Transient absorption from OPA549 battery sag |
| LM2596 output | (per datasheet) | Verify ESR requirement | Buck loop stability |

**IMPORTANT**: Follow the AP2112K (or chosen LDO) datasheet for minimum output capacitance and ESR requirements. The AP2112K requires a minimum of 1 uF output capacitance with ESR < 5 Ohm for stability.

### Issue 9: OPA549 EMI TX Isolation (CONSENSUS — ADEQUATE WITH PRECAUTIONS)

**Current Design**: OPA549 fed directly from LiPo battery, not through the analog supply chain.

**Consensus Assessment**: Correct approach. The direct battery feed provides inherent isolation from the analog supply chain. However, three additional precautions are needed:

1. **Input bulk capacitance at LM2596**: Add 100 uF electrolytic near the LM2596 input. During OPA549 TX phase (500 mA draw), battery voltage sags by I x R_batt (approximately 0.5A x 0.1 Ohm = 50 mV). This sag appears at the LM2596 input. The buck converter's line regulation plus the subsequent filter chain provides >80 dB attenuation, but local bulk capacitance reduces the disturbance at the source.

2. **Separate high-current ground return routing**: The OPA549 TX current (up to 500 mA pulsed) must return to the battery negative via a dedicated trace/pour that does NOT pass through or near the analog ground area. Route the OPA549 power return directly to the battery star point on its own copper path.

3. **Pi-filter at OPA549 input (optional)**: Consider a ferrite bead + capacitor pi-filter at the OPA549 power input to confine high-frequency switching transients within the TX circuit. This is belt-and-suspenders but inexpensive insurance.

---

## Additional Findings

### Power Budget Errors

The power budget table in the architecture document contains inconsistencies:

| Issue | Details |
|-------|---------|
| **RPLiDAR C1 (700 mA, 3.5W)** | Listed in the table but NOT included in "Total (continuous) ~2.1W". RPLiDAR must be on its own regulator directly from LiPo or buck output — it CANNOT run through the analog supply chain |
| **ESP32-CAM (300 mA, 1.5W)** | Same issue — needs separate regulation, not on analog supply path |
| **Regulator losses not included** | LDO drops (5.5V -> 3.3V for ESP32 = 0.53W at 240mA) and buck efficiency (~85%) reduce actual battery runtime |
| **TPS7A49 current margin** | With per-sensor LDOs moved to post-LC rail, TPS7A49 only supplies ~25 mA (ADS1115 + LM2917) — well within 150 mA limit |

### Corrected Power Budget

| Subsystem | Rail | Current | Power | Regulator |
|-----------|------|---------|-------|-----------|
| 8x FG-3+ (via AP2112K x8) | 5.5V -> 5.0V | 96 mA | 0.48W | 8x AP2112K-5.0 |
| 2x ADS1115 | 5.0V (TPS7A49) | 0.4 mA | 0.002W | TPS7A49 |
| 8x LM2917 | 5.0V (TPS7A49) | 24 mA | 0.12W | TPS7A49 |
| ESP32 | 3.3V | 240 mA | 0.79W | 3.3V LDO (or dedicated buck) |
| ZED-F9P GPS | 3.3V | 68 mA | 0.22W | 3.3V LDO |
| BNO055 IMU | 3.3V | 12 mA | 0.04W | 3.3V LDO |
| MLX90614 IR | 3.3V | 1.5 mA | 0.005W | 3.3V LDO |
| **Analog subtotal** | | ~120 mA | ~0.60W | TPS7A49 + AP2112Ks (from post-LC 5.5V) |
| **Digital 3.3V subtotal** | | ~321 mA | ~1.06W | 3.3V regulator |
| **Continuous total** | | | **~1.66W** | |
| | | | | |
| AD9833 + OPA549 (TX) | Direct LiPo | 500 mA | 2.5W | Direct battery |
| RPLiDAR C1 | 5V (separate reg) | 700 mA | 3.5W | Separate buck from LiPo |
| ESP32-CAM | 5V (separate reg) | 300 mA | 1.5W | Separate regulator |
| **Peak total (all active)** | | | **~9.16W** | |

### 3.3V Rail Thermal Concern (NEW — from GPT-5.2-pro)

With ESP32 drawing 240 mA (peaks higher during WiFi TX), a linear LDO from 5.5V -> 3.3V dissipates:
```
P_dissipation = (5.5V - 3.3V) x 0.24A = 0.53W
```

This is significant for a small LDO in an enclosed space. Options:
1. **Accept it**: 0.53W is manageable with adequate copper pour for heat sinking and SOT-223 or similar package with exposed pad
2. **Dedicated 3.3V buck converter**: TPS62162 (1A) or TPS62130 (3A) — higher efficiency, less heat, but adds a switching noise source to the digital domain (acceptable since digital domain is isolated from analog)

**Recommendation**: Use a dedicated 3.3V buck converter for the digital rail. The added switching noise is on the digital side only and does not affect analog measurements. The thermal savings and efficiency improvement are worth the additional component.

---

## Consolidated Component Recommendations (BOM Changes)

### Components to REMOVE

| Qty | Part | Reason |
|-----|------|--------|
| 8 | LM78L05 (TO-92) | Cannot regulate at 5.5V input |
| 1 | BLM18PG601SN1 (0603 ferrite) | Undersized for current, wrong frequency profile |

### Components to ADD

| Qty | Part Number | Description | Approx Unit Cost |
|-----|-------------|-------------|-----------------|
| 8 | AP2112K-5.0TRG1 | 5.0V LDO, 600mA, SOT-23-5 | $0.35 |
| 1 | BLM31PG601SN1L | 600 Ohm @ 100 MHz ferrite bead, 1206, 2A | $0.10 |
| 1 | MSS1260-333ML | 33 uH shielded inductor, 2.3A sat | $2.50 |
| 1 | 0.3 Ohm resistor (0805) | LC filter damping resistor | $0.02 |
| 8 | 1 uF X7R ceramic (0402/0603) | AP2112K input capacitors | $0.01 |
| 1 | 100 uF electrolytic (low ESR) | LM2596 input bulk capacitor | $0.15 |
| 1 | TPS62162DSG (optional) | 3.3V 1A buck for digital rail | $1.80 |

**Net BOM impact**: Approximately +$3-5 depending on whether the 3.3V buck is added. The AP2112K units ($2.80 total) replace the LM78L05 units at similar cost.

---

## Revised Architecture Diagram

```
                                                    PRECISION ANALOG RAIL
                                                    ┌─────────────────────┐
                                                    │  TPS7A49            │
                                              ┌────>│  5.5V -> 5.0V      │───> ADS1115, LM2917 ref
                                              │     │  ~25 mA load       │     (ultra-low-noise)
                                              │     └─────────────────────┘
                                              │
LiPo ──> LM2596 ──> [Ferrite + LC + Damping] ─┤     PER-SENSOR ISOLATION
7.4V     -> 5.5V     BLM31 + 10uH + 100uF     │     ┌─────────────────────┐
2S       150 kHz     + 0.3 Ohm damping         ├────>│  AP2112K-5.0 x8     │───> FG-3+ sensors x8
         MSS1260                                │     │  5.5V -> 5.0V       │     (12 mA each)
         -333ML                                 │     │  500mV headroom     │
                                                │     └─────────────────────┘
                                                │
                                                │     DIGITAL RAIL
                                                │     ┌─────────────────────┐
                                                └────>│  TPS62162 (buck)    │───> ESP32, ZED-F9P,
                                                      │  5.5V -> 3.3V      │     BNO055, BMP390,
                                                      │  ~321 mA load      │     MLX90614, SD card
                                                      └─────────────────────┘

LiPo ──> Direct ──────────────────────────────────> OPA549 (EMI TX, 500mA pulsed)
7.4V     (100uF bulk cap at LM2596 input)           (separate ground return to battery)

LiPo ──> Separate Buck ───────────────────────────> RPLiDAR C1 (700mA, 5V)
7.4V     (dedicated regulator)                       ESP32-CAM (300mA, 5V)
```

---

## Summary of Consensus Decisions

| # | Issue | Severity | Decision | Confidence |
|---|-------|----------|----------|------------|
| 1 | LM78L05 dropout | CRITICAL | Replace with AP2112K-5.0 | High (unanimous) |
| 2 | Replacement LDO selection | CRITICAL | AP2112K-5.0 as primary; MCP1700 or TPS7A20 as alternatives | High |
| 3 | Shielded inductor | Standard | Coilcraft MSS1260-333ML, 33 uH, 2.3A | High |
| 4 | Ferrite bead | Moderate | Upgrade to BLM31PG601SN1L (1206, 2A) | High |
| 5 | LC filter damping | CRITICAL | Add 0.3 Ohm series damping resistor | High |
| 6 | PSRR total claim | Moderate | Downgrade from 120 dB to ~100 dB (Modeled); validate with bench measurement | High |
| 7 | Star grounding | Moderate | Continuous ground plane with net-tie, not hard split | High (GPT-5.2-pro approach preferred) |
| 8 | Decoupling | Standard | Current spec adequate; add 1 uF input caps per AP2112K, 100 uF at LM2596 input | High |
| 9 | OPA549 isolation | Standard | Adequate with separate ground return and input bulk cap | High |
| -- | 3.3V rail thermal | NEW | Consider dedicated 3.3V buck (TPS62162) for digital rail | Moderate |
| -- | Power budget errors | NEW | RPLiDAR and ESP32-CAM need separate regulators; totals need correction | High |

---

## Open Items for Bench Validation

1. **AP2112K PSRR verification**: Measure output noise spectrum of AP2112K at 5.5V -> 5.0V with 12mA load; verify PSRR at 150 kHz harmonics
2. **LC filter transient response**: Apply step load and measure ringing at LC filter output with 0.3 Ohm damping resistor; adjust R value if needed
3. **LM2596 loop stability**: Verify buck converter stability with the LC filter stage as load; check for oscillation or excessive ringing at startup and load transients
4. **Total PSRR measurement**: Use spectrum analyzer at ADS1115 reference pin to measure actual supply noise at 150 kHz and harmonics; compare against ~100 dB (Modeled) estimate
5. **FG-3+ crosstalk with AP2112K**: Measure conducted crosstalk between two FG-3+ sensors on adjacent AP2112K regulators; confirm isolation is adequate for gradiometer-grade measurements
6. **Thermal measurement**: Monitor AP2112K junction temperature at 12mA load and TPS62162 temperature at 321mA load in enclosed spaces

---

## References

- LM2596 datasheet: Texas Instruments SNVS124E
- TPS7A49 datasheet: Texas Instruments SBVS163B
- AP2112K datasheet: Diodes Incorporated (see Output Capacitor Stability Requirements section)
- MCP1700 datasheet: Microchip DS20001826
- MSS1260 datasheet: Coilcraft Document 361 (see DC Bias curves for saturation verification)
- BLM31PG601SN1L datasheet: Murata (see impedance vs frequency curves)
- TPS62162 datasheet: Texas Instruments SLVSB16
- TI Application Note SLYT107: Power Supply Design for Mixed-Signal Systems

---

*Generated by PAL consensus validation, 2026-02-18*
*Models: Claude Opus 4.6 (independent) + openai/gpt-5.2-pro (neutral, 8/10 confidence)*
*gemini-3-pro-preview: unavailable (API quota exhausted)*
