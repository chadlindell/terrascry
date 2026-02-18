# EMI Coil Prototype Specifications

## Research Task R3 -- PAL Consensus Validation

**Date:** 2026-02-18
**Task:** CRITICAL MATH -- EMI (FDEM) Coil Prototype Design Validation
**Method:** PAL Multi-Model Consensus
**Models Consulted:**
- Claude Opus 4.6 (independent analysis)
- openai/gpt-5.2-pro (neutral stance) -- **successful**
- gemini-3-pro-preview (neutral stance) -- **UNAVAILABLE** (429 RESOURCE_EXHAUSTED, daily quota exceeded)

**Consensus Confidence:** High (2 of 3 models, strong cross-validation with detailed calculations)

---

## Design Under Evaluation

| Parameter | Value |
|-----------|-------|
| Operating frequency | 15 kHz |
| TX coil | 30 turns, 12 cm diameter, enameled copper wire |
| RX coil | 30 turns, 8 cm diameter |
| Coil separation | 1.0 m (HCP geometry) |
| Exploration depth | ~1.5 m (1.5x separation for HCP) |
| TX drive current | 100 mA (target) |

### Signal Chain

```
AD9833 DDS (15 kHz) --> OPA549 Power Amp --> TX Coil (30T, 12cm)
                                              |
                                         [Primary field through ground]
                                              |
RX Coil (30T, 8cm) --> AD8421 Preamp (G=100) --> AD630 Phase Detector --> RC LPF (10 Hz) --> ADS1115 ADC
                                                       |
                                                  [Bucking coil cancels primary]
```

---

## Q1: TX Coil Inductance -- DESIGN ERROR IDENTIFIED

### Error in Original Formulation

The proposed formula `L = u0 * N^2 * A / l` is for a **long solenoid** where `l >> diameter`. For this coil geometry (l/d ratio ~ 0.25), this formula **grossly overestimates** inductance.

### Corrected Calculation (Wheeler's Approximation)

Wheeler's single-layer formula (inputs in inches, output in uH):

```
L(uH) = r^2 * N^2 / (9r + 10l)
```

**TX Coil Parameters:**
- Diameter = 12 cm, radius r = 6 cm = 2.362 inches
- N = 30 turns
- Winding length l (depends on wire gauge):
  - 0.6 mm wire+insulation: l = 18 mm = 0.709 in
  - 1.0 mm wire+insulation: l = 30 mm = 1.181 in

| Wire assumption | Winding length | Wheeler L | Solenoid L (wrong) |
|-----------------|---------------|-----------|---------------------|
| 0.6 mm pitch | 18 mm | **177 uH** | 710 uH |
| 1.0 mm pitch | 30 mm | **152 uH** | 426 uH |

**Consensus value: L_TX = 150--180 uH** (Modeled)

### Resonant Capacitor at 15 kHz

```
C = 1 / (4 * pi^2 * f^2 * L)
```

| Inductance | Resonant C |
|-----------|-----------|
| 150 uH | 750 nF |
| 177 uH | 636 nF |
| 180 uH | 625 nF |

**Consensus value: C_res = 0.56--0.75 uF** (Modeled)

**Note:** Resonance is not strictly required to operate -- the OPA549 can drive the coil directly -- but series resonance reduces amplifier VA requirements and stabilizes TX current. If using resonance, temperature drift of L and C must be managed (NP0/C0G capacitors recommended).

---

## Q2: Primary Field at RX -- CRITICAL GEOMETRY ERROR IDENTIFIED

### Error in Original Formulation

The formula `B = u0*N*I*A / (2*(r^2 + d^2)^(3/2))` calculates the **on-axis (axial)** field of a circular coil. In HCP (Horizontal Coplanar) geometry, the TX and RX coils are **side-by-side in the same horizontal plane** -- this is the **equatorial** position, NOT the on-axis position.

### On-Axis vs. Equatorial Field

For a magnetic dipole with moment m = NIA at distance d:

```
B_axial     = (u0 / 4pi) * 2m / d^3    (on-axis, along dipole direction)
B_equatorial = (u0 / 4pi) *  m / d^3    (side-by-side, perpendicular to dipole)
```

**The equatorial field is exactly HALF the axial field.**

But for HCP geometry, the RX coil axis is vertical (parallel to TX coil axis), sensing the vertical component of the field. The relevant component at the equatorial position is:

```
B_z(equatorial) = -(u0 / 4pi) * m / r^3
```

### Corrected Calculation

```
TX parameters:
  A_TX = pi * (0.06)^2 = 1.131e-2 m^2
  I = 100 mA = 0.1 A
  N = 30
  m = N * I * A = 30 * 0.1 * 1.131e-2 = 0.03393 A*m^2

At d = 1.0 m (equatorial, HCP):
  B_primary = (u0 / 4pi) * m / d^3
            = 1e-7 * 0.03393 / 1.0
            = 3.39e-9 T
            = 3.4 nT
```

| Calculation | B at RX (1m) | Error factor |
|------------|-------------|-------------|
| **Equatorial (CORRECT for HCP)** | **3.4 nT** | -- |
| On-axis (WRONG for HCP) | 6.8 nT | 2x overestimate |
| Original doc formula | 21.2 nT | ~6x overestimate |

### Primary-Induced EMF in RX Coil

```
V_primary = N_RX * A_RX * omega * B_primary  (peak)

Where:
  N_RX = 30
  A_RX = pi * (0.04)^2 = 5.027e-3 m^2
  omega = 2 * pi * 15000 = 94,248 rad/s
  B_primary = 3.4e-9 T

V_primary = 30 * 5.027e-3 * 94248 * 3.4e-9
          = 48 uV peak
          = 34 uV RMS
```

**Consensus value: V_primary at RX = 48 uV peak / 34 uV RMS** (Modeled)

---

## Q3: Bucking Coil Specification

### Requirement

Cancel the 3.4 nT primary field at the RX coil location. The bucking coil is wound in opposition to TX and positioned between TX and RX, close to RX.

### Design Calculation

For a small bucking coil (dipole approximation) at distance d_b from RX center:

```
B_buck = (u0 * N_b * I_b * A_b) / (2 * d_b^3)

Need: B_buck = B_primary = 3.4 nT
```

If the bucking coil carries the same current as TX (series-connected, I_b = 0.1 A):

| Bucking coil radius | Distance from RX | Required turns |
|---------------------|-------------------|---------------|
| 5 mm | 5 cm | 0.4 turns (impractical) |
| 5 mm | 7 cm | 1.1 turns |
| 5 mm | 10 cm | 3.2 turns |
| 10 mm | 7 cm | 0.3 turns (impractical) |
| 10 mm | 10 cm | 0.8 turns |

### Consensus Recommendation

**STRONG CONSENSUS: Fixed geometric bucking is impractical.**

Both analyses independently concluded:

1. The required turns count is fractional or near-unity, making fixed geometry extremely sensitive to position errors.
2. Field scales as 1/d^3 -- a 1 mm positioning error at 7 cm causes ~4% field change.
3. Temperature drift will shift coil dimensions and change coupling.

**Recommended approach: Tunable bucking injection.**

```
Design:
  - Small bucking coil: 1 turn, ~10 mm diameter, at 7-10 cm from RX
  - Connected via adjustable attenuator (trim potentiometer on a separate winding
    or variable resistor in a current divider)
  - Optional: phase trim network for fine null adjustment
  - Mount on adjustable slide along the crossbar for coarse positioning
  - Fine adjustment via electrical trimming
  - Target: >99% primary cancellation (residual <34 ppm of primary)
  - Residual primary removed by phase-sensitive detector (AD630)
```

---

## Q4: Expected Signal Levels -- DOC TABLE ERROR IDENTIFIED

### LIN Formula Calculation

```
Hs/Hp = i * omega * u0 * sigma * s^2 / 4

Where:
  omega = 2 * pi * 15000 = 94,248 rad/s
  u0 = 4 * pi * 1e-7 = 1.2566e-6 H/m
  sigma = 50 mS/m = 0.05 S/m
  s = 1.0 m

|Hs/Hp| = 94248 * 1.2566e-6 * 0.05 * 1.0 / 4
        = 5.924e-3 / 4
        = 1.481e-3
        = 1481 ppm (quadrature component)
```

### Corrected Signal Level Table

The table in `emi-coil-design.md` (lines 42-49) shows ~500 ppm for 50 mS/m. This is **inconsistent** with the LIN formula at s=1.0 m and f=15 kHz. The table values appear to correspond to either s ~ 0.58 m or f ~ 5 kHz.

**Corrected table for s = 1.0 m, f = 15 kHz:**

| Soil conductivity | sigma (S/m) | Hs/Hp (ppm) | Notes |
|-------------------|-------------|-------------|-------|
| 10 mS/m (dry sand) | 0.01 | **296 ppm** | Weak signal, near noise floor |
| 50 mS/m (loam) | 0.05 | **1481 ppm** | Moderate |
| 100 mS/m (wet clay) | 0.10 | **2962 ppm** | Strong |
| 300 mS/m (saline) | 0.30 | **8886 ppm** | Very strong |

### Voltage at AD8421 Output

Using the corrected equatorial primary field:

```
V_secondary = |Hs/Hp| * V_primary_RMS

For sigma = 50 mS/m:
  V_secondary = 1.481e-3 * 34 uV = 50.4 nV RMS  (at RX coil)

After AD8421 (gain = 100):
  V_out = 100 * 50.4 nV = 5.04 uV RMS
```

| Conductivity | V at RX coil | V after AD8421 (G=100) |
|-------------|-------------|----------------------|
| 10 mS/m | 10.1 nV RMS | 1.01 uV RMS |
| 50 mS/m | 50.4 nV RMS | 5.04 uV RMS |
| 100 mS/m | 100.7 nV RMS | 10.07 uV RMS |
| 300 mS/m | 302.1 nV RMS | 30.21 uV RMS |

**Consensus values confirmed by both models.** (Modeled)

---

## Q5: AD630 Phase Detection -- TWO CHANNELS REQUIRED

### Consensus Finding

A single AD630 outputs a signal proportional to the input multiplied by the reference:

```
V_out = V_in * sign(V_ref)  (simplified)
```

This extracts **one phase component only** (I or Q, depending on reference phase).

### Requirements for Simultaneous I/Q

**Option A (RECOMMENDED for prototype): Dual AD630**
```
RX coil --> AD8421 (G=100) --+--> AD630 #1 (0 deg ref)   --> RC LPF --> ADS1115 Ch0 (In-phase)
                              |
                              +--> AD630 #2 (90 deg ref)  --> RC LPF --> ADS1115 Ch1 (Quadrature)
```
- Requires 90-degree phase shift of reference signal from AD9833
- AD9833 can output two synchronized signals; or use RC all-pass network for 90-deg shift
- Two ADS1115 channels needed (differential inputs preferred)

**Option B (alternative): Digital Lock-In**
```
RX coil --> AD8421 (G=100) --> ADC (>=100 ksps) --> Digital I/Q demodulation on ESP32
```
- Eliminates AD630 entirely
- Simpler calibration, easier drift correction
- Requires faster ADC than ADS1115 (which maxes at 860 SPS)
- Would need external ADC like ADS131M04 (32 ksps) or similar

**Option C (time-multiplexed): Single AD630**
- Alternate between 0-deg and 90-deg reference each measurement cycle
- Halves effective integration time
- Acceptable if measurement rate is not critical

### ADS1115 Caveat

The ADS1115 multiplexes between channels -- it does **not** sample simultaneously. For I/Q measurements where phase accuracy matters, ensure adequate settling time between channel switches, or use two separate ADS1115 devices for truly independent sampling.

---

## Q6: Noise Floor and Minimum Detectable Conductivity

### Input-Referred Noise

```
AD8421 voltage noise: e_n = 3 nV/sqrt(Hz)
Post-demod bandwidth: BW = 10 Hz (RC LPF cutoff)

Noise (input-referred, in detection bandwidth):
  V_noise = e_n * sqrt(BW) = 3 * sqrt(10) = 9.49 nV RMS
```

### Signal-to-Noise Ratio

Using corrected equatorial field values:

| Conductivity | V_signal (at RX) | SNR | Detectable? |
|-------------|-----------------|-----|------------|
| 1 mS/m | 1.01 nV | 0.11 | No |
| 5 mS/m | 5.04 nV | 0.53 | No |
| **10 mS/m** | **10.07 nV** | **1.06** | **Marginal (SNR=1)** |
| 20 mS/m | 20.15 nV | 2.12 | Yes (marginal) |
| 50 mS/m | 50.37 nV | 5.31 | Yes |
| 100 mS/m | 100.7 nV | 10.6 | Yes (good) |

### Minimum Detectable Conductivity

```
At SNR = 1:  sigma_min = ~10 mS/m
At SNR = 3:  sigma_min = ~30 mS/m  (more realistic detection threshold)
```

**This is MARGINAL.** The EM38 achieves ~1 mS/m sensitivity, making this prototype roughly 10-30x less sensitive.

### Real-World Noise Sources (not included above)

The 9.49 nV floor assumes **only** AD8421 voltage noise. In practice:
- Environmental EMI (power lines, radio) can dominate
- Mechanical vibration causing coil-position changes (microphonic coupling)
- Residual primary field leakage through imperfect bucking
- AD8421 current noise (0.2 pA/sqrt(Hz)) across coil impedance
- 1/f noise corner of AD8421 (~10 Hz) is right at the detection band

**Realistic minimum detectable: ~20-50 mS/m** without careful shielding, mechanical stabilization, and multi-cycle averaging.

### Strategies to Improve SNR

1. **Increase TX dipole moment** (most effective):
   - Increase current: 100 mA --> 500 mA (5x improvement, OPA549 can handle it)
   - Increase TX turns: 30 --> 60 turns (2x, but doubles inductance)
   - Increase TX diameter: 12 cm --> 20 cm (2.8x area improvement)
2. **Reduce noise bandwidth**: 10 Hz --> 1 Hz LPF (sqrt(10) = 3.2x improvement)
3. **Use series resonance** to boost TX current at constant amplifier power
4. **Digital lock-in** with longer integration time
5. **Averaging**: N measurements --> sqrt(N) improvement

---

## Q7: OPA549 Suitability

### Electrical Analysis

```
TX coil impedance at 15 kHz:
  X_L = 2 * pi * 15000 * 177e-6 = 16.7 ohms (reactive)
  R_coil ~ 2-5 ohms (estimated, 30 turns of ~0.4m circumference = 12m wire)
  |Z| = sqrt(R^2 + X_L^2) ~ 17-18 ohms

For I = 100 mA peak:
  V_required = |Z| * I = 18 * 0.1 = 1.8 V peak

OPA549 capabilities:
  Output voltage swing: +/- (Vs - 4V), so with +/-12V supply: +/-8V -- ADEQUATE
  Output current: up to 8A continuous -- MASSIVELY OVERKILL for 100mA
  Slew rate: 9 V/us
  Required slew rate: 2*pi*15000*1.8 = 0.17 V/us -- ADEQUATE (53x margin)
  Bandwidth: 900 kHz -- ADEQUATE (60x margin)
```

### Concerns

1. **Quiescent current**: OPA549 draws ~30 mA quiescent. For a battery-powered portable instrument, this is significant.
2. **Thermal shutdown pin**: Available (useful for TDM gating during fluxgate window).
3. **Inductive load stability**: The OPA549 may oscillate driving a purely inductive load. **Mandatory: add output Zobel network** (series R-C from output to ground, e.g., 10 ohm + 100 nF).
4. **Current regulation**: OPA549 is a voltage-mode amplifier. For stable magnetic dipole moment, add a **sense resistor (1-10 ohm) in series with the TX coil** and monitor the current. Better yet, use the sense voltage as feedback for a current-control loop.

### Verdict

**OPA549 works but is overkill.** For a prototype, it is acceptable. For a production/field version, consider:
- OPA548 (similar but lower quiescent)
- LT6020 (lower power, lower noise)
- MOSFET H-bridge with DDS-driven gate control (best efficiency for battery operation)
- Current-mode drive with sense feedback regardless of amplifier choice

---

## Summary of Design Errors Found

| # | Error | Severity | Impact |
|---|-------|----------|--------|
| 1 | Solenoid inductance formula used for short coil | **Moderate** | 2-3x overestimate of L; wrong resonant cap value |
| 2 | On-axis field formula used for HCP equatorial geometry | **CRITICAL** | ~6x overestimate of primary field; all downstream signal/noise calculations affected |
| 3 | PPM table in emi-coil-design.md inconsistent with LIN formula | **Moderate** | Misleading signal level expectations; values ~3x too low for s=1.0m at 15kHz |
| 4 | Single AD630 shown for I and Q extraction | **Moderate** | Cannot extract both simultaneously; need 2x AD630 or digital lock-in |
| 5 | Bucking coil described as fixed geometry | **Minor** | Will require tunable approach in practice |

---

## Consensus Design Recommendations

### Immediate Corrections Required

1. **Update `emi-coil-design.md` ppm table** to match the LIN formula at s=1.0 m, f=15 kHz.
2. **Use equatorial field formula** for all HCP primary field calculations.
3. **Use Wheeler's formula** (or measure directly) for coil inductance.
4. **Add second AD630** to signal chain diagram for simultaneous I/Q.

### Prototype Build Specifications (Modeled)

```
TX COIL:
  - 30 turns, 12 cm diameter (6 cm radius)
  - AWG 24 enameled copper wire (~0.56 mm dia with insulation)
  - Single-layer winding, ~18 mm winding length
  - Expected inductance: ~177 uH (measure with LCR meter before assembly)
  - Optional: series resonant capacitor 636 nF (use 680 nF standard + trim)
  - DC resistance: ~2-3 ohms

RX COIL:
  - 30 turns, 8 cm diameter (4 cm radius)
  - AWG 28 enameled copper wire
  - Single-layer winding
  - Expected inductance: ~65 uH (Wheeler estimate)

BUCKING COIL:
  - 1 turn, 10 mm diameter
  - Position: 7-10 cm from RX center, between TX and RX
  - Mounted on adjustable slide (coarse) + electrical trim (fine)
  - Series resistor (trimpot, 0-100 ohm) for current attenuation
  - Target: >40 dB primary suppression

SEPARATION:
  - TX-to-RX: 1.0 m center-to-center
  - HCP geometry (both coils horizontal, parallel to ground)
  - Mounted on rigid crossbar (carbon fiber or aluminum)
  - Mechanical rigidity is CRITICAL -- any flex changes primary coupling

DRIVE:
  - AD9833 DDS at 15.000 kHz
  - OPA549 in voltage-follower configuration with Zobel network
  - 1 ohm sense resistor in series with TX coil
  - Target TX current: 100-500 mA (higher = better SNR)
  - Add shutdown control for TDM gating

RECEIVE:
  - AD8421 (G=100, set by R_G = 505 ohms)
  - 2x AD630 phase detectors (0-deg and 90-deg reference)
  - 2x RC LPF (fc = 10 Hz: R = 16k, C = 1 uF)
  - 2x ADS1115 channels (I and Q)

EXPECTED PERFORMANCE (Modeled):
  Primary at RX:      3.4 nT / 48 uV peak EMF in RX
  Signal at 50 mS/m:  ~50 nV RMS at RX / ~5 uV after preamp
  Noise floor:        ~9.5 nV RMS (input-referred, 10 Hz BW)
  Min detectable:     ~10 mS/m (SNR=1), ~30 mS/m (SNR=3)
  Exploration depth:  ~1.5 m (HCP, s=1.0m)
```

### Future Improvements (Phase 2)

1. **Increase TX moment**: Move to 500 mA drive (5x SNR improvement --> sigma_min ~2 mS/m)
2. **Digital lock-in**: Replace AD630s with fast ADC + ESP32 DSP
3. **Reduce LPF bandwidth**: 10 Hz --> 1 Hz for stationary measurements
4. **Multi-frequency**: AD9833 can sweep frequencies for depth sounding
5. **Active bucking**: Closed-loop primary cancellation using DSP

---

## Consensus Agreement Matrix

| Question | Claude Opus 4.6 | openai/gpt-5.2-pro | Agreement |
|----------|-----------------|---------------------|-----------|
| Q1: Inductance formula error | Yes | Yes | FULL |
| Q1: Wheeler L ~ 150-180 uH | 152 uH | 177 uH | Agree (range) |
| Q1: Resonant C ~ 0.6-0.75 uF | 740 nF | 630 nF | Agree (range) |
| Q2: HCP = equatorial field | Initially used on-axis | Corrected to equatorial | CORRECTED |
| Q2: B_primary ~ 3.4 nT | (21.2 nT on-axis) | 3.4 nT equatorial | GPT-5.2-pro correct |
| Q3: Tunable bucking required | Yes | Yes | FULL |
| Q4: Hs/Hp ~ 1480 ppm @ 50 mS/m | 1481 ppm | 1480 ppm | FULL |
| Q4: Doc table is wrong | Yes | Yes | FULL |
| Q5: Need 2x AD630 for I/Q | Yes | Yes | FULL |
| Q6: Min detectable ~ 10 mS/m | ~1 mS/m (on-axis) --> 10 mS/m (corrected) | ~10 mS/m | FULL (after correction) |
| Q7: OPA549 works but overkill | Yes | Yes | FULL |

---

## References

- McNeill, J.D. (1980). *Electromagnetic terrain conductivity measurement at low induction numbers.* Technical Note TN-6, Geonics Limited.
- Wheeler, H.A. (1928). Simple Inductance Formulas for Radio Coils. *Proceedings of the IRE*, 16(10), 1398-1400.
- Ward, S.H., & Hohmann, G.W. (1988). Electromagnetic theory for geophysical applications. *Electromagnetic Methods in Applied Geophysics*, Vol. 1, SEG.
- Analog Devices AD8421 datasheet (Rev. B).
- Analog Devices AD630 datasheet (Rev. E).
- Texas Instruments OPA549 datasheet (Rev. D).

---

*Generated by PAL Consensus Validation (R3). Two of three requested models responded successfully. All critical math validated with detailed unit-tracked calculations.*
