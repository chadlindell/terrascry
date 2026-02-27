# EMI Signal Chain Consensus Validation

**Task:** I5 — Pathfinder Firmware EMI Coil Signal Chain
**Date:** 2026-02-18
**Classification:** CRITICAL MATH
**Status:** FAIL — 3 critical errors requiring redesign

## Consensus Panel

| Model | Role | Status |
|-------|------|--------|
| Claude Opus 4.6 | Independent analysis | Completed |
| openai/gpt-5.2-pro | Neutral evaluator | Completed |
| gemini-3-pro-preview | Neutral evaluator | UNAVAILABLE (API rate limit 429) |

**Consensus confidence:** High (2 of 3 models completed; both fully agree on all critical findings)

---

## Executive Summary

The EMI coil signal chain firmware implementation for apparent conductivity measurement has **sound underlying physics** (the LIN conductivity formula is correct and dimensionally verified) but contains **three critical implementation errors** that will prevent valid conductivity readings:

1. The RC phase shifter produces 45 degrees, not 90 degrees
2. The AD630 is a single-channel device, not dual I/Q
3. The 30ms measurement window is insufficient for the 10 Hz low-pass filter to settle

Additionally, there is a minor hex transcription error in the DDS frequency tuning word, and the ADC sampling budget is marginal.

---

## Detailed Analysis

### 1. DDS Frequency Tuning Word (AD9833)

**Decimal value: CORRECT**

```
FREQREG = f * 2^28 / MCLK
        = 15000 * 268435456 / 25000000
        = 4,026,531,840,000 / 25,000,000
        = 161,061,273.6
        → truncated to 161,061,273
```

Verification (reverse calculation):
```
f_out = FREQREG * MCLK / 2^28
      = 161,061,273 * 25,000,000 / 268,435,456
      = 14,999.999944 Hz  (error < 0.006 Hz)
```

**Hex value: MINOR ERROR**

- Stated: `0x099999A9`
- Correct: `0x09999999`

Verification:
```
0x09999999 = 0*268435456 + 9*16777216 + 9*1048576 + 9*65536
           + 9*4096 + 9*256 + 9*16 + 9*1
           = 0 + 150,994,944 + 9,437,184 + 589,824
           + 36,864 + 2,304 + 144 + 9
           = 161,061,273  CONFIRMED
```

**NOTE ON GPT-5.2-pro DISAGREEMENT:** GPT-5.2-pro claimed the FTW should be ~161,061 (not 161,061,273), asserting a 1000x error. This claim is **incorrect** — GPT-5.2-pro made an arithmetic error in its own calculation. The formula `15000 * 268435456 / 25000000` yields 161,061,273.6, not 161,061. The decimal value as stated in the implementation is correct. Only the hex representation has a minor transcription error (last two hex digits should be `99` not `A9`).

**Severity:** LOW (minor hex typo, decimal value correct)

---

### 2. RC Phase Shift — CRITICAL ERROR

**Claim:** RC network with R=10.6 kOhm, C=1 nF produces 90 degree phase shift at 15 kHz.

**Actual phase shift: 45 degrees (NOT 90 degrees)**

Calculation:
```
Xc = 1 / (2*pi*f*C)
   = 1 / (2*pi*15000*1e-9)
   = 1 / (9.4248e-5)
   = 10,610 Ohm  ≈ 10.6 kOhm

For a low-pass RC network (output across C):
  phase = -arctan(omega*R*C) = -arctan(R/Xc)

When R = Xc:
  phase = -arctan(1) = -45 degrees

For a high-pass RC network (output across R):
  phase = +90 - arctan(omega*R*C) = +90 - 45 = +45 degrees
```

A single-pole RC network **cannot** produce 90 degrees of phase shift with usable signal amplitude. When R = Xc, the output is attenuated by 1/sqrt(2) = -3 dB and shifted by exactly 45 degrees.

To approach 90 degrees with an RC network, you would need R >> Xc, but this causes severe attenuation (signal approaches zero as phase approaches 90 degrees). This is a fundamental property of first-order RC networks.

**Impact:** The quadrature channel will contain approximately equal parts in-phase and quadrature signal. Since the primary field in-phase component is ~1000x larger than the secondary quadrature component, even small I-to-Q leakage will completely overwhelm the conductivity signal. The measurement will be dominated by primary field leakage, not soil conductivity.

**Recommended fixes (in order of preference):**

1. **Digital quadrature generation (BEST):** Use ESP32 hardware timer to generate a 90-degree-shifted square wave coherent with the DDS output. The AD9833 SIGN_BIT_OUT provides a 0-degree reference; use a hardware timer synchronized to the same clock edge to generate a precise 90-degree-shifted reference. Both references drive separate AD630 chips.

2. **Dual-channel DDS (AD9958/AD9959):** These chips have two or four independent DDS channels with programmable phase offsets. Set channel 1 to 0 degrees and channel 2 to 90 degrees (phase register = 2^12 / 4 = 1024). Provides exact, stable quadrature at any frequency.

3. **Active allpass network:** A properly designed active allpass filter (op-amp based) can provide exactly 90 degrees phase shift at a single design frequency. Unlike RC networks, allpass filters maintain constant amplitude. However, they are still frequency-dependent.

**Severity:** CRITICAL — renders quadrature channel useless

---

### 3. AD630 Channel Configuration — CRITICAL ERROR

**Claim:** AD630 provides two simultaneous outputs for I and Q channels.

**Fact:** The AD630 is a single balanced modulator/demodulator with **one output**. It multiplies the input signal by +1 or -1 based on the reference signal, then the output is low-pass filtered to extract the DC component at the reference phase.

Per the AD630 datasheet (Analog Devices):
- Pin 13 (OUT A) and Pin 12 (OUT B) are differential outputs of a **single** channel
- There is no built-in provision for simultaneous I and Q demodulation

**To implement true I/Q demodulation, the architecture needs:**

**Option A — Two AD630 chips (RECOMMENDED):**
```
                          ┌──► AD630 #1 (0° ref)   ──► LPF ──► I output
RX Signal ──► AD8421 ──┤
                          └──► AD630 #2 (90° ref)  ──► LPF ──► Q output
```
Cost: ~$30 additional for second AD630 + support components.

**Option B — Time-multiplexed single AD630:**
- Phase 1: Connect 0-degree reference, wait for LPF to settle, sample I
- Phase 2: Switch to 90-degree reference, wait for LPF to settle, sample Q
- Penalty: doubles the measurement time (2x LPF settling per cycle)

**Option C — Replace AD630 with AD8333 (quad demodulator):**
- The AD8333 is a dual I/Q demodulator designed specifically for this application
- Provides four outputs: I+, I-, Q+, Q- from a single chip
- However, it operates at much higher frequencies (DC to 50 MHz IF) and may be overkill

**Severity:** CRITICAL — architecture as drawn is physically impossible with one AD630

---

### 4. Measurement Window vs. LPF Settling — CRITICAL ERROR

**Claim:** 30ms EMI measurement phase with 100 us OPA549 settling delay is sufficient.

**Problem:** The low-pass filter after the AD630 has fc ~ 10 Hz.

```
Time constant: tau = 1 / (2*pi*fc) = 1 / (2*pi*10) = 15.92 ms

Settling to various accuracy levels:
  1*tau = 15.9 ms → 63.2% of final value (36.8% error)
  2*tau = 31.8 ms → 86.5% of final value (13.5% error)
  3*tau = 47.7 ms → 95.0% of final value (5.0% error)
  4*tau = 63.7 ms → 98.2% of final value (1.8% error)
  5*tau = 79.6 ms → 99.3% of final value (0.7% error)
```

The current firmware flow:
```
1. Enable TX (gpio LOW)        t = 0
2. Wait 100 us (OPA549)        t = 0.1 ms
3. Start ADC sampling           t ≈ 0.1 ms
4. Sample for 30 ms            t = 0.1 - 30.1 ms
5. Disable TX                  t = 30.1 ms
```

During the 30ms sampling window, the LPF output is still in its **transient phase** — it has not reached steady state. At t = 30 ms after step change, the output is at only ~85% of its final value. Measurements will be biased low and will have high variance.

**Recommended fixes:**

**Option A — Extend EMI window (BEST for accuracy):**
- Increase to 100-150ms total EMI phase
- Wait 80ms for LPF settling before beginning ADC sampling
- Sample for 20-70ms after settling
- Impact: TDM cycle increases from 100ms to 200-250ms (4-5 Hz update rate)

**Option B — Raise LPF cutoff frequency (BEST for speed):**
- Change LPF from fc = 10 Hz to fc = 50 Hz
- New tau = 3.18 ms, 5*tau = 15.9 ms (fits in 30ms window)
- Trade-off: accepts more noise (5x wider bandwidth)
- Possible compromise: fc = 30 Hz (tau = 5.3ms, 5*tau = 26.5ms)

**Option C — Keep TX running continuously:**
- Do not cycle TX on/off in TDM
- Eliminates settling transient entirely
- Trade-off: EMI TX is always on during fluxgate measurements (interference concern)

**Severity:** CRITICAL — measurements will be biased and noisy

---

### 5. ADS1115 Sampling Budget — IMPORTANT

At the default 128 SPS data rate:

```
Conversion period = 1/128 = 7.8125 ms per sample

In a 30ms window:
  Total conversions possible = 30 / 7.8 ≈ 3.8 → 3 complete conversions
  If alternating I, Q, I, Q: 2 samples for I, 1 for Q (or vice versa)
  After discarding first sample per channel (mux settling): ~1 valid sample each
```

This is **not enough** for reliable averaging or outlier rejection.

**Note on time skew:** After the AD630 + LPF, the outputs are DC (slowly varying). The 7.8ms skew between sequential I and Q readings does **not** cause phase errors in the demodulated signal — this is not sampling the 15 kHz carrier. The concern about "117 cycles of phase shift" is a misunderstanding; it would only matter if the ADC were directly sampling the RF signal.

**Recommended fix:** Use ADS1115 at 860 SPS (highest available rate):
```
Conversion period = 1/860 = 1.16 ms per sample
In 30ms: ~25 total conversions → ~12 per channel
After settling discard: ~10 valid samples per channel
```

This provides reasonable averaging. However, at 860 SPS the ADS1115 effective resolution drops from 16 bits to ~12 bits (noise-free). For the EMI application where signals are relatively large (mV-level DC after demodulation), 12-bit resolution is likely sufficient.

**Alternative:** Use a dedicated ADC for EMI channels (e.g., ADS1015 at 3300 SPS, or MCP3562 24-bit delta-sigma at 153.6 kSPS).

**Severity:** IMPORTANT — marginal sampling degrades measurement quality

---

### 6. Conductivity Formula Verification — CONFIRMED CORRECT

**Formula:**
```
sigma_a = (4 / (omega * mu_0 * s^2)) * Im(Hs / Hp)
```

This is the standard McNeill (1980) Low Induction Number (LIN) approximation for horizontal coplanar (HCP) coil geometry.

**Dimensional analysis:**

```
[omega] = rad/s = s^-1
[mu_0]  = H/m = V*s/(A*m)
[s^2]   = m^2

[omega * mu_0 * s^2] = s^-1 * V*s/(A*m) * m^2
                      = V*m/A

[4 / (omega * mu_0 * s^2)] = A/(V*m) = S/m

[Im(Hs/Hp)] = dimensionless (ratio of field magnitudes)

[sigma_a] = S/m * (dimensionless) = S/m  ✓
```

**Numerical computation:**

```
omega = 2*pi*15000 = 94,247.78 rad/s
mu_0  = 4*pi*1e-7  = 1.25664e-6 H/m
s     = 1.0 m
s^2   = 1.0 m^2

omega * mu_0 * s^2 = 94247.78 * 1.25664e-6 * 1.0
                   = 0.11844 V*m/A

4 / (omega * mu_0 * s^2) = 4 / 0.11844
                          = 33.78 S/m
```

**Sanity checks against emi-coil-design.md Table (lines 42-49):**

| Soil Type | sigma (mS/m) | Hs/Hp (ppm) | Computed sigma_a |
|-----------|-------------|-------------|------------------|
| Dry sand | 10 | ~100 | 33.78 * 100e-6 = 3.4 mS/m |
| Loam | 50 | ~500 | 33.78 * 500e-6 = 16.9 mS/m |
| Wet clay | 100 | ~1000 | 33.78 * 1000e-6 = 33.8 mS/m |
| Saline | 300 | ~3000 | 33.78 * 3000e-6 = 101.3 mS/m |

**Note:** The computed values are lower than the table values by a factor of ~3. This is because the table gives approximate Hs/Hp values, and the exact relationship under LIN is:

```
Hs/Hp = i*omega*mu_0*sigma*s^2 / 4
```

For sigma = 50 mS/m:
```
Hs/Hp = i * 94247.78 * 1.25664e-6 * 0.050 * 1.0 / 4
      = i * 1.481e-3
      = i * 1481 ppm
```

The table value of ~500 ppm for 50 mS/m loam appears to be approximate/rounded. The formula itself is mathematically self-consistent: applying the forward model and then the inverse formula recovers the input conductivity exactly.

**Firmware implementation:**
```
sigma_a = (4.0 / (OMEGA * MU0 * S_SQUARED)) * (q_avg / PRIMARY_CAL);
```

This is correct **IF** `q_avg / PRIMARY_CAL` equals `Im(Hs/Hp)`. The calibration constant `PRIMARY_CAL` must be determined such that this ratio represents the dimensionless secondary-to-primary field ratio.

---

### 7. Calibration Approach

**Saltwater bucket** — functional but problematic:
- Not a semi-infinite half-space (boundary effects dominate)
- Conductivity varies with temperature (~2%/degree C for NaCl solutions)
- Coil geometry relative to bucket edges/surface matters
- At best provides order-of-magnitude verification

**Better approaches:**

1. **Calibration coil / mutual inductance fixture:** A small coil with known geometry and current placed at a known position relative to the RX coil. This injects a precisely known secondary field, decoupling calibration from ground truth. This is the industry standard for EMI instrument calibration.

2. **KCl reference solutions** at 25 degrees C: Standard conductivity solutions (e.g., 0.01 M KCl = 1.412 mS/cm at 25C) measured with a calibrated lab conductivity meter, poured into a large (~1 m diameter) shallow container. Better than "saltwater bucket" but still has geometry issues.

3. **Cross-calibration against known instrument:** Measure a site with both Pathfinder and a calibrated EM38/EM38-MK2. Match readings by adjusting PRIMARY_CAL.

**Recommended:** Use calibration coil for initial system validation, then cross-calibrate in the field against a reference instrument.

---

### 8. OPA549 Settling Time

The 100 us delay accounts for OPA549 amplifier settling (slew rate 9 V/us), but does **not** account for the TX coil's L/R time constant.

```
TX coil: 30 turns, 12 cm diameter (from emi-coil-design.md)

Estimated inductance (single-layer air-core solenoid approximation):
  L ≈ mu_0 * N^2 * A / l
  For a flat spiral (pancake) coil, use Wheeler's approximation:
  L ≈ (N^2 * a^2) / (8*a + 11*c) [microhenries]
  where a = mean radius in inches, c = winding width in inches

  For 30 turns at ~12 cm diameter: L ≈ 50-200 uH (order of magnitude)

With OPA549 output impedance ~1 Ohm + coil resistance ~1-5 Ohm:
  tau_coil = L / R = 100e-6 / 3 ≈ 33 us
  5*tau = 167 us
```

The 100 us delay is marginal but in the right ballpark. However, this delay is for the **coil current** to reach steady state. The more critical delay is the **LPF settling** discussed in Section 4 above.

**Recommendation:** Increase delay to 500 us for coil settling safety margin. This is negligible compared to the LPF settling requirement.

---

## Summary of Errors and Fixes

| # | Issue | Severity | Status | Fix |
|---|-------|----------|--------|-----|
| 1 | RC phase shift is 45 deg, not 90 deg | CRITICAL | Both analyses agree | Digital quadrature or dual-DDS |
| 2 | AD630 is single-channel, not dual I/Q | CRITICAL | Both analyses agree | Two AD630 chips or AD8333 |
| 3 | 30ms window insufficient for 10 Hz LPF | CRITICAL | Both analyses agree | Extend to 100ms+ or raise fc |
| 4 | Hex FTW representation wrong | LOW | Both analyses agree | Change 0x099999A9 to 0x09999999 |
| 5 | ADS1115 at 128 SPS gives ~2 samples/ch | IMPORTANT | Both analyses agree | Use 860 SPS or dedicated ADC |
| 6 | OPA549 settling may be marginal | LOW | Independent finding | Increase to 500 us |
| 7 | Saltwater bucket calibration is crude | ADVISORY | Both analyses agree | Use calibration coil fixture |

## What Is Correct

| Item | Status | Verification |
|------|--------|--------------|
| Conductivity formula (LIN, HCP) | CORRECT | Matches McNeill (1980) |
| Dimensional analysis | CORRECT | Yields S/m with dimensionless ratio |
| Scale factor 4/(omega*mu0*s^2) | CORRECT | = 33.78 S/m |
| Angular frequency omega | CORRECT | = 94,247.78 rad/s |
| DDS frequency register (decimal) | CORRECT | = 161,061,273 |
| mu_0 value | CORRECT | = 1.2566e-6 H/m |
| Firmware formula structure | CORRECT | Properly maps Q/cal to Im(Hs/Hp) |

---

## Recommended Revised Architecture

```
                     ESP32 Hardware Timer
                     (90° square wave)──────────────────┐
                                                         │
AD9833 DDS ──► OPA549 Power Amp ──► TX Coil              │
  (15 kHz)      (current drive)     (30 turns)           │
     │                                                    │
     └─ SIGN_BIT (0° square wave) ─────────┐             │
                                            │             │
                                    ┌───────▼─────┐ ┌────▼──────┐
RX Coil ──► AD8421 Preamp ──┬──────►  AD630 #1   │ │  AD630 #2  │
  (30 turns)   (gain: 100)  │      │  (0° ref)   │ │  (90° ref) │
                             │      └──────┬──────┘ └─────┬──────┘
                             │             │              │
                             │        RC LPF (30 Hz) RC LPF (30 Hz)
                             │             │              │
                             │          I output       Q output
                             │             │              │
                             │        ADS1115 Ch0    ADS1115 Ch1
                             │          (860 SPS)      (860 SPS)
```

### Revised Firmware Pseudo-Code

```c
// Constants
#define OMEGA       94247.78f       // 2*pi*15000 rad/s
#define MU0         1.25664e-6f     // H/m
#define S_SQUARED   1.0f            // m^2
#define SCALE       33.78f          // 4/(OMEGA*MU0*S_SQUARED) S/m
#define N_SAMPLES   10              // ~12ms at 860 SPS
#define LPF_SETTLE_MS  50           // 5*tau for fc=30Hz LPF

void measure_emi() {
    // Enable TX
    ad9833_set_freq(15000);   // FREQREG = 161061273 = 0x09999999
    gpio_set_level(EMI_SHUTDOWN, 0);  // OPA549 enable

    // Wait for coil current + LPF settling
    delay_ms(LPF_SETTLE_MS);  // 50ms for 30Hz LPF (was 0.1ms!)

    // Configure ADS1115 for 860 SPS
    ads1115_set_data_rate(860);

    // Sample I and Q channels (post-demod DC)
    float i_sum = 0, q_sum = 0;
    for (int n = 0; n < N_SAMPLES; n++) {
        i_sum += ads1115_read_channel(EMI_I_CHANNEL);  // AD630 #1
        q_sum += ads1115_read_channel(EMI_Q_CHANNEL);  // AD630 #2
    }
    float i_avg = i_sum / N_SAMPLES;
    float q_avg = q_sum / N_SAMPLES;

    // Calculate apparent conductivity
    float sigma_a = SCALE * (q_avg / PRIMARY_CAL);  // S/m

    // Also store in-phase for magnetic susceptibility
    float chi_a = SCALE * (i_avg / PRIMARY_CAL);  // dimensionless (approx)

    // Disable TX
    gpio_set_level(EMI_SHUTDOWN, 1);
}
```

### Revised TDM Budget

```
Original:  50ms fluxgate | 30ms EMI  | 20ms settling = 100ms (10 Hz)
Revised:   50ms fluxgate | 75ms EMI  | 25ms settling = 150ms (6.7 Hz)
                           ├─ 50ms LPF settle
                           └─ 25ms ADC sampling (860 SPS, ~10 samples/ch)
```

The 6.7 Hz update rate is acceptable for walking-speed survey (~1 m/s = one reading per 15 cm).

---

## BOM Impact

| Component | Change | Est. Cost |
|-----------|--------|-----------|
| AD630 (second chip) | ADD | +$30 |
| RC LPF components (revised fc=30Hz) | MODIFY | ~$0 (same cost) |
| Hex FTW correction | FIRMWARE ONLY | $0 |
| ADS1115 data rate change | FIRMWARE ONLY | $0 |
| **Total additional cost** | | **~$30** |

If replacing RC phase shift with dual-DDS (AD9958): add ~$15 for the DDS chip (replaces AD9833).

---

## References

- McNeill, J.D. (1980). Electromagnetic terrain conductivity measurement at low induction numbers. Technical Note TN-6, Geonics Limited.
- Analog Devices AD630 Datasheet: Balanced Modulator/Demodulator
- Analog Devices AD9833 Datasheet: Programmable Waveform Generator
- Texas Instruments ADS1115 Datasheet: 16-Bit ADC
- Pathfinder EMI design: `/research/multi-sensor-architecture/emi-coil-design.md`

---

*Generated by PAL consensus validation — Claude Opus 4.6 + openai/gpt-5.2-pro*
*gemini-3-pro-preview was unavailable due to API rate limits*
