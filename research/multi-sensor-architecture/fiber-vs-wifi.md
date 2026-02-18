# R1 Consensus Validation: Fiber Optics vs WiFi for Field Data Transmission

## Consensus Metadata

| Field | Value |
|-------|-------|
| **Research Task** | R1: Fiber Optics vs WiFi for Field Data Transmission |
| **Date** | 2026-02-18 |
| **Models Consulted** | Claude Opus 4.6 (independent analysis), openai/gpt-5.2 (neutral), openai/gpt-5.2-pro (neutral, substitute) |
| **Models Unavailable** | gemini-3-pro-preview (429 RESOURCE_EXHAUSTED -- Gemini API quota exceeded on all retry attempts) |
| **Consensus Confidence** | 7.5 / 10 |
| **Overall Verdict** | **WiFi with TDM is sufficient for standard operation; fiber recommended as upgrade path** |

## Source Documents

- `../../GeoSim/docs/research/fiber-vs-wifi-analysis.md` -- Original analysis document
- `interference-matrix.md` -- Cross-sensor electromagnetic interference matrix
- `tdm-firmware-design.md` -- TDM firmware timing and architecture

---

## Question Under Evaluation

Should we use ESP32 WiFi (2.4 GHz, built-in) or fiber optic data links for transmitting measurement data from the handheld Pathfinder instrument to a Jetson edge computer?

### Context Summary

- FG-3+ self-oscillating fluxgate sensors (50-120 kHz output) are the primary measurement
- ESP32 WiFi TX is 2.4 GHz at up to 20 dBm (100 mW)
- TDM firmware disables WiFi TX during the 50 ms fluxgate measurement phase
- LM2917 F-to-V converter has ~10 Hz bandwidth (63 dB rejection at 15 kHz, >100 dB at 2.4 GHz)
- Metal enclosure provides ~20 dB shielding at 2.4 GHz
- Estimated WiFi interference after TDM + shielding: <0.01 nT **(Modeled)** (system noise floor: 0.5 nT)
- Fiber options: USB-to-fiber ($80-150), SFP ($50-70), POF ($40-60)
- Operational concern: trailing fiber cable limits terrain flexibility

---

## Consensus Findings

### Finding 1: WiFi with TDM Is Sufficient for Standard Operation (UNANIMOUS)

All analyses agree that WiFi with TDM gating is adequate for standard Pathfinder surveying. The reasoning:

- **TDM eliminates the primary interference mechanism.** WiFi TX is completely OFF during the 50 ms fluxgate measurement phase. No 2.4 GHz emission occurs during measurement. This alone provides effectively infinite attenuation of the dominant interference source.
- **Defense-in-depth provides multiple safety margins.** Even if residual RF is present (WiFi RX local oscillator, CPU clock harmonics), the mitigation chain provides cumulative rejection:
  - Metal enclosure: ~20 dB at 2.4 GHz
  - LM2917 inherent rejection: >100 dB at 2.4 GHz
  - 3-stage power supply PSRR: >120 dB at >1 kHz
  - RC input filtering on analog amplifiers (100 pF, fc = 160 MHz)
- **Operational advantages of WiFi are significant.** Zero setup time, no cable management, full terrain flexibility, no snag risk in brush/forest. These are not marginal benefits -- they directly affect survey productivity and safety.
- **Fiber does not eliminate the need for TDM.** The TDM architecture is required regardless of data link choice because of other CRITICAL interference paths (EMI TX coil at 2,100 nT, LiDAR motor at 100-300 nT). Fiber only addresses one row of the interference matrix.

### Finding 2: The <0.01 nT Estimate Is Plausible but Optimistic (UNANIMOUS)

All analyses agree the modeled <0.01 nT figure should be treated as an aspirational target, not a validated specification.

**Why it may hold:**
- TDM completely eliminates WiFi TX during measurement (the dominant mechanism)
- The LM2917's >100 dB rejection at 2.4 GHz makes direct RF coupling negligible
- The 3-stage power supply chain provides >120 dB PSRR for conducted interference

**Why it may not hold:**
- **RF rectification/downconversion is the real risk** (identified by all analyses): 2.4 GHz RF coupling into high-impedance analog nodes can be rectified by semiconductor junctions (AD8421, LM2917 input diodes), producing baseband (DC to 10 Hz) offset terms that land directly in the science band. The LM2917's >100 dB rejection at 2.4 GHz does not protect against this mechanism because the rectification occurs before the frequency-dependent filtering.
- **TDM "TX off" may not mean "RF quiet"**: ESP32's WiFi stack may keep PLLs active, emit management frames/beacons, or transmit probe responses even with TX power set to 0. The `esp_wifi_set_max_tx_power(0)` API call may not fully suppress all RF emission.
- **20 dB enclosure shielding is fragile**: Actual shielding depends on seam integrity, cable penetrations, and antenna proximity. Cables exiting the enclosure act as antennas, and the ESP32's PCB antenna radiates through any aperture.

**Revised estimate**: A more realistic prior for a well-designed but imperfect system is **0.01 to 0.1 nT**. Even at 0.1 nT, this remains 5x below the 0.5 nT system noise floor and is acceptable per the decision framework.

### Finding 3: Additional Unmodeled Interference Mechanisms (CONSENSUS)

The following mechanisms were identified across analyses but are not fully addressed in the current documentation:

| Mechanism | Identified By | Risk Level | Notes |
|-----------|--------------|------------|-------|
| RF rectification at semiconductor junctions producing DC/baseband offsets | All three | HIGH | Primary concern -- bypasses frequency-domain rejection |
| Phase-boundary transients (WiFi enable/disable bleeding into next measurement window) | All three | MODERATE-HIGH | LM2917 RC filter tau=15ms, only 26% attenuation after 20ms settling |
| ESP32 clock harmonics (40 MHz crystal, 240 MHz CPU) as continuous emitters | Claude Opus 4.6 | LOW-MODERATE | 60th harmonic of 40 MHz = 2.4 GHz; always present regardless of WiFi state |
| WiFi management frames/beacons emitting with TX power = 0 | Claude Opus 4.6, GPT-5.2 | MODERATE | Firmware must verify true RF silence during Phase 1 |
| Jetson-side WiFi/BT emissions | GPT-5.2 | MODERATE | Jetson at 0.5-2m can become the dominant RF source if its radios are unmanaged |
| Cable/antenna near-field magnetic coupling and common-mode currents | GPT-5.2 | LOW-MODERATE | Currents on antenna feed create local H-fields coupling into sensor leads |
| Enclosure shielding fragility at seams and cable penetrations | GPT-5.2 | MODERATE | Real shielding path is often cables, not enclosure walls |
| Conducted coupling via shared digital/analog ground | GPT-5.2 | MODERATE | ESP32 digital activity creates broadband supply noise even with TX off |
| Environmental EMI and motion-induced coupling | GPT-5.2-Pro | HIGH (field) | Often the real noise floor limit in practice, dwarfing WiFi self-interference |

### Finding 4: Fiber/Wired Fallback Should Be Maintained (UNANIMOUS)

All analyses recommend maintaining a fiber option as an upgrade path. The recommended configuration:

**Primary link**: WiFi with TDM (standard operation, all terrain types)

**Upgrade option**: POF (Plastic Optical Fiber) kit, $40-60

The fiber option serves three distinct purposes:
1. **Diagnostic baseline**: "If noise disappears on fiber, it is RF/coupling" -- this is the most valuable use case for isolating interference sources during development and troubleshooting.
2. **High-precision applications**: Users working near the detection limit (graves, subtle fired clay anomalies) on flat/cleared terrain where cable management is practical.
3. **RF-hostile environments**: Military/restricted areas where wireless transmission is prohibited.

**Note**: Fiber does not eliminate the need for TDM. The EMI TX coil and LiDAR motor interference paths are orders of magnitude larger than WiFi and require TDM regardless. Fiber only addresses the WiFi emitter row in the interference matrix.

### Finding 5: Enhanced Bench Test Protocol (CONSENSUS)

All analyses agreed on the need for rigorous bench validation. The following protocol synthesizes recommendations from all sources.

#### Test Conditions

| Condition | WiFi State | Purpose |
|-----------|-----------|---------|
| **A** | WiFi fully disabled (`esp_wifi_deinit()`) | Baseline noise floor |
| **B** | WiFi initialized, RX only (TX power = 0) | Tests RX local oscillator leakage |
| **C** | WiFi with TDM gating (normal operation) | Production configuration |
| **D** | WiFi TX active during measurement (no gating) | Worst case -- quantifies TDM benefit |
| **E** | Fiber/wired link, WiFi hardware disabled | Reference baseline independent of radio stack |

#### Measurement Protocol

1. **Environment**: Magnetically quiet location (wooden table, no nearby ferrous objects, no AC power lines within 5m)
2. **Duration**: Minimum 10 minutes per condition (600 measurement cycles at 10 Hz)
3. **Worst-case traffic**: During conditions C and D, force continuous UDP upload (iperf-style) at maximum data rate with maximum TX power on a crowded 2.4 GHz channel

#### Metrics

- **RMS noise** in science band (0.1-10 Hz) for each condition
- **Welch PSD** (power spectral density) to identify discrete spectral lines from WiFi-related frequencies
- **Allan deviation** to capture low-frequency wander and rectification effects
- **Supply rail monitoring** with oscilloscope + FFT (correlate supply transients with WiFi events)
- **Near-field probe** with spectrum analyzer around enclosure seams and cable exits to locate coupling points

#### Parametric Sweeps

- **Settling time**: Vary Phase 3 duration (10, 20, 30, 40 ms) and measure noise in subsequent Phase 1
- **WiFi edge timing**: Deliberately shift WiFi enable/disable edges relative to fluxgate sampling to detect edge-coupled artifacts
- **Physical configuration**: Test with enclosure open vs closed, vary antenna placement and harness cable routing
- **Jetson distance**: Vary distance (0.5, 1, 2, 5 m) and test with Jetson radios on vs off

#### Acceptance Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| **Primary**: (C - A) RMS noise | < 0.1 nT | Aligns with decision framework: WiFi interference below "marginal" threshold |
| **Secondary**: No discrete WiFi spectral lines | Visual inspection of PSD | Tones could alias into mapping products even if RMS is low |
| **Tertiary**: (C - A) Allan deviation | No divergence at tau > 1s | Rules out low-frequency rectification drift |
| **Informational**: (D - A) RMS noise | Record value | Quantifies TDM benefit; informs risk if TDM timing slips |
| **Informational**: (E - A) RMS noise | < 0.01 nT | Confirms fiber provides the expected zero-interference baseline |

---

## Points of Agreement (All Models)

1. WiFi with TDM is the correct default for a field-portable instrument
2. The <0.01 nT estimate needs bench validation before the decision is finalized
3. RF rectification/downconversion at semiconductor junctions is the primary unmodeled risk
4. POF fiber should be offered as an optional upgrade kit
5. The bench test must include worst-case RF traffic conditions, not just idle WiFi
6. Phase-boundary transient settling requires explicit characterization
7. Environmental EMI and motion-induced coupling likely dominate the real-world noise budget

## Points of Disagreement

**None substantive.** All analyses converged on the same recommendation. Minor differences in emphasis:
- GPT-5.2 placed more emphasis on Jetson-side RF as a potential dominant source
- Claude Opus 4.6 emphasized ESP32 clock harmonics as continuous emitters
- GPT-5.2-Pro emphasized that environmental EMI typically dominates component-level self-noise in field instruments

These are complementary observations, not contradictions.

---

## Final Consolidated Recommendation

### Decision: WiFi with TDM as Primary, POF Fiber as Upgrade

**Ship with WiFi as the default data link**, contingent on passing the bench test protocol above. The TDM approach combined with metal enclosure shielding and the LM2917's inherent narrowband filtering provides sufficient interference suppression for standard geophysical surveying.

**Offer a POF fiber kit ($40-60) as an optional accessory** for high-precision work and as a diagnostic tool during development. Design the ESP32 firmware to support a `WIFI_DISABLED` mode that disables all radio hardware when fiber is connected.

### Actionable Next Steps

1. **Immediate**: Add `esp_wifi_set_max_tx_power(0)` verification test -- confirm with spectrum analyzer that no RF emission occurs during Phase 1 when TX power is set to 0
2. **Immediate**: Add Jetson WiFi/BT management to the system architecture -- Jetson's radios must be gated or disabled during Pathfinder measurements, or the Jetson must be positioned far enough away (>2m) to be negligible
3. **Before prototype**: Execute the 5-condition bench test protocol (A through E) and record results
4. **Before prototype**: Characterize phase-boundary settling by sweeping Phase 3 duration from 10-40 ms
5. **Design**: Add 100 pF bypass capacitors on AD8421 and LM2917 analog inputs if not already present (fc = 160 MHz with 10 kohm source impedance)
6. **Design**: Ensure ESP32 antenna is positioned on the far side of the enclosure from the analog electronics board
7. **Future**: Consider shielded USB/Ethernet (copper) as an intermediate option for handheld-to-backpack distances (<2m) -- simpler than fiber media converters while still removing RF TX

### Critical Risks to Monitor

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RF rectification produces >0.1 nT baseband offset | Low-Moderate | HIGH | Bench test condition D will reveal this; add input RC filters if needed |
| 20 ms settling insufficient for WiFi transients | Moderate | MODERATE | Settling time sweep will quantify; extend Phase 3 if needed (reduces measurement rate) |
| Jetson WiFi becomes dominant RF source | Moderate | HIGH | Manage Jetson radios or increase separation distance |
| ESP32 firmware update changes WiFi behavior | Low | HIGH | Regression test: re-run bench conditions A-C after any ESP-IDF update |

---

## Appendix: Model Response Summaries

### Claude Opus 4.6 (Independent Analysis)

**Verdict**: WiFi with TDM is sufficient. <0.01 nT is reasonable but optimistic -- weakest link is the settling time assumption (LM2917 tau=15ms, only 26% attenuation after 20ms). Identified ESP32 clock harmonics and WiFi management frames as additional concerns. Recommended 6-condition bench test with settling time sweep.

### openai/gpt-5.2 (Neutral)

**Verdict**: WiFi with TDM is "probably sufficient" but <0.01 nT is optimistic without bench evidence. Confidence 7/10. Key contribution: identified RF rectification/downconversion as the real risk (not direct 2.4 GHz passthrough), flagged Jetson-side RF as an uncontrolled source, and proposed 4-condition (A/B/C/D) bench test with PSD, Allan deviation, and supply rail correlation. Recommended WiFi default + fiber/wired "RF-quiet mode" for diagnostics and troubleshooting.

### openai/gpt-5.2-pro (Neutral, Substitute for Gemini)

**Verdict**: Provided analysis primarily focused on EMI coil design and system-level interference. Relevant insight: environmental EMI and motion-induced coupling are often the real noise floor limits in field instruments, not component-level self-noise. This indirectly supports the WiFi-is-sufficient conclusion, since WiFi interference (even at 0.1 nT) would be dwarfed by real-world environmental noise sources.

### gemini-3-pro-preview (Unavailable)

**Status**: Model was unavailable due to Gemini API quota exhaustion (429 RESOURCE_EXHAUSTED). Free-tier daily request limit for gemini-3-pro was exceeded. All retry attempts (>14) failed with the same error. This consensus proceeds with two of the three requested models.

---

*Generated by PAL Consensus Tool -- 2026-02-18*
*Models: Claude Opus 4.6, openai/gpt-5.2, openai/gpt-5.2-pro (substitute)*
*Continuation ID: b9e7aba9-885e-4002-b6db-628003fd3270*
