# Pathfinder Noise Floor Analysis

## Overview

This document provides a quantitative end-to-end noise budget for the Pathfinder gradiometer signal chain. Pathfinder's detection depth claims in [design-concept.md](design-concept.md) are listed as **(Target)** without supporting physics. This analysis establishes the expected noise floor and derives detection capabilities from first principles.

All values in this document are **(Modeled)** estimates based on published component specifications and comparable sensor data. Field validation is required before these can be upgraded to **(Measured)**.

---

## Signal Chain Noise Budget

The Pathfinder signal chain consists of four stages, each contributing noise that propagates to the final gradient measurement. The total noise budget is dominated by the weakest link in this chain.

### 1. FG-3+ Fluxgate Sensor

The FG-3+ fluxgate sensor outputs a frequency proportional to the ambient magnetic field. The sensor's intrinsic noise sets the theoretical floor for the entire system.

- **Output**: Frequency proportional to magnetic field
- **Typical fluxgate noise density**: ~10-20 pT/sqrt(Hz) at 1 Hz (based on comparable sensors) **(Modeled)**
- **At 10 Hz bandwidth**: noise = 10-20 pT x sqrt(10) = 30-60 pT RMS **(Modeled)**
- **Note**: FG-3+ specific noise specs are not published by the manufacturer; this analysis uses comparable fluxgate estimates from similar low-cost fluxgate sensors. The actual noise may differ.

At the pT level, sensor noise is well below the downstream electronics noise. The fluxgate itself is not the limiting factor in this signal chain.

### 2. LM2917 Frequency-to-Voltage Converter

The LM2917 converts the fluxgate's frequency output to a proportional voltage for ADC sampling. This component introduces the largest noise and drift contribution in the analog chain.

- **Non-linearity**: typically +/-0.3% of full scale **(Modeled)**
- **Temperature coefficient**: ~0.3%/deg C **(Modeled)**
- **Temperature drift over a survey day**: At Delta-T = 20 deg C (morning to afternoon in field conditions), the drift is approximately 0.3% x 20 = 6% if uncorrected **(Modeled)**
- **Output noise**: Depends on timing capacitor and load resistor values; typically 10-50 mV RMS **(Modeled)**

**This is the dominant noise source in the analog chain.** The LM2917's temperature sensitivity is particularly problematic for field work where the electronics enclosure may heat significantly in direct sunlight. The 6% drift figure represents a worst case without any temperature compensation or recalibration during the survey.

### 3. ADS1115 ADC (16-bit)

The ADS1115 provides 16-bit analog-to-digital conversion over I2C. At the gain and sample rate settings used by Pathfinder, the effective resolution is less than the nominal 16 bits.

- **At GAIN_ONE (+/-4.096V range)**: LSB = 125 uV
- **Noise-free resolution**: ~14.5 bits at 128 SPS, yielding effective noise of approximately 180 uV RMS **(Modeled)**
- **In magnetic units**: Depends on the F-to-V conversion gain, but approximately 0.1-0.5 nT equivalent **(Modeled)**

The ADC contributes a modest but non-negligible amount of noise. At 14.5 effective bits, the quantization noise is roughly comparable to the LM2917 output noise when referred back to the magnetic field.

### 4. Gradiometer Subtraction

The gradiometer measurement subtracts the top sensor reading from the bottom sensor reading. This cancels common-mode signals (diurnal variation, regional geology) but causes the uncorrelated noise from each sensor channel to add in quadrature.

- **Common-mode rejection**: Diurnal magnetic variation and regional geological gradients cancel effectively
- **Residual after subtraction**: Sensor noise adds in quadrature (sqrt of sum of squares)
- **Gradient noise floor**: sqrt(2) x single sensor noise = approximately 0.15-0.7 nT **(Modeled)**

The factor of sqrt(2) = 1.41 is the unavoidable penalty for differencing two noisy measurements. For a single-channel noise floor of ~0.1-0.5 nT, the gradient noise floor becomes ~0.15-0.7 nT.

### Noise Budget Summary

| Stage | Noise Contribution | Dominant? |
|-------|-------------------|-----------|
| FG-3+ fluxgate sensor | 30-60 pT RMS (at 10 Hz BW) | No — well below electronics noise |
| LM2917 F-to-V converter | 10-50 mV RMS (output referred) | **Yes — dominant source** |
| ADS1115 ADC | ~180 uV RMS (~0.1-0.5 nT equiv.) | Moderate — comparable to F-to-V |
| Gradiometer subtraction | sqrt(2) penalty on total noise | Adds ~41% to single-channel noise |
| **Total gradient noise floor** | **~0.15-0.7 nT** | |

All values **(Modeled)**.

---

## Detection Capability

The following table estimates the magnetic gradient anomaly produced by representative targets at various depths, assuming a vertical gradiometer with 0.35 m baseline. Values assume dipole field decay (1/r^3) and are order-of-magnitude estimates.

| Target | Depth 0.5 m | Depth 1.0 m | Depth 1.5 m | Depth 2.0 m |
|--------|-------------|-------------|-------------|-------------|
| 500 lb bomb (iron, ~0.5 m dia) | >1000 nT | ~200 nT | ~40 nT | ~10 nT |
| Small iron object (1 kg) | ~100 nT | ~15 nT | ~3 nT | <1 nT |
| Fired clay (kiln, 1 m dia) | ~20 nT | ~5 nT | ~1 nT | <0.5 nT |
| Grave (disturbed soil) | ~5 nT | ~1 nT | <0.5 nT | <0.2 nT |

*Gradient values for vertical gradiometer with 0.35 m baseline. Assumes dipole field decay (1/r^3). All values are order-of-magnitude estimates* **(Modeled)**.

### Detection Depth Estimates

With a noise floor of approximately 0.5 nT and a detection threshold of SNR > 3 (signal at least 3x the noise floor, i.e., >1.5 nT):

| Target Type | Estimated Detection Depth | Confidence |
|-------------|--------------------------|------------|
| 500 lb bomb (iron, ~0.5 m dia) | ~1.5-2.0 m | Good — signal well above noise at these depths **(Modeled)** |
| Small iron object (1 kg) | ~0.5-1.0 m | Moderate — approaches noise floor at 1 m **(Modeled)** |
| Fired clay (kiln, 1 m dia) | ~0.5 m | Marginal — near noise floor even at shallow depth **(Modeled)** |
| Grave (disturbed soil) | At detection limit even at shallow depth | Poor — signal comparable to noise floor **(Modeled)** |

These estimates are consistent with commercial fluxgate gradiometer performance for similar sensor configurations. The detection depths listed in [design-concept.md](design-concept.md) as **(Target)** are broadly supported by this analysis, though graves and fired clay are at the margins of detectability.

---

## Recommendations

### 1. Address LM2917 Temperature Drift (Highest Priority)

The LM2917 frequency-to-voltage converter's temperature coefficient (~0.3%/deg C) is the largest error source in the signal chain. Over a 20 deg C temperature swing during a field day, this produces approximately 6% drift if uncorrected. Mitigation options:

- **Thermal insulation**: Wrap the electronics enclosure in reflective material to reduce solar heating
- **Temperature logging**: Add a thermistor to the electronics board and record temperature alongside magnetic data; apply post-processing correction
- **Temperature compensation circuit**: Add a temperature-dependent resistor in the LM2917 timing network to partially cancel the drift
- **Frequent recalibration**: Return to a known reference point every 30-60 minutes and apply drift correction in post-processing

### 2. Noise Reduction Through Averaging

Multiple stacked readings (averaging) reduce random noise by a factor of sqrt(N), where N is the number of samples averaged:

- At 10 Hz sample rate, a 10-sample running average gives sqrt(10) = 3.2x noise reduction at an effective 1 Hz output rate **(Modeled)**
- At walking speed (1 m/s), 1 Hz output gives one reading per meter, which is adequate spatial sampling for most targets
- A 20-sample average (0.5 Hz output, 2 readings per meter at walking speed) gives sqrt(20) = 4.5x noise reduction **(Modeled)**

This is the simplest and most effective noise reduction strategy. The firmware should implement configurable moving-average or boxcar filtering.

### 3. Consider ADC Upgrade for Future Revisions

The ADS1115 at 16 bits provides adequate but not exceptional performance. For a future revision:

- **ADS1256** (24-bit, SPI): Significantly lower noise floor, but more complex firmware integration
- **ADS1220** (24-bit, I2C): Drop-in upgrade path with much better noise performance
- Either option would shift the noise bottleneck entirely to the LM2917, making temperature compensation even more important

### 4. Field Validation Protocol

This entire analysis is **(Modeled)**. Before relying on these detection depth estimates, the following measurements should be performed:

1. **Bench noise measurement**: Record data with sensors stationary in a magnetically quiet environment for 10 minutes; compute RMS noise of the gradient channels
2. **Temperature drift test**: Record data over 4-6 hours with known temperature variation; quantify drift rate
3. **Known-target test**: Bury ferrous objects of known mass at known depths; compare measured gradient with predicted values from the table above
4. **Detection threshold test**: Determine the minimum detectable signal empirically by varying target depth until signal disappears into noise

---

---

## Multi-Sensor Interference Noise Budget

This section extends the noise analysis to account for the multi-sensor Pathfinder architecture. Cross-sensor electromagnetic interference adds noise sources not present in the original single-modality design.

### FG-3+ Self-Oscillating Characteristics

The FG-3+ is a **self-oscillating** fluxgate, not a traditional driven-excitation design. Key implications:

- **Output is a frequency** (50-120 kHz), not a voltage. The LM2917 is the primary demodulator.
- **The LM2917 acts as a natural EMI filter**: with BW ≈ 10 Hz (R1=100kΩ, C2=0.15μF), interference at 15 kHz (EMI TX) is rejected by 63 dB — a factor of ~1400×.
- **Self-oscillation is supply-sensitive**: power supply noise directly modulates the oscillation frequency, making the 3-stage power supply critical.
- **Thermal drift**: ~46 nT/°F — largely common-mode (cancels in gradiometer subtraction), but sensor-to-sensor mismatch contributes residual drift.

See `research/multi-sensor-architecture/fg3-plus-specifications.md` for full characterization.

### LM2917 as Natural EMI Filter

The LM2917 charge pump architecture provides inherent low-pass filtering that benefits the multi-sensor design:

| Interference Source | Frequency | LM2917 Rejection | Residual |
|-------------------|-----------|-------------------|----------|
| EMI TX coil | 15 kHz | 63 dB | <0.001 nT **(Modeled)** |
| FG-3+ oscillation ripple | 80 kHz | 78 dB | 4.2 mV p-p (above ADC Nyquist) |
| Buck converter | 150 kHz | 83 dB | Negligible |
| WiFi | 2.4 GHz | >100 dB | Negligible |

The LM2917's 10 Hz bandwidth means that even without TDM, the EMI TX signal at 15 kHz would be attenuated by 63 dB. With TDM (EMI TX off during measurement), the rejection is effectively infinite.

See `research/multi-sensor-architecture/lm2917-analysis.md` for bandwidth calculations.

### Cross-Sensor Interference Budget

The complete 15-pair interference matrix is documented in `research/multi-sensor-architecture/interference-matrix.md`. Summary of contributions to fluxgate noise floor:

| Source | Mechanism | Estimated Contribution | Mitigation |
|--------|-----------|----------------------|------------|
| EMI TX (15 kHz) | Direct magnetic field | <0.01 nT **(Target)** | TDM + 1.25m separation |
| LiDAR motor DC field | Permanent magnet | <1 nT **(Target)** | 1.25-2.0m separation + gradiometer subtraction |
| WiFi TX (2.4 GHz) | RF rectification | 0.01-0.1 nT **(Modeled)** | TDM + shielding + LM2917 filter |
| Buck converter | Supply ripple | <0.001 nT **(Target)** | 95-110 dB PSRR (ferrite + LC + AP2112K) |
| I2C/SPI digital | Capacitive coupling | <0.01 nT **(Target)** | PCB layout, guard traces |
| **Combined interference** | | **<1 nT** **(Target)** | |

### Power Supply Noise Contribution

The 3-stage power supply chain provides strong noise rejection. **Note**: Consensus validation (R5) corrected the original LM78L05 per-sensor regulators to AP2112K-5.0 (the LM78L05 has 1.7V dropout, incompatible with the 5.5V intermediate rail). The corrected chain:

```
Buck converter → Ferrite+LC filter → AP2112K-5.0 (per sensor)
  30-50 mV ripple   60 dB rejection    70 dB PSRR @ 1kHz
                    = 30-50 μV         = ~10 nV
```

Consensus also corrected the overall PSRR claim from 120 dB to **95-110 dB** (realistic range). The LC filter requires damping (resistor in series with capacitor) to prevent ringing during transients (undamped Q > 100).

At the sensor, power supply noise contributes ~10 nV of supply ripple — still well below the system noise floor.

See `research/multi-sensor-architecture/power-supply-design.md` for consensus-validated design.

### Updated Noise Budget (Multi-Sensor)

| Stage | Noise Contribution | Change from Original |
|-------|-------------------|---------------------|
| FG-3+ fluxgate sensor | 30-60 pT RMS | Unchanged |
| LM2917 F-to-V converter | 10-50 mV RMS | Unchanged (dominant) |
| ADS1115 ADC | ~180 μV RMS (~0.1-0.5 nT) | Unchanged |
| Cross-sensor interference | <1 nT total **(Target)** | **New** — from multi-sensor |
| Power supply noise | <0.00001 nT | **New** — negligible |
| Gradiometer subtraction | √2 penalty | Unchanged |
| **Total gradient noise floor** | **~0.5-1.2 nT** **(Target)** | Slightly increased by interference |

The multi-sensor interference adds up to ~1 nT in the worst case (LiDAR motor), but this is largely DC/common-mode and cancels in the gradiometer. The effective noise floor increase is estimated at <0.3 nT above the single-modality baseline.

---

## References

- ADS1115 datasheet: [Texas Instruments SBAS444C](https://www.ti.com/lit/ds/symlink/ads1115.pdf)
- LM2917 datasheet: [Texas Instruments SNAS555C](https://www.ti.com/lit/ds/symlink/lm2917-n.pdf)
- Fluxgate noise characteristics: Ripka, P. (2001). *Magnetic Sensors and Magnetometers*. Artech House.
- Archaeological gradiometer detection depths: David, A. et al. (2008). *Geophysical Survey in Archaeological Field Evaluation*. English Heritage.
- FG-3+ specifications: See `research/multi-sensor-architecture/fg3-plus-specifications.md`
- Interference matrix: See `research/multi-sensor-architecture/interference-matrix.md`
- Power supply design: See `research/multi-sensor-architecture/power-supply-architecture.md`
