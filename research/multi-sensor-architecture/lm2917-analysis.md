# LM2917 Frequency-to-Voltage Converter Analysis

## Overview

The LM2917 is the primary demodulation stage for the FG-3+ self-oscillating fluxgate. It converts the sensor's frequency output (50-120 kHz) to a proportional voltage for ADC sampling. Understanding its bandwidth and filtering characteristics is critical for the multi-sensor Pathfinder design.

## Operating Principle: Charge Pump Architecture

The LM2917 uses a charge pump architecture:

1. An internal comparator detects zero-crossings of the input frequency.
2. Each zero-crossing transfers a fixed charge packet (set by timing capacitor C1) to an output RC filter.
3. The average output voltage is proportional to the input frequency: `V_out = V_cc × f_in × C1 × R1`
4. The RC filter (R1, C2) smooths the charge pump ripple to produce a DC-proportional output.

## Bandwidth Analysis

The LM2917's output bandwidth is determined by the RC filter components:

```
BW = 1 / (2π × R1 × C2)
```

### Typical Configuration

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| R1 | 100 kΩ | Output load resistor |
| C1 | 0.01 μF | Timing capacitor |
| C2 | 1 μF | Filter capacitor |
| BW | ~1.6 Hz | 1/(2π × 100k × 1μF) |

For Pathfinder's 10 Hz sample rate, we want BW ≈ 10 Hz:

| R1 | C2 | BW | Ripple at 80 kHz |
|----|----|----|------------------|
| 100 kΩ | 0.1 μF | 16 Hz | Higher ripple |
| 100 kΩ | 0.15 μF | 10.6 Hz | Good balance |
| 100 kΩ | 1.0 μF | 1.6 Hz | Very low ripple, too slow |

**Recommended**: R1 = 100 kΩ, C2 = 0.15 μF for ~10 Hz bandwidth.

## EMI Rejection Properties

The LM2917's charge pump architecture provides inherent low-pass filtering. At the recommended 10 Hz bandwidth:

### Rejection at Key Interference Frequencies

| Frequency | Source | Rejection | Calculation |
|-----------|--------|-----------|-------------|
| 15 kHz | EMI TX coil | **63 dB** | 20 × log10(15000/10) |
| 80 kHz | FG-3+ oscillation ripple | **78 dB** | 20 × log10(80000/10) |
| 150 kHz | LM2596 buck converter | **83 dB** | 20 × log10(150000/10) |
| 2.4 GHz | WiFi TX | **>100 dB** | Well beyond analog bandwidth |

The 63 dB rejection at 15 kHz means the EMI transmitter's signal is attenuated by a factor of ~1400 before reaching the ADC. Combined with physical separation and TDM (EMI TX off during fluxgate measurement), the EMI coil should produce negligible interference in the fluxgate channel.

## Ripple Analysis

The charge pump produces ripple at the input frequency:

```
V_ripple = V_cc / (f_in × R1 × C2)
```

At 80 kHz input, 100 kΩ, 0.15 μF:
```
V_ripple = 5V / (80000 × 100000 × 0.15e-6) = 5V / 1200 ≈ 4.2 mV p-p
```

This 4.2 mV ripple at 80 kHz is well above the ADS1115's Nyquist frequency and will alias to near-DC if not filtered. The ADS1115's internal PGA filter provides additional attenuation, but an external RC anti-alias filter (10 kΩ + 0.1 μF, fc = 160 Hz) at the ADC input is recommended.

## 2-Pole Butterworth Option

For applications requiring sharper roll-off (e.g., if EMI TX frequency is moved closer to the measurement bandwidth), a 2-pole Butterworth filter can be implemented using an op-amp after the LM2917:

```
                    R
    LM2917 out ───/\/\/──┬──── Op-amp ──── To ADC
                         │
                        C1
                         │
                        GND
```

A Sallen-Key topology with fc = 10 Hz provides 40 dB/decade roll-off instead of the single-pole 20 dB/decade. This doubles the rejection at 15 kHz to ~126 dB.

**Current recommendation**: The single-pole RC filter (63 dB at 15 kHz) is sufficient given TDM mitigation. The 2-pole option is reserved for future if bench testing reveals insufficient rejection.

## Component Selection Guide

| Component | Recommended Part | Value | Notes |
|-----------|-----------------|-------|-------|
| C1 (timing) | Polypropylene film | 0.01 μF ±1% | Low drift, temperature stable |
| C2 (filter) | X7R ceramic or film | 0.15 μF | Sets bandwidth |
| R1 (load) | Metal film | 100 kΩ ±1% | Low tempco |
| Anti-alias RC | Metal film + X7R | 10 kΩ + 0.1 μF | fc ≈ 160 Hz at ADC input |

### Temperature Considerations

- LM2917 tempco: ~0.3%/°C **(Modeled)**
- C1 tempco (polypropylene): ~100 ppm/°C
- R1 tempco (metal film): ~25 ppm/°C
- **Combined drift over 20°C**: ~6% on LM2917 + <0.5% on passive components **(Modeled)**
- Gradiometer subtraction cancels most common-mode drift

## References

- LM2917 datasheet: Texas Instruments SNAS555C
- Application Note AN-162: LM2907/LM2917 Frequency to Voltage Converter
