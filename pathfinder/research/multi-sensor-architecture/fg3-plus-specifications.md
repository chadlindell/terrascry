# FG-3+ Self-Oscillating Fluxgate Specifications

## Overview

The FG-3+ from FG Sensors is a self-oscillating fluxgate magnetometer. Unlike traditional driven-excitation fluxgates that use a separate excitation oscillator and sense winding with synchronous demodulation, the FG-3+ uses a single toroidal core that self-oscillates at a frequency determined by the ambient magnetic field. This fundamentally changes the signal chain design compared to a conventional fluxgate.

## Operating Principle

The FG-3+ operates on the self-oscillating (or auto-oscillating) fluxgate principle:

1. A toroidal permalloy core is wound with a single coil that forms part of an LC oscillator circuit.
2. The core saturates twice per oscillation cycle, creating a frequency that depends on the external magnetic field.
3. The output is a **frequency** (not a voltage), typically in the 50-120 kHz range.
4. The frequency shift is proportional to the ambient field component along the sensor axis.

### Key Implication

Because the output is a frequency, the LM2917 frequency-to-voltage converter is not just a convenience — it is the primary demodulation stage. The LM2917's bandwidth and noise characteristics directly determine the measurement performance.

## Electrical Specifications

| Parameter | Value | Source |
|-----------|-------|--------|
| Output type | Frequency (self-oscillating) | FG Sensors documentation |
| Output frequency range | 50-120 kHz (field-dependent) | **(Modeled)** from similar sensors |
| Supply current | ~12 mA per sensor | FG Sensors documentation |
| Supply voltage | 5V nominal | FG Sensors documentation |
| Bandwidth (DC to) | ~20 kHz (oscillation-limited) | **(Modeled)** |
| Sensitivity | 0.118 μs/μT (period sensitivity) | FG Sensors documentation |
| Core type | Toroidal permalloy | FG Sensors documentation |

## Noise Characteristics

### Intrinsic Sensor Noise

- **Noise density**: ~10-20 pT/√Hz at 1 Hz **(Modeled)** — based on comparable low-cost self-oscillating fluxgates
- **At 10 Hz bandwidth**: 30-60 pT RMS **(Modeled)**
- **Note**: The manufacturer does not publish detailed noise specifications. These estimates are based on published data from similar self-oscillating fluxgate designs (e.g., Primdahl 1979, Ripka 2001).

### Thermal Drift

- **Temperature coefficient**: ~46 nT/°F **(Modeled)** — derived from comparable sensors
- **Over 20°C field day swing**: ~1,600 nT total drift if uncorrected **(Modeled)**
- **Implication**: Gradiometer subtraction cancels most drift (common-mode), but residual from sensor-to-sensor mismatch requires attention

### Conducted Crosstalk

When multiple FG-3+ sensors share a power supply, their self-oscillation frequencies can couple through the supply rail:

- **Mechanism**: Each sensor draws pulsed current at its oscillation frequency; this modulates the supply voltage, which affects neighboring sensors
- **Severity**: Potentially significant if supply impedance is not low enough
- **Mitigation**: Individual LM78L05 voltage regulators per sensor (see power-supply-architecture.md)

### Power Supply Sensitivity

The self-oscillation frequency is sensitive to supply voltage variations:

- Supply voltage ripple directly modulates the core saturation point
- Buck converter switching noise (typically 100-500 kHz) can inject into the oscillation circuit
- **Mitigation**: Dedicated low-noise LDO per sensor, ferrite bead + LC filter upstream

## Implications for Pathfinder Design

1. **LM2917 is the demodulator**: The frequency-to-voltage conversion is not optional — it IS the measurement. The LM2917's bandwidth and linearity directly limit performance.
2. **Individual power supplies required**: Shared power causes conducted crosstalk between sensors.
3. **EMI sensitivity**: The self-oscillating circuit acts as an antenna at 50-120 kHz. External EMI at or near this frequency range can corrupt the oscillation — the EMI conductivity coil operating at 15 kHz is safely outside this range but harmonics must be considered.
4. **TDM timing**: The settling time after power-on or after EMI TX shutdown must be characterized for the TDM firmware.

## References

- FG Sensors product page: https://www.fgsensors.com/
- Primdahl, F. (1979). The fluxgate magnetometer. *Journal of Physics E*, 12(4), 241.
- Ripka, P. (2001). *Magnetic Sensors and Magnetometers*. Artech House.
