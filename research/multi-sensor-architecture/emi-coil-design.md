# EMI Conductivity Coil Design

## Overview

The EMI (Electromagnetic Induction) conductivity channel adds below-ground electrical conductivity mapping to Pathfinder's above-ground magnetic gradiometry. This creates a "above ground + below ground" multi-physics survey capability in a single pass.

## Operating Principle

The EMI channel uses the frequency-domain electromagnetic (FDEM) method, similar to the Geonics EM38:

1. A transmitter (TX) coil driven at a fixed frequency (e.g., 14.6 kHz for EM38) generates a primary magnetic field in the ground.
2. This time-varying field induces eddy currents in conductive subsurface materials.
3. The eddy currents generate a secondary magnetic field that is detected by a receiver (RX) coil.
4. The secondary field is phase-shifted relative to the primary: the in-phase component relates to magnetic susceptibility, and the quadrature (90°) component relates to electrical conductivity.

### Low Induction Number (LIN) Approximation

For the operating frequencies and coil separations used in portable instruments, the LIN approximation applies. This allows direct calculation of apparent conductivity:

```
σ_a = (4 / (ω × μ₀ × s²)) × Im(Hs/Hp)
```

Where:
- σ_a = apparent conductivity (S/m)
- ω = 2πf (angular frequency)
- μ₀ = 4π × 10⁻⁷ (permeability of free space)
- s = coil separation (m)
- Hs/Hp = ratio of secondary to primary field at the receiver

### Secondary-to-Primary Ratio

Under LIN conditions:
```
Hs/Hp = iωμ₀σs² / 4
```

This is purely imaginary (quadrature), confirming that the quadrature component is the conductivity signal.

## Signal Levels

For typical soils (σ = 10-100 mS/m), the secondary-to-primary ratio is:

| Soil conductivity | Hs/Hp (ppm) | Notes |
|-------------------|-------------|-------|
| 10 mS/m (dry sand) | ~100 ppm | Very weak signal |
| 50 mS/m (loam) | ~500 ppm | Moderate |
| 100 mS/m (wet clay) | ~1000 ppm | Strong |
| 300 mS/m (saline) | ~3000 ppm | Very strong |

These are parts-per-million of the primary field. The receiver must detect a signal that is 1000-10000× weaker than the primary field. This is why primary field cancellation (bucking coil) is essential.

## Coil Geometry Options

### Horizontal Coplanar (HCP)

Both TX and RX coils horizontal (parallel to ground). Effective exploration depth ≈ 1.5 × s (coil separation).

### Vertical Coplanar (VCP)

Both TX and RX coils vertical. Effective exploration depth ≈ 0.75 × s. More sensitive to near-surface layers.

### Recommended for Pathfinder

HCP geometry with s = 1.0 m (matching EM38 configuration). This provides:
- Exploration depth ≈ 1.5 m (HCP mode)
- Matches Pathfinder's magnetic detection depth range
- TX and RX can be mounted on the existing crossbar

## Bucking Coil Concept

The primary field at the RX coil is ~10⁶× stronger than the secondary field signal. A bucking coil (also called a compensation or nulling coil) placed between TX and RX generates a field that cancels the primary at the RX location:

1. Small coil wound in opposition to TX, positioned at a calculated distance from RX.
2. Cancels >99% of primary field coupling.
3. Residual primary is removed by the phase-sensitive detector (AD630).
4. Allows the receive amplifier chain to operate at higher gain without saturation.

Detailed bucking coil calculations are in `emi-coil-prototype-spec.md` (Phase 3, R3).

## Signal Chain

```
AD9833 DDS ──► OPA549 Power Amp ──► TX Coil
  (15 kHz)      (current drive)     (30 turns)

                                    RX Coil ──► AD8421 Preamp ──► AD630 Phase Detector
                                   (30 turns)   (gain: 100)       (I and Q channels)
                                                                        │
                                                                   RC Low-Pass Filter
                                                                   (fc ~ 10 Hz)
                                                                        │
                                                                   ADS1115 ADC
                                                                   (I and Q channels)
```

### Component Roles

| Component | Function | Key Spec |
|-----------|----------|----------|
| AD9833 | Direct Digital Synthesis, generates precise 15 kHz sine wave | 0.1 Hz resolution, SPI control |
| OPA549 | High-current op-amp drives TX coil | Up to 8A output, shutdown pin for TDM |
| TX Coil | Generates primary field | 30 turns, 12 cm diameter |
| Bucking Coil | Cancels primary at RX | Calculated turns/position |
| RX Coil | Detects secondary field | 30 turns, 8 cm diameter |
| AD8421 | Low-noise instrumentation amplifier | 3 nV/√Hz, gain = 100 |
| AD630 | Balanced modulator/demodulator for phase-sensitive detection | Extracts I (in-phase) and Q (quadrature) |
| RC LPF | Removes carrier frequency, passes DC conductivity signal | fc ~ 10 Hz |
| ADS1115 | 16-bit ADC for I and Q channels | Shared with fluxgate ADC bank |

## Open-Source Reference Designs

### CSM-EM (Colorado School of Mines)

- Open-source FDEM instrument design
- Uses similar LIN approach
- Published schematics and firmware
- Reference: https://github.com/csm-em

### MEMIS (Modular Electromagnetic Induction System)

- Modular open-source EM instrument
- Multiple coil geometries supported
- SimPEG-compatible data format

## SimPEG FDEM Forward Modeling API

GeoSim can validate EMI measurements using SimPEG's FDEM module:

```python
from SimPEG.electromagnetics import frequency_domain as fdem

# Define survey
survey = fdem.Survey(source_list)

# Define 1D layered model
sigma_map = maps.ExpMap(nP=n_layers)

# Forward model
simulation = fdem.Simulation1DLayered(
    survey=survey,
    sigmaMap=sigma_map,
)
d_pred = simulation.dpred(model)
```

This enables synthetic testing of the EMI channel before hardware is built.

## References

- McNeill, J.D. (1980). Electromagnetic terrain conductivity measurement at low induction numbers. Technical Note TN-6, Geonics Limited.
- Geonics EM38 manual: Operating principles and specifications.
- Ward, S.H., & Hohmann, G.W. (1988). Electromagnetic theory for geophysical applications. *Electromagnetic Methods in Applied Geophysics*, Vol. 1, SEG.
