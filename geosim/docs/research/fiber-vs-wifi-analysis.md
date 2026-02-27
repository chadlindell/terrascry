# Fiber Optic vs WiFi for Field Data Transmission

## Overview

This document evaluates whether Pathfinder should use fiber optic data links instead of (or in addition to) ESP32 WiFi for transmitting measurement data to the Jetson edge computer. The question is motivated by potential WiFi RF interference with sensitive fluxgate magnetometer measurements.

## Context

The Ukraine-Russia conflict demonstrated fiber optic tethers for FPV drones to defeat RF jamming. This concept—replacing wireless links with fiber to eliminate RF—is directly applicable to geophysical instruments where RF interference from the data link itself can corrupt sensitive measurements.

### The Interference Concern

ESP32 WiFi transmits at 2.4 GHz with up to 20 dBm (100 mW) output power. Potential interference mechanisms:

1. **Direct coupling**: 2.4 GHz RF enters analog amplifier inputs and is rectified by semiconductor junctions
2. **Intermodulation**: WiFi signal mixes with fluxgate oscillation frequency, producing in-band products
3. **Supply modulation**: WiFi TX current draw (peak ~300 mA) modulates power supply voltage
4. **Ground bounce**: High-frequency return currents create voltage drops in shared ground plane

## Decision Framework

| WiFi Interference Level (after TDM) | Decision |
|--------------------------------------|----------|
| < 0.1 nT at fluxgate | WiFi is fine — no action needed |
| 0.1 - 1.0 nT | Marginal — fiber recommended for high-precision work |
| > 1.0 nT | Fiber required for useful measurements |

## Analysis: WiFi Interference After TDM Mitigation

### TDM Mitigation

In the TDM firmware, WiFi TX is **disabled** during the fluxgate measurement phase (50 ms). This means:

- No 2.4 GHz RF emission during measurement
- No WiFi TX current draw during measurement
- WiFi is only active during the settling/communications phase (20 ms)

### Residual Concerns

Even with TDM:

1. **WiFi receiver**: The ESP32 WiFi receiver is still active (listening for NTRIP corrections, MQTT ACKs). The local oscillator in the receiver emits low-level RF.
2. **Settling transients**: When WiFi TX re-enables after measurement, the transient may affect the start of the next measurement cycle.
3. **Power supply ripple**: WiFi TX current draw creates supply transients that may not fully settle within the 20 ms settling phase.

### Quantitative Estimate

| Mechanism | Estimated Interference | After TDM + Shielding |
|-----------|----------------------|----------------------|
| WiFi TX at 2.4 GHz | 1-100 mV at amp input | **0** (TX off during measurement) |
| WiFi RX local oscillator | ~0.01 mV at amp input | ~0.001 mV (enclosure shielding) |
| Supply modulation | ~5 mV on supply rail | ~0.5 μV (after 120 dB PSRR chain) |
| LM2917 rejection at 2.4 GHz | >100 dB inherent | Negligible |
| **Combined residual** | | **< 0.01 nT** **(Modeled)** |

**Conclusion**: With TDM + metal enclosure + power supply filtering, WiFi interference is estimated at < 0.01 nT — well below the 0.1 nT threshold. **WiFi appears sufficient.**

## Fiber Optic Options (If Needed)

### Option 1: USB-to-Fiber Media Converter

| Parameter | Value |
|-----------|-------|
| Type | USB 2.0 over fiber media converter pair |
| Fiber | Multimode OM3 (50/125 μm) or POF |
| Range | Up to 500m |
| Cost | $80-150 per pair |
| Data rate | Up to 480 Mbps (USB 2.0) |
| Power | 5V, ~100 mA per unit |

**Pro**: Plug-and-play, no firmware changes. ESP32's USB-serial output goes through fiber to Jetson.
**Con**: Two extra boxes, fiber cable to manage, power for converters.

### Option 2: SFP Module on Jetson

| Parameter | Value |
|-----------|-------|
| Type | SFP transceiver on Jetson's Ethernet |
| Fiber | Single-mode or multimode |
| Range | Up to 10 km |
| Cost | $20-40 per SFP, $30 for media converter on ESP32 side |
| Data rate | 1 Gbps |

**Pro**: High bandwidth, low latency, industrial-grade.
**Con**: Requires Ethernet-capable interface on instrument side (ESP32 doesn't have native Ethernet).

### Option 3: Plastic Optical Fiber (POF)

| Parameter | Value |
|-----------|-------|
| Type | PMMA plastic fiber, 1mm core |
| Range | Up to 50m |
| Cost | $5-10/m cable, $20-30 per transceiver |
| Data rate | Up to 100 Mbps |
| Advantage | Cheap, flexible, easy to terminate (just cut and polish) |

**Pro**: Cheapest, most field-friendly (flexible, won't break like glass fiber).
**Con**: Limited range (50m vs 500m for glass). Adequate for Pathfinder use.

### Option 4: Hybrid — Fiber for Data, WiFi Disabled Entirely

| Parameter | Value |
|-----------|-------|
| Approach | ESP32 WiFi completely disabled; all data over fiber |
| NTRIP corrections | Via Jetson Ethernet → fiber → ESP32 |
| EMI from ESP32 | Minimal — no RF transmission at all |

**Pro**: Absolute guarantee of zero WiFi interference.
**Con**: Trailing cable is operationally burdensome. Cannot be used in all terrain.

## Operational Comparison

| Factor | WiFi | Fiber |
|--------|------|-------|
| Setup time | 0 (built-in) | 5-10 min (cable deployment) |
| Cable management | None | Must manage 10-100m fiber reel |
| Terrain flexibility | Any terrain | Limited by cable (snag risk in brush/forest) |
| Reliability | Occasional dropouts | Near-perfect link reliability |
| EMI impact | < 0.01 nT with TDM **(Modeled)** | 0 nT (no RF) |
| Cost | $0 (built-in) | $50-200 (transceivers + cable) |
| Weight | 0 | ~200g (cable + reel) |
| Range | ~30m (open field) | 50-500m (depending on fiber type) |

## Recommendation

**Primary: WiFi with TDM is sufficient for standard surveying.**

The TDM approach (WiFi off during measurement) combined with metal enclosure shielding and the LM2917's inherent narrowband filtering reduces WiFi interference to < 0.01 nT. This is 50× below the 0.5 nT system noise floor.

**Secondary: POF fiber as an upgrade path for high-precision applications.**

For users requiring the absolute lowest noise floor (e.g., detecting graves or subtle fired clay anomalies near the detection limit), a POF fiber kit could be offered as an optional upgrade. The hybrid approach (fiber for data, WiFi disabled entirely) eliminates all RF interference.

**This recommendation requires bench validation** — see R1 consensus task. The quantitative estimate of < 0.01 nT WiFi interference must be confirmed by measurement before this decision is finalized.

## Bench Test Protocol

To validate the WiFi interference estimate:

1. Set up Pathfinder in a magnetically quiet environment (wooden table, no nearby ferrous objects)
2. Record 60 seconds of gradient data with WiFi OFF (baseline noise floor)
3. Record 60 seconds with WiFi ON, TDM enabled (normal operation)
4. Record 60 seconds with WiFi ON, TDM disabled (worst case — WiFi active during measurement)
5. Compute RMS noise and spectral content for each condition
6. The difference between conditions 2 and 1 is the actual WiFi interference with TDM

If condition 2 - condition 1 < 0.1 nT, WiFi is confirmed adequate.

## References

- ESP32 WiFi technical reference: Espressif Systems
- Plastic optical fiber communications: Koike, Y. (2015). Fundamentals of Plastic Optical Fibers. Wiley.
- Ukraine drone fiber optic tether concept: Various defense analysis reports, 2023-2024
