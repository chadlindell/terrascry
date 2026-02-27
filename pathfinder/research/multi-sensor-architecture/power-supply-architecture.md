# Power Supply Architecture

## Overview

The Pathfinder multi-sensor system requires a carefully designed power supply architecture that prevents switching noise from corrupting sensitive analog measurements. The design uses a 3-stage approach: efficient switching conversion, followed by passive filtering, followed by ultra-low-noise linear regulation.

## Design Philosophy

1. **Separate analog and digital power domains** — switching noise from digital circuits must not reach analog sensor circuits
2. **Individual regulation per fluxgate sensor** — prevents conducted crosstalk between self-oscillating FG-3+ sensors
3. **Star grounding** — all ground returns to a single point to prevent ground loop currents
4. **Shielded inductors** — all switching converter inductors must be shielded types to minimize radiated EMI

## 3-Stage Power Supply Chain

```
LiPo Battery     Stage 1: Buck       Stage 2: LC Filter     Stage 3: LDO
  7.4V           Converter            Passive                Ultra-Low Noise
  2000mAh        (efficient)          (EMI removal)          (clean output)

  ┌─────┐      ┌──────────┐      ┌──────────────┐      ┌───────────┐
  │     │──►──│ LM2596    │──►──│ Ferrite Bead │──►──│ TPS7A49   │──► Analog 5V Rail
  │ 7.4V│      │ 7.4V→5.5V│      │ + LC filter  │      │ 5.5V→5.0V│    (to LM78L05s)
  │     │      │ 150 kHz  │      │ 10μH + 100μF │      │ 60dB PSRR│
  └─────┘      └──────────┘      └──────────────┘      └───────────┘
                    │
                    └──────────────────────────────────────────────────► Digital 5V Rail
                                                                         (ESP32, SD, etc.)
```

### Stage 1: LM2596 Buck Converter

| Parameter | Value |
|-----------|-------|
| Input | 7.4V LiPo (6.0-8.4V range) |
| Output | 5.5V (headroom for LDO) |
| Switching frequency | 150 kHz |
| Efficiency | ~85% at 500 mA |
| Output ripple | 30-50 mV p-p (before filtering) |
| Inductor | **Shielded** 33 μH (Coilcraft MSS1260 or Wurth WE-PD) |

**Critical**: The inductor MUST be a shielded type. Unshielded inductors radiate switching-frequency magnetic fields that directly corrupt fluxgate readings.

### Stage 2: Passive LC Filter

After the buck converter, a ferrite bead + LC filter removes switching frequency ripple:

| Component | Value | Purpose |
|-----------|-------|---------|
| Ferrite bead | 600Ω @ 100 MHz (e.g., BLM18PG601SN1) | High-frequency broadband attenuation |
| L (inductor) | 10 μH shielded | LC filter resonance |
| C (capacitor) | 100 μF low-ESR electrolytic + 10 μF X7R ceramic | Energy storage + high-frequency bypass |

Filter cutoff: fc = 1/(2π√(LC)) = 1/(2π√(10μ × 100μ)) ≈ 5 kHz

Attenuation at 150 kHz: ~60 dB (from LC filter alone)

### Stage 3: TPS7A49 Ultra-Low-Noise LDO

| Parameter | Value |
|-----------|-------|
| Input | 5.5V (from LC filter) |
| Output | 5.0V |
| Dropout | 310 mV typical |
| Output noise | 4.4 μV RMS (10 Hz - 100 kHz) |
| PSRR | >60 dB at 150 kHz |
| Output current | Up to 150 mA |

**Combined attenuation at 150 kHz (buck switching frequency):**
- LC filter: ~60 dB
- LDO PSRR: ~60 dB
- **Total: ~120 dB** (1,000,000:1 rejection)

The 30-50 mV ripple from the buck converter is reduced to ~30-50 pV at the analog rail — well below any measurement significance.

## Individual Sensor Regulators

Each FG-3+ fluxgate sensor gets its own LM78L05 voltage regulator:

```
Analog 5V Rail ──► LM78L05 #1 ──► FG-3+ Sensor 1
(from TPS7A49)  ──► LM78L05 #2 ──► FG-3+ Sensor 2
                ──► LM78L05 #3 ──► FG-3+ Sensor 3
                ──► LM78L05 #4 ──► FG-3+ Sensor 4
                ──► LM78L05 #5 ──► FG-3+ Sensor 5
                ──► LM78L05 #6 ──► FG-3+ Sensor 6
                ──► LM78L05 #7 ──► FG-3+ Sensor 7
                ──► LM78L05 #8 ──► FG-3+ Sensor 8
```

### Why Individual Regulators?

The FG-3+ is a self-oscillating sensor — its oscillation frequency is sensitive to supply voltage. When multiple sensors share a regulator:
1. Sensor A draws pulsed current at its oscillation frequency (e.g., 80 kHz)
2. This modulates the shared supply voltage by a small amount
3. Sensor B's oscillation frequency shifts in response
4. This creates **conducted crosstalk** between sensors

Individual LM78L05 regulators isolate each sensor's power supply, eliminating this coupling path.

### LM78L05 Specifications

| Parameter | Value |
|-----------|-------|
| Output voltage | 5.0V ±4% |
| Output current | 100 mA max (FG-3+ needs ~12 mA) |
| Dropout | ~1.7V (input must be >6.7V) — we supply 5.0V from TPS7A49, so we need 5.5V from buck or use 3.3V regulators |
| Ripple rejection | 51 dB at 120 Hz |
| Output noise | ~40 μV RMS |

**Note**: The LM78L05 dropout voltage (1.7V) means it needs >6.7V input to regulate at 5.0V. With our 5.0V TPS7A49 output, the LM78L05 cannot regulate. **Resolution options**:

1. Set TPS7A49 output to 5.0V and use LM78L33 (3.3V) regulators if FG-3+ can operate at 3.3V
2. Set buck converter output higher (7V) and feed LM78L05s directly from the LC filter stage
3. Use LDO regulators with lower dropout (e.g., MCP1700, 178 mV dropout) instead of LM78L05

**Recommended**: Option 2 — Feed the LM78L05 individual regulators from the post-LC-filter 5.5V rail, and use the TPS7A49 only for the precision analog rail (ADS1115, LM2917 reference). This gives the LM78L05s sufficient headroom while still benefiting from the LC filter stage.

## Revised Architecture

```
LiPo 7.4V ──► LM2596 (→5.5V) ──► Ferrite+LC ──┬──► TPS7A49 (→5.0V) ──► ADS1115, LM2917 ref
                                                 │
                                                 ├──► LM78L05 ×8 ──► FG-3+ sensors
                                                 │    (5.5V input, marginal but functional)
                                                 │
                                                 └──► 3.3V LDO ──► ESP32, digital sensors

LiPo 7.4V ──► Direct ──► OPA549 power amp (EMI TX coil driver)
```

**Note**: 5.5V into LM78L05 is marginal (only 0.5V headroom vs 1.7V dropout spec). For reliable operation, either:
- Increase buck output to 7V and accept slightly lower efficiency
- Replace LM78L05 with low-dropout regulators (MCP1700, AP2112)

This design decision requires bench testing and is flagged for R5 consensus validation.

## Star Grounding

```
                    ┌─── Analog GND (fluxgates, ADC, LM2917)
                    │
Battery GND ──── STAR ──── Digital GND (ESP32, SD card, GPS)
                    │
                    └─── Power GND (buck converter, motor drivers)
```

All ground returns connect at a single star point near the battery. No analog signal current flows through digital ground traces. On the PCB, this means:
- Separate ground pours for analog and digital
- Single connection point between the two pours
- No traces crossing the ground split

## Decoupling Specifications

| Location | Capacitor | Type | Purpose |
|----------|-----------|------|---------|
| Each LM78L05 output | 100 nF + 10 μF | X7R ceramic + electrolytic | Local bypass |
| Each ADS1115 VDD | 100 nF | X7R ceramic | High-frequency bypass |
| TPS7A49 output | 10 μF + 1 μF | Ceramic X7R | LDO stability |
| ESP32 VDD | 100 nF + 10 μF | X7R ceramic + tantalum | Bulk + HF bypass |
| AD9833 VDD/AGND | 100 nF + 10 μF | X7R ceramic | DDS quiet supply |
| AD8421 supply | 100 nF | X7R ceramic | Preamp supply bypass |

## Power Budget

| Subsystem | Voltage | Current | Power |
|-----------|---------|---------|-------|
| 8× FG-3+ sensors | 5V | 96 mA | 0.48W |
| 2× ADS1115 | 5V | 0.4 mA | 0.002W |
| 8× LM2917 | 5V | 24 mA | 0.12W |
| ESP32 | 3.3V | 240 mA (WiFi active) | 0.79W |
| ZED-F9P GPS | 3.3V | 68 mA | 0.22W |
| BNO055 IMU | 3.3V | 12 mA | 0.04W |
| MLX90614 IR | 3.3V | 1.5 mA | 0.005W |
| AD9833 + OPA549 | 5V | 500 mA (TX active) | 2.5W (intermittent) |
| RPLiDAR C1 | 5V | 700 mA | 3.5W |
| ESP32-CAM | 5V | 300 mA | 1.5W (intermittent) |
| **Total (continuous)** | | | **~2.1W** |
| **Total (peak, all active)** | | | **~9.2W** |

### Battery Life Estimate

With 7.4V × 2000 mAh = 14.8 Wh battery:
- Continuous operation (fluxgate + GPS + comms): 14.8 / 2.1 ≈ **7 hours**
- Heavy use (all sensors, frequent EMI + camera): 14.8 / 5.0 ≈ **3 hours**
- Peak (everything simultaneously): 14.8 / 9.2 ≈ **1.6 hours**

TDM helps significantly — the EMI TX and camera are only active for brief phases.

## References

- LM2596 datasheet: Texas Instruments SNVS124E
- TPS7A49 datasheet: Texas Instruments SBVS163B
- LM78L05 datasheet: Texas Instruments SNOS764F
- Application Note: Power Supply Design for Mixed-Signal Systems (TI SLYT107)
