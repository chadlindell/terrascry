# Pathfinder Design Concept

## Overview

Pathfinder is a harness-supported multi-channel fluxgate gradiometer for rapid magnetic reconnaissance. The design follows the proven "trapeze" configuration used by commercial systems like the Bartington Grad601, but extends it to 4 sensor pairs for wider swath coverage.

## Physical Configuration

```
                    PADDED HARNESS
                   (backpack straps)
                          |
              +-----------+-----------+
              |     bungee/elastic    |
              |      suspension       |
              +-----------+-----------+
                          |
    ======================+=======================
              CARBON FIBER OR ALUMINUM TUBE
                    (1.5 - 2.0 m)
        |         |         |         |
       +-+       +-+       +-+       +-+
       |T|       |T|       |T|       |T|    <- TOP sensors (reference)
       | |       | |       | |       | |       mounted on crossbar
       | |       | |       | |       | |    <- PVC drop tubes (50cm)
       |B|       |B|       |B|       |B|    <- BOTTOM sensors (signal)
       +-+       +-+       +-+       +-+       ~15-20 cm above ground

       <--50cm--><--50cm--><--50cm-->
                 1.5 m swath
```

## Gradiometer Principle

Each gradiometer pair consists of two vertically-separated fluxgate sensors:

- **Top sensor (reference)**: Measures background magnetic field (~50 cm above ground)
- **Bottom sensor (signal)**: Measures field including near-surface anomalies (~15-20 cm above ground)
- **Gradient**: Difference between sensors cancels regional field, isolates local anomalies

This configuration is insensitive to:
- Diurnal magnetic field variations
- Regional geological gradients
- Operator movement through Earth's field

## Why Multiple Pairs Work

Fluxgate sensors are **passive receivers** - they don't emit fields that could interfere with neighbors. Commercial systems routinely operate 8-10 pairs simultaneously:

| System | Channels | Spacing | Reference |
|--------|----------|---------|-----------|
| Foerster 10-channel | 9-10 pairs | 25 cm | [ResearchGate](https://www.researchgate.net/figure/Magnetic-survey-with-motorized-10-channel-Foerster-gradiometer-array-mounted-with-25-cm_fig1_261061128) |
| tMag (Bartington) | 8 pairs | 50 cm | [Geoscientific Instrumentation](https://gi.copernicus.org/articles/10/313/2021/) |
| Bartington Grad601-2 | 2 pairs | 100 cm | [Bartington](https://www.bartington.com/products/grad601/) |

Crosstalk only becomes significant at <2 cm spacing (medical MEG applications). At 50 cm spacing, it's negligible.

## Harness System

Based on the [Bartington Grad601 trapeze design](https://arf.berkeley.edu/files/attachments/books/Bartington_Grad601_2_Setup_And_Operation_1.pdf):

### Key Features

1. **Shoulder straps**: Padded backpack-style straps distribute weight
2. **Waist belt**: Stabilizes load, prevents swinging
3. **Elastic suspension**: Bungee cords isolate sensor vibration from walking motion
4. **Quick-release**: Carabiners allow rapid don/doff
5. **Height adjustment**: Straps set sensor height to 15-20 cm above ground

### Ergonomics

- Operator walks at steady pace (~1 m/s)
- Hands rest lightly on crossbar for guidance only
- Arms never bear weight
- Audio beeper marks pace (one beep per meter)

## Multi-Sensor Electronics Architecture

The Pathfinder electronics architecture has been expanded from a single-modality magnetic gradiometer to a multi-sensor platform controlled by an ESP32 dual-core MCU with time-division multiplexed (TDM) measurement cycles.

```
                                    ┌─────────────────────┐
SENSOR POD (shared w/ HIRT)         │                     │
┌────────────────────┐   Cat5 STP   │      ESP32          │
│ ZED-F9P RTK GPS    │◄────────────►│   (Dual Core)       │
│ BNO055 IMU         │  PCA9615     │                     │
│ BMP390 Baro        │  diff I2C    │ Core 0: WiFi/MQTT   │
│ DS3231 RTC         │  (Bus 1)     │   NTRIP, SD, GPS    │
└────────────────────┘              │                     │
                                    │ Core 1: TDM engine  │
Fluxgate Pairs 1-4  ──► LM2917 ──► │   ADC, EMI, tilt    │
(8× FG-3+ sensors)      (F-to-V)   │                     │
                                    │ I2C Bus 0:          │
ADS1115 ×2 (0x48,0x49) ◄───────────│   Local sensors     │
MLX90614 IR (0x5A)  ◄──────────────│                     │
                                    └──────┬──────────────┘
                                           │
EMI TX Coil ◄── OPA549 ◄── AD9833 DDS     │ WiFi / MQTT
EMI RX Coil ──► AD8421 ──► AD630 (I/Q) ──►│
                                           ▼
RPLiDAR C1 ──── USB ──────────────► Jetson (edge compute)
ESP32-CAM  ──── GPIO trigger        MQTT broker + storage
                                    Anomaly detection
LiPo Battery 7.4V 2000mAh          LiDAR → DEM
├── LM2596 buck (5.5V)
│   ├── Ferrite+LC filter
│   │   ├── TPS7A49 LDO (5.0V analog rail)
│   │   └── AP2112K-5.0 ×8 (individual per FG-3+)
│   └── 3.3V LDO (ESP32, digital)
└── Direct to OPA549 (EMI TX power)
```

### Time-Division Multiplexing (TDM)

The TDM firmware prevents electromagnetic interference between sensors by ensuring only compatible sensors operate simultaneously. Each 100 ms cycle:

| Phase | Duration | Active Sensors | Disabled |
|-------|----------|---------------|----------|
| Fluxgate | 50 ms | Fluxgates, ADC, IMU, IR, GPS RX | EMI TX, WiFi TX, SD card |
| EMI TX/RX | 30 ms | EMI coils, ADC (I/Q), lock-in | WiFi TX |
| Settling | 20 ms | WiFi TX, MQTT, SD write, NTRIP | EMI TX |

See `research/multi-sensor-architecture/tdm-firmware-design.md` for full implementation details.

### EMI Conductivity Channel

The EMI channel adds subsurface electrical conductivity mapping using the FDEM method:

- **TX**: AD9833 DDS generates 15 kHz sine → OPA549 drives TX coil (30 turns, 12 cm dia)
- **RX**: AD8421 preamp (gain 100) → AD630 phase detector → I and Q channels → ADS1115
- **Bucking coil**: Cancels primary field at RX location (>99% cancellation)
- **Output**: Apparent conductivity σ_a = (4/(ωμ₀s²)) × Q/primary

See `research/multi-sensor-architecture/emi-coil-design.md` for signal chain details.

### Physical Layout (Crossbar)

Sensor placement is optimized to minimize electromagnetic interference:

```
-100cm    -50cm     0cm      +15cm  +25cm +50cm +75cm +100cm
  │         │        │         │      │     │     │      │
LiDAR    EMI TX  Electronics  IR   FG-1  FG-2  FG-3   FG-4
              EMI RX (-10cm)  CAM
              GPS mast (above)
              Sensor pod
```

See `research/multi-sensor-architecture/crossbar-physical-layout.md` for detailed layout.

### Interference Mitigation

15 source-victim interference paths have been analyzed and mitigated:

- **CRITICAL**: EMI TX → fluxgate (2100 nT at 1m) → TDM + physical separation
- **CRITICAL**: LiDAR motor → fluxgate (100-300 nT) → 1.25-2.0m separation + gradiometer subtraction
- **HIGH**: Buck converter → ADC → 3-stage power supply (95-110 dB PSRR **(Modeled)**)
- See `research/multi-sensor-architecture/interference-matrix.md` for complete analysis.

### Consensus-Validated Design Corrections

Multi-model consensus validation (GPT-5.2, GPT-5.2-Pro, Grok 4.1 Fast) identified the following critical corrections to the initial design:

| Issue | Severity | Correction |
|-------|----------|------------|
| LM78L05 dropout at 5.5V input (needs 6.7V+) | **CRITICAL** | Replace with AP2112K-5.0 (250mV dropout) — feeds from 5.5V post-LC rail |
| ADS1115 "DMA" claim over I2C impossible | **CRITICAL** | Use ALERT/RDY pin interrupt-driven continuous mode; 860 SPS for EMI phase |
| WiFi `esp_wifi_stop/start` too slow for 100ms TDM | **HIGH** | Use `esp_wifi_set_max_tx_power(0)` + application-level send blocking |
| M8 4-pin connector insufficient for PCA9615 | **CRITICAL** | Upgrade to M8 8-pin (PCA9615 needs 6 conductors: 2×SDA, 2×SCL, VCC, GND) |
| Solenoid inductance formula wrong for short coils | **HIGH** | Wheeler's approximation: L_TX = 150-180 μH (not 710 μH) |
| Single AD630 cannot produce simultaneous I/Q | **HIGH** | Need two AD630s or digital lock-in approach |
| 20ms settling insufficient (26% LM2917 residual) | **MODERATE** | Increase to 45ms (3τ) or use active discharge/discard strategy |
| GPIO 16/17 conflict (I2C Bus 1 vs UART2) | **HIGH** | Reassign GPS UART2 to GPIO 32/33 |
| BMP390 in sealed IP67 enclosure invalid | **HIGH** | Add IP67 vent membrane (Gore PolyVent or Amphenol) |

See `research/multi-sensor-architecture/` consensus files for detailed analysis and recommendations.

### ADC Requirements

- 8 channels for fluxgates (4 pairs × 2 sensors) via 2× ADS1115
- 2 additional channels for EMI I/Q (shared ADS1115 MUX or dedicated 3rd unit)
- 16-bit resolution adequate for gradiometry
- Sample rate: 10 Hz (TDM-limited, 128 SPS raw ADC rate)

## Bill of Materials (Estimated)

The multi-sensor upgrade adds 7× sensing capability for approximately 60% more cost. Full BOM details in `research/multi-sensor-architecture/updated-bom.md`.

| Category | Est. Cost | Key Components |
|----------|-----------|----------------|
| **Fluxgate sensors** | $480-640 | 8× FG-3+ (4 gradiometer pairs) |
| **Frame + harness** | $85-115 | Carbon fiber tube, PVC drops, harness, bungee |
| **ESP32 + original electronics** | $57 | ESP32, 2× ADS1115, 8× LM2917, SD card, battery |
| **Power supply upgrade** | $18 | LM2596 + ferrite+LC + TPS7A49 + 8× AP2112K-5.0 |
| **EMI conductivity channel** | $71 | AD9833 + OPA549 + AD8421 + AD630 + coils |
| **IR temperature** | $15 | MLX90614xAC |
| **Camera** | $15 | ESP32-CAM standalone |
| **LiDAR** | $73-93 | RPLiDAR C1 |
| **Sensor pod (50% shared)** | $182 | ZED-F9P + BNO055 + BMP390 + DS3231 + PCA9615 |
| **EMI mitigation** | $22 | M8 8-pin connectors, shielded cable, bypass caps, vent membrane |
| **Total** | **$1,052-1,262** | |

## Weight Budget

| Component | Weight |
|-----------|--------|
| 8 fluxgate sensors | ~400g |
| Carbon tube + PVC drops | ~300g |
| Electronics + battery | ~250g |
| Harness + suspension | ~200g |
| Cables | ~100g |
| **Total** | **~1.25 kg** |

Commercial comparison:
- Bartington Grad601: 1.6 kg (2 sensors)
- SENSYS MagWalk: 1.6 kg (2 sensors)

Pathfinder achieves **4x the sensors at similar weight**.

## Performance Estimates

### Coverage Rate

| Configuration | Swath | Speed | Area/Hour |
|---------------|-------|-------|-----------|
| 2-pair (1.0 m) | 0.5 m | 3.6 km/h | ~1,800 m² |
| 4-pair (1.5 m) | 1.5 m | 3.6 km/h | ~5,400 m² |

### Detection Capability

Based on fluxgate gradiometer physics:

| Target | Estimated Detection Depth |
|--------|--------------------------|
| Small ferrous (nail, shell casing) | 20-40 cm |
| Medium ferrous (helmet, tool) | 50-100 cm |
| Large ferrous (engine block, UXO) | 1-2 m |
| Fired clay (kiln, hearth) | 30-60 cm |
| Disturbed soil (burial, pit) | 20-50 cm |

**Note**: These are estimates based on similar commercial systems. Actual performance requires field validation.

## Data Format

CSV output with columns:

```
timestamp,lat,lon,g1_top,g1_bot,g1_grad,g2_top,g2_bot,g2_grad,g3_top,g3_bot,g3_grad,g4_top,g4_bot,g4_grad
2024-01-15T10:23:45.123,51.234567,18.345678,48234,48456,222,48189,48401,212,...
```

- Timestamp: ISO 8601 with milliseconds
- Position: Decimal degrees (WGS84)
- Readings: Raw ADC counts (calibration applied in post-processing)
- Gradient: Bottom minus top (computed on-device for quick review)

## Post-Processing Workflow

1. **Download**: Transfer CSV from SD card
2. **Import**: Load into QGIS, Google Earth, or custom Python script
3. **Grid**: Interpolate point data to regular grid
4. **Visualize**: Generate gradient magnitude map
5. **Identify**: Mark anomalies for HIRT follow-up

## Joint Inversion with HIRT

Pathfinder's surface measurements serve as boundary conditions for HIRT's subsurface 3D tomographic inversion. The operational workflow:

1. **Pathfinder survey** (30-60 min): Walk site, collect magnetics + EMI + IR + LiDAR
2. **Anomaly flagging**: Jetson detects anomalies in real-time
3. **HIRT deployment** (2-4 hr per anomaly): Insert probes around flagged targets
4. **Joint inversion**: Combine Pathfinder surface data + HIRT crosshole data in SimPEG

The shared sensor pod ensures GPS coordinate registration between instruments. Cross-gradient regularization couples the magnetic susceptibility and conductivity models. See `GeoSim/docs/research/joint-inversion-concept.md`.

## Open Questions

1. ~~**Sensor selection**: FG Sensors fluxgates vs. alternatives~~ → Resolved: FG-3+ selected
2. **ADC noise floor**: Is ADS1115 adequate, or need 24-bit ADS1256? (see noise-analysis.md)
3. **Crossbar material**: Carbon fiber for fluxgate section, fiberglass elsewhere (aluminum causes eddy currents for EMI)
4. ~~**LM78L05 dropout**: 5.5V input marginal~~ → Resolved: Replace with AP2112K-5.0 (250mV dropout, feeds from 5.5V post-LC rail) (R5 consensus)
5. ~~**WiFi vs fiber**: TDM + shielding estimated < 0.01 nT interference~~ → Resolved: WiFi+TDM sufficient; realistic estimate 0.01-0.1 nT; fiber as optional upgrade (R1 consensus)
6. ~~**EMI coil bucking**: Precise turns/position calculation needed~~ → Resolved: Wheeler L_TX=150-180 μH, resonant C=625-750 nF, need two AD630s for I/Q (R3 consensus)
7. **LiDAR TDM gating**: Motor DC field at fluxgate — need bench characterization
8. **ADS1115 timing**: Cannot achieve 5 cycles/50ms at 128 SPS — need ALERT/RDY interrupt mode (R2 consensus)
9. **M8 connector upgrade**: 4-pin → 8-pin for PCA9615 differential I2C (R4 consensus)

## References

- [Bartington Grad601 Manual](https://www.bartingtondownloads.com/wp-content/uploads/OM1800.pdf)
- [Berkeley Grad601 Field Guide](https://arf.berkeley.edu/files/attachments/books/Bartington_Grad601_2_Setup_And_Operation_1.pdf)
- [tMag Array Paper](https://gi.copernicus.org/articles/10/313/2021/)
- [FG Sensors DIY Kit](https://www.fgsensors.com/diy-gradiometer-kit)
- [SENSYS MagWalk](https://sensysmagnetometer.com/products/magwalk-magnetometer-survey-kit/)
