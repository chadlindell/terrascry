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

## Electronics Architecture

```
+-------------------+
|   GPS Module      |----+
|   (NEO-6M)        |    |
+-------------------+    |
                         |    +------------------+
+-------------------+    +--->|                  |
|   Fluxgate Pair 1 |--->|    |   Arduino Nano   |
+-------------------+    |    |                  |
                         |    |   - Read 8 ADC   |
+-------------------+    |    |   - Log to SD    |
|   Fluxgate Pair 2 |--->|--->|   - GPS parse    |
+-------------------+    |    |   - Pace beeper  |
                         |    |                  |
+-------------------+    |    +--------+---------+
|   Fluxgate Pair 3 |--->|             |
+-------------------+    |             v
                         |    +------------------+
+-------------------+    |    |   SD Card        |
|   Fluxgate Pair 4 |--->+    |   (CSV logging)  |
+-------------------+         +------------------+

+-------------------+         +------------------+
|   LiPo Battery    |-------->|   Speaker/Buzzer |
|   7.4V 2000mAh    |         |   (pace beeper)  |
+-------------------+         +------------------+
```

### ADC Requirements

- 8 channels (4 pairs x 2 sensors)
- 16-bit resolution adequate for gradiometry
- 2x ADS1115 modules provide 8 channels total
- Sample rate: 10-50 Hz sufficient for walking pace

## Bill of Materials (Estimated)

| Component | Qty | Est. Cost | Notes |
|-----------|-----|-----------|-------|
| **Sensors** |
| Fluxgate sensors (FG Sensors or equiv.) | 8 | $480-640 | 4 pairs x 2 sensors |
| **Frame** |
| Carbon fiber tube 2m x 25mm | 1 | $40 | Main crossbar |
| PVC conduit 20mm x 50cm | 4 | $10 | Sensor drop tubes |
| 3D printed mounts | 8 | $20 | Sensor clips |
| **Harness** |
| Backpack harness (salvage) | 1 | $0-30 | Old backpack straps |
| Bungee cord | 2m | $5 | Vibration dampening |
| Carabiners | 4 | $10 | Quick-release |
| **Electronics** |
| Arduino Nano | 1 | $5-25 | |
| ADS1115 16-bit ADC | 2 | $10 | 4 channels each |
| GPS module (NEO-6M) | 1 | $15 | Position tagging |
| SD card module | 1 | $5 | Data logging |
| LiPo 7.4V 2000mAh | 1 | $20 | Belt-mounted |
| Speaker/buzzer | 1 | $2 | Pace beeper |
| Enclosure (IP65) | 1 | $15 | Electronics housing |
| Cables, connectors | - | $25 | |
| **Total** | | **$660-870** | |

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

## Open Questions

1. **Sensor selection**: FG Sensors fluxgates vs. alternatives (cost/performance tradeoff)
2. **ADC noise floor**: Is ADS1115 adequate, or need better ADC?
3. **Crossbar material**: Carbon fiber vs. fiberglass vs. aluminum (weight/cost/rigidity)
4. **Enclosure design**: Belt-mounted vs. integrated into crossbar
5. **Calibration procedure**: Factory calibration vs. field calibration protocol

## References

- [Bartington Grad601 Manual](https://www.bartingtondownloads.com/wp-content/uploads/OM1800.pdf)
- [Berkeley Grad601 Field Guide](https://arf.berkeley.edu/files/attachments/books/Bartington_Grad601_2_Setup_And_Operation_1.pdf)
- [tMag Array Paper](https://gi.copernicus.org/articles/10/313/2021/)
- [FG Sensors DIY Kit](https://www.fgsensors.com/diy-gradiometer-kit)
- [SENSYS MagWalk](https://sensysmagnetometer.com/products/magwalk-magnetometer-survey-kit/)
