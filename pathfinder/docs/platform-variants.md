# Pathfinder Platform Variants and Positioning

## Why This Document Exists

Pathfinder targets a gap in the geophysical instrument market: no open-source, multi-sensor fluxgate gradiometer exists that a competent maker can build for under $1,000. Commercial equivalents cost 10-50x more, and the few DIY magnetometer projects are single-sensor educational builds — not field-ready multi-channel survey tools.

This document defines how Pathfinder compares to commercial alternatives, how position accuracy affects data quality, and how the core electronics support three deployment platforms: handheld trapeze, backpack, and drone.

---

## Part 1: DIY vs Commercial — Where Pathfinder Fits

### The Market Landscape

| System | Type | Channels | Weight | GPS | Approx. Cost | Availability |
|--------|------|----------|--------|-----|-------------|--------------|
| **Pathfinder** | DIY fluxgate gradiometer | 4 pairs (8 sensors) | <1.5 kg (Target) | NEO-6M (2-5 m) or RTK | **$620-820** | Open-source |
| Bartington Grad601-2 | Commercial fluxgate | 2 pairs | ~1.6 kg | Optional add-on | ~$9,000 used; $15-25k new (est.) | Quote only |
| Foerster FEREX 4.032 | Commercial fluxgate | 1-4 channels | 4.1 kg | Optional | ~$15-30k (est.) | Quote only |
| SENSYS MXPDA | Cart-based fluxgate | Multi-sensor | Cart | RTK GNSS option | ~$20-40k (est.) | Quote only |
| Geometrics MagArrow II | Drone cesium vapor | 1 total-field | 1.2 kg | Integrated | ~$2-10k (est.) | Quote only |
| GEM GSM-19T | Proton precession | 1-2 | ~3.5 kg | Integrated | ~$15-25k (est.) | Quote only |
| Magnetometer-Kit.com | DIY single-pair kit | 1 pair | N/A | None | ~$205 | Direct order |

**Note**: Commercial pricing is estimated from used-market listings, rental rates, and industry context. Manufacturers do not publish prices. "(est.)" = estimated, not confirmed.

### Pathfinder's Value Proposition

1. **2x the swath width** of a Grad601-2 (4 pairs vs 2 pairs), at <5% of the cost
2. **Open hardware and firmware** — no vendor lock-in, fully repairable in the field
3. **Modular GPS** — start with $15 NEO-6M, upgrade to $185 RTK when precision matters
4. **Multi-platform** — same electronics board works handheld, backpack, or drone-mounted
5. **$620-820 total build cost** vs $9,000+ for the cheapest used commercial equivalent

### What Pathfinder Is NOT

Pathfinder is a **screening tool**, not a survey-grade instrument. It trades:
- Noise floor for cost (ADS1115 16-bit vs 24-bit ADCs in commercial units)
- Position accuracy for accessibility (consumer GPS vs integrated DGPS/RTK)
- Certification for openness (no traceable calibration, no CE/FCC marking)

For detailed characterization of anomalies Pathfinder identifies, use HIRT, GPR, or a rented commercial gradiometer.

### Cost Comparison: 7-Day Survey Campaign

| Approach | Equipment | GPS/Position | Total |
|----------|-----------|-------------|-------|
| **Build Pathfinder (basic)** | $620-820 | Included (NEO-6M) | **$620-820** |
| **Build Pathfinder (RTK)** | $620-820 | +$185-300 RTK kit | **$805-1,120** |
| **Rent Foerster FEREX** | $150/day x 7 = $1,050 | Included | **$1,110** (+ shipping) |
| **Buy used Grad601-2** | ~$9,200 | +$500 GPS add-on | **~$9,700** |
| **Buy new commercial** | $15,000-40,000 | Included or +$2-5k | **$15,000-45,000** |

Pathfinder pays for itself after a single week compared to rental, and immediately compared to purchase.

---

## Part 2: Position Accuracy

### Why Position Matters

Gradiometer data without position is a time series. With position, it becomes a map. The quality of that map depends directly on GPS accuracy:

| GPS Module | Accuracy | Grid Cell Size | Use Case |
|-----------|----------|----------------|----------|
| NEO-6M (standard) | 2-5 m CEP | 5 m practical minimum | Anomaly flagging, area screening |
| NEO-M8N/M8P | 1-2.5 m CEP | 2-3 m practical | Better screening, rough gridding |
| ZED-F9P (RTK float) | 30-60 cm | 50 cm practical | Useful gridded maps |
| **ZED-F9P (RTK fix)** | **1-2 cm** | **25 cm practical** | **Survey-grade gridded maps** |

**CEP** = Circular Error Probable (50% of fixes within this radius).

### GPS Upgrade Path

Pathfinder uses UART GPS via SoftwareSerial. Any NMEA-compatible GPS module can be swapped in without firmware changes — only the baud rate and pin wiring may differ.

#### Tier 1: NEO-6M (Default — $10-18)

- Included in the base BOM
- 2-5 m accuracy, adequate for "where is the anomaly" screening
- Good enough for QGIS point overlay on satellite imagery
- **Limitation**: Cannot produce meaningful interpolated grid maps at 50 cm resolution

#### Tier 2: NEO-M9N ($25-40)

- Drop-in replacement, same footprint, same UART protocol
- 1.5 m CEP with SBAS corrections
- Multi-constellation (GPS + GLONASS + Galileo + BeiDou)
- **Good for**: Improved point maps, rough gridding at 2-3 m cells

#### Tier 3: ZED-F9P RTK ($185-300 with antenna)

- ArduSimple simpleRTK2B or SparkFun GPS-RTK2 breakout
- **1-2 cm accuracy** with RTK correction stream (NTRIP via phone hotspot or radio link)
- UART output at 115200 baud, NMEA compatible
- **Good for**: Survey-grade gridded maps, GIS integration, academic publication
- **Adds**: ~$185 for module + antenna, plus NTRIP subscription ($0-50/month depending on country)

#### Tier 4: Post-Processed Kinematic (PPK) — No Additional Hardware

- Log raw GNSS observations from ZED-F9P alongside Pathfinder CSV
- Post-process against CORS base station data (free in most countries)
- Achieves cm-level accuracy without real-time correction link
- **Best for**: Drone surveys where RTK radio link is impractical

### Firmware Implications

The current firmware reads NMEA sentences via `TinyGPSPlus` over SoftwareSerial. This works with all four GPS tiers above. Changes needed for RTK support:

| Change | Scope | Required? |
|--------|-------|-----------|
| Increase `GPS_BAUD` to 115200 | `config.h` | Yes, for ZED-F9P |
| Switch to hardware Serial (D0/D1) for faster baud | `main.cpp` | Recommended for 115200 baud (SoftwareSerial unreliable above 57600) |
| Add `fix_quality` column to CSV | `main.cpp`, `config.h` | Recommended — distinguishes no-fix / autonomous / DGPS / RTK float / RTK fix |
| Log HDOP (horizontal dilution of precision) | `main.cpp` | Optional — useful for filtering low-quality fixes in post-processing |

**These changes are backward-compatible**: the NEO-6M still works at 9600 baud over SoftwareSerial when `GPS_BAUD=9600`.

---

## Part 3: Platform Variants

### Core Electronics (Shared)

All three platforms use the same firmware codebase with compile-time configuration:

- Arduino Nano + 1-2x ADS1115 + GPS module + SD card + buzzer + LED
- `NUM_SENSOR_PAIRS` (1-4) controls how many pairs are read, logged, and plotted
- `PLATFORM_HANDHELD`, `PLATFORM_BACKPACK`, or `PLATFORM_DRONE` sets platform defaults
- CSV column count adjusts automatically (fewer pairs = narrower CSV)
- Python tools auto-detect pair count from CSV columns
- Builds with `NUM_SENSOR_PAIRS <= 2` compile out the second ADS1115 entirely

Only the **frame**, **power system**, **GPS tier**, **sensor count**, and **build flags** differ.

### Variant A: Handheld Trapeze (Primary Design)

The current Pathfinder design. Operator walks with harness-supported crossbar.

```
                    PADDED HARNESS
                   (backpack straps)
                          |
              +-----------+-----------+
              |   bungee suspension   |
              +-----------+-----------+
                          |
    ======================+=======================
         CARBON FIBER CROSSBAR (1.5-2.0m)
        |         |         |         |
       [T]       [T]       [T]       [T]    <- Top sensors
        |         |         |         |
       [B]       [B]       [B]       [B]    <- Bottom sensors
                                                15-20cm above ground
```

| Parameter | Value |
|-----------|-------|
| Sensor pairs | 4 |
| Sensor spacing | 50 cm |
| Vertical baseline | 30-35 cm |
| Bottom sensor height | 15-20 cm AGL |
| Swath width | 1.5 m |
| Weight | <1.5 kg (Target) |
| Speed | ~1 m/s (walking) |
| Coverage rate | ~5,400 m²/hr (Target) |
| GPS tier | 1 (NEO-6M) or 2-3 |
| Battery | 7.4V 2S LiPo 2000 mAh |
| Runtime | ~5 hours (Modeled) |

**Best for**: Rough terrain, forests, uneven ground, quick screening.

### Variant B: Backpack

Electronics and battery in a small backpack. Crossbar extends below on telescoping poles or a rigid frame. Useful for longer surveys where belt-mounted electronics are uncomfortable, or when carrying additional equipment (RTK radio, tablet).

```
              +------------------+
              |    BACKPACK      |
              |  [Electronics]   |
              |  [Battery]       |
              |  [RTK radio]     |
              +--------+---------+
                       |
              cable harness down
                       |
    ===================+====================
         CROSSBAR (same as handheld)
```

| Parameter | Value |
|-----------|-------|
| Sensor pairs | 2-4 |
| Additional payload capacity | Yes — RTK base/radio, tablet, spare battery |
| Weight on harness | Similar to handheld + backpack contents |
| GPS tier | 2-3 recommended (justifies backpack for RTK radio) |

**Changes from handheld**:
- Longer sensor cables (backpack to crossbar: ~1.5 m vs ~0.5 m)
- Optional: larger battery (3S or USB powerbank) since weight is on back not hips
- Optional: Bluetooth/WiFi module for live data to tablet in backpack

**Firmware changes**: None. Same board, same code.

### Variant C: Drone-Mounted

The Pathfinder electronics board and 1-2 sensor pairs mounted below a multirotor drone. Eliminates operator fatigue, enables access to hazardous or difficult terrain, and achieves uniform coverage patterns.

```
          +------------------+
          |    MULTIROTOR    |
          |   (>2 kg payload)|
          +--------+---------+
                   |
            tether/mount (1-2m below)
                   |
    ===============+================
         SHORT CROSSBAR (0.5-1.0m)
        |         |
       [T]       [T]
        |         |
       [B]       [B]
```

| Parameter | Value |
|-----------|-------|
| Sensor pairs | 1-2 (weight limited) |
| Crossbar length | 0.5-1.0 m |
| Vertical baseline | 30-50 cm |
| Sensor height AGL | 2-5 m (flight altitude limited) |
| Total payload | <800 g (Target) for sensors + electronics |
| Speed | 2-5 m/s |
| Coverage rate | ~10,000-36,000 m²/hr (Modeled, altitude-dependent) |
| GPS tier | 3-4 (RTK or PPK — required for meaningful gridded data) |
| Battery | Drone provides 5V via BEC, or separate small LiPo |
| Runtime | Limited by drone endurance (15-30 min typical) |

**Critical differences from handheld**:

1. **Magnetic interference**: Drone motors produce strong magnetic fields. Sensors must be towed on a 1-2 m cable/rod below the drone, or mounted on a rigid boom with magnetic compensation.
2. **Reduced detection depth**: Higher altitude (2-5 m AGL vs 0.15 m) means weaker gradient signal. Shallow targets (<30 cm) may be undetectable. Drone gradiometry is best for medium-large ferrous targets.
3. **RTK GPS is essential**: At 3 m/s flight speed, 2-5 m GPS error makes data unmappable. ZED-F9P RTK (Tier 3-4) is required.
4. **No pace beeper**: Disable in config. Drone follows pre-programmed flight path.
5. **Higher sample rate**: At 3 m/s, 10 Hz gives 30 cm sample spacing. Consider `SAMPLE_RATE_HZ=20` or higher.
6. **Weight budget**: Every gram matters. Remove buzzer, LED, harness. Use lighter battery or draw from drone BEC.

**Build command**: `pio run -e nano_drone`

All drone-specific behavior is handled by compile flags — no code changes needed:

```
; platformio.ini [env:nano_drone]
-D PLATFORM_DRONE=1      # Auto-disables beeper, enables GPS quality logging
-D NUM_SENSOR_PAIRS=2    # Only reads/logs 2 pairs (ADC2 compiled out)
-D GPS_BAUD=115200       # For ZED-F9P RTK module
-D GPS_LOG_QUALITY=1     # Adds fix_quality, hdop, altitude to CSV
```

### Platform Comparison Summary

| Feature | Handheld | Backpack | Drone |
|---------|----------|----------|-------|
| `NUM_SENSOR_PAIRS` | 4 (default) | 2-4 | 2 (default) |
| ADS1115 modules | 2 | 1-2 | 1 |
| Swath | 1.5 m | 0.5-1.5 m | 0.5-1.0 m |
| Coverage rate | ~5,400 m²/hr | ~5,400 m²/hr | ~10,000-36,000 m²/hr |
| Terrain | Any walkable | Any walkable | Any (including hazardous) |
| Detection depth | Best (sensors at 15-20 cm) | Best | Reduced (sensors at 2-5 m) |
| GPS requirement | Tier 1 OK | Tier 2-3 | Tier 3-4 required |
| CSV columns | 15 | 9-15 | 12 (with GPS quality) |
| Total cost | $620-820 | $400-1,120 | $400-1,120 + drone |
| PlatformIO env | `nanoatmega328` | `nanoatmega328` | `nano_drone` |
| Operator skill | Low | Low | Drone pilot required |

---

## Part 4: Implementation Status

### Implemented

- **`NUM_SENSOR_PAIRS`** (1-4) in `config.h` — controls pair count at compile time
- **Platform flags**: `PLATFORM_HANDHELD`, `PLATFORM_BACKPACK`, `PLATFORM_DRONE`
- **`NEEDS_ADC2`** derived flag — compiles out ADC2 when `NUM_SENSOR_PAIRS <= 2`
- **`GPS_LOG_QUALITY`** flag — adds `fix_quality`, `hdop`, `altitude` columns to CSV
- **Runtime CSV header generation** — column count adjusts to pair count and GPS quality
- **Array-based firmware** — `readGradiometers()`, `logReading()`, `printDebugInfo()` all loop over `NUM_SENSOR_PAIRS`
- **PlatformIO build targets**: `nanoatmega328` (handheld, 4 pairs) and `nano_drone` (drone, 2 pairs)
- **Python tools**: `detect_pairs()` auto-discovers pair count from CSV columns; plots adapt layout

### Future Work (Requires Hardware)

**Phase 6C: Drone Mounting Guide**

Create `hardware/cad/drone-mount.md`:
- Boom/tether design for motor interference rejection
- Weight budget for common drones (DJI M300, Matrice 350, Tarot 680)
- Flight planning for survey grid coverage
- Magnetic compensation notes

**Phase 6D: RTK GPS Integration Guide**

Create `docs/rtk-gps-guide.md`:
- Wiring ZED-F9P to Arduino Nano (UART)
- NTRIP setup via phone hotspot
- PPK workflow with RTKLIB
- Expected accuracy at each correction level

---

## References

- [ArduSimple simpleRTK2B](https://www.ardusimple.com/product/simplertk2b/) — RTK GNSS module
- [Bartington Grad601](https://www.bartington.com/products/grad601/) — Commercial reference design
- [Foerster FEREX 4.034 Datasheet](https://www.ndt-instruments.com/wp-content/uploads/2020/10/FOERSTER_FEREX_4.034_EN.pdf)
- [Geometrics MagArrow II](https://www.geometrics.com/product/magarrow/) — Drone magnetometer
- [SENSYS MagDrone R3](https://sensysmagnetometer.com/products/magdrone-r3-magnetometer-for-drone/)
- [Drone-Borne Magnetic Gradiometry (2024)](https://www.mdpi.com/1424-8220/24/13/4270) — Academic survey of UAV mag methods
- [RTKLIB](http://www.rtklib.com/) — Open-source RTK/PPK processing
- [u-blox ZED-F9P Integration Manual](https://www.u-blox.com/en/docs/UBX-18010854) — Hardware integration reference

---

**Document Status**: Design concept — not yet implemented
**Validation Status**: All specifications are **(Target)** or **(Modeled)** unless noted otherwise
