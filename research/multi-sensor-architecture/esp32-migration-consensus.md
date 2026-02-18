# ESP32 Migration Consensus — I1: Pathfinder Firmware ESP32 Migration

**Date**: 2026-02-18
**Task**: I1 — Pathfinder Firmware ESP32 Migration
**Method**: PAL Multi-Model Consensus Validation
**Models Consulted**: Claude Opus 4.6 (independent analysis), openai/gpt-5.2 (neutral)
**Models Unavailable**: gemini-3-pro-preview (Gemini API free-tier daily quota exhausted — 429 RESOURCE_EXHAUSTED)
**Consensus Confidence**: 8.5/10
**Reference Files**:
- `tdm-firmware-design.md`
- `i2c-address-map.md`

---

## Executive Summary

The firmware migration from Arduino Nano (ATmega328P) to ESP32-DevKitC V4 (ESP-WROOM-32) is **technically feasible and architecturally sound**. Both models unanimously agree on the viability of the migration, but identify **three critical issues** that must be resolved before implementation:

1. **GPIO 16/17 hard conflict** between I2C Bus 1 and UART2 defaults
2. **GPIO 5 strapping pin risk** for SD card chip select
3. **GPIO 12 bootstrap pin risk** if HSPI is used for AD9833

All three have clear, well-established resolutions documented below.

---

## Concern-by-Concern Analysis

### Concern 1: GPIO 16/17 Conflict (I2C Bus 1 vs GPS UART2)

**Severity**: CRITICAL — Hard pin conflict
**Consensus**: UNANIMOUS

**Problem**: The `i2c-address-map.md` (lines 112-114) assigns I2C Bus 1 to GPIO 16 (SDA) and GPIO 17 (SCL) for the sensor pod connection via PCA9615. The ESP32 UART2 also defaults to these same pins.

**Resolution**: Keep I2C Bus 1 on GPIO 16/17 (as documented) and reassign UART2 to GPIO 32/33.

| Signal | Original Plan | Consensus Assignment | Rationale |
|--------|--------------|---------------------|-----------|
| I2C Bus 1 SDA | GPIO 16 | GPIO 16 (unchanged) | Documented in i2c-address-map.md, fewer downstream changes |
| I2C Bus 1 SCL | GPIO 17 | GPIO 17 (unchanged) | Same as above |
| GPS UART2 RX | GPIO 16 | **GPIO 32** | Safe, non-strapping, input-capable |
| GPS UART2 TX | GPIO 17 | **GPIO 33** | Safe, non-strapping, output-capable |

**Additional Note (Claude)**: GPIO 16/17 are connected to PSRAM on some ESP32 module variants (ESP-WROOM-32**D** with PSRAM). Confirm the specific module is standard ESP-WROOM-32 (no PSRAM) before committing to this pin assignment.

**Additional Note (Claude)**: Since the ZED-F9P GPS replaces the NEO-6M and communicates via I2C (address 0x42 on Bus 1), the UART2 GPS pins may only be needed for legacy NEO-6M backward compatibility or for feeding RTCM corrections. If no UART peripheral is required at all, the conflict is moot.

### Concern 2: ADS1115 Library Compatibility with Dual Wire Instances

**Severity**: LOW — Confirmed compatible
**Consensus**: UNANIMOUS

The Adafruit_ADS1X15 library accepts a `TwoWire*` parameter in its `begin()` method:

```c
Adafruit_ADS1115 ads1;
Adafruit_ADS1115 ads2;

ads1.begin(0x48, &I2C_Local);  // Bus 0
ads2.begin(0x49, &I2C_Local);  // Bus 0
```

**GPT-5.2 caveat**: If pinned to an older library version, the `TwoWire*` parameter may not be available. Ensure `lib_deps` specifies a recent version (>=2.0.0).

**Additional finding (both models)**: The TDM firmware design document (line 36) mentions "DMA-triggered continuous sampling of 8 fluxgate channels via ADS1115." This is **misleading** — the ADS1115 communicates over I2C and cannot support true DMA streaming. The implementation must use **timer-triggered polling** at the desired sample rate. At 128 SPS, each ADS1115 conversion takes ~7.8 ms; the I2C transaction overhead adds ~0.2-0.5 ms per read. This fits within the 50 ms Phase 1 window for 5 complete conversion cycles per ADC.

**Action item**: Correct `tdm-firmware-design.md` line 36 — change "DMA-triggered continuous sampling" to "Timer-triggered polling" to avoid implementation confusion.

### Concern 3: SD Card Library (SD_MMC vs SPI Mode)

**Severity**: LOW — Clear recommendation
**Consensus**: UNANIMOUS

**Recommendation**: Use **SPI mode** via `SdFat` library (preferred) or Arduino `SD.h`.

| Option | Pins Used | Compatibility | Recommendation |
|--------|-----------|--------------|----------------|
| SD_MMC (SDIO mode) | GPIO 2, 4, 12, 13, 14, 15 | Conflicts with bootstrap pins, other peripherals | **Do NOT use** |
| SD.h via VSPI | GPIO 18, 19, 23, 5 | Matches Nano SPI approach | Acceptable |
| SdFat via VSPI | GPIO 18, 19, 23, CS (see below) | Better performance, more features | **Preferred** |

**GPT-5.2 raised**: GPIO 5 is a **strapping pin** on the ESP32. If the SD card module or wiring pulls CS (GPIO 5) low during boot, it can cause boot mode failures. This is a common "works on the bench, fails in the field" issue.

**Consensus**: Move SD CS from GPIO 5 to **GPIO 13** (non-strapping, general purpose).

### Concern 4: WiFi Initialization Timing

**Severity**: MEDIUM — Design decision
**Consensus**: UNANIMOUS

**Recommendation**: Initialize WiFi stack at startup, but **gate TX power** per TDM cycle rather than performing full start/stop cycles.

The TDM firmware design document (line 224) already documents this approach:
> "For TDM gating, use `esp_wifi_set_max_tx_power(0)` / `esp_wifi_set_max_tx_power(78)` which is faster (<1 ms)."

| Approach | Latency | TDM Compatible | Recommendation |
|----------|---------|----------------|----------------|
| `esp_wifi_stop()` / `esp_wifi_start()` | 10-100 ms variable | Too slow for 20 ms Phase 3 | **Do NOT use** |
| `esp_wifi_set_max_tx_power(0/78)` | <1 ms | Fits within Phase 3 | **Use this** |

**Startup sequence**:
1. Initialize I2C buses and sensors (Core 1 ready)
2. Initialize WiFi stack (non-blocking, on Core 0)
3. Begin TDM measurement cycles immediately
4. WiFi association and MQTT connection happen asynchronously during Phase 3 windows

### Concern 5: Memory and Buffer Sizes

**Severity**: LOW — Straightforward upgrade
**Consensus**: UNANIMOUS

ESP32 has 520 KB SRAM vs Arduino Nano's 2 KB. No risk of memory pressure, but buffers should be **intentionally sized** rather than arbitrarily maximized.

| Buffer | Nano Size | ESP32 Recommended | Rationale |
|--------|-----------|-------------------|-----------|
| ADC sample buffer | 8-16 samples | 512 samples | 5 complete ADS1115 cycles per phase |
| NMEA parse buffer | 64 bytes | 1024 bytes | Full multi-constellation sentences |
| CSV write buffer | 128 bytes | 4096 bytes | Reduce SD write frequency |
| MQTT message buffer | N/A | 2048 bytes | JSON payload with all sensor channels |
| FreeRTOS measurement queue | N/A | Depth: 4-8 structs | Decouple Core 1 measurement from Core 0 I/O |

**GPT-5.2 recommendation**: Prefer **static/ring buffers** over dynamic allocation to avoid heap fragmentation on long field runs. Avoid Arduino `String` class in production firmware.

### Concern 6: PlatformIO Dual Environments

**Severity**: LOW — Standard practice
**Consensus**: UNANIMOUS

Yes, `platformio.ini` supports multiple `[env:xxx]` sections. Use conditional compilation for platform-specific code:

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
lib_deps =
    adafruit/Adafruit ADS1X15@^2.5.0
    greiman/SdFat@^2.2.0
build_flags = -DESP32_BUILD

[env:nanoatmega328]
platform = atmelavr
board = nanoatmega328
framework = arduino
lib_deps =
    adafruit/Adafruit ADS1X15@^2.5.0
    SD
```

Source code uses `#ifdef ESP32` / `#ifdef __AVR__` for platform-specific sections (dual Wire, FreeRTOS tasks, WiFi, etc.).

### Concern 7: Boot Time

**Severity**: LOW — Manageable
**Consensus**: UNANIMOUS (with nuance)

| Phase | Duration | Notes |
|-------|----------|-------|
| ESP32 ROM bootloader | ~50 ms | Fixed |
| Second-stage bootloader | ~200-300 ms | Loads app from flash |
| App init (sensors, I2C) | ~100-200 ms | Fast |
| WiFi stack init | ~1-2 s | Can be deferred |
| WiFi association | ~1-3 s | Asynchronous on Core 0 |
| **Total to first measurement** | **~500 ms** | Without WiFi |
| **Total to full operation** | **~2-3 s** | With WiFi + MQTT |

**Field implications**: Start measurement and SD logging immediately on boot. WiFi/MQTT connection happens asynchronously. Add a startup LED sequence (e.g., 3 rapid blinks) to indicate "ready" state.

---

## Consensus GPIO Assignment (Final)

### Complete Pin Map

```
ESP32-DevKitC V4 (ESP-WROOM-32) GPIO Assignment
================================================

I2C Bus 0 (Local Sensors — Main PCB)
  GPIO 21 ── SDA  ──> ADS1115 #1 (0x48), ADS1115 #2 (0x49), MLX90614 (0x5A)
  GPIO 22 ── SCL  ──> 400 kHz, short PCB traces

I2C Bus 1 (Sensor Pod — via PCA9615 differential I2C over Cat5)
  GPIO 16 ── SDA  ──> ZED-F9P (0x42), BNO055 (0x29), BMP390 (0x77), DS3231 (0x68)
  GPIO 17 ── SCL  ──> 100 kHz, conservative for 1-2m cable

GPS UART2 (if needed for legacy NEO-6M or RTCM forwarding)
  GPIO 32 ── RX
  GPIO 33 ── TX

SD Card (VSPI — SPI Mode)
  GPIO 18 ── SCK
  GPIO 19 ── MISO
  GPIO 23 ── MOSI
  GPIO 13 ── CS     (moved from GPIO 5 to avoid strapping pin risk)

AD9833 DDS (Bitbang SPI — write-only, no MISO needed)
  GPIO 27 ── SCK
  GPIO 26 ── MOSI (SDATA)
  GPIO 25 ── CS (FSYNC)

Control Signals
  GPIO  4 ── EMI TX shutdown (OPA549 shutdown pin)
  GPIO 34 ── GPS PPS input (input-only, interrupt capable)
  GPIO  2 ── Status LED (onboard DevKitC LED)
  GPIO 35 ── ESP32-CAM trigger (input-only, or use GPIO 15 if output needed)

UART0 (USB Debug)
  GPIO  1 ── TX (default, USB-Serial)
  GPIO  3 ── RX (default, USB-Serial)

Reserved / Unused
  GPIO  0 ── Strapping pin (BOOT button) — do not use
  GPIO  5 ── Strapping pin (freed from SD CS) — leave floating or use with caution
  GPIO 12 ── Strapping pin (MTDI) — avoid, can cause 1.8V VDD_SDIO if HIGH at boot
  GPIO 14 ── Available (spare)
  GPIO 15 ── Available (spare, strapping pin but less critical)
  GPIO 36 ── Available (input-only, ADC1_CH0)
  GPIO 39 ── Available (input-only, ADC1_CH3)
```

### Pin Conflict Resolution Summary

| Conflict | Pins | Resolution | Risk Level |
|----------|------|------------|------------|
| I2C Bus 1 vs UART2 default | GPIO 16, 17 | UART2 moved to GPIO 32/33 | Resolved |
| SD CS on strapping pin | GPIO 5 | CS moved to GPIO 13 | Resolved |
| HSPI MISO bootstrap risk | GPIO 12 | AD9833 uses bitbang on GPIO 25/26/27 | Resolved |
| PSRAM conflict (some modules) | GPIO 16, 17 | Verify ESP-WROOM-32 variant has no PSRAM | Verify before fabrication |

---

## Additional Findings

### Test Adaptation for FreeRTOS (GPT-5.2)

Existing unit tests (`test_adc`, `test_gps`, `test_config`) assume a single-threaded blocking loop (Arduino Nano model). On ESP32 with FreeRTOS:

- Tests must either run within a FreeRTOS task context, or
- Provide a "single-thread compatibility mode" build flag that bypasses task creation for testing, or
- Use PlatformIO's `test_framework = unity` with ESP32 native test runner

**Recommended approach**: Use `#ifdef UNIT_TEST` to run sensor tests in a simplified single-task mode on ESP32, bypassing the full TDM state machine.

### Documentation Corrections Needed

1. **tdm-firmware-design.md line 36**: Change "DMA-triggered continuous sampling" to "Timer-triggered polling" for ADS1115 reads
2. **tdm-firmware-design.md line 230**: Add note that ADS1115 "DMA" is not literally DMA over I2C — it is timer-interrupt-driven conversion reads
3. **i2c-address-map.md lines 112-114**: Add note about PSRAM variant incompatibility with GPIO 16/17

### Library Recommendations

| Component | Library | Version | Notes |
|-----------|---------|---------|-------|
| ADS1115 | Adafruit ADS1X15 | >=2.5.0 | Supports TwoWire* parameter |
| SD Card | SdFat | >=2.2.0 | Better performance than SD.h |
| GPS (if UART) | TinyGPSPlus | >=1.0.3 | Lightweight NMEA parser |
| BNO055 | Adafruit BNO055 | >=1.6.0 | Supports TwoWire* |
| MLX90614 | Adafruit MLX90614 | >=2.1.0 | Supports TwoWire* |
| WiFi/MQTT | PubSubClient | >=2.8 | Or AsyncMqttClient for non-blocking |
| FreeRTOS | Built into ESP-IDF | N/A | Included with espressif32 platform |

---

## Points of Agreement (All Models)

1. Migration is technically feasible and a strong architectural upgrade
2. ESP32 dual-core maps naturally to TDM measurement (Core 1) vs I/O (Core 0)
3. GPIO 16/17 conflict resolved by moving UART2 to GPIO 32/33
4. SPI-mode SD (SdFat) is the correct choice, not SD_MMC
5. WiFi TX power gating is preferred over start/stop cycles
6. ADS1115 over I2C is timer-polled, not DMA
7. Dual PlatformIO environments in one platformio.ini is standard practice
8. Boot time to first measurement is ~500 ms (acceptable for field use)
9. Buffer sizes should be intentionally chosen, not maximized
10. Static/ring buffers preferred over dynamic allocation for long runs

## Points Requiring Further Validation

1. Exact ESP-WROOM-32 module variant (PSRAM presence) — affects GPIO 16/17 availability
2. Specific SD card module behavior at boot with GPIO 13 as CS (verify no unexpected pull)
3. Adafruit_ADS1X15 library version currently in use (ensure >=2.0.0 for TwoWire* support)
4. SdFat compatibility with ESP32 Arduino core version in use
5. WiFi TX power gating electromagnetic compatibility with fluxgate readings during Phase 1

---

## Actionable Next Steps

1. **Verify module variant**: Confirm ESP-WROOM-32 (no PSRAM) to validate GPIO 16/17 for I2C Bus 1
2. **Update i2c-address-map.md**: Add PSRAM variant warning note
3. **Update tdm-firmware-design.md**: Correct "DMA" references to "timer-triggered polling"
4. **Create platformio.ini dual environment**: Add `[env:esp32dev]` alongside existing `[env:nanoatmega328]`
5. **Implement GPIO map**: Apply the consensus pin assignment from this document
6. **Adapt unit tests**: Add `#ifdef UNIT_TEST` single-thread mode for ESP32 test execution
7. **Bench test SD CS on GPIO 13**: Verify reliable boot with SD module connected
8. **Bench test WiFi TX power gating**: Measure fluxgate noise with WiFi power cycling to validate TDM isolation

---

## Consensus Process Notes

- **gemini-3-pro-preview** was requested but unavailable due to Gemini API free-tier daily quota exhaustion (429 RESOURCE_EXHAUSTED). The model returned errors on all retry attempts across both direct Gemini API and OpenRouter routing.
- The consensus was completed with two perspectives: Claude Opus 4.6 (independent analysis) and openai/gpt-5.2 (neutral evaluation).
- Despite the missing third model, both available perspectives showed strong agreement on all 7 concerns, giving high confidence in the recommendations.
- Consider re-running with gemini-3-pro-preview when API quota resets for additional validation.
