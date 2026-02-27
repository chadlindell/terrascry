# Sensor Pod Firmware Consensus Validation

## Implementation Task: I3 — Pathfinder Firmware Sensor Pod Detection + Integration

**Date**: 2026-02-18
**Validation Method**: PAL Multi-Model Consensus
**Models Consulted**:
- openai/gpt-5.2 (for stance) — **Successful**, confidence 8/10
- gemini-3-pro-preview (against stance) — **Unavailable** (429 RESOURCE_EXHAUSTED, API quota exceeded)
- Claude Opus 4.6 — Independent analysis (acting as critical counter-perspective)

**Overall Consensus Confidence**: 8/10

---

## Executive Summary

The sensor pod auto-detection and integration firmware plan is **architecturally sound** but contains **one critical bug** and **one high-risk design choice** that must be resolved before implementation. The modular pod concept, graceful degradation hierarchy, and shared-pod interoperability are all well-designed. Five specific issues were identified, with unanimous agreement between GPT-5.2 and independent analysis on all five.

**Verdict: APPROVED WITH MANDATORY MODIFICATIONS**

---

## Proposal Under Review

### Pod Sensors (I2C Bus 1 via PCA9615 Differential Buffer)

| Sensor | I2C Address | Rate | Function |
|--------|-------------|------|----------|
| ZED-F9P GPS | 0x42 | 10 Hz | RTK positioning, NTRIP client via WiFi |
| BNO055 IMU | 0x29 | 100 Hz | IMUPLUS mode (accel+gyro, mag disabled) |
| BMP390 Barometer | 0x77 | 1 Hz | Pressure + temperature |
| DS3231 RTC | 0x68 | On-demand | Precision timestamps |

### Auto-Detection Logic

1. At startup, scan I2C Bus 1 for all 4 addresses
2. ZED-F9P found --> use RTK GPS (fall back to NEO-6M on UART if not)
3. BNO055 found --> enable tilt correction (disable if not)
4. DS3231 found --> use for timestamps (fall back to GPS time if not)
5. None found --> log warning, operate in degraded mode (original Nano functionality)

### Tilt Correction (as proposed)

```
gradient_corrected = gradient_raw * cos(pitch) * cos(roll)
Flag readings where |pitch| > 15 deg or |roll| > 15 deg as unreliable
```

### ZED-F9P RTK Configuration

- 10 Hz update rate via UBX protocol
- NTRIP client on ESP32 Core 0
- RTCM3 corrections forwarded to ZED-F9P via I2C write
- Fix quality monitoring: 0=no fix through 6=fixed RTK

---

## Consensus Findings

### Finding 1: CRITICAL BUG — Tilt Correction Formula Direction is Wrong

**Severity**: CRITICAL
**Agreement**: Unanimous (GPT-5.2 + independent analysis)
**Confidence**: 9/10

#### Problem

The proposed formula `gradient_corrected = gradient_raw * cos(pitch) * cos(roll)` has the correction direction inverted. For a vertical-baseline gradiometer, when the instrument tilts:

- The effective vertical separation between top and bottom sensors **decreases** by a factor of `cos(tilt)`
- This means the **measured gradient is smaller** than the true vertical gradient
- Therefore, correction requires **dividing** by the cosine terms, not multiplying

Multiplying would make the corrected gradient *smaller* than the raw measurement, introducing a systematic under-correction that worsens with tilt angle.

#### Corrected Formula (First-Order)

```
gradient_true = gradient_measured / (cos(pitch) * cos(roll))
```

#### Recommended Formula (Proper Rotation)

For better accuracy, use the BNO055 quaternion output (available in IMUPLUS mode) and apply a full rotation matrix to project the gradient vector onto the true vertical axis:

```c
// Using BNO055 quaternion output
Quaternion q = bno055_get_quaternion();

// Sensor baseline vector in instrument frame (assuming Z-up)
Vector3 baseline = {0, 0, 1};

// Rotate baseline to world frame
Vector3 baseline_world = quaternion_rotate(q, baseline);

// The measured gradient is along the rotated baseline
// True vertical gradient = measured / dot(baseline_world, vertical)
float vertical_projection = baseline_world.z;  // dot product with (0,0,1)

if (fabs(vertical_projection) > cos(15.0 * DEG_TO_RAD)) {
    gradient_corrected = gradient_measured / vertical_projection;
    quality_flag = GOOD;
} else {
    gradient_corrected = gradient_measured;  // uncorrected
    quality_flag = UNRELIABLE_TILT;
}
```

#### Additional Note on Heading

Heading relative to Earth's magnetic field is **not required** for correcting a vertical gradiometer (the gradient is a difference measurement, not absolute field). However, **frame alignment** between the IMU coordinate system and the sensor baseline must be calibrated. The BNO055 X-axis should align with the crossbar axis, Z-axis vertical.

---

### Finding 2: HIGH RISK — ZED-F9P I2C Bandwidth Insufficient for 10 Hz NAV + RTCM3

**Severity**: HIGH
**Agreement**: Unanimous (GPT-5.2 + independent analysis)
**Confidence**: 8/10

#### Problem

At 100 kHz I2C (the maximum recommended for ZED-F9P), theoretical throughput is approximately 10 KB/s. The required data flows are:

| Data Stream | Direction | Size | Rate | Bandwidth |
|-------------|-----------|------|------|-----------|
| UBX NAV-PVT | F9P --> ESP32 | ~100 bytes | 10 Hz | ~1.0 KB/s |
| RTCM3 corrections | ESP32 --> F9P | ~200-500 bytes | 1-5 Hz | ~1.0-2.5 KB/s |
| BNO055 Euler/Quat | BNO055 --> ESP32 | ~8 bytes | 100 Hz | ~0.8 KB/s |
| BMP390 | BMP390 --> ESP32 | ~6 bytes | 1 Hz | ~0.006 KB/s |
| DS3231 | DS3231 --> ESP32 | ~7 bytes | 1 Hz | ~0.007 KB/s |
| I2C overhead | (addressing, ACK, etc.) | ~30% | — | ~1.1 KB/s |
| **Total** | | | | **~5.0-5.5 KB/s** |

While this appears within the 10 KB/s budget, real-world factors reduce headroom significantly:
- Clock stretching by ZED-F9P during data preparation
- ESP32 I2C driver latency and interrupt handling
- Burst characteristics of RTCM3 data (1-3 KB arriving in sub-second bursts)
- Bus arbitration when multiple devices need attention simultaneously

At 50-80% bus utilization, occasional overruns and dropped data become likely.

#### Recommended Solution

**Move ZED-F9P to UART for NAV output and RTCM3 injection.** Keep I2C for configuration only.

This is the **industry standard** approach for RTK rovers. Almost all commercial RTK integrations use UART or USB for correction data and navigation output.

**Implementation options**:

1. **Preferred**: Add UART lines to the pod cable. The Cat5 STP cable has 4 twisted pairs; currently only 3 are used (SDA_D+/-, SCL_D+/-, VCC/GND). Use the spare pair (Pair 4, currently "Reserved/Shield drain") for UART TX/RX to ZED-F9P.

2. **Alternative**: If the single-cable constraint is firm, switch ZED-F9P to SPI (which the F9P supports at up to 5.5 MHz) and use the differential I2C only for the low-rate sensors (BNO055, BMP390, DS3231). This requires a separate SPI-to-differential converter or a more capable pod cable.

3. **Fallback**: If staying on I2C, disable NMEA entirely, use UBX binary only, reduce update rate to 5 Hz, and implement chunked RTCM writes with backpressure handling and dropped-byte metrics.

---

### Finding 3: MEDIUM RISK — NTRIP over WiFi Conflicts with TDM WiFi Gating

**Severity**: MEDIUM
**Agreement**: Unanimous (GPT-5.2 + independent analysis)
**Confidence**: 7/10

#### Problem

The TDM cycle (100ms total: 50ms fluxgate + 30ms EMI + 20ms settling/comms) requires WiFi TX to be disabled during the 50ms fluxgate measurement window to prevent electromagnetic interference. However, the NTRIP client needs a persistent TCP connection:

- TCP requires timely ACKs (typically within seconds, configurable)
- WiFi radio may emit management frames even in "RX only" mode (beacon responses, probe requests, association keepalives)
- The 20ms comms window may be insufficient for receiving RTCM3 burst data (typical RTCM3 message sets are 200-500 bytes per epoch)
- TCP keepalive timeouts need careful tuning to avoid disconnection

#### Recommended Solutions (in order of preference)

1. **Quiet-window scheduling** (preferred): Instead of hard WiFi gating, schedule measurement during WiFi quiet periods. Configure the ESP32 WiFi to use power-save mode with listen intervals aligned to TDM cycles. Measure EMI impact empirically — the 25 cm separation between pod and nearest fluxgate may provide sufficient attenuation.

2. **RTCM buffering with gated WiFi**: If WiFi EMI is confirmed problematic:
   - Keep WiFi active during the 30ms EMI + 20ms comms windows (50ms total)
   - Buffer incoming RTCM3 data in a ring buffer on ESP32
   - Forward buffered RTCM3 to ZED-F9P during the next available slot
   - RTCM corrections are only needed every 1-10 seconds, so many missed windows are tolerable
   - Configure TCP keepalive interval to 30+ seconds to survive gating gaps
   - Monitor RTK fix quality; reconnect NTRIP if fix degrades to float/standalone

3. **Separate comms radio**: Use an external ESP32 or cellular modem (physically separated from fluxgates) dedicated to NTRIP. Forward RTCM3 to the main ESP32 via UART. Most complex but eliminates the conflict entirely.

---

### Finding 4: RECOMMENDED — Persist BNO055 Calibration Offsets in NVS

**Severity**: RECOMMENDED (affects data quality during warm-up)
**Agreement**: Unanimous (GPT-5.2 + independent analysis)
**Confidence**: 9/10

#### Problem

In IMUPLUS mode (accel + gyro fusion, magnetometer disabled):
- **Gyroscope**: Auto-calibrates within 5-10 seconds when the device is stationary. Relatively fast.
- **Accelerometer**: Auto-calibrates by experiencing different orientations/accelerations. Can take 30-60+ seconds of varied movement. Slower and less predictable.

Without persisted calibration, every power-on requires a warm-up period during which tilt correction accuracy is degraded.

#### Recommended Implementation

```c
// Calibration persistence flow
void bno055_init() {
    bno055_set_mode(IMUPLUS);

    // Try to restore saved calibration
    if (nvs_has_key("bno055_cal")) {
        uint8_t cal_data[22];
        nvs_read("bno055_cal", cal_data, 22);
        bno055_write_calibration(cal_data);
        calibration_state = CAL_RESTORED;
    } else {
        calibration_state = CAL_UNCALIBRATED;
    }
}

void bno055_update() {
    uint8_t sys_cal, gyro_cal, accel_cal, mag_cal;
    bno055_get_calibration_status(&sys_cal, &gyro_cal, &accel_cal, &mag_cal);

    // In IMUPLUS, only gyro and accel matter
    if (gyro_cal >= 2 && accel_cal >= 2) {
        calibration_state = CAL_ADEQUATE;
        tilt_correction_enabled = true;
    }

    if (gyro_cal == 3 && accel_cal == 3) {
        // Fully calibrated — save offsets
        if (calibration_state != CAL_SAVED) {
            uint8_t cal_data[22];
            bno055_read_calibration(cal_data);
            nvs_write("bno055_cal", cal_data, 22);
            calibration_state = CAL_SAVED;
        }
    }
}
```

#### Field Procedure

Add to the operator startup checklist:
1. Power on instrument
2. Hold stationary for 5 seconds (gyro calibration)
3. Slowly tilt instrument in multiple directions (accel calibration)
4. Wait for "Tilt Correction Active" indicator (LED or display)
5. Begin survey

After first full calibration, subsequent startups with restored offsets should reach CAL_ADEQUATE in under 5 seconds.

---

### Finding 5: RECOMMENDED — I2C Bus Recovery and Pod Health Monitoring

**Severity**: RECOMMENDED (affects field reliability)
**Agreement**: Unanimous (GPT-5.2 + independent analysis)
**Confidence**: 8/10

#### Problem

PCA9615 differential I2C over 1-2m Cat5 STP cable at 100 kHz is a known-workable configuration, but field conditions (cable strain, connector moisture, ESD events) can cause:
- SDA stuck low (most common I2C fault)
- Bus lockup from unfinished transactions (power glitch mid-transfer)
- Individual sensor hangs that block the entire bus

#### Recommended Implementation

**A. Stuck-Bus Recovery (firmware)**

```c
bool i2c_bus_recover(int sda_pin, int scl_pin) {
    // Detect stuck bus
    if (gpio_get_level(sda_pin) == 0) {
        // SDA stuck low — toggle SCL up to 9 times
        gpio_set_direction(scl_pin, GPIO_MODE_OUTPUT);
        for (int i = 0; i < 9; i++) {
            gpio_set_level(scl_pin, 1);
            delay_us(5);
            gpio_set_level(scl_pin, 0);
            delay_us(5);
            if (gpio_get_level(sda_pin) == 1) break;
        }
        // Generate STOP condition
        gpio_set_level(sda_pin, 0);
        gpio_set_level(scl_pin, 1);
        delay_us(5);
        gpio_set_level(sda_pin, 1);
        delay_us(5);

        // Re-initialize I2C peripheral
        i2c_driver_delete(I2C_NUM_1);
        i2c_driver_install(I2C_NUM_1, ...);
        return true;
    }
    return false;
}
```

**B. Hardware Pod Power-Cycle Capability**

Add a P-channel MOSFET (or load switch IC like TPS22918) on the host side, controlled by a GPIO, to power-cycle the entire pod:

```
ESP32 GPIO --> Gate driver --> P-FET --> Pod VCC (M8 pin 1)
```

This enables recovery from hard sensor lockups that survive bus recovery. Typical power-cycle sequence: disable for 100ms, re-enable, wait 50ms, re-scan I2C.

**C. Per-Device Health Monitoring**

Implement separate tracking for each sensor:

```c
typedef struct {
    bool detected;          // Found during initial scan
    bool healthy;           // Responding to recent pings
    uint32_t nack_count;    // Cumulative NACK errors
    uint32_t timeout_count; // Cumulative timeout errors
    uint32_t last_ok_ms;    // Timestamp of last successful read
    uint8_t state;          // ACTIVE, DEGRADED, DISABLED
} sensor_health_t;

sensor_health_t pod_sensors[4]; // ZED-F9P, BNO055, BMP390, DS3231
```

**D. Pullup Resistor Validation**

The PCA9615 pod-side output drives standard I2C. With 4 sensors + ESD protection (~5 pF each = 20 pF total) + PCB traces (~10 pF), total bus capacitance is approximately 30-40 pF. Standard 4.7 kOhm pullups are appropriate. Do NOT add pullups on the differential (cable) side — PCA9615 handles that internally.

---

## Additional Improvements Identified

### 6. Use UBX Binary Protocol Instead of NMEA

Even if ZED-F9P remains on I2C (not recommended), switch from NMEA to UBX binary protocol:
- UBX NAV-PVT message: 92 bytes per fix (contains lat, lon, alt, fix quality, PDOP, num satellites)
- Equivalent NMEA GGA+RMC: ~350-500 bytes per fix
- Bandwidth savings: approximately 70%
- No string parsing required — direct struct mapping

### 7. Add BMP390 Altitude Cross-Check

Use barometric altitude as a sanity check against GPS altitude:
- If |baro_altitude - gps_altitude| > 10m, flag potential GPS multipath or poor fix
- Temperature reading from BMP390 useful for thermal drift compensation of fluxgate/LM2917 signal chain
- Log both altitudes for post-processing altitude fusion

### 8. Diagnostic Counters for Post-Survey Analysis

Log the following counters to SD card CSV and/or MQTT telemetry:
- I2C NACK count per sensor per survey
- Bus recovery events (count and timestamps)
- Pod power-cycle events
- BNO055 calibration state transitions
- RTK fix quality transitions (e.g., fixed --> float --> standalone)
- NTRIP connection state (connected, reconnecting, failed)
- WiFi RSSI during comms windows

### 9. Separate Pod Detection from Pod Health

Implement a two-phase sensor management approach:

```
PHASE 1 (Boot): I2C scan --> detect which sensors are present
PHASE 2 (Runtime): Periodic health pings (every 1-5 seconds)
  - If sensor stops responding: increment error counter
  - After 3 consecutive failures: mark DEGRADED, attempt bus recovery
  - After recovery failure: mark DISABLED, attempt pod power-cycle
  - After power-cycle: re-scan, attempt re-detection
```

---

## Summary of Required Actions

| # | Finding | Severity | Action Required |
|---|---------|----------|-----------------|
| 1 | Tilt correction formula inverted | **CRITICAL** | Change multiply to divide; prefer quaternion-based projection |
| 2 | ZED-F9P I2C bandwidth insufficient | **HIGH** | Move ZED-F9P to UART using spare Cat5 pair |
| 3 | NTRIP/WiFi gating conflict | **MEDIUM** | Implement RTCM buffering + quiet-window scheduling |
| 4 | BNO055 calibration not persisted | **RECOMMENDED** | Store/restore calibration in NVS; add calibration state machine |
| 5 | No I2C bus recovery mechanism | **RECOMMENDED** | Add stuck-bus recovery + hardware pod power-cycle + per-device health |

---

## Responses to Original Concerns

### Concern 1: BNO055 Calibration Time

**Answer**: In IMUPLUS mode, gyro calibrates in 5-10 seconds (stationary), accel takes 30-60+ seconds (requires varied orientations). **Store calibration offsets in ESP32 NVS** and restore on boot to reduce warm-up to under 5 seconds. Implement a calibration state machine that gates tilt correction on calibration confidence level.

### Concern 2: ZED-F9P I2C Bandwidth

**Answer**: At 100 kHz, bandwidth is technically sufficient (~5 KB/s used of ~10 KB/s theoretical) but leaves insufficient margin for real-world factors (clock stretching, bus contention, burst RTCM data). **Move to UART** using the spare Cat5 pair. This is the industry standard for RTK integration. If constrained to I2C, use UBX binary only and reduce to 5 Hz.

### Concern 3: NTRIP Over WiFi with TDM Gating

**Answer**: RTCM corrections are only needed every 1-10 seconds, so brief WiFi outages (50-80ms per TDM cycle) are tolerable **if** TCP keepalive is configured appropriately (30+ seconds). Use an RTCM ring buffer on ESP32. Prefer quiet-window scheduling over hard WiFi gating. Monitor RTK fix quality as the ground truth indicator of NTRIP health.

### Concern 4: Tilt Correction Formula

**Answer**: The proposed formula `gradient * cos(pitch) * cos(roll)` is **wrong** — it should be `gradient / (cos(pitch) * cos(roll))` for first-order correction. The better approach is quaternion-based rotation projection onto the true vertical axis. Heading relative to Earth's field is NOT needed for gradient correction, but IMU-to-sensor-baseline frame alignment must be calibrated.

### Concern 5: PCA9615 Reliability

**Answer**: PCA9615 over 1-2m Cat5 STP at 100 kHz is a proven configuration with no known systematic issues. Required mitigations: (a) I2C stuck-bus recovery in firmware (9 SCL pulses + STOP), (b) hardware pod power-cycle capability (MOSFET switch), (c) ESD protection at connector (TPD4E05U06, 5 pF is acceptable at 100 kHz), (d) shield grounded at host end only to prevent ground loops.

---

## Model Availability Note

The gemini-3-pro-preview model was requested as the "against" stance evaluator but was unavailable throughout all retry attempts due to API quota exhaustion (HTTP 429 RESOURCE_EXHAUSTED). The consensus was formed using GPT-5.2 (for stance) and Claude Opus 4.6 independent analysis as the critical counter-perspective. Despite the missing adversarial model, confidence in the findings is high (8/10) because both available analyses independently converged on the same five issues.

---

## References

- `/research/multi-sensor-architecture/sensor-pod-design.md` — Pod physical design, PCA9615 bus, cable specifications
- `/research/multi-sensor-architecture/sensor-component-specs.md` — Sensor datasheets, I2C addresses, power budgets
- u-blox ZED-F9P Integration Manual (F9 HPS) — I2C/UART interface specifications
- Bosch BNO055 Datasheet (BST-BNO055-DS000) — Calibration procedure, IMUPLUS mode
- NXP PCA9615 Datasheet — Differential I2C operation, termination requirements
