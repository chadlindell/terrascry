# PAL Consensus Validation: I6 -- HIRT Firmware Sensor Pod Support

**Date:** 2026-02-18
**Task:** I6 -- Add sensor pod support to HIRT firmware
**Consensus Tool:** mcp__pal__consensus
**Models Consulted:**
- openai/gpt-5.2 (FOR stance) -- **Responded successfully**
- gemini-3-pro-preview (AGAINST stance) -- **Unavailable** (429 RESOURCE_EXHAUSTED, daily free-tier API quota exceeded after 12+ retry attempts)
**Supplemental Critical Perspective:** Claude Opus 4.6 independent analysis (filling the adversarial gap)

**Reference Document:** `sensor-pod-integration.md`

---

## Proposal Under Evaluation

Add sensor pod support to the HIRT (Hybrid Inductive-Resistivity Tomography) instrument firmware, enabling the shared sensor pod (ZED-F9P RTK GPS, BNO055 IMU, BMP390 barometer, DS3231 RTC) to connect to the HIRT base hub for GPS-based probe position recording.

### Implementation Plan Summary

1. Add M8 4-pin female connector to HIRT base hub enclosure
2. Mount PCA9615 breakout inside base hub, wire to ESP32 I2C Bus 1 (GPIO 16/17)
3. Firmware: I2C bus scan at startup to detect pod
4. If pod detected: enable GPS position recording mode
5. GPS workflow: operator places pod at probe location, presses button, records RTK position, stores in metadata CSV
6. Metadata CSV format: `probe_id, timestamp, lat, lon, alt_m, fix_type, hdop, satellites`
7. Graceful degradation: if pod absent, warn user, accept manual position entry from survey plan

### HIRT-Specific Concerns Addressed

1. I2C Bus 0 (AD7124-8 ADC, ADXL345 inclinometer) vs Bus 1 (pod sensors) -- architecture conflicts?
2. GPS position recording between measurements, not during -- timing concerns?
3. BNO055 IMU configuration: IMUPLUS mode for both instruments?
4. Power budget: 3.3V LDO headroom for pod?
5. Data integration: timestamp format consistency for inversion pipeline?

---

## Consensus Results

### Overall Verdict: PROCEED WITH IMPLEMENTATION

**Confidence Score: 8/10**

The sensor pod integration plan is architecturally sound and technically feasible. The dual I2C bus separation is the correct approach, timing isolation is well-designed, and the graceful degradation model is robust. Seven specific concerns must be addressed before field deployment.

---

## Points of Agreement (All Perspectives)

### 1. Dual I2C Bus Architecture Is Sound

The ESP32 provides two independent hardware I2C controllers. Using Bus 0 for HIRT's native sensors (AD7124-8 ADC, ADXL345 inclinometer) and Bus 1 for the sensor pod eliminates any possibility of I2C address conflicts or bus contention. The PCA9615 differential buffer is a proven solution for 1-2 m cable runs over Cat5 STP.

### 2. Timing Isolation Is Well-Designed

GPS position recording occurs between HIRT measurement sequences (seconds per electrode combination), not during active measurements. This creates natural timing windows with no contention for CPU, I2C bus, or power resources. The button-press-triggered discrete GPS read is fundamentally different from Pathfinder's continuous real-time streaming.

### 3. Graceful Degradation Model Is Correct

The plan for HIRT to function normally without the pod (manual position entry from survey plan) is essential. The pod is a shared resource between HIRT and Pathfinder, so it will not always be available. Startup-based detection with user warning is the right baseline.

### 4. CSV Data Integration Is Straightforward

ISO 8601 timestamps in both the position CSV and measurement data enable straightforward merging. The `probe_id` field maps directly to electrode identifiers in the inversion pipeline. SimPEG reads electrode coordinates first to build the mesh, then overlays measurement data.

### 5. BNO055 IMUPLUS Mode Is Acceptable

While HIRT only uses the IMU for tilt verification during position recording (not real-time correction as on Pathfinder), keeping IMUPLUS mode simplifies the shared pod firmware. The BNO055 draws approximately 12 mA in IMUPLUS -- negligible during HIRT's intermittent position recording events.

---

## Critical Concerns Requiring Resolution

### Concern 1: Hot-Plug Protection (HARDWARE)

**Risk:** No TVS diode protection on M8 connector I2C lines. Disconnecting the pod during an active I2C transaction could cause bus lockup or damage the PCA9615/ESP32.

**Recommendation:** Add bidirectional TVS diodes (e.g., PESD5V0S2BT) on SDA_D+ and SCL_D- lines at the M8 connector. This is a minor BOM addition (approximately $0.50) with significant reliability benefit.

### Concern 2: Startup-Only Pod Detection (FIRMWARE)

**Risk:** If the pod is connected after boot, it remains undetected until the next power cycle. Field operators may power on HIRT first, then realize they need GPS.

**Recommendation:** Implement periodic re-scan of I2C Bus 1 (e.g., every 30 seconds when idle) or add a "Detect Pod" menu option in the HIRT controller UI. A GPIO interrupt from the M8 connector's power sense line is also viable but adds hardware complexity.

### Concern 3: Terminology Inconsistency in CSV Format (DOCUMENTATION)

**Risk:** The integration document uses `fix_quality` in some places and `fix_type` in the plan. This will cause confusion in the inversion pipeline.

**Recommendation:** Standardize on `fix_type` with enumerated values: `NO_FIX`, `AUTONOMOUS`, `DGPS`, `RTK_FLOAT`, `RTK_FIXED`. Update both `sensor-pod-integration.md` and the firmware CSV writer to use identical column names.

### Concern 4: NTRIP Corrections Delivery (OPERATIONAL)

**Risk:** The ZED-F9P requires RTCM3 corrections from an NTRIP caster for cm-accuracy RTK. The plan does not specify how HIRT provides internet connectivity in remote field sites.

**Recommendation:** Document three fallback strategies:
1. **Primary:** Operator phone Wi-Fi tethering to ESP32, which forwards NTRIP to ZED-F9P
2. **Secondary:** Portable base station (second ZED-F9P on tripod) broadcasting local corrections via UHF or LoRa
3. **Tertiary:** Log raw GNSS observations on the ZED-F9P for post-processed kinematic (PPK) correction after the survey

### Concern 5: Power Budget Verification (HARDWARE)

**Risk:** Pod peak current is approximately 150-200 mA (ZED-F9P dominates at approximately 70 mA tracking, approximately 130 mA acquisition). HIRT's 3.3V LDO headroom is not documented.

**Recommendation:** Measure HIRT's existing 3.3V rail current consumption during measurement sequences. Verify LDO output capability exceeds existing load + 200 mA pod overhead. If insufficient, add a dedicated LDO for the pod powered from the battery rail. Note that GPS recording does not overlap with high-current measurement phases, which reduces the concern.

### Concern 6: PCA9615 Single Point of Failure (HARDWARE)

**Risk:** If the PCA9615 differential I2C buffer fails (either hub-side or pod-side), the entire sensor pod is lost with no partial recovery possible.

**Recommendation:** Accept this risk for now -- the PCA9615 is a well-proven IC and field replacement is straightforward (swappable breakout board). Add the PCA9615 breakout to the HIRT field spares kit. Future revision could add a bypass jumper for direct (short-cable) I2C connection without the differential buffer.

### Concern 7: Cable Strain in Field Conditions (MECHANICAL)

**Risk:** The M8 connector is rated IP67 when mated, but field conditions (muddy ground, tripping hazards, cable pulls) stress the connector and cable.

**Recommendation:** Add a cable strain relief (cable gland or P-clip) adjacent to the M8 connector on the base hub enclosure. Use a coiled/retractable cable if the pod will be frequently moved during the position recording workflow.

---

## Recommended Firmware Architecture

GPT-5.2 proposed a clean abstraction that both perspectives endorsed:

### SensorPod Abstraction Layer

```c
// sensor_pod.h -- shared between HIRT and Pathfinder firmware

typedef struct {
    bool present;
    bool gps_has_fix;
    uint8_t fix_type;       // 0=none, 1=autonomous, 4=RTK_float, 5=RTK_fixed
    float hdop;
    uint8_t satellites;
    double latitude;
    double longitude;
    float altitude_m;
    float roll_deg;         // from BNO055
    float pitch_deg;        // from BNO055
    float pressure_hpa;     // from BMP390
    uint32_t rtc_unix;      // from DS3231
    char firmware_version[16];
} SensorPodState;

// API methods
bool     pod_init(TwoWire *bus);         // Initialize on specified I2C bus
bool     pod_detect(void);               // Scan for pod presence
bool     pod_is_present(void);           // Cached presence flag
void     pod_read_position(SensorPodState *state);   // One-shot GPS read
void     pod_read_orientation(SensorPodState *state); // One-shot IMU read
void     pod_sleep(void);                // Power-save when not in use
```

### State Machine for HIRT Position Recording

```
IDLE ──[button press]──> POSITION_RECORDING
    │                         │
    │                    [read GPS + IMU]
    │                         │
    │                    [quality gate: RTK_FIXED?]
    │                     /          \
    │                   yes           no
    │                    │             │
    │               [store CSV]   [warn user, retry?]
    │                    │             │
    │                    v             v
    └──────────── READY_FOR_MEASUREMENT
                         │
                    [run ERT/MIT sequence]
                         │
                    DATA_STORE ──> IDLE
```

### Position Quality Gate

For critical surveys (forensic, UXO), the firmware should enforce a minimum GPS quality threshold before accepting a position record:

| Fix Type | HDOP Threshold | Action |
|----------|---------------|--------|
| RTK_FIXED | < 2.0 | Accept, mark as high-confidence |
| RTK_FLOAT | < 3.0 | Accept with warning, mark as moderate-confidence |
| AUTONOMOUS | any | Warn user, allow override, mark as low-confidence |
| NO_FIX | -- | Reject, require manual entry or retry |

### Enhanced CSV Format

```csv
# HIRT Probe Position Log
# Generated: 2026-02-18T14:30:00Z
# Instrument: HIRT
# Pod Firmware: v1.2.3
# RTK Source: UNAVCO CORS / local base
probe_id,timestamp,latitude,longitude,altitude_m,fix_type,hdop,satellites,pitch_deg,roll_deg,confidence
P1,2026-02-18T14:30:12Z,51.2345678,-1.4567890,102.34,RTK_FIXED,0.8,14,0.3,0.2,high
P2,2026-02-18T14:31:45Z,51.2345712,-1.4567823,102.31,RTK_FIXED,0.7,15,0.5,0.1,high
P3,2026-02-18T14:33:22Z,51.2345690,-1.4567756,102.35,RTK_FLOAT,1.4,13,0.2,0.4,moderate
```

Additions over the original format:
- `pitch_deg`, `roll_deg` -- pod orientation at time of recording (verify level placement)
- `confidence` -- derived from fix_type + HDOP quality gate
- Header includes pod firmware version and RTK correction source

---

## Additional Recommendations from GPT-5.2

### Data Integrity

- Add sequence numbers and optional CRC to position records for data integrity verification
- Implement monotonic timestamp validation (reject out-of-order records)

### Testing Additions

- **Fault injection test:** Disconnect pod during I2C read, verify bus recovery
- **Time sync test:** Validate DS3231 RTC vs ESP32 system time, ensure cross-device alignment within 20 ms
- **Range test:** Verify PCA9615 differential I2C over 1 m, 1.5 m, and 2 m Cat5 STP cables
- **Power test:** Measure 3.3V rail droop during simultaneous GPS acquisition + ERT measurement

### Cross-Instrument Coordinate Registration

When both Pathfinder and HIRT use the same sensor pod (sequentially), their GPS data shares the same ZED-F9P receiver and NTRIP correction stream. This eliminates coordinate transformation errors between instruments -- a significant advantage over using separate GPS units.

---

## Model Availability Note

The gemini-3-pro-preview model was requested as the AGAINST stance evaluator but was unavailable due to Gemini API free-tier quota exhaustion (HTTP 429 RESOURCE_EXHAUSTED). The tool retried 12+ times before the consensus proceeded with GPT-5.2's response and Claude Opus 4.6's independent critical analysis filling the adversarial perspective.

For a complete two-model consensus with the intended adversarial balance, re-run this validation when the gemini-3-pro-preview quota resets, or substitute an alternative model (e.g., gemini-2.5-pro).

---

## Action Items

| # | Item | Priority | Owner | Status |
|---|------|----------|-------|--------|
| 1 | Add TVS diodes to M8 I2C lines | High | Electronics | Pending |
| 2 | Implement periodic pod re-scan or UI trigger | Medium | Firmware | Pending |
| 3 | Standardize CSV column names (`fix_type`) | High | Documentation + Firmware | Pending |
| 4 | Document NTRIP delivery strategy (3 fallback levels) | Medium | Operations | Pending |
| 5 | Measure 3.3V LDO headroom, verify pod power budget | High | Electronics | Pending |
| 6 | Add PCA9615 breakout to field spares kit | Low | Logistics | Pending |
| 7 | Add cable strain relief to M8 connector mounting | Medium | Mechanical | Pending |
| 8 | Implement SensorPod abstraction layer | High | Firmware | Pending |
| 9 | Implement position quality gate | Medium | Firmware | Pending |
| 10 | Add pitch/roll/confidence columns to CSV format | Low | Firmware + Docs | Pending |
