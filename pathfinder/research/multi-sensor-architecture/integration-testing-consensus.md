# I10: Integration Testing -- Multi-Model Consensus Validation

**Task:** I10 Integration Testing
**Date:** 2026-02-18
**Consensus Tool:** PAL `consensus`
**Models Consulted:** openai/gpt-5.2 (neutral), gemini-3-pro-preview (neutral -- unavailable, quota exhausted)
**Orchestrating Model:** Claude Opus 4.6
**Overall Consensus Confidence:** 8/10

---

## 1. Executive Summary

The integration testing plan for the multi-instrument geophysical survey system (Pathfinder + HIRT + GeoSim + shared sensor pod) was evaluated through a multi-model consensus process. **The plan is structurally solid** with good coverage of core integration boundaries, but has several critical gaps in pass criteria realism, fault tolerance testing, time synchronization validation, and automated data quality verification. Nine additional test cases are recommended.

**Note:** The gemini-3-pro-preview model was unavailable due to API quota exhaustion (429 RESOURCE_EXHAUSTED). The synthesis below is based on openai/gpt-5.2 (confidence 8/10) and independent analysis by the orchestrating model (Claude Opus 4.6).

---

## 2. Models Consulted

| Model | Stance | Status | Confidence |
|-------|--------|--------|------------|
| openai/gpt-5.2 | Neutral | Success | 8/10 |
| gemini-3-pro-preview | Neutral | Error (429 quota) | N/A |
| Claude Opus 4.6 (orchestrator) | Independent analysis | Success | 8/10 |

---

## 3. Evaluation of Original Test Plan (T1--T5)

### 3.1 Strengths

1. **Good architecture boundary coverage** -- Tests map well to the actual system integration points:
   - Pathfinder <-> MQTT <-> GeoSim (T1, T4)
   - Shared sensor pod interoperability (T2, T3)
   - End-to-end cross-instrument registration (T5)
2. **Quantitative pass criteria** -- 99.9% delivery, <10ms latency, <5cm GPS, <2m anomaly position are measurable
3. **Hot-swap testing (T2)** covers a critical field reliability scenario
4. **LWT testing (T1)** validates disconnect detection, important for field operations
5. **End-to-end workflow (T5)** validates the full operational cycle: survey -> detect -> investigate

### 3.2 Issues Identified in Original Tests

#### T1: Pathfinder Standalone Streaming

| Issue | Severity | Detail |
|-------|----------|--------|
| Latency criterion unrealistic | HIGH | <10ms end-to-end is unrealistic over WiFi + Linux scheduling + MQTT serialization, especially when camera/LiDAR payloads share the link. Feasible to measure but likely to fail for non-critical channels. |
| Global latency target inappropriate | MEDIUM | Different sensor channels have fundamentally different latency requirements; a single target hides this. |
| Missing QoS specification | MEDIUM | No per-topic QoS level defined (QoS 0 for high-rate best-effort vs QoS 1 for alerts/status). |

**Recommendation:** Replace the single `<10ms` global latency with per-topic SLAs:
- Magnetics/EMI: p95 < 50ms, p99 < 100ms
- Status/heartbeat: p95 < 250ms
- Camera/LiDAR: best-effort, track throughput instead
- Anomaly alerts: QoS 1, p95 < 500ms

#### T2: Sensor Pod Hot-Swap

| Issue | Severity | Detail |
|-------|----------|--------|
| No rapid cycling test | LOW | Should test rapid connect/disconnect cycling (10x in 60s) to catch resource leaks. |
| Missing I2C bus recovery | MEDIUM | Hot-swap on I2C can leave the bus in a hung state; test should verify bus recovery. |

#### T3: HIRT Probe GPS Registration

| Issue | Severity | Detail |
|-------|----------|--------|
| Fix type inconsistency | HIGH | Test specifies "fix type 6" but pass criteria says "fix quality >= 5". These NMEA GGA fix quality values should be consistent. Type 4 = RTK fixed, Type 5 = RTK float. Clarify which is required. |
| Ground truth methodology undefined | HIGH | <5cm accuracy requires surveyed reference markers, not phone-based coordinates. |
| BMP390 altitude untested | MEDIUM | Barometric altitude from BMP390 in the sensor pod is not validated anywhere. |

**Recommendation:** Define ground truth as surveyed markers with known WGS84 coordinates. Use a total station or post-processed RTK baseline to establish marker positions to <1cm.

#### T4: GeoSim Anomaly Detection

| Issue | Severity | Detail |
|-------|----------|--------|
| No false positive testing | HIGH | Only tests detection over a known target. Must also verify NO false alerts over blank terrain. |
| No sensitivity characterization | MEDIUM | Only tests one target type/depth. Need multiple scenarios to understand detection envelope. |
| Traceability gap | MEDIUM | No requirement to trace from alert back to raw sensor data for replay/verification. |

#### T5: End-to-End Workflow

| Issue | Severity | Detail |
|-------|----------|--------|
| Coordinate frame validation vague | MEDIUM | "Same coordinate frame" needs specific definition: WGS84? Local ENU? What transform chain? |
| No time alignment requirement | HIGH | Pathfinder and HIRT data must be temporally aligned for correlation; no timing requirement stated. |

---

## 4. Critical Gaps Identified (Consensus)

All evaluators agreed on the following critical gaps:

### 4.1 No Time Synchronization Testing

The DS3231 RTC exists in the sensor pod but is never tested as a system requirement. Cross-instrument timing alignment is essential for correlating Pathfinder survey data with HIRT measurements. GPS PPS, DS3231 RTC, and Jetson NTP must all be validated against each other.

### 4.2 No Fault Injection / Negative Testing

No tests cover what happens when things go wrong in the field:
- WiFi drops mid-survey
- Mosquitto broker crashes/restarts
- Power failure during data logging
- SD card fills up
- Malformed MQTT messages
- GPS jamming or signal loss

### 4.3 No Stress / Throughput Testing

No test verifies the system under worst-case load: all 8 fluxgates + EMI coil + RTK GPS + IMU + IR temp + LiDAR + camera streaming simultaneously. The Jetson must handle aggregate throughput without dropping critical telemetry.

### 4.4 No Data Schema / Contract Validation

No test verifies that MQTT payloads match expected schemas, that CSV metadata has correct column formats, or that firmware version mismatches are handled gracefully.

### 4.5 No Automated Data Quality Validation

Current tests verify that data is "publishing" but not that it contains physically reasonable values. Need automated sanity checks: range validation, monotonic timestamps, cross-sensor consistency checks (GPS speed vs IMU acceleration, baro altitude vs GNSS altitude trend, magnetics stability when stationary).

### 4.6 No EMI / Interference Matrix Testing

Fluxgate magnetometers are extremely sensitive. No test verifies that Pathfinder's own electronics (WiFi radio, LiDAR, camera, power converters) do not corrupt magnetometer readings during simultaneous operation.

---

## 5. Recommended Additional Test Cases

### T6: Network / Broker Fault Injection

**Purpose:** Validate system resilience to communication failures matching field reality.

| Step | Action | Verify |
|------|--------|--------|
| 1 | Drop WiFi for 30s during active survey | Pathfinder buffers to SD; no data loss |
| 2 | Drop WiFi for 120s | LWT published; MQTT auto-reconnect on restore |
| 3 | Kill Mosquitto process during streaming | Client reconnect; no data corruption |
| 4 | Restart Mosquitto | Retained messages restored; clients re-subscribe |
| 5 | Fill SD card to 95%, then 100% | Graceful warning; oldest data rotation or clean stop |

**Pass criteria:** Zero data corruption in any scenario; auto-reconnect within 10s; LWT published within 1.5x keepalive interval; SD card overflow handled without crash.

### T7: Time Synchronization Validation

**Purpose:** Verify cross-device timing alignment for data fusion.

| Step | Action | Verify |
|------|--------|--------|
| 1 | Compare DS3231 time vs GPS PPS | Drift < 2ms over 1 hour |
| 2 | Compare Jetson system time vs NTP reference | Offset < 50ms |
| 3 | Verify timestamps are monotonically increasing | No reversals in any channel |
| 4 | Correlate Pathfinder and HIRT timestamps for same GPS fix event | Cross-device alignment < 20ms |
| 5 | Power-cycle DS3231, verify time recovery from GPS | RTC re-syncs within 60s of GPS fix |

**Pass criteria:** All timestamps monotonic; cross-device alignment < 20ms for fusion-critical channels; DS3231 drift < 2ppm.

### T8: Throughput / Backpressure Stress Test

**Purpose:** Verify system stability under maximum sensor load.

| Step | Action | Verify |
|------|--------|--------|
| 1 | Enable all sensors at full rate for 4+ hours | No memory leaks (RSS stable) |
| 2 | Monitor Jetson CPU, RAM, disk I/O, temperature | CPU < 80%, no thermal throttling |
| 3 | Add camera + LiDAR worst-case payloads | Critical telemetry (magnetics) not starved |
| 4 | Monitor MQTT broker queue depths | No unbounded growth |
| 5 | Verify SD card write rates keep pace | No write buffer overflow |

**Pass criteria:** 4-hour continuous run with all sensors; Jetson CPU < 80%; no thermal throttling; zero OOM events; magnetics latency SLA maintained throughout.

### T9: EMI / Interference Matrix Validation

**Purpose:** Verify fluxgate magnetometer data integrity during multi-sensor operation.

| Step | Action | Verify |
|------|--------|--------|
| 1 | Record fluxgate baseline (all other sensors OFF) in magnetically quiet environment | Establish noise floor |
| 2 | Enable WiFi radio only | Noise increase < 0.5 nT |
| 3 | Enable LiDAR only | Noise increase < 0.5 nT |
| 4 | Enable camera only | Noise increase < 0.5 nT |
| 5 | Enable all sensors simultaneously | Total noise increase < 1.0 nT vs baseline |
| 6 | Enable EMI coil with TDM sequencing | Verify TDM windows clean for fluxgate reads |

**Pass criteria:** Fluxgate noise floor degradation < 1.0 nT with all subsystems active; no coherent interference patterns; TDM timing verified with logic analyzer.

### T10: Data Schema and Integrity Validation

**Purpose:** Verify data formats, detect transport errors, and handle version mismatches.

| Step | Action | Verify |
|------|--------|--------|
| 1 | Validate all MQTT payloads against defined schema | 100% compliance |
| 2 | Inject malformed MQTT messages into broker | GeoSim rejects with clear error log; no crash |
| 3 | Add sequence numbers to payloads; verify at receiver | Zero gaps, duplicates, or reordering under normal operation |
| 4 | Verify CSV metadata column headers and data types | Match specification exactly |
| 5 | Test with mismatched firmware versions (old Pathfinder, new GeoSim) | Graceful degradation; clear version mismatch warning |

**Pass criteria:** Zero schema violations in normal operation; malformed data rejected without crash; sequence number gaps detected and logged at reconnect boundaries.

### T11: WiFi Range / Degradation Profiling

**Purpose:** Characterize MQTT reliability as a function of distance and obstructions.

| Distance | LOS | Non-LOS | Measure |
|-----------|-----|---------|---------|
| 10m | Test | Test | RSSI, message delivery %, latency p95 |
| 30m | Test | Test | RSSI, message delivery %, latency p95 |
| 60m | Test | -- | RSSI, message delivery %, latency p95 |
| 100m | Test | -- | RSSI, message delivery %, latency p95 |

**Pass criteria:** Define operational envelope: e.g., 99.9% delivery at 30m LOS, 99% at 60m LOS. Document degradation curve for field planning.

### T12: Multi-Client Broker Simultaneous Operation

**Purpose:** Verify Mosquitto handles Pathfinder + HIRT + monitoring clients concurrently.

| Step | Action | Verify |
|------|--------|--------|
| 1 | Connect Pathfinder streaming at full rate | Baseline broker metrics |
| 2 | Add HIRT client publishing measurement sequences | No impact on Pathfinder delivery |
| 3 | Add monitoring laptop subscribing to all topics | No backpressure on publishers |
| 4 | HIRT runs high-frequency sequence while Pathfinder streams | Neither client starved |
| 5 | Verify client ID uniqueness and topic namespace isolation | No cross-contamination |

**Pass criteria:** Zero message cross-contamination; no client starvation; broker CPU < 50% on Jetson; unique client IDs enforced.

### T13: Power Interruption Recovery

**Purpose:** Verify data integrity and system recovery after unplanned power loss.

| Step | Action | Verify |
|------|--------|--------|
| 1 | Kill power to Pathfinder during active survey | SD card data intact up to last flush |
| 2 | Restore power | Boot, auto-connect to broker, resume logging |
| 3 | Kill power to Jetson during data logging | Filesystem integrity (journaled); no DB corruption |
| 4 | Restore Jetson | Mosquitto restarts; clients reconnect; data logging resumes |
| 5 | Kill power to sensor pod during GPS fix | Pathfinder detects loss, falls back to NEO-6M |

**Pass criteria:** Zero data corruption from power loss; auto-recovery within 30s of power restore; SD card filesystem intact.

### T14: Barometric Altitude Cross-Validation (BMP390)

**Purpose:** Validate BMP390 altitude data from sensor pod against GNSS altitude.

| Step | Action | Verify |
|------|--------|--------|
| 1 | Record BMP390 pressure + temperature at known elevation | Altitude matches within +/-5m of reference |
| 2 | Walk survey with elevation changes (>10m vertical) | BMP390 altitude trend matches GNSS altitude trend |
| 3 | Verify BMP390 data published over MQTT | Topic exists, publishes at expected rate |
| 4 | Compare BMP390 relative altitude precision over short intervals | Relative precision < 0.5m over 10 minutes |

**Pass criteria:** BMP390 altitude trend correlates with GNSS altitude (r > 0.9); relative precision < 0.5m; data published at 1 Hz minimum.

---

## 6. Concerns Resolution

### Concern 1: Test Environment / Buried Targets

**Consensus recommendation:**
- Use a small test lane (e.g., 20m x 5m) with known targets at multiple depths: 0.2m, 0.5m, and 1.0m
- Include ferrous targets (rebar, iron pipe), non-ferrous targets (copper pipe), and a blank control zone
- If burial is impractical for initial testing, start with above-ground surrogates (targets on surface) to validate detection + registration logic, then graduate to buried targets
- Mark target positions with surveyed reference stakes (<1cm accuracy)

### Concern 2: GPS Accuracy / RTK Without Cellular

**Consensus recommendation:**
- Bring a portable RTK base station (second ZED-F9P on tripod with averaged/surveyed position) broadcasting corrections locally via WiFi or serial radio
- This eliminates NTRIP/cellular dependency entirely for testing
- Additionally, log raw GNSS observations (RAWX/SFRBX) on ZED-F9P for post-processed kinematic (PPK) as a "truth recovery" path if real-time RTK fails
- This is the standard approach in precision agriculture and surveying when cellular is unavailable

### Concern 3: MQTT Reliability / WiFi Range

**Consensus recommendation:**
- Define an explicit operational envelope by testing at 10m, 30m, 60m line-of-sight, plus one non-LOS (through vegetation/vehicle) case
- Record RSSI, retry rates, message gaps, and latency at each distance
- Specify per-topic QoS levels: QoS 0 for high-rate sensor data (best-effort), QoS 1 for anomaly alerts and status messages
- Document the degradation curve so field operators know the practical working radius

### Concern 4: Simultaneous Pathfinder + HIRT on One Broker

**Consensus recommendation:**
- Yes, this is feasible and standard for Mosquitto
- Must enforce: unique client IDs per device, per-device topic prefixes (e.g., `pathfinder/mag/...`, `hirt/ert/...`), and verify no namespace collision
- Test that HIRT measurement sequences (which may be bursty) do not starve Pathfinder continuous telemetry
- Monitor broker queue depths and CPU under simultaneous load (see T12)

### Concern 5: Physically Reasonable Values

**Consensus recommendation:**
- Implement automated "sanity + invariants" validation scripts that run post-collection or in real-time on GeoSim:
  - **Range checks:** Magnetic field 20,000--70,000 nT (Earth's field); temperature -40 to +85C; pressure 300--1100 hPa; GPS coordinates within expected bounding box
  - **Monotonic timestamps:** No time reversals in any channel
  - **Cross-sensor consistency:** GPS speed vs IMU-derived velocity; BMP390 altitude trend vs GNSS altitude trend; magnetics stability when stationary (std dev < threshold)
  - **Rate validation:** Each topic publishes within 10% of its expected rate
- These checks transform testing from "is it publishing?" to "is it publishing correct data?"

---

## 7. Revised Pass Criteria for Original Tests

### T1 (Revised)

| Metric | Original | Revised |
|--------|----------|---------|
| Message delivery | 99.9% over 1 hour | 99.9% for QoS 1 topics; 99% for QoS 0 topics over 1 hour |
| Latency | <10ms end-to-end | Magnetics: p95 < 50ms; Status: p95 < 250ms; Camera/LiDAR: track throughput |
| SD card backup | Same data as MQTT | Same data as MQTT; verify with sequence number comparison |
| LWT | Published on disconnect | Published within 1.5x keepalive interval |

### T3 (Revised)

| Metric | Original | Revised |
|--------|----------|---------|
| Fix type | Type 6 | Fix quality 4 (RTK fixed) or 5 (RTK float); document which is minimum |
| Position accuracy | <5cm | <5cm vs surveyed reference markers (not phone GPS) |
| Ground truth | Undefined | Surveyed markers with total station or PPK baseline |

### T4 (Revised -- add negative test)

| Metric | Original | Revised |
|--------|----------|---------|
| Detection | Within 2m of actual | Within 2m of actual; AND zero false positives over 50m of blank terrain |
| Traceability | Not specified | Alert links to raw data window for replay |

---

## 8. Test Execution Priority

Recommended execution order based on dependency and risk:

| Priority | Test | Rationale |
|----------|------|-----------|
| 1 | T1 (revised) | Foundation: validates basic data transport |
| 2 | T7 (time sync) | Critical for all data fusion; blocks T5 |
| 3 | T9 (EMI matrix) | Validates sensor data quality; blocks T4 |
| 4 | T2 (hot-swap) | Tests shared pod mechanism used by T3, T5 |
| 5 | T10 (schema/integrity) | Validates data contracts used everywhere |
| 6 | T11 (WiFi range) | Defines operational envelope for all field tests |
| 7 | T3 (revised) | GPS registration for HIRT |
| 8 | T8 (stress test) | Validates long-duration stability |
| 9 | T4 (revised) | Anomaly detection with negative testing |
| 10 | T12 (multi-client) | Validates simultaneous operation |
| 11 | T6 (fault injection) | Resilience testing |
| 12 | T13 (power recovery) | Field failure recovery |
| 13 | T14 (barometric) | Sensor pod altitude validation |
| 14 | T5 (end-to-end) | Capstone: requires all other tests passing |

---

## 9. Infrastructure Requirements

Before any testing can begin, the following must be in place:

1. **Test site** with surveyed reference markers and buried/surface targets
2. **Portable RTK base station** (second ZED-F9P + antenna + tripod) for NTRIP-independent corrections
3. **Record/replay harness** for MQTT streams (enables deterministic regression testing)
4. **Automated validation scripts** for data quality checks (range, timestamps, cross-sensor)
5. **MQTT topic contract document** defining schema, units, coordinate frame, and timestamp format for every topic
6. **Logic analyzer** for TDM timing verification (T9)

---

## 10. Long-Term Recommendations

1. **Build a record/replay harness** -- Record MQTT streams + raw sensor logs in a "golden run", then replay into GeoSim for deterministic anomaly detection testing. Without this, integration testing becomes unrepeatable "hero runs" that cannot scale with firmware evolution.

2. **Define explicit topic contracts** -- Schema + units + coordinate frame + timestamp definition for every MQTT topic. This reduces future regressions more effectively than adding one-off field tests.

3. **Automate regression suite** -- Tests T1, T7, T10 can be largely automated on the bench. Only T4, T5, T14 require field execution.

4. **Version the test plan** -- As firmware and software evolve, the test plan must evolve. Tie test case versions to firmware release milestones.

---

## Appendix: Raw Model Responses

### openai/gpt-5.2 (Neutral)

**Verdict:** Strong integration test outline with clear end-to-end coverage, but several pass criteria and validation methods are underspecified/unrealistic for field conditions and it needs additional fault-injection, time-sync/coordinate-frame, and data-quality tests to de-risk deployment.

**Confidence:** 8/10

**Key points:**
- Feasible overall; MQTT + Jetson edge compute + multi-instrument workflow is standard in field robotics/IoT
- <10ms latency unrealistic over WiFi + Linux; replace with per-channel p95/p99 SLAs
- <5cm GPS accuracy needs surveyed references, not phone coordinates
- Missing time sync, schema validation, fault injection, and automated data quality checks
- Recommended record/replay harness and portable base station for test independence
- Industry best practice: test transport reliability, data integrity, and domain validity as three distinct layers
- Proposed additional tests: T6 (fault injection), T7 (time sync), T8 (stress), T9 (schema compatibility), T10 (data integrity with sequence numbers/CRC)

### gemini-3-pro-preview (Neutral)

**Status:** UNAVAILABLE -- 429 RESOURCE_EXHAUSTED (Gemini API daily quota exceeded). Multiple retry attempts failed. This model's perspective was not obtained.

### Claude Opus 4.6 -- Orchestrator Independent Analysis

**Key points not covered by GPT-5.2:**
- EMI/interference matrix testing is critical for fluxgate magnetometers (Pathfinder's own electronics may corrupt readings)
- BMP390 barometric altitude is in the sensor pod but untested in any scenario
- Camera/LiDAR produce large data volumes; need explicit storage and bandwidth validation
- I2C bus recovery after hot-swap should be explicitly tested
- Power failure recovery is a distinct concern from network failure (SD card filesystem integrity)
- WiFi range profiling should include non-line-of-sight scenarios (vegetation, vehicles)
