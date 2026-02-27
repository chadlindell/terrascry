# I4: Pathfinder Firmware MQTT Streaming -- Multi-Model Consensus Validation

**Date:** 2026-02-18
**Task:** I4 -- Pathfinder Firmware MQTT Streaming
**Consensus Models:** openai/gpt-5.2 (FOR stance), gemini-3-pro-preview (AGAINST stance -- UNAVAILABLE, see note)
**Synthesiser:** Claude Opus 4.6

---

## Model Availability Note

- **openai/gpt-5.2** -- Successfully consulted. Confidence: 8/10. Provided comprehensive analysis.
- **gemini-3-pro-preview** -- UNAVAILABLE. Returned 429 RESOURCE_EXHAUSTED on all retry attempts (daily quota exceeded on free tier). The "against" perspective has been supplemented by the synthesiser's own adversarial analysis to maintain balance.

---

## 1. Executive Summary

**Consensus Verdict: APPROVED WITH MODIFICATIONS**

The MQTT streaming plan for Pathfinder is technically sound and follows established IoT telemetry patterns. The plan is approved with one **critical architectural modification**: MQTT publishing must be decoupled from the TDM timing path using a FreeRTOS ring buffer and a dedicated publisher task. Publishing directly during the 20 ms settling phase risks TDM timing violations.

**Overall Confidence: 8/10**

---

## 2. Plan Under Review

### Proposed Architecture
- ESP32 MQTT client using **esp-mqtt** (ESP-IDF native)
- Publish `PathfinderRawMessage` JSON on each TDM cycle (10 Hz) to topic `geosim/pathfinder/data/raw`
- QoS 1 for measurement data (at-least-once delivery)
- QoS 0 for status/heartbeat (1 Hz)
- Last Will and Testament (LWT) on `geosim/status/pathfinder`
- SD card logging continues as backup
- MQTT configurable via `config.h` compile flag

### Message Format (~250 bytes)
```json
{
  "ts": "2026-02-18T14:30:00.123Z",
  "lat": 51.2345678, "lon": -1.4567890, "alt": 102.34,
  "fix": 6, "hdop": 0.8,
  "g1": 222, "g2": 212, "g3": 198, "g4": 205,
  "emi_i": 0.0023, "emi_q": 0.0456, "sigma": 0.052,
  "pitch": 2.3, "roll": -1.1, "heading": 145.6,
  "ir_obj": 18.5, "ir_amb": 22.1
}
```

---

## 3. Concern-by-Concern Analysis

### Concern 1: JSON Serialization Overhead (ArduinoJson vs snprintf)

| Aspect | Recommendation |
|--------|---------------|
| **Consensus** | Use `snprintf()` into a fixed stack buffer |
| **Rationale** | At 10 Hz, determinism matters more than developer convenience. `snprintf` has zero heap allocation, predictable execution time, and compiles to smaller code. ArduinoJson's `StaticJsonDocument` is acceptable if you are already in Arduino-land, but adds ~6 KB flash overhead for no real benefit at this message complexity. |
| **Precision** | Lat/lon: 7 decimal places. Angles: 1 decimal. EMI: 4 decimals. This keeps payload size stable at ~250 bytes. |

**GPT-5.2:** "Prefer manual `snprintf` into a fixed buffer (predictable, no allocations). Overhead is manageable; prioritize determinism over micro-optimizing CPU cycles."

**Adversarial counterpoint:** If the message format evolves (new fields, nested objects), snprintf format strings become fragile and error-prone. Consider a thin wrapper function that validates field count at compile time.

### Concern 2: WiFi Reliability and Reconnection

| Aspect | Recommendation |
|--------|---------------|
| **Consensus** | Decouple WiFi from TDM via ring buffer + publisher task |
| **Rationale** | ESP32 WiFi can drop during TDM gating transitions. The TDM task must never block on network I/O. |
| **Architecture** | TDM task writes `PathfinderRawMessage` struct to a lock-free ring buffer in RAM. A separate FreeRTOS publisher task drains the buffer and calls `esp_mqtt_client_publish()`. |
| **Buffer policy** | Size ring buffer for ~100 samples (10 seconds). When full: drop oldest, increment a `dropped_count` field in the next published message. |
| **Reconnection** | Use ESP-IDF WiFi event handlers with auto-reconnect + exponential backoff. Disable WiFi power save (`esp_wifi_set_ps(WIFI_PS_NONE)`) to minimise latency. |

**GPT-5.2:** "Don't publish from the TDM timing path. TDM task writes a struct to a lock-free/ring buffer (RAM). A publisher task drains the buffer and calls esp-mqtt."

**Adversarial counterpoint:** Ring buffer on ESP32 consumes precious RAM. 100 samples x ~64 bytes struct = 6.4 KB, which is acceptable. However, if buffer overflow is frequent, it indicates a systemic WiFi problem that buffering cannot solve. Log overflow events to SD with timestamps for post-hoc diagnosis.

### Concern 3: MQTT Broker Address (Hardcoded vs mDNS)

| Aspect | Recommendation |
|--------|---------------|
| **Consensus** | Static IP as primary, mDNS as optional fallback |
| **Rationale** | mDNS is unreliable on ESP32 in noisy RF environments and during initial boot. A static IP or DHCP reservation for the Jetson is more deterministic. |
| **Implementation** | Compile-time default in `config.h` (e.g., `#define MQTT_BROKER_IP "192.168.4.1"`). Long-term: add NVS runtime configuration for field flexibility. |

**GPT-5.2:** "Default to a static IP/hostname configured in NVS (or compile-time default). Optionally support mDNS discovery as a convenience, but keep a deterministic fallback."

**Adversarial counterpoint:** If the Jetson acts as a WiFi AP (likely for field use), the gateway IP is known and fixed. mDNS adds complexity for zero benefit in this topology. Skip mDNS entirely for v1.

### Concern 4: Message Size and MQTT Overhead

| Aspect | Recommendation |
|--------|---------------|
| **Consensus** | Current sizing is fine; focus on buffer configuration |
| **Rationale** | 250 B payload x 10 Hz = 2.5 KB/s. MQTT fixed header adds 2-5 bytes. Variable header (topic) adds ~30 bytes. Total: ~285 B/message, ~2.85 KB/s. This is <1% of ESP32 WiFi capacity. |
| **Buffer config** | Set esp-mqtt outbox size to handle bursts: `outbox_limit = 32768` (enough for ~100 queued messages during brief RF fades). |

**GPT-5.2:** "WiFi capacity isn't the issue. More relevant is buffer sizing: configure esp-mqtt outbox and TX buffers to comfortably hold bursts during brief RF fades."

**Adversarial counterpoint:** The real overhead concern is not bandwidth but latency variance. QoS 1 requires a PUBACK round-trip. On a congested WiFi channel (2.4 GHz), this can spike to 50-200 ms. This is why the async publisher task is mandatory -- the outbox handles retransmission without blocking.

### Concern 5: TDM Interaction (20 ms Settling Phase)

| Aspect | Recommendation |
|--------|---------------|
| **Consensus** | **CRITICAL: Do NOT publish directly in the settling phase** |
| **Rationale** | 20 ms is NOT reliably enough for QoS 1 publish + PUBACK over WiFi. TCP retransmits, WiFi arbitration delays, and Linux scheduling jitter on the broker side all contribute to unpredictable latency. |
| **Architecture** | The settling phase should only perform a `memcpy` of the measurement struct into the ring buffer (~1 us). The actual MQTT publish happens asynchronously in the publisher task, which runs on a different FreeRTOS core (or at lower priority). |

**GPT-5.2:** "If you *wait* for PUBACK in that window: no, that's risky. If you enqueue asynchronously and return immediately: yes, 20 ms is plenty because the network stack can complete later."

**Adversarial counterpoint:** Even the `esp_mqtt_client_publish()` call itself can block briefly if the TCP send buffer is full (e.g., during WiFi reconnection when the outbox is saturated). This is another reason the publisher task must be on a separate core or at minimum a separate FreeRTOS task with its own stack, so TDM timing is never affected.

### Concern 6: SD Card Fallback and Replay

| Aspect | Recommendation |
|--------|---------------|
| **Consensus** | No automatic replay. Resume live streaming on reconnect. |
| **Rationale** | SD card is the ground-truth record. MQTT streaming is best-effort live telemetry for dashboards and monitoring. Replaying missed messages would flood the broker, create out-of-order data, and confuse downstream consumers. |
| **If replay is needed later** | Publish on a separate topic (`geosim/pathfinder/data/replay`), rate-limit to 100 msg/s, include `seq` field for deduplication. But this is a v2 feature. |

**GPT-5.2:** "SD should remain authoritative; default to 'no replay' and optionally add a controlled backfill mode on a separate topic."

**Adversarial counterpoint:** Without replay, any WiFi dropout creates gaps in the live stream that the Jetson/GeoSim cannot fill. For anomaly detection, this means missed detections during dropouts. Accept this trade-off for v1, but ensure the SD card path is always active (not just a fallback) so post-survey analysis has complete data.

### Concern 7: esp-mqtt vs PubSubClient

| Aspect | Recommendation |
|--------|---------------|
| **Consensus** | **esp-mqtt is the correct choice** |
| **Rationale** | esp-mqtt is event-driven, integrates natively with FreeRTOS and ESP-IDF event loops, has a built-in outbox for QoS 1 retransmission, and supports async operation. PubSubClient blocks on publish and requires manual reconnection -- both are disqualifying for TDM firmware. |

**GPT-5.2:** "For your TDM constraints, esp-mqtt is the better choice: event-driven, integrates with FreeRTOS, supports asynchronous behavior and internal outbox, avoids the common 'publish blocks and ruins timing' pitfalls."

**Adversarial counterpoint:** esp-mqtt's event-driven model means you must handle `MQTT_EVENT_ERROR`, `MQTT_EVENT_DISCONNECTED`, and `MQTT_EVENT_PUBLISHED` callbacks correctly. Improper error handling in callbacks can cause memory leaks (outbox growth) or silent data loss. Implement explicit outbox size monitoring and log warnings when outbox exceeds 50% capacity.

---

## 4. Additional Recommendations (Beyond Original Concerns)

### 4a. Add Sequence Numbers and Boot ID

Add `seq` (monotonically increasing per boot) and `boot_id` (random uint16 generated at startup) fields to every message:

```json
{
  "seq": 14523,
  "boot_id": 42817,
  "ts": "2026-02-18T14:30:00.123Z",
  ...
}
```

**Rationale:** QoS 1 guarantees at-least-once delivery, meaning duplicates are possible. Sequence numbers allow downstream consumers to detect duplicates and gaps. Boot ID distinguishes sequence resets across device reboots.

### 4b. Security Considerations

The plan does not mention TLS or authentication. For a local Jetson-to-ESP32 WiFi link:
- **v1:** Plain MQTT (port 1883) is acceptable if the WiFi AP uses WPA2/WPA3 with a strong passphrase. TLS on ESP32 adds ~40 KB RAM overhead and measurable latency.
- **v2:** Add MQTT username/password authentication at minimum. Consider TLS with PSK (pre-shared key) rather than full certificate chain to minimise overhead.

### 4c. MQTT Protocol Version

Use **MQTT 3.1.1** (not MQTT 5). esp-mqtt supports both, but MQTT 5 features (shared subscriptions, topic aliases, reason codes) are unnecessary for this point-to-point telemetry use case. MQTT 3.1.1 is simpler and better tested with Mosquitto.

### 4d. Consider Binary Format for Future Optimisation

JSON at 250 bytes is fine for v1. If bandwidth or serialization time becomes a concern at higher rates:
- **MessagePack:** ~40% smaller than JSON, trivial to implement (`msgpack-c` or manual packing).
- **CBOR:** Similar size reduction, IETF standard (RFC 8949).
- **Custom binary struct:** Smallest possible (~64 bytes), but sacrifices human-readability and requires matched parser on the Jetson.

### 4e. FreeRTOS Task Architecture

Recommended task layout for MQTT-enabled firmware:

| Task | Core | Priority | Stack | Role |
|------|------|----------|-------|------|
| TDM Scheduler | Core 1 | configMAX_PRIORITIES - 1 | 4096 | Timing-critical sensor multiplexing |
| MQTT Publisher | Core 0 | tskIDLE_PRIORITY + 3 | 4096 | Drains ring buffer, publishes to broker |
| WiFi/MQTT Events | Core 0 | tskIDLE_PRIORITY + 2 | 4096 | Handles connect/disconnect/error events |
| SD Logger | Core 0 | tskIDLE_PRIORITY + 1 | 4096 | Writes CSV to SD card |

Pin the TDM task to Core 1 so WiFi interrupts (which run on Core 0) never preempt sensor timing.

---

## 5. Synthesised Architecture Diagram

```
                        ESP32 Core 1                 ESP32 Core 0
                    +------------------+         +-------------------+
                    |  TDM Scheduler   |         | MQTT Publisher    |
                    |  (10 Hz cycle)   |         | Task              |
                    |                  |         |                   |
                    |  1. Read sensors |         | 1. Wait on ring   |
                    |  2. Build struct |         |    buffer         |
                    |  3. memcpy to  ------>---- | 2. Serialize JSON |
                    |     ring buffer  |  RING   | 3. esp_mqtt_      |
                    |  4. Signal SD    |  BUFFER |    client_publish |
                    |     logger       |  (RAM)  | 4. Monitor outbox |
                    +------------------+         +-------------------+
                            |                            |
                            v                            v
                    +------------------+         +-------------------+
                    |  SD Logger Task  |         |  WiFi/MQTT Event  |
                    |  (writes CSV)    |         |  Handler          |
                    +------------------+         +-------------------+
                                                         |
                                                         v
                                                 +-------------------+
                                                 | Mosquitto Broker  |
                                                 | (Jetson)          |
                                                 +-------------------+
```

---

## 6. Recommended MQTT Topic Structure

| Topic | QoS | Rate | Content |
|-------|-----|------|---------|
| `geosim/pathfinder/data/raw` | 1 | 10 Hz | PathfinderRawMessage JSON |
| `geosim/pathfinder/status` | 0 | 1 Hz | Heartbeat: uptime, free heap, WiFi RSSI, SD space, dropped_count |
| `geosim/status/pathfinder` | 1 | LWT | `{"status":"offline","ts":"..."}` (retained) |
| `geosim/pathfinder/config` | 1 | On-demand | Subscribe for runtime config commands (v2) |

---

## 7. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | QoS 1 PUBACK blocks TDM | High (if direct publish) | Critical | Ring buffer + async publisher task (MANDATORY) |
| R2 | WiFi drops during survey | Medium | Low | SD card ground truth; ring buffer absorbs gaps |
| R3 | esp-mqtt outbox memory exhaustion | Low | Medium | Monitor outbox size; cap at 32 KB; drop oldest on overflow |
| R4 | JSON format string bugs | Medium | Medium | Unit test snprintf output; validate with JSON parser in CI |
| R5 | SD + MQTT dual-write contention | Low | Low | SD writes on Core 0 at low priority; SPI mutex already exists |
| R6 | WiFi interference from TDM gating | Low | Medium | EMI coil gating is low-frequency; 2.4 GHz WiFi unaffected |
| R7 | Broker unavailable at boot | Medium | Low | Retry with backoff; SD logging starts immediately regardless |

---

## 8. Implementation Checklist

- [ ] Add `MQTT_ENABLED` compile flag to `config.h`
- [ ] Add `MQTT_BROKER_IP` and `MQTT_BROKER_PORT` to `config.h`
- [ ] Create `PathfinderRawMessage` struct in shared header
- [ ] Implement lock-free ring buffer (or use `xRingbufferCreate`)
- [ ] Create MQTT publisher FreeRTOS task pinned to Core 0
- [ ] Implement `snprintf`-based JSON serializer with fixed buffer
- [ ] Add `seq` and `boot_id` fields to message format
- [ ] Configure esp-mqtt: outbox 32 KB, QoS 1, LWT, clean session
- [ ] Add MQTT event handler: log connect/disconnect/error to serial
- [ ] Add outbox size monitoring to status/heartbeat message
- [ ] Add `dropped_count` tracking when ring buffer overflows
- [ ] Test with `mosquitto_sub` on Jetson at 10 Hz for 1 hour
- [ ] Verify SD card logging unaffected when MQTT enabled
- [ ] Verify TDM timing unaffected when MQTT enabled (oscilloscope)
- [ ] Verify LWT published on power-off and WiFi disconnect

---

## 9. Consensus Decision Matrix

| Concern | GPT-5.2 (FOR) | Adversarial (AGAINST) | Final Decision |
|---------|---------------|----------------------|----------------|
| Library choice | esp-mqtt (strong yes) | esp-mqtt (agreed, with error handling caveat) | **esp-mqtt** |
| JSON serializer | snprintf preferred | snprintf (agreed, add validation wrapper) | **snprintf with compile-time field count check** |
| QoS level | QoS 1 for data | QoS 1 acceptable (note PUBACK latency) | **QoS 1 for data, QoS 0 for status** |
| Broker discovery | Static IP + optional mDNS | Static IP only for v1 | **Static IP only (v1), NVS config (v2)** |
| Publish timing | Async via ring buffer | Async mandatory (agreed) | **Ring buffer + publisher task (MANDATORY)** |
| SD replay | No auto-replay | No auto-replay (agreed) | **No replay; SD is ground truth** |
| Message format | JSON fine for v1 | JSON fine (add seq + boot_id) | **JSON with seq + boot_id fields** |
| Security | Not addressed | Plain MQTT v1, add auth v2 | **WPA2 WiFi encryption for v1; MQTT auth for v2** |

---

## 10. Final Recommendation

**Proceed with the MQTT streaming plan as described, implementing the mandatory ring buffer + publisher task architecture.** The plan is well-suited to the Pathfinder's operational requirements. The esp-mqtt library, QoS 1 for measurement data, LWT for disconnect detection, and SD card as ground-truth backup form a solid telemetry stack.

The single most important takeaway: **never call `esp_mqtt_client_publish()` from the TDM timing path.** All MQTT I/O must be isolated to a separate FreeRTOS task on Core 0, communicating with the TDM scheduler via a lock-free ring buffer.

---

*Generated by PAL Consensus Validation | openai/gpt-5.2 + Claude Opus 4.6 adversarial synthesis*
*gemini-3-pro-preview was unavailable (API quota exhausted)*
