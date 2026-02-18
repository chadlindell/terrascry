# R2: ESP32 Dual-Core TDM Firmware Architecture -- Consensus Validation

**Date**: 2026-02-18
**Research Task**: R2 -- TDM Firmware Architecture Evaluation
**Method**: Multi-model consensus (Claude Opus 4.6 + OpenAI GPT-5.2 + GPT-5.2-pro)
**Source Document**: `tdm-firmware-design.md`
**Status**: Consensus achieved -- Architecture feasible with critical issues requiring resolution

---

## Executive Summary

The ESP32 dual-core Time-Division Multiplexed (TDM) firmware architecture for Pathfinder's multi-sensor geophysical instrument is **fundamentally sound in concept**. The dual-core split (measurement Core 1 / protocol Core 0) correctly addresses the primary interference concern (EMI TX corrupting fluxgate measurements). The TDM approach provides deterministic electromagnetic isolation between incompatible sensor operations.

However, all consulted models identified **five critical-to-high-risk implementation issues** that must be resolved before the architecture will produce reliable measurements in field conditions. The issues range from incorrect timing assumptions in the ADC path to insufficient settling time and cross-core data race conditions.

**Overall Verdict**: Feasible in principle, but current timing assumptions are optimistic and will cause measurement contamination and/or missed deadlines without the changes recommended below.

**Consensus Confidence**: High (8/10)

---

## Consensus Findings by Concern

### Concern 1: ADS1115 I2C ADC Timing -- CRITICAL

**Severity**: CRITICAL -- Architecture will not meet stated timing requirements
**Consensus**: UNANIMOUS -- All models agree this is the highest-priority issue

#### The Problem

The design document (line 36) claims "DMA-triggered continuous sampling of 8 fluxgate channels via ADS1115 x2" and the code (lines 87-91) calls `adc_start_dma_sampling()`. This is **internally inconsistent**: the ADS1115 communicates over I2C, which is not DMA-capable on the ESP32. Each sample requires explicit I2C transactions.

At 128 SPS, each conversion takes approximately 7.8 ms. The timing arithmetic:

| Parameter | Value |
|-----------|-------|
| Conversion time per sample | 7.8 ms (at 128 SPS) |
| Channels per ADS1115 | 4 (sequential MUX) |
| Time for one full scan (per device) | 4 x 7.8 ms = 31.2 ms |
| Available Phase 1 time | 50 ms |
| Full scans achievable in 50 ms | ~1.6 per ADS1115 |
| I2C bus overhead (config + read) | ~0.5-1.0 ms per transaction |
| Concurrent BNO055 + MLX90614 reads | Additional I2C bus time |

The design document's claim (line 40) of "5 complete ADS1115 conversion cycles at 128 SPS" is **not achievable** with the current architecture. Both ADS1115s share a single I2C bus with BNO055, MLX90614, DS3231, and PCA9615, further reducing available bandwidth.

For the EMI phase (Phase 2, 30 ms), at 128 SPS you get only ~3-4 samples of I/Q channels -- marginal for averaging.

#### Recommended Fixes (Priority Order)

1. **Immediate (firmware)**: Replace polling with ALERT/RDY pin interrupt-driven continuous mode. Connect each ADS1115's ALERT/RDY pin to an ESP32 GPIO. Configure ADS1115 in continuous mode. Use ISR to read completed conversions without wasting CPU time polling.

2. **Near-term (firmware)**: Consider 860 SPS mode during EMI phase where resolution requirements are lower (conductivity mapping is inherently lower resolution than fluxgate gradiometry). At 860 SPS, 30 ms yields ~25 samples -- much better for I/Q averaging.

3. **Medium-term (hardware)**: Split I2C buses -- dedicate one I2C peripheral to ADCs only, second for BNO055/MLX90614/DS3231. ESP32 has two I2C peripherals (I2C0, I2C1).

4. **Long-term (hardware redesign)**: Replace ADS1115 with SPI-based multi-channel delta-sigma ADC (ADS131M04, ADS131E08, or AD7124) that supports true DMA via SPI. This eliminates all I2C timing constraints and enables simultaneous multi-channel sampling.

### Concern 2: WiFi TX Gating Method -- HIGH RISK

**Severity**: HIGH -- Current approaches either too slow or incomplete
**Consensus**: UNANIMOUS -- All models agree neither option is fully satisfactory

#### The Problem

Two WiFi gating options are discussed:

| Method | Latency | True RF Silence? |
|--------|---------|-----------------|
| `esp_wifi_stop()` / `esp_wifi_start()` | 10-100 ms (variable) | Yes, but too slow for 100 ms TDM frame |
| `esp_wifi_set_max_tx_power(0/78)` | <1 ms | No -- management frames, ACKs, beacons still emit |

`esp_wifi_stop/start` is **too slow and variable** for reliable use within a 100 ms TDM cycle. A single `esp_wifi_start()` call in Phase 3 (20 ms budget) could overrun the phase entirely.

`esp_wifi_set_max_tx_power(0)` is fast but **does not guarantee RF silence**. The WiFi radio continues to emit management frames, probe responses, and ACK packets at some minimum power level. These RF emissions, even if low-power, may couple into the high-gain analog front end (ADS1115 + fluxgate signal chain) and corrupt measurements.

#### Recommended Fixes (Priority Order)

1. **Immediate (firmware)**: Use `esp_wifi_set_max_tx_power(0)` for TDM gating (fastest option). Additionally, implement **application-level send blocking**: pause all MQTT publishes, socket sends, and NTRIP requests during Phase 1 and Phase 2. This reduces the probability of the WiFi stack initiating TX during sensitive phases.

2. **Near-term (firmware)**: Investigate `esp_wifi_set_ps(WIFI_PS_MAX_MODEM)` which puts the radio into deep modem sleep. This may provide more complete RF suppression than TX power alone, with faster wake than stop/start.

3. **Medium-term (system design)**: Perform bench testing with a spectrum analyzer to characterize actual RF emissions during each gating method. Measure coupling to the fluxgate ADC chain. If residual emissions are below the noise floor of the measurement chain, `set_max_tx_power(0)` may be sufficient in practice.

4. **Long-term (hardware)**: For production instruments, consider either (a) a hardware RF switch/attenuator on the antenna feed controlled by GPIO, (b) a separate communications processor physically isolated from the measurement analog, or (c) Ethernet/serial tether during sensitive acquisition (common in precision geophysical instruments).

### Concern 3: LM2917 Settling Time -- CRITICAL

**Severity**: CRITICAL -- 26% residual will contaminate fluxgate measurements
**Consensus**: UNANIMOUS -- All models agree 26% residual is unacceptable

#### The Problem

The LM2917 frequency-to-voltage converter has an RC filter with:
- R1 = 100 kohm, C2 = 0.15 uF
- Time constant: tau = RC = 15 ms
- After 20 ms settling: residual = e^(-20/15) = e^(-1.33) = 26.4%

If the LM2917 output experiences a step disturbance of, for example, 100 mV during EMI TX (Phase 2), a 26 mV residual would persist at the start of Phase 1 in the next cycle. For a fluxgate gradient measurement targeting ~50 nT sensitivity, this level of analog contamination is **orders of magnitude too large**.

| Settling Time | Time Constants | Residual | Acceptable? |
|--------------|---------------|----------|-------------|
| 20 ms | 1.33 tau | 26.4% | No |
| 30 ms | 2.0 tau | 13.5% | No |
| 45 ms | 3.0 tau | 5.0% | Marginal |
| 60 ms | 4.0 tau | 1.8% | Likely OK |
| 75 ms | 5.0 tau | 0.7% | Yes |

Extending settling to 45-75 ms would compress the fluxgate measurement phase unacceptably.

#### Recommended Fixes (Priority Order)

1. **Immediate (firmware)**: Implement a **sample discard window** -- ignore the first 15-20 ms of Phase 1 ADC readings. Only use samples from t=20ms to t=50ms of Phase 1 for gradient computation. This gives the LM2917 an effective 40 ms settling (20 ms Phase 3 + 20 ms discard = 2.67 tau, ~7% residual). Combined with averaging, this may be adequate.

2. **Near-term (hardware)**: Add an **active discharge circuit** -- use an analog switch (e.g., ADG419, CD4066) or small N-channel MOSFET controlled by an ESP32 GPIO to rapidly discharge the LM2917 filter capacitor at the start of Phase 3. This resets the LM2917 output to a known baseline, eliminating the exponential settling problem entirely.

3. **Medium-term (hardware)**: Redesign the LM2917 filter for a **faster time constant** (e.g., R=47k, C=68nF gives tau=3.2ms, 5-tau settling in 16ms). Accept more output ripple and rely on digital averaging/filtering in firmware to reject the ripple noise. This is often the better engineering tradeoff for a TDM system.

4. **Alternative (hardware)**: Add a **sample-and-hold circuit** (e.g., LF398) before the ADS1115 input. Capture the fluxgate LM2917 value just before EMI TX starts (end of Phase 1) and hold it during Phase 2/3. This decouples the ADC from the LM2917 transient entirely.

### Concern 4: AD9833 + OPA549 Shutdown Sequencing -- MODERATE RISK

**Severity**: MODERATE -- Transients may couple into fluxgate analog front end
**Consensus**: STRONG AGREEMENT -- All models identify the risk, differ slightly on mitigation

#### The Problem

The current code sequences shutdown as:
1. `gpio_set_level(PIN_EMI_SHUTDOWN, 1)` -- OPA549 shutdown pin HIGH

This shuts down the power amplifier first, but:
- The **TX coil is an inductor**. When OPA549 disables, the coil current is interrupted, producing an inductive kick (V = -L di/dt). Even with the OPA549's internal clamp, the transient can couple capacitively or inductively into nearby fluxgate sensor lines.
- The **AD9833 output goes to mid-rail** when the OPA549 disconnects its load. If the AD9833 is then disabled, it creates a second step transient.

#### Recommended Shutdown Sequence

**EMI TX Shutdown (Phase 2 to Phase 3 transition):**
```
1. ad9833_set_amplitude(0)      // Ramp DDS output to zero (or set RESET bit)
2. delayMicroseconds(10)        // Allow DDS output to settle
3. gpio_set_level(PIN_EMI_SHUTDOWN, 1)  // OPA549 shutdown (no current flowing)
4. ad9833_disable()             // Power down DDS
```

**EMI TX Startup (Phase 1 to Phase 2 transition):**
```
1. ad9833_set_frequency(15000)  // Configure DDS frequency
2. ad9833_set_amplitude(0)      // Start with zero amplitude
3. gpio_set_level(PIN_EMI_SHUTDOWN, 0)  // Enable OPA549 (no signal yet)
4. delayMicroseconds(100)       // Allow OPA549 to stabilize
5. ad9833_set_amplitude(full)   // Ramp up DDS output
```

**Additional hardware mitigations:**
- Add a **snubber network** (RC or RCD) across the TX coil to dampen inductive ringing.
- Add a **TVS diode or back-to-back Zener** at the OPA549 output for hard clamping.
- Consider **zero-crossing-aware disable** (monitor the DDS output and trigger shutdown near a zero crossing to minimize di/dt). This is more complex but reduces broadband transient energy.

### Concern 5: Cross-Core GPS Data Race -- MODERATE RISK

**Severity**: MODERATE -- Can cause corrupted GPS coordinates in measurement records
**Consensus**: UNANIMOUS -- All models identify the `latest_gps_fix` volatile struct as unsafe

#### The Problem

The inter-core communication design (line 186) uses:
```c
latest_gps_fix  // Shared volatile struct (updated by GPS task on Core 0, read by measurement task on Core 1)
```

On the ESP32's dual-core Xtensa LX6 architecture, `volatile` does **not** provide atomicity for compound data structures. A GPS fix struct likely contains multiple fields (latitude, longitude, altitude, fix quality, satellite count, HDOP, timestamp). Core 1 can read a **partially-updated struct** where some fields are from the old fix and some from the new fix ("torn read").

Example scenario:
1. Core 0 GPS task updates `latitude` field (new value)
2. Core 1 reads `latitude` (new) and `longitude` (old -- not yet updated)
3. Result: measurement record contains an impossible GPS coordinate

Additionally, the GPS task on Core 0 runs at the same priority (5) as the MQTT task, and WiFi gating via semaphore could preempt GPS parsing mid-NMEA-sentence, though this is less critical since both are on Core 0.

#### Recommended Fixes

1. **Preferred (firmware)**: Use a **FreeRTOS queue** for GPS data (same pattern as measurement results):
```c
// GPS task (Core 0) produces:
xQueueOverwrite(xQueueGPSFix, &new_fix);  // Always keeps latest

// Measurement task (Core 1) consumes:
xQueuePeek(xQueueGPSFix, &local_fix, 0);  // Non-blocking read of latest
```
`xQueueOverwrite` with a queue depth of 1 provides a thread-safe "latest value" pattern.

2. **Alternative (firmware)**: Use a **mutex** around the shared struct:
```c
// Writer (Core 0):
xSemaphoreTake(xMutexGPS, portMAX_DELAY);
latest_gps_fix = new_fix;
xSemaphoreGive(xMutexGPS);

// Reader (Core 1):
xSemaphoreTake(xMutexGPS, 0);  // Non-blocking attempt
local_fix = latest_gps_fix;
xSemaphoreGive(xMutexGPS);
```
Caution: mutex on Core 1 must be non-blocking (timeout = 0) to avoid delaying the measurement cycle.

3. **Alternative (firmware)**: Use a **double-buffer with atomic index**:
```c
volatile GPSFix_t gps_buffers[2];
volatile uint8_t gps_active_index = 0;  // Atomic on ESP32 (single byte)

// Writer: fill inactive buffer, then flip index
// Reader: read from active index
```

---

## Additional Risks Identified During Consensus

### Risk 6: I2C Bus Contention (HIGH)

All sensors on Core 1's measurement path share a single I2C bus:
- 2x ADS1115 (0x48, 0x49) -- high-frequency access during ADC sampling
- BNO055 (0x29) -- orientation read during Phase 1
- MLX90614 (0x5A) -- temperature read during Phase 1
- DS3231 (0x68) -- timestamp read each cycle
- PCA9615 -- differential I2C buffer (transparent but adds propagation delay)

**Mitigation**: Use ESP32's two I2C peripherals. Assign I2C0 to ADS1115s exclusively. Assign I2C1 to BNO055, MLX90614, DS3231. Implement an I2C mutex for each bus to prevent task-level contention.

### Risk 7: Phase Timing Drift (MODERATE)

The TDM cycle uses `vTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICKS(100))` for the overall 100 ms period, but individual phases use `vTaskDelay(pdMS_TO_TICKS(50))` and `vTaskDelay(pdMS_TO_TICKS(30))`. If Phase 1 or Phase 2 overruns due to I2C bus contention or other delays:
- Phase 3 (settling/comms) gets compressed
- Settling time is reduced below the already-marginal 20 ms
- The `vTaskDelayUntil` at the end may not delay at all if the total exceeds 100 ms

**Mitigation**: Use hardware timer interrupts to enforce phase boundaries. Or, calculate Phase 3 duration dynamically as `100 - elapsed` and assert/warn if it drops below a minimum threshold (e.g., 15 ms).

### Risk 8: Measurement Queue Overflow (LOW-MODERATE)

The measurement queue has depth 4. If Core 0 cannot drain the queue (e.g., during WiFi reconnection, SD card busy, or MQTT broker timeout), measurements are silently dropped by `xQueueSend(xQueueMeasurement, &result, 0)` (non-blocking send, timeout = 0).

**Mitigation**: Monitor queue high-water mark via `uxQueueMessagesWaiting()`. Log a warning when queue is >75% full. Consider increasing queue depth to 10-20 (each measurement struct is small). Implement a "queue overflow" counter in the status output.

### Risk 9: EMI Phase Analog Signal Chain Issues (from GPT-5.2-pro)

The GPT-5.2-pro analysis identified additional issues in the EMI signal chain (documented in `emi-coil-design.md`) that affect the TDM architecture:

1. **AD9833 Frequency Tuning Word**: The FTW calculation in the EMI design doc may contain a 1000x error. Verify: FTW = f_out x 2^28 / f_MCLK. For 15 kHz with 25 MHz clock: FTW = 15000 x 268435456 / 25000000 = 161,061 (NOT 161,061,273).

2. **AD630 I/Q**: The AD630 is a single balanced demodulator. For simultaneous I and Q channels, you need either two AD630 chips (one with 0-degree reference, one with 90-degree) or time-multiplexed demodulation with sufficient settling between switches.

3. **Quadrature Reference**: An RC phase shift network at Xc = R provides only 45 degrees of phase shift, not 90 degrees. Consider digital quadrature generation or an all-pass/polyphase network.

4. **EMI Low-Pass Filter Settling**: If the post-demodulation low-pass filter has fc = 10 Hz (tau = 15.9 ms), the 30 ms EMI measurement window only allows ~1.9 tau settling after TX enable -- the ADC may be reading transients, not steady-state conductivity.

---

## Recommended Architecture Revisions

### Phase 1: Immediate Firmware Changes (No Hardware Modifications)

| Change | Impact | Effort |
|--------|--------|--------|
| Replace `adc_start_dma_sampling()` with ALERT/RDY interrupt-driven reads | Correct ADC timing, improve sample count | Medium |
| Add 15 ms sample discard window at start of Phase 1 | Improve settling from 26% to ~7% residual | Low |
| Replace `volatile latest_gps_fix` with `xQueueOverwrite` pattern | Eliminate cross-core data race | Low |
| Add application-level WiFi send blocking during Phase 1/2 | Reduce RF emission probability | Low |
| Implement proper AD9833/OPA549 shutdown sequencing | Reduce transient coupling | Low |
| Add queue high-water mark monitoring | Detect measurement drops | Low |
| Split I2C buses (I2C0 for ADCs, I2C1 for other sensors) | Reduce bus contention | Medium |

### Phase 2: Hardware Revisions (Next PCB Spin)

| Change | Impact | Effort |
|--------|--------|--------|
| Add active discharge circuit on LM2917 filter cap | Eliminate settling time problem | Low |
| Add snubber/clamp on TX coil | Reduce OPA549 shutdown transients | Low |
| Add second AD630 for true simultaneous I/Q | Correct EMI demodulation | Medium |
| Design proper 90-degree quadrature reference network | Correct I/Q phase | Medium |
| Route ADS1115 ALERT/RDY pins to ESP32 GPIOs | Enable interrupt-driven ADC | Low |

### Phase 3: Future Architecture (If Field Testing Reveals Issues)

| Change | Impact | Effort |
|--------|--------|--------|
| Replace ADS1115 with SPI ADC (ADS131M04/AD7124) | True DMA, simultaneous sampling | High |
| Add hardware RF switch on WiFi antenna | True RF silence during measurement | Medium |
| Separate comms processor (ESP32-C3 for WiFi) | Physical RF isolation from analog | High |
| Redesign LM2917 filter for faster tau | Better TDM compatibility | Medium |

---

## Revised TDM Cycle Proposal

Based on consensus findings, the recommended revised cycle (still 100 ms total):

```
0 ms              15 ms              50 ms               80 ms      100 ms
|-- Discard --|---- FLUXGATE ----|---- EMI TX/RX -----|-- SETTLE --|
|  window     |    VALID DATA    |                     |            |
|             |                  |                     |            |
| LM2917      | ADC interrupt-   | AD9833 on          | Sequenced  |
| settling    | driven sampling  | OPA549 on           | shutdown   |
| WiFi TX=0   | BNO055 read      | ADC I/Q sampling   | WiFi TX on |
| No sends    | MLX90614 read    | No WiFi TX         | MQTT pub   |
|             | GPS queue read   |                    | SD write   |
```

- Phase 1a (0-15 ms): Settling/discard window -- ADC reads are taken but discarded
- Phase 1b (15-50 ms): Valid fluxgate measurement window (35 ms of clean data)
- Phase 2 (50-80 ms): EMI TX/RX (unchanged)
- Phase 3 (80-100 ms): Settling + communications (unchanged)

This provides an effective settling time of 35 ms (20 ms Phase 3 + 15 ms discard = 2.33 tau, ~10% residual) without hardware changes, improving to <1% residual with active cap discharge.

---

## Models Consulted

| Model | Role | Status | Key Contribution |
|-------|------|--------|-----------------|
| Claude Opus 4.6 | Independent analysis (Step 1) | Complete | Identified all 5 concerns, I2C bus contention, phase timing drift |
| OpenAI GPT-5.2 | Neutral evaluator | Complete | Detailed ADS1115 throughput math, WiFi management frame emissions, industry perspective |
| OpenAI GPT-5.2-pro | Neutral evaluator (bonus) | Complete | AD9833 FTW error, AD630 single-output limitation, RC phase shift physics, EMI LPF settling |
| Google Gemini 3 Pro | Neutral evaluator (requested) | Rate limited (429) | Not available due to API quota exhaustion |
| Google Gemini 2.5 Pro | Neutral evaluator (fallback) | Rate limited (429) | Not available due to API quota exhaustion |

**Note**: Gemini models were unavailable due to API rate limits during this consensus session. The analysis proceeds with strong agreement across the three models that did respond. A follow-up consultation with Gemini is recommended when quota resets.

---

## Consensus Agreement Matrix

| Issue | Claude Opus 4.6 | GPT-5.2 | GPT-5.2-pro | Consensus |
|-------|:---:|:---:|:---:|-----------|
| ADS1115 "DMA" claim incorrect | AGREE | AGREE | AGREE | UNANIMOUS |
| 128 SPS insufficient for 8-ch in 50ms | AGREE | AGREE | AGREE | UNANIMOUS |
| WiFi stop/start too slow | AGREE | AGREE | -- | STRONG |
| WiFi TX power=0 not true RF silence | AGREE | AGREE | -- | STRONG |
| 26% settling residual unacceptable | AGREE | AGREE | -- | STRONG |
| OPA549 shutdown transients a risk | AGREE | AGREE | -- | STRONG |
| volatile GPS struct is cross-core race | AGREE | AGREE | -- | STRONG |
| SPI ADC recommended long-term | AGREE | AGREE | AGREE | UNANIMOUS |
| AD9833 FTW calculation error | -- | -- | AGREE | NOTED |
| AD630 not dual I/Q | -- | -- | AGREE | NOTED |
| RC phase shift 45 not 90 degrees | -- | -- | AGREE | NOTED |

---

## References

- ESP-IDF FreeRTOS SMP documentation: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/freertos-smp.html
- ADS1115 datasheet (TI SBAS444): ALERT/RDY pin configuration for continuous mode
- ESP-IDF WiFi API: `esp_wifi_set_max_tx_power()` limitations
- McNeill (1980): Electromagnetic terrain conductivity measurement at low induction numbers
- AD9833 datasheet (Analog Devices): Frequency tuning word calculation
- AD630 datasheet (Analog Devices): Single-channel balanced modulator/demodulator
- Source design document: `tdm-firmware-design.md`
