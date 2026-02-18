# TDM State Machine Consensus Validation

## Implementation Task: I2 - Pathfinder Firmware TDM State Machine

**Date**: 2026-02-18
**Source Document**: `tdm-firmware-design.md`
**Validation Method**: PAL Multi-Model Consensus
**Confidence**: 7.5/10

---

## Models Consulted

| Model | Stance | Status | Confidence |
|-------|--------|--------|------------|
| openai/gpt-5.2 | For (advocate) | Success | 8/10 |
| gemini-3-pro-preview | Against (adversary) | FAILED (429 quota exhausted) | N/A |
| claude-opus-4-6 | Independent analysis (substituting adversarial perspective) | Success | 8/10 |

> **Note**: The gemini-3-pro-preview model was unavailable due to Gemini API rate limits (429 RESOURCE_EXHAUSTED). The adversarial perspective was provided by the orchestrating model's independent critical analysis. Confidence score reduced from 8/10 to 7.5/10 to reflect the incomplete multi-model validation.

---

## Overall Verdict

**FEASIBLE WITH MANDATORY MODIFICATIONS**

The TDM state machine architecture is sound in its fundamental approach -- dual-core ESP32, ISR-driven phase transitions, FreeRTOS semaphore/task notification signaling, and clean separation of measurement (Core 1) from protocol (Core 0). However, the current design has one **blocking timing issue** (ADS1115 conversion time exceeds Phase 1 budget), one **documentation inconsistency** (DMA references for I2C ADC), and several **implementation details** that require resolution before coding begins.

---

## Concern-by-Concern Analysis

### Concern 1: I2C in ISR Context

**Severity**: LOW (already correctly handled in the design)
**Consensus**: UNANIMOUS AGREEMENT

The proposed ISR structure is correct: the ISR only performs a phase variable flip and semaphore give. No I2C transactions occur in interrupt context.

**Recommended improvement**: Replace `xSemaphoreGiveFromISR()` with `vTaskNotifyGiveFromISR()` for lower overhead (~30% faster on ESP32). The measurement task then uses `ulTaskNotifyTake(pdTRUE, portMAX_DELAY)` to wait for phase transitions.

```c
// Improved ISR
void IRAM_ATTR onTimer() {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    switch(currentPhase) {
        case FLUXGATE: currentPhase = EMI_TXRX; break;
        case EMI_TXRX: currentPhase = SETTLING; break;
        case SETTLING: currentPhase = FLUXGATE; break;
    }
    vTaskNotifyGiveFromISR(xMeasurementTaskHandle, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

### Concern 2: ADS1115 Conversion Timing (CRITICAL BLOCKER)

**Severity**: CRITICAL -- must be resolved before implementation
**Consensus**: UNANIMOUS AGREEMENT on the problem; strong convergence on solutions

**The problem**: At 128 SPS, each ADS1115 conversion takes 7.8 ms. With 8 channels requiring MUX switching (which restarts the conversion), total sequential time = 8 x 7.8 ms = 62.4 ms. This exceeds the 50 ms Phase 1 budget by 12.4 ms (25% overrun).

**Additional inconsistency identified**: The design document references "DMA-triggered continuous sampling" (line 36) but line 230 states "ADS1115 DMA: Not natively supported over I2C." The ADS1115 communicates via I2C, which has no DMA path on ESP32. The correct terminology is "timer-triggered polling" or "interrupt-driven polling."

**Ranked solutions (in order of recommendation)**:

#### Solution A: Dual ADS1115 on Separate I2C Buses (RECOMMENDED)

The ESP32 has two I2C controllers (I2C_NUM_0 and I2C_NUM_1). Place one ADS1115 on each bus:

- ADS1115 #1 (I2C_NUM_0, addr 0x48): Fluxgate channels 0-3
- ADS1115 #2 (I2C_NUM_1, addr 0x49): Fluxgate channels 4-7

Each device reads 4 channels: 4 x 7.8 ms = 31.2 ms per device. Because the I2C buses are independent hardware, both conversions run in parallel. Wall time = ~31.2 ms + I2C overhead (~2 ms) = ~33 ms. This comfortably fits within 50 ms.

**Implementation with pipelining and ALERT/RDY**:

```c
// Phase 1: Pipelined dual-ADS1115 acquisition
// Step 1: Start conversions on both chips simultaneously
ads1115_start_conversion(I2C_NUM_0, ADS1115_MUX_CH0);
ads1115_start_conversion(I2C_NUM_1, ADS1115_MUX_CH0);

// Step 2: While waiting for conversion, read BNO055 + MLX90614
bno055_read_quaternion(&orientation);   // ~1 ms
mlx90614_read_temperature(&ir_temp);     // ~1 ms

// Step 3: Read results when ALERT/RDY pins go low (or poll config register)
for (int ch = 0; ch < 4; ch++) {
    wait_for_alert(ADS1115_RDY_PIN_0);
    adc_buffer[ch] = ads1115_read_result(I2C_NUM_0);
    wait_for_alert(ADS1115_RDY_PIN_1);
    adc_buffer[ch + 4] = ads1115_read_result(I2C_NUM_1);

    if (ch < 3) {  // Start next conversion
        ads1115_start_conversion(I2C_NUM_0, next_mux[ch]);
        ads1115_start_conversion(I2C_NUM_1, next_mux[ch]);
    }
}
```

**Hardware requirement**: Wire ALERT/RDY pin from each ADS1115 to an ESP32 GPIO. This avoids fixed `vTaskDelay(8)` waits and reduces total acquisition time.

#### Solution B: Increase Data Rate to 250 SPS

At 250 SPS, conversion time = 4 ms. Total for 8 channels (single bus) = 32 ms. For 2x ADS1115 on one bus = still 32 ms (sequential). On two buses = 16 ms.

**Trade-off**: ADS1115 noise at 250 SPS is higher (7.81 uV RMS vs 3.91 uV at 128 SPS). At 10 Hz output, oversampling provides ~4x noise reduction (2 conversions per output), so effective noise is 5.5 uV -- likely acceptable for fluxgate gradients where the LM2917 F-to-V output has mV-scale signals.

#### Solution C: Interleave Channels Across Cycles

Read channels 0-3 in even cycles, channels 4-7 in odd cycles. Time per cycle = 4 x 7.8 ms = 31.2 ms. Each channel updates at 5 Hz instead of 10 Hz.

**Critical caveat**: This introduces 100 ms temporal skew between channel sets. For gradient computation (difference between sensor pairs), both sensors in a pair MUST be read in the same cycle. Acceptable only if pairs are assigned to the same cycle (e.g., pair 1 and pair 2 in even cycles, pair 3 and pair 4 in odd cycles).

#### Solution D: Migrate to SPI ADC (Long-term)

Replace ADS1115 with a multichannel SPI ADC (e.g., ADS131M08, MCP3208). SPI offers MHz-rate transfers and simultaneous sampling. This eliminates per-channel latency entirely. Reserve for v2 if channel count or sample rate requirements grow.

### Concern 3: AD9833 SPI vs I2C Bus Contention

**Severity**: LOW
**Consensus**: UNANIMOUS AGREEMENT -- no contention

SPI (AD9833) and I2C (ADS1115, BNO055, MLX90614) are independent hardware peripherals on the ESP32. There is no electrical or bus-level contention.

**Caveat**: If both SPI and I2C transactions are initiated from the same FreeRTOS task (as in the current vTaskMeasurement design), they execute sequentially within that task -- no concurrency risk. If future refactoring splits them into separate tasks on Core 1, guard with mutexes or ensure non-overlapping phase access.

### Concern 4: OPA549 Enable/Disable Settling

**Severity**: LOW
**Consensus**: UNANIMOUS AGREEMENT -- negligible but add guard

The OPA549 shutdown pin provides <1 us disable. Output settling after enable is ~10 us. Against a 30 ms EMI phase, this is negligible (0.03% of phase time).

**Recommended action**: Discard the first 0.5-2 ms of EMI ADC samples after OPA549 enable. This accounts for the full analog chain settling: OPA549 output stage (~10 us), AD8421 preamp (~5 us), AD630 lock-in detector (~100 us), and any downstream RC filtering.

```c
// Phase 2: EMI TX/RX
gpio_set_level(PIN_EMI_SHUTDOWN, 0);   // OPA549 enable
vTaskDelay(pdMS_TO_TICKS(2));          // 2ms analog settling guard
adc_start_emi_sampling();              // Begin sampling after settling
vTaskDelay(pdMS_TO_TICKS(28));         // Remaining 28ms of sampling
adc_stop_emi_sampling();
```

### Concern 5: WiFi TX Gating Reliability

**Severity**: MEDIUM
**Consensus**: UNANIMOUS AGREEMENT on approach

`esp_wifi_stop()` / `esp_wifi_start()` has 10-100 ms latency -- far too slow for TDM gating. The design document already identifies the better approach: `esp_wifi_set_max_tx_power(0)` / `esp_wifi_set_max_tx_power(78)`, which executes in <1 ms.

**Remaining risks**:
1. **Mid-packet abort**: Setting TX power to 0 during an active WiFi frame may cause the frame to be truncated. The 802.11 MAC will handle retransmission, but there may be a brief RF burst before the power change takes effect.
2. **Not a true RF mute**: TX power 0 on ESP32 still allows minimal emissions from the RF front-end.

**Recommended approach**:
1. Use `esp_wifi_set_max_tx_power(0)` with a **2-5 ms guard band** before fluxgate sampling begins. Signal Core 0 to disable WiFi TX 5 ms before Phase 1 starts (at the end of Phase 3).
2. Schedule all MQTT publishes and NTRIP traffic to occur only during the first 15 ms of Phase 3, leaving 5 ms for WiFi to complete any pending frames before Phase 1.
3. If bench testing reveals fluxgate noise from residual WiFi emissions, add a **hardware RF switch** (e.g., SKY13286) on the ESP32 antenna path, controlled via GPIO.

### Concern 6: FreeRTOS Queue Overflow

**Severity**: MEDIUM-HIGH
**Consensus**: UNANIMOUS AGREEMENT -- increase depth and add policy

Queue depth 4 at 10 Hz = 400 ms of buffering. Common SD card stalls (FATFS sector allocation, wear leveling) can last 100-500 ms; occasional stalls can reach 1-2 seconds. WiFi reconnection after dropout can take 5+ seconds.

**Recommended changes**:

1. **Increase queue depth to 16** (1.6 seconds of buffering): `xQueueCreate(16, sizeof(MeasurementResult_t))`
2. **Use non-blocking send with overflow counter**:

```c
if (xQueueSend(xQueueMeasurement, &result, 0) != pdPASS) {
    overflow_count++;  // Increment diagnostic counter
    // Optionally: overwrite oldest with xQueueOverwrite for 1-deep mailbox
}
```

3. **Add PSRAM ring buffer for extended buffering** (if ESP32-WROVER with PSRAM):
   - Primary path: FreeRTOS queue (depth 16) for real-time MQTT streaming
   - Secondary path: PSRAM circular buffer (depth 1000+) for SD card write-behind
   - SD task drains PSRAM buffer in bursts, tolerating multi-second stalls

4. **Monitor and report**: Include `overflow_count` and `queue_high_watermark` in the 1 Hz status telemetry.

---

## Additional Issues Identified

### Issue 7: Documentation Inconsistency -- DMA Reference

**Location**: `tdm-firmware-design.md` line 36 vs line 230
**Problem**: Line 36 states "DMA-triggered continuous sampling of 8 fluxgate channels via ADS1115 x2" but line 230 states "ADS1115 DMA: Not natively supported over I2C; use timer-triggered polling at desired sample rate."
**Action required**: Correct line 36 to read "Timer-triggered polling of 8 fluxgate channels via ADS1115 x2" and ensure all references to DMA in the context of ADS1115 are removed.

### Issue 8: LM2917 Settling Time May Be Insufficient

**Location**: `tdm-firmware-design.md` lines 222-223
**Problem**: The LM2917 F-to-V converter has RC time constant tau = R1 x C2 = 100k x 0.15 uF = 15 ms. After 20 ms settling (Phase 3), residual from EMI TX shutdown transient is e^(-20/15) = 26.4%. Over one-quarter of the disturbance energy remains.
**Impact**: If the LM2917 output hasn't settled before Phase 1 begins, fluxgate readings in the first few milliseconds of Phase 1 may contain EMI artifacts.
**Options**:
1. Extend settling phase to 30 ms (e^(-30/15) = 13.5% residual). Requires reducing Phase 1 to 40 ms or Phase 2 to 20 ms.
2. Add active discharge circuit to LM2917 output during Phase 3 (MOSFET switch to discharge C2).
3. Discard first 5-10 ms of Phase 1 fluxgate readings (dead time).
4. Accept 26% residual if the absolute magnitude is below the noise floor of the ADC chain.
**Recommendation**: Bench-test first. Measure LM2917 output with oscilloscope at Phase 1 boundary. If residual exceeds ADS1115 LSB at PGA gain setting, implement option 3 as quickest fix.

### Issue 9: GPS Shared Volatile Struct Race Condition

**Location**: `tdm-firmware-design.md` line 186
**Problem**: `latest_gps_fix` is a shared volatile struct written by Core 0 (GPS task) and read by Core 1 (measurement task). A volatile qualifier alone does not prevent torn reads on multi-word structs. If Core 0 writes the latitude while Core 1 reads longitude, the result may be an inconsistent position.
**Solution**: Use a FreeRTOS mutex, critical section, or double-buffering with atomic index swap:

```c
// Double-buffer approach (lock-free)
volatile GPSFix_t gps_fix_buffers[2];
volatile uint8_t gps_fix_active = 0;  // Atomic on ESP32 (single byte)

// Core 0 (writer):
uint8_t write_idx = 1 - gps_fix_active;
gps_fix_buffers[write_idx] = new_fix;  // Write to inactive buffer
gps_fix_active = write_idx;             // Atomic switch

// Core 1 (reader):
GPSFix_t fix = gps_fix_buffers[gps_fix_active];  // Read active buffer
```

### Issue 10: vTaskDelay Phase Timing Precision

**Problem**: The current `vTaskMeasurement` code uses `vTaskDelay(pdMS_TO_TICKS(50))` for Phase 1 timing. FreeRTOS tick resolution on ESP32 is typically 1 ms (configTICK_RATE_HZ = 1000), but `vTaskDelay` has +/- 1 tick jitter. Combined with task scheduling overhead, phase boundaries may drift by 1-2 ms.
**Impact**: For most use cases, 1-2 ms jitter on a 50 ms phase is acceptable (2-4%). But if precise phase timing is critical for EMI TX window alignment, this could matter.
**Alternative**: The ISR-based timer approach in the state machine proposal provides more precise phase boundaries (~us precision with hardware timer). Keep the ISR for phase transitions; use the task loop only for sensor reads within each phase.

---

## Points of Agreement (All Sources)

1. The dual-core ESP32 architecture with measurement on Core 1 and protocol on Core 0 is correct and follows industry best practices for EMI-sensitive instruments.
2. The ISR must only signal (phase flip + notification); all bus I/O must remain in task context.
3. The ADS1115 timing problem is the single blocking issue and is solvable without changing hardware, primarily via dual I2C buses + pipelining.
4. SPI and I2C have no bus-level contention on ESP32.
5. WiFi gating via TX power control is the right approach; esp_wifi_stop/start is too slow.
6. Queue depth 4 is insufficient and needs a clear overflow policy.

## Points Requiring Further Validation

1. **ADS1115 noise at 250 SPS**: If Solution B is chosen, verify that the increased ADC noise at 250 SPS does not degrade gradient sensitivity below the target specification.
2. **LM2917 settling residual**: Bench measurement needed to determine if 26% residual at 20 ms is within acceptable limits.
3. **WiFi TX power gating**: Verify with spectrum analyzer that `esp_wifi_set_max_tx_power(0)` provides sufficient RF attenuation in the fluxgate frequency band.
4. **ESP32 I2C clock stretching**: Some I2C devices (BNO055 in particular) use clock stretching. Verify that concurrent I2C operations on both buses don't cause unexpected delays.

---

## Recommended Revised Timing Budget

```
Phase 0 (guard):   -5 ms to 0 ms    WiFi TX power -> 0, wait for pending frames
Phase 1 (fluxgate): 0 ms to 50 ms   Dual ADS1115 pipelined reads (~33 ms)
                                      + BNO055 (~1 ms) + MLX90614 (~1 ms)
                                      Margin: ~15 ms
Phase 2 (EMI):      50 ms to 80 ms   2 ms settling guard, 28 ms sampling
Phase 3 (settling): 80 ms to 95 ms   Package + queue results, WiFi TX on,
                                      MQTT publish, SD write trigger
Phase 0 (guard):    95 ms to 100 ms  WiFi TX power -> 0 (pre-Phase 1 guard)
```

---

## Actionable Next Steps

1. **IMMEDIATE**: Correct DMA references in `tdm-firmware-design.md` (documentation fix)
2. **HARDWARE**: Wire ALERT/RDY pins from both ADS1115 devices to ESP32 GPIOs
3. **HARDWARE**: Assign ADS1115 devices to separate I2C buses (I2C_NUM_0 and I2C_NUM_1)
4. **FIRMWARE**: Implement pipelined dual-ADS1115 acquisition with ALERT/RDY interrupts
5. **FIRMWARE**: Replace xSemaphoreGiveFromISR with vTaskNotifyGiveFromISR
6. **FIRMWARE**: Add 2 ms EMI settling guard at start of Phase 2
7. **FIRMWARE**: Increase queue depth to 16, add non-blocking send with overflow counter
8. **FIRMWARE**: Implement double-buffered GPS fix struct for safe cross-core access
9. **FIRMWARE**: Add WiFi TX disable guard band (5 ms before Phase 1)
10. **BENCH TEST**: Measure LM2917 output settling with oscilloscope at phase boundaries
11. **BENCH TEST**: Verify WiFi TX power gating RF attenuation with spectrum analyzer
12. **BENCH TEST**: Measure actual ADS1115 acquisition time with pipelining on both I2C buses

---

## Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| ADS1115 timing overrun | HIGH (certain without fix) | CRITICAL (breaks TDM cycle) | Dual I2C buses + pipelining |
| Queue overflow on SD stall | MEDIUM | HIGH (data loss) | Depth 16 + PSRAM buffer |
| WiFi residual emission | LOW-MEDIUM | MEDIUM (fluxgate noise) | Guard band + bench test |
| LM2917 settling insufficient | MEDIUM | MEDIUM (first-sample artifact) | Bench test, then discard or extend |
| GPS struct torn read | LOW-MEDIUM | LOW (occasional bad position) | Double-buffer pattern |
| I2C clock stretching delays | LOW | MEDIUM (phase overrun) | Timeout + bench characterization |

---

*Generated by PAL Consensus Validation (claude-opus-4-6 + openai/gpt-5.2)*
*gemini-3-pro-preview was unavailable (API quota exhausted)*
