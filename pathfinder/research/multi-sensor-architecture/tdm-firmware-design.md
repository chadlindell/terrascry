# Time-Division Multiplexing (TDM) Firmware Design

## Overview

The TDM firmware is the central coordination mechanism that prevents electromagnetic interference between Pathfinder's multiple sensors. By ensuring that only compatible sensors operate simultaneously, TDM eliminates the CRITICAL interference paths identified in `interference-matrix.md` without sacrificing measurement throughput.

## TDM Cycle Structure

Each measurement cycle is 100 ms (10 Hz output rate), divided into three phases:

```
0 ms                    50 ms                80 ms      100 ms
├────── Phase 1 ────────┤──── Phase 2 ───────┤── Phase 3 ─┤
│   FLUXGATE READ       │   EMI TX/RX        │  SETTLING  │
│                       │                     │            │
│ ✓ Fluxgate sensors    │ ✓ AD9833 DDS on    │ ✓ WiFi TX  │
│ ✓ ADS1115 sampling    │ ✓ OPA549 TX amp on │ ✓ MQTT pub │
│ ✓ BNO055 orientation  │ ✓ ADC I/Q sample   │ ✓ SD write │
│ ✓ MLX90614 IR         │ ✓ Lock-in detect   │ ✓ NTRIP RX │
│ ✓ GPS RX (passive)    │                     │ ✓ GPS fix  │
│                       │ ✗ Fluxgate (ignore) │            │
│ ✗ EMI TX (OFF)        │ ✗ WiFi TX (OFF)    │ ✗ EMI TX   │
│ ✗ WiFi TX (OFF)       │                     │            │
│ ✗ SD card (no writes) │                     │            │
└───────────────────────┘─────────────────────┘────────────┘
```

### Phase 1: Fluxgate Measurement (50 ms)

**Purpose**: Acquire clean magnetic gradient measurements free from EMI TX and WiFi interference.

| Action | Detail |
|--------|--------|
| EMI TX shutdown | OPA549 shutdown pin HIGH (hard disable) |
| WiFi TX gate | `esp_wifi_stop()` or TX power = 0 |
| ADC sampling | DMA-triggered continuous sampling of 8 fluxgate channels via ADS1115 ×2 |
| BNO055 read | Orientation quaternion for tilt correction |
| MLX90614 read | Ground surface temperature |
| GPS receive | Passive — ZED-F9P continues tracking, no TX from ESP32 |
| Duration rationale | 50 ms provides 5 complete ADS1115 conversion cycles at 128 SPS, sufficient for averaging |

### Phase 2: EMI TX/RX (30 ms)

**Purpose**: Acquire ground conductivity measurement using the EMI channel.

| Action | Detail |
|--------|--------|
| AD9833 enable | DDS outputs 15 kHz sine wave |
| OPA549 enable | Shutdown pin LOW, TX coil driven |
| ADC sampling | Switch ADS1115 MUX to I and Q channels from AD630 |
| Lock-in processing | Average I and Q samples over 30 ms window |
| Fluxgate channels | Still connected but readings are corrupted — ignored by firmware |
| WiFi TX gate | Still disabled — WiFi could affect AD630 phase detection |
| Duration rationale | 30 ms = 450 cycles at 15 kHz, sufficient for lock-in averaging |

### Phase 3: Settling/Communications (20 ms)

**Purpose**: Transmit data, write to SD card, and allow EMI transients to settle before next fluxgate measurement.

| Action | Detail |
|--------|--------|
| EMI TX shutdown | OPA549 shutdown pin HIGH |
| WiFi TX enable | `esp_wifi_start()` or TX power restored |
| MQTT publish | Send previous cycle's measurement to broker |
| SD card write | Append CSV row for previous cycle |
| NTRIP receive | Fetch RTK corrections for GPS |
| GPS fix | Process accumulated NMEA sentences |
| Settling | 20 ms allows LM2917 output to stabilize after EMI TX shutdown |
| Duration rationale | 20 ms is sufficient for MQTT publish (<5 ms) + SD write (<10 ms) |

## ESP32 Dual-Core Architecture

### Core 1: Measurement Core (Real-Time)

Core 1 runs the time-critical measurement state machine. It must not be interrupted by networking or file I/O.

```c
// xTaskMeasurement — Core 1, Priority: configMAX_PRIORITIES - 1
void vTaskMeasurement(void *pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();

    while (1) {
        // Phase 1: Fluxgate (50 ms)
        gpio_set_level(PIN_EMI_SHUTDOWN, 1);  // EMI TX off
        xSemaphoreGive(xSemWifiOff);          // Signal Core 0 to disable WiFi

        adc_start_dma_sampling();              // Begin continuous ADC reads
        bno055_read_quaternion(&orientation);
        mlx90614_read_temperature(&ir_temp);
        vTaskDelay(pdMS_TO_TICKS(50));
        adc_stop_dma_sampling();
        compute_gradients(adc_buffer, &gradients);
        apply_tilt_correction(&gradients, &orientation);

        // Phase 2: EMI TX/RX (30 ms)
        ad9833_set_frequency(15000);
        gpio_set_level(PIN_EMI_SHUTDOWN, 0);   // EMI TX on
        adc_switch_to_emi_channels();
        adc_start_dma_sampling();
        vTaskDelay(pdMS_TO_TICKS(30));
        adc_stop_dma_sampling();
        compute_conductivity(adc_buffer, &conductivity);

        // Phase 3: Settling (20 ms)
        gpio_set_level(PIN_EMI_SHUTDOWN, 1);   // EMI TX off
        xSemaphoreGive(xSemWifiOn);           // Signal Core 0 to enable WiFi

        // Package and queue result
        MeasurementResult_t result = {
            .timestamp = ds3231_get_time(),
            .gradients = gradients,
            .conductivity = conductivity,
            .orientation = orientation,
            .ir_temperature = ir_temp,
            .gps_position = latest_gps_fix,
        };
        xQueueSend(xQueueMeasurement, &result, 0);

        vTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICKS(100));  // 10 Hz cycle
    }
}
```

### Core 0: Protocol Core (Background)

Core 0 handles all networking, file I/O, and non-time-critical tasks.

```c
// xTaskMQTT — Core 0, Priority: 5
void vTaskMQTT(void *pvParameters) {
    while (1) {
        MeasurementResult_t result;
        if (xQueueReceive(xQueueMeasurement, &result, portMAX_DELAY)) {
            mqtt_publish("geosim/pathfinder/data/raw", &result, QOS_1);
        }
    }
}

// xTaskGPS — Core 0, Priority: 5
void vTaskGPS(void *pvParameters) {
    while (1) {
        nmea_parse_available();           // Parse buffered NMEA sentences
        if (ntrip_corrections_available()) {
            zedf9p_send_rtcm(ntrip_buffer);  // Forward RTK corrections
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

// xTaskSD — Core 0, Priority: 3
void vTaskSD(void *pvParameters) {
    while (1) {
        xSemaphoreTake(xSemSDWrite, portMAX_DELAY);
        sd_append_csv_row(&latest_result);
        sd_flush();
    }
}

// xTaskStatus — Core 0, Priority: 1
void vTaskStatus(void *pvParameters) {
    while (1) {
        led_update_status();
        serial_debug_output();
        vTaskDelay(pdMS_TO_TICKS(1000));    // 1 Hz status updates
    }
}
```

## FreeRTOS Task Summary

| Task | Core | Priority | Rate | Function |
|------|------|----------|------|----------|
| xTaskMeasurement | 1 | MAX-1 | 10 Hz | TDM state machine, all sensor reads |
| xTaskMQTT | 0 | 5 | Event-driven | Publish measurements to MQTT |
| xTaskGPS | 0 | 5 | 10 Hz | NMEA parsing, NTRIP corrections |
| xTaskSD | 0 | 3 | Event-driven | SD card CSV logging |
| xTaskStatus | 0 | 1 | 1 Hz | LED, serial debug |

## Inter-Core Communication

| Mechanism | Direction | Purpose |
|-----------|-----------|---------|
| `xQueueMeasurement` | Core 1 → Core 0 | Completed measurement structs (queue depth: 4) |
| `xSemWifiOff` | Core 1 → Core 0 | Signal to disable WiFi TX |
| `xSemWifiOn` | Core 1 → Core 0 | Signal to re-enable WiFi TX |
| `latest_gps_fix` | Core 0 → Core 1 | Shared volatile struct (updated by GPS task, read by measurement task) |

## Timing Diagram

```
Time (ms):  0    10   20   30   40   50   60   70   80   90  100
            ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
Fluxgate:   ████████████████████████████
EMI TX:                                  ██████████████████
WiFi TX:                                                    ████████
SD Write:                                                   ████████
MQTT:                                                       ████████
ADC Mux:    [fluxgate channels        ] [EMI I/Q channels]
BNO055:     ██ (single read)
MLX90614:      ██ (single read)
GPS Parse:                                                  ████████
```

## Configuration Options (config.h)

```c
#define TDM_CYCLE_MS        100     // Total cycle time (10 Hz)
#define TDM_FLUXGATE_MS     50      // Phase 1 duration
#define TDM_EMI_MS          30      // Phase 2 duration
#define TDM_SETTLING_MS     20      // Phase 3 duration

#define EMI_ENABLED         1       // Set to 0 to disable EMI channel
#define WIFI_TDM_GATING     1       // Set to 0 to keep WiFi always on
#define MQTT_ENABLED        1       // Set to 0 for SD-only logging
#define LIDAR_TDM_GATING    0       // Set to 1 if LiDAR motor causes issues
```

## Settling Time Characterization

The 20 ms settling phase must be validated by bench testing:

1. **LM2917 settling after EMI TX shutdown**: The LM2917 RC filter (R1=100k, C2=0.15μF) has a time constant of 15 ms. After 20 ms settling, the residual from a step disturbance is attenuated to e^(-20/15) ≈ 26% — may need to extend settling if this is insufficient.
2. **AD9833/OPA549 shutdown transient**: The OPA549 shutdown pin provides a fast (<1 μs) disable. The AD9833 output goes to mid-rail when disabled. Combined with the LM2917 filter, the transient should be fully settled within 20 ms.
3. **WiFi TX re-enable**: `esp_wifi_start()` has variable latency (10-100 ms). For TDM gating, use `esp_wifi_set_max_tx_power(0)` / `esp_wifi_set_max_tx_power(78)` which is faster (<1 ms).

## References

- ESP-IDF FreeRTOS documentation: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/system/freertos.html
- ESP32 dual-core programming guide: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/freertos-smp.html
- ADS1115 DMA: Not natively supported over I2C; use timer-triggered polling at desired sample rate
