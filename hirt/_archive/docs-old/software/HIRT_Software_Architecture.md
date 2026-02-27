# HIRT System Software Architecture Specification

**Version:** 0.1 (Draft)
**Date:** 2026-01-29
**Status:** Planning Phase

---

## 1. System Overview

The HIRT software stack is a distributed control system consisting of two primary tiers:
1.  **Real-Time Firmware (ESP32):** Handles microsecond-precision signal generation (DDS), acquisition (ADC), and digital signal processing (Lock-In).
2.  **Field Controller (Raspberry Pi/Tablet):** Handles survey orchestration, data storage, user interface, and preliminary inversion.

## 2. Firmware Architecture (ESP32)

**Framework:** ESP-IDF (Espressif IoT Development Framework)
**Language:** C/C++
**Rationale:** Required for dual-core task pinning and deterministic interrupt handling needed for software-defined Lock-In Amplification.

### 2.1 Core Responsibilities
- **Signal Generation:** Control AD9833 via high-speed SPI (VSPI).
- **Data Acquisition:** Capture ADS1256 data at 30 kSPS via SPI (HSPI).
- **DSP:** Implement digital Lock-In Amplifier (LIA) to extract amplitude/phase at 2-50 kHz.
- **Multiplexing:** Control CD4051 GPIOs to switch probe pairs.

### 2.2 Task Allocation
| Task Name | Core | Priority | Frequency | Description |
|-----------|------|----------|-----------|-------------|
| `task_adc_isr` | 1 | High (ISR) | 30 kHz | Reads 24-bit sample from ADS1256 DRDY interrupt. |
| `task_dsp_lia` | 1 | High | 30 kHz | Multiplies sample by Ref_Sin/Ref_Cos look-up tables. |
| `task_dds_mgr` | 0 | Medium | On-Change | Updates AD9833 frequency/phase. |
| `task_mux_mgr` | 0 | Low | 1-2 Hz | Switches CD4051 channels for next pair. |
| `task_comms` | 0 | Low | 10 Hz | Packets averaged data and sends to Pi via UART/WiFi. |

### 2.3 Data Protocol (UART)
**Baud Rate:** 921,600 baud
**Packet Structure (Binary):**
```c
struct DataPacket {
    uint8_t sync_byte;      // 0xAA
    uint8_t msg_type;       // 0x01 = Measurement, 0x02 = Status
    uint16_t tx_id;         // Probe transmitting
    uint16_t rx_id;         // Probe receiving
    uint32_t freq_hz;       // Measurement frequency
    float    amplitude;     // Calculated amplitude
    float    phase;         // Calculated phase (radians)
    float    noise_floor;   // StdDev of measurement
    uint16_t crc16;         // Checksum
};
```

---

## 3. Field Controller Architecture (Raspberry Pi 4/5)

**OS:** Linux (Raspberry Pi OS Lite)
**Runtime:** Python 3.11+
**Rationale:** Rich ecosystem for science (NumPy, SciPy) and geophysics (SimPEG, pyGIMLi).

### 3.1 Software Modules
1.  **`hirt-daemon`:** Background service that manages the serial connection to the ESP32.
    *   Auto-detects ESP32 port.
    *   Buffers incoming binary packets.
    *   Writes raw data to HDF5.
2.  **`hirt-survey-mgr`:** The "Brain" of the operation.
    *   Loads `survey_config.json` (grid size, spacing).
    *   Generates the "Scan Plan" (list of TX-RX pairs).
    *   Commands the ESP32 to execute the plan.
3.  **`hirt-ui-server`:** Lightweight Web Server (FastAPI).
    *   Hosts a mobile-friendly Web UI.
    *   Allows operator to Start/Stop scans, view live heatmaps, and download data.
    *   **Why Web UI?** Allows control via any smartphone/tablet connected to the Pi's WiFi hotspot. No app installation needed.

### 3.2 Data Storage Strategy
*   **Metadata:** SQLite (`survey.db`)
    *   Tables: `Sites`, `Surveys`, `Probes`, `Events`
*   **Bulk Data:** HDF5 (`survey_data.h5`)
    *   Hierarchical structure: `/survey_id/frequency/tx_id/rx_id`
    *   Stores raw time-series I/Q data (optional) and averaged final values.
*   **Export:** CSV exporter provided for legacy compatibility (Excel/Matlab).

---

## 4. Inversion Pipeline (Post-Processing)

**Libraries:**
*   **SimPEG:** For primary MIT/ERT inversion.
*   **pyGIMLi:** Alternative cross-hole ERT inversion.

**Workflow:**
1.  **Pre-processing:** `hirt-process.py` reads HDF5, applies calibration factors, and filters outliers (reciprocity check).
2.  **Mesh Generation:** Dynamically builds a 3D Tetrahedral mesh based on GPS coordinates of probes.
3.  **Inversion:** Runs `InversionDirective` (SimPEG).
    *   *Note:* On RPi 4, this may take minutes. On RPi 5/Laptop, seconds.
4.  **Visualization:** Exports `.vtu` (Paraview) or slices to `.png` for the Web UI.

---

## 5. Development Roadmap

### Phase 1: The "Mock" System (Week 1-2)
- [ ] Write `hirt-daemon` in Python.
- [ ] Create a "Virtual ESP32" script that spits out fake binary packets.
- [ ] Verify the HDF5 storage and CSV export.

### Phase 2: Firmware Core (Week 3-4)
- [ ] Set up ESP-IDF environment.
- [ ] Implement `task_dds_control` (SPI).
- [ ] Implement `task_adc_read` (SPI + ISR).
- [ ] Verify 30kSPS throughput on bench.

### Phase 3: The "Lock-In" (Week 5-6)
- [ ] Implement fixed-point math LIA (Lock-In Amplifier).
- [ ] Test linearity and dynamic range on signal generator.

### Phase 4: Integration (Week 7+)
- [ ] Connect Real ESP32 to Real Pi.
- [ ] Run full "Scan Plan" on bench with resistor phantom.
