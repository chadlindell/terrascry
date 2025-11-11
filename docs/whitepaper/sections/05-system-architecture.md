# 5. System Architecture

## 5.1 Probe Overview (identical units)

Each probe carries:

### MIT Coil Set
- **1× TX coil** + **1× RX coil** (or shared coil in time‑division)
- Mounted on ferrite cores
- Orthogonal orientation or slightly separated to reduce direct coupling

### ERT‑Lite Rings
- Two stainless/copper bands at fixed depths
- Standard positions: **0.5 m & 1.5 m** from tip
- Add deeper ring at **3.0 m** for longer rods

### Electronics Pod
- **DDS sine source** (e.g., AD9833)
- **Low‑noise RX chain** (preamp + instrumentation amp)
- **ADC/lock‑in** (digital or analog)
- **MCU** (ESP32 or STM32)
- **Power/bus connectors**

### Rod
- **Fiberglass or PVC** segments
- Threaded couplers for length adjustment
- **Metal pilot rod** used only to make the hole (removed before sensor insertion)

## 5.2 Base/Hub

### Sync/Timebase
- Simple wired sync line or PPS clock shared to all probes
- Ensures phase coherence for MIT measurements
- Enables synchronized data collection

### Power
- **12 V or 5 V battery pack(s)**
- Capacity: 10–20 Ah for field operations
- Distribution via cables or wireless power (future)

### Communications
- **Cabled bus** (RJ45/CAT5) for reliable data transfer
- **Short‑range wireless** (LoRa/BLE) option depending on site conditions
- Data logging to tablet/computer

## 5.3 Frequency Plan

### MIT Sweeps
- Typical frequencies: **~2, 5, 10, 20, 50 kHz**
- Choose 3–5 points based on depth/resolution requirements
- Lower frequencies for deeper targets
- Higher frequencies for near-surface detail

### ERT Configuration
- **DC with polarity reversal** (e.g., every 1–2 s)
- **Low‑freq AC** option (e.g., 8–16 Hz) to reduce polarization
- Current levels: 0.5–2 mA

## System Block Diagram

```
           ┌──────────────────────────────┐
           │  Probe Electronics (each)    │
DDS → TX Amp → TX Coil  ─┐                │
                         │ magnetic field │
RX Coil → LNA → IA → ADC → MCU (ESP32) → Comms/Sync

ERT Rings → MUX → Diff Amp → ADC → MCU → Comms/Sync

           └──────────────────────────────┘
Base Hub: Battery | Sync/Clock | ERT Current Source | Logger/Tablet
```

