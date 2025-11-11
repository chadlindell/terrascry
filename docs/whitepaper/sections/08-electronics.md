# 8. Electronics (block‑level)

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

## Key Components

### DDS Sine Generator
- **Purpose:** Generate stable, frequency-agile sine waves
- **Example:** AD9833 or similar
- **Output:** 0–5 Vrms into TX coil
- **Frequency range:** 2–50 kHz

### TX Driver
- **Purpose:** Amplify DDS output to drive TX coil
- **Options:**
  - Op‑amp based (linear, low distortion)
  - Small class‑D (efficient, higher power)
- **Output:** 0–5 Vrms into coil

### RX Chain
- **LNA (Low-Noise Amplifier):** First stage amplification
- **IA (Instrumentation Amplifier):** Differential amplification, common-mode rejection
- **ADC:** Convert analog signal to digital
  - Option A: 24‑bit ADC (e.g., ADS1256) for digital lock‑in
  - Option B: Modest ADC with analog lock‑in (AD630)

### Lock‑in Detection

**Option A: Digital Lock‑in**
- Sample RX at ≥2–5 kS/s
- Demodulate at reference frequency in MCU
- Advantages: Flexible, software-configurable
- Requires: High-speed ADC, DSP capability

**Option B: Analog Lock‑in**
- Analog multiplier (e.g., AD630)
- Modest ADC for digitization
- Advantages: Lower computational load, proven design
- Requires: Analog reference signal

### ERT Measurement Chain
- **MUX:** Select which ERT ring(s) to measure
- **Diff Amp:** Differential voltage measurement
- **ADC:** Digitize voltage readings
- **Current source:** Located at base hub

### MCU (Microcontroller)
- **Options:** ESP32 or STM32
- **Functions:**
  - Control DDS frequency
  - Read ADC values
  - Implement lock‑in (if digital)
  - Handle communications
  - Manage timing/synchronization

### Synchronization
- **Simplest:** Wired reference from DDS TX to all receivers
- **Alternative:** Distribute DDS reference clock
- **Purpose:** Ensure phase coherence for MIT measurements

## Shielding & Noise Reduction

### Best Practices
- **Twist and shield** RX leads
- Keep TX/RX physically separated in capsule
- **Single-point ground** to avoid ground loops
- Use shielded cables throughout
- Minimize loop areas in wiring

### Common Issues
- **Direct coupling:** TX→RX coupling without ground interaction
  - *Fix:* Increase separation, orthogonal orientation
- **Ground loops:** Multiple ground paths causing noise
  - *Fix:* Single-point grounding, isolation where needed
- **EMI:** External electromagnetic interference
  - *Fix:* Shielding, filtering, proper grounding

## Power Management

- **Probe power:** Typically 5V or 12V
- **Base hub:** 12V battery pack (10–20 Ah)
- **Distribution:** Via cables or wireless (future)
- **Efficiency:** Low-power design extends battery life

## Communications

### Options
- **Cabled bus:** RJ45/CAT5 (reliable, low latency)
- **Wireless:** LoRa/BLE (flexible, but requires power)
- **Protocol:** Custom or standard (e.g., Modbus, CAN)

### Data Rate Requirements
- MIT: Moderate (sweep takes seconds per frequency)
- ERT: Low (voltage readings every 1–2 seconds)
- Total: <100 kbps per probe typically sufficient

