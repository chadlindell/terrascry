# 8. Electronics - Central Electronics Hub

## Design Philosophy

**Centralized Electronics:** All electronics are at surface in central hub. Probes are passive (coils and electrodes only). One hub serves all probes (20–24 typical).

## System Block Diagram

### Old Design (Deprecated - Electronics in Each Probe)
```
[NOT USED - Each probe had its own electronics]
```

### New Design (Micro-Probe - Central Electronics)

```
Surface - Central Electronics Hub:
┌─────────────────────────────────────────┐
│     Central Electronics Hub             │
│                                         │
│  DDS → TX Amp ──┐                      │
│                 │                       │
│  TX MUX ────────┼──→ Select Probe TX   │
│                 │                       │
│  RX MUX ←───────┼──→ Select Probe RX   │
│                 │                       │
│  RX Chain ←─────┼──→ ADC → MCU         │
│  (LNA, IA)      │                       │
│                 │                       │
│  ERT Current ───┼──→ MUX → Diff Amp  │
│  Source         │      ↓                │
│                  │    ADC → MCU         │
│                  │                       │
│  Sync/Clock ─────┼───────────────────   │
│  Power ──────────┼───────────────────   │
│  Comms ──────────┼───────────────────   │
└──────────────────┼──────────────────────┘
                   │
        Multi-Probe Cable Harness
                   │
        ┌──────────┴──────────┐
        │                     │
    ┌───▼───┐            ┌───▼───┐
    │ Probe │            │ Probe │
    │   1   │            │   2   │
    └───┬───┘            └───┬───┘
        │                    │
Downhole (Passive):
    ┌───┴───┐            ┌───┴───┐
    │  TX   │            │  TX   │
    │ Coil  │            │ Coil  │
    │       │            │       │
    │  RX   │            │  RX   │
    │ Coil  │            │ Coil  │
    │       │            │       │
    │ Ring  │            │ Ring  │
    │ Ring  │            │ Ring  │
    └───────┘            └───────┘
```

## Key Components (Central Hub)

### DDS Sine Generator
- **Purpose:** Generate stable, frequency-agile sine waves
- **Component:** AD9833 or similar
- **Output:** 0–5 Vrms (drives all probe TX coils via multiplexer)
- **Frequency range:** 2–50 kHz (lower frequencies preferred for micro-probes)
- **Location:** Central hub (one for all probes)

### TX Driver and Multiplexer
- **TX Driver:** Op-amp based (OPA2277) - amplifies DDS output
- **TX Multiplexer:** CD4051 or similar - selects which probe TX to drive
- **Output:** 0–5 Vrms into selected probe TX coil
- **Purpose:** Drive one probe TX at a time, measure all RX

### RX Chain and Multiplexer
- **RX Multiplexer:** CD4051 or similar - selects which probe RX to read
- **LNA (Low-Noise Amplifier):** AD620 - first stage amplification
- **IA (Instrumentation Amplifier):** INA128 - differential amplification, common-mode rejection
- **ADC:** 24-bit ADS1256 for digital lock-in
- **Purpose:** Receive from selected probe RX, amplify, digitize

### Lock-in Detection

**Option A: Digital Lock-in (Recommended)**
- Sample RX at ≥2–5 kS/s
- Demodulate at reference frequency in MCU
- Advantages: Flexible, software-configurable, good for micro-probes
- Requires: High-speed ADC, DSP capability

**Option B: Analog Lock-in**
- Analog multiplier (AD630)
- Modest ADC for digitization
- Advantages: Lower computational load, proven design
- Requires: Analog reference signal

### ERT System

**Current Source:**
- Precision op-amp (OPA177) with voltage reference (REF5025)
- Programmable current: 0.5–2 mA
- Polarity reversal (DPDT relay) for DC measurements
- Located in central hub

**Voltage Measurement:**
- Multiplexer: Select which probe/ring pairs to measure
- Differential amplifier: INA128 for high precision
- ADC: 24-bit ADS1256 for voltage readings
- Purpose: Measure voltage between ring pairs

### MCU (Microcontroller)
- **Component:** ESP32 or STM32
- **Functions:**
  - Control DDS frequency
  - Control multiplexers (TX, RX, ERT)
  - Read ADC values
  - Implement digital lock-in (if Option A)
  - Handle communications
  - Manage timing/synchronization
  - Data logging

### Synchronization
- **Method:** Wired reference from DDS TX to all receivers
- **Alternative:** Distribute DDS reference clock
- **Purpose:** Ensure phase coherence for MIT measurements
- **Distribution:** Via multi-probe cable harness

## Probe Components (Passive)

### MIT Coils
- **TX Coil:** Ferrite core (6–8 mm) with many turns (200–400)
- **RX Coil:** Ferrite core (6–8 mm) with many turns (200–400)
- **Wire:** 34–38 AWG fine wire
- **Connection:** Thin cable to central hub
- **No electronics:** Just passive coils

### ERT Rings
- **Rings:** Narrow bands (3–5 mm) of stainless steel or copper
- **Connection:** Thin twisted pair to central hub
- **No electronics:** Just passive electrodes

### Surface Junction Box
- **Function:** Terminal block for connections
- **No electronics:** Just wiring connections
- **Purpose:** Organize probe connections, strain relief

## Shielding & Noise Reduction

### Best Practices
- **Twist and shield** all signal cables
- **Single-point ground** to avoid ground loops
- Use **shielded cables** throughout multi-probe harness
- Minimize **loop areas** in wiring
- **Separate TX and RX cables** in harness
- **Proper termination** of shields

### Common Issues
- **Direct coupling:** TX→RX coupling (mitigate with orthogonal coils)
- **Ground loops:** Multiple ground paths (single-point ground)
- **EMI:** External electromagnetic interference (shielding, filtering)
- **Cable crosstalk:** Multiple probes in same harness (proper cable routing)

## Power Management

### Central Hub Power
- **Battery:** 12V 12Ah SLA or LiFePO4
- **Regulators:** 12V → 5V (LM2596), 5V → 3.3V (AMS1117)
- **Distribution:** Via multi-probe harness
- **Efficiency:** Centralized power management

### Probe Power
- **No power needed in probe** (passive design)
- Power only needed for:
  - Cable drivers (if long cables)
  - Optional buffer amps (if needed)
- Much lower power per probe than old design

## Communications

### Options
- **Cabled bus:** RS485 over CAT5 (reliable, low latency)
- **Ethernet:** Direct Ethernet connection (if using RPi4)
- **Wireless:** LoRa/BLE (optional, flexible)
- **Protocol:** Custom or standard (e.g., Modbus)

### Data Rate Requirements
- **MIT:** Moderate (sweep takes seconds per frequency)
- **ERT:** Low (voltage readings every 1–2 seconds)
- **Total:** <100 kbps per probe typically sufficient
- **Central hub:** Handles all probes sequentially

## Multi-Probe Architecture

### Cable Harness
- **TX cables:** Connect central TX driver to all probe TX coils
- **RX cables:** Connect all probe RX coils to central RX chain
- **ERT cables:** Connect all probe rings to central ERT system
- **Sync/Clock:** Distribute sync signal to all probes
- **Power:** Distribute power (if needed)

### Measurement Sequence
1. Select probe TX (via TX multiplexer)
2. Drive TX at desired frequency
3. Select probe RX (via RX multiplexer)
4. Measure RX signal (amplify, digitize, lock-in)
5. Repeat for all TX/RX pairs
6. Switch to ERT measurements
7. Select probe/ring pairs (via ERT multiplexer)
8. Inject current, measure voltage
9. Repeat for all ring pairs

## Advantages of Centralized Design

1. **Lower Cost:** One set of electronics vs many
2. **Easier Maintenance:** All electronics accessible
3. **Better Reliability:** Passive probes more robust
4. **Easier Updates:** Firmware updates in one place
5. **Better Power Management:** Centralized power control
6. **Simpler Probes:** No electronics to fail in probe

## Trade-offs

1. **More Cables:** Multi-probe harness needed
2. **Central Failure Point:** Hub failure affects all probes (mitigate with redundancy)
3. **Cable Management:** More cables to manage in field
4. **Sequential Measurement:** Measure one probe at a time (acceptable for most applications)

---

*For mechanical assembly, see Section 7. For BOM, see Section 6.*
