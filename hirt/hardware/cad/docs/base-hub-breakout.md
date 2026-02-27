# Base Hub Backplane & Harness Breakout

This document captures the physical decomposition of the centralized HIRT base hub so hardware and wiring can be fabricated consistently across builds. It complements the electrical BOM and the system architecture write-up.

## 1. Stack Overview

```
┌─────────────────────────────┐
│   Weatherproof enclosure    │  Bud NBF-32016 or similar
├─────────────────────────────┤
│  Front panel (glands, UI)   │  Probe harness bulkhead, RJ45, power
├─────────────────────────────┤
│  Backplane PCB (160×120 mm) │  DDS, TX/RX chain, ERT source, MCU
├─────────────────────────────┤
│  Harness strain relief bar  │  Latches shielded cables + labels
├─────────────────────────────┤
│  Power shelf (battery/fuse) │  LiFePO₄ pack, fuse block, charger lead
└─────────────────────────────┘
```

- **Segregated grounds:** the backplane zones route analog front-ends on the left, digital/MCU on the right, and power across the rear in a star topology to keep MIT/ERT measurements quiet.
- **Field-serviceable harness:** each probe plugs into a removable Phoenix Contact 12-position terminal block; replacements require no soldering in the field.

## 2. Backplane Zones

| Zone | Function | Key Parts |
|------|----------|-----------|
| DDS / TX Driver | Generates sweep (2–50 kHz) and pushes selected TX coil | AD9833, OPA454, CD4051 TX mux, sense resistors |
| RX Front End | Gains/filters return signals before lock-in | Dual AD620 → INA128 stages, ADG706 mux |
| ERT Current Source | Push-pull injection with polarity reversal | OPA177, REF5025, RN73 sense resistor, G5V-2 relays |
| Lock-In / ADC | Digitizes MIT amplitude/phase + ERT voltages | ADS1256 (or AD630 + ADS1115 option) |
| Control / Sync | Schedules channel matrix, logs data | ESP32 DevKit, MAXM22511 isolated RS-485, SN74HC244 clock buffer |
| Power | Regulates ±12 V analog, +5 V, +3.3 V rails | PDQE10 DC‑DC, LM2596 buck, AP7361 LDO, fuse block |

## 3. Probe Harness Connector

Each probe terminates in a 12-pin plug (Phoenix Contact 1757248) that mates with a pluggable header on the backplane.

| Pin | Signal | Notes |
|-----|--------|-------|
| 1 | TX+ | From DDS/TX driver through probe TX coil |
| 2 | TX− | Return path; twisted pair with Pin 1 |
| 3 | RX+ | Differential RX coil lead |
| 4 | RX− | Differential RX coil return |
| 5 | Guard / Shield | Foil/drain wire tied to hub analog ground near entry |
| 6 | Ring A | Upper ERT electrode (0.5 m) |
| 7 | Ring B | Mid ERT electrode (1.5 m) |
| 8 | Ring C (optional) | Deep electrode (2.5 m) or spare |
| 9 | ID Sense | Future option (resistor ladder for auto-ID) |
| 10 | Spare Diff+ | Reserved for future sensors |
| 11 | Spare Diff− | Reserved for future sensors |
| 12 | Cable shield clamp | Mechanically clamped to strain relief bar |

### Cable Spec
- **Type:** Belden 3066A (12 shielded pairs) trimmed to 3–5 m per probe.
- **Labeling:** Heat-shrink ID near both ends (`P01`, `P02`, …) + ring color bands.
- **Strain relief:** Cable clamp plate immediately behind panel glands to avoid stressing PCB headers.

## 4. Front Panel Layout

1. **Left column:** three M20 glands for probe bundle quadrants (8 probes per gland). Each gland feeds a strain relief comb before reaching backplane headers.
2. **Center column:** RJ45 (sync/data), USB-C (tablet tether), and optional SMA for GPS PPS.
3. **Right column:** DC jack for charger, Anderson SB-50 for battery swap, status LEDs (Power, TX active, Fault), and a recessed reset switch.

Panel drawings reside in `hardware/cad/step` (DXF/STEP pending) and should be printed at 1:1 before drilling the enclosure.

## 5. Test & Bring-Up Checklist

- [ ] **Continuity:** Verify each probe header pin maps to the correct Phoenix plug using a breakout harness before installing the PCB.
- [ ] **Isolation:** Measure >1 MΩ between TX bundle, RX bundle, and shield to confirm no nicked insulation.
- [ ] **TX sweep:** With dummy load installed, sweep 2–50 kHz and confirm current sense matches expected amplitude (log data via ESP32).
- [ ] **RX noise:** Short RX inputs, capture 60 s baseline, ensure noise floor <1% of full scale.
- [ ] **ERT current:** Inject into 1 kΩ dummy load, reverse polarity every 2 s, verify 0.5–2 mA settings.
- [ ] **Harness stress:** Tug-test each cable with 10 lb force; ensure strain relief takes load, not the PCB header.

## 6. Files & Next Steps

- **PCB:** `hardware/cad/step/backplane_160x120.step` *(placeholder – to be generated after schematic capture)*.
- **Panel DXF:** `hardware/cad/step/base_hub_panel.dxf` *(pending)*.
- **Harness schematic:** `docs/whitepaper/sections/08-electronics.md` (update once PCB netlist is finalized).

Contributions: open a PR with updated drawings or Gerbers and append an entry to `docs/IP-LOG.md` if it constitutes a new disclosure.

