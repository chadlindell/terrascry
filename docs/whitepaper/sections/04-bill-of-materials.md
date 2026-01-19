# 4. Bill of Materials

## Overview

This section provides comprehensive bills of materials (BOMs) for building the HIRT system, including per-probe components, base hub components, and shared equipment. The micro-probe design (16mm OD, passive probes) minimizes cost while maintaining measurement quality.

---

## Cost Summary

### System Cost Estimate (25-Probe Array)

| Category | Cost Range | Notes |
|----------|------------|-------|
| Probes (25 units) | $1,500-2,500 | Passive design |
| Base Hub | $300-450 | Complete unit |
| Cables/Connectors | $400-600 | All probe cables |
| Tools/Equipment | $500-1,000 | One-time purchase |
| Consumables | $100-200 | Initial stock |
| **Total** | **$2,800-4,750** | |

### Cost Per Probe Breakdown

| Component | Cost |
|-----------|------|
| Mechanical (rod, tip, coupler) | $40-60 |
| ERT (rings, collars) | $5-10 |
| Coils (ferrite, wire) | $10-15 |
| Cable/Connector | $15-25 |
| Hardware (epoxy, o-rings) | $5-10 |
| **Total per probe** | **$75-120** |

### Cost Reduction Options

1. **Bulk ordering:** 20%+ savings on quantities >50
2. **Local sourcing:** Reduce shipping costs
3. **Simpler design:** Passive probes vs active
4. **DIY coils:** Wind your own vs purchased
5. **Generic parts:** Where precision not critical

---

## Per-Probe BOM

### Mechanical Components

| Ref | Component | Description | Qty | Unit Cost | Notes |
|-----|-----------|-------------|-----|-----------|-------|
| ROD1 | Fiberglass Tube | 16mm OD x 12mm ID x 1.5m | 2 | $15-25 ea | Rod sections |
| TIP1 | Probe Tip | 3D printed PETG | 1 | $1-2 | Nose cone |
| CPL1 | Rod Coupler | 3D printed/CNC | 1 | $2-5 | Joins sections |
| JB1 | Junction Box | 3D printed PETG | 1 | $3-5 | Surface enclosure |

### Coil Components

| Ref | Component | Description | Qty | Unit Cost | Notes |
|-----|-----------|-------------|-----|-----------|-------|
| L1 | Ferrite Rod | 6-8mm x 40-80mm MnZn | 1-2 | $2-5 | TX/RX cores |
| W1 | Magnet Wire | 28-32 AWG, 10m | 1 | $3-5 | For coil winding |

### ERT Components

| Ref | Component | Description | Qty | Unit Cost | Notes |
|-----|-----------|-------------|-----|-----------|-------|
| R1-R3 | ERT Rings | Stainless steel 3-5mm bands | 2-3 | $1-2 ea | Electrodes |
| C1-C3 | Ring Collars | 3D printed PETG | 2-3 | $0.50 ea | Ring mounts |

### Hardware

| Ref | Component | Description | Qty | Unit Cost | Notes |
|-----|-----------|-------------|-----|-----------|-------|
| HW1 | O-Rings | M12 size, nitrile | 4 | $0.50 ea | Sealing |
| HW2 | Epoxy | 2-part structural | - | $5/probe | Assembly |

### Cable and Connectors

| Ref | Component | Description | Qty | Unit Cost | Notes |
|-----|-----------|-------------|-----|-----------|-------|
| CBL1 | Shielded Cable | 6-conductor, 3-5m | 1 | $10-15 | Probe to hub |
| CON1 | Connector | 12-pin Phoenix | 1 | $5-8 | Hub connection |

**Total Per Probe (Passive): ~$60-100**

### Active Probe Electronics (Optional)

If building active probes with in-probe electronics:

| Ref | Component | Part Number | Qty | Unit Cost |
|-----|-----------|-------------|-----|-----------|
| U1 | MCU | ESP32-WROOM-32 | 1 | $5-8 |
| U2 | DDS | AD9833BRMZ | 1 | $8-12 |
| U3 | TX Op-Amp | OPA454AIDDAR | 1 | $6-10 |
| U4 | RX Preamp | AD620ARZ | 1 | $6-10 |
| U5 | Inst Amp | INA128PAG4 | 1 | $6-10 |
| U6 | ADC | ADS1256IDBR | 1 | $10-15 |
| U7 | Mux | CD4051BE | 1 | $1-2 |
| U8 | LDO | AMS1117-3.3 | 1 | $0.50 |
| PCB | Custom PCB | - | 1 | $5-10 |
| | Passives | Resistors, caps | - | $5 |

**Additional per Active Probe: ~$50-80**

---

## Base Hub BOM

### Power System

| Ref | Component | Part Number | Qty | Unit Cost | Notes |
|-----|-----------|-------------|-----|-----------|-------|
| BAT1 | Battery | 12V 12Ah LiFePO4 | 1 | $60-100 | Main power |
| F1 | Fuse Holder | 0287005.PXCN | 1 | $3 | Panel mount |
| F2 | Fuse | 5A fast-blow | 5 | $1 ea | Spares |
| REG1 | 5V Regulator | LM2596 Module | 1 | $3-5 | Buck converter |
| REG2 | 3.3V Regulator | AMS1117-3.3 | 1 | $0.50 | LDO |
| SW1 | Power Switch | DPST 10A | 1 | $3-5 | Main switch |
| TB1 | Terminal Block | Multi-position | 1 | $10-15 | Distribution |

### ERT Current Source

| Ref | Component | Part Number | Qty | Unit Cost |
|-----|-----------|-------------|-----|-----------|
| U1 | Voltage Ref | REF5025AIDGKR | 1 | $4-6 |
| U2 | Op-Amp | OPA277PAG4 | 1 | $4-6 |
| U3 | Inst Amp | INA128PAG4 | 1 | $6-10 |
| K1 | Relay | G5V-2-H1 | 1 | $3-5 |
| R1-R4 | Precision R | 0.1% various | 10 | $0.50 ea |
| R5 | Sense R | 10 ohm 0.1% | 1 | $1 |

### Sync/Clock Distribution

| Ref | Component | Part Number | Qty | Unit Cost |
|-----|-----------|-------------|-----|-----------|
| Y1 | Oscillator | ECS-100-10-30B-TR | 1 | $3-5 |
| U1-U3 | Buffer | SN74HC244N | 3 | $1 ea |

### Communication

| Ref | Component | Part Number | Qty | Unit Cost |
|-----|-----------|-------------|-----|-----------|
| U1 | RS485 | MAX485ESA+ | 1 | $2-4 |
| U2 | USB-Serial | CP2102 Module | 1 | $3-5 |
| J1 | RJ45 Jack | - | 1 | $2 |

### Control

| Ref | Component | Part Number | Qty | Unit Cost |
|-----|-----------|-------------|-----|-----------|
| U1 | MCU | ESP32 DevKit | 1 | $8-12 |
| U2 | ADC | ADS1256IDBR | 1 | $10-15 |
| SD1 | SD Card | Micro SD module | 1 | $3-5 |

### Enclosure and Connectors

| Ref | Component | Description | Qty | Unit Cost |
|-----|-----------|-------------|-----|-----------|
| ENC1 | Enclosure | IP65 200x150x100mm | 1 | $30-50 |
| PG1-PG20 | Cable Glands | PG11 or M20 | 20 | $1 ea |
| CON1-CON20 | Probe Connectors | 12-pin Phoenix | 20 | $5 ea |

**Total Base Hub: ~$300-450**

---

## Shared Equipment BOM

### Connectors and Cables

| Ref | Component | Description | Qty | Unit Cost |
|-----|-----------|-------------|-----|-----------|
| CBL1 | Probe Cable | Belden 3066A 12-pair, 5m | 20 | $15 ea |
| CBL2 | Power Cable | 14 AWG 2-conductor | 10m | $10 |
| CON1 | Phoenix Headers | 12-pos pluggable | 20 | $5 ea |
| CON2 | DC Jack | 5.5x2.1mm panel | 1 | $2 |

### Test Equipment (Recommended)

| Item | Description | Est. Cost | Notes |
|------|-------------|-----------|-------|
| DMM | Digital Multimeter | $50-100 | Fluke or equivalent |
| LCR | LCR Meter | $100-300 | For coil testing |
| Scope | Oscilloscope | $300-500 | 2-ch, 50MHz min |
| PS | Bench Power Supply | $50-100 | Adjustable, current limit |

### Tools

| Item | Description | Est. Cost |
|------|-------------|-----------|
| Soldering | Iron + solder | $50-100 |
| Tap/Die | M12x1.75 set | $30-50 |
| Crimpers | For connectors | $30-50 |
| Heat Gun | For shrink tubing | $30-50 |
| Hand Tools | Screwdrivers, pliers | $50 |

### Consumables

| Item | Description | Est. Cost |
|------|-------------|-----------|
| Solder | 60/40 or lead-free | $15 |
| Flux | Rosin flux | $10 |
| Heat Shrink | Assorted sizes | $15 |
| Epoxy | 2-part structural | $20 |
| Cable Ties | Assorted | $10 |
| IPA | Isopropyl alcohol | $10 |

---

## Procurement Guide

### Recommended Suppliers

**Electronics:**
- DigiKey (www.digikey.com) - Wide selection, fast shipping
- Mouser (www.mouser.com) - Good for precision components
- Newark (www.newark.com) - Alternative source

**Mechanical:**
- McMaster-Carr (www.mcmaster.com) - Hardware, tubing
- Grainger (www.grainger.com) - Industrial supplies
- Amazon - General supplies

**3D Printing:**
- Local print shop
- Shapeways (www.shapeways.com) - Online service
- JLCPCB (www.jlcpcb.com) - Also offers 3D printing

**PCB Fabrication:**
- JLCPCB (www.jlcpcb.com) - Low cost, fast
- PCBWay (www.pcbway.com) - Good quality
- OSH Park (oshpark.com) - US-based, quality

### Key Part Numbers Reference

| Component | DigiKey PN | Mouser PN |
|-----------|------------|-----------|
| AD9833BRMZ | AD9833BRMZ-REEL | 584-AD9833BRMZ |
| AD620ARZ | AD620ARZ-ND | 584-AD620ARZ |
| INA128PAG4 | INA128PAG4-ND | 595-INA128PAG4 |
| ADS1256IDBR | ADS1256IDBR-ND | 595-ADS1256IDBR |
| OPA454AIDDAR | OPA454AIDDAR-ND | 595-OPA454AIDDAR |
| REF5025AIDGKR | REF5025AIDGKR-ND | 595-REF5025AIDGKR |
| ESP32-WROOM-32 | 1904-1009-1-ND | 356-ESP32-WROOM-32 |

### Procurement Tips

1. **Order extras:** Add 10-20% for spares/mistakes
2. **Check MOQ:** Some parts have minimum order quantities
3. **Lead times:** Check availability before ordering
4. **Substitutes:** Have backup part numbers identified
5. **Consolidate:** Combine orders to reduce shipping costs

---

## Alternative Components

### Coil Alternatives

| Original | Alternative | Notes |
|----------|-------------|-------|
| 6-8mm ferrite rod | 10mm rod | Larger = more signal, larger diameter |
| 30 AWG magnet wire | 28-34 AWG | Trade-off: turns vs resistance |

### ERT Ring Alternatives

| Original | Alternative | Notes |
|----------|-------------|-------|
| Stainless steel band | Copper tape | Lower cost, easier to work with |
| 3D printed collar | Heat shrink tube | Simpler mounting |

### Electronics Alternatives

| Original | Alternative | Notes |
|----------|-------------|-------|
| AD9833 DDS | Si5351 | More outputs, different interface |
| AD620 preamp | INA217 | Different specifications |
| ADS1256 ADC | ADS1115 | Lower resolution, lower cost |

---

*For assembly procedures, see Section 7: Assembly and Wiring. For mechanical specifications, see Section 5: Mechanical Design.*
