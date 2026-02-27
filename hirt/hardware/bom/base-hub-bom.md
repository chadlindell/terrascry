# Base Hub BOM (Bill of Materials) - Central Electronics Hub

**Central Electronics Hub** - Estimated cost: $300–600 (serves all probes)

## Design Note

**Centralized Electronics:** All electronics are at surface in central hub. Probes are passive (see [Probe BOM](probe-bom.md)). One hub serves all probes (20–24 typical).

## MIT Driver/Receiver System (2026 Modernized)

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| DDS generator | Low-power DDS | 1 | AD9837BCPZ-REEL7 | Digi-Key | $8.00 | $8.00 | 8.5mW low-power upgrade (LFCSP) |
| TX driver | Precision Op-amp | 1 | OPA2186 | Digi-Key | $2.50 | $2.50 | Zero-drift, 90uA power (90% reduction) |
| RX preamp | Low-noise In-Amp | 1 | AD8421ARZ | Digi-Key | $9.50 | $9.50 | 3 nV/sqrt(Hz) noise floor (3x better) |
| Instrumentation amp | INA128 | 1 | INA128PAG4 | Digi-Key | $7.50 | $7.50 | Precision instrumentation amp |
| Lock-in / ADC | Integrated 24-bit AFE | 1 | AD7124-8 | Digi-Key | $12.50 | $12.50 | Integrated PGA + Ref + ADC (low power) |
| Lock-in (Legacy) | Analog multiplier AD630 | 0 | AD630ANZ | Digi-Key | $12.00 | $0.00 | Deprecated in favor of digital lock-in |
| TX multiplexer | Analog mux for TX selection | 1 | CD4051BE | Digi-Key | $1.50 | $1.50 | Select which probe TX to drive |
| RX multiplexer | Analog mux for RX selection | 1 | CD4051BE | Digi-Key | $1.50 | $1.50 | Select which probe RX to read |
| Probe Connectors | IP68 Circular 26-pin | 20 | Weipu SP29-26 | AliExpress | $8.00 | $160.00 | Waterproof replacement for DB25 |
| **Subtotal MIT System** | | | | | | **$203.00** | Includes ruggedized connectors |

## ERT System

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Current source op-amp | Precision op-amp OPA177 | 1 | OPA177GP | Digi-Key | $4.50 | $4.50 | Current source amplifier |
| Voltage reference | REF5025 2.5V | 1 | REF5025AIDGKT | Digi-Key | $3.50 | $3.50 | Precision 2.5V reference |
| Current sense resistor | 1.25kΩ 0.1% precision | 1 | RN73R2BTTD1251B10 | Digi-Key | $1.50 | $1.50 | For 2mA full scale |
| Polarity switch | DPDT relay 5V | 1 | G5V-2-5DC | Digi-Key | $3.50 | $3.50 | Polarity reversal |
| Relay driver | Transistor/MOSFET | 1 | 2N7002 | Digi-Key | $0.25 | $0.25 | Drive relay coil |
| Differential voltmeter | INA128 precision | 1 | INA128PAG4 | Digi-Key | $7.50 | $7.50 | High-precision diff amp |
| ERT ADC | 24-bit ADS1256 | 1 | ADS1256IDBR | Digi-Key | $18.50 | $18.50 | High resolution ADC |
| ERT multiplexer | CD4051 8-channel | 2–3 | CD4051BE | Digi-Key | $1.50 | $3–4.50 | Channel selection (ring pairs) |
| **Subtotal ERT System** | | | | | | **$43.25–45.75** | |

## Control and Data Acquisition

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| MCU | ESP32 DevKit | 1 | ESP32-DEVKITC-32E | Adafruit/SparkFun | $10.00 | $10.00 | WiFi/Bluetooth, or ESP32-WROOM-32 |
| Voltage regulator | 3.3V LDO AMS1117 | 2 | AMS1117-3.3 | Digi-Key | $0.50 | $1.00 | 3.3V regulator |
| Voltage regulator | 5V LDO AMS1117 | 2 | AMS1117-5.0 | Digi-Key | $0.50 | $1.00 | 5V regulator |
| PCB | Perfboard or custom | 1 | — | Custom | $5–15 | $5–15 | Larger board for all electronics |
| Connectors | Headers, sockets | 1 set | Various | Digi-Key | $5–10 | $5–10 | 2.54mm pitch headers |
| **Subtotal Control** | | | | | | **$22–36** | |

## Sync/Clock Distribution

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Crystal oscillator | 10 MHz oscillator | 1 | ECS-100-10-30B-TR | Digi-Key | $2.50 | $2.50 | Reference clock (or use DDS) |
| Clock buffer | 74HC244 octal buffer | 1–2 | SN74HC244N | Digi-Key | $1.50 | $1.50–3 | Drive multiple outputs |
| Sync cables | Thin shielded cable | 20–24 | Custom | Digi-Key/Mouser | $1–2 | $20–48 | Sync distribution to probes |
| Connectors | Small connectors | 20–24 | Various | Digi-Key | $0.50 | $10–12 | For sync cables |
| **Subtotal Sync** | | | | | | **$34–65.50** | |

## Power System

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Battery | 12V 12Ah SLA | 1 | UB12120 | Digi-Key/Amazon | $25–35 | $25–35 | Sealed lead-acid, or LiFePO4 |
| Battery charger | 12V SLA charger | 1 | Various | Amazon/Digi-Key | $15–25 | $15–25 | Automatic charger |
| Fuse holder | Panel mount fuse holder | 1 | 0287005.PXCN | Digi-Key | $2.00 | $2.00 | 5A fuse holder |
| Fuses | 5A fast-blow fuses | 5 | 0034.5002 | Digi-Key | $0.50 | $2.50 | Spare fuses |
| Distribution block | Terminal block | 1 | Various | Digi-Key | $3–5 | $3–5 | Power distribution |
| Voltage regulator | 12V to 5V buck | 1 | LM2596 module | Amazon | $3–5 | $3–5 | 5V regulator module |
| **Subtotal Power** | | | | | | **$50.50–74.50** | |

## Communications/Data Logging

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Data logger | Raspberry Pi 4 or tablet | 1 | Various | Various | $50–200 | $50–200 | RPi4 + SD card, or Android tablet |
| RS485 transceiver | MAX485 module | 1 | MAX485ESA+ | Digi-Key | $2.50 | $2.50 | RS485 interface (if used) |
| Data cables | CAT5 or custom cable | 20–24 | Custom | Digi-Key/Mouser | $2–4 | $40–96 | Data/power cables to probes |
| Connectors | RJ45 or custom | 20–24 | Various | Digi-Key | $0.50 | $10–12 | For data cables |
| Power connectors | DC power jacks | 20–24 | PJ-002A | Digi-Key | $0.50 | $10–12 | Power connectors |
| Wireless option | LoRa module (optional) | 1 | SX1278 module | Amazon | $8–12 | $8–12 | Optional LoRa |
| **Subtotal Comms** | | | | | | **$120.50–332.50** | |

## Enclosure and Mounting

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Enclosure | Weatherproof box | 1 | Various | Digi-Key/Amazon | $30–60 | $30–60 | IP65 rated, ~300×200×150mm (larger for all electronics) |
| Mounting hardware | Screws, brackets | 1 set | Various | McMaster/Digi-Key | $5–10 | $5–10 | M3 screws, brackets |
| Cable management | Strain reliefs, glands | 1 set | Various | Digi-Key | $10–15 | $10–15 | PG glands, strain reliefs (more needed) |
| Terminal blocks | For probe connections | 2–3 | Various | Digi-Key | $5–10 | $10–30 | Terminal blocks for probe wiring |
| **Subtotal Enclosure** | | | | | | **$55–115** | |

## Total Base Hub Cost

| Category | Cost Range |
|----------|------------|
| MIT System | $56.50–66.50 |
| ERT System | $43.25–45.75 |
| Control | $22–36 |
| Sync/Clock | $34–65.50 |
| Power System | $50.50–74.50 |
| Communications/Data Logging | $120.50–332.50 |
| Enclosure and Mounting | $55–115 |
| **Total** | **$381.75–736.75** |

**Typical cost:** ~$400–600 (depending on data logger choice - RPi4 vs tablet)

## Cost Comparison

**Old Design:** $250–450 per base hub (simpler, fewer functions)
**New Design:** $400–600 per base hub (includes all probe electronics)

**Note:** While hub cost is higher, total system cost is lower because:
- Old: $130–180 per probe × 20 = $2,600–3,600 + $250–450 hub = $2,850–4,050
- New: $40–60 per probe × 20 = $800–1,200 + $400–600 hub = $1,200–1,800

**Total System Savings:** ~$1,650–2,250 (significant reduction)

## Notes

- Data logger cost varies widely (tablet vs. dedicated logger)
- Wireless option adds cost but increases flexibility
- Power system can be simplified if using USB power banks
- Some components can be integrated into single PCB (recommended)
- Consider modular design for easier upgrades
- Larger enclosure needed for all electronics

## Simplified Option

For lower cost, consider:
- Use existing laptop/tablet instead of dedicated logger: **-$100–200**
- Use USB power banks instead of large battery: **-$20–50**
- Simplified sync (wired only): **-$10–20**
- **Reduced total:** ~$270–480

## Integration Notes

- All electronics in one location (easier maintenance)
- Can use single PCB for all electronics (recommended)
- Modular design allows upgrades
- Easier troubleshooting (all electronics accessible)
- Better power management (centralized)

---

*For passive probe components, see [Probe BOM](probe-bom.md)*
