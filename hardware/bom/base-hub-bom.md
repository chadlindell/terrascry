# Base/Hub BOM (Bill of Materials)

**Shared Components** - Estimated cost: $200–500

## ERT Current Source

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Op-amp | Precision op-amp OPA177 | 1 | OPA177GP | Digi-Key | $4.50 | $4.50 | Current source amplifier |
| Voltage reference | REF5025 2.5V | 1 | REF5025AIDGKT | Digi-Key | $3.50 | $3.50 | Precision 2.5V reference |
| Current sense resistor | 1.25kΩ 0.1% precision | 1 | RN73R2BTTD1251B10 | Digi-Key | $1.50 | $1.50 | For 2mA full scale |
| Polarity switch | DPDT relay 5V | 1 | G5V-2-5DC | Digi-Key | $3.50 | $3.50 | Polarity reversal |
| Relay driver | Transistor/MOSFET | 1 | 2N7002 | Digi-Key | $0.25 | $0.25 | Drive relay coil |
| **Subtotal ERT Source** | | | | | | **$13.25** | |

## Differential Voltmeter

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Instrumentation amp | INA128 precision | 1 | INA128PAG4 | Digi-Key | $7.50 | $7.50 | High-precision diff amp |
| ADC | 24-bit ADS1256 | 1 | ADS1256IDBR | Digi-Key | $18.50 | $18.50 | High resolution ADC |
| Multiplexer | CD4051 8-channel | 1 | CD4051BE | Digi-Key | $1.50 | $1.50 | Channel selection (if needed) |
| **Subtotal Voltmeter** | | | | | | **$27.50** | |

## Sync/Clock Distribution

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Crystal oscillator | 10 MHz oscillator | 1 | ECS-100-10-30B-TR | Digi-Key | $2.50 | $2.50 | Reference clock (or use DDS) |
| Clock buffer | 74HC244 octal buffer | 1 | SN74HC244N | Digi-Key | $1.50 | $1.50 | Drive multiple outputs |
| Cables | CAT5 cable 20m | 1 | Custom | Digi-Key/Mouser | $10–20 | $10–20 | Sync distribution (20 probes) |
| Connectors | RJ45 connectors | 20 | Various | Digi-Key | $0.50 | $10 | For sync cables |
| **Subtotal Sync** | | | | | | **$24.50–34.50** | |

## Power System

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Battery | 12V 12Ah SLA | 1 | UB12120 | Digi-Key/Amazon | $25–35 | $25–35 | Sealed lead-acid, or LiFePO4 alternative |
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
| Cables | CAT5 cable 50m | 1 | Custom | Digi-Key/Mouser | $15–25 | $15–25 | Data/power cables |
| Connectors | RJ45 connectors | 20 | Various | Digi-Key | $0.50 | $10 | For data cables |
| Power connectors | DC power jacks | 20 | PJ-002A | Digi-Key | $0.50 | $10 | Power connectors |
| Wireless option | LoRa module (optional) | 1 | SX1278 module | Amazon | $8–12 | $8–12 | Optional LoRa |
| **Subtotal Comms** | | | | | | **$95.50–257.50** | |

## Enclosure and Mounting

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Enclosure | Weatherproof box | 1 | Various | Digi-Key/Amazon | $25–50 | $25–50 | IP65 rated, ~200×150×100mm |
| Mounting hardware | Screws, brackets | 1 set | Various | McMaster/Digi-Key | $5–10 | $5–10 | M3 screws, brackets |
| Cable management | Strain reliefs, glands | 1 set | Various | Digi-Key | $5–10 | $5–10 | PG glands, strain reliefs |
| **Subtotal Enclosure** | | | | | | **$35–70** | |

## Total Base/Hub Cost

| Category | Cost Range |
|----------|------------|
| ERT Current Source | $13.25 |
| Differential Voltmeter | $27.50 |
| Sync/Clock Distribution | $24.50–34.50 |
| Power System | $50.50–74.50 |
| Communications/Data Logging | $95.50–257.50 |
| Enclosure and Mounting | $35–70 |
| **Total** | **$246.25–477.75** |

**Typical cost:** ~$250–450 (depending on data logger choice - RPi4 vs tablet)

## Notes

- Data logger cost varies widely (tablet vs. dedicated logger)
- Wireless option adds cost but increases flexibility
- Power system can be simplified if using USB power banks
- Some components can be integrated into single PCB
- Consider modular design for easier upgrades

## Simplified Option

For lower cost, consider:
- Use existing laptop/tablet instead of dedicated logger: **-$100–200**
- Use USB power banks instead of large battery: **-$20–50**
- Simplified sync (wired only): **-$10–20**
- **Reduced total:** ~$150–300

