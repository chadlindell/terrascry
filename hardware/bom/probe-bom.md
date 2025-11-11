# Probe BOM (Bill of Materials)

**Per-Probe Components** - Estimated cost: $70–150 per probe

## Mechanical Components

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Rod sections | Fiberglass rod 1m, Ø25mm | 1–3 | Custom | Local/McMaster | $12–18 | $12–54 | Fiberglass tube, 1m sections. McMaster 8537K11 or equivalent |
| Threaded couplers | M/F couplers, nylon | 0–2 | Custom | 3D print/McMaster | $3–5 | $0–10 | Glass-filled nylon, custom thread |
| Nose capsule | 3D-printed capsule | 1 | Custom | 3D print | $5–8 | $5–8 | PETG or ABS, Ø30mm × 100mm |
| Seals | O-rings, various | 2–4 | AS568-XXX | McMaster | $0.50–1 | $1–4 | Size depends on design |
| Potting silicone | Neutral cure RTV | 50 ml | Dow Corning 3145 | Digi-Key/Mouser | $8–12 | $8–12 | 50ml tube, neutral cure |
| Cable glands | PG7 or M12 gland | 1–2 | Various | Digi-Key | $3–5 | $3–10 | Waterproof entry, size depends on cable |
| Strain relief | Heat-shrink, clamps | 1 set | Various | Digi-Key | $2–4 | $2–4 | Assorted sizes |
| **Subtotal Mechanical** | | | | | | **$31–92** | |

## MIT-3D Electronics

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Ferrite rod cores | Ø10 mm × 100 mm | 2 | FER-10-100 | Custom/Mouser | $3–5 | $6–10 | One TX, one RX. Alternative: Fair-Rite 2643250002 |
| Magnet wire | 34 AWG enameled, 50m | 1 | MW-34-50M | Digi-Key/Mouser | $8–12 | $8–12 | Belden 8055 or equivalent |
| DDS generator | AD9833 DDS | 1 | AD9833BRMZ-REEL7 | Digi-Key | $8.50 | $8.50 | 25 MHz DDS, SPI interface |
| TX driver | Op-amp OPA2277 | 1 | OPA2277PA | Digi-Key | $4.50 | $4.50 | Precision op-amp, dual channel |
| RX preamp | Low-noise AD620 | 1 | AD620ANZ | Digi-Key | $6.50 | $6.50 | Instrumentation amp, low noise |
| Instrumentation amp | INA128 | 1 | INA128PAG4 | Digi-Key | $7.50 | $7.50 | Precision instrumentation amp |
| Lock-in (Option A) | 24-bit ADC ADS1256 | 1 | ADS1256IDBR | Digi-Key | $18.50 | $18.50 | Digital lock-in, SPI interface |
| Lock-in (Option B) | Analog multiplier AD630 | 1 | AD630ANZ | Digi-Key | $12.00 | $12.00 | Analog lock-in multiplier |
| ADC (if Option B) | 16-bit ADC ADS1115 | 1 | ADS1115IDGST | Digi-Key | $6.00 | $6.00 | I2C interface, optional |
| MCU | ESP32 DevKit | 1 | ESP32-DEVKITC-32E | Adafruit/SparkFun | $10.00 | $10.00 | WiFi/Bluetooth, or ESP32-WROOM-32 |
| Voltage regulator | 3.3V LDO AMS1117 | 1 | AMS1117-3.3 | Digi-Key | $0.50 | $0.50 | 3.3V regulator |
| Voltage regulator | 5V LDO AMS1117 | 1 | AMS1117-5.0 | Digi-Key | $0.50 | $0.50 | 5V regulator |
| PCB | Perfboard or custom | 1 | — | Custom | $3–8 | $3–8 | 50×50 mm perfboard or custom PCB |
| Connectors | Headers, sockets | 1 set | Various | Digi-Key | $3–5 | $3–5 | 2.54mm pitch headers |
| **Subtotal MIT Electronics** | | | | | | **$78–95** | Option A (digital lock-in) recommended |

## ERT-Lite Components

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Ring electrodes | Stainless steel strip 12mm | 2–3 | Custom | Local/McMaster | $2–3 | $4–9 | 304SS, 0.5mm thick, custom cut |
| Electrode wire | Shielded twisted pair 3m | 1 | Custom | Digi-Key/Mouser | $3–5 | $3–5 | 24 AWG, shielded, Belden 8723 or equivalent |
| Multiplexer | CD4051 analog mux | 1 | CD4051BE | Digi-Key | $1.50 | $1.50 | 8-channel analog multiplexer |
| Diff amp | INA128 (shared with MIT) | 1 | INA128PAG4 | Digi-Key | $7.50 | $7.50 | Can share with MIT subsystem |
| **Subtotal ERT** | | | | | | **$16–22** | |

## Total per Probe

| Category | Cost Range |
|----------|------------|
| Mechanical | $31–92 |
| MIT Electronics | $78–95 |
| ERT Components | $16–22 |
| **Total** | **$125–209** |

**Typical cost:** ~$130–180 per probe (with specific components, quantity discounts may reduce by 10–15%)

**Note:** Costs updated with specific part numbers. Bulk ordering (20+ probes) recommended for best pricing.

## Notes

- Costs are indicative and vary by supplier, quantity, and component choices
- Bulk ordering (20+ probes) may reduce costs by 10–20%
- Some components can be sourced used/refurbished for lower cost
- 3D printing costs depend on whether you have access to a printer
- Consider buying some components in larger quantities (e.g., wire, connectors)

## Procurement Tips

- Order all electronic components from major distributors (Digi-Key, Mouser) for consistency
- Mechanical components may be available locally or from specialized suppliers
- Consider prototyping with cheaper alternatives before final build
- Keep spares of critical components (10–20% extra)

