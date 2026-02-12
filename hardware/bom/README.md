# Pathfinder Bill of Materials

Complete parts list for building the Pathfinder 4-pair fluxgate gradiometer.
**Estimated total cost: $622 - $818**

*Generated from `pathfinder-bom.csv` by `generate_bom.py`.*

## Sensors ($436 - $464)

| Item | Spec | Qty | Unit Cost | Ext. Cost | Supplier | Notes |
|------|------|----:|----------:|----------:|----------|-------|
| Fluxgate sensor | FG Sensors FG-3+ (frequency output) | 8 | $52.00-54.00 | $416.00-432.00 | FG Sensors, SparkFun | Primary recommendation; requires LM2917 F-to-V conversion |
| LM2917 F-to-V converter | LM2917N-8 frequency-to-voltage IC | 8 | $2.00-3.00 | $16.00-24.00 | Mouser, DigiKey | Signal conditioning for FG-3+ frequency output |
| F-to-V support components | Resistors, capacitors per LM2917 datasheet | 8 | $0.50-1.00 | $4.00-8.00 | Mouser, DigiKey | Per-channel: timing R/C + output filter |

## Electronics ($33 - $76)

| Item | Spec | Qty | Unit Cost | Ext. Cost | Supplier | Notes |
|------|------|----:|----------:|----------:|----------|-------|
| Arduino Nano | ATmega328P 5V/16MHz | 1 | $5.00-25.00 | $5.00-25.00 | Arduino Store, AliExpress | Clone ~$5; genuine ~$25 |
| ADS1115 ADC module | 16-bit I2C ADC breakout | 2 | $4.00-6.00 | $8.00-12.00 | Adafruit, AliExpress | 4 channels each; 0x48 and 0x49 |
| NEO-6M GPS module | UART GPS with antenna | 1 | $10.00-18.00 | $10.00-18.00 | AliExpress, Amazon | Includes ceramic patch antenna |
| SD card module | SPI microSD breakout | 1 | $3.00-6.00 | $3.00-6.00 | Adafruit, AliExpress | FAT32 formatted card required |
| microSD card | Class 10 8-32 GB FAT32 | 1 | $5.00-10.00 | $5.00-10.00 | Any retailer |  |
| Piezo buzzer | Passive 5V piezo or magnetic buzzer | 1 | $1.00-3.00 | $1.00-3.00 | Mouser, AliExpress | Pace beeper |
| Status LED | 5mm red LED | 1 | $0.10-0.50 | $0.10-0.50 | Mouser, AliExpress |  |
| 220 ohm resistor | 1/4W through-hole | 1 | $0.05-0.10 | $0.05-0.10 | Mouser, DigiKey | LED current limiter |
| 2N7000 MOSFET | N-channel TO-92 | 1 | $0.30-0.60 | $0.30-0.60 | Mouser, DigiKey | Buzzer driver |
| 10k ohm resistor | 1/4W through-hole | 2 | $0.05-0.10 | $0.10-0.20 | Mouser, DigiKey | MOSFET gate pulldown + bias |
| 4.7k ohm resistor | 1/4W through-hole | 2 | $0.05-0.10 | $0.10-0.20 | Mouser, DigiKey | I2C pullups (if not on ADS1115 modules) |
| 0.1uF ceramic capacitor | Bypass caps | 4 | $0.05-0.10 | $0.20-0.40 | Mouser, DigiKey | One per ADS1115 + one per sensor power rail |

## Power ($20 - $43)

| Item | Spec | Qty | Unit Cost | Ext. Cost | Supplier | Notes |
|------|------|----:|----------:|----------:|----------|-------|
| LiPo battery | 7.4V 2S 2000-3000 mAh | 1 | $15.00-30.00 | $15.00-30.00 | HobbyKing, Amazon | Belt-mounted; use LiPo-safe bag |
| 5V buck converter | LM2596-based module >600mA | 1 | $3.00-8.00 | $3.00-8.00 | AliExpress, Amazon | Preferred over linear 7805 for efficiency |
| Power switch | SPST toggle or rocker | 1 | $1.00-3.00 | $1.00-3.00 | Hardware store | Panel-mount |
| XT60 or JST connector | Battery connector pair | 1 | $1.00-2.00 | $1.00-2.00 | HobbyKing, Amazon | Match battery connector |

## Frame ($67 - $109)

| Item | Spec | Qty | Unit Cost | Ext. Cost | Supplier | Notes |
|------|------|----:|----------:|----------:|----------|-------|
| Carbon fiber tube | 25mm OD x 2mm wall x 2m | 1 | $35.00-45.00 | $35.00-45.00 | DragonPlate, RockWest | Primary crossbar; aluminum EMT 1.25in is $15 budget alt |
| PVC conduit | 3/4in Schedule 40 x 50cm | 4 | $1.50-2.50 | $6.00-10.00 | Hardware store | Sensor drop tubes |
| 3D-printed sensor clips | PLA/PETG clip mounts | 8 | $1.50-3.00 | $12.00-24.00 | Self-printed or Shapeways | Top + bottom sensor mounts |
| End caps | 3D-printed crossbar end caps | 4 | $0.50-1.50 | $2.00-6.00 | Self-printed |  |
| M5 x 60mm bolts + locknuts | Stainless steel | 4 | $0.75-1.25 | $3.00-5.00 | Hardware store | Drop tube attachment |
| Nylon spacers | M5 ID x 10mm | 4 | $0.50-1.00 | $2.00-4.00 | Hardware store |  |
| Center mount D-ring | 25mm stainless | 1 | $2.00-4.00 | $2.00-4.00 | Hardware store | Harness attachment point |
| Pipe clamp collar | 25mm | 1 | $2.00-5.00 | $2.00-5.00 | Hardware store | Center reinforcement |
| Foam padding sheet | Self-adhesive 5mm EVA | 1 | $3.00-6.00 | $3.00-6.00 | Hardware store | Sensor cushioning |

## Harness ($12 - $36)

| Item | Spec | Qty | Unit Cost | Ext. Cost | Supplier | Notes |
|------|------|----:|----------:|----------:|----------|-------|
| Shoulder straps | Padded 50mm wide (salvaged from backpack) | 1 | $0.00-10.00 | $0.00-10.00 | Salvaged or outdoor store |  |
| Bungee cord | 6mm diameter x 1m | 1 | $2.00-4.00 | $2.00-4.00 | Hardware store | Vibration isolation |
| Carabiners | 5kN rated spring-gate | 4 | $2.00-4.00 | $8.00-16.00 | Hardware store | Quick-release connections |
| D-rings | 25mm stainless | 2 | $0.75-1.50 | $1.50-3.00 | Hardware store | Harness attachment |
| Spreader bar | 20cm aluminum rod | 1 | $1.00-3.00 | $1.00-3.00 | Hardware store |  |

## Enclosure ($21 - $36)

| Item | Spec | Qty | Unit Cost | Ext. Cost | Supplier | Notes |
|------|------|----:|----------:|----------:|----------|-------|
| IP65 enclosure | Hammond 1554K or equivalent ~150x100x50mm | 1 | $12.00-18.00 | $12.00-18.00 | Mouser, Amazon | Electronics housing |
| PG7 cable glands | Waterproof cable entry | 2 | $1.50-3.00 | $3.00-6.00 | Mouser, Amazon | Sensor cable + power entry |
| Gore-Tex vent | Pressure equalization | 1 | $2.00-4.00 | $2.00-4.00 | Mouser, Amazon | Prevents condensation |
| Velcro straps | 25mm x 30cm | 2 | $1.50-3.00 | $3.00-6.00 | Hardware store | Belt mounting |
| M3 standoffs | 10mm nylon M3 | 8 | $0.15-0.30 | $1.20-2.40 | Mouser, Amazon | PCB mounting |

## Cables ($32 - $54)

| Item | Spec | Qty | Unit Cost | Ext. Cost | Supplier | Notes |
|------|------|----:|----------:|----------:|----------|-------|
| 4-conductor shielded cable | 22-24 AWG shielded | 15 | $0.80-1.20 | $12.00-18.00 | Mouser (per meter) | Sensor connections |
| JST-XH 4-pin connector sets | Male + female pairs | 10 | $0.50-1.00 | $5.00-10.00 | Mouser, AliExpress | Sensor cable connectors |
| Spiral cable wrap | 10mm diameter | 2 | $2.00-3.00 | $4.00-6.00 | Hardware store | Per meter; cable management |
| Heat shrink tubing kit | Assorted sizes | 1 | $4.00-6.00 | $4.00-6.00 | Mouser, Amazon |  |
| Cable ties UV-resistant | 200mm nylon | 50 | $0.04-0.08 | $2.00-4.00 | Hardware store |  |
| Label tape | Self-laminating wire labels | 1 | $5.00-10.00 | $5.00-10.00 | Brady, Amazon | Sensor identification |

## Cost Summary

| Category | Low | High |
|----------|----:|-----:|
| Sensors | $436 | $464 |
| Electronics | $33 | $76 |
| Power | $20 | $43 |
| Frame | $67 | $109 |
| Harness | $12 | $36 |
| Enclosure | $21 | $36 |
| Cables | $32 | $54 |
| **Total** | **$622** | **$818** |

## Substitution Notes

- **Crossbar**: Aluminum EMT conduit 1.25" (~$15) is a viable budget substitute for carbon fiber (~$40)
- **Arduino Nano**: Clones from AliExpress (~$5) work identically to genuine (~$25)
- **Shoulder straps**: Salvaging from an old backpack saves $10-15
- **3D-printed parts**: Can be ordered from Shapeways/JLCPCB if no printer available
- **Fluxgate sensors**: The FG-3+ is recommended; Magnetometer-Kit.com FGM-3 PRO is a compatible alternative at ~20% higher cost

## Source Data

This file was generated from `pathfinder-bom.csv`. To regenerate after editing the CSV:
```bash
python generate_bom.py -o README.md
```
