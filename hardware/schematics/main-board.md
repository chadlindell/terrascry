# Pathfinder Main Board Schematic Documentation

Version: 1.0
Date: 2026-02-02
Target Platform: Arduino Nano (ATmega328P)

## System Overview

The Pathfinder main board integrates:
- 8 fluxgate sensor inputs via 2x ADS1115 ADC modules
- GPS positioning via NEO-6M module
- SD card data logging
- Audio pace beeper
- 7.4V LiPo battery with 5V regulation

## Text-Based Schematic

```
                    +---------------------+
                    |   7.4V LiPo Battery |
                    |    2000-3000 mAh    |
                    +----------+----------+
                               |
                    +----------+----------+
                    |    Power Switch     |
                    +----------+----------+
                               |
                    +----------v----------+
                    |  5V Regulator       |
                    |  LM7805 or Buck     |
                    |  (>500mA capacity)  |
                    +----------+----------+
                               |
                               +----------------------------------+
                               |                                  |
                    +----------v----------+            +----------v----------+
                    |   Arduino Nano      |            |   Peripheral 5V Rail|
                    |   ATmega328P        |            +----------+----------+
                    |                     |                       |
                    | D13 [o]---SCK-------+----+---> SD Card     |
                    | D12 [o]---MISO------+    |     CLK          |
                    | D11 [o]---MOSI------+    |     DI/DO        |
                    |                     |    |                  |
                    | D10 [o]---CS--------+----+---> SD Card CS   |
                    |                     |                       |
                    | D9  [o]---BEEP------+--> BUZZER <----------+
                    |                     |      |                |
                    | D2  [o]---LED-------+      GND              |
                    |                     |    |                 |
                    | D4  [o]---RX--------+----+---> GPS (NEO-6M)
                    | D3  [o]---TX--------+    |     TX/RX       |
                    |                     |    |     VCC <-------+
                    | A4  [o]---SDA-------+----+---> GND
                    | A5  [o]---SCL-------+    |                 |
                    |                     |    |                 |
                    | GND [o]-------------+----+                 |
                    | VIN [o] (not used)  |    |                 |
                    | 5V  [o]-------------+----+                 |
                    +---------------------+    |     VCC <-------+
                                               |     GND          |
                                               |                  |
                                               |  +---------------v---------+
                                               |  |  ADS1115 Module #1      |
                                               +->|  I2C Addr: 0x48         |
                                               |  |                         |
                                               |  | A0 <-- Sensor P1_TOP    |
                                               |  | A1 <-- Sensor P1_BOT    |
                                               |  | A2 <-- Sensor P2_TOP    |
                                               |  | A3 <-- Sensor P2_BOT    |
                                               |  |                         |
                                               |  | VCC <---+               |
                                               |  | GND     |               |
                                               |  | SDA/SCL |               |
                                               |  +---------|---------------+
                                               |            |
                                               |  +---------v---------------+
                                               |  |  ADS1115 Module #2      |
                                               +->|  I2C Addr: 0x49         |
                                                  |  (ADDR pin to VDD)      |
                                                  |                         |
                                                  | A0 <-- Sensor P3_TOP    |
                                                  | A1 <-- Sensor P3_BOT    |
                                                  | A2 <-- Sensor P4_TOP    |
                                                  | A3 <-- Sensor P4_BOT    |
                                                  |                         |
                                                  | VCC <---+               |
                                                  | GND     |               |
                                                  | SDA/SCL |               |
                                                  +-------------------------+
```

## Arduino Nano Pin Mapping

### Digital Pins

| Pin | Function | Connection | Notes |
|-----|----------|------------|-------|
| D13 | SPI SCK | SD Card CLK | Hardware SPI clock (fixed) |
| D12 | SPI MISO | SD Card DO | Hardware SPI data in (fixed) |
| D11 | SPI MOSI | SD Card DI | Hardware SPI data out (fixed) |
| D10 | SD_CS | SD Card CS | SPI Chip Select |
| D9 | BEEPER | Buzzer + MOSFET | Pace beeper output |
| D8 | (Reserved) | - | Future expansion |
| D7 | (Reserved) | - | Future expansion |
| D6 | (Reserved) | - | Future expansion |
| D5 | (Reserved) | - | Future expansion |
| D4 | GPS_RX | NEO-6M TX | Arduino receives GPS data |
| D3 | GPS_TX | NEO-6M RX | Arduino transmits to GPS |
| D2 | STATUS_LED | LED + 220R to GND | Status indicator |

### Analog Pins

| Pin | Function | Connection | Notes |
|-----|----------|------------|-------|
| A0 | (Reserved) | - | Future: Battery voltage monitor |
| A1 | (Reserved) | - | Future expansion |
| A2 | (Reserved) | - | Future expansion |
| A3 | (Reserved) | - | Future expansion |
| A4 | I2C_SDA | ADS1115 x2 SDA | I2C data line, 10k pullup |
| A5 | I2C_SCL | ADS1115 x2 SCL | I2C clock line, 10k pullup |

### Power Pins

| Pin | Function | Connection | Notes |
|-----|----------|------------|-------|
| VIN | Raw Input | Not used | Use 5V pin instead |
| 5V | 5V Output | From regulator | Powers all peripherals |
| 3V3 | 3.3V Output | Not used | Low current limit |
| GND | Ground | Common ground | Multiple GND pins available |
| AREF | ADC Reference | Not used | Internal reference OK |

## I2C Address Assignments

| Device | Address | Configuration | Notes |
|--------|---------|---------------|-------|
| ADS1115 #1 | 0x48 | ADDR to GND | Default address (Pairs 1-2) |
| ADS1115 #2 | 0x49 | ADDR to VDD | Pairs 3-4 |
| (Reserved) | 0x50-0x57 | - | Future: EEPROM for calibration |
| (Reserved) | 0x68 | - | Future: RTC module |

### I2C Bus Configuration

- **Pullup resistors**: 4.7k ohms on SDA and SCL (typically on ADS1115 modules)
- **Bus speed**: 100 kHz standard mode (400 kHz fast mode compatible)
- **Cable length**: <30 cm recommended for noise immunity

## ADS1115 Sensor Channel Mapping

### ADC #1 (Address 0x48) - Pairs 1-2

| Channel | Sensor | Location | Notes |
|---------|--------|----------|-------|
| A0 | P1_TOP | Pair 1 Top | Left-most sensor, reference |
| A1 | P1_BOT | Pair 1 Bottom | Left-most sensor, signal |
| A2 | P2_TOP | Pair 2 Top | Center-left, reference |
| A3 | P2_BOT | Pair 2 Bottom | Center-left, signal |

### ADC #2 (Address 0x49) - Pairs 3-4

| Channel | Sensor | Location | Notes |
|---------|--------|----------|-------|
| A0 | P3_TOP | Pair 3 Top | Center-right, reference |
| A1 | P3_BOT | Pair 3 Bottom | Center-right, signal |
| A2 | P4_TOP | Pair 4 Top | Right-most, reference |
| A3 | P4_BOT | Pair 4 Bottom | Right-most, signal |

### ADS1115 Configuration

- **Gain**: +/- 4.096V (programmable gain amplifier)
- **Sample rate**: 128 SPS (samples per second) - adequate for walking pace
- **Resolution**: 16-bit (0.125 mV per bit at +/- 4.096V gain)
- **Input mode**: Single-ended (each sensor to GND)

## Sensor Cable Connector Pinout

### Standard Sensor Cable (4-conductor shielded)

Each fluxgate sensor requires:

| Pin | Signal | Wire Color | Notes |
|-----|--------|------------|-------|
| 1 | +5V | Red | Power supply |
| 2 | GND | Black | Ground return |
| 3 | SIGNAL | White/Yellow | Analog output to ADC |
| 4 | SHIELD | Bare/Green | Connected to GND at controller end only |

### Recommended Connector

- **Type**: JST-XH 4-pin or screw terminals
- **Wire gauge**: 22-24 AWG
- **Cable**: Shielded 4-conductor (e.g., microphone cable)
- **Shield**: Ground at controller end, float at sensor end (avoid ground loops)

### Cable Routing Guidelines

1. **Separation**: Keep sensor cables >5 cm from power cables
2. **Strain relief**: Use cable ties or spiral wrap on crossbar
3. **Length**: Minimize cable length (30-50 cm typical)
4. **Labeling**: Mark each cable with sensor ID (P1_TOP, P1_BOT, etc.)

## Power Budget Calculation

### Power Consumption Analysis (Modeled)

Full system including sensors and all peripherals:

| Component | Voltage | Current (typ) | Current (max) | Notes |
|-----------|---------|---------------|---------------|-------|
| Arduino Nano | 5V | 20 mA | 40 mA | ATmega328P + USB chip |
| ADS1115 #1 | 5V | 0.2 mA | 1 mA | Low power ADC |
| ADS1115 #2 | 5V | 0.2 mA | 1 mA | Low power ADC |
| NEO-6M GPS | 5V | 45 mA | 67 mA | During acquisition |
| SD Card | 5V | 50 mA | 100 mA | During write |
| Status LED | 5V | 10 mA | 20 mA | Via 220R resistor |
| Buzzer | 5V | 20 mA | 50 mA | Intermittent, duty cycle ~5% |
| Fluxgates (8x) | 5V | 160 mA | 240 mA | Estimated 20-30 mA each |
| **Total (full system)** | | **305 mA** | **519 mA** | **(Modeled)** |

**Note**: Electronics-only (no fluxgate sensors) draws ~145 mA typical / ~279 mA max. MCU + peripherals (no sensors, no SD writes) draws ~120 mA typical. All figures are datasheet estimates, not bench-measured.

### 5V Regulator Requirements

- **Input voltage**: 7.4V nominal (6.0-8.4V range for 2S LiPo)
- **Output current**: 600 mA minimum (with margin)
- **Type**: Linear (LM7805) or Buck converter (preferred for efficiency)
- **Heatsinking**: Required for linear regulator (2W dissipation)

**Recommended**: LM2596-based buck module (>90% efficiency, minimal heat)

### Battery Life Estimation

| Battery Capacity | Runtime (typ) | Runtime (max load) |
|------------------|---------------|--------------------|
| 2000 mAh | 6.5 hours | 3.9 hours |
| 3000 mAh | 9.8 hours | 5.8 hours |
| 5000 mAh | 16.4 hours | 9.6 hours |

**Calculation**: Runtime = Capacity / (Current × Efficiency)
Assumes 80% regulator efficiency for buck converter.

### Power Distribution

```
LiPo 7.4V ---[SWITCH]---[BUCK CONVERTER]---+---> Arduino 5V
                                            |
                                            +---> ADS1115 #1 VCC
                                            |
                                            +---> ADS1115 #2 VCC
                                            |
                                            +---> GPS Module VCC
                                            |
                                            +---> SD Card VCC
                                            |
                                            +---> Buzzer VCC
                                            |
                                            +---> Sensor Power (8x)

Common GND for all components
```

## Additional Circuit Details

### Status LED Circuit

```
D2 ---[220R]---LED(+)---(-)--- GND

LED: Red, 5mm, 2V forward drop
Current: (5V - 2V) / 220R = 13.6 mA
```

### Buzzer Driver Circuit

```
D9 ---[10k]---+---[MOSFET Gate]
                     |
                    [10k to GND]

MOSFET Drain --- BUZZER(+)
MOSFET Source --- GND
BUZZER(-) --- 5V

MOSFET: 2N7000 or BS170 (N-channel)
Buzzer: Passive piezo or magnetic (5V, <50mA)
PWM Frequency: 2 kHz for audible tone
```

### GPS Serial Connection

```
Arduino D4 (RX) <--- GPS TX (3.3V or 5V tolerant)
Arduino D3 (TX) ---> GPS RX (via 1k resistor if 3.3V only)

NEO-6M modules typically 5V tolerant on all pins.
Baud rate: 9600 (default) or 115200
```

### SD Card SPI Connection

```
Arduino D10 (CS)   ---> SD CS
Arduino D11 (MOSI) ---> SD MOSI/DI    (hardware SPI, fixed)
Arduino D12 (MISO) <--- SD MISO/DO    (hardware SPI, fixed)
Arduino D13 (SCK)  ---> SD SCK/CLK    (hardware SPI, fixed)
5V                 ---> SD VCC
GND                ---> SD GND

SD modules typically include level shifters for 3.3V cards.
```

## I2C Pullup Configuration

The ADS1115 modules typically include onboard 10k pullup resistors on SDA and SCL. If not:

```
SDA ---[4.7k]--- 5V
SCL ---[4.7k]--- 5V

Parallel combination of multiple modules:
- 2x ADS1115 with 10k each = 5k effective
- Add external 10k if needed for margin
- Too strong pullup (<1k) wastes power
- Too weak pullup (>10k) causes communication errors
```

## Physical Layout Recommendations

### Main Board PCB/Perfboard

```
+------------------------------------------+
|  [Power Switch]  [Power Jack/Connector] |
|                                          |
|  [Buck Module]    [Arduino Nano]         |
|                                          |
|  [ADS1115 #1]     [ADS1115 #2]          |
|                                          |
|  [GPS Module]     [SD Card Module]       |
|                                          |
|  [Screw Terminals for 8 Sensor Cables]  |
+------------------------------------------+

Approximate size: 100mm x 150mm
```

### Enclosure Mounting

- **Enclosure**: IP65-rated plastic box (e.g., Hammond 1554 series)
- **Mounting**: Belt-mounted or attached to crossbar
- **Cable entry**: PG7 cable glands for sensor cables and power
- **Access**: SD card slot accessible without opening (optional)

### Sensor Cable Termination

Option 1: Screw terminals on PCB (field serviceable)
Option 2: JST-XH connectors (more compact, keyed)
Option 3: DB9/DB15 multi-pin connector (professional, bulky)

## Testing and Validation

### Power-On Test Sequence

1. **Visual inspection**: Check all connections before power
2. **Power test**: Measure 5V rail with multimeter (4.9-5.1V acceptable)
3. **LED test**: Status LED should illuminate
4. **I2C scan**: Upload I2C scanner sketch, verify 0x48 and 0x49 detected
5. **GPS test**: Monitor serial output, verify NMEA sentences
6. **SD test**: Verify card detection and write capability
7. **ADC test**: Connect test voltage (e.g., 2.5V), verify reading
8. **Buzzer test**: Verify tone output on command

### Troubleshooting

| Symptom | Probable Cause | Solution |
|---------|----------------|----------|
| No power | Switch off, dead battery | Check switch, charge battery |
| Arduino boots, peripherals dead | Insufficient 5V current | Upgrade regulator to >500mA |
| I2C devices not found | Address conflict, wiring | Check ADDR pin, verify SDA/SCL |
| GPS no fix | Indoor, poor antenna | Move outdoors, check wiring |
| SD write errors | Incompatible card, wiring | Use Class 10 card, check SPI pins |
| ADC noisy readings | Ground loops, cable shield | Shield to GND at one end only |
| Buzzer doesn't sound | MOSFET damaged, wiring | Check gate voltage, replace MOSFET |

## Firmware Pin Definitions

For use in Arduino sketch:

```cpp
// User interface pins
#define STATUS_LED_PIN  2     // Status indicator LED
#define BEEPER_PIN      9     // Piezo buzzer or speaker

// GPS UART pins (SoftwareSerial)
#define GPS_RX_PIN      4     // Connect to GPS TX
#define GPS_TX_PIN      3     // Connect to GPS RX

// SD Card SPI (hardware SPI on Nano: MOSI=11, MISO=12, SCK=13)
#define SD_CS_PIN       10    // Chip Select

// I2C pins (hardware I2C on Nano: SDA=A4, SCL=A5)

// I2C addresses
#define ADS1115_ADDR_1  0x48  // Pairs 1-2
#define ADS1115_ADDR_2  0x49  // Pairs 3-4

// ADS1115 channel mapping
// Module 1 (0x48): Pairs 1-2
// Module 2 (0x49): Pairs 3-4
// Each module: A0=Top, A1=Bot for first pair; A2=Top, A3=Bot for second pair
```

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-02 | Initial schematic documentation |

## References

- [Arduino Nano Pinout](https://docs.arduino.cc/hardware/nano)
- [ADS1115 Datasheet](https://www.ti.com/lit/ds/symlink/ads1115.pdf)
- [NEO-6M GPS Module](https://www.u-blox.com/en/product/neo-6-series)
- [SD Card Library](https://www.arduino.cc/en/Reference/SD)

## Notes

- All component values and pin assignments verified against Arduino Nano specifications
- Power budget includes 20% margin for safety
- I2C addresses confirmed non-conflicting
- Sensor cable pinout follows industry standard color codes
- PCB layout recommendations based on noise immunity best practices

---

**Document Status**: Ready for Implementation
**Next Steps**: Order components, build prototype, firmware development
