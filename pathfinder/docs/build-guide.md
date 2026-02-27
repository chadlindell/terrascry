# Pathfinder Build Guide

Complete step-by-step instructions for building the Pathfinder 4-pair fluxgate gradiometer from scratch. This guide consolidates information from the wiring guide, BOM, frame design, and schematics into a single build sequence.

**Document version**: 1.0 (2026-02-18)

---

## Overview

The Pathfinder is a harness-supported, multi-channel fluxgate gradiometer for rapid magnetic reconnaissance. It carries 4 gradiometer pairs (8 sensors total) on a horizontal crossbar, suspended from padded shoulder straps with bungee isolation. The electronics fit inside a belt-mounted IP65 enclosure and log georeferenced magnetic gradient data to an SD card.

### Build Summary

| Parameter | Value |
|-----------|-------|
| **Total build time** | 6-10 hours (first build) |
| **Skill level** | Intermediate (basic soldering, 3D printing helpful but not required) |
| **Total cost** | $620-$820 (see BOM); up to $1,050 if purchasing premium components |
| **Tools needed** | Soldering iron, multimeter, wire strippers, heat gun, 3D printer (optional) |
| **Firmware** | PlatformIO on Arduino Nano (ATmega328P, 5V/16MHz) |

### What You Will Build

```
                    PADDED HARNESS
                   (backpack straps)
                          |
              +-----------+-----------+
              |   bungee suspension   |
              +-----------+-----------+
                          |
    ======================+=======================
         CARBON FIBER CROSSBAR (2.0m x 25mm)
        |         |         |         |
       +-+       +-+       +-+       +-+
       |T|       |T|       |T|       |T|    <- TOP sensors (reference)
       | |       | |       | |       | |
       | |       | |       | |       | |    <- PVC drop tubes (50cm)
       |B|       |B|       |B|       |B|    <- BOTTOM sensors (signal)
       +-+       +-+       +-+       +-+       15-20cm above ground

       <--50cm--><--50cm--><--50cm-->
                 1.5m swath
```

### Build Phases at a Glance

| Phase | Description | Time Estimate |
|-------|-------------|---------------|
| 1. Procurement | Order components, print parts | 1-2 weeks |
| 2. Electronics Assembly | Solder and wire all modules | 3-4 hours |
| 3. Sensor Preparation | Test sensors, build F-to-V boards, make cables | 1-2 hours |
| 4. Frame Assembly | Cut crossbar, attach drop tubes, mount enclosure | 1-2 hours |
| 5. Harness System | Straps, bungees, fitting adjustments | 30-60 minutes |
| 6. Initial Testing | Flash firmware, bench test, first field walk | 30-60 minutes |

---

## Phase 1: Procurement (1-2 weeks)

### Full Bill of Materials

The complete BOM with pricing is maintained in `hardware/bom/pathfinder-bom.csv` and the formatted version in `hardware/bom/README.md`. The cost summary is:

| Category | Low Estimate | High Estimate |
|----------|--------------|---------------|
| Sensors (FG-3+ x8 + LM2917 x8) | $436 | $464 |
| Electronics (Arduino, ADCs, GPS, SD, passives) | $33 | $76 |
| Power (LiPo, regulator, switch) | $20 | $43 |
| Frame (crossbar, PVC, mounts) | $67 | $109 |
| Harness (straps, bungees, carabiners) | $12 | $36 |
| Enclosure (IP65 box, glands, standoffs) | $21 | $36 |
| Cables (shielded wire, connectors, labels) | $32 | $54 |
| **Total** | **$622** | **$818** |

### Long Lead-Time Components

Order these first, as they may take 1-3 weeks to arrive:

1. **Fluxgate sensors (FG-3+)**: Order from FG Sensors directly. These are specialty items and typically ship in 5-10 business days. Quantity: 8.
2. **LM2917N-8 frequency-to-voltage ICs**: Available from Mouser or DigiKey. Usually in stock, but verify before ordering. Quantity: 8.
3. **Carbon fiber tube (25mm OD x 2mm wall x 2m)**: Order from DragonPlate or RockWest Composites. These are cut-to-order and may take 1-2 weeks. Quantity: 1.
4. **3D-printed sensor clips and end caps**: If you do not have a 3D printer, order from Shapeways or JLCPCB 3D printing service. STL files are in `hardware/cad/`. Allow 1-2 weeks for printing and shipping.

### Readily Available Components

These are available from common retailers (Amazon, AliExpress, hardware stores) and typically arrive in 2-5 days:

- Arduino Nano (clone or genuine)
- ADS1115 ADC breakout modules (x2)
- NEO-6M GPS module with antenna
- SD card module and microSD card
- LiPo battery (7.4V 2S, 2000-3000 mAh)
- LM2596 buck converter module
- PVC conduit from any hardware store
- All resistors, capacitors, connectors
- Hammond 1554K enclosure (or equivalent)

### Substitute Components

If specific parts are unavailable, these substitutes work:

| Original | Substitute | Trade-off |
|----------|------------|-----------|
| Carbon fiber tube 25mm OD | Aluminum EMT conduit 1.25" | Heavier (+130g), cheaper ($15 vs $40) |
| Genuine Arduino Nano | Clone Arduino Nano (CH340) | Same functionality, $5 vs $25 |
| FG-3+ sensors | Magnetometer-Kit.com FGM-3 PRO | ~20% higher cost, compatible pinout |
| 3D-printed clips | Cable ties + foam padding | Less secure, but functional |
| Hammond 1554K enclosure | Any IP65 box ~150x100x50mm | Verify internal dimensions fit components |
| 2N7000 MOSFET (buzzer driver) | BS170 N-channel MOSFET | Pin-compatible, equivalent specs |

### Tools Required

Gather these before starting assembly:

**Essential tools:**
- Soldering iron (temperature-controlled, chisel tip recommended)
- Solder (60/40 or 63/37 leaded, 0.8mm diameter for through-hole work)
- Multimeter (for voltage, continuity, and resistance checks)
- Wire strippers (22-24 AWG range)
- Flush cutters (for trimming leads)
- Heat gun or lighter (for heat shrink tubing)
- Small Phillips and flathead screwdrivers
- Hacksaw or pipe cutter (for PVC tubes)
- Drill with 5mm bit (for crossbar mounting holes)

**Helpful but not essential:**
- Third-hand or PCB holder
- Solder wick and/or solder sucker (for mistakes)
- 3D printer (PLA or PETG filament)
- Label maker or self-laminating wire labels
- Hot glue gun (for strain relief)

### Pre-Build Preparation

Before starting assembly:

1. **Verify all components received.** Check against BOM. Test that each module powers up if possible.
2. **Format the microSD card.** Use FAT32 format, card must be 32 GB or smaller. Class 10 recommended.
3. **Install PlatformIO.** Follow instructions at https://platformio.org/install. You will need this to flash firmware.
4. **Print 3D parts** (if printing yourself). Use PETG for outdoor durability or PLA for prototyping. See STL files in `hardware/cad/`.

---

## Phase 2: Electronics Assembly (3-4 hours)

This phase covers all electronics: from preparing individual modules to wiring the complete system inside the enclosure. Work through each step in order. Each step lists the components you need, the connections to make, and a verification test to confirm the step succeeded before moving on.

### Exploded Electronics Layout

When fully assembled, the electronics sit inside the Hammond enclosure in this arrangement:

```
+------------------------------------------+
|  [Power Switch]  [Battery Connector]     |
|                                          |
|  [Buck Converter]    [Arduino Nano]      |
|                                          |
|  [ADS1115 #1]        [ADS1115 #2]       |
|                                          |
|  [GPS Module]        [SD Card Module]    |
|                                          |
|  [Screw Terminals / JST Headers for     |
|   8 Sensor Cables]                       |
+------------------------------------------+

Approximate PCB/perfboard size: 100mm x 150mm
Mount on M3 nylon standoffs (10mm) inside enclosure.
```

All modules mount on a single perfboard or directly on standoffs within the enclosure. Wiring runs underneath or along the edges.

### Step 2.1: Arduino Nano Preparation

**Components needed:**
- 1x Arduino Nano (ATmega328P, 5V/16MHz)
- Header pins (if bare board -- most modules come with headers pre-soldered)
- USB Mini-B cable

**Procedure:**

1. If the Arduino Nano came without headers soldered, solder male header pins to both rows. Use a breadboard as a jig to keep the pins aligned while soldering.
2. Connect the Nano to your computer via USB.
3. Open a serial terminal (PlatformIO Serial Monitor or Arduino IDE Serial Monitor) at 115200 baud.
4. Upload the I2C scanner sketch to verify the board is working:
   ```cpp
   #include <Wire.h>
   void setup() {
     Wire.begin();
     Serial.begin(115200);
     Serial.println("I2C Scanner Ready");
   }
   void loop() {
     for (byte addr = 1; addr < 127; addr++) {
       Wire.beginTransmission(addr);
       if (Wire.endTransmission() == 0) {
         Serial.print("Found device at 0x");
         Serial.println(addr, HEX);
       }
     }
     Serial.println("Scan complete.");
     delay(5000);
   }
   ```
5. The scanner should report no devices found (nothing connected yet). This confirms the board programs and communicates correctly.

**Verification:** Serial output shows "I2C Scanner Ready" and "Scan complete." with no errors.

**Soldering tips for headers:**
- Apply flux to pads before soldering if your solder does not contain flux.
- Heat the pad and pin simultaneously for 2-3 seconds, then touch solder to the junction (not the iron tip).
- A good joint is shiny and concave (volcano-shaped). A cold joint is dull and blobby.
- If you bridge two pins, use solder wick to remove the excess.

### Step 2.2: ADS1115 ADC Modules

**Components needed:**
- 2x ADS1115 16-bit ADC breakout modules
- Header pins (if not pre-soldered)
- 2x 4.7k ohm resistors (if your ADS1115 modules lack onboard I2C pullups)
- 4x 0.1 uF ceramic capacitors (bypass caps)

**Procedure:**

1. Solder headers onto both ADS1115 modules if needed. The standard breakout has pins for VDD, GND, SCL, SDA, ADDR, ALRT, A0, A1, A2, A3.

2. **Set I2C addresses.** This is critical -- the two modules must have different addresses:
   - **Module 1 (0x48):** Connect the ADDR pin to GND. On most breakout boards, this is the default (ADDR is pulled to GND via onboard resistor). Verify with your multimeter: ADDR pin should read 0V.
   - **Module 2 (0x49):** Connect the ADDR pin to VDD (5V). Solder a wire jumper from ADDR to VDD, or bridge the address solder jumper if your board has one.

3. **Check for onboard pullups.** Most ADS1115 breakout boards from Adafruit and similar include 10k pullup resistors on SDA and SCL. Look for two small resistors near the I2C pins. If your boards do not have them, you will add external 4.7k pullups in Step 2.4.

4. Solder a 0.1 uF ceramic capacitor across the VDD and GND pins of each module, as close to the chip as possible. These bypass capacitors reduce noise.

**Verification:** Do not connect yet. Set aside. We will test after wiring the I2C bus in Step 2.4.

### Step 2.3: Power Distribution

**Components needed:**
- 1x 7.4V 2S LiPo battery (2000-3000 mAh)
- 1x 5V buck converter module (LM2596-based, rated >600 mA)
- 1x SPST power switch (toggle or rocker, panel-mount)
- 1x XT60 or JST battery connector pair (match your battery)
- 22 AWG stranded wire (red and black, ~50 cm each)

**Procedure:**

1. **Wire the battery connector.** Solder red wire to the positive terminal and black wire to the negative terminal of the female battery connector. Apply heat shrink tubing to each solder joint.

2. **Wire the power switch.** Connect the red (positive) wire from the battery connector to one terminal of the SPST switch. Solder a second red wire from the other terminal of the switch to the input positive (VIN+) of the buck converter.

3. **Wire the buck converter input.** Connect the black (negative) wire from the battery connector directly to the input negative (VIN-) of the buck converter.

4. **Adjust the buck converter output.** Before connecting any loads:
   - Connect the battery.
   - Turn the power switch on.
   - Measure the output voltage with your multimeter on the VOUT+ and VOUT- terminals.
   - Turn the adjustment potentiometer on the buck converter until the output reads **5.0V** (acceptable range: 4.9V-5.1V).
   - Turn the power switch off.

5. **Create a power bus.** You need to distribute 5V and GND to all modules. Options:
   - **Perfboard approach:** Solder two copper bus strips on your perfboard -- one for 5V, one for GND. Connect the buck converter output to these buses.
   - **Terminal block approach:** Use 5mm screw terminal blocks. One row for 5V distribution, one for GND.

   The following modules will connect to the 5V/GND bus:
   - Arduino Nano (5V pin and GND pin)
   - ADS1115 Module 1 (VDD and GND)
   - ADS1115 Module 2 (VDD and GND)
   - GPS module (VCC and GND)
   - SD card module (VCC and GND)
   - Sensor power (8x fluxgate sensors)

6. **Add a power indicator (optional but recommended).** Wire a spare LED with a 1k ohm resistor between the 5V bus and GND. This gives you a quick visual check that power is reaching the bus.

**Power distribution wiring diagram:**

```
[7.4V LiPo Battery]
       |
   [XT60 Connector]
       |
       +--[Power Switch]--[Buck Converter IN+]
       |                        |
       +--[Buck Converter IN-]  [Buck Converter OUT+]---+--- 5V BUS
                                |                       |
                        [Buck Converter OUT-]---+--- GND BUS
```

**Verification:**
- Turn on the power switch. Measure 5.0V (+/- 0.1V) on the 5V bus with your multimeter.
- Measure continuity between all GND points on the bus.
- Turn off the switch. Voltage should drop to 0V.
- Current draw with no modules connected should be near zero (just the buck converter quiescent current, typically <5 mA).

### Step 2.4: I2C Bus Wiring

**Components needed:**
- Arduino Nano (from Step 2.1)
- 2x ADS1115 modules (from Step 2.2, with addresses configured)
- 2x 4.7k ohm resistors (if modules lack onboard pullups)
- 24 AWG stranded wire (2 colors: e.g., blue for SDA, yellow for SCL)

**Procedure:**

1. **Connect power to all three boards.** Wire VDD/5V and GND from the power bus (Step 2.3) to:
   - Arduino Nano 5V pin and GND pin
   - ADS1115 Module 1 VDD and GND
   - ADS1115 Module 2 VDD and GND

2. **Wire SDA (data line).** Connect:
   - Arduino Nano pin A4 (SDA) --> ADS1115 Module 1 SDA --> ADS1115 Module 2 SDA
   - This is a bus topology: all three SDA pins share a single wire.

3. **Wire SCL (clock line).** Connect:
   - Arduino Nano pin A5 (SCL) --> ADS1115 Module 1 SCL --> ADS1115 Module 2 SCL
   - Same bus topology as SDA.

4. **Add pullup resistors (if needed).** If neither ADS1115 module has onboard pullup resistors:
   - Solder a 4.7k ohm resistor from SDA to 5V.
   - Solder a 4.7k ohm resistor from SCL to 5V.
   - If both modules already have 10k onboard pullups, the effective parallel resistance is 5k, which is fine. Do NOT add external pullups in this case (too-strong pullups waste power and can cause signal issues).

5. **Verify ADDR pins:**
   - Module 1: ADDR pin connected to GND (address 0x48).
   - Module 2: ADDR pin connected to VDD (address 0x49).

**I2C bus wiring diagram:**

```
                    5V
                    |
              [4.7k] [4.7k]     <-- only if no onboard pullups
                |       |
Arduino A4 ----+--------)---------- ADS1 SDA -------- ADS2 SDA
               |        |
Arduino A5 ----+--------+---------- ADS1 SCL -------- ADS2 SCL
```

**Keep I2C wires short** -- under 30 cm total bus length. Use twisted pair if possible (twist SDA and SCL together) to reduce noise pickup.

**Verification:**
1. Power on the system.
2. Upload the I2C scanner sketch from Step 2.1 to the Arduino Nano.
3. Open the serial monitor at 115200 baud.
4. You should see two devices reported:
   ```
   Found device at 0x48
   Found device at 0x49
   Scan complete.
   ```
5. If only one device appears, check the ADDR pin wiring on the missing module. If neither appears, check SDA/SCL wiring and power connections.

### Step 2.5: GPS Module

**Components needed:**
- 1x NEO-6M GPS module (with ceramic patch antenna)
- 24 AWG stranded wire

**Procedure:**

1. **Check voltage compatibility.** Most NEO-6M breakout modules accept 5V on the VCC pin (they have an onboard 3.3V regulator). Verify with your module's documentation. If your module requires 3.3V, use the Arduino Nano 3V3 pin instead of 5V -- but note the 3V3 pin has limited current (50 mA).

2. **Wire power:** Connect GPS VCC to the 5V bus and GPS GND to the GND bus.

3. **Wire serial data:**
   - GPS TX pin --> Arduino Nano D4 (this is the Arduino's receive line in software serial)
   - GPS RX pin --> Arduino Nano D3 (not used by default firmware, but wire it for future use)

   **Important: cross-connect.** The GPS TX goes to the Arduino RX pin, and vice versa. This is the most common wiring mistake with serial connections.

4. **Mount the GPS antenna.** The ceramic patch antenna should face upward with clear sky view. If the antenna is attached via a u.FL connector, keep the coaxial cable short and avoid sharp bends.

5. **GPS placement considerations:**
   - Mount the GPS module or its antenna on the outside of the enclosure lid if possible, or near a window/opening.
   - Keep the antenna at least 10 cm from metal objects and electronics.
   - The antenna can be mounted on the crossbar for better sky view, with a cable running to the enclosure.

**Verification:**
1. Power on the system.
2. Upload a simple GPS test sketch or use the serial monitor to observe raw data:
   ```cpp
   #include <SoftwareSerial.h>
   SoftwareSerial gpsSerial(4, 3); // RX=D4, TX=D3
   void setup() {
     Serial.begin(115200);
     gpsSerial.begin(9600);
     Serial.println("GPS Test");
   }
   void loop() {
     while (gpsSerial.available()) {
       Serial.write(gpsSerial.read());
     }
   }
   ```
3. You should see NMEA sentences scrolling on the serial monitor. They look like:
   ```
   $GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*47
   $GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A
   ```
4. **Indoors, the GPS will not get a fix** (latitude/longitude will be 0 or empty). That is normal. If you see NMEA sentences at all, the wiring is correct.
5. If no data appears: check that TX goes to D4 (not D3), verify baud rate is 9600, and confirm power is reaching the GPS module.

### Step 2.6: SD Card Module

**Components needed:**
- 1x SPI microSD breakout module
- 1x microSD card (Class 10, 8-32 GB, formatted FAT32)
- 24 AWG stranded wire

**Procedure:**

1. **Wire power:** Connect SD module VCC to the 5V bus and GND to the GND bus.

2. **Wire SPI connections.** These use hardware SPI pins on the Arduino Nano, which are fixed:

   | SD Module Pin | Arduino Nano Pin | SPI Signal |
   |---------------|------------------|------------|
   | CS | D10 | Chip Select |
   | MOSI (DI) | D11 | Master Out Slave In |
   | MISO (DO) | D12 | Master In Slave Out |
   | SCK (CLK) | D13 | Serial Clock |

3. **Insert the microSD card.** Ensure it is formatted as FAT32.

**Voltage note:** Some SD card modules are 3.3V only and do not have onboard level shifters. If your module has both a 3.3V regulator and level shifters (most common breakout boards do), then 5V is fine. If the module has only a raw SD card socket with no regulator, you must power it from 3.3V and add level shifters. The Adafruit MicroSD breakout and most AliExpress modules include the necessary circuitry for 5V operation.

**Verification:**
1. Upload an SD card test sketch:
   ```cpp
   #include <SPI.h>
   #include <SD.h>
   void setup() {
     Serial.begin(115200);
     Serial.print("Initializing SD card...");
     if (!SD.begin(10)) { // CS on pin 10
       Serial.println("FAILED!");
       return;
     }
     Serial.println("OK");
     File f = SD.open("TEST.TXT", FILE_WRITE);
     if (f) {
       f.println("Hello from Pathfinder");
       f.close();
       Serial.println("Write successful");
     }
   }
   void loop() {}
   ```
2. The serial monitor should show "Initializing SD card...OK" and "Write successful".
3. Remove the SD card, insert it in a computer, and verify `TEST.TXT` exists and contains "Hello from Pathfinder".
4. If initialization fails: try a different SD card, verify CS is on D10, check SPI wiring, and ensure the card is FAT32.

### Step 2.7: Beeper and Status LED

**Components needed:**
- 1x Piezo buzzer (passive) or magnetic buzzer
- 1x 2N7000 N-channel MOSFET (TO-92 package)
- 2x 10k ohm resistors
- 1x 5mm red LED
- 1x 220 ohm resistor
- 24 AWG stranded wire

**Procedure:**

#### Status LED

1. Connect the LED anode (longer leg, +) to Arduino pin D2 through a 220 ohm resistor.
2. Connect the LED cathode (shorter leg, flat side, -) to GND.

```
Arduino D2 ----[220 ohm]---->(+LED-)---->GND
```

Current calculation: (5V - 2V forward drop) / 220 ohm = 13.6 mA. This is within safe limits for both the LED and the Arduino pin.

#### Pace Beeper

For a louder buzzer (recommended for field use), use the MOSFET driver circuit:

1. Connect Arduino pin D9 to a 10k ohm resistor. Connect the other end of this resistor to the gate of the 2N7000 MOSFET.
2. Connect a second 10k ohm resistor from the MOSFET gate to GND (this is a pulldown to ensure the MOSFET stays off when the Arduino pin is floating during boot).
3. Connect the MOSFET drain to the buzzer negative terminal.
4. Connect the buzzer positive terminal to 5V.
5. Connect the MOSFET source to GND.

```
           5V
            |
        [Buzzer+]
        [Buzzer-]
            |
    MOSFET Drain
            |
    MOSFET Source --- GND
            |
    MOSFET Gate ----[10k]---- Arduino D9
            |
          [10k]
            |
           GND
```

**MOSFET pin identification (2N7000, TO-92 package, flat side facing you, pins down):**
- Left pin: Source
- Center pin: Gate
- Right pin: Drain

**Alternative for passive buzzer (simpler, quieter):** Connect the buzzer directly between D9 and GND, without the MOSFET. This works but produces a quieter sound since the Arduino pin can only source ~20 mA.

**Verification:**
1. Upload a quick test:
   ```cpp
   void setup() {
     pinMode(2, OUTPUT);
     pinMode(9, OUTPUT);
   }
   void loop() {
     digitalWrite(2, HIGH); // LED on
     tone(9, 2000, 50);    // 2kHz beep for 50ms
     delay(1000);
     digitalWrite(2, LOW);  // LED off
     delay(1000);
   }
   ```
2. The LED should blink on/off every second. The buzzer should beep once per second.
3. If the LED does not light: check polarity (swap the legs). If the buzzer does not sound: verify MOSFET pinout and gate resistor wiring.

### Step 2.8: Sensor Cable Connectors

**Components needed:**
- 10x JST-XH 4-pin connector sets (male + female pairs; 8 needed, 2 spares)
- 24 AWG stranded wire

**Procedure:**

1. **Install JST-XH male headers on the perfboard/enclosure.** These are the connectors where sensor cables plug in. Install 8 headers -- one per sensor. Label them clearly:
   - P1_TOP, P1_BOT (Pair 1 top and bottom)
   - P2_TOP, P2_BOT (Pair 2)
   - P3_TOP, P3_BOT (Pair 3)
   - P4_TOP, P4_BOT (Pair 4)

2. **Wire each JST header to the corresponding ADS1115 input and power:**

   **Standard sensor cable pinout (JST-XH 4-pin):**

   | Pin | Signal | Wire Color | Connection |
   |-----|--------|------------|------------|
   | 1 | +5V | Red | 5V bus |
   | 2 | GND | Black | GND bus |
   | 3 | Signal | White/Yellow | ADS1115 analog input |
   | 4 | Shield | Bare/Green | GND bus (controller end only) |

3. **Wire the signal pins to the correct ADS1115 analog inputs:**

   | Sensor | ADS Module | Channel | Notes |
   |--------|------------|---------|-------|
   | P1_TOP | ADS1 (0x48) | A0 | Left-most pair, reference |
   | P1_BOT | ADS1 (0x48) | A1 | Left-most pair, signal |
   | P2_TOP | ADS1 (0x48) | A2 | Center-left pair, reference |
   | P2_BOT | ADS1 (0x48) | A3 | Center-left pair, signal |
   | P3_TOP | ADS2 (0x49) | A0 | Center-right pair, reference |
   | P3_BOT | ADS2 (0x49) | A1 | Center-right pair, signal |
   | P4_TOP | ADS2 (0x49) | A2 | Right-most pair, reference |
   | P4_BOT | ADS2 (0x49) | A3 | Right-most pair, signal |

4. **Double-check every connection.** A miswired sensor channel produces incorrect gradient data that may not be obvious until you try to interpret survey results.

**Verification:**
1. With no sensors connected, power on and read ADC values via a test sketch. Each channel should read near 0V (floating input will show some random noise).
2. Touch a wire from the 5V bus to each analog input pin, one at a time. The ADC reading should jump to near full-scale (~32767 at GAIN_ONE). This confirms each channel is wired correctly.

### Complete Wiring Reference

After completing all steps above, your wiring should match this diagram:

```
                           +5V BUS
                              |
        +---------------------+---------------------+
        |          |          |          |          |
    [Arduino]  [ADS1]    [ADS2]     [GPS]       [SD]
        |          |          |          |          |
        +---------------------+---------------------+
                          GND BUS

ARDUINO NANO PIN ASSIGNMENTS:
  A4 (SDA) ---------- ADS1 SDA -------- ADS2 SDA
  A5 (SCL) ---------- ADS1 SCL -------- ADS2 SCL
  D4 (GPS RX) ------- GPS TX
  D3 (GPS TX) ------- GPS RX
  D9 (Beeper) ------- MOSFET Gate (via 10k)
  D2 (LED) ---------- LED Anode (via 220 ohm)
  D10 (SD CS) ------- SD Card CS
  D11 (MOSI) -------- SD Card MOSI
  D12 (MISO) -------- SD Card MISO
  D13 (SCK) --------- SD Card SCK

ADS1115 #1 (ADDR=GND, address 0x48):
  A0 <-- Sensor P1_TOP
  A1 <-- Sensor P1_BOT
  A2 <-- Sensor P2_TOP
  A3 <-- Sensor P2_BOT

ADS1115 #2 (ADDR=VDD, address 0x49):
  A0 <-- Sensor P3_TOP
  A1 <-- Sensor P3_BOT
  A2 <-- Sensor P4_TOP
  A3 <-- Sensor P4_BOT
```

### Electronics Assembly Checklist

Before moving on, verify:

- [ ] 5V bus reads 5.0V +/- 0.1V with multimeter
- [ ] I2C scanner finds both ADS1115 modules (0x48 and 0x49)
- [ ] GPS outputs NMEA sentences on serial monitor
- [ ] SD card initializes and writes a test file successfully
- [ ] Status LED lights on D2
- [ ] Beeper sounds on D9
- [ ] All sensor JST headers wired to correct ADS1115 channels
- [ ] All solder joints inspected (no cold joints, no bridges)
- [ ] Total current draw with no sensors: ~120-150 mA (Modeled)

---

## Phase 3: Sensor Preparation (1-2 hours)

This phase covers preparing the FG-3+ fluxgate sensors and their signal conditioning circuitry. Each sensor outputs a frequency proportional to the magnetic field; the LM2917 converts this frequency to a voltage that the ADS1115 can read.

### Step 3.1: Fluxgate Sensor Testing

**Components needed:**
- 8x FG-3+ fluxgate sensors
- Power supply (5V regulated -- use the buck converter from Phase 2)
- Multimeter with frequency measurement, or oscilloscope

**Procedure:**

1. **Test each sensor individually** before building the full system. Power one sensor at a time.
2. Connect the sensor to 5V and GND.
3. The FG-3+ outputs a frequency signal (typically 49-120 kHz range depending on the ambient magnetic field).
4. Measure the output frequency with an oscilloscope or frequency counter. If using a multimeter with frequency mode, verify it can measure frequencies in the 50-120 kHz range.
5. The exact frequency does not matter at this point -- you are confirming that each sensor produces a stable frequency output when powered.
6. **Label each sensor** with a unique identifier (1T, 1B, 2T, 2B, 3T, 3B, 4T, 4B). Use the self-laminating wire labels.
7. **Check for defective sensors.** A sensor is defective if:
   - It draws no current (open circuit internally)
   - It draws excessive current (>40 mA)
   - Its output is stuck at 0V or 5V (no frequency output)
   - Its output frequency is unstable or jittery (>1% variation in a static field)

**Verification:** All 8 sensors produce a stable frequency output when individually powered.

### Step 3.2: LM2917 Frequency-to-Voltage Boards

**Components needed (per channel, x8 total):**
- 1x LM2917N-8 IC (8-pin DIP)
- Timing resistor and capacitor (values per LM2917 datasheet, chosen for the FG-3+ frequency range)
- Output filter capacitor
- 8-pin DIP socket (recommended -- do not solder the IC directly)

The LM2917 converts the frequency output of each FG-3+ sensor into a proportional DC voltage that the ADS1115 ADC can read.

**Procedure:**

1. **Determine component values.** The LM2917 output voltage is: V_out = f_in x C_t x R_t x V_cc, where:
   - f_in = sensor output frequency
   - C_t = timing capacitor
   - R_t = timing resistor
   - V_cc = supply voltage

   Choose R_t and C_t so that the output voltage spans 0.5V-4.0V over the expected frequency range. Consult the LM2917 datasheet application notes for recommended values for your specific FG-3+ frequency range.

2. **Build 8 identical frequency-to-voltage circuits.** You can build these on small perfboard sections, or order a simple PCB if building multiples. Each circuit is small (about 20mm x 15mm).

3. **If using pre-built F-to-V modules**, verify that the timing components are appropriate for the FG-3+ frequency range. Generic modules may need component swaps.

4. **Install DIP sockets first**, then insert the LM2917 ICs. This protects the ICs from soldering heat and allows easy replacement.

5. **Wire each F-to-V circuit:**
   - Input: from the FG-3+ frequency output
   - Output: to the corresponding JST connector / ADS1115 input
   - Power: from the 5V/GND bus

**Verification:**
1. Power one F-to-V circuit with a sensor connected.
2. Measure the DC output voltage with your multimeter. It should be a stable voltage in the 0.5-4.0V range.
3. Move a magnet near the sensor. The output voltage should change smoothly.
4. Repeat for all 8 channels.

### Step 3.3: Sensor Cable Fabrication

**Components needed:**
- 15 meters of 4-conductor shielded cable (22-24 AWG)
- 8x JST-XH 4-pin female connectors (with crimp terminals)
- JST crimp tool or small needle-nose pliers
- Heat shrink tubing
- Self-laminating wire labels

**Procedure:**

1. **Cut cables to length.** Each sensor cable runs from the sensor position on the frame to the electronics enclosure. Measure based on your frame layout:
   - For sensors near the center of the crossbar: ~60 cm
   - For sensors at the ends of the crossbar: ~120 cm
   - Add 15 cm of slack to each cable for strain relief and routing

   Typical cable lengths for a 2m crossbar with belt-mounted enclosure:
   | Sensor | Cable Length |
   |--------|-------------|
   | P1_TOP, P1_BOT (left end) | 120 cm |
   | P2_TOP, P2_BOT (center-left) | 80 cm |
   | P3_TOP, P3_BOT (center-right) | 80 cm |
   | P4_TOP, P4_BOT (right end) | 120 cm |

2. **Strip and terminate each cable.**
   - Strip the outer jacket back 25 mm at each end.
   - Separate the 4 conductors and the shield braid.
   - Strip each conductor 3 mm.
   - Crimp JST-XH female terminals onto each conductor.
   - Insert terminals into the JST-XH 4-pin housing in the correct order (Pin 1: 5V red, Pin 2: GND black, Pin 3: Signal white/yellow, Pin 4: Shield bare/green).

3. **Shield grounding (critical for noise performance):**
   - At the **enclosure end**: connect the cable shield to the GND bus.
   - At the **sensor end**: leave the shield wire disconnected (floating). Fold it back and cover with heat shrink.
   - Grounding the shield at only one end prevents ground loop currents that cause noise.

4. **Label both ends of each cable** with the sensor identifier (P1_TOP, P1_BOT, etc.).

5. **Continuity test every cable.** Use your multimeter on continuity mode to verify:
   - Pin 1 at one end connects to Pin 1 at the other end (not to any other pin).
   - No shorts between any pair of pins.
   - Shield is continuous from end to end.

**Verification:** All 8 cables pass continuity test with no shorts and correct pin-to-pin mapping.

### Step 3.4: Sensor Mounting Hardware

**Components needed:**
- 8x 3D-printed sensor clips (or cable ties + foam padding)
- EVA foam padding sheet (5mm, self-adhesive)

**Procedure:**

1. **If using 3D-printed clips:** Print from STL files in `hardware/cad/`. Use PETG filament for outdoor durability. Each clip friction-fits around the sensor body and snaps onto the PVC drop tube or crossbar.

2. **If using cable ties:** Wrap each sensor in a small piece of EVA foam padding (to prevent rattling and protect the sensor body), then secure to the drop tube or crossbar with two UV-resistant cable ties per sensor.

3. **Sensor orientation.** All sensors in a pair must be oriented the same way (same axis aligned with Earth's field). Mark the sensing axis on each sensor housing. When mounting, ensure all 8 sensors point in the same direction.

**Verification:** Each sensor sits snugly in its mount with no wobble. Mounts grip the drop tube securely without cracking.

---

## Phase 4: Frame Assembly (1-2 hours)

### Step 4.1: Carbon Fiber Crossbar Preparation

**Components needed:**
- 1x Carbon fiber tube (25mm OD x 2mm wall x 2m) or aluminum EMT conduit (1.25" x 2m)
- Drill with 5mm bit
- Tape measure
- Marker or pencil
- Fine-grit sandpaper (220 grit)
- Masking tape

**Procedure:**

1. **Measure and mark sensor positions.** From the left end of the crossbar, mark the following positions for the 4 sensor pairs:

   | Position | Distance from Left End | Pair |
   |----------|------------------------|------|
   | Pair 1 center | 25 cm | Left-most |
   | Pair 2 center | 75 cm | Center-left |
   | Pair 3 center | 125 cm | Center-right |
   | Pair 4 center | 175 cm | Right-most |
   | Center (D-ring) | 100 cm | Harness attachment |

   This produces 50 cm center-to-center spacing between pairs, with the total sensor span being 1.5 m.

2. **Drill mounting holes.** At each of the 4 sensor positions, drill a 5mm hole through both walls of the tube (the bolt will pass through the tube and the drop tube clamp).

   **Carbon fiber drilling tips:**
   - Wrap the tube with masking tape at the drill location to prevent splintering.
   - Use a sharp 5mm bit. Drill slowly with moderate pressure.
   - Support the tube from below to prevent cracking.
   - Clean up any fraying with fine sandpaper.

   **Aluminum drilling tips:**
   - Use a center punch to mark the hole before drilling.
   - Deburr the holes with a countersink or round file.

3. **Sand the ends** of the crossbar lightly with 220-grit sandpaper to remove sharp edges. If carbon fiber, seal the cut ends with CA glue (superglue) or clear nail polish to prevent fiber fraying.

4. **Install end caps.** Press-fit the 3D-printed end caps onto both ends of the crossbar. If they are loose, apply a drop of CA glue. End caps prevent water and debris from entering the tube.

5. **Install the center D-ring.** At the 100 cm mark (center of crossbar):
   - Place the pipe clamp collar around the tube and tighten.
   - Attach the 25mm D-ring to the clamp collar.
   - This is the main harness attachment point.

**Safety note:** Carbon fiber dust is an irritant. Wear a dust mask and eye protection when cutting or drilling carbon fiber. Work outdoors or in a well-ventilated area.

**Verification:**
- Crossbar is 2.0 m long with clean, capped ends.
- 4 mounting holes drilled at 25, 75, 125, and 175 cm from the left end.
- Center D-ring installed securely at 100 cm.
- No cracks, splits, or excessive fraying.

### Step 4.2: Sensor Drop Tubes

**Components needed:**
- 4x PVC conduit (3/4" Schedule 40, cut to 50 cm each)
- 4x M5 x 60mm stainless steel bolts with locknuts
- 4x Nylon spacers (M5 ID x 10mm)
- Pipe cutter or hacksaw
- Sandpaper (to deburr)

**Procedure:**

1. **Cut 4 PVC tubes to 50 cm length.** Deburr the cut ends with sandpaper.

2. **Drill a 5mm mounting hole** in each tube, centered 2 cm from the top end. This hole aligns with the crossbar mounting hole.

3. **Attach each drop tube to the crossbar.**
   - Insert the M5 bolt through: the crossbar hole, a nylon spacer, the PVC tube hole, a second nylon spacer (if needed for alignment), and secure with a locknut.
   - The nylon spacers prevent metal-on-composite contact and allow slight angular adjustment.
   - Tighten until snug but not so tight that the PVC tube cracks.

4. **Install sensor mounts.** For each drop tube:
   - **Bottom sensor:** Mount the bottom sensor at the bottom end of the PVC tube using a 3D-printed clip or cable tie. The sensor should be recessed about 2 cm inside the tube end, with the sensing element pointing downward.
   - **Top sensor:** Mount the top sensor on the crossbar, directly above the drop tube, using a 3D-printed clip or cable tie. The sensor sits at crossbar height.
   - The vertical separation between top and bottom sensors is approximately 48 cm (crossbar to bottom of drop tube minus sensor recesses).

5. **Route sensor cables.** Run each bottom sensor's cable up through the inside of the PVC drop tube, then along the crossbar toward the center. Top sensor cables route directly along the crossbar.

**Exploded view of a single drop tube assembly:**

```
                [Crossbar]
                    |
    [Top sensor]----+----[3D clip or cable tie]
                    |
         [M5 bolt + nylon spacer]
                    |
              [PVC tube 50cm]
                    |
                    |    <-- sensor cable routed inside tube
                    |
    [Bottom sensor]-+----[3D clip or cable tie]
                    |
              ~15-20 cm above ground (when worn)
```

**Verification:**
- All 4 drop tubes are securely bolted to the crossbar with no wobble.
- Sensors mount firmly in clips at top and bottom positions.
- Cables route cleanly through the tubes without pinching or kinking.
- The assembly is symmetric: all drop tubes hang vertically when the crossbar is held level.

### Step 4.3: Electronics Enclosure Mounting

**Components needed:**
- 1x Hammond 1554K IP65 enclosure (or equivalent, ~150x100x50mm)
- 2x PG7 cable glands
- 1x Gore-Tex pressure vent
- 8x M3 nylon standoffs (10mm)
- 2x Velcro straps (25mm x 30cm)
- Drill with step bit or appropriate hole saw

**Procedure:**

1. **Drill cable entry holes** in the enclosure:
   - Drill 2 holes for PG7 cable glands (12mm diameter each) on the bottom face of the enclosure.
   - One gland is for sensor cables (bundle all 8 through one gland, or use two glands -- one per side of the crossbar).
   - One gland is for the battery power cable.

2. **Drill the vent hole.** Drill one small hole (per vent specs) on the side of the enclosure for the Gore-Tex pressure vent. This prevents condensation buildup inside the sealed box.

3. **Install the cable glands and vent.** Thread them into the drilled holes and tighten the lockrings from inside.

4. **Mount nylon standoffs inside the enclosure.** Use M3 screws to secure 4-8 standoffs to the base of the enclosure. These support the perfboard or directly mount the Arduino Nano and modules.

5. **Install the electronics assembly** (from Phase 2) inside the enclosure:
   - Secure the perfboard to the standoffs with M3 screws.
   - Route sensor cables through the cable glands, leaving enough slack inside to reach the JST connectors.
   - Route the battery power cable through its cable gland.
   - Tighten the cable glands to create a water-resistant seal around the cables.

6. **Verify enclosure fit.** Close the lid and confirm:
   - The lid closes fully without pinching any wires.
   - All cables exit cleanly through cable glands.
   - The SD card can be accessed. Ideally, position the SD card module near the enclosure edge so you can swap cards without full disassembly. Alternatively, cut a small slot in the enclosure wall for card access (seal with tape during operation).
   - The status LED is visible through the enclosure wall. If not, drill a small hole and insert a clear LED light pipe, or mount the LED externally.

7. **Mount the enclosure for carrying.** Two options:
   - **Belt mount (recommended):** Use the Velcro straps to attach the enclosure to a belt or waist strap. This keeps weight on the hips and allows easy access.
   - **Crossbar mount:** Attach the enclosure to the center of the crossbar using Velcro straps or cable ties. This simplifies cable routing but adds weight to the crossbar.

**Verification:**
- Enclosure lid closes and seals properly.
- Cable glands grip all cables firmly.
- All connectors are accessible for plugging/unplugging sensors.
- SD card is accessible for data retrieval.
- Status LED visible from outside.
- Enclosure mounts securely in the chosen carry position.

### Step 4.4: Cable Routing and Management

**Components needed:**
- 2 meters of 10mm spiral cable wrap
- 50x UV-resistant cable ties (200mm)
- Self-laminating wire labels

**Procedure:**

1. **Bundle sensor cables along the crossbar.** Route all cables from the sensor positions toward the enclosure. Use spiral cable wrap or cable ties every 15-20 cm to keep cables neat and prevent snagging.

2. **Create a drip loop.** Where cables enter the enclosure (through the cable glands), form a small downward loop in each cable before it enters the gland. This prevents water from running along the cable and into the enclosure.

3. **Separate sensor cables from power cables.** Keep signal-carrying sensor cables at least 5 cm from power wires (battery leads, 5V bus wires) to minimize electrical noise pickup.

4. **Verify cable labels.** Each cable should be labeled at both the sensor end and the enclosure end with its identifier (P1_TOP, P1_BOT, etc.). This is essential for troubleshooting in the field.

5. **Secure cables to prevent mechanical stress on solder joints.** Use cable ties or hot glue to create strain relief points where cables connect to the enclosure and at each sensor mount.

**Verification:**
- All cables neatly routed and secured.
- No cables hanging loose that could snag on vegetation.
- All cables labeled at both ends.
- Drip loops formed at enclosure entry points.

---

## Phase 5: Harness System (30-60 minutes)

### Step 5.1: Shoulder Straps

**Components needed:**
- 1 set of padded shoulder straps (50mm wide -- salvaged from an old backpack is ideal)
- Alternatively, purchase padded camera strap or backpack replacement straps

**Procedure:**

1. **If salvaging from a backpack:** Remove the shoulder straps, keeping the upper attachment points and lower adjustment buckles intact. You need straps that can be adjusted for length and have some form of upper attachment loop or ring.

2. **If purchasing new:** Look for padded replacement backpack straps or dual camera straps with adjustable length. They should have D-ring or loop attachment points at the top.

3. **Attach the straps** to a common point at the back (a triangular plate, a webbing junction, or a salvaged backpack frame clip). The straps should meet at a point approximately at shoulder-blade height on the operator's back.

### Step 5.2: Bungee Isolation System

**Components needed:**
- 1m of 6mm bungee cord
- 4x spring-gate carabiners (5kN rated)
- 2x 25mm stainless steel D-rings
- 1x 20cm aluminum rod (spreader bar)

**Procedure:**

The bungee isolation system absorbs the vertical motion of walking so that sensor vibrations are damped rather than transmitted directly into the data. This is one of the most important features of the Pathfinder design.

1. **Attach D-rings to the bottom of the shoulder straps.** Sew or bolt a D-ring to each strap, at approximately waist level on the operator. These are the upper attachment points for the bungee system.

2. **Cut two bungee cord sections.** Each should be approximately 40 cm long (adjust based on operator height -- see fitting guide below).

3. **Attach the spreader bar.** The aluminum rod acts as a spreader to prevent the bungee cords from converging to a single point, which would allow the crossbar to swing side-to-side:
   - Drill a small hole near each end of the 20cm rod.
   - Thread a carabiner through each hole.
   - Clip the top carabiners to the D-rings on the shoulder straps.

4. **Connect the bungees.**
   - Tie one end of each bungee cord to the ends of the spreader bar (or to the carabiners).
   - Tie the other ends of the bungee cords to a central carabiner.
   - Clip the central carabiner to the D-ring on the center of the crossbar.

5. **Assembly order (top to bottom):**
   ```
   Shoulder straps
        |
   D-rings (sewn to straps)
        |
   Carabiners
        |
   Spreader bar (20cm aluminum rod)
        |
   Bungee cords (40cm each, one per side)
        |
   Central carabiner
        |
   Crossbar center D-ring
   ```

### Step 5.3: Fitting Guide

Proper harness fit is critical for data quality. If the sensors bounce excessively or drag on the ground, the data will be degraded.

**Target:** Bottom sensors 15-20 cm above ground during normal walking.

#### Fitting Procedure

1. **Don the harness** with the crossbar attached via the bungee system.
2. Stand upright on flat, level ground.
3. Have an assistant measure the height of the bottom sensors above the ground.
4. Adjust the shoulder strap length:
   - **Tighten straps** to raise the crossbar (increase ground clearance).
   - **Loosen straps** to lower the crossbar (decrease ground clearance).
5. Walk 10 paces at survey speed (~1 m/s). The sensors should gently bob up and down no more than 5 cm. If they bounce more than that, shorten the bungee cords.
6. Mark the final strap position with a permanent marker so you can reproduce the setting.

#### Height Adjustments by Operator

| Operator Height | Bungee Length | Strap Setting | Expected Ground Clearance |
|-----------------|---------------|---------------|---------------------------|
| 160-170 cm | 35 cm | Shorter (tighter) | 15-20 cm |
| 170-180 cm | 40 cm | Medium | 15-20 cm |
| 180-190 cm | 45 cm | Medium-long | 15-20 cm |
| 190-200 cm | 50 cm | Longer (looser) | 15-20 cm |

The target is the same for all operators (15-20 cm). Taller operators need longer bungee cords and more strap length to achieve the same sensor height.

#### Terrain Adaptation

- **Tall grass or rough ground:** Raise sensors to 25-30 cm to prevent snagging. This reduces near-surface sensitivity slightly but prevents damage.
- **Smooth, mown ground:** Lower sensors to 10-15 cm for maximum sensitivity.
- **Slopes:** Walk across-slope (contour lines) rather than up/down when possible. The bungee system compensates for moderate slopes.

#### Fit Verification Checklist

- [ ] Weight is on shoulders and hips, not on arms
- [ ] Hands rest lightly on crossbar for guidance only (no gripping)
- [ ] Bottom sensors 15-20 cm above ground on level surface
- [ ] Walking at 1 m/s, sensor bounce is less than 5 cm
- [ ] Crossbar is level (not tilted left or right)
- [ ] Operator can don and doff the harness in under 30 seconds using carabiners

---

## Phase 6: Initial Testing (30-60 minutes)

### Step 6.1: Flash Firmware

**Prerequisites:**
- PlatformIO installed (VS Code with PlatformIO extension, or PlatformIO CLI)
- Arduino Nano connected via USB
- All wiring from Phase 2 complete

**Procedure:**

1. Connect the Arduino Nano to your computer via USB.
2. Navigate to the firmware directory and build/upload:
   ```bash
   cd firmware && pio run -t upload
   ```
3. Wait for the upload to complete. PlatformIO will auto-detect the serial port in most cases.
4. If the upload fails with a serial port error, specify the port manually:
   ```bash
   pio run -t upload --upload-port /dev/ttyUSB0   # Linux
   pio run -t upload --upload-port COM3            # Windows
   ```

### Step 6.2: Power-On Verification

1. Disconnect USB (you will now run on battery power).
2. Connect the LiPo battery.
3. Turn on the power switch.
4. **Observe the status LED sequence:**

   | LED Pattern | Meaning | Duration |
   |-------------|---------|----------|
   | Fast blink (100ms) | Startup / initialization | 2-3 seconds |
   | Medium blink (500ms) | Waiting for GPS lock | Until GPS locks (may be minutes) |
   | Slow blink (2000ms) | Normal logging | Continuous during operation |
   | Fast blink (100ms) | Error condition | Continuous until resolved |

5. If the LED shows fast blink after startup and does not transition to medium blink, there is an initialization error. Connect via USB and check the serial debug output.

### Step 6.3: Serial Debug Output

1. Connect the Arduino Nano via USB (you can run on USB power or battery simultaneously).
2. Open a serial monitor at **115200 baud**.
3. You should see initialization messages:
   ```
   Pathfinder Gradiometer v1.4.0
   Initializing...
   ADS1115 #1 (0x48): OK
   ADS1115 #2 (0x49): OK
   SD card: OK
   GPS: Waiting for fix...
   Logging to PATH0001.CSV
   ```
4. Verify all modules report OK. Any module that reports FAIL needs its wiring checked (refer to Troubleshooting section).

### Step 6.4: Sensor Readings Check

1. With all 8 sensors connected and the system powered on, observe the serial debug output.
2. Every 10 samples (approximately once per second at 10 Hz), the firmware prints current readings:
   ```
   P1: T=16234 B=16890 G=656
   P2: T=16180 B=16802 G=622
   P3: T=16301 B=16945 G=644
   P4: T=16155 B=16778 G=623
   ```
   - T = top sensor raw ADC value
   - B = bottom sensor raw ADC value
   - G = gradient (bottom minus top)

3. **What to look for:**
   - All 8 channels should show non-zero readings.
   - Gradient values should be roughly similar across all 4 pairs (within ~20% of each other in a uniform field).
   - Readings should be stable (not jumping wildly between samples).
   - No channel should be at 0 or 32767 (saturation).

4. **Move a ferrous object (screwdriver, key) near one sensor pair.** The gradient for that pair should change noticeably while the others remain stable. This confirms the channels are independent and correctly mapped.

### Step 6.5: First Field Walk

1. Take the fully assembled Pathfinder outdoors to an open area with GPS sky view.
2. Power on and wait for GPS lock (LED transitions from medium to slow blink). This may take 1-2 minutes for a cold start.
3. Walk a short test line (20-30 meters) at a steady pace (~1 m/s). The beeper should sound once per second to help maintain pace.
4. After the walk, turn off the system.
5. Remove the SD card and insert it in your computer.
6. Look for the CSV data file (e.g., `PATH0001.CSV`).
7. Visualize the test data:
   ```bash
   python firmware/tools/visualize_data.py PATH0001.CSV
   ```
   This produces a time-series plot of all 4 gradients. You should see smooth traces with some variation where you walked over buried objects or changes in soil.

8. For a spatial map (requires GPS fix):
   ```bash
   python firmware/tools/visualize_data.py PATH0001.CSV --map
   ```

### Step 6.6: Calibration

After verifying that the system collects data correctly, perform the calibration procedure. Calibration ensures that all 4 sensor pairs produce consistent gradient readings for the same magnetic anomaly.

**Note:** A dedicated calibration document is under development. In the meantime, the basic calibration procedure is:

1. Find a location with known or no magnetic anomalies (open field, away from buildings and metal fences).
2. Collect data while walking a straight line at constant pace.
3. Compute the mean and standard deviation of each pair's gradient in the quiet zone.
4. Apply offset corrections so all pairs read approximately zero gradient in the quiet zone.
5. Walk over a known ferrous target (e.g., a buried nail at 20 cm depth) and verify all pairs produce similar gradient magnitudes when passing directly over it.

---

## Troubleshooting Build Issues

### ADS1115 ADC Not Found on I2C Bus

**Symptom:** I2C scanner does not find 0x48 or 0x49. Serial output shows "ADS1115 not found."

**Solutions (check in order):**
1. Verify power: measure 5V on the ADS1115 VDD pin with a multimeter.
2. Verify ground: confirm continuity between ADS1115 GND and Arduino GND.
3. Check SDA/SCL wiring: confirm A4 connects to all SDA pins and A5 connects to all SCL pins.
4. Verify ADDR pin: Module 1 ADDR should measure 0V (GND). Module 2 ADDR should measure 5V (VDD).
5. Check for solder bridges on the ADS1115 module.
6. Try each ADS1115 module individually (disconnect one, scan for the other).
7. Upload the I2C scanner sketch and check all detected addresses. If the module appears at an unexpected address, the ADDR wiring is wrong.

### GPS Not Receiving Data

**Symptom:** No NMEA sentences on serial monitor. GPS power LED may or may not be lit.

**Solutions:**
1. Verify power: GPS VCC should measure 5V (or 3.3V if using the 3V3 pin).
2. Check TX/RX wiring: GPS TX must connect to Arduino D4 (RX). This is the most common mistake -- people connect TX to TX.
3. Verify baud rate: the NEO-6M default is 9600. The firmware uses `GPS_BAUD` defined in `config.h` (default 9600). If your GPS module uses a different baud rate, update the config.
4. Try powering the GPS module on its own with just VCC and GND, and probe the TX pin with an oscilloscope or multimeter on AC mode. You should see activity.
5. Ensure the GPS antenna has a clear view of the sky. It will not get a fix indoors, and the first cold-start fix can take 1-2 minutes.

### SD Card Not Initializing

**Symptom:** Serial output shows "SD card failed" or similar.

**Solutions:**
1. Verify the CS pin is connected to D10 (not another pin).
2. Check all SPI wiring: MOSI (D11), MISO (D12), SCK (D13).
3. Verify power to the SD module.
4. Try a different SD card. Some cards are incompatible with certain modules.
5. Ensure the card is formatted as FAT32. Cards larger than 32 GB often come formatted as exFAT, which the Arduino SD library cannot read. Use a formatting tool to reformat as FAT32.
6. Check that the card is fully seated in the module's socket.
7. Try a Class 10 card if you are using a slower card.

### No Gradient Readings (All Zeros or Static Values)

**Symptom:** ADC values are stuck at 0, 32767, or a constant value that does not change when moving a magnet near the sensors.

**Solutions:**
1. Check sensor power: each fluxgate sensor should be receiving 5V and GND.
2. Verify the LM2917 frequency-to-voltage converter is working: measure the DC output voltage with a multimeter. It should be between 0.5V and 4.0V and should change when you move a magnet near the sensor.
3. Check the signal path from the LM2917 output through the cable to the ADS1115 analog input.
4. If the ADC reads 32767 (full scale), the input voltage is above the ADC range. Check the LM2917 output voltage and adjust timing components if needed.
5. If the ADC reads 0, the signal wire may be disconnected or shorted to GND.
6. Check the channel mapping: verify that each sensor cable goes to the correct ADS1115 input (see the table in Step 2.8).

### Inconsistent Readings Between Pairs

**Symptom:** One or two pairs show much noisier or offset readings compared to others, even in a uniform magnetic field.

**Solutions:**
1. **Cable routing:** If a sensor cable runs close to the power cable or buck converter, it will pick up switching noise. Re-route sensor cables at least 5 cm from power wires.
2. **Sensor alignment:** Verify that all sensors in a pair are oriented with the same axis. A rotated sensor will read a different component of the magnetic field.
3. **Loose connections:** Wiggle each connector and cable while observing serial output. If readings jump when you move a cable, there is a bad connection.
4. **Shield grounding:** Verify that each cable shield is grounded at the enclosure end only, not at the sensor end.
5. **Bypass capacitors:** Ensure 0.1 uF caps are installed on each ADS1115 VDD pin and on the sensor power rail.
6. **LM2917 component variation:** If using hand-selected timing components, slight differences in resistor/capacitor values between channels will produce offset differences. This is corrected during calibration.

### Beeper Not Sounding

**Symptom:** No audible tone during operation.

**Solutions:**
1. Check that `ENABLE_BEEPER` is set to 1 in `config.h`.
2. Verify the MOSFET is correctly oriented (check pinout: Source, Gate, Drain for TO-92).
3. Measure the gate voltage with a multimeter while the beeper should be active. It should pulse between 0V and ~5V.
4. Check the buzzer polarity (positive terminal to 5V, negative to MOSFET drain).
5. Try connecting the buzzer directly between D9 and GND temporarily to rule out MOSFET issues.

### Status LED Not Working

**Symptom:** LED never illuminates.

**Solutions:**
1. Check LED polarity (anode to D2 via resistor, cathode to GND).
2. Verify the 220 ohm resistor is correct (red-red-brown-gold).
3. Measure voltage on D2 with a multimeter when the LED should be on. It should read ~5V.
4. Try a different LED.

### System Resets or Freezes

**Symptom:** The system restarts unexpectedly or stops logging data.

**Solutions:**
1. **Insufficient power:** The buck converter may not supply enough current when all sensors are active. Measure the 5V rail under full load. If it droops below 4.5V, you need a higher-rated regulator (>600 mA output).
2. **Loose battery connection:** Vibration during walking can cause intermittent power drops. Secure the battery connector with tape or a cable tie.
3. **SD card write failures:** If the SD card is slow or failing, the system may stall. Try a different Class 10 card.
4. **Memory issues:** The Arduino Nano has only 2 KB of SRAM. If the firmware is modified with large strings or arrays, it may run out of memory. Check the build output for SRAM usage -- it should be below 75%.

---

## Safety Notes

### Battery Safety

- Use a proper LiPo balance charger (not a generic power adapter).
- Never charge the battery unattended.
- Store the battery in a LiPo-safe bag when not in use.
- Inspect the battery before each use: do not use if swollen, dented, or showing damaged insulation.
- Disconnect the battery from the system when not in use.
- Operating temperature range: 0 to 45 degrees C. Do not charge below 0 degrees C.

### Electrical Safety

- Double-check polarity before connecting the battery for the first time. Reverse polarity will destroy the buck converter and potentially other modules.
- Use a fuse or polyfuse on the battery positive lead (1A recommended) to protect against short circuits.
- Cover all exposed connections with heat shrink tubing or electrical tape.
- Work on a non-conductive surface when soldering or wiring.

### Field Safety

- Use quick-release carabiners so the operator can doff the harness quickly in an emergency.
- Do not survey during thunderstorms. The carbon fiber crossbar can attract lightning.
- Take 10-minute rest breaks every hour to prevent back and shoulder fatigue.
- Verify proper harness fit before each survey session to prevent strain injuries.
- Stay aware of terrain hazards (holes, roots, fences) since your attention will be partly on the instrument.

### Carbon Fiber Safety

- Wear a dust mask and safety glasses when cutting or drilling carbon fiber.
- Carbon fiber splinters are extremely fine and difficult to remove from skin. Handle cut edges carefully.
- Work in a well-ventilated area or outdoors.
- Clean up all carbon fiber dust and debris after cutting.

---

## Pre-Survey Checklist

Run through this list before every survey session:

### Hardware Checks
- [ ] All crossbar bolts tight (drop tubes do not wobble)
- [ ] Bungee cords show no fraying or wear
- [ ] All carabiners close and lock properly
- [ ] Power switch functions (LED responds)
- [ ] Battery charged (>7.0V measured at connector)
- [ ] SD card inserted with free space (>100 MB)
- [ ] All sensor cables securely connected
- [ ] All sensor cables labeled correctly
- [ ] Sensors aligned (same orientation)
- [ ] Cable glands tight on enclosure

### Software Checks
- [ ] GPS acquires fix (LED slow blink)
- [ ] Serial debug shows all modules OK
- [ ] ADC readings from all 8 channels are reasonable
- [ ] Beeper sounds at correct interval
- [ ] New log file created on SD card

### Fit Checks
- [ ] Harness adjusted for operator
- [ ] Bottom sensors at 15-20 cm height
- [ ] Crossbar level
- [ ] Can walk at 1 m/s without sensors dragging

---

## Field Repair Kit

Carry these items in a belt pouch during surveys:

| Item | Quantity | Purpose |
|------|----------|---------|
| Multi-tool | 1 | General repairs |
| Spare carabiners | 2 | Replace failed clip |
| Cable ties (200mm) | 10 | Emergency cable/sensor mounting |
| Duct tape | 1 small roll | Temporary fixes |
| Spare bungee cord (1m) | 1 | Replace snapped bungee |
| Spare JST connectors | 2 sets | Replace damaged sensor cables |
| Electrical tape | 1 small roll | Insulate exposed wires |
| Spare SD card (formatted FAT32) | 1 | Replace failed card |
| Spare LiPo battery | 1 | Extended survey sessions |
| Spare fuse (1A) | 2 | Replace blown fuse |
| Small screwdriver set | 1 | Tighten terminal blocks |

---

## Appendix A: Firmware Configuration Reference

Key settings in `firmware/include/config.h` that you may want to adjust:

| Setting | Default | Description |
|---------|---------|-------------|
| `NUM_SENSOR_PAIRS` | 4 | Number of active sensor pairs (1-4) |
| `SAMPLE_RATE_HZ` | 10 | Readings per second |
| `GPS_BAUD` | 9600 | GPS module baud rate |
| `ADC_GAIN` | GAIN_ONE | ADS1115 gain (+/- 4.096V range) |
| `ADC_DATA_RATE` | 128 | ADS1115 samples per second per channel |
| `ENABLE_BEEPER` | 1 | Enable/disable pace beeper |
| `BEEP_INTERVAL_MS` | 1000 | Milliseconds between beeps |
| `SERIAL_DEBUG` | 1 | Enable serial debug output (disable to save power) |
| `SD_FLUSH_INTERVAL` | 10 | Flush SD buffer every N samples |
| `ENABLE_WATCHDOG` | 0 | Hardware watchdog timer (test before enabling) |
| `ENABLE_RTC` | 0 | DS3231 real-time clock for absolute timestamps |

---

## Appendix B: Pin Reference Card

Print this and tape it inside the enclosure lid for field reference.

```
PATHFINDER PIN REFERENCE - Arduino Nano
=======================================
DIGITAL PINS:
  D2  = Status LED (via 220R)
  D3  = GPS TX (to GPS RX, unused by default)
  D4  = GPS RX (from GPS TX)
  D9  = Beeper (via MOSFET)
  D10 = SD Card CS
  D11 = SD Card MOSI (fixed HW SPI)
  D12 = SD Card MISO (fixed HW SPI)
  D13 = SD Card SCK  (fixed HW SPI)

ANALOG PINS:
  A4  = I2C SDA (to both ADS1115)
  A5  = I2C SCL (to both ADS1115)

I2C ADDRESSES:
  0x48 = ADS1115 #1 (ADDR=GND) - Pairs 1-2
  0x49 = ADS1115 #2 (ADDR=VDD) - Pairs 3-4

ADC CHANNEL MAP:
  ADS1 A0 = Pair 1 Top    ADS2 A0 = Pair 3 Top
  ADS1 A1 = Pair 1 Bot    ADS2 A1 = Pair 3 Bot
  ADS1 A2 = Pair 2 Top    ADS2 A2 = Pair 4 Top
  ADS1 A3 = Pair 2 Bot    ADS2 A3 = Pair 4 Bot

POWER:
  7.4V LiPo -> Switch -> Buck -> 5V bus
  Serial Debug: 115200 baud
```

---

## Appendix C: Related Documentation

| Document | Location | Contents |
|----------|----------|----------|
| Wiring Guide | `firmware/WIRING.md` | Detailed pin-by-pin wiring reference |
| Bill of Materials | `hardware/bom/README.md` | Full priced BOM with suppliers |
| BOM Source Data | `hardware/bom/pathfinder-bom.csv` | Raw BOM data (CSV) |
| Frame Design | `hardware/cad/frame-design.md` | Frame dimensions, materials, parts |
| Main Board Schematic | `hardware/schematics/main-board.md` | Circuit diagrams, power budget |
| Design Concept | `docs/design-concept.md` | System architecture, performance targets |
| Firmware Config | `firmware/include/config.h` | All configurable parameters |
| Data Visualization | `firmware/tools/visualize_data.py` | Python tool for plotting survey data |
| Platform Variants | `docs/platform-variants.md` | Backpack and drone configurations |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-18 | Initial comprehensive build guide |
