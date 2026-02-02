# Pathfinder Frame and Harness Design

## Overview

This document specifies the physical harness and frame system for the Pathfinder 4-channel fluxgate gradiometer. The design follows the proven "trapeze" configuration used by commercial systems (Bartington Grad601, SENSYS MagWalk) but extends it to 4 sensor pairs for wider swath coverage.

## Design Goals

- **Weight**: Frame/harness under 500g (1.15 kg total with sensors + electronics)
- **Swath**: 1.5m coverage with 4 sensor pairs at 50cm spacing
- **Ergonomics**: Hands-free operation, weight on shoulders/hips
- **Cost**: Frame + harness under $200
- **Field-rugged**: IP65 electronics, weather-resistant materials

## Physical Configuration

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
       |T|       |T|       |T|       |T|    <- TOP sensors
       | |       | |       | |       | |
       | |       | |       | |       | |    <- PVC drop tubes (50cm)
       |B|       |B|       |B|       |B|    <- BOTTOM sensors
       +-+       +-+       +-+       +-+       15-20cm above ground

       <--50cm--><--50cm--><--50cm-->
                 1.5m swath
```

## Component Selection

### Crossbar

| Material | Weight | Cost | Recommendation |
|----------|--------|------|----------------|
| Carbon fiber 25mm OD x 2mm wall | 150g | $40 | **Primary choice** |
| Aluminum EMT 1.25" conduit | 280g | $15 | Budget alternative |
| Fiberglass tube | 200g | $30 | Good compromise |

**Source**: DragonPlate, RockWest Composites, or hardware store (aluminum)

### Sensor Drop Tubes

- **Material**: 3/4" Schedule 40 PVC electrical conduit
- **Length**: 50cm each (4 tubes)
- **Attachment**: M5 through-bolts with nylon spacers
- **Weight**: ~120g total

### Harness System

**Recommended approach**: Salvage straps from old backpack

Components:
- Padded shoulder straps (50mm wide)
- Waist belt for stability
- 2x bungee cords (40cm, 6mm diameter)
- Spreader bar (20cm aluminum rod)
- 4x climbing carabiners (5kN rated)

### Electronics Housing

- **Enclosure**: Hammond 1554K or equivalent IP65 box (~$15)
- **Mounting**: Belt-mount recommended (easy access, good GPS reception)
- **Cable entry**: PG7 cable glands with drip loop

## Parts List

### Crossbar Assembly ($27-52)

| Part | Qty | Est. Cost |
|------|-----|-----------|
| Carbon fiber tube 2m x 25mm | 1 | $40 |
| End caps (3D-printed) | 4 | $5 |
| Center mount D-ring | 1 | $3 |
| Pipe clamp collar | 1 | $4 |

### Sensor Drop Tubes ($28)

| Part | Qty | Est. Cost |
|------|-----|-----------|
| PVC conduit 3/4" x 50cm | 4 | $8 |
| M5 x 60mm bolts + locknuts | 4 | $4 |
| Nylon spacers | 4 | $3 |
| 3D-printed sensor clips | 8 | $8 |
| Foam padding sheet | 1 | $5 |

### Harness System ($19-29)

| Part | Qty | Est. Cost |
|------|-----|-----------|
| Shoulder straps (salvaged) | 1 set | $0-10 |
| Bungee cord 6mm x 1m | 1 | $3 |
| Carabiners 5kN | 4 | $12 |
| D-rings 25mm | 2 | $2 |
| Spreader bar | 1 | $2 |

### Electronics Housing ($28)

| Part | Qty | Est. Cost |
|------|-----|-----------|
| IP65 enclosure | 1 | $15 |
| PG7 cable glands | 2 | $4 |
| Gore-Tex vent | 1 | $3 |
| Velcro straps | 2 | $4 |
| M3 standoffs | 8 | $2 |

### Cables and Connectors ($44)

| Part | Qty | Est. Cost |
|------|-----|-----------|
| 4-conductor shielded cable | 15m | $15 |
| JST-XH 4-pin connectors | 10 sets | $8 |
| Spiral cable wrap | 2m | $5 |
| Heat shrink tubing kit | 1 | $5 |
| Cable ties (UV-resistant) | 50 | $3 |
| Label tape | 1 | $8 |

## Cost Summary

| Category | Cost Range |
|----------|------------|
| Crossbar assembly | $27-52 |
| Sensor drop tubes | $28 |
| Harness system | $19-29 |
| Electronics housing | $28 |
| Cables and connectors | $44 |
| **Frame/Harness Total** | **$146-181** |

## Assembly Overview

### 1. Crossbar Preparation
- Cut tube to 2.0m length
- Mark sensor positions: 25cm, 75cm, 125cm, 175cm from left end
- Drill 5mm mounting holes at each position
- Install center mount D-ring at 100cm (center)
- Install end caps

### 2. Drop Tube Assembly
- Cut 4x PVC tubes to 50cm
- Drill mounting holes 2cm from top
- Install bottom sensor mounts (friction-fit)
- Attach to crossbar with through-bolts

### 3. Sensor Installation
- Mount top sensors on crossbar with clips
- Mount bottom sensors in PVC tube ends
- Route cables inside tubes and along crossbar
- Bundle cables with spiral wrap

### 4. Harness Integration
- Attach D-rings to harness back panel
- Connect bungee cords through spreader bar
- Attach to crossbar center via carabiner
- Adjust height for 15-20cm ground clearance

### 5. Electronics Mounting
- Mount Arduino, ADC, GPS in enclosure
- Attach enclosure to waist belt
- Connect sensor cables through cable glands
- Install pressure vent

## Height Adjustment

Target: Bottom sensors 15-20cm above ground

1. Don harness
2. Attach crossbar via carabiner
3. Stand upright on level ground
4. Measure bottom sensor height
5. Adjust shoulder straps until correct
6. Mark strap position with marker

**Terrain adaptation**:
- Tall grass: Raise sensors to 25-30cm
- Smooth ground: Lower to 10-15cm for better sensitivity

## Pre-Survey Checklist

- [ ] All bolts tight
- [ ] Bungee cords show no wear
- [ ] Carabiners close and lock
- [ ] Power switch functions
- [ ] SD card has free space
- [ ] Cables securely connected
- [ ] Sensors aligned

## Field Repair Kit

Carry in belt pouch:
- Multi-tool
- Spare carabiners (2x)
- Zip ties (10x)
- Duct tape
- Spare bungee cord (1m)
- JST connectors (2 sets)
- Electrical tape
- Spare batteries
- Spare SD card

## Safety Notes

- Use quick-release carabiners for emergency doffing
- Do not survey during thunderstorms (carbon fiber)
- Take 10-minute breaks every hour
- Verify proper harness fit to prevent back strain

## References

- [Bartington Grad601 Manual](https://www.bartingtondownloads.com/wp-content/uploads/OM1800.pdf)
- [Berkeley Field Guide](https://arf.berkeley.edu/files/attachments/books/Bartington_Grad601_2_Setup_And_Operation_1.pdf)
- [SENSYS MagWalk](https://sensysmagnetometer.com/products/magwalk-magnetometer-survey-kit/)
