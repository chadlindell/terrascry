# Crossbar Physical Layout

## Overview

The Pathfinder crossbar must accommodate multiple sensor types with vastly different electromagnetic signatures. This document specifies the physical placement of every component along the crossbar, optimized to minimize cross-sensor interference while maintaining practical weight distribution and ergonomics.

## Coordinate System

Position is measured from the electronics enclosure (center of crossbar), which is defined as **0 cm**. Negative positions are to the operator's left; positive positions extend to the right toward the fluxgate sensor array.

```
         LEFT (−)                    CENTER (0)                    RIGHT (+)
         ◄──────────────────────────────┼──────────────────────────────────►

  LiDAR    EMI TX     Electronics    EMI RX     Fluxgate Array
  -100cm   -50cm       0cm          +10cm    +25cm  +50cm  +75cm  +100cm
    ●        ◉          █              ◉        ╫      ╫      ╫      ╫
                                                ║      ║      ║      ║
                                               ╫╫    ╫╫    ╫╫    ╫╫
                                              (pairs, top+bottom)

         GPS antenna on vertical mast above center (0cm, +50cm height)
         Sensor pod at center (IMU, GPS receiver, barometer, RTC)
         Camera below center, pointing down
         IR temperature sensor between pod and first fluxgate pair
```

## Position Table

| Position | Component | Reason for Placement |
|----------|-----------|---------------------|
| -100 cm | RPLiDAR C1 | Maximum separation from fluxgates (2m); motor magnetic field falls off as 1/r³ |
| -50 cm | EMI TX coil | 1.5m from nearest fluxgate; 1.0m from EMI RX (standard EM38 geometry) |
| -10 cm | EMI RX coil | 1.0m from TX coil; placed near center to minimize cantilever moment |
| 0 cm | Electronics enclosure | Center of mass for ergonomic balance; contains ESP32, ADC, power supply |
| 0 cm (mast) | GPS antenna (on vertical mast) | Elevated above crossbar for clear sky view; centered for balance |
| 0 cm (pod) | Sensor pod | IMU at center for best tilt measurement; GPS receiver near antenna |
| +15 cm | MLX90614 IR sensor | Between electronics and first fluxgate; pointing down at ground |
| +20 cm | ESP32-CAM | Near center, pointing down; shielded metal enclosure |
| +25 cm | Fluxgate Pair 1 | Start of fluxgate array; 75 cm from EMI RX, 1.25m from EMI TX |
| +50 cm | Fluxgate Pair 2 | 50 cm spacing between pairs |
| +75 cm | Fluxgate Pair 3 | Continuing array |
| +100 cm | Fluxgate Pair 4 | End of array; maximum separation from all interference sources |

## Separation Distances (Key Pairs)

| Source | Victim | Distance | Adequate? |
|--------|--------|----------|-----------|
| EMI TX (-50) | Fluxgate 1 (+25) | 75 cm | Marginal — TDM required |
| EMI TX (-50) | Fluxgate 4 (+100) | 150 cm | Good with TDM |
| LiDAR (-100) | Fluxgate 1 (+25) | 125 cm | Good — motor field ~3 nT at this distance |
| LiDAR (-100) | Fluxgate 4 (+100) | 200 cm | Excellent — motor field <0.5 nT |
| EMI TX (-50) | EMI RX (-10) | 100 cm | By design (EM38 equivalent geometry) |
| GPS antenna (0, +50h) | EMI TX (-50) | ~71 cm | Adequate with LP filter on TX |
| Electronics (0) | Fluxgate 1 (+25) | 25 cm | Adequate — digital noise shielded by enclosure |

## Crossbar Construction

### Material Options

| Material | Weight (2m) | Stiffness | Cost | Notes |
|----------|-------------|-----------|------|-------|
| Carbon fiber tube 25mm OD | ~200g | Excellent | $40 | Non-magnetic, non-conductive |
| Aluminum tube 25mm OD | ~350g | Good | $15 | Non-magnetic, conductive (eddy current concern for EMI) |
| Fiberglass tube 25mm OD | ~300g | Good | $20 | Non-magnetic, non-conductive |

**Recommendation**: Carbon fiber for the fluxgate section (+25 to +100 cm). Fiberglass or carbon fiber for the rest. Avoid aluminum near the EMI coils (eddy currents in aluminum tube would affect conductivity measurements).

### Mounting Hardware

| Component | Mount Type | Connector |
|-----------|-----------|-----------|
| Fluxgate pairs | PVC drop tubes, clamped to crossbar | 3D-printed clips |
| EMI TX/RX coils | Clamped at fixed positions | 3D-printed brackets with set screws |
| LiDAR | Bracket at crossbar end | 3D-printed mount |
| Electronics enclosure | Center clamp + harness attachment | Quick-release plate |
| Sensor pod | Clamp at center, above electronics | 3D-printed cradle |
| GPS antenna mast | Vertical tube, 50 cm above crossbar | Clamp + telescoping tube |
| Camera | Under-crossbar bracket at center | 3D-printed mount |
| IR sensor | Clip mount between electronics and fluxgates | 3D-printed clip |

## Cable Routing Rules

To minimize electromagnetic coupling between cables:

1. **Analog signal cables** (fluxgate → LM2917 → ADC): Route along the RIGHT side of the crossbar, bundled together. Use shielded cable (braided shield grounded at the electronics enclosure end only).
2. **Digital cables** (I2C to sensor pod, UART to GPS, SPI to SD card): Route along the LEFT side of the crossbar, or through the center of the tube.
3. **Power cables**: Route through the center of the crossbar tube or along the bottom. Use twisted pairs for supply + return to minimize loop area.
4. **EMI TX drive cable**: Coaxial or tightly twisted pair from electronics to TX coil at -50 cm. Keep >10 cm from all other cables.
5. **EMI RX signal cable**: Shielded twisted pair (STP) from RX coil to electronics. This carries μV-level signals — maximum separation from TX drive cable and digital cables.
6. **LiDAR USB cable**: Standard USB cable routed to the left, away from analog paths.
7. **Sensor pod cable**: Cat5 STP cable, 1-2m, from center to pod location. Carries differential I2C (PCA9615).

### Cable Crossing Rule

If any cable must cross from left to right of the crossbar, it crosses at 90° (perpendicular) to minimize inductive coupling. Never run analog and digital cables in parallel for more than 5 cm.

## M8 IP67 Connectors

All external connections use M8 IP67 circular connectors for weatherproofing and quick field assembly:

| Connector | Location | Pins | Function |
|-----------|----------|------|----------|
| M8-4F #1 | Electronics enclosure | 4 | Sensor pod (VCC, GND, SDA_D+, SCL_D-) |
| M8-4F #2 | Electronics enclosure | 4 | EMI TX coil (TX+, TX-, Shield, NC) |
| M8-4F #3 | Electronics enclosure | 4 | EMI RX coil (RX+, RX-, Shield, NC) |
| M8-8F #4 | Electronics enclosure | 8 | Fluxgate Pair 1+2 (F1+, F1-, F2+, F2-, VCC1, VCC2, GND, GND) |
| M8-8F #5 | Electronics enclosure | 8 | Fluxgate Pair 3+4 (same pinout) |
| USB-C | Electronics enclosure | - | LiDAR (pass-through to Jetson) |
| SMA | Sensor pod | 1 | GPS antenna |

## Weight Distribution

| Section | Components | Weight |
|---------|-----------|--------|
| Left (-100 to -50) | LiDAR, EMI TX coil | ~250g |
| Center (-50 to +25) | Electronics, EMI RX, pod, camera, IR | ~600g |
| Right (+25 to +100) | 4 fluxgate pairs + cables | ~500g |
| GPS mast (above center) | Antenna + mast | ~100g |
| **Total crossbar assembly** | | **~1450g** |

Balance point is slightly right of center due to fluxgate weight. The harness attachment point should be at approximately +5 cm to compensate.

## References

- Geonics EM38 manual (coil separation geometry)
- Bartington Grad601 sensor array layout
- IP67 M8 connector specifications (TE Connectivity, Binder)
