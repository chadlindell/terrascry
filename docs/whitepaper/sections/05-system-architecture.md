# 5. System Architecture - Micro-Probe Design

## 5.1 Design Philosophy

**"Archaeologist brain first, engineer brain second"**

- **Goal:** Smallest possible hole (10-16 mm OD, target: 12 mm)
- **Reality:** Need enough physics (coil area, electrode contact) for good signal
- **Constraint:** Many thin, gentle holes rather than few big ones

## 5.2 Probe Overview (Passive Micro-Probes)

Each probe is **passive** - no electronics downhole. Only sensors and wiring.

### MIT Coil Set
- **1× TX coil** + **1× RX coil** on ferrite cores
- **Ferrite cores:** 6-8 mm diameter × 40-80 mm long
- **Mounting:** Glued along rod (not in bulky head)
- **Orientation:** Orthogonal or slightly separated to reduce direct coupling
- **Wire:** Fine wire (34-38 AWG), many turns for signal strength

### ERT-Lite Rings
- **2-3 narrow ring electrodes** (3-5 mm wide bands)
- Standard positions: **0.5 m & 1.5 m** from tip
- Add deeper ring at **2.5-3.0 m** for longer rods
- **Material:** Stainless steel or copper foil
- **Mounting:** Bonded with epoxy, flush with rod surface

### Rod
- **Fiberglass or carbon-fiber** segments
- **OD:** 10-16 mm (target: 12 mm)
- **Length:** 1.5 m segments, threaded couplers to reach 3 m
- **Weight:** ~50-100 g per meter (much lighter than 25mm design)
- **Metal pilot rod** used only to make the hole (removed before sensor insertion)

### Surface Junction Box
- **Small IP-rated box** at top of each rod
- **Terminal block** for coil and electrode leads
- **Optional small buffer amp** (if needed)
- **Cable strain relief** and probe ID labeling
- **No heavy electronics** - just connections

## 5.3 Central Electronics Hub

### MIT Driver/Receiver
- **Central DDS sine source** (e.g., AD9833)
- **TX driver amplifier** - drives all probe TX coils via cables
- **RX low-noise amplifier chain** - receives from all probe RX coils
- **ADC/lock-in detection** (digital or analog)
- **MCU** (ESP32 or STM32) for control and data acquisition
- **Multi-probe cable harness** - connects to all probes

### ERT System
- **Central current source** - 0.5-2 mA, programmable
- **Voltage measurement** - differential amplifier + ADC
- **Multiplexer** - selects which probe/ring pairs to measure
- **Multi-probe cable harness** - connects to all probe rings

### Sync/Timebase
- **Wired sync line** or PPS clock shared to all probes
- Ensures phase coherence for MIT measurements
- Enables synchronized data collection
- Distributed from central hub

### Power
- **12 V or 5 V battery pack(s)**
- Capacity: 10-20 Ah for field operations
- Distribution via cables to junction boxes
- Much lower power per probe (passive probes)

### Communications
- **Cabled bus** (RJ45/CAT5) for reliable data transfer
- **Short-range wireless** (LoRa/BLE) option depending on site conditions
- Data logging to tablet/computer
- Centralized control and monitoring

## 5.4 System Block Diagram

### Old Design (Deprecated - 25mm with electronics in probe)
```
[NOT USED - Electronics in probe head]
```

### New Design (Micro-Probe - Passive + Central Electronics)

```
Surface:
┌─────────────────────────────────────────┐
│     Central Electronics Hub            │
│                                         │
│  DDS → TX Amp ──┐                      │
│                 │                       │
│  RX Chain ←─────┼──→ ADC → MCU         │
│  (LNA, IA)      │                       │
│                 │                       │
│  ERT Current ───┼──→ MUX → Diff Amp  │
│  Source         │      ↓                │
│                  │    ADC → MCU         │
│                  │                       │
│  Sync/Clock ─────┼───────────────────   │
│  Power ──────────┼───────────────────   │
│  Comms ──────────┼───────────────────   │
└──────────────────┼──────────────────────┘
                   │
        Multi-Probe Cable Harness
                   │
        ┌──────────┴──────────┐
        │                     │
    ┌───▼───┐            ┌───▼───┐
    │ Probe │            │ Probe │
    │   1   │            │   2   │
    └───┬───┘            └───┬───┘
        │                    │
Downhole (Passive):
    ┌───┴───┐            ┌───┴───┐
    │  TX   │            │  TX   │
    │ Coil  │            │ Coil  │
    │       │            │       │
    │  RX   │            │  RX   │
    │ Coil  │            │ Coil  │
    │       │            │       │
    │ Ring  │            │ Ring  │
    │ Ring  │            │ Ring  │
    └───────┘            └───────┘
```

## 5.5 Frequency Plan

### MIT Sweeps
- Typical frequencies: **~2, 5, 10, 20, 50 kHz**
- **Lower frequencies preferred** (2-10 kHz) for deeper penetration with smaller coils
- Choose 3-5 points based on depth/resolution requirements
- Longer integration times compensate for smaller coil area

### ERT Configuration
- **DC with polarity reversal** (e.g., every 1-2 s)
- **Low-freq AC** option (e.g., 8-16 Hz) to reduce polarization
- Current levels: 0.5-2 mA
- Narrow rings (3-5 mm) work well with slurry/water in hole

## 5.6 Performance Trade-offs

### Smaller Coil Area
**Challenge:** Smaller radius → smaller coil area → weaker coupling

**Compensation:**
- More turns on coil (fine wire, 34-38 AWG, many turns)
- Lower frequency (2-10 kHz for deeper penetration)
- Longer integration time (lock-in detection can average more)
- Careful noise control and shielding

**Result:** Acceptable SNR with longer dwell times and careful design

### Centralized Electronics
**Advantages:**
- Easier maintenance and troubleshooting
- Lower cost per probe (passive probes are cheaper)
- Better power management
- Easier firmware updates

**Challenges:**
- More cables to surface
- Central failure point (mitigate with redundancy)
- Cable management in field

## 5.7 Array Layout

### Field Geometry
- **Standard:** 10×10 m section, 2 m spacing
- **Dense:** Can go to 1-1.5 m spacing (easier with micro-probes)
- **Probe count:** 20-24 probes (standard), up to 50+ for dense arrays

### Visual Impact
- Probes look like **tent stakes**, not construction
- Minimal visual disturbance
- Easy to backfill after removal
- Acceptable for sensitive archaeological contexts

### Disturbance
- **Hole size:** 12-18 mm (vs 50 mm for old design)
- **Per hole:** ~0.5 liters displaced (vs ~6 liters)
- **Total (25 probes):** ~12-15 liters (vs ~150 liters)
- **Reduction:** ~10× less disturbance

## 5.8 Insertion Methods

1. **Hand Auger:** 10-20 mm hand auger, create hole, insert probe
2. **Pilot Rod:** 8-10 mm steel rod, drive to depth, wiggle to 12-14 mm, remove, insert probe
3. **Direct Push:** In sandy loam, may push probe directly (requires robust tip)
4. **Water-Jet:** In sand, use water lance to fluidize, insert probe, water drains

## 5.9 Advantages of Micro-Probe Design

1. **Minimal Intrusion:** ~10× less disturbance than 25mm design
2. **Easy Insertion:** Lightweight, easy to handle, minimal force required
3. **Better Contact:** Slurry/water in hole improves ERT contact
4. **Flexible Deployment:** Can go denser spacing, easy to remove and backfill
5. **Simplified Electronics:** Centralized (easier maintenance), passive probes (more reliable)
6. **Archaeology-Friendly:** Acceptable for sensitive contexts, minimal visual impact

---

*This design keeps the science of HIRT while making the mechanics feel like archaeology, not construction.*
