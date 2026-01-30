# 3. System Architecture - Micro-Probe Design

## 3.1 Design Philosophy

**"Archaeologist brain first, engineer brain second"**

- **Goal:** Smallest possible hole (10-16 mm OD, target: 12 mm)
- **Reality:** Need enough physics (coil area, electrode contact) for good signal
- **Constraint:** Many thin, gentle holes rather than few big ones

## 3.2 Probe Overview (Passive Micro-Probes)

Each probe is **passive** - no electronics downhole. Only sensors and wiring.

### MIT Coil Set
- **1x TX coil** + **1x RX coil** on ferrite cores
- **Ferrite cores:** 6-8 mm diameter x 40-80 mm long
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

## 3.3 Central Electronics Hub

### MIT Driver/Receiver
- **Central DDS sine source** (e.g., AD9833)
- **TX driver amplifier** - drives all probe TX coils via trunk cables
- **RX low-noise amplifier chain** - receives from all probe RX coils
- **ADC/lock-in detection** (digital or analog)
- **MCU** (ESP32 or STM32) for control and data acquisition
- **High-Density Trunk Interface** - Connects to Zone Hubs (not individual probes)

### ERT System
- **Central current source** - 0.5-2 mA, programmable
- **Voltage measurement** - differential amplifier + ADC
- **Multiplexer** - High-density matrix to switch between trunk lines
- **High-Density Trunk Interface** - Connects to Zone Hubs

### Power
- **12 V or 5 V battery pack(s)**
- Capacity: 10-20 Ah for field operations
- Distribution: Power remains at the Hub; only signals traverse the trunks.

### Communications
- **Data Logging:** USB or WiFi to field tablet
- **Control:** Centralized MCU handles all scheduling

## 3.4 System Block Diagram

### Scalable Zone Architecture

```
Surface - Central Electronics Hub:
+---------------------------------------------+
|     Central Electronics Hub                 |
|                                             |
|  [DDS/TX Drive] ----+                       |
|                     |                       |
|  [RX Signal Chain] <+--> [Analog Matrix]    |
|                     |        |              |
|  [ERT Source/Meas] -+        |              |
|                              |              |
|  [MCU Control] --------------+              |
+----------------------+----------------------+
                       |
             High-Density Trunk Cables
             (DB25 / Centronics)
                       |
      +----------------+----------------+
      |                                 |
+-----v-----+                     +-----v-----+
| Zone Hub  | (Passive Breakout)  | Zone Hub  |
|  (Zone A) |                     |  (Zone B) |
+-----+-----+                     +-----+-----+
      |                                 |
      +--[Probe A1]                     +--[Probe B1]
      +--[Probe A2]                     +--[Probe B2]
      +--[Probe A3]                     +--[Probe B3]
      +--[Probe A4]                     +--[Probe B4]

Downhole (Passive):
    +---+---+
    |  TX   |
    | Coil  |
    |       |
    |  RX   |
    | Coil  |
    |       |
    | Ring  |
    | Ring  |
    +-------+
```

## 3.5 Scalability Architecture: The Zone Strategy

To scale the system from a small prototype (4-8 probes) to a field-ready array (20-50+ probes) without creating an unmanageable "rat's nest" of cables, HIRT employs a **Zone Wiring Strategy**.

### The Challenge
A single passive probe requires 8-12 analog conductors (TX coil pair, RX coil pair, 3x ERT rings, Shield, ID). Connecting 25 probes directly to a central hub would require terminating ~250-300 conductors at a single panel, creating severe mechanical congestion and crosstalk risks.

### The Solution: Passive Zone Hubs
The array is divided into logical "Zones," each consisting of 4 probes.
1.  **Probes connect to a local Zone Hub:** A small, passive IP65 box placed on the ground near the probe cluster.
2.  **Zone Hubs connect to the Main Unit:** Via a single high-quality, shielded **Trunk Cable** (e.g., DB25 or 37-pin military connector).
3.  **Central Hub manages Zones:** The hub switches between Trunk lines to address specific probes.

**Benefits:**
*   **Modular Deployment:** Setup involves running a few thick cables rather than dozens of thin ones.
*   **Signal Integrity:** Trunk cables use individual shielded twisted pairs for all analog lines.
*   **Noise Rejection:** Differential signaling (balanced TX drive, instrumentation amp RX) rejects common-mode noise.
*   **Field Repair:** A damaged probe cable only affects one local Zone, not the main harness.

## 3.6 Frequency Plan

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

## 3.6 Performance Trade-offs

### Smaller Coil Area: Quantified SNR Degradation

**Challenge:** Smaller radius -> smaller coil area -> weaker coupling

#### SNR Loss Analysis: 25mm -> 12mm Coils

| Metric | 25mm | 12mm | Loss |
|--------|------|------|------|
| Coil Area | 490.9 mm^2 | 113.1 mm^2 | 77% reduction |
| TX-RX Coupling (M) | Baseline | ~5.3% | **-12.8 dB** |
| Target-RX Coupling | Baseline | ~23% | **-6.4 dB** |
| **Combined SNR Loss** | **Baseline** | **~1.3%** | **-19 dB** |

#### Compensation Strategies

| Strategy | Recovery |
|----------|----------|
| Higher turn count (300-400 vs 100-150) | +4-6 dB |
| Lower frequency (2-10 kHz) | +3-6 dB |
| Centralized low-noise electronics | +3-6 dB |
| Extended integration (5-15 sec vs 1-2 sec) | +10-15 dB |
| **Net Recovery** | **+15-22 dB** |

**Result:** 3-10 dB remaining penalty -> **Survey time increases 5-10x** but remains field-practical.

#### Electronics Performance Comparison

| Parameter | HIRT | Commercial Lock-in |
|-----------|------|-------------------|
| Cost | $45 | $20,000-50,000 |
| Phase accuracy | +/-5 degrees | +/-0.5 degrees |
| Noise floor | ~100 nV | ~10 nV |
| Integration time | 1-2s (up to 15s) | 0.1s |
| Effective SNR | ~85 dB | ~100 dB |

**Assessment:** HIRT electronics adequate for UXO/target detection (SNR limited by coil size, not electronics) but compromised for precise material characterization.

**Compensation (detailed):**
- More turns on coil (fine wire, 34-38 AWG, many turns)
- Lower frequency (2-10 kHz for deeper penetration)
- Longer integration time (lock-in detection can average more)
- Careful noise control and shielding

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

## 3.7 Array Layout

### Field Geometry
- **Standard:** 10x10 m section, 2 m spacing
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
- **Reduction:** ~10x less disturbance

## 3.8 Optimal Array Configuration

### Optimal Probe Spacing

Probe spacing determines lateral resolution and coverage. The Nyquist-like sampling requirement for subsurface tomography is: **spacing <= 2x smallest feature to resolve**.

| Target Depth | Target Size | Recommended Spacing | Array Size | Rationale |
|--------------|-------------|---------------------|------------|-----------|
| 0.5-1.5m (shallow) | Small (<0.3m) | **0.75-1.0m** | 5x5 min | Nyquist for small targets |
| 0.5-1.5m (shallow) | Medium (0.3-1m) | **1.0-1.5m** | 4x4 to 5x5 | Adequate coverage |
| 1.5-3m (mid) | Any | **1.5-2.0m** | 5x5 to 6x6 | Balance resolution/coverage |
| 3-4m (deep) | Large (>0.5m) | **2.0-2.5m** | 4x4 to 5x5 | Wider spacing, deeper sensitivity |
| 4-6m (very deep) | Large | **2.5-3.0m** | 4x4 min | Maximum DOI configuration |

**Rule of thumb:** Spacing is approximately 0.5-0.75x target size for optimal resolution

### Optimal Probe Depth

Probe depth determines the depth of investigation (DOI). For crosshole tomography, probes should extend **at or below** the target depth for optimal sensitivity.

| Investigation Depth | Minimum Probe Depth | Optimal Probe Depth | Confidence |
|--------------------|---------------------|---------------------|------------|
| 1-2m | 1.5m | 1.5-2m (1.2x target) | HIGH |
| 2-3m | 2.5m | 3m (1.2x target) | HIGH |
| 3-4m | 3m | 3.5-4m (1.2x target) | MEDIUM |
| 4-6m | 3m minimum | 4-5m | LOW |

**Rule of thumb:** Probe depth is approximately 1.2-1.5x investigation depth for optimal sensitivity

### Application-Specific Configurations

#### WWII Crash Site (targets at 2-4m depth)

| Parameter | Recommendation |
|-----------|----------------|
| Probe depth | **3m** (3.5m if achievable) |
| Spacing | **1.5-2.0m** |
| Array size | **5x5 to 6x6** probes (25-36 total) |
| MIT frequencies | **2, 5, 10 kHz** (emphasize lower for depth) |
| Integration time | **5-10 seconds** per measurement |
| Expected resolution | **1-1.5m lateral, 0.5m vertical** |
| Survey time | 4-6 hours (including insertion) |

#### Shallow Burial / Woods Context (targets at 0.5-1.5m depth)

| Parameter | Recommendation |
|-----------|----------------|
| Probe depth | **1.5m** |
| Spacing | **1.0-1.5m** |
| Array size | **4x4 to 5x5** probes (16-25 total) |
| MIT frequencies | **10, 20, 50 kHz** (higher for shallow detail) |
| Integration time | **2-5 seconds** |
| Expected resolution | **0.5-1m lateral, 0.3m vertical** |
| Survey time | 2-3 hours |

#### Deep Crater Investigation (targets at 4-6m depth)

| Parameter | Recommendation |
|-----------|----------------|
| Probe depth | **Maximum achievable (4-5m)** |
| Spacing | **2.0-3.0m** |
| Array size | **4x4 minimum** (16+ probes) |
| MIT frequencies | **2, 5 kHz only** |
| Integration time | **10-30 seconds** |
| Expected resolution | **1.5-2m** |
| Confidence | **LOW** at 6m depth |
| Note | ERT will outperform MIT at these depths |

### Configuration Trade-offs

| Tighter Spacing (1m) | Wider Spacing (2.5m) |
|---------------------|---------------------|
| Better lateral resolution | Deeper sensitivity |
| More probes required | Fewer probes needed |
| Longer deployment time | Faster deployment |
| Better for shallow targets | Better for deep targets |
| Higher ray density | Lower ray density |

### Array Geometry Options

**Square Grid (Standard)**
```
x   x   x   x   x
x   x   x   x   x
x   x   x   x   x
x   x   x   x   x
x   x   x   x   x
```
- Best for unknown target locations
- Uniform ray coverage

**Perimeter Array (UXO-safe)**
```
x   x   x   x   x
x               x
x   [EXCLUSION] x
x               x
x   x   x   x   x
```
- Safe standoff from suspected ordnance
- Reduced center resolution but still viable

**Dense Center (Known Anomaly)**
```
x       x       x
    x   x   x
x   x   x   x   x
    x   x   x
x       x       x
```
- Concentrate probes around known anomaly
- Maximum resolution at target location

## 3.9 Insertion Methods

1. **Hand Auger:** 10-20 mm hand auger, create hole, insert probe
2. **Pilot Rod:** 8-10 mm steel rod, drive to depth, wiggle to 12-14 mm, remove, insert probe
3. **Direct Push:** In sandy loam, may push probe directly (requires robust tip)
4. **Water-Jet:** In sand, use water lance to fluidize, insert probe, water drains

## 3.10 Advantages of Micro-Probe Design

1. **Minimal Intrusion:** ~10x less disturbance than 25mm design
2. **Easy Insertion:** Lightweight, easy to handle, minimal force required
3. **Better Contact:** Slurry/water in hole improves ERT contact
4. **Flexible Deployment:** Can go denser spacing, easy to remove and backfill
5. **Simplified Electronics:** Centralized (easier maintenance), passive probes (more reliable)
6. **Archaeology-Friendly:** Acceptable for sensitive contexts, minimal visual impact

## 3.11 Software Pipeline Roadmap

While this document focuses on hardware, the HIRT system is designed to output data compatible with established open-source inversion frameworks. The 'Stage 2' software pipeline is currently in development to leverage these powerful tools:

- **EIDORS:** Primary target for MIT-3D image reconstruction using finite element models.
- **pyGIMLi:** Target framework for coupled ERT/MIT inversion and multi-physics mesh generation.
- **ResIPy:** User-friendly interface for ERT component processing.

---

*This design keeps the science of HIRT while making the mechanics feel like archaeology, not construction.*

