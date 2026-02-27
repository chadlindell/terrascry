# I9: Pathfinder Main PCB Redesign - Consensus Validation

## Metadata

| Field | Value |
|-------|-------|
| Task ID | I9 |
| Task Type | CRITICAL ENGINEERING |
| Date | 2026-02-18 |
| Models Requested | openai/gpt-5.2-pro, gemini-3-pro-preview |
| Models Consulted | openai/gpt-5.2-pro (neutral, 8/10 confidence) |
| Models Failed | gemini-3-pro-preview (429 RESOURCE_EXHAUSTED - API quota) |
| Additional Analysis | claude-opus-4-6 (independent), PAL expert layer |
| Consensus Tool | mcp__pal__consensus |
| Continuation ID | 8175d261-b22b-4307-8a21-4ba217a4c93d |
| Reference Files | power-supply-architecture.md, i2c-address-map.md |

---

## Consensus Verdict: CONDITIONALLY APPROVED

The PCB redesign plan is technically sound and feasible for a 100x80mm 4-layer board. However, **three critical changes must be made before layout begins** to avoid a first-article failure. The overall architecture (3-stage power supply, dual I2C bus, analog/digital partitioning) is well-engineered and appropriate for this mixed-signal geophysical instrument.

**Overall Confidence: 8/10**

---

## Critical Changes Required (BLOCKERS)

### BLOCKER 1: Fix LM78L05 Voltage Headroom

**Severity: SHOWSTOPPER** -- Highest-probability first-article failure mode.

The current design feeds 5.5V from the LM2596 buck converter into 8x LM78L05 regulators that need to output 5.0V. The LM78L05 has a 1.7V dropout voltage, requiring a minimum input of 6.7V. At 5.5V input, the regulators **cannot regulate** and will pass through whatever voltage arrives, minus their saturation drop. This reintroduces the exact conducted crosstalk between FG-3+ sensors that the individual regulators were meant to prevent.

**Resolution Options (pick one):**

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| A: Increase buck to 7.0-7.5V | LM78L05 works as designed; no BOM change | Slightly lower efficiency; TPS7A49 input increases (still within spec) | **Preferred if sticking with LM78L05** |
| B: Replace LM78L05 with MCP1700-5002E/TT | 178mV dropout; SOT-23 footprint; works at 5.5V input | Different BOM; lower output current (250mA, but FG-3+ only needs 12mA) | **Preferred overall -- simpler, lower power** |
| C: Replace LM78L05 with AP2112K-5.0 | 250mV dropout; SOT-23-5; good PSRR | Slightly more expensive | Good alternative to MCP1700 |

**Recommended**: Option B (MCP1700-5002E/TT). The 178mV dropout at 12mA load means the regulator works perfectly at 5.5V input (5.5 - 0.178 = 5.32V headroom, well above 5.0V output). SOT-23 footprint saves board space compared to TO-92 LM78L05. The existing 3-stage supply chain remains unchanged.

**Source**: power-supply-architecture.md lines 107-136, which already flag this issue and recommend this resolution path.

### BLOCKER 2: Change Ground Plane Strategy

**Severity: HIGH** -- Incorrect implementation will create worse noise than no separation at all.

The original plan calls for a **split ground plane** with separate analog and digital ground pours connected at a single star point. While conceptually correct for wire-level design, this approach is **counterproductive on a dense mixed-signal PCB** because:

1. SPI traces to the ADS1115 and AD9833 **must** cross from the digital domain to the analog domain. If they cross a ground plane split, their return currents are forced to detour around the split, creating large current loops that radiate EMI and increase susceptibility.
2. I2C traces (Bus 0) connect ESP32 (digital) to ADS1115 (analog) and similarly cross domains.
3. The split forces return currents to flow through the single star-point connection, which becomes a noise bottleneck.

**Corrected Approach:**
- **L2 (Ground plane): Solid, unbroken, continuous** -- no splits, no cuts, no moats
- Partition noise by **component placement and routing discipline**, not by physical ground plane division
- If schematic requires separate AGND/DGND nets, implement as a **net-tie** near the ADC analog power entry, but keep L2 physically continuous
- Split/island the **power plane (L3)** instead, using ferrite beads to create quiet analog power islands

This aligns with industry best practice for mixed-signal PCB design (TI Application Note SLYT107, Analog Devices MT-031).

### BLOCKER 3: Move ESP32 to Board Edge

**Severity: MEDIUM-HIGH** -- Antenna performance and routing density.

The original plan places the ESP32-WROOM-32 at the center of the board. This creates two problems:

1. The antenna keep-out zone (minimum 10x10mm, practically larger once margins are added) consumes valuable interior routing area on a dense 100x80mm board.
2. WiFi antenna performance degrades when surrounded by copper on all sides. The ESP32-WROOM-32 antenna is designed to extend beyond the PCB edge.

**Corrected Placement:**
- Place ESP32 at one **board edge** with the antenna section extending beyond or flush with the edge
- Maintain copper-free zone on **all layers** (including L2 and L3) under the antenna per Espressif hardware design guidelines
- Route digital buses (SPI, I2C, UART) from ESP32 inward toward the board center

---

## Approved Design Elements

The following aspects of the original plan are validated and should proceed as designed:

### 4-Layer Stackup: JUSTIFIED

4-layer is **unequivocally the correct choice** for this design. The cost premium over 2-layer at JLCPCB (~$5-10 extra for 5 boards) is trivial compared to:
- Continuous ground plane under all analog signal traces (essential for fluxgate noise floor)
- Controlled impedance for SPI bus
- Proper power distribution
- Reduced debug time on first article

### 3-Stage Power Supply: APPROVED

The LM2596 buck (150kHz) -> Ferrite+LC filter (5kHz cutoff, ~60dB attenuation) -> TPS7A49 LDO (~60dB PSRR) chain delivers ~120dB combined rejection at the switching frequency. The 30-50mV ripple from the buck is reduced to ~30-50pV at the analog rail. This exceeds the requirements for ADS1115 and LM2917 precision by a wide margin.

**Note**: The shielded inductor requirement (Coilcraft MSS1260 or equivalent) is critical. Distance from the fluxgate analog region matters even more than shielding -- maximize physical separation between the buck converter and the LM2917/ADS1115 section.

### Individual Per-Sensor Regulators: APPROVED

The rationale for individual regulators per FG-3+ sensor is sound: the self-oscillating nature of the FG-3+ means supply voltage modulation from one sensor's pulsed current draw can shift another sensor's oscillation frequency, creating conducted crosstalk. Individual regulators provide supply isolation.

**Change required**: Replace LM78L05 with MCP1700 (see Blocker 1).

### Dual I2C Bus Architecture: APPROVED

- Bus 0 (GPIO 21/22) at 400kHz for local PCB sensors: no issues with short traces
- Bus 1 (GPIO 16/17) at 100kHz for sensor pod via PCA9615: conservative and correct for 1-2m cable
- No address conflicts on either bus (verified per i2c-address-map.md)
- Future expansion capacity available (0x4A, 0x4B on Bus 0 for EMI I/Q ADC)

### Component Placement Zones: APPROVED (with ESP32 modification)

- LM2917 array along one edge, analog ground pour beneath: **Good**
- ADS1115 adjacent to LM2917 outputs: **Good** -- add anti-alias RC filter
- AD9833 on digital side, SPI routed away from analog: **Good**
- Power supply in corner, input near edge connector: **Good**
- M8 connectors on board edge: **Good**
- JST-XH on opposite edge, grouped by sensor pair: **Good**

---

## Recommended Layer Stackup

```
L1 (Top):     Components + short critical analog routes (LM2917 -> ADS1115)
              Keep LM2596 SW node copper area minimal
              Place bypass caps within 1-2mm of IC supply pins

L2:           SOLID UNBROKEN GROUND PLANE
              No splits. No cuts. No moats.
              Via stitching around board perimeter and between domains
              Copper-free zone under ESP32 antenna ONLY

L3:           POWER PLANE (partitioned)
              Separate power pours: 5.5V (or 7V), 5.0V_A (TPS7A49), 3.3V_D
              Ferrite beads between power zones for quiet analog islands
              Local pours to each LM2917/ADC cluster

L4 (Bottom):  Digital routing (SPI, I2C), control traces
              Connector breakouts and test points
              Secondary ground pour where possible
```

### Board Layout Zones (100x80mm)

```
+------------------------------------------------------------------------+
|  POWER SUPPLY CORNER    |           DIGITAL ZONE                       |
|  LM2596 + LC filter     |  SD card (SPI)                              |
|  TPS7A49 LDO            |  USB-C                     [ESP32-WROOM-32] |
|  Ferrite bead            |  AD9833 (SPI)              Antenna ->  |||  |
|--------------------------|  M8 (sensor pod, I2C Bus 1)                 |
|                          |                                             |
|  ANALOG ZONE             |          TRANSITION ZONE                    |
|                          |                                             |
|  [LM2917] [LM2917] [LM2917] [LM2917] [LM2917] [LM2917] [LM2917] [LM2917]
|  [ADS1115 #1]  [ADS1115 #2]         |                                 |
|  [MCP1700 x8 (one per sensor)]      |                                 |
|                                      |                                 |
|  JST-XH  JST-XH  JST-XH  JST-XH   JST-XH  JST-XH  JST-XH  JST-XH  |
|  (Sensor 1-2)    (Sensor 3-4)       (Sensor 5-6)    (Sensor 7-8)     |
+------------------------------------------------------------------------+
          M8 (EMI TX)                          M8 (EMI RX)
```

---

## Detailed Answers to Original Concerns

### Concern 1: 4-Layer vs 2-Layer

**Answer: 4-layer is justified and necessary.**

JLCPCB 4-layer pricing for 5 boards at 100x80mm is approximately $15-25 USD (vs ~$5-10 for 2-layer). The ~$15 premium is negligible compared to:
- Risk of noise-induced re-spin ($15-25 + 1-2 weeks)
- Debug time savings from proper ground plane
- Controlled impedance SPI routing

### Concern 2: Ground Plane Split Location

**Answer: Do not split the ground plane.** (See Blocker 2 above.)

Keep L2 as a solid, continuous ground plane. Partition noise by placement discipline:
- All analog components (LM2917, ADS1115, MCP1700 regulators) on one side of the board
- All digital components (ESP32, SD card, AD9833, USB-C) on the other side
- Minimize the number of traces that cross between zones
- Where traces must cross (I2C to ADS1115, SPI to AD9833), ensure they always run over the continuous L2 ground

### Concern 3: ESP32 Antenna Keep-Out

**Answer: Move ESP32 to board edge.** (See Blocker 3 above.)

With the ESP32 at a board edge, the keep-out zone extends beyond the PCB rather than consuming interior space. This solves the space conflict and improves antenna performance.

### Concern 4: LM2917 Thermal

**Answer: No special thermal pads needed.**

8 units at 60mW each = 480mW total. At ~80mm edge length, this is ~6mW/mm of linear board edge. Even in still air, the temperature rise is negligible (~2-3C above ambient with basic ground plane copper). Ensure:
- Each LM2917 has some copper connected to L2 via stitching vias
- Do not isolate them in narrow copper islands
- If using SOIC packages, exposed pad to L2 via thermal vias (if applicable)

### Concern 5: Mixed-Signal Trace Routing

**Answer: Keep all traces over continuous L2 ground.**

Traces that cross between analog and digital zones:
- **I2C Bus 0** (SDA/SCL from ESP32 to ADS1115): Route over solid L2 ground. Keep traces paired and close together.
- **SPI** (to AD9833): Route on L4 (bottom), over L2 ground reference. Keep far from LM2917/ADS1115.
- **Power connections**: Route on L3 through ferrite bead transitions between power zones.

Because L2 is continuous (no split), there is no "crossing" problem. Return currents follow the signal traces naturally through the solid ground plane.

### Concern 6: Decoupling Placement

**Answer: Follow these rules.**

| Component | Bypass Cap | Distance | Via to L2 |
|-----------|-----------|----------|-----------|
| Each MCP1700 output | 100nF X7R + 10uF electrolytic | At output pin | Own via at cap ground pad |
| Each ADS1115 VDD | 100nF X7R | <2mm from VDD pin | Own via at cap ground pad |
| TPS7A49 output | 10uF + 1uF X7R ceramic | Per datasheet (tight) | Short, direct |
| ESP32 VDD | 100nF + 10uF X7R | <2mm from VDD pin | Own via |
| AD9833 VDD/AGND | 100nF + 10uF X7R | <2mm from pins | Own via |
| Each LM2917 VCC | 100nF X7R | <2mm from VCC pin | Own via to L2 |
| LM2596 input | 100uF electrolytic + 100nF ceramic | At input pin | Short path |
| LM2596 output | 220uF electrolytic + 100nF ceramic | At output pin | Short path |

**Rule of thumb**: Each bypass cap gets its own via to L2, placed at the cap's ground pad. Do not share vias between multiple bypass caps. The via should be as close to the cap as physically possible.

### Concern 7: JLCPCB Manufacturing

**Answer: 4-layer is standard at JLCPCB; verify component availability.**

- **4-layer capability**: Standard service, 5-7 day turnaround, well-proven
- **Component availability risks** (check LCSC stock before finalizing BOM):
  - LM2917: May need to source through-hole DIP-8 version (LM2917N-8) or find SMD alternative. LCSC stock varies.
  - TPS7A49: Check specific package (SOT-223 or SOIC). May need TPS7A4901 variant.
  - MCP1700-5002E/TT: Well-stocked on LCSC in SOT-23
  - ADS1115: MSOP-10, generally available
  - AD9833: MSOP-10, check stock
  - ESP32-WROOM-32: Module, well-stocked
  - JST-XH: Widely available
  - M8 connectors: Likely not on LCSC; order separately and hand-solder

**Recommendation**: Run a LCSC availability check on the full BOM before finalizing the schematic. Identify backup parts for anything with <100 units in stock.

### Concern 8: Test Points

**Answer: The following test points are required for first-article debug.**

#### Power Rail Test Points (minimum set)
| Test Point | Signal | Purpose |
|-----------|--------|---------|
| TP1 | Battery input (7.4V) | Verify input voltage |
| TP2 | LM2596 output (5.5V or 7V) | Verify buck regulation |
| TP3 | Post-LC filter output | Verify ripple attenuation |
| TP4 | TPS7A49 output (5.0V_A) | Verify precision analog rail |
| TP5 | 3.3V digital rail | Verify ESP32 supply |
| TP6-TP7 | 2-3 representative MCP1700 outputs | Spot-check sensor regulators |

#### Signal Test Points (minimum set)
| Test Point | Signal | Purpose |
|-----------|--------|---------|
| TP8-TP9 | I2C Bus 0 SDA/SCL | Debug ADS1115 communication |
| TP10-TP13 | SPI (SCK/MOSI/MISO/CS) | Debug AD9833 + SD card |
| TP14-TP15 | 2x LM2917 outputs (ch1, ch5) | Verify F-to-V conversion |
| TP16 | AD9833 VOUT | Verify DDS waveform |
| TP17 | ADS1115 #1 ALERT/RDY | Verify conversion timing |

#### Ground Test Points
| Test Point | Signal | Purpose |
|-----------|--------|---------|
| TP_GND_A | Analog ground (near ADS1115) | Oscilloscope ground reference |
| TP_GND_D | Digital ground (near ESP32) | Oscilloscope ground reference |
| TP_GND_P | Power ground (near LM2596) | Noise measurement reference |

**Implementation**: Use 1mm plated through-hole pads (standard test point). For signals that need frequent probing, use loop-style test points that accept oscilloscope probe hooks.

---

## Alternative Approaches Considered

### Digital Frequency Measurement Instead of LM2917

GPT-5.2-pro raised the possibility of replacing the 8x LM2917 with digital frequency measurement (timer capture / counter IC / mux + counter). This would reduce BOM and board area significantly.

**Trade-offs:**
- Pros: Fewer analog components, simpler PCB, less noise sensitivity
- Cons: Higher firmware complexity, jitter sensitivity at high frequencies, potential EMI susceptibility on long sensor cables feeding digital inputs

**Decision**: Retain LM2917 for this revision. The analog F-to-V approach is proven, well-understood, and documented in `lm2917-analysis.md`. Digital frequency measurement is a valid R&D track for a future board revision once the analog baseline is established and characterized.

### RS-485/UART Instead of Differential I2C for Sensor Pod

GPT-5.2-pro noted that RS-485/UART might be more robust than PCA9615 differential I2C for the 1-2m pod cable, especially for RTCM correction streaming to the ZED-F9P.

**Decision**: Retain PCA9615 for this revision. The sensor pod design is already documented, validated for HIRT compatibility, and the ZED-F9P I2C interface is sufficient for position/velocity data at the current update rate. RTCM corrections can be injected via the ZED-F9P's separate UART if needed.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LM78L05 fails to regulate (if not replaced) | **Certain** | **Critical** | Replace with MCP1700 (Blocker 1) |
| Ground plane split creates noise loops | **High** | **High** | Use solid L2 ground (Blocker 2) |
| ESP32 antenna degraded by center placement | **Medium** | **Medium** | Move to board edge (Blocker 3) |
| LCSC stock-out on LM2917 or TPS7A49 | **Medium** | **Medium** | Pre-check stock; identify alternates |
| Board area insufficient for all components | **Low-Medium** | **Medium** | ESP32 at edge frees interior; increase to 120x80mm if needed |
| SPI bus contention (AD9833 vs SD card) | **Low** | **Low** | Proper CS management, separate CS lines, firmware mutex |
| LM2596 EMI coupling to analog traces | **Low** | **Medium** | Shielded inductor + max physical separation + L2 ground continuity |
| Thermal gradient affecting LM2917 accuracy | **Low** | **Low** | 480mW total is negligible; adequate copper to L2 |

---

## Action Items

| Priority | Action | Status |
|----------|--------|--------|
| P0 | Replace 8x LM78L05 with MCP1700-5002E/TT in schematic | **PENDING** |
| P0 | Update power-supply-architecture.md with MCP1700 specifications | **PENDING** |
| P0 | Change ground strategy: solid L2, partitioned L3 | **PENDING** |
| P0 | Move ESP32 placement to board edge in layout plan | **PENDING** |
| P1 | Run LCSC availability check on full BOM | **PENDING** |
| P1 | Add anti-alias RC filters between LM2917 outputs and ADS1115 | **PENDING** |
| P1 | Define test point locations in schematic | **PENDING** |
| P1 | Add ESD protection (TVS/TPD4E05U06) at JST-XH and M8 connectors | **PENDING** |
| P2 | Calculate controlled impedance for SPI traces on L1/L4 | **PENDING** |
| P2 | Verify ESP32 antenna keep-out dimensions for chosen edge placement | **PENDING** |
| P2 | Determine MCP1700 output capacitor stability requirements | **PENDING** |

---

## References

- power-supply-architecture.md (lines 107-136: LM78L05 dropout issue)
- i2c-address-map.md (complete bus architecture, no conflicts)
- TI Application Note SLYT107: Power Supply Design for Mixed-Signal Systems
- Analog Devices MT-031: Grounding Data Converters and Solving the Mystery of "AGND" and "DGND"
- Espressif ESP32-WROOM-32 Hardware Design Guidelines (antenna keep-out)
- Microchip MCP1700 Datasheet (DS20001826)
- JLCPCB 4-Layer PCB Capabilities: https://jlcpcb.com/capabilities/pcb-capabilities

---

*Consensus validation performed 2026-02-18 using PAL consensus tool.*
*Models: openai/gpt-5.2-pro (success), gemini-3-pro-preview (unavailable -- quota exhausted).*
*Note: Gemini perspective was not obtained. Consider re-running with an alternative second model (e.g., gemini-2.5-pro or openai/gpt-5.1-codex) for full dual-model consensus.*
