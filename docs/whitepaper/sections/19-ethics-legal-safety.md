# 19. Ethics, Legal, and Safety

## Human Remains

Treat as a **forensic/archaeological context**:
- Obtain all **permits/permissions** before deployment
- Follow jurisdictional requirements (e.g., heritage boards, war graves authorities)
- Maintain proper chain of custody for any findings
- Document all activities thoroughly

## UXO Risk

**CRITICAL:** WWII sites can contain live ordnance.

### Pre-Survey Requirements

- **Do not drive or hammer probes** until the area is **cleared by qualified EOD/UXO professionals**
- Coordinate with explosive ordnance disposal teams
- Follow established UXO clearance protocols
- Maintain safe standoff distances during operations

### UXO-Specific Safety Protocols

**WARNING:** The standard insertion methodology ("drive pilot rod to depth, wiggle to 12-14 mm, remove") is **DANGEROUS** at UXO sites.

#### Critical Problems

1. **Mechanical trigger risk:** Driving a steel rod into ground containing potential ordnance could trigger detonation
2. **No risk analysis:** Standard procedures lack risk analysis for probe insertion near UXB
3. **Electrical interaction:** ERT current injection (0.5-2 mA) could theoretically interact with sensitive fuzes
4. **Transient hazards:** No overvoltage protection specified (12V/0.5 ohm = 24A transient on short)

#### Required Safety Measures for UXO Sites

| Protocol | Action |
|----------|--------|
| Pre-survey EOD clearance | Professional EOD sweep before ANY insertion |
| Safe standoff perimeter | Minimum 100m exclusion zone during insertion |
| Soft insertion only | Hand auger or water-jet only (NO hammering) |
| Conductivity pre-check | Check soil conductivity before deep insertion |
| Personnel limits | Minimum personnel in hazard zone |
| Communications | Constant radio contact with safe zone |

### Rim-Only Deployment (High-Risk Craters)

For craters with suspected UXB at center, deploy probes **around the perimeter only**:

```
        x                              x

    x       [EXCLUSION        x
                 ZONE]
        x                          x

            x              x
```

**Benefits:**
- Survey geometry still enables 3D reconstruction
- No probe insertion directly over suspected UXB
- Reduced disturbance volume: ~12-15 liters (vs 150 liters traditional)
- Maintains safe standoff from ordnance

**Limitations:**
- Reduced resolution at crater center
- May need wider spacing to maintain safe perimeter

### Conductivity Threshold Monitoring Protocol

**Background:** Research (Waga et al. 2026) indicates that conductivity >6,000 uS/cm at WWII bomb craters signals chemical activation risk, potentially leading to spontaneous explosions.

#### Safety Thresholds

| Conductivity | Status | Action |
|--------------|--------|--------|
| <3,000 uS/cm | **GREEN** (Safe) | Normal operations |
| 3,000-5,500 uS/cm | **YELLOW** (Caution) | Increased monitoring, limit insertion depth |
| >5,500 uS/cm | **RED** (Halt) | **STOP all insertion**, evacuate to safe zone, consult EOD |

#### Time-Lapse Monitoring Schedule

| Phase | Timing | Metrics | Action |
|-------|--------|---------|--------|
| Baseline | Day 0 | Full MIT + ERT survey | Establish reference values |
| Early detection | Day 7-14 | Conductivity only | Alert if >10% change |
| Long-term | Monthly | Full ERT | Track redox boundary evolution |
| Pre-excavation | 24-48h before | Full survey | Final safety check |

#### HIRT Early Warning Capability

HIRT's ERT-Lite system can detect conductivity changes that may precede chemical activation:

- **Volumetric 3D resistivity** (vs. single-point measurements)
- **Gradient detection** identifies active redox boundaries
- **Non-magnetic** operation safe near UXB
- **"Set Once; Measure Many"** enables time-lapse without repeated insertion

**Recommended Protocol:**
1. Baseline ERT survey of crater perimeter
2. Weekly monitoring of conductivity trends
3. Alert threshold at >10% change from baseline
4. Halt excavation at >5,500 uS/cm

## Minimal Intrusion

- Prefer **rim/perimeter probing** over direct insertion into suspected areas
- Use **shallow depths** when possible
- Employ **pilot holes** to minimize ground disturbance
- **Avoid inserting probes into suspected burial voids** without authorization
- Document all probe insertion points and depths

## Field Safety

- Maintain clear communication protocols
- Use appropriate personal protective equipment (PPE)
- Have emergency contact information readily available
- Follow local environmental regulations
- Respect site boundaries and access restrictions

