# Pathfinder Vision

## Mission

Enable rapid magnetic reconnaissance of survey areas, reducing time-to-decision for deploying detailed investigation systems.

## Design Philosophy

### Harness-First
The operator's arms should never bear the weight. A shoulder harness with elastic suspension carries the sensor array; hands provide guidance only. This enables sustained walking surveys without fatigue.

### Speed Over Resolution
Pathfinder prioritizes coverage rate over measurement precision. A "good enough" anomaly detection in 5 minutes beats a perfect measurement in 2 hours. Detailed characterization is HIRT's job.

### Field-Rugged
Must survive real field conditions: rain, mud, brush, uneven terrain. No carts, no wheels, no flat-ground assumptions.

### DIY-Accessible
Total build cost under $1,000. Components available globally. No specialized manufacturing required. Complete documentation for self-builders.

## Goals

| Goal | Target | Rationale |
|------|--------|-----------|
| Coverage rate | >3,000 m²/hour | Screen typical survey area in <15 min |
| Swath width | 1.5-2.0 m | 3-4 gradiometer pairs at 50 cm spacing |
| Detection depth | 0.5-1.5 m | Adequate for shallow anomaly flagging |
| System weight | <1.5 kg | Comfortable for 2+ hour surveys |
| Build cost | <$900 | Accessible to research groups |
| Setup time | <2 min | Don harness, power on, walk |

## Non-Goals

- **Not a replacement for detailed survey**: Pathfinder finds anomalies; HIRT/GPR/etc. characterize them
- **Not a metal detector**: Gradiometers measure magnetic field gradients, not conductivity
- **Not ferrous-only**: Fluxgate gradiometers detect any magnetic anomaly (ferrous, fired clay, disturbed soil)
- **Not cart-based**: Must work in forests, brush, and uneven terrain

## Target Users

1. **HIRT operators**: Pre-screen before deploying crosshole probes
2. **Archaeological teams**: Rapid site assessment before excavation planning
3. **Forensic investigators**: Search area prioritization
4. **UXO survey teams**: Initial screening before detailed clearance
5. **Citizen scientists**: Low-cost entry to geophysical survey

## Constraints

### Physical
- Single operator, no assistant required
- Walkable terrain (not vehicle-dependent)
- All-weather operation (IP65 electronics minimum)

### Technical
- Fluxgate sensors (not optically-pumped or SQUID)
- Arduino-class microcontroller (accessible, well-documented)
- Standard GPS module (2-5 m accuracy acceptable for screening)
- SD card data logging (no real-time link required)

### Budget
- Electronics: <$700
- Frame/harness: <$100
- Tools/misc: <$100
- **Total: <$900**

## Success Criteria

Pathfinder is successful when:

1. A single operator can screen a 20×20 m area in under 10 minutes
2. Anomalies detectable by commercial gradiometers are also detected by Pathfinder
3. Data can be exported and visualized in standard GIS software
4. Total build time is under 20 hours for a competent maker
5. The system survives a full day of field use without failure
