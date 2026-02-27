# M8 8-Pin Connector Specification

Standard connector for the TERRASCRY shared sensor pod. Used on both Pathfinder and HIRT instruments.

## Why M8 8-Pin (Not 4-Pin)

The PCA9615 differential I2C buffer requires 6 conductors minimum:
- 2x differential SDA (D+, D-)
- 2x differential SCL (D+, D-)
- VCC
- GND

Plus we need VBUS (5V input) and cable shield, totaling 8 conductors. An M8 4-pin connector is insufficient.

This was identified as a **CRITICAL** correction during consensus validation (see `pathfinder/research/multi-sensor-architecture/sensor-pod-pcb-requirements.md`, CRITICAL-1).

## Pinout

| Pin | Signal | Wire Color (Cat5 T568B) | Description |
|-----|--------|------------------------|-------------|
| 1 | VCC | Orange/White | 3.3V regulated output (from pod LDO) |
| 2 | GND | Orange | Ground return |
| 3 | SDA_D+ | Green/White | PCA9615 differential SDA positive |
| 4 | SDA_D- | Green | PCA9615 differential SDA negative |
| 5 | SCL_D+ | Blue/White | PCA9615 differential SCL positive |
| 6 | SCL_D- | Blue | PCA9615 differential SCL negative |
| 7 | VBUS | Brown/White | 5V power input to pod |
| 8 | SHIELD | Brown | Cable shield / drain wire |

## Cable

- **Type:** Cat5 STP (shielded twisted pair)
- **Length:** 1-2m (PCA9615 supports up to 5m)
- **Pairs used:**
  - Pair 1 (Orange): VCC + GND (power)
  - Pair 2 (Green): SDA differential
  - Pair 3 (Blue): SCL differential
  - Pair 4 (Brown): VBUS + shield

## Mating Specifications

- **Standard:** IEC 61076-2-104
- **IP Rating:** IP67 when mated
- **Mating cycles:** >500
- **Current rating:** 4A per pin (adequate for pod's ~200 mA peak)

## Strain Relief

Add cable gland or P-clip at M8 connector mounting point on both pod and instrument enclosures. The M8 connector alone does not provide adequate strain relief for field use.

## Suppliers

- Binder (Series 768)
- TE Connectivity (M8 8-pin)
- Phoenix Contact (SACC-M8)
- Amphenol (M8 series)
