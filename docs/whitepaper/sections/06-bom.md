# 6. Bill of Materials (BOM) - Micro-Probe Design

**Note:** This section covers the micro-probe design (12mm OD, passive probes). See [Probe BOM](../../hardware/bom/probe-bom.md) and [Base Hub BOM](../../hardware/bom/base-hub-bom.md) for detailed component lists. — per probe (indicative costs)

## Mechanical Components

| Item | Description | Cost Range |
|------|-------------|------------|
| Rod sections | Fiberglass/PVC rod sections (1 m each), threaded couplers | $20–40 |
| Nose capsule | 3D‑printed/PVC nose capsule, seals, potting silicone | $10–20 |
| Cable management | Cable glands, strain relief, heat‑shrink | $5–10 |

## MIT‑3D Electronics

| Item | Description | Cost Range |
|------|-------------|------------|
| Ferrite cores | Ferrite rod cores (Ø8–10 mm × 100–120 mm) | $5–10 |
| Magnet wire | Magnet wire (32–36 AWG) | $5–10 |
| DDS generator | DDS sine generator (e.g., AD9833 or similar) | $6–15 |
| TX driver | TX driver (op‑amp or small class‑D, 0–5 Vrms into coil) | $8–20 |
| RX preamp | RX preamp + instrumentation amp (low‑noise op‑amps) | $10–20 |
| Lock‑in (Option A) | Digital lock‑in via 24‑bit ADC (e.g., ADS1256) + MCU DSP | $18–35 |
| Lock‑in (Option B) | Analog lock‑in (e.g., AD630) + modest ADC | $25–45 |
| MCU | MCU (ESP32 or STM32), breakout PCB, connectors | $10–25 |

## ERT‑Lite Components

| Item | Description | Cost Range |
|------|-------------|------------|
| Electrodes | 2–3 stainless/copper ring electrodes + wiring | $8–12 |
| Multiplexer | Solid‑state switches/multiplexer (probe channel select) | $5–12 |

## Total per Probe

**Estimated cost:** ~**$70–150** DIY

*Note: Costs are indicative and vary by supplier, quantity, and component choices.*

## Shared/Base Components

| Item | Description | Cost Range |
|------|-------------|------------|
| ERT hub | Current source + differential voltmeter (ERT hub) | $40–80 |
| Sync/clock | Sync/clock distribution | $15–40 |
| Cables | Cables (CAT5, power leads) or LoRa modules | $20–60 |
| Battery | Field battery (12 V/10–20 Ah or USB packs) | $30–120 |

## Complete System Cost Estimate

See [Cost & Build Plan](14-cost-build-plan.md) for complete system cost breakdown.

