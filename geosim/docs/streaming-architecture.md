# GeoSim Real-Time Streaming Architecture

## Overview

This document describes the real-time data streaming and continuous analysis
pipeline for HIRT and Pathfinder geophysical instruments. The architecture
enables live sensor data to flow from instruments through edge processing to
cloud-based inversion, with sub-second anomaly detection at the edge.

---

## System Architecture

```
Instrument Hardware          Edge Compute              Cloud / Base Station
+------------------+      +-------------------+      +--------------------+
| Pathfinder       |      | NVIDIA Jetson     |      | Workstation / HPC  |
|  8x FG-3+       |      | AGX Orin 64GB     |      | Full inversion     |
|  EMI coil (FDEM) | WiFi |                   | Star |  SimPEG / pyGIMLi  |
|  ZED-F9P RTK GPS | MQTT | - TDM data ingest | link |  Joint inversion   |
|  BNO055 IMU      |----->| - Tilt correction |----->|  3D model update   |
|  MLX90614 IR     |      | - Anomaly detect  |      |  Visualization     |
|  RPLiDAR C1      | USB  | - MQTT broker     |      +--------------------+
|  ESP32-CAM       |----->| - LiDAR→DEM       |
+------------------+      | - Data logging    |      +--------------------+
                           |                   |      | MQTT Cloud Bridge  |
+------------------+      | Docker containers |      | AWS IoT Core or    |
| HIRT             |      | per instrument    |----->| self-hosted         |
|  MIT transmitter |----->|                   |      +--------------------+
|  ERT electrodes  | SPI  | Shared sensor pod |
|  Inclinometers   |      | provides GPS      |
+------------------+      | registration      |
                           +-------------------+
```

### Data Flow

1. **Pathfinder ESP32** produces TDM-synchronized sensor readings at 10 Hz:
   magnetic gradients (4 channels), EMI conductivity (I/Q), RTK GPS, orientation,
   IR temperature. Published via MQTT over WiFi to Jetson.
2. **RPLiDAR C1** streams 360° point clouds at 10 Hz directly to Jetson via USB
   (not through ESP32 — avoids bandwidth bottleneck and EMI).
3. **Edge compute** (Jetson) receives all data, applies tilt correction,
   runs anomaly detection (multi-physics: magnetics + conductivity + thermal),
   publishes corrected data to local MQTT broker.
4. **Local subscribers** (field laptop, tablet) receive corrected data and
   anomaly alerts via MQTT for live visualization.
5. **Cloud bridge** (when Starlink available) forwards data to cloud for
   full-physics inversion and long-term storage.

### TDM Timing and Data Flow

The ESP32 firmware uses a 100 ms Time-Division Multiplexed cycle to prevent
cross-sensor electromagnetic interference:

```
0 ms              50 ms           80 ms    100 ms
├── Phase 1 ──────┤── Phase 2 ────┤─ P3 ───┤
│ FLUXGATE READ   │  EMI TX/RX    │ SETTLE │
│                 │               │        │
│ 8× fluxgate ADC│  AD9833 DDS   │ WiFi TX│
│ BNO055 IMU     │  AD630 I/Q    │ MQTT   │
│ MLX90614 IR    │  Conductivity │ SD card│
│ GPS RX (pass.) │  calculation  │ NTRIP  │
│                 │               │        │
│ EMI TX: OFF    │  WiFi TX: OFF │ EMI:OFF│
│ WiFi TX: OFF   │               │        │
└─────────────────┴───────────────┴────────┘
```

Each MQTT message contains the complete TDM cycle result: gradients, conductivity,
position, orientation, temperature, and timestamp. See
`../Pathfinder/research/multi-sensor-architecture/tdm-firmware-design.md`.

### Updated Sensor Data Rates

| Sensor | Data Rate | Transport |
|---|---|---|
| Fluxgate gradients (4 ch) | 160 B/s | MQTT (ESP32 → Jetson) |
| EMI conductivity (I, Q, σ) | 120 B/s | MQTT |
| RTK GPS position | 240 B/s | MQTT |
| BNO055 orientation | 160 B/s | MQTT |
| MLX90614 IR temperature | 80 B/s | MQTT |
| **MQTT subtotal** | **~1 KB/s** | WiFi |
| RPLiDAR C1 point clouds | ~400 KB/s | USB direct to Jetson |
| ESP32-CAM images | ~50 KB/s | WiFi/SD |
| **Total** | **~451 KB/s** | |
| **Per hour** | **~1.6 GB** | ~180M data points |

See `docs/research/data-rates-analysis.md` for full breakdown.

---

## MQTT Topic Hierarchy

All topics are rooted under `geosim/`.

| Topic Pattern | Description | Rate |
|---|---|---|
| `geosim/pathfinder/data/raw` | Raw gradiometer + EMI + IMU + GPS + IR | 10 Hz |
| `geosim/pathfinder/data/corrected` | After tilt correction | 10 Hz |
| `geosim/pathfinder/data/emi` | EMI conductivity (I, Q, σ_a) | 10 Hz |
| `geosim/pathfinder/data/thermal` | IR temperature (object + ambient) | 10 Hz |
| `geosim/pathfinder/anomaly/detected` | Anomaly alerts (multi-physics) | Event-driven |
| `geosim/hirt/data/mit/raw` | MIT quadrature measurements | Per sequence |
| `geosim/hirt/data/ert/raw` | ERT four-electrode readings | Per sequence |
| `geosim/hirt/model/update` | Latest inversion result | ~1/min |
| `geosim/hirt/probe/orientation` | Inclinometer readings | 1 Hz |
| `geosim/hirt/probe/position` | GPS position of probe insertion points | Event-driven |
| `geosim/status/{instrument}` | System health / heartbeat | 1 Hz |

### Message Format

All messages are JSON-serialized Python dataclasses (see `geosim.streaming.messages`).
Timestamps are ISO 8601 UTC. Numeric values use SI-compatible units:
nanotesla (nT) for gradients, degrees for angles, meters for positions.

### QoS Levels

- **Raw data**: QoS 0 (at most once) -- high rate, loss tolerable
- **Corrected data**: QoS 1 (at least once) -- logged to disk
- **Anomaly alerts**: QoS 2 (exactly once) -- critical notifications
- **Status/health**: QoS 1

---

## Hardware Requirements

### Recommended Edge Platform: NVIDIA Jetson AGX Orin 64GB

| Spec | Value | Justification |
|---|---|---|
| GPU | 2048 CUDA cores, 64 Tensor cores | CuPy acceleration for inversion |
| CPU | 12-core Arm Cortex-A78AE | Concurrent MQTT + processing |
| RAM | 64 GB LPDDR5 | Full inversion mesh in memory |
| Storage | 64 GB eMMC + NVMe SSD (1 TB) | Survey data logging |
| Power | 15-60 W configurable | Battery/solar operation |
| I/O | USB 3.2, SPI, I2C, UART, Ethernet | Direct instrument connection |
| Size | 100 x 87 mm module | Field-deployable enclosure |

### Minimum Viable: Jetson Orin Nano 8GB

Suitable for Pathfinder-only deployment (tilt correction + anomaly detection).
Not sufficient for on-device HIRT inversion.

### Hardware Additions BOM

| Item | Qty | Est. Cost | Purpose |
|---|---|---|---|
| Jetson AGX Orin 64GB Developer Kit | 1 | $1,999 | Edge compute |
| Samsung 980 PRO 1TB NVMe SSD | 1 | $110 | Survey data logging |
| Starlink Mini | 1 | $599 + $50/mo | Cloud connectivity |
| Pelican 1150 case | 1 | $40 | Weatherproof enclosure |
| USB-C hub (industrial) | 1 | $60 | Instrument connections |
| LiPo battery pack 24V 20Ah | 1 | $300 | 8+ hr field operation |
| DC-DC converter (24V to 19V) | 1 | $25 | Jetson power supply |
| BNO055 IMU breakout (spare) | 2 | $30 ea | Pathfinder tilt sensing |
| Panel-mount Ethernet connector | 2 | $15 ea | Weatherproof I/O |
| **Total** | | **~$3,250** | |

---

## Software Stack

### Edge (Jetson)

| Component | Version | Purpose |
|---|---|---|
| JetPack | 6.0+ | Base OS with CUDA |
| Python | 3.10+ | Application runtime |
| Eclipse Mosquitto | 2.x | Local MQTT broker |
| NumPy | 1.24+ | Numeric computation |
| CuPy | 13.x | GPU-accelerated NumPy |
| ONNX Runtime | 1.17+ | Surrogate model inference |
| paho-mqtt | 2.x | MQTT client library |
| Docker | 24.x | Container isolation |

### Cloud / Base Station

| Component | Version | Purpose |
|---|---|---|
| SimPEG | 0.20+ | EM/ERT forward modeling and inversion |
| pyGIMLi | 1.4+ | ERT inversion (alternative) |
| PyVista | 0.42+ | 3D visualization |
| PostgreSQL + TimescaleDB | 16+ | Time-series data storage |
| Grafana | 10+ | Real-time dashboards |

---

## Two-Tier Compute Strategy

### Tier 1: Edge (Always Available)

Runs on the Jetson regardless of connectivity. Sub-second latency.

| Task | Latency Target | Method |
|---|---|---|
| Pathfinder tilt correction | < 1 ms | NumPy vectorized |
| Anomaly detection (running stats) | < 1 ms | Rolling mean/std |
| Anomaly detection (ONNX surrogate) | < 10 ms | ONNX Runtime on GPU |
| Data logging to NVMe | < 1 ms | Append-only Parquet |
| MQTT publish | < 2 ms | Local broker |

### Tier 2: Cloud (When Connected)

Full-physics inversion when Starlink is available.

| Task | Latency Target | Method |
|---|---|---|
| HIRT progressive inversion | 30-120 s | SimPEG on GPU workstation |
| Full survey reprocessing | 5-30 min | Batch on HPC |
| Model archival | async | TimescaleDB + S3 |

### Graceful Degradation

- **Full connectivity**: Edge correction + cloud inversion in real time.
- **Intermittent**: Edge queues data, cloud processes when link available.
- **No connectivity**: Full edge pipeline runs independently. Data syncs later.

---

## Deployment: Docker Containers on Jetson

```yaml
# docker-compose.yml (simplified)
version: "3.8"

services:
  mosquitto:
    image: eclipse-mosquitto:2
    ports:
      - "1883:1883"
    volumes:
      - ./mosquitto.conf:/mosquitto/config/mosquitto.conf
      - mosquitto-data:/mosquitto/data

  pathfinder-pipeline:
    build: ./containers/pathfinder
    depends_on: [mosquitto]
    devices:
      - /dev/ttyUSB0:/dev/ttyUSB0   # GPS
      - /dev/i2c-1:/dev/i2c-1       # IMU
    environment:
      MQTT_BROKER: mosquitto
      MQTT_PORT: 1883
    volumes:
      - survey-data:/data

  hirt-pipeline:
    build: ./containers/hirt
    depends_on: [mosquitto]
    runtime: nvidia                   # GPU access for CuPy
    environment:
      MQTT_BROKER: mosquitto
    volumes:
      - survey-data:/data

  cloud-bridge:
    build: ./containers/bridge
    depends_on: [mosquitto]
    environment:
      MQTT_LOCAL: mosquitto:1883
      MQTT_CLOUD: ${CLOUD_MQTT_ENDPOINT}

volumes:
  mosquitto-data:
  survey-data:
```

Each container is independently restartable. The MQTT broker provides
decoupling between producers and consumers.

---

## Performance Benchmarks

Target benchmarks for edge compute on Jetson AGX Orin:

| Operation | Target | Measured | Status |
|---|---|---|---|
| Tilt correction (single sample) | < 1 ms | -- | Pending |
| Tilt correction (batch, 1000 samples) | < 5 ms | -- | Pending |
| Anomaly detection (single sample) | < 1 ms | -- | Pending |
| MQTT publish latency | < 2 ms | -- | Pending |
| MQTT end-to-end (pub to sub) | < 5 ms | -- | Pending |
| Pathfinder full pipeline (raw to corrected) | < 10 ms | -- | Pending |
| HIRT progressive inversion (100 cells) | < 30 s | -- | Pending |
| ONNX surrogate inference | < 10 ms | -- | Pending |
| Data logging (Parquet append) | < 1 ms | -- | Pending |
| Memory usage (idle) | < 2 GB | -- | Pending |

---

## Progressive Inversion Strategy for HIRT

HIRT collects data sequentially (one electrode combination at a time).
Rather than waiting for a complete survey, we use progressive inversion:

1. **Bootstrap phase** (first 20 measurements): Use a homogeneous
   half-space starting model. Inversion is underdetermined but provides
   a first approximation.

2. **Incremental update** (each new measurement): Add the new datum to the
   existing dataset and re-invert using the previous model as starting model.
   This converges faster than starting from scratch.

3. **Periodic full re-inversion** (every 50 measurements or on demand):
   Full inversion from scratch to prevent error accumulation from
   incremental updates.

4. **Model publication**: After each update, publish the current model to
   `geosim/hirt/model/update` as a serialized mesh with resistivity values.

### Probe Tilt Compensation

HIRT probes in boreholes may not be perfectly vertical. Inclinometer
readings from each probe are published to `geosim/hirt/probe/orientation`
and used to correct the geometric factor in ERT apparent resistivity
calculations:

```
K_corrected = K_geometric * cos(tilt_magnitude)
```

This ensures that the inversion uses accurate electrode positions even
when probes deflect from vertical.

---

## Surrogate Model / PINN Future Roadmap

For Phase 2, we plan to replace or supplement physics-based inversion with
learned models for faster edge inference:

### ONNX Surrogate Models

1. **Training**: Run thousands of SimPEG forward models with varied
   subsurface configurations. Train a neural network to predict the
   forward response (or its Jacobian) from a resistivity model.

2. **Export**: Convert trained PyTorch/TensorFlow model to ONNX format.

3. **Inference**: Run on Jetson using ONNX Runtime with CUDA execution
   provider. Target < 10 ms per forward evaluation (vs. seconds for
   full physics).

4. **Use case**: Real-time "what-if" visualization in the field.
   Operator adjusts model, sees predicted data response instantly.

### Physics-Informed Neural Networks (PINNs)

Embed Maxwell's equations as loss terms during training so the network
respects physics even in regions with sparse data:

- **Advantage**: Generalizes better than pure data-driven surrogates.
- **Challenge**: Training is slower; inference speed is similar to standard NN.
- **Timeline**: Phase 3, after surrogate model validation.

### Hybrid Approach

Use surrogate for rapid approximate inversion at the edge, then refine
with full SimPEG inversion in the cloud. The surrogate provides an
excellent starting model, reducing cloud inversion iterations by 3-5x.

---

---

## Fiber Optic Alternative

WiFi (ESP32 built-in) is the primary data link between Pathfinder and the Jetson.
However, for applications requiring absolute minimum RF interference with fluxgate
measurements, a fiber optic alternative has been evaluated.

### Decision Framework

| WiFi Interference (after TDM) | Recommendation |
|---|---|
| < 0.1 nT | WiFi sufficient (current design) |
| 0.1 - 1.0 nT | Fiber recommended for high-precision |
| > 1.0 nT | Fiber required |

**Current estimate**: < 0.01 nT with TDM + shielding **(Modeled)**. WiFi is sufficient.

### Fiber Options (If Needed)

| Option | Cost | Data Rate | Cable |
|---|---|---|---|
| USB-to-fiber media converter | $80-150/pair | 480 Mbps | Glass multimode |
| SFP on Jetson Ethernet | $50-70 | 1 Gbps | Glass single-mode |
| Plastic Optical Fiber (POF) | $40-60 | 100 Mbps | PMMA 1mm (flexible) |

POF is recommended for field use: cheap, flexible, easy to terminate, adequate bandwidth.

### Hybrid Approach

Fiber for data, WiFi disabled entirely during survey. ESP32 sends all data over
fiber-to-USB converter. NTRIP corrections flow from Jetson via fiber to ESP32.
Zero RF emission from instrument.

**Operational downside**: Trailing 10-100m fiber cable requires reel, limits terrain flexibility.

See `docs/research/fiber-vs-wifi-analysis.md` for full analysis.

---

## EMI Conductivity Channel Data Flow

The EMI (FDEM) conductivity channel adds below-ground electrical conductivity
to the streaming pipeline. Data flow:

```
AD9833 DDS (15 kHz) → OPA549 → TX Coil
                                    ↓ (ground eddy currents)
                              RX Coil → AD8421 → AD630 → ADS1115
                                                   ↓
                                        I (in-phase) → susceptibility
                                        Q (quadrature) → conductivity
                                                   ↓
                        σ_a = (4/(ωμ₀s²)) × Q/primary
                                                   ↓
                        MQTT: geosim/pathfinder/data/emi
                                                   ↓
                        Jetson: conductivity map overlay
                                                   ↓
                        SimPEG: FDEM forward model validation
```

The conductivity data enables:
- Real-time conductivity mapping alongside magnetics
- Detection of soil disturbance invisible to magnetics (e.g., clay-filled features)
- Boundary condition for HIRT ERT inversion (surface conductivity constrains top layers)

---

## References

- Eclipse Mosquitto: https://mosquitto.org/
- NVIDIA Jetson AGX Orin: https://developer.nvidia.com/embedded/jetson-agx-orin
- SimPEG: https://simpeg.xyz/
- pyGIMLi: https://www.pygimli.org/
- ONNX Runtime: https://onnxruntime.ai/
- Paho MQTT: https://eclipse.dev/paho/
- CuPy: https://cupy.dev/
- Pathfinder TDM design: `../Pathfinder/research/multi-sensor-architecture/tdm-firmware-design.md`
- Data rates analysis: `docs/research/data-rates-analysis.md`
- Fiber vs WiFi: `docs/research/fiber-vs-wifi-analysis.md`
