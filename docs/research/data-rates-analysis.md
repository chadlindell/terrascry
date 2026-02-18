# Data Rates Analysis: Multi-Sensor Pathfinder + HIRT

## Overview

The multi-sensor Pathfinder generates substantially more data than the original single-modality design. This document quantifies data rates, storage requirements, and processing implications for the complete instrument suite.

## Per-Sensor Data Rates

### Pathfinder Sensors

| Sensor | Channels | Sample Rate | Bytes/Sample | Data Rate |
|--------|----------|-------------|--------------|-----------|
| Fluxgate (8 sensors) | 8 | 10 Hz | 2 bytes (16-bit ADC) | 160 B/s |
| Fluxgate gradients (4 pairs) | 4 | 10 Hz | 4 bytes (float32) | 160 B/s |
| EMI I channel | 1 | 10 Hz | 4 bytes (float32) | 40 B/s |
| EMI Q channel | 1 | 10 Hz | 4 bytes (float32) | 40 B/s |
| Conductivity (computed) | 1 | 10 Hz | 4 bytes (float32) | 40 B/s |
| BNO055 orientation | 4 (quaternion) | 100 Hz | 8 bytes (2×float16) | 3,200 B/s |
| BNO055 (downsampled to TDM) | 4 | 10 Hz | 16 bytes (4×float32) | 160 B/s |
| ZED-F9P GPS | 3 (lat,lon,alt) | 10 Hz | 24 bytes (3×double) | 240 B/s |
| BMP390 pressure/temp | 2 | 1 Hz | 8 bytes | 8 B/s |
| MLX90614 IR temperature | 2 (obj+amb) | 10 Hz | 8 bytes | 80 B/s |
| DS3231 timestamp | 1 | 10 Hz | 8 bytes (epoch ms) | 80 B/s |
| RPLiDAR C1 | 5000 pts/scan | 10 Hz scans | 8 bytes/point (angle+dist) | 400,000 B/s |
| ESP32-CAM image | 1 frame | 1 Hz | ~50,000 bytes (JPEG) | 50,000 B/s |

### HIRT Sensors (for comparison)

| Sensor | Channels | Sample Rate | Data Rate |
|--------|----------|-------------|-----------|
| MIT quadrature | 2 (I,Q) per pair | Per sequence (~1/s) | ~100 B/s |
| ERT voltage | 1 per combination | Per sequence (~1/s) | ~50 B/s |
| ADXL345 inclinometer | 3 per probe | 1 Hz | ~48 B/s |
| Pod GPS (position recording) | 3 | Event-driven | ~24 B/event |

## Aggregate Data Rates

### Pathfinder Total

| Component | Data Rate | Notes |
|-----------|-----------|-------|
| Measurement data (non-LiDAR, non-camera) | ~1,008 B/s | All sensor channels at TDM rate |
| LiDAR point clouds | ~400,000 B/s | 5000 pts × 10 Hz × 8 B/pt |
| Camera images | ~50,000 B/s | 1 JPEG frame per second |
| **Total raw** | **~451 KB/s** | **~1.6 GB/hour** |
| **Without LiDAR/camera** | **~1 KB/s** | **~3.6 MB/hour** |

### Per-Hour Volume Breakdown

| Data Type | Volume/Hour | Readings/Hour |
|-----------|-------------|---------------|
| Fluxgate readings | 1.15 MB | 288,000 (8 ch × 10 Hz × 3600 s) |
| Gradient values | 0.58 MB | 144,000 |
| EMI conductivity | 0.43 MB | 108,000 |
| Orientation | 0.58 MB | 144,000 |
| GPS positions | 0.86 MB | 36,000 |
| Temperature (IR + baro) | 0.32 MB | 39,600 |
| **Subtotal (core sensors)** | **~3.9 MB** | **~760,000 readings** |
| LiDAR scans | ~1.4 GB | 36,000 scans (180M points) |
| Camera frames | ~180 MB | 3,600 frames |
| **Grand total** | **~1.6 GB** | **~180M+ data points** |

## Storage Requirements

### SD Card (Local Backup)

| Survey Duration | Core Data | With LiDAR | With Everything |
|----------------|-----------|------------|-----------------|
| 1 hour | 4 MB | 1.4 GB | 1.6 GB |
| 4 hours | 16 MB | 5.6 GB | 6.4 GB |
| 8 hours (full day) | 32 MB | 11.2 GB | 12.8 GB |

**Recommendation**: 32 GB SD card for core sensor data (years of surveys). 128 GB or larger if LiDAR/camera data is stored locally.

### Jetson NVMe SSD

| Duration | Data Volume | 1 TB SSD Capacity |
|----------|-------------|-------------------|
| 1 day (8 hr) | ~13 GB | ~75 days |
| 1 week (40 hr) | ~65 GB | ~15 weeks |
| 1 month | ~260 GB | ~4 months |

With 1 TB NVMe, the Jetson can store approximately 4 months of continuous daily surveying before requiring data offload.

## MQTT Bandwidth Requirements

### Over WiFi (ESP32 → Jetson)

| Data Stream | Bandwidth | QoS | Notes |
|-------------|-----------|-----|-------|
| Core measurement MQTT | ~2 KB/s | 1 | JSON overhead adds ~2× raw size |
| Status/heartbeat | 0.1 KB/s | 1 | 1 Hz |
| **Total MQTT** | **~2.1 KB/s** | | Well within WiFi capacity |

LiDAR and camera data do NOT flow through MQTT. They connect directly to the Jetson via USB (LiDAR) and WiFi/SD (camera).

### Over Starlink (Jetson → Cloud)

| Data Stream | Bandwidth | Priority |
|-------------|-----------|----------|
| Corrected measurements | ~2 KB/s | High |
| Anomaly alerts | Event-driven | Critical |
| LiDAR point clouds | ~400 KB/s (if streamed) | Low |
| Camera frames | ~50 KB/s | Low |
| **Total upstream** | **~450 KB/s peak** | |

Starlink Mini provides 50-200 Mbps downstream, 10-20 Mbps upstream. At 450 KB/s (~3.6 Mbps), we use <20% of upstream capacity. LiDAR streaming is optional and can be deferred to batch upload.

## Processing Implications

### Edge (Jetson) Real-Time Requirements

| Operation | Input Rate | CPU Requirement |
|-----------|-----------|----------------|
| Tilt correction | 10 Hz × 4 channels | Trivial (<0.1 ms) |
| Anomaly detection | 10 Hz | <1 ms (rolling statistics) |
| LiDAR → DEM | 10 Hz scans | ~50 ms/scan (point cloud processing) |
| Image georeferencing | 1 Hz | <10 ms (EXIF write) |
| MQTT routing | ~2 KB/s | Trivial |

Total CPU load: <10% of Jetson's capacity. The Jetson is significantly over-provisioned for edge processing, leaving headroom for on-device inversion (HIRT) and ONNX inference.

### Cloud/Workstation Post-Processing

| Operation | Data Volume | Processing Time |
|-----------|-------------|----------------|
| Grid interpolation (magnetics) | 36K points/hr | ~10 s |
| EMI conductivity mapping | 36K points/hr | ~10 s |
| LiDAR → DEM generation | 180M points/hr | ~5-10 min |
| Joint inversion (SimPEG) | All modalities | 30-60 min |
| Full survey visualization | All data | ~2-5 min |

## Big Data Considerations

At 180 million data points per hour, the multi-sensor Pathfinder generates what qualifies as "big data" for a portable geophysical instrument:

1. **Storage format**: Parquet (columnar, compressed) reduces storage by ~5× compared to CSV
2. **Time-series database**: TimescaleDB on the Jetson or workstation for efficient temporal queries
3. **Spatial indexing**: PostGIS or H3 hexagonal indexing for spatial queries
4. **Streaming analytics**: Apache Kafka or MQTT + InfluxDB for real-time dashboarding
5. **Archive**: S3-compatible object storage for long-term survey archive

## References

- RPLiDAR C1 specifications: Slamtec
- ESP32 WiFi throughput: Espressif Systems
- Starlink Mini specifications: SpaceX
- TimescaleDB: https://www.timescale.com/
