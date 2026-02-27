# TERRASCRY CSV Data Schemas

Shared CSV column definitions for instrument data logging and GeoSim ingest.

## Pathfinder Survey CSV

Logged to SD card at 10 Hz. Each row is one TDM cycle (100 ms).

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp_utc` | string | ISO 8601 | UTC timestamp |
| `seq` | int | — | Sequence number (monotonic, wraps at 2^32) |
| `lat` | float | degrees | WGS84 latitude |
| `lon` | float | degrees | WGS84 longitude |
| `alt_m` | float | m | Altitude above ellipsoid |
| `fix_type` | int | enum | 0=none, 1=autonomous, 2=DGPS, 4=RTK_float, 5=RTK_fixed |
| `hdop` | float | — | Horizontal dilution of precision |
| `satellites` | int | — | Number of satellites tracked |
| `grad_1_nt` | float | nT | Gradient, sensor pair 1 (bottom - top) |
| `grad_2_nt` | float | nT | Gradient, sensor pair 2 |
| `grad_3_nt` | float | nT | Gradient, sensor pair 3 |
| `grad_4_nt` | float | nT | Gradient, sensor pair 4 |
| `top_1_raw` | int | ADC counts | Top sensor 1, raw ADC value |
| `top_2_raw` | int | ADC counts | Top sensor 2 |
| `top_3_raw` | int | ADC counts | Top sensor 3 |
| `top_4_raw` | int | ADC counts | Top sensor 4 |
| `bot_1_raw` | int | ADC counts | Bottom sensor 1 |
| `bot_2_raw` | int | ADC counts | Bottom sensor 2 |
| `bot_3_raw` | int | ADC counts | Bottom sensor 3 |
| `bot_4_raw` | int | ADC counts | Bottom sensor 4 |
| `emi_i` | float | V | EMI in-phase component |
| `emi_q` | float | V | EMI quadrature component |
| `emi_sigma_a` | float | S/m | Apparent conductivity |
| `pitch_deg` | float | degrees | BNO055 pitch |
| `roll_deg` | float | degrees | BNO055 roll |
| `heading_deg` | float | degrees | BNO055 heading |
| `ir_object_c` | float | Celsius | MLX90614 object temperature |
| `ir_ambient_c` | float | Celsius | MLX90614 ambient temperature |

### Example

```csv
timestamp_utc,seq,lat,lon,alt_m,fix_type,hdop,satellites,grad_1_nt,grad_2_nt,grad_3_nt,grad_4_nt,top_1_raw,top_2_raw,top_3_raw,top_4_raw,bot_1_raw,bot_2_raw,bot_3_raw,bot_4_raw,emi_i,emi_q,emi_sigma_a,pitch_deg,roll_deg,heading_deg,ir_object_c,ir_ambient_c
2026-02-18T14:30:12.345Z,12345,51.2345678,-1.4567890,102.34,5,0.8,14,12.3,-5.1,8.7,-2.4,16384,16512,16256,16400,16520,16480,16380,16440,0.0023,0.0015,0.045,1.2,-0.8,45.3,12.3,15.1
```

## HIRT Probe Position CSV

One row per probe insertion point. Recorded when operator places sensor pod at each probe location.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `probe_id` | string | — | Probe identifier (e.g., "P1", "P2") |
| `timestamp_utc` | string | ISO 8601 | Time of GPS fix |
| `lat` | float | degrees | WGS84 latitude |
| `lon` | float | degrees | WGS84 longitude |
| `alt_m` | float | m | Altitude above ellipsoid |
| `fix_type` | enum | string | NO_FIX, AUTONOMOUS, DGPS, RTK_FLOAT, RTK_FIXED |
| `hdop` | float | — | Horizontal dilution of precision |
| `satellites` | int | — | Satellites tracked |
| `pitch_deg` | float | degrees | Surface pitch at insertion point |
| `roll_deg` | float | degrees | Surface roll at insertion point |
| `confidence` | string | enum | low, medium, high (based on fix_type + HDOP) |

### Example

```csv
probe_id,timestamp_utc,lat,lon,alt_m,fix_type,hdop,satellites,pitch_deg,roll_deg,confidence
P1,2026-02-18T14:30:12Z,51.2345678,-1.4567890,102.34,RTK_FIXED,0.8,14,0.3,0.2,high
P2,2026-02-18T14:31:05Z,51.2345690,-1.4567850,102.28,RTK_FIXED,0.9,13,0.5,0.1,high
P3,2026-02-18T14:32:18Z,51.2345701,-1.4567810,102.41,RTK_FLOAT,1.2,11,0.8,0.4,medium
```

## HIRT MIT Measurement CSV

One row per TX-RX measurement.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp_utc` | string | ISO 8601 | Measurement time |
| `tx_probe` | int | — | Transmitter probe index |
| `rx_probe` | int | — | Receiver probe index |
| `frequency_hz` | int | Hz | Excitation frequency |
| `amplitude` | float | V | Received signal amplitude |
| `phase_deg` | float | degrees | Phase relative to transmit |

## HIRT ERT Measurement CSV

One row per four-electrode measurement.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp_utc` | string | ISO 8601 | Measurement time |
| `a` | int | — | Current injection electrode A |
| `b` | int | — | Current injection electrode B |
| `m` | int | — | Potential measurement electrode M |
| `n` | int | — | Potential measurement electrode N |
| `voltage_mv` | float | mV | Measured potential difference |
| `current_ma` | float | mA | Injected current |
| `apparent_resistivity_ohm_m` | float | ohm-m | Calculated apparent resistivity |

## GeoSim Ingest Compatibility

GeoSim's `geosim/formats.py` and `geosim/survey_format.py` handle loading these CSV formats. The column names defined here are the canonical schema — any changes must be synchronized with GeoSim's loader code.
