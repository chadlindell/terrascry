# 11. Data Recording

## Overview

This section specifies the data formats and organization for HIRT field measurements. Consistent data recording ensures reliable post-processing and long-term data management.

---

## 11.1 MIT Record Format

Each MIT measurement should record:

| Field | Description | Units/Format |
|-------|-------------|--------------|
| `timestamp` | Measurement time | ISO 8601 or Unix timestamp |
| `section_id` | Survey section identifier | String (e.g., "S01") |
| `zone_id` | Identifier for the Zone Hub used | String (e.g., "ZA", "ZB") |
| `tx_probe_id` | Transmitting probe ID | String (e.g., "P01") |
| `rx_probe_id` | Receiving probe ID | String (e.g., "P02") |
| `freq_hz` | Measurement frequency | Hz |
| `amp` | Amplitude | V or normalized |
| `phase_deg` | Phase | Degrees |
| `tx_current_mA` | TX coil current | mA |

---

## 11.2 ERT Record Format

Each ERT measurement should record:

| Field | Description | Units/Format |
|-------|-------------|--------------|
| `timestamp` | Measurement time | ISO 8601 |
| `section_id` | Survey section identifier | String |
| `zone_id` | Identifier for the Zone Hub used | String |
| `inject_pos_id` | Positive current probe ID | String |
| `inject_neg_id` | Negative current probe ID | String |
| `sense_id` | Voltage sensing probe ID | String |
| `volt_mV` | Measured voltage | mV |
| `current_mA` | Injected current | mA |

### Example ERT Record

```csv
timestamp,section_id,inject_pos_probe_id,inject_neg_probe_id,sense_probe_id,volt_mV,current_mA,polarity,notes
2024-03-15T10:45:12Z,S01,P01,P20,P05,12.5,1.2,+1,
2024-03-15T10:45:13Z,S01,P01,P20,P06,8.3,1.2,+1,
2024-03-15T10:45:14Z,S01,P01,P20,P05,-12.4,1.2,-1,reversed polarity
```

---

## 11.3 Probe Registry

Each probe should have a registry entry:

| Field | Description | Units/Format |
|-------|-------------|--------------|
| `probe_id` | Unique probe identifier | String |
| `coil_L_mH` | TX coil inductance | mH |
| `coil_Q` | Coil Q factor | Dimensionless |
| `rx_gain_dB` | RX amplifier gain | dB |
| `ring_depths_m` | ERT ring depths | m (comma-separated, e.g., "0.5,1.5") |
| `firmware_rev` | Firmware version | String |
| `calibration_date` | Last calibration date | YYYY-MM-DD |
| `notes` | Additional notes | Free text |

### Example Probe Registry

```csv
probe_id,coil_L_mH,coil_Q,rx_gain_dB,ring_depths_m,firmware_rev,calibration_date,notes
P01,1.2,25,40,0.5,1.5,v1.2,2024-03-10,
P02,1.15,28,40,0.5,1.5,v1.2,2024-03-10,
P03,1.18,26,40,0.5,1.5,v1.2,2024-03-10,
```

---

## 11.4 Data Storage

### File Organization

- **One CSV file per section** for MIT data
- **One CSV file per section** for ERT data
- **One registry file** for all probes (shared)
- **Paper log** for conditions and notes

### Naming Convention

- MIT: `MIT_S{section_id}_{date}.csv`
- ERT: `ERT_S{section_id}_{date}.csv`
- Registry: `probe_registry.csv`

### Example Directory Structure

```
data/
├── 2024-03-15/
│   ├── MIT_S01_2024-03-15.csv
│   ├── ERT_S01_2024-03-15.csv
│   ├── MIT_S02_2024-03-15.csv
│   └── ERT_S02_2024-03-15.csv
├── probe_registry.csv
└── field_log_2024-03-15.txt
```

---

## 11.5 Metadata

### Site Information

- Site name/location
- Survey date/time
- Team members
- Weather conditions
- Soil type/moisture
- Site access notes

### Measurement Parameters

- Frequency list (MIT)
- Current levels (ERT)
- Probe spacing
- Insertion depths
- Grid coordinates/orientation

---

## 11.6 Data Quality Notes

Record in paper log or notes field:

- Soil moisture changes during survey
- Temperature variations
- Disturbances (vehicles, people)
- Equipment issues
- Anomalous readings (with context)

## 11.7 Software Compatibility Roadmap

The CSV data formats described above are designed as raw intermediate storage. Future software tools will provide import scripts to convert these raw logs into standard formats for inversion:

*   **EIDORS (Matlab/Octave):** Import script will generate `.mat` structures for MIT-3D finite element reconstruction.
*   **pyGIMLi (Python):** Converters will map ERT data to the Unified Data Format (Unified Data Format) for coupled inversion.
*   **ResIPy:** Direct import support for ERT `.csv` files is planned.
