# HIRT Software Stack

This directory contains the modernized software stack for the HIRT system (2026 Revision).

## Architecture

The system consists of two main components:
1.  **HIRT Daemon (`hirt-daemon`):** A Python service that manages the serial connection to the ESP32 firmware (or mock device), handles data ingestion, and writes to HDF5 storage.
2.  **HIRT UI (`hirt-ui`):** A FastAPI-based web server that provides a mobile-friendly dashboard for the field operator.

## Directory Structure

```
software/
├── hirt-daemon/
│   ├── src/
│   │   ├── daemon.py       # Core logic (HDF5 writer)
│   │   └── mock_device.py  # Hardware simulator
│   └── tests/
├── hirt-ui/
│   ├── main.py             # FastAPI backend
│   └── static/             # HTML/JS frontend
└── requirements.txt        # Python dependencies
```

## Quick Start (Development)

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Mock System:**
    Start the UI server (which auto-starts the mock daemon in Phase 1):
    ```bash
    cd hirt-ui
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

3.  **Access the Dashboard:**
    Open `http://localhost:8000` in your browser.

## Data Format

Data is saved to `data_out/` in HDF5 format.
*   **Group:** `/raw_data`
*   **Dataset:** `mit_measurements` (Compound type: TX, RX, Freq, Amp, Phase, Noise)

## Next Steps
*   Implement real serial driver in `daemon.py`.
*   Port `mock_device.py` logic to C++ for ESP32 firmware.
