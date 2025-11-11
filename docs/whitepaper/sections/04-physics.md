# 4. Physics (practical, field-level)

## 4.1 MIT‑3D (low‑frequency EM)

### Operating Principle

- **TX coil** drives a stable sine wave at **2–50 kHz**
- **RX coils** measure magnetic field amplitude/phase
- Conductive objects (metal) create **eddy currents** → cause **attenuation & phase lag** along TX→RX paths

### Frequency Selection

- **Lower frequency** → **deeper penetration**
- **Higher frequency** → **sharper sensitivity** near the probes
- Typical range: **2–50 kHz** (choose 3–5 points)

### Measurement Geometry

With many TX→RX pairs around/through the site, the dataset samples the volume like a **CT scan**:
- Each TX→RX pair provides a path-integrated measurement
- Multiple paths through the same volume provide redundancy
- Inversion algorithms reconstruct 3D conductivity distribution

## 4.2 ERT‑Lite (galvanic)

### Operating Principle

- Two electrodes **inject constant current**
- Voltage measured at other electrodes yields **apparent resistivity**
- **Disturbed fill, moisture, compacted layers, voids** produce detectable contrasts

### Current Specifications

- Small, safe currents: **0.5–2 mA**
- **Polarity reversal** periodically to minimize polarization artifacts
- Low‑freq AC option: **8–16 Hz** to reduce polarization

### Measurement Patterns

- **Long baselines** (corner‑to‑corner, edge‑to‑edge, center‑to‑edge)
- Multiple injection pairs for redundancy
- All probes log voltages simultaneously

## Depth of Investigation (rule‑of‑thumb)

- With rods inserted **~3 m**: **sensitivity volume** typically extends **~3–6 m** deep depending on soil and probe spacing
- With **~1.5 m rods** (woods): expect **~2–3 m** effective sensitivity
- Actual depth depends on:
  - Soil conductivity
  - Probe spacing
  - Measurement frequency (for MIT)
  - Current injection geometry (for ERT)

