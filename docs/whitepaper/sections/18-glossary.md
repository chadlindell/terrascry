# 18. Glossary (quick reference)

## Acronyms and Terms

### A–D

**ADC (Analog-to-Digital Converter)**
- Converts analog voltage signals to digital values for processing

**Amplitude**
- Magnitude of a signal, typically measured in volts or normalized units

**Baseline**
- Distance/geometry between source and receiver probes
- Longer baselines provide deeper investigation depth

**BOM (Bill of Materials)**
- Complete list of components needed to build the system

**Crosshole**
- Measurement geometry where sensors are placed in separate boreholes/probes
- Provides true tomographic coverage through the volume

### E–H

**ERT (Electrical Resistivity Tomography)**
- Method using current injection and voltage measurement to map soil resistivity
- Detects moisture, disturbance, voids, and soil variations

**Eddy Currents**
- Electrical currents induced in conductive materials by changing magnetic fields
- Cause attenuation and phase lag in MIT measurements

**Ferrite Core**
- Magnetic material used in coils to increase inductance and efficiency
- Typically rod-shaped for probe applications

**Frequency**
- Number of cycles per second, measured in Hz (Hertz)
- Lower frequencies penetrate deeper; higher frequencies provide better resolution

### I–L

**Inductance**
- Property of a coil that resists changes in current
- Measured in Henries (H) or millihenries (mH)

**Inversion**
- Mathematical process to reconstruct 3D property distribution from measurements
- Software step (not covered in this hardware guide)

**Lock‑in Detection**
- Technique to extract small signals at a known reference frequency with high SNR
- Can be analog (AD630) or digital (DSP-based)

### M–P

**MIT‑3D (Magneto‑Inductive Tomography)**
- Low-frequency electromagnetic method using coils
- Measures amplitude and phase changes caused by conductive objects

**MCU (Microcontroller Unit)**
- Small computer on a chip (e.g., ESP32, STM32)
- Controls probe operation and data collection

**Phase**
- Timing relationship between transmitted and received signals
- Measured in degrees, indicates conductivity and distance

**Probe**
- Complete sensor unit inserted into the ground
- Contains MIT coils, ERT rings, and electronics

### Q–T

**Q Factor**
- Quality factor of a coil, indicates efficiency
- Higher Q = lower losses, better performance

**Reciprocity**
- Principle that TX→RX measurement should equal RX→TX measurement
- Used for quality control

**Resistivity**
- Property of material to resist electrical current flow
- Measured in ohm-meters (Ω·m)
- High resistivity: dry soil, voids
- Low resistivity: wet soil, clay, metal

**RX (Receive/Receiver)**
- Receiving coil or probe that measures signals

**Sensitivity Volume**
- The 3D region that contributes most to a particular measurement
- Depends on probe spacing, frequency, and soil properties

### U–Z

**TX (Transmit/Transmitter)**
- Transmitting coil or probe that generates signals

**Tomography**
- Imaging method that reconstructs 3D structure from multiple measurements
- Similar to medical CT scanning

**UXO (Unexploded Ordnance)**
- Live explosive devices that may be present at WWII sites
- Requires EOD clearance before deployment

## Measurement Terms

**Apparent Resistivity**
- Calculated resistivity from voltage/current measurements
- May differ from true resistivity due to measurement geometry

**Attenuation**
- Reduction in signal amplitude
- Indicates presence of conductive objects or losses

**Common-Mode Rejection**
- Ability to reject signals common to both inputs
- Important for differential measurements

**SNR (Signal-to-Noise Ratio)**
- Ratio of signal strength to noise level
- Higher SNR = better data quality

## Field Terms

**Section**
- Grid area surveyed in one deployment cycle
- Typically 10×10 m, manageable by small team

**Node**
- Probe insertion point in the grid
- Spacing determines resolution and depth

**Pilot Rod**
- Metal rod used to create hole for probe insertion
- Removed before inserting sensor rod

**Rim Deployment**
- Placing probes around perimeter rather than throughout area
- Reduces ground disturbance

