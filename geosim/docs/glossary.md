# GeoSim Consolidated Glossary

Terms used across the HIRT, Pathfinder, and GeoSim projects. Each term is
tagged with which project(s) it applies to.

**Legend:** [H] = HIRT, [P] = Pathfinder, [G] = GeoSim, [All] = all projects

---

## A

**ADC (Analog-to-Digital Converter)** [All]:
Device that converts continuous analog voltage signals to discrete digital
values for processing by a microcontroller. In Pathfinder, the ADS1115
provides 16-bit resolution over two I2C channels. In HIRT, a delta-sigma
ADC provides high-resolution sampling for both MIT and ERT channels.

**ADC Saturation** [All]:
Condition where the input signal exceeds the ADC's measurement range,
causing readings to clip at the maximum or minimum digital value. In
Pathfinder firmware, the saturation threshold is +/-32000 counts. Saturated
readings are flagged in the GeoSim Survey Format via `QualityFlags`.

**ADS1115** [P]:
16-bit sigma-delta ADC with I2C interface, used in Pathfinder to digitise
fluxgate sensor outputs. Two ADS1115 chips share the I2C bus, each serving
two differential input channels (four sensor pairs total).

**Amplitude** [H][G]:
Magnitude of a signal, typically measured in volts or normalised units. In
HIRT MIT measurements, amplitude changes indicate the presence of
conductive objects in the subsurface.

**Anomaly** [All]:
A localised deviation from the expected background field or resistivity.
Pathfinder detects magnetic anomalies as gradient peaks; HIRT resolves
their 3D shape and conductivity/resistivity contrast through tomographic
imaging.

**Apparent Resistivity** [H][G]:
Resistivity value calculated from voltage and current measurements assuming
a homogeneous half-space. May differ significantly from the true
resistivity of any particular soil layer. Used as input to inversion
algorithms that recover the true resistivity distribution.

**Attenuation** [H][G]:
Reduction in signal amplitude as it passes through a medium. In MIT
measurements, attenuation of the transmitted field indicates the presence
of conductive objects or lossy soil between transmitter and receiver.

---

## B

**Baseline** [H][P]:
*Disambiguation -- this term has different meanings in each project:*

- **HIRT**: The distance between source and receiver probes in a crosshole
  measurement. Longer baselines provide greater investigation depth but
  reduced spatial resolution.
- **Pathfinder**: The vertical separation between top and bottom sensors
  in a gradiometer pair. The standard Pathfinder baseline is 0.35 m
  (35 cm).

**Below Noise Floor** [All]:
Condition where the measured signal magnitude is smaller than the estimated
noise level, making the measurement unreliable. Flagged in the GeoSim
Survey Format via `QualityFlags.below_noise_floor`.

**Bill of Materials (BOM)** [H][P]:
Complete list of components, quantities, and approximate costs needed to
build the system. Both projects maintain BOMs in their respective
`hardware/bom/` directories.

**Bottom Sensor** [P]:
The lower magnetometer in a Pathfinder gradiometer pair, positioned
approximately 15--20 cm above the ground surface. This sensor is closer to
near-surface anomalies and therefore has a stronger anomaly signal than the
top sensor. See also *Top Sensor*, *Gradiometer*.

---

## C

**Calibration** [All]:
Process of adjusting instrument readings against a known reference to
ensure accuracy. In Pathfinder, calibration involves zeroing sensor offsets
in a magnetically quiet environment. In HIRT, calibration includes both
electronic offset correction and geometric calibration of probe positions.
The GeoSim Survey Format tracks calibration state via `CalibrationStatus`
(raw or calibrated).

**Central Hub** [H]:
The above-ground electronics enclosure in the HIRT system that houses the
microcontroller (ESP32), signal generation (DDS), multiplexing, and data
acquisition circuits. All active electronics are located in the Central
Hub; probes are passive.

**Cole-Cole Model** [H][G]:
A frequency-dependent electrical conductivity model for soils and rocks.
Describes how conductivity varies with excitation frequency due to
polarisation effects. Used in advanced HIRT inversion to distinguish
different soil and target types.

**Common-Mode Rejection** [H][P]:
Ability of a differential measurement to reject signals that are identical
on both inputs. In Pathfinder, the gradiometer configuration provides
common-mode rejection of the uniform Earth field and diurnal variations.
In HIRT, differential amplifiers reject common-mode noise on long cable
runs.

**Coordinate Convention** [All]:
GeoSim uses a right-handed coordinate system: X = East, Y = North, Z = Up.
All positions are in metres, magnetic moments in A*m^2, fields in Tesla,
gradients in T/m. SI units throughout. GPS coordinates use WGS84 decimal
degrees.

**Coverage Rate** [P]:
Area of ground surveyed per unit time. Pathfinder targets >3000 m^2/hour
by using a wide swath (4 sensor pairs at 0.50 m spacing) and a walking
pace of ~1 m/s.

**Crosshole** [H][G]:
Measurement geometry where sensors are placed in separate boreholes or
probes inserted into the ground, providing true tomographic coverage
through the intervening soil volume. Distinguishes HIRT from surface-only
survey instruments like Pathfinder.

**Crosstalk** [P]:
Electromagnetic interference between adjacent sensor channels. At
Pathfinder's 50 cm sensor spacing, crosstalk between fluxgate pairs is
negligible. Crosstalk becomes significant only at spacings below ~2 cm
(e.g. medical MEG applications).

**CSV (Comma-Separated Values)** [All]:
Plain-text tabular data format used for field data logging. Pathfinder
firmware writes CSV files to SD card with columns for timestamp, GPS
coordinates, and per-pair sensor readings. The GeoSim Survey Format (GSF)
provides a structured JSON alternative for cross-instrument analysis.

---

## D

**Data Fusion** [G]:
The process of combining measurements from multiple instruments (e.g.
Pathfinder magnetic gradients and HIRT resistivity/EM data) into a single
analysis. The GeoSim Survey Format enables data fusion by providing a
common coordinate system and record structure.

**DDS (Direct Digital Synthesis)** [H]:
Technique for generating precise, frequency-stable excitation waveforms
digitally. Used in the HIRT Central Hub to create stable sinusoidal signals
for both the MIT and ERT channels.

**Delta-Sigma ADC** [H]:
Type of analog-to-digital converter that uses oversampling and noise
shaping to achieve very high resolution (typically 24-bit). Used in HIRT
for precision measurement of small induced voltages.

**Detection Depth** [P][H]:
Maximum depth at which a target of a given size can be reliably detected.
Pathfinder targets 0.5--1.5 m detection depth for ferrous objects.
HIRT achieves greater depths through crosshole geometry.

**DGPS (Differential GPS)** [P]:
GPS technique that uses a nearby reference station to improve position
accuracy from ~3 m (standard GPS) to ~0.5 m or better. Reflected in the
GeoSim Survey Format as `gps_fix_quality = 2`.

**Dipole Model** [P][G]:
Mathematical representation of a magnetic source as a point dipole,
producing a field that falls off as 1/r^3. GeoSim uses the dipole
approximation for Pathfinder simulation:
`B(r) = mu_0/4pi * [3(m . r_hat)r_hat - m] / r^3`.

**Diurnal Drift** [P][G]:
Slow variation in the Earth's magnetic field over the course of a day
(typically 20--50 nT), caused by solar-wind-driven ionospheric currents.
The gradiometer configuration cancels most diurnal drift because both
sensors experience the same variation.

---

## E

**Earth Field** [P][G]:
The ambient geomagnetic field at the survey location. GeoSim uses a
default Earth field of [0, 20 uT, 45 uT] representing a mid-latitude
site with ~65 deg inclination. Pathfinder measures perturbations to this
field caused by buried ferrous objects.

**Eddy Currents** [H][G]:
Electrical currents induced in conductive materials by time-varying
magnetic fields. In HIRT MIT measurements, eddy currents in buried
conductors cause attenuation and phase shift of the transmitted signal,
enabling detection and characterisation.

**ERT (Electrical Resistivity Tomography)** [H][G]:
Geophysical method that injects small electrical currents into the ground
and measures the resulting voltage distribution to map subsurface
resistivity. Sensitive to moisture, soil disturbance, voids, and clay
content. Implemented in HIRT as the "ERT-Lite" channel.

**ERT-Lite** [H]:
The resistivity measurement channel in HIRT, using ring electrodes on the
probe surface with small injection currents (0.5--2 mA). "Lite" indicates
reduced complexity compared to full ERT arrays.

**ESP32** [H]:
Microcontroller used in the HIRT Central Hub. Manages signal generation
via DDS, multiplexing across probe channels, and data acquisition.

---

## F

**Ferrite Core** [H]:
Magnetic ceramic material (typically MnZn ferrite) shaped as a rod and
used inside HIRT probe coils to increase inductance and improve coupling
efficiency. Concentrates magnetic flux within the coil.

**Ferrous** [P][G]:
Containing iron or iron alloys; strongly magnetic. Ferrous targets produce
large magnetic anomalies detectable by Pathfinder. Non-ferrous metals
(aluminium, copper) are generally invisible to passive magnetometry but
detectable by HIRT's active MIT channel.

**Fluxgate** [P]:
Type of vector magnetometer sensor that measures the component of the
magnetic field along its sensitive axis. Pathfinder uses FG-3+ fluxgate
sensors arranged as vertical pairs. Fluxgates are passive receivers that
do not emit fields.

**Forward Model** [G]:
A physics simulation that predicts instrument readings given a known
subsurface model. GeoSim implements forward models for magnetic dipoles
(Pathfinder) and, as stubs, for EM induction and resistivity (HIRT via
SimPEG/pyGIMLi).

**Frequency** [H][G]:
Number of cycles per second of an oscillating signal, measured in Hz.
In HIRT, lower excitation frequencies penetrate deeper into the ground;
higher frequencies provide better near-surface resolution. Typical HIRT
range: 2--50 kHz.

---

## G

**GeoSim Survey Format (GSF)** [G]:
JSON-based intermediate data format defined in `geosim/survey_format.py`.
Provides a common record structure (`SurveyRecord`) and file wrapper
(`SurveyFile`) for exchanging data between HIRT, Pathfinder, and
post-processing tools. See also `formats.py` for the lighter-weight
dict-based record format.

**GPS (Global Positioning System)** [P][G]:
Satellite navigation system providing position fixes. Pathfinder uses a
NEO-6M GPS module for georeferencing survey data. GPS coordinates are
stored in WGS84 decimal degrees and can be projected to local grid
coordinates via `geosim.coordinates`.

**Gradient** [P][G]:
The spatial rate of change of a field quantity. In Pathfinder, the vertical
magnetic gradient is computed as `delta_B = B(bottom) - B(top)`, measured
in nanotesla (nT) or ADC counts. Gradient measurements suppress the
uniform background field and enhance near-surface anomaly signals.

**Gradiometer** [P][G]:
An instrument that measures the spatial gradient of a field by using two
sensors at a known separation (the baseline). Pathfinder is a fluxgate
gradiometer with four vertical sensor pairs.

**Grid Coordinates** [All]:
Local Cartesian coordinate system (X East, Y North, in metres) used for
survey layout. Derived from GPS coordinates via a tangent-plane projection
centred on a chosen grid origin. See `geosim.coordinates.GridOrigin`.

**Grid Origin** [All]:
The reference point (in WGS84 lat/lon) at which the local survey grid has
coordinates (0, 0). Defined by `GridOrigin` in `geosim/coordinates.py`
and by `Location.grid_origin_lat/lon` in the GeoSim Survey Format.

---

## H

**Harness** [P]:
Padded backpack-style suspension system that supports the Pathfinder
sensor bar. Distributes weight across the operator's shoulders and waist so
that arms never bear instrument weight. Includes bungee isolation to
decouple walking vibration from the sensors.

**HDOP (Horizontal Dilution of Precision)** [P][G]:
A dimensionless measure of GPS position accuracy based on satellite
geometry. Lower HDOP indicates better precision. Stored in the GeoSim
Survey Format as `QualityFlags.hdop`.

**Heading Error** [P][G]:
Systematic error in magnetometer readings that depends on the sensor
orientation relative to the Earth's field. GeoSim models heading error as
one of three noise sources applied during Pathfinder survey simulation.

**HIRT (Hybrid Inductive-Resistivity Tomography)** [H][G]:
Dual-channel subsurface imaging system combining MIT (electromagnetic
induction) and ERT (electrical resistivity) in a crosshole geometry. Uses
passive probes inserted into the ground and a Central Hub for signal
generation and acquisition.

**Howland Current Source** [H]:
Precision voltage-controlled current source circuit topology used in the
HIRT ERT channel. Maintains constant injection current independent of
varying soil impedance (load).

---

## I

**Inclination (Magnetic)** [P][G]:
The angle between the magnetic field vector and the horizontal plane.
GeoSim's default Earth field has ~65 deg inclination, typical of
mid-latitude European sites.

**Inductance** [H]:
Property of a coil that opposes changes in current, measured in Henries (H)
or millihenries (mH). HIRT probe coils are designed for specific inductance
values to optimise coupling at the operating frequency.

**Inversion** [H][G]:
Mathematical process of reconstructing the 3D distribution of subsurface
properties (conductivity, resistivity) from a set of measurements. Not
covered by hardware; performed in post-processing software (e.g. SimPEG,
pyGIMLi).

**ISO 8601** [All]:
International standard for date and time representation. The GeoSim Survey
Format stores all timestamps as ISO 8601 strings in UTC (e.g.
`"2025-06-15T14:30:00+00:00"`). Pathfinder firmware uses millisecond
offsets that are converted to ISO 8601 during GSF export.

---

## J-K

**JSON (JavaScript Object Notation)** [G]:
Lightweight text-based data interchange format used by the GeoSim Survey
Format for structured survey data files (`.gsf.json`). Chosen over CSV for
its ability to represent nested metadata, typed enums, and quality flags.

---

## L

**Lock-in Detection** [H]:
Signal processing technique that extracts a small signal at a known
reference frequency from a noisy background. Can be implemented in analog
hardware (e.g. AD630 demodulator) or digitally. Provides high SNR for HIRT
MIT measurements.

**Local Grid** [All]:
See *Grid Coordinates*.

---

## M

**Magnetic Moment** [P][G]:
Vector quantity (in A*m^2) characterising the strength and orientation of a
magnetic dipole source. GeoSim represents buried targets as magnetic
sources with specified position and moment vectors. Moment can be set
explicitly or computed from target susceptibility and radius.

**Magnetometer** [P]:
Instrument that measures magnetic field strength or direction. Pathfinder
uses fluxgate magnetometers; each sensor measures one component of the
field along its sensitive axis.

**MCU (Microcontroller Unit)** [H][P]:
Small embedded computer. HIRT uses an ESP32 in its Central Hub. Pathfinder
uses an Arduino Nano. In both projects, the MCU manages sensor reading,
data logging, and (in HIRT) signal generation.

**MIT / MIT-3D (Magnetic Induction Tomography)** [H][G]:
Low-frequency electromagnetic method using transmitter and receiver coils.
Measures amplitude and phase changes caused by eddy currents in conductive
objects. "MIT-3D" specifically refers to HIRT's crosshole implementation
providing 3D volumetric imaging.

**Multiplexing** [H]:
Sequential switching between multiple probe channels using electronic
switches (multiplexers) in the HIRT Central Hub. Allows a single ADC to
serve many probes by reading them in rapid succession.

---

## N

**NEO-6M** [P]:
u-blox GPS receiver module used in Pathfinder for georeferencing survey
data. Provides standard NMEA output at 1 Hz update rate with typical
horizontal accuracy of ~2.5 m CEP.

**NMEA** [P]:
Standard protocol for GPS receiver output. Pathfinder parses NMEA GGA
sentences for latitude, longitude, altitude, fix quality, and HDOP.

**Node** [H]:
A probe insertion point in the HIRT survey grid. Node spacing determines
the spatial resolution and investigation depth of the tomographic image.

**Noise Floor** [P][G]:
The minimum detectable signal level, determined by the combined electronic,
environmental, and quantisation noise. Pathfinder's target noise floor is
~50 nT for gradient measurements.

**Noise Model** [G]:
GeoSim's parameterised noise simulation, combining three independent
sources: `SensorNoise` (white + 1/f electronic noise), `DiurnalDrift`
(geomagnetic variation), and `HeadingError` (orientation-dependent
systematic). Applied after clean physics computation.

---

## O

**Ohm-meter (ohm*m)** [H][G]:
SI unit of electrical resistivity. Typical values: dry sand 1000--10000
ohm*m, wet clay 1--10 ohm*m, metal <<1 ohm*m. HIRT ERT measurements
produce apparent resistivity in ohm*m.

**Operator** [All]:
The person conducting the field survey. Stored as optional metadata in the
GeoSim Survey Format (`SurveyRecord.operator`).

---

## P

**Pace Beeper** [P]:
Audio metronome built into Pathfinder firmware that beeps at regular
distance intervals (typically once per metre) to help the operator maintain
a consistent walking pace during survey.

**Pair** [P]:
See *Sensor Pair*.

**Pathfinder** [P][G]:
Handheld multi-sensor fluxgate gradiometer for rapid magnetic
reconnaissance. Identifies magnetic anomalies quickly so that detailed
investigation systems (like HIRT) can be deployed to specific locations.

**Phase** [H][G]:
The angular offset between transmitted and received sinusoidal signals,
measured in degrees or radians. In HIRT MIT measurements, phase changes
indicate the conductivity and size of buried objects.

**Pilot Rod** [H]:
Metal rod used to create an insertion hole in the ground for HIRT probes.
The pilot rod is removed before inserting the sensor probe to avoid
metallic contamination of measurements.

**PlatformIO** [P]:
Build system and IDE used for Pathfinder firmware development (Arduino Nano
target). Commands: `pio run` (build), `pio run -t upload` (flash),
`pio test` (unit tests).

**Probe** [H][P]:
*Disambiguation:*

- **HIRT**: The complete passive sensor assembly inserted into the ground,
  containing coils (MIT) and ring electrodes (ERT) but no active
  electronics. Multiple probes are connected to the Central Hub via Zone
  Hubs and trunk cables.
- **Pathfinder**: Not typically used. The equivalent component is the
  *sensor pair* (fluxgate top + bottom mounted on a drop tube).

**pyGIMLi** [H][G]:
Open-source Python library for geophysical inversion and modelling.
GeoSim uses pyGIMLi (optionally) for ERT forward modelling. Requires
separate installation via the `hirt` extras.

**PyVista** [G]:
3D visualisation library used by GeoSim for rendering terrain, buried
objects, magnetic field volumes, and survey scenes. Optional dependency
installed via the `viz` extras.

---

## Q

**Q Factor** [H]:
Quality factor of a resonant circuit or coil, defined as the ratio of
stored energy to energy dissipated per cycle. Higher Q indicates lower
losses and better measurement sensitivity. Important for HIRT probe coil
design.

**Quality Flags** [G]:
Structured metadata attached to each GeoSim Survey Format record
(`QualityFlags` dataclass) indicating data reliability: GPS fix quality,
HDOP, ADC saturation, and noise floor status.

**Quarto** [H]:
Publishing system used to render HIRT's technical manual from `.qmd`
(Quarto Markdown) source files into HTML and PDF. Commands: `quarto render`
(build), `quarto preview` (live reload).

---

## R

**Reciprocity** [H]:
Principle that a TX-to-RX measurement should equal the corresponding
RX-to-TX measurement (swapping transmitter and receiver roles). Used as a
data quality check in HIRT surveys -- large reciprocity errors indicate
electrode contact problems or instrument drift.

**Record** [G]:
A single measurement sample in the GeoSim Survey Format
(`SurveyRecord` dataclass), containing timestamp, location, instrument
type, measurement values, calibration status, and quality flags.

**Resistivity** [H][G]:
Intrinsic property of a material quantifying its resistance to electrical
current flow, measured in ohm*m. High resistivity: dry soil, voids, rock.
Low resistivity: wet soil, clay, metal. HIRT's ERT channel maps
resistivity distribution in the subsurface.

**Rim Deployment** [H]:
HIRT deployment strategy where probes are placed only around the perimeter
of the investigation area rather than throughout it. Reduces ground
disturbance within the area of interest while still providing tomographic
coverage.

**Rod Segment** [H]:
Individual fiberglass tube section that stacks with others to form the
structural body of a HIRT probe. Rod segments carry the coils and
electrodes at specified depths.

**RTK (Real-Time Kinematic)** [P]:
High-precision GPS technique achieving centimetre-level accuracy using
carrier-phase measurements and a local base station. Reflected in the
GeoSim Survey Format as `gps_fix_quality = 4`.

**RX (Receiver)** [H][G]:
The receiving coil or probe in a HIRT measurement. The RX coil measures
the field produced by eddy currents induced by the TX coil.

---

## S

**Scenario** [G]:
A JSON file defining the ground truth for a GeoSim simulation: target
positions, magnetic moments, soil properties, and survey parameters. The
scenario is the single source of truth; all simulated sensor data derives
from it.

**SD Card** [P]:
Removable flash storage medium used by Pathfinder firmware for CSV data
logging during field surveys.

**Section** [H]:
A grid area surveyed in one HIRT deployment cycle, typically 10x10 m.
Sized for a small field team to deploy and recover probes within a single
session.

**Sensitivity Volume** [H][G]:
The 3D region of the subsurface that contributes most to a particular
measurement. Depends on probe geometry, spacing, excitation frequency, and
soil properties. Determines the effective resolution of the tomographic
image.

**Sensor Noise** [P][G]:
Random electronic noise from the sensor and its signal-conditioning
circuitry. GeoSim models sensor noise as a combination of white noise and
1/f (pink) noise, parameterised by noise density and corner frequency.

**Sensor Pair** [P]:
A pair of vertically separated fluxgate magnetometers forming one
gradiometer channel. Pathfinder has four sensor pairs spaced at 0.50 m
horizontally, each with a 0.35 m vertical baseline.

**Signal-to-Noise Ratio (SNR)** [All]:
Ratio of the desired signal amplitude to the noise amplitude. Higher SNR
means better data quality. Expressed in dB or as a dimensionless ratio.

**SimPEG** [H][G]:
Open-source Python framework for simulation and parameter estimation in
geophysics. GeoSim uses SimPEG (optionally) for electromagnetic forward
modelling. Requires separate installation via the `hirt` extras.

**Site Name** [All]:
Human-readable identifier for the survey location. Stored as metadata in
the GeoSim Survey Format (`SurveyFile.site_name`).

**Survey** [All]:
*Disambiguation:*

- **General**: A systematic collection of geophysical measurements over an
  area.
- **Pathfinder**: Walking a series of parallel traverses across a site,
  collecting continuous gradient data.
- **HIRT**: Deploying probes in a grid and running a programmed sequence
  of transmit-receive measurements.
- **GeoSim**: A simulated survey generated by the physics engine, or a
  collection of `SurveyRecord` entries in the GSF.

**Survey Stake** [H]:
Temporary marker placed at grid node positions during HIRT site layout.
Distinct from the sensor probe -- survey stakes are used for positioning
only and are removed or offset before measurements.

**Susceptibility (Magnetic)** [P][G]:
Dimensionless property of a material describing how strongly it is
magnetised in an applied field. GeoSim can compute a target's magnetic
moment from its susceptibility and volume if the moment is not specified
directly.

**Swath** [P]:
The width of ground covered in a single survey pass. Pathfinder's swath
is determined by the number of sensor pairs and their spacing: four pairs
at 0.50 m gives a 1.5 m swath.

---

## T

**Tangent-Plane Projection** [G]:
The coordinate transformation used by `geosim.coordinates.gps_to_grid` to
convert WGS84 lat/lon to local grid metres. Uses the WGS84 ellipsoid
radii of curvature at the grid origin. Accurate to <1 mm for survey areas
up to ~10 km across.

**Timestamp** [All]:
Time associated with a measurement. Pathfinder firmware records timestamps
as milliseconds since power-on. The GeoSim Survey Format normalises all
timestamps to ISO 8601 UTC strings.

**Tomography** [H][G]:
Imaging method that reconstructs 2D or 3D internal structure from multiple
external or crosshole measurements. Analogous to medical CT scanning.
HIRT uses tomographic inversion of MIT and ERT data to produce volumetric
conductivity and resistivity maps.

**Top Sensor** [P]:
The upper magnetometer in a Pathfinder gradiometer pair, positioned at the
crossbar (~50 cm above ground). Serves as the reference sensor; its
reading is subtracted from the bottom sensor to compute the gradient. See
also *Bottom Sensor*, *Gradiometer*.

**Trapeze** [P]:
The horizontal crossbar (carbon fibre or aluminium tube, 1.5--2.0 m) from
which Pathfinder's sensor pairs are suspended. Named after the
configuration used by commercial systems such as the Bartington Grad601.

**Trunk Cable** [H]:
High-density shielded multi-core cable (e.g. DB25 connector) that carries
analog signals from a Zone Hub to the HIRT Central Hub. Each trunk cable
serves four probes aggregated by one Zone Hub.

**TX (Transmitter)** [H][G]:
The transmitting coil or probe in a HIRT measurement. Generates the
excitation field that induces eddy currents in subsurface conductors.

---

## U

**Unexploded Ordnance (UXO)** [H][P]:
Military munitions that failed to detonate and remain live in the ground.
A primary target type for both HIRT (detailed investigation) and Pathfinder
(rapid screening). UXO sites require EOD (Explosive Ordnance Disposal)
clearance before any ground-intrusive work.

---

## V

**Vacuum Permeability** [G]:
Fundamental physical constant `mu_0 = 4*pi * 10^-7 T*m/A` used in
magnetic field calculations. Appears in the dipole field formula
implemented in `geosim/magnetics/dipole.py`.

---

## W

**Walk Path** [P][G]:
The trajectory followed by the operator during a Pathfinder survey.
GeoSim's `generate_walk_path()` function creates simulated walk paths
defined by start point, end point, speed, and sample rate.

**WGS84** [All]:
World Geodetic System 1984 -- the reference coordinate system used by GPS.
All lat/lon coordinates in the GeoSim ecosystem are in WGS84 decimal
degrees. The WGS84 ellipsoid parameters (semi-major axis 6378137 m,
flattening 1/298.257223563) are used for coordinate projections.

---

## X-Y-Z

**ZeroMQ (ZMQ)** [G]:
High-performance asynchronous messaging library used by `geosim-server`
to expose the physics engine to external clients (e.g. Godot visualisation
frontend) via a REQ-REP socket pattern.

**Zone Hub** [H]:
A passive breakout box located near a group of four HIRT probes. Aggregates
the analog signals from four probes into a single trunk cable for
connection to the Central Hub. Contains no active electronics.
