# Comparable Open Source Projects Catalog

**Research Date:** 2026-02-01
**Purpose:** Reference library of open-source scientific hardware projects comparable to HIRT

---

## Summary

This catalog documents open-source hardware projects relevant to HIRT's dual-modality subsurface imaging approach. Projects are categorized by sensing modality and rated for documentation quality, academic validation, and buildability.

**Key Finding:** No existing open-source project combines MIT (inductive metal detection) with ERT (electrical resistivity tomography) in a single integrated system. HIRT fills a unique gap in the open hardware landscape.

---

## Tier 1: Direct Comparisons (Geophysics Instrumentation)

### OhmPi - Open Source Resistivity Meter
**The closest comparable project to HIRT's ERT channel**

| Attribute | Details |
|-----------|---------|
| **URL** | https://ohmpi.org/ |
| **Repository** | https://github.com/ohmpi/ohmpi (mirror of GitLab) |
| **Primary GitLab** | https://gitlab.irstea.fr/reversaal/ohmpi |
| **Publication** | [HardwareX 2020](https://doi.org/10.1016/j.ohx.2020.e00122), [HardwareX 2023 update](https://pubmed.ncbi.nlm.nih.gov/35498256/) |
| **What it measures** | Electrical resistivity tomography (ERT), induced polarization (IP) |
| **Electrode count** | Up to 64 electrodes (v2024) |
| **Cost** | <$500 for 32-electrode system |
| **Documentation** | **Excellent** - Full build guides, API reference, tutorials |
| **Hardware** | Raspberry Pi-based, I2C multiplexers, custom acquisition board |
| **Software** | Python API, web interface, MQTT IoT support |
| **License** | Open source (check repo for specifics) |

**Why it matters for HIRT:** OhmPi is the gold standard for open-source ERT. Study their documentation structure, HardwareX submission, and community engagement model.

---

### OpenEIT - Electrical Impedance Tomography
**Closest to HIRT's multi-electrode tomographic approach**

| Attribute | Details |
|-----------|---------|
| **URL** | https://openeit.github.io/ |
| **GitHub Org** | https://github.com/OpenEIT |
| **Key Repos** | [OpenEIT](https://github.com/OpenEIT/OpenEIT) (dashboard), [EIT_EE](https://github.com/OpenEIT/EIT_EE) (PCB), [EIT_Firmware](https://github.com/OpenEIT/EIT_Firmware), [pyEIT](https://github.com/OpenEIT/pyEIT) (algorithms) |
| **Publication** | [Open Source Imaging listing](https://www.opensourceimaging.org/project/open-eit-electrical-impedance-tomography/) |
| **What it measures** | Electrical impedance tomography (biomedical focus, but transferable) |
| **Electrode count** | Up to 32 electrodes |
| **Frequency range** | 80 Hz - 80 kHz |
| **Cost** | ~$200-500 depending on configuration |
| **Documentation** | **Good** - Modular repos, some gaps in integration docs |
| **Hardware** | ADuCM350 precision analog impedance analyzer, 2" square PCB, Bluetooth |
| **Reconstruction** | Back Projection, Graz Consensus, Gauss-Newton |
| **License** | CC BY-NC-SA 4.0 |

**Why it matters for HIRT:** Excellent reference for multi-channel electrode front-end design, noise/CMRR optimization, and tomographic reconstruction algorithms. The pyEIT library is directly applicable.

---

## Tier 2: Metal Detection / Inductive Sensing

### Pulse Induction Metal Detectors (Multiple Projects)

| Project | URL | MCU | Documentation | Notes |
|---------|-----|-----|---------------|-------|
| **balanced_pulse_induction** | [GitHub](https://github.com/Petersdavis/balanced_pulse_induction_metal_detector) | Analog + discrete | **Good** | Balanced coil design, LTSpice models |
| **pi-metal-detector** | [GitHub](https://github.com/anorimaki/pi-metal-detector) | PIC | **Good** | Schematics, simulations, documentation |
| **teensyPi** | [GitHub](https://github.com/msilveus/teensyPi) | Teensy 4.1 | **Moderate** | Modern MCU, LTSpice models |
| **tiny-metal-detector** | [GitHub](https://github.com/lpodkalicki/tiny-metal-detector) | ATtiny13 | **Minimal** | Ultra-simple, educational |
| **Hackaday PI Detector** | [Hackaday.io](https://hackaday.io/project/179588-pulse-induction-metal-detector) | Various | **Good** | Multiple schematic variations |
| **FoxyPI** | [Arduino Hub](https://projecthub.arduino.cc/FoxyLab/d92e7fd8-60b0-4bd6-bd2b-a95d1472aa3c) | Arduino Nano | **Good** | Well-documented build |

**Why it matters for HIRT:** These projects demonstrate coil driver circuits, receive amplification, and signal processing approaches applicable to MIT-3D channel design.

---

### VLF Metal Detectors

| Project | URL | Notes |
|---------|-----|-------|
| **Geotech1 Forums** | http://www.geotech1.com/forums/ | Engineering-level discussion of coil design, VLF discrimination |

**Note:** VLF detectors offer frequency discrimination capabilities useful for material identification, complementing pulse induction approaches.

---

## Tier 3: Magnetometry

### PyPPM - Proton Precession Magnetometer
**Serious scientific instrument, fully open source**

| Attribute | Details |
|-----------|---------|
| **URL** | http://geekysuavo.github.io/ppm/ |
| **GitHub** | https://github.com/geekysuavo/pyppm |
| **Hackaday** | https://hackaday.io/project/1376-pyppm-a-proton-precession-magnetometer-for-all |
| **What it measures** | Earth's magnetic field strength (30-60 uT range) |
| **Output** | ~2 kHz Larmor precession frequency |
| **Documentation** | **Excellent** - Complete hardware designs, Python API |
| **Hardware** | Single-board design, USB connection to Linux/Mac |
| **Variants** | PyPPMv1 (basic PPM), PyPPMv2 (Earth's Field NMR Spectrometer) |
| **License** | GPLv3 (code), CC BY-SA (hardware) |

**Why it matters for HIRT:** Archaeological magnetometry reference. Though not directly applicable to HIRT's modalities, demonstrates open-source scientific instrument rigor.

### MrSzymonello/ppm
| Attribute | Details |
|-----------|---------|
| **GitHub** | https://github.com/MrSzymonello/ppm |
| **What it measures** | Proton precession magnetometer |
| **Documentation** | **Moderate** |

---

## Tier 4: Ground Penetrating Radar (GPR)

### Open GPR Projects

| Project | URL | Frequency | Cost | Documentation |
|---------|-----|-----------|------|---------------|
| **Open Ground Penetrating Radar (oGPR)** | [Hackaday.io](https://hackaday.io/project/4440-open-ground-penetrating-radar) | Various | ~$500 target | **Good** |
| **GPRino** | [Hackaday.io](https://hackaday.io/project/175115-gprino) | 250 MHz - 10 GHz | Moderate | **Good** |
| **Educational GPR (Polish)** | [PDF Guide](https://gpradar.eu/onewebmedia/TU1208_GPRforeducationaluse_November2017_FerraraChizhPietrelli.pdf) | Various | ~600 EUR | **Excellent** |

**Technical Notes:**
- GPR penetration depth varies dramatically: meters in arctic regions at 100 MHz, centimeters in concrete at GHz
- Resolution proportional to bandwidth
- Lower frequencies = deeper penetration but worse resolution

**Why it matters for HIRT:** GPR is a complementary subsurface imaging modality. Understanding its limitations helps position HIRT's advantages (works in conductive soils where GPR fails).

---

## Tier 5: Environmental Monitoring Platforms

### FieldKit
**Gold standard for modular open-source environmental sensing**

| Attribute | Details |
|-----------|---------|
| **URL** | https://www.fieldkit.org/ |
| **Hackaday** | https://hackaday.io/project/26354-fieldkit |
| **What it measures** | Modular: water chemistry (pH, DO, EC, temp), distance, weather |
| **Cost** | Varies by configuration |
| **Documentation** | **Excellent** - Hardware, software, data portal |
| **Hardware** | SAM D51 MCU, ATWINC1500 WiFi, GPS, modular sensor backplane |
| **Recognition** | 2019 Hackaday Grand Prize winner |
| **Deployment** | 24+ countries, partnerships with UCLA, WCS, FIU |
| **License** | Fully open source |

**Why it matters for HIRT:** FieldKit demonstrates how to build a modular, ruggedized field platform with excellent documentation and community engagement. Their approach to sensor modularity is directly applicable.

---

### Cave Pearl Project
**Long-term underwater environmental logging**

| Attribute | Details |
|-----------|---------|
| **URL** | https://thecavepearlproject.org/ |
| **Hackaday** | https://hackaday.io/project/6961-the-cave-pearl-project |
| **Publication** | [MDPI Sensors 2018](https://www.mdpi.com/1424-8220/18/2/530) |
| **What it measures** | Conductivity, temperature, pressure, flow (underwater caves) |
| **Battery life** | >1 year on 3x AA batteries |
| **Documentation** | **Excellent** - Narrative blog format, deep technical detail |
| **Hardware** | Arduino-based, PVC housings, breakout board approach |
| **EC Range** | 0.05-55 mS/cm (freshwater to seawater) |

**Why it matters for HIRT:** Demonstrates long-term field deployment, power optimization, and conductivity measurement in harsh environments. Their approach to electrode corrosion and ground loops is directly relevant.

---

### OpenCTD - Conductivity/Temperature/Depth Profiler

| Attribute | Details |
|-----------|---------|
| **GitHub** | https://github.com/OceanographyforEveryone/OpenCTD |
| **What it measures** | Conductivity (salinity), temperature, depth |
| **Cost** | $100-1000 depending on configuration |
| **Documentation** | **Good** |

---

## Tier 6: Seismology

### Raspberry Shake
**Professional-grade seismograph, partially open**

| Attribute | Details |
|-----------|---------|
| **URL** | https://raspberryshake.org/ |
| **What it measures** | Seismic/ground motion, infrasound |
| **Cost** | From ~$175 (basic geophone + board) |
| **Documentation** | **Excellent** - Tutorials, DIY guides |
| **Hardware** | Geophone sensor + Raspberry Pi HAT |
| **Software** | Uses USGS open-source seismology software |
| **Data format** | SEED (standard seismology format) |
| **DIY files** | Laser cutter and 3D printer files for enclosures |

### Seisberry (Fully Open Source Alternative)

| Attribute | Details |
|-----------|---------|
| **GitHub** | Fork of will127534/RaspberryPi-seismograph |
| **Hardware** | Raspberry Pi 3B+, Waveshare AD/AC board, geophones |
| **Cost** | <$100 (1-component), <$160 (3-component) |
| **Documentation** | **Good** |

---

## Tier 7: Oil & Gas Borehole Technology

**Key Finding:** Oil & gas has extensive open-source **software** for borehole data processing, but virtually no open-source **hardware**. The actual downhole sensors (wireline tools, LWD) are 100% proprietary from Schlumberger, Baker Hughes, Halliburton, etc. However, the software and theory are directly applicable to HIRT.

### Seismic Processing Frameworks

#### Madagascar
**The most comprehensive open-source seismic package**

| Attribute | Details |
|-----------|---------|
| **URL** | https://ahay.org/wiki/Main_Page |
| **GitHub** | https://github.com/ahay/src |
| **SourceForge** | https://sourceforge.net/projects/rsf/ |
| **Scale** | 1000+ main programs, 5000+ tests |
| **History** | Started 2003, public 2006, builds on 30 years of SEPlib/SU |
| **License** | GPL |
| **Languages** | C (low-level), Python (workflows) |
| **Capabilities** | Full seismic processing: geometry, deconvolution, F-K filter, velocity analysis, NMO, migration |

**Why it matters for HIRT:** Madagascar's reproducible workflow philosophy and data format design are excellent models. Its integration with OpenDTect demonstrates how processing and interpretation can work together.

---

#### Seismic Unix (SU)
**The original open-source seismic package**

| Attribute | Details |
|-----------|---------|
| **URL** | https://wiki.seismic-unix.org/ |
| **SEG Wiki** | https://wiki.seg.org/wiki/Seismic_Unix |
| **Origin** | Colorado School of Mines, Center for Wave Phenomena |
| **History** | Late 1970s (Stanford), expanded at CSM 1986+ |
| **Maintainer** | John Stockwell |
| **Recognition** | 2014 SEG Presidential Award, 2002 SEG Special Commendation |
| **Platforms** | Unix, Linux, Mac OS X, Cygwin (Windows) |

**Why it matters for HIRT:** SU established many conventions still used in geophysics software. Understanding its data formats (SU, SEG-Y) aids interoperability.

---

#### OpenDTect
**Commercial-quality open-source seismic interpretation**

| Attribute | Details |
|-----------|---------|
| **URL** | https://dgbes.com/software/opendtect |
| **Developer** | dGB Earth Sciences |
| **License** | GPL, commercial, and academic licenses |
| **Capabilities** | 2D/3D/4D seismic visualization, attribute analysis, horizon tracking |
| **Integration** | Madagascar GUI plugin available |
| **Data Repository** | Open Seismic Repository (F3 North Sea demo, etc.) |

**Why it matters for HIRT:** OpenDTect demonstrates that open-source can succeed commercially in oil & gas. Their "freemium" model and extensive documentation are worth studying.

---

### Crosshole & Borehole Tomography

#### bh_tomo
**Directly applicable to HIRT's crosshole geometry**

| Attribute | Details |
|-----------|---------|
| **GitHub** | https://github.com/groupeLIAMG/bh_tomo |
| **Origin** | Ecole Polytechnique of Montreal (LIAMG group) |
| **Platform** | MATLAB 2015b+ |
| **License** | GNU |
| **Capabilities** | Borehole georadar AND seismic tomography, ray-based 2D/3D inversion |
| **Data types** | Traveltime and amplitude datasets |

**Why it matters for HIRT:** This is the most directly applicable software for HIRT's crosshole ERT processing. Key algorithms:
- Ray-path computation (straight and curved)
- Crosshole geometry handling
- Tomographic inversion with constraints
- Anisotropy support

---

### Well Log Data Libraries (Python)

| Library | Purpose | URL/Package |
|---------|---------|-------------|
| **lasio** | Read/write LAS files (CWLS standard) | https://lasio.readthedocs.io/ |
| **welly** | High-level well log analysis, QC, synthetics | `pip install welly` |
| **dlisio** | Read DLIS/LIS binary files (Equinor) | `pip install dlisio` |
| **PetroPy** | Petrophysical workflows, log viewer | https://github.com/toddheitmann/PetroPy |

**Learning Resource:** [Petrophysics-Python-Series](https://github.com/andymcdgeo/Petrophysics-Python-Series) - Jupyter notebooks for well log analysis

**Why it matters for HIRT:** These libraries handle industry-standard borehole data formats. If HIRT data needs to interoperate with oil & gas workflows, understanding LAS/DLIS formats is valuable.

---

### Distributed Acoustic Sensing (DAS)

**Emerging technology using fiber optics as distributed sensors**

| Tool | Type | Notes |
|------|------|-------|
| **ExploreDAS** | MATLAB tool | Open-source modeling/imaging for DAS seismic |
| **Commercial DAS** | Hardware | Interrogators cost $50k-500k+ (no open alternatives) |

**DAS Concept:** Fiber optic cable becomes the sensor - can be kilometers long, deployed in boreholes, trenches, or seafloor. Measures acoustic waves via Rayleigh backscattering.

**Why it matters for HIRT:** DAS represents a different approach to distributed subsurface sensing. While hardware is commercial, the signal processing concepts (distributed strain measurement, spatial sampling) may inform future HIRT developments.

---

### Oil & Gas Hardware Gap

| Component | Open Source? | Reality |
|-----------|--------------|---------|
| Wireline logging tools | No | Schlumberger, Baker Hughes, Halliburton |
| LWD (Logging While Drilling) | No | Same vendors, $100k+ tools |
| Downhole sensors | No | Proprietary, harsh environment rated |
| DAS interrogators | No | Commercial only |
| Surface seismic equipment | Partial | Some DIY geophones exist |

**HIRT fills a genuine gap:** Open-source software assumes commercial hardware. HIRT provides the missing open hardware layer for subsurface probe sensing.

---

## Tier 8: Geophysics Inversion & Modeling Libraries

### pyGIMLi
**Primary recommendation for HIRT data processing**

| Attribute | Details |
|-----------|---------|
| **URL** | https://www.pygimli.org/ |
| **GitHub** | https://github.com/gimli-org/pyGIMLi |
| **Publication** | [Computers & Geosciences 2017](https://doi.org/10.1016/j.cageo.2017.07.011) |
| **Capabilities** | Forward modeling, inversion, mesh generation (2D/3D) |
| **Methods** | ERT, seismics, magnetics, and more |
| **License** | Apache 2.0 |

### SimPEG
**Alternative framework for geophysical inversion**

| Attribute | Details |
|-----------|---------|
| **URL** | https://simpeg.xyz/ |
| **Capabilities** | Modular inversion framework, multiple geophysical methods |

### ResIPy

| Attribute | Details |
|-----------|---------|
| **GitLab** | https://gitlab.com/hkex/resipy |
| **Capabilities** | 2D resistivity inversion, crosshole capable |
| **Relevance** | Directly applicable to HIRT's crosshole geometry |

### bh_tomo - Borehole Georadar Tomography

| Attribute | Details |
|-----------|---------|
| **Origin** | Ecole Polytechnique of Montreal |
| **What it does** | Borehole georadar data processing, ray-based 2D tomography |
| **License** | GNU (free software) |
| **Platform** | MATLAB |
| **Relevance** | Crosshole tomography algorithms applicable to HIRT geometry |

### WuMapPy

| Attribute | Details |
|-----------|---------|
| **Type** | Python package for sub-surface geophysical survey data processing |
| **Focus** | Ground survey data, geophysical map generation |
| **Output** | GIS-compatible maps |

### ArchaeoFusion

| Attribute | Details |
|-----------|---------|
| **Purpose** | Data fusion for archaeological ground-penetrating sensors |
| **Capabilities** | Loads multi-source data, processes, integrates into subsurface maps |
| **Relevance** | Multi-sensor fusion approach similar to HIRT's dual-modality concept |

### Geophysics-OpenSource GitHub Organization

| Attribute | Details |
|-----------|---------|
| **GitHub** | https://github.com/Geophysics-OpenSource |
| **Contains** | cuQRTM, Elastic FWI, 3D/2D FD modeling, Marchenko algorithms |
| **Focus** | Collection of geophysical open-source software |

---

## Project Comparison Matrix

| Project | Modality | Open HW | Open SW | Peer Review | Cost | Doc Quality |
|---------|----------|---------|---------|-------------|------|-------------|
| **OhmPi** | ERT | Yes | Yes | HardwareX | <$500 | Excellent |
| **OpenEIT** | EIT | Yes | Yes | Partial | ~$300 | Good |
| **PyPPM** | Magnetometry | Yes | Yes | No | ~$200 | Excellent |
| **FieldKit** | Environmental | Yes | Yes | No | Varies | Excellent |
| **Cave Pearl** | EC/Temp | Yes | Yes | MDPI | <$100 | Excellent |
| **Raspberry Shake** | Seismic | Partial | Yes | No | $175+ | Excellent |
| **Open GPR** | GPR | Yes | Yes | No | ~$500 | Good |
| **PI Metal Detectors** | Inductive | Yes | Yes | No | <$100 | Varies |

---

## Gap Analysis: What HIRT Offers

Based on this catalog, HIRT fills a unique niche:

1. **No existing dual-modality system** - No open project combines MIT + ERT
2. **Crosshole geometry** - Most ERT systems are surface-based; HIRT's crosshole approach enables true 3D tomography
3. **Archaeological/forensic focus** - Most open geophysics tools target geology or environmental monitoring
4. **Integrated probe design** - Existing systems require separate instruments for each modality

---

## Recommended Study Order

For HIRT development, study these projects in order:

1. **OhmPi** - Documentation structure, HardwareX format, ERT implementation
2. **OpenEIT** - Multi-electrode front-end, reconstruction algorithms
3. **FieldKit** - Modular hardware architecture, field ruggedization
4. **Cave Pearl** - Conductivity measurement, power optimization
5. **PI Metal Detectors** - Coil driver circuits, receive signal processing

---

## Community Resources

### Journals
- [HardwareX](https://www.sciencedirect.com/journal/hardwarex) - Primary target for HIRT publication
- [Journal of Open Hardware](https://openhardware.metajnl.com/)
- [Archaeological Prospection](https://onlinelibrary.wiley.com/journal/10990763)

### Communities
- [GOSH (Gathering for Open Science Hardware)](https://openhardware.science/)
- [Open Source Imaging](https://www.opensourceimaging.org/)
- [Hackaday.io](https://hackaday.io/)

### Forums
- [Geotech1 Forums](http://www.geotech1.com/forums/) - Metal detector engineering
- [EnviroDIY](https://github.com/EnviroDIY) - Environmental monitoring

---

## References

### Academic Papers
- Clement, R. et al. (2020). "OhmPi: An open source data logger for dedicated applications of electrical resistivity imaging." *HardwareX*, 8, e00122.
- Rucker, C. et al. (2017). "pyGIMLi: An open-source library for modelling and inversion in geophysics." *Computers & Geosciences*, 109, 106-123.
- Beddows, P.A. & Mallon, E.K. (2018). "Cave Pearl Data Logger: A Flexible Arduino-Based Logging Platform for Long-Term Monitoring in Harsh Environments." *Sensors*, 18(2), 530.

### Documentation Sites
- OhmPi: https://ohmpi.org/
- pyGIMLi: https://www.pygimli.org/
- FieldKit: https://www.fieldkit.org/
- OpenEIT: https://github.com/OpenEIT

---

*Last updated: 2026-02-01*
