# Open Source Scientific Hardware Documentation Research

**Research Date:** 2026-01-30
**Purpose:** Establish documentation standards and terminology for HIRT project

---

## Key Finding: "Whitepaper" is Wrong Terminology

In open-source scientific hardware (OSH), "whitepaper" implies a sales/marketing document. The community uses different terms:

| Term | Use Case | Example |
|------|----------|---------|
| **Technical Manual / Instrument Handbook** | Living documentation (build, operate, maintain) | OhmPi Documentation |
| **Design Article / Hardware Article** | Peer-reviewed publication (HardwareX) | OhmPi HardwareX paper |
| **Methodology Paper** | Application-focused publication | Archaeological Prospection journal |

**Recommendation for HIRT:**
- Rename "whitepaper" to **"HIRT Technical Manual"** or **"HIRT Instrument Handbook"**
- Target a **HardwareX Design Article** as the citable academic record

---

## Comparable Open Source Projects

### Direct Comparisons (Geophysics Instrumentation)

#### OhmPi - Open Source Resistivity Meter
- **URL:** https://ohmpi.org/
- **GitHub:** https://gitlab.irstea.fr/reversaal/ohmpi
- **Publication:** [HardwareX 2020](https://doi.org/10.1016/j.ohx.2020.e00122)
- **Cost:** <$500 for 32 electrodes
- **Documentation Structure:**
  1. Introduction / System Overview
  2. Build OhmPi Boards (electronic design, board specs, versions)
  3. Build OhmPi Systems (assembly guides by configuration)
  4. Operate OhmPi (software, installation, API reference)
  5. Troubleshooting & Examples
- **Why It's Relevant:** Almost identical use case (ERT), published in HardwareX, excellent documentation

#### OpenEIT - Electrical Impedance Tomography
- **URL:** https://openeit.github.io/
- **GitHub:** https://github.com/OpenEIT/OpenEIT
- **Documentation Structure:**
  - Separate repos for: Dashboard (Python), Firmware (C), PCB Design, Algorithms (pyEIT)
  - Tutorials organized by: Time Series, Bioimpedance Spectroscopy, EIT
  - Multiple reconstruction algorithms documented (Back Projection, Gauss-Newton, Graz Consensus)
- **Why It's Relevant:** Tomographic imaging, similar signal processing challenges

### Software Libraries (Inversion & Modeling)

#### pyGIMLi
- **URL:** https://www.pygimli.org/
- **GitHub:** https://github.com/gimli-org/pyGIMLi
- **Publication:** [Computers & Geosciences 2017](https://doi.org/10.1016/j.cageo.2017.07.011)
- **Documentation Structure:**
  - Home (showcase of 40+ publications using it)
  - Tutorials (progressive: mesh basics → FEM → inversion)
  - Examples (by method: ERT, seismics, magnetics)
  - API Reference
- **Why It's Relevant:** HIRT data will be inverted with pyGIMLi/SimPEG

#### SimPEG
- **URL:** https://simpeg.xyz/
- **GitHub:** https://github.com/simpeg/simpeg
- **Why It's Relevant:** Alternative inversion framework, excellent documentation patterns

### Related Hardware Projects

#### OpenFlexure Microscope
- **URL:** https://openflexure.org/
- **Why It's Relevant:** Gold standard for documenting complex mechanical+optical assemblies

#### Red Pitaya
- **URL:** https://redpitaya.com/
- **Documentation:** https://redpitaya.readthedocs.io/
- **Why It's Relevant:** Professional open-source test equipment documentation model

#### Geotech1 Forums (Metal Detection)
- **URL:** http://www.geotech1.com/forums/
- **Why It's Relevant:** Engineering-level discussion of coil design, pulse induction physics

---

## Documentation Architecture: The "Onion Model"

Successful OSH projects use layered documentation for different audiences:

### Layer 1: Field Operations (The User)
- Quick Start Guide
- Safety procedures
- Data acquisition protocols
- Troubleshooting

### Layer 2: Build Guide (The Maker)
- Bill of Materials (interactive HTML tables)
- Electronics assembly (PCB ordering, soldering)
- Mechanical assembly (waterproofing, cable management)
- Testing & verification procedures

### Layer 3: Theory Guide (The Scientist)
- Physics principles (MIT + ERT fusion)
- Forward modeling & inversion
- Calibration methodology
- Uncertainty analysis

### Layer 4: Developer Guide (The Contributor)
- API reference
- Firmware architecture
- Communication protocols
- Contribution guidelines

---

## Target Journals for HIRT

### Primary: HardwareX (Elsevier)
- **URL:** https://www.sciencedirect.com/journal/hardwarex
- **Guide for Authors:** https://www.sciencedirect.com/journal/hardwarex/publish/guide-for-authors
- **Focus:** Open source scientific hardware with demonstrated application
- **Required Sections:**
  - Hardware description
  - Design files (must be openly available)
  - Bill of Materials
  - Build instructions
  - Operation instructions
  - Validation and characterization
- **Review:** Single-blind peer review
- **Why First:** Establishes academic credibility, forces rigorous documentation

### Secondary: Archaeological Prospection
- **URL:** https://onlinelibrary.wiley.com/journal/10990763
- **Focus:** Application-focused, methodology papers
- **Content:** Less circuit diagrams, more tomography results vs. excavation data
- **Why:** Demonstrates real-world archaeological value

### Tertiary: Journal of Open Hardware
- **URL:** https://openhardware.metajnl.com/
- **Focus:** Broader open hardware community
- **Why:** Additional visibility in maker/citizen science space

### Other Options
- **Sensors (MDPI):** Novel sensor configurations, SNR analysis
- **Measurement Science and Technology (IOP):** Engineering focus
- **Journal of Applied Geophysics:** Field methodology

---

## Recommended Documentation Platform

### OhmPi Model (Recommended)
- **Primary:** Read the Docs / MkDocs site linked to repository
- **Supplements:**
  - GitHub Wiki for community contributions
  - HardwareX paper as citable design document

### Structure for HIRT
```
hirt-docs/              # Dedicated documentation site
├── index.md            # Project overview
├── getting-started/    # Quick start, safety
├── theory/             # Physics, inversion, calibration
├── build-guide/
│   ├── electronics/    # PCB assembly, testing
│   ├── mechanical/     # Probe construction
│   └── integration/    # System assembly
├── field-operations/   # Deployment, data acquisition
├── data-processing/    # pyGIMLi integration, interpretation
├── troubleshooting/
├── api-reference/      # Firmware, software APIs
└── contributing/       # How to contribute
```

---

## Action Items for HIRT

1. **Rename "whitepaper" directory** → Consider "docs" or "manual"
2. **Restructure for dual audiences:**
   - Technical Manual (living documentation)
   - HardwareX Article (peer-reviewed design document)
3. **Add HardwareX-required sections:**
   - Validation & characterization (compare to commercial instrument)
   - Complete design files availability statement
4. **Create comparison table** showing HIRT vs. OhmPi vs. commercial alternatives
5. **Plan validation experiments** for HardwareX submission

---

## References

### Published Papers
- Clement, R. et al. (2020). "OhmPi: An open source data logger for dedicated applications of electrical resistivity imaging." *HardwareX*, 8, e00122. https://doi.org/10.1016/j.ohx.2020.e00122
- Rücker, C. et al. (2017). "pyGIMLi: An open-source library for modelling and inversion in geophysics." *Computers & Geosciences*, 109, 106-123. https://doi.org/10.1016/j.cageo.2017.07.011

### Documentation Sites
- OhmPi Documentation: https://ohmpi.org/
- pyGIMLi Documentation: https://www.pygimli.org/
- OpenEIT Documentation: https://openeit.github.io/
- OpenFlexure Documentation: https://openflexure.org/

### Journals
- HardwareX: https://www.sciencedirect.com/journal/hardwarex
- Journal of Open Hardware: https://openhardware.metajnl.com/
- Archaeological Prospection: https://onlinelibrary.wiley.com/journal/10990763

### Community Forums
- Geotech1 Forums: http://www.geotech1.com/forums/

---

## Additional Projects (Grok-4 & DeepSeek Research)

### Metal Detectors with Serious Documentation
| Project | URL | Notes |
|---------|-----|-------|
| Smart Metal Detector (SMD) | https://github.com/alex4ip/Smart-Metal-Detector | Raspberry Pi-based, pulse induction, full schematics |
| Open Metal Detector | https://github.com/AlexSidorov/Open-Metal-Detector | Arduino-based, PCB designs, discrimination features |

### DIY GPR Projects
| Project | URL | Notes |
|---------|-----|-------|
| OpenGPR | https://github.com/OpenGPR/OpenGPR | SDR-based (HackRF), antenna designs, Python processing |
| Ettus USRP GPR Examples | https://kb.ettus.com/Examples/GPR | MATLAB/Python scripts for tomography |

### Environmental & Field Sensing
| Project | URL | Notes |
|---------|-----|-------|
| FieldKit | https://fieldkit.org/docs/ | Modular environmental sensing, excellent docs |
| SeisComP | https://www.seiscomp.de/doc/ | Seismology processing, operational manuals |
| ArduPilot Geophysics | https://ardupilot.org/ | Drone-based magnetometry/ERT for archaeology |

### Inversion & Processing Software
| Project | URL | Notes |
|---------|-----|-------|
| ResIPy | https://gitlab.com/hkex/resipy | Python 2D resistivity inversion, crosshole capable |
| pyGIMLi Borehole Examples | https://www.pygimli.org/examples.html#borehole | Directly applicable to HIRT |

---

## HardwareX Submission Requirements (Detailed)

Based on [HardwareX Guide for Authors](https://www.elsevier.com/journals/hardwarex/2468-0672/guide-for-authors):

### Required Sections
1. **Abstract** (200-300 words)
2. **Keywords** (4-6 terms)
3. **Design Files Summary** - List all files with GitHub repository links
4. **Bill of Materials (BOM)** - Detailed cost breakdown, suppliers, quantities
5. **Build Instructions** - Step-by-step with photos/videos
6. **Operation Instructions** - Usage guide, safety notes
7. **Validation and Characterization** - Experimental results, comparisons to commercial tools
8. **Discussion** - Limitations, future work
9. **References** (20-50 citations)

### Success Factors
- **Novelty:** Emphasize HIRT's hybrid MIT-ERT crosshole integration
- **Reproducibility:** All files CC-BY licensed, include validation scripts
- **Impact:** Quantify cost savings (e.g., <$500 vs $10k commercial)
- **Completeness:** Incomplete BOMs or untested designs lead to rejection
- **Length:** 10-20 pages typical; peer review takes 2-4 months

### Example Papers to Study
- OhmPi: https://doi.org/10.1016/j.ohx.2020.e00122
- OpenEIT: https://doi.org/10.3389/fphy.2015.00015
- pyGIMLi: https://doi.org/10.5194/gmd-10-2861-2017

---

## Documentation Hosting Platforms

| Platform | Pros | Used By |
|----------|------|---------|
| **ReadTheDocs** | Auto-builds from repo, versioning, search | OhmPi, pyGIMLi |
| **GitHub Pages** | Free, simple, Markdown/Jekyll | OpenEIT, Arduino |
| **MkDocs** | Material theme, clean design | Newer projects |
| **Sphinx** | Best for API docs, Python integration | Scientific libraries |

**Recommendation for HIRT:** ReadTheDocs with MkDocs Material theme, mirroring OhmPi's structure.

---

## Consensus Across Models

All three AI models (Gemini 3 Pro, Grok-4, DeepSeek R1) agreed on:

1. **"Whitepaper" is wrong terminology** - Use "Documentation" or "Technical Manual"
2. **HardwareX is the primary target journal** for peer-reviewed credibility
3. **OhmPi is the closest comparable project** - Study their structure
4. **Modular documentation** works best - Separate theory/build/operate/interpret
5. **ReadTheDocs or GitHub Pages** for hosting
6. **Validation section is critical** - Compare against commercial instruments
