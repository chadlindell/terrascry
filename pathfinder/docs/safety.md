# Pathfinder Safety Documentation

## Why This Document Exists

Pathfinder is designed for archaeological reconnaissance, forensic site screening, and UXO pre-survey work. These environments carry real hazards — unexploded ordnance, crime scene protocols, extreme weather, and remote locations. This document addresses the safety considerations specific to operating Pathfinder in the field, with particular attention to electromagnetic compatibility near ordnance and the legal requirements for forensic applications.

For general technical details about the instrument, see [Design Concept](design-concept.md).

---

## 1. General Safety

Pathfinder is a passive magnetic sensor system. The fluxgate gradiometers measure the Earth's existing magnetic field — they do not transmit energy, generate ionizing radiation, or operate at high voltages. The 7.4 V LiPo battery is the only stored energy source in the system.

The primary hazards during Pathfinder operations are environmental, not instrument-related:

- **Terrain**: Uneven ground, concealed holes, vegetation, slopes
- **Weather**: Lightning, heat exposure, cold, rain
- **Site hazards**: UXO, contaminated soil, restricted areas
- **Ergonomic**: Sustained walking with harness-mounted equipment

The instrument itself poses negligible risk to the operator under normal use. A healthy respect for the operating environment, not the instrument, is what keeps people safe.

---

## 2. Electromagnetic Compatibility Near Ordnance

### Why This Matters

Pathfinder is frequently deployed at sites with known or suspected unexploded ordnance. Any electronic device brought onto a UXO site must be assessed for its potential to trigger electro-explosive devices (EEDs) in ordnance fuzing systems. This assessment follows HERO (Hazards of Electromagnetic Radiation to Ordnance) principles defined in MIL-HDBK-240 and NATO AASTP-1.

### Pathfinder's Electromagnetic Profile

Pathfinder is a **passive** gradiometer. The fluxgate sensors measure existing magnetic fields; they do not generate significant fields of their own. This is fundamentally different from active systems like HIRT's MIT channel, which deliberately transmits electromagnetic energy into the ground.

However, Pathfinder does contain active electronics that produce incidental electromagnetic emissions:

| Source | Frequency | Nature | Estimated Power |
|--------|-----------|--------|-----------------|
| Arduino Nano (ATmega328P) | 16 MHz crystal oscillator, digital switching harmonics | Digital clock, GPIO switching | < 1 mW radiated |
| GPS module (NEO-6M) | 1575.42 MHz L1 receiver | Passive antenna, but local oscillator generates weak emissions | < 0.1 mW |
| ADS1115 ADCs (x2) | I2C bus up to 400 kHz | Short-range digital bus, shielded by enclosure | Negligible |
| SD card module | SPI bus up to 4 MHz | Short-range digital bus, shielded by enclosure | Negligible |

All electronics are housed in an IP65 enclosure, which provides partial shielding of digital emissions.

### HERO Assessment

The HERO framework evaluates whether electromagnetic emissions from a device can induce currents in ordnance firing circuits that exceed the Maximum No-Fire Current (MNFC) threshold. The critical requirement from MIL-HDBK-240 is that induced currents must remain 16.5 dB (approximately 7:1 current ratio) below the MNFC.

**Pathfinder's assessment:**

1. **Total radiated power is extremely low.** All Pathfinder emissions combined are well under 10 mW — comparable to a digital wristwatch or fitness tracker. For reference, HERO safe distances are calculated for radar transmitters and radio equipment operating at watts to kilowatts.

2. **No intentional transmitter.** Unlike HIRT's MIT channel (which deliberately generates AC magnetic fields at 2-50 kHz to induce eddy currents), Pathfinder has no transmit function. The GPS module receives only; its local oscillator emission at 1575 MHz is far above the frequency range where EED coupling is efficient (DC to ~100 kHz).

3. **Field strength at distance.** At 1 m from the electronics enclosure, the magnetic field from Pathfinder's digital electronics is orders of magnitude below the levels needed to induce meaningful current in an EED firing circuit. The 16 MHz clock and its harmonics produce fields comparable to background urban electromagnetic noise.

4. **No low-frequency AC magnetic field.** The most dangerous frequency range for EED induction is DC to 100 kHz, where firing circuit wiring can act as an efficient coupling loop. Pathfinder produces no intentional emissions in this range. The I2C bus (400 kHz) and SPI bus (4 MHz) are both above this critical range and are extremely low power.

**Conclusion:** Pathfinder poses no credible HERO threat to any known ordnance type. **(Modeled)**

This assessment is based on the system's electromagnetic characteristics and comparison to HERO threshold levels. It has not been validated by formal HERO testing at a certified facility. For operations at sites where formal HERO certification is required, consult the site UXO authority.

### Mandatory Compliance

Regardless of this assessment, **always follow site-specific ordnance safety protocols.** If a site UXO officer requires power-down of all electronic equipment, comply immediately. The instrument assessment does not override the authority of qualified ordnance disposal personnel.

---

## 3. UXO Site Procedures

Pathfinder is a screening tool. It identifies magnetic anomalies. It does **not** determine whether those anomalies are ordnance, whether ordnance is safe, or whether ground disturbance is permissible. These determinations belong to qualified UXO/EOD professionals.

### Operating Under UXO Authority

When surveying known or suspected UXO sites:

- Always operate under the direction of a qualified UXO safety officer
- Attend the site safety briefing before commencing any survey
- Understand the emergency evacuation routes and rally points
- Carry a communication device (radio or phone) at all times
- Know the location of the nearest first aid and emergency equipment

### Walking Survey Protocol on UXO Sites

The harness-supported walking survey is Pathfinder's primary operating mode. On UXO sites, this requires additional discipline:

1. **Stay on cleared paths when possible.** If the site has been partially cleared, survey cleared areas first. Only enter uncleared areas with explicit authorization from the UXO officer.

2. **Maintain minimum probe-to-ground clearance.** The bottom sensors operate at 15-20 cm above ground level. This clearance prevents physical contact with surface or near-surface ordnance. Never lower the trapeze below the design clearance.

3. **If the instrument contacts an unknown object — STOP.** Do not attempt to identify the object. Mark the location with tape or a flag placed at a safe distance (do not place markers directly on the object). Withdraw along your approach path. Report to the UXO officer immediately.

4. **The harness design keeps hands free.** This is a deliberate safety feature. Operators can carry safety equipment (radio, marking supplies, water) without compromising instrument handling.

5. **Walk at a steady, deliberate pace.** Rushing increases the risk of tripping and falling onto surface ordnance. The pace beeper helps maintain consistent speed.

### What Pathfinder Data Does NOT Tell You

- Whether an anomaly is ordnance (it could be a nail, pipe, or geological feature)
- Whether ordnance is live, safe, or fuzed
- The depth of a buried object (gradient amplitude is a rough proxy, not a measurement)
- Whether it is safe to excavate

**Never excavate based on Pathfinder data alone.** Pathfinder anomaly maps inform where to investigate further, not where to dig.

---

## 4. Operator Health and Ergonomics

### Fatigue Management

The harness distributes the instrument's weight (~1.25 kg) across the shoulders and waist, but sustained walking over uneven terrain is physically demanding regardless of load.

- **Maximum recommended continuous survey: 2 hours**, followed by a minimum 15-minute rest break
- Hydration is critical — carry water and drink regularly, especially in hot conditions
- Sun protection (hat, sunscreen, long sleeves) for outdoor surveys
- Watch for repetitive strain from maintaining the trapeze level over extended periods. Wrist and forearm fatigue is the most common complaint with trapeze-style instruments
- Fatigue degrades data quality. A tired operator walks unevenly, introduces motion noise, and is more likely to miss survey lines. Rest breaks improve both safety and data

### Heat and Cold Exposure

Pathfinder operations expose the operator to ambient weather conditions for extended periods. Dress for the environment, not for the instrument — Pathfinder has no temperature requirements that constrain operator clothing.

**Battery performance**, however, is temperature-sensitive:

- LiPo capacity degrades significantly below 0 C
- LiPo batteries must **never be charged below 0 C** (lithium plating causes internal short circuits and fire risk)
- Above 45 C, battery degradation accelerates and thermal runaway risk increases
- In cold conditions, keep the battery warm (inside a jacket pocket) until deployment
- In hot conditions, avoid leaving the instrument in direct sun during breaks

---

## 5. Battery Safety

Pathfinder uses a 7.4 V 2S LiPo (lithium polymer) battery pack, typically 2000 mAh. LiPo batteries store significant energy for their size and require careful handling.

### Charging

- Charge only with a LiPo-compatible charger set to 2S (7.4 V) mode
- Never charge unattended — monitor for swelling, heat, or unusual odor
- Charge on a fireproof surface (concrete, ceramic tile, LiPo charge bag)
- Do not charge below 0 C ambient temperature
- Do not charge immediately after heavy use — allow the battery to cool to ambient temperature first

### Storage

- For storage longer than a few days, charge to storage voltage: 3.8 V per cell (7.6 V total)
- Store in a cool, dry location away from flammable materials
- Use a fireproof LiPo storage bag
- Check stored batteries monthly for swelling or voltage drift

### Handling

- Never puncture, crush, drop, or short-circuit the battery
- Keep away from water (short-circuit risk through contaminated water)
- Do not expose to temperatures above 60 C (e.g., inside a closed vehicle in summer)
- Inspect before each use — reject any battery that is swollen, dented, or has damaged wiring

### Transport

- Transport in a fireproof LiPo bag, especially during air travel or long drives
- Airlines have specific rules for lithium batteries — check before flying
- For vehicle transport, keep batteries in a ventilated area (not sealed in the boot/trunk in hot weather)

### Disposal

- Do not dispose of LiPo batteries in regular waste
- Discharge fully (to 0 V) before disposal, or tape terminals to prevent short circuit
- Take to an electronics recycling facility that accepts lithium batteries
- A damaged or swollen battery should be placed in a sand-filled container until it can be safely disposed of

---

## 6. Forensic and Legal Considerations

Pathfinder may be used for forensic screening — for example, identifying potential burial locations during criminal investigations, or locating disturbed ground at heritage sites. When survey data may be used as evidence or in legal proceedings, additional protocols apply.

### Chain of Custody

If Pathfinder is deployed in support of a forensic investigation:

1. **Log instrument details**: Record all serial numbers (Arduino, GPS module, ADC modules), firmware version, and most recent calibration date
2. **Record operator information**: Full name, qualifications, affiliation, and role in the investigation
3. **Document survey parameters**: Sensor height, sample rate, GPS mode, date/time, grid coordinates
4. **Preserve raw data**: Copy the original CSV files from the SD card immediately after survey. Keep originals unmodified — perform all analysis on copies
5. **Document environmental conditions**: Temperature, recent rainfall, ground conditions, nearby magnetic interference sources (fences, vehicles, buried utilities)
6. **Compute file hashes**: Generate MD5 and/or SHA-256 hashes of raw CSV files immediately after download. Record these hashes in the survey log

### Limitations as Evidence

Pathfinder results are **screening-level data**, not forensic-grade evidence. The instrument:

- Has no traceable calibration certificate (DIY build)
- Uses consumer-grade GPS (2-5 m accuracy without RTK)
- Has not undergone formal validation testing against known targets
- Produces gradient maps that indicate anomalies, not identifications

In legal proceedings, Pathfinder data is most appropriately presented as preliminary reconnaissance that guided subsequent investigation, not as standalone evidence of what lies underground.

### Permits and Legal Requirements

Geophysical surveys may require permits or permissions depending on jurisdiction:

- **Heritage sites**: Many countries require specific permits for any geophysical investigation at protected sites. Contact the relevant heritage authority (e.g., Historic England, state historic preservation offices in the US, or national equivalents)
- **Private land**: Always obtain written landowner permission before surveying
- **Crime scenes**: Operate only under direction of the investigating authority. Do not conduct independent surveys at crime scenes
- **UXO-contaminated land**: May require a Detailed Risk Assessment (DRA) and Permit to Dig even for non-intrusive surveys, depending on jurisdiction. In the UK, CIRIA C681/C785 provides the framework; other countries have equivalent regulations
- **Environmental regulations**: Some protected natural areas restrict the use of electronic equipment or restrict access during certain seasons (e.g., nesting periods)

---

## 7. Emergency Procedures

### Ordnance Discovery

If you encounter what may be unexploded ordnance during a survey:

1. **Stop immediately.** Do not touch, move, or further investigate the object
2. **Mark the location** with tape or a flag placed at a safe distance — do not place anything on or directly adjacent to the object
3. **Withdraw along your approach path.** Do not take a new route that may cross uncleared ground
4. **Notify the UXO safety officer** or site supervisor immediately
5. **Do not re-enter the area** until cleared by qualified EOD personnel

### Equipment Fire (LiPo Battery)

LiPo fires produce toxic fumes and are difficult to extinguish. If the battery catches fire:

1. **Remove the harness immediately** — use the quick-release buckles
2. **Move upwind** of the burning equipment
3. **Do not use water** — water reacts with burning lithium and can cause violent splattering
4. **Use sand or a Class D (metal fire) extinguisher** if available
5. **If no extinguisher is available, let it burn out** in a clear area away from flammable materials
6. **Do not attempt to recover equipment** until fully cooled (minimum 30 minutes)

### Medical Emergency

1. **Remove the harness** using quick-release buckles to provide clear access to the patient
2. **Administer standard first aid** as trained
3. **Call emergency services** — ensure you know the site grid reference or GPS coordinates for the ambulance
4. **Send someone to meet the ambulance** at the site access point if the survey area is remote

### Lightning

Carbon fiber and aluminum — both candidate materials for the Pathfinder trapeze — are electrically conductive. A person carrying a horizontal conductive bar in an open field is an elevated lightning target.

1. **Cease survey immediately** when lightning is observed within 10 km (roughly: if the time between flash and thunder is less than 30 seconds)
2. **Remove the harness** and set down the instrument
3. **Move to shelter** — a vehicle or substantial building, not a tree or open shelter
4. **Wait 30 minutes** after the last observed lightning before resuming operations
5. **Do not shelter under the instrument** or near it during a storm

---

## 8. Pre-Survey Safety Checklist

Complete this checklist before commencing any Pathfinder survey. Items marked with an asterisk (*) are mandatory for UXO sites.

### Site Preparation

- [ ] Site hazard assessment completed (terrain, access, known risks)
- [ ] * UXO safety officer briefed and authorization obtained (if applicable)
- [ ] * Emergency contacts recorded and shared with team
- [ ] Emergency evacuation route identified
- [ ] First aid kit available on site
- [ ] * Communication device (phone/radio) charged and tested

### Equipment

- [ ] Battery charged and visually inspected (no swelling, no damage, no exposed wiring)
- [ ] SD card inserted with sufficient free space
- [ ] GPS acquiring satellites (check fix quality before starting)
- [ ] Harness quick-release buckles tested
- [ ] Sensor drop tubes secure, no loose fittings
- [ ] Electronics enclosure sealed (check gasket if wet conditions expected)

### Operator

- [ ] * Metal check completed — remove belt buckles, keys, phones from pockets, steel-toe boots if possible. Ferrous objects on the operator produce systematic noise in gradiometer data and can mask real anomalies
- [ ] Weather forecast checked — no lightning risk, acceptable temperature range
- [ ] Adequate water and sun protection for planned survey duration
- [ ] Operator familiar with site emergency procedures
- [ ] Survey plan agreed with site supervisor or UXO officer

### Data Integrity (Forensic Sites Only)

- [ ] Instrument serial numbers and firmware version logged
- [ ] Operator name, qualifications, and role recorded
- [ ] Chain of custody forms prepared
- [ ] Hash computation procedure agreed (who, when, what algorithm)
- [ ] Raw data preservation plan in place (original SD card retained or imaged)

---

## References

- MIL-HDBK-240: Hazards of Electromagnetic Radiation to Ordnance (HERO)
- NATO AASTP-1: Manual of NATO Safety Principles for the Storage of Military Ammunition and Explosives
- CIRIA C681 (2009) / C785 (2019): Unexploded Ordnance — A Guide for the Construction Industry (UK)
- ISO/IEC 27037:2012: Guidelines for identification, collection, acquisition, and preservation of digital evidence
- ASTM E1492-23: Standard Practice for Receiving, Documenting, Storing, and Retrieving Evidence in a Forensic Science Laboratory
