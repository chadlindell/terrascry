# Probe BOM (Bill of Materials) - Modular Micro-Probe Design

**Per-Probe Components** - Estimated cost: $30–55 per probe (passive design)

## Design Note

**Micro-Probe Architecture:** Probes are **passive** - no electronics downhole. Electronics are centralized at surface (see [Base Hub BOM](base-hub-bom.md)). This reduces per-probe cost and complexity.
**Modular Design:** Probes are assembled from 16mm OD fiberglass segments and 3D-printed sensor modules.

## Mechanical Components

| Component | Description | Quantity | Part Number | Supplier | Example Source (real-world) | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------------------------|-----------|------------|-------|
| Rod sections | Fiberglass tube 0.5–1.0 m, Ø16 mm OD / 12 mm ID | 2–3 | Custom | Local/McMaster | McMaster 5/8" OD fiberglass tube (or similar 16mm metric) | $6–10 | $12–30 | Cut to 0.5m or 1.0m lengths |
| Male Insert Plugs | 3D printed threaded plug | 2–3 | Custom | 3D print | Printed per `hardware/cad/openscad/modular_flush_connector.scad` | $0.50 | $1.50 | Epoxied into rod ends. **Print with 100% Infill** |
| Sensor Modules | 3D printed sensor housing (Female thread) | 2–3 | Custom | 3D print | Printed per `hardware/cad/openscad/modular_flush_connector.scad` | $1–2 | $2–6 | Houses coils/electrodes. **Print with 100% Infill** |
| Probe tip | 3D-printed nose cone (Threaded) | 1 | Custom | 3D print | Custom SCAD | $1 | $1 | Screws into bottom rod |
| O-rings | Seals for threaded joints | 2–4 | AS568-014 | McMaster | Buna-N O-ring for 12mm thread seal | $0.10 | $0.40 | Weatherproofing joints |
| Epoxy | 2-part structural epoxy | 1 | Various | Hardware store | Loctite Marine Epoxy | $5 | $5 | For bonding inserts to rods |
| **Subtotal Mechanical** | | | | | | | **$21.90–43.90** | |

## Manufacturing Tools (One-time purchase)

| Tool | Description | Quantity | Supplier | Cost | Notes |
|------|-------------|----------|----------|------|-------|
| Tap | M12×1.75 Plug Tap | 1 | Amazon/McMaster | $10–15 | To cut threads in female modules |
| Die | M12×1.75 Round Die | 1 | Amazon/McMaster | $10–15 | To cut threads on male plugs |
| Die Stock / Tap Handle | Handle for tools | 1 set | Amazon | $15–20 | |

## MIT-3D Components (Integrated into Modules)

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Ferrite cores | Ø6–8 mm × 40 mm | 2 | Custom | Mouser/Digi-Key | $2 | $4 | Inserted into sensor module body |
| Magnet wire | 34–38 AWG enameled | 1 | MW-34 | Digi-Key | $5 | $5 | Wound on module body |
| **Subtotal MIT** | | | | | | **$9** | |

## ERT Components (Integrated into Modules)

| Component | Description | Quantity | Part Number | Supplier | Unit Cost | Total Cost | Notes |
|-----------|-------------|----------|-------------|----------|-----------|------------|-------|
| Electrodes | Copper/SS tape or wire | 2–3 | Custom | McMaster | $1 | $2–3 | Wrapped in grooves on sensor module |
| Wire | Thin twisted pair | 3–5m | Custom | Digi-Key | $2 | $2 | Runs through center of rod |
| **Subtotal ERT** | | | | | | **$4–5** | |

## Total per Probe

| Category | Cost Range |
|----------|------------|
| Mechanical | $22–44 |
| MIT Components | $9 |
| ERT Components | $4–5 |
| **Total** | **$35–58** |

**Typical cost:** ~$45 per probe

## Procurement Tips
- **Rod Stock:** Buy standard 5/8" or 16mm fiberglass tubing in bulk lengths (e.g., 8ft or 2m) and cut to size.
- **3D Printing:** This design relies heavily on printed parts. Use PETG or ASA for strength and outdoor durability. **Set slicer to 100% Infill.**
- **O-rings:** Don't skip these. They keep groundwater out of the wiring channel.
