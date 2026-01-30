# Measurement Geometry Diagrams

## Overview

Visual representations of probe deployment geometry, measurement patterns, and depth sensitivity for the HIRT system.

## Standard Grid Layout (10×10 m Section)

```
Top View - 10×10 m Section with 2 m Spacing

    0m    2m    4m    6m    8m   10m
  ┌─────┬─────┬─────┬─────┬─────┐
0m│  P01 │  P02 │  P03 │  P04 │  P05 │
  ├─────┼─────┼─────┼─────┼─────┤
2m│  P06 │  P07 │  P08 │  P09 │  P10 │
  ├─────┼─────┼─────┼─────┼─────┤
4m│  P11 │  P12 │  P13 │  P14 │  P15 │
  ├─────┼─────┼─────┼─────┼─────┤
6m│  P16 │  P17 │  P18 │  P19 │  P20 │
  └─────┴─────┴─────┴─────┴─────┘

Probe Count: 20 probes (4×5 grid)
Spacing: 2 m
Section Size: 10×10 m
```

## MIT Measurement Patterns

### TX→RX Measurement Paths

```
Example: P01 as TX, measuring to all RX probes

     P01 (TX) ────────────────→ P05 (RX)
        │                           │
        │                           │
        ├──────────→ P10 (RX)        │
        │                           │
        ├──────────→ P15 (RX)        │
        │                           │
        └──────────→ P20 (RX)        │
                                    │
     P01 → P02 (short baseline)     │
     P01 → P03 (medium baseline)     │
     P01 → P04 (long baseline)      │
     P01 → P05 (longest baseline) ──┘

Each probe takes turn as TX; all others record as RX
Total measurements: 20 TX × 19 RX = 380 pairs per frequency
```

### Crosshole Measurement Geometry

```
Side View - Crosshole Measurement Paths

Surface ────────────────────────────────
         │                               │
         │  P01 (TX)                    │  P05 (RX)
         │    │                          │    │
         │    │                          │    │
   1m ───┼────┼──────────────────────────┼────┼───
         │    │                          │    │
         │    │  ──── Magnetic Field ────│    │
         │    │         (attenuated)      │    │
         │    │                          │    │
   2m ───┼────┼──────────────────────────┼────┼───
         │    │                          │    │
         │    │    ┌─────────────────┐   │    │
         │    │    │  Conductive     │   │    │
         │    │    │  Target (metal)  │   │    │
         │    │    └─────────────────┘   │    │
   3m ───┼────┼──────────────────────────┼────┼───
         │    │                          │    │
         │    │                          │    │
         │    │                          │    │
   4m ───┴────┴──────────────────────────┴────┴───

Depth: Probes inserted to 3 m
Sensitivity: Extends to ~4-5 m depth
```

## ERT Measurement Patterns

### Current Injection Patterns

```
Pattern 1: Corner-to-Corner (Long Baseline)

  P01 (+) ──────────────────────────────── P20 (-)
    │                                         │
    │                                         │
    │         Voltage measured at all        │
    │         other probes (P02-P19)          │
    │                                         │
    └─────────────────────────────────────────┘

Pattern 2: Edge-to-Edge

  P05 (+) ──────────────────────────────── P16 (-)
    │                                         │
    │         Voltage measured at all        │
    │         other probes                   │
    │                                         │
    └─────────────────────────────────────────┘

Pattern 3: Center-to-Edge

        P13 (+) ──────────────── P01 (-)
            │                         │
            │    Voltage measured     │
            │    at all other probes   │
            │                         │
            └─────────────────────────┘

Multiple injection pairs provide redundancy and depth coverage
```

### ERT Depth Sensitivity

```
Side View - ERT Current Paths

Surface ────────────────────────────────
         │                               │
         │  P01 (+)                      │  P20 (-)
         │    │                          │    │
         │    │                          │    │
   1m ───┼────┼──────────────────────────┼────┼───
         │    │                          │    │
         │    │  ╱─── Current Path ────╲ │    │
         │    │ ╱                         ╲   │
         │    ││                           ││ │
   2m ───┼────┼│───────────────────────────│┼─┼───
         │    ││                           ││ │
         │    │ ╲                         ╱ │    │
         │    │  ╲─── Current Path ────╱  │    │
         │    │                          │    │
   3m ───┼────┼──────────────────────────┼────┼───
         │    │                          │    │
         │    │                          │    │
   4m ───┴────┴──────────────────────────┴────┴───

Current flows through volume between injection electrodes
Sensitivity extends deeper with longer baselines
```

## Depth Sensitivity Visualization

### Sensitivity Volume (Rule of Thumb)

```
3D View - Sensitivity Volume

        ┌─────────────────────────┐
        │                         │
        │   High Sensitivity      │
        │   (Near probes)         │
        │                         │
        ├─────────────────────────┤
        │                         │
        │   Medium Sensitivity    │
        │   (Mid-depth)           │
        │                         │
        ├─────────────────────────┤
        │                         │
        │   Lower Sensitivity    │
        │   (Deep, far from probes)│
        │                         │
        └─────────────────────────┘

With 3 m probes and 2 m spacing:
- High sensitivity: 0-2 m depth
- Medium sensitivity: 2-4 m depth  
- Lower sensitivity: 4-6 m depth
```

## Scenario-Specific Layouts

### Woods Burials (8×8 m, 1.5 m spacing)

```
    0m   1.5m  3m   4.5m  6m   7.5m
  ┌────┬────┬────┬────┬────┐
0m│ P01│ P02│ P03│ P04│ P05│
  ├────┼────┼────┼────┼────┤
1.5│ P06│ P07│ P08│ P09│ P10│
  ├────┼────┼────┼────┼────┤
3m│ P11│ P12│ P13│ P14│ P15│
  └────┴────┴────┴────┴────┘

Probe Count: 12-15 probes
Spacing: 1-1.5 m
Depth Target: 0-2 m
Rod Length: 1.6 m
```

### Bomb Crater (15×15 m, 2 m spacing)

```
    0m    2m    4m    6m    8m   10m   12m   14m
  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐
0m│ P01 │ P02 │ P03 │ P04 │ P05 │ P06 │ P07 │
  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
2m│ P08 │ P09 │ P10 │ P11 │ P12 │ P13 │ P14 │
  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
4m│ P15 │ P16 │ P17 │ P18 │ P19 │ P20 │ P21 │
  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤
6m│ P22 │ P23 │ P24 │ P25 │ P26 │ P27 │ P28 │
  └─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Probe Count: 28 probes (4×7 grid) or multiple sections
Spacing: 2 m
Depth Target: 0-4+ m
Rod Length: 3.0 m
Covers crater + rim
```

### Rim-Only Deployment (Minimal Intrusion)

```
Top View - Rim Deployment Around Crater

        P01 ──────────── P02
       ╱                    ╲
      │                      │
   P03│                      │P04
      │                      │
      │      Crater Area      │
      │      (not probed)     │
      │                      │
   P05│                      │P06
      │                      │
       ╲                    ╱
        P07 ──────────── P08

Probe Count: 8-12 probes
Spacing: Variable (2-4 m)
Approach: Perimeter only, angled inward
```

## Measurement Sequence

### MIT Sweep Sequence

```
For each frequency (2, 5, 10, 20, 50 kHz):

1. Set P01 = TX
   - Measure P01 → P02, P03, ..., P20
   - Record amplitude and phase

2. Set P02 = TX
   - Measure P02 → P01, P03, ..., P20
   - Record amplitude and phase

3. Repeat for P03, P04, ..., P20

Total: 20 TX × 19 RX × 5 frequencies = 1,900 measurements
```

### ERT Pattern Sequence

```
Pattern 1: P01 (+) → P20 (-)
  - Measure voltage at P02-P19
  - Reverse polarity
  - Repeat measurement

Pattern 2: P05 (+) → P16 (-)
  - Measure voltage at all other probes
  - Reverse polarity
  - Repeat

Pattern 3: P13 (+) → P01 (-)
  - Measure voltage at all other probes
  - Reverse polarity
  - Repeat

Total: 3 patterns × 2 polarities × ~18 measurements = ~108 ERT measurements
```

## Overlap Between Sections

```
Section 1                    Section 2
┌─────────────┐            ┌─────────────┐
│  P01  P02   │            │  P02  P03   │
│  P06  P07   │            │  P07  P08   │
│  P11  P12   │            │  P12  P13   │
│  P16  P17   │            │  P17  P18   │
└─────────────┘            └─────────────┘
      │                            │
      └──────── Overlap ───────────┘
      (One column shared for continuity)

Benefits:
- Ensures data continuity
- Provides redundancy
- Helps with quality control
```

## Notes

- **Lateral Resolution:** Approximately 0.5-1.5 × spacing
- **Depth Resolution:** Best near probes, decreases with depth
- **Measurement Time:** ~2-3 hours per section (including setup/extraction)
- **Data Volume:** ~2,000 measurements per section (MIT + ERT)

