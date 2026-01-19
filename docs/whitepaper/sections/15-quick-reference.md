# 15. Quick Reference Card

## HIRT Field Quick Reference

*Print this section as a laminated card for field use.*

---

## Grid Layout (Standard 10x10 m)

```
Spacing: 2 meters
Probes: 25 (5x5 grid)
Depth: 2.5-3 m

    0m    2m    4m    6m    8m
 0m  P01   P02   P03   P04   P05
 2m  P06   P07   P08   P09   P10
 4m  P11   P12   P13   P14   P15
 6m  P16   P17   P18   P19   P20
 8m  P21   P22   P23   P24   P25
```

---

## Key Parameters

### MIT (Magnetic Induction)
| Parameter | Value |
|-----------|-------|
| Frequency Range | 2-50 kHz |
| TX Current | 10-50 mA |
| Integration Time | 1-5 sec |

### ERT (Resistivity)
| Parameter | Value |
|-----------|-------|
| Injection Current | 0.5-2 mA |
| Ring Positions | 0.5, 1.5, 2.5 m |
| Polarity Reversal | Every 2 sec |

---

## Power-Up Sequence

1. Connect all probe cables
2. Turn ON base hub
3. Wait 15 sec for init
4. Run probe scan
5. Verify all probes green
6. Start measurement

---

## Shutdown Sequence

1. Complete final measurement
2. Save data to backup
3. Power OFF base hub
4. Disconnect cables
5. Extract probes (straight pull)
6. Fill holes per permit

---

## Troubleshooting

| Symptom | Check | Fix |
|---------|-------|-----|
| Probe offline | Cable connection | Reseat/swap cable |
| Noisy data | EMI sources | Move away, shield |
| No ERT signal | Ring contact | Add water to soil |
| Drift | Temperature | Wait 10 min stabilize |
| No power | Battery/fuse | Charge/replace fuse |

---

## Emergency Contacts

| Role | Name | Phone |
|------|------|-------|
| PI | _______ | _______ |
| Safety | _______ | _______ |
| Local | _______ | _______ |

---

## Coil Specifications

| Parameter | TX | RX |
|-----------|-----|-----|
| Inductance | 1-2 mH | 1-2 mH |
| Q Factor | >20 | >20 |
| DC Resist. | <10 ohm | <10 ohm |

---

## Data File Naming

`SITE_YYYYMMDD_SCAN##.dat`

Example: `ROMA_20240615_SCAN01.dat`

---

## Measurement Modes

**MIT Full Matrix:**
- All TX-RX pairs
- Time: 10-15 min
- Use: Metal detection

**ERT Wenner:**
- Sequential injection
- Time: 5-10 min
- Use: Soil resistivity

**Combined:**
- MIT + ERT interleaved
- Time: 15-20 min
- Use: Full characterization

---

## Status LEDs

| LED | Solid | Blink | Off |
|-----|-------|-------|-----|
| PWR | OK | Low batt | No power |
| TX | Active | Scanning | Idle |
| COM | Connected | Data xfer | No link |
| ERR | Fault | Warning | OK |

---

## Cable Color Code

| Wire | Function |
|------|----------|
| Red | Power + |
| Black | Power GND |
| White | TX+ |
| Green | TX- |
| Blue | RX+ |
| Yellow | RX- |
| Shield | Ground |

---

## Probe Connector Pinout

| Pin | Signal |
|-----|--------|
| 1 | TX+ |
| 2 | TX- |
| 3 | RX+ |
| 4 | RX- |
| 5 | Guard |
| 6 | Ring A |
| 7 | Ring B |
| 8 | Ring C |

---

## Soil Type Guidelines

| Soil | Insertion | ERT Contact |
|------|-----------|-------------|
| Sand | Direct push | Add water |
| Clay | Pre-drill | Good |
| Rocky | Careful auger | Variable |
| Wet | Easy | Excellent |

---

## Safety Checklist

- [ ] Site access authorized
- [ ] Utilities located/marked
- [ ] First aid kit available
- [ ] Weather checked
- [ ] Contact person informed
- [ ] Extraction plan ready

---

## Notes

_______________________________________

_______________________________________

_______________________________________

_______________________________________

---

*For detailed procedures, see Section 10: Field Operation Manual.*

