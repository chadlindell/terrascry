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

1. Connect all Trunk cables to Central Hub.
2. Connect Probes to Zone Boxes.
3. Turn ON base hub.
4. Wait 15 sec for init.
5. Verify Central Hub "Ready" LED.
6. Run System Continuity Check (Software Diagnostic).
7. Start measurement.

---

## Status LEDs (Central Hub Only)

| LED | Solid | Blink | Off |
|-----|-------|-------|-----|
| PWR | OK | Low batt | No power |
| TX | Active | Scanning | Idle |
| LOG | Log Active | SD Error | No Log |
| ERR | System Fault | Port Warning | OK |

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

