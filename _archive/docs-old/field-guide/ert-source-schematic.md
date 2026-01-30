# ERT Current Source Schematic

## Overview

This document describes the design of a simple, field-ready constant-current source for ERT (Electrical Resistivity Tomography) measurements.

## Requirements

- **Current Range:** 0.5–2 mA (adjustable)
- **Accuracy:** ±1% or better
- **Polarity Reversal:** Capable of reversing current direction
- **Current Monitoring:** Measure actual current output
- **Stability:** Stable over temperature and time
- **Safety:** Current limiting, overvoltage protection

## Design Approach

### Option 1: Op-Amp Based Current Source

**Components:**
- Precision op-amp (e.g., OP177, LT1012)
- Precision voltage reference (e.g., REF5025, 2.5V)
- Current sense resistor (precision, 1% or better)
- Polarity switch (relay or solid-state)
- Current monitoring (ADC or meter)

**Circuit Concept:**
```
Vref → Op-amp → Current Sense Resistor → Output
                ↓
            Feedback to op-amp
```

**Advantages:**
- Simple design
- Good accuracy
- Easy to adjust
- Low cost

### Option 2: Dedicated Current Source IC

**Components:**
- Current source IC (e.g., LM334, REF200)
- Voltage reference
- Polarity switch
- Current monitoring

**Advantages:**
- Very stable
- Temperature compensated
- Lower component count

## Detailed Design (Op-Amp Based)

### Voltage Reference
- **IC:** REF5025 (2.5V precision reference)
- **Accuracy:** ±0.05%
- **Temperature Coefficient:** 5 ppm/°C

### Current Source Op-Amp
- **IC:** OP177 or LT1012 (precision op-amp)
- **Offset Voltage:** <10 µV
- **Input Bias Current:** <2 nA

### Current Sense Resistor
- **Value:** 1 kΩ (for 1 mA = 1V drop)
- **Tolerance:** 0.1% or better
- **Power Rating:** 1/4 W sufficient
- **Type:** Metal film precision resistor

### Current Calculation
- I = Vref / Rsense
- For Vref = 2.5V, Rsense = 1.25 kΩ → I = 2 mA
- For Vref = 2.5V, Rsense = 2.5 kΩ → I = 1 mA
- For Vref = 2.5V, Rsense = 5 kΩ → I = 0.5 mA

### Adjustability
- Use potentiometer or digital potentiometer
- Or switch between fixed resistors
- Or use DAC for voltage reference control

## Polarity Reversal

### Relay-Based (Simple)
- **Component:** DPDT relay
- **Control:** MCU or manual switch
- **Switching:** Every 1–2 seconds
- **Advantages:** Simple, reliable
- **Disadvantages:** Mechanical wear, slower switching

### Solid-State (Advanced)
- **Component:** H-bridge or analog switch
- **Control:** MCU
- **Switching:** Fast, no wear
- **Advantages:** Fast, no moving parts
- **Disadvantages:** More complex, requires drive circuit

## Current Monitoring

### ADC Measurement
- Measure voltage across sense resistor
- Calculate: I = V / Rsense
- Use 24-bit ADC (ADS1256) for precision
- Log current with each measurement

### Analog Meter (Optional)
- Simple voltmeter across sense resistor
- Scale: 1V = 1 mA (if Rsense = 1 kΩ)
- For field verification

## Safety Features

### Current Limiting
- Maximum current limit (e.g., 5 mA)
- Prevents damage to probes or soil
- Use current-limiting circuit or software limit

### Overvoltage Protection
- Limit maximum output voltage
- Protect against open-circuit conditions
- Use zener diodes or clamp circuit

### Fuse Protection
- Fuse in series with output
- Rating: 100–500 mA (safety margin)
- Protects against shorts

## PCB Layout Considerations

- Keep sense resistor close to op-amp
- Minimize trace resistance
- Use ground plane
- Shield sensitive nodes
- Keep power supply clean (filtered)

## Calibration

### Procedure
1. Set reference voltage accurately
2. Measure sense resistor precisely
3. Verify current output with precision ammeter
4. Adjust if needed (trim pot or software)
5. Document calibration values

### Verification
- Test at multiple current levels
- Verify polarity reversal
- Check stability over time
- Test at different temperatures (if possible)

## Bill of Materials

| Component | Part Number | Quantity | Notes |
|-----------|-------------|----------|-------|
| Op-amp | OP177GPZ | 1 | Precision op-amp |
| Voltage reference | REF5025AIDGKT | 1 | 2.5V reference |
| Sense resistor | 1.25 kΩ, 0.1% | 1 | For 2 mA |
| Polarity switch | DPDT relay | 1 | Or solid-state |
| Current monitor | ADC input | 1 | Or voltmeter |
| Protection | Fuse, zener | 1 set | Safety |

## Implementation Notes

- Can be built on perfboard or custom PCB
- Integrate into base hub design
- Consider remote control via MCU
- Add status LEDs for current/polarity
- Include test points for debugging

## Field Testing

- Verify current accuracy with precision meter
- Test polarity reversal timing
- Check stability during long measurements
- Verify current limiting works
- Test with actual probe load

## References

- Application notes for op-amp current sources
- Precision voltage reference datasheets
- ERT measurement standards (if applicable)

