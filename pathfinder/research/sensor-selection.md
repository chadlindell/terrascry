# Pathfinder Sensor Selection Research

## Overview

This document evaluates fluxgate sensor options for the Pathfinder gradiometer project, targeting 8 sensors (4 gradiometer pairs) within a budget of ~$480-640.

## Requirements

From design-concept.md:
- **Configuration**: 4 gradiometer pairs (8 total sensors)
- **Budget target**: $60-80 per sensor
- **Integration**: DIY-friendly, Arduino ADC compatible
- **Application**: Archaeological reconnaissance gradiometry

## Sensor Options Evaluated

### 1. FG Sensors FG-3+ (PRIMARY RECOMMENDATION)

**Specifications:**
- Measurement range: +/-50 uT (+/-50,000 nT)
- Output type: Digital 5V square-wave (frequency proportional to field)
- Frequency range: ~50 KHz to ~120 KHz
- Sensitivity: ~1.6 nT/mV (after F-V conversion)
- Power: 5V @ 12 mA

**Pricing:**
- Direct from FG Sensors: EUR 49.50 (~$52-54 USD)
- SparkFun/distributors: $52-60 USD
- **8 sensors: $416-480 USD**

**Pros:**
- Proven track record (replacement for legacy FGM-1/FGM-3)
- Available from multiple distributors
- DIY gradiometer kit available demonstrates feasibility
- Free Android app for testing/calibration
- Adequate sensitivity for archaeological targets

**Cons:**
- Frequency output requires LM2917 frequency-to-voltage conversion
- More complex signal conditioning than direct analog

**Sources:**
- https://www.fgsensors.com/product-page/fg-3
- https://www.sparkfun.com/fgsensors-fg-3-sensor.html

### 2. Magnetometer-Kit.com FGM-3 PRO

**Specifications:**
- Measurement range: +/-50 uT
- Output type: Frequency (similar to FG-3+)
- Matched pairs available

**Pricing:**
- Single sensor: EUR 60 (~$65 USD)
- Matched pair: EUR 120 (~$130 USD)
- Complete gradiometer kit: EUR 199 (~$215 USD)
- **8 sensors: ~$520 USD**

**Pros:**
- Complete gradiometer kit provides reference design
- Matched pairs for better gradient measurement

**Cons:**
- 20% more expensive than FG-3+
- Single source supplier
- Limited documentation

**Source:** https://magnetometer-kit.com/

### 3. Bartington Mag-03 Series

**Specifications:**
- Measurement range: +/-100 uT
- Noise floor: 6 pT/sqrt(Hz) @ 1 Hz (extremely low)
- Output: Analog voltage (+/-10V)
- 3-axis configuration

**Pricing:**
- Estimated: $800-1500+ per sensor
- **8 sensors: $6,400-12,000 USD**

**Assessment:** NOT VIABLE - far exceeds budget (10-20x over)

**Source:** https://www.bartington.com/products/mag-03/

### 4. Stefan Mayer Instruments (FLC-100)

**Specifications:**
- Miniature sensor, low power
- Suited for multi-sensor arrays
- Sub-nanotesla resolution available

**Pricing:**
- Not publicly available
- Estimated: $200-500 per sensor
- **8 sensors: $1,600-4,000 USD (estimated)**

**Assessment:** MARGINAL - likely exceeds budget

**Source:** https://stefan-mayer.com/en/products/magnetometers-and-sensors/

### 5. Chinese/Alibaba Alternatives

**Availability:** Multiple suppliers on Alibaba

**Pricing:**
- Estimated: $20-80 per sensor
- **8 sensors: $160-640 USD**

**Assessment:** HIGH RISK
- Quality highly variable
- Poor/absent documentation
- No community support
- Unknown performance characteristics

### 6. DIY Homemade Fluxgate

**Materials Cost:** $10-30 per sensor
**Time Investment:** 4-8 hours per sensor
**8 sensors: $80-240 USD + 32-64 hours**

**Assessment:** Viable for extremely budget-constrained projects but time-intensive

## Comparison Table

| Sensor | Price/Unit | Total (8) | Output | DIY-Friendly | Recommendation |
|--------|------------|-----------|--------|--------------|----------------|
| **FG-3+** | $52-54 | $416-432 | Frequency | Good | **PRIMARY** |
| FGM-3 PRO | $65 | $520 | Frequency | Good | Secondary |
| Bartington | $800-1500 | $6,400+ | Analog | Easy | Not viable |
| Stefan Mayer | $200-500 | $1,600+ | Unknown | Unknown | Not viable |
| Chinese | $20-80 | $160-640 | Unknown | Poor | High risk |
| DIY | $10-30 | $80-240 | Analog | Poor | Time intensive |

## Recommendation

**PRIMARY: FG Sensors FG-3+**

**Rationale:**
1. Budget fit: $416-432 for 8 sensors (within $480-640 target)
2. Proven performance in archaeological applications
3. Multiple distributors ensure availability
4. DIY gradiometer kit validates feasibility
5. Adequate sensitivity (~1.6 nT/mV) for target applications

**Implementation Path:**
1. Purchase 2x FG-3+ sensors (~$105) for prototype
2. Design/test LM2917 frequency-to-voltage converter
3. Validate ADS1115 ADC integration
4. Scale to 8 sensors once signal chain validated

## Signal Conditioning Note

The FG-3+ outputs a frequency signal (not voltage), requiring conversion:

```
FG-3+ (freq) --> LM2917 F-to-V --> ADS1115 ADC --> Arduino
```

The LM2917 is a well-documented frequency-to-voltage converter (~$2-3 each). Total additional cost for 8 converters: ~$20-30.

## References

- FG Sensors: https://www.fgsensors.com
- SparkFun FG-3+: https://www.sparkfun.com/fgsensors-fg-3-sensor.html
- Magnetometer Kit: https://magnetometer-kit.com/
- Bartington: https://www.bartington.com/products/mag-03/
- Stefan Mayer: https://stefan-mayer.com/en/
