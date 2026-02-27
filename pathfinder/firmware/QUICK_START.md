# Pathfinder Quick Start Guide

## Pre-Flight Checklist

- [ ] Battery charged and connected
- [ ] SD card inserted and formatted (FAT32)
- [ ] GPS antenna has clear sky view
- [ ] All sensor cables connected
- [ ] Harness adjusted for comfort

## Power-On Sequence

1. **Power on** - Status LED starts blinking
2. **Wait for slow blink** (2 second interval) = GPS locked and logging
3. **Listen for beeps** - One beep per second = system ready

## Status LED Meanings

| Pattern | Meaning | Action |
|---------|---------|--------|
| **Fast blink (100ms)** | Error - SD card failed | Check SD card |
| **Medium blink (500ms)** | No GPS fix | Wait for sky view |
| **Slow blink (2s)** | Normal - logging data | Ready to survey |

## Survey Operation

### Walking Technique

1. Walk at steady pace matching beeper (1 beep/second ≈ 1 m/s)
2. Keep crossbar level and parallel to ground
3. Maintain constant height (~15-20 cm above ground)
4. Avoid rapid turns or sudden stops
5. Mark corners with flags for georeferencing

### Grid Pattern

```
Start →→→→→→→→→→ End
       ↓
      ←←←←←←←←←← Turn
       ↓
      →→→→→→→→→→ Continue
       ↓
```

Walk parallel lines with 1-2 m spacing between passes.

## Data Download

1. **Power off** system
2. **Remove SD card**
3. **Insert** into computer
4. **Copy** `PATHXXXX.CSV` files
5. Files auto-increment: PATH0001, PATH0002, etc.

## Troubleshooting

### LED Won't Stop Fast Blinking

**Problem**: SD card error

**Solutions**:
- Remove and reinsert SD card
- Format card as FAT32 (not exFAT)
- Try different SD card
- Check SD module connections

### No GPS Lock (Medium Blink)

**Problem**: Cannot see satellites

**Solutions**:
- Move to open area (away from trees/buildings)
- Wait 30-60 seconds for cold start
- Check GPS antenna connection
- Verify GPS power LED is on

### Erratic Readings

**Problem**: Interference or bad connections

**Solutions**:
- Check all sensor cable connections
- Keep electronics box away from sensors
- Ensure battery voltage >7.0V
- Shield sensor cables from electronics

### No Beeps

**Problem**: Beeper disconnected or disabled

**Solutions**:
- Check beeper wire connection (pin 9)
- Verify beeper polarity
- Test beeper with multimeter
- Reflash firmware

## Field Tips

### Battery Management

- Bring spare battery for full-day surveys
- Monitor voltage (should stay >7.0V)
- Cold weather reduces capacity - keep battery warm

### Data Quality

- Walk at consistent speed
- Avoid metal objects on body (keys, phone)
- Remove watches and jewelry
- Keep separation from other operators (>5m)

### Weather Considerations

- **Rain**: Seal electronics in waterproof bag
- **Heat**: Shade electronics, prevent overheating
- **Cold**: Insulate battery, GPS may be slower to lock
- **Wind**: Brace crossbar to prevent swinging

## Post-Survey

1. Power off before moving to vehicle
2. Disassemble sensors from crossbar if transporting
3. Download data immediately (don't rely on SD card alone)
4. Check data quality before leaving site
5. Document any anomalies or issues in field notes

## Emergency Contacts

- Technical Support: [your contact info]
- Hardware Issues: [your contact info]
- Firmware Updates: [GitHub repo or website]

## Appendix: Data Format

CSV columns:
```
timestamp,lat,lon,g1_top,g1_bot,g1_grad,g2_top,g2_bot,g2_grad,g3_top,g3_bot,g3_grad,g4_top,g4_bot,g4_grad
```

- **timestamp**: Milliseconds since power-on
- **lat/lon**: GPS coordinates (decimal degrees)
- **gN_top/bot**: Raw ADC counts from sensors
- **gN_grad**: Gradient (bottom - top)

Import into QGIS, Python, or MATLAB for visualization.
