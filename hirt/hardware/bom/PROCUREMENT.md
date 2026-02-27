# Procurement Workflow Guide

## Overview

This guide provides step-by-step instructions for ordering components for the HIRT system using the BOM files and CSV order sheets.

## Prerequisites

- Review [Probe BOM](probe-bom.md) and [Base Hub BOM](base-hub-bom.md)
- Determine quantity needed (typically 20 probes + base hub)
- Set procurement budget
- Identify preferred suppliers

## Step 1: Review BOM Files

### 1.1 Understand Component Requirements

- **Probe Components:** See [probe-bom.md](probe-bom.md)
  - Mechanical: Rods, capsules, seals, etc.
  - MIT Electronics: DDS, amplifiers, ADC, MCU
  - ERT Components: Electrodes, mux, amplifiers

- **Base Hub Components:** See [base-hub-bom.md](base-hub-bom.md)
  - ERT current source
  - Differential voltmeter
  - Sync/clock distribution
  - Power system
  - Communications/data logging
  - Enclosure

### 1.2 Check for Alternatives

- Some components have alternatives listed in notes
- Consider cost vs. performance trade-offs
- Verify compatibility if substituting parts

## Step 2: Prepare Order Lists

### 2.1 Use CSV Order Sheets

The CSV files in [order-sheets/](order-sheets/) are ready for import:

- **probe-order-sheet.csv** - Components for probes
- **base-hub-order-sheet.csv** - Base hub components
- **complete-kit-order-sheet.csv** - Summary of all components

### 2.2 Calculate Quantities

**For 20 probes:**
- Multiply "Quantity per Probe" by 20
- Add 10-20% spares for critical components
- Order wire, connectors, etc. in bulk

**Example:**
- AD9833 DDS: 1 per probe × 20 = 20 units
- Add 2 spares = **22 total**
- ESP32: 1 per probe × 20 = 20 units
- Add 2 spares = **22 total**

### 2.3 Create Supplier-Specific Lists

Group components by supplier:
- **Digi-Key:** Most electronic components
- **Mouser:** Alternative for electronics
- **McMaster-Carr:** Mechanical components
- **Amazon:** Batteries, enclosures, common items
- **Local/3D Print:** Custom parts (capsules, couplers)

## Step 3: Supplier Selection

### 3.1 Major Distributors

**Digi-Key Electronics**
- **Website:** digikey.com
- **Best for:** ICs, passives, connectors, wire
- **Shipping:** Fast, reliable
- **Minimum order:** Usually none
- **Bulk discounts:** Available for large quantities

**Mouser Electronics**
- **Website:** mouser.com
- **Best for:** ICs, passives (alternative to Digi-Key)
- **Shipping:** Fast, reliable
- **Price comparison:** Compare with Digi-Key

**McMaster-Carr**
- **Website:** mcmaster.com
- **Best for:** Mechanical components, rods, hardware
- **Shipping:** Fast, reliable
- **No minimum:** Usually none

**Amazon**
- **Best for:** Batteries, enclosures, common items
- **Convenience:** Fast shipping, easy returns
- **Caution:** Verify part numbers match

### 3.2 Specialty Suppliers

- **Adafruit/SparkFun:** ESP32 modules, breakout boards
- **Local suppliers:** Fiberglass rods, custom machining
- **3D printing services:** Probe capsules, couplers

## Step 4: Ordering Process

### 4.1 Create Shopping Carts

**Digi-Key Process:**
1. Go to digikey.com
2. Search for part number (e.g., "AD9833BRMZ-REEL7")
3. Add to cart with correct quantity
4. Repeat for all components
5. Review cart for accuracy
6. Apply quantity discounts if available
7. Check shipping options and costs

**Mouser Process:**
1. Similar to Digi-Key
2. Compare prices if ordering from both
3. Consider consolidating to one supplier for simplicity

**McMaster-Carr Process:**
1. Search by part number or description
2. Select appropriate size/material
3. Add to cart
4. Note: May need to call for custom items

### 4.2 Verify Part Numbers

**Before ordering, verify:**
- Part number matches BOM exactly
- Package type (SOIC, DIP, etc.) is correct
- Quantity matches requirements
- Lead time is acceptable
- Stock availability

### 4.3 Check Alternatives

If a part is out of stock or expensive:
- Check notes in BOM for alternatives
- Search distributor websites for equivalents
- Verify pin compatibility
- Update BOM if substituting

## Step 5: Bulk Ordering Tips

### 5.1 Quantity Discounts

- Order all probes' components at once (20× quantities)
- Many distributors offer 10-20% discount at quantity breaks
- Check pricing tiers before ordering

### 5.2 Spare Components

**Recommended spares (10-20% extra):**
- Critical ICs (DDS, ADC, MCU): 2-4 spares
- Connectors, headers: 10-20% extra
- Fuses, small components: 20-30% extra
- Wire, cable: Order extra (useful for repairs)

### 5.3 Common Components

Order in larger quantities:
- Headers, sockets: Used throughout
- Resistors, capacitors: Common values
- Wire, cable: Useful for prototyping and repairs

## Step 6: Cost Tracking

### 6.1 Create Spreadsheet

Track orders with columns:
- Component name
- Part number
- Supplier
- Quantity ordered
- Unit cost
- Total cost
- Order date
- Expected delivery
- Actual delivery
- Notes

### 6.2 Budget Management

- Set budget limits for each category
- Track actual vs. estimated costs
- Note any significant cost differences
- Update BOM with actual costs for future reference

## Step 7: Ordering Checklist

### Pre-Order Checklist

- [ ] BOM files reviewed
- [ ] Quantities calculated (including spares)
- [ ] Part numbers verified
- [ ] Suppliers selected
- [ ] Budget approved
- [ ] Shopping carts prepared
- [ ] Shipping addresses confirmed
- [ ] Payment method ready

### Order Verification

- [ ] All part numbers correct
- [ ] Quantities match requirements
- [ ] Shipping addresses correct
- [ ] Payment processed
- [ ] Order confirmations received
- [ ] Expected delivery dates noted

### Post-Order

- [ ] Track shipments
- [ ] Verify received items match orders
- [ ] Check for damaged items
- [ ] Update inventory spreadsheet
- [ ] Store components properly
- [ ] Note any backorders or delays

## Step 8: Handling Issues

### 8.1 Out of Stock Items

- Check alternative suppliers
- Look for equivalent parts
- Consider backorder if acceptable
- Update BOM with substitutions

### 8.2 Wrong Parts Received

- Contact supplier immediately
- Request return/replacement
- Verify correct part number
- Update records

### 8.3 Cost Overruns

- Review actual vs. estimated costs
- Identify high-cost items
- Consider alternatives for future orders
- Update BOM with actual costs

## Step 9: Component Storage

### 9.1 Organization

- Organize by component type
- Use labeled bins/containers
- Keep ESD-sensitive parts in anti-static bags
- Store in dry, temperature-controlled area

### 9.2 Inventory Management

- Track what's been used
- Keep spares organized
- Label probe-specific components
- Maintain inventory spreadsheet

## Step 10: Future Orders

### 10.1 Learn from Experience

- Note which suppliers were best
- Record actual costs vs. estimates
- Identify components that needed alternatives
- Update BOM files with lessons learned

### 10.2 Refinement

- Update BOMs with verified part numbers
- Add notes about alternatives that worked
- Document any issues encountered
- Share updates with team

## Cost Tracking Template

```csv
Component,Part Number,Supplier,Quantity,Unit Cost,Total Cost,Order Date,Delivery Date,Status,Notes
AD9833 DDS,AD9833BRMZ-REEL7,Digi-Key,22,$8.50,$187.00,2024-03-15,2024-03-20,Received,
ESP32 DevKit,ESP32-DEVKITC-32E,Adafruit,22,$10.00,$220.00,2024-03-15,2024-03-18,Received,
...
```

## Supplier Contact Information

### Digi-Key Electronics
- **Website:** www.digikey.com
- **Phone:** 1-800-344-4539
- **Email:** support@digikey.com
- **Hours:** 24/7 online ordering

### Mouser Electronics
- **Website:** www.mouser.com
- **Phone:** 1-800-346-6873
- **Email:** support@mouser.com
- **Hours:** 24/7 online ordering

### McMaster-Carr
- **Website:** www.mcmaster.com
- **Phone:** 1-630-833-0300
- **Email:** Contact via website
- **Hours:** Business hours

## Notes

- **Lead Times:** Some components may have long lead times; order early
- **Minimum Orders:** Some suppliers have minimum order values; consolidate orders
- **Shipping Costs:** Consider shipping costs when selecting suppliers
- **Tax:** Sales tax applies in many jurisdictions
- **Import Duties:** International orders may incur customs/duties

## Quick Reference

**Typical Order Timeline:**
1. Review BOM: 1-2 hours
2. Prepare order lists: 2-4 hours
3. Create shopping carts: 2-3 hours
4. Place orders: 1 hour
5. Receive components: 3-7 days (domestic)

**Total Procurement Time:** ~1-2 weeks from start to receipt

**Estimated Total Cost (20 probes + base hub):**
- Probes: $2,500-3,600 (20 × $125-180)
- Base Hub: $250-450
- Tools/Supplies: $200-400
- **Total: $2,950-4,450**

*Note: Actual costs vary by supplier, quantity discounts, and component choices.*

