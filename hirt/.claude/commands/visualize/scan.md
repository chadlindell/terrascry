---
description: Scan for visual placeholders without expanding them
---

Scan the whitepaper for `.visual-placeholder` blocks and report what visualizations are needed.

Target: $ARGUMENTS

This is a read-only operation. For each placeholder found, report:
- Location (file and line)
- Figure concept/title
- Type (diagram, schematic, chart, etc.)
- Complexity estimate

Use this to plan visualization work before running `/visualize:expand`.
