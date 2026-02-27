---
description: Expand visual placeholders into matplotlib diagram code
---

Activate the visualize agent to find and expand `.visual-placeholder` blocks.

Target: $ARGUMENTS

The agent will:
1. Search for visual placeholders in the whitepaper sections
2. Study existing diagram patterns in `docs/hirt-whitepaper/diagrams/`
3. Create new diagram functions following project conventions
4. Replace placeholders with proper Quarto figure blocks

If a specific file or section is provided, focus only on placeholders in that location.
Otherwise, scan all sections and report what was found.
