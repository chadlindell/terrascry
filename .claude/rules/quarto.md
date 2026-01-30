---
paths:
  - "docs/**/*.qmd"
  - "docs/**/*.py"
---

# Quarto Documentation Rules

## Writing Quality Check
Before writing or editing any .qmd content, verify:
- Am I writing narrative prose or falling into list-of-lists?
- Does this section flow from the previous and into the next?
- Would a visual help here? If so, add a `.visual-placeholder` block.
- Is the audience (technical academics) kept in mind?

## Prose Standards
- **Narrative over enumeration**: Restructure bullet cascades as flowing paragraphs
- **Active voice**: "The circuit amplifies..." not "The signal is amplified by..."
- **Terminology**: MIT-3D, ERT-Lite, probe, crosshole (see CLAUDE.md for full list)
- **Rigor**: Back claims with equations, measurements, or citations

## Visual Placeholders
When a concept needs illustration, insert:
```markdown
::: {.visual-placeholder}
**Figure Concept**: [title]
[Rich description of what to visualize]
**Type**: [diagram | schematic | chart | cross-section | exploded-view]
:::
```
These will be expanded into Python diagram code by a visualization agent.

## Diagrams (Python)
- All diagram code in `docs/hirt-whitepaper/diagrams/*.py`
- Use matplotlib with consistent style
- Return BytesIO buffer from diagram functions
- Figure labels: `fig-<descriptive-name>`

## Formatting
- No Unicode symbols in prose (use LaTeX: $\geq$, $\pm$, $\Omega$)
- Cross-references: @fig-name, @tbl-name, @sec-name
- Equations in display mode for important relationships

## After Editing
Run `quarto render` from `docs/hirt-whitepaper/` to rebuild
