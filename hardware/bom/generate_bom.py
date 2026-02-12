#!/usr/bin/env python3
"""
Pathfinder BOM Generator

Reads pathfinder-bom.csv and generates a formatted markdown summary
with subtotals by category and overall cost range.

Usage:
    python generate_bom.py                    # Print to stdout
    python generate_bom.py -o README.md       # Write to file
"""

import argparse
import csv
import sys
from pathlib import Path


def load_bom(csv_path):
    """Load BOM from CSV file, return list of dicts."""
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['Qty'] = int(row['Qty'])
            row['Unit_Cost_Low'] = float(row['Unit_Cost_Low'])
            row['Unit_Cost_High'] = float(row['Unit_Cost_High'])
            row['Ext_Low'] = row['Qty'] * row['Unit_Cost_Low']
            row['Ext_High'] = row['Qty'] * row['Unit_Cost_High']
            rows.append(row)
    return rows


def group_by_category(rows):
    """Group rows by Category, preserving order of first appearance."""
    categories = {}
    order = []
    for row in rows:
        cat = row['Category']
        if cat not in categories:
            categories[cat] = []
            order.append(cat)
        categories[cat].append(row)
    return [(cat, categories[cat]) for cat in order]


def format_markdown(grouped, total_low, total_high):
    """Generate markdown BOM table."""
    lines = []
    lines.append("# Pathfinder Bill of Materials")
    lines.append("")
    lines.append("Complete parts list for building the Pathfinder 4-pair fluxgate gradiometer.")
    lines.append(f"**Estimated total cost: ${total_low:.0f} - ${total_high:.0f}**")
    lines.append("")
    lines.append("*Generated from `pathfinder-bom.csv` by `generate_bom.py`.*")
    lines.append("")

    for category, items in grouped:
        cat_low = sum(r['Ext_Low'] for r in items)
        cat_high = sum(r['Ext_High'] for r in items)

        lines.append(f"## {category} (${cat_low:.0f} - ${cat_high:.0f})")
        lines.append("")
        lines.append("| Item | Spec | Qty | Unit Cost | Ext. Cost | Supplier | Notes |")
        lines.append("|------|------|----:|----------:|----------:|----------|-------|")

        for r in items:
            unit_str = f"${r['Unit_Cost_Low']:.2f}-{r['Unit_Cost_High']:.2f}"
            ext_str = f"${r['Ext_Low']:.2f}-{r['Ext_High']:.2f}"
            lines.append(
                f"| {r['Item']} | {r['Specification']} | {r['Qty']} "
                f"| {unit_str} | {ext_str} | {r['Supplier']} | {r['Notes']} |"
            )

        lines.append("")

    lines.append("## Cost Summary")
    lines.append("")
    lines.append("| Category | Low | High |")
    lines.append("|----------|----:|-----:|")

    for category, items in grouped:
        cat_low = sum(r['Ext_Low'] for r in items)
        cat_high = sum(r['Ext_High'] for r in items)
        lines.append(f"| {category} | ${cat_low:.0f} | ${cat_high:.0f} |")

    lines.append(f"| **Total** | **${total_low:.0f}** | **${total_high:.0f}** |")
    lines.append("")

    lines.append("## Substitution Notes")
    lines.append("")
    lines.append("- **Crossbar**: Aluminum EMT conduit 1.25\" (~$15) is a viable budget substitute for carbon fiber (~$40)")
    lines.append("- **Arduino Nano**: Clones from AliExpress (~$5) work identically to genuine (~$25)")
    lines.append("- **Shoulder straps**: Salvaging from an old backpack saves $10-15")
    lines.append("- **3D-printed parts**: Can be ordered from Shapeways/JLCPCB if no printer available")
    lines.append("- **Fluxgate sensors**: The FG-3+ is recommended; Magnetometer-Kit.com FGM-3 PRO is a compatible alternative at ~20% higher cost")
    lines.append("")
    lines.append("## Source Data")
    lines.append("")
    lines.append("This file was generated from `pathfinder-bom.csv`. To regenerate after editing the CSV:")
    lines.append("```bash")
    lines.append("python generate_bom.py -o README.md")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate Pathfinder BOM summary from CSV')
    parser.add_argument('-o', '--output', type=str, help='Output markdown file (default: stdout)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to BOM CSV (default: pathfinder-bom.csv in same directory)')
    args = parser.parse_args()

    # Resolve CSV path
    script_dir = Path(__file__).parent
    csv_path = Path(args.csv) if args.csv else script_dir / 'pathfinder-bom.csv'

    if not csv_path.exists():
        print(f"Error: BOM CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_bom(csv_path)
    grouped = group_by_category(rows)

    total_low = sum(r['Ext_Low'] for r in rows)
    total_high = sum(r['Ext_High'] for r in rows)

    markdown = format_markdown(grouped, total_low, total_high)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / output_path
        output_path.write_text(markdown)
        print(f"BOM written to {output_path}")
        print(f"Total cost range: ${total_low:.0f} - ${total_high:.0f}")
        print(f"Items: {len(rows)} across {len(grouped)} categories")
    else:
        print(markdown)


if __name__ == '__main__':
    main()
