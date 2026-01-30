#!/usr/bin/env python3
"""
HIRT PDF Generator - Section 16: Field Checklists

Publication-quality PDF for field operations checklists including:
- Pre-deployment checklist
- On-site checklist
- Post-deployment checklist
- Emergency procedures
- Field log template

Optimized for printable checklist formatting with checkbox characters.

Usage:
    python section_16_field_checklists.py
"""

import os
import sys

# Add parent directory to path for package imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from lib import SectionPDFBuilder, CONTENT_WIDTH, PRIMARY, SECONDARY, ACCENT, WARNING, SUCCESS
from lib.diagrams.flowcharts import create_safety_checklist_visual

from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle, Spacer, Paragraph, KeepTogether, HRFlowable, CondPageBreak
from reportlab.lib.colors import HexColor
from reportlab.lib import colors


# Checkbox character for printable checklists
CHECKBOX = "\u2610"  # Empty checkbox: [ ]
CHECKBOX_BULLET = f"{CHECKBOX}  "


class ChecklistPDFBuilder(SectionPDFBuilder):
    """Extended PDF builder with checklist-specific formatting."""

    def add_checklist_items(self, items, indent_level=0):
        """
        Add checklist items with checkbox characters.

        Args:
            items: List of checklist item strings
            indent_level: Indentation level (0=normal, 1=sub-item)
        """
        left_indent = 25 + (indent_level * 20)

        for item in items:
            # Check if item has sub-items (starts with spaces in original)
            self.story.append(Paragraph(
                f"{CHECKBOX_BULLET}{item}",
                self._get_checklist_style(indent_level)
            ))

    def _get_checklist_style(self, indent_level=0):
        """Get paragraph style for checklist items."""
        from reportlab.lib.styles import ParagraphStyle

        base_indent = 20 + (indent_level * 15)
        return ParagraphStyle(
            f'ChecklistItem{indent_level}',
            fontName='Times-Roman',
            fontSize=10,
            leading=14,
            leftIndent=base_indent,
            firstLineIndent=0,
            spaceAfter=3,
        )

    def add_checklist_section(self, title, items, header_color=None):
        """
        Add a complete checklist section with header and items.

        Args:
            title: Section title
            items: List of (item_text, is_sub_item) tuples or just strings
            header_color: Optional header background color
        """
        if header_color is None:
            header_color = SECONDARY

        # Create section header box
        header_data = [[title]]
        header_table = Table(header_data, colWidths=[CONTENT_WIDTH])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), header_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#ffffff')),
            ('FONTNAME', (0, 0), (-1, -1), 'Times-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))

        self.story.append(Spacer(1, 8))
        self.story.append(header_table)
        self.story.append(Spacer(1, 4))

        # Add checklist items
        for item in items:
            if isinstance(item, tuple):
                text, is_sub = item
                self.add_checklist_items([text], indent_level=1 if is_sub else 0)
            else:
                self.add_checklist_items([item])

        self.story.append(Spacer(1, 6))

    def add_fill_in_field(self, label, width_chars=30):
        """
        Add a fill-in-the-blank field for field log templates.

        Args:
            label: Field label
            width_chars: Approximate width in characters for underline
        """
        underline = "_" * width_chars
        self.story.append(Paragraph(
            f"<b>{label}:</b> {underline}",
            self._get_field_style()
        ))

    def _get_field_style(self):
        """Get paragraph style for fill-in fields."""
        from reportlab.lib.styles import ParagraphStyle

        return ParagraphStyle(
            'FillInField',
            fontName='Times-Roman',
            fontSize=10,
            leading=18,
            leftIndent=10,
            spaceAfter=2,
        )

    def add_notes_area(self, lines=3):
        """
        Add a notes area with blank lines.

        Args:
            lines: Number of blank lines
        """
        for _ in range(lines):
            self.story.append(Paragraph(
                "_" * 80,
                self._get_field_style()
            ))


def create_section_16_pdf():
    """Generate Section 16: Field Checklists PDF."""

    # Initialize builder
    builder = ChecklistPDFBuilder(
        section_num=16,
        title="Field Checklists"
    )

    # Title block
    builder.add_title_block(
        subtitle="Pre-Deployment, On-Site, and Post-Deployment Procedures"
    )

    # Introduction
    builder.add_body_text(
        "This section provides comprehensive checklists for HIRT field operations. "
        "These checklists ensure consistent, safe, and thorough survey procedures "
        "from initial planning through final data archival. Print these pages for "
        "use in the field.",
        first_paragraph=True
    )

    builder.add_spacer(6)

    # =========================================================================
    # PRE-DEPLOYMENT CHECKLIST
    # =========================================================================
    builder.add_section_header("16.1 Pre-Deployment Checklist", level=1)

    # Permits and Legal
    builder.add_checklist_section(
        "Permits and Legal",
        [
            "Permits/ethics & UXO clearance confirmed",
            "Site access permissions obtained",
            "Team members briefed on legal/ethical requirements",
            "Emergency contacts documented",
        ],
        header_color=WARNING
    )

    # Equipment Preparation
    builder.add_checklist_section(
        "Equipment Preparation",
        [
            "Probe calibration sheet packed",
            "All probes tested and functional",
            "Batteries charged; spares packed",
            "Base hub tested and ready",
            "Cables tested for continuity",
            "Spare probes available (2-4 recommended)",
        ],
        header_color=SECONDARY
    )

    # Tools and Supplies
    builder.add_checklist_section(
        "Tools and Supplies",
        [
            "Pilot rods available",
            "Driver/extraction tools",
            "Flags, tapes for marking",
            "GPS/total station for coordinates",
            "Notebooks and data loggers",
            "Field diagnostic kit (multimeter, spare parts)",
        ],
        header_color=SECONDARY
    )

    # Documentation
    builder.add_checklist_section(
        "Documentation",
        [
            "Field log templates prepared",
            "Probe registry updated",
            "Site maps/plans available",
            "Previous survey data (if applicable)",
        ],
        header_color=SECONDARY
    )

    # =========================================================================
    # ON-SITE CHECKLIST
    # =========================================================================
    builder.story.append(CondPageBreak(5*inch))
    builder.add_section_header("16.2 On-Site Checklist", level=1)

    # Site Setup
    builder.add_checklist_section(
        "Site Setup",
        [
            "Establish control area (background scan location)",
            "Lay out section grid",
            "Record coordinates (GPS/total station)",
            "Document site conditions (soil, moisture, weather)",
            "Mark probe positions with flags",
        ],
        header_color=SUCCESS
    )

    # Probe Deployment
    builder.add_checklist_section(
        "Probe Deployment",
        [
            "Insert probes systematically",
            "Verify depths & IDs match records",
            "Connect Probes to Zone Boxes",
            "Connect Zone Boxes to Central Hub via Trunk Cables",
            "Verify Trunk Cable connectors are tight/screwed in",
            "Run diagnostic continuity scan from Hub",
        ],
        header_color=SUCCESS
    )

    # Data Collection
    builder.add_checklist_section(
        "Data Collection",
        [
            "Run background scan (control area)",
            "Run MIT sweep (all TX; multi-freq)",
            ("Verify all frequencies measured", True),
            ("Check for saturation or errors", True),
            "Run ERT patterns (2-4 baselines; flip polarity)",
            ("Verify current injection working", True),
            ("Check voltage readings reasonable", True),
            "QC repeats (5-10% of pairs)",
            ("Verify reciprocity (A->B approx B->A)", True),
            ("Check consistency", True),
        ],
        header_color=ACCENT
    )

    # Quality Control
    builder.add_checklist_section(
        "Quality Control",
        [
            "Monitor data quality in real-time",
            "Note any anomalies or issues",
            "Document any disturbances",
            "Verify time synchronization",
            "Check for ground loops or noise",
        ],
        header_color=ACCENT
    )

    # Section Completion
    builder.add_checklist_section(
        "Section Completion",
        [
            "All measurements completed",
            "Data backed up (if possible on-site)",
            "Extract probes carefully",
            "Move to next section (keep overlap if possible)",
            "Update field log",
        ],
        header_color=SUCCESS
    )

    # =========================================================================
    # POST-DEPLOYMENT CHECKLIST
    # =========================================================================
    builder.story.append(CondPageBreak(5*inch))
    builder.add_section_header("16.3 Post-Deployment Checklist", level=1)

    # Data Management
    builder.add_checklist_section(
        "Data Management",
        [
            "Back up CSVs to secure storage",
            "Photo log of the grid (if taken)",
            "Quick sanity plots (amp/phase vs distance)",
            "Verify data completeness",
            "Organize files by section/date",
        ],
        header_color=SECONDARY
    )

    # Equipment Care
    builder.add_checklist_section(
        "Equipment Care",
        [
            "Decontaminate/inspect probes",
            "Note any repairs needed",
            "Clean connectors and cables",
            "Check for damage",
            "Store equipment properly",
        ],
        header_color=SECONDARY
    )

    # Documentation
    builder.add_checklist_section(
        "Post-Survey Documentation",
        [
            "Complete field notes",
            "Update probe registry with any changes",
            "Document any issues encountered",
            "Note lessons learned",
            "Prepare preliminary report (if needed)",
        ],
        header_color=SECONDARY
    )

    # Follow-Up
    builder.add_checklist_section(
        "Follow-Up Actions",
        [
            "Review data quality",
            "Identify any missing measurements",
            "Plan next steps (additional sections, processing)",
            "Schedule equipment maintenance",
            "Update procedures based on experience",
        ],
        header_color=SECONDARY
    )

    # =========================================================================
    # EMERGENCY PROCEDURES
    # =========================================================================
    builder.story.append(CondPageBreak(5*inch))
    builder.add_section_header("16.4 Emergency Procedures", level=1)

    builder.add_body_text(
        "In the event of an emergency, use these checklists to ensure proper "
        "response and documentation. Safety is the top priority in all situations.",
        first_paragraph=True
    )

    # Safety Incidents
    builder.add_checklist_section(
        "Safety Incidents",
        [
            "Stop work immediately if unsafe conditions",
            "Evacuate if necessary",
            "Contact emergency services",
            "Document incident",
            "Report to site supervisor",
        ],
        header_color=WARNING
    )

    # Equipment Failure
    builder.add_checklist_section(
        "Equipment Failure",
        [
            "Isolate failed component",
            "Use spare if available",
            "Document failure for repair",
            "Continue with remaining probes if possible",
            "Adjust survey plan if needed",
        ],
        header_color=HexColor('#ed8936')  # Orange
    )

    # Data Loss
    builder.add_checklist_section(
        "Data Loss Response",
        [
            "Attempt recovery if possible",
            "Document what was lost",
            "Re-measure if critical",
            "Improve backup procedures",
            "Learn from incident",
        ],
        header_color=HexColor('#ed8936')  # Orange
    )

    # =========================================================================
    # SAFETY CHECKLIST VISUAL (Optional diagram)
    # =========================================================================
    builder.add_spacer(12)
    builder.add_section_header("16.5 UXO Site Safety Visual", level=1)

    builder.add_body_text(
        "For humanitarian demining or UXO-sensitive sites, the following visual "
        "checklist provides a quick reference for critical safety requirements.",
        first_paragraph=True
    )

    # Add the safety checklist visual diagram
    safety_visual = create_safety_checklist_visual()
    builder.add_figure(
        safety_visual,
        "UXO Site Safety Checklist - Complete ALL items before operations.",
        width=5 * inch,
        height=4.4 * inch  # Maintain aspect ratio of 8:7 from the diagram
    )

    # =========================================================================
    # FIELD LOG TEMPLATE
    # =========================================================================
    builder.story.append(CondPageBreak(5*inch))
    builder.add_section_header("16.6 Field Log Template", level=1)

    builder.add_body_text(
        "Use this template to record daily survey activities. Make copies as needed "
        "for each day of fieldwork. Complete entries help with data processing and "
        "future reference.",
        first_paragraph=True
    )

    builder.add_spacer(8)

    # Daily Entry section
    builder.add_section_header("Daily Entry", level=2)
    builder.add_fill_in_field("Date", 25)
    builder.add_fill_in_field("Site", 25)
    builder.add_fill_in_field("Section", 25)
    builder.add_fill_in_field("Team", 35)
    builder.add_fill_in_field("Weather", 30)
    builder.add_fill_in_field("Soil conditions", 30)

    builder.add_spacer(12)

    # Measurements section
    builder.add_section_header("Measurements", level=2)
    builder.add_fill_in_field("Probes deployed", 25)
    builder.add_fill_in_field("MIT frequencies", 30)
    builder.add_fill_in_field("ERT baselines", 25)
    builder.add_fill_in_field("Start time", 20)
    builder.add_fill_in_field("End time", 20)
    builder.add_fill_in_field("Issues encountered", 40)

    builder.add_spacer(12)

    # Notes section
    builder.add_section_header("Notes", level=2)
    builder.add_notes_area(6)

    builder.add_spacer(12)

    # Sign-off
    builder.add_horizontal_rule()
    builder.add_fill_in_field("Completed by", 30)
    builder.add_fill_in_field("Signature", 30)
    builder.add_fill_in_field("Date", 20)

    # =========================================================================
    # QUICK REFERENCE SUMMARY
    # =========================================================================
    builder.add_spacer(16)
    builder.add_section_header("16.7 Quick Reference Summary", level=1)

    builder.add_body_text(
        "Summary of critical items that must be verified at each phase:",
        first_paragraph=True
    )

    # Create summary table
    summary_data = [
        ["Phase", "Critical Items", "Sign-Off"],
        ["Pre-Deployment",
         "Permits confirmed\nEquipment tested\nBatteries charged\nSpares packed",
         CHECKBOX],
        ["Site Setup",
         "Grid marked\nCoordinates recorded\nConditions documented",
         CHECKBOX],
        ["Data Collection",
         "Background scan done\nMIT sweep complete\nERT patterns run\nQC repeats done",
         CHECKBOX],
        ["Post-Survey",
         "Data backed up\nEquipment cleaned\nNotes completed",
         CHECKBOX],
    ]

    col_widths = [1.3 * inch, 3.5 * inch, 0.8 * inch]
    summary_table = Table(summary_data, colWidths=col_widths)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (1, -1), 'LEFT'),
        ('ALIGN', (2, 1), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f7fafc')),
        ('FONTNAME', (0, 1), (0, -1), 'Times-Bold'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    builder.story.append(Spacer(1, 10))
    builder.story.append(summary_table)

    builder.add_spacer(16)

    # Important notes
    builder.add_info_box(
        "Important Notes",
        [
            "Print multiple copies of this section for field use",
            "Laminate checklists for durability in wet conditions",
            "Complete ALL items before moving to next phase",
            "Document any deviations from standard procedures",
            "Keep completed logs for project records",
        ]
    )

    # Build the PDF
    output_path = builder.build()
    return output_path


def main():
    """Main entry point."""
    print("Generating Section 16: Field Checklists PDF...")
    print("=" * 60)

    try:
        output_path = create_section_16_pdf()
        print("=" * 60)
        print(f"Success! PDF generated at:")
        print(f"  {output_path}")
        return 0
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
