#!/usr/bin/env python3
"""
HIRT PDF Generator - Section 00: Index / Table of Contents

Generates a publication-quality PDF for the HIRT Whitepaper Index,
including document structure overview and section dependency diagrams.

Usage:
    python section_00_index.py

Output:
    ../output/sections/00-index.pdf
"""

import os
import sys

# Add parent directory to path for lib imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import numpy as np
from io import BytesIO

from reportlab.lib.units import inch
from reportlab.platypus import Spacer, Table, TableStyle, KeepTogether, Paragraph
from reportlab.lib import colors

from lib.pdf_builder import SectionPDFBuilder
from lib.styles import (
    get_styles, CONTENT_WIDTH, PRIMARY, SECONDARY, ACCENT,
    SUCCESS, WARNING, LIGHT_BG, COLOR_PALETTE
)


# =============================================================================
# FIGURE 1: DOCUMENT STRUCTURE OVERVIEW
# =============================================================================
def create_document_structure_diagram():
    """
    Creates a visual representation of the HIRT whitepaper structure
    showing the 5 parts and 19 sections organized hierarchically.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Color scheme for parts
    part_colors = {
        'I': '#2c5282',    # Navy - Foundations
        'II': '#38a169',   # Green - Building
        'III': '#ed8936',  # Orange - Field Operations
        'IV': '#805ad5',   # Purple - Reference
    }

    # Title
    ax.text(5, 7.7, 'HIRT Whitepaper v2.0 - Document Structure',
            fontsize=14, fontweight='bold', ha='center', color='#1a365d')
    ax.text(5, 7.35, '5 Parts | 19 Sections | ~195 KB',
            fontsize=10, ha='center', color='#4a5568', style='italic')

    # Part I: Foundations (3 sections)
    part1_x, part1_y = 0.3, 5.5
    ax.add_patch(FancyBboxPatch((part1_x, part1_y), 2.0, 1.6,
                                boxstyle="round,pad=0.03",
                                facecolor='#ebf8ff', edgecolor=part_colors['I'],
                                linewidth=2))
    ax.text(part1_x + 1.0, part1_y + 1.4, 'PART I', fontsize=9,
            fontweight='bold', ha='center', color=part_colors['I'])
    ax.text(part1_x + 1.0, part1_y + 1.15, 'FOUNDATIONS', fontsize=8,
            ha='center', color=part_colors['I'])

    sections_p1 = ['01 Executive Summary', '02 Physics & Theory', '03 System Architecture']
    for i, sec in enumerate(sections_p1):
        ax.text(part1_x + 0.15, part1_y + 0.85 - i*0.28, sec, fontsize=7,
                color='#2d3748', va='center')

    # Part II: Building (6 sections)
    part2_x, part2_y = 2.6, 4.8
    ax.add_patch(FancyBboxPatch((part2_x, part2_y), 2.0, 2.3,
                                boxstyle="round,pad=0.03",
                                facecolor='#f0fff4', edgecolor=part_colors['II'],
                                linewidth=2))
    ax.text(part2_x + 1.0, part2_y + 2.1, 'PART II', fontsize=9,
            fontweight='bold', ha='center', color=part_colors['II'])
    ax.text(part2_x + 1.0, part2_y + 1.85, 'BUILDING', fontsize=8,
            ha='center', color=part_colors['II'])

    sections_p2 = ['04 Bill of Materials', '05 Mechanical Design',
                   '06 Electronics & Circuits', '07 Assembly & Wiring',
                   '08 Testing & Verification', '09 Calibration']
    for i, sec in enumerate(sections_p2):
        ax.text(part2_x + 0.15, part2_y + 1.55 - i*0.26, sec, fontsize=7,
                color='#2d3748', va='center')

    # Part III: Field Operations (4 sections)
    part3_x, part3_y = 5.0, 5.2
    ax.add_patch(FancyBboxPatch((part3_x, part3_y), 2.0, 1.9,
                                boxstyle="round,pad=0.03",
                                facecolor='#fffaf0', edgecolor=part_colors['III'],
                                linewidth=2))
    ax.text(part3_x + 1.0, part3_y + 1.7, 'PART III', fontsize=9,
            fontweight='bold', ha='center', color=part_colors['III'])
    ax.text(part3_x + 1.0, part3_y + 1.45, 'FIELD OPERATIONS', fontsize=8,
            ha='center', color=part_colors['III'])

    sections_p3 = ['10 Field Operations', '11 Data Recording',
                   '12 Data Interpretation', '13 Troubleshooting']
    for i, sec in enumerate(sections_p3):
        ax.text(part3_x + 0.15, part3_y + 1.15 - i*0.28, sec, fontsize=7,
                color='#2d3748', va='center')

    # Part IV: Reference (6 sections)
    part4_x, part4_y = 7.4, 4.8
    ax.add_patch(FancyBboxPatch((part4_x, part4_y), 2.3, 2.3,
                                boxstyle="round,pad=0.03",
                                facecolor='#faf5ff', edgecolor=part_colors['IV'],
                                linewidth=2))
    ax.text(part4_x + 1.15, part4_y + 2.1, 'PART IV', fontsize=9,
            fontweight='bold', ha='center', color=part_colors['IV'])
    ax.text(part4_x + 1.15, part4_y + 1.85, 'REFERENCE', fontsize=8,
            ha='center', color=part_colors['IV'])

    sections_p4 = ['14 Glossary', '15 Quick Reference',
                   '16 Field Checklists', '17 Application Scenarios',
                   '18 Future Development', '19 Ethics, Legal & Safety']
    for i, sec in enumerate(sections_p4):
        ax.text(part4_x + 0.15, part4_y + 1.55 - i*0.26, sec, fontsize=7,
                color='#2d3748', va='center')

    # Connection arrows between parts
    arrow_style = dict(arrowstyle='->', color='#718096', lw=1.5,
                       connectionstyle='arc3,rad=0.1')

    # Part I -> Part II
    ax.annotate('', xy=(part2_x, part2_y + 1.6), xytext=(part1_x + 2.0, part1_y + 0.8),
                arrowprops=arrow_style)

    # Part II -> Part III
    ax.annotate('', xy=(part3_x, part3_y + 0.95), xytext=(part2_x + 2.0, part2_y + 1.15),
                arrowprops=arrow_style)

    # Part III -> Part IV
    ax.annotate('', xy=(part4_x, part4_y + 1.15), xytext=(part3_x + 2.0, part3_y + 0.95),
                arrowprops=arrow_style)

    # Reader Paths section at bottom
    ax.axhline(y=3.8, xmin=0.03, xmax=0.97, color='#e2e8f0', linewidth=1)
    ax.text(5, 3.5, 'READER PATHS', fontsize=10, fontweight='bold',
            ha='center', color='#1a365d')

    # Path boxes
    paths = [
        ('System Builder', '#2c5282', '01 > 03 > 04 > 05 > 06 > 07 > 08 > 09', 0.5),
        ('Field Operator', '#38a169', '15 > 16 > 10 > 11 > 13 > 19', 2.7),
        ('Data Analyst', '#ed8936', '02 > 11 > 12 > 17', 5.0),
        ('Quick Start', '#805ad5', '01 > 15 > 10', 7.3),
    ]

    for name, color, path, x in paths:
        ax.add_patch(FancyBboxPatch((x, 2.0), 2.2, 1.3,
                                    boxstyle="round,pad=0.03",
                                    facecolor='white', edgecolor=color,
                                    linewidth=1.5))
        ax.text(x + 1.1, 3.05, name, fontsize=8, fontweight='bold',
                ha='center', color=color)
        ax.text(x + 1.1, 2.65, path, fontsize=6, ha='center', color='#4a5568')

        # Small icon
        ax.add_patch(Circle((x + 1.1, 2.35), 0.15, facecolor=color, alpha=0.3))

    # Legend for section sizes
    ax.text(0.5, 1.5, 'Documentation Sizes:', fontsize=8, fontweight='bold', color='#1a365d')
    sizes = [('I: 26 KB', '#2c5282'), ('II: 95 KB', '#38a169'),
             ('III: 37 KB', '#ed8936'), ('IV: 37 KB', '#805ad5')]
    for i, (size, color) in enumerate(sizes):
        ax.add_patch(Rectangle((0.5 + i*1.8, 1.0), 0.3, 0.3, facecolor=color, alpha=0.7))
        ax.text(0.9 + i*1.8, 1.15, size, fontsize=7, va='center', color='#4a5568')

    ax.text(8.5, 1.15, 'Total: ~195 KB', fontsize=8, fontweight='bold',
            ha='center', color='#1a365d')

    # Version note
    ax.text(5, 0.3, 'Consolidated from 33 sections (v1.0) to 19 sections (v2.0)',
            fontsize=7, ha='center', color='#718096', style='italic')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# =============================================================================
# FIGURE 2: SECTION DEPENDENCY MAP
# =============================================================================
def create_dependency_map():
    """
    Creates a visual diagram showing how sections relate to each other
    and the recommended reading order for different use cases.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.7, 'Section Dependencies & Cross-References',
            fontsize=12, fontweight='bold', ha='center', color='#1a365d')

    # Define section positions in a flowchart layout
    positions = {
        # Row 1: Entry point
        '01': (5.0, 5.8),

        # Row 2: Foundation branches
        '02': (2.0, 4.8),
        '03': (5.0, 4.8),
        '19': (8.0, 4.8),

        # Row 3: Build sequence
        '04': (1.5, 3.6),
        '05': (3.0, 3.6),
        '06': (4.5, 3.6),
        '07': (6.0, 3.6),
        '08': (7.5, 2.8),
        '09': (7.5, 3.6),

        # Row 4: Field operations
        '10': (2.5, 2.0),
        '11': (4.0, 2.0),
        '12': (5.5, 2.0),
        '13': (7.0, 2.0),

        # Row 5: Reference materials
        '14': (1.5, 0.8),
        '15': (3.0, 0.8),
        '16': (4.5, 0.8),
        '17': (6.0, 0.8),
        '18': (7.5, 0.8),
    }

    # Section colors by part
    colors_by_section = {
        '01': '#2c5282', '02': '#2c5282', '03': '#2c5282',
        '04': '#38a169', '05': '#38a169', '06': '#38a169',
        '07': '#38a169', '08': '#38a169', '09': '#38a169',
        '10': '#ed8936', '11': '#ed8936', '12': '#ed8936', '13': '#ed8936',
        '14': '#805ad5', '15': '#805ad5', '16': '#805ad5',
        '17': '#805ad5', '18': '#805ad5', '19': '#805ad5',
    }

    # Short names
    names = {
        '01': 'Exec\nSummary', '02': 'Physics', '03': 'System\nArch',
        '04': 'BOM', '05': 'Mech', '06': 'Elec', '07': 'Assy',
        '08': 'Test', '09': 'Cal',
        '10': 'Field\nOps', '11': 'Data\nRec', '12': 'Interp', '13': 'Troubl',
        '14': 'Gloss', '15': 'Quick\nRef', '16': 'Check',
        '17': 'Scenarios', '18': 'Future', '19': 'Ethics\n& Safety',
    }

    # Draw section boxes
    for sec, (x, y) in positions.items():
        color = colors_by_section[sec]
        ax.add_patch(FancyBboxPatch((x-0.45, y-0.35), 0.9, 0.7,
                                    boxstyle="round,pad=0.02",
                                    facecolor='white', edgecolor=color,
                                    linewidth=1.5))
        ax.text(x, y + 0.1, sec, fontsize=8, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x, y - 0.15, names[sec], fontsize=5, ha='center',
                va='center', color='#4a5568')

    # Primary dependencies (solid arrows)
    primary_deps = [
        ('01', '02'), ('01', '03'), ('01', '19'),
        ('03', '04'), ('04', '05'), ('05', '06'), ('06', '07'),
        ('07', '08'), ('08', '09'), ('07', '09'),
        ('09', '10'), ('10', '11'), ('11', '12'), ('10', '13'),
        ('02', '11'), ('02', '12'),
    ]

    for src, dst in primary_deps:
        sx, sy = positions[src]
        dx, dy = positions[dst]

        # Adjust start/end points to box edges
        if sy > dy:
            sy -= 0.35
            dy += 0.35
        elif sy < dy:
            sy += 0.35
            dy -= 0.35
        else:
            if sx < dx:
                sx += 0.45
                dx -= 0.45
            else:
                sx -= 0.45
                dx += 0.45

        ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color='#a0aec0', lw=1,
                                   connectionstyle='arc3,rad=0.1'))

    # Cross-reference connections (dashed)
    cross_refs = [
        ('05', '09'), ('06', '08'), ('11', '17'), ('10', '15'),
        ('10', '16'), ('13', '19'),
    ]

    for src, dst in cross_refs:
        sx, sy = positions[src]
        dx, dy = positions[dst]

        if sy > dy:
            sy -= 0.35
            dy += 0.35
        elif sy < dy:
            sy += 0.35
            dy -= 0.35
        else:
            if sx < dx:
                sx += 0.45
                dx -= 0.45
            else:
                sx -= 0.45
                dx += 0.45

        ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color='#cbd5e0', lw=0.8,
                                   linestyle='dashed',
                                   connectionstyle='arc3,rad=0.2'))

    # Legend
    ax.add_patch(Rectangle((0.2, 6.2), 0.3, 0.2, facecolor='#2c5282', alpha=0.8))
    ax.text(0.6, 6.3, 'Part I: Foundations', fontsize=7, va='center')

    ax.add_patch(Rectangle((2.2, 6.2), 0.3, 0.2, facecolor='#38a169', alpha=0.8))
    ax.text(2.6, 6.3, 'Part II: Building', fontsize=7, va='center')

    ax.add_patch(Rectangle((4.2, 6.2), 0.3, 0.2, facecolor='#ed8936', alpha=0.8))
    ax.text(4.6, 6.3, 'Part III: Field Ops', fontsize=7, va='center')

    ax.add_patch(Rectangle((6.2, 6.2), 0.3, 0.2, facecolor='#805ad5', alpha=0.8))
    ax.text(6.6, 6.3, 'Part IV: Reference', fontsize=7, va='center')

    # Arrow legend
    ax.annotate('', xy=(8.7, 6.35), xytext=(8.2, 6.35),
                arrowprops=dict(arrowstyle='->', color='#a0aec0', lw=1))
    ax.text(8.8, 6.3, 'Primary', fontsize=6, va='center')

    ax.annotate('', xy=(9.7, 6.35), xytext=(9.2, 6.35),
                arrowprops=dict(arrowstyle='->', color='#cbd5e0', lw=0.8,
                               linestyle='dashed'))
    ax.text(9.8, 6.3, 'Cross-ref', fontsize=6, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# =============================================================================
# DOCUMENT BUILDER
# =============================================================================
def build_section_00():
    """Build the Section 00: Index PDF document."""

    print("Generating Section 00: Index / Table of Contents...")

    # Create PDF builder
    builder = SectionPDFBuilder(
        section_num=0,
        title="Index"
    )
    styles = builder.styles

    # === TITLE BLOCK ===
    builder.add_title_block(
        subtitle="Hybrid Impedance-Resistivity Tomography System - Whitepaper Index"
    )

    # === ABOUT THIS DOCUMENT ===
    builder.add_section_header("About This Document")
    builder.add_body_text(
        "This whitepaper provides complete documentation for the HIRT system, a hybrid "
        "geophysical survey tool combining Magnetic Induction Tomography (MIT) and "
        "Electrical Resistivity Tomography (ERT) for subsurface imaging in forensic, "
        "archaeological, and environmental applications.",
        first_paragraph=True
    )

    builder.add_section_header("Key Applications", level=2)
    builder.add_bullet_list([
        "<b>WWII bomb crater investigation</b> - UXO assessment and burial detection",
        "<b>Woodland burial search</b> - Clandestine grave detection under tree cover",
        "<b>Wetland/swamp surveys</b> - Subsurface mapping in challenging terrain",
    ])

    builder.add_spacer(8)

    # === DOCUMENT STRUCTURE DIAGRAM ===
    print("  Creating document structure diagram...")
    fig_structure = create_document_structure_diagram()
    builder.add_figure(
        fig_structure,
        "HIRT Whitepaper v2.0 document organization. The whitepaper is divided into "
        "5 parts containing 19 sections, with recommended reader paths for different "
        "user roles. Total documentation size is approximately 195 KB, consolidated "
        "from the original 33-section v1.0 structure.",
        width=CONTENT_WIDTH
    )

    builder.add_conditional_page_break(3*inch)

    # === PART I: FOUNDATIONS ===
    builder.add_section_header("Part I: Foundations")
    builder.add_body_text(
        "The Foundations part introduces the HIRT system, its theoretical basis, "
        "and overall architecture. These sections provide essential context for "
        "understanding the system design.",
        first_paragraph=True
    )

    part1_data = [
        ['#', 'Section', 'Description'],
        ['01', 'Executive Summary', 'What is HIRT, use cases, capabilities, target audience'],
        ['02', 'Physics & Theory', 'MIT and ERT principles, frequency selection, measurement geometry'],
        ['03', 'System Architecture', 'Micro-probe design, centralized hub, array configurations'],
    ]
    builder.add_table(part1_data,
                      col_widths=[CONTENT_WIDTH*0.08, CONTENT_WIDTH*0.28, CONTENT_WIDTH*0.64],
                      caption="Part I: Foundations - 3 sections, ~26 KB")

    # === PART II: BUILDING ===
    builder.add_section_header("Part II: Building")
    builder.add_body_text(
        "The Building part provides comprehensive instructions for constructing a "
        "complete HIRT system. These sections should be followed sequentially when "
        "building a new system from components.",
        first_paragraph=True
    )

    part2_data = [
        ['#', 'Section', 'Description'],
        ['04', 'Bill of Materials', 'Complete BOM with costs ($1,800-3,900), part numbers, suppliers'],
        ['05', 'Mechanical Design', 'Rod segments, coils, ERT rings, junction box, 3D prints, CAD'],
        ['06', 'Electronics & Circuits', 'MIT circuit, ERT circuit, power, complete schematics'],
        ['07', 'Assembly & Wiring', 'Step-by-step assembly, wiring diagrams, waterproofing'],
        ['08', 'Testing & Verification', 'Test procedures, QC checklist, pass/fail criteria'],
        ['09', 'Calibration', 'Coil, TX, RX, ERT calibration; field quick-check; schedule'],
    ]
    builder.add_table(part2_data,
                      col_widths=[CONTENT_WIDTH*0.08, CONTENT_WIDTH*0.28, CONTENT_WIDTH*0.64],
                      caption="Part II: Building - 6 sections, ~95 KB")

    # === PART III: FIELD OPERATIONS ===
    builder.add_section_header("Part III: Field Operations")
    builder.add_body_text(
        "The Field Operations part covers deployment, data collection, and "
        "troubleshooting in the field. These sections are essential references "
        "for field operators.",
        first_paragraph=True
    )

    part3_data = [
        ['#', 'Section', 'Description'],
        ['10', 'Field Operations', 'Planning, grid design, installation, measurement protocols'],
        ['11', 'Data Recording', 'MIT/ERT file formats, probe registry, metadata'],
        ['12', 'Data Interpretation', 'Resolution, detection limits, combined analysis'],
        ['13', 'Troubleshooting', 'Diagnostics, repairs, when to abort'],
    ]
    builder.add_table(part3_data,
                      col_widths=[CONTENT_WIDTH*0.08, CONTENT_WIDTH*0.28, CONTENT_WIDTH*0.64],
                      caption="Part III: Field Operations - 4 sections, ~37 KB")

    # === PART IV: REFERENCE ===
    builder.add_section_header("Part IV: Reference")
    builder.add_body_text(
        "The Reference part contains supplementary materials including glossary, "
        "quick reference cards, checklists, detailed application scenarios, and "
        "important safety and ethical considerations.",
        first_paragraph=True
    )

    part4_data = [
        ['#', 'Section', 'Description'],
        ['14', 'Glossary', 'Acronyms and terminology'],
        ['15', 'Quick Reference', 'Printable field card'],
        ['16', 'Field Checklists', 'Pre/on-site/post deployment checklists'],
        ['17', 'Application Scenarios', 'Detailed playbooks for crater, woods, swamp'],
        ['18', 'Future Development', 'Software roadmap, hardware improvements, manufacturing status'],
        ['19', 'Ethics, Legal & Safety', 'Permits, UXO protocols, conductivity monitoring'],
    ]
    builder.add_table(part4_data,
                      col_widths=[CONTENT_WIDTH*0.08, CONTENT_WIDTH*0.28, CONTENT_WIDTH*0.64],
                      caption="Part IV: Reference - 6 sections, ~37 KB")

    builder.add_conditional_page_break(3*inch)

    # === SECTION DEPENDENCY MAP ===
    builder.add_section_header("Section Dependencies")
    builder.add_body_text(
        "The following diagram illustrates the relationships between sections and "
        "recommended reading sequences. Solid arrows indicate primary dependencies "
        "(prerequisite reading), while dashed arrows show cross-references.",
        first_paragraph=True
    )

    print("  Creating dependency map...")
    fig_deps = create_dependency_map()
    builder.add_figure(
        fig_deps,
        "Section dependency map showing primary reading sequences (solid arrows) and "
        "cross-reference relationships (dashed arrows). Colors indicate the part each "
        "section belongs to. The Build sequence (04-09) is typically followed linearly, "
        "while Reference sections (14-19) can be accessed independently as needed.",
        width=CONTENT_WIDTH
    )

    # === READER PATHS ===
    builder.add_section_header("Reader Paths")
    builder.add_body_text(
        "Different users should follow different paths through the documentation "
        "based on their role and objectives:",
        first_paragraph=True
    )

    builder.add_section_header("Path A: System Builder", level=2)
    builder.add_body_text(
        "For building a HIRT system from scratch, follow this sequence: "
        "01 Executive Summary (overview) -> 03 System Architecture (understand design) -> "
        "04 Bill of Materials (order parts) -> 05 Mechanical Design (manufacture components) -> "
        "06 Electronics & Circuits (assemble PCBs) -> 07 Assembly & Wiring (integration) -> "
        "08 Testing & Verification (quality control) -> 09 Calibration (prepare for deployment).",
        first_paragraph=True
    )

    builder.add_section_header("Path B: Field Operator", level=2)
    builder.add_body_text(
        "For operating an existing HIRT system: "
        "15 Quick Reference (keep on hand) -> 16 Field Checklists (pre-deployment) -> "
        "10 Field Operations (detailed procedures) -> 11 Data Recording (data formats) -> "
        "13 Troubleshooting (when issues arise) -> 19 Ethics, Legal & Safety (UXO sites).",
        first_paragraph=True
    )

    builder.add_section_header("Path C: Data Analyst", level=2)
    builder.add_body_text(
        "For processing and interpreting HIRT data: "
        "02 Physics & Theory (understand measurements) -> 11 Data Recording (format specs) -> "
        "12 Data Interpretation (analysis methods) -> 17 Application Scenarios (interpretation context).",
        first_paragraph=True
    )

    builder.add_section_header("Path D: Quick Start", level=2)
    builder.add_body_text(
        "Minimal reading for experienced users: "
        "01 Executive Summary (5 min overview) -> 15 Quick Reference (field card) -> "
        "10 Field Operations (detailed if needed).",
        first_paragraph=True
    )

    builder.add_conditional_page_break(3*inch)

    # === QUICK TOPIC LOOKUP ===
    builder.add_section_header("Quick Topic Lookup")
    builder.add_body_text(
        "Use this reference table to quickly locate information on specific topics:",
        first_paragraph=True
    )

    lookup_data = [
        ['Topic', 'Primary Section', 'Related Sections'],
        ['Coil winding', '05 Mechanical Design', '09 Calibration'],
        ['Current source (ERT)', '06 Electronics', '08 Testing'],
        ['Data formats', '11 Data Recording', '12 Interpretation'],
        ['DDS/TX circuit', '06 Electronics', '09 Calibration'],
        ['Frequency selection', '02 Physics', '10 Field Ops'],
        ['Grid layout', '10 Field Operations', '15 Quick Reference'],
        ['Lock-in detection', '06 Electronics', '02 Physics'],
        ['Part numbers', '04 Bill of Materials', '-'],
        ['PCB layout', '06 Electronics', '07 Assembly'],
        ['Probe insertion', '10 Field Operations', '05 Mechanical'],
        ['QC checklist', '08 Testing', '16 Checklists'],
        ['Reciprocity', '09 Calibration', '12 Interpretation'],
        ['Ring electrodes', '05 Mechanical', '06 Electronics'],
        ['Schematics', '06 Electronics', '04 BOM'],
        ['Skin depth', '02 Physics', '12 Interpretation'],
        ['STL files', '05 Mechanical', '18 Future Dev'],
        ['Time-lapse', '19 Ethics/Safety', '10 Field Ops'],
        ['UXO safety', '19 Ethics/Safety', '10 Field Ops'],
    ]
    builder.add_table(lookup_data,
                      col_widths=[CONTENT_WIDTH*0.28, CONTENT_WIDTH*0.36, CONTENT_WIDTH*0.36],
                      caption="Quick topic lookup for common subjects across the whitepaper")

    # === DOCUMENT CONVENTIONS ===
    builder.add_section_header("Document Conventions")

    builder.add_section_header("File Naming", level=2)
    builder.add_body_text(
        "Section files follow the format <b>XX-topic-name.md</b> where XX is the section "
        "number (00-19). All filenames use lowercase with hyphens.",
        first_paragraph=True
    )

    builder.add_section_header("Cross-References", level=2)
    builder.add_body_text(
        "Internal links use the format [Section Title](XX-filename.md). Section references "
        "in text follow the pattern: 'See Section 10: Field Operations'.",
        first_paragraph=True
    )

    builder.add_section_header("Measurement Units", level=2)
    builder.add_bullet_list([
        "<b>Length:</b> meters (m), millimeters (mm)",
        "<b>Frequency:</b> kilohertz (kHz), hertz (Hz)",
        "<b>Current:</b> milliamps (mA), microamps (uA)",
        "<b>Resistance:</b> ohms, kilohms (k-ohm), megohms (M-ohm)",
        "<b>Conductivity:</b> microsiemens/cm (uS/cm)",
    ])

    builder.add_spacer(12)

    # === VERSION HISTORY ===
    builder.add_section_header("Version History")

    version_data = [
        ['Version', 'Date', 'Changes'],
        ['2.0', '2026-01', 'Consolidated from 33 to 19 sections'],
        ['1.0', '2025-01', 'Complete whitepaper package (33 sections)'],
        ['0.9', '2024-12', 'Manufacturing release (16mm modular design)'],
        ['0.5', '2024-11', 'Initial documentation structure'],
    ]
    builder.add_table(version_data,
                      col_widths=[CONTENT_WIDTH*0.15, CONTENT_WIDTH*0.20, CONTENT_WIDTH*0.65],
                      caption="Document version history")

    builder.add_spacer(12)
    builder.add_note(
        "This is the master index for the HIRT whitepaper v2.0. All 19 sections are "
        "contained in the docs/whitepaper/sections/ directory."
    )

    # Build PDF
    output_path = builder.build()
    print(f"Section 00 PDF created: {output_path}")
    return output_path


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    build_section_00()
