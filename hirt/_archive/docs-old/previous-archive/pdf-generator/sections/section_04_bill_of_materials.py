#!/usr/bin/env python3
"""
HIRT Section 04: Bill of Materials - Publication-Quality PDF Generator

Generates a professional PDF with:
- Cost breakdown pie chart
- Component categories visual
- Comprehensive parts tables
- Procurement guide

Usage:
    python section_04_bill_of_materials.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge
import numpy as np
from io import BytesIO

from lib.pdf_builder import SectionPDFBuilder
from lib.styles import (
    CONTENT_WIDTH, PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING,
    LIGHT_BG, COLOR_PALETTE, ORANGE, PURPLE, GRAY_DARK, GRAY_MED
)


def create_cost_breakdown_pie_chart():
    """
    Create a pie chart showing the cost breakdown for a 25-probe HIRT system.
    Uses mid-range estimates from the BOM.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Cost categories with mid-range estimates
    categories = ['Probes\n(25 units)', 'Base Hub', 'Cables/\nConnectors',
                  'Tools/\nEquipment', 'Consumables']
    costs = [2000, 375, 500, 750, 150]  # Mid-range values
    total = sum(costs)

    # Colors from the HIRT palette
    colors = [
        COLOR_PALETTE['secondary'],   # Probes - blue
        COLOR_PALETTE['success'],     # Base Hub - green
        COLOR_PALETTE['orange'],      # Cables - orange
        COLOR_PALETTE['accent'],      # Tools - light blue
        COLOR_PALETTE['purple']       # Consumables - purple
    ]

    # Explode the largest segment slightly
    explode = (0.05, 0, 0, 0, 0)

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        costs,
        explode=explode,
        labels=categories,
        colors=colors,
        autopct=lambda pct: f'${int(pct/100*total):,}\n({pct:.1f}%)',
        startangle=90,
        pctdistance=0.65,
        labeldistance=1.15
    )

    # Style the text
    for text in texts:
        text.set_fontsize(9)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Add center circle for donut effect
    center_circle = plt.Circle((0, 0), 0.35, fc='white')
    ax.add_patch(center_circle)

    # Add total in center
    ax.text(0, 0.05, f'Total', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLOR_PALETTE['primary'])
    ax.text(0, -0.12, f'~$3,775', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLOR_PALETTE['primary'])

    ax.set_title('25-Probe HIRT System Cost Distribution',
                 fontsize=12, fontweight='bold', color=COLOR_PALETTE['primary'], pad=15)

    # Add note about cost range
    ax.text(0, -1.5, 'Estimated range: $2,800 - $4,750 depending on sourcing',
            ha='center', va='center', fontsize=8, style='italic',
            color=COLOR_PALETTE['text_muted'], transform=ax.transData)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf


def create_component_categories_diagram():
    """
    Create a visual diagram showing the component categories and their relationships.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 7.5, 'HIRT Component Categories', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLOR_PALETTE['primary'])

    # Define category boxes
    categories = [
        {
            'name': 'Per-Probe\nComponents',
            'x': 1.5, 'y': 5.5, 'width': 3, 'height': 1.8,
            'color': COLOR_PALETTE['secondary'],
            'items': ['Mechanical (rod, tip)', 'Coils (ferrite, wire)',
                     'ERT rings/collars', 'Cable/connector']
        },
        {
            'name': 'Base Hub\nComponents',
            'x': 5.5, 'y': 5.5, 'width': 3, 'height': 1.8,
            'color': COLOR_PALETTE['success'],
            'items': ['Power system', 'ERT current source',
                     'Sync/clock', 'Control MCU']
        },
        {
            'name': 'Shared\nEquipment',
            'x': 1.5, 'y': 2.5, 'width': 3, 'height': 1.8,
            'color': COLOR_PALETTE['accent'],
            'items': ['Test equipment', 'Tools',
                     'Consumables', 'Bulk cables']
        },
        {
            'name': 'Optional\nActive Electronics',
            'x': 5.5, 'y': 2.5, 'width': 3, 'height': 1.8,
            'color': COLOR_PALETTE['orange'],
            'items': ['MCU (ESP32)', 'DDS signal gen',
                     'Preamp/ADC', 'Custom PCB']
        }
    ]

    for cat in categories:
        # Main box
        box = FancyBboxPatch(
            (cat['x'], cat['y'] - cat['height']/2),
            cat['width'], cat['height'],
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor=cat['color'],
            edgecolor='white',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(box)

        # Category name
        ax.text(cat['x'] + cat['width']/2, cat['y'] + 0.5,
                cat['name'], ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

        # Item list
        item_text = '\n'.join([f'  {item}' for item in cat['items']])
        ax.text(cat['x'] + 0.15, cat['y'] - 0.1, item_text,
                ha='left', va='top', fontsize=7, color='white',
                linespacing=1.3)

    # Cost annotations
    cost_info = [
        ('1.5', '4.7', '$60-100\nper probe'),
        ('5.5', '4.7', '$300-450\ntotal'),
        ('1.5', '1.7', '$500-1,000\none-time'),
        ('5.5', '1.7', '+$50-80\nper probe')
    ]

    for x, y, text in cost_info:
        ax.text(float(x) + 1.5, float(y), text, ha='center', va='top',
                fontsize=8, fontweight='bold', color=COLOR_PALETTE['primary'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=COLOR_PALETTE['gray_med'], alpha=0.9))

    # Connection arrows
    arrow_style = dict(arrowstyle='->', color=COLOR_PALETTE['gray_dark'],
                      lw=1.5, mutation_scale=15)

    # Probes to Hub connection
    ax.annotate('', xy=(5.5, 5.5), xytext=(4.5, 5.5),
                arrowprops=dict(arrowstyle='<->', color=COLOR_PALETTE['gray_dark'], lw=2))
    ax.text(5, 5.9, 'Connects via\ncables', ha='center', va='bottom', fontsize=7,
            color=COLOR_PALETTE['gray_dark'])

    # Legend/note at bottom
    ax.text(5, 0.5,
            'Passive probes: Lower cost, simpler assembly, recommended for most applications\n'
            'Active probes: Higher performance, more complex, for demanding applications',
            ha='center', va='center', fontsize=8, style='italic',
            color=COLOR_PALETTE['text_muted'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_PALETTE['light_bg'],
                     edgecolor='none'))

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf


def build_section_04_pdf():
    """
    Build the complete Section 04: Bill of Materials PDF.
    """
    # Initialize builder
    builder = SectionPDFBuilder(
        section_num=4,
        title="Bill of Materials"
    )

    # Title block
    builder.add_title_block(
        subtitle="Complete Parts List for HIRT System Construction",
        version="2.0"
    )

    # === Overview ===
    builder.add_section_header("4.1 Overview", level=1)
    builder.add_body_text(
        "This section provides comprehensive bills of materials (BOMs) for building "
        "the HIRT system, including per-probe components, base hub components, and "
        "shared equipment. The micro-probe design (16mm OD, passive probes) minimizes "
        "cost while maintaining measurement quality.",
        first_paragraph=True
    )
    builder.add_body_text(
        "The modular architecture allows builders to scale the system from a minimal "
        "single-probe test setup to full 25+ probe arrays. Component selection prioritizes "
        "availability, cost-effectiveness, and ease of assembly while meeting performance "
        "specifications."
    )

    # === Cost Summary ===
    builder.add_section_header("4.2 Cost Summary", level=1)

    builder.add_section_header("System Cost Estimate (25-Probe Array)", level=2)
    builder.add_body_text(
        "The following table presents the estimated cost ranges for a complete 25-probe "
        "HIRT system. Actual costs may vary based on sourcing, quantity discounts, and "
        "regional availability.",
        first_paragraph=True
    )

    # System cost table
    system_cost_data = [
        ['Category', 'Cost Range', 'Notes'],
        ['Probes (24 units)', '$1,400-2,400', 'Passive design'],
        ['Central Hub', '$200-300', 'Electronics + Enclosure'],
        ['Zone Wiring', '$150-300', 'Hubs + Trunk Cables'],
        ['Tools/Equipment', '$100-200', 'Assembly tools'],
        ['Consumables', '$50-100', 'Epoxy, solder, etc.'],
        ['Total', '$1,800-3,900', '95%+ Savings']
    ]
    builder.add_table(
        system_cost_data,
        col_widths=[CONTENT_WIDTH * 0.35, CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.40],
        caption="24-Probe HIRT System Cost Estimate"
    )

    # Cost breakdown pie chart
    builder.add_spacer(12)
    pie_chart = create_cost_breakdown_pie_chart()
    builder.add_figure(
        pie_chart,
        "Cost distribution for a 25-probe HIRT system showing the relative investment "
        "in each component category. Probes represent the largest expense due to quantity.",
        width=CONTENT_WIDTH * 0.85
    )

    # Cost per probe breakdown
    builder.add_section_header("Cost Per Probe Breakdown", level=2)
    builder.add_body_text(
        "Each passive probe consists of mechanical, sensing, and interconnection components. "
        "The following breakdown shows typical costs for a single probe unit.",
        first_paragraph=True
    )

    probe_cost_data = [
        ['Component', 'Cost'],
        ['Mechanical (rod, tip, coupler)', '$35-55'],
        ['ERT (rings, collars)', '$5-10'],
        ['Coils (ferrite, wire)', '$10-15'],
        ['Short Cable (1m)', '$10-15'],
        ['Hardware (epoxy, o-rings)', '$5-10'],
        ['Total per probe', '$70-120']
    ]
    builder.add_table(
        probe_cost_data,
        col_widths=[CONTENT_WIDTH * 0.6, CONTENT_WIDTH * 0.4],
        caption="Per-Probe Cost Breakdown (Passive Design)"
    )

    # Cost reduction tips
    builder.add_section_header("Cost Reduction Options", level=2)
    builder.add_bullet_list([
        "<b>Bulk ordering:</b> 20%+ savings on quantities >50 units",
        "<b>Local sourcing:</b> Reduce shipping costs for common components",
        "<b>Simpler design:</b> Passive probes vs active for significant savings",
        "<b>DIY coils:</b> Wind your own coils vs purchasing pre-wound",
        "<b>Generic parts:</b> Use non-branded components where precision is not critical"
    ])

    # === Component Categories Diagram ===
    builder.add_spacer(12)
    categories_diagram = create_component_categories_diagram()
    builder.add_figure(
        categories_diagram,
        "HIRT component categories showing the relationship between per-probe components, "
        "base hub electronics, shared equipment, and optional active probe upgrades.",
        width=CONTENT_WIDTH
    )

    # === Per-Probe BOM ===
    builder.add_page_break()
    builder.add_section_header("4.3 Per-Probe Bill of Materials", level=1)

    # Mechanical Components
    builder.add_section_header("Mechanical Components", level=2)
    mechanical_data = [
        ['Ref', 'Component', 'Description', 'Qty', 'Unit Cost'],
        ['ROD1', 'Fiberglass Tube', '16mm OD x 12mm ID x 1.5m', '2', '$15-25 ea'],
        ['TIP1', 'Probe Tip', '3D printed PETG', '1', '$1-2'],
        ['CPL1', 'Rod Coupler', '3D printed/CNC', '1', '$2-5'],
        ['JB1', 'Junction Box', '3D printed PETG', '1', '$3-5']
    ]
    builder.add_table(
        mechanical_data,
        col_widths=[CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.18,
                   CONTENT_WIDTH * 0.40, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.26],
        caption="Per-Probe Mechanical Components"
    )

    # Coil Components
    builder.add_section_header("Coil Components", level=2)
    coil_data = [
        ['Ref', 'Component', 'Description', 'Qty', 'Unit Cost'],
        ['L1', 'Ferrite Rod', '6-8mm x 40-80mm MnZn', '1-2', '$2-5'],
        ['W1', 'Magnet Wire', '34-38 AWG, 50m', '1', '$3-5']
    ]
    builder.add_table(
        coil_data,
        col_widths=[CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.18,
                   CONTENT_WIDTH * 0.40, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.26],
        caption="Per-Probe Coil Components"
    )

    # ERT Components
    builder.add_section_header("ERT Components", level=2)
    ert_data = [
        ['Ref', 'Component', 'Description', 'Qty', 'Unit Cost'],
        ['R1-R3', 'ERT Rings', 'Stainless steel 3-5mm bands', '2-3', '$1-2 ea'],
        ['C1-C3', 'Ring Collars', '3D printed PETG', '2-3', '$0.50 ea']
    ]
    builder.add_table(
        ert_data,
        col_widths=[CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.18,
                   CONTENT_WIDTH * 0.40, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.26],
        caption="Per-Probe ERT Components"
    )

    # Hardware
    builder.add_section_header("Hardware", level=2)
    hardware_data = [
        ['Ref', 'Component', 'Description', 'Qty', 'Unit Cost'],
        ['HW1', 'O-Rings', 'M12 size, nitrile', '4', '$0.50 ea'],
        ['HW2', 'Epoxy', '2-part structural', '-', '$5/probe']
    ]
    builder.add_table(
        hardware_data,
        col_widths=[CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.18,
                   CONTENT_WIDTH * 0.40, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.26],
        caption="Per-Probe Hardware"
    )

    # Cable and Connectors
    builder.add_section_header("Cable and Connectors", level=2)
    cable_data = [
        ['Ref', 'Component', 'Description', 'Qty', 'Unit Cost'],
        ['CBL1', 'Shielded Cable', '6-conductor, 3-5m', '1', '$10-15'],
        ['CON1', 'Connector', '12-pin Phoenix', '1', '$5-8']
    ]
    builder.add_table(
        cable_data,
        col_widths=[CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.18,
                   CONTENT_WIDTH * 0.40, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.26],
        caption="Per-Probe Cable and Connectors"
    )

    builder.add_info_box("Total Per Passive Probe", [
        "Estimated cost: ~$60-100 per passive probe",
        "Assembly time: 2-4 hours per probe",
        "Required skills: Basic soldering, 3D printing, mechanical assembly"
    ])

    # === Active Probe Electronics (Optional) ===
    builder.add_section_header("Active Probe Electronics (Optional)", level=2)
    builder.add_body_text(
        "For applications requiring higher performance, active probes with in-probe "
        "electronics can be built. These include signal generation, amplification, "
        "and digitization circuitry.",
        first_paragraph=True
    )

    active_data = [
        ['Ref', 'Component', 'Part Number', 'Qty', 'Unit Cost'],
        ['U1', 'MCU', 'ESP32-WROOM-32', '1', '$5-8'],
        ['U2', 'DDS', 'AD9833BRMZ', '1', '$8-12'],
        ['U3', 'TX Op-Amp', 'OPA454AIDDAR', '1', '$6-10'],
        ['U4', 'RX Preamp', 'AD620ARZ', '1', '$6-10'],
        ['U5', 'Inst Amp', 'INA128PAG4', '1', '$6-10'],
        ['U6', 'ADC', 'ADS1256IDBR', '1', '$10-15'],
        ['U7', 'Mux', 'CD4051BE', '1', '$1-2'],
        ['U8', 'LDO', 'AMS1117-3.3', '1', '$0.50'],
        ['PCB', 'Custom PCB', '-', '1', '$5-10'],
        ['-', 'Passives', 'Resistors, caps', '-', '$5']
    ]
    builder.add_table(
        active_data,
        col_widths=[CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.15,
                   CONTENT_WIDTH * 0.35, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.34],
        caption="Active Probe Electronics (Optional Add-on)"
    )

    builder.add_note(
        "Additional cost per active probe: ~$50-80. Active probes provide improved "
        "signal-to-noise ratio and reduced cable effects but require more complex assembly."
    )

    # === Base Hub BOM ===
    builder.add_page_break()
    builder.add_section_header("4.4 Base Hub Bill of Materials", level=1)
    builder.add_body_text(
        "The base hub provides power distribution, synchronization, and data collection "
        "for all connected probes. One hub supports up to 25 probes.",
        first_paragraph=True
    )

    # Power System
    builder.add_section_header("Power System", level=2)
    power_data = [
        ['Ref', 'Component', 'Part Number', 'Qty', 'Unit Cost'],
        ['BAT1', 'Battery', '12V 12Ah LiFePO4', '1', '$60-100'],
        ['F1', 'Fuse Holder', '0287005.PXCN', '1', '$3'],
        ['F2', 'Fuse', '5A fast-blow', '5', '$1 ea'],
        ['REG1', '5V Regulator', 'LM2596 Module', '1', '$3-5'],
        ['REG2', '3.3V Regulator', 'AMS1117-3.3', '1', '$0.50'],
        ['SW1', 'Power Switch', 'DPST 10A', '1', '$3-5'],
        ['TB1', 'Terminal Block', 'Multi-position', '1', '$10-15']
    ]
    builder.add_table(
        power_data,
        col_widths=[CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.17,
                   CONTENT_WIDTH * 0.33, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.34],
        caption="Base Hub Power System Components"
    )

    # ERT Current Source
    builder.add_section_header("ERT Current Source", level=2)
    ert_source_data = [
        ['Ref', 'Component', 'Part Number', 'Qty', 'Unit Cost'],
        ['U1', 'Voltage Ref', 'REF5025AIDGKR', '1', '$4-6'],
        ['U2', 'Op-Amp', 'OPA277PAG4', '1', '$4-6'],
        ['U3', 'Inst Amp', 'INA128PAG4', '1', '$6-10'],
        ['K1', 'Relay', 'G5V-2-H1', '1', '$3-5'],
        ['R1-R4', 'Precision R', '0.1% various', '10', '$0.50 ea'],
        ['R5', 'Sense R', '10 ohm 0.1%', '1', '$1']
    ]
    builder.add_table(
        ert_source_data,
        col_widths=[CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.17,
                   CONTENT_WIDTH * 0.33, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.34],
        caption="ERT Current Source Components"
    )

    # Sync/Clock Distribution
    builder.add_section_header("Sync/Clock Distribution", level=2)
    sync_data = [
        ['Ref', 'Component', 'Part Number', 'Qty', 'Unit Cost'],
        ['Y1', 'Oscillator', 'ECS-100-10-30B-TR', '1', '$3-5'],
        ['U1-U3', 'Buffer', 'SN74HC244N', '3', '$1 ea']
    ]
    builder.add_table(
        sync_data,
        col_widths=[CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.17,
                   CONTENT_WIDTH * 0.33, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.34],
        caption="Sync/Clock Distribution Components"
    )

    # Communication
    builder.add_section_header("Communication", level=2)
    comm_data = [
        ['Ref', 'Component', 'Part Number', 'Qty', 'Unit Cost'],
        ['U1', 'RS485', 'MAX485ESA+', '1', '$2-4'],
        ['U2', 'USB-Serial', 'CP2102 Module', '1', '$3-5'],
        ['J1', 'RJ45 Jack', '-', '1', '$2']
    ]
    builder.add_table(
        comm_data,
        col_widths=[CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.17,
                   CONTENT_WIDTH * 0.33, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.34],
        caption="Communication Interface Components"
    )

    # Control
    builder.add_section_header("Control", level=2)
    control_data = [
        ['Ref', 'Component', 'Part Number', 'Qty', 'Unit Cost'],
        ['U1', 'MCU', 'ESP32 DevKit', '1', '$8-12'],
        ['U2', 'ADC', 'ADS1256IDBR', '1', '$10-15'],
        ['SD1', 'SD Card', 'Micro SD module', '1', '$3-5']
    ]
    builder.add_table(
        control_data,
        col_widths=[CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.17,
                   CONTENT_WIDTH * 0.33, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.34],
        caption="Control System Components"
    )

    # Enclosure and Connectors
    builder.add_section_header("Enclosure and Connectors", level=2)
    enclosure_data = [
        ['Ref', 'Component', 'Description', 'Qty', 'Unit Cost'],
        ['ENC1', 'Enclosure', 'IP65 200x150x100mm', '1', '$30-50'],
        ['PG1-PG20', 'Cable Glands', 'PG11 or M20', '20', '$1 ea'],
        ['CON1-CON20', 'Probe Connectors', '12-pin Phoenix', '20', '$5 ea']
    ]
    builder.add_table(
        enclosure_data,
        col_widths=[CONTENT_WIDTH * 0.12, CONTENT_WIDTH * 0.18,
                   CONTENT_WIDTH * 0.30, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.32],
        caption="Enclosure and Connector Components"
    )

    builder.add_info_box("Total Base Hub Cost", [
        "Estimated cost: ~$300-450 for complete base hub",
        "Assembly time: 8-12 hours",
        "Testing and calibration: Additional 4-6 hours"
    ])

    # === Shared Equipment BOM ===
    builder.add_page_break()
    builder.add_section_header("4.5 Shared Equipment Bill of Materials", level=1)

    # Connectors and Cables
    builder.add_section_header("Connectors and Cables", level=2)
    shared_cables_data = [
        ['Ref', 'Component', 'Description', 'Qty', 'Unit Cost'],
        ['CBL1', 'Probe Cable', 'Belden 3066A 12-pair, 5m', '20', '$15 ea'],
        ['CBL2', 'Power Cable', '14 AWG 2-conductor', '10m', '$10'],
        ['CON1', 'Phoenix Headers', '12-pos pluggable', '20', '$5 ea'],
        ['CON2', 'DC Jack', '5.5x2.1mm panel', '1', '$2']
    ]
    builder.add_table(
        shared_cables_data,
        col_widths=[CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.18,
                   CONTENT_WIDTH * 0.38, CONTENT_WIDTH * 0.08, CONTENT_WIDTH * 0.28],
        caption="Shared Cables and Connectors"
    )

    # Test Equipment
    builder.add_section_header("Test Equipment (Recommended)", level=2)
    test_equip_data = [
        ['Item', 'Description', 'Est. Cost', 'Notes'],
        ['DMM', 'Digital Multimeter', '$50-100', 'Fluke or equivalent'],
        ['LCR', 'LCR Meter', '$100-300', 'For coil testing'],
        ['Scope', 'Oscilloscope', '$300-500', '2-ch, 50MHz min'],
        ['PS', 'Bench Power Supply', '$50-100', 'Adjustable, current limit']
    ]
    builder.add_table(
        test_equip_data,
        col_widths=[CONTENT_WIDTH * 0.12, CONTENT_WIDTH * 0.28,
                   CONTENT_WIDTH * 0.20, CONTENT_WIDTH * 0.40],
        caption="Recommended Test Equipment"
    )

    # Tools
    builder.add_section_header("Tools", level=2)
    tools_data = [
        ['Item', 'Description', 'Est. Cost'],
        ['Soldering', 'Iron + solder', '$50-100'],
        ['Tap/Die', 'M12x1.75 set', '$30-50'],
        ['Crimpers', 'For connectors', '$30-50'],
        ['Heat Gun', 'For shrink tubing', '$30-50'],
        ['Hand Tools', 'Screwdrivers, pliers', '$50']
    ]
    builder.add_table(
        tools_data,
        col_widths=[CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.45, CONTENT_WIDTH * 0.30],
        caption="Required Tools"
    )

    # Consumables
    builder.add_section_header("Consumables", level=2)
    consumables_data = [
        ['Item', 'Description', 'Est. Cost'],
        ['Solder', '60/40 or lead-free', '$15'],
        ['Flux', 'Rosin flux', '$10'],
        ['Heat Shrink', 'Assorted sizes', '$15'],
        ['Epoxy', '2-part structural', '$20'],
        ['Cable Ties', 'Assorted', '$10'],
        ['IPA', 'Isopropyl alcohol', '$10']
    ]
    builder.add_table(
        consumables_data,
        col_widths=[CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.45, CONTENT_WIDTH * 0.30],
        caption="Consumables"
    )

    # === Procurement Guide ===
    builder.add_page_break()
    builder.add_section_header("4.6 Procurement Guide", level=1)

    # Recommended Suppliers
    builder.add_section_header("Recommended Suppliers", level=2)

    builder.add_section_header("Electronics", level=3)
    builder.add_bullet_list([
        "<b>DigiKey</b> (www.digikey.com) - Wide selection, fast shipping",
        "<b>Mouser</b> (www.mouser.com) - Good for precision components",
        "<b>Newark</b> (www.newark.com) - Alternative source"
    ])

    builder.add_section_header("Mechanical", level=3)
    builder.add_bullet_list([
        "<b>McMaster-Carr</b> (www.mcmaster.com) - Hardware, tubing",
        "<b>Grainger</b> (www.grainger.com) - Industrial supplies",
        "<b>Amazon</b> - General supplies"
    ])

    builder.add_section_header("3D Printing", level=3)
    builder.add_bullet_list([
        "Local print shop",
        "<b>Shapeways</b> (www.shapeways.com) - Online service",
        "<b>JLCPCB</b> (www.jlcpcb.com) - Also offers 3D printing"
    ])

    builder.add_section_header("PCB Fabrication", level=3)
    builder.add_bullet_list([
        "<b>JLCPCB</b> (www.jlcpcb.com) - Low cost, fast",
        "<b>PCBWay</b> (www.pcbway.com) - Good quality",
        "<b>OSH Park</b> (oshpark.com) - US-based, quality"
    ])

    # Key Part Numbers Reference
    builder.add_section_header("Key Part Numbers Reference", level=2)
    part_numbers_data = [
        ['Component', 'DigiKey PN', 'Mouser PN'],
        ['AD9833BRMZ', 'AD9833BRMZ-REEL', '584-AD9833BRMZ'],
        ['AD620ARZ', 'AD620ARZ-ND', '584-AD620ARZ'],
        ['INA128PAG4', 'INA128PAG4-ND', '595-INA128PAG4'],
        ['ADS1256IDBR', 'ADS1256IDBR-ND', '595-ADS1256IDBR'],
        ['OPA454AIDDAR', 'OPA454AIDDAR-ND', '595-OPA454AIDDAR'],
        ['REF5025AIDGKR', 'REF5025AIDGKR-ND', '595-REF5025AIDGKR'],
        ['ESP32-WROOM-32', '1904-1009-1-ND', '356-ESP32-WROOM-32']
    ]
    builder.add_table(
        part_numbers_data,
        col_widths=[CONTENT_WIDTH * 0.35, CONTENT_WIDTH * 0.35, CONTENT_WIDTH * 0.30],
        caption="Key Part Numbers for Major Suppliers"
    )

    # Procurement Tips
    builder.add_section_header("Procurement Tips", level=2)
    builder.add_numbered_list([
        "<b>Order extras:</b> Add 10-20% for spares and mistakes",
        "<b>Check MOQ:</b> Some parts have minimum order quantities",
        "<b>Lead times:</b> Check availability before ordering",
        "<b>Substitutes:</b> Have backup part numbers identified",
        "<b>Consolidate:</b> Combine orders to reduce shipping costs"
    ])

    # === Alternative Components ===
    builder.add_section_header("4.7 Alternative Components", level=1)
    builder.add_body_text(
        "The following tables provide alternative component options for situations "
        "where primary components are unavailable or when different performance "
        "characteristics are desired.",
        first_paragraph=True
    )

    # Coil Alternatives
    builder.add_section_header("Coil Alternatives", level=2)
    coil_alt_data = [
        ['Original', 'Alternative', 'Notes'],
        ['6-8mm ferrite rod', '10mm rod', 'Larger = more signal, larger diameter'],
        ['30 AWG magnet wire', '28-34 AWG', 'Trade-off: turns vs resistance']
    ]
    builder.add_table(
        coil_alt_data,
        col_widths=[CONTENT_WIDTH * 0.30, CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.45],
        caption="Coil Component Alternatives"
    )

    # ERT Ring Alternatives
    builder.add_section_header("ERT Ring Alternatives", level=2)
    ert_alt_data = [
        ['Original', 'Alternative', 'Notes'],
        ['Stainless steel band', 'Copper tape', 'Lower cost, easier to work with'],
        ['3D printed collar', 'Heat shrink tube', 'Simpler mounting']
    ]
    builder.add_table(
        ert_alt_data,
        col_widths=[CONTENT_WIDTH * 0.30, CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.45],
        caption="ERT Ring Alternatives"
    )

    # Electronics Alternatives
    builder.add_section_header("Electronics Alternatives", level=2)
    elec_alt_data = [
        ['Original', 'Alternative', 'Notes'],
        ['AD9833 DDS', 'Si5351', 'More outputs, different interface'],
        ['AD620 preamp', 'INA217', 'Different specifications'],
        ['ADS1256 ADC', 'ADS1115', 'Lower resolution, lower cost']
    ]
    builder.add_table(
        elec_alt_data,
        col_widths=[CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.50],
        caption="Electronics Alternatives"
    )

    # Cross-references
    builder.add_spacer(12)
    builder.add_horizontal_rule()
    builder.add_note(
        "For assembly procedures, see Section 7: Assembly and Wiring. "
        "For mechanical specifications, see Section 5: Mechanical Design. "
        "For electronics schematics, see Section 6: Electronics and Circuits."
    )

    # Build the PDF
    output_path = builder.build()
    return output_path


if __name__ == "__main__":
    print("Generating HIRT Section 04: Bill of Materials PDF...")
    output = build_section_04_pdf()
    print(f"PDF generated successfully: {output}")
