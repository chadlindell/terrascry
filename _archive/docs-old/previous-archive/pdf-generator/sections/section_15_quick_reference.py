#!/usr/bin/env python3
"""
HIRT Section 15: Quick Reference Card - PDF Generator

Creates a compact, publication-quality reference document with:
- Quick setup flowchart diagram
- Key specifications visual table
- Color-coded reference tables for field use

Usage:
    python section_15_quick_reference.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch
import numpy as np
from io import BytesIO

from reportlab.lib.units import inch
from reportlab.platypus import Spacer, Table, TableStyle, KeepTogether, Paragraph
from reportlab.lib import colors
from reportlab.lib.colors import HexColor

from lib import (
    SectionPDFBuilder, CONTENT_WIDTH, PRIMARY, SECONDARY, ACCENT,
    SUCCESS, WARNING, LIGHT_BG, COLOR_PALETTE
)
from lib.diagrams import save_figure_to_buffer, COLORS


# ============================================================================
# DIAGRAM 1: QUICK SETUP FLOWCHART
# ============================================================================
def create_quick_setup_diagram():
    """
    Create a compact visual flowchart of the power-up sequence.
    Shows the 7-step process from cable connection to measurement start.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 4.7, 'HIRT Quick Setup Sequence', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['primary'])

    # Define step positions (compact 2-row layout)
    step_positions = [
        (1.2, 3.3),   # Step 1
        (3.6, 3.3),   # Step 2
        (6.0, 3.3),   # Step 3
        (8.4, 3.3),   # Step 4
        (2.4, 1.2),   # Step 5
        (5.0, 1.2),   # Step 6
        (7.6, 1.2),   # Step 7
    ]

    step_labels = [
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
    ]

    step_texts = [
        'Connect\nTrunk Cables',
        'Connect\nProbes',
        'Power ON\nBase Hub',
        'Wait 15s\nfor Init',
        'Verify\n"Ready" LED',
        'Run System\nDiagnostic',
        'Start\nMeasurement',
    ]

    step_icons = ['cable', 'probe', 'power', 'timer', 'led', 'check', 'play']

    # Color scheme for steps
    step_colors = [
        COLORS['secondary'],   # Cable
        COLORS['secondary'],   # Probe
        COLORS['success'],     # Power
        COLORS['orange'],      # Wait
        COLORS['accent'],      # LED
        COLORS['accent'],      # Diagnostic
        COLORS['success'],     # Start
    ]

    # Draw steps
    box_width = 1.8
    box_height = 1.4

    for i, ((x, y), label, text, icon, color) in enumerate(
            zip(step_positions, step_labels, step_texts, step_icons, step_colors)):

        # Draw box
        box = FancyBboxPatch((x - box_width/2, y - box_height/2), box_width, box_height,
                             boxstyle="round,pad=0.03,rounding_size=0.15",
                             facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(box)

        # Step number circle
        circle = Circle((x - box_width/2 + 0.2, y + box_height/2 - 0.2), 0.18,
                        facecolor=color, edgecolor='white', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x - box_width/2 + 0.2, y + box_height/2 - 0.2, label,
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        # Draw icon representation
        icon_y = y + 0.15
        if icon == 'cable':
            # Cable connector icon
            ax.plot([x-0.3, x+0.3], [icon_y, icon_y], color=color, lw=3, solid_capstyle='round')
            ax.plot([x-0.15, x-0.15], [icon_y-0.1, icon_y+0.1], color=color, lw=2)
            ax.plot([x+0.15, x+0.15], [icon_y-0.1, icon_y+0.1], color=color, lw=2)
        elif icon == 'probe':
            # Probe icon
            ax.add_patch(Rectangle((x-0.08, icon_y-0.25), 0.16, 0.5,
                                   facecolor=color, edgecolor='none'))
            ax.add_patch(Circle((x, icon_y+0.25), 0.12, facecolor='white', edgecolor=color, lw=1.5))
        elif icon == 'power':
            # Power symbol
            ax.add_patch(Circle((x, icon_y), 0.22, facecolor='none', edgecolor=color, lw=2))
            ax.plot([x, x], [icon_y-0.05, icon_y+0.25], color=color, lw=2.5)
        elif icon == 'timer':
            # Timer/clock icon
            ax.add_patch(Circle((x, icon_y), 0.22, facecolor='none', edgecolor=color, lw=2))
            ax.plot([x, x], [icon_y, icon_y+0.15], color=color, lw=2)
            ax.plot([x, x+0.12], [icon_y, icon_y], color=color, lw=2)
        elif icon == 'led':
            # LED indicator
            ax.add_patch(Circle((x, icon_y), 0.2, facecolor=COLORS['success'],
                                edgecolor='white', lw=2))
            ax.text(x, icon_y, 'OK', ha='center', va='center', fontsize=6,
                    fontweight='bold', color='white')
        elif icon == 'check':
            # Checkmark
            ax.plot([x-0.15, x-0.02, x+0.2], [icon_y, icon_y-0.15, icon_y+0.2],
                    color=color, lw=3, solid_capstyle='round', solid_joinstyle='round')
        elif icon == 'play':
            # Play triangle
            triangle = Polygon([(x-0.12, icon_y-0.18), (x-0.12, icon_y+0.18), (x+0.2, icon_y)],
                              facecolor=color, edgecolor='none')
            ax.add_patch(triangle)

        # Step text
        ax.text(x, y - 0.45, text, ha='center', va='center', fontsize=8,
                color=COLORS['gray_dark'], linespacing=1.2)

    # Draw arrows between steps
    arrow_style = mpatches.ArrowStyle('->', head_length=0.15, head_width=0.1)

    # Top row arrows (1->2->3->4)
    for i in range(3):
        x1, y1 = step_positions[i]
        x2, y2 = step_positions[i + 1]
        arrow = FancyArrowPatch((x1 + box_width/2 + 0.08, y1),
                                (x2 - box_width/2 - 0.08, y2),
                                arrowstyle=arrow_style, color=COLORS['gray_med'],
                                lw=1.5, mutation_scale=12)
        ax.add_patch(arrow)

    # Arrow from step 4 down to step 5 (curved)
    ax.annotate('', xy=(step_positions[4][0] + box_width/2, step_positions[4][1] + 0.1),
                xytext=(step_positions[3][0], step_positions[3][1] - box_height/2 - 0.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_med'],
                               connectionstyle='arc3,rad=0.3', lw=1.5))

    # Bottom row arrows (5->6->7)
    for i in range(4, 6):
        x1, y1 = step_positions[i]
        x2, y2 = step_positions[i + 1]
        arrow = FancyArrowPatch((x1 + box_width/2 + 0.08, y1),
                                (x2 - box_width/2 - 0.08, y2),
                                arrowstyle=arrow_style, color=COLORS['gray_med'],
                                lw=1.5, mutation_scale=12)
        ax.add_patch(arrow)

    # Add timing note
    ax.text(5, 0.15, 'Total Setup Time: ~2-3 minutes (excluding probe installation)',
            ha='center', va='center', fontsize=9, style='italic', color=COLORS['gray_dark'])

    plt.tight_layout()
    return save_figure_to_buffer(fig, dpi=200)


# ============================================================================
# DIAGRAM 2: KEY SPECIFICATIONS VISUAL
# ============================================================================
def create_key_specs_visual():
    """
    Create a visual summary of key HIRT specifications organized by category.
    Shows MIT parameters, ERT parameters, and grid specifications.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'HIRT Key Specifications at a Glance', ha='center', va='center',
            fontsize=14, fontweight='bold', color=COLORS['primary'])

    # Define three main categories with their specs
    categories = [
        {
            'title': 'MIT Parameters',
            'color': COLORS['accent'],
            'x': 1.7,
            'specs': [
                ('Frequency', '2-50 kHz'),
                ('TX Current', '10-50 mA'),
                ('Integration', '1-5 sec'),
            ]
        },
        {
            'title': 'ERT Parameters',
            'color': COLORS['orange'],
            'x': 5.0,
            'specs': [
                ('Injection', '0.5-2 mA'),
                ('Ring Positions', '0.5,  1.5,  2.5 m'),
                ('Polarity Rev.', 'Every 2 sec'),
            ]
        },
        {
            'title': 'Grid Layout',
            'color': COLORS['success'],
            'x': 8.3,
            'specs': [
                ('Spacing', '2 meters'),
                ('Standard Grid', '5 x 5  (25 probes)'),
                ('Probe Depth', '2.5-3 m'),
            ]
        },
    ]

    box_width = 3.4
    box_height = 3.0
    top_y = 4.8

    for cat in categories:
        x = cat['x']

        # Category header box
        header_height = 0.6
        header = FancyBboxPatch((x - box_width/2, top_y - header_height),
                                box_width, header_height,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=cat['color'], edgecolor='none')
        ax.add_patch(header)
        ax.text(x, top_y - header_height/2, cat['title'],
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')

        # Specs box
        specs_box = FancyBboxPatch((x - box_width/2, top_y - box_height),
                                   box_width, box_height - header_height,
                                   boxstyle="round,pad=0.02,rounding_size=0.1",
                                   facecolor='white', edgecolor=cat['color'], linewidth=1.5)
        ax.add_patch(specs_box)

        # Draw specs - stacked layout (label on top, value below)
        spec_y = top_y - header_height - 0.4
        for param, value in cat['specs']:
            # Parameter name (centered, smaller)
            ax.text(x, spec_y, param + ':',
                    ha='center', va='center', fontsize=8, color=COLORS['gray_dark'])
            # Value below (centered, bold)
            ax.text(x, spec_y - 0.32, value,
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color=COLORS['primary'])
            spec_y -= 0.78

    # Draw mini grid visualization
    grid_x, grid_y = 5, 0.7
    grid_size = 1.0
    spacing = grid_size / 4

    # Grid background
    ax.add_patch(Rectangle((grid_x - grid_size/2 - 0.1, grid_y - grid_size/2 - 0.1),
                           grid_size + 0.2, grid_size + 0.2,
                           facecolor=COLORS['ground_tan'], alpha=0.3, edgecolor='none'))

    # Draw 5x5 probe dots
    for i in range(5):
        for j in range(5):
            px = grid_x - grid_size/2 + i * spacing
            py = grid_y - grid_size/2 + j * spacing
            ax.add_patch(Circle((px, py), 0.035, facecolor=COLORS['secondary'],
                                edgecolor='white', lw=0.5))

    ax.text(grid_x, grid_y - grid_size/2 - 0.28, '5x5 Grid Layout (25 probes)',
            ha='center', va='center', fontsize=8, color=COLORS['gray_dark'])

    # Dimension labels
    ax.annotate('', xy=(grid_x + grid_size/2 + 0.05, grid_y - grid_size/2),
                xytext=(grid_x - grid_size/2 - 0.05, grid_y - grid_size/2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray_med'], lw=1))
    ax.text(grid_x, grid_y - grid_size/2 - 0.12, '10m', ha='center', va='top',
            fontsize=7, color=COLORS['gray_dark'])

    plt.tight_layout()
    return save_figure_to_buffer(fig, dpi=250)


# ============================================================================
# PDF BUILDER CLASS
# ============================================================================
class Section15Builder(SectionPDFBuilder):
    """Builder for Section 15: Quick Reference Card."""

    def __init__(self):
        super().__init__(
            section_num=15,
            title="Quick Reference"
        )

    def build_content(self):
        """Build the complete section content."""

        # Title block
        self.add_title_block(
            subtitle="Field Reference Guide for HIRT System Operation",
            version="2.0"
        )

        # Introduction
        self.add_body_text(
            "This quick reference card provides essential information for field deployment "
            "of the HIRT system. Print this section as a laminated card for use in the field.",
            first_paragraph=True
        )

        self.add_spacer(8)

        # Figure 1: Quick Setup Diagram
        self.add_section_header("Power-Up Sequence", level=1)
        setup_diagram = create_quick_setup_diagram()
        self.add_figure(
            setup_diagram,
            "HIRT quick setup sequence showing the seven-step power-up procedure "
            "from cable connection to measurement initiation.",
            width=CONTENT_WIDTH * 0.95
        )

        # Figure 2: Key Specifications Visual
        self.add_section_header("Key Parameters", level=1)
        specs_visual = create_key_specs_visual()
        self.add_figure(
            specs_visual,
            "Key HIRT specifications for MIT (magnetic induction tomography), "
            "ERT (electrical resistivity tomography), and standard grid layout.",
            width=CONTENT_WIDTH * 0.95
        )

        # Grid Layout Section
        self.add_section_header("Standard Grid Layout (10x10 m)", level=2)
        self.add_code_block(
            "Spacing: 2 meters    Probes: 25 (5x5 grid)    Depth: 2.5-3 m\n"
            "\n"
            "      0m    2m    4m    6m    8m\n"
            " 0m   P01   P02   P03   P04   P05\n"
            " 2m   P06   P07   P08   P09   P10\n"
            " 4m   P11   P12   P13   P14   P15\n"
            " 6m   P16   P17   P18   P19   P20\n"
            " 8m   P21   P22   P23   P24   P25"
        )

        # Status LEDs Table
        self.add_section_header("Status LEDs (Central Hub)", level=2)
        led_data = [
            ['LED', 'Solid', 'Blink', 'Off'],
            ['PWR', 'OK', 'Low Battery', 'No Power'],
            ['TX', 'Active', 'Scanning', 'Idle'],
            ['LOG', 'Log Active', 'SD Error', 'No Log'],
            ['ERR', 'System Fault', 'Port Warning', 'OK'],
        ]
        self.add_table(led_data, caption="Central Hub LED status indicators")

        # Cable Color Code Table
        self.add_section_header("Cable Color Code", level=2)
        cable_data = [
            ['Wire Color', 'Function'],
            ['Red', 'Power +'],
            ['Black', 'Power GND'],
            ['White', 'TX+'],
            ['Green', 'TX-'],
            ['Blue', 'RX+'],
            ['Yellow', 'RX-'],
            ['Shield', 'Ground'],
        ]
        col_widths = [CONTENT_WIDTH * 0.4, CONTENT_WIDTH * 0.6]
        self.add_table(cable_data, col_widths=col_widths,
                      caption="Standard cable wire color assignments")

        # Probe Connector Pinout
        self.add_section_header("Probe Connector Pinout", level=2)
        pinout_data = [
            ['Pin', 'Signal', 'Pin', 'Signal'],
            ['1', 'TX+', '5', 'Guard'],
            ['2', 'TX-', '6', 'Ring A'],
            ['3', 'RX+', '7', 'Ring B'],
            ['4', 'RX-', '8', 'Ring C'],
        ]
        col_widths = [CONTENT_WIDTH * 0.15, CONTENT_WIDTH * 0.35,
                     CONTENT_WIDTH * 0.15, CONTENT_WIDTH * 0.35]
        self.add_table(pinout_data, col_widths=col_widths,
                      caption="8-pin probe connector pinout")

        # Soil Type Guidelines
        self.add_section_header("Soil Type Guidelines", level=2)
        soil_data = [
            ['Soil Type', 'Insertion Method', 'ERT Contact'],
            ['Sand', 'Direct push', 'Add water'],
            ['Clay', 'Pre-drill', 'Good'],
            ['Rocky', 'Careful auger', 'Variable'],
            ['Wet', 'Easy', 'Excellent'],
        ]
        self.add_table(soil_data, caption="Soil-specific probe installation guidance")

        # Safety Checklist
        self.add_section_header("Pre-Deployment Safety Checklist", level=1)
        safety_items = [
            "Site access authorized",
            "Underground utilities located and marked",
            "First aid kit available on site",
            "Weather conditions checked and acceptable",
            "Contact person informed of work location",
            "Equipment extraction plan ready",
        ]
        self.add_warning_box("SAFETY CHECK (Complete Before Deployment)", safety_items)

        # Reference note
        self.add_note(
            "For detailed procedures and troubleshooting, see Section 10: Field Operation Manual "
            "and Section 13: Troubleshooting Guide."
        )

        return self.build()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main():
    """Generate the Section 15 Quick Reference PDF."""
    print("Generating Section 15: Quick Reference Card...")

    builder = Section15Builder()
    output_path = builder.build_content()

    print(f"Successfully generated: {output_path}")
    return output_path


if __name__ == '__main__':
    main()
