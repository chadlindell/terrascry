#!/usr/bin/env python3
"""
HIRT Whitepaper - Section 07: Assembly and Wiring
Publication-quality PDF generator for assembly instructions and wiring documentation.

This section covers:
- Parts lists and tools required
- Rod segment preparation
- Sensor module assembly
- Field assembly procedures
- Zone wiring architecture
- Quality checks and troubleshooting
"""

import os
import sys

# Add parent directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch, Arc
)
import numpy as np
from io import BytesIO

from lib.pdf_builder import SectionPDFBuilder
from reportlab.lib.units import inch
from lib.styles import CONTENT_WIDTH, PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING
from lib.diagrams.field_ops import (
    create_cable_color_coding,
    create_insertion_procedure
)

# Color palette for diagrams
COLORS = {
    'primary': '#1a365d',
    'secondary': '#2c5282',
    'accent': '#3182ce',
    'success': '#38a169',
    'warning': '#c53030',
    'orange': '#ed8936',
    'purple': '#805ad5',
    'gray_dark': '#4a5568',
    'gray_med': '#718096',
    'gray_light': '#a0aec0',
    'light_green': '#c6f6d5',
    'light_orange': '#feebc8',
    'light_purple': '#e9d8fd',
    'light_blue': '#ebf8ff',
    'light_red': '#fed7d7',
    'ground_tan': '#d4a373',
    'white': '#ffffff',
    'black': '#000000',
}


# ============================================================================
# DIAGRAM 0: Zone Architecture (local implementation)
# ============================================================================
def create_zone_architecture():
    """
    Create zone wiring architecture diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Central hub
    ax.add_patch(FancyBboxPatch((0.5, 2), 2.0, 2.0, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_blue'], edgecolor=COLORS['secondary'],
                                linewidth=2))
    ax.text(1.5, 3, 'Central\nHub', ha='center', va='center',
           fontsize=10, fontweight='bold')

    # Trunk cables
    trunk_y = [4.5, 3.0, 1.5]
    for y in trunk_y:
        ax.plot([2.5, 4.0], [y, y], color=COLORS['gray_dark'], lw=4)

    ax.text(3.25, 5.0, 'Trunk Cables\n(DB25)', ha='center', fontsize=8,
           color=COLORS['gray_dark'])

    # Zone hubs
    zone_positions = [(4.0, 4.0), (4.0, 2.5), (4.0, 1.0)]
    for i, (x, y) in enumerate(zone_positions):
        ax.add_patch(FancyBboxPatch((x, y), 1.5, 1.0, boxstyle="round,pad=0.03",
                                    facecolor='white', edgecolor=COLORS['gray_dark'],
                                    linewidth=1.5))
        ax.text(x + 0.75, y + 0.5, f'Zone {i+1}\nHub', ha='center', va='center',
               fontsize=8)

    # Probe cables from each zone
    for zi, (zx, zy) in enumerate(zone_positions):
        for pi in range(4):
            px = 7.0 + (pi % 2) * 1.5
            py = zy + 0.2 + (pi // 2) * 0.4
            ax.plot([zx + 1.5, px], [zy + 0.5, py], color=COLORS['gray_light'], lw=1)
            ax.add_patch(Circle((px, py), 0.2, facecolor=COLORS['secondary'],
                               edgecolor='black', lw=0.5))

    ax.text(7.75, 5.5, 'Probes\n(4 per zone)', ha='center', fontsize=8)

    # Annotations
    ax.annotate('High-density\nmultiplexing', xy=(1.5, 2), xytext=(0.5, 0.5),
               fontsize=7, ha='center',
               arrowprops=dict(arrowstyle='->', color=COLORS['accent']))

    ax.annotate('Passive\njunction', xy=(4.75, 4.5), xytext=(4.75, 5.5),
               fontsize=7, ha='center',
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

    # Title
    ax.text(5, 0.2, 'Zone Wiring Architecture', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# DIAGRAM 1: Wiring Harness Overview
# ============================================================================
def create_wiring_harness_diagram():
    """
    Create wiring harness layout diagram showing probe-to-hub cable routing.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Wiring Harness Architecture', fontsize=12,
            fontweight='bold', ha='center', color=COLORS['primary'])

    # === Probe (Left Side) ===
    # Probe rod
    ax.add_patch(Rectangle((0.5, 1.0), 0.4, 3.5,
                           facecolor=COLORS['secondary'], edgecolor='black', lw=1.5))
    ax.text(0.7, 4.7, 'Probe', ha='center', fontsize=9, fontweight='bold')

    # Sensor positions on probe
    sensor_info = [
        (4.0, 'TX Coil', COLORS['success']),
        (3.2, 'ERT Ring 1', COLORS['orange']),
        (2.4, 'RX Coil', COLORS['accent']),
        (1.6, 'ERT Ring 2', COLORS['orange']),
    ]

    for y, label, color in sensor_info:
        ax.add_patch(Rectangle((0.45, y - 0.1), 0.5, 0.2,
                               facecolor=color, edgecolor='black', lw=0.5))

    # === Junction Box ===
    ax.add_patch(FancyBboxPatch((1.2, 3.0), 1.5, 1.5, boxstyle="round,pad=0.05",
                                facecolor=COLORS['gray_dark'], edgecolor='black', lw=1.5))
    ax.text(1.95, 4.3, 'Junction\nBox', ha='center', fontsize=8, color='white')

    # Internal connections from probe to junction box
    for y, label, color in sensor_info:
        ax.plot([0.9, 1.2], [y, 3.75], color=color, lw=1.5, alpha=0.7)

    # === Multi-conductor Cable ===
    cable_x = np.linspace(2.7, 5.5, 50)
    cable_y = 3.75 + 0.3 * np.sin(np.linspace(0, 2*np.pi, 50))
    ax.plot(cable_x, cable_y, color=COLORS['gray_dark'], lw=6, solid_capstyle='round')
    ax.plot(cable_x, cable_y, color=COLORS['gray_light'], lw=4, solid_capstyle='round')

    ax.text(4.1, 4.4, 'Multi-conductor\nShielded Cable', ha='center', fontsize=8)
    ax.annotate('2-4mm dia', xy=(4.1, 3.4), fontsize=7, ha='center', color=COLORS['gray_dark'])

    # === Cable internal conductors (breakout view) ===
    breakout_x = 4.1
    breakout_y = 2.5
    ax.add_patch(FancyBboxPatch((breakout_x - 0.8, breakout_y - 0.8), 1.6, 1.4,
                                boxstyle="round,pad=0.02",
                                facecolor='white', edgecolor=COLORS['gray_med'],
                                lw=1, linestyle='--'))
    ax.text(breakout_x, breakout_y + 0.9, 'Cable Cross-Section', fontsize=7,
            ha='center', color=COLORS['gray_med'])

    # Conductor colors
    conductor_colors = [
        ('TX+', 'red'), ('TX-', 'black'), ('RX+', 'white'), ('RX-', 'green'),
        ('Ring A', 'blue'), ('Ring B', 'orange'), ('Shield', COLORS['gray_light'])
    ]

    for i, (name, color) in enumerate(conductor_colors[:6]):
        row = i // 3
        col = i % 3
        cx = breakout_x - 0.5 + col * 0.5
        cy = breakout_y + 0.2 - row * 0.6
        ax.add_patch(Circle((cx, cy), 0.1, facecolor=color, edgecolor='black', lw=0.5))

    # Shield (outer ring)
    ax.add_patch(Circle((breakout_x, breakout_y - 0.1), 0.6,
                        facecolor='none', edgecolor=COLORS['gray_light'], lw=2))

    # === Zone Box (Right Side) ===
    ax.add_patch(FancyBboxPatch((5.8, 2.5), 1.8, 2.0, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=COLORS['gray_dark'], lw=1.5))
    ax.text(6.7, 4.3, 'Zone Box', ha='center', fontsize=9, fontweight='bold')

    # Cable glands on Zone Box
    for i in range(4):
        gy = 2.8 + i * 0.4
        ax.add_patch(Circle((5.8, gy), 0.12, facecolor=COLORS['gray_med'],
                           edgecolor='black', lw=0.5))
    ax.text(5.3, 3.3, 'Cable\nGlands', ha='center', fontsize=6, va='center')

    # Terminal strip inside Zone Box
    ax.add_patch(Rectangle((6.2, 2.8), 0.8, 1.4, facecolor=COLORS['light_blue'],
                           edgecolor=COLORS['accent'], lw=1))
    ax.text(6.6, 3.5, 'Terminal\nStrip', ha='center', fontsize=6, va='center')

    # === Trunk Cable to Hub ===
    ax.plot([7.6, 9.0], [3.5, 3.5], color=COLORS['gray_dark'], lw=8, solid_capstyle='round')
    ax.plot([7.6, 9.0], [3.5, 3.5], color=COLORS['gray_light'], lw=5, solid_capstyle='round')
    ax.text(8.3, 3.9, 'Trunk Cable\n(to Main Hub)', ha='center', fontsize=8)

    # DB25 connector
    ax.add_patch(FancyBboxPatch((7.2, 3.2), 0.6, 0.6, boxstyle="round,pad=0.02",
                                facecolor=COLORS['gray_med'], edgecolor='black', lw=1))
    ax.text(7.5, 3.5, 'DB25', ha='center', fontsize=6, color='white', fontweight='bold')

    # === Legend ===
    legend_y = 0.5
    legend_items = [
        ('TX Signal', COLORS['success']),
        ('RX Signal', COLORS['accent']),
        ('ERT Electrodes', COLORS['orange']),
        ('Shield/Ground', COLORS['gray_light']),
    ]

    for i, (label, color) in enumerate(legend_items):
        lx = 1.5 + i * 2.3
        ax.add_patch(Rectangle((lx, legend_y), 0.3, 0.2, facecolor=color, edgecolor='black', lw=0.5))
        ax.text(lx + 0.4, legend_y + 0.1, label, fontsize=7, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# DIAGRAM 2: 12-Pin Connector Pinout Visual
# ============================================================================
def create_connector_pinout_diagram():
    """
    Create 12-pin Phoenix connector pinout diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(4.5, 5.7, '12-Pin Phoenix Connector Pinout (1757248)', fontsize=12,
            fontweight='bold', ha='center', color=COLORS['primary'])

    # === Connector Body ===
    conn_x, conn_y = 2.0, 2.0
    conn_w, conn_h = 3.5, 2.5

    # Outer housing
    ax.add_patch(FancyBboxPatch((conn_x, conn_y), conn_w, conn_h,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['gray_dark'], edgecolor='black', lw=2))

    # Inner cavity
    ax.add_patch(FancyBboxPatch((conn_x + 0.2, conn_y + 0.2), conn_w - 0.4, conn_h - 0.4,
                                boxstyle="round,pad=0.05",
                                facecolor=COLORS['gray_med'], edgecolor='black', lw=1))

    # Pin arrangement (2 rows of 6)
    pin_colors = {
        1: ('red', 'TX+'),
        2: ('black', 'TX-'),
        3: ('white', 'RX+'),
        4: ('green', 'RX-'),
        5: ('yellow', 'Guard'),
        6: ('blue', 'Ring A'),
        7: ('orange', 'Ring B'),
        8: ('#8B4513', 'Ring C'),  # Brown
        9: ('purple', 'ID Sense'),
        10: ('gray', 'Spare+'),
        11: ('pink', 'Spare-'),
        12: (COLORS['gray_light'], 'Shield'),
    }

    pin_radius = 0.15
    for pin in range(1, 13):
        row = (pin - 1) // 6
        col = (pin - 1) % 6
        px = conn_x + 0.5 + col * 0.5
        py = conn_y + conn_h - 0.7 - row * 0.6

        color, _ = pin_colors[pin]
        ax.add_patch(Circle((px, py), pin_radius, facecolor=color,
                           edgecolor='black', lw=1))

        # Pin number
        text_color = 'white' if color in ['black', 'purple', 'blue', '#8B4513'] else 'black'
        ax.text(px, py, str(pin), ha='center', va='center', fontsize=6,
                fontweight='bold', color=text_color)

    # Row labels
    ax.text(conn_x + 0.15, conn_y + conn_h - 0.7, '1-6', fontsize=6,
            va='center', color='white')
    ax.text(conn_x + 0.15, conn_y + conn_h - 1.3, '7-12', fontsize=6,
            va='center', color='white')

    # === Pinout Table ===
    table_x = 6.0
    table_y = 4.8
    row_height = 0.35

    ax.text(table_x + 1.2, table_y + 0.3, 'Signal Assignment', fontsize=10,
            fontweight='bold', ha='center', color=COLORS['primary'])

    # Table header
    ax.add_patch(Rectangle((table_x, table_y - row_height), 2.4, row_height,
                           facecolor=COLORS['primary'], edgecolor='black', lw=0.5))
    ax.text(table_x + 0.25, table_y - row_height/2, 'Pin', fontsize=7,
            color='white', va='center', fontweight='bold')
    ax.text(table_x + 0.8, table_y - row_height/2, 'Signal', fontsize=7,
            color='white', va='center', fontweight='bold')
    ax.text(table_x + 1.7, table_y - row_height/2, 'Color', fontsize=7,
            color='white', va='center', fontweight='bold')

    # Table rows
    for i, (pin, (color, signal)) in enumerate(pin_colors.items()):
        row_y = table_y - (i + 2) * row_height
        bg_color = COLORS['light_blue'] if i % 2 == 0 else 'white'

        ax.add_patch(Rectangle((table_x, row_y), 2.4, row_height,
                               facecolor=bg_color, edgecolor=COLORS['gray_light'], lw=0.5))

        ax.text(table_x + 0.25, row_y + row_height/2, str(pin), fontsize=6, va='center')
        ax.text(table_x + 0.8, row_y + row_height/2, signal, fontsize=6, va='center')

        # Color swatch
        ax.add_patch(Rectangle((table_x + 1.55, row_y + 0.08), 0.25, row_height - 0.16,
                               facecolor=color, edgecolor='black', lw=0.5))

    # === Notes ===
    ax.text(0.3, 0.8, 'Notes:', fontsize=8, fontweight='bold', color=COLORS['primary'])
    notes = [
        '- Pins 1-4: MIT coil differential pairs',
        '- Pins 5: Guard/reference electrode',
        '- Pins 6-8: ERT ring electrodes',
        '- Pin 9: Probe ID sense (optional)',
        '- Pins 10-11: Reserved for future use',
        '- Pin 12: Cable shield (drain wire)',
    ]
    for i, note in enumerate(notes):
        ax.text(0.3, 0.5 - i * 0.25, note, fontsize=7, color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# DIAGRAM 3: Assembly Sequence Diagram
# ============================================================================
def create_assembly_sequence_diagram():
    """
    Create step-by-step assembly sequence diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    steps = [
        ('1. Prepare Tip', 'tip'),
        ('2. First Segment', 'segment1'),
        ('3. Add Sensor Module', 'sensor'),
        ('4. Thread O-Ring', 'oring'),
        ('5. Stack Segments', 'stack'),
        ('6. Install Top Cap', 'cap'),
    ]

    for ax, (title, step) in zip(axes.flat, steps):
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, 4)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10, fontweight='bold', color=COLORS['primary'], pad=8)
        ax.axis('off')

        if step == 'tip':
            # Probe tip
            tip_points = [(-0.15, 0), (0.15, 0), (0.15, 0.8), (0, 1.2), (-0.15, 0.8)]
            ax.add_patch(Polygon(tip_points, facecolor=COLORS['secondary'],
                                edgecolor='black', lw=1.5))
            ax.text(0, 0.5, 'Tip', ha='center', fontsize=8, color='white')

            # Thread indicator
            ax.annotate('M12 Female\nThreads', xy=(0.15, 0.6), xytext=(1.0, 0.8),
                       fontsize=7, arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

            # Inspection checklist
            ax.text(0, 3.5, 'Check:', fontsize=8, fontweight='bold', ha='center')
            ax.text(0, 3.1, '- No damage', fontsize=7, ha='center')
            ax.text(0, 2.7, '- Thread clean', fontsize=7, ha='center')

        elif step == 'segment1':
            # Rod segment
            ax.add_patch(Rectangle((-0.15, 0.5), 0.3, 2.5,
                                  facecolor=COLORS['secondary'], edgecolor='black', lw=1.5))
            ax.text(0, 1.75, 'Rod', ha='center', fontsize=8, color='white', rotation=90)

            # Male thread at bottom
            ax.add_patch(Rectangle((-0.1, 0.2), 0.2, 0.3,
                                  facecolor=COLORS['gray_med'], edgecolor='black', lw=1))
            ax.annotate('Male\nInsert', xy=(0.1, 0.35), xytext=(1.0, 0.5),
                       fontsize=7, arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

            # Cable through
            ax.plot([0, 0], [0.5, 3.0], color=COLORS['orange'], lw=2, linestyle='--')
            ax.annotate('Thread\ncable', xy=(0, 2.5), xytext=(1.0, 2.5),
                       fontsize=7, arrowprops=dict(arrowstyle='->', color=COLORS['orange']))

        elif step == 'sensor':
            # Sensor module body
            ax.add_patch(Rectangle((-0.25, 1.2), 0.5, 1.0,
                                  facecolor=COLORS['success'], edgecolor='black', lw=1.5))
            ax.text(0, 1.7, 'Sensor\nModule', ha='center', fontsize=7, color='white')

            # Coil inside
            ax.add_patch(Rectangle((-0.15, 1.4), 0.3, 0.4,
                                  facecolor=COLORS['light_green'], edgecolor='black', lw=0.5))

            # Threads at top and bottom
            ax.add_patch(Rectangle((-0.1, 1.0), 0.2, 0.2,
                                  facecolor=COLORS['gray_med'], edgecolor='black', lw=0.5))
            ax.add_patch(Rectangle((-0.1, 2.2), 0.2, 0.2,
                                  facecolor=COLORS['gray_med'], edgecolor='black', lw=0.5))

            ax.annotate('Female (top)', xy=(0.1, 2.3), xytext=(0.8, 2.6),
                       fontsize=6, arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))
            ax.annotate('Male (bottom)', xy=(0.1, 1.1), xytext=(0.8, 0.8),
                       fontsize=6, arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

        elif step == 'oring':
            # Thread detail with O-ring
            ax.add_patch(Rectangle((-0.2, 1.0), 0.4, 1.5,
                                  facecolor=COLORS['secondary'], edgecolor='black', lw=1.5))

            # Thread pattern
            for i in range(6):
                ty = 1.2 + i * 0.2
                ax.plot([-0.2, 0.2], [ty, ty], color=COLORS['gray_med'], lw=0.5)

            # O-ring
            ax.add_patch(Circle((0, 2.6), 0.1, facecolor='black', edgecolor='black'))
            ax.add_patch(Circle((0, 2.6), 0.06, facecolor=COLORS['gray_dark']))

            ax.annotate('O-Ring\n(10mm ID x 1.5mm)', xy=(0.1, 2.6), xytext=(0.8, 3.0),
                       fontsize=7, arrowprops=dict(arrowstyle='->', color='black'))

            ax.text(0, 0.5, 'Seat O-ring on\nthread shoulder', ha='center', fontsize=7)

        elif step == 'stack':
            # Stacked segments
            segment_colors = [COLORS['secondary'], COLORS['success'], COLORS['secondary']]
            segment_labels = ['Segment', 'Sensor', 'Segment']

            y_pos = 0.3
            for i, (color, label) in enumerate(zip(segment_colors, segment_labels)):
                height = 0.8 if label == 'Sensor' else 1.0
                ax.add_patch(Rectangle((-0.15, y_pos), 0.3, height,
                                       facecolor=color, edgecolor='black', lw=1))
                y_pos += height

            # Joint indicators
            ax.annotate('', xy=(-0.4, 1.3), xytext=(-0.4, 1.5),
                       arrowprops=dict(arrowstyle='<->', color=COLORS['warning']))
            ax.text(-0.6, 1.4, 'Joint', fontsize=6, va='center', ha='right',
                    color=COLORS['warning'])

            ax.text(0, 3.5, 'Repeat for\ndesired depth', ha='center', fontsize=8)

        elif step == 'cap':
            # Complete probe with top cap
            ax.add_patch(Rectangle((-0.15, 0.3), 0.3, 2.5,
                                  facecolor=COLORS['secondary'], edgecolor='black', lw=1))

            # Top cap
            ax.add_patch(FancyBboxPatch((-0.25, 2.8), 0.5, 0.5, boxstyle="round,pad=0.02",
                                        facecolor=COLORS['gray_dark'], edgecolor='black', lw=1.5))
            ax.text(0, 3.05, 'Cap', ha='center', fontsize=7, color='white')

            # Cable exit
            ax.plot([0, 0.8], [3.3, 3.5], color=COLORS['orange'], lw=2)
            ax.annotate('Cable exit\nwith strain relief', xy=(0.5, 3.4), xytext=(1.0, 3.6),
                       fontsize=6, arrowprops=dict(arrowstyle='->', color=COLORS['orange']))

            # Test reminder
            ax.add_patch(FancyBboxPatch((-1.5, 0.3), 1.0, 0.6, boxstyle="round,pad=0.02",
                                        facecolor=COLORS['light_green'], edgecolor=COLORS['success']))
            ax.text(-1.0, 0.6, 'Test all\nconnections', ha='center', fontsize=6)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# DIAGRAM 4: Probe-to-Hub Connection Diagram
# ============================================================================
def create_probe_to_hub_connection():
    """
    Create detailed probe-to-hub signal routing diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5.5, 4.7, 'Probe-to-Hub Signal Routing', fontsize=12,
            fontweight='bold', ha='center', color=COLORS['primary'])

    # === Probe Section ===
    probe_x = 0.5
    ax.add_patch(Rectangle((probe_x, 0.5), 0.3, 3.5,
                           facecolor=COLORS['secondary'], edgecolor='black', lw=1.5))
    ax.text(probe_x + 0.15, 4.2, 'Probe', ha='center', fontsize=9, fontweight='bold')

    # Sensors on probe
    sensor_specs = [
        (3.5, 'TX', COLORS['success'], ['TX+', 'TX-']),
        (2.5, 'RX', COLORS['accent'], ['RX+', 'RX-']),
        (1.5, 'ERT', COLORS['orange'], ['A', 'B', 'C']),
    ]

    # === Junction Box ===
    jbox_x = 1.5
    ax.add_patch(FancyBboxPatch((jbox_x, 1.5), 1.2, 2.2, boxstyle="round,pad=0.03",
                                facecolor=COLORS['gray_dark'], edgecolor='black', lw=1.5))
    ax.text(jbox_x + 0.6, 3.5, 'Junction', ha='center', fontsize=7, color='white')

    # Connector at junction box
    ax.add_patch(Rectangle((jbox_x + 0.3, 1.7), 0.6, 0.4,
                           facecolor=COLORS['gray_med'], edgecolor='black', lw=1))
    ax.text(jbox_x + 0.6, 1.9, '12P', ha='center', fontsize=6, color='white')

    # Draw signal lines from sensors through junction
    wire_colors = {
        'TX+': 'red', 'TX-': 'black',
        'RX+': 'white', 'RX-': 'green',
        'A': 'blue', 'B': COLORS['orange'], 'C': '#8B4513'
    }

    y_offset = 0
    for sy, name, color, wires in sensor_specs:
        ax.add_patch(Rectangle((probe_x - 0.05, sy - 0.15), 0.4, 0.3,
                               facecolor=color, edgecolor='black', lw=0.5))
        for i, wire in enumerate(wires):
            wy = sy + 0.1 - i * 0.1
            ax.plot([probe_x + 0.3, jbox_x], [wy, 2.0 + y_offset],
                    color=wire_colors[wire], lw=1, alpha=0.8)
            y_offset += 0.15

    # === Multi-conductor Cable ===
    cable_x = np.linspace(2.7, 5.0, 30)
    cable_y = 2.5 * np.ones_like(cable_x)
    ax.plot(cable_x, cable_y, color=COLORS['gray_dark'], lw=8)
    ax.plot(cable_x, cable_y, color=COLORS['gray_light'], lw=5)
    ax.text(3.85, 2.9, 'Shielded Cable (3-5m)', ha='center', fontsize=8)

    # === Zone Box ===
    zbox_x = 5.2
    ax.add_patch(FancyBboxPatch((zbox_x, 1.0), 2.0, 3.0, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=COLORS['gray_dark'], lw=1.5))
    ax.text(zbox_x + 1.0, 3.8, 'Zone Box', ha='center', fontsize=9, fontweight='bold')

    # Terminal blocks in Zone Box
    for i in range(4):
        ty = 1.3 + i * 0.6
        ax.add_patch(Rectangle((zbox_x + 0.2, ty), 1.6, 0.4,
                               facecolor=COLORS['light_blue'], edgecolor=COLORS['accent'], lw=0.5))
        ax.text(zbox_x + 1.0, ty + 0.2, f'Probe {i+1}', ha='center', fontsize=6)

    # Highlighted current probe connection
    ax.add_patch(Rectangle((zbox_x + 0.15, 3.15), 1.7, 0.5,
                           facecolor='none', edgecolor=COLORS['success'], lw=2))

    # === Trunk Cable ===
    trunk_x = np.linspace(7.2, 9.0, 20)
    trunk_y = 2.5 * np.ones_like(trunk_x)
    ax.plot(trunk_x, trunk_y, color=COLORS['gray_dark'], lw=12)
    ax.plot(trunk_x, trunk_y, color=COLORS['gray_light'], lw=8)
    ax.text(8.1, 3.0, 'Trunk Cable\n(DB25, 10-20m)', ha='center', fontsize=8)

    # === Main Hub ===
    hub_x = 9.2
    ax.add_patch(FancyBboxPatch((hub_x, 0.5), 1.6, 4.0, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_blue'], edgecolor=COLORS['secondary'], lw=2))
    ax.text(hub_x + 0.8, 4.3, 'Main Hub', ha='center', fontsize=10, fontweight='bold')

    # Hub internals
    hub_blocks = [
        (3.6, 'MUX Array', COLORS['light_purple']),
        (2.8, 'MIT TX/RX', COLORS['light_green']),
        (2.0, 'ERT Circuits', COLORS['light_orange']),
        (1.2, 'MCU/ADC', COLORS['light_purple']),
    ]

    for by, label, color in hub_blocks:
        ax.add_patch(Rectangle((hub_x + 0.1, by), 1.4, 0.6,
                               facecolor=color, edgecolor=COLORS['gray_med'], lw=0.5))
        ax.text(hub_x + 0.8, by + 0.3, label, ha='center', fontsize=6)

    # === Signal Flow Arrows ===
    ax.annotate('', xy=(1.5, 2.5), xytext=(0.9, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))
    ax.annotate('', xy=(5.2, 2.5), xytext=(2.7, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))
    ax.annotate('', xy=(9.2, 2.5), xytext=(7.2, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# DIAGRAM 5: Base Hub Stack Overview
# ============================================================================
def create_base_hub_stack_diagram():
    """
    Create base hub internal stack layout diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Base Hub Internal Stack', fontsize=12,
            fontweight='bold', ha='center', color=COLORS['primary'])

    # === Enclosure Outline ===
    ax.add_patch(FancyBboxPatch((1.0, 0.5), 5.0, 4.8, boxstyle="round,pad=0.1",
                                facecolor='white', edgecolor=COLORS['gray_dark'], lw=2))
    ax.text(3.5, 5.15, 'Weatherproof Enclosure (Bud NBF-32016)',
            ha='center', fontsize=8, color=COLORS['gray_dark'])

    # Stack layers (bottom to top)
    stack_layers = [
        (0.7, 0.8, 'Power Shelf', COLORS['light_orange'],
         'LiFePO4 Battery, Fuse Block'),
        (1.6, 0.6, 'Harness Strain Relief', COLORS['gray_light'],
         'Internal ribbon cables'),
        (2.3, 1.2, 'Backplane PCB (160x120mm)', COLORS['light_green'],
         'DDS, TX/RX, ERT, MCU'),
        (3.6, 0.7, 'Front Panel', COLORS['light_blue'],
         'DB25 Trunk Ports'),
        (4.4, 0.6, 'Top Cover', COLORS['gray_light'],
         'Weatherproof seal'),
    ]

    for y, h, name, color, desc in stack_layers:
        ax.add_patch(Rectangle((1.2, y), 4.6, h,
                               facecolor=color, edgecolor='black', lw=1))
        ax.text(3.5, y + h/2, name, ha='center', va='center',
                fontsize=9, fontweight='bold')
        ax.text(6.0, y + h/2, desc, va='center', fontsize=7, color=COLORS['gray_dark'])

    # === Backplane Detail (Right Side) ===
    detail_x = 7.0
    ax.add_patch(FancyBboxPatch((detail_x, 0.8), 2.8, 4.2, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=COLORS['gray_med'], lw=1,
                                linestyle='--'))
    ax.text(detail_x + 1.4, 4.85, 'Backplane Zones', ha='center', fontsize=9,
            fontweight='bold', color=COLORS['primary'])

    backplane_zones = [
        ('DDS/TX Driver', COLORS['light_green'], 'AD9833, OPA454'),
        ('RX Front End', COLORS['light_green'], 'AD620, INA128'),
        ('ERT Source', COLORS['light_orange'], 'OPA177, REF5025'),
        ('Lock-In/ADC', COLORS['light_purple'], 'ADS1256'),
        ('Control/Sync', COLORS['light_purple'], 'ESP32, USB-UART'),
        ('Power', COLORS['light_orange'], 'DC-DC, LDO'),
    ]

    zone_y = 4.4
    for name, color, parts in backplane_zones:
        zone_y -= 0.55
        ax.add_patch(Rectangle((detail_x + 0.1, zone_y), 2.6, 0.45,
                               facecolor=color, edgecolor=COLORS['gray_med'], lw=0.5))
        ax.text(detail_x + 0.2, zone_y + 0.22, name, fontsize=7, va='center', fontweight='bold')
        ax.text(detail_x + 2.6, zone_y + 0.22, parts, fontsize=6, va='center',
                ha='right', color=COLORS['gray_dark'])

    # === Front Panel Detail ===
    ax.annotate('6x DB25 Female\n(24 probes max)', xy=(1.2, 3.95), xytext=(0.2, 4.5),
                fontsize=7, arrowprops=dict(arrowstyle='->', color=COLORS['accent']))

    # Connector symbols on front panel layer
    for i in range(6):
        cx = 1.5 + i * 0.7
        ax.add_patch(Rectangle((cx, 3.7), 0.4, 0.25,
                               facecolor=COLORS['gray_med'], edgecolor='black', lw=0.5))

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# MAIN DOCUMENT BUILDER
# ============================================================================
def build_section_07():
    """Build Section 07: Assembly and Wiring PDF."""

    print("Generating Section 07: Assembly and Wiring...")
    print("Creating diagrams...")

    # Generate all diagrams
    fig_harness = create_wiring_harness_diagram()
    fig_pinout = create_connector_pinout_diagram()
    fig_zone_arch = create_zone_architecture()
    fig_cable_colors = create_cable_color_coding()
    fig_assembly = create_assembly_sequence_diagram()
    fig_probe_hub = create_probe_to_hub_connection()
    fig_hub_stack = create_base_hub_stack_diagram()
    fig_insertion = create_insertion_procedure()

    print("Building PDF...")

    # Create PDF builder
    builder = SectionPDFBuilder(
        section_num=7,
        title="Assembly and Wiring"
    )

    # Title block
    builder.add_title_block(
        subtitle="Comprehensive Assembly Instructions for HIRT Modular Micro-Probe System",
        version="2.0"
    )

    # === Overview ===
    builder.add_section_header("7.1 Overview", level=1)
    builder.add_body_text(
        "This section provides comprehensive step-by-step assembly instructions for the "
        "HIRT modular micro-probe system (16mm). The modular design allows probes to be "
        "built by stacking fiberglass rod segments and 3D-printed sensor modules. The "
        "assembly process is designed for field conditions with minimal tooling requirements.",
        first_paragraph=True
    )

    # === Parts List ===
    builder.add_section_header("7.2 Parts List", level=1)

    builder.add_section_header("7.2.1 Printed Parts (PETG/ASA)", level=2)
    printed_parts = [
        ['Part', 'Qty/Probe', 'Notes'],
        ['Male Insert Plug', '2', 'Threaded male screw end'],
        ['Sensor Module (Female)', '2-3', 'Sensor body with female threads'],
        ['Probe Tip', '1', 'Pointed nose cone'],
        ['Top Cap', '1', 'Cable exit/handle'],
    ]
    builder.add_table(printed_parts, caption="3D-printed probe components")

    builder.add_section_header("7.2.2 Hardware", level=2)
    hardware_parts = [
        ['Part', 'Qty/Probe', 'Notes'],
        ['Fiberglass Tube', '2-3 sections', '16mm OD x 12mm ID'],
        ['Epoxy', 'As needed', '2-part structural (Loctite Marine, JB Weld)'],
        ['O-rings', '4-6', 'Size for M12 thread shoulder (10mm ID x 1.5mm)'],
        ['Ferrite Cores', '1-2', 'For MIT coils (6-8mm x 40-80mm)'],
        ['Magnet Wire', '10-20m', '30-34 AWG for coil winding'],
        ['ERT Ring Material', '2-3 bands', 'Stainless steel or copper tape'],
        ['Cable', '3-5m', 'Multi-conductor shielded'],
    ]
    builder.add_table(hardware_parts, caption="Hardware and consumable materials")

    # === Tools Required ===
    builder.add_section_header("7.3 Tools Required", level=1)

    tools_data = [
        ['Tool', 'Purpose'],
        ['Hacksaw or Tube Cutter', 'Cutting fiberglass tubing'],
        ['400-grit Sandpaper', 'Surface preparation'],
        ['M12x1.75 Tap', 'Threading female parts'],
        ['M12x1.75 Die', 'Threading male parts'],
        ['Tap Handle', 'For tap operation'],
        ['Mixing Cups', 'For epoxy'],
        ['Nitrile Gloves', 'Epoxy handling'],
        ['Soldering Iron', 'Wire connections'],
        ['Multimeter', 'Testing continuity'],
        ['Calipers', 'Measuring dimensions'],
    ]
    builder.add_table(tools_data, caption="Essential tools for assembly")

    builder.add_info_box("Recommended Additional Tools", [
        "Bench Vise - Holding parts during tapping",
        "Thread Cutting Oil - Lubrication for tap/die",
        "LCR Meter - Coil testing",
        "Heat Gun - Heat shrink tubing",
        "Wire Strippers - Cable preparation"
    ])

    # === Wiring Architecture ===
    builder.add_conditional_page_break(4*inch)
    builder.add_section_header("7.4 Wiring Architecture", level=1)

    builder.add_body_text(
        "The HIRT system uses a hierarchical wiring architecture designed to manage the "
        "complexity of connecting 20-50 passive probes. Rather than routing all probe cables "
        "directly to the main hub (which would require 600+ conductors), the system uses a "
        "<b>Zone Wiring Strategy</b> that aggregates signals through intermediate junction boxes.",
        first_paragraph=True
    )

    # Zone Architecture Diagram
    builder.add_figure(fig_zone_arch,
        "Zone wiring architecture showing probe groupings, zone boxes, and trunk cable "
        "routing to the central hub. Each zone aggregates 4 probes through a passive junction box."
    )

    builder.add_section_header("7.4.1 System Topology", level=2)
    builder.add_body_text(
        "The zone architecture divides the probe array into manageable groups of 4 probes each. "
        "Each group connects to a Zone Box (small passive hub) via individual probe cables. "
        "The Zone Box then connects to the Main Hub via a single high-density Trunk Cable. "
        "This reduces the number of cables entering the main enclosure from 20+ to just 5-6 trunk cables.",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "<b>Zone Box:</b> Small IP65 enclosure (100x100mm) with 4 cable glands for probes and 1 DB25 connector for trunk output",
        "<b>Trunk Cable:</b> High-quality shielded 25-conductor cable, 10-20 meters, carrying signals for 4 probes",
        "<b>Advantages:</b> Modular expansion, clean installation, field-ready deployment"
    ])

    # === Wiring Harness ===
    builder.add_section_header("7.4.2 Wiring Harness Layout", level=2)

    builder.add_figure(fig_harness,
        "Wiring harness architecture from probe sensors through junction box and zone box to the main hub. "
        "The multi-conductor shielded cable (2-4mm diameter) carries all sensor signals."
    )

    # === Connector Pinout ===
    builder.add_section_header("7.4.3 Connector Pinout Reference", level=2)

    builder.add_body_text(
        "The system uses 12-pin Phoenix Contact connectors (part number 1757248) or equivalent "
        "for probe-side connections. The pinout is standardized across all probes to ensure "
        "interchangeability and simplify field maintenance.",
        first_paragraph=True
    )

    builder.add_figure(fig_pinout,
        "12-pin Phoenix connector pinout showing signal assignments and recommended wire colors. "
        "Pins 1-4 carry MIT coil differential pairs, pins 6-8 carry ERT electrode signals."
    )

    # === Cable Color Coding ===
    builder.add_section_header("7.4.4 Cable Color Coding", level=2)

    builder.add_figure(fig_cable_colors,
        "Cable color coding reference for consistent wiring across all probes. Following this "
        "standard simplifies troubleshooting and maintenance."
    )

    # === Probe-to-Hub Connection ===
    builder.add_conditional_page_break(4*inch)
    builder.add_section_header("7.4.5 Probe-to-Hub Signal Routing", level=2)

    builder.add_figure(fig_probe_hub,
        "Complete signal routing from probe sensors through the junction box, zone box, and trunk "
        "cable to the main hub electronics. The hierarchical structure reduces wiring complexity."
    )

    # === Assembly Procedures ===
    builder.add_section_header("7.5 Assembly Procedures", level=1)

    builder.add_section_header("7.5.1 Preparation Steps", level=2)
    builder.add_numbered_list([
        "<b>Inspect All Parts:</b> Verify all printed parts are present and undamaged, no visible layer separation or cracks",
        "<b>Prepare Workspace:</b> Clean, well-lit, well-ventilated area with protected work surface",
        "<b>Test-Fit Parts (Dry Run):</b> Test thread engagement on all threaded parts, verify tube sections fit inserts, check O-ring sizing"
    ])

    builder.add_section_header("7.5.2 Rod Segment Preparation", level=2)
    builder.add_body_text(
        "Cut fiberglass tubing to desired lengths (50 cm or 100 cm segments are standard). "
        "Use steady, even strokes with the hacksaw and rotate the tube to keep the cut square. "
        "After cutting, deburr the ends by removing fiberglass fibers with a file and sanding smooth with 400-grit sandpaper.",
        first_paragraph=True
    )

    builder.add_info_box("Rod Assembly Convention", [
        "Rods have Male threads at bottom, Female at top",
        "Apply epoxy to flange of insert before installation",
        "Ensure shoulder sits flush against tube cut",
        "Allow 24 hours cure time before stressing threads"
    ])

    # === Assembly Sequence ===
    builder.add_section_header("7.5.3 Field Assembly Sequence", level=2)

    builder.add_body_text(
        "Assembly proceeds from the bottom (tip) and works upward. Each joint uses thread sealant "
        "and an O-ring for waterproofing. The cable is threaded through each segment before connecting.",
        first_paragraph=True
    )

    builder.add_figure(fig_assembly,
        "Six-step assembly sequence: (1) Prepare probe tip, (2) Thread cable through first segment, "
        "(3) Add sensor module, (4) Seat O-ring on thread shoulder, (5) Stack additional segments, "
        "(6) Install top cap with cable strain relief."
    )

    # Insertion Procedure
    builder.add_figure(fig_insertion,
        "Probe insertion procedure showing pilot hole preparation, segment insertion, segment connection, "
        "and final installation with surface junction box."
    )

    # === Base Hub Assembly ===
    builder.add_conditional_page_break(4*inch)
    builder.add_section_header("7.6 Base Hub Assembly", level=1)

    builder.add_body_text(
        "The base hub houses all electronics in a weatherproof enclosure. The internal layout "
        "follows a stacked architecture for easy assembly and maintenance. The backplane PCB "
        "(160x120mm) contains all signal processing circuits organized into functional zones.",
        first_paragraph=True
    )

    builder.add_figure(fig_hub_stack,
        "Base hub internal stack layout showing the layered architecture from power shelf at bottom "
        "to weatherproof cover at top. The backplane PCB contains all analog and digital circuits."
    )

    # Backplane zones table
    backplane_data = [
        ['Zone', 'Function', 'Key Parts'],
        ['DDS/TX Driver', 'Sweep generation', 'AD9833, OPA454'],
        ['RX Front End', 'Signal conditioning', 'AD620, INA128'],
        ['ERT Source', 'Current injection', 'OPA177, REF5025'],
        ['Lock-In/ADC', 'Digitization', 'ADS1256'],
        ['Control/Sync', 'Scheduling, logging', 'ESP32, USB-UART'],
        ['Power', 'Regulation', 'DC-DC, LDO'],
    ]
    builder.add_table(backplane_data, caption="Backplane PCB functional zones")

    # === Quality Checks ===
    builder.add_section_header("7.7 Quality Checks", level=1)

    builder.add_section_header("7.7.1 During Assembly", level=2)
    builder.add_bullet_list([
        "O-ring properly seated at each joint",
        "No gaps between components",
        "Threads fully engaged (hand tight + 1/4 turn)",
        "Cable has slack (not stretched at any joint)",
        "Joints are flush (smooth to touch)"
    ])

    builder.add_section_header("7.7.2 After Complete Assembly", level=2)

    builder.add_body_text("<b>Mechanical Checks:</b>", no_indent=True)
    builder.add_bullet_list([
        "Total length correct for intended depth",
        "All joints tight with no visible gaps",
        "Probe straight (no bends from misaligned joints)",
        "Cable secure at strain relief"
    ])

    builder.add_body_text("<b>Electrical Checks:</b>", no_indent=True)
    builder.add_bullet_list([
        "All coil leads continuous (no opens) - verify with multimeter",
        "Coil inductance in spec (1-2 mH) - verify with LCR meter",
        "ERT ring isolation >1 M-ohm between all ring pairs",
        "No shorts between any conductors",
        "Shield continuity verified end-to-end"
    ])

    # === Troubleshooting ===
    builder.add_section_header("7.8 Troubleshooting", level=1)

    builder.add_section_header("7.8.1 Thread Issues", level=2)
    troubleshoot_threads = [
        ['Problem', 'Solution', 'Prevention'],
        ['Threads too tight', 'Chase threads with tap/die', 'Print at correct tolerance, post-process'],
        ['Threads too loose', 'Apply thread sealant (Teflon tape)', 'Check print settings, use proper tolerances'],
        ['Threads stripped', 'Replace part', 'Hand-tight + 1/4 turn max, no over-tightening'],
    ]
    builder.add_table(troubleshoot_threads, caption="Thread issue troubleshooting guide")

    builder.add_section_header("7.8.2 Electrical Issues", level=2)
    troubleshoot_electrical = [
        ['Problem', 'Solution', 'Check'],
        ['Open circuit in coil', 'Check for broken wire, resolder', 'Wire may be damaged during assembly'],
        ['Short between rings', 'Check for bridging, clean thoroughly', 'Conductive debris in gaps'],
        ['Low coil Q factor', 'Rewrap coil more neatly', 'Shorted turns from damaged insulation'],
    ]
    builder.add_table(troubleshoot_electrical, caption="Electrical issue troubleshooting guide")

    # === Assembly Tips ===
    builder.add_section_header("7.9 Assembly Tips", level=1)

    builder.add_info_box("General Assembly Tips", [
        "Work clean - Fiberglass dust and epoxy do not mix well",
        "Test often - Verify continuity at each stage",
        "Do not rush - Allow full cure time for epoxy (24 hours)",
        "Label everything - Mark probe ID on each segment",
        "Document - Record any deviations or issues"
    ])

    builder.add_info_box("Epoxy Tips", [
        "Mix thoroughly (2+ minutes of stirring)",
        "Apply thin coats - too much is messy and adds weight",
        "Work in well-ventilated area",
        "Clean up drips immediately with IPA",
        "Allow full cure before stressing joints"
    ])

    builder.add_info_box("Thread Tips", [
        "Use cutting oil with tap/die",
        "Back out tap every half-turn to clear chips",
        "If stuck, back out and clear - do not force",
        "Test fit with mating part before committing",
        "Apply light lubricant before final assembly"
    ])

    # === Post-Assembly ===
    builder.add_section_header("7.10 Post-Assembly", level=1)
    builder.add_body_text(
        "After complete assembly, proceed to the following sections for system verification:",
        first_paragraph=True
    )
    builder.add_numbered_list([
        "Complete Testing Procedures (Section 8: Testing and Verification)",
        "Complete Calibration Procedures (Section 9: Calibration)",
        "Label and register probe in system database",
        "Store properly in protective case until deployment"
    ])

    builder.add_warning_box("Important Reminders", [
        "Allow epoxy to fully cure (24 hours) before field deployment",
        "Verify all electrical connections before sealing junction boxes",
        "Document probe configuration (sensor positions, coil specs) for each unit",
        "Store probes vertically to prevent cable stress at joints"
    ])

    # Build the PDF
    output_path = builder.build()
    print(f"Section 07 PDF created: {output_path}")
    return output_path


if __name__ == "__main__":
    build_section_07()
