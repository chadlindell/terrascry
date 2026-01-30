#!/usr/bin/env python3
"""
HIRT Section 03: System Architecture - Professional PDF Generator

This module generates a publication-quality PDF for Section 03 of the HIRT
whitepaper, covering system architecture including probe design, electronics
hub, zone wiring strategy, and array configurations.

Features 10+ detailed diagrams covering:
- System block diagram (Hub -> Zone Hubs -> Probes)
- Probe cross-section with TX/RX coils and ERT rings
- Zone Hub topology
- Array configuration options
- Component positioning along rod
- Electronics hub internal layout
- Trunk cable connections
- MIT signal flow
- ERT signal flow
- Multiplexer switching topology
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, Ellipse, FancyArrowPatch, Arc
)
import numpy as np
from io import BytesIO
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pdf_builder import SectionPDFBuilder, CONTENT_WIDTH
from lib.styles import COLOR_PALETTE, PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING


# ============================================================================
# COLOR PALETTE (matplotlib hex strings)
# ============================================================================
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
    'ground_dark': '#654321',
    'sky': '#e8f4f8',
    'tx_coil': '#38a169',
    'rx_coil': '#3182ce',
    'ert_ring': '#ed8936',
    'probe_body': '#2c5282',
    'connector': '#4a5568',
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def draw_block(ax, x, y, width, height, text, color='white', text_color='black',
               fontsize=8, edgecolor='black', linewidth=1, rounded=True):
    """Draw a block diagram element."""
    left = x - width/2
    bottom = y - height/2

    if rounded:
        patch = FancyBboxPatch((left, bottom), width, height,
                               boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor=edgecolor,
                               linewidth=linewidth)
    else:
        patch = Rectangle((left, bottom), width, height,
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=linewidth)
    ax.add_patch(patch)

    ax.text(x, y, text, ha='center', va='center',
           fontsize=fontsize, color=text_color)

    return patch


def draw_connection(ax, start, end, style='->', color='black', lw=1.5,
                    connectionstyle=None):
    """Draw a connection/arrow between two points."""
    if connectionstyle:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                  connectionstyle=connectionstyle))
    else:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# ============================================================================
# FIGURE 1: SYSTEM BLOCK DIAGRAM
# ============================================================================
def create_system_block_diagram():
    """
    Create complete HIRT system block diagram showing:
    Central Hub -> Zone Hubs -> Probes hierarchy
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    # Central Hub box
    hub = FancyBboxPatch((0.3, 2.5), 4.0, 4.0, boxstyle="round,pad=0.1",
                         facecolor=COLORS['light_blue'], edgecolor=COLORS['secondary'],
                         linewidth=2)
    ax.add_patch(hub)
    ax.text(2.3, 6.3, 'Central Electronics Hub', fontsize=11, fontweight='bold',
           ha='center', color=COLORS['primary'])

    # MCU block
    draw_block(ax, 2.3, 5.5, 2.0, 0.7, 'MCU (STM32/ESP32)\nControl + DSP', COLORS['light_purple'])

    # MIT TX
    draw_block(ax, 1.3, 4.3, 1.6, 0.7, 'MIT TX\n(DDS + Driver)', COLORS['light_green'])

    # MIT RX
    draw_block(ax, 3.3, 4.3, 1.6, 0.7, 'MIT RX\n(Preamp + ADC)', COLORS['light_green'])

    # ERT Source
    draw_block(ax, 1.3, 3.2, 1.6, 0.7, 'ERT Source\n(Howland)', COLORS['light_orange'])

    # ERT Measure
    draw_block(ax, 3.3, 3.2, 1.6, 0.7, 'ERT Measure\n(Diff Amp)', COLORS['light_orange'])

    # Mux
    draw_block(ax, 2.3, 2.0, 2.5, 0.6, 'High-Density Analog MUX\n(CD74HC4067)', COLORS['gray_light'])

    # Internal connections
    ax.plot([2.3, 2.3], [5.1, 4.7], 'k-', lw=1)
    ax.plot([1.3, 1.3], [4.7, 3.9], 'k-', lw=1)
    ax.plot([3.3, 3.3], [4.7, 3.9], 'k-', lw=1)
    ax.plot([1.3, 1.3], [3.6, 2.3], 'k-', lw=1)
    ax.plot([3.3, 3.3], [3.6, 2.3], 'k-', lw=1)

    # Trunk cable to Zone Hub A
    draw_connection(ax, (4.3, 4.0), (5.5, 4.5), style='-', lw=3, color=COLORS['gray_dark'])
    ax.text(4.7, 4.5, 'Trunk A\n(DB25)', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Trunk cable to Zone Hub B
    draw_connection(ax, (4.3, 3.0), (5.5, 2.5), style='-', lw=3, color=COLORS['gray_dark'])
    ax.text(4.7, 2.3, 'Trunk B\n(DB25)', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Zone Hub A
    zone_hub_a = FancyBboxPatch((5.5, 4.0), 1.6, 1.2, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=COLORS['gray_dark'],
                                linewidth=1.5)
    ax.add_patch(zone_hub_a)
    ax.text(6.3, 4.6, 'Zone Hub A\n(Passive)', fontsize=8, fontweight='bold',
           ha='center', color=COLORS['gray_dark'])

    # Zone Hub B
    zone_hub_b = FancyBboxPatch((5.5, 1.8), 1.6, 1.2, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=COLORS['gray_dark'],
                                linewidth=1.5)
    ax.add_patch(zone_hub_b)
    ax.text(6.3, 2.4, 'Zone Hub B\n(Passive)', fontsize=8, fontweight='bold',
           ha='center', color=COLORS['gray_dark'])

    # Probes from Zone A
    for i, py in enumerate([5.2, 4.8, 4.4, 4.0]):
        ax.plot([7.1, 7.8], [4.6, py], 'k-', lw=1)
        draw_block(ax, 9.0, py, 1.6, 0.35, f'Probe A{i+1}', COLORS['light_green'], fontsize=7)

    # Probes from Zone B
    for i, py in enumerate([3.0, 2.6, 2.2, 1.8]):
        ax.plot([7.1, 7.8], [2.4, py], 'k-', lw=1)
        draw_block(ax, 9.0, py, 1.6, 0.35, f'Probe B{i+1}', COLORS['light_green'], fontsize=7)

    # Ground representation
    ax.fill_between([7.8, 10.5], [1.4, 1.4], [0.8, 0.8], color=COLORS['ground_tan'], alpha=0.3)
    ax.axhline(1.4, xmin=0.71, xmax=0.95, color=COLORS['ground_dark'], lw=2)
    ax.text(9.0, 1.1, 'Ground Surface', fontsize=8, ha='center', color=COLORS['ground_dark'])

    # USB/Power connection
    draw_block(ax, 2.3, 0.8, 1.5, 0.5, 'USB/Power', COLORS['light_red'], fontsize=8)
    ax.plot([2.3, 2.3], [1.3, 1.7], 'k-', lw=1.5)
    draw_connection(ax, (2.3, 1.7), (2.3, 2.5), style='->', color='black')

    # Data output
    draw_connection(ax, (0.8, 5.5), (0.3, 5.5), style='->', color=COLORS['accent'])
    ax.text(0.1, 5.5, 'Data\nOut', fontsize=7, ha='right', color=COLORS['accent'])

    # Title
    ax.text(5.5, 6.8, 'HIRT System Block Diagram', fontsize=13,
           fontweight='bold', ha='center', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 2: PROBE CROSS-SECTION
# ============================================================================
def create_probe_cross_section():
    """
    Create detailed probe cross-section showing TX/RX coils and ERT rings.
    """
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(4, 8.7, 'HIRT Passive Micro-Probe Cross-Section', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Main probe body (centered)
    probe_cx = 3.0
    probe_width = 0.5
    probe_length = 7.5

    # Fiberglass rod body
    ax.add_patch(Rectangle((probe_cx - probe_width/2, 0.5), probe_width, probe_length,
                           color=COLORS['probe_body'], ec='black', lw=1.5, alpha=0.8))

    # Junction box at top
    ax.add_patch(FancyBboxPatch((probe_cx - 0.6, 7.8), 1.2, 0.6, boxstyle="round,pad=0.02",
                                facecolor=COLORS['connector'], ec='black', lw=1.5))
    ax.text(probe_cx, 8.1, 'Junction Box\n(Surface)', fontsize=7, ha='center', va='center', color='white')

    # Components from top to bottom with annotations
    components = [
        (7.5, 'M12x1.75 Threaded Joint', COLORS['gray_light'], 0.15, 'joint'),
        (6.8, 'TX Coil (Ferrite Core)', COLORS['tx_coil'], 0.6, 'tx'),
        (6.0, 'M12x1.75 Threaded Joint', COLORS['gray_light'], 0.15, 'joint'),
        (5.2, 'RX Coil (Ferrite Core)', COLORS['rx_coil'], 0.6, 'rx'),
        (4.3, 'ERT Ring (0.5m from tip)', COLORS['ert_ring'], 0.15, 'ert'),
        (3.5, 'M12x1.75 Threaded Joint', COLORS['gray_light'], 0.15, 'joint'),
        (2.5, 'ERT Ring (1.5m from tip)', COLORS['ert_ring'], 0.15, 'ert'),
        (1.5, 'ERT Ring (2.5m from tip)', COLORS['ert_ring'], 0.15, 'ert'),
        (0.7, 'Probe Tip', COLORS['primary'], 0.3, 'tip'),
    ]

    for y, label, color, height, comp_type in components:
        if comp_type == 'tip':
            # Pointed tip
            points = [(probe_cx - probe_width/2, y + height),
                     (probe_cx + probe_width/2, y + height),
                     (probe_cx, y - 0.2)]
            ax.add_patch(Polygon(points, color=color, ec='black', lw=1))
        elif comp_type == 'ert':
            # ERT ring - wider than probe body
            ax.add_patch(Rectangle((probe_cx - probe_width/2 - 0.05, y - height/2),
                                   probe_width + 0.1, height,
                                   color=color, ec='black', lw=1))
        elif comp_type == 'joint':
            # Threaded joint
            ax.add_patch(Rectangle((probe_cx - probe_width/2, y - height/2),
                                   probe_width, height,
                                   color=color, ec='black', lw=0.5))
        else:
            # TX/RX coil - shown as filled area
            ax.add_patch(Rectangle((probe_cx - probe_width/2 + 0.05, y - height/2),
                                   probe_width - 0.1, height,
                                   color=color, ec='black', lw=1))

        # Annotation with leader line
        ax.annotate(label, (probe_cx + probe_width/2, y),
                   (probe_cx + 1.5, y),
                   fontsize=8, va='center',
                   arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Dimension annotations
    # Overall length
    ax.annotate('', xy=(probe_cx - 1.0, 0.5), xytext=(probe_cx - 1.0, 8.0),
               arrowprops=dict(arrowstyle='<->', color=COLORS['gray_dark'], lw=1))
    ax.text(probe_cx - 1.3, 4.25, '1.5m\nsegment', fontsize=8, ha='center', va='center',
           color=COLORS['gray_dark'], rotation=90)

    # Probe diameter
    ax.annotate('', xy=(probe_cx - probe_width/2, 8.5), xytext=(probe_cx + probe_width/2, 8.5),
               arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=1))
    ax.text(probe_cx, 8.65, '16mm OD', fontsize=9, ha='center', fontweight='bold',
           color=COLORS['primary'])

    # Wire routing indication
    ax.plot([probe_cx - 0.15, probe_cx - 0.15], [0.9, 7.8], 'k--', lw=0.5, alpha=0.5)
    ax.text(probe_cx - 0.3, 4.5, 'Wires', fontsize=6, rotation=90, va='center', alpha=0.5)

    # Legend
    legend_y = 2.0
    legend_x = 6.0
    ax.text(legend_x, legend_y + 1.5, 'Legend:', fontsize=9, fontweight='bold')

    legend_items = [
        (COLORS['tx_coil'], 'TX Coil'),
        (COLORS['rx_coil'], 'RX Coil'),
        (COLORS['ert_ring'], 'ERT Ring'),
        (COLORS['gray_light'], 'Thread Joint'),
        (COLORS['probe_body'], 'Fiberglass Rod'),
    ]
    for i, (color, label) in enumerate(legend_items):
        ax.add_patch(Rectangle((legend_x, legend_y + 1.0 - i*0.35), 0.3, 0.25,
                               color=color, ec='black', lw=0.5))
        ax.text(legend_x + 0.45, legend_y + 1.0 - i*0.35 + 0.12, label, fontsize=7, va='center')

    # Specifications box
    specs_x = 5.5
    specs_y = 5.5
    ax.add_patch(FancyBboxPatch((specs_x, specs_y), 2.3, 2.0, boxstyle="round,pad=0.05",
                                facecolor='#f7fafc', ec=COLORS['gray_med'], lw=1))
    ax.text(specs_x + 1.15, specs_y + 1.8, 'Specifications', fontsize=8, fontweight='bold',
           ha='center', color=COLORS['primary'])

    specs = [
        'OD: 16 mm (10-16mm range)',
        'Material: G10 Fiberglass',
        'Segments: 50cm, 100cm',
        'Weight: 50-100 g/m',
        'Coil: 34-38 AWG wire',
    ]
    for i, spec in enumerate(specs):
        ax.text(specs_x + 0.1, specs_y + 1.45 - i*0.28, spec, fontsize=7)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 3: ZONE HUB TOPOLOGY
# ============================================================================
def create_zone_hub_topology():
    """
    Create Zone Hub topology diagram showing passive breakout architecture.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Zone Hub Wiring Topology', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Central zone hub box
    hub_x, hub_y = 3.5, 3.0
    zone_hub = FancyBboxPatch((hub_x - 1.2, hub_y - 1.5), 2.4, 3.0, boxstyle="round,pad=0.08",
                               facecolor='white', edgecolor=COLORS['secondary'], linewidth=2)
    ax.add_patch(zone_hub)
    ax.text(hub_x, hub_y + 1.2, 'Zone Hub\n(Passive IP65 Box)', fontsize=9, fontweight='bold',
           ha='center', color=COLORS['secondary'])

    # DB25 connector (input)
    ax.add_patch(Rectangle((hub_x - 0.8, hub_y - 1.3), 1.6, 0.4,
                           color=COLORS['gray_dark'], ec='black', lw=1))
    ax.text(hub_x, hub_y - 1.1, 'DB25 Input', fontsize=7, ha='center', color='white')

    # Internal terminal blocks
    for i in range(4):
        y_pos = hub_y + 0.6 - i * 0.4
        ax.add_patch(Rectangle((hub_x - 0.5, y_pos - 0.1), 1.0, 0.25,
                               color=COLORS['light_green'], ec='black', lw=0.5))
        ax.text(hub_x - 0.7, y_pos, f'P{i+1}', fontsize=6, ha='right', va='center')

    ax.text(hub_x, hub_y, 'Terminal\nBlocks', fontsize=7, ha='center', va='center')

    # Trunk cable from hub
    ax.annotate('', xy=(hub_x - 1.2, hub_y - 1.1), xytext=(0.5, hub_y - 1.1),
               arrowprops=dict(arrowstyle='-', lw=4, color=COLORS['gray_dark']))
    ax.text(1.2, hub_y - 0.7, 'To Central Hub\n(DB25 Trunk Cable)', fontsize=8,
           ha='center', color=COLORS['gray_dark'])

    # Probe cables
    probe_positions = [(7.5, 4.5), (7.5, 3.5), (7.5, 2.5), (7.5, 1.5)]
    for i, (px, py) in enumerate(probe_positions):
        # Cable from zone hub to probe
        ax.plot([hub_x + 1.2, px - 1.0], [hub_y + 0.5 - i*0.4, py], 'k-', lw=1.5)

        # Probe representation
        ax.add_patch(Rectangle((px - 0.15, py - 0.8), 0.3, 1.0,
                               color=COLORS['probe_body'], ec='black', lw=1))
        ax.add_patch(Rectangle((px - 0.25, py + 0.1), 0.5, 0.3,
                               color=COLORS['connector'], ec='black', lw=1))

        # Label
        ax.text(px + 0.4, py + 0.2, f'Probe {i+1}', fontsize=8, va='center')

        # Cable strain relief symbol
        ax.add_patch(Circle((px, py + 0.25), 0.08, color='black'))

    # Ground line
    ax.axhline(0.8, xmin=0.65, xmax=0.85, color=COLORS['ground_dark'], lw=2)
    ax.fill_between([6.5, 8.5], [0.8, 0.8], [0.3, 0.3], color=COLORS['ground_tan'], alpha=0.3)
    ax.text(7.5, 0.55, 'Ground', fontsize=8, ha='center', color=COLORS['ground_dark'])

    # Benefits callout
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 2.5, 1.5, boxstyle="round,pad=0.05",
                                facecolor='#f0fff4', ec=COLORS['success'], lw=1))
    ax.text(1.45, 1.55, 'Zone Benefits:', fontsize=8, fontweight='bold', color=COLORS['success'])
    benefits = [
        'Modular deployment',
        'Shielded trunk cables',
        'Easy field repair',
        '4 probes per zone',
    ]
    for i, benefit in enumerate(benefits):
        ax.text(0.4, 1.25 - i*0.25, f'  {benefit}', fontsize=7)

    # Wire count annotation
    ax.add_patch(FancyBboxPatch((0.2, 4.5), 2.0, 1.0, boxstyle="round,pad=0.03",
                                facecolor='#fffbeb', ec=COLORS['orange'], lw=1))
    ax.text(1.2, 5.25, 'Per Probe Wiring:', fontsize=7, fontweight='bold')
    ax.text(1.2, 4.95, 'TX: 2 wires', fontsize=6, ha='center')
    ax.text(1.2, 4.75, 'RX: 2 wires', fontsize=6, ha='center')
    ax.text(1.2, 4.55, 'ERT: 3 rings = 3 wires', fontsize=6, ha='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 4: ARRAY CONFIGURATION OPTIONS
# ============================================================================
def create_array_configurations():
    """
    Create diagram showing different array geometry options.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    configs = [
        ('Square Grid (Standard)', 'square'),
        ('Perimeter Array (UXO-Safe)', 'perimeter'),
        ('Dense Center (Known Anomaly)', 'dense'),
    ]

    for ax, (title, config_type) in zip(axes, configs):
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 5.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10, fontweight='bold', color=COLORS['primary'], pad=8)

        # Grid background
        ax.set_facecolor('#f7fafc')

        if config_type == 'square':
            # 5x5 uniform grid
            for i in range(5):
                for j in range(5):
                    ax.add_patch(Circle((i+0.5, j+0.5), 0.15,
                                       color=COLORS['secondary'], ec='black', lw=1))

        elif config_type == 'perimeter':
            # Perimeter only with exclusion zone
            ax.add_patch(Rectangle((1.2, 1.2), 2.6, 2.6,
                                   facecolor=COLORS['light_red'], ec=COLORS['warning'],
                                   lw=2, linestyle='--', alpha=0.5))
            ax.text(2.5, 2.5, 'EXCLUSION\nZONE', fontsize=8, ha='center', va='center',
                   color=COLORS['warning'], fontweight='bold')

            # Perimeter probes
            for i in range(5):
                ax.add_patch(Circle((i+0.5, 0.5), 0.15, color=COLORS['secondary'], ec='black', lw=1))
                ax.add_patch(Circle((i+0.5, 4.5), 0.15, color=COLORS['secondary'], ec='black', lw=1))
            for j in range(1, 4):
                ax.add_patch(Circle((0.5, j+0.5), 0.15, color=COLORS['secondary'], ec='black', lw=1))
                ax.add_patch(Circle((4.5, j+0.5), 0.15, color=COLORS['secondary'], ec='black', lw=1))

        elif config_type == 'dense':
            # Dense center pattern
            # Corner probes
            corners = [(0.5, 0.5), (0.5, 4.5), (4.5, 0.5), (4.5, 4.5)]
            for x, y in corners:
                ax.add_patch(Circle((x, y), 0.15, color=COLORS['secondary'], ec='black', lw=1))

            # Edge midpoints
            edges = [(2.5, 0.5), (2.5, 4.5), (0.5, 2.5), (4.5, 2.5)]
            for x, y in edges:
                ax.add_patch(Circle((x, y), 0.15, color=COLORS['secondary'], ec='black', lw=1))

            # Dense center grid
            for i in range(3):
                for j in range(3):
                    ax.add_patch(Circle((i+1.5, j+1.5), 0.15,
                                       color=COLORS['success'], ec='black', lw=1))

            # Highlight center
            ax.add_patch(Circle((2.5, 2.5), 0.5,
                               facecolor='none', ec=COLORS['warning'], lw=2, linestyle='--'))
            ax.text(2.5, 2.5, 'Target', fontsize=7, ha='center', va='center', color=COLORS['warning'])

        # Grid lines
        for i in range(6):
            ax.axhline(i, color='gray', lw=0.3, alpha=0.5)
            ax.axvline(i, color='gray', lw=0.3, alpha=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Add overall title
    fig.suptitle('Array Configuration Options', fontsize=12, fontweight='bold',
                color=COLORS['primary'], y=1.02)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 5: COMPONENT POSITIONING ALONG ROD
# ============================================================================
def create_component_positioning():
    """
    Create diagram showing component positioning along probe rod.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 4.7, 'Component Positioning Along Probe Rod', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Horizontal probe representation
    probe_y = 2.5
    probe_start = 0.5
    probe_end = 9.5
    probe_height = 0.3

    # Main rod
    ax.add_patch(Rectangle((probe_start, probe_y - probe_height/2), probe_end - probe_start, probe_height,
                           color=COLORS['probe_body'], ec='black', lw=1.5))

    # Components with positions
    components = [
        (0.7, 'Tip', COLORS['primary'], 'Hardened\ntip'),
        (1.5, 'ERT 3', COLORS['ert_ring'], '2.5m'),
        (3.0, 'ERT 2', COLORS['ert_ring'], '1.5m'),
        (4.5, 'Joint', COLORS['gray_light'], 'M12x1.75'),
        (5.5, 'RX Coil', COLORS['rx_coil'], '6-8mm\nferrite'),
        (6.5, 'Joint', COLORS['gray_light'], 'M12x1.75'),
        (7.5, 'TX Coil', COLORS['tx_coil'], '6-8mm\nferrite'),
        (8.2, 'ERT 1', COLORS['ert_ring'], '0.5m'),
        (9.2, 'Junction', COLORS['connector'], 'Surface'),
    ]

    for x, label, color, desc in components:
        if 'Coil' in label:
            ax.add_patch(Rectangle((x - 0.3, probe_y - probe_height/2 - 0.02),
                                   0.6, probe_height + 0.04,
                                   color=color, ec='black', lw=1))
        elif 'ERT' in label:
            ax.add_patch(Rectangle((x - 0.1, probe_y - probe_height/2 - 0.05),
                                   0.2, probe_height + 0.1,
                                   color=color, ec='black', lw=1))
        elif 'Joint' in label:
            ax.add_patch(Rectangle((x - 0.15, probe_y - probe_height/2),
                                   0.3, probe_height,
                                   color=color, ec='black', lw=0.5))
        elif 'Tip' in label:
            points = [(x - 0.2, probe_y + probe_height/2),
                     (x - 0.2, probe_y - probe_height/2),
                     (x + 0.3, probe_y)]
            ax.add_patch(Polygon(points, color=color, ec='black', lw=1))
        elif 'Junction' in label:
            ax.add_patch(FancyBboxPatch((x - 0.25, probe_y - 0.25), 0.5, 0.5,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, ec='black', lw=1))

        # Label above
        ax.text(x, probe_y + 0.6, label, fontsize=7, ha='center', fontweight='bold')
        # Description below
        ax.text(x, probe_y - 0.7, desc, fontsize=6, ha='center', color=COLORS['gray_dark'])

    # Distance scale
    ax.annotate('', xy=(0.7, 1.3), xytext=(9.2, 1.3),
               arrowprops=dict(arrowstyle='<->', color=COLORS['gray_dark'], lw=1))
    ax.text(5, 1.1, 'Total length: 3m (two 1.5m segments)', fontsize=8, ha='center',
           color=COLORS['gray_dark'])

    # Depth markers
    depth_labels = ['3m depth', '2.5m', '2m', '1.5m', '1m', '0.5m', '0m (surface)']
    for i, label in enumerate(depth_labels):
        x_pos = 0.7 + i * (8.5 / 6)
        ax.plot([x_pos, x_pos], [1.5, 1.7], 'k-', lw=0.5)
        ax.text(x_pos, 1.9, label, fontsize=6, ha='center', rotation=45, color=COLORS['gray_med'])

    # Wire routing note
    ax.add_patch(FancyBboxPatch((0.2, 3.8), 3.5, 0.6, boxstyle="round,pad=0.03",
                                facecolor='#fffbeb', ec=COLORS['orange'], lw=1))
    ax.text(2.0, 4.1, 'Internal wiring: Fine wire (34-38 AWG) routed through rod bore',
           fontsize=7, ha='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 6: ELECTRONICS HUB INTERNAL LAYOUT
# ============================================================================
def create_electronics_hub_layout():
    """
    Create detailed electronics hub internal layout diagram.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 6.7, 'Central Electronics Hub - Internal Layout', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Main enclosure
    enclosure = FancyBboxPatch((0.5, 0.5), 9.0, 5.8, boxstyle="round,pad=0.1",
                                facecolor='#f7fafc', edgecolor=COLORS['secondary'], linewidth=2.5)
    ax.add_patch(enclosure)

    # Power section (left)
    ax.add_patch(FancyBboxPatch((0.8, 4.5), 2.0, 1.5, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_orange'], ec=COLORS['orange'], lw=1.5))
    ax.text(1.8, 5.8, 'Power Supply', fontsize=8, fontweight='bold', ha='center')
    draw_block(ax, 1.8, 5.2, 1.4, 0.4, '12V Input', COLORS['light_orange'], fontsize=7)
    draw_block(ax, 1.8, 4.7, 1.4, 0.4, '5V/3.3V LDO', COLORS['light_orange'], fontsize=7)

    # MCU section (center-top)
    ax.add_patch(FancyBboxPatch((3.2, 4.2), 2.5, 1.8, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_purple'], ec=COLORS['purple'], lw=1.5))
    ax.text(4.45, 5.8, 'Control Unit', fontsize=8, fontweight='bold', ha='center')
    draw_block(ax, 4.45, 5.2, 2.0, 0.5, 'MCU\n(ESP32/STM32)', COLORS['light_purple'], fontsize=7)
    draw_block(ax, 4.45, 4.5, 2.0, 0.5, 'USB/WiFi\nInterface', COLORS['light_purple'], fontsize=7)

    # MIT section (center-left)
    ax.add_patch(FancyBboxPatch((0.8, 2.2), 2.8, 1.8, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_green'], ec=COLORS['success'], lw=1.5))
    ax.text(2.2, 3.8, 'MIT Analog Front-End', fontsize=8, fontweight='bold', ha='center')
    draw_block(ax, 1.5, 3.2, 1.0, 0.5, 'DDS\n(AD9833)', COLORS['light_green'], fontsize=6)
    draw_block(ax, 2.8, 3.2, 1.0, 0.5, 'TX\nDriver', COLORS['light_green'], fontsize=6)
    draw_block(ax, 1.5, 2.5, 1.0, 0.5, 'Preamp\n(INA)', COLORS['light_green'], fontsize=6)
    draw_block(ax, 2.8, 2.5, 1.0, 0.5, 'Lock-in\nDetect', COLORS['light_green'], fontsize=6)

    # ERT section (center-right)
    ax.add_patch(FancyBboxPatch((4.0, 2.2), 2.4, 1.8, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_orange'], ec=COLORS['orange'], lw=1.5))
    ax.text(5.2, 3.8, 'ERT Measurement', fontsize=8, fontweight='bold', ha='center')
    draw_block(ax, 4.6, 3.2, 0.9, 0.5, 'Current\nSource', COLORS['light_orange'], fontsize=6)
    draw_block(ax, 5.7, 3.2, 0.9, 0.5, 'Polarity\nSwitch', COLORS['light_orange'], fontsize=6)
    draw_block(ax, 4.6, 2.5, 0.9, 0.5, 'Diff\nAmp', COLORS['light_orange'], fontsize=6)
    draw_block(ax, 5.7, 2.5, 0.9, 0.5, 'ADC\n(16-bit)', COLORS['light_orange'], fontsize=6)

    # Multiplexer section (bottom)
    ax.add_patch(FancyBboxPatch((0.8, 0.7), 5.8, 1.2, boxstyle="round,pad=0.05",
                                facecolor=COLORS['gray_light'], ec=COLORS['gray_dark'], lw=1.5))
    ax.text(3.7, 1.7, 'High-Density Analog Multiplexer Array', fontsize=8, fontweight='bold', ha='center')
    for i in range(6):
        draw_block(ax, 1.3 + i*0.9, 1.0, 0.7, 0.4, f'MUX\n{i+1}', COLORS['gray_light'], fontsize=5)

    # Connector panel (right side)
    ax.add_patch(FancyBboxPatch((7.0, 0.7), 2.2, 5.3, boxstyle="round,pad=0.05",
                                facecolor='white', ec=COLORS['gray_dark'], lw=1.5))
    ax.text(8.1, 5.8, 'Trunk Interface', fontsize=8, fontweight='bold', ha='center')

    # DB25 connectors
    for i, label in enumerate(['Zone A', 'Zone B', 'Zone C', 'Zone D']):
        y = 5.0 - i * 1.1
        ax.add_patch(Rectangle((7.3, y - 0.3), 1.6, 0.6,
                               color=COLORS['connector'], ec='black', lw=1))
        ax.text(8.1, y, label, fontsize=7, ha='center', color='white')

    # USB connector
    ax.add_patch(Rectangle((7.5, 1.0), 1.2, 0.4,
                           color='black', ec='black', lw=1))
    ax.text(8.1, 1.2, 'USB', fontsize=7, ha='center', color='white')

    # Power LED indicators
    for i, (color, label) in enumerate([('green', 'PWR'), ('blue', 'TX'), ('orange', 'RX')]):
        ax.add_patch(Circle((8.8, 5.0 - i*0.3), 0.08, color=color))
        ax.text(8.6, 5.0 - i*0.3, label, fontsize=5, ha='right', va='center')

    # Signal flow arrows
    ax.annotate('', xy=(3.3, 2.85), xytext=(3.95, 2.85),
               arrowprops=dict(arrowstyle='<->', color=COLORS['gray_dark'], lw=1))
    ax.annotate('', xy=(4.45, 4.1), xytext=(4.45, 3.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1))
    ax.annotate('', xy=(6.5, 1.0), xytext=(7.0, 1.0),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.5))

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 7: TRUNK CABLE CONNECTIONS
# ============================================================================
def create_trunk_cable_diagram():
    """
    Create trunk cable connection diagram showing DB25 pinout.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Trunk Cable Connections (DB25)', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # DB25 connector representation
    conn_x = 2.5
    conn_y = 3.0
    ax.add_patch(FancyBboxPatch((conn_x - 1.5, conn_y - 1.2), 3.0, 2.4, boxstyle="round,pad=0.1",
                                facecolor=COLORS['connector'], ec='black', lw=2))
    ax.text(conn_x, conn_y + 0.9, 'DB25 Connector', fontsize=9, ha='center', color='white', fontweight='bold')

    # Pin rows
    for row in range(2):
        num_pins = 13 if row == 0 else 12
        y_pos = conn_y + 0.3 - row * 0.6
        for i in range(num_pins):
            x_pos = conn_x - 1.2 + i * 0.2
            ax.add_patch(Circle((x_pos, y_pos), 0.06, color='gold', ec='black', lw=0.5))

    # Pin assignment table (right side)
    table_x = 5.5
    table_y = 4.8

    ax.text(table_x + 1.5, table_y + 0.3, 'Pin Assignments (4 Probes/Zone)', fontsize=9,
           fontweight='bold', ha='center', color=COLORS['primary'])

    assignments = [
        ('1-2', 'Probe 1 TX Coil', COLORS['tx_coil']),
        ('3-4', 'Probe 1 RX Coil', COLORS['rx_coil']),
        ('5-7', 'Probe 1 ERT (3 rings)', COLORS['ert_ring']),
        ('8-9', 'Probe 2 TX Coil', COLORS['tx_coil']),
        ('10-11', 'Probe 2 RX Coil', COLORS['rx_coil']),
        ('12-14', 'Probe 2 ERT', COLORS['ert_ring']),
        ('15-16', 'Probe 3 TX Coil', COLORS['tx_coil']),
        ('17-18', 'Probe 3 RX Coil', COLORS['rx_coil']),
        ('19-21', 'Probe 3 ERT', COLORS['ert_ring']),
        ('22-23', 'Probe 4 TX Coil', COLORS['tx_coil']),
        ('24', 'Shield/GND', COLORS['gray_dark']),
        ('25', 'Probe ID', COLORS['gray_med']),
    ]

    for i, (pins, desc, color) in enumerate(assignments):
        y = table_y - i * 0.35
        ax.add_patch(Rectangle((table_x, y - 0.12), 0.6, 0.25,
                               color=color, ec='black', lw=0.5))
        ax.text(table_x + 0.3, y, pins, fontsize=6, ha='center', va='center',
               color='white' if color in [COLORS['tx_coil'], COLORS['connector'], COLORS['gray_dark']] else 'black')
        ax.text(table_x + 0.75, y, desc, fontsize=7, va='center')

    # Cable representation
    ax.plot([conn_x, conn_x], [conn_y - 1.2, 0.5], color=COLORS['gray_dark'], lw=4)
    ax.text(conn_x, 0.3, 'To Zone Hub', fontsize=8, ha='center', color=COLORS['gray_dark'])

    # Shielding note
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 2.5, 0.8, boxstyle="round,pad=0.03",
                                facecolor='#f0fff4', ec=COLORS['success'], lw=1))
    ax.text(1.45, 0.75, 'Cable Specifications:', fontsize=7, fontweight='bold')
    ax.text(1.45, 0.45, 'Shielded, twisted pairs\n10-20m max length', fontsize=6, ha='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 8: MIT SIGNAL FLOW DIAGRAM
# ============================================================================
def create_mit_signal_flow():
    """
    Create MIT signal path diagram.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5.5, 3.7, 'MIT Signal Flow Path', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # TX Chain (top)
    ax.text(0.5, 2.8, 'TX Chain:', fontsize=9, fontweight='bold', color=COLORS['tx_coil'])

    tx_blocks = [
        (1.5, 2.5, 1.2, 0.6, 'MCU\n(DDS)', COLORS['light_purple']),
        (3.2, 2.5, 1.0, 0.6, 'DAC', COLORS['light_blue']),
        (4.7, 2.5, 1.0, 0.6, 'LP Filter', COLORS['light_green']),
        (6.2, 2.5, 1.0, 0.6, 'Driver', COLORS['light_orange']),
        (7.8, 2.5, 1.0, 0.6, 'MUX', COLORS['gray_light']),
        (9.5, 2.5, 1.2, 0.6, 'TX Coil\n(Ground)', COLORS['tx_coil']),
    ]

    for x, y, w, h, text, color in tx_blocks:
        draw_block(ax, x, y, w, h, text, color=color, fontsize=7)

    # TX connections
    tx_connections = [(2.1, 2.5), (2.7, 2.5), (3.7, 2.5), (4.2, 2.5),
                      (5.2, 2.5), (5.7, 2.5), (6.7, 2.5), (7.3, 2.5),
                      (8.3, 2.5), (8.9, 2.5)]
    for i in range(0, len(tx_connections), 2):
        draw_connection(ax, tx_connections[i], tx_connections[i+1], color=COLORS['gray_dark'])

    # Signal labels
    ax.text(2.4, 2.9, 'SPI', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(3.95, 2.9, 'Analog', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(5.45, 2.9, 'Clean', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(7.0, 2.9, '10-50mA', fontsize=6, ha='center', color=COLORS['gray_med'])

    # RX Chain (bottom)
    ax.text(0.5, 1.0, 'RX Chain:', fontsize=9, fontweight='bold', color=COLORS['rx_coil'])

    rx_blocks = [
        (9.5, 1.2, 1.2, 0.6, 'RX Coil\n(Ground)', COLORS['rx_coil']),
        (7.8, 1.2, 1.0, 0.6, 'MUX', COLORS['gray_light']),
        (6.2, 1.2, 1.0, 0.6, 'Preamp', COLORS['light_green']),
        (4.7, 1.2, 1.0, 0.6, 'Inst Amp', COLORS['light_green']),
        (3.2, 1.2, 1.0, 0.6, 'BP Filter', COLORS['light_blue']),
        (1.5, 1.2, 1.2, 0.6, 'ADC\n(16-bit)', COLORS['light_purple']),
    ]

    for x, y, w, h, text, color in rx_blocks:
        draw_block(ax, x, y, w, h, text, color=color, fontsize=7)

    # RX connections (reversed)
    rx_connections = [(8.9, 1.2), (8.3, 1.2), (7.3, 1.2), (6.7, 1.2),
                      (5.7, 1.2), (5.2, 1.2), (4.2, 1.2), (3.7, 1.2),
                      (2.7, 1.2), (2.1, 1.2)]
    for i in range(0, len(rx_connections), 2):
        draw_connection(ax, rx_connections[i], rx_connections[i+1], color=COLORS['gray_dark'])

    # RX signal labels
    ax.text(8.0, 0.7, 'uV', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(5.45, 0.7, 'mV', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(3.95, 0.7, '100mV', fontsize=6, ha='center', color=COLORS['gray_med'])

    # Reference signal path
    ax.plot([1.5, 1.5, 10.0, 10.0], [2.2, 0.5, 0.5, 1.2], '--', color=COLORS['orange'], lw=1)
    ax.text(5.5, 0.3, 'Reference signal for lock-in detection', fontsize=7,
           ha='center', color=COLORS['orange'], style='italic')

    # Frequency annotation
    ax.add_patch(FancyBboxPatch((0.2, 3.1), 2.0, 0.5, boxstyle="round,pad=0.03",
                                facecolor='#f0fff4', ec=COLORS['success'], lw=1))
    ax.text(1.2, 3.35, 'Frequency: 2-50 kHz', fontsize=7, ha='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 9: ERT SIGNAL FLOW DIAGRAM
# ============================================================================
def create_ert_signal_flow():
    """
    Create ERT signal path diagram.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5.5, 4.2, 'ERT Signal Flow Path', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Current injection path (top)
    ax.text(0.3, 3.3, 'Current Injection:', fontsize=9, fontweight='bold', color=COLORS['orange'])

    inj_blocks = [
        (1.5, 3.0, 1.2, 0.6, 'MCU\nControl', COLORS['light_purple']),
        (3.2, 3.0, 1.2, 0.6, 'DAC\n(V_ref)', COLORS['light_blue']),
        (5.0, 3.0, 1.4, 0.6, 'Howland\nSource', COLORS['light_orange']),
        (6.8, 3.0, 1.0, 0.6, 'Polarity', COLORS['light_orange']),
        (8.3, 3.0, 1.0, 0.6, 'MUX', COLORS['gray_light']),
        (9.8, 3.0, 1.0, 0.6, 'ERT\nRings', COLORS['ert_ring']),
    ]

    for x, y, w, h, text, color in inj_blocks:
        draw_block(ax, x, y, w, h, text, color=color, fontsize=7)

    # Injection connections
    for i in range(len(inj_blocks) - 1):
        x1 = inj_blocks[i][0] + inj_blocks[i][2]/2
        x2 = inj_blocks[i+1][0] - inj_blocks[i+1][2]/2
        draw_connection(ax, (x1, 3.0), (x2, 3.0), color=COLORS['gray_dark'])

    # Labels
    ax.text(2.35, 3.45, 'I2C', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(4.1, 3.45, '0-2.5V', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(5.9, 3.45, '0.5-2mA', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(7.55, 3.45, '+/-', fontsize=6, ha='center', color=COLORS['gray_med'])

    # Voltage measurement path (bottom)
    ax.text(0.3, 1.4, 'Voltage Measure:', fontsize=9, fontweight='bold', color=COLORS['orange'])

    meas_blocks = [
        (9.8, 1.7, 1.0, 0.6, 'ERT\nRings', COLORS['ert_ring']),
        (8.3, 1.7, 1.0, 0.6, 'MUX', COLORS['gray_light']),
        (6.6, 1.7, 1.2, 0.6, 'Diff Amp\n(INA128)', COLORS['light_green']),
        (4.8, 1.7, 1.2, 0.6, 'LP Filter\n(10 Hz)', COLORS['light_blue']),
        (3.0, 1.7, 1.2, 0.6, 'ADC\n(24-bit)', COLORS['light_purple']),
        (1.5, 1.7, 1.0, 0.6, 'MCU', COLORS['light_purple']),
    ]

    for x, y, w, h, text, color in meas_blocks:
        draw_block(ax, x, y, w, h, text, color=color, fontsize=7)

    # Measurement connections (reversed)
    for i in range(len(meas_blocks) - 1):
        x1 = meas_blocks[i][0] - meas_blocks[i][2]/2
        x2 = meas_blocks[i+1][0] + meas_blocks[i+1][2]/2
        draw_connection(ax, (x1, 1.7), (x2, 1.7), color=COLORS['gray_dark'])

    # Labels
    ax.text(7.45, 1.2, 'uV-mV', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(5.7, 1.2, 'Filtered', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(3.9, 1.2, 'SPI', fontsize=6, ha='center', color=COLORS['gray_med'])

    # Current flow indication
    ax.annotate('', xy=(10.3, 2.4), xytext=(10.3, 3.0),
               arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=2))
    ax.text(10.5, 2.7, 'I', fontsize=10, color=COLORS['warning'], fontweight='bold')

    # Ground symbol
    ax.plot([10.3, 10.3], [2.0, 2.4], 'k-', lw=1.5)
    ax.text(10.5, 2.2, 'Ground', fontsize=6, color=COLORS['gray_dark'])

    # Polarity reversal note
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 3.5, 0.6, boxstyle="round,pad=0.03",
                                facecolor='#fffbeb', ec=COLORS['orange'], lw=1))
    ax.text(2.0, 0.5, 'DC with polarity reversal (every 1-2s)\nor low-freq AC (8-16 Hz)',
           fontsize=7, ha='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 10: MULTIPLEXER SWITCHING TOPOLOGY
# ============================================================================
def create_mux_topology():
    """
    Create multiplexer switching topology diagram.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 6.7, 'Multiplexer Switching Topology', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # MCU control
    draw_block(ax, 1.5, 5.5, 1.8, 1.0, 'MCU\n(Control)', COLORS['light_purple'])

    # Address/control lines
    ax.plot([2.4, 3.5], [5.5, 5.5], 'k-', lw=1)
    ax.text(2.95, 5.7, 'Addr (4-bit)', fontsize=6, color=COLORS['gray_med'])

    # Main MUX array
    ax.add_patch(FancyBboxPatch((3.5, 3.0), 3.5, 3.5, boxstyle="round,pad=0.1",
                                facecolor=COLORS['gray_light'], ec=COLORS['gray_dark'], lw=2))
    ax.text(5.25, 6.2, 'Analog MUX Array', fontsize=9, fontweight='bold', ha='center')

    # MIT TX MUX
    draw_block(ax, 4.25, 5.3, 1.0, 0.6, 'MUX\nMIT TX', COLORS['light_green'], fontsize=6)

    # MIT RX MUX
    draw_block(ax, 5.75, 5.3, 1.0, 0.6, 'MUX\nMIT RX', COLORS['light_green'], fontsize=6)

    # ERT MUXes (4 for 4 probes with 3 rings each)
    for i in range(4):
        draw_block(ax, 4.0 + i*0.8, 4.2, 0.7, 0.5, f'ERT\n{i+1}', COLORS['light_orange'], fontsize=5)

    # Output stage
    draw_block(ax, 4.25, 3.4, 1.0, 0.5, 'TX Out', COLORS['tx_coil'], fontsize=7)
    draw_block(ax, 5.75, 3.4, 1.0, 0.5, 'RX In', COLORS['rx_coil'], fontsize=7)

    # Probe connections (right side)
    ax.text(8.5, 6.0, 'Trunk Cables', fontsize=8, fontweight='bold', ha='center')

    probe_y = [5.2, 4.4, 3.6, 2.8]
    for i, y in enumerate(probe_y):
        # Connection line
        ax.plot([7.0, 7.8], [4.0, y], 'k-', lw=1)

        # Probe representation
        ax.add_patch(Rectangle((7.8, y - 0.25), 1.8, 0.5,
                               facecolor='white', ec=COLORS['gray_dark'], lw=1))
        ax.text(8.7, y, f'Zone {chr(65+i)} (4 probes)', fontsize=7, ha='center', va='center')

    # Signal paths
    ax.annotate('', xy=(4.25, 4.8), xytext=(4.25, 5.0),
               arrowprops=dict(arrowstyle='->', color=COLORS['tx_coil'], lw=1.5))
    ax.annotate('', xy=(5.75, 5.0), xytext=(5.75, 4.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['rx_coil'], lw=1.5))

    # ERT connections
    for i in range(4):
        ax.plot([4.0 + i*0.8, 4.0 + i*0.8], [3.7, 3.95], 'k-', lw=0.5)

    # Channel count annotation
    ax.add_patch(FancyBboxPatch((0.3, 0.5), 3.5, 2.0, boxstyle="round,pad=0.05",
                                facecolor='#f7fafc', ec=COLORS['secondary'], lw=1))
    ax.text(2.05, 2.25, 'Channel Capacity:', fontsize=8, fontweight='bold', ha='center')
    channel_info = [
        'MIT: 16 TX + 16 RX = 32 channels',
        'ERT: 4 zones x 4 probes x 3 rings = 48',
        'Total: 80 analog channels',
        'MUX IC: CD74HC4067 (16:1)',
    ]
    for i, info in enumerate(channel_info):
        ax.text(2.05, 1.9 - i*0.35, info, fontsize=7, ha='center')

    # Switching speed note
    ax.add_patch(FancyBboxPatch((0.3, 3.0), 2.8, 1.2, boxstyle="round,pad=0.03",
                                facecolor='#fffbeb', ec=COLORS['orange'], lw=1))
    ax.text(1.7, 3.95, 'Switching Speed:', fontsize=7, fontweight='bold', ha='center')
    ax.text(1.7, 3.6, 'MIT: <1us (fast scan)', fontsize=6, ha='center')
    ax.text(1.7, 3.3, 'ERT: <10us (slower)', fontsize=6, ha='center')

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
def main():
    """Build the complete Section 03 PDF document."""

    print("=" * 60)
    print("HIRT Section 03: System Architecture - PDF Generator")
    print("=" * 60)

    # Initialize the PDF builder
    builder = SectionPDFBuilder(
        section_num=3,
        title="System Architecture"
    )

    # Add title block
    builder.add_title_block(
        subtitle="Micro-Probe Design and Zone Wiring Strategy"
    )

    # =========================================================================
    # 3.1 DESIGN PHILOSOPHY
    # =========================================================================
    builder.add_section_header("3.1 Design Philosophy", level=1)
    builder.add_body_text(
        "<b>\"Archaeologist brain first, engineer brain second\"</b> - This guiding principle "
        "shapes every aspect of the HIRT system architecture. The primary goal is to minimize "
        "ground disturbance while maintaining sufficient measurement quality for target detection. "
        "The result is a probe design with 10-16 mm outer diameter (target: 12-16 mm), representing "
        "approximately 10x less ground disturbance than traditional 25+ mm geophysical probes.",
        first_paragraph=True
    )
    builder.add_body_text(
        "The design philosophy embraces a key constraint: many thin, gentle holes are preferable "
        "to fewer large ones. This approach is particularly important for sensitive archaeological "
        "contexts where visible disturbance must be minimized and backfilling should leave no "
        "lasting trace of the survey."
    )

    # =========================================================================
    # 3.2 PROBE OVERVIEW
    # =========================================================================
    builder.add_section_header("3.2 Probe Overview (Passive Micro-Probes)", level=1)
    builder.add_body_text(
        "Each HIRT probe is fundamentally <b>passive</b> - containing no active electronics "
        "downhole. Only sensors and wiring reside within the probe body, with all signal "
        "conditioning and processing occurring at the central electronics hub. This architecture "
        "offers significant advantages in reliability, cost, and field serviceability.",
        first_paragraph=True
    )

    # Generate and add probe cross-section figure
    print("Generating Figure 1: Probe Cross-Section...")
    fig_probe = create_probe_cross_section()
    builder.add_figure(
        fig_probe,
        "HIRT passive micro-probe cross-section showing internal component arrangement. "
        "TX and RX coils are wound on ferrite cores (6-8 mm diameter) and mounted along the "
        "fiberglass rod. ERT rings are flush-mounted at standard positions (0.5m, 1.5m, 2.5m from tip). "
        "Modular M12x1.75 threaded joints allow field assembly of multiple segments.",
        width=CONTENT_WIDTH * 0.75
    )

    builder.add_section_header("MIT Coil Set", level=2)
    builder.add_bullet_list([
        "<b>1x TX coil + 1x RX coil</b> wound on ferrite cores",
        "<b>Ferrite cores:</b> 6-8 mm diameter x 40-80 mm long",
        "<b>Mounting:</b> Glued along rod (not in bulky head) for streamlined profile",
        "<b>Orientation:</b> Orthogonal or slightly separated to reduce direct coupling",
        "<b>Wire:</b> Fine wire (34-38 AWG), 200-400 turns for adequate signal strength"
    ])

    builder.add_section_header("ERT-Lite Rings", level=2)
    builder.add_bullet_list([
        "<b>2-3 narrow ring electrodes</b> (3-5 mm wide bands)",
        "<b>Standard positions:</b> 0.5 m and 1.5 m from tip",
        "<b>Deep extension:</b> Add third ring at 2.5-3.0 m for longer rods",
        "<b>Material:</b> Stainless steel or copper foil",
        "<b>Mounting:</b> Bonded with epoxy, flush with rod surface"
    ])

    builder.add_section_header("Rod Construction", level=2)
    builder.add_bullet_list([
        "<b>Material:</b> Fiberglass (G10) or carbon-fiber segments",
        "<b>OD:</b> 10-16 mm (target: 16 mm with flush joints)",
        "<b>Segments:</b> 50 cm or 100 cm lengths with M12x1.75 threaded couplers",
        "<b>Total depth:</b> Up to 3 m with coupled segments",
        "<b>Weight:</b> ~50-100 g per meter (much lighter than 25mm design)"
    ])

    # Component positioning diagram
    print("Generating Figure 2: Component Positioning...")
    fig_components = create_component_positioning()
    builder.add_figure(
        fig_components,
        "Component positioning along a 3-meter probe assembly. Sensors are distributed to maximize "
        "coverage: TX coil near surface for strong drive, RX coil at mid-depth, and ERT rings at "
        "three depths for resistivity profiling. Threaded joints allow field disassembly for transport."
    )

    # =========================================================================
    # 3.3 CENTRAL ELECTRONICS HUB
    # =========================================================================
    builder.add_section_header("3.3 Central Electronics Hub", level=1)
    builder.add_body_text(
        "All active electronics reside in a central hub unit, typically housed in a rugged "
        "IP65-rated enclosure. This centralized architecture simplifies probe design, reduces "
        "per-probe cost, and enables sophisticated signal processing that would be impractical "
        "in distributed electronics.",
        first_paragraph=True
    )

    # Electronics hub layout
    print("Generating Figure 3: Electronics Hub Layout...")
    fig_hub_layout = create_electronics_hub_layout()
    builder.add_figure(
        fig_hub_layout,
        "Central Electronics Hub internal layout. The hub contains power supply, MCU (ESP32 or STM32), "
        "MIT analog front-end (DDS transmitter and lock-in receiver), ERT current source and measurement "
        "circuitry, and the high-density analog multiplexer array. DB25 connectors provide interface "
        "to Zone Hub trunk cables."
    )

    builder.add_section_header("MIT Driver/Receiver", level=2)
    builder.add_bullet_list([
        "<b>Central DDS sine source</b> (e.g., AD9833) for 2-50 kHz operation",
        "<b>TX driver amplifier</b> - drives all probe TX coils via trunk cables (10-50 mA)",
        "<b>RX low-noise amplifier chain</b> - instrumentation amplifier with G=1000",
        "<b>ADC/lock-in detection</b> - digital synchronous demodulation",
        "<b>MCU</b> (ESP32 or STM32) for control, DSP, and data acquisition"
    ])

    builder.add_section_header("ERT System", level=2)
    builder.add_bullet_list([
        "<b>Howland current source</b> - 0.5-2 mA, programmable via DAC",
        "<b>Polarity reversal</b> - H-bridge for DC measurements with polarity cycling",
        "<b>Voltage measurement</b> - differential amplifier (INA128) + 24-bit ADC",
        "<b>Multiplexer</b> - High-density matrix (CD74HC4067) to switch electrode pairs"
    ])

    builder.add_section_header("Power and Communications", level=2)
    builder.add_bullet_list([
        "<b>Power:</b> 12V or 5V battery pack, 10-20 Ah for full-day field operations",
        "<b>Distribution:</b> Power remains at hub; only signals traverse trunk cables",
        "<b>Data logging:</b> USB or WiFi connection to field tablet",
        "<b>Control:</b> Centralized MCU handles all scheduling and sequencing"
    ])

    # =========================================================================
    # 3.4 SYSTEM BLOCK DIAGRAM
    # =========================================================================
    builder.add_section_header("3.4 System Block Diagram", level=1)
    builder.add_body_text(
        "The complete HIRT system follows a hierarchical architecture: Central Electronics Hub "
        "connects to multiple Zone Hubs via high-density trunk cables, with each Zone Hub "
        "serving as a passive breakout for 4 individual probes. This scalable design supports "
        "arrays from 4 probes (single zone) to 50+ probes (12+ zones).",
        first_paragraph=True
    )

    # System block diagram
    print("Generating Figure 4: System Block Diagram...")
    fig_system = create_system_block_diagram()
    builder.add_figure(
        fig_system,
        "HIRT system block diagram showing the hierarchical architecture. The Central Electronics Hub "
        "contains all active circuitry. High-density trunk cables (DB25) connect to passive Zone Hubs, "
        "which distribute signals to individual probes. This architecture minimizes cable complexity "
        "while maintaining signal integrity over 10-20m cable runs."
    )

    # =========================================================================
    # 3.5 ZONE WIRING STRATEGY
    # =========================================================================
    builder.add_section_header("3.5 Scalability Architecture: The Zone Strategy", level=1)
    builder.add_body_text(
        "To scale the system from a small prototype (4-8 probes) to a field-ready array "
        "(20-50+ probes) without creating an unmanageable cable harness, HIRT employs a "
        "<b>Zone Wiring Strategy</b>. This approach divides the array into logical zones, "
        "each containing 4 probes connected through a local Zone Hub.",
        first_paragraph=True
    )

    builder.add_section_header("The Challenge", level=2)
    builder.add_body_text(
        "A single passive probe requires 8-12 analog conductors (TX coil pair, RX coil pair, "
        "3x ERT rings, Shield, ID). Connecting 25 probes directly to a central hub would require "
        "terminating ~250-300 conductors at a single panel, creating severe mechanical congestion "
        "and crosstalk risks.",
        first_paragraph=True
    )

    builder.add_section_header("The Solution: Passive Zone Hubs", level=2)
    builder.add_numbered_list([
        "<b>Probes connect to a local Zone Hub:</b> A small, passive IP65 box placed on the ground near the probe cluster.",
        "<b>Zone Hubs connect to the Main Unit:</b> Via a single high-quality, shielded trunk cable (DB25 or 37-pin connector).",
        "<b>Central Hub manages Zones:</b> The hub switches between trunk lines to address specific probes via the multiplexer array."
    ])

    # Zone Hub topology
    print("Generating Figure 5: Zone Hub Topology...")
    fig_zone = create_zone_hub_topology()
    builder.add_figure(
        fig_zone,
        "Zone Hub wiring topology. Each passive Zone Hub aggregates connections from 4 probes "
        "into a single DB25 trunk cable. Terminal blocks inside the Zone Hub provide strain relief "
        "and easy field replacement of individual probe cables. This modular approach dramatically "
        "simplifies field deployment."
    )

    # Trunk cable connections
    print("Generating Figure 6: Trunk Cable Connections...")
    fig_trunk = create_trunk_cable_diagram()
    builder.add_figure(
        fig_trunk,
        "Trunk cable DB25 pin assignments for a 4-probe zone. Each probe uses 7 conductors "
        "(TX pair, RX pair, 3 ERT rings), with shared shield and zone ID pins. Shielded twisted-pair "
        "cables maintain signal integrity over 10-20m runs from Zone Hub to Central Hub.",
        width=CONTENT_WIDTH * 0.9
    )

    builder.add_info_box(
        "ZONE WIRING BENEFITS & NOISE IMMUNITY",
        [
            "Modular Deployment: Setup involves running a few thick cables rather than dozens of thin ones",
            "Signal Integrity: Trunk cables utilize individual shielded twisted pairs for all analog lines",
            "Noise Rejection: Differential signaling (balanced TX drive, instrumentation amp RX) rejects common-mode noise",
            "Field Repair: A damaged probe cable only affects one local zone, not the main harness",
            "Scalability: Add zones as needed without redesigning the core system"
        ]
    )

    # =========================================================================
    # 3.6 SIGNAL FLOW DIAGRAMS
    # =========================================================================
    builder.add_section_header("3.6 Signal Flow Paths", level=1)

    builder.add_section_header("MIT Signal Path", level=2)
    builder.add_body_text(
        "The MIT measurement chain begins with the MCU generating a sine wave via Direct Digital "
        "Synthesis (DDS), which is converted to analog, filtered, and amplified to drive the TX coil. "
        "The induced signal in the RX coil passes through a high-gain amplifier chain before "
        "synchronous demodulation extracts amplitude and phase information.",
        first_paragraph=True
    )

    # MIT signal flow
    print("Generating Figure 7: MIT Signal Flow...")
    fig_mit = create_mit_signal_flow()
    builder.add_figure(
        fig_mit,
        "MIT signal flow from DDS generation through TX drive and RX amplification. The lock-in "
        "detection uses the reference signal from the DDS to perform synchronous demodulation, "
        "extracting both in-phase (resistive) and quadrature (reactive) components. Operating "
        "frequency range: 2-50 kHz."
    )

    builder.add_section_header("ERT Signal Path", level=2)
    builder.add_body_text(
        "The ERT system injects a known current (0.5-2 mA) between selected electrode pairs and "
        "measures the resulting voltage distribution. A Howland current source provides stable "
        "current injection regardless of load impedance, while a precision differential amplifier "
        "rejects common-mode noise from the measurement electrodes.",
        first_paragraph=True
    )

    # ERT signal flow
    print("Generating Figure 8: ERT Signal Flow...")
    fig_ert = create_ert_signal_flow()
    builder.add_figure(
        fig_ert,
        "ERT signal flow showing current injection and voltage measurement paths. The Howland "
        "current source provides output impedance >1M ohm, ensuring stable injection regardless "
        "of electrode contact resistance. Polarity reversal (every 1-2 seconds) cancels electrode "
        "polarization effects."
    )

    # =========================================================================
    # 3.7 MULTIPLEXER TOPOLOGY
    # =========================================================================
    builder.add_section_header("3.7 Multiplexer Switching Topology", level=1)
    builder.add_body_text(
        "The analog multiplexer array is central to HIRT's ability to address any probe in "
        "the array. Using CD74HC4067 16:1 analog multiplexers, the system can route TX drive "
        "signals to any probe's TX coil and simultaneously route any RX coil to the receiver "
        "chain. Similar switching enables flexible ERT electrode pair selection.",
        first_paragraph=True
    )

    # MUX topology
    print("Generating Figure 9: Multiplexer Topology...")
    fig_mux = create_mux_topology()
    builder.add_figure(
        fig_mux,
        "Multiplexer switching topology showing channel allocation across MIT and ERT subsystems. "
        "The MCU controls all MUX address lines, enabling any-to-any probe addressing. Total "
        "capacity: 80+ analog channels (32 MIT + 48+ ERT) using cascaded 16:1 multiplexers."
    )

    # =========================================================================
    # 3.8 ARRAY CONFIGURATIONS
    # =========================================================================
    builder.add_section_header("3.8 Array Configuration Options", level=1)
    builder.add_body_text(
        "The HIRT system supports multiple array geometries optimized for different investigation "
        "scenarios. Array configuration affects both lateral resolution and the distribution of "
        "ray paths through the target volume. The following standard configurations address "
        "common field requirements.",
        first_paragraph=True
    )

    # Array configurations
    print("Generating Figure 10: Array Configurations...")
    fig_arrays = create_array_configurations()
    builder.add_figure(
        fig_arrays,
        "Standard array configuration options. (a) Square grid provides uniform coverage for "
        "unknown target locations. (b) Perimeter array enables safe standoff from suspected "
        "UXO while maintaining imaging capability. (c) Dense center configuration concentrates "
        "probes around a known anomaly for maximum resolution.",
        width=CONTENT_WIDTH * 0.95
    )

    # Configuration table
    config_data = [
        ['Configuration', 'Best For', 'Probe Count', 'Spacing'],
        ['Square Grid', 'Unknown target locations', '16-36 (4x4 to 6x6)', '1.5-2.0 m'],
        ['Perimeter', 'UXO/hazard standoff', '12-16', '1.5-2.0 m'],
        ['Dense Center', 'Known anomaly detail', '17-25', '0.75-1.0 m center'],
    ]
    builder.add_table(config_data, caption="Array configuration selection guide.")

    # =========================================================================
    # 3.9 INSERTION METHODS
    # =========================================================================
    builder.add_section_header("3.9 Insertion Methods", level=1)
    builder.add_body_text(
        "The slim 16mm probe profile enables multiple insertion methods depending on soil "
        "conditions. All methods prioritize minimal disturbance and safe operation.",
        first_paragraph=True
    )

    builder.add_numbered_list([
        "<b>Hand Auger:</b> 18-20 mm hand auger creates clearance hole, probe inserted without force",
        "<b>Pilot Rod:</b> 8-10 mm steel rod driven to depth, wiggled to ~14 mm, removed before probe insertion",
        "<b>Direct Push:</b> In sandy loam, probe may push directly (requires hardened tip)",
        "<b>Water-Jet:</b> In sand, water lance fluidizes soil; probe inserted as water drains"
    ])

    builder.add_warning_box(
        "INSERTION SAFETY - UXO SITES",
        [
            "Never hammer or drive probes directly - use pilot hole methods only",
            "Professional EOD clearance required before any insertion operations",
            "Maintain 100m exclusion zone during pilot rod operations",
            "Use perimeter-only deployment when UXB suspected at center"
        ]
    )

    # =========================================================================
    # 3.10 ADVANTAGES
    # =========================================================================
    builder.add_section_header("3.10 Advantages of Micro-Probe Design", level=1)
    builder.add_numbered_list([
        "<b>Minimal Intrusion:</b> ~10x less ground disturbance than 25mm designs",
        "<b>Easy Insertion:</b> Lightweight probes (~100g/m) require minimal force",
        "<b>Better Contact:</b> Slurry/water in insertion hole improves ERT electrode contact",
        "<b>Flexible Deployment:</b> Can achieve denser spacing; easy removal and backfill",
        "<b>Simplified Electronics:</b> Centralized hub is easier to maintain; passive probes are more reliable",
        "<b>Archaeology-Friendly:</b> Acceptable for sensitive contexts with minimal visual impact"
    ])

    builder.add_section_header("3.11 Software Pipeline Roadmap", level=1)
    builder.add_body_text(
        "While this document focuses on hardware, the HIRT system is designed to output data compatible "
        "with established open-source inversion frameworks. The 'Stage 2' software pipeline is currently "
        "in development to leverage these powerful tools:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "<b>EIDORS (Electrical Impedance Tomography and Diffuse Optical Tomography Reconstruction Software):</b> "
        "Primary target for MIT-3D image reconstruction using finite element models.",
        "<b>pyGIMLi (Geophysical Inversion and Modeling Library):</b> "
        "Target framework for coupled ERT/MIT inversion and multi-physics mesh generation.",
        "<b>ResIPy:</b> User-friendly interface for the ERT component processing and inversion."
    ])

    # =========================================================================
    # REFERENCES
    # =========================================================================
    builder.add_references([
        "[1] Daily, W., Ramirez, A., & Johnson, R. (1998). Electrical impedance tomography of a "
        "perchloroethylene release. Journal of Environmental and Engineering Geophysics, 2(3), 189-201.",

        "[2] Binley, A., & Kemna, A. (2005). DC resistivity and induced polarization methods. "
        "In Y. Rubin & S. Hubbard (Eds.), Hydrogeophysics (pp. 129-156). Springer.",

        "[3] Wait, J. R. (1982). Geo-Electromagnetism. Academic Press.",

        "[4] Telford, W. M., Geldart, L. P., & Sheriff, R. E. (1990). Applied Geophysics. "
        "Cambridge University Press.",

        "[5] Butler, D. K. (2001). Potential fields methods for location of unexploded ordnance. "
        "The Leading Edge, 20(8), 890-895.",
    ])

    # Build the PDF
    print("\nBuilding PDF document...")
    output_path = builder.build()
    print(f"\nSuccess! PDF generated at: {output_path}")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    main()
