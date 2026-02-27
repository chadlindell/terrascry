"""
HIRT Whitepaper - System Architecture Diagrams Module

Functions for creating system block diagrams, probe cross-sections, zone hub topology,
array configurations, component positioning, and signal flow diagrams.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch
)
import matplotlib.patheffects as path_effects
import numpy as np
from io import BytesIO

# Color palette
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
    'light_blue': '#ebf8ff',
    'light_green': '#c6f6d5',
    'light_orange': '#feebc8',
    'light_purple': '#faf5ff',
    'light_red': '#fff5f5',
    'ground_tan': '#d4a373',
    'ground_dark': '#654321',
    'probe_body': '#e8d4b8',
    'connector': '#4a5568',
    'tx_coil': '#38a169',
    'rx_coil': '#3182ce',
    'ert_ring': '#ed8936',
}


def draw_block(ax, x, y, width, height, text, color='white', text_color='black',
               fontsize=7, edgecolor='black', linewidth=0.8, rounded=True, zorder=10):
    """Draw a block diagram element with drop shadow and halo text."""
    left = x - width/2
    bottom = y - height/2
    shadow_off = 0.05

    # Drop Shadow
    if rounded:
        shadow = FancyBboxPatch((left + shadow_off, bottom - shadow_off), width, height,
                               boxstyle="round,pad=0.02",
                               facecolor='#00000030', edgecolor='none',
                               zorder=zorder-1)
    else:
        shadow = Rectangle((left + shadow_off, bottom - shadow_off), width, height,
                         facecolor='#00000030', edgecolor='none',
                         zorder=zorder-1)
    ax.add_patch(shadow)

    # Main Block
    if rounded:
        patch = FancyBboxPatch((left, bottom), width, height,
                               boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor=edgecolor,
                               linewidth=linewidth, zorder=zorder)
    else:
        patch = Rectangle((left, bottom), width, height,
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=linewidth, zorder=zorder)
    ax.add_patch(patch)

    # Text with Halo
    txt = ax.text(x, y, text, ha='center', va='center',
                 fontsize=fontsize, color=text_color, zorder=zorder+1)
    txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white', alpha=0.7)])

    return patch


def draw_connection(ax, start, end, style='->', color='black', lw=0.8,
                    connectionstyle=None, manhattan=False):
    """Draw a connection/arrow between two points."""
    if manhattan and connectionstyle is None:
        connectionstyle = "angle,angleA=0,angleB=90,rad=0"

    if connectionstyle:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                  connectionstyle=connectionstyle),
                   zorder=5)
    else:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                   zorder=5)


def create_zone_hub_topology():
    """Create Zone Hub topology diagram showing passive breakout architecture."""
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
        ax.plot([hub_x + 1.2, px - 1.0], [hub_y + 0.5 - i*0.4, py], 'k-', lw=1.5)

        # Probe representation
        ax.add_patch(Rectangle((px - 0.15, py - 0.8), 0.3, 1.0,
                               color=COLORS['probe_body'], ec='black', lw=1))
        ax.add_patch(Rectangle((px - 0.25, py + 0.1), 0.5, 0.3,
                               color=COLORS['connector'], ec='black', lw=1))

        ax.text(px + 0.4, py + 0.2, f'Probe {i+1}', fontsize=8, va='center')
        ax.add_patch(Circle((px, py + 0.25), 0.08, color='black'))

    # Ground line
    ax.axhline(0.8, xmin=0.65, xmax=0.85, color=COLORS['ground_dark'], lw=2)
    ax.fill_between([6.5, 8.5], [0.8, 0.8], [0.3, 0.3], color=COLORS['ground_tan'], alpha=0.3)
    ax.text(7.5, 0.55, 'Ground', fontsize=8, ha='center', color=COLORS['ground_dark'])

    # Benefits callout
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 2.5, 1.5, boxstyle="round,pad=0.05",
                                facecolor='#f0fff4', ec=COLORS['success'], lw=1))
    ax.text(1.45, 1.55, 'Zone Benefits:', fontsize=8, fontweight='bold', color=COLORS['success'])
    benefits = ['Modular deployment', 'Shielded trunk cables', 'Easy field repair', '4 probes per zone']
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


def create_array_configurations():
    """Create diagram showing different array geometry options."""
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
        ax.set_facecolor('#f7fafc')

        if config_type == 'square':
            for i in range(5):
                for j in range(5):
                    ax.add_patch(Circle((i+0.5, j+0.5), 0.15,
                                       color=COLORS['secondary'], ec='black', lw=1))

        elif config_type == 'perimeter':
            ax.add_patch(Rectangle((1.2, 1.2), 2.6, 2.6,
                                   facecolor=COLORS['light_red'], ec=COLORS['warning'],
                                   lw=2, linestyle='--', alpha=0.5))
            ax.text(2.5, 2.5, 'EXCLUSION\nZONE', fontsize=8, ha='center', va='center',
                   color=COLORS['warning'], fontweight='bold')

            for i in range(5):
                ax.add_patch(Circle((i+0.5, 0.5), 0.15, color=COLORS['secondary'], ec='black', lw=1))
                ax.add_patch(Circle((i+0.5, 4.5), 0.15, color=COLORS['secondary'], ec='black', lw=1))
            for j in range(1, 4):
                ax.add_patch(Circle((0.5, j+0.5), 0.15, color=COLORS['secondary'], ec='black', lw=1))
                ax.add_patch(Circle((4.5, j+0.5), 0.15, color=COLORS['secondary'], ec='black', lw=1))

        elif config_type == 'dense':
            corners = [(0.5, 0.5), (0.5, 4.5), (4.5, 0.5), (4.5, 4.5)]
            for x, y in corners:
                ax.add_patch(Circle((x, y), 0.15, color=COLORS['secondary'], ec='black', lw=1))

            edges = [(2.5, 0.5), (2.5, 4.5), (0.5, 2.5), (4.5, 2.5)]
            for x, y in edges:
                ax.add_patch(Circle((x, y), 0.15, color=COLORS['secondary'], ec='black', lw=1))

            for i in range(3):
                for j in range(3):
                    ax.add_patch(Circle((i+1.5, j+1.5), 0.15,
                                       color=COLORS['success'], ec='black', lw=1))

            ax.add_patch(Circle((2.5, 2.5), 0.5,
                               facecolor='none', ec=COLORS['warning'], lw=2, linestyle='--'))
            ax.text(2.5, 2.5, 'Target', fontsize=7, ha='center', va='center', color=COLORS['warning'])

        for i in range(6):
            ax.axhline(i, color='gray', lw=0.3, alpha=0.5)
            ax.axvline(i, color='gray', lw=0.3, alpha=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle('Array Configuration Options', fontsize=12, fontweight='bold',
                color=COLORS['primary'], y=1.02)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_component_positioning():
    """Create diagram showing component positioning along probe rod."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(5, 4.7, 'Component Positioning Along Probe Rod', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    probe_y = 2.5
    probe_start = 0.5
    probe_end = 9.5
    probe_height = 0.3

    ax.add_patch(Rectangle((probe_start, probe_y - probe_height/2), probe_end - probe_start, probe_height,
                           color=COLORS['probe_body'], ec='black', lw=1.5))

    # Standardized coil specs: 34 AWG, 200-300 turns, 8mm x 80mm ferrite core
    components = [
        (0.7, 'Tip', COLORS['primary'], 'Hardened\ntip'),
        (1.5, 'ERT 3', COLORS['ert_ring'], '2.5m'),
        (3.0, 'ERT 2', COLORS['ert_ring'], '1.5m'),
        (4.5, 'Joint', COLORS['gray_light'], 'M12x1.75'),
        (5.5, 'RX Coil', COLORS['rx_coil'], '8x80mm\nferrite'),
        (6.5, 'Joint', COLORS['gray_light'], 'M12x1.75'),
        (7.5, 'TX Coil', COLORS['tx_coil'], '8x80mm\nferrite'),
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

        ax.text(x, probe_y + 0.6, label, fontsize=7, ha='center', fontweight='bold')
        ax.text(x, probe_y - 0.7, desc, fontsize=6, ha='center', color=COLORS['gray_dark'])

    ax.annotate('', xy=(0.7, 1.3), xytext=(9.2, 1.3),
               arrowprops=dict(arrowstyle='<->', color=COLORS['gray_dark'], lw=1))
    ax.text(5, 1.1, 'Total length: 3m (two 1.5m segments)', fontsize=8, ha='center',
           color=COLORS['gray_dark'])

    depth_labels = ['3m depth', '2.5m', '2m', '1.5m', '1m', '0.5m', '0m (surface)']
    for i, label in enumerate(depth_labels):
        x_pos = 0.7 + i * (8.5 / 6)
        ax.plot([x_pos, x_pos], [1.5, 1.7], 'k-', lw=0.5)
        ax.text(x_pos, 1.9, label, fontsize=6, ha='center', rotation=45, color=COLORS['gray_med'])

    ax.add_patch(FancyBboxPatch((0.2, 3.8), 3.5, 0.6, boxstyle="round,pad=0.03",
                                facecolor='#fffbeb', ec=COLORS['orange'], lw=1))
    # Standardized to 34 AWG for consistency
    ax.text(2.0, 4.1, 'Internal wiring: 34 AWG magnet wire routed through rod bore',
           fontsize=7, ha='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_electronics_hub_layout():
    """Create detailed electronics hub internal layout diagram."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(5, 6.7, 'Central Electronics Hub - Internal Layout', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    enclosure = FancyBboxPatch((0.5, 0.5), 9.0, 5.8, boxstyle="round,pad=0.1",
                                facecolor='#f7fafc', edgecolor=COLORS['secondary'], linewidth=2.5)
    ax.add_patch(enclosure)

    # Power section
    ax.add_patch(FancyBboxPatch((0.8, 4.5), 2.0, 1.5, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_orange'], ec=COLORS['orange'], lw=1.5))
    ax.text(1.8, 5.8, 'Power Supply', fontsize=8, fontweight='bold', ha='center')
    draw_block(ax, 1.8, 5.2, 1.4, 0.4, '12V Input', COLORS['light_orange'], fontsize=7)
    draw_block(ax, 1.8, 4.7, 1.4, 0.4, '5V/3.3V LDO', COLORS['light_orange'], fontsize=7)

    # MCU section
    ax.add_patch(FancyBboxPatch((3.2, 4.2), 2.5, 1.8, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_purple'], ec=COLORS['purple'], lw=1.5))
    ax.text(4.45, 5.8, 'Control Unit', fontsize=8, fontweight='bold', ha='center')
    draw_block(ax, 4.45, 5.2, 2.0, 0.5, 'MCU\n(ESP32/STM32)', COLORS['light_purple'], fontsize=7)
    draw_block(ax, 4.45, 4.5, 2.0, 0.5, 'USB/WiFi\nInterface', COLORS['light_purple'], fontsize=7)

    # MIT section
    ax.add_patch(FancyBboxPatch((0.8, 2.2), 2.8, 1.8, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_green'], ec=COLORS['success'], lw=1.5))
    ax.text(2.2, 3.8, 'MIT Analog Front-End', fontsize=8, fontweight='bold', ha='center')
    draw_block(ax, 1.5, 3.2, 1.0, 0.5, 'DDS\n(AD9833)', COLORS['light_green'], fontsize=6)
    draw_block(ax, 2.8, 3.2, 1.0, 0.5, 'TX\nDriver', COLORS['light_green'], fontsize=6)
    draw_block(ax, 1.5, 2.5, 1.0, 0.5, 'Preamp\n(INA)', COLORS['light_green'], fontsize=6)
    draw_block(ax, 2.8, 2.5, 1.0, 0.5, 'Lock-in\nDetect', COLORS['light_green'], fontsize=6)

    # ERT section
    ax.add_patch(FancyBboxPatch((4.0, 2.2), 2.4, 1.8, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_orange'], ec=COLORS['orange'], lw=1.5))
    ax.text(5.2, 3.8, 'ERT Measurement', fontsize=8, fontweight='bold', ha='center')
    draw_block(ax, 4.6, 3.2, 0.9, 0.5, 'Current\nSource', COLORS['light_orange'], fontsize=6)
    draw_block(ax, 5.7, 3.2, 0.9, 0.5, 'Polarity\nSwitch', COLORS['light_orange'], fontsize=6)
    draw_block(ax, 4.6, 2.5, 0.9, 0.5, 'Diff\nAmp', COLORS['light_orange'], fontsize=6)
    draw_block(ax, 5.7, 2.5, 0.9, 0.5, 'ADC\n(16-bit)', COLORS['light_orange'], fontsize=6)

    # Multiplexer section
    ax.add_patch(FancyBboxPatch((0.8, 0.7), 5.8, 1.2, boxstyle="round,pad=0.05",
                                facecolor=COLORS['gray_light'], ec=COLORS['gray_dark'], lw=1.5))
    ax.text(3.7, 1.7, 'High-Density Analog Multiplexer Array', fontsize=8, fontweight='bold', ha='center')
    for i in range(6):
        draw_block(ax, 1.3 + i*0.9, 1.0, 0.7, 0.4, f'MUX\n{i+1}', COLORS['gray_light'], fontsize=5)

    # Connector panel
    ax.add_patch(FancyBboxPatch((7.0, 0.7), 2.2, 5.3, boxstyle="round,pad=0.05",
                                facecolor='white', ec=COLORS['gray_dark'], lw=1.5))
    ax.text(8.1, 5.8, 'Trunk Interface', fontsize=8, fontweight='bold', ha='center')

    for i, label in enumerate(['Zone A', 'Zone B', 'Zone C', 'Zone D']):
        y = 5.0 - i * 1.1
        ax.add_patch(Rectangle((7.3, y - 0.3), 1.6, 0.6,
                               color=COLORS['connector'], ec='black', lw=1))
        ax.text(8.1, y, label, fontsize=7, ha='center', color='white')

    ax.add_patch(Rectangle((7.5, 1.0), 1.2, 0.4, color='black', ec='black', lw=1))
    ax.text(8.1, 1.2, 'USB', fontsize=7, ha='center', color='white')

    for i, (color, label) in enumerate([('green', 'PWR'), ('blue', 'TX'), ('orange', 'RX')]):
        ax.add_patch(Circle((8.8, 5.0 - i*0.3), 0.08, color=color))
        ax.text(8.6, 5.0 - i*0.3, label, fontsize=5, ha='right', va='center')

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


def create_trunk_cable_diagram():
    """Create trunk cable connection diagram showing DB25 pinout."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(5, 5.7, 'Trunk Cable Connections (DB25)', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    conn_x = 2.5
    conn_y = 3.0
    ax.add_patch(FancyBboxPatch((conn_x - 1.5, conn_y - 1.2), 3.0, 2.4, boxstyle="round,pad=0.1",
                                facecolor=COLORS['connector'], ec='black', lw=2))
    ax.text(conn_x, conn_y + 0.9, 'DB25 Connector', fontsize=9, ha='center', color='white', fontweight='bold')

    for row in range(2):
        num_pins = 13 if row == 0 else 12
        y_pos = conn_y + 0.3 - row * 0.6
        for i in range(num_pins):
            x_pos = conn_x - 1.2 + i * 0.2
            ax.add_patch(Circle((x_pos, y_pos), 0.06, color='gold', ec='black', lw=0.5))

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

    ax.plot([conn_x, conn_x], [conn_y - 1.2, 0.5], color=COLORS['gray_dark'], lw=4)
    ax.text(conn_x, 0.3, 'To Zone Hub', fontsize=8, ha='center', color=COLORS['gray_dark'])

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


def create_mit_signal_flow():
    """Create MIT signal path diagram."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(5.5, 3.7, 'MIT Signal Flow Path', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

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

    tx_connections = [(2.1, 2.5), (2.7, 2.5), (3.7, 2.5), (4.2, 2.5),
                      (5.2, 2.5), (5.7, 2.5), (6.7, 2.5), (7.3, 2.5),
                      (8.3, 2.5), (8.9, 2.5)]
    for i in range(0, len(tx_connections), 2):
        draw_connection(ax, tx_connections[i], tx_connections[i+1], color=COLORS['gray_dark'])

    ax.text(2.4, 2.9, 'SPI', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(3.95, 2.9, 'Analog', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(5.45, 2.9, 'Clean', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(7.0, 2.9, '10-50mA', fontsize=6, ha='center', color=COLORS['gray_med'])

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

    rx_connections = [(8.9, 1.2), (8.3, 1.2), (7.3, 1.2), (6.7, 1.2),
                      (5.7, 1.2), (5.2, 1.2), (4.2, 1.2), (3.7, 1.2),
                      (2.7, 1.2), (2.1, 1.2)]
    for i in range(0, len(rx_connections), 2):
        draw_connection(ax, rx_connections[i], rx_connections[i+1], color=COLORS['gray_dark'])

    ax.text(8.0, 0.7, 'uV', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(5.45, 0.7, 'mV', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(3.95, 0.7, '100mV', fontsize=6, ha='center', color=COLORS['gray_med'])

    ax.plot([1.5, 1.5, 10.0, 10.0], [2.2, 0.5, 0.5, 1.2], '--', color=COLORS['orange'], lw=1)
    ax.text(5.5, 0.3, 'Reference signal for lock-in detection', fontsize=7,
           ha='center', color=COLORS['orange'], style='italic')

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


def create_ert_signal_flow():
    """Create ERT signal path diagram."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(5.5, 4.2, 'ERT Signal Flow Path', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

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

    for i in range(len(inj_blocks) - 1):
        x1 = inj_blocks[i][0] + inj_blocks[i][2]/2
        x2 = inj_blocks[i+1][0] - inj_blocks[i+1][2]/2
        draw_connection(ax, (x1, 3.0), (x2, 3.0), color=COLORS['gray_dark'])

    ax.text(2.35, 3.45, 'I2C', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(4.1, 3.45, '0-2.5V', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(5.9, 3.45, '0.5-2mA', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(7.55, 3.45, '+/-', fontsize=6, ha='center', color=COLORS['gray_med'])

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

    for i in range(len(meas_blocks) - 1):
        x1 = meas_blocks[i][0] - meas_blocks[i][2]/2
        x2 = meas_blocks[i+1][0] + meas_blocks[i+1][2]/2
        draw_connection(ax, (x1, 1.7), (x2, 1.7), color=COLORS['gray_dark'])

    ax.text(7.45, 1.2, 'uV-mV', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(5.7, 1.2, 'Filtered', fontsize=6, ha='center', color=COLORS['gray_med'])
    ax.text(3.9, 1.2, 'SPI', fontsize=6, ha='center', color=COLORS['gray_med'])

    ax.annotate('', xy=(10.3, 2.4), xytext=(10.3, 3.0),
               arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=2))
    ax.text(10.5, 2.7, 'I', fontsize=10, color=COLORS['warning'], fontweight='bold')

    ax.plot([10.3, 10.3], [2.0, 2.4], 'k-', lw=1.5)
    ax.text(10.5, 2.2, 'Ground', fontsize=6, color=COLORS['gray_dark'])

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


def create_method_comparison_table():
    """Create visual comparison of HIRT vs alternative geophysical methods."""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(5.5, 6.7, 'Geophysical Method Comparison', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Create visual comparison bars for key parameters
    methods = ['HIRT', 'GPR', 'Magnetometry', 'Surface ERT', 'EMI/Metal Detector']
    method_colors = [COLORS['secondary'], COLORS['accent'], COLORS['purple'],
                     COLORS['success'], COLORS['orange']]

    # Metrics to compare (normalized 0-10 scale)
    depth_capability = [8, 6, 8, 9, 4]  # Depth range capability
    resolution = [9, 7, 3, 2, 3]  # Spatial resolution
    metal_sensitivity = [10, 7, 5, 0, 8]  # Metal detection capability
    soil_sensitivity = [10, 2, 0, 9, 0]  # Soil disturbance sensitivity
    deployment_speed = [3, 7, 9, 7, 10]  # Ease and speed of deployment
    relative_cost = [8, 5, 2, 5, 2]  # Equipment and operational cost

    metrics = [
        ('Depth Capability\n(1-5m)', depth_capability),
        ('Resolution\n(<0.5m)', resolution),
        ('Metal Sensitivity\n(Al+Fe)', metal_sensitivity),
        ('Soil Disturbance\nSensitivity', soil_sensitivity),
        ('Deployment Speed\n(field ready)', deployment_speed),
        ('Relative Cost\n(equipment)', relative_cost),
    ]

    # Draw comparison bars
    bar_height = 0.12
    spacing = 0.25
    metric_y_start = 5.8

    for metric_idx, (metric_name, values) in enumerate(metrics):
        y_base = metric_y_start - metric_idx * 0.9

        # Metric label
        ax.text(0.3, y_base, metric_name, fontsize=7, fontweight='bold',
               va='center', color=COLORS['gray_dark'])

        # Draw bars for each method
        for method_idx, (method, value, color) in enumerate(zip(methods, values, method_colors)):
            y_pos = y_base - method_idx * spacing

            # Background bar (max scale)
            ax.add_patch(Rectangle((2.0, y_pos - bar_height/2), 5.0, bar_height,
                                   facecolor=COLORS['gray_light'], ec='none', alpha=0.3))

            # Value bar
            bar_width = (value / 10.0) * 5.0
            ax.add_patch(Rectangle((2.0, y_pos - bar_height/2), bar_width, bar_height,
                                   facecolor=color, ec='none', alpha=0.8))

            # Method label (only for first metric)
            if metric_idx == 0:
                ax.text(1.8, y_pos, method, fontsize=7, ha='right', va='center',
                       color=color, fontweight='bold')

            # Value label
            if value > 0:
                ax.text(2.0 + bar_width + 0.1, y_pos, f'{value}/10',
                       fontsize=5, va='center', color=COLORS['gray_dark'])

    # Draw legend for bar scale
    ax.text(4.5, 0.3, '0 = Poor/Low    5 = Moderate    10 = Excellent/High',
           fontsize=7, ha='center', style='italic', color=COLORS['gray_med'])

    # Add key strengths callouts
    callout_x = 8.0

    ax.add_patch(FancyBboxPatch((callout_x, 5.5), 2.7, 1.0, boxstyle="round,pad=0.05",
                                facecolor='#f0fff4', ec=COLORS['success'], lw=1.5))
    ax.text(callout_x + 1.35, 6.3, 'HIRT Strengths', fontsize=8,
           fontweight='bold', ha='center', color=COLORS['success'])
    strengths = ['Dual-mode sensing', '3D tomography', 'Al + Fe detection']
    for i, strength in enumerate(strengths):
        ax.text(callout_x + 0.15, 6.0 - i*0.25, f'  {strength}', fontsize=6)

    ax.add_patch(FancyBboxPatch((callout_x, 4.0), 2.7, 1.0, boxstyle="round,pad=0.05",
                                facecolor='#fff5f5', ec=COLORS['warning'], lw=1.5))
    ax.text(callout_x + 1.35, 4.8, 'HIRT Limitations', fontsize=8,
           fontweight='bold', ha='center', color=COLORS['warning'])
    limits = ['Requires probe insertion', 'Slow deployment (2-4h)', 'Higher cost']
    for i, limit in enumerate(limits):
        ax.text(callout_x + 0.15, 4.5 - i*0.25, f'  {limit}', fontsize=6)

    ax.add_patch(FancyBboxPatch((callout_x, 2.5), 2.7, 1.0, boxstyle="round,pad=0.05",
                                facecolor='#fffbeb', ec=COLORS['orange'], lw=1.5))
    ax.text(callout_x + 1.35, 3.3, 'When to Use HIRT', fontsize=8,
           fontweight='bold', ha='center', color=COLORS['orange'])
    when_use = ['Soft soils (1-3m)', 'Need metal + soil data', 'Crosshole geometry']
    for i, condition in enumerate(when_use):
        ax.text(callout_x + 0.15, 3.0 - i*0.25, f'  {condition}', fontsize=6)

    ax.add_patch(FancyBboxPatch((callout_x, 1.0), 2.7, 1.0, boxstyle="round,pad=0.05",
                                facecolor='#faf5ff', ec=COLORS['purple'], lw=1.5))
    ax.text(callout_x + 1.35, 1.8, 'Avoid HIRT When', fontsize=8,
           fontweight='bold', ha='center', color=COLORS['purple'])
    avoid = ['Rocky ground', 'Saline/saturated soil', 'EMI interference zones']
    for i, condition in enumerate(avoid):
        ax.text(callout_x + 0.15, 1.5 - i*0.25, f'  {condition}', fontsize=6)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_mux_topology():
    """Create multiplexer switching topology diagram."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(5, 6.7, 'Multiplexer Switching Topology', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    draw_block(ax, 1.5, 5.5, 1.8, 1.0, 'MCU\n(Control)', COLORS['light_purple'])

    ax.plot([2.4, 3.5], [5.5, 5.5], 'k-', lw=1)
    ax.text(2.95, 5.7, 'Addr (4-bit)', fontsize=6, color=COLORS['gray_med'])

    ax.add_patch(FancyBboxPatch((3.5, 3.0), 3.5, 3.5, boxstyle="round,pad=0.1",
                                facecolor=COLORS['gray_light'], ec=COLORS['gray_dark'], lw=2))
    ax.text(5.25, 6.2, 'Analog MUX Array', fontsize=9, fontweight='bold', ha='center')

    draw_block(ax, 4.25, 5.3, 1.0, 0.6, 'MUX\nMIT TX', COLORS['light_green'], fontsize=6)
    draw_block(ax, 5.75, 5.3, 1.0, 0.6, 'MUX\nMIT RX', COLORS['light_green'], fontsize=6)

    for i in range(4):
        draw_block(ax, 4.0 + i*0.8, 4.2, 0.7, 0.5, f'ERT\n{i+1}', COLORS['light_orange'], fontsize=5)

    draw_block(ax, 4.25, 3.4, 1.0, 0.5, 'TX Out', COLORS['tx_coil'], fontsize=7)
    draw_block(ax, 5.75, 3.4, 1.0, 0.5, 'RX In', COLORS['rx_coil'], fontsize=7)

    ax.text(8.5, 6.0, 'Trunk Cables', fontsize=8, fontweight='bold', ha='center')

    probe_y = [5.2, 4.4, 3.6, 2.8]
    for i, y in enumerate(probe_y):
        ax.plot([7.0, 7.8], [4.0, y], 'k-', lw=1)
        ax.add_patch(Rectangle((7.8, y - 0.25), 1.8, 0.5,
                               facecolor='white', ec=COLORS['gray_dark'], lw=1))
        ax.text(8.7, y, f'Zone {chr(65+i)} (4 probes)', fontsize=7, ha='center', va='center')

    ax.annotate('', xy=(4.25, 4.8), xytext=(4.25, 5.0),
               arrowprops=dict(arrowstyle='->', color=COLORS['tx_coil'], lw=1.5))
    ax.annotate('', xy=(5.75, 5.0), xytext=(5.75, 4.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['rx_coil'], lw=1.5))

    for i in range(4):
        ax.plot([4.0 + i*0.8, 4.0 + i*0.8], [3.7, 3.95], 'k-', lw=0.5)

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
