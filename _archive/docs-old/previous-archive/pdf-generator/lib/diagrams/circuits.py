"""
HIRT PDF Generator - Circuit Diagrams Module

Functions for creating schematics, block diagrams, and signal flow diagrams.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch,
    Arc, Wedge, PathPatch
)
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
import numpy as np
from io import BytesIO

# Configure Matplotlib for Scientific Publication Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.8,
    'patch.linewidth': 0.5,
    'text.color': '#222222',
    'axes.labelcolor': '#222222',
    'xtick.color': '#222222',
    'ytick.color': '#222222',
})

# Color palette (Muted Academic)
COLORS = {
    'primary': '#1a365d',      # Navy
    'secondary': '#2c5282',    # Muted Blue
    'accent': '#3182ce',       # Sky Blue
    'success': '#2f855a',      # Forest Green (Darker)
    'warning': '#c53030',      # Brick Red
    'orange': '#c05621',       # Burnt Orange
    'purple': '#6b46c1',       # Muted Purple
    'gray_dark': '#2d3748',    # Slate
    'gray_med': '#718096',     # Cool Gray
    'gray_light': '#cbd5e0',   # Light Gray
    'light_green': '#f0fff4',
    'light_orange': '#fffaf0',
    'light_purple': '#faf5ff',
    'light_blue': '#ebf8ff',
    'light_red': '#fff5f5',
    'ground_tan': '#d4a373',
}


def draw_block(ax, x, y, width, height, text, color='white', text_color='black',
               fontsize=7, edgecolor='black', linewidth=0.8, rounded=True, zorder=10):
    """
    Draw a block diagram element with drop shadow and halo text.
    """
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
    """
    Draw a connection/arrow between two points.
    """
    if manhattan and connectionstyle is None:
        # Default orthogonal routing: horizontal then vertical
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


def draw_signal_flow(ax, points, labels=None, style='->', color='black', lw=0.8):
    """
    Draw a signal flow path through multiple points.
    """
    for i in range(len(points) - 1):
        draw_connection(ax, points[i], points[i+1], style=style, color=color, lw=lw)

    if labels:
        for point, label in zip(points, labels):
            ax.text(point[0], point[1] + 0.3, label, ha='center', fontsize=7)


def draw_resistor(ax, x, y, width=0.6, height=0.2, vertical=False):
    """Draw a resistor symbol."""
    if vertical:
        width, height = height, width

    zigzag_points = []
    n_zigs = 5
    for i in range(n_zigs * 2 + 1):
        px = x - width/2 + (width * i) / (n_zigs * 2)
        py = y + height/2 * (1 if i % 2 == 1 else -1) if 0 < i < n_zigs*2 else y
        zigzag_points.append((px, py))

    xs, ys = zip(*zigzag_points)
    ax.plot(xs, ys, 'k-', lw=1.5)


def draw_capacitor(ax, x, y, width=0.3, gap=0.08):
    """Draw a capacitor symbol."""
    # Two parallel lines
    ax.plot([x - gap/2, x - gap/2], [y - width/2, y + width/2], 'k-', lw=2)
    ax.plot([x + gap/2, x + gap/2], [y - width/2, y + width/2], 'k-', lw=2)


def draw_inductor(ax, x, y, width=0.6, height=0.15, n_loops=4):
    """Draw an inductor symbol (series of arcs)."""
    loop_width = width / n_loops
    for i in range(n_loops):
        cx = x - width/2 + loop_width/2 + i * loop_width
        arc = Arc((cx, y), loop_width * 0.8, height * 2,
                 angle=0, theta1=0, theta2=180, lw=1.5)
        ax.add_patch(arc)


def draw_opamp(ax, x, y, size=0.5, flip=False):
    """
    Draw an op-amp triangle symbol.

    Args:
        ax: matplotlib Axes object
        x, y: Center position
        size: Triangle size
        flip: If True, flip horizontally
    """
    if flip:
        points = [(x + size, y), (x - size/2, y + size/2), (x - size/2, y - size/2)]
    else:
        points = [(x - size, y), (x + size/2, y + size/2), (x + size/2, y - size/2)]

    triangle = Polygon(points, facecolor='white', edgecolor='black', lw=1.5)
    ax.add_patch(triangle)

    # + and - inputs
    if flip:
        ax.text(x - size/2 + 0.1, y + size/4, '-', fontsize=8, va='center')
        ax.text(x - size/2 + 0.1, y - size/4, '+', fontsize=8, va='center')
    else:
        ax.text(x - size/2 - 0.05, y + size/4, '+', fontsize=8, va='center', ha='right')
        ax.text(x - size/2 - 0.05, y - size/4, '-', fontsize=8, va='center', ha='right')


def draw_coil_symbol(ax, x, y, width=0.4, n_turns=5):
    """Draw a simplified coil symbol."""
    for i in range(n_turns):
        cx = x - width/2 + (width * i) / (n_turns - 1)
        circle = Circle((cx, y), 0.05, facecolor='none', edgecolor='black', lw=1)
        ax.add_patch(circle)


def draw_ground_symbol(ax, x, y, size=0.15):
    """Draw an electrical ground symbol."""
    ax.plot([x, x], [y, y - size/2], 'k-', lw=1.5)
    for i, w in enumerate([size, size*0.7, size*0.4]):
        ax.plot([x - w/2, x + w/2], [y - size/2 - i*size/4, y - size/2 - i*size/4],
               'k-', lw=1.5)


def draw_vcc_symbol(ax, x, y, label='VCC'):
    """Draw a power supply symbol."""
    ax.plot([x, x], [y, y + 0.15], 'k-', lw=1.5)
    ax.plot([x - 0.1, x + 0.1], [y + 0.15, y + 0.15], 'k-', lw=2)
    ax.text(x, y + 0.25, label, ha='center', fontsize=7)


def create_mit_tx_chain():
    """
    Create MIT transmitter chain block diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Block positions
    blocks = [
        (1.0, 1.5, 1.4, 0.8, 'MCU\n(DDS)', COLORS['light_purple']),
        (2.8, 1.5, 1.2, 0.8, 'DAC', COLORS['light_blue']),
        (4.4, 1.5, 1.4, 0.8, 'LP Filter\n(fc=50kHz)', COLORS['light_green']),
        (6.2, 1.5, 1.4, 0.8, 'Power\nDriver', COLORS['light_orange']),
        (8.0, 1.5, 1.2, 0.8, 'TX Coil', COLORS['success']),
    ]

    for x, y, w, h, text, color in blocks:
        draw_block(ax, x, y, w, h, text, color=color, fontsize=8)

    # Connections
    connections = [
        ((1.7, 1.5), (2.2, 1.5)),
        ((3.4, 1.5), (3.7, 1.5)),
        ((5.1, 1.5), (5.5, 1.5)),
        ((6.9, 1.5), (7.4, 1.5)),
    ]
    for start, end in connections:
        draw_connection(ax, start, end, color=COLORS['gray_dark'])

    # Signal labels
    ax.text(1.95, 1.85, 'SPI', fontsize=7, ha='center', color=COLORS['gray_dark'])
    ax.text(3.55, 1.85, 'Analog', fontsize=7, ha='center', color=COLORS['gray_dark'])
    ax.text(5.3, 1.85, 'Clean\nSine', fontsize=6, ha='center', color=COLORS['gray_dark'])
    ax.text(7.15, 1.85, '10-50mA', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Title
    ax.text(5, 2.7, 'MIT Transmitter Signal Chain', fontsize=11,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Frequency annotation
    ax.text(1.0, 0.7, 'f = 2-50 kHz', fontsize=8, ha='center',
           style='italic', color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_mit_rx_chain():
    """
    Create MIT receiver chain block diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Block positions
    blocks = [
        (1.0, 2.0, 1.2, 0.8, 'RX Coil', COLORS['accent']),
        (2.6, 2.0, 1.2, 0.8, 'Preamp\n(G=10)', COLORS['light_green']),
        (4.2, 2.0, 1.4, 0.8, 'Inst Amp\n(G=100)', COLORS['light_green']),
        (5.9, 2.0, 1.2, 0.8, 'BP Filter', COLORS['light_blue']),
        (7.5, 2.0, 1.2, 0.8, 'ADC\n(16-bit)', COLORS['light_purple']),
        (9.0, 2.0, 0.8, 0.8, 'MCU', COLORS['light_purple']),
    ]

    for x, y, w, h, text, color in blocks:
        draw_block(ax, x, y, w, h, text, color=color, fontsize=8)

    # Lock-in detection branch
    draw_block(ax, 7.5, 0.8, 1.4, 0.7, 'Lock-in\nDetection', COLORS['light_orange'], fontsize=8)

    # Main connections
    connections = [
        ((1.6, 2.0), (2.0, 2.0)),
        ((3.2, 2.0), (3.5, 2.0)),
        ((4.9, 2.0), (5.3, 2.0)),
        ((6.5, 2.0), (6.9, 2.0)),
        ((8.1, 2.0), (8.6, 2.0)),
    ]
    for start, end in connections:
        draw_connection(ax, start, end, color=COLORS['gray_dark'])

    # Lock-in connections
    ax.plot([7.5, 7.5], [1.6, 1.15], 'k-', lw=1)
    ax.plot([8.1, 8.4], [0.8, 0.8], 'k-', lw=1)
    ax.plot([8.4, 8.4], [0.8, 1.6], 'k-', lw=1)
    draw_connection(ax, (8.4, 1.6), (9.0, 1.6), color=COLORS['gray_dark'])

    # Signal labels
    ax.text(1.8, 2.35, 'uV', fontsize=7, ha='center', color=COLORS['gray_dark'])
    ax.text(3.35, 2.35, 'mV', fontsize=7, ha='center', color=COLORS['gray_dark'])
    ax.text(5.1, 2.35, '100mV', fontsize=7, ha='center', color=COLORS['gray_dark'])
    ax.text(6.7, 2.35, 'Filtered', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Reference signal
    ax.annotate('REF from TX', xy=(6.8, 0.8), xytext=(5.5, 0.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['orange']),
               fontsize=7, color=COLORS['orange'])

    # Title
    ax.text(5, 3.2, 'MIT Receiver Signal Chain', fontsize=11,
           fontweight='bold', ha='center', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_ert_current_source():
    """
    Create ERT Howland current source schematic.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Op-amp
    draw_opamp(ax, 3.5, 2.5, size=0.6)

    # Input resistors
    draw_resistor(ax, 1.5, 3.0, width=0.5)
    ax.text(1.5, 3.4, 'R1', fontsize=8, ha='center')
    ax.plot([0.5, 1.25], [3.0, 3.0], 'k-', lw=1)
    ax.plot([1.75, 2.9], [3.0, 3.0], 'k-', lw=1)
    ax.plot([2.9, 2.9], [3.0, 2.7], 'k-', lw=1)

    draw_resistor(ax, 1.5, 2.0, width=0.5)
    ax.text(1.5, 1.6, 'R2', fontsize=8, ha='center')
    ax.plot([0.5, 1.25], [2.0, 2.0], 'k-', lw=1)
    ax.plot([1.75, 2.9], [2.0, 2.0], 'k-', lw=1)
    ax.plot([2.9, 2.9], [2.0, 2.3], 'k-', lw=1)

    # Feedback resistors
    draw_resistor(ax, 5.0, 3.5, width=0.5)
    ax.text(5.0, 3.9, 'R3', fontsize=8, ha='center')
    ax.plot([4.1, 4.75], [2.7, 2.7], 'k-', lw=1)
    ax.plot([4.75, 4.75], [2.7, 3.5], 'k-', lw=1)
    ax.plot([5.25, 5.5], [3.5, 3.5], 'k-', lw=1)
    ax.plot([5.5, 5.5], [3.5, 2.5], 'k-', lw=1)

    draw_resistor(ax, 5.0, 1.5, width=0.5)
    ax.text(5.0, 1.1, 'R4', fontsize=8, ha='center')
    ax.plot([4.1, 4.75], [2.3, 2.3], 'k-', lw=1)
    ax.plot([4.75, 4.75], [2.3, 1.5], 'k-', lw=1)
    ax.plot([5.25, 6.0], [1.5, 1.5], 'k-', lw=1)

    # Output
    ax.plot([4.1, 5.5], [2.5, 2.5], 'k-', lw=1)
    ax.plot([5.5, 6.0], [2.5, 2.5], 'k-', lw=1)

    # Load
    draw_resistor(ax, 6.0, 2.0, width=0.5, vertical=True)
    ax.text(6.4, 2.0, 'Z_load\n(Ground)', fontsize=7, va='center')

    # Ground symbols
    draw_ground_symbol(ax, 6.0, 1.4)
    draw_ground_symbol(ax, 0.5, 1.5)
    ax.plot([0.5, 0.5], [2.0, 1.65], 'k-', lw=1)

    # Input labels
    ax.text(0.3, 3.0, 'V_in+', fontsize=8, ha='right')
    ax.text(0.3, 2.0, 'V_in-', fontsize=8, ha='right')

    # Output current label
    ax.annotate('I_out = V_in / R', xy=(5.5, 2.5), xytext=(5.8, 3.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['success']),
               fontsize=9, color=COLORS['success'])

    # Title
    ax.text(3.5, 4.7, 'Howland Current Source (ERT)', fontsize=11,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Notes
    ax.text(3.5, 0.3, 'R1 = R2 = R3 = R4 for ideal current source\nI_out = 0.5-2 mA typical',
           fontsize=8, ha='center', color=COLORS['gray_dark'], style='italic')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_system_block_diagram():
    """
    Create complete HIRT system block diagram with upgraded visuals.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Central Hub box (Background container)
    hub = FancyBboxPatch((0.3, 2.5), 4.5, 4.5, boxstyle="round,pad=0.1",
                         facecolor=COLORS['light_blue'], edgecolor=COLORS['secondary'],
                         linewidth=2, zorder=1)
    ax.add_patch(hub)
    ax.text(2.55, 6.7, 'Central Electronics Hub', fontsize=12, fontweight='bold',
           ha='center', color=COLORS['primary'], zorder=20)

    # MCU block
    draw_block(ax, 2.55, 5.8, 2.2, 0.8, 'MCU (STM32/ESP32)\nControl + DSP', COLORS['light_purple'])

    # MIT TX
    draw_block(ax, 1.4, 4.5, 1.6, 0.8, 'MIT TX\n(DDS + Driver)', COLORS['light_green'])

    # MIT RX
    draw_block(ax, 3.7, 4.5, 1.6, 0.8, 'MIT RX\n(Preamp + ADC)', COLORS['light_green'])

    # ERT
    draw_block(ax, 1.4, 3.2, 1.6, 0.8, 'ERT Source\n(Howland)', COLORS['light_orange'])

    # ERT RX
    draw_block(ax, 3.7, 3.2, 1.6, 0.8, 'ERT Measure\n(Diff Amp)', COLORS['light_orange'])

    # Mux
    draw_block(ax, 2.55, 1.8, 3.0, 0.7, 'High-Density Analog MUX\n(CD74HC4067)', COLORS['gray_light'])

    # Internal connections (Manhattan)
    # MCU to Mux
    draw_connection(ax, (2.55, 5.4), (2.55, 2.15), style='->', lw=1, color=COLORS['gray_dark'])
    
    # MCU to TX/RX
    draw_connection(ax, (2.55, 5.4), (1.4, 4.9), style='->', lw=1, color=COLORS['gray_dark'], manhattan=True)
    draw_connection(ax, (2.55, 5.4), (3.7, 4.9), style='->', lw=1, color=COLORS['gray_dark'], manhattan=True)

    # TX/RX to Mux
    draw_connection(ax, (1.4, 4.1), (1.4, 2.15), style='->', lw=1, color=COLORS['gray_dark'])
    draw_connection(ax, (3.7, 4.1), (3.7, 2.15), style='->', lw=1, color=COLORS['gray_dark'])
    draw_connection(ax, (1.4, 2.8), (1.4, 2.15), style='->', lw=1, color=COLORS['gray_dark'])
    draw_connection(ax, (3.7, 2.8), (3.7, 2.15), style='->', lw=1, color=COLORS['gray_dark'])

    # Trunk cable to Zone Hub
    draw_connection(ax, (4.8, 3.5), (6.0, 3.5), style='-', lw=4, color=COLORS['gray_dark'])
    ax.text(5.4, 3.8, 'Trunk\nCable', fontsize=8, ha='center', color=COLORS['gray_dark'], zorder=20)

    # Zone Hub
    zone_hub = FancyBboxPatch((6.0, 2.5), 2.0, 2.2, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor=COLORS['gray_dark'],
                              linewidth=1.5, zorder=5)
    ax.add_patch(zone_hub)
    ax.text(7.0, 4.5, 'Zone Hub\n(Passive)', fontsize=10, fontweight='bold',
           ha='center', color=COLORS['gray_dark'], zorder=20)
    ax.text(7.0, 3.2, 'DB25\nConnector\n+\nTerminals', fontsize=8, ha='center', zorder=20)

    # Probe connections from Zone Hub
    probe_y = [4.2, 3.7, 3.2, 2.7]
    for i, py in enumerate(probe_y):
        draw_connection(ax, (8.0, py), (9.0, py - 0.2), style='-', lw=1, color=COLORS['gray_dark'])

    # Probes (Shaded Cylinders)
    for i in range(4):
        py = 2.2 + i * 0.8
        # Draw probe as block for now, but shaded
        draw_block(ax, 10.0, py, 1.8, 0.5, f'Probe {4-i}\n(Passive)', COLORS['light_green'], fontsize=8)

    # Ground representation
    ax.fill_between([8.5, 11.5], [1.2, 1.2], [0.5, 0.5], color=COLORS['ground_tan'], alpha=0.3, zorder=0)
    ax.axhline(1.2, xmin=0.71, xmax=0.95, color='#654321', lw=2, zorder=1)
    ax.text(10.0, 0.85, 'Ground', fontsize=9, ha='center', color='#654321', zorder=20)

    # USB/Power connection
    draw_block(ax, 2.55, 0.8, 1.8, 0.6, 'USB/Power', COLORS['light_red'], fontsize=9)
    draw_connection(ax, (2.55, 1.1), (2.55, 1.8), style='->', color='black', lw=1.5)

    # Data output arrow
    draw_connection(ax, (1.4, 5.8), (0.2, 5.8), style='->', color=COLORS['accent'], lw=2)
    ax.text(0.1, 6.0, 'Data\nOut', fontsize=9, ha='left', color=COLORS['accent'], fontweight='bold', zorder=20)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_power_distribution():
    """
    Create power distribution diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Power input
    draw_block(ax, 1.0, 2.5, 1.2, 1.2, 'Battery\n12V\n(LiPo 3S)', COLORS['light_orange'], fontsize=8)

    # Main regulators
    draw_block(ax, 3.0, 3.8, 1.4, 0.7, '5V Reg\n(LDO)', COLORS['light_green'], fontsize=8)
    draw_block(ax, 3.0, 2.5, 1.4, 0.7, '3.3V Reg\n(LDO)', COLORS['light_blue'], fontsize=8)
    draw_block(ax, 3.0, 1.2, 1.4, 0.7, '-5V Gen\n(Charge Pump)', COLORS['light_purple'], fontsize=8)

    # Connections from battery
    ax.plot([1.6, 2.0], [2.5, 2.5], 'k-', lw=2)
    ax.plot([2.0, 2.0], [1.2, 3.8], 'k-', lw=2)
    ax.plot([2.0, 2.3], [3.8, 3.8], 'k-', lw=1.5)
    ax.plot([2.0, 2.3], [2.5, 2.5], 'k-', lw=1.5)
    ax.plot([2.0, 2.3], [1.2, 1.2], 'k-', lw=1.5)

    # Loads
    draw_block(ax, 5.5, 4.2, 2.0, 0.5, 'Power Driver (MIT TX)', COLORS['gray_light'], fontsize=7)
    draw_block(ax, 5.5, 3.5, 2.0, 0.5, 'MCU + Digital', COLORS['gray_light'], fontsize=7)
    draw_block(ax, 5.5, 2.8, 2.0, 0.5, 'ADC + Reference', COLORS['gray_light'], fontsize=7)
    draw_block(ax, 5.5, 2.1, 2.0, 0.5, 'Analog Front-End', COLORS['gray_light'], fontsize=7)
    draw_block(ax, 5.5, 1.0, 2.0, 0.5, 'Op-Amp Rails', COLORS['gray_light'], fontsize=7)

    # Power rails
    ax.plot([3.7, 4.5], [3.8, 3.8], 'r-', lw=1.5)
    ax.plot([4.5, 4.5], [3.5, 4.2], 'r-', lw=1.5)
    ax.text(4.1, 4.0, '5V', fontsize=7, color='red')

    ax.plot([3.7, 4.5], [2.5, 2.5], 'b-', lw=1.5)
    ax.plot([4.5, 4.5], [2.1, 3.5], 'b-', lw=1.5)
    ax.text(4.1, 2.7, '3.3V', fontsize=7, color='blue')

    ax.plot([3.7, 4.5], [1.2, 1.2], 'm-', lw=1.5)
    ax.plot([4.5, 4.5], [1.0, 2.1], 'm-', lw=1.5)
    ax.text(4.1, 1.4, '-5V', fontsize=7, color='purple')

    # Load connections
    for y in [4.2, 3.5, 2.8, 2.1, 1.0]:
        ax.plot([4.5, 4.5], [y, y], 'k-', lw=1)

    # Current annotations
    ax.text(7.8, 4.2, '~50mA', fontsize=7, color=COLORS['gray_dark'])
    ax.text(7.8, 3.5, '~100mA', fontsize=7, color=COLORS['gray_dark'])
    ax.text(7.8, 2.8, '~20mA', fontsize=7, color=COLORS['gray_dark'])
    ax.text(7.8, 2.1, '~30mA', fontsize=7, color=COLORS['gray_dark'])
    ax.text(7.8, 1.0, '~10mA', fontsize=7, color=COLORS['gray_dark'])

    # Title
    ax.text(4.5, 4.8, 'Power Distribution', fontsize=11,
           fontweight='bold', ha='center', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf
