"""
HIRT PDF Generator - Field Operations Diagrams Module

Functions for creating grid layouts, deployment scenarios, and field operation diagrams.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, Ellipse, FancyArrowPatch
)
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
    'ground_tan': '#d4a373',
}


def create_grid_layout(rows=5, cols=5, spacing=2.0, show_zones=True):
    """
    Create a probe grid layout diagram.

    Args:
        rows: Number of rows
        cols: Number of columns
        spacing: Probe spacing in meters
        show_zones: Whether to show zone groupings

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    total_width = (cols - 1) * spacing
    total_height = (rows - 1) * spacing

    ax.set_xlim(-2, total_width + 2)
    ax.set_ylim(-2, total_height + 2)
    ax.set_aspect('equal')

    # Draw probe positions
    probe_num = 1
    for i in range(rows):
        for j in range(cols):
            x = j * spacing
            y = i * spacing

            # Zone coloring
            if show_zones:
                zone = (i // 2) * ((cols + 1) // 2) + (j // 2)
                zone_colors = [COLORS['accent'], COLORS['success'],
                              COLORS['orange'], COLORS['purple'],
                              COLORS['warning'], COLORS['gray_med']]
                color = zone_colors[zone % len(zone_colors)]
            else:
                color = COLORS['secondary']

            # Probe marker
            ax.add_patch(Circle((x, y), 0.3, facecolor=color,
                               edgecolor='black', lw=1, zorder=10))
            ax.text(x, y, str(probe_num), ha='center', va='center',
                   fontsize=7, color='white', fontweight='bold', zorder=11)
            probe_num += 1

    # Grid lines
    for i in range(rows):
        ax.axhline(i * spacing, color=COLORS['gray_light'], linestyle=':', alpha=0.5)
    for j in range(cols):
        ax.axvline(j * spacing, color=COLORS['gray_light'], linestyle=':', alpha=0.5)

    # Zone boxes
    if show_zones:
        zone_num = 1
        for zi in range(0, rows, 2):
            for zj in range(0, cols, 2):
                x0 = zj * spacing - 0.5
                y0 = zi * spacing - 0.5
                w = min(2, cols - zj) * spacing
                h = min(2, rows - zi) * spacing
                ax.add_patch(Rectangle((x0, y0), w, h,
                                       facecolor='none', edgecolor=COLORS['gray_dark'],
                                       linestyle='--', lw=1.5, zorder=5))
                ax.text(x0 + 0.2, y0 + h - 0.3, f'Zone {zone_num}',
                       fontsize=7, color=COLORS['gray_dark'])
                zone_num += 1

    # Dimension annotations
    ax.annotate('', xy=(spacing, -1.2), xytext=(0, -1.2),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(spacing/2, -1.5, f'{spacing}m spacing', ha='center', fontsize=9)

    ax.annotate('', xy=(total_width + 1.2, total_height), xytext=(total_width + 1.2, 0),
               arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(total_width + 1.5, total_height/2, f'{total_height}m', va='center',
           fontsize=9, rotation=90)

    # Legend
    ax.text(total_width/2, total_height + 1.5,
           f'{rows}x{cols} Grid ({rows*cols} probes, {spacing}m spacing)',
           ha='center', fontsize=11, fontweight='bold', color=COLORS['primary'])

    ax.set_xlabel('Distance (m)', fontsize=10)
    ax.set_ylabel('Distance (m)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_measurement_sequence():
    """
    Create measurement sequence timeline diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Timeline base
    ax.plot([1, 9], [2.5, 2.5], 'k-', lw=2)

    # Time markers
    times = [1, 3, 5, 7, 9]
    labels = ['0s', '30s', '60s', '90s', '120s']
    for t, label in zip(times, labels):
        ax.plot([t, t], [2.3, 2.7], 'k-', lw=2)
        ax.text(t, 2.0, label, ha='center', fontsize=8)

    # MIT measurement blocks
    mit_times = [(1.2, 2.5), (3.2, 4.5), (5.2, 6.5), (7.2, 8.5)]
    for start, end in mit_times:
        ax.add_patch(Rectangle((start, 3.0), end-start, 0.6,
                               facecolor=COLORS['success'], edgecolor='black', lw=1))
    ax.text(1.5, 3.8, 'MIT Scan', fontsize=8, color=COLORS['success'])

    # Frequency sweep indicator
    ax.annotate('2→50 kHz', xy=(2.0, 3.3), xytext=(2.0, 4.2),
               fontsize=7, ha='center', color=COLORS['success'],
               arrowprops=dict(arrowstyle='->', color=COLORS['success']))

    # ERT measurement blocks
    ert_times = [(2.5, 3.0), (4.5, 5.0), (6.5, 7.0)]
    for start, end in ert_times:
        ax.add_patch(Rectangle((start, 1.5), end-start, 0.6,
                               facecolor=COLORS['orange'], edgecolor='black', lw=1))
    ax.text(2.5, 0.9, 'ERT Measure', fontsize=8, color=COLORS['orange'])

    # Multiplexer switching
    for t in [3.0, 5.0, 7.0]:
        ax.annotate('', xy=(t, 2.7), xytext=(t, 3.0),
                   arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.5))
        ax.annotate('', xy=(t, 2.3), xytext=(t, 2.1),
                   arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.5))
    ax.text(5.0, 1.2, 'MUX Switch', fontsize=7, ha='center', color=COLORS['purple'])

    # Probe pairs indicator
    probe_labels = ['P1-P2', 'P2-P3', 'P3-P4', 'P4-P5']
    for i, (start, end) in enumerate(mit_times):
        if i < len(probe_labels):
            ax.text((start+end)/2, 3.75, probe_labels[i], ha='center',
                   fontsize=7, color='white', fontweight='bold')

    # Title
    ax.text(5, 4.7, 'Measurement Sequence Timeline (One Cycle)',
           ha='center', fontsize=11, fontweight='bold', color=COLORS['primary'])

    # Legend
    legend_y = 0.3
    ax.add_patch(Rectangle((1.5, legend_y), 0.4, 0.3, facecolor=COLORS['success']))
    ax.text(2.0, legend_y + 0.15, 'MIT Scan', fontsize=8, va='center')
    ax.add_patch(Rectangle((4.0, legend_y), 0.4, 0.3, facecolor=COLORS['orange']))
    ax.text(4.5, legend_y + 0.15, 'ERT Measure', fontsize=8, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_ert_injection_patterns():
    """
    Create ERT injection pattern options diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    patterns = [
        ('Wenner-Alpha', [(0, 3), (1, 2)], 'ABMN spacing = 1:1:1'),
        ('Dipole-Dipole', [(0, 1), (2, 3)], 'AB and MN separated'),
        ('Pole-Dipole', [(0, None), (2, 3)], 'Remote current electrode'),
    ]

    for ax, (name, pairs, desc) in zip(axes, patterns):
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')

        # Ground line
        ax.axhline(1, color='#654321', lw=2)
        ax.fill_between([-1, 5], [1, 1], [-1, -1], color=COLORS['ground_tan'], alpha=0.3)

        # Electrode positions
        positions = [0.5, 1.5, 2.5, 3.5]
        labels = ['A', 'B', 'M', 'N']

        for i, (pos, label) in enumerate(zip(positions, labels)):
            is_current = label in ['A', 'B']
            color = COLORS['warning'] if is_current else COLORS['accent']

            # Skip if remote electrode
            if pairs[0][1] is None and label == 'B':
                ax.text(pos, 1.4, '(remote)', fontsize=7, ha='center',
                       color=COLORS['gray_med'])
                continue

            ax.add_patch(Circle((pos, 1), 0.2, facecolor=color,
                               edgecolor='black', lw=1))
            ax.text(pos, 1, label, ha='center', va='center',
                   fontsize=9, color='white', fontweight='bold')

        # Connection lines
        for pair, color in [(pairs[0], COLORS['warning']), (pairs[1], COLORS['accent'])]:
            if pair[0] is not None and pair[1] is not None:
                x1, x2 = positions[pair[0]], positions[pair[1]]
                ax.plot([x1, x2], [2.0, 2.0], color=color, lw=2)
                ax.plot([x1, x1], [1.2, 2.0], color=color, lw=1)
                ax.plot([x2, x2], [1.2, 2.0], color=color, lw=1)

        # Current flow indication
        ax.annotate('', xy=(positions[pairs[0][0]] + 0.5, 0.3),
                   xytext=(positions[pairs[0][0]], 0.3),
                   arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=1.5))

        ax.set_title(name, fontsize=10, fontweight='bold', color=COLORS['primary'])
        ax.text(2, -0.5, desc, ha='center', fontsize=8, color=COLORS['gray_dark'])
        ax.axis('off')

    # Legend
    fig.text(0.5, 0.02, 'Red = Current injection (A,B)  |  Blue = Voltage measurement (M,N)',
            ha='center', fontsize=9, color=COLORS['gray_dark'])

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_deployment_scenarios():
    """
    Create multi-panel deployment scenarios diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    scenarios = [
        ('(a) Square Array', 'square'),
        ('(b) Perimeter Only', 'perimeter'),
        ('(c) Dense Center', 'dense_center'),
    ]

    for ax, (title, pattern) in zip(axes, scenarios):
        ax.set_xlim(-1, 9)
        ax.set_ylim(-1, 9)
        ax.set_aspect('equal')

        # Grid background
        ax.fill_between([-1, 9], [-1, -1], [9, 9], color=COLORS['ground_tan'], alpha=0.2)

        # Probe positions based on pattern
        if pattern == 'square':
            positions = [(i*2, j*2) for i in range(5) for j in range(5)]
        elif pattern == 'perimeter':
            positions = (
                [(0, j*2) for j in range(5)] +  # Left
                [(8, j*2) for j in range(5)] +  # Right
                [(i*2, 0) for i in range(1, 4)] +  # Bottom
                [(i*2, 8) for i in range(1, 4)]    # Top
            )
        else:  # dense_center
            # Outer ring
            positions = [(0, j*2) for j in range(5)] + [(8, j*2) for j in range(5)]
            positions += [(i*2, 0) for i in range(1, 4)] + [(i*2, 8) for i in range(1, 4)]
            # Inner dense
            positions += [(3, 3), (5, 3), (3, 5), (5, 5), (4, 4)]

        # Draw probes
        for i, (x, y) in enumerate(positions):
            ax.add_patch(Circle((x, y), 0.35, facecolor=COLORS['secondary'],
                               edgecolor='black', lw=1))

        # Target zone for perimeter and dense
        if pattern in ['perimeter', 'dense_center']:
            ax.add_patch(Ellipse((4, 4), 4, 4, facecolor=COLORS['warning'],
                                alpha=0.2, edgecolor=COLORS['warning'], linestyle='--'))
            ax.text(4, 4, 'Target\nZone', ha='center', va='center',
                   fontsize=8, color=COLORS['warning'])

        # Scale bar
        ax.plot([0, 2], [-0.5, -0.5], 'k-', lw=2)
        ax.text(1, -0.8, '2m', ha='center', fontsize=8)

        ax.set_title(title, fontsize=10, fontweight='bold', color=COLORS['primary'])
        ax.set_xlabel(f'{len(positions)} probes', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_insertion_procedure():
    """
    Create probe insertion procedure diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 5))

    steps = [
        ('1. Pilot Hole', 'auger'),
        ('2. First Segment', 'insert1'),
        ('3. Connect Segment', 'connect'),
        ('4. Full Insertion', 'complete'),
    ]

    for ax, (title, step) in zip(axes, steps):
        ax.set_xlim(-2, 2)
        ax.set_ylim(-4, 2)
        ax.set_aspect('equal')

        # Ground surface
        ax.axhline(0, color='#654321', lw=2)
        ax.fill_between([-2, 2], [0, 0], [-4, -4], color=COLORS['ground_tan'], alpha=0.3)

        if step == 'auger':
            # Hand auger
            ax.add_patch(Rectangle((-0.15, -1.5), 0.3, 2.5,
                                   facecolor=COLORS['gray_dark'], edgecolor='black', lw=1))
            ax.text(0, 1.3, 'Hand\nAuger', ha='center', fontsize=8)
            # Hole
            ax.add_patch(Rectangle((-0.2, -1.5), 0.4, 1.5,
                                   facecolor='white', edgecolor=COLORS['gray_med']))

        elif step == 'insert1':
            # First segment being inserted
            ax.add_patch(Rectangle((-0.15, -1.5), 0.3, 2.0,
                                   facecolor=COLORS['secondary'], edgecolor='black', lw=1))
            # Arrow showing insertion
            ax.annotate('', xy=(0, -1.5), xytext=(0, 0.5),
                       arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))

        elif step == 'connect':
            # Bottom segment in ground
            ax.add_patch(Rectangle((-0.15, -2.5), 0.3, 1.5,
                                   facecolor=COLORS['secondary'], edgecolor='black', lw=1))
            # Top segment being connected
            ax.add_patch(Rectangle((-0.15, -0.5), 0.3, 1.5,
                                   facecolor=COLORS['secondary'], edgecolor='black', lw=1))
            # Thread connection point
            ax.add_patch(Circle((0, -1.0), 0.2, facecolor=COLORS['orange'],
                               edgecolor='black', lw=1))
            ax.annotate('Thread\nJoint', xy=(0, -1.0), xytext=(1.0, -0.5),
                       fontsize=7, arrowprops=dict(arrowstyle='->', color=COLORS['orange']))

        elif step == 'complete':
            # Full probe in ground
            ax.add_patch(Rectangle((-0.15, -3.5), 0.3, 3.7,
                                   facecolor=COLORS['secondary'], edgecolor='black', lw=1))
            # Junction box
            ax.add_patch(FancyBboxPatch((-0.3, 0.2), 0.6, 0.4, boxstyle="round,pad=0.02",
                                        facecolor=COLORS['gray_dark'], edgecolor='black'))
            # Cable
            ax.plot([0.3, 1.5], [0.4, 0.4], color=COLORS['gray_dark'], lw=2)
            ax.text(1.6, 0.4, 'To\nHub', fontsize=7, va='center')

        ax.set_title(title, fontsize=10, fontweight='bold', color=COLORS['primary'])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


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


def create_cable_color_coding():
    """
    Create cable color coding reference diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Color coding table
    codes = [
        ('TX Signal', '#38a169', 'Green'),
        ('RX Signal', '#3182ce', 'Blue'),
        ('ERT Current', '#c53030', 'Red'),
        ('ERT Voltage', '#ed8936', 'Orange'),
        ('Ground/Shield', '#000000', 'Black'),
        ('Power (+12V)', '#805ad5', 'Purple'),
        ('Power (GND)', '#4a5568', 'Gray'),
    ]

    y_start = 5.0
    for i, (function, color, name) in enumerate(codes):
        y = y_start - i * 0.7

        # Color swatch
        ax.add_patch(Rectangle((0.5, y - 0.2), 1.0, 0.4,
                               facecolor=color, edgecolor='black', lw=1))

        # Color name
        ax.text(1.8, y, name, fontsize=10, va='center', fontweight='bold')

        # Function
        ax.text(4.0, y, function, fontsize=10, va='center')

    # Header
    ax.text(1.0, 5.7, 'Color', fontsize=10, fontweight='bold', ha='center')
    ax.text(1.8, 5.7, 'Name', fontsize=10, fontweight='bold')
    ax.text(4.0, 5.7, 'Function', fontsize=10, fontweight='bold')

    # Separator line
    ax.plot([0.3, 7.5], [5.5, 5.5], 'k-', lw=1)

    # Title
    ax.text(4, 0.3, 'Cable Color Coding Reference', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf
