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
from matplotlib.path import Path
import matplotlib.patheffects as path_effects
import numpy as np
from io import BytesIO

# Import colorblind-safe palette from parent module
try:
    from . import WONG_PALETTE
except ImportError:
    # Fallback if imported standalone
    WONG_PALETTE = {
        'orange': '#E69F00',
        'sky_blue': '#56B4E9',
        'bluish_green': '#009E73',
        'yellow': '#F0E442',
        'blue': '#0072B2',
        'vermillion': '#D55E00',
        'reddish_purple': '#CC79A7',
        'black': '#000000',
    }

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
    'light_blue': '#ebf8ff',
    'light_green': '#c6f6d5',
    'light_orange': '#feebc8',
}

# Connector type definitions for cable routing
CONNECTOR_TYPES = {
    'DB25': {'shape': 'D', 'color': COLORS['gray_dark'], 'label': 'DB25'},
    'DB9': {'shape': 'D', 'color': COLORS['gray_dark'], 'label': 'DB9'},
    'RJ45': {'shape': 'rect', 'color': COLORS['accent'], 'label': 'RJ45'},
    'BNC': {'shape': 'circle', 'color': COLORS['gray_med'], 'label': 'BNC'},
    'screw': {'shape': 'circle', 'color': COLORS['success'], 'label': 'Screw'},
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


def _draw_timing_annotation(ax, x, y, duration_text, color='black'):
    """
    Draw a timing annotation with duration label.

    Args:
        ax: matplotlib Axes object
        x, y: Position for the annotation
        duration_text: Text showing duration (e.g., '~25s')
        color: Text color
    """
    # Small clock icon (circle with hands)
    clock_radius = 0.12
    ax.add_patch(Circle((x - 0.2, y), clock_radius, facecolor='white',
                        edgecolor=color, lw=0.8))
    # Clock hands
    ax.plot([x - 0.2, x - 0.2], [y, y + clock_radius * 0.7], color=color, lw=0.8)
    ax.plot([x - 0.2, x - 0.2 + clock_radius * 0.5], [y, y], color=color, lw=0.8)
    # Duration text
    ax.text(x + 0.05, y, duration_text, ha='left', va='center', fontsize=6, color=color)


def _draw_duration_bar(ax, x_start, x_end, y, duration_text, color):
    """
    Draw a small duration bar under a measurement block.

    Args:
        ax: matplotlib Axes object
        x_start, x_end: Horizontal extent
        y: Vertical position
        duration_text: Duration label
        color: Bar color
    """
    bar_height = 0.08
    ax.add_patch(Rectangle((x_start, y - bar_height/2), x_end - x_start, bar_height,
                           facecolor=color, edgecolor='none', alpha=0.6))
    ax.text((x_start + x_end) / 2, y - 0.2, duration_text, ha='center',
           va='top', fontsize=5, color=color, style='italic')


def create_measurement_sequence():
    """
    Create measurement sequence timeline diagram with timing annotations.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 5.2)
    ax.axis('off')

    # Timeline base
    ax.plot([1, 9], [2.5, 2.5], 'k-', lw=2)

    # Time markers with enhanced styling
    times = [1, 3, 5, 7, 9]
    labels = ['0s', '30s', '60s', '90s', '120s']
    for t, label in zip(times, labels):
        ax.plot([t, t], [2.3, 2.7], 'k-', lw=2)
        ax.text(t, 2.0, label, ha='center', fontsize=8, fontweight='bold')
        # Add subtle tick marks between major markers
        if t < 9:
            for minor in [0.5, 1.0, 1.5]:
                ax.plot([t + minor, t + minor], [2.4, 2.6], 'k-', lw=0.5, alpha=0.5)

    # MIT measurement blocks with timing annotations
    mit_times = [(1.2, 2.5), (3.2, 4.5), (5.2, 6.5), (7.2, 8.5)]
    mit_durations = ['~25s', '~25s', '~25s', '~25s']
    for (start, end), duration in zip(mit_times, mit_durations):
        ax.add_patch(Rectangle((start, 3.0), end-start, 0.6,
                               facecolor=COLORS['success'], edgecolor='black', lw=1))
        # Add duration bar below the block
        _draw_duration_bar(ax, start, end, 2.85, duration, COLORS['success'])

    ax.text(1.5, 3.85, 'MIT Scan', fontsize=8, color=COLORS['success'], fontweight='bold')

    # Frequency sweep indicator with timing
    ax.annotate('2→50 kHz sweep', xy=(2.0, 3.3), xytext=(2.0, 4.3),
               fontsize=7, ha='center', color=COLORS['success'],
               arrowprops=dict(arrowstyle='->', color=COLORS['success']))
    _draw_timing_annotation(ax, 2.5, 4.5, '~20s/sweep', COLORS['success'])

    # ERT measurement blocks with timing annotations
    ert_times = [(2.5, 3.0), (4.5, 5.0), (6.5, 7.0)]
    ert_durations = ['~5s', '~5s', '~5s']
    for (start, end), duration in zip(ert_times, ert_durations):
        ax.add_patch(Rectangle((start, 1.5), end-start, 0.6,
                               facecolor=COLORS['orange'], edgecolor='black', lw=1))
        _draw_duration_bar(ax, start, end, 1.35, duration, COLORS['orange'])

    ax.text(2.5, 0.75, 'ERT Measure', fontsize=8, color=COLORS['orange'], fontweight='bold')

    # Multiplexer switching with timing
    mux_switch_time = '~0.5s'
    for i, t in enumerate([3.0, 5.0, 7.0]):
        ax.annotate('', xy=(t, 2.7), xytext=(t, 3.0),
                   arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.5))
        ax.annotate('', xy=(t, 2.3), xytext=(t, 2.1),
                   arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.5))
        if i == 1:  # Only label middle switch
            ax.text(t, 2.5, mux_switch_time, ha='left', va='center',
                   fontsize=5, color=COLORS['purple'], style='italic')

    ax.text(5.0, 1.0, 'MUX Switch', fontsize=7, ha='center', color=COLORS['purple'])

    # Probe pairs indicator
    probe_labels = ['P1-P2', 'P2-P3', 'P3-P4', 'P4-P5']
    for i, (start, end) in enumerate(mit_times):
        if i < len(probe_labels):
            ax.text((start+end)/2, 3.3, probe_labels[i], ha='center',
                   fontsize=7, color='white', fontweight='bold')

    # Title
    ax.text(5, 5.0, 'Measurement Sequence Timeline (One Cycle)',
           ha='center', fontsize=11, fontweight='bold', color=COLORS['primary'])

    # Total cycle time annotation
    ax.add_patch(FancyBboxPatch((7.5, 4.2), 2.2, 0.7, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_blue'], edgecolor=COLORS['accent'], lw=1))
    ax.text(8.6, 4.55, 'Total: ~120s/cycle', ha='center', va='center',
           fontsize=8, fontweight='bold', color=COLORS['primary'])

    # Legend with timing info
    legend_y = 0.0
    ax.add_patch(Rectangle((1.2, legend_y), 0.4, 0.3, facecolor=COLORS['success']))
    ax.text(1.7, legend_y + 0.15, 'MIT Scan (~25s)', fontsize=7, va='center')
    ax.add_patch(Rectangle((4.0, legend_y), 0.4, 0.3, facecolor=COLORS['orange']))
    ax.text(4.5, legend_y + 0.15, 'ERT Measure (~5s)', fontsize=7, va='center')
    ax.add_patch(Rectangle((6.8, legend_y), 0.4, 0.3, facecolor=COLORS['purple']))
    ax.text(7.3, legend_y + 0.15, 'MUX Switch (~0.5s)', fontsize=7, va='center')

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
    Create multi-panel deployment scenarios diagram with side-by-side comparison layout.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5))

    scenarios = [
        ('(a) Square Array', 'square', {
            'coverage': 'Uniform',
            'resolution': 'Medium',
            'setup_time': '~2h',
            'use_case': 'General survey',
        }),
        ('(b) Perimeter Only', 'perimeter', {
            'coverage': 'Edge-focused',
            'resolution': 'Low center',
            'setup_time': '~1h',
            'use_case': 'Known target area',
        }),
        ('(c) Dense Center', 'dense_center', {
            'coverage': 'Target-focused',
            'resolution': 'High center',
            'setup_time': '~1.5h',
            'use_case': 'Suspected UXO location',
        }),
    ]

    # Use colorblind-friendly colors for the scenarios
    scenario_colors = [WONG_PALETTE['blue'], WONG_PALETTE['orange'], WONG_PALETTE['bluish_green']]

    for ax, (title, pattern, specs), sc_color in zip(axes, scenarios, scenario_colors):
        ax.set_xlim(-1.5, 10)
        ax.set_ylim(-2.5, 10)
        ax.set_aspect('equal')

        # Grid background
        ax.fill_between([-0.5, 8.5], [-0.5, -0.5], [8.5, 8.5], color=COLORS['ground_tan'], alpha=0.2)

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

        # Draw probes with scenario color
        for i, (x, y) in enumerate(positions):
            ax.add_patch(Circle((x, y), 0.35, facecolor=sc_color,
                               edgecolor='black', lw=1, alpha=0.8))

        # Target zone for perimeter and dense
        if pattern in ['perimeter', 'dense_center']:
            ax.add_patch(Ellipse((4, 4), 4, 4, facecolor=COLORS['warning'],
                                alpha=0.15, edgecolor=COLORS['warning'], linestyle='--', lw=1.5))
            ax.text(4, 4, 'Target\nZone', ha='center', va='center',
                   fontsize=7, color=COLORS['warning'], fontweight='bold')

        # Scale bar
        ax.plot([0, 2], [-0.8, -0.8], 'k-', lw=2)
        ax.text(1, -1.1, '2m', ha='center', fontsize=7)

        # Title with color indicator
        ax.add_patch(Rectangle((-1.2, 9.2), 0.4, 0.4, facecolor=sc_color, edgecolor='black', lw=0.5))
        ax.text(-0.6, 9.4, title, fontsize=10, fontweight='bold', color=COLORS['primary'], va='center')

        # Specifications box (comparison info)
        spec_y = -1.8
        spec_box_height = 0.9
        ax.add_patch(FancyBboxPatch((-1, spec_y - spec_box_height/2), 10, spec_box_height,
                                    boxstyle="round,pad=0.02", facecolor='white',
                                    edgecolor=COLORS['gray_light'], lw=1))

        spec_text = f"Probes: {len(positions)}  |  Coverage: {specs['coverage']}  |  Setup: {specs['setup_time']}"
        ax.text(4, spec_y, spec_text, ha='center', va='center', fontsize=6)
        ax.text(4, spec_y - 0.35, f"Best for: {specs['use_case']}", ha='center', va='center',
               fontsize=6, style='italic', color=COLORS['gray_dark'])

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Add comparison summary at bottom
    fig.text(0.5, 0.02, 'Comparison: Square = baseline coverage | Perimeter = reduced setup | Dense Center = maximum resolution at suspected location',
            ha='center', fontsize=8, style='italic', color=COLORS['gray_dark'])

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_deployment_comparison():
    """
    Create a side-by-side deployment scenario comparison table/diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5.5, 5.7, 'Deployment Scenario Comparison', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Column headers
    headers = ['Scenario', 'Layout', 'Probes', 'Coverage', 'Resolution', 'Setup Time', 'Use Case']
    col_x = [0.5, 2.0, 4.0, 5.2, 6.4, 7.6, 9.0]

    for hx, header in zip(col_x, headers):
        ax.text(hx, 5.2, header, ha='center', va='center', fontsize=8,
               fontweight='bold', color=COLORS['primary'])

    # Header underline
    ax.plot([0.2, 10.8], [5.0, 5.0], color=COLORS['gray_dark'], lw=1.5)

    # Scenario data
    scenarios = [
        ('Square\nArray', 'square', '25', 'Uniform', 'Medium', '~2h', 'General'),
        ('Perimeter\nOnly', 'perimeter', '16', 'Edge', 'Low center', '~1h', 'Known target'),
        ('Dense\nCenter', 'dense_center', '21', 'Target', 'High center', '~1.5h', 'Suspected UXO'),
    ]

    row_colors = [WONG_PALETTE['blue'], WONG_PALETTE['orange'], WONG_PALETTE['bluish_green']]

    for row_idx, ((name, pattern, probes, coverage, resolution, time, use_case), row_color) in enumerate(zip(scenarios, row_colors)):
        row_y = 4.2 - row_idx * 1.4

        # Row background
        ax.add_patch(Rectangle((0.1, row_y - 0.55), 10.8, 1.1,
                               facecolor=row_color, edgecolor='none', alpha=0.1))

        # Name column
        ax.text(0.5, row_y, name, ha='center', va='center', fontsize=7, fontweight='bold')

        # Mini layout diagram
        layout_x, layout_y = 2.0, row_y
        layout_scale = 0.08
        if pattern == 'square':
            positions = [(i, j) for i in range(5) for j in range(5)]
        elif pattern == 'perimeter':
            positions = ([(0, j) for j in range(5)] + [(4, j) for j in range(5)] +
                        [(i, 0) for i in range(1, 4)] + [(i, 4) for i in range(1, 4)])
        else:
            positions = ([(0, j) for j in range(5)] + [(4, j) for j in range(5)] +
                        [(i, 0) for i in range(1, 4)] + [(i, 4) for i in range(1, 4)] +
                        [(1.5, 1.5), (2.5, 1.5), (1.5, 2.5), (2.5, 2.5), (2, 2)])

        for px, py in positions:
            ax.add_patch(Circle((layout_x + (px - 2) * layout_scale * 5,
                                layout_y + (py - 2) * layout_scale * 5),
                               layout_scale * 1.5, facecolor=row_color,
                               edgecolor='none', alpha=0.8))

        # Data columns
        data_values = [probes, coverage, resolution, time, use_case]
        for val, cx in zip(data_values, col_x[2:]):
            ax.text(cx, row_y, val, ha='center', va='center', fontsize=7)

    # Bottom recommendation
    ax.add_patch(FancyBboxPatch((0.5, 0.2), 10, 0.6, boxstyle="round,pad=0.03",
                                facecolor=COLORS['light_blue'], edgecolor=COLORS['accent'], lw=1))
    ax.text(5.5, 0.5, 'Recommendation: Start with Square Array for initial surveys, switch to Dense Center for suspected anomalies',
           ha='center', va='center', fontsize=7, style='italic')

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


def _draw_connector_symbol(ax, x, y, connector_type, size=0.15):
    """
    Draw a connector symbol at the specified position.

    Args:
        ax: matplotlib Axes object
        x, y: Center position
        connector_type: One of 'DB25', 'DB9', 'RJ45', 'BNC', 'screw'
        size: Symbol size

    Returns:
        The patch object
    """
    conn_info = CONNECTOR_TYPES.get(connector_type, CONNECTOR_TYPES['screw'])

    if conn_info['shape'] == 'D':
        # D-sub connector (trapezoid approximation)
        verts = [
            (x - size, y - size * 0.6),
            (x - size * 0.7, y + size * 0.6),
            (x + size * 0.7, y + size * 0.6),
            (x + size, y - size * 0.6),
            (x - size, y - size * 0.6),
        ]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        path = Path(verts, codes)
        from matplotlib.patches import PathPatch
        patch = PathPatch(path, facecolor=conn_info['color'], edgecolor='black', lw=0.8)
        ax.add_patch(patch)
    elif conn_info['shape'] == 'rect':
        patch = Rectangle((x - size, y - size * 0.5), size * 2, size,
                          facecolor=conn_info['color'], edgecolor='black', lw=0.8)
        ax.add_patch(patch)
    else:  # circle
        patch = Circle((x, y), size * 0.6, facecolor=conn_info['color'],
                       edgecolor='black', lw=0.8)
        ax.add_patch(patch)

    return patch


def _draw_cable_route(ax, points, color, lw=2, style='-', label=None):
    """
    Draw a cable route through multiple points with optional bends.

    Args:
        ax: matplotlib Axes object
        points: List of (x, y) tuples for cable path
        color: Cable color
        lw: Line width
        style: Line style ('-', '--', etc.)
        label: Optional label at midpoint
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, lw=lw, linestyle=style, solid_capstyle='round')

    # Add label at midpoint if specified
    if label and len(points) >= 2:
        mid_idx = len(points) // 2
        mid_x = (points[mid_idx][0] + points[mid_idx - 1][0]) / 2
        mid_y = (points[mid_idx][1] + points[mid_idx - 1][1]) / 2
        ax.text(mid_x, mid_y + 0.15, label, ha='center', va='bottom',
               fontsize=5, color=color, style='italic')


def create_zone_architecture():
    """
    Create zone wiring architecture diagram with cable routing and connector types.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    # Central hub with more detail
    ax.add_patch(FancyBboxPatch((0.2, 2.2), 2.2, 2.2, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_blue'], edgecolor=COLORS['secondary'],
                                linewidth=2))
    ax.text(1.3, 3.3, 'Central\nHub', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.text(1.3, 2.6, '(MCU + MUX)', ha='center', va='center',
           fontsize=7, color=COLORS['gray_med'])

    # Power input to central hub
    ax.plot([0.2, -0.3], [4.0, 4.0], color=COLORS['warning'], lw=3)
    ax.text(-0.4, 4.0, '+12V', ha='right', va='center', fontsize=7, color=COLORS['warning'])
    _draw_connector_symbol(ax, 0.2, 4.0, 'screw', size=0.12)

    # USB output
    ax.plot([0.2, -0.3], [2.6, 2.6], color=COLORS['accent'], lw=2)
    ax.text(-0.4, 2.6, 'USB', ha='right', va='center', fontsize=7, color=COLORS['accent'])
    _draw_connector_symbol(ax, 0.2, 2.6, 'RJ45', size=0.1)

    # Trunk cables with detailed routing and connectors
    trunk_y_positions = [5.0, 3.3, 1.6]
    trunk_colors = [WONG_PALETTE['blue'], WONG_PALETTE['bluish_green'], WONG_PALETTE['orange']]

    for i, (trunk_y, trunk_color) in enumerate(zip(trunk_y_positions, trunk_colors)):
        # Cable from hub to zone
        cable_points = [
            (2.4, 3.3),  # Exit from hub
            (2.8, 3.3),  # First bend point
            (2.8, trunk_y),  # Vertical route
            (4.2, trunk_y),  # To zone hub
        ]
        _draw_cable_route(ax, cable_points, trunk_color, lw=3.5, label=f'Trunk {i+1}')

        # Draw DB25 connector at hub end
        _draw_connector_symbol(ax, 2.4, 3.3 + (i - 1) * 0.25, 'DB25', size=0.12)

    ax.text(2.8, 5.8, 'Trunk Cables', ha='center', fontsize=8,
           fontweight='bold', color=COLORS['gray_dark'])
    ax.text(2.8, 5.5, '(DB25 shielded)', ha='center', fontsize=6,
           color=COLORS['gray_med'])

    # Zone hubs with connector details
    zone_positions = [(4.0, 4.5), (4.0, 2.8), (4.0, 1.1)]
    zone_hub_colors = [COLORS['light_green'], COLORS['light_orange'], '#e9d8fd']

    for i, ((x, y), zone_color) in enumerate(zip(zone_positions, zone_hub_colors)):
        # Zone hub box
        ax.add_patch(FancyBboxPatch((x, y), 1.8, 1.0, boxstyle="round,pad=0.03",
                                    facecolor=zone_color, edgecolor=COLORS['gray_dark'],
                                    linewidth=1.5))
        ax.text(x + 0.9, y + 0.65, f'Zone {i+1}', ha='center', va='center',
               fontsize=9, fontweight='bold')
        ax.text(x + 0.9, y + 0.35, 'Junction Box', ha='center', va='center',
               fontsize=6, color=COLORS['gray_med'])

        # DB25 input connector
        _draw_connector_symbol(ax, x, y + 0.5, 'DB25', size=0.12)

        # Probe cables with proper routing
        probe_colors = [WONG_PALETTE['vermillion'], WONG_PALETTE['sky_blue'],
                       WONG_PALETTE['yellow'], WONG_PALETTE['reddish_purple']]

        for pi in range(4):
            px = 8.0 + (pi % 2) * 1.2
            py = y + 0.2 + (pi // 2) * 0.5

            # Route cable with bends for clarity
            cable_points = [
                (x + 1.8, y + 0.5),  # Exit zone hub
                (x + 2.2 + pi * 0.15, y + 0.5),  # Spread out
                (x + 2.2 + pi * 0.15, py),  # Vertical
                (px - 0.3, py),  # To probe
            ]
            _draw_cable_route(ax, cable_points, probe_colors[pi], lw=1.2)

            # Screw terminal at zone hub
            _draw_connector_symbol(ax, x + 1.8, y + 0.3 + pi * 0.15, 'screw', size=0.06)

            # Probe marker
            ax.add_patch(Circle((px, py), 0.22, facecolor=COLORS['secondary'],
                               edgecolor='black', lw=0.8, zorder=10))
            ax.text(px, py, f'{i*4 + pi + 1}', ha='center', va='center',
                   fontsize=6, color='white', fontweight='bold', zorder=11)

    # Probe label
    ax.text(8.6, 6.2, 'Probes', ha='center', fontsize=9, fontweight='bold')
    ax.text(8.6, 5.9, '(12 total, 4/zone)', ha='center', fontsize=6, color=COLORS['gray_med'])

    # Cable legend
    legend_y = 0.0
    legend_items = [
        ('Trunk (DB25)', COLORS['gray_dark'], 3.5),
        ('Probe (Screw)', COLORS['gray_light'], 1.5),
        ('Power', COLORS['warning'], 2.5),
        ('Data', COLORS['accent'], 1.5),
    ]
    legend_x = 0.5
    for label, color, lw in legend_items:
        ax.plot([legend_x, legend_x + 0.6], [legend_y, legend_y], color=color, lw=lw)
        ax.text(legend_x + 0.75, legend_y, label, fontsize=6, va='center')
        legend_x += 2.3

    # Annotations with connector callouts
    ax.annotate('DB25 (25-pin\nshielded)', xy=(2.4, 3.1), xytext=(1.0, 1.0),
               fontsize=6, ha='center',
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_med'], lw=0.8))

    ax.annotate('Screw\nTerminals', xy=(5.8, 3.0), xytext=(6.5, 1.5),
               fontsize=6, ha='center',
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_med'], lw=0.8))

    # Title
    ax.text(5, 6.6, 'Zone Wiring Architecture with Cable Routing', ha='center',
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
