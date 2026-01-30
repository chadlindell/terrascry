"""
HIRT PDF Generator - Flowcharts Module

Functions for creating decision trees, workflows, timelines, and process diagrams.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch,
    Ellipse, PathPatch
)
from matplotlib.path import Path
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
    'light_green': '#c6f6d5',
    'light_blue': '#ebf8ff',
    'light_orange': '#feebc8',
    'light_red': '#fed7d7',
    'light_purple': '#e9d8fd',
}


def draw_process_box(ax, x, y, width, height, text, color='white'):
    """Draw a process box (rectangle with rounded corners)."""
    patch = FancyBboxPatch((x - width/2, y - height/2), width, height,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(patch)
    ax.text(x, y, text, ha='center', va='center', fontsize=8,
           wrap=True)
    return patch


def draw_decision_diamond(ax, x, y, size, text, color='white'):
    """Draw a decision diamond."""
    half = size / 2
    points = [(x, y + half), (x + half, y), (x, y - half), (x - half, y)]
    patch = Polygon(points, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(patch)
    ax.text(x, y, text, ha='center', va='center', fontsize=7)
    return patch


def draw_terminal(ax, x, y, width, height, text, color='white'):
    """Draw a terminal/start/end box (pill shape)."""
    patch = FancyBboxPatch((x - width/2, y - height/2), width, height,
                           boxstyle="round,pad=0.3",
                           facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(patch)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    return patch


def draw_flow_arrow(ax, start, end, label=None, color='black'):
    """Draw a flow arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
               arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x + 0.1, mid_y, label, fontsize=7, color=color)


def create_soil_type_decision_tree():
    """
    Create soil type insertion decision tree.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Start
    draw_terminal(ax, 5, 7.5, 2, 0.5, 'START', COLORS['light_green'])

    # First decision: Soil hardness
    draw_decision_diamond(ax, 5, 6, 1.5, 'Soil\nHard?', COLORS['light_blue'])
    draw_flow_arrow(ax, (5, 7.25), (5, 6.75))

    # Hard soil path
    draw_flow_arrow(ax, (5.75, 6), (7.5, 6), 'Yes')
    draw_process_box(ax, 8.5, 6, 1.8, 0.8, 'Use hand\nauger\n(18mm)', COLORS['light_orange'])

    # Soft soil path
    draw_flow_arrow(ax, (5, 5.25), (5, 4.5), 'No')
    draw_decision_diamond(ax, 5, 4, 1.5, 'Water\nTable?', COLORS['light_blue'])

    # High water table
    draw_flow_arrow(ax, (5.75, 4), (7.5, 4), 'High')
    draw_process_box(ax, 8.5, 4, 1.8, 0.8, 'Direct push\n(wet method)', COLORS['light_orange'])

    # Low water table
    draw_flow_arrow(ax, (5, 3.25), (5, 2.5), 'Low')
    draw_decision_diamond(ax, 5, 2, 1.5, 'Stones/\nRoots?', COLORS['light_blue'])

    # With obstructions
    draw_flow_arrow(ax, (5.75, 2), (7.5, 2), 'Yes')
    draw_process_box(ax, 8.5, 2, 1.8, 0.8, 'Careful\nauger +\nprobe', COLORS['light_orange'])

    # Clear path
    draw_flow_arrow(ax, (5, 1.25), (5, 0.5))
    draw_process_box(ax, 5, 0.5, 2.0, 0.6, 'Direct push\nonly', COLORS['light_green'])

    # Convergence arrows to success
    for y in [6, 4, 2]:
        draw_flow_arrow(ax, (8.5, y - 0.4), (8.5, 0.7), '')

    # Success terminal
    draw_terminal(ax, 8.5, 0.5, 1.8, 0.5, 'INSERT', COLORS['light_green'])

    # Title
    ax.text(5, 7.9, 'Soil Insertion Decision Tree', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    # Notes
    ax.text(1.5, 1.5, 'NOTES:\n\u2022 Never hammer probes\n\u2022 Max force: hand pressure\n\u2022 If blocked, relocate',
           fontsize=7, va='top', bbox=dict(boxstyle='round', facecolor='white',
                                          edgecolor=COLORS['gray_med']))

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_calibration_workflow():
    """
    Create calibration procedure workflow.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    # Workflow steps
    steps = [
        (2, 6, 'Air\nCalibration', COLORS['light_blue']),
        (5, 6, 'Known\nResistor\nTest', COLORS['light_blue']),
        (8, 6, 'Verify\nReadings', COLORS['light_green']),
        (2, 4, 'Water\nBucket\nTest', COLORS['light_blue']),
        (5, 4, 'Known\nConductivity\nSolution', COLORS['light_blue']),
        (8, 4, 'Record\nBaseline', COLORS['light_green']),
        (2, 2, 'Metal\nTarget\nTest', COLORS['light_orange']),
        (5, 2, 'Distance\nSweep', COLORS['light_orange']),
        (8, 2, 'Build\nLookup\nTable', COLORS['light_green']),
    ]

    for x, y, text, color in steps:
        draw_process_box(ax, x, y, 2.0, 1.0, text, color)

    # Horizontal arrows
    for y in [6, 4, 2]:
        draw_flow_arrow(ax, (3, y), (4, y))
        draw_flow_arrow(ax, (6, y), (7, y))

    # Vertical arrows
    draw_flow_arrow(ax, (8, 5.5), (8, 4.5))
    draw_flow_arrow(ax, (8, 3.5), (8, 2.5))

    # Phase labels
    ax.text(0.5, 6, 'MIT\nCalibration', fontsize=8, va='center',
           fontweight='bold', color=COLORS['accent'])
    ax.text(0.5, 4, 'ERT\nCalibration', fontsize=8, va='center',
           fontweight='bold', color=COLORS['accent'])
    ax.text(0.5, 2, 'Target\nResponse', fontsize=8, va='center',
           fontweight='bold', color=COLORS['orange'])

    # Checkmarks for verification steps
    for x, y in [(8, 6), (8, 4), (8, 2)]:
        ax.text(x + 0.8, y + 0.3, '\u2713', fontsize=14, color=COLORS['success'])

    # Title
    ax.text(5, 6.8, 'Calibration Workflow', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_troubleshooting_flowchart():
    """
    Create troubleshooting decision flowchart.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # Start
    draw_terminal(ax, 6, 8.5, 2.5, 0.5, 'PROBLEM', COLORS['light_red'])

    # First decision: Power?
    draw_decision_diamond(ax, 6, 7, 1.5, 'Power\nLED?', COLORS['light_blue'])
    draw_flow_arrow(ax, (6, 8.25), (6, 7.75))

    # No power path
    draw_flow_arrow(ax, (5.25, 7), (3, 7), 'No')
    draw_process_box(ax, 2, 7, 1.8, 0.8, 'Check\nbattery/\nconnector', COLORS['light_orange'])
    draw_flow_arrow(ax, (2, 6.6), (2, 6))
    draw_decision_diamond(ax, 2, 5.5, 1.2, 'Fixed?', COLORS['light_blue'])
    draw_flow_arrow(ax, (2.6, 5.5), (4, 5.5), 'No')
    draw_process_box(ax, 5, 5.5, 1.5, 0.6, 'Replace\npower\nsupply', COLORS['light_red'])

    # Power OK path
    draw_flow_arrow(ax, (6, 6.25), (6, 5.5), 'Yes')
    draw_decision_diamond(ax, 6, 5, 1.5, 'Data\nOutput?', COLORS['light_blue'])

    # No data path
    draw_flow_arrow(ax, (6.75, 5), (8.5, 5), 'No')
    draw_process_box(ax, 9.5, 5, 1.8, 0.8, 'Check\nUSB/\ncables', COLORS['light_orange'])
    draw_flow_arrow(ax, (9.5, 4.6), (9.5, 4))
    draw_decision_diamond(ax, 9.5, 3.5, 1.2, 'Fixed?', COLORS['light_blue'])

    # Data OK path
    draw_flow_arrow(ax, (6, 4.25), (6, 3.5), 'Yes')
    draw_decision_diamond(ax, 6, 3, 1.5, 'Valid\nReadings?', COLORS['light_blue'])

    # Invalid readings
    draw_flow_arrow(ax, (5.25, 3), (3.5, 3), 'No')
    draw_decision_diamond(ax, 2.5, 3, 1.2, 'MIT or\nERT?', COLORS['light_blue'])

    # MIT issue
    draw_flow_arrow(ax, (2.5, 2.4), (2.5, 1.5), 'MIT')
    draw_process_box(ax, 2.5, 1, 1.8, 0.8, 'Check coil\nconnections\n& frequency', COLORS['light_orange'])

    # ERT issue
    draw_flow_arrow(ax, (3.1, 3), (5, 3))
    draw_flow_arrow(ax, (5, 3), (5, 1.5), 'ERT')
    draw_process_box(ax, 5, 1, 1.8, 0.8, 'Check ERT\nrings &\ncurrent src', COLORS['light_orange'])

    # Valid readings - success
    draw_flow_arrow(ax, (6, 2.25), (6, 1.5), 'Yes')
    draw_terminal(ax, 6, 1, 2.5, 0.5, 'SYSTEM OK', COLORS['light_green'])

    # Title
    ax.text(6, 8.9, 'Quick Troubleshooting Guide', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_data_processing_pipeline():
    """
    Create data processing pipeline diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Pipeline stages
    stages = [
        (1, 2.5, 'Raw\nData', COLORS['light_blue']),
        (3, 2.5, 'Filter &\nAverage', COLORS['light_green']),
        (5, 2.5, 'Background\nRemoval', COLORS['light_green']),
        (7, 2.5, 'Tomographic\nInversion', COLORS['light_orange']),
        (9, 2.5, '3D\nModel', COLORS['light_purple']),
    ]

    for x, y, text, color in stages:
        draw_process_box(ax, x, y, 1.6, 1.2, text, color)

    # Flow arrows
    for i in range(len(stages) - 1):
        draw_flow_arrow(ax, (stages[i][0] + 0.8, 2.5), (stages[i+1][0] - 0.8, 2.5))

    # Data types below
    data_types = [
        (1, 1.2, 'ADC\nValues'),
        (3, 1.2, 'Noise\nReduced'),
        (5, 1.2, 'Anomaly\nData'),
        (7, 1.2, 'Voxel\nGrid'),
        (9, 1.2, 'Isosurface'),
    ]
    for x, y, text in data_types:
        ax.text(x, y, text, ha='center', va='center', fontsize=7,
               color=COLORS['gray_dark'], style='italic')

    # Input/Output
    ax.text(0.2, 2.5, 'INPUT', fontsize=8, ha='left', color=COLORS['gray_med'])
    draw_flow_arrow(ax, (0.6, 2.5), (0.2, 2.5))
    ax.text(10.8, 2.5, 'OUTPUT', fontsize=8, ha='right', color=COLORS['gray_med'])
    draw_flow_arrow(ax, (9.8, 2.5), (10.6, 2.5))

    # Processing time indicators
    ax.text(1, 4, '~1s', fontsize=7, ha='center', color=COLORS['accent'])
    ax.text(3, 4, '~5s', fontsize=7, ha='center', color=COLORS['accent'])
    ax.text(5, 4, '~2s', fontsize=7, ha='center', color=COLORS['accent'])
    ax.text(7, 4, '~30s', fontsize=7, ha='center', color=COLORS['accent'])
    ax.text(9, 4, '~10s', fontsize=7, ha='center', color=COLORS['accent'])

    # Title
    ax.text(5.5, 4.5, 'Data Processing Pipeline', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_survey_workflow():
    """
    Create complete field survey workflow.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Phases with vertical layout
    phases = [
        ('PLANNING', 7.5, [
            'Site reconnaissance',
            'Grid design',
            'Equipment check'
        ], COLORS['light_blue']),
        ('DEPLOYMENT', 5.5, [
            'Mark positions',
            'Insert probes',
            'Connect cables'
        ], COLORS['light_green']),
        ('ACQUISITION', 3.5, [
            'System test',
            'Run survey',
            'QC checks'
        ], COLORS['light_orange']),
        ('RECOVERY', 1.5, [
            'Data backup',
            'Remove probes',
            'Site restoration'
        ], COLORS['light_purple']),
    ]

    for phase_name, y, steps, color in phases:
        # Phase header
        ax.add_patch(FancyBboxPatch((0.5, y - 0.3), 2.5, 0.6,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', lw=1.5))
        ax.text(1.75, y, phase_name, ha='center', va='center',
               fontsize=10, fontweight='bold')

        # Steps
        for i, step in enumerate(steps):
            x = 4.5 + i * 2
            ax.add_patch(FancyBboxPatch((x - 0.8, y - 0.25), 1.6, 0.5,
                                        boxstyle="round,pad=0.02",
                                        facecolor='white', edgecolor=color, lw=1))
            ax.text(x, y, step, ha='center', va='center', fontsize=7)

            # Arrow to next step
            if i < len(steps) - 1:
                draw_flow_arrow(ax, (x + 0.8, y), (x + 1.2, y), color=color)

    # Phase transitions
    for i in range(len(phases) - 1):
        draw_flow_arrow(ax, (1.75, phases[i][1] - 0.3),
                       (1.75, phases[i+1][1] + 0.3), color=COLORS['gray_dark'])

    # Time estimates
    ax.text(9.5, 7.5, '~1-2h', fontsize=8, color=COLORS['gray_med'])
    ax.text(9.5, 5.5, '~2-4h', fontsize=8, color=COLORS['gray_med'])
    ax.text(9.5, 3.5, '~1-2h', fontsize=8, color=COLORS['gray_med'])
    ax.text(9.5, 1.5, '~1h', fontsize=8, color=COLORS['gray_med'])

    # Title
    ax.text(5, 8, 'Field Survey Workflow', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_safety_checklist_visual():
    """
    Create visual safety checklist diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal') # Prevent distortion
    ax.axis('off')

    # Warning header
    ax.add_patch(FancyBboxPatch((0.5, 6), 7, 0.8,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS['warning'], edgecolor='black', lw=2))
    ax.text(4, 6.4, '\u26A0  UXO SITE SAFETY CHECKLIST', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')

    # Checklist items
    items = [
        ('EOD clearance obtained', True),
        ('Site perimeter marked', True),
        ('100m exclusion zone established', True),
        ('Communication plan in place', True),
        ('First aid kit on site', True),
        ('Emergency contacts posted', True),
        ('Weather conditions checked', True),
        ('Soft insertion tools only', False),
    ]

    for i, (item, is_critical) in enumerate(items):
        y = 5.3 - i * 0.6
        color = COLORS['warning'] if is_critical else COLORS['success']

        # Checkbox
        ax.add_patch(Rectangle((0.8, y - 0.15), 0.3, 0.3,
                               facecolor='white', edgecolor=color, lw=2))

        # Item text
        ax.text(1.3, y, f'\u2610  {item}', fontsize=9, va='center')

        # Critical marker
        if is_critical:
            ax.text(7.2, y, '\u2605', fontsize=10, va='center', color=COLORS['warning'])

    # Legend
    ax.text(0.8, 0.5, '\u2605 = Critical safety item', fontsize=8, color=COLORS['warning'])

    # Title
    ax.text(4, 0.2, 'Complete ALL items before operations', ha='center',
           fontsize=9, style='italic', color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf
