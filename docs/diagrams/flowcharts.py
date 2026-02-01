"""
HIRT PDF Generator - Flowcharts Module

Functions for creating decision trees, workflows, timelines, and process diagrams.
Includes icons/pictograms, safety warning symbols, and progress indicators.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch,
    Ellipse, PathPatch, RegularPolygon, Wedge
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
    'light_green': '#c6f6d5',
    'light_blue': '#ebf8ff',
    'light_orange': '#feebc8',
    'light_red': '#fed7d7',
    'light_purple': '#e9d8fd',
    'caution_yellow': '#fef3c7',
    'caution_orange': '#ffedd5',
}

# Icon definitions for process step types
STEP_ICONS = {
    'start': 'play',
    'end': 'stop',
    'process': 'gear',
    'decision': 'question',
    'data': 'database',
    'manual': 'hand',
    'document': 'file',
    'test': 'check',
    'calibrate': 'target',
    'measure': 'ruler',
    'warning': 'alert',
    'wait': 'clock',
}


def _draw_icon(ax, x, y, icon_type, size=0.15, color='black'):
    """
    Draw a simple geometric icon/pictogram.

    Args:
        ax: matplotlib Axes object
        x, y: Center position
        icon_type: One of 'gear', 'check', 'target', 'ruler', 'clock', 'hand',
                   'file', 'database', 'question', 'play', 'stop', 'alert'
        size: Icon size
        color: Icon color
    """
    s = size  # shorthand

    if icon_type == 'gear':
        # Simple gear (octagon with center hole)
        ax.add_patch(RegularPolygon((x, y), 8, radius=s * 0.8, facecolor=color))
        ax.add_patch(Circle((x, y), s * 0.3, facecolor='white', edgecolor='none'))

    elif icon_type == 'check':
        # Checkmark
        check_x = [x - s*0.5, x - s*0.1, x + s*0.5]
        check_y = [y, y - s*0.3, y + s*0.5]
        ax.plot(check_x, check_y, color=color, lw=2, solid_capstyle='round')

    elif icon_type == 'target':
        # Target/crosshairs
        ax.add_patch(Circle((x, y), s * 0.7, facecolor='none', edgecolor=color, lw=1))
        ax.add_patch(Circle((x, y), s * 0.35, facecolor='none', edgecolor=color, lw=1))
        ax.add_patch(Circle((x, y), s * 0.1, facecolor=color, edgecolor='none'))

    elif icon_type == 'ruler':
        # Ruler/measure
        ax.add_patch(Rectangle((x - s*0.7, y - s*0.2), s*1.4, s*0.4,
                               facecolor=color, edgecolor='none'))
        for i in range(5):
            tick_x = x - s*0.6 + i * s*0.3
            tick_h = s*0.15 if i % 2 == 0 else s*0.1
            ax.plot([tick_x, tick_x], [y + s*0.2, y + s*0.2 - tick_h], color='white', lw=1)

    elif icon_type == 'clock':
        # Clock face
        ax.add_patch(Circle((x, y), s * 0.7, facecolor='white', edgecolor=color, lw=1.5))
        ax.plot([x, x], [y, y + s*0.4], color=color, lw=1.5)
        ax.plot([x, x + s*0.3], [y, y], color=color, lw=1.5)

    elif icon_type == 'hand':
        # Hand/manual (simplified)
        ax.add_patch(Circle((x, y), s * 0.5, facecolor=color, edgecolor='none', alpha=0.7))
        ax.plot([x, x], [y - s*0.5, y - s*0.9], color=color, lw=2)

    elif icon_type == 'file':
        # Document
        verts = [(x - s*0.4, y - s*0.5), (x - s*0.4, y + s*0.5),
                 (x + s*0.2, y + s*0.5), (x + s*0.4, y + s*0.3),
                 (x + s*0.4, y - s*0.5), (x - s*0.4, y - s*0.5)]
        patch = Polygon(verts, facecolor='white', edgecolor=color, lw=1)
        ax.add_patch(patch)
        # Folded corner
        ax.plot([x + s*0.2, x + s*0.2, x + s*0.4], [y + s*0.5, y + s*0.3, y + s*0.3], color=color, lw=1)

    elif icon_type == 'database':
        # Database cylinder (simplified)
        ax.add_patch(Ellipse((x, y + s*0.4), s*0.8, s*0.3, facecolor=color, edgecolor='none'))
        ax.add_patch(Rectangle((x - s*0.4, y - s*0.3), s*0.8, s*0.7, facecolor=color, edgecolor='none'))
        ax.add_patch(Ellipse((x, y - s*0.3), s*0.8, s*0.3, facecolor=color, edgecolor='none'))

    elif icon_type == 'question':
        # Question mark
        ax.text(x, y, '?', ha='center', va='center', fontsize=int(s*60),
               fontweight='bold', color=color)

    elif icon_type == 'play':
        # Play triangle
        verts = [(x - s*0.3, y - s*0.4), (x - s*0.3, y + s*0.4), (x + s*0.4, y)]
        patch = Polygon(verts, facecolor=color, edgecolor='none')
        ax.add_patch(patch)

    elif icon_type == 'stop':
        # Stop square
        ax.add_patch(Rectangle((x - s*0.3, y - s*0.3), s*0.6, s*0.6,
                               facecolor=color, edgecolor='none'))

    elif icon_type == 'alert':
        # Alert triangle with !
        verts = [(x, y + s*0.5), (x - s*0.5, y - s*0.4), (x + s*0.5, y - s*0.4)]
        patch = Polygon(verts, facecolor=WONG_PALETTE['yellow'], edgecolor=color, lw=1.5)
        ax.add_patch(patch)
        ax.text(x, y - s*0.05, '!', ha='center', va='center', fontsize=int(s*50),
               fontweight='bold', color=color)


def _draw_warning_triangle(ax, x, y, size=0.3, text='!'):
    """
    Draw a warning/caution triangle symbol.

    Args:
        ax: matplotlib Axes object
        x, y: Center position
        size: Triangle size
        text: Text inside triangle (default '!')
    """
    s = size
    # Triangle points (pointing up)
    verts = [(x, y + s*0.7), (x - s*0.6, y - s*0.4), (x + s*0.6, y - s*0.4)]
    # Outer triangle (dark border)
    patch = Polygon(verts, facecolor=WONG_PALETTE['yellow'], edgecolor=WONG_PALETTE['orange'], lw=2)
    ax.add_patch(patch)
    # Exclamation mark
    ax.text(x, y, text, ha='center', va='center', fontsize=int(s*35),
           fontweight='bold', color='black')


def _draw_progress_bar(ax, x, y, width, progress, height=0.1, label=None):
    """
    Draw a small progress bar.

    Args:
        ax: matplotlib Axes object
        x, y: Lower-left corner
        width: Total bar width
        progress: Progress value (0.0 to 1.0)
        height: Bar height
        label: Optional label (e.g., '~5min')
    """
    # Background bar
    ax.add_patch(Rectangle((x, y), width, height,
                           facecolor=COLORS['gray_light'], edgecolor='none', alpha=0.5))
    # Progress fill
    fill_width = width * min(1.0, max(0.0, progress))
    ax.add_patch(Rectangle((x, y), fill_width, height,
                           facecolor=WONG_PALETTE['bluish_green'], edgecolor='none'))
    # Border
    ax.add_patch(Rectangle((x, y), width, height,
                           facecolor='none', edgecolor=COLORS['gray_dark'], lw=0.5))
    # Label
    if label:
        ax.text(x + width + 0.05, y + height/2, label, ha='left', va='center',
               fontsize=6, color=COLORS['gray_dark'])


def _draw_duration_label(ax, x, y, duration_text, color=None):
    """
    Draw a duration/time label with clock icon.

    Args:
        ax: matplotlib Axes object
        x, y: Position
        duration_text: Duration string (e.g., '~5min')
        color: Text color
    """
    if color is None:
        color = COLORS['accent']
    _draw_icon(ax, x - 0.15, y, 'clock', size=0.08, color=color)
    ax.text(x + 0.05, y, duration_text, ha='left', va='center',
           fontsize=6, color=color)


def draw_process_box(ax, x, y, width, height, text, color='white',
                     icon=None, is_warning=False, duration=None, progress=None):
    """
    Draw a process box (rectangle with rounded corners) with optional icon,
    warning indicator, and progress/duration display.

    Args:
        ax: matplotlib Axes object
        x, y: Center position
        width, height: Box dimensions
        text: Label text
        color: Fill color
        icon: Icon type to draw (see STEP_ICONS)
        is_warning: If True, add warning styling
        duration: Duration string (e.g., '~5min')
        progress: Progress value 0.0-1.0 for progress bar
    """
    # Adjust color for warning steps
    if is_warning:
        color = COLORS['caution_yellow']
        edge_color = WONG_PALETTE['orange']
        edge_width = 2
    else:
        edge_color = 'black'
        edge_width = 1

    patch = FancyBboxPatch((x - width/2, y - height/2), width, height,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor=edge_color, linewidth=edge_width)
    ax.add_patch(patch)

    # Draw icon if specified
    text_offset = 0
    if icon:
        icon_x = x - width/2 + 0.2
        icon_y = y
        _draw_icon(ax, icon_x, icon_y, icon, size=0.12, color=COLORS['gray_dark'])
        text_offset = 0.15  # Shift text right

    # Draw warning triangle if warning step
    if is_warning:
        _draw_warning_triangle(ax, x + width/2 - 0.15, y + height/2 - 0.15, size=0.12)

    # Main text
    ax.text(x + text_offset, y, text, ha='center', va='center', fontsize=8, wrap=True)

    # Duration label or progress bar below
    if duration:
        _draw_duration_label(ax, x - width/4, y - height/2 - 0.12, duration)
    if progress is not None:
        _draw_progress_bar(ax, x - width/3, y - height/2 - 0.08, width*0.6, progress)

    return patch


def draw_decision_diamond(ax, x, y, size, text, color='white', is_warning=False):
    """
    Draw a decision diamond with optional warning styling.

    Args:
        ax: matplotlib Axes object
        x, y: Center position
        size: Diamond size
        text: Label text
        color: Fill color
        is_warning: If True, add warning styling
    """
    half = size / 2

    if is_warning:
        color = COLORS['caution_orange']
        edge_color = WONG_PALETTE['vermillion']
        edge_width = 2
    else:
        edge_color = 'black'
        edge_width = 1

    points = [(x, y + half), (x + half, y), (x, y - half), (x - half, y)]
    patch = Polygon(points, facecolor=color, edgecolor=edge_color, linewidth=edge_width)
    ax.add_patch(patch)

    # Add question mark icon for decisions
    _draw_icon(ax, x, y + half*0.5, 'question', size=0.08, color=COLORS['gray_med'])

    ax.text(x, y - 0.05, text, ha='center', va='center', fontsize=7)
    return patch


def draw_terminal(ax, x, y, width, height, text, color='white', is_start=True):
    """
    Draw a terminal/start/end box (pill shape) with icon.

    Args:
        ax: matplotlib Axes object
        x, y: Center position
        width, height: Box dimensions
        text: Label text
        color: Fill color
        is_start: If True, show play icon; if False, show stop icon
    """
    patch = FancyBboxPatch((x - width/2, y - height/2), width, height,
                           boxstyle="round,pad=0.3",
                           facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(patch)

    # Add start/stop icon
    icon_type = 'play' if is_start else 'stop'
    icon_color = COLORS['success'] if is_start else COLORS['warning']
    _draw_icon(ax, x - width/3, y, icon_type, size=0.1, color=icon_color)

    ax.text(x + 0.1, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
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
    Create soil type insertion decision tree with icons and warning indicators.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 8.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Start
    draw_terminal(ax, 5, 7.5, 2, 0.5, 'START', COLORS['light_green'], is_start=True)

    # First decision: Soil hardness
    draw_decision_diamond(ax, 5, 6, 1.5, 'Soil\nHard?', COLORS['light_blue'])
    draw_flow_arrow(ax, (5, 7.25), (5, 6.75))

    # Hard soil path
    draw_flow_arrow(ax, (5.75, 6), (7.5, 6), 'Yes')
    draw_process_box(ax, 8.5, 6, 1.8, 0.8, 'Use hand\nauger\n(18mm)', COLORS['light_orange'],
                    icon='hand', duration='~5min')

    # Soft soil path
    draw_flow_arrow(ax, (5, 5.25), (5, 4.5), 'No')
    draw_decision_diamond(ax, 5, 4, 1.5, 'Water\nTable?', COLORS['light_blue'])

    # High water table - warning for wet conditions
    draw_flow_arrow(ax, (5.75, 4), (7.5, 4), 'High')
    draw_process_box(ax, 8.5, 4, 1.8, 0.8, 'Direct push\n(wet method)', COLORS['light_orange'],
                    icon='hand', is_warning=True, duration='~3min')

    # Low water table
    draw_flow_arrow(ax, (5, 3.25), (5, 2.5), 'Low')
    draw_decision_diamond(ax, 5, 2, 1.5, 'Stones/\nRoots?', COLORS['light_blue'])

    # With obstructions - warning
    draw_flow_arrow(ax, (5.75, 2), (7.5, 2), 'Yes')
    draw_process_box(ax, 8.5, 2, 1.8, 0.8, 'Careful\nauger +\nprobe', COLORS['light_orange'],
                    icon='target', is_warning=True, duration='~7min')

    # Clear path
    draw_flow_arrow(ax, (5, 1.25), (5, 0.5))
    draw_process_box(ax, 5, 0.5, 2.0, 0.6, 'Direct push\nonly', COLORS['light_green'],
                    icon='check', duration='~2min')

    # Convergence arrows to success
    for y in [6, 4, 2]:
        draw_flow_arrow(ax, (8.5, y - 0.4), (8.5, 0.7), '')

    # Success terminal
    draw_terminal(ax, 8.5, 0.5, 1.8, 0.5, 'INSERT', COLORS['light_green'], is_start=False)

    # Title
    ax.text(5, 8.1, 'Soil Insertion Decision Tree', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    # Notes box with warning styling
    notes_box = FancyBboxPatch((0.5, 0.8), 2.8, 1.6, boxstyle='round,pad=0.05',
                                facecolor=COLORS['caution_yellow'], edgecolor=WONG_PALETTE['orange'], lw=1.5)
    ax.add_patch(notes_box)
    _draw_warning_triangle(ax, 0.9, 2.1, size=0.12)
    ax.text(1.9, 2.1, 'SAFETY NOTES', fontsize=8, fontweight='bold', va='center')
    ax.text(0.7, 1.65, '\u2022 Never hammer probes', fontsize=6, va='center')
    ax.text(0.7, 1.35, '\u2022 Max force: hand pressure', fontsize=6, va='center')
    ax.text(0.7, 1.05, '\u2022 If blocked, relocate', fontsize=6, va='center')

    # Legend
    ax.add_patch(FancyBboxPatch((0.5, -0.2), 3.5, 0.5, boxstyle="round,pad=0.02",
                                facecolor='white', edgecolor=COLORS['gray_light'], lw=1))
    _draw_icon(ax, 0.8, 0.05, 'clock', size=0.1, color=COLORS['accent'])
    ax.text(1.05, 0.05, '= Est. time', fontsize=6, va='center')
    _draw_warning_triangle(ax, 2.2, 0.05, size=0.1)
    ax.text(2.5, 0.05, '= Caution required', fontsize=6, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_calibration_workflow():
    """
    Create calibration procedure workflow with icons, progress indicators, and timing.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 7.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Workflow steps with icons, durations, and progress
    # (x, y, text, color, icon, duration, progress, is_warning)
    steps = [
        (2, 6, 'Air\nCalibration', COLORS['light_blue'], 'calibrate', '~2min', 0.15, False),
        (5, 6, 'Known\nResistor\nTest', COLORS['light_blue'], 'measure', '~3min', 0.35, False),
        (8, 6, 'Verify\nReadings', COLORS['light_green'], 'check', '~1min', 0.45, False),
        (2, 4, 'Water\nBucket\nTest', COLORS['light_blue'], 'measure', '~5min', 0.55, False),
        (5, 4, 'Known\nConductivity\nSolution', COLORS['light_blue'], 'calibrate', '~5min', 0.70, True),  # Warning: chemical
        (8, 4, 'Record\nBaseline', COLORS['light_green'], 'file', '~2min', 0.80, False),
        (2, 2, 'Metal\nTarget\nTest', COLORS['light_orange'], 'target', '~5min', 0.85, False),
        (5, 2, 'Distance\nSweep', COLORS['light_orange'], 'ruler', '~10min', 0.95, False),
        (8, 2, 'Build\nLookup\nTable', COLORS['light_green'], 'database', '~3min', 1.0, False),
    ]

    for x, y, text, color, icon, duration, progress, is_warning in steps:
        draw_process_box(ax, x, y, 2.0, 1.0, text, color,
                        icon=icon, is_warning=is_warning, duration=duration, progress=progress)

    # Horizontal arrows
    for y in [6, 4, 2]:
        draw_flow_arrow(ax, (3, y), (4, y))
        draw_flow_arrow(ax, (6, y), (7, y))

    # Vertical arrows
    draw_flow_arrow(ax, (8, 5.5), (8, 4.5))
    draw_flow_arrow(ax, (8, 3.5), (8, 2.5))

    # Phase labels with icons
    ax.text(0.5, 6, 'MIT\nCalibration', fontsize=8, va='center',
           fontweight='bold', color=WONG_PALETTE['blue'])
    _draw_icon(ax, 0.2, 6.4, 'gear', size=0.1, color=WONG_PALETTE['blue'])

    ax.text(0.5, 4, 'ERT\nCalibration', fontsize=8, va='center',
           fontweight='bold', color=WONG_PALETTE['blue'])
    _draw_icon(ax, 0.2, 4.4, 'ruler', size=0.1, color=WONG_PALETTE['blue'])

    ax.text(0.5, 2, 'Target\nResponse', fontsize=8, va='center',
           fontweight='bold', color=WONG_PALETTE['orange'])
    _draw_icon(ax, 0.2, 2.4, 'target', size=0.1, color=WONG_PALETTE['orange'])

    # Checkmarks for verification steps (enhanced)
    for x, y in [(8, 6), (8, 4), (8, 2)]:
        ax.add_patch(Circle((x + 0.85, y + 0.35), 0.12, facecolor=COLORS['success'],
                           edgecolor='none'))
        ax.text(x + 0.85, y + 0.35, '\u2713', fontsize=9, ha='center', va='center',
               color='white', fontweight='bold')

    # Total time summary
    ax.add_patch(FancyBboxPatch((0.3, 0.0), 2.5, 0.5, boxstyle="round,pad=0.03",
                                facecolor=COLORS['light_blue'], edgecolor=COLORS['accent'], lw=1))
    _draw_icon(ax, 0.6, 0.25, 'clock', size=0.12, color=COLORS['accent'])
    ax.text(1.55, 0.25, 'Total: ~35 min', ha='center', va='center',
           fontsize=8, fontweight='bold', color=COLORS['primary'])

    # Warning legend
    ax.add_patch(FancyBboxPatch((6.5, 0.0), 3.0, 0.5, boxstyle="round,pad=0.03",
                                facecolor=COLORS['caution_yellow'], edgecolor=WONG_PALETTE['orange'], lw=1))
    _draw_warning_triangle(ax, 6.8, 0.25, size=0.1)
    ax.text(8.0, 0.25, '= Requires caution', ha='center', va='center',
           fontsize=7, color=COLORS['gray_dark'])

    # Title
    ax.text(5, 7.1, 'Calibration Workflow', ha='center',
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
    Create troubleshooting decision flowchart with icons and warning indicators.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(12, 9.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.3, 9.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Start with warning icon
    draw_terminal(ax, 6, 8.5, 2.5, 0.5, 'PROBLEM', COLORS['light_red'], is_start=False)
    _draw_warning_triangle(ax, 4.9, 8.5, size=0.15)

    # First decision: Power?
    draw_decision_diamond(ax, 6, 7, 1.5, 'Power\nLED?', COLORS['light_blue'])
    draw_flow_arrow(ax, (6, 8.25), (6, 7.75))

    # No power path
    draw_flow_arrow(ax, (5.25, 7), (3, 7), 'No')
    draw_process_box(ax, 2, 7, 1.8, 0.8, 'Check\nbattery/\nconnector', COLORS['light_orange'],
                    icon='gear', duration='~2min')
    draw_flow_arrow(ax, (2, 6.6), (2, 6))
    draw_decision_diamond(ax, 2, 5.5, 1.2, 'Fixed?', COLORS['light_blue'])
    draw_flow_arrow(ax, (2.6, 5.5), (4, 5.5), 'No')
    draw_process_box(ax, 5, 5.5, 1.5, 0.6, 'Replace\npower\nsupply', COLORS['light_red'],
                    icon='alert', is_warning=True)

    # Power OK path
    draw_flow_arrow(ax, (6, 6.25), (6, 5.5), 'Yes')
    draw_decision_diamond(ax, 6, 5, 1.5, 'Data\nOutput?', COLORS['light_blue'])

    # No data path
    draw_flow_arrow(ax, (6.75, 5), (8.5, 5), 'No')
    draw_process_box(ax, 9.5, 5, 1.8, 0.8, 'Check\nUSB/\ncables', COLORS['light_orange'],
                    icon='gear', duration='~3min')
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
    draw_process_box(ax, 2.5, 1, 1.8, 0.8, 'Check coil\nconnections\n& frequency', COLORS['light_orange'],
                    icon='target', duration='~5min')

    # ERT issue
    draw_flow_arrow(ax, (3.1, 3), (5, 3))
    draw_flow_arrow(ax, (5, 3), (5, 1.5), 'ERT')
    draw_process_box(ax, 5, 1, 1.8, 0.8, 'Check ERT\nrings &\ncurrent src', COLORS['light_orange'],
                    icon='ruler', duration='~5min')

    # Valid readings - success
    draw_flow_arrow(ax, (6, 2.25), (6, 1.5), 'Yes')
    draw_terminal(ax, 6, 1, 2.5, 0.5, 'SYSTEM OK', COLORS['light_green'], is_start=False)
    _draw_icon(ax, 7.1, 1, 'check', size=0.15, color=COLORS['success'])

    # Title
    ax.text(6, 9.1, 'Quick Troubleshooting Guide', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    # Legend at bottom
    ax.add_patch(FancyBboxPatch((0.5, -0.2), 4.0, 0.5, boxstyle="round,pad=0.02",
                                facecolor='white', edgecolor=COLORS['gray_light'], lw=1))
    _draw_icon(ax, 0.85, 0.05, 'gear', size=0.1, color=COLORS['gray_med'])
    ax.text(1.15, 0.05, '= Action step', fontsize=6, va='center')
    _draw_icon(ax, 2.3, 0.05, 'question', size=0.1, color=COLORS['gray_med'])
    ax.text(2.6, 0.05, '= Decision', fontsize=6, va='center')
    _draw_warning_triangle(ax, 3.6, 0.05, size=0.1)
    ax.text(3.9, 0.05, '= Critical', fontsize=6, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_data_processing_pipeline():
    """
    Create data processing pipeline diagram with icons and progress indicators.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.2, 5.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Pipeline stages with icons and timing
    # (x, y, text, color, icon, duration, cumulative_progress)
    stages = [
        (1, 2.5, 'Raw\nData', COLORS['light_blue'], 'database', '~1s', 0.02),
        (3, 2.5, 'Filter &\nAverage', COLORS['light_green'], 'gear', '~5s', 0.12),
        (5, 2.5, 'Background\nRemoval', COLORS['light_green'], 'gear', '~2s', 0.16),
        (7, 2.5, 'Tomographic\nInversion', COLORS['light_orange'], 'gear', '~30s', 0.78),
        (9, 2.5, '3D\nModel', COLORS['light_purple'], 'file', '~10s', 1.0),
    ]

    for x, y, text, color, icon, duration, progress in stages:
        # Draw box with icon
        draw_process_box(ax, x, y, 1.6, 1.2, text, color, icon=icon)

        # Progress bar above showing cumulative progress
        _draw_progress_bar(ax, x - 0.6, y + 0.75, 1.2, progress, height=0.1)

        # Duration label with clock icon
        _draw_duration_label(ax, x - 0.4, y + 1.0, duration, color=WONG_PALETTE['blue'])

    # Flow arrows with data transformation labels
    transformations = ['Sample', 'Smooth', 'Subtract', 'Invert']
    for i in range(len(stages) - 1):
        draw_flow_arrow(ax, (stages[i][0] + 0.8, 2.5), (stages[i+1][0] - 0.8, 2.5))
        # Arrow label
        mid_x = (stages[i][0] + stages[i+1][0]) / 2
        ax.text(mid_x, 2.8, transformations[i], ha='center', va='bottom',
               fontsize=5, color=COLORS['gray_med'], style='italic')

    # Data types below with icons
    data_types = [
        (1, 1.1, 'ADC\nValues', 'database'),
        (3, 1.1, 'Noise\nReduced', 'check'),
        (5, 1.1, 'Anomaly\nData', 'target'),
        (7, 1.1, 'Voxel\nGrid', 'database'),
        (9, 1.1, 'Isosurface', 'file'),
    ]
    for x, y, text, icon in data_types:
        _draw_icon(ax, x - 0.4, y + 0.15, icon, size=0.06, color=COLORS['gray_light'])
        ax.text(x + 0.1, y, text, ha='center', va='center', fontsize=6,
               color=COLORS['gray_dark'], style='italic')

    # Input/Output with icons
    _draw_icon(ax, 0.15, 2.5, 'play', size=0.1, color=COLORS['success'])
    ax.text(0.15, 2.15, 'INPUT', fontsize=7, ha='center', color=COLORS['gray_med'])
    draw_flow_arrow(ax, (0.35, 2.5), (0.2, 2.5))

    _draw_icon(ax, 10.85, 2.5, 'stop', size=0.1, color=WONG_PALETTE['vermillion'])
    ax.text(10.85, 2.15, 'OUTPUT', fontsize=7, ha='center', color=COLORS['gray_med'])
    draw_flow_arrow(ax, (9.8, 2.5), (10.65, 2.5))

    # Total time summary at bottom
    ax.add_patch(FancyBboxPatch((4.0, 0.0), 3.0, 0.5, boxstyle="round,pad=0.03",
                                facecolor=COLORS['light_blue'], edgecolor=COLORS['accent'], lw=1))
    _draw_icon(ax, 4.3, 0.25, 'clock', size=0.12, color=COLORS['accent'])
    ax.text(5.5, 0.25, 'Total: ~48s', ha='center', va='center',
           fontsize=8, fontweight='bold', color=COLORS['primary'])

    # Title
    ax.text(5.5, 4.9, 'Data Processing Pipeline', ha='center',
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
    Create complete field survey workflow with icons, progress indicators, and timing.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 8.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Phases with vertical layout - now including icons and warnings
    # (phase_name, y, [(step_name, icon, is_warning), ...], color, duration, cumulative_progress)
    phases = [
        ('PLANNING', 7.5, [
            ('Site reconnaissance', 'target', False),
            ('Grid design', 'file', False),
            ('Equipment check', 'check', False)
        ], COLORS['light_blue'], '~1-2h', 0.15),
        ('DEPLOYMENT', 5.5, [
            ('Mark positions', 'ruler', False),
            ('Insert probes', 'hand', True),  # Warning: careful insertion
            ('Connect cables', 'gear', False)
        ], COLORS['light_green'], '~2-4h', 0.50),
        ('ACQUISITION', 3.5, [
            ('System test', 'check', False),
            ('Run survey', 'database', False),
            ('QC checks', 'check', False)
        ], COLORS['light_orange'], '~1-2h', 0.80),
        ('RECOVERY', 1.5, [
            ('Data backup', 'file', False),
            ('Remove probes', 'hand', False),
            ('Site restoration', 'check', False)
        ], COLORS['light_purple'], '~1h', 1.0),
    ]

    phase_icons = ['file', 'gear', 'measure', 'check']

    for (phase_name, y, steps, color, duration, progress), phase_icon in zip(phases, phase_icons):
        # Phase header with icon
        ax.add_patch(FancyBboxPatch((0.5, y - 0.35), 2.5, 0.7,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', lw=1.5))
        _draw_icon(ax, 0.85, y + 0.1, phase_icon, size=0.12, color=COLORS['gray_dark'])
        ax.text(1.85, y + 0.05, phase_name, ha='center', va='center',
               fontsize=10, fontweight='bold')

        # Progress bar under phase header
        _draw_progress_bar(ax, 0.7, y - 0.5, 2.1, progress, height=0.08)

        # Steps with icons
        for i, (step_name, step_icon, is_warning) in enumerate(steps):
            x = 4.5 + i * 2

            # Step box with warning styling if needed
            if is_warning:
                box_color = COLORS['caution_yellow']
                edge_color = WONG_PALETTE['orange']
                edge_width = 1.5
            else:
                box_color = 'white'
                edge_color = color
                edge_width = 1

            ax.add_patch(FancyBboxPatch((x - 0.85, y - 0.3), 1.7, 0.6,
                                        boxstyle="round,pad=0.02",
                                        facecolor=box_color, edgecolor=edge_color, lw=edge_width))

            # Icon
            _draw_icon(ax, x - 0.55, y + 0.1, step_icon, size=0.08, color=COLORS['gray_med'])

            # Warning triangle if needed
            if is_warning:
                _draw_warning_triangle(ax, x + 0.6, y + 0.15, size=0.08)

            ax.text(x + 0.1, y - 0.05, step_name, ha='center', va='center', fontsize=6)

            # Arrow to next step
            if i < len(steps) - 1:
                draw_flow_arrow(ax, (x + 0.85, y), (x + 1.15, y), color=color)

        # Duration label with clock
        _draw_duration_label(ax, 9.3, y, duration, color=COLORS['gray_med'])

    # Phase transitions with progress indication
    for i in range(len(phases) - 1):
        draw_flow_arrow(ax, (1.75, phases[i][1] - 0.5),
                       (1.75, phases[i+1][1] + 0.35), color=COLORS['gray_dark'])

    # Title
    ax.text(5, 8.1, 'Field Survey Workflow', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    # Total time summary
    ax.add_patch(FancyBboxPatch((3.5, 0.0), 3.0, 0.5, boxstyle="round,pad=0.03",
                                facecolor=COLORS['light_blue'], edgecolor=COLORS['accent'], lw=1))
    _draw_icon(ax, 3.8, 0.25, 'clock', size=0.12, color=COLORS['accent'])
    ax.text(5.0, 0.25, 'Total: ~5-9 hours', ha='center', va='center',
           fontsize=8, fontweight='bold', color=COLORS['primary'])

    # Warning legend
    ax.add_patch(FancyBboxPatch((7.0, 0.0), 2.5, 0.5, boxstyle="round,pad=0.03",
                                facecolor=COLORS['caution_yellow'], edgecolor=WONG_PALETTE['orange'], lw=1))
    _draw_warning_triangle(ax, 7.3, 0.25, size=0.1)
    ax.text(8.25, 0.25, '= Caution step', ha='center', va='center',
           fontsize=7, color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_safety_checklist_visual():
    """
    Create visual safety checklist diagram with enhanced warning symbols.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.3, 7.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Warning header with large warning triangle
    ax.add_patch(FancyBboxPatch((0.5, 6), 7, 0.9,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS['warning'], edgecolor='black', lw=2))
    _draw_warning_triangle(ax, 1.2, 6.45, size=0.25, text='!')
    ax.text(4.2, 6.45, 'UXO SITE SAFETY CHECKLIST', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')

    # Checklist items with icons and warning indicators
    items = [
        ('EOD clearance obtained', True, 'check'),
        ('Site perimeter marked', True, 'target'),
        ('100m exclusion zone established', True, 'ruler'),
        ('Communication plan in place', True, 'file'),
        ('First aid kit on site', True, 'check'),
        ('Emergency contacts posted', True, 'file'),
        ('Weather conditions checked', True, 'clock'),
        ('Soft insertion tools only', False, 'hand'),
    ]

    for i, (item, is_critical, icon) in enumerate(items):
        y = 5.4 - i * 0.6

        # Row background for critical items
        if is_critical:
            ax.add_patch(Rectangle((0.5, y - 0.25), 7, 0.5,
                                   facecolor=COLORS['caution_yellow'], edgecolor='none', alpha=0.3))

        color = COLORS['warning'] if is_critical else COLORS['success']

        # Checkbox with check icon inside
        ax.add_patch(Rectangle((0.7, y - 0.15), 0.3, 0.3,
                               facecolor='white', edgecolor=color, lw=2))

        # Small icon
        _draw_icon(ax, 1.25, y, icon, size=0.08, color=COLORS['gray_med'])

        # Item text
        ax.text(1.5, y, item, fontsize=9, va='center')

        # Warning triangle for critical items
        if is_critical:
            _draw_warning_triangle(ax, 7.0, y, size=0.12)

    # Legend with icons
    legend_y = 0.3

    # Critical item legend
    _draw_warning_triangle(ax, 0.9, legend_y, size=0.12)
    ax.text(1.2, legend_y, '= Critical safety item (must complete)', fontsize=7, va='center')

    # Normal item legend
    _draw_icon(ax, 4.5, legend_y, 'check', size=0.1, color=COLORS['success'])
    ax.text(4.8, legend_y, '= Standard item', fontsize=7, va='center')

    # Bottom warning
    ax.add_patch(FancyBboxPatch((1.5, -0.2), 5, 0.4, boxstyle="round,pad=0.02",
                                facecolor=COLORS['light_red'], edgecolor=COLORS['warning'], lw=1))
    ax.text(4, 0, 'Complete ALL items before operations', ha='center', va='center',
           fontsize=9, fontweight='bold', color=COLORS['warning'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf
