"""
HIRT PDF Generator - Diagrams Module

Common utilities and functions for creating matplotlib diagrams
for the HIRT whitepaper PDF generator.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, Wedge,
    FancyArrowPatch, Arc, Ellipse, PathPatch
)
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patheffects as path_effects
import numpy as np
from io import BytesIO

# Import specialized diagram modules
from . import subsurface
from . import circuits
from . import physics
from . import mechanical
from . import field_ops
from . import flowcharts

# Color palette for consistent styling
COLORS = {
    'primary': '#1a365d',
    'secondary': '#2c5282',
    'accent': '#3182ce',
    'success': '#38a169',
    'warning': '#c53030',
    'light_bg': '#f7fafc',
    'ground_tan': '#d4a373',
    'orange': '#ed8936',
    'purple': '#805ad5',
    'gray_dark': '#4a5568',
    'gray_med': '#718096',
    'gray_light': '#a0aec0',
    'sky': '#e8f4f8',
    'ground_dark': '#654321',
    'tx_coil': '#38a169',
    'rx_coil': '#3182ce',
    'ert_ring': '#ed8936',
    'probe_body': '#2c5282',
    'connector': '#4a5568',
    'white': '#ffffff',
    'black': '#000000',
}


def save_figure_to_buffer(fig, dpi=200):
    """
    Save a matplotlib figure to a BytesIO buffer as PNG.

    Args:
        fig: matplotlib Figure object
        dpi: Resolution in dots per inch (default: 200)

    Returns:
        BytesIO buffer containing the PNG image
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_figure(width=10, height=6, subplots=None):
    """
    Create a new figure with consistent styling.

    Args:
        width: Figure width in inches
        height: Figure height in inches
        subplots: Tuple of (rows, cols) for subplot grid, or None for single axes

    Returns:
        Tuple of (fig, ax) or (fig, axes) if subplots specified
    """
    if subplots:
        fig, axes = plt.subplots(subplots[0], subplots[1], figsize=(width, height))
        return fig, axes
    else:
        fig, ax = plt.subplots(figsize=(width, height))
        return fig, ax


def setup_axes_clean(ax, xlim=None, ylim=None, aspect='equal'):
    """
    Set up axes with clean styling (no spines, no ticks).

    Args:
        ax: matplotlib Axes object
        xlim: Tuple of (xmin, xmax) or None
        ylim: Tuple of (ymin, ymax) or None
        aspect: Aspect ratio ('equal', 'auto', or numeric)
    """
    ax.set_aspect(aspect)
    ax.axis('off')
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def setup_axes_scientific(ax, xlabel='', ylabel='', title=''):
    """
    Set up axes with scientific styling (visible spines, gridlines).

    Args:
        ax: matplotlib Axes object
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Axes title
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle=':', alpha=0.5)
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', color=COLORS['primary'], pad=10)


def add_title(ax, text, fontsize=11, pad=10):
    """Add a title to an axes with consistent styling."""
    ax.set_title(text, fontsize=fontsize, fontweight='bold',
                 color=COLORS['primary'], pad=pad)


def add_legend(ax, handles=None, labels=None, loc='best', fontsize=8):
    """Add a legend with consistent styling."""
    if handles and labels:
        ax.legend(handles, labels, loc=loc, fontsize=fontsize, framealpha=0.9)
    else:
        ax.legend(loc=loc, fontsize=fontsize, framealpha=0.9)


def draw_ground_surface(ax, xmin, xmax, y=0, color=None):
    """
    Draw a ground surface line.

    Args:
        ax: matplotlib Axes object
        xmin: Left extent
        xmax: Right extent
        y: Y position of ground surface
        color: Line color (default: brown)
    """
    if color is None:
        color = COLORS['ground_dark']
    ax.axhline(y, xmin=0, xmax=1, color=color, linewidth=2.5, zorder=5)


def draw_ground_layers(ax, xmin, xmax, depths, alphas=None, color=None):
    """
    Draw ground layers with gradient alpha.

    Args:
        ax: matplotlib Axes object
        xmin: Left extent
        xmax: Right extent
        depths: List of (top_depth, bottom_depth) tuples
        alphas: List of alpha values (default: gradual increase)
        color: Fill color (default: tan)
    """
    if color is None:
        color = COLORS['ground_tan']
    if alphas is None:
        alphas = [0.3 + 0.1*i for i in range(len(depths))]

    for (d1, d2), alpha in zip(depths, alphas):
        ax.fill_between([xmin, xmax], [d1, d1], [d2, d2],
                       color=color, alpha=alpha)


def draw_sky(ax, xmin, xmax, ymin=0, ymax=1, color=None):
    """
    Draw sky/air region above ground.

    Args:
        ax: matplotlib Axes object
        xmin, xmax: Horizontal extent
        ymin: Bottom of sky (usually ground surface)
        ymax: Top of sky
        color: Fill color (default: light blue)
    """
    if color is None:
        color = COLORS['sky']
    ax.fill_between([xmin, xmax], [ymin, ymin], [ymax, ymax],
                   color=color, alpha=0.5)


def draw_depth_markers(ax, xpos, depths, fontsize=7):
    """
    Draw horizontal depth marker lines with labels.

    Args:
        ax: matplotlib Axes object
        xpos: X position for depth labels
        depths: List of depth values (positive numbers, plotted as negative)
    """
    for d in depths:
        ax.axhline(-d, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.text(xpos, -d, f'{d}m', fontsize=fontsize, va='center', color='gray')


def draw_arrow(ax, start, end, color='black', style='->', lw=1.5, **kwargs):
    """
    Draw an arrow between two points.

    Args:
        ax: matplotlib Axes object
        start: (x, y) tuple for start point
        end: (x, y) tuple for end point
        color: Arrow color
        style: Arrow style (e.g., '->', '<->', '-')
        lw: Line width
        **kwargs: Additional arguments passed to annotate
    """
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                **kwargs)


def draw_dimension_line(ax, start, end, label, offset=0.1, fontsize=8):
    """
    Draw a dimension line with arrows and centered label.

    Args:
        ax: matplotlib Axes object
        start: (x, y) tuple for start point
        end: (x, y) tuple for end point
        label: Dimension label text
        offset: Vertical offset for label
        fontsize: Label font size
    """
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2

    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray_med'], lw=1))
    ax.text(mid_x, mid_y + offset, label, ha='center', fontsize=fontsize,
            fontweight='bold')


def draw_box(ax, x, y, width, height, text='', color=None, edgecolor='black',
             fontsize=8, text_color=None, rounded=True, **kwargs):
    """
    Draw a box with optional text label.

    Args:
        ax: matplotlib Axes object
        x, y: Lower-left corner coordinates
        width, height: Box dimensions
        text: Label text
        color: Fill color
        edgecolor: Edge color
        fontsize: Label font size
        text_color: Label color
        rounded: Whether to use rounded corners
        **kwargs: Additional patch arguments
    """
    if rounded:
        patch = FancyBboxPatch((x, y), width, height,
                               boxstyle="round,pad=0.02",
                               facecolor=color or 'white',
                               edgecolor=edgecolor, linewidth=1, **kwargs)
    else:
        patch = Rectangle((x, y), width, height,
                         facecolor=color or 'white',
                         edgecolor=edgecolor, linewidth=1, **kwargs)
    ax.add_patch(patch)

    if text:
        if text_color is None:
            text_color = 'black'
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
               fontsize=fontsize, color=text_color)

    return patch


def draw_circle(ax, x, y, radius, color=None, edgecolor='black', **kwargs):
    """
    Draw a circle.

    Args:
        ax: matplotlib Axes object
        x, y: Center coordinates
        radius: Circle radius
        color: Fill color
        edgecolor: Edge color
        **kwargs: Additional Circle arguments

    Returns:
        Circle patch object
    """
    circle = Circle((x, y), radius, facecolor=color or 'white',
                   edgecolor=edgecolor, linewidth=1, **kwargs)
    ax.add_patch(circle)
    return circle


def draw_ellipse(ax, x, y, width, height, color=None, alpha=0.3, **kwargs):
    """
    Draw an ellipse.

    Args:
        ax: matplotlib Axes object
        x, y: Center coordinates
        width, height: Ellipse dimensions
        color: Fill color
        alpha: Transparency
        **kwargs: Additional Ellipse arguments

    Returns:
        Ellipse patch object
    """
    ellipse = Ellipse((x, y), width, height, facecolor=color or COLORS['success'],
                     alpha=alpha, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


# Export all colors and functions
__all__ = [
    'COLORS',
    'save_figure_to_buffer',
    'create_figure',
    'setup_axes_clean',
    'setup_axes_scientific',
    'add_title',
    'add_legend',
    'draw_ground_surface',
    'draw_ground_layers',
    'draw_sky',
    'draw_depth_markers',
    'draw_arrow',
    'draw_dimension_line',
    'draw_box',
    'draw_circle',
    'draw_ellipse',
    'subsurface',
    'circuits',
    'physics',
    'mechanical',
    'field_ops',
    'flowcharts',
]
