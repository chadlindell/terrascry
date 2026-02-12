"""
Pathfinder Diagram Utilities

Shared colors, scales, and drawing functions for assembly diagrams.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection

# Color palette for Pathfinder components
COLORS = {
    'carbon_fiber': '#2D2D2D',      # Dark gray
    'carbon_fiber_light': '#4A4A4A', # Lighter for gradients
    'pvc': '#E8E8E8',               # Light gray
    'pvc_dark': '#C0C0C0',          # Darker for gradients
    'sensor': '#3D6B99',            # Steel blue
    'sensor_light': '#5A8BC2',      # Lighter for gradients
    'harness': '#8B4513',           # Saddle brown
    'harness_light': '#A0522D',     # Sienna
    'metal': '#B8B8C0',             # Silver
    'metal_dark': '#888890',        # Darker silver
    'bungee': '#333333',            # Near black
    'cable': '#1A1A1A',             # Black
    'ground': '#8B7355',            # Earth brown
    'operator': '#666666',          # Gray silhouette
    'annotation': '#333333',        # Dark gray for text
    'dimension': '#E63946',         # Red for dimension lines
    'callout': '#2196F3',           # Blue for callouts
    'background': '#FFFFFF',        # White background
}

# Scale factor: mm to display units
SCALE = 0.01  # 1mm = 0.01 display units (100mm = 1 unit)

# Standard dimensions (mm)
DIMENSIONS = {
    'crossbar_length': 2000,
    'crossbar_od': 25,
    'crossbar_wall': 2,
    'drop_tube_length': 500,
    'drop_tube_od': 21.3,
    'sensor_diameter': 16,
    'sensor_length': 80,
    'sensor_positions': [250, 750, 1250, 1750],
    'center_mount': 1000,
    'bottom_sensor_height': 175,
    'operator_height': 1750,
}


def mm_to_units(mm):
    """Convert millimeters to display units."""
    return mm * SCALE


def draw_cylinder_gradient(ax, x, y, width, height, color_dark, color_light,
                           horizontal=True, zorder=1):
    """
    Draw a 3D-shaded cylinder representation.

    Args:
        ax: Matplotlib axes
        x, y: Bottom-left corner position
        width, height: Dimensions
        color_dark, color_light: Gradient colors
        horizontal: True for horizontal cylinder, False for vertical
        zorder: Draw order

    Returns:
        List of patches added
    """
    patches = []
    n_strips = 10

    if horizontal:
        strip_height = height / n_strips
        for i in range(n_strips):
            # Create gradient from dark at edges to light in center
            t = abs(i - n_strips/2) / (n_strips/2)
            r = int(int(color_light[1:3], 16) * (1-t) + int(color_dark[1:3], 16) * t)
            g = int(int(color_light[3:5], 16) * (1-t) + int(color_dark[3:5], 16) * t)
            b = int(int(color_light[5:7], 16) * (1-t) + int(color_dark[5:7], 16) * t)
            color = f'#{r:02x}{g:02x}{b:02x}'

            rect = plt.Rectangle((x, y + i * strip_height), width, strip_height,
                                 facecolor=color, edgecolor='none', zorder=zorder)
            ax.add_patch(rect)
            patches.append(rect)
    else:
        strip_width = width / n_strips
        for i in range(n_strips):
            t = abs(i - n_strips/2) / (n_strips/2)
            r = int(int(color_light[1:3], 16) * (1-t) + int(color_dark[1:3], 16) * t)
            g = int(int(color_light[3:5], 16) * (1-t) + int(color_dark[3:5], 16) * t)
            b = int(int(color_light[5:7], 16) * (1-t) + int(color_dark[5:7], 16) * t)
            color = f'#{r:02x}{g:02x}{b:02x}'

            rect = plt.Rectangle((x + i * strip_width, y), strip_width, height,
                                 facecolor=color, edgecolor='none', zorder=zorder)
            ax.add_patch(rect)
            patches.append(rect)

    # Add outline
    outline = plt.Rectangle((x, y), width, height,
                            facecolor='none', edgecolor=color_dark,
                            linewidth=0.5, zorder=zorder+1)
    ax.add_patch(outline)
    patches.append(outline)

    return patches


def draw_dimension_line(ax, start, end, text, offset=0.3, fontsize=8,
                        color=None, text_offset=0.15):
    """
    Draw a dimension annotation line with measurement text.

    Args:
        ax: Matplotlib axes
        start: (x, y) start point
        end: (x, y) end point
        text: Dimension text (e.g., "500 mm")
        offset: Distance from object to dimension line
        fontsize: Text size
        color: Line/text color (default: dimension red)
        text_offset: Distance from line to text
    """
    if color is None:
        color = COLORS['dimension']

    x1, y1 = start
    x2, y2 = end

    # Determine if horizontal or vertical
    if abs(x2 - x1) > abs(y2 - y1):
        # Horizontal dimension
        ax.annotate('', xy=(x2, y1 + offset), xytext=(x1, y1 + offset),
                   arrowprops=dict(arrowstyle='<->', color=color, lw=1))
        # Extension lines
        ax.plot([x1, x1], [y1, y1 + offset + 0.05], color=color, lw=0.5)
        ax.plot([x2, x2], [y2, y2 + offset + 0.05], color=color, lw=0.5)
        # Text
        ax.text((x1 + x2)/2, y1 + offset + text_offset, text,
               ha='center', va='bottom', fontsize=fontsize, color=color)
    else:
        # Vertical dimension
        ax.annotate('', xy=(x1 + offset, y2), xytext=(x1 + offset, y1),
                   arrowprops=dict(arrowstyle='<->', color=color, lw=1))
        # Extension lines
        ax.plot([x1, x1 + offset + 0.05], [y1, y1], color=color, lw=0.5)
        ax.plot([x2, x2 + offset + 0.05], [y2, y2], color=color, lw=0.5)
        # Text
        ax.text(x1 + offset + text_offset, (y1 + y2)/2, text,
               ha='left', va='center', fontsize=fontsize, color=color,
               rotation=90)


def draw_circled_number(ax, x, y, number, radius=0.2, fontsize=10,
                        bg_color=None, text_color='white'):
    """
    Draw a callout marker with a circled number.

    Args:
        ax: Matplotlib axes
        x, y: Center position
        number: Number to display
        radius: Circle radius
        fontsize: Text size
        bg_color: Background color (default: callout blue)
        text_color: Text color

    Returns:
        Circle patch
    """
    if bg_color is None:
        bg_color = COLORS['callout']

    circle = Circle((x, y), radius, facecolor=bg_color, edgecolor='white',
                   linewidth=1.5, zorder=100)
    ax.add_patch(circle)

    ax.text(x, y, str(number), ha='center', va='center',
           fontsize=fontsize, fontweight='bold', color=text_color, zorder=101)

    return circle


def draw_callout_line(ax, start, end, color=None):
    """
    Draw a leader line from callout to component.

    Args:
        ax: Matplotlib axes
        start: (x, y) callout position
        end: (x, y) component position
        color: Line color
    """
    if color is None:
        color = COLORS['callout']

    ax.plot([start[0], end[0]], [start[1], end[1]],
           color=color, lw=1, linestyle='-', zorder=99)

    # Small circle at component end
    ax.plot(end[0], end[1], 'o', markersize=4,
           markerfacecolor='white', markeredgecolor=color, zorder=99)


def setup_figure(figsize=(12, 9), title=None):
    """
    Create a figure with standard settings.

    Args:
        figsize: Figure size in inches
        title: Optional title

    Returns:
        fig, ax tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_facecolor(COLORS['background'])

    # Remove axes for cleaner diagrams
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    return fig, ax


def add_legend(ax, items, loc='upper right'):
    """
    Add a component legend to the diagram.

    Args:
        ax: Matplotlib axes
        items: List of (color, label) tuples
        loc: Legend location
    """
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=color, edgecolor='gray', label=label)
               for color, label in items]

    ax.legend(handles=handles, loc=loc, framealpha=0.9, fontsize=9)


def save_figure(fig, filename, dpi=150):
    """
    Save figure with standard settings.

    Args:
        fig: Matplotlib figure
        filename: Output path
        dpi: Resolution
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
