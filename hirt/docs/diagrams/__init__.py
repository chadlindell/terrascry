"""
HIRT Whitepaper - Diagrams Module

Provides matplotlib-based diagram generation functions for use in Quarto documents.
This module re-exports functions from the original pdf-generator library with
modifications for Quarto/matplotlib integration.

Usage in .qmd files:
    ```{python}
    #| label: fig-system-overview
    #| fig-cap: "HIRT system overview"
    from diagrams import subsurface
    subsurface.create_crosshole_comparison_figure()
    plt.show()
    ```
"""

import matplotlib
# Only force Agg backend when not running inside Jupyter/IPython,
# where the inline backend is needed for figure capture.
try:
    get_ipython()
except NameError:
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

# Color palette for consistent styling across all diagrams
COLORS = {
    'primary': '#1a365d',      # Navy - main headings
    'secondary': '#2c5282',    # Blue - subheadings, probes
    'accent': '#3182ce',       # Light blue - RX coils, highlights
    'success': '#38a169',      # Green - TX coils, positive
    'warning': '#c53030',      # Red - warnings, targets
    'light_bg': '#f7fafc',     # Light gray - table backgrounds
    'ground_tan': '#d4a373',   # Tan - ground/soil
    'orange': '#ed8936',       # ERT rings
    'purple': '#805ad5',       # Disturbed soil
    'gray_dark': '#4a5568',    # Dark gray - connectors
    'gray_med': '#718096',     # Medium gray - borders
    'gray_light': '#a0aec0',   # Light gray - inactive
    'sky': '#e8f4f8',          # Sky background
    'ground_dark': '#654321',  # Dark brown - ground surface
    'tx_coil': '#38a169',      # TX coil color
    'rx_coil': '#3182ce',      # RX coil color
    'ert_ring': '#ed8936',     # ERT ring color
    'probe_body': '#2c5282',   # Probe body color
    'connector': '#4a5568',    # Connector color
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


# =============================================================================
# Phase 2 Utility Enhancements: Advanced Rendering Functions
# =============================================================================

# Wong colorblind-safe palette (8 colors)
# Reference: Bang Wong, Nature Methods 8, 441 (2011)
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

# Standardized line weights for consistent diagrams
LINE_WEIGHTS = {
    'fine': 0.5,      # Grid lines, depth markers
    'normal': 1.0,    # Standard borders, secondary elements
    'emphasis': 1.5,  # Primary elements, callouts
    'bold': 2.0,      # Ground surface, major divisions
}

# =============================================================================
# HIRT Diagram Style Standards
# =============================================================================
# Colors: Use COLORS dict for consistent palette, WONG_PALETTE for scientific plots
# Line weights: Use LINE_WEIGHTS dict (fine=0.5, normal=1.0, emphasis=1.5, bold=2.0)
# Font sizes: Titles=11pt, Labels=8pt, Annotations=7pt
# Arrows: Dimension arrows <-> lw=1.0, Callout arrows -> lw=1.5
# =============================================================================


def get_colorblind_cmap(n_colors, palette='wong'):
    """
    Get a list of colorblind-friendly colors.

    Args:
        n_colors: Number of colors needed
        palette: 'wong' for Wong palette

    Returns:
        List of hex color strings
    """
    if palette == 'wong':
        colors = list(WONG_PALETTE.values())[:n_colors]
        # Cycle if more colors needed
        while len(colors) < n_colors:
            colors.extend(list(WONG_PALETTE.values())[:n_colors - len(colors)])
        return colors
    return [COLORS['primary']] * n_colors


def draw_cylinder_gradient(ax, x, y, width, height, base_color, light_angle=45, n_strips=20):
    """
    Draw a 3D-shaded cylinder using vertical gradient strips.

    Args:
        ax: matplotlib Axes object
        x, y: Lower-left corner coordinates
        width: Cylinder width
        height: Cylinder height
        base_color: Base color (hex or name)
        light_angle: Light source angle in degrees (0=right, 90=top)
        n_strips: Number of vertical strips for shading

    Returns:
        List of Rectangle patches
    """
    import matplotlib.colors as mcolors

    # Convert base color to RGB
    rgb = mcolors.to_rgb(base_color)

    patches = []
    strip_width = width / n_strips

    for i in range(n_strips):
        # Position along width (0 to 1)
        t = i / n_strips

        # Cosine shading - brighter in center, darker at edges
        # Shift based on light angle
        angle_shift = (light_angle - 90) / 180
        shade = 0.5 + 0.5 * np.cos(np.pi * (t - 0.5 + angle_shift))

        # Blend between darker (0.4) and lighter (1.2) versions
        factor = 0.4 + 0.8 * shade
        shaded_rgb = tuple(min(1, c * factor) for c in rgb)

        rect = Rectangle((x + i * strip_width, y), strip_width, height,
                         facecolor=shaded_rgb, edgecolor='none')
        ax.add_patch(rect)
        patches.append(rect)

    # Add subtle edge lines
    ax.plot([x, x], [y, y + height], color='black', linewidth=0.5, alpha=0.5)
    ax.plot([x + width, x + width], [y, y + height], color='black', linewidth=0.5, alpha=0.5)

    return patches


def draw_sphere_gradient(ax, x, y, radius, base_color, light_offset=(-0.3, 0.3), n_rings=15):
    """
    Draw a 3D-shaded sphere using concentric circles with highlight.

    Args:
        ax: matplotlib Axes object
        x, y: Center coordinates
        radius: Sphere radius
        base_color: Base color (hex or name)
        light_offset: (dx, dy) offset for highlight position (-1 to 1 range)
        n_rings: Number of concentric rings

    Returns:
        List of Circle patches
    """
    import matplotlib.colors as mcolors

    rgb = mcolors.to_rgb(base_color)
    patches = []

    # Draw from outside in
    for i in range(n_rings, 0, -1):
        t = i / n_rings  # 1 at edge, small at center
        r = radius * t

        # Shade: darker at edges, lighter toward highlight
        shade = 0.5 + 0.5 * (1 - t)
        factor = 0.5 + 0.7 * shade
        shaded_rgb = tuple(min(1, c * factor) for c in rgb)

        circle = Circle((x, y), r, facecolor=shaded_rgb, edgecolor='none')
        ax.add_patch(circle)
        patches.append(circle)

    # Add highlight spot
    highlight_x = x + light_offset[0] * radius * 0.5
    highlight_y = y + light_offset[1] * radius * 0.5
    highlight = Circle((highlight_x, highlight_y), radius * 0.15,
                       facecolor='white', edgecolor='none', alpha=0.6)
    ax.add_patch(highlight)
    patches.append(highlight)

    return patches


def create_metal_gradient(base_color, style='brushed'):
    """
    Create a color mapping function for metal appearance.

    Args:
        base_color: Base metal color
        style: 'brushed', 'polished', or 'matte'

    Returns:
        Function that takes position (0-1) and returns color
    """
    import matplotlib.colors as mcolors

    rgb = mcolors.to_rgb(base_color)

    if style == 'polished':
        # High contrast, sharp highlight
        def color_func(t):
            shine = 0.3 + 0.7 * (0.5 + 0.5 * np.cos(2 * np.pi * t))
            return tuple(min(1, c * shine + 0.3 * shine) for c in rgb)
    elif style == 'brushed':
        # Subtle linear gradient
        def color_func(t):
            shine = 0.6 + 0.4 * np.sin(np.pi * t)
            return tuple(min(1, c * shine) for c in rgb)
    else:  # matte
        def color_func(t):
            return rgb

    return color_func


def draw_metal_surface(ax, x, y, width, height, base_color='#888888',
                       style='brushed', orientation='horizontal'):
    """
    Draw a metal surface with realistic shading.

    Args:
        ax: matplotlib Axes object
        x, y: Lower-left corner
        width, height: Dimensions
        base_color: Metal base color
        style: 'brushed', 'polished', or 'matte'
        orientation: 'horizontal' or 'vertical' for brush direction

    Returns:
        List of Rectangle patches
    """
    color_func = create_metal_gradient(base_color, style)
    patches = []
    n_strips = 25

    if orientation == 'horizontal':
        strip_height = height / n_strips
        for i in range(n_strips):
            t = i / n_strips
            rect = Rectangle((x, y + i * strip_height), width, strip_height,
                             facecolor=color_func(t), edgecolor='none')
            ax.add_patch(rect)
            patches.append(rect)
    else:
        strip_width = width / n_strips
        for i in range(n_strips):
            t = i / n_strips
            rect = Rectangle((x + i * strip_width, y), strip_width, height,
                             facecolor=color_func(t), edgecolor='none')
            ax.add_patch(rect)
            patches.append(rect)

    return patches


def draw_soil_texture_advanced(ax, xmin, xmax, ytop, ybottom, soil_type='mixed',
                               moisture_gradient=True, seed=42):
    """
    Draw realistic soil texture with noise, inclusions, and moisture effects.

    Args:
        ax: matplotlib Axes object
        xmin, xmax: Horizontal extent
        ytop, ybottom: Vertical extent (ytop > ybottom for downward)
        soil_type: 'topsoil', 'sand', 'clay', 'gravel', or 'mixed'
        moisture_gradient: Whether to darken with depth
        seed: Random seed for reproducibility

    Returns:
        Collection of patches
    """
    np.random.seed(seed)

    # Base colors for soil types
    soil_colors = {
        'topsoil': '#5c4033',  # Dark brown
        'sand': '#c2b280',     # Sandy tan
        'clay': '#a67b5b',     # Reddish brown
        'gravel': '#808080',   # Gray
        'mixed': COLORS['ground_tan'],
    }

    base_color = soil_colors.get(soil_type, soil_colors['mixed'])

    # Draw base fill with depth-dependent shading
    n_layers = 20
    layer_height = abs(ytop - ybottom) / n_layers

    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(base_color)

    for i in range(n_layers):
        y_layer = ytop - (i + 1) * layer_height if ytop > ybottom else ybottom + i * layer_height
        y_next = ytop - i * layer_height if ytop > ybottom else ybottom + (i + 1) * layer_height

        # Darken with depth if moisture gradient enabled
        if moisture_gradient:
            depth_factor = 1.0 - 0.3 * (i / n_layers)
        else:
            depth_factor = 1.0

        # Add slight random variation
        noise = np.random.uniform(-0.05, 0.05)
        factor = depth_factor + noise
        layer_color = tuple(max(0, min(1, c * factor)) for c in rgb)

        ax.fill_between([xmin, xmax], [y_layer, y_layer], [y_next, y_next],
                       color=layer_color, edgecolor='none')

    # Add soil-specific features
    patches = []

    if soil_type == 'gravel' or soil_type == 'mixed':
        # Add gravel inclusions
        n_gravel = int(abs(xmax - xmin) * abs(ytop - ybottom) * 3)
        for _ in range(n_gravel):
            gx = np.random.uniform(xmin, xmax)
            gy = np.random.uniform(min(ytop, ybottom), max(ytop, ybottom))
            gr = np.random.uniform(0.02, 0.08)
            gray = np.random.uniform(0.4, 0.7)
            gravel = Circle((gx, gy), gr, facecolor=(gray, gray, gray),
                           edgecolor='none', alpha=0.6)
            ax.add_patch(gravel)
            patches.append(gravel)

    if soil_type == 'sand':
        # Add subtle stippling
        n_dots = int(abs(xmax - xmin) * abs(ytop - ybottom) * 50)
        xs = np.random.uniform(xmin, xmax, n_dots)
        ys = np.random.uniform(min(ytop, ybottom), max(ytop, ybottom), n_dots)
        ax.scatter(xs, ys, s=0.5, c='#a08060', alpha=0.3)

    # Add stratification lines for clay
    if soil_type == 'clay' or soil_type == 'mixed':
        n_lines = int(abs(ytop - ybottom) * 2)
        for i in range(n_lines):
            line_y = np.random.uniform(min(ytop, ybottom), max(ytop, ybottom))
            # Wavy line
            xs = np.linspace(xmin, xmax, 50)
            ys = line_y + 0.02 * np.sin(xs * 10 + np.random.uniform(0, 2*np.pi))
            ax.plot(xs, ys, color='#654321', linewidth=0.3, alpha=0.3)

    return patches


# =============================================================================
# Import specialized diagram modules
# =============================================================================
from . import subsurface
from . import circuits
from . import physics
from . import mechanical
from . import field_ops
from . import flowcharts
from . import executive_summary
from . import architecture
from . import testing
from . import calibration
from . import assembly_3d
from . import sensor_modalities
from . import uncertainty


# Export all colors and functions
__all__ = [
    # Core colors and palettes
    'COLORS',
    'WONG_PALETTE',
    'LINE_WEIGHTS',
    'get_colorblind_cmap',
    # Figure utilities
    'save_figure_to_buffer',
    'create_figure',
    'setup_axes_clean',
    'setup_axes_scientific',
    'add_title',
    'add_legend',
    # Ground/environment drawing
    'draw_ground_surface',
    'draw_ground_layers',
    'draw_sky',
    'draw_depth_markers',
    'draw_soil_texture_advanced',
    # Shape drawing
    'draw_arrow',
    'draw_dimension_line',
    'draw_box',
    'draw_circle',
    'draw_ellipse',
    # 3D shading utilities
    'draw_cylinder_gradient',
    'draw_sphere_gradient',
    'create_metal_gradient',
    'draw_metal_surface',
    # Diagram modules
    'subsurface',
    'circuits',
    'physics',
    'mechanical',
    'field_ops',
    'flowcharts',
    'executive_summary',
    'architecture',
    'testing',
    'calibration',
    'assembly_3d',
    'sensor_modalities',
    'uncertainty',
]
