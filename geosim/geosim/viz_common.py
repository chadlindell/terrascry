"""
Shared visualization utilities for HIRT and Pathfinder diagram generators.

Provides common color palettes, soil texture rendering, subsurface cross-section
layouts, and matplotlib style helpers used across both projects' documentation.

Both HIRT (docs/diagrams/) and Pathfinder (docs/diagrams/) define their own
specialized drawing functions. This module extracts the *shared* patterns:
consistent soil colors, instrument palettes, cross-section primitives, and
plan-view helpers so that documentation diagrams across both projects look
unified without duplicating low-level rendering code.

Usage from either project's diagram module::

    from geosim.viz_common import (
        SOIL_COLORS, INSTRUMENT_COLORS, setup_figure,
        draw_soil_layers, draw_buried_target, draw_probe,
    )
"""

import matplotlib
try:
    get_ipython()  # type: ignore[name-defined]
except NameError:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import numpy as np
from io import BytesIO
from typing import Optional


# ============================================================================
# COLOR PALETTES
# ============================================================================

# Soil layer colors (consistent across both projects)
SOIL_COLORS = {
    'topsoil': '#8B7355',
    'clay': '#CD853F',
    'sand': '#F4E4C1',
    'gravel': '#A9A9A9',
    'bedrock': '#696969',
    'water_table': '#4169E1',
    'disturbed': '#D2691E',      # For grave shafts, backfill
    'void': '#FFFFFF',
}

# Instrument colors
INSTRUMENT_COLORS = {
    'pathfinder': '#2196F3',     # Blue - screening
    'hirt_mit': '#4CAF50',       # Green - electromagnetic
    'hirt_ert': '#FF9800',       # Orange - resistivity
    'combined': '#9C27B0',       # Purple - joint interpretation
    'anomaly': '#F44336',        # Red - detected anomaly
    'background': '#607D8B',     # Gray - background/reference
}

# Anomaly strength colormap (for heatmaps)
ANOMALY_CMAP = LinearSegmentedColormap.from_list(
    'anomaly',
    ['#2196F3', '#4CAF50', '#FFEB3B', '#FF9800', '#F44336'],
    N=256
)

# Confidence level colors
CONFIDENCE_COLORS = {
    'high': '#4CAF50',
    'medium': '#FF9800',
    'low': '#F44336',
}

# Wong colorblind-safe palette (shared with HIRT diagrams/__init__.py)
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


# ============================================================================
# MATPLOTLIB RCPARAMS (Scientific Publication Style)
# ============================================================================

SCIENTIFIC_RCPARAMS = {
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
}


# ============================================================================
# STYLE HELPERS
# ============================================================================

def apply_scientific_style() -> None:
    """Apply the shared scientific publication rcParams to matplotlib."""
    plt.rcParams.update(SCIENTIFIC_RCPARAMS)


def setup_figure(width: float = 10, height: float = 6, dpi: int = 150,
                 style: str = 'scientific') -> tuple[plt.Figure, plt.Axes]:
    """Create a consistently styled figure.

    Parameters
    ----------
    width : float
        Figure width in inches.
    height : float
        Figure height in inches.
    dpi : int
        Resolution in dots per inch.
    style : str
        One of ``'scientific'`` (grid, visible spines) or ``'clean'``
        (no axes, white background -- suited to schematic diagrams).

    Returns
    -------
    tuple[Figure, Axes]
        The created figure and axes.
    """
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')

    if style == 'clean':
        ax.axis('off')
    elif style == 'scientific':
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.grid(True, linestyle=':', alpha=0.4)
        ax.yaxis.grid(True, linestyle=':', alpha=0.4)
        ax.set_axisbelow(True)

    return fig, ax


def figure_to_buffer(fig: plt.Figure, dpi: int = 150,
                     fmt: str = 'png') -> BytesIO:
    """Save figure to BytesIO buffer (for Quarto embedding).

    Parameters
    ----------
    fig : Figure
        The matplotlib figure to serialize.
    dpi : int
        Output resolution.
    fmt : str
        Image format (``'png'``, ``'svg'``, ``'pdf'``).

    Returns
    -------
    BytesIO
        Seeked-to-zero buffer containing the rendered image.
    """
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches='tight', dpi=dpi,
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def add_depth_scale(ax: plt.Axes, max_depth: float, side: str = 'left',
                    num_ticks: int = 5, units: str = 'm') -> None:
    """Add a depth scale bar to a cross-section plot.

    Draws a vertical ruler along the specified side of the axes showing
    depth below surface (positive downward).  Tick marks are evenly spaced
    from 0 to *max_depth*.

    Parameters
    ----------
    ax : Axes
        Target axes (should already have its y-limits set so that the
        surface is at y=0 and depth increases downward).
    max_depth : float
        Maximum depth value to label (in *units*).
    side : str
        ``'left'`` or ``'right'`` -- which side of the axes to place the scale.
    num_ticks : int
        Number of labelled tick marks (including 0 and max_depth).
    units : str
        Unit string appended to the deepest label (e.g. ``'m'``).
    """
    xlim = ax.get_xlim()
    x_pos = xlim[0] - 0.03 * (xlim[1] - xlim[0]) if side == 'left' \
        else xlim[1] + 0.03 * (xlim[1] - xlim[0])

    depths = np.linspace(0, max_depth, num_ticks)
    bar_x = x_pos

    # Vertical bar
    ax.plot([bar_x, bar_x], [0, -max_depth], color='#333333', linewidth=1.5,
            clip_on=False, zorder=50)

    for d in depths:
        tick_len = 0.015 * (xlim[1] - xlim[0])
        if side == 'left':
            ax.plot([bar_x - tick_len, bar_x], [-d, -d], color='#333333',
                    linewidth=1.0, clip_on=False, zorder=50)
            label = f'{d:.1f}' if d > 0 else '0'
            if d == max_depth:
                label += f' {units}'
            ax.text(bar_x - tick_len * 1.5, -d, label, ha='right',
                    va='center', fontsize=7, color='#333333', clip_on=False)
        else:
            ax.plot([bar_x, bar_x + tick_len], [-d, -d], color='#333333',
                    linewidth=1.0, clip_on=False, zorder=50)
            label = f'{d:.1f}' if d > 0 else '0'
            if d == max_depth:
                label += f' {units}'
            ax.text(bar_x + tick_len * 1.5, -d, label, ha='left',
                    va='center', fontsize=7, color='#333333', clip_on=False)

    # "Depth" label rotated vertically
    mid_depth = -max_depth / 2
    offset = -0.07 * (xlim[1] - xlim[0]) if side == 'left' \
        else 0.07 * (xlim[1] - xlim[0])
    ax.text(bar_x + offset, mid_depth, 'Depth', ha='center', va='center',
            fontsize=8, rotation=90, color='#333333', clip_on=False)


def add_north_arrow(ax: plt.Axes, x: float, y: float,
                    size: float = 0.05) -> None:
    """Add a north arrow to a plan view.

    Parameters
    ----------
    ax : Axes
        Target axes (should be a plan / map view).
    x, y : float
        Position of the arrow base in axes-fraction coordinates (0-1).
    size : float
        Relative size of the arrow in axes-fraction units.
    """
    # Work in axes fraction coordinates via ax.transAxes
    arrow_length = size
    head_width = size * 0.4
    head_length = size * 0.3

    # Shaft
    ax.annotate(
        '', xy=(x, y + arrow_length), xytext=(x, y),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                        mutation_scale=15),
    )

    # "N" label
    ax.text(x, y + arrow_length + 0.02, 'N', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color='black')

    # Small crossbar for east-west
    bar_half = size * 0.25
    mid_y = y + arrow_length * 0.35
    ax.plot([x - bar_half, x + bar_half], [mid_y, mid_y],
            transform=ax.transAxes, color='black', linewidth=1.0,
            clip_on=False)


# ============================================================================
# SUBSURFACE CROSS-SECTION HELPERS
# ============================================================================

def draw_soil_layers(ax: plt.Axes, layers: list[dict],
                     x_range: tuple[float, float] = (0, 10)) -> list[mpatches.Rectangle]:
    """Draw horizontal soil layers on a cross-section.

    Parameters
    ----------
    ax : Axes
        Target axes.  The convention is y=0 at the surface with depth
        increasing *downward* (negative y values).
    layers : list[dict]
        Each dict must contain:
        - ``'name'``: str -- soil type key (looked up in :data:`SOIL_COLORS`)
        - ``'top_m'``: float -- depth to top of layer (positive number, metres)
        - ``'bottom_m'``: float -- depth to bottom of layer (positive number)
        Optionally:
        - ``'color'``: str -- override colour
    x_range : tuple[float, float]
        Horizontal extent of the cross-section.

    Returns
    -------
    list[Rectangle]
        The patch objects added to *ax*.
    """
    patches: list[mpatches.Rectangle] = []
    for layer in layers:
        color = layer.get('color', SOIL_COLORS.get(layer['name'], '#CCCCCC'))
        top = -layer['top_m']       # convert depth to y coordinate
        bottom = -layer['bottom_m']
        height = bottom - top       # negative (downward extent)
        rect = mpatches.Rectangle(
            (x_range[0], top),
            x_range[1] - x_range[0],
            height,
            facecolor=color, edgecolor='none', alpha=0.6,
        )
        ax.add_patch(rect)
        patches.append(rect)

        # Label centred vertically within the layer
        label_y = (top + bottom) / 2
        ax.text(
            x_range[0] + 0.2, label_y,
            layer['name'].replace('_', ' ').title(),
            fontsize=8, va='center', alpha=0.7,
        )

    return patches


def draw_soil_texture(ax: plt.Axes, x_range: tuple[float, float],
                      y_top: float, y_bottom: float,
                      soil_type: str = 'mixed', seed: int = 42) -> None:
    """Add realistic texture (stippling, stratification, gravel) to a soil region.

    This is a lightweight version of HIRT's ``draw_soil_texture_advanced``
    intended for shared cross-section diagrams.

    Parameters
    ----------
    ax : Axes
        Target axes.
    x_range : tuple[float, float]
        Horizontal extent.
    y_top, y_bottom : float
        Vertical extent (y_top > y_bottom for standard orientation).
    soil_type : str
        One of ``'topsoil'``, ``'sand'``, ``'clay'``, ``'gravel'``, ``'mixed'``.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    xmin, xmax = x_range
    height = abs(y_top - y_bottom)
    width = abs(xmax - xmin)

    if soil_type in ('gravel', 'mixed'):
        n_gravel = int(width * height * 3)
        for _ in range(n_gravel):
            gx = rng.uniform(xmin, xmax)
            gy = rng.uniform(min(y_top, y_bottom), max(y_top, y_bottom))
            gr = rng.uniform(0.02, 0.06)
            gray = rng.uniform(0.4, 0.7)
            stone = plt.Circle((gx, gy), gr, facecolor=(gray, gray, gray),
                               edgecolor='none', alpha=0.5)
            ax.add_patch(stone)

    if soil_type == 'sand':
        n_dots = int(width * height * 40)
        xs = rng.uniform(xmin, xmax, n_dots)
        ys = rng.uniform(min(y_top, y_bottom), max(y_top, y_bottom), n_dots)
        ax.scatter(xs, ys, s=0.3, c='#a08060', alpha=0.3, zorder=2)

    if soil_type in ('clay', 'mixed'):
        n_lines = max(1, int(height * 2))
        for _ in range(n_lines):
            line_y = rng.uniform(min(y_top, y_bottom), max(y_top, y_bottom))
            xs = np.linspace(xmin, xmax, 50)
            ys = line_y + 0.02 * np.sin(xs * 10 + rng.uniform(0, 2 * np.pi))
            ax.plot(xs, ys, color='#654321', linewidth=0.3, alpha=0.3, zorder=2)


def draw_water_table(ax: plt.Axes, depth: float,
                     x_range: tuple[float, float] = (0, 10),
                     label: bool = True) -> None:
    """Draw a water table line on a cross-section.

    Parameters
    ----------
    ax : Axes
        Target axes.
    depth : float
        Depth of water table in metres (positive number).
    x_range : tuple[float, float]
        Horizontal extent.
    label : bool
        Whether to add a text label.
    """
    y = -depth
    color = SOIL_COLORS['water_table']
    ax.plot(x_range, [y, y], color=color, linewidth=1.5, linestyle='--',
            alpha=0.8, zorder=10)

    # Blue shading below water table (subtle)
    ax.fill_between(x_range, [y, y], [ax.get_ylim()[0]] * 2,
                    color=color, alpha=0.05, zorder=1)

    if label:
        ax.text(x_range[1] - 0.1, y + 0.1, 'Water Table',
                ha='right', va='bottom', fontsize=7, color=color,
                fontweight='bold', fontstyle='italic')


def draw_buried_target(ax: plt.Axes, x: float, depth: float,
                       width: float = 0.5, height: float = 0.3,
                       label: str = '', color: str = '#F44336',
                       style: str = 'box') -> mpatches.Patch:
    """Draw a buried target object on a cross-section.

    Parameters
    ----------
    ax : Axes
        Target axes.
    x : float
        Horizontal centre position.
    depth : float
        Centre depth in metres (positive number).
    width, height : float
        Target dimensions in metres.
    label : str
        Optional label above the target.
    color : str
        Fill colour.
    style : str
        ``'box'`` for a rounded rectangle, ``'circle'`` for a sphere-like target.

    Returns
    -------
    Patch
        The patch object added to *ax*.
    """
    y = -depth

    if style == 'circle':
        radius = max(width, height) / 2
        patch = plt.Circle((x, y), radius, facecolor=color,
                           edgecolor='black', linewidth=1.5, zorder=20)
    else:
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.05", facecolor=color,
            edgecolor='black', linewidth=1.5, zorder=20,
        )
    ax.add_patch(patch)

    if label:
        ax.text(x, y + height / 2 + 0.15, label, ha='center',
                fontsize=8, fontweight='bold', zorder=21)

    return patch


def draw_probe(ax: plt.Axes, x: float, depth: float,
               label: str = '', instrument: str = 'hirt_mit',
               show_sensors: bool = True) -> None:
    """Draw a probe/sensor on a cross-section.

    Renders a vertical rod from the surface down to *depth* with optional
    sensor markers and a surface-level indicator triangle.

    Parameters
    ----------
    ax : Axes
        Target axes.
    x : float
        Horizontal position.
    depth : float
        Maximum depth of the probe tip in metres (positive number).
    label : str
        Optional label above the probe.
    instrument : str
        Key into :data:`INSTRUMENT_COLORS` (determines colour).
    show_sensors : bool
        If True, draw small circles along the rod to indicate sensor
        positions (evenly spaced).
    """
    color = INSTRUMENT_COLORS.get(instrument, '#333333')
    y_surface = 0
    y_tip = -depth

    # Probe rod
    ax.plot([x, x], [y_surface, y_tip], color=color, linewidth=2,
            solid_capstyle='round', zorder=15)

    # Surface indicator (downward-pointing triangle)
    ax.plot(x, y_surface, 'v', color=color, markersize=8, zorder=16)

    # Tip marker
    ax.plot(x, y_tip, 'o', color=color, markersize=6, zorder=16)

    # Sensor markers along the rod
    if show_sensors and depth > 0.3:
        n_sensors = max(2, int(depth / 0.25))
        sensor_depths = np.linspace(0.1, depth - 0.05, n_sensors)
        for sd in sensor_depths:
            ax.plot(x, -sd, 's', color='white', markersize=4,
                    markeredgecolor=color, markeredgewidth=1.0, zorder=17)

    if label:
        ax.text(x, 0.15, label, ha='center', fontsize=7, color=color,
                fontweight='bold', zorder=18)


def draw_anomaly_zone(ax: plt.Axes, x: float, depth: float,
                      radius: float = 0.5,
                      confidence: str = 'high') -> mpatches.Ellipse:
    """Draw a semi-transparent anomaly zone (ellipse) on a cross-section.

    Parameters
    ----------
    ax : Axes
        Target axes.
    x : float
        Horizontal centre.
    depth : float
        Centre depth in metres.
    radius : float
        Approximate radius of the anomaly zone.
    confidence : str
        ``'high'``, ``'medium'``, or ``'low'`` -- selects colour from
        :data:`CONFIDENCE_COLORS`.

    Returns
    -------
    Ellipse
        The patch object added to *ax*.
    """
    color = CONFIDENCE_COLORS.get(confidence, '#FF9800')
    y = -depth
    ellipse = mpatches.Ellipse(
        (x, y), width=radius * 2, height=radius * 1.4,
        facecolor=color, edgecolor=color, linewidth=1.5,
        alpha=0.2, linestyle='--', zorder=5,
    )
    ax.add_patch(ellipse)
    return ellipse


# ============================================================================
# PLAN VIEW HELPERS
# ============================================================================

def draw_survey_grid(ax: plt.Axes, origin: tuple[float, float],
                     nx: int, ny: int, spacing: float,
                     color: str = '#999999',
                     show_labels: bool = False) -> None:
    """Draw a survey grid on a plan view.

    Parameters
    ----------
    ax : Axes
        Target axes.
    origin : tuple[float, float]
        (x, y) of the grid origin (lower-left corner).
    nx, ny : int
        Number of grid points in the x and y directions.
    spacing : float
        Distance between grid points (metres).
    color : str
        Marker colour.
    show_labels : bool
        If True, label every other grid intersection with its coordinates.
    """
    for i in range(nx):
        for j in range(ny):
            x = origin[0] + i * spacing
            y = origin[1] + j * spacing
            ax.plot(x, y, '+', color=color, markersize=8,
                    markeredgewidth=1, zorder=5)

            if show_labels and i % 2 == 0 and j % 2 == 0:
                ax.text(x + spacing * 0.1, y + spacing * 0.1,
                        f'({x:.0f},{y:.0f})', fontsize=5,
                        color=color, alpha=0.7)

    # Draw light grid lines connecting points
    for i in range(nx):
        x = origin[0] + i * spacing
        y0 = origin[1]
        y1 = origin[1] + (ny - 1) * spacing
        ax.plot([x, x], [y0, y1], color=color, linewidth=0.3,
                alpha=0.3, zorder=3)

    for j in range(ny):
        y = origin[1] + j * spacing
        x0 = origin[0]
        x1 = origin[0] + (nx - 1) * spacing
        ax.plot([x0, x1], [y, y], color=color, linewidth=0.3,
                alpha=0.3, zorder=3)


def draw_anomaly_heatmap(ax: plt.Axes, x: np.ndarray, y: np.ndarray,
                         values: np.ndarray,
                         cmap: Optional[LinearSegmentedColormap] = None,
                         size: float = 50,
                         add_colorbar: bool = True,
                         label: str = '') -> plt.matplotlib.collections.PathCollection:
    """Draw an interpolated anomaly heatmap on plan view.

    Parameters
    ----------
    ax : Axes
        Target axes.
    x, y : ndarray
        1-D arrays of measurement positions.
    values : ndarray
        1-D array of anomaly strength values.
    cmap : LinearSegmentedColormap or None
        Colourmap to use; defaults to :data:`ANOMALY_CMAP`.
    size : float
        Marker size for the scatter points.
    add_colorbar : bool
        Whether to append a colour bar to the axes.
    label : str
        Colour bar label text.

    Returns
    -------
    PathCollection
        The scatter object (useful for further customisation).
    """
    if cmap is None:
        cmap = ANOMALY_CMAP
    scatter = ax.scatter(x, y, c=values, cmap=cmap, s=size,
                         edgecolors='black', linewidth=0.5, zorder=10)
    if add_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        if label:
            cbar.set_label(label, fontsize=8)
    return scatter


def draw_walk_path(ax: plt.Axes, waypoints: np.ndarray,
                   color: str = '#2196F3', linewidth: float = 1.5,
                   show_direction: bool = True) -> None:
    """Draw a survey walk path with direction arrows.

    Parameters
    ----------
    ax : Axes
        Target axes.
    waypoints : ndarray
        Shape ``(N, 2)`` array of (x, y) positions in metres.
    color : str
        Path colour.
    linewidth : float
        Line width.
    show_direction : bool
        If True, add small arrowheads at intervals along the path.
    """
    if len(waypoints) < 2:
        return

    ax.plot(waypoints[:, 0], waypoints[:, 1], color=color,
            linewidth=linewidth, zorder=8, solid_capstyle='round')

    # Start / end markers
    ax.plot(waypoints[0, 0], waypoints[0, 1], 'o', color=color,
            markersize=7, zorder=9)
    ax.plot(waypoints[-1, 0], waypoints[-1, 1], 's', color=color,
            markersize=7, zorder=9)

    if show_direction and len(waypoints) > 4:
        # Place direction arrows at ~25% intervals
        step = max(1, len(waypoints) // 4)
        for idx in range(step, len(waypoints) - 1, step):
            dx = waypoints[idx, 0] - waypoints[idx - 1, 0]
            dy = waypoints[idx, 1] - waypoints[idx - 1, 1]
            ax.annotate(
                '', xy=(waypoints[idx, 0], waypoints[idx, 1]),
                xytext=(waypoints[idx, 0] - dx * 0.3,
                        waypoints[idx, 1] - dy * 0.3),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth),
                zorder=9,
            )


# ============================================================================
# COMPOSITE / CONVENIENCE FUNCTIONS
# ============================================================================

def create_cross_section(layers: list[dict],
                         targets: Optional[list[dict]] = None,
                         probes: Optional[list[dict]] = None,
                         water_table_depth: Optional[float] = None,
                         x_range: tuple[float, float] = (0, 10),
                         max_depth: float = 3.0,
                         title: str = '',
                         width: float = 12,
                         height: float = 6) -> tuple[plt.Figure, plt.Axes]:
    """Create a complete subsurface cross-section diagram.

    This is a high-level convenience function that composes soil layers,
    buried targets, probes, and a water table into a single figure.

    Parameters
    ----------
    layers : list[dict]
        Passed to :func:`draw_soil_layers`.
    targets : list[dict] or None
        Each dict should have keys ``'x'``, ``'depth'`` and optionally
        ``'width'``, ``'height'``, ``'label'``, ``'color'``, ``'style'``.
    probes : list[dict] or None
        Each dict should have ``'x'``, ``'depth'`` and optionally
        ``'label'``, ``'instrument'``.
    water_table_depth : float or None
        If provided, draws a water table at this depth.
    x_range : tuple[float, float]
        Horizontal extent.
    max_depth : float
        Maximum depth shown.
    title : str
        Figure title.
    width, height : float
        Figure dimensions in inches.

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = setup_figure(width=width, height=height, style='scientific')

    # Draw soil layers
    draw_soil_layers(ax, layers, x_range=x_range)

    # Water table
    if water_table_depth is not None:
        draw_water_table(ax, water_table_depth, x_range=x_range)

    # Targets
    if targets:
        for t in targets:
            draw_buried_target(
                ax, t['x'], t['depth'],
                width=t.get('width', 0.5),
                height=t.get('height', 0.3),
                label=t.get('label', ''),
                color=t.get('color', INSTRUMENT_COLORS['anomaly']),
                style=t.get('style', 'box'),
            )

    # Probes
    if probes:
        for p in probes:
            draw_probe(
                ax, p['x'], p['depth'],
                label=p.get('label', ''),
                instrument=p.get('instrument', 'hirt_mit'),
            )

    # Surface line
    ax.axhline(0, color='#654321', linewidth=2.5, zorder=25)
    ax.text(x_range[0] + 0.1, 0.12, 'Ground Surface', fontsize=7,
            color='#654321', fontweight='bold')

    # Axis labels and limits
    ax.set_xlim(x_range[0] - 0.5, x_range[1] + 0.5)
    ax.set_ylim(-max_depth - 0.3, max_depth * 0.25)
    ax.set_xlabel('Distance (m)', fontsize=9)
    ax.set_ylabel('Elevation (m)', fontsize=9)

    # Depth scale
    add_depth_scale(ax, max_depth, side='right')

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12)

    fig.tight_layout()
    return fig, ax


def create_plan_view(grid_origin: tuple[float, float] = (0, 0),
                     grid_nx: int = 10, grid_ny: int = 10,
                     grid_spacing: float = 1.0,
                     title: str = '',
                     width: float = 8,
                     height: float = 8) -> tuple[plt.Figure, plt.Axes]:
    """Create a plan-view figure with a survey grid and north arrow.

    Parameters
    ----------
    grid_origin : tuple[float, float]
        Lower-left corner of the survey grid.
    grid_nx, grid_ny : int
        Number of grid points in x and y.
    grid_spacing : float
        Spacing between grid points (metres).
    title : str
        Figure title.
    width, height : float
        Figure dimensions in inches.

    Returns
    -------
    tuple[Figure, Axes]
    """
    fig, ax = setup_figure(width=width, height=height, style='scientific')
    ax.set_aspect('equal')

    draw_survey_grid(ax, grid_origin, grid_nx, grid_ny, grid_spacing)
    add_north_arrow(ax, x=0.95, y=0.85, size=0.06)

    ax.set_xlabel('Easting (m)', fontsize=9)
    ax.set_ylabel('Northing (m)', fontsize=9)

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12)

    fig.tight_layout()
    return fig, ax
