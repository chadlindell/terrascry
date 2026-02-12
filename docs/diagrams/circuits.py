"""
HIRT PDF Generator - Circuit Diagrams Module

Functions for creating schematics, block diagrams, and signal flow diagrams.
Includes IEEE-standard component symbols, signal chain waveform insets,
Manhattan routing, and noise figure cascade annotations.
"""

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch,
    Arc, Wedge, PathPatch
)
import matplotlib.path as mpath
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from io import BytesIO

# Import colorblind-friendly palette from module __init__
try:
    from . import WONG_PALETTE
except ImportError:
    # Fallback if imported directly
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


# =============================================================================
# Signal Waveform Inset Utilities
# =============================================================================

def add_waveform_inset(ax, position, waveform_type='sine', amplitude=1.0,
                       frequency=1.0, noise_level=0.0, distortion=0.0,
                       label=None, color=None, width=0.08, height=0.12):
    """
    Add a small inset axes showing a signal waveform at a specific position.

    Args:
        ax: Parent matplotlib Axes object
        position: (x, y) position in data coordinates for inset center
        waveform_type: 'sine', 'square', 'triangle', 'noise', 'filtered'
        amplitude: Signal amplitude (0-1 scale for display)
        frequency: Number of cycles to show
        noise_level: Amount of noise to add (0-1)
        distortion: Amount of harmonic distortion (0-1)
        label: Optional label text below inset
        color: Waveform color (default: uses WONG_PALETTE)
        width: Inset width in axes fraction
        height: Inset height in axes fraction

    Returns:
        The inset Axes object
    """
    if color is None:
        color = WONG_PALETTE['blue']

    # Convert data coords to axes coords
    x_data, y_data = position
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_ax = (x_data - xlim[0]) / (xlim[1] - xlim[0])
    y_ax = (y_data - ylim[0]) / (ylim[1] - ylim[0])

    # Create inset axes
    inset = ax.inset_axes([x_ax - width/2, y_ax - height/2, width, height],
                          transform=ax.transAxes)

    # Generate waveform
    t = np.linspace(0, 2 * np.pi * frequency, 200)

    if waveform_type == 'sine':
        y = amplitude * np.sin(t)
        if distortion > 0:
            # Add odd harmonics for distortion
            y += distortion * amplitude * 0.3 * np.sin(3 * t)
            y += distortion * amplitude * 0.1 * np.sin(5 * t)
    elif waveform_type == 'square':
        y = amplitude * np.sign(np.sin(t))
    elif waveform_type == 'triangle':
        y = amplitude * (2 * np.abs(2 * (t / (2*np.pi) - np.floor(t / (2*np.pi) + 0.5))) - 1)
    elif waveform_type == 'noise':
        y = amplitude * np.random.randn(len(t))
    elif waveform_type == 'filtered':
        # Low-pass filtered signal appearance
        y = amplitude * np.sin(t) * np.exp(-0.1 * t)
    else:
        y = amplitude * np.sin(t)

    # Add noise
    if noise_level > 0:
        y += noise_level * amplitude * np.random.randn(len(t))

    # Plot waveform
    inset.plot(t, y, color=color, linewidth=0.8)
    inset.axhline(0, color='gray', linewidth=0.3, alpha=0.5)

    # Style inset
    inset.set_xlim(t[0], t[-1])
    inset.set_ylim(-1.5 * amplitude, 1.5 * amplitude)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color(COLORS['gray_med'])

    # Add background
    inset.set_facecolor('white')

    # Add label if provided
    if label:
        ax.text(x_data, y_data - 0.25, label, ha='center', va='top',
                fontsize=5, color=COLORS['gray_dark'])

    return inset


def add_signal_quality_annotation(ax, position, snr_db=None, thd_percent=None,
                                   quality='good'):
    """
    Add signal quality annotation near a waveform inset.

    Args:
        ax: matplotlib Axes object
        position: (x, y) position in data coordinates
        snr_db: Signal-to-noise ratio in dB
        thd_percent: Total harmonic distortion percentage
        quality: 'good', 'fair', or 'poor' for color coding
    """
    quality_colors = {
        'good': WONG_PALETTE['bluish_green'],
        'fair': WONG_PALETTE['orange'],
        'poor': WONG_PALETTE['vermillion'],
    }
    color = quality_colors.get(quality, COLORS['gray_dark'])

    text_parts = []
    if snr_db is not None:
        text_parts.append(f'SNR: {snr_db} dB')
    if thd_percent is not None:
        text_parts.append(f'THD: {thd_percent}%')

    if text_parts:
        ax.text(position[0], position[1], '\n'.join(text_parts),
                ha='center', va='top', fontsize=5, color=color,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                         edgecolor=color, alpha=0.9, linewidth=0.5))


# =============================================================================
# IEEE-Standard Component Symbols
# =============================================================================

def draw_resistor_ieee(ax, x, y, width=0.6, height=0.15, vertical=False,
                       n_zigs=6, linewidth=1.2, color='black'):
    """
    Draw an IEEE-standard resistor symbol with proper zigzag pattern.

    The IEEE standard specifies a rectangular zigzag with 6 peaks,
    equal amplitude on both sides of centerline.

    Args:
        ax: matplotlib Axes object
        x, y: Center position of resistor
        width: Total width of resistor symbol
        height: Peak-to-peak height of zigzag
        vertical: If True, orient vertically
        n_zigs: Number of complete zigzag cycles (IEEE standard is 6)
        linewidth: Line width
        color: Line color
    """
    # Calculate zigzag points
    # IEEE standard: straight lead-in, then zigzag, then straight lead-out
    lead_fraction = 0.1  # 10% of width for leads on each end
    zigzag_width = width * (1 - 2 * lead_fraction)

    points = []

    # Left lead
    if vertical:
        points.append((x, y - width/2))
        points.append((x, y - width/2 + width * lead_fraction))
    else:
        points.append((x - width/2, y))
        points.append((x - width/2 + width * lead_fraction, y))

    # Zigzag section: 2 * n_zigs + 1 points
    n_points = 2 * n_zigs + 1
    for i in range(n_points):
        frac = i / (n_points - 1)
        if vertical:
            py = y - width/2 + width * lead_fraction + zigzag_width * frac
            px = x + height/2 * (1 if i % 2 == 1 else -1 if i % 2 == 0 and 0 < i < n_points-1 else 0)
        else:
            px = x - width/2 + width * lead_fraction + zigzag_width * frac
            py = y + height/2 * (1 if i % 2 == 1 else -1 if i % 2 == 0 and 0 < i < n_points-1 else 0)

        # First and last points should be on centerline
        if i == 0 or i == n_points - 1:
            if vertical:
                px = x
            else:
                py = y
        points.append((px, py))

    # Right lead
    if vertical:
        points.append((x, y + width/2 - width * lead_fraction))
        points.append((x, y + width/2))
    else:
        points.append((x + width/2 - width * lead_fraction, y))
        points.append((x + width/2, y))

    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=linewidth, solid_capstyle='round')


def draw_capacitor_ieee(ax, x, y, width=0.25, gap=0.06, plate_style='parallel',
                        linewidth=1.5, color='black'):
    """
    Draw an IEEE-standard capacitor symbol.

    Args:
        ax: matplotlib Axes object
        x, y: Center position
        width: Height of capacitor plates
        gap: Gap between plates
        plate_style: 'parallel' for non-polarized, 'curved' for electrolytic
        linewidth: Line width
        color: Line color
    """
    if plate_style == 'parallel':
        # Two parallel straight lines (non-polarized capacitor)
        ax.plot([x - gap/2, x - gap/2], [y - width/2, y + width/2],
                color=color, linewidth=linewidth, solid_capstyle='butt')
        ax.plot([x + gap/2, x + gap/2], [y - width/2, y + width/2],
                color=color, linewidth=linewidth, solid_capstyle='butt')
    elif plate_style == 'curved':
        # One straight, one curved (electrolytic/polarized)
        ax.plot([x - gap/2, x - gap/2], [y - width/2, y + width/2],
                color=color, linewidth=linewidth, solid_capstyle='butt')
        # Curved plate using arc
        arc = Arc((x + gap/2 + 0.03, y), 0.08, width,
                  angle=0, theta1=90, theta2=270, color=color, linewidth=linewidth)
        ax.add_patch(arc)


def draw_inductor_ieee(ax, x, y, width=0.6, height=0.12, n_loops=4,
                       linewidth=1.2, color='black', core=None):
    """
    Draw an IEEE-standard inductor symbol with proper semicircular arcs.

    Args:
        ax: matplotlib Axes object
        x, y: Center position
        width: Total width of inductor
        height: Height of arcs
        n_loops: Number of loops/arcs
        linewidth: Line width
        color: Line color
        core: None, 'iron', or 'ferrite' for core type indicator
    """
    loop_width = width / n_loops

    # Draw semicircular arcs
    for i in range(n_loops):
        cx = x - width/2 + loop_width/2 + i * loop_width
        arc = Arc((cx, y), loop_width * 0.9, height * 2,
                  angle=0, theta1=0, theta2=180, color=color, linewidth=linewidth)
        ax.add_patch(arc)

    # Add core indicator if specified
    if core == 'iron':
        # Two parallel lines above the coils
        ax.plot([x - width/2, x + width/2], [y + height + 0.03, y + height + 0.03],
                color=color, linewidth=linewidth * 0.8)
        ax.plot([x - width/2, x + width/2], [y + height + 0.06, y + height + 0.06],
                color=color, linewidth=linewidth * 0.8)
    elif core == 'ferrite':
        # Dashed line above the coils
        ax.plot([x - width/2, x + width/2], [y + height + 0.04, y + height + 0.04],
                color=color, linewidth=linewidth * 0.8, linestyle='--')


def draw_ground_ieee(ax, x, y, size=0.15, style='earth'):
    """
    Draw an IEEE-standard ground symbol.

    Args:
        ax: matplotlib Axes object
        x, y: Position (top of ground symbol)
        size: Overall size
        style: 'earth' (3 lines), 'chassis' (triangle), or 'signal' (single line)
    """
    if style == 'earth':
        # Three horizontal lines of decreasing width
        ax.plot([x, x], [y, y - size/3], 'k-', linewidth=1.5)
        for i, width_frac in enumerate([1.0, 0.65, 0.3]):
            line_y = y - size/3 - i * size/4
            w = size * width_frac
            ax.plot([x - w/2, x + w/2], [line_y, line_y], 'k-', linewidth=1.5)
    elif style == 'chassis':
        # Triangle pointing down
        ax.plot([x, x], [y, y - size/3], 'k-', linewidth=1.5)
        triangle = Polygon([(x - size/2, y - size/3),
                           (x + size/2, y - size/3),
                           (x, y - size)],
                          facecolor='none', edgecolor='black', linewidth=1.5)
        ax.add_patch(triangle)
    elif style == 'signal':
        # Single horizontal line with triangle
        ax.plot([x, x], [y, y - size/2], 'k-', linewidth=1.5)
        ax.plot([x - size/2, x + size/2], [y - size/2, y - size/2], 'k-', linewidth=1.5)


def draw_vcc_ieee(ax, x, y, label='VCC', style='bar'):
    """
    Draw an IEEE-standard power supply symbol.

    Args:
        ax: matplotlib Axes object
        x, y: Position (bottom of power symbol, connects to circuit)
        label: Power rail label
        style: 'bar' (horizontal line), 'arrow' (upward arrow), or 'circle'
    """
    if style == 'bar':
        ax.plot([x, x], [y, y + 0.12], 'k-', linewidth=1.5)
        ax.plot([x - 0.12, x + 0.12], [y + 0.12, y + 0.12], 'k-', linewidth=2.0)
        ax.text(x, y + 0.2, label, ha='center', va='bottom', fontsize=7, fontweight='bold')
    elif style == 'arrow':
        ax.annotate('', xy=(x, y + 0.2), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text(x + 0.08, y + 0.15, label, ha='left', va='center', fontsize=7, fontweight='bold')
    elif style == 'circle':
        ax.plot([x, x], [y, y + 0.1], 'k-', linewidth=1.5)
        circle = Circle((x, y + 0.15), 0.05, facecolor='white', edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y + 0.15, label[0], ha='center', va='center', fontsize=6, fontweight='bold')


# =============================================================================
# Manhattan Routing Utilities
# =============================================================================

def draw_manhattan_wire(ax, start, end, style='h_first', color='black',
                        linewidth=1.0, corner_radius=0):
    """
    Draw a Manhattan-routed wire (orthogonal routing with 90-degree bends).

    Args:
        ax: matplotlib Axes object
        start: (x, y) start point
        end: (x, y) end point
        style: 'h_first' (horizontal then vertical), 'v_first' (vertical then horizontal),
               'auto' (choose based on distance)
        color: Wire color
        linewidth: Line width
        corner_radius: Radius for rounded corners (0 for sharp corners)

    Returns:
        List of (x, y) points defining the path
    """
    x1, y1 = start
    x2, y2 = end

    if style == 'auto':
        # Choose based on which direction has greater distance
        style = 'h_first' if abs(x2 - x1) > abs(y2 - y1) else 'v_first'

    if style == 'h_first':
        mid_x = x2
        mid_y = y1
        points = [(x1, y1), (mid_x, mid_y), (x2, y2)]
    else:  # v_first
        mid_x = x1
        mid_y = y2
        points = [(x1, y1), (mid_x, mid_y), (x2, y2)]

    # Remove duplicate points (when start/end aligned)
    points = [p for i, p in enumerate(points)
              if i == 0 or (p[0] != points[i-1][0] or p[1] != points[i-1][1])]

    if len(points) < 2:
        return points

    # Draw with sharp corners (simplified for reliability)
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, linewidth=linewidth, solid_capstyle='round')

    return points


def draw_manhattan_bus(ax, start, end, n_wires=3, spacing=0.08, style='h_first',
                       color='black', linewidth=0.8):
    """
    Draw a Manhattan-routed bus (multiple parallel wires).

    Args:
        ax: matplotlib Axes object
        start: (x, y) start point for center wire
        end: (x, y) end point for center wire
        n_wires: Number of wires in bus
        spacing: Spacing between wires
        style: Routing style ('h_first' or 'v_first')
        color: Wire color
        linewidth: Line width
    """
    for i in range(n_wires):
        offset = (i - (n_wires - 1) / 2) * spacing
        if style == 'h_first':
            wire_start = (start[0], start[1] + offset)
            wire_end = (end[0], end[1] + offset)
        else:
            wire_start = (start[0] + offset, start[1])
            wire_end = (end[0] + offset, end[1])
        draw_manhattan_wire(ax, wire_start, wire_end, style=style,
                           color=color, linewidth=linewidth)


def draw_connection_dot(ax, x, y, radius=0.04, color='black'):
    """
    Draw a connection/junction dot at wire intersections.

    Args:
        ax: matplotlib Axes object
        x, y: Position
        radius: Dot radius
        color: Dot color
    """
    dot = Circle((x, y), radius, facecolor=color, edgecolor='none')
    ax.add_patch(dot)


# =============================================================================
# Noise Figure Cascade Annotations
# =============================================================================

def calculate_noise_figure_cascade(stages):
    """
    Calculate cumulative noise figure through a cascade of stages.

    Uses Friis formula: NF_total = NF1 + (NF2-1)/G1 + (NF3-1)/(G1*G2) + ...

    Args:
        stages: List of dicts with 'nf' (noise figure in dB) and 'gain' (in dB)

    Returns:
        List of cumulative NF values in dB at each stage output
    """
    cumulative_nf = []
    total_nf_linear = 1.0  # Start with NF = 0 dB (linear = 1)
    cumulative_gain_linear = 1.0

    for i, stage in enumerate(stages):
        nf_linear = 10 ** (stage['nf'] / 10)
        gain_linear = 10 ** (stage['gain'] / 10)

        if i == 0:
            total_nf_linear = nf_linear
        else:
            # Friis formula
            total_nf_linear += (nf_linear - 1) / cumulative_gain_linear

        cumulative_gain_linear *= gain_linear
        cumulative_nf.append(10 * np.log10(total_nf_linear))

    return cumulative_nf


def draw_noise_figure_annotation(ax, x, y, nf_db, label=None, arrow_up=True):
    """
    Draw a noise figure annotation with value and optional label.

    Args:
        ax: matplotlib Axes object
        x, y: Position
        nf_db: Noise figure value in dB
        label: Optional label (e.g., "Cumulative NF")
        arrow_up: If True, annotation arrow points up; else down
    """
    color = WONG_PALETTE['vermillion'] if nf_db > 3.0 else WONG_PALETTE['bluish_green']

    text = f'NF: {nf_db:.1f} dB'
    if label:
        text = f'{label}\n{text}'

    offset_y = 0.3 if arrow_up else -0.3
    ax.annotate(text, xy=(x, y), xytext=(x, y + offset_y),
                ha='center', va='bottom' if arrow_up else 'top',
                fontsize=6, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                         edgecolor=color, alpha=0.9, linewidth=0.5))


def draw_dynamic_range_bar(ax, x, y, width=0.12, height=1.0,
                           signal_level_db=-60, noise_floor_db=-100,
                           max_level_db=-20, show_labels=True):
    """
    Draw a dynamic range indicator bar showing signal headroom and SNR.

    Args:
        ax: matplotlib Axes object
        x, y: Lower-left position
        width: Bar width
        height: Bar height
        signal_level_db: Current signal level in dB
        noise_floor_db: Noise floor in dB
        max_level_db: Maximum level before clipping in dB
        show_labels: Whether to show dB labels
    """
    total_range_db = max_level_db - noise_floor_db

    # Background bar (full range)
    bar_bg = Rectangle((x, y), width, height, facecolor=COLORS['gray_light'],
                       edgecolor=COLORS['gray_dark'], linewidth=0.5)
    ax.add_patch(bar_bg)

    # Signal level
    signal_frac = (signal_level_db - noise_floor_db) / total_range_db
    signal_height = height * signal_frac
    signal_bar = Rectangle((x, y), width, signal_height,
                           facecolor=WONG_PALETTE['bluish_green'], alpha=0.7,
                           edgecolor='none')
    ax.add_patch(signal_bar)

    # SNR indicator
    snr_db = signal_level_db - noise_floor_db
    headroom_db = max_level_db - signal_level_db

    if show_labels:
        ax.text(x + width + 0.03, y + signal_height/2,
                f'{snr_db:.0f}dB', fontsize=5, va='center',
                color=WONG_PALETTE['bluish_green'])
        ax.text(x + width + 0.03, y + height - (height - signal_height)/2,
                f'{headroom_db:.0f}dB', fontsize=5, va='center',
                color=WONG_PALETTE['orange'])


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
    Create MIT transmitter chain block diagram with signal waveform insets.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Block positions (using WONG_PALETTE for colorblind-friendly colors)
    blocks = [
        (1.0, 1.5, 1.4, 0.8, 'MCU\n(DDS)', COLORS['light_purple']),
        (2.8, 1.5, 1.2, 0.8, 'DAC', COLORS['light_blue']),
        (4.4, 1.5, 1.4, 0.8, 'LP Filter\n(fc=50kHz)', COLORS['light_green']),
        (6.2, 1.5, 1.4, 0.8, 'Power\nDriver', COLORS['light_orange']),
        (8.0, 1.5, 1.2, 0.8, 'TX Coil', COLORS['success']),
    ]

    for x, y, w, h, text, color in blocks:
        draw_block(ax, x, y, w, h, text, color=color, fontsize=8)

    # Manhattan-style connections using new utility
    connection_points = [
        ((1.7, 1.5), (2.2, 1.5)),
        ((3.4, 1.5), (3.7, 1.5)),
        ((5.1, 1.5), (5.5, 1.5)),
        ((6.9, 1.5), (7.4, 1.5)),
    ]
    for start, end in connection_points:
        draw_manhattan_wire(ax, start, end, color=COLORS['gray_dark'], linewidth=1.0)
        # Add arrowhead
        draw_connection(ax, start, end, color=COLORS['gray_dark'])

    # Signal labels
    ax.text(1.95, 1.9, 'SPI', fontsize=7, ha='center', color=COLORS['gray_dark'])
    ax.text(3.55, 1.9, 'Analog', fontsize=7, ha='center', color=COLORS['gray_dark'])
    ax.text(5.3, 1.95, 'Clean\nSine', fontsize=6, ha='center', color=COLORS['gray_dark'])
    ax.text(7.15, 1.9, '10-50mA', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Add waveform insets showing signal at key stages
    # After DAC: stepped/quantized sine
    add_waveform_inset(ax, (2.8, 0.7), waveform_type='sine', frequency=2,
                       noise_level=0.1, color=WONG_PALETTE['blue'], label='Quantized')

    # After LP Filter: clean sine
    add_waveform_inset(ax, (4.4, 0.7), waveform_type='sine', frequency=2,
                       noise_level=0.0, color=WONG_PALETTE['bluish_green'], label='Filtered')

    # After Power Driver: high current sine
    add_waveform_inset(ax, (6.2, 0.7), waveform_type='sine', frequency=2,
                       amplitude=1.2, color=WONG_PALETTE['vermillion'], label='Amplified')

    # Title
    ax.text(5, 3.2, 'MIT Transmitter Signal Chain', fontsize=11,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Frequency annotation
    ax.text(1.0, 0.5, 'f = 2-50 kHz', fontsize=8, ha='center',
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
    Create MIT receiver chain block diagram with waveform insets and
    noise figure cascade annotations.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Block positions
    blocks = [
        (1.0, 2.5, 1.2, 0.8, 'RX Coil', COLORS['accent']),
        (2.6, 2.5, 1.2, 0.8, 'Preamp\n(G=10)', COLORS['light_green']),
        (4.2, 2.5, 1.4, 0.8, 'Inst Amp\n(G=100)', COLORS['light_green']),
        (5.9, 2.5, 1.2, 0.8, 'BP Filter', COLORS['light_blue']),
        (7.5, 2.5, 1.2, 0.8, 'ADC\n(16-bit)', COLORS['light_purple']),
        (9.0, 2.5, 0.8, 0.8, 'MCU', COLORS['light_purple']),
    ]

    for x, y, w, h, text, color in blocks:
        draw_block(ax, x, y, w, h, text, color=color, fontsize=8)

    # Lock-in detection branch
    draw_block(ax, 7.5, 1.2, 1.4, 0.7, 'Lock-in\nDetection', COLORS['light_orange'], fontsize=8)

    # Main connections using Manhattan routing
    connection_points = [
        ((1.6, 2.5), (2.0, 2.5)),
        ((3.2, 2.5), (3.5, 2.5)),
        ((4.9, 2.5), (5.3, 2.5)),
        ((6.5, 2.5), (6.9, 2.5)),
        ((8.1, 2.5), (8.6, 2.5)),
    ]
    for start, end in connection_points:
        draw_manhattan_wire(ax, start, end, color=COLORS['gray_dark'], linewidth=1.0)
        draw_connection(ax, start, end, color=COLORS['gray_dark'])

    # Lock-in connections with Manhattan routing
    draw_manhattan_wire(ax, (7.5, 2.1), (7.5, 1.55), style='v_first',
                       color=COLORS['gray_dark'], linewidth=1.0)
    draw_manhattan_wire(ax, (8.2, 1.2), (8.4, 1.2), color=COLORS['gray_dark'], linewidth=1.0)
    draw_manhattan_wire(ax, (8.4, 1.2), (8.4, 2.1), style='v_first',
                       color=COLORS['gray_dark'], linewidth=1.0)
    draw_connection(ax, (8.4, 2.1), (9.0, 2.1), color=COLORS['gray_dark'])

    # Signal labels with voltage levels
    ax.text(1.8, 2.9, 'uV', fontsize=7, ha='center', color=COLORS['gray_dark'])
    ax.text(3.35, 2.9, 'mV', fontsize=7, ha='center', color=COLORS['gray_dark'])
    ax.text(5.1, 2.9, '100mV', fontsize=7, ha='center', color=COLORS['gray_dark'])
    ax.text(6.7, 2.9, 'Filtered', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Add waveform insets showing signal quality at each stage
    # RX Coil output: very noisy signal
    add_waveform_inset(ax, (1.0, 3.5), waveform_type='sine', frequency=2,
                       noise_level=0.5, amplitude=0.3, color=WONG_PALETTE['blue'],
                       label='Noisy')

    # After Preamp: less noisy, amplified
    add_waveform_inset(ax, (2.6, 3.5), waveform_type='sine', frequency=2,
                       noise_level=0.2, amplitude=0.7, color=WONG_PALETTE['blue'],
                       label='Amplified')

    # After Inst Amp: cleaner signal
    add_waveform_inset(ax, (4.2, 3.5), waveform_type='sine', frequency=2,
                       noise_level=0.1, amplitude=0.9, color=WONG_PALETTE['bluish_green'],
                       label='Clean')

    # After BP Filter: very clean
    add_waveform_inset(ax, (5.9, 3.5), waveform_type='sine', frequency=2,
                       noise_level=0.02, amplitude=0.95, color=WONG_PALETTE['bluish_green'],
                       label='Filtered')

    # Calculate and display cumulative noise figures
    # Typical RX chain stage parameters
    rx_stages = [
        {'nf': 1.0, 'gain': 0},    # RX Coil (passive, unity gain)
        {'nf': 2.5, 'gain': 20},   # Preamp (NF=2.5dB, G=20dB)
        {'nf': 3.0, 'gain': 40},   # Inst Amp (NF=3dB, G=40dB)
        {'nf': 1.5, 'gain': 0},    # BP Filter (NF=1.5dB, passive)
        {'nf': 6.0, 'gain': 0},    # ADC (quantization noise equivalent)
    ]
    cumulative_nfs = calculate_noise_figure_cascade(rx_stages)

    # Display NF annotations at key stages
    nf_positions = [(2.6, 1.7), (4.2, 1.7), (5.9, 1.7)]
    nf_labels = ['Preamp', 'Inst Amp', 'Filter']
    for i, (pos, lbl) in enumerate(zip(nf_positions, nf_labels)):
        draw_noise_figure_annotation(ax, pos[0], pos[1], cumulative_nfs[i+1],
                                     label=f'After {lbl}', arrow_up=False)

    # Reference signal annotation
    ax.annotate('REF from TX', xy=(6.8, 1.2), xytext=(5.5, 0.8),
               arrowprops=dict(arrowstyle='->', color=WONG_PALETTE['orange']),
               fontsize=7, color=WONG_PALETTE['orange'])

    # Dynamic range bar
    draw_dynamic_range_bar(ax, 9.5, 2.0, width=0.15, height=1.2,
                          signal_level_db=-58, noise_floor_db=-92,
                          max_level_db=-20, show_labels=True)
    ax.text(9.6, 3.3, 'DR', fontsize=6, ha='center', color=COLORS['gray_dark'])

    # Title
    ax.text(5, 4.2, 'MIT Receiver Signal Chain', fontsize=11,
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
    Create ERT Howland current source schematic with IEEE-style components.

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

    # Input resistors using IEEE-style zigzag
    draw_resistor_ieee(ax, 1.5, 3.0, width=0.5, height=0.12)
    ax.text(1.5, 3.3, 'R1', fontsize=8, ha='center')
    # Manhattan routing for wires
    draw_manhattan_wire(ax, (0.5, 3.0), (1.25, 3.0), color='black', linewidth=1.0)
    draw_manhattan_wire(ax, (1.75, 3.0), (2.9, 3.0), color='black', linewidth=1.0)
    draw_manhattan_wire(ax, (2.9, 3.0), (2.9, 2.7), style='v_first', color='black', linewidth=1.0)

    draw_resistor_ieee(ax, 1.5, 2.0, width=0.5, height=0.12)
    ax.text(1.5, 1.7, 'R2', fontsize=8, ha='center')
    draw_manhattan_wire(ax, (0.5, 2.0), (1.25, 2.0), color='black', linewidth=1.0)
    draw_manhattan_wire(ax, (1.75, 2.0), (2.9, 2.0), color='black', linewidth=1.0)
    draw_manhattan_wire(ax, (2.9, 2.0), (2.9, 2.3), style='v_first', color='black', linewidth=1.0)

    # Feedback resistors using IEEE-style
    draw_resistor_ieee(ax, 5.0, 3.5, width=0.5, height=0.12)
    ax.text(5.0, 3.8, 'R3', fontsize=8, ha='center')
    draw_manhattan_wire(ax, (4.1, 2.7), (4.75, 2.7), color='black', linewidth=1.0)
    draw_manhattan_wire(ax, (4.75, 2.7), (4.75, 3.5), style='v_first', color='black', linewidth=1.0)
    draw_manhattan_wire(ax, (5.25, 3.5), (5.5, 3.5), color='black', linewidth=1.0)
    draw_manhattan_wire(ax, (5.5, 3.5), (5.5, 2.5), style='v_first', color='black', linewidth=1.0)

    draw_resistor_ieee(ax, 5.0, 1.5, width=0.5, height=0.12)
    ax.text(5.0, 1.2, 'R4', fontsize=8, ha='center')
    draw_manhattan_wire(ax, (4.1, 2.3), (4.75, 2.3), color='black', linewidth=1.0)
    draw_manhattan_wire(ax, (4.75, 2.3), (4.75, 1.5), style='v_first', color='black', linewidth=1.0)
    draw_manhattan_wire(ax, (5.25, 1.5), (6.0, 1.5), color='black', linewidth=1.0)

    # Output
    draw_manhattan_wire(ax, (4.1, 2.5), (5.5, 2.5), color='black', linewidth=1.0)
    draw_manhattan_wire(ax, (5.5, 2.5), (6.0, 2.5), color='black', linewidth=1.0)

    # Connection dots at junctions
    draw_connection_dot(ax, 5.5, 2.5)
    draw_connection_dot(ax, 4.75, 2.7)
    draw_connection_dot(ax, 4.75, 2.3)

    # Load resistor (vertical, IEEE-style)
    draw_resistor_ieee(ax, 6.0, 2.0, width=0.5, height=0.12, vertical=True)
    ax.text(6.35, 2.0, 'Z_load', fontsize=7, va='center')

    # IEEE-style ground symbols
    draw_ground_ieee(ax, 6.0, 1.45, style='earth')
    draw_ground_ieee(ax, 0.5, 1.55, style='earth')
    draw_manhattan_wire(ax, (0.5, 2.0), (0.5, 1.7), style='v_first', color='black', linewidth=1.0)

    # VCC power symbols for op-amp
    draw_vcc_ieee(ax, 3.5, 3.2, label='+V', style='bar')
    ax.plot([3.5, 3.5], [3.1, 3.05], 'k-', lw=1)

    # Input labels
    ax.text(0.3, 3.0, 'V_in+', fontsize=8, ha='right', color=WONG_PALETTE['vermillion'])
    ax.text(0.3, 2.0, 'V_in-', fontsize=8, ha='right', color=WONG_PALETTE['blue'])

    # Output current label
    ax.annotate('I_out = Vref / R_sense', xy=(5.5, 2.5), xytext=(5.8, 3.5),
               arrowprops=dict(arrowstyle='->', color=WONG_PALETTE['bluish_green']),
               fontsize=9, color=WONG_PALETTE['bluish_green'], fontweight='bold')

    # Title
    ax.text(3.5, 4.7, 'Howland Current Source (ERT)', fontsize=11,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Notes
    ax.text(3.5, 0.3, 'For matched R1=R2=R3=R4: I_out = Vref/R_sense\nTypical: 0.5-2 mA, 0.1% resistors',
           fontsize=8, ha='center', color=COLORS['gray_dark'], style='italic')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_lock_in_detection_diagram():
    """
    Create detailed lock-in detection block diagram showing
    digital lock-in with waveform insets at each stage.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 5.2, 'Lock-In Detection (Digital Implementation)',
           fontsize=12, fontweight='bold', ha='center', color=COLORS['primary'])

    # Input signal path
    draw_block(ax, 1.0, 3.0, 1.2, 0.7, 'RX\nSignal', COLORS['light_green'], fontsize=8)

    # Multiplier blocks (I and Q channels)
    draw_block(ax, 3.0, 3.8, 1.0, 0.6, 'Multiply', COLORS['light_blue'], fontsize=8)
    draw_block(ax, 3.0, 2.2, 1.0, 0.6, 'Multiply', COLORS['light_blue'], fontsize=8)

    # Reference signal generators
    draw_block(ax, 3.0, 1.0, 1.2, 0.6, 'sin(wt)\nRef', COLORS['light_purple'], fontsize=7)
    draw_block(ax, 3.0, 4.6, 1.2, 0.6, 'cos(wt)\nRef', COLORS['light_purple'], fontsize=7)

    # Low-pass filters
    draw_block(ax, 5.0, 3.8, 1.2, 0.6, 'LP Filter', COLORS['light_orange'], fontsize=8)
    draw_block(ax, 5.0, 2.2, 1.2, 0.6, 'LP Filter', COLORS['light_orange'], fontsize=8)

    # Integration blocks
    draw_block(ax, 7.0, 3.8, 1.0, 0.6, 'Integrate', COLORS['light_green'], fontsize=7)
    draw_block(ax, 7.0, 2.2, 1.0, 0.6, 'Integrate', COLORS['light_green'], fontsize=7)

    # Output computation
    draw_block(ax, 9.0, 3.0, 1.6, 1.4, 'Compute\nA = sqrt(I^2+Q^2)\nPhi = atan2(Q,I)',
              COLORS['light_purple'], fontsize=7)

    # Manhattan-style connections - main signal path
    draw_manhattan_wire(ax, (1.6, 3.0), (2.0, 3.0), color=COLORS['gray_dark'], linewidth=1.2)
    draw_manhattan_wire(ax, (2.0, 3.0), (2.0, 3.8), style='v_first', color=COLORS['gray_dark'], linewidth=1.2)
    draw_manhattan_wire(ax, (2.0, 3.0), (2.0, 2.2), style='v_first', color=COLORS['gray_dark'], linewidth=1.2)
    draw_connection_dot(ax, 2.0, 3.0)

    draw_connection(ax, (2.0, 3.8), (2.5, 3.8), color=COLORS['gray_dark'])
    draw_connection(ax, (2.0, 2.2), (2.5, 2.2), color=COLORS['gray_dark'])

    # Reference connections with Manhattan routing
    draw_manhattan_wire(ax, (3.0, 4.3), (3.0, 4.1), style='v_first', color=WONG_PALETTE['blue'], linewidth=1.0)
    draw_manhattan_wire(ax, (3.0, 1.3), (3.0, 1.9), style='v_first', color=WONG_PALETTE['blue'], linewidth=1.0)

    # Through filters
    draw_connection(ax, (3.5, 3.8), (4.4, 3.8), color=COLORS['gray_dark'])
    draw_connection(ax, (3.5, 2.2), (4.4, 2.2), color=COLORS['gray_dark'])

    # Through integrators
    draw_connection(ax, (5.6, 3.8), (6.5, 3.8), color=COLORS['gray_dark'])
    draw_connection(ax, (5.6, 2.2), (6.5, 2.2), color=COLORS['gray_dark'])

    # To output with Manhattan routing
    draw_manhattan_wire(ax, (7.5, 3.8), (8.0, 3.8), color=COLORS['gray_dark'], linewidth=1.0)
    draw_manhattan_wire(ax, (8.0, 3.8), (8.0, 3.4), style='v_first', color=COLORS['gray_dark'], linewidth=1.0)
    draw_connection(ax, (8.0, 3.4), (8.2, 3.4), color=COLORS['gray_dark'])

    draw_manhattan_wire(ax, (7.5, 2.2), (8.0, 2.2), color=COLORS['gray_dark'], linewidth=1.0)
    draw_manhattan_wire(ax, (8.0, 2.2), (8.0, 2.6), style='v_first', color=COLORS['gray_dark'], linewidth=1.0)
    draw_connection(ax, (8.0, 2.6), (8.2, 2.6), color=COLORS['gray_dark'])

    # Waveform insets showing signal transformation
    # Input: noisy signal
    add_waveform_inset(ax, (0.5, 3.0), waveform_type='sine', frequency=3,
                       noise_level=0.4, color=WONG_PALETTE['vermillion'])

    # After multiply (I-channel): mixed signal
    add_waveform_inset(ax, (3.8, 4.5), waveform_type='sine', frequency=1,
                       noise_level=0.2, amplitude=0.8, color=WONG_PALETTE['blue'])

    # After LP filter (I-channel): DC-like
    add_waveform_inset(ax, (5.8, 4.5), waveform_type='filtered', frequency=0.5,
                       noise_level=0.05, color=WONG_PALETTE['bluish_green'])

    # Channel labels
    ax.text(1.8, 4.15, 'I-channel', fontsize=7, color=WONG_PALETTE['blue'], fontweight='bold')
    ax.text(1.8, 1.85, 'Q-channel', fontsize=7, color=WONG_PALETTE['vermillion'], fontweight='bold')

    # Output labels
    ax.text(9.0, 1.8, 'Amplitude &\nPhase Output', fontsize=8, ha='center',
           color=WONG_PALETTE['bluish_green'], fontweight='bold')

    # Signal quality annotations
    add_signal_quality_annotation(ax, (1.6, 2.5), snr_db=10, quality='poor')
    add_signal_quality_annotation(ax, (9.0, 2.0), snr_db=50, quality='good')

    # Annotations
    ax.text(5, 0.4, 'Digital lock-in extracts signal at reference frequency,\n'
           'rejecting noise at other frequencies (SNR improvement > 40 dB)',
           fontsize=8, ha='center', style='italic', color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_signal_level_progression():
    """
    Create signal level progression chart showing voltage levels
    through the MIT RX chain.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Stages and their signal levels (in dBV)
    stages = ['RX Coil', 'Preamp\n(G=10)', 'Inst Amp\n(G=100)', 'Filter', 'ADC\nInput']
    levels_dbv = [-100, -80, -60, -58, -58]  # dBV
    levels_v = ['1 uV', '10 uV', '1 mV', '0.8 mV', '0.8 mV']
    noise_floor = [-120, -110, -90, -92, -92]  # dBV noise floor

    x = np.arange(len(stages))
    width = 0.35

    # Signal bars
    bars1 = ax.bar(x - width/2, [l + 120 for l in levels_dbv], width,
                   label='Signal Level', color=COLORS['success'], alpha=0.8)

    # Noise floor bars
    bars2 = ax.bar(x + width/2, [n + 120 for n in noise_floor], width,
                   label='Noise Floor', color=COLORS['warning'], alpha=0.5)

    # Add voltage labels on signal bars
    for bar, v_label in zip(bars1, levels_v):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               v_label, ha='center', va='bottom', fontsize=8, fontweight='bold')

    # SNR annotation line
    ax.annotate('', xy=(4.2, 62), xytext=(4.2, 28),
               arrowprops=dict(arrowstyle='<->', color=COLORS['accent'], lw=2))
    ax.text(4.4, 45, 'SNR\n~34 dB', fontsize=8, color=COLORS['accent'], fontweight='bold')

    ax.set_ylabel('Signal Level (dBV + 120)', fontsize=10)
    ax.set_title('MIT RX Signal Level Progression', fontsize=12,
                fontweight='bold', color=COLORS['primary'])
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, 80)

    # Grid
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    # Secondary y-axis with actual dBV
    ax2 = ax.twinx()
    ax2.set_ylim(-120, -40)
    ax2.set_ylabel('Actual Level (dBV)', fontsize=10)

    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_adc_interface_diagram():
    """
    Create ADC interface diagram showing connections to ADS1256.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(4.5, 4.7, 'ADC Interface (ADS1256)', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # ADC chip block
    adc_box = FancyBboxPatch((3.0, 1.5), 3.0, 2.5, boxstyle="round,pad=0.05",
                             facecolor=COLORS['light_purple'], edgecolor=COLORS['secondary'],
                             linewidth=2)
    ax.add_patch(adc_box)
    ax.text(4.5, 3.5, 'ADS1256', fontsize=11, fontweight='bold', ha='center')
    ax.text(4.5, 3.0, '24-bit ADC\n30 kSPS\nPGA 1-64x', fontsize=8, ha='center')

    # Input side labels
    inputs = [
        (2.8, 3.8, 'AIN0+', 'MIT RX'),
        (2.8, 3.4, 'AIN0-', ''),
        (2.8, 2.8, 'AIN1+', 'ERT V+'),
        (2.8, 2.4, 'AIN1-', 'ERT V-'),
        (2.8, 1.8, 'AINCOM', 'AGND'),
    ]

    for x, y, label, src in inputs:
        ax.text(x, y, label, fontsize=7, ha='right', va='center')
        ax.plot([x + 0.05, x + 0.2], [y, y], 'k-', lw=1)
        if src:
            ax.text(0.5, y, src, fontsize=8, ha='left', va='center', color=COLORS['gray_dark'])
            ax.plot([1.3, 2.8], [y, y], 'k-', lw=1)

    # Output side labels (SPI)
    outputs = [
        (6.2, 3.6, 'SCLK'),
        (6.2, 3.2, 'DIN (MOSI)'),
        (6.2, 2.8, 'DOUT (MISO)'),
        (6.2, 2.4, 'CS'),
        (6.2, 2.0, 'DRDY'),
    ]

    for x, y, label in outputs:
        ax.text(x, y, label, fontsize=7, ha='left', va='center')
        ax.plot([x - 0.05, x - 0.2], [y, y], 'k-', lw=1)

    # MCU block
    draw_block(ax, 8.0, 2.8, 1.4, 1.8, 'MCU\n(ESP32)\nSPI\nMaster', COLORS['light_blue'], fontsize=7)

    # SPI connections
    for y in [3.6, 3.2, 2.8, 2.4, 2.0]:
        ax.plot([6.5, 7.3], [y, y], 'b-', lw=1)

    # Power connections
    ax.text(4.5, 1.3, '+3.3V', fontsize=8, ha='center', color='red')
    ax.plot([4.5, 4.5], [1.5, 1.1], 'r-', lw=1.5)

    # Reference voltage
    draw_block(ax, 1.0, 1.2, 1.0, 0.6, 'Vref\n2.5V', COLORS['light_orange'], fontsize=7)
    ax.plot([1.5, 3.0], [1.2, 1.8], 'k--', lw=1)
    ax.text(2.0, 1.6, 'VREFP', fontsize=7, color=COLORS['gray_dark'])

    # Annotations
    ax.text(4.5, 0.5, 'Differential inputs | Internal MUX | PGA for gain adjustment',
           fontsize=8, ha='center', style='italic', color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_multiplexer_switching_diagram():
    """
    Create multiplexer switching topology diagram showing
    how probes are selected for TX/RX operations.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Multiplexer Switching Topology', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # TX Multiplexer section
    draw_block(ax, 1.5, 4.5, 1.4, 0.7, 'TX\nDriver', COLORS['light_green'], fontsize=8)

    # TX MUX
    mux_tx = FancyBboxPatch((2.5, 3.5), 1.5, 1.8, boxstyle="round,pad=0.03",
                            facecolor=COLORS['light_blue'], edgecolor=COLORS['secondary'],
                            linewidth=1.5)
    ax.add_patch(mux_tx)
    ax.text(3.25, 5.1, 'TX MUX', fontsize=9, ha='center', fontweight='bold')
    ax.text(3.25, 4.5, 'CD4051\n8:1', fontsize=8, ha='center')

    # TX MUX outputs
    tx_outputs = ['TX1', 'TX2', 'TX3', '...', 'TX8']
    for i, label in enumerate(tx_outputs):
        y = 5.0 - i * 0.35
        ax.plot([4.0, 4.5], [y, y], 'k-', lw=1)
        ax.text(4.6, y, label, fontsize=7, va='center')

    # RX Multiplexer section
    draw_block(ax, 1.5, 1.5, 1.4, 0.7, 'RX\nPreamp', COLORS['light_orange'], fontsize=8)

    # RX MUX
    mux_rx = FancyBboxPatch((2.5, 0.7), 1.5, 1.8, boxstyle="round,pad=0.03",
                            facecolor=COLORS['light_blue'], edgecolor=COLORS['secondary'],
                            linewidth=1.5)
    ax.add_patch(mux_rx)
    ax.text(3.25, 2.3, 'RX MUX', fontsize=9, ha='center', fontweight='bold')
    ax.text(3.25, 1.7, 'CD4051\n8:1', fontsize=8, ha='center')

    # RX MUX outputs
    rx_outputs = ['RX1', 'RX2', 'RX3', '...', 'RX8']
    for i, label in enumerate(rx_outputs):
        y = 2.2 - i * 0.35
        ax.plot([4.0, 4.5], [y, y], 'k-', lw=1)
        ax.text(4.6, y, label, fontsize=7, va='center')

    # Connections to driver/preamp
    draw_connection(ax, (2.2, 4.5), (2.5, 4.5), color=COLORS['gray_dark'])
    draw_connection(ax, (2.5, 1.5), (2.2, 1.5), color=COLORS['gray_dark'])

    # Probe array representation
    probe_box = FancyBboxPatch((5.5, 0.5), 3.5, 5.0, boxstyle="round,pad=0.05",
                               facecolor='white', edgecolor=COLORS['gray_med'],
                               linewidth=1, linestyle='--')
    ax.add_patch(probe_box)
    ax.text(7.25, 5.3, 'Probe Array (8-24 probes)', fontsize=9, ha='center',
           fontweight='bold', color=COLORS['gray_dark'])

    # Individual probes
    probe_y = [4.5, 3.7, 2.9, 2.1, 1.3]
    probe_labels = ['Probe 1', 'Probe 2', 'Probe 3', '...', 'Probe N']
    for i, (y, label) in enumerate(zip(probe_y, probe_labels)):
        draw_block(ax, 7.25, y, 1.8, 0.5, label, COLORS['light_green'], fontsize=7)
        # TX connection
        ax.plot([5.0, 5.5], [5.0 - i*0.35, y + 0.1], 'g-', lw=0.8, alpha=0.6)
        # RX connection
        ax.plot([5.0, 5.5], [2.2 - i*0.35, y - 0.1], 'b-', lw=0.8, alpha=0.6)

    # MCU control
    draw_block(ax, 1.5, 3.0, 1.2, 0.5, 'MCU\nGPIO', COLORS['light_purple'], fontsize=7)
    ax.plot([2.1, 2.5], [3.2, 4.0], 'k--', lw=1)
    ax.plot([2.1, 2.5], [2.8, 2.0], 'k--', lw=1)
    ax.text(2.0, 3.6, 'A0-A2', fontsize=6, color=COLORS['gray_dark'])

    # Annotations
    ax.text(5, 0.2, 'Sequential TX/RX switching allows full tomographic measurement matrix',
           fontsize=8, ha='center', style='italic', color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_noise_filtering_diagram():
    """
    Create noise filtering stages diagram showing the
    filtering at each stage with waveform insets demonstrating noise reduction.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 4.2, 'Noise Filtering Stages', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Stage 1: Input filter
    draw_block(ax, 1.0, 2.0, 1.4, 1.0, 'Input\nFilter\n(RC LP)', COLORS['light_green'], fontsize=7)
    ax.text(1.0, 1.3, 'fc = 100 kHz', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Stage 2: Preamp with high-pass
    draw_block(ax, 3.0, 2.0, 1.4, 1.0, 'Preamp\n+\nHP Filter', COLORS['light_blue'], fontsize=7)
    ax.text(3.0, 1.3, 'fc = 100 Hz', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Stage 3: Band-pass filter
    draw_block(ax, 5.0, 2.0, 1.4, 1.0, 'Active\nBand-Pass\nFilter', COLORS['light_orange'], fontsize=7)
    ax.text(5.0, 1.3, '1-50 kHz', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Stage 4: Anti-aliasing
    draw_block(ax, 7.0, 2.0, 1.4, 1.0, 'Anti-Alias\nFilter\n(4th order)', COLORS['light_purple'], fontsize=7)
    ax.text(7.0, 1.3, 'fc = 15 kHz', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Stage 5: Digital filter (in DSP)
    draw_block(ax, 9.0, 2.0, 1.2, 1.0, 'Digital\nFilter\n(DSP)', COLORS['light_green'], fontsize=7)
    ax.text(9.0, 1.3, 'Matched', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Manhattan-style connections
    connection_x = [(1.7, 2.3), (3.7, 4.3), (5.7, 6.3), (7.7, 8.4)]
    for start_x, end_x in connection_x:
        draw_manhattan_wire(ax, (start_x, 2.0), (end_x, 2.0),
                           color=COLORS['gray_dark'], linewidth=1.0)
        draw_connection(ax, (start_x, 2.0), (end_x, 2.0), color=COLORS['gray_dark'])

    # Waveform insets showing progressive noise reduction
    # Before filtering: very noisy
    add_waveform_inset(ax, (0.3, 2.0), waveform_type='sine', frequency=2,
                       noise_level=0.6, color=WONG_PALETTE['vermillion'])

    # After stage 1: reduced HF noise
    add_waveform_inset(ax, (2.0, 3.3), waveform_type='sine', frequency=2,
                       noise_level=0.4, color=WONG_PALETTE['orange'])

    # After stage 3: much cleaner
    add_waveform_inset(ax, (6.0, 3.3), waveform_type='sine', frequency=2,
                       noise_level=0.1, color=WONG_PALETTE['bluish_green'])

    # After all stages: very clean
    add_waveform_inset(ax, (9.7, 2.0), waveform_type='sine', frequency=2,
                       noise_level=0.02, color=WONG_PALETTE['bluish_green'])

    # Noise sources labeled above (using WONG colors)
    noise_sources = [
        (1.0, 'EMI\n60 Hz', WONG_PALETTE['vermillion']),
        (3.0, 'DC\nOffset', WONG_PALETTE['orange']),
        (5.0, 'Wideband\nNoise', WONG_PALETTE['orange']),
        (7.0, 'Aliasing', WONG_PALETTE['reddish_purple']),
        (9.0, 'Quantization', WONG_PALETTE['reddish_purple']),
    ]
    for x, label, color in noise_sources:
        ax.text(x, 2.9, label, fontsize=6, ha='center', color=color,
               fontweight='bold')
        ax.annotate('', xy=(x, 2.55), xytext=(x, 2.75),
                   arrowprops=dict(arrowstyle='->', color=color, lw=1))

    # Noise rejection values (dB)
    rejection_db = ['>40', '>20', '>30', '>50', '>20']
    for i, (x, _, _) in enumerate(noise_sources):
        ax.text(x, 0.9, f'-{rejection_db[i]} dB', fontsize=6, ha='center',
               color=WONG_PALETTE['bluish_green'], fontweight='bold')

    # Rejection summary
    ax.text(5, 0.5, 'Each stage targets specific noise sources: Total rejection > 60 dB in measurement band',
           fontsize=8, ha='center', style='italic', color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_ground_loop_prevention():
    """
    Create ground loop prevention diagram showing proper
    grounding techniques with IEEE-style ground symbols.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # === Left Panel: Ground Loop Problem ===
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 4.5)
    ax1.set_aspect('equal')
    ax1.axis('off')

    ax1.set_title('(a) Ground Loop Problem', fontsize=10, fontweight='bold',
                 color=WONG_PALETTE['vermillion'], pad=10)

    # Two circuit blocks
    draw_block(ax1, 1.0, 3.0, 1.2, 0.8, 'Circuit\nA', COLORS['light_red'], fontsize=8)
    draw_block(ax1, 4.0, 3.0, 1.2, 0.8, 'Circuit\nB', COLORS['light_red'], fontsize=8)

    # Signal connection
    draw_manhattan_wire(ax1, (1.6, 3.0), (3.4, 3.0), color='black', linewidth=1.5)
    ax1.text(2.5, 3.2, 'Signal', fontsize=7, ha='center')

    # Multiple ground paths (the problem) - using Manhattan routing
    # Path 1 - direct
    draw_manhattan_wire(ax1, (1.0, 2.6), (1.0, 1.5), style='v_first',
                       color=WONG_PALETTE['vermillion'], linewidth=2.0)
    draw_manhattan_wire(ax1, (4.0, 2.6), (4.0, 1.5), style='v_first',
                       color=WONG_PALETTE['vermillion'], linewidth=2.0)
    draw_manhattan_wire(ax1, (1.0, 1.5), (4.0, 1.5),
                       color=WONG_PALETTE['vermillion'], linewidth=2.0)

    # Path 2 - through chassis/earth
    ax1.plot([1.0, 1.0], [1.5, 0.8], color=WONG_PALETTE['vermillion'],
             linestyle='--', lw=1.5)
    ax1.plot([4.0, 4.0], [1.5, 0.8], color=WONG_PALETTE['vermillion'],
             linestyle='--', lw=1.5)
    ax1.plot([1.0, 4.0], [0.8, 0.8], color=WONG_PALETTE['vermillion'],
             linestyle='--', lw=1.5)

    # IEEE-style ground symbols
    draw_ground_ieee(ax1, 1.0, 0.5, style='earth')
    draw_ground_ieee(ax1, 4.0, 0.5, style='earth')

    # Connection dots
    draw_connection_dot(ax1, 1.0, 1.5, color=WONG_PALETTE['vermillion'])
    draw_connection_dot(ax1, 4.0, 1.5, color=WONG_PALETTE['vermillion'])

    # Loop current indicator
    ax1.annotate('', xy=(2.0, 1.2), xytext=(3.0, 1.2),
                arrowprops=dict(arrowstyle='->', color=WONG_PALETTE['vermillion'], lw=2))
    ax1.text(2.5, 0.9, 'Loop\nCurrent', fontsize=7, ha='center',
            color=WONG_PALETTE['vermillion'])

    # Noise injection annotation
    ax1.text(2.5, 1.8, 'EMI pickup', fontsize=7, ha='center',
            color=WONG_PALETTE['vermillion'], style='italic')

    # === Right Panel: Star Ground Solution ===
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 4.5)
    ax2.set_aspect('equal')
    ax2.axis('off')

    ax2.set_title('(b) Star Ground Solution', fontsize=10, fontweight='bold',
                 color=WONG_PALETTE['bluish_green'], pad=10)

    # Two circuit blocks
    draw_block(ax2, 1.0, 3.0, 1.2, 0.8, 'Circuit\nA', COLORS['light_green'], fontsize=8)
    draw_block(ax2, 4.0, 3.0, 1.2, 0.8, 'Circuit\nB', COLORS['light_green'], fontsize=8)

    # Signal connection with shield (coaxial representation)
    draw_manhattan_wire(ax2, (1.6, 3.0), (3.4, 3.0), color='black', linewidth=1.5)
    ax2.plot([1.6, 3.4], [3.12, 3.12], color=WONG_PALETTE['bluish_green'],
             linestyle='--', lw=1)
    ax2.plot([1.6, 3.4], [2.88, 2.88], color=WONG_PALETTE['bluish_green'],
             linestyle='--', lw=1)
    ax2.text(2.5, 3.35, 'Shielded Signal', fontsize=7, ha='center')

    # Star ground point
    star_x, star_y = 2.5, 1.2
    ax2.add_patch(Circle((star_x, star_y), 0.15, facecolor=WONG_PALETTE['bluish_green'],
                        edgecolor='black', lw=2))
    ax2.text(star_x, 0.65, 'Star Ground\nPoint', fontsize=7, ha='center',
            fontweight='bold', color=WONG_PALETTE['bluish_green'])

    # Individual ground connections (star pattern)
    ax2.plot([1.0, star_x], [2.6, star_y + 0.15], color=WONG_PALETTE['bluish_green'], lw=2)
    ax2.plot([4.0, star_x], [2.6, star_y + 0.15], color=WONG_PALETTE['bluish_green'], lw=2)
    ax2.plot([2.5, star_x], [2.88, star_y + 0.15], color=WONG_PALETTE['bluish_green'], lw=1)

    # Single earth ground with IEEE symbol
    ax2.plot([star_x, star_x], [star_y - 0.15, 0.45], color=WONG_PALETTE['bluish_green'], lw=2)
    draw_ground_ieee(ax2, star_x, 0.3, style='earth')

    # Success annotation
    ax2.text(0.5, 4.2, 'No loop!', fontsize=9, color=WONG_PALETTE['bluish_green'],
            fontweight='bold')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_ert_polarity_reversal():
    """
    Create ERT polarity reversal circuit diagram showing
    the relay-based switching for electrode polarization reduction.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(4, 4.7, 'ERT Polarity Reversal Circuit', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Current source
    draw_block(ax, 1.5, 2.5, 1.4, 1.0, 'Current\nSource\n(1 mA)', COLORS['light_orange'], fontsize=8)

    # DPDT Relay
    relay_box = FancyBboxPatch((3.0, 1.5), 2.0, 2.0, boxstyle="round,pad=0.05",
                               facecolor=COLORS['light_blue'], edgecolor=COLORS['secondary'],
                               linewidth=1.5)
    ax.add_patch(relay_box)
    ax.text(4.0, 3.3, 'DPDT Relay', fontsize=9, ha='center', fontweight='bold')
    ax.text(4.0, 2.9, '(G5V-2)', fontsize=7, ha='center')

    # Relay internal connections (simplified)
    # Common contacts
    ax.plot([3.2, 3.5], [2.5, 2.5], 'k-', lw=1.5)
    ax.plot([3.2, 3.5], [2.0, 2.0], 'k-', lw=1.5)

    # Switching contacts (shown in one position)
    ax.plot([3.5, 4.0], [2.5, 2.7], 'k-', lw=1.5)
    ax.plot([3.5, 4.0], [2.0, 1.8], 'k-', lw=1.5)
    ax.add_patch(Circle((3.5, 2.5), 0.05, facecolor='black'))
    ax.add_patch(Circle((3.5, 2.0), 0.05, facecolor='black'))

    # NC/NO contacts
    ax.add_patch(Circle((4.0, 2.7), 0.05, facecolor='black'))
    ax.add_patch(Circle((4.0, 2.3), 0.05, facecolor='black'))
    ax.add_patch(Circle((4.0, 2.0), 0.05, facecolor='black'))
    ax.add_patch(Circle((4.0, 1.8), 0.05, facecolor='black'))

    # Cross-wiring
    ax.plot([4.0, 4.8], [2.7, 2.7], 'b-', lw=1)
    ax.plot([4.0, 4.8], [1.8, 1.8], 'b-', lw=1)
    ax.plot([4.0, 4.5], [2.3, 2.3], 'r-', lw=1)
    ax.plot([4.5, 4.5], [2.3, 1.8], 'r-', lw=1)
    ax.plot([4.0, 4.6], [2.0, 2.0], 'r-', lw=1)
    ax.plot([4.6, 4.6], [2.0, 2.7], 'r-', lw=1)

    # Input from current source
    draw_connection(ax, (2.2, 2.7), (3.2, 2.7), color='black')
    ax.plot([3.2, 3.2], [2.7, 2.5], 'k-', lw=1)
    draw_connection(ax, (2.2, 2.3), (3.2, 2.3), color='black')
    ax.plot([3.2, 3.2], [2.3, 2.0], 'k-', lw=1)

    # Output to electrodes
    draw_block(ax, 6.5, 2.7, 1.0, 0.5, 'Ring A', COLORS['light_green'], fontsize=8)
    draw_block(ax, 6.5, 1.8, 1.0, 0.5, 'Ring B', COLORS['light_green'], fontsize=8)

    ax.plot([4.8, 6.0], [2.7, 2.7], 'k-', lw=1)
    ax.plot([4.8, 6.0], [1.8, 1.8], 'k-', lw=1)

    # MCU control
    draw_block(ax, 4.0, 0.8, 1.0, 0.5, 'MCU', COLORS['light_purple'], fontsize=8)
    ax.plot([4.0, 4.0], [1.05, 1.5], 'k--', lw=1)
    ax.text(4.2, 1.3, 'Control', fontsize=7, color=COLORS['gray_dark'])

    # Timing annotation
    ax.text(4, 4.2, 'Reversal frequency: 0.5 Hz (every 2 seconds)',
           fontsize=8, ha='center', style='italic', color=COLORS['gray_dark'])

    # Polarity indicators
    ax.text(2.5, 3.0, 'I+', fontsize=9, color='red', fontweight='bold')
    ax.text(2.5, 2.0, 'I-', fontsize=9, color='blue', fontweight='bold')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_component_location_diagram():
    """
    Create PCB component location diagram showing the
    recommended layout for the electronics hub.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'PCB Layout Guidelines', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # PCB outline
    pcb = FancyBboxPatch((0.5, 0.5), 9.0, 5.0, boxstyle="round,pad=0.02",
                         facecolor='#c6f6d5', edgecolor='black', linewidth=2)
    ax.add_patch(pcb)

    # Analog section (shielded)
    analog_box = FancyBboxPatch((0.8, 2.5), 3.5, 2.7, boxstyle="round,pad=0.02",
                                facecolor='#ebf8ff', edgecolor=COLORS['accent'],
                                linewidth=1.5, linestyle='--')
    ax.add_patch(analog_box)
    ax.text(2.55, 5.0, 'ANALOG SECTION', fontsize=8, ha='center',
           fontweight='bold', color=COLORS['accent'])
    ax.text(2.55, 4.7, '(Shielded)', fontsize=7, ha='center', color=COLORS['accent'])

    # Analog components
    draw_block(ax, 1.5, 4.0, 1.0, 0.6, 'Preamp\nAD620', COLORS['light_green'], fontsize=7)
    draw_block(ax, 3.2, 4.0, 1.0, 0.6, 'Inst Amp\nINA128', COLORS['light_green'], fontsize=7)
    draw_block(ax, 1.5, 3.0, 1.0, 0.6, 'Vref\n2.5V', COLORS['light_orange'], fontsize=7)
    draw_block(ax, 3.2, 3.0, 1.0, 0.6, 'ERT\nSource', COLORS['light_orange'], fontsize=7)

    # Digital section
    digital_box = FancyBboxPatch((4.5, 2.5), 3.5, 2.7, boxstyle="round,pad=0.02",
                                 facecolor='#e9d8fd', edgecolor=COLORS['purple'],
                                 linewidth=1.5, linestyle='--')
    ax.add_patch(digital_box)
    ax.text(6.25, 5.0, 'DIGITAL SECTION', fontsize=8, ha='center',
           fontweight='bold', color=COLORS['purple'])

    # Digital components
    draw_block(ax, 5.3, 4.0, 1.2, 0.6, 'MCU\nESP32', COLORS['light_purple'], fontsize=7)
    draw_block(ax, 7.0, 4.0, 1.0, 0.6, 'ADC\nADS1256', COLORS['light_blue'], fontsize=7)
    draw_block(ax, 5.3, 3.0, 1.0, 0.6, 'DDS\nAD9833', COLORS['light_blue'], fontsize=7)
    draw_block(ax, 7.0, 3.0, 1.0, 0.6, 'MUX\nCD4051', COLORS['light_blue'], fontsize=7)

    # Power section
    power_box = FancyBboxPatch((8.2, 2.5), 1.0, 2.7, boxstyle="round,pad=0.02",
                               facecolor='#fed7d7', edgecolor=COLORS['warning'],
                               linewidth=1.5, linestyle='--')
    ax.add_patch(power_box)
    ax.text(8.7, 5.0, 'PWR', fontsize=7, ha='center',
           fontweight='bold', color=COLORS['warning'])
    draw_block(ax, 8.7, 4.0, 0.6, 0.5, '5V', COLORS['light_red'], fontsize=7)
    draw_block(ax, 8.7, 3.3, 0.6, 0.5, '3.3V', COLORS['light_red'], fontsize=7)

    # Ground plane indication
    ax.fill_between([0.8, 8.8], [0.7, 0.7], [2.3, 2.3], color=COLORS['gray_light'], alpha=0.5)
    ax.text(4.8, 1.5, 'GROUND PLANE (Bottom Layer)', fontsize=9, ha='center',
           fontweight='bold', color=COLORS['gray_dark'])

    # Star ground point
    ax.add_patch(Circle((4.3, 1.0), 0.15, facecolor=COLORS['success'],
                        edgecolor='black', lw=2))
    ax.text(4.3, 0.6, 'Star GND', fontsize=7, ha='center', color=COLORS['success'])

    # Separation line between analog/digital grounds
    ax.plot([4.3, 4.3], [1.0, 2.3], 'g--', lw=2)
    ax.text(4.1, 1.8, 'A/D\nSplit', fontsize=6, ha='right', color=COLORS['success'])

    # Connectors
    draw_block(ax, 1.5, 1.5, 1.2, 0.5, 'Input\nConnector', COLORS['gray_light'], fontsize=7)
    draw_block(ax, 7.5, 1.5, 1.2, 0.5, 'USB\nPort', COLORS['gray_light'], fontsize=7)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_system_block_diagram():
    """
    Create complete HIRT system block diagram with Manhattan routing
    and enhanced visuals using WONG_PALETTE.
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

    # Internal connections using Manhattan routing
    # MCU to Mux (vertical bus)
    draw_manhattan_wire(ax, (2.55, 5.4), (2.55, 2.15), style='v_first',
                       color=COLORS['gray_dark'], linewidth=1.2)
    draw_connection(ax, (2.55, 2.3), (2.55, 2.15), style='->', color=COLORS['gray_dark'])

    # MCU to TX - Manhattan style (go down then left)
    draw_manhattan_wire(ax, (1.8, 5.4), (1.4, 4.9), style='v_first',
                       color=WONG_PALETTE['bluish_green'], linewidth=1.0)
    draw_connection(ax, (1.4, 5.0), (1.4, 4.9), style='->', color=WONG_PALETTE['bluish_green'])

    # MCU to RX - Manhattan style (go down then right)
    draw_manhattan_wire(ax, (3.3, 5.4), (3.7, 4.9), style='v_first',
                       color=WONG_PALETTE['blue'], linewidth=1.0)
    draw_connection(ax, (3.7, 5.0), (3.7, 4.9), style='->', color=WONG_PALETTE['blue'])

    # TX/RX to Mux with Manhattan routing
    draw_manhattan_wire(ax, (1.4, 4.1), (1.4, 2.15), style='v_first',
                       color=WONG_PALETTE['bluish_green'], linewidth=1.0)
    draw_manhattan_wire(ax, (3.7, 4.1), (3.7, 2.15), style='v_first',
                       color=WONG_PALETTE['blue'], linewidth=1.0)

    # ERT to Mux
    draw_manhattan_wire(ax, (1.4, 2.8), (1.2, 2.15), style='v_first',
                       color=WONG_PALETTE['orange'], linewidth=1.0)
    draw_manhattan_wire(ax, (3.7, 2.8), (3.9, 2.15), style='v_first',
                       color=WONG_PALETTE['orange'], linewidth=1.0)

    # Connection dots at junctions
    draw_connection_dot(ax, 2.55, 5.4, radius=0.06)

    # Trunk cable to Zone Hub (data bus representation)
    draw_manhattan_bus(ax, (4.8, 3.5), (6.0, 3.5), n_wires=4, spacing=0.1,
                      color=COLORS['gray_dark'], linewidth=1.0)
    ax.text(5.4, 4.0, 'Trunk\nCable', fontsize=8, ha='center', color=COLORS['gray_dark'], zorder=20)

    # Zone Hub
    zone_hub = FancyBboxPatch((6.0, 2.5), 2.0, 2.2, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor=COLORS['gray_dark'],
                              linewidth=1.5, zorder=5)
    ax.add_patch(zone_hub)
    ax.text(7.0, 4.5, 'Zone Hub\n(Passive)', fontsize=10, fontweight='bold',
           ha='center', color=COLORS['gray_dark'], zorder=20)
    ax.text(7.0, 3.2, 'DB25\nConnector\n+\nTerminals', fontsize=8, ha='center', zorder=20)

    # Probe connections from Zone Hub with Manhattan routing
    probe_y = [4.2, 3.7, 3.2, 2.7]
    probe_colors = [WONG_PALETTE['blue'], WONG_PALETTE['bluish_green'],
                   WONG_PALETTE['orange'], WONG_PALETTE['vermillion']]
    for i, (py, pc) in enumerate(zip(probe_y, probe_colors)):
        draw_manhattan_wire(ax, (8.0, py), (9.0, py - 0.2), style='h_first',
                           color=pc, linewidth=1.0)

    # Probes
    for i in range(4):
        py = 2.2 + i * 0.8
        draw_block(ax, 10.0, py, 1.8, 0.5, f'Probe {4-i}\n(Passive)',
                  COLORS['light_green'], fontsize=8)

    # Ground representation
    ax.fill_between([8.5, 11.5], [1.2, 1.2], [0.5, 0.5],
                   color=COLORS['ground_tan'], alpha=0.3, zorder=0)
    ax.axhline(1.2, xmin=0.71, xmax=0.95, color='#654321', lw=2, zorder=1)
    ax.text(10.0, 0.85, 'Ground', fontsize=9, ha='center', color='#654321', zorder=20)

    # USB/Power connection
    draw_block(ax, 2.55, 0.8, 1.8, 0.6, 'USB/Power', COLORS['light_red'], fontsize=9)
    draw_manhattan_wire(ax, (2.55, 1.1), (2.55, 1.45), style='v_first',
                       color=WONG_PALETTE['vermillion'], linewidth=1.5)
    draw_connection(ax, (2.55, 1.4), (2.55, 1.45), style='->', color=WONG_PALETTE['vermillion'])

    # Data output arrow
    draw_manhattan_wire(ax, (1.4, 5.8), (0.3, 5.8),
                       color=WONG_PALETTE['blue'], linewidth=2.0)
    draw_connection(ax, (0.4, 5.8), (0.2, 5.8), style='->', color=WONG_PALETTE['blue'], lw=2)
    ax.text(0.1, 6.0, 'Data\nOut', fontsize=9, ha='left',
           color=WONG_PALETTE['blue'], fontweight='bold', zorder=20)

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
