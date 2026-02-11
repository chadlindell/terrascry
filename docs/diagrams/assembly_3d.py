"""
HIRT Technical Manual - 3D Assembly Diagrams Module

Functions for creating 3D-style assembly visualizations that complement
the OpenSCAD renders. These provide annotated schematic views suitable
for documentation.

Provides:
- Exploded views with component callouts and dimensions
- Deployment scene showing probes in ground with hub
- Thread engagement detail for assembly precision
- Component identification diagrams
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, Arc, Wedge,
    FancyArrowPatch, Ellipse, PathPatch
)
from matplotlib.path import Path
import numpy as np
from io import BytesIO

# Import utility functions from parent module
from . import (
    COLORS as BASE_COLORS,
    WONG_PALETTE,
    LINE_WEIGHTS,
    draw_cylinder_gradient,
    draw_metal_surface,
    create_figure,
    setup_axes_clean,
    add_title,
    draw_dimension_line,
)

# Extended COLORS with additional colors needed for assembly diagrams
COLORS = dict(BASE_COLORS)
COLORS['copper'] = '#b87333'  # Copper wire color

# Import from mechanical module
from .mechanical import (
    draw_fiberglass_texture,
    draw_iso_thread_profile,
    draw_chamfer_detail,
    draw_circled_number,
    draw_dimension,
)


# =============================================================================
# Constants matching OpenSCAD design (in mm, scaled for display)
# =============================================================================

# Component dimensions (actual mm values from OpenSCAD)
ROD_OD = 16.0
ROD_ID = 12.0
SENSOR_BODY_LEN = 70.0
THREAD_LEN = 15.0
ROD_INSERT_DEPTH = 20.0
FLANGE_THICKNESS = 2.0
TIP_LENGTH = 30.0
JUNCTION_CAP_LEN = 35.0
THREAD_HOLE_DEPTH = 25.0

# Display scale (mm to display units)
SCALE = 0.05  # 1mm = 0.05 display units


def _mm(value):
    """Convert mm to display units."""
    return value * SCALE


# =============================================================================
# Main Diagram Functions
# =============================================================================

def create_probe_assembly_exploded():
    """
    Create exploded view diagram of probe assembly with callouts.

    Shows all probe components separated vertically with leader lines,
    dimensions, and part identification labels. Correctly shows fiberglass
    rod segments with male caps epoxied into each end.

    Structure from bottom to top:
    1. Probe Tip (female M12 socket)
    2. Rod Segment #1: Male cap (threads) + Fiberglass tube + Male cap (threads)
    3. Sensor Body (dual female M12 sockets)
    4. Rod Segment #2: Male cap (threads) + Fiberglass tube + Male cap (threads)
    5. Junction Cap (female M12 socket)

    Returns:
        BytesIO buffer containing PNG image
    """
    fig, ax = plt.subplots(figsize=(11, 16))
    ax.set_xlim(-4, 9)
    ax.set_ylim(-1, 18)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(2.5, 17.5, 'HIRT Probe Assembly - Exploded View',
            fontsize=14, fontweight='bold', ha='center', color=COLORS['primary'])

    # Component colors
    colors = {
        'tip': COLORS['gray_dark'],
        'cap': COLORS['secondary'],
        'sensor': '#38a169',  # Green for sensor bodies
        'junction': COLORS['gray_dark'],
        'rod': '#e8d4b8',  # Fiberglass tan
        'epoxy': '#8B4513',  # Epoxy zone indication
    }

    # Dimensions (scaled for visibility)
    cap_w = _mm(ROD_OD)
    rod_w = cap_w  # Same OD as caps
    thread_h = _mm(THREAD_LEN)
    insert_h = _mm(ROD_INSERT_DEPTH)
    flange_h = _mm(FLANGE_THICKNESS)
    sensor_h = _mm(SENSOR_BODY_LEN)
    tip_h = _mm(TIP_LENGTH)
    junc_h = _mm(JUNCTION_CAP_LEN)

    # Exploded gap between components
    gap = 0.6
    rod_gap = 0.15  # Smaller gap to show rod connects to caps

    # Center x position
    cx = 2.5

    # Draw components from bottom to top
    y_pos = 0
    step = 1

    # =========================================================================
    # 1. Probe Tip (female socket at top)
    # =========================================================================
    # Tapered nose
    nose_points = [
        (cx - cap_w/2, y_pos + tip_h * 0.4),
        (cx + cap_w/2, y_pos + tip_h * 0.4),
        (cx, y_pos),
    ]
    ax.add_patch(Polygon(nose_points, facecolor=colors['tip'], edgecolor='black', lw=1.5))
    # Body
    ax.add_patch(Rectangle((cx - cap_w/2, y_pos + tip_h * 0.4), cap_w, tip_h * 0.6,
                            facecolor=colors['tip'], edgecolor='black', lw=1.5))
    draw_circled_number(ax, -1.0, y_pos + tip_h/2, step, radius=0.3)
    ax.text(5.0, y_pos + tip_h/2, 'Probe Tip\n30mm tapered\nFemale M12 socket (top)',
            fontsize=9, va='center')
    ax.plot([cx + cap_w/2 + 0.1, 4.8], [y_pos + tip_h/2, y_pos + tip_h/2], 'k-', lw=0.5)
    y_pos += tip_h + gap
    step += 1

    # =========================================================================
    # 2. Rod Segment #1 (male cap + rod + male cap)
    # =========================================================================
    segment_start = y_pos

    # 2a. Male cap bottom (threads pointing down into tip)
    cap_total_h = thread_h + flange_h + insert_h
    draw_cylinder_gradient(ax, cx - cap_w/2, y_pos, cap_w, cap_total_h, colors['cap'], n_strips=12)
    ax.add_patch(Rectangle((cx - cap_w/2, y_pos), cap_w, cap_total_h,
                            facecolor='none', edgecolor='black', lw=1.5))
    # Thread marks
    for i in range(4):
        ty = y_pos + i * thread_h/4
        ax.plot([cx - cap_w/2, cx - cap_w/2 - 0.08], [ty, ty + 0.04], 'k-', lw=0.6)
        ax.plot([cx + cap_w/2, cx + cap_w/2 + 0.08], [ty, ty + 0.04], 'k-', lw=0.6)
    # Flange line
    ax.plot([cx - cap_w/2, cx + cap_w/2], [y_pos + thread_h, y_pos + thread_h], 'k-', lw=0.8)
    y_pos += cap_total_h + rod_gap

    # 2b. Fiberglass rod (solid, visible)
    rod_h = 2.0  # Display height for 50cm segment
    draw_fiberglass_texture(ax, cx - rod_w/2, y_pos, rod_w, rod_h)
    ax.add_patch(Rectangle((cx - rod_w/2, y_pos), rod_w, rod_h,
                            facecolor='none', edgecolor='black', lw=1.5))
    ax.text(cx, y_pos + rod_h/2, '50cm', ha='center', va='center',
            fontsize=8, fontweight='bold', color=COLORS['gray_dark'])
    y_pos += rod_h + rod_gap

    # 2c. Male cap top (threads pointing up into sensor body)
    draw_cylinder_gradient(ax, cx - cap_w/2, y_pos, cap_w, cap_total_h, colors['cap'], n_strips=12)
    ax.add_patch(Rectangle((cx - cap_w/2, y_pos), cap_w, cap_total_h,
                            facecolor='none', edgecolor='black', lw=1.5))
    # Thread marks on top portion
    thread_start = y_pos + insert_h + flange_h
    for i in range(4):
        ty = thread_start + i * thread_h/4
        ax.plot([cx - cap_w/2, cx - cap_w/2 - 0.08], [ty, ty + 0.04], 'k-', lw=0.6)
        ax.plot([cx + cap_w/2, cx + cap_w/2 + 0.08], [ty, ty + 0.04], 'k-', lw=0.6)
    ax.plot([cx - cap_w/2, cx + cap_w/2], [y_pos + insert_h, y_pos + insert_h], 'k-', lw=0.8)

    # Label for entire rod segment
    segment_mid = segment_start + (y_pos + cap_total_h - segment_start) / 2
    draw_circled_number(ax, -1.0, segment_mid, step, radius=0.3)
    ax.text(5.0, segment_mid, 'Rod Segment #1\nMale caps epoxied\ninto each end of\nfiberglass tube',
            fontsize=9, va='center')
    ax.plot([cx + cap_w/2 + 0.1, 4.8], [segment_mid, segment_mid], 'k-', lw=0.5)
    y_pos += cap_total_h + gap
    step += 1

    # =========================================================================
    # 3. Sensor Body (dual female M12 sockets)
    # =========================================================================
    draw_cylinder_gradient(ax, cx - cap_w/2, y_pos, cap_w, sensor_h, colors['sensor'], n_strips=12)
    ax.add_patch(Rectangle((cx - cap_w/2, y_pos), cap_w, sensor_h,
                            facecolor='none', edgecolor='black', lw=1.5))
    # ERT ring groove
    ert_y = y_pos + sensor_h * 0.25
    ax.add_patch(Rectangle((cx - cap_w/2 - 0.05, ert_y - 0.05), cap_w + 0.1, 0.1,
                            facecolor=WONG_PALETTE['orange'], edgecolor='black', lw=0.8))
    # MIT coil zone
    mit_y = y_pos + sensor_h * 0.6
    mit_h = sensor_h * 0.25
    ax.add_patch(Rectangle((cx - cap_w/2 + 0.02, mit_y), cap_w - 0.04, mit_h,
                            facecolor=COLORS['accent'], edgecolor='none', alpha=0.4))
    # Socket indicators (female threads)
    ax.plot([cx - cap_w/4, cx + cap_w/4], [y_pos + 0.05, y_pos + 0.05], 'k--', lw=0.6)
    ax.plot([cx - cap_w/4, cx + cap_w/4], [y_pos + sensor_h - 0.05, y_pos + sensor_h - 0.05], 'k--', lw=0.6)

    draw_circled_number(ax, -1.0, y_pos + sensor_h/2, step, radius=0.3)
    ax.text(5.0, y_pos + sensor_h/2, 'Sensor Body\n70mm length\nDual female M12 sockets\nERT ring + MIT coil',
            fontsize=9, va='center')
    ax.plot([cx + cap_w/2 + 0.1, 4.8], [y_pos + sensor_h/2, y_pos + sensor_h/2], 'k-', lw=0.5)
    y_pos += sensor_h + gap
    step += 1

    # =========================================================================
    # 4. Rod Segment #2 (male cap + rod + male cap)
    # =========================================================================
    segment_start = y_pos

    # 4a. Male cap bottom
    draw_cylinder_gradient(ax, cx - cap_w/2, y_pos, cap_w, cap_total_h, colors['cap'], n_strips=12)
    ax.add_patch(Rectangle((cx - cap_w/2, y_pos), cap_w, cap_total_h,
                            facecolor='none', edgecolor='black', lw=1.5))
    for i in range(4):
        ty = y_pos + i * thread_h/4
        ax.plot([cx - cap_w/2, cx - cap_w/2 - 0.08], [ty, ty + 0.04], 'k-', lw=0.6)
        ax.plot([cx + cap_w/2, cx + cap_w/2 + 0.08], [ty, ty + 0.04], 'k-', lw=0.6)
    ax.plot([cx - cap_w/2, cx + cap_w/2], [y_pos + thread_h, y_pos + thread_h], 'k-', lw=0.8)
    y_pos += cap_total_h + rod_gap

    # 4b. Fiberglass rod
    draw_fiberglass_texture(ax, cx - rod_w/2, y_pos, rod_w, rod_h)
    ax.add_patch(Rectangle((cx - rod_w/2, y_pos), rod_w, rod_h,
                            facecolor='none', edgecolor='black', lw=1.5))
    ax.text(cx, y_pos + rod_h/2, '50cm', ha='center', va='center',
            fontsize=8, fontweight='bold', color=COLORS['gray_dark'])
    y_pos += rod_h + rod_gap

    # 4c. Male cap top
    draw_cylinder_gradient(ax, cx - cap_w/2, y_pos, cap_w, cap_total_h, colors['cap'], n_strips=12)
    ax.add_patch(Rectangle((cx - cap_w/2, y_pos), cap_w, cap_total_h,
                            facecolor='none', edgecolor='black', lw=1.5))
    thread_start = y_pos + insert_h + flange_h
    for i in range(4):
        ty = thread_start + i * thread_h/4
        ax.plot([cx - cap_w/2, cx - cap_w/2 - 0.08], [ty, ty + 0.04], 'k-', lw=0.6)
        ax.plot([cx + cap_w/2, cx + cap_w/2 + 0.08], [ty, ty + 0.04], 'k-', lw=0.6)
    ax.plot([cx - cap_w/2, cx + cap_w/2], [y_pos + insert_h, y_pos + insert_h], 'k-', lw=0.8)

    segment_mid = segment_start + (y_pos + cap_total_h - segment_start) / 2
    draw_circled_number(ax, -1.0, segment_mid, step, radius=0.3)
    ax.text(5.0, segment_mid, 'Rod Segment #2\n(additional segments\ncan be added)',
            fontsize=9, va='center')
    ax.plot([cx + cap_w/2 + 0.1, 4.8], [segment_mid, segment_mid], 'k-', lw=0.5)
    y_pos += cap_total_h + gap
    step += 1

    # =========================================================================
    # 5. Junction Cap (female socket at bottom, cable exit at top)
    # =========================================================================
    boss_w = _mm(20)
    draw_cylinder_gradient(ax, cx - cap_w/2, y_pos, cap_w, junc_h * 0.6, colors['junction'], n_strips=12)
    draw_cylinder_gradient(ax, cx - boss_w/2, y_pos + junc_h * 0.6, boss_w, junc_h * 0.4,
                          colors['junction'], n_strips=15)
    ax.add_patch(Rectangle((cx - cap_w/2, y_pos), cap_w, junc_h * 0.6,
                            facecolor='none', edgecolor='black', lw=1.5))
    ax.add_patch(Rectangle((cx - boss_w/2, y_pos + junc_h * 0.6), boss_w, junc_h * 0.4,
                            facecolor='none', edgecolor='black', lw=1.5))
    # Socket indicator
    ax.plot([cx - cap_w/4, cx + cap_w/4], [y_pos + 0.05, y_pos + 0.05], 'k--', lw=0.6)

    draw_circled_number(ax, -1.0, y_pos + junc_h/2, step, radius=0.3)
    ax.text(5.0, y_pos + junc_h/2, 'Junction Cap\nFemale M12 socket (bottom)\nCable gland (top)\nWeather sealed',
            fontsize=9, va='center')
    ax.plot([cx + cap_w/2 + 0.1, 4.8], [y_pos + junc_h/2, y_pos + junc_h/2], 'k-', lw=0.5)

    # =========================================================================
    # Assembly arrows (between major components)
    # =========================================================================
    arrow_positions = [
        tip_h + gap/2,  # Between tip and rod segment 1
        tip_h + gap + cap_total_h + rod_gap + rod_h + rod_gap + cap_total_h + gap/2,  # Between rod 1 and sensor
    ]
    # More arrows would clutter; these show the key connections

    # =========================================================================
    # Legend
    # =========================================================================
    legend_x = -3.5
    legend_y = 1.0
    ax.add_patch(FancyBboxPatch((legend_x - 0.2, legend_y - 0.3), 2.6, 3.8,
                                boxstyle="round,pad=0.05", facecolor='white',
                                edgecolor=COLORS['gray_med'], lw=1))
    ax.text(legend_x, legend_y + 3.2, 'Legend', fontsize=10, fontweight='bold')
    legend_items = [
        (colors['tip'], 'Tip/Junction (3D printed)'),
        (colors['cap'], 'Male Rod Caps (3D printed)'),
        (colors['rod'], 'Fiberglass Tube (G10)'),
        (colors['sensor'], 'Sensor Body (3D printed)'),
        (WONG_PALETTE['orange'], 'ERT Ring Electrode'),
        (COLORS['accent'], 'MIT Coil Winding Zone'),
    ]
    for i, (color, label) in enumerate(legend_items):
        y = legend_y + 2.6 - i * 0.45
        ax.add_patch(Rectangle((legend_x, y - 0.1), 0.25, 0.25,
                                facecolor=color, edgecolor='black', lw=0.5))
        ax.text(legend_x + 0.4, y + 0.02, label, fontsize=7, va='center')

    # =========================================================================
    # Dimension: overall width
    # =========================================================================
    draw_dimension(ax, (cx - cap_w/2, -0.5), (cx + cap_w/2, -0.5), -0.3, '16mm OD')

    # =========================================================================
    # Notes
    # =========================================================================
    ax.text(cx, -0.9, 'Male caps are epoxied into fiberglass tube ends before assembly',
            ha='center', fontsize=8, style='italic', color=COLORS['gray_med'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_deployment_scene():
    """
    Create deployment scene showing probes in ground with hub and cables.

    Shows a top-down view with 4 probes in a grid pattern, cables running
    to a central hub, and ground texture. Suitable for conveying the
    system-level deployment concept.

    Returns:
        BytesIO buffer containing PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(0, 3.7, 'HIRT Deployment Configuration - Top View',
            fontsize=14, fontweight='bold', ha='center', color=COLORS['primary'])

    # Ground plane (subtle texture)
    ground = Rectangle((-4.5, -3.5), 9, 7, facecolor='#d4a373', alpha=0.3, edgecolor='none')
    ax.add_patch(ground)

    # Grid lines (subtle)
    for x in np.arange(-4, 5, 1):
        ax.axvline(x, color='#654321', lw=0.3, alpha=0.3)
    for y in np.arange(-3, 4, 1):
        ax.axhline(y, color='#654321', lw=0.3, alpha=0.3)

    # Probe positions (2x2 grid, 1m spacing)
    probe_positions = [(-1.5, 1.5), (1.5, 1.5), (-1.5, -1.5), (1.5, -1.5)]
    probe_radius = 0.25

    for i, (px, py) in enumerate(probe_positions):
        # Probe hole (darker circle)
        ax.add_patch(Circle((px, py), probe_radius * 1.3,
                            facecolor='#4a3520', edgecolor='#2d1f12', lw=2))
        # Probe top (junction cap visible)
        ax.add_patch(Circle((px, py), probe_radius,
                            facecolor=COLORS['gray_dark'], edgecolor='black', lw=1.5))
        # Cable gland boss
        ax.add_patch(Circle((px, py), probe_radius * 0.5,
                            facecolor=COLORS['gray_med'], edgecolor='black', lw=1))
        # Probe label
        ax.text(px, py - 0.5, f'P{i+1}', ha='center', fontsize=9, fontweight='bold')

        # Cable running to hub
        hub_x, hub_y = 0, 0
        # Curved cable path
        mid_x = (px + hub_x) / 2 + 0.3 * np.sign(px)
        mid_y = (py + hub_y) / 2 + 0.3 * np.sign(py)
        ax.plot([px, mid_x, hub_x], [py, mid_y, hub_y],
                color='black', lw=3, solid_capstyle='round', zorder=1)
        ax.plot([px, mid_x, hub_x], [py, mid_y, hub_y],
                color=COLORS['gray_light'], lw=2, solid_capstyle='round', zorder=2)

    # Central hub
    hub = FancyBboxPatch((-0.6, -0.4), 1.2, 0.8, boxstyle="round,pad=0.05",
                         facecolor=COLORS['gray_dark'], edgecolor='black', lw=2, zorder=5)
    ax.add_patch(hub)
    ax.text(0, 0, 'HUB', ha='center', va='center', fontsize=10,
            fontweight='bold', color='white', zorder=6)

    # Dimension annotations
    # Probe spacing
    ax.annotate('', xy=(-1.5, 2.3), xytext=(1.5, 2.3),
                arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=1.5))
    ax.text(0, 2.5, '1.0m (typical)', ha='center', fontsize=9, fontweight='bold',
            color=COLORS['primary'])

    # Grid indicator
    ax.annotate('', xy=(-1.5, -2.8), xytext=(1.5, -2.8),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray_med'], lw=1))
    ax.text(0, -3.1, 'Survey grid spacing adjustable', ha='center', fontsize=8,
            color=COLORS['gray_med'])

    # Legend
    ax.add_patch(FancyBboxPatch((-4.3, 1.5), 2.0, 1.8, boxstyle="round,pad=0.03",
                                facecolor='white', edgecolor=COLORS['gray_med'], lw=1, alpha=0.9))
    ax.text(-3.3, 3.0, 'Legend', fontsize=9, fontweight='bold')
    ax.add_patch(Circle((-4.0, 2.5), 0.15, facecolor=COLORS['gray_dark'], edgecolor='black', lw=1))
    ax.text(-3.7, 2.5, 'Probe (top view)', fontsize=8, va='center')
    ax.plot([-4.1, -3.8], [2.1, 2.1], color='black', lw=2)
    ax.text(-3.7, 2.1, 'Cable', fontsize=8, va='center')
    ax.add_patch(Rectangle((-4.1, 1.65), 0.3, 0.2, facecolor=COLORS['gray_dark'], edgecolor='black', lw=1))
    ax.text(-3.7, 1.75, 'Hub', fontsize=8, va='center')

    # Scale bar
    ax.plot([3, 4], [-3.2, -3.2], 'k-', lw=3)
    ax.text(3.5, -3.4, '1 meter', ha='center', fontsize=8)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_thread_engagement_detail():
    """
    Create detailed diagram showing thread engagement between components.

    Shows cross-section of male cap threading into sensor body female socket,
    with critical dimensions and engagement specifications.

    Returns:
        BytesIO buffer containing PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(-1.5, 6)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(2.25, 4.7, 'Thread Engagement Detail - M12x1.75',
            fontsize=12, fontweight='bold', ha='center', color=COLORS['primary'])

    # Component dimensions (scaled)
    body_w = 1.0  # 16mm OD scaled
    thread_w = 0.75  # ~12mm thread diameter scaled
    socket_depth = 1.5  # 25mm scaled
    thread_len = 0.9  # 15mm scaled
    engagement = 0.6  # 10mm engagement scaled

    # Female component (sensor body - left side, sectioned)
    ax.add_patch(Rectangle((0, 0), body_w * 0.5, 4,
                            facecolor='#38a169', edgecolor='black', lw=1.5, alpha=0.8))
    ax.add_patch(Rectangle((0, 0), body_w * 0.5, socket_depth,
                            facecolor='white', edgecolor='black', lw=1))
    # Thread profile on socket
    draw_iso_thread_profile(ax, 0.35, socket_depth - 0.1, socket_depth - 0.2,
                           pitch=1.75, side='right', show_annotations=False)
    ax.text(-0.3, 2, 'Sensor\nBody', ha='right', va='center', fontsize=9, fontweight='bold',
            color='#38a169')

    # Male component (rod cap - shown engaged)
    engaged_start = socket_depth - engagement
    ax.add_patch(Rectangle((0.6, engaged_start), thread_w * 0.5, thread_len,
                            facecolor=COLORS['secondary'], edgecolor='black', lw=1.5))
    # Male thread profile
    draw_iso_thread_profile(ax, 0.6, engaged_start + thread_len - 0.1, thread_len - 0.2,
                           pitch=1.75, side='left', show_annotations=False)
    # Rod body above thread
    ax.add_patch(Rectangle((0.5, engaged_start + thread_len), body_w * 0.5, 2,
                            facecolor=COLORS['secondary'], edgecolor='black', lw=1.5, alpha=0.6))
    ax.text(1.2, 3, 'Male\nRod Cap', ha='left', va='center', fontsize=9, fontweight='bold',
            color=COLORS['secondary'])

    # Engagement zone highlight
    ax.add_patch(Rectangle((0.35, engaged_start), 0.3, engagement,
                            facecolor=COLORS['accent'], edgecolor='none', alpha=0.3))

    # Center wire channel indication
    ax.add_patch(Rectangle((0.42, 0), 0.08, 4, facecolor='white', edgecolor=COLORS['gray_light'], lw=0.5))
    ax.annotate('6mm wire\nchannel', xy=(0.46, 0.3), xytext=(1.8, 0.3),
                fontsize=7, va='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_med'], lw=0.8))

    # Dimension annotations
    # Engagement depth
    ax.annotate('', xy=(2.5, engaged_start), xytext=(2.5, socket_depth),
                arrowprops=dict(arrowstyle='<->', color=COLORS['warning'], lw=1.5))
    ax.text(2.7, engaged_start + engagement/2, '10mm\nmin engagement',
            fontsize=8, va='center', color=COLORS['warning'], fontweight='bold')

    # Socket depth
    ax.annotate('', xy=(3.5, 0), xytext=(3.5, socket_depth),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray_med'], lw=1))
    ax.text(3.7, socket_depth/2, '25mm\nsocket depth', fontsize=8, va='center',
            color=COLORS['gray_med'])

    # Thread pitch callout
    ax.annotate('P = 1.75mm\n(~6 threads engaged)', xy=(0.5, engaged_start + 0.3),
                xytext=(-1.0, 0.8), fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    # Specifications box
    spec_box = FancyBboxPatch((3.8, 2.5), 2.0, 1.8, boxstyle="round,pad=0.05",
                               facecolor='#f7fafc', edgecolor=COLORS['primary'], lw=1)
    ax.add_patch(spec_box)
    ax.text(4.8, 4.0, 'Thread Specifications', fontsize=9, fontweight='bold',
            ha='center', color=COLORS['primary'])
    specs = [
        'M12 x 1.75 ISO Metric',
        'Class 6H/6g fit',
        '60 deg thread form',
        'Min engagement: 10mm',
        'Hand-tight + 1/4 turn',
    ]
    for i, spec in enumerate(specs):
        ax.text(3.9, 3.6 - i * 0.25, '- ' + spec, fontsize=7)

    # O-ring location callout
    ax.add_patch(Circle((0.46, engaged_start - 0.1), 0.06,
                        facecolor='black', edgecolor='none'))
    ax.annotate('O-ring seal\n(10mm x 1.5mm)', xy=(0.46, engaged_start - 0.1),
                xytext=(-0.8, engaged_start + 0.3), fontsize=7, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_sensor_body_detail():
    """
    Create detailed diagram of the sensor body showing all features.

    Cross-section view highlighting ERT groove, MIT coil zone, thread
    sockets, and wire routing.

    Returns:
        BytesIO buffer containing PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-3, 7)
    ax.set_ylim(-1, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(2, 5.7, 'Sensor Body Detail - Cross Section',
            fontsize=12, fontweight='bold', ha='center', color=COLORS['primary'])

    # Dimensions (scaled)
    body_len = _mm(SENSOR_BODY_LEN) * 1.5  # Scaled up for visibility
    body_w = _mm(ROD_OD) * 1.5
    socket_depth = _mm(THREAD_HOLE_DEPTH) * 1.5

    # Center the body
    cx = 2
    cy = 2.5

    # Main body (cross-section, half shown)
    body_left = cx - body_w/2
    body_bottom = cy - body_len/2

    # Full body outline
    ax.add_patch(Rectangle((body_left, body_bottom), body_w, body_len,
                            facecolor='#38a169', edgecolor='black', lw=2, alpha=0.8))

    # Bottom socket
    socket_w = _mm(THREAD_DIA_MINOR if 'THREAD_DIA_MINOR' in dir() else 10.4) * 1.5
    ax.add_patch(Rectangle((cx - socket_w/2, body_bottom), socket_w, socket_depth,
                            facecolor='white', edgecolor='black', lw=1))

    # Top socket
    ax.add_patch(Rectangle((cx - socket_w/2, cy + body_len/2 - socket_depth),
                            socket_w, socket_depth,
                            facecolor='white', edgecolor='black', lw=1))

    # Center wire channel
    channel_w = _mm(6) * 1.5
    ax.add_patch(Rectangle((cx - channel_w/2, body_bottom + socket_depth),
                            channel_w, body_len - 2*socket_depth,
                            facecolor='white', edgecolor=COLORS['gray_light'], lw=0.5))

    # Sensor zone (between sockets)
    sensor_zone_start = body_bottom + socket_depth
    sensor_zone_len = body_len - 2*socket_depth

    # ERT ring groove (at 1/4 position)
    ert_y = sensor_zone_start + sensor_zone_len * 0.25
    ert_h = 0.15
    ax.add_patch(Rectangle((body_left - 0.1, ert_y - ert_h/2), body_w + 0.2, ert_h,
                            facecolor=WONG_PALETTE['orange'], edgecolor='black', lw=1))

    # MIT coil zone (at 3/4 position)
    mit_y = sensor_zone_start + sensor_zone_len * 0.65
    mit_h = sensor_zone_len * 0.25
    ax.add_patch(Rectangle((body_left + 0.1, mit_y), body_w - 0.2, mit_h,
                            facecolor=COLORS['accent'], edgecolor=COLORS['primary'],
                            lw=1, alpha=0.4))
    # Coil winding indication
    for i in range(6):
        coil_y = mit_y + 0.1 + i * (mit_h - 0.2) / 5
        ax.plot([body_left + 0.15, cx - channel_w/2 - 0.05], [coil_y, coil_y],
                color=COLORS['copper'], lw=1.5)
        ax.plot([cx + channel_w/2 + 0.05, body_left + body_w - 0.15], [coil_y, coil_y],
                color=COLORS['copper'], lw=1.5)

    # Radial wire entry holes
    hole_positions = [
        (ert_y, 0),          # ERT wire
        (mit_y + 0.1, 60),   # MIT start
        (mit_y + mit_h - 0.1, 120),  # MIT end
    ]
    for hy, angle in hole_positions:
        hx = cx + body_w/2 * np.cos(np.radians(angle)) * 0.7
        ax.add_patch(Circle((body_left + body_w, hy), 0.06,
                            facecolor='white', edgecolor='black', lw=0.5))

    # Orientation flat indicator
    ax.add_patch(Rectangle((body_left + body_w - 0.08, cy - 0.15), 0.15, 0.3,
                            facecolor=COLORS['gray_med'], edgecolor='black', lw=0.5))

    # Dimension annotations
    # Total length
    ax.annotate('', xy=(-1, body_bottom), xytext=(-1, body_bottom + body_len),
                arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=1.5))
    ax.text(-1.3, cy, '70mm', fontsize=10, va='center', ha='right', fontweight='bold',
            color=COLORS['primary'], rotation=90)

    # Socket depth
    ax.annotate('', xy=(4.5, body_bottom), xytext=(4.5, body_bottom + socket_depth),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray_med'], lw=1))
    ax.text(4.7, body_bottom + socket_depth/2, '25mm\nsocket', fontsize=8, va='center')

    # OD
    ax.annotate('', xy=(body_left, body_bottom - 0.3), xytext=(body_left + body_w, body_bottom - 0.3),
                arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=1))
    ax.text(cx, body_bottom - 0.5, '16mm OD', fontsize=9, ha='center', fontweight='bold')

    # Feature callouts
    callouts = [
        (ert_y, 'ERT Ring Groove\n4mm wide x 1mm deep'),
        (mit_y + mit_h/2, 'MIT Coil Zone\n15mm winding area'),
        (sensor_zone_start + sensor_zone_len/2, '6mm wire channel'),
        (body_bottom + socket_depth/2, 'M12x1.75 female\nthread socket'),
    ]

    callout_x = 5.0
    for cy_pos, text in callouts:
        ax.plot([body_left + body_w + 0.1, callout_x - 0.1], [cy_pos, cy_pos], 'k-', lw=0.5)
        ax.text(callout_x, cy_pos, text, fontsize=8, va='center')

    # Legend
    ax.add_patch(FancyBboxPatch((-2.5, 0), 2.0, 1.5, boxstyle="round,pad=0.03",
                                facecolor='white', edgecolor=COLORS['gray_med'], lw=1))
    ax.text(-1.5, 1.3, 'Features', fontsize=9, fontweight='bold', ha='center')
    legend_items = [
        ('#38a169', 'PETG Body'),
        (WONG_PALETTE['orange'], 'ERT Electrode'),
        (COLORS['accent'], 'MIT Zone'),
        (COLORS['copper'], 'Coil Winding'),
    ]
    for i, (color, label) in enumerate(legend_items):
        y = 1.0 - i * 0.3
        ax.add_patch(Rectangle((-2.4, y - 0.08), 0.2, 0.16,
                                facecolor=color, edgecolor='black', lw=0.5))
        ax.text(-2.1, y, label, fontsize=7, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    'create_probe_assembly_exploded',
    'create_deployment_scene',
    'create_thread_engagement_detail',
    'create_sensor_body_detail',
]
