"""
HIRT PDF Generator - Mechanical Diagrams Module

Functions for creating CAD-style views, exploded assemblies, and
mechanical detail drawings.

Enhanced with:
- 3D cylinder gradient shading
- ISO-standard thread profiles (60-degree form)
- Realistic material textures (fiberglass weave, metal surfaces)
- CAD-style section indicators and annotations
"""

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, Arc, Wedge,
    FancyArrowPatch, Ellipse, PathPatch, FancyBboxPatch
)
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import numpy as np
from io import BytesIO

# Import utility functions from parent module
from . import (
    draw_cylinder_gradient,
    draw_metal_surface,
    WONG_PALETTE,
    COLORS as BASE_COLORS,
)

# Color palette - extended for mechanical diagrams
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
    'fiberglass': '#e8d4b8',
    'ferrite': '#333333',
    'copper': '#b87333',
    'steel': '#71797E',
    # Additional colors for enhanced diagrams
    'epoxy': '#c9a959',          # Amber/honey color for epoxy
    'bobbin': '#f5f5dc',         # Beige for coil bobbins
    'section_line': '#0000ff',   # Blue for CAD section lines
    'chamfer': '#a0a0a0',        # Gray for chamfer details
}


# =============================================================================
# Enhanced Utility Functions for Realistic Rendering
# =============================================================================

def draw_fiberglass_texture(ax, x, y, width, height, base_color=None, weave_spacing=0.08):
    """
    Draw a fiberglass surface with weave pattern suggestion.

    Args:
        ax: matplotlib Axes object
        x, y: Lower-left corner
        width, height: Dimensions
        base_color: Base color (default: fiberglass tan)
        weave_spacing: Spacing between weave lines
    """
    if base_color is None:
        base_color = COLORS['fiberglass']

    # Draw base with gradient shading
    draw_cylinder_gradient(ax, x, y, width, height, base_color, light_angle=60, n_strips=15)

    # Add subtle diagonal weave pattern (45-degree crosshatch)
    # First direction
    spacing = weave_spacing
    for offset in np.arange(-height, width + height, spacing * 2):
        x1 = x + max(0, offset)
        y1 = y + max(0, -offset)
        x2 = x + min(width, offset + height)
        y2 = y + min(height, height - offset)
        if x1 < x + width and x2 > x and y1 < y + height and y2 > y:
            ax.plot([x1, x2], [y1, y2], color='#d4c4a8', lw=0.3, alpha=0.4)

    # Second direction (perpendicular)
    for offset in np.arange(0, width + height, spacing * 2):
        x1 = x + min(width, offset)
        y1 = y + max(0, offset - width)
        x2 = x + max(0, offset - height)
        y2 = y + min(height, offset)
        if x1 > x and x2 < x + width and y1 < y + height and y2 > y:
            ax.plot([x1, x2], [y1, y2], color='#c4b498', lw=0.3, alpha=0.3)


def draw_section_line_indicator(ax, x, y1, y2, label='A', fontsize=10):
    """
    Draw a CAD-style section line indicator with arrows and circles.

    Args:
        ax: matplotlib Axes object
        x: X position of section line
        y1, y2: Start and end Y positions
        label: Section label (e.g., 'A' for Section A-A)
    """
    # Section line (dashed with alternating long-short pattern)
    ax.plot([x, x], [y1, y2], color=COLORS['section_line'], lw=1.5,
            linestyle=(0, (10, 3, 2, 3)), zorder=10)

    # Circle indicators at ends
    circle_radius = 0.15
    for cy, direction in [(y1, -1), (y2, 1)]:
        # Circle with label
        circle = Circle((x, cy), circle_radius, facecolor='white',
                        edgecolor=COLORS['section_line'], lw=1.5, zorder=11)
        ax.add_patch(circle)
        ax.text(x, cy, label, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=COLORS['section_line'], zorder=12)

        # Arrow indicating viewing direction
        arrow_y = cy + direction * 0.25
        ax.annotate('', xy=(x + direction * 0.3, arrow_y),
                   xytext=(x, arrow_y),
                   arrowprops=dict(arrowstyle='->', color=COLORS['section_line'], lw=1.5),
                   zorder=11)


def draw_iso_thread_profile(ax, x, y, length, pitch=1.75, major_d=12, side='right',
                            show_annotations=True, thread_angle=60):
    """
    Draw an ISO-standard thread profile with accurate 60-degree thread form.

    Args:
        ax: matplotlib Axes object
        x, y: Start position (top of thread)
        length: Thread length in display units
        pitch: Thread pitch (for annotation)
        major_d: Major diameter (for annotation)
        side: 'right' or 'left'
        show_annotations: Whether to show thread geometry annotations
        thread_angle: Thread included angle (60 degrees for ISO metric)

    Returns:
        Dictionary with thread geometry details
    """
    scale = 0.1
    n_threads = int(length / (pitch * scale))

    # ISO metric thread geometry
    # H = 0.866 * P (theoretical thread height)
    # Actual engagement depth = 0.6495 * P
    H = 0.866 * pitch * scale
    thread_depth = 0.6495 * pitch * scale
    half_angle = thread_angle / 2  # 30 degrees for ISO

    thread_points = []

    for i in range(n_threads):
        base_y = y - i * pitch * scale

        if side == 'right':
            # Root
            thread_points.append((x, base_y))
            # Crest (peak)
            thread_points.append((x + thread_depth, base_y - pitch * scale * 0.25))
            # Root of next thread
            thread_points.append((x, base_y - pitch * scale * 0.5))
        else:
            thread_points.append((x, base_y))
            thread_points.append((x - thread_depth, base_y - pitch * scale * 0.25))
            thread_points.append((x, base_y - pitch * scale * 0.5))

    if thread_points:
        xs, ys = zip(*thread_points)
        ax.plot(xs, ys, 'k-', lw=1.0)

        # Add root radius (small fillet at thread root)
        for i in range(n_threads):
            base_y = y - i * pitch * scale
            root_y = base_y - pitch * scale * 0.5
            fillet_r = 0.01
            if side == 'right':
                arc = Arc((x + fillet_r, root_y), fillet_r * 2, fillet_r * 2,
                         angle=0, theta1=90, theta2=180, color='k', lw=0.5)
            else:
                arc = Arc((x - fillet_r, root_y), fillet_r * 2, fillet_r * 2,
                         angle=0, theta1=0, theta2=90, color='k', lw=0.5)
            ax.add_patch(arc)

    # Thread geometry annotations
    if show_annotations and n_threads >= 2:
        anno_y = y - pitch * scale * 0.5
        anno_x = x + (0.3 if side == 'right' else -0.3)

        # Pitch dimension
        ax.annotate('', xy=(anno_x, anno_y), xytext=(anno_x, anno_y - pitch * scale),
                   arrowprops=dict(arrowstyle='<->', color=COLORS['warning'], lw=0.8))

        # 60-degree angle indicator
        if side == 'right':
            angle_x = x + thread_depth * 0.5
            angle_y = anno_y - pitch * scale * 0.125
            ax.text(angle_x + 0.15, angle_y, f'{thread_angle}deg',
                   fontsize=6, color=COLORS['gray_med'], style='italic')

    return {
        'pitch': pitch,
        'major_diameter': major_d,
        'thread_depth': thread_depth / scale,
        'n_threads': n_threads,
        'engagement_depth': thread_depth / scale * 0.75,  # 75% engagement typical
    }


def draw_chamfer_detail(ax, x, y, chamfer_size, direction='down-right'):
    """
    Draw a chamfer/lead-in detail at thread start.

    Args:
        ax: matplotlib Axes object
        x, y: Corner position
        chamfer_size: Size of chamfer (45-degree)
        direction: 'down-right', 'down-left', 'up-right', 'up-left'
    """
    if direction == 'down-right':
        points = [(x, y), (x + chamfer_size, y - chamfer_size), (x, y - chamfer_size)]
    elif direction == 'down-left':
        points = [(x, y), (x - chamfer_size, y - chamfer_size), (x, y - chamfer_size)]
    elif direction == 'up-right':
        points = [(x, y), (x + chamfer_size, y + chamfer_size), (x, y + chamfer_size)]
    else:  # up-left
        points = [(x, y), (x - chamfer_size, y + chamfer_size), (x, y + chamfer_size)]

    ax.add_patch(Polygon(points, facecolor=COLORS['chamfer'], edgecolor='black', lw=0.5))

    # Chamfer annotation line
    mid_x = sum(p[0] for p in points) / 3
    mid_y = sum(p[1] for p in points) / 3
    ax.plot([mid_x - chamfer_size * 0.3, mid_x + chamfer_size * 0.3],
            [mid_y - chamfer_size * 0.3, mid_y + chamfer_size * 0.3],
            color=COLORS['gray_med'], lw=0.5, linestyle=':')


def draw_coil_winding_layers(ax, center_x, y_start, y_end, inner_radius, outer_radius,
                              n_layers=3, turns_per_layer=8, wire_diameter=0.2):
    """
    Draw multi-layer coil winding with individual wire visualization.

    Args:
        ax: matplotlib Axes object
        center_x: X position of coil center
        y_start, y_end: Vertical extent of winding area
        inner_radius: Inner winding radius (bobbin OD)
        outer_radius: Outer winding radius
        n_layers: Number of winding layers
        turns_per_layer: Turns per layer (for visualization)
        wire_diameter: Wire diameter in display units
    """
    winding_height = abs(y_end - y_start)
    wire_spacing = winding_height / turns_per_layer
    layer_thickness = (outer_radius - inner_radius) / n_layers

    # Draw each layer
    for layer in range(n_layers):
        layer_radius = inner_radius + (layer + 0.5) * layer_thickness

        # Alternate starting position for hexagonal packing effect
        y_offset = wire_spacing * 0.5 if layer % 2 == 1 else 0

        for turn in range(turns_per_layer):
            wire_y = y_start + turn * wire_spacing + y_offset
            if wire_y > y_end:
                break

            # Left side of coil
            left_x = center_x - layer_radius
            # Right side of coil
            right_x = center_x + layer_radius

            # Color gradient: inner layers slightly darker
            brightness = 0.85 + 0.15 * (layer / max(1, n_layers - 1))
            wire_color = tuple(c * brightness for c in mcolors.to_rgb(COLORS['copper']))

            # Draw wire cross-sections
            ax.add_patch(Circle((left_x, wire_y), wire_diameter / 2,
                               facecolor=wire_color, edgecolor='#8b5a2b', lw=0.3))
            ax.add_patch(Circle((right_x, wire_y), wire_diameter / 2,
                               facecolor=wire_color, edgecolor='#8b5a2b', lw=0.3))


def draw_bobbin_with_flanges(ax, center_x, y_bottom, y_top, tube_od, flange_od,
                              flange_thickness=0.1, tube_id=None):
    """
    Draw a coil bobbin/former with flanges.

    Args:
        ax: matplotlib Axes object
        center_x: X position of bobbin center
        y_bottom, y_top: Vertical extent
        tube_od: Tube outer diameter
        flange_od: Flange outer diameter
        flange_thickness: Thickness of flanges
        tube_id: Tube inner diameter (for hollow core)
    """
    tube_radius = tube_od / 2
    flange_radius = flange_od / 2
    height = y_top - y_bottom

    # Main tube body
    draw_cylinder_gradient(ax, center_x - tube_radius, y_bottom + flange_thickness,
                          tube_od, height - 2 * flange_thickness,
                          COLORS['bobbin'], light_angle=60, n_strips=10)

    # Bottom flange
    ax.add_patch(Rectangle((center_x - flange_radius, y_bottom),
                           flange_od, flange_thickness,
                           facecolor=COLORS['bobbin'], edgecolor='black', lw=0.8))

    # Top flange
    ax.add_patch(Rectangle((center_x - flange_radius, y_top - flange_thickness),
                           flange_od, flange_thickness,
                           facecolor=COLORS['bobbin'], edgecolor='black', lw=0.8))

    # Hollow center if specified
    if tube_id is not None:
        inner_radius = tube_id / 2
        ax.add_patch(Rectangle((center_x - inner_radius, y_bottom),
                               tube_id, height,
                               facecolor='white', edgecolor=COLORS['gray_light'], lw=0.5))


def draw_epoxy_potting_level(ax, x_left, x_right, y_bottom, fill_level, max_level):
    """
    Draw epoxy potting indication with fill level.

    Args:
        ax: matplotlib Axes object
        x_left, x_right: Horizontal bounds
        y_bottom: Bottom of potting area
        fill_level: Current fill level (0-1)
        max_level: Maximum fill height
    """
    fill_height = fill_level * max_level
    width = x_right - x_left

    # Epoxy fill
    ax.add_patch(Rectangle((x_left, y_bottom), width, fill_height,
                           facecolor=COLORS['epoxy'], edgecolor='none', alpha=0.6))

    # Meniscus effect at top
    meniscus_height = 0.02
    xs = np.linspace(x_left, x_right, 20)
    ys = y_bottom + fill_height + meniscus_height * np.sin(np.pi * (xs - x_left) / width)
    ax.fill_between(xs, [y_bottom + fill_height] * len(xs), ys,
                   color=COLORS['epoxy'], alpha=0.4)

    # Fill level indicator line
    ax.plot([x_right + 0.05, x_right + 0.15],
            [y_bottom + fill_height, y_bottom + fill_height],
            color=COLORS['epoxy'], lw=1, linestyle='--')
    ax.text(x_right + 0.18, y_bottom + fill_height,
            f'Epoxy fill\n({int(fill_level*100)}%)',
            fontsize=6, va='center', color=COLORS['gray_dark'])


def draw_circled_number(ax, x, y, number, radius=0.25, bg_color=None, text_color='white'):
    """
    Draw a circled assembly step number.

    Args:
        ax: matplotlib Axes object
        x, y: Center position
        number: Step number to display
        radius: Circle radius
        bg_color: Background color (default: secondary blue)
        text_color: Text color
    """
    if bg_color is None:
        bg_color = COLORS['secondary']

    ax.add_patch(Circle((x, y), radius, facecolor=bg_color,
                       edgecolor='black', lw=1.5, zorder=20))
    ax.text(x, y, str(number), ha='center', va='center',
           fontsize=10, fontweight='bold', color=text_color, zorder=21)


def draw_torque_spec(ax, x, y, torque_nm, direction='right'):
    """
    Draw a torque specification annotation.

    Args:
        ax: matplotlib Axes object
        x, y: Position near threaded joint
        torque_nm: Torque value in N-m
        direction: 'left' or 'right' for annotation placement
    """
    # Torque symbol (curved arrow)
    offset = 0.3 if direction == 'right' else -0.3

    # Arc for torque symbol
    arc = Arc((x + offset, y), 0.2, 0.2, angle=0,
              theta1=0 if direction == 'right' else 180,
              theta2=270 if direction == 'right' else 450,
              color=COLORS['warning'], lw=1.5)
    ax.add_patch(arc)

    # Arrow head
    arrow_x = x + offset + (0.1 if direction == 'right' else -0.1)
    ax.annotate('', xy=(arrow_x, y + 0.05), xytext=(arrow_x, y - 0.05),
               arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=1))

    # Torque value
    text_x = x + (0.5 if direction == 'right' else -0.5)
    ax.text(text_x, y, f'{torque_nm} N-m', fontsize=7,
            fontweight='bold', color=COLORS['warning'],
            ha='left' if direction == 'right' else 'right', va='center',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                     edgecolor=COLORS['warning'], alpha=0.9))


def draw_alignment_indicator(ax, x1, x2, y, style='centerline'):
    """
    Draw alignment feature indicators.

    Args:
        ax: matplotlib Axes object
        x1, x2: Horizontal extent
        y: Y position
        style: 'centerline', 'keyway', or 'pin'
    """
    if style == 'centerline':
        # Center-line symbol (alternating long-short dash)
        ax.axhline(y, xmin=(x1 - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                  xmax=(x2 - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
                  color=COLORS['accent'], lw=0.8, linestyle=(0, (10, 3, 2, 3)))
        # CL symbol
        mid_x = (x1 + x2) / 2
        ax.text(mid_x, y + 0.08, 'CL', fontsize=6, ha='center',
               color=COLORS['accent'], style='italic')
    elif style == 'keyway':
        # Small rectangle for keyway
        key_width = (x2 - x1) * 0.1
        key_height = 0.05
        ax.add_patch(Rectangle((x1, y - key_height/2), key_width, key_height,
                               facecolor=COLORS['gray_dark'], edgecolor='black', lw=0.5))
    elif style == 'pin':
        # Alignment pin holes
        pin_radius = 0.03
        ax.add_patch(Circle((x1, y), pin_radius, facecolor='white',
                           edgecolor='black', lw=0.5))
        ax.add_patch(Circle((x2, y), pin_radius, facecolor='white',
                           edgecolor='black', lw=0.5))


# =============================================================================
# Original Utility Functions (maintained for backwards compatibility)
# =============================================================================


def draw_dimension(ax, start, end, offset, text, fontsize=8, horizontal=True):
    """
    Draw a dimension line with arrows and centered text.

    Args:
        ax: matplotlib Axes object
        start: Start point (x, y)
        end: End point (x, y)
        offset: Perpendicular offset for dimension line
        text: Dimension text
        fontsize: Text font size
        horizontal: True if dimension is horizontal
    """
    if horizontal:
        y = start[1] + offset
        ax.plot([start[0], start[0]], [start[1], y], 'k-', lw=0.5)
        ax.plot([end[0], end[0]], [end[1], y], 'k-', lw=0.5)
        ax.annotate('', xy=(end[0], y), xytext=(start[0], y),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
        ax.text((start[0] + end[0])/2, y + 0.05, text, ha='center',
               fontsize=fontsize, fontweight='bold')
    else:
        x = start[0] + offset
        ax.plot([start[0], x], [start[1], start[1]], 'k-', lw=0.5)
        ax.plot([end[0], x], [end[1], end[1]], 'k-', lw=0.5)
        ax.annotate('', xy=(x, end[1]), xytext=(x, start[1]),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
        ax.text(x + 0.1, (start[1] + end[1])/2, text, va='center',
               fontsize=fontsize, fontweight='bold', rotation=90)


def draw_thread_profile(ax, x, y, length, pitch=1.75, od=12, side='right'):
    """
    Draw a thread profile cross-section.

    Args:
        ax: matplotlib Axes object
        x, y: Start position
        length: Thread length
        pitch: Thread pitch in mm
        od: Outer diameter in mm
        side: 'right' or 'left' for which side to draw
    """
    scale = 0.1  # Scale factor for display
    n_threads = int(length / (pitch * scale))

    thread_height = pitch * 0.65 * scale
    thread_points = []

    for i in range(n_threads):
        base_y = y - i * pitch * scale
        if side == 'right':
            thread_points.extend([
                (x, base_y),
                (x + thread_height, base_y - pitch * scale * 0.25),
                (x, base_y - pitch * scale * 0.5),
            ])
        else:
            thread_points.extend([
                (x, base_y),
                (x - thread_height, base_y - pitch * scale * 0.25),
                (x, base_y - pitch * scale * 0.5),
            ])

    if thread_points:
        xs, ys = zip(*thread_points)
        ax.plot(xs, ys, 'k-', lw=0.8)


def create_probe_cross_section(enhanced=False):
    """
    Create detailed probe cross-section diagram with 3D-like shading.

    Args:
        enhanced: If True, use advanced rendering with fiberglass texture,
                  gradient shading, CAD-style section indicators, and
                  detailed coil winding visualization.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(7, 10) if enhanced else (6, 10))
    ax.set_xlim(-3.5 if enhanced else -3, 3.5 if enhanced else 3)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # Main probe body (16mm OD = 0.8 display units at scale)
    probe_width = 0.8
    probe_height = 10
    inner_width = 0.5

    # Shading helper (legacy mode)
    def draw_shaded_cyl(x, y, w, h, color, alpha=1.0):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='black', lw=1, alpha=alpha))
        ax.add_patch(Rectangle((x + w*0.1, y), w*0.2, h, facecolor='white', alpha=0.3, edgecolor='none'))
        ax.add_patch(Rectangle((x + w*0.7, y), w*0.2, h, facecolor='black', alpha=0.2, edgecolor='none'))

    if enhanced:
        # === ENHANCED MODE: Fiberglass with weave texture ===
        draw_fiberglass_texture(ax, -probe_width/2, 0, probe_width, probe_height,
                               COLORS['fiberglass'], weave_spacing=0.06)

        # CAD-style section line indicator
        draw_section_line_indicator(ax, -2.8, -0.5, 10.5, label='A')
        ax.text(0, -0.8, 'SECTION A-A', ha='center', fontsize=9,
               fontweight='bold', color=COLORS['section_line'])
    else:
        # Legacy: Simple shaded cylinder
        draw_shaded_cyl(-probe_width/2, 0, probe_width, probe_height, COLORS['fiberglass'])

    # Inner cavity (cutaway view)
    ax.add_patch(Rectangle((-inner_width/2, 0.3), inner_width, probe_height - 0.6,
                           facecolor='white', edgecolor=COLORS['gray_light'], lw=0.5))

    # === Components from bottom to top ===

    # Tip (pointed) with optional 3D effect
    tip_points = [(-probe_width/2, 0.3), (probe_width/2, 0.3), (0, -0.2)]
    if enhanced:
        draw_metal_surface(ax, -probe_width/2, -0.2, probe_width, 0.5,
                          COLORS['primary'], style='polished', orientation='vertical')
        ax.add_patch(Polygon(tip_points, facecolor='none', edgecolor='black', lw=1.2))
    else:
        ax.add_patch(Polygon(tip_points, facecolor=COLORS['primary'], edgecolor='black', lw=1))
    ax.text(1.6, 0.1, 'Probe Tip\n(Tapered)', fontsize=10, va='center', fontweight='bold')

    # ERT Ring 1 (bottom)
    ert_y1 = 1.0
    if enhanced:
        draw_metal_surface(ax, -probe_width/2 - 0.05, ert_y1, probe_width + 0.1, 0.15,
                          WONG_PALETTE['orange'], style='brushed', orientation='horizontal')
        ax.add_patch(Rectangle((-probe_width/2 - 0.05, ert_y1), probe_width + 0.1, 0.15,
                               facecolor='none', edgecolor='black', lw=0.8))
    else:
        draw_shaded_cyl(-probe_width/2 - 0.05, ert_y1, probe_width + 0.1, 0.15, COLORS['orange'])
    ax.text(1.6, ert_y1 + 0.07, 'ERT Ring #1\n(316L SS)', fontsize=10, va='center')

    # Thread joint 1 with ISO profile (enhanced)
    thread_y1 = 2.0
    if enhanced:
        draw_cylinder_gradient(ax, -probe_width/2, thread_y1, probe_width, 0.3,
                              COLORS['gray_light'], light_angle=50, n_strips=12)
        draw_iso_thread_profile(ax, probe_width/2, thread_y1 + 0.28, 0.25,
                               pitch=1.75, side='right', show_annotations=False)
        draw_chamfer_detail(ax, probe_width/2, thread_y1 + 0.3, 0.04, direction='down-right')
    else:
        draw_shaded_cyl(-probe_width/2, thread_y1, probe_width, 0.3, COLORS['gray_light'])
    ax.text(1.6, thread_y1 + 0.15, 'M12x1.75\nFlush Joint', fontsize=9, va='center', style='italic')

    # RX Coil (lower) - Enhanced with winding layers
    coil_y = 3.5
    if enhanced:
        # Bobbin/former with flanges
        draw_bobbin_with_flanges(ax, 0, coil_y - 0.4, coil_y + 0.4,
                                tube_od=0.25, flange_od=0.35, flange_thickness=0.05, tube_id=0.16)
        # Individual wire windings
        draw_coil_winding_layers(ax, 0, coil_y - 0.35, coil_y + 0.35,
                                inner_radius=0.13, outer_radius=0.22,
                                n_layers=3, turns_per_layer=6, wire_diameter=0.04)
        # Epoxy potting level indication
        draw_epoxy_potting_level(ax, -inner_width/2 + 0.06, inner_width/2 - 0.06,
                                coil_y - 0.38, fill_level=0.7, max_level=0.76)
    else:
        ax.add_patch(Rectangle((-inner_width/2 + 0.05, coil_y - 0.4), inner_width - 0.1, 0.8,
                               facecolor=COLORS['accent'], edgecolor='black', lw=1, hatch='///'))
        ax.add_patch(Rectangle((-0.08, coil_y - 0.35), 0.16, 0.7,
                               facecolor=COLORS['ferrite'], edgecolor='black', lw=0.5))
    ax.text(1.6, coil_y, 'RX Coil\n(250T, 34AWG)', fontsize=10, va='center', fontweight='bold',
           color=WONG_PALETTE['blue'] if enhanced else 'black')

    # ERT Ring 2 (middle)
    ert_y2 = 4.5
    if enhanced:
        draw_metal_surface(ax, -probe_width/2 - 0.05, ert_y2, probe_width + 0.1, 0.15,
                          WONG_PALETTE['orange'], style='brushed', orientation='horizontal')
        ax.add_patch(Rectangle((-probe_width/2 - 0.05, ert_y2), probe_width + 0.1, 0.15,
                               facecolor='none', edgecolor='black', lw=0.8))
    else:
        draw_shaded_cyl(-probe_width/2 - 0.05, ert_y2, probe_width + 0.1, 0.15, COLORS['orange'])
    ax.text(1.6, ert_y2 + 0.07, 'ERT Ring #2', fontsize=10, va='center')

    # Thread joint 2
    thread_y2 = 5.5
    if enhanced:
        draw_cylinder_gradient(ax, -probe_width/2, thread_y2, probe_width, 0.3,
                              COLORS['gray_light'], light_angle=50, n_strips=12)
    else:
        draw_shaded_cyl(-probe_width/2, thread_y2, probe_width, 0.3, COLORS['gray_light'])

    # TX Coil - Enhanced
    coil_y = 7.0
    if enhanced:
        draw_bobbin_with_flanges(ax, 0, coil_y - 0.4, coil_y + 0.4,
                                tube_od=0.25, flange_od=0.35, flange_thickness=0.05, tube_id=0.16)
        draw_coil_winding_layers(ax, 0, coil_y - 0.35, coil_y + 0.35,
                                inner_radius=0.13, outer_radius=0.22,
                                n_layers=4, turns_per_layer=8, wire_diameter=0.035)
        draw_epoxy_potting_level(ax, -inner_width/2 + 0.06, inner_width/2 - 0.06,
                                coil_y - 0.38, fill_level=0.8, max_level=0.76)
    else:
        ax.add_patch(Rectangle((-inner_width/2 + 0.05, coil_y - 0.4), inner_width - 0.1, 0.8,
                               facecolor=COLORS['success'], edgecolor='black', lw=1, hatch='///'))
        ax.add_patch(Rectangle((-0.08, coil_y - 0.35), 0.16, 0.7,
                               facecolor=COLORS['ferrite'], edgecolor='black', lw=0.5))
    ax.text(1.6, coil_y, 'TX Coil\n(300T, 34AWG)', fontsize=10, va='center', fontweight='bold',
           color=WONG_PALETTE['bluish_green'] if enhanced else 'black')

    # ERT Ring 3 (top)
    ert_y3 = 8.0
    if enhanced:
        draw_metal_surface(ax, -probe_width/2 - 0.05, ert_y3, probe_width + 0.1, 0.15,
                          WONG_PALETTE['orange'], style='brushed', orientation='horizontal')
        ax.add_patch(Rectangle((-probe_width/2 - 0.05, ert_y3), probe_width + 0.1, 0.15,
                               facecolor='none', edgecolor='black', lw=0.8))
    else:
        draw_shaded_cyl(-probe_width/2 - 0.05, ert_y3, probe_width + 0.1, 0.15, COLORS['orange'])

    # Thread joint 3 (top)
    thread_y3 = 9.0
    if enhanced:
        draw_cylinder_gradient(ax, -probe_width/2, thread_y3, probe_width, 0.3,
                              COLORS['gray_light'], light_angle=50, n_strips=12)
    else:
        draw_shaded_cyl(-probe_width/2, thread_y3, probe_width, 0.3, COLORS['gray_light'])

    # Junction box
    if enhanced:
        draw_cylinder_gradient(ax, -0.6, 9.5, 1.2, 0.8, COLORS['gray_dark'],
                              light_angle=45, n_strips=18)
        ax.add_patch(Rectangle((-0.6, 9.5), 1.2, 0.8, facecolor='none',
                               edgecolor='black', lw=1.2))
    else:
        draw_shaded_cyl(-0.6, 9.5, 1.2, 0.8, COLORS['gray_dark'])
    ax.text(0, 9.9, 'Junction\nBox', ha='center', va='center',
           fontsize=8, color='white', fontweight='bold')
    ax.text(1.6, 9.9, '12-Pin\nConnector', fontsize=10, va='center')

    # Wiring harness (internal)
    ax.plot([0, 0], [1.0, 9.5], color=COLORS['copper'], lw=2, alpha=0.6, linestyle=':')

    # Dimensions
    draw_dimension(ax, (-probe_width/2, -0.5), (probe_width/2, -0.5),
                  -0.3, '16mm OD')

    draw_dimension(ax, (-1.8 if enhanced else -1.5, 0), (-1.8 if enhanced else -1.5, 10),
                  0.2, '50cm\nSegment', horizontal=False)

    # Material legend (enhanced mode only)
    if enhanced:
        legend_x = -3.2
        legend_y = 5.5
        ax.text(legend_x, legend_y + 1.5, 'Materials:', fontsize=8, fontweight='bold')
        materials = [
            (COLORS['fiberglass'], 'G10 Fiberglass'),
            (WONG_PALETTE['orange'], '316L Stainless'),
            (COLORS['copper'], '34 AWG Cu Wire'),
            (COLORS['epoxy'], 'Epoxy Potting'),
        ]
        for i, (color, name) in enumerate(materials):
            y = legend_y + 1.0 - i * 0.35
            ax.add_patch(Rectangle((legend_x, y - 0.08), 0.2, 0.16,
                                   facecolor=color, edgecolor='black', lw=0.5))
            ax.text(legend_x + 0.3, y, name, fontsize=7, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_exploded_assembly(enhanced=False):
    """
    Create exploded view of probe assembly.

    Args:
        enhanced: If True, add circled assembly order numbers, torque specs,
                  and alignment indicators.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(11, 11) if enhanced else (10, 10))
    ax.set_xlim(-3 if enhanced else -2, 9 if enhanced else 8)
    ax.set_ylim(-1.5 if enhanced else -1, 10 if enhanced else 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # Exploded components with vertical separation
    # Format: (x, y, width, height, color, label, step_num)
    components = [
        (2, 0.5, 1.5, 0.4, COLORS['primary'], 'Probe Tip', 1),
        (2, 1.5, 1.5, 3.0, COLORS['fiberglass'], 'Bottom Segment\n(with RX coil)', 2),
        (2, 5.0, 1.5, 3.0, COLORS['fiberglass'], 'Top Segment\n(with TX coil)', 3),
        (2, 8.5, 1.8, 0.6, COLORS['gray_dark'], 'Junction Box', 4),
    ]

    for x, y, w, h, color, label, step in components:
        if enhanced:
            # 3D gradient shading
            draw_cylinder_gradient(ax, x - w/2, y, w, h, color, light_angle=55, n_strips=18)
            ax.add_patch(FancyBboxPatch((x - w/2, y), w, h, boxstyle="round,pad=0.02",
                                        facecolor='none', edgecolor='black', lw=1.5))
            # Circled step number
            draw_circled_number(ax, x - w/2 - 0.5, y + h/2, step, radius=0.3,
                               bg_color=WONG_PALETTE['blue'])
        else:
            ax.add_patch(FancyBboxPatch((x - w/2, y), w, h, boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor='black', lw=1.5))
        ax.text(x + w/2 + 0.5, y + h/2, label, fontsize=9, va='center')

    # Thread detail callouts with ISO spec (enhanced)
    if enhanced:
        # Male thread with chamfer detail
        ax.annotate('M12x1.75 ISO\nMale Thread\n(60deg form)', xy=(2, 4.4), xytext=(5.5, 4.3),
                   fontsize=8, ha='left',
                   arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))
        # Female thread
        ax.annotate('M12x1.75 ISO\nFemale Thread\n(12mm engagement)', xy=(2, 5.1), xytext=(5.5, 5.8),
                   fontsize=8, ha='left',
                   arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

        # Torque specification
        draw_torque_spec(ax, 5.5, 4.75, 3.0, direction='right')

        # Alignment indicator between segments
        draw_alignment_indicator(ax, 1.25, 2.75, 4.7, style='centerline')
    else:
        ax.annotate('M12x1.75\nMale Thread', xy=(2, 4.4), xytext=(5, 4.5),
                   fontsize=8, ha='left',
                   arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))
        ax.annotate('M12x1.75\nFemale Thread', xy=(2, 5.1), xytext=(5, 5.8),
                   fontsize=8, ha='left',
                   arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

    # Internal component callouts
    ax.add_patch(Circle((2, 2.5), 0.25, facecolor=COLORS['accent'],
                       edgecolor='black', lw=1))
    ax.annotate('RX Coil\n(250T, 34AWG)', xy=(2, 2.5), xytext=(-1.0 if enhanced else -0.5, 2.5),
               fontsize=8, ha='right',
               arrowprops=dict(arrowstyle='->', color=COLORS['accent']))

    ax.add_patch(Circle((2, 6.5), 0.25, facecolor=COLORS['success'],
                       edgecolor='black', lw=1))
    ax.annotate('TX Coil\n(300T, 34AWG)', xy=(2, 6.5), xytext=(-1.0 if enhanced else -0.5, 6.5),
               fontsize=8, ha='right',
               arrowprops=dict(arrowstyle='->', color=COLORS['success']))

    # ERT rings with metal surface (enhanced)
    for i, y in enumerate([1.2, 3.2, 5.8, 7.8]):
        if enhanced:
            draw_metal_surface(ax, 1.15, y, 1.7, 0.12, WONG_PALETTE['orange'],
                              style='brushed', orientation='horizontal')
            ax.add_patch(Rectangle((1.15, y), 1.7, 0.12,
                                   facecolor='none', edgecolor='black', lw=0.5))
        else:
            ax.add_patch(Rectangle((1.15, y), 1.7, 0.12,
                                   facecolor=COLORS['orange'], edgecolor='black', lw=0.5))
    ax.annotate('ERT Rings\n(4 per probe)\n316L Stainless' if enhanced else 'ERT Rings\n(4 per probe)',
               xy=(1.15, 3.2), xytext=(-1.0 if enhanced else -0.5, 3.5),
               fontsize=8, ha='right',
               arrowprops=dict(arrowstyle='->', color=COLORS['orange']))

    # Assembly arrows
    ax.annotate('', xy=(2, 4.6), xytext=(2, 4.9),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=2))
    ax.annotate('', xy=(2, 1.0), xytext=(2, 1.4),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=2))
    ax.annotate('', xy=(2, 8.1), xytext=(2, 8.4),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=2))

    # Assembly order legend (enhanced)
    if enhanced:
        ax.add_patch(FancyBboxPatch((-2.5, -1.2), 3.5, 1.0, boxstyle="round,pad=0.03",
                                    facecolor='#f7fafc', edgecolor=COLORS['accent'], lw=1))
        ax.text(-0.75, -0.4, 'Assembly Order:', fontsize=8, fontweight='bold',
               ha='center', color=COLORS['primary'])
        ax.text(-0.75, -0.75, 'Tip -> Bottom -> Top -> Junction', fontsize=7,
               ha='center', color=COLORS['gray_dark'])
        ax.text(-0.75, -1.0, '(All threaded connections)', fontsize=6,
               ha='center', color=COLORS['gray_med'])

    # Title
    title_text = 'Probe Assembly - Exploded View (with Assembly Order)' if enhanced else 'Probe Assembly - Exploded View'
    ax.text(3, -0.5 if not enhanced else -1.4, title_text, ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_thread_detail(enhanced=False):
    """
    Create detailed thread profile diagram with ISO-standard geometry.

    Args:
        enhanced: If True, use accurate 60-degree thread form with
                  engagement calculations and chamfer details.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6) if enhanced else (10, 5))

    # === Male Thread ===
    ax1.set_xlim(-1, 3.5 if enhanced else 3)
    ax1.set_ylim(-0.8 if enhanced else -0.5, 3)
    ax1.set_aspect('equal')

    # Thread body with gradient shading (enhanced)
    if enhanced:
        draw_cylinder_gradient(ax1, 0, 0, 0.8, 2.5, COLORS['fiberglass'],
                              light_angle=60, n_strips=12)
        ax1.add_patch(Rectangle((0, 0), 0.8, 2.5, facecolor='none', edgecolor='black', lw=1))
    else:
        ax1.add_patch(Rectangle((0, 0), 0.8, 2.5,
                                facecolor=COLORS['fiberglass'], edgecolor='black', lw=1))

    pitch = 0.175  # 1.75mm at scale
    n_threads = int(1.5 / pitch)

    if enhanced:
        # ISO 60-degree thread profile with full annotations
        thread_info = draw_iso_thread_profile(ax1, 0.8, 2.0, 1.5, pitch=1.75,
                                              side='right', show_annotations=True, thread_angle=60)

        # Chamfer at thread start (lead-in)
        draw_chamfer_detail(ax1, 0.8, 2.0, 0.06, direction='down-right')

        # Thread engagement annotation box
        ax1.add_patch(FancyBboxPatch((1.4, 1.8), 1.8, 0.8, boxstyle="round,pad=0.03",
                                     facecolor='#f7fafc', edgecolor=COLORS['accent'], lw=1))
        ax1.text(2.3, 2.4, 'Thread Engagement:', fontsize=7, fontweight='bold',
                ha='center', color=COLORS['primary'])
        ax1.text(2.3, 2.15, f'Min: 8mm (4.5 turns)', fontsize=6, ha='center')
        ax1.text(2.3, 1.95, f'Rec: 12mm (6.9 turns)', fontsize=6, ha='center',
                color=COLORS['success'], fontweight='bold')

        # 60-degree angle callout
        angle_arc = Arc((0.85, 1.7), 0.15, 0.15, angle=0, theta1=-30, theta2=30,
                       color=COLORS['warning'], lw=1.5)
        ax1.add_patch(angle_arc)
        ax1.text(1.05, 1.7, '60deg', fontsize=7, color=COLORS['warning'], fontweight='bold')
    else:
        # Legacy simple zigzag
        for i in range(n_threads):
            y_base = 0.5 + i * pitch
            ax1.plot([0.8, 0.9, 0.8], [y_base, y_base + pitch/2, y_base + pitch],
                    'k-', lw=1)

    # Dimensions
    ax1.annotate('', xy=(0.8, 0.5), xytext=(0.9, 0.5 + pitch/2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['warning'], lw=1))
    ax1.text(1.1, 0.6, 'P=1.75mm', fontsize=8, color=COLORS['warning'])

    ax1.annotate('', xy=(0, -0.3), xytext=(0.8, -0.3),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax1.text(0.4, -0.45, 'M12', fontsize=9, ha='center', fontweight='bold')

    if enhanced:
        # Major/minor diameter annotations
        ax1.text(0.4, -0.65, '(Major: 12mm, Minor: 10.1mm)', fontsize=6,
                ha='center', color=COLORS['gray_med'])

    ax1.set_title('(a) Male Thread Profile (ISO Metric)', fontsize=10, fontweight='bold',
                 color=COLORS['primary'])
    ax1.axis('off')

    # === Female Thread (Cross-section) ===
    ax2.set_xlim(-1, 3.5 if enhanced else 3)
    ax2.set_ylim(-0.8 if enhanced else -0.5, 3)
    ax2.set_aspect('equal')

    # Outer body
    if enhanced:
        draw_fiberglass_texture(ax2, 0, 0, 1.2, 2.5, COLORS['fiberglass'], weave_spacing=0.1)
        ax2.add_patch(Rectangle((0, 0), 1.2, 2.5, facecolor='none', edgecolor='black', lw=1))
    else:
        ax2.add_patch(Rectangle((0, 0), 1.2, 2.5,
                                facecolor=COLORS['fiberglass'], edgecolor='black', lw=1))

    # Inner bore
    ax2.add_patch(Rectangle((0.2, 0), 0.8, 2.5,
                            facecolor='white', edgecolor='black', lw=0.5))

    if enhanced:
        # ISO thread profiles on both sides
        draw_iso_thread_profile(ax2, 0.2, 2.0, 1.5, pitch=1.75,
                               side='left', show_annotations=False)
        draw_iso_thread_profile(ax2, 1.0, 2.0, 1.5, pitch=1.75,
                               side='right', show_annotations=False)

        # Chamfers at bore entry
        draw_chamfer_detail(ax2, 0.2, 2.5, 0.05, direction='down-left')
        draw_chamfer_detail(ax2, 1.0, 2.5, 0.05, direction='down-right')
    else:
        # Legacy zigzag
        for i in range(n_threads):
            y_base = 0.5 + i * pitch
            ax2.plot([0.2, 0.1, 0.2], [y_base, y_base + pitch/2, y_base + pitch],
                    'k-', lw=1)
            ax2.plot([1.0, 1.1, 1.0], [y_base, y_base + pitch/2, y_base + pitch],
                    'k-', lw=1)

    # Dimensions
    ax2.annotate('', xy=(0.2, -0.2), xytext=(1.0, -0.2),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax2.text(0.6, -0.35, 'ID 10.6mm', fontsize=8, ha='center')

    ax2.annotate('', xy=(0, -0.2), xytext=(1.2, -0.2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray_med'], lw=1))
    ax2.text(0.6, -0.5, '(OD 16mm)', fontsize=7, ha='center', color=COLORS['gray_med'])

    if enhanced:
        # Thread depth annotation
        ax2.annotate('Thread\ndepth\n0.94mm', xy=(0.13, 1.5), xytext=(-0.5, 1.5),
                    fontsize=6, ha='center', va='center',
                    arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=0.8))

    ax2.set_title('(b) Female Thread (Section)', fontsize=10, fontweight='bold',
                 color=COLORS['primary'])
    ax2.axis('off')

    # Overall title
    title_text = 'M12x1.75 ISO Metric Thread Profile' if enhanced else 'M12x1.75 Thread Profile Detail'
    fig.suptitle(title_text, fontsize=11, fontweight='bold',
                y=0.98, color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_coil_mounting_detail(enhanced=False):
    """
    Create coil mounting and ferrite core detail.

    Args:
        enhanced: If True, show individual wire layers, bobbin with flanges,
                  and epoxy potting level indication.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 7) if enhanced else (8, 6))
    ax.set_xlim(-2.5 if enhanced else -2, 6.5 if enhanced else 6)
    ax.set_ylim(-1.5 if enhanced else -1, 5.5 if enhanced else 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Probe body cross-section
    if enhanced:
        draw_fiberglass_texture(ax, 0, 0, 4, 4, COLORS['fiberglass'], weave_spacing=0.12)
        ax.add_patch(Rectangle((0, 0), 4, 4, facecolor='none', edgecolor='black', lw=1.5))
    else:
        ax.add_patch(Rectangle((0, 0), 4, 4,
                               facecolor=COLORS['fiberglass'], edgecolor='black', lw=1.5))
    ax.add_patch(Rectangle((0.3, 0.3), 3.4, 3.4,
                           facecolor='white', edgecolor=COLORS['gray_light'], lw=0.5))

    if enhanced:
        # === ENHANCED: Bobbin with flanges ===
        draw_bobbin_with_flanges(ax, 2, 0.6, 3.4, tube_od=0.7, flange_od=1.0,
                                flange_thickness=0.15, tube_id=0.5)

        # Ferrite core inside bobbin
        ax.add_patch(Rectangle((1.75, 0.75), 0.5, 2.5,
                               facecolor=COLORS['ferrite'], edgecolor='black', lw=1))
        ax.text(2, 4.5, 'Ferrite Rod\n(8mm x 80mm)', ha='center', fontsize=8)

        # Multi-layer windings with individual wires
        draw_coil_winding_layers(ax, 2, 0.9, 3.1, inner_radius=0.4, outer_radius=0.85,
                                n_layers=4, turns_per_layer=10, wire_diameter=0.12)

        # Epoxy potting level indication
        draw_epoxy_potting_level(ax, 0.4, 3.6, 0.35, fill_level=0.75, max_level=3.3)

        # Bobbin flange callout
        ax.annotate('Bobbin Flange\n(retains winding)', xy=(1.5, 3.35), xytext=(-1.5, 3.8),
                   fontsize=7, ha='center',
                   arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

        # Layer annotation
        ax.add_patch(FancyBboxPatch((4.2, 2.2), 2.0, 1.2, boxstyle="round,pad=0.03",
                                    facecolor='#f7fafc', edgecolor=COLORS['copper'], lw=1))
        ax.text(5.2, 3.2, 'Winding Layers:', fontsize=7, fontweight='bold',
               ha='center', color=COLORS['primary'])
        ax.text(5.2, 2.9, '4 layers, ~75 turns/layer', fontsize=6, ha='center')
        ax.text(5.2, 2.65, '34 AWG (0.16mm) Cu', fontsize=6, ha='center')
        ax.text(5.2, 2.4, 'Total: 250-300 turns', fontsize=6, ha='center',
               fontweight='bold', color=COLORS['success'])
        ax.annotate('', xy=(2.8, 2.0), xytext=(4.2, 2.8),
                   arrowprops=dict(arrowstyle='->', color=COLORS['copper']))
    else:
        # Legacy: Simple representation
        ax.add_patch(Rectangle((1.7, 0.8), 0.6, 2.4,
                               facecolor=COLORS['ferrite'], edgecolor='black', lw=1))
        ax.text(2, 4.3, 'Ferrite Rod\n(8mm x 80mm)', ha='center', fontsize=8)

        # Simple coil windings
        for y in np.linspace(1.0, 2.8, 12):
            ax.add_patch(Circle((1.4, y), 0.08, facecolor=COLORS['copper'],
                               edgecolor='black', lw=0.3))
            ax.add_patch(Circle((2.6, y), 0.08, facecolor=COLORS['copper'],
                               edgecolor='black', lw=0.3))

        ax.annotate('34 AWG Magnet Wire\n(0.16mm enameled)\n200-300 turns', xy=(2.6, 2.0),
                   xytext=(4.5, 2.5), fontsize=8, ha='left',
                   arrowprops=dict(arrowstyle='->', color=COLORS['copper']))

    # Mounting supports
    ax.add_patch(Rectangle((0.5, 0.8), 0.6, 0.3,
                           facecolor=COLORS['gray_light'], edgecolor='black', lw=0.5))
    ax.add_patch(Rectangle((2.9, 0.8), 0.6, 0.3,
                           facecolor=COLORS['gray_light'], edgecolor='black', lw=0.5))
    ax.add_patch(Rectangle((0.5, 2.9), 0.6, 0.3,
                           facecolor=COLORS['gray_light'], edgecolor='black', lw=0.5))
    ax.add_patch(Rectangle((2.9, 2.9), 0.6, 0.3,
                           facecolor=COLORS['gray_light'], edgecolor='black', lw=0.5))
    ax.annotate('Support\nBrackets', xy=(0.8, 0.95), xytext=(-1.5 if enhanced else -1.0, 1.0),
               fontsize=8, ha='right',
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

    # Lead wires
    ax.plot([1.4, 1.4, 0.5], [3.1, 3.5, 3.5], color=COLORS['copper'], lw=1.5)
    ax.plot([2.6, 2.6, 3.5], [3.1, 3.5, 3.5], color=COLORS['copper'], lw=1.5)
    ax.text(0.3, 3.7, 'Lead\nWires', fontsize=7, ha='center')

    # Dimensions
    draw_dimension(ax, (0, -0.3), (4, -0.3), -0.3, '16mm (OD)')
    draw_dimension(ax, (1.7, -0.3), (2.3, -0.3), -0.6 if enhanced else -0.6, '8mm core', fontsize=7)

    if enhanced:
        # Additional dimension for winding OD
        draw_dimension(ax, (1.15, -0.3), (2.85, -0.3), -0.9, 'Winding OD ~12mm', fontsize=6)

    # Title
    title_text = 'Coil Assembly Detail (Cross-Section)' if enhanced else 'Coil Assembly Detail'
    ax.text(2, 5.0 if enhanced else 4.8, title_text, ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_disturbance_comparison():
    """
    Create soil disturbance comparison diagram.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6)) # Widened from default to ensure circles stay round

    for ax, title, rod_dia, hole_dia, vol in [
        (ax1, 'HIRT Micro-Probe', 16, 20, '~1.0 L'),
        (ax2, 'Standard Probe', 25, 50, '~6.0 L')
    ]:
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_aspect('equal') # Critical for circles
        ax.axis('off')

        # Ground surface
        ax.add_patch(Rectangle((-40, -40), 80, 80, facecolor='#d4a373', alpha=0.3))

        # Disturbed zone (colorblind-safe reddish purple from Wong palette)
        dist_rad = hole_dia / 2
        ax.add_patch(Circle((0, 0), dist_rad, facecolor=WONG_PALETTE['reddish_purple'], alpha=0.4))
        ax.add_patch(Circle((0, 0), dist_rad, fill=False, edgecolor=WONG_PALETTE['reddish_purple'], linestyle='--', lw=1))
        
        # Rod
        rod_rad = rod_dia / 2
        ax.add_patch(Circle((0, 0), rod_rad, facecolor='#2c5282', edgecolor='black', lw=1.5))
        
        # Labels
        ax.text(0, 32, title, ha='center', fontsize=11, fontweight='bold', color='#1a365d')
        ax.text(0, -32, f'Hole: {hole_dia}mm', ha='center', fontsize=10)
        ax.text(0, -38, f'Vol: {vol}', ha='center', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_ert_ring_detail():
    """
    Create ERT ring collar assembly detail.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Probe body (cross-section)
    ax.add_patch(Rectangle((0, 0.5), 5, 2,
                           facecolor=COLORS['fiberglass'], edgecolor='black', lw=1.5))

    # ERT ring (external collar)
    ring_inner = 0.4
    ax.add_patch(Rectangle((-0.1, 1.0), 5.2, 1,
                           facecolor=COLORS['orange'], edgecolor='black', lw=1))
    ax.add_patch(Rectangle((0, 1.1), 5, 0.8,
                           facecolor=COLORS['fiberglass'], edgecolor='none'))

    # Contact points
    for x in [0.5, 2.5, 4.5]:
        ax.add_patch(Circle((x, 1.5), 0.15, facecolor=COLORS['steel'],
                           edgecolor='black', lw=0.5))
    ax.annotate('Spring-loaded\ncontact pins', xy=(0.5, 1.5), xytext=(-0.5, 2.8),
               fontsize=8, ha='center',
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

    # Wire connection
    ax.plot([4.5, 4.5, 5.5], [1.5, 2.8, 2.8], color=COLORS['copper'], lw=1.5)
    ax.text(5.7, 2.8, 'To harness', fontsize=8, va='center')

    # Dimensions
    draw_dimension(ax, (0, 0.2), (5, 0.2), -0.3, '~50mm length')
    draw_dimension(ax, (-0.1, 1.0), (-0.1, 2.0), -0.5, '10mm\nheight', horizontal=False)

    # Material callouts
    ax.text(2.5, 3.2, 'Stainless Steel Ring\n(316L, flush mount)',
           ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor=COLORS['orange'], alpha=0.3))

    # Title
    ax.text(2.5, -0.5, 'ERT Ring Assembly Detail', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_material_specs_diagram():
    """
    Create material specifications comparison diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.5, 'Material Selection Guide', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Materials data (colorblind-safe palette for categorical comparison)
    materials = [
        ('Fiberglass (G10)', 'Preferred', 7, [
            'Non-conductive', 'RF transparent', 'High strength',
            'Moderate cost', 'Excellent durability'
        ], WONG_PALETTE['bluish_green']),  # Colorblind-safe green
        ('Carbon Fiber', 'Premium', 5.2, [
            'Very high strength', 'Lightweight',
            'Conductive (!)', 'Expensive'
        ], WONG_PALETTE['blue']),  # Colorblind-safe blue
        ('PVC', 'Budget', 3.4, [
            'Low cost', 'Easy to machine',
            'Lower strength', 'Shallow use only'
        ], WONG_PALETTE['orange']),  # Colorblind-safe orange
        ('Metal (Al/Steel)', 'Avoid', 1.6, [
            'Strong', 'Durable',
            'EM interference', 'Not recommended'
        ], WONG_PALETTE['vermillion']),  # Colorblind-safe red
    ]

    for name, rec, y, props, color in materials:
        # Material box
        ax.add_patch(FancyBboxPatch((0.5, y - 0.5), 2.5, 1.0,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='black',
                                    lw=1.5, alpha=0.3))
        ax.text(1.75, y, name, ha='center', va='center',
               fontsize=10, fontweight='bold', color=COLORS['primary'])

        # Recommendation badge (colorblind-safe)
        badge_color = WONG_PALETTE['bluish_green'] if rec == 'Preferred' else (
            WONG_PALETTE['blue'] if rec == 'Premium' else (
            WONG_PALETTE['orange'] if rec == 'Budget' else WONG_PALETTE['vermillion']))
        ax.add_patch(FancyBboxPatch((3.2, y - 0.25), 1.2, 0.5,
                                    boxstyle="round,pad=0.02",
                                    facecolor=badge_color, edgecolor='black',
                                    lw=1, alpha=0.8))
        ax.text(3.8, y, rec, ha='center', va='center',
               fontsize=8, fontweight='bold', color='white')

        # Properties
        for i, prop in enumerate(props):
            x_pos = 5.0 + (i % 3) * 1.8
            y_off = 0.15 if i < 3 else -0.25
            ax.text(x_pos, y + y_off, f'- {prop}', fontsize=7,
                   va='center', color=COLORS['gray_dark'])

    # Legend (colorblind-safe)
    ax.text(0.5, 0.8, 'Legend:', fontsize=9, fontweight='bold')
    for i, (label, color) in enumerate([
        ('Preferred', WONG_PALETTE['bluish_green']),
        ('Premium', WONG_PALETTE['blue']),
        ('Budget', WONG_PALETTE['orange']),
        ('Avoid', WONG_PALETTE['vermillion'])
    ]):
        ax.add_patch(Rectangle((1.8 + i*2, 0.5), 0.3, 0.3,
                               facecolor=color, edgecolor='black', lw=0.5))
        ax.text(2.2 + i*2, 0.65, label, fontsize=7, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_modular_segment_connection():
    """
    Create modular segment connection detail diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(4, 7.5, 'Modular Flush-Mount Connector System', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    # === Left side: Assembled view ===
    ax.text(1.5, 6.8, '(a) Assembled', ha='center', fontsize=10, fontweight='bold')

    # Upper rod segment
    ax.add_patch(Rectangle((0.7, 4.5), 1.6, 2.0,
                           facecolor=COLORS['fiberglass'], edgecolor='black', lw=1.5))
    ax.text(1.5, 5.5, 'Rod\nSegment 1', ha='center', va='center', fontsize=8)

    # Connection zone (flush)
    ax.add_patch(Rectangle((0.7, 3.8), 1.6, 0.7,
                           facecolor=COLORS['gray_light'], edgecolor='black', lw=1))
    ax.text(1.5, 4.15, 'Flush\nJoint', ha='center', va='center', fontsize=7)

    # Lower rod segment
    ax.add_patch(Rectangle((0.7, 1.8), 1.6, 2.0,
                           facecolor=COLORS['fiberglass'], edgecolor='black', lw=1.5))
    ax.text(1.5, 2.8, 'Rod\nSegment 2', ha='center', va='center', fontsize=8)

    # OD dimension
    draw_dimension(ax, (0.7, 1.5), (2.3, 1.5), -0.3, '16mm OD')

    # Flush callout
    ax.annotate('No snag points\n(flush profile)', xy=(2.3, 4.15),
               xytext=(3.5, 4.15), fontsize=8, ha='left',
               arrowprops=dict(arrowstyle='->', color=COLORS['success']))

    # === Right side: Exploded view ===
    ax.text(6.5, 6.8, '(b) Exploded', ha='center', fontsize=10, fontweight='bold')

    # Upper rod with male insert
    ax.add_patch(Rectangle((5.5, 5.5), 2.0, 1.0,
                           facecolor=COLORS['fiberglass'], edgecolor='black', lw=1.5))
    ax.add_patch(Rectangle((6.0, 4.7), 1.0, 0.8,
                           facecolor=COLORS['gray_med'], edgecolor='black', lw=1))
    ax.text(6.5, 6.0, 'Rod 1', ha='center', va='center', fontsize=8)
    ax.text(6.5, 5.1, 'Male\nInsert', ha='center', va='center', fontsize=7,
           color='white')

    # Thread visualization on male
    for i in range(4):
        y = 4.75 + i * 0.15
        ax.plot([6.95, 7.05, 6.95], [y, y + 0.075, y + 0.15],
               'k-', lw=0.8)

    # Arrow showing connection
    ax.annotate('', xy=(6.5, 4.0), xytext=(6.5, 4.6),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    ax.text(7.3, 4.3, 'M12x1.75\nThread', fontsize=8, va='center')

    # Lower rod with female insert
    ax.add_patch(Rectangle((5.5, 1.8), 2.0, 1.5,
                           facecolor=COLORS['fiberglass'], edgecolor='black', lw=1.5))
    ax.add_patch(Rectangle((6.0, 3.3), 1.0, 0.5,
                           facecolor=COLORS['gray_med'], edgecolor='black', lw=1))
    # Hollow center for female
    ax.add_patch(Rectangle((6.15, 3.35), 0.7, 0.45,
                           facecolor='white', edgecolor='black', lw=0.5))
    ax.text(6.5, 2.5, 'Rod 2', ha='center', va='center', fontsize=8)
    ax.text(6.5, 3.55, 'Female', ha='center', va='center', fontsize=6)

    # Wiring channel
    ax.add_patch(Circle((6.5, 5.0), 0.15, facecolor='white',
                       edgecolor=COLORS['copper'], lw=1.5))
    ax.add_patch(Circle((6.5, 2.5), 0.15, facecolor='white',
                       edgecolor=COLORS['copper'], lw=1.5))
    ax.annotate('6mm wiring\nchannel', xy=(6.5, 2.5), xytext=(8.0, 2.0),
               fontsize=7, ha='left',
               arrowprops=dict(arrowstyle='->', color=COLORS['copper']))

    # Specifications box
    specs_text = [
        'Thread: M12x1.75 ISO',
        'Engagement: 12-15mm',
        'Male OD: 12.2mm (print)',
        'Female ID: 10.5mm (print)',
        'Post-process: Tap/Die'
    ]
    ax.add_patch(FancyBboxPatch((0.3, -0.5), 4.0, 1.3, boxstyle="round,pad=0.05",
                                facecolor='#f7fafc',
                                edgecolor=COLORS['gray_med'], lw=1))
    ax.text(2.3, 0.6, 'Specifications:', fontsize=8, fontweight='bold',
           ha='center', color=COLORS['primary'])
    for i, spec in enumerate(specs_text):
        ax.text(0.5, 0.3 - i*0.2, f'- {spec}', fontsize=7)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_junction_box_diagram():
    """
    Create junction box design detail diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # === Left: External view ===
    ax1.set_xlim(-1.5, 3.5)
    ax1.set_ylim(-1, 4)
    ax1.set_aspect('equal')
    ax1.axis('off')

    ax1.text(1, 3.7, '(a) External View', ha='center', fontsize=10, fontweight='bold')

    # Cable gland
    ax1.add_patch(FancyBboxPatch((0.6, 2.8), 0.8, 0.5, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['gray_dark'], edgecolor='black', lw=1))
    ax1.text(1, 3.05, 'Cable\nGland', ha='center', va='center',
            fontsize=6, color='white')

    # Main body
    ax1.add_patch(FancyBboxPatch((0.2, 1.0), 1.6, 1.8, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['gray_dark'], edgecolor='black', lw=1.5))
    ax1.text(1, 1.9, 'Junction\nBox Body\n(PETG)', ha='center', va='center',
            fontsize=8, color='white')

    # Thread region
    ax1.add_patch(Rectangle((0.6, 0.5), 0.8, 0.5,
                            facecolor=COLORS['gray_med'], edgecolor='black', lw=1))
    ax1.text(1, 0.75, 'Thread', ha='center', va='center', fontsize=7)

    # Rod connection
    ax1.add_patch(Rectangle((0.6, -0.3), 0.8, 0.8,
                            facecolor=COLORS['fiberglass'], edgecolor='black', lw=1))
    ax1.text(1, 0.1, 'Rod', ha='center', va='center', fontsize=7)

    # Dimensions
    draw_dimension(ax1, (0.2, 0.3), (1.8, 0.3), -0.4, '25mm')
    draw_dimension(ax1, (2.0, 1.0), (2.0, 2.8), 0.3, '35mm', horizontal=False)

    # === Right: Cutaway view ===
    ax2.set_xlim(-1.5, 4.5)
    ax2.set_ylim(-1, 4)
    ax2.set_aspect('equal')
    ax2.axis('off')

    ax2.text(1.5, 3.7, '(b) Internal Cutaway', ha='center', fontsize=10, fontweight='bold')

    # Outer shell (cutaway)
    ax2.add_patch(Rectangle((0, 0.8), 0.2, 2.2,
                            facecolor=COLORS['gray_dark'], edgecolor='black', lw=1))
    ax2.add_patch(Rectangle((2.6, 0.8), 0.2, 2.2,
                            facecolor=COLORS['gray_dark'], edgecolor='black', lw=1))
    ax2.add_patch(Rectangle((0, 2.8), 2.8, 0.2,
                            facecolor=COLORS['gray_dark'], edgecolor='black', lw=1))

    # Internal cavity
    ax2.add_patch(Rectangle((0.2, 0.8), 2.4, 2.0,
                            facecolor='white', edgecolor=COLORS['gray_light'], lw=0.5))

    # Terminal block
    ax2.add_patch(FancyBboxPatch((0.5, 1.5), 1.8, 1.0, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['accent'], edgecolor='black', lw=1))
    ax2.text(1.4, 2.0, 'Terminal\nBlock', ha='center', va='center',
            fontsize=8, color='white')

    # Connection points on terminal block
    for x in [0.7, 1.1, 1.5, 1.9]:
        ax2.add_patch(Circle((x, 1.7), 0.08, facecolor=COLORS['steel'],
                            edgecolor='black', lw=0.5))

    # Wire routing
    ax2.plot([1.4, 1.4], [1.5, 0.5], color=COLORS['copper'], lw=2)

    # Thread
    ax2.add_patch(Rectangle((0.8, 0.3), 1.2, 0.5,
                            facecolor=COLORS['gray_light'], edgecolor='black', lw=0.5))
    for i in range(4):
        y = 0.35 + i * 0.1
        ax2.plot([0.8, 0.7, 0.8], [y, y + 0.05, y + 0.1], 'k-', lw=0.5)
        ax2.plot([2.0, 2.1, 2.0], [y, y + 0.05, y + 0.1], 'k-', lw=0.5)

    # Labels
    ax2.annotate('MIT coil leads (2x2)', xy=(0.7, 1.7), xytext=(3.2, 2.5),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))
    ax2.annotate('ERT electrode leads (3-4)', xy=(1.5, 1.7), xytext=(3.2, 1.8),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))
    ax2.annotate('M12x1.75\nInternal', xy=(1.4, 0.55), xytext=(3.2, 0.5),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

    # Key features box
    ax2.add_patch(FancyBboxPatch((3.0, -0.5), 1.3, 0.8, boxstyle="round,pad=0.03",
                                 facecolor='#e8f4f8', edgecolor=COLORS['accent'], lw=1))
    ax2.text(3.65, 0.1, 'Key Features:', fontsize=7, fontweight='bold', ha='center')
    ax2.text(3.65, -0.1, 'Passive only', fontsize=6, ha='center')
    ax2.text(3.65, -0.25, 'Weatherproof', fontsize=6, ha='center')
    ax2.text(3.65, -0.4, 'Field serviceable', fontsize=6, ha='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_tool_requirements_diagram():
    """
    Create tool requirements diagram for probe assembly.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.5, 'Tool Requirements for Probe Assembly', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Categories
    categories = [
        ('Thread Cutting', 5.2, [
            ('M12x1.75 Tap', 'Female threads'),
            ('M12x1.75 Die', 'Male threads'),
            ('Tap wrench', 'T-handle style'),
            ('Cutting oil', 'Reduces friction'),
        ], COLORS['success']),
        ('Assembly', 3.4, [
            ('Epoxy (2-part)', 'Bonding inserts'),
            ('CA glue (thin)', 'Thread hardening'),
            ('Heat shrink', 'Wire protection'),
            ('Multimeter', 'Continuity check'),
        ], COLORS['accent']),
        ('Finishing', 1.6, [
            ('Flush cutters', 'Brim removal'),
            ('Fine sandpaper', '220-400 grit'),
            ('Deburring tool', 'Thread cleanup'),
            ('Calipers', 'Dimension check'),
        ], COLORS['orange']),
    ]

    for cat_name, y_base, tools, color in categories:
        # Category header
        ax.add_patch(FancyBboxPatch((0.3, y_base + 0.3), 2.2, 0.5,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', lw=1))
        ax.text(1.4, y_base + 0.55, cat_name, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

        # Tools
        for i, (tool, desc) in enumerate(tools):
            x = 2.8 + i * 1.8
            ax.add_patch(Rectangle((x, y_base + 0.1), 1.6, 0.7,
                                   facecolor='white', edgecolor=color, lw=1))
            ax.text(x + 0.8, y_base + 0.55, tool, ha='center', va='center',
                   fontsize=8, fontweight='bold')
            ax.text(x + 0.8, y_base + 0.25, desc, ha='center', va='center',
                   fontsize=6, color=COLORS['gray_dark'])

    # 3D Printing section
    ax.add_patch(FancyBboxPatch((0.3, 0.2), 9.4, 1.0, boxstyle="round,pad=0.03",
                                facecolor='#e8f4f8', edgecolor=COLORS['primary'], lw=1.5))
    ax.text(5, 1.0, '3D Printer Settings (Bambu Lab / FDM)', ha='center',
           fontsize=9, fontweight='bold', color=COLORS['primary'])

    settings = [
        'Material: PETG/ASA',
        'Layer: 0.12mm',
        'Infill: 100%',
        'Walls: 6 loops',
        'Supports: OFF',
        'Speed: 50mm/s (outer)'
    ]
    for i, setting in enumerate(settings):
        ax.text(0.8 + i * 1.55, 0.55, setting, fontsize=7, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_assembly_sequence_diagram(enhanced=False):
    """
    Create assembly sequence diagram showing bottom-to-top order.

    Args:
        enhanced: If True, add torque specifications on threaded joints,
                  alignment indicators, and improved step numbering.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 11) if enhanced else (6.67, 10))
    ax.set_xlim(-2.5 if enhanced else -2, 7.5 if enhanced else 6)
    ax.set_ylim(-1.5 if enhanced else -1, 11.5 if enhanced else 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(2.5 if enhanced else 2, 11.0 if enhanced else 10.5,
           'Assembly Sequence (Bottom to Top)', ha='center',
           fontsize=12 if enhanced else 11, fontweight='bold', color=COLORS['primary'])

    # Components from bottom to top with step numbers
    # Format: (y, name, color, desc, has_thread, torque_nm)
    components = [
        (0.5, 'Probe Tip', COLORS['primary'], '(pointed, screws on)', True, 2.5),
        (2.0, 'Rod Section 1', COLORS['fiberglass'], '(1.5m with RX coil)', False, None),
        (3.5, 'ERT Ring #1', COLORS['orange'], '(epoxy bond at 0.5m)', False, None),
        (5.0, 'Coupler', COLORS['gray_light'], '(M12 threads both ends)', True, 3.0),
        (6.5, 'Rod Section 2', COLORS['fiberglass'], '(1.5m with TX coil)', False, None),
        (8.0, 'ERT Ring #2 & #3', COLORS['orange'], '(at 1.5m, 2.5m)', False, None),
        (9.5, 'Junction Box', COLORS['gray_dark'], '(at surface, passive)', True, 2.0),
    ]

    for i, (y, name, color, desc, has_thread, torque) in enumerate(components):
        # Step number with enhanced styling
        if enhanced:
            draw_circled_number(ax, -1.0, y, i + 1, radius=0.35,
                               bg_color=WONG_PALETTE['blue'], text_color='white')
        else:
            ax.add_patch(Circle((-0.5, y), 0.3, facecolor=COLORS['secondary'],
                               edgecolor='black', lw=1))
            ax.text(-0.5, y, str(i+1), ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')

        # Component bar with gradient (enhanced)
        bar_x = 0.0 if enhanced else 0.3
        bar_width = 2.2 if enhanced else 2.0
        if enhanced:
            draw_cylinder_gradient(ax, bar_x, y - 0.3, bar_width, 0.6, color,
                                  light_angle=60, n_strips=15)
            ax.add_patch(FancyBboxPatch((bar_x, y - 0.3), bar_width, 0.6,
                                        boxstyle="round,pad=0.02",
                                        facecolor='none', edgecolor='black', lw=1.5))
        else:
            ax.add_patch(FancyBboxPatch((bar_x, y - 0.3), bar_width, 0.6,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor='black', lw=1.5))

        text_color = 'white' if color in [COLORS['primary'], COLORS['gray_dark'], COLORS['secondary']] else 'black'
        ax.text(bar_x + bar_width/2, y, name, ha='center', va='center',
               fontsize=9, fontweight='bold', color=text_color)

        # Description
        desc_x = bar_x + bar_width + 0.3
        ax.text(desc_x, y, desc, fontsize=8, va='center', color=COLORS['gray_dark'])

        # Torque specification (enhanced mode)
        if enhanced and has_thread and torque is not None:
            draw_torque_spec(ax, desc_x + 2.5, y, torque, direction='right')

        # Connecting arrow (except last)
        if i < len(components) - 1:
            arrow_x = bar_x + bar_width/2
            ax.annotate('', xy=(arrow_x, y + 0.6), xytext=(arrow_x, y + 1.1),
                       arrowprops=dict(arrowstyle='->', color=COLORS['gray_med'], lw=1.5))

    # Alignment indicator (enhanced mode)
    if enhanced:
        draw_alignment_indicator(ax, -0.2, 2.4, 1.25, style='centerline')
        ax.text(-1.8, 1.25, 'Align\nwiring\nchannel', fontsize=6, ha='center',
               va='center', color=COLORS['accent'])

    # Ground level indicator
    ground_y = 9.0
    ax.axhline(ground_y, xmin=0.05, xmax=0.55 if enhanced else 0.7, color='#654321', lw=3)
    ax.text(-1.3 if enhanced else -0.8, ground_y, 'Ground\nLevel', fontsize=8,
           ha='center', va='center', color='#654321')

    # Key points box
    box_x = -2.2 if enhanced else -1.8
    box_width = 9.4 if enhanced else 7.4
    ax.add_patch(FancyBboxPatch((box_x, -1.0 if enhanced else -0.5), box_width,
                                1.6 if enhanced else 1.2, boxstyle="round,pad=0.03",
                                facecolor='#fff5f5', edgecolor=COLORS['warning'], lw=1))
    ax.text(box_x + 0.3, 0.35 if enhanced else 0.5, 'Key Points:', fontsize=9,
           fontweight='bold', color=COLORS['warning'])
    ax.text(box_x + 0.3, 0.0 if enhanced else 0.15,
           '- All parts screw together (modular)   - Tip points DOWN   - Wires route through center',
           fontsize=7, color=COLORS['gray_dark'])
    ax.text(box_x + 0.3, -0.35 if enhanced else -0.15,
           '- Junction box stays at surface       - ERT collars bond with epoxy',
           fontsize=7, color=COLORS['gray_dark'])

    if enhanced:
        # Additional torque note
        ax.text(box_x + 0.3, -0.7,
               '- Hand-tight + 1/4 turn for all threaded joints (torque values shown)',
               fontsize=7, color=COLORS['warning'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_insertion_methods_diagram():
    """
    Create insertion methods comparison diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    methods = [
        ('(a) Hand Auger', axes[0, 0], 'Best for most soils'),
        ('(b) Pilot Rod', axes[0, 1], 'Clay/compact soils'),
        ('(c) Direct Push', axes[1, 0], 'Sandy soils only'),
        ('(d) Water-Jet', axes[1, 1], 'Excellent for sand'),
    ]

    for title, ax, desc in methods:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-3, 1)
        ax.set_aspect('equal')

        # Ground
        ax.axhline(0, color='#654321', lw=2)
        ax.fill_between([-2, 2], [0, 0], [-3, -3], color='#d4a373', alpha=0.4)

        ax.set_title(title, fontsize=10, fontweight='bold', color=COLORS['primary'])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(0, -2.8, desc, ha='center', fontsize=8, style='italic',
               color=COLORS['gray_dark'])

    # (a) Hand Auger
    ax = axes[0, 0]
    # Auger tool
    ax.add_patch(Rectangle((-0.15, -1.5), 0.3, 2.0,
                           facecolor=COLORS['gray_med'], edgecolor='black', lw=1))
    ax.add_patch(Circle((0, -1.5), 0.3, facecolor=COLORS['gray_dark'],
                       edgecolor='black', lw=1))
    # Spiral suggestion
    for i in range(3):
        y = -0.5 - i * 0.3
        ax.plot([-0.15, -0.3, -0.15], [y, y - 0.1, y - 0.2],
               color=COLORS['gray_dark'], lw=1)
    # Hole
    ax.add_patch(Rectangle((-0.25, -2.2), 0.5, 0.7,
                           facecolor='white', edgecolor=COLORS['gray_light'], lw=1, alpha=0.5))
    ax.text(1.2, -0.8, '10-20mm\nauger', fontsize=7, va='center')

    # (b) Pilot Rod
    ax = axes[0, 1]
    # Steel rod
    ax.add_patch(Rectangle((-0.1, -2.0), 0.2, 2.5,
                           facecolor=COLORS['steel'], edgecolor='black', lw=1))
    ax.text(0, 0.7, 'Steel', ha='center', fontsize=7, color=COLORS['steel'])
    # Warning
    ax.add_patch(FancyBboxPatch((0.5, -1.5), 1.3, 0.5, boxstyle="round,pad=0.02",
                                facecolor='#fff5f5', edgecolor=COLORS['warning'], lw=1))
    ax.text(1.15, -1.25, 'Remove\nbefore survey!', ha='center', fontsize=6,
           color=COLORS['warning'])
    ax.annotate('', xy=(0.15, -1.5), xytext=(0.5, -1.25),
               arrowprops=dict(arrowstyle='->', color=COLORS['warning']))

    # (c) Direct Push
    ax = axes[1, 0]
    # Probe
    ax.add_patch(Rectangle((-0.12, -1.8), 0.24, 2.0,
                           facecolor=COLORS['secondary'], edgecolor='black', lw=1))
    # Tip
    tip_points = [(-0.12, -1.8), (0.12, -1.8), (0, -2.1)]
    ax.add_patch(Polygon(tip_points, facecolor=COLORS['primary'],
                        edgecolor='black', lw=1))
    # Force arrow
    ax.annotate('', xy=(0, 0), xytext=(0, 0.6),
               arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))
    ax.text(0, 0.75, 'Push', ha='center', fontsize=8, fontweight='bold',
           color=COLORS['success'])
    ax.text(0.8, -1.0, 'Sand/loam\nonly', fontsize=7, va='center')

    # (d) Water-Jet
    ax = axes[1, 1]
    # Water lance
    ax.add_patch(Rectangle((-0.08, -1.5), 0.16, 1.8,
                           facecolor=COLORS['accent'], edgecolor='black', lw=1))
    # Water spray effect
    for angle in [-30, 0, 30]:
        rad = np.radians(angle - 90)
        dx = 0.5 * np.cos(rad)
        dy = 0.5 * np.sin(rad)
        ax.annotate('', xy=(dx, -1.5 + dy), xytext=(0, -1.5),
                   arrowprops=dict(arrowstyle='->', color='#90cdf4', lw=1))
    # Fluidized zone
    ax.add_patch(Ellipse((0, -2.0), 0.6, 0.8,
                        facecolor='#90cdf4', edgecolor='none', alpha=0.4))
    ax.text(1.0, -1.8, 'Fluidized\nsand', fontsize=7, va='center',
           color=COLORS['accent'])
    ax.text(-1.5, -1.0, 'Water\nlance', fontsize=7, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_advantages_diagram():
    """
    Create micro-probe advantages infographic.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.5, 'Micro-Probe Design Advantages', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Central probe image
    ax.add_patch(Rectangle((4.5, 2), 1.0, 4.5,
                           facecolor=COLORS['secondary'], edgecolor='black', lw=2))
    ax.add_patch(Polygon([(4.5, 2), (5.5, 2), (5, 1.5)],
                        facecolor=COLORS['primary'], edgecolor='black', lw=2))
    ax.text(5, 3.5, '16mm\nProbe', ha='center', va='center',
           fontsize=9, fontweight='bold', color='white', rotation=90)

    # Advantage callouts arranged around the probe (colorblind-safe alternating colors)
    advantages = [
        (1.5, 6.5, '10x Less Disturbance', '~3 cm^2 vs 20 cm^2 hole', WONG_PALETTE['bluish_green']),
        (8.5, 6.5, 'Modular Design', 'Field-serviceable segments', WONG_PALETTE['blue']),
        (1.5, 5.0, '50% Lighter', '50-100 g/m vs 200+ g/m', WONG_PALETTE['bluish_green']),
        (8.5, 5.0, 'M12 Threads', 'Standard, robust connection', WONG_PALETTE['blue']),
        (1.5, 3.5, 'Passive Probes', 'No electronics downhole', WONG_PALETTE['bluish_green']),
        (8.5, 3.5, 'Flush Profile', 'No snag points', WONG_PALETTE['blue']),
        (1.5, 2.0, '70% Cost Reduction', '$40-60 vs $130-180', WONG_PALETTE['bluish_green']),
        (8.5, 2.0, 'Better Reliability', 'Simpler = more robust', WONG_PALETTE['blue']),
    ]

    for x, y, title, desc, color in advantages:
        # Connector line
        probe_x = 4.5 if x < 5 else 5.5
        ax.plot([x, probe_x], [y, y], color=color, lw=1, alpha=0.5)

        # Advantage box
        ax.add_patch(FancyBboxPatch((x - 1.3, y - 0.4), 2.6, 0.8,
                                    boxstyle="round,pad=0.03",
                                    facecolor=color, edgecolor='black', lw=1, alpha=0.2))
        ax.add_patch(FancyBboxPatch((x - 1.3, y - 0.4), 2.6, 0.8,
                                    boxstyle="round,pad=0.03",
                                    facecolor='none', edgecolor=color, lw=1.5))
        ax.text(x, y + 0.1, title, ha='center', va='center',
               fontsize=8, fontweight='bold', color=COLORS['primary'])
        ax.text(x, y - 0.2, desc, ha='center', va='center',
               fontsize=6, color=COLORS['gray_dark'])

    # Summary box at bottom
    ax.add_patch(FancyBboxPatch((0.5, 0.2), 9.0, 0.9, boxstyle="round,pad=0.03",
                                facecolor=COLORS['primary'], edgecolor='black', lw=2, alpha=0.1))
    ax.text(5, 0.65, 'Result: Professional-grade subsurface imaging at 95% lower cost',
           ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# =============================================================================
# Prototyping and First-Test Diagrams
# =============================================================================

def create_prototyping_workflow():
    """
    Create rapid prototyping workflow diagram showing MVP build process.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.7, 'Rapid Prototyping Workflow', ha='center',
           fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax.text(5, 6.35, 'Minimum Viable Prototype (MVP) — First Data in 2 Days', ha='center',
           fontsize=9, style='italic', color=COLORS['gray_dark'])

    # Workflow steps (colorblind-safe sequential colors)
    steps = [
        ('1', 'PRINT', '4-6 hrs', 'Parts', WONG_PALETTE['blue'],
         ['Sensor body', 'Male cap', 'Probe tip']),
        ('2', 'WIND', '1-1.5 hrs', 'Coil', WONG_PALETTE['bluish_green'],
         ['300 turns', '34 AWG', 'Ferrite core']),
        ('3', 'ASSEMBLE', '30 min', 'Probe', WONG_PALETTE['orange'],
         ['Thread fit', 'Dry assembly', 'Wire routing']),
        ('4', 'TEST', '1 hr', 'Data', WONG_PALETTE['reddish_purple'],
         ['Inductance', 'Q factor', 'Metal response']),
    ]

    box_width = 2.0
    box_height = 3.5
    start_x = 0.5
    spacing = 2.4

    for i, (num, title, time, output, color, details) in enumerate(steps):
        x = start_x + i * spacing
        y = 1.5

        # Main box with shadow
        shadow = FancyBboxPatch((x + 0.05, y - 0.05), box_width, box_height,
                                boxstyle="round,pad=0.05",
                                facecolor='#00000020', edgecolor='none')
        ax.add_patch(shadow)

        box = FancyBboxPatch((x, y), box_width, box_height,
                             boxstyle="round,pad=0.05",
                             facecolor='white', edgecolor=color, lw=2)
        ax.add_patch(box)

        # Step number circle
        circle = Circle((x + 0.35, y + box_height - 0.35), 0.25,
                        facecolor=color, edgecolor='white', lw=2)
        ax.add_patch(circle)
        ax.text(x + 0.35, y + box_height - 0.35, num, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

        # Title
        ax.text(x + box_width/2 + 0.15, y + box_height - 0.4, title, ha='center',
               fontsize=11, fontweight='bold', color=color)

        # Time
        ax.text(x + box_width/2, y + box_height - 0.8, time, ha='center',
               fontsize=8, color=COLORS['gray_dark'])

        # Divider
        ax.plot([x + 0.2, x + box_width - 0.2], [y + box_height - 1.0, y + box_height - 1.0],
               color=COLORS['gray_light'], lw=1)

        # Details
        for j, detail in enumerate(details):
            ax.text(x + 0.3, y + box_height - 1.4 - j * 0.35, f"• {detail}",
                   fontsize=8, color=COLORS['gray_dark'])

        # Output label
        ax.add_patch(FancyBboxPatch((x + 0.3, y + 0.15), box_width - 0.6, 0.4,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='none', alpha=0.2))
        ax.text(x + box_width/2, y + 0.35, f"→ {output}", ha='center',
               fontsize=8, fontweight='bold', color=color)

        # Arrow to next step
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + box_width + 0.3, y + box_height/2),
                       xytext=(x + box_width + 0.1, y + box_height/2),
                       arrowprops=dict(arrowstyle='->', color=COLORS['gray_med'], lw=2))

    # Bottom summary
    ax.add_patch(FancyBboxPatch((0.5, 0.3), 9.0, 0.8, boxstyle="round,pad=0.03",
                                facecolor='#e8f4e8', edgecolor=COLORS['success'], lw=1.5))
    ax.text(5, 0.7, 'Success Criteria: L = 1.5mH ±20%  •  Q ≥ 15 @ 10kHz  •  Visible metal response',
           ha='center', fontsize=9, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_quick_test_setup():
    """
    Create bench test setup diagram for functional testing.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(5, 4.7, 'Quick Functional Test Setup', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Function Generator
    ax.add_patch(FancyBboxPatch((0.3, 1.5), 2.0, 2.0, boxstyle="round,pad=0.05",
                                facecolor='#e8e8e8', edgecolor=COLORS['gray_dark'], lw=2))
    ax.text(1.3, 3.2, 'Function', ha='center', fontsize=9, fontweight='bold')
    ax.text(1.3, 2.9, 'Generator', ha='center', fontsize=9, fontweight='bold')
    ax.text(1.3, 2.4, '10 kHz', ha='center', fontsize=8, color=COLORS['accent'])
    ax.text(1.3, 2.1, '2V p-p', ha='center', fontsize=8, color=COLORS['accent'])
    # Output jack
    ax.add_patch(Circle((2.3, 2.5), 0.1, facecolor=COLORS['warning'], edgecolor='black'))

    # Series resistor
    ax.add_patch(FancyBboxPatch((2.8, 2.2), 0.8, 0.6, boxstyle="round,pad=0.02",
                                facecolor='#f5e6d3', edgecolor=COLORS['orange'], lw=1.5))
    ax.text(3.2, 2.5, '100Ω', ha='center', fontsize=8, fontweight='bold')

    # Wire from func gen to resistor
    ax.plot([2.4, 2.8], [2.5, 2.5], color='black', lw=2)

    # Probe/Coil (central element)
    probe_x, probe_y = 4.5, 1.2
    # Coil representation
    ax.add_patch(FancyBboxPatch((probe_x, probe_y), 1.2, 2.2, boxstyle="round,pad=0.05",
                                facecolor=COLORS['fiberglass'], edgecolor=COLORS['primary'], lw=2))
    # Ferrite core
    ax.add_patch(Rectangle((probe_x + 0.35, probe_y + 0.4), 0.5, 1.4,
                           facecolor=COLORS['ferrite'], edgecolor='black', lw=1))
    # Windings suggestion
    for i in range(8):
        wy = probe_y + 0.5 + i * 0.15
        ax.plot([probe_x + 0.3, probe_x + 0.9], [wy, wy], color=COLORS['copper'], lw=1.5)
    ax.text(probe_x + 0.6, probe_y + 2.0, 'COIL', ha='center', fontsize=8,
           fontweight='bold', color=COLORS['primary'])
    ax.text(probe_x + 0.6, probe_y - 0.2, '1.5 mH', ha='center', fontsize=7,
           color=COLORS['gray_dark'])

    # Wire from resistor to coil
    ax.plot([3.6, probe_x], [2.5, 2.5], color='black', lw=2)
    ax.plot([probe_x, probe_x + 0.1], [2.5, probe_y + 2.0], color='black', lw=2)

    # Wire from coil to scope
    ax.plot([probe_x + 1.1, probe_x + 1.2], [probe_y + 2.0, 2.5], color='black', lw=2)
    ax.plot([probe_x + 1.2, 6.5], [2.5, 2.5], color='black', lw=2)

    # Ground return
    ax.plot([probe_x + 0.1, probe_x + 0.1], [probe_y + 0.2, 0.8], color='black', lw=2)
    ax.plot([probe_x + 0.1, 7.0], [0.8, 0.8], color='black', lw=2)
    ax.plot([7.0, 7.0], [0.8, 1.5], color='black', lw=2)

    # Oscilloscope
    ax.add_patch(FancyBboxPatch((6.5, 1.5), 2.5, 2.0, boxstyle="round,pad=0.05",
                                facecolor='#1a1a2e', edgecolor=COLORS['gray_dark'], lw=2))
    ax.text(7.75, 3.2, 'Oscilloscope', ha='center', fontsize=9, fontweight='bold', color='white')
    # Screen
    ax.add_patch(Rectangle((6.7, 1.7), 1.6, 1.2, facecolor='#0a1628', edgecolor='#333'))
    # Sine wave on screen
    x_wave = np.linspace(6.8, 8.2, 50)
    y_wave = 2.3 + 0.4 * np.sin((x_wave - 6.8) * 8)
    ax.plot(x_wave, y_wave, color='#00ff00', lw=1.5)
    # Input jack
    ax.add_patch(Circle((6.5, 2.5), 0.1, facecolor=COLORS['success'], edgecolor='black'))

    # Test target with stick
    target_x = 4.0
    ax.add_patch(FancyBboxPatch((target_x - 0.6, 3.5), 0.5, 0.8, boxstyle="round,pad=0.02",
                                facecolor=COLORS['steel'], edgecolor='black', lw=1))
    ax.plot([target_x - 0.35, target_x - 0.35], [4.3, 4.8], color='#8B4513', lw=3)
    ax.text(target_x - 0.35, 3.3, 'Metal\ntarget', ha='center', fontsize=7,
           color=COLORS['gray_dark'])

    # Arrow showing target approach
    ax.annotate('', xy=(probe_x + 0.2, 3.0), xytext=(target_x - 0.1, 3.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['warning'],
                              lw=2, connectionstyle='arc3,rad=-0.2'))
    ax.text(3.8, 3.9, 'Approach', fontsize=7, color=COLORS['warning'], style='italic')

    # Labels
    ax.text(1.3, 1.2, 'Signal\nSource', ha='center', fontsize=7, color=COLORS['gray_dark'])
    ax.text(7.75, 1.2, 'Monitor\nResponse', ha='center', fontsize=7, color=COLORS['gray_dark'])

    # Bottom note
    ax.add_patch(FancyBboxPatch((0.3, 0.1), 9.4, 0.5, boxstyle="round,pad=0.02",
                                facecolor='#fff8e8', edgecolor=COLORS['orange'], lw=1))
    ax.text(5, 0.35, 'PASS: Visible amplitude/phase change when metal approaches coil',
           ha='center', fontsize=8, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_coil_winding_steps():
    """
    Create step-by-step coil winding visual guide.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, axes = plt.subplots(1, 4, figsize=(10, 4))

    steps = [
        ('1. Prepare Core', 'Clean ferrite rod\nMark center 50mm zone'),
        ('2. Start Winding', 'Leave 150mm tail\nSecure with tape'),
        ('3. Wind 300 Turns', 'Single layer, close-wound\nMaintain tension'),
        ('4. Finish & Test', 'Secure with CA glue\nMeasure: L ~1.5mH'),
    ]

    for ax, (title, desc) in zip(axes, steps):
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 3)
        ax.axis('off')

        # Title
        ax.text(1, 2.85, title, ha='center', fontsize=9, fontweight='bold',
               color=COLORS['primary'])

        # Ferrite rod
        ax.add_patch(Rectangle((0.3, 1.0), 1.4, 0.4,
                               facecolor=COLORS['ferrite'], edgecolor='black', lw=1.5))

        # Description
        ax.text(1, 0.4, desc, ha='center', va='center', fontsize=7,
               color=COLORS['gray_dark'], linespacing=1.5)

    # Step 1: Clean rod with marks
    ax = axes[0]
    ax.plot([0.5, 0.5], [0.95, 1.45], color=COLORS['warning'], lw=2, linestyle='--')
    ax.plot([1.5, 1.5], [0.95, 1.45], color=COLORS['warning'], lw=2, linestyle='--')
    ax.text(1, 1.55, '50mm', ha='center', fontsize=7, color=COLORS['warning'])

    # Step 2: Wire tail
    ax = axes[1]
    ax.plot([0.3, 0.1, 0.1], [1.2, 1.2, 0.7], color=COLORS['copper'], lw=2)
    ax.text(0.15, 0.55, 'Tail', fontsize=6, color=COLORS['copper'])
    # Tape
    ax.add_patch(Rectangle((0.4, 1.1), 0.2, 0.2,
                           facecolor='#f0e68c', edgecolor='black', lw=0.5, alpha=0.8))

    # Step 3: Windings
    ax = axes[2]
    for i in range(12):
        wx = 0.5 + i * 0.08
        ax.add_patch(Rectangle((wx, 1.0), 0.06, 0.4,
                               facecolor=COLORS['copper'], edgecolor='#8B5A2B', lw=0.5))

    # Step 4: Complete with glue and measurement
    ax = axes[3]
    for i in range(12):
        wx = 0.5 + i * 0.08
        ax.add_patch(Rectangle((wx, 1.0), 0.06, 0.4,
                               facecolor=COLORS['copper'], edgecolor='#8B5A2B', lw=0.5))
    # Glue indication
    ax.add_patch(FancyBboxPatch((0.45, 0.95), 1.1, 0.5, boxstyle="round,pad=0.02",
                                facecolor='none', edgecolor=COLORS['success'],
                                lw=1.5, linestyle='--'))
    # Checkmark
    ax.text(1.7, 1.2, '✓', fontsize=14, color=COLORS['success'], fontweight='bold')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_prototype_decision_tree():
    """
    Create decision tree for prototype test results.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Prototype Test Decision Tree', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Start node (colorblind-safe blue)
    ax.add_patch(FancyBboxPatch((3.5, 4.5), 3.0, 0.8, boxstyle="round,pad=0.05",
                                facecolor=WONG_PALETTE['blue'], edgecolor='black', lw=2))
    ax.text(5, 4.9, 'Run Tests', ha='center', fontsize=10, fontweight='bold', color='white')

    # Decision diamonds
    def draw_diamond(dx, dy, text, color):
        diamond = Polygon([(dx, dy+0.4), (dx+0.6, dy), (dx, dy-0.4), (dx-0.6, dy)],
                         facecolor='#fff8e8', edgecolor=color, lw=2)
        ax.add_patch(diamond)
        ax.text(dx, dy, text, ha='center', va='center', fontsize=7, fontweight='bold')

    # Coil OK? (colorblind-safe orange)
    draw_diamond(3, 3.5, 'Coil\nL, Q?', WONG_PALETTE['orange'])
    ax.plot([5, 5, 3], [4.5, 4.0, 4.0], color='black', lw=1.5)
    ax.plot([3, 3], [4.0, 3.9], color='black', lw=1.5)

    # Threads OK? (colorblind-safe orange)
    draw_diamond(5, 2.5, 'Threads\nOK?', WONG_PALETTE['orange'])
    ax.plot([3.6, 5], [3.5, 3.5], color='black', lw=1.5)
    ax.text(4.3, 3.65, 'Yes', fontsize=7, color=WONG_PALETTE['bluish_green'])
    ax.plot([5, 5], [3.5, 2.9], color='black', lw=1.5)

    # Detection OK? (colorblind-safe orange)
    draw_diamond(7, 1.5, 'Metal\nDetect?', WONG_PALETTE['orange'])
    ax.plot([5.6, 7], [2.5, 2.5], color='black', lw=1.5)
    ax.text(6.3, 2.65, 'Yes', fontsize=7, color=WONG_PALETTE['bluish_green'])
    ax.plot([7, 7], [2.5, 1.9], color='black', lw=1.5)

    # Outcomes
    # Re-wind coil (colorblind-safe vermillion for error)
    ax.add_patch(FancyBboxPatch((0.5, 3.0), 1.8, 0.8, boxstyle="round,pad=0.03",
                                facecolor='#ffe8e8', edgecolor=WONG_PALETTE['vermillion'], lw=1.5))
    ax.text(1.4, 3.4, 'Re-wind\nCoil', ha='center', fontsize=8, fontweight='bold',
           color=WONG_PALETTE['vermillion'])
    ax.plot([2.4, 2.4], [3.5, 3.4], color='black', lw=1.5)
    ax.text(2.1, 3.65, 'No', fontsize=7, color=WONG_PALETTE['vermillion'])

    # Reprint parts (colorblind-safe vermillion for error)
    ax.add_patch(FancyBboxPatch((3.1, 1.5), 1.8, 0.8, boxstyle="round,pad=0.03",
                                facecolor='#ffe8e8', edgecolor=WONG_PALETTE['vermillion'], lw=1.5))
    ax.text(4.0, 1.9, 'Reprint\nParts', ha='center', fontsize=8, fontweight='bold',
           color=WONG_PALETTE['vermillion'])
    ax.plot([4.4, 4.4], [2.5, 2.3], color='black', lw=1.5)
    ax.text(4.1, 2.35, 'No', fontsize=7, color=WONG_PALETTE['vermillion'])

    # Debug setup (colorblind-safe orange for warning)
    ax.add_patch(FancyBboxPatch((5.6, 0.3), 1.8, 0.8, boxstyle="round,pad=0.03",
                                facecolor='#fff0e8', edgecolor=WONG_PALETTE['orange'], lw=1.5))
    ax.text(6.5, 0.7, 'Debug\nSetup', ha='center', fontsize=8, fontweight='bold',
           color=WONG_PALETTE['orange'])
    ax.plot([6.4, 6.4], [1.5, 1.1], color='black', lw=1.5)
    ax.text(6.1, 1.35, 'No', fontsize=7, color=WONG_PALETTE['vermillion'])

    # SUCCESS (colorblind-safe bluish green)
    ax.add_patch(FancyBboxPatch((7.8, 0.3), 1.8, 0.8, boxstyle="round,pad=0.03",
                                facecolor='#e8f4e8', edgecolor=WONG_PALETTE['bluish_green'], lw=2))
    ax.text(8.7, 0.7, 'SUCCESS\nPhase 2', ha='center', fontsize=9, fontweight='bold',
           color=WONG_PALETTE['bluish_green'])
    ax.plot([7.6, 8.3], [1.5, 1.1], color='black', lw=1.5)
    ax.text(8.0, 1.35, 'Yes', fontsize=7, color=WONG_PALETTE['bluish_green'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf
