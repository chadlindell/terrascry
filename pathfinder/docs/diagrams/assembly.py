"""
Pathfinder Assembly Diagrams

Generate annotated assembly diagrams using matplotlib.

Usage:
    from docs.diagrams.assembly import create_exploded_view, create_assembled_view

    fig = create_exploded_view()
    fig.savefig('exploded.png', dpi=150)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyBboxPatch
from matplotlib.lines import Line2D

from . import (
    COLORS, SCALE, DIMENSIONS, mm_to_units,
    draw_cylinder_gradient, draw_dimension_line, draw_circled_number,
    draw_callout_line, setup_figure, add_legend, save_figure
)


def draw_crossbar(ax, x, y, length, od, zorder=10):
    """Draw the carbon fiber crossbar."""
    l = mm_to_units(length)
    d = mm_to_units(od)

    draw_cylinder_gradient(ax, x - l/2, y - d/2, l, d,
                          COLORS['carbon_fiber'], COLORS['carbon_fiber_light'],
                          horizontal=True, zorder=zorder)
    return (x, y)


def draw_drop_tube(ax, x, y, length, od, zorder=10):
    """Draw a PVC drop tube."""
    l = mm_to_units(length)
    d = mm_to_units(od)

    draw_cylinder_gradient(ax, x - d/2, y - l, d, l,
                          COLORS['pvc_dark'], COLORS['pvc'],
                          horizontal=False, zorder=zorder)
    return (x, y - l/2)


def draw_sensor(ax, x, y, length, diameter, zorder=15):
    """Draw a fluxgate sensor."""
    l = mm_to_units(length)
    d = mm_to_units(diameter)

    draw_cylinder_gradient(ax, x - d/2, y - l/2, d, l,
                          COLORS['sensor'], COLORS['sensor_light'],
                          horizontal=False, zorder=zorder)

    # End caps
    cap_h = mm_to_units(5)
    cap_d = mm_to_units(diameter + 2)
    rect_top = Rectangle((x - cap_d/2, y + l/2), cap_d, cap_h,
                         facecolor=COLORS['sensor'], edgecolor=COLORS['sensor'],
                         zorder=zorder)
    rect_bot = Rectangle((x - cap_d/2, y - l/2 - cap_h), cap_d, cap_h,
                         facecolor=COLORS['sensor'], edgecolor=COLORS['sensor'],
                         zorder=zorder)
    ax.add_patch(rect_top)
    ax.add_patch(rect_bot)

    return (x, y)


def draw_center_mount(ax, x, y, zorder=20):
    """Draw the center mount D-ring."""
    # Base plate
    plate_w = mm_to_units(40)
    plate_h = mm_to_units(3)
    rect = Rectangle((x - plate_w/2, y), plate_w, plate_h,
                    facecolor=COLORS['metal'], edgecolor=COLORS['metal_dark'],
                    zorder=zorder)
    ax.add_patch(rect)

    # D-ring (simplified as arc)
    ring_r = mm_to_units(15)
    theta = np.linspace(0, np.pi, 30)
    ring_x = x + ring_r * np.cos(theta)
    ring_y = y + plate_h + ring_r * np.sin(theta)
    ax.plot(ring_x, ring_y, color=COLORS['metal_dark'], linewidth=3, zorder=zorder+1)

    # Straight bottom of D
    ax.plot([x - ring_r, x + ring_r], [y + plate_h, y + plate_h],
           color=COLORS['metal_dark'], linewidth=3, zorder=zorder+1)

    return (x, y + plate_h + ring_r)


def draw_harness(ax, x, y, zorder=5):
    """Draw simplified harness straps."""
    strap_w = mm_to_units(50)
    strap_h = mm_to_units(300)
    spacing = mm_to_units(200)

    # Left strap
    rect_l = Rectangle((x - spacing/2 - strap_w/2, y), strap_w, strap_h,
                       facecolor=COLORS['harness'], edgecolor=COLORS['harness_light'],
                       alpha=0.7, zorder=zorder)
    ax.add_patch(rect_l)

    # Right strap
    rect_r = Rectangle((x + spacing/2 - strap_w/2, y), strap_w, strap_h,
                       facecolor=COLORS['harness'], edgecolor=COLORS['harness_light'],
                       alpha=0.7, zorder=zorder)
    ax.add_patch(rect_r)

    # Back panel
    panel_w = mm_to_units(250)
    panel_h = mm_to_units(100)
    rect_p = Rectangle((x - panel_w/2, y + strap_h - panel_h), panel_w, panel_h,
                       facecolor=COLORS['harness'], edgecolor=COLORS['harness_light'],
                       alpha=0.7, zorder=zorder)
    ax.add_patch(rect_p)

    return (x, y + strap_h/2)


def draw_bungee(ax, start, end, zorder=8):
    """Draw a bungee cord."""
    ax.plot([start[0], end[0]], [start[1], end[1]],
           color=COLORS['bungee'], linewidth=4, solid_capstyle='round',
           zorder=zorder)


def draw_operator_silhouette(ax, x, y, height, zorder=1):
    """Draw a simplified human silhouette."""
    h = mm_to_units(height)

    # Body proportions
    head_r = h * 0.06
    torso_h = h * 0.3
    torso_w = h * 0.15
    leg_h = h * 0.5
    leg_w = h * 0.06

    # Head
    head = Circle((x, y + h - head_r), head_r,
                 facecolor=COLORS['operator'], edgecolor='none',
                 alpha=0.5, zorder=zorder)
    ax.add_patch(head)

    # Torso
    torso = FancyBboxPatch((x - torso_w/2, y + leg_h), torso_w, torso_h,
                           boxstyle="round,pad=0.02",
                           facecolor=COLORS['operator'], edgecolor='none',
                           alpha=0.5, zorder=zorder)
    ax.add_patch(torso)

    # Legs
    for lx in [-1, 1]:
        leg = Rectangle((x + lx * leg_w - leg_w/2, y), leg_w, leg_h,
                        facecolor=COLORS['operator'], edgecolor='none',
                        alpha=0.5, zorder=zorder)
        ax.add_patch(leg)

    return (x, y + h/2)


def draw_ground_plane(ax, x_range, y, zorder=0):
    """Draw a ground reference plane."""
    rect = Rectangle((x_range[0], y - 0.1), x_range[1] - x_range[0], 0.1,
                     facecolor=COLORS['ground'], edgecolor='none',
                     alpha=0.5, zorder=zorder)
    ax.add_patch(rect)

    # Ground hatch lines
    for gx in np.arange(x_range[0], x_range[1], 0.3):
        ax.plot([gx, gx + 0.15], [y, y - 0.15],
               color=COLORS['ground'], linewidth=0.5, alpha=0.5, zorder=zorder)


def create_exploded_view(output_path=None):
    """
    Create an exploded view diagram showing all components separated.

    Args:
        output_path: Optional path to save PNG

    Returns:
        matplotlib Figure
    """
    fig, ax = setup_figure(figsize=(14, 10), title='Pathfinder Gradiometer - Exploded View')

    # Vertical spacing for explosion
    gap = 1.5

    # Component positions (from top to bottom in exploded view)
    harness_y = 8
    bungee_y = 6
    mount_y = 4.5
    crossbar_y = 3
    tube_top_y = 1.5
    top_sensor_y = 2.5
    bottom_sensor_y = -1

    # Center X
    cx = 0

    # Draw components

    # 1. Harness straps
    draw_harness(ax, cx, harness_y)
    draw_circled_number(ax, cx + 2.5, harness_y + 1.5, 1)
    draw_callout_line(ax, (cx + 2.5, harness_y + 1.5), (cx + 1.3, harness_y + 1.5))

    # 2. Bungee cords
    bungee_w = mm_to_units(100)
    ax.plot([cx - bungee_w, cx + bungee_w], [bungee_y, bungee_y],
           color=COLORS['bungee'], linewidth=6, solid_capstyle='round', zorder=10)
    draw_circled_number(ax, cx + 2.5, bungee_y, 2)
    draw_callout_line(ax, (cx + 2.5, bungee_y), (cx + 1.2, bungee_y))

    # 3. Center mount
    draw_center_mount(ax, cx, mount_y)
    draw_circled_number(ax, cx + 2.5, mount_y + 0.3, 3)
    draw_callout_line(ax, (cx + 2.5, mount_y + 0.3), (cx + 0.3, mount_y + 0.3))

    # 4. Crossbar
    draw_crossbar(ax, cx, crossbar_y, DIMENSIONS['crossbar_length'], DIMENSIONS['crossbar_od'])
    draw_circled_number(ax, cx + 11, crossbar_y, 4)
    draw_callout_line(ax, (cx + 11, crossbar_y), (cx + 10, crossbar_y))

    # Sensor positions
    positions = DIMENSIONS['sensor_positions']
    crossbar_half = mm_to_units(DIMENSIONS['crossbar_length']) / 2

    for i, pos in enumerate(positions):
        sensor_x = cx - crossbar_half + mm_to_units(pos)

        # 5. Top sensors
        draw_sensor(ax, sensor_x, top_sensor_y,
                   DIMENSIONS['sensor_length'], DIMENSIONS['sensor_diameter'])

        # 6. Drop tubes
        draw_drop_tube(ax, sensor_x, tube_top_y,
                      DIMENSIONS['drop_tube_length'], DIMENSIONS['drop_tube_od'])

        # 7. Bottom sensors
        draw_sensor(ax, sensor_x, bottom_sensor_y,
                   DIMENSIONS['sensor_length'], DIMENSIONS['sensor_diameter'])

    # Callouts for sensor pairs (just first one)
    first_x = cx - crossbar_half + mm_to_units(positions[0])
    draw_circled_number(ax, first_x - 1.5, top_sensor_y, 5)
    draw_callout_line(ax, (first_x - 1.5, top_sensor_y), (first_x - 0.3, top_sensor_y))

    draw_circled_number(ax, first_x - 1.5, tube_top_y - 2, 6)
    draw_callout_line(ax, (first_x - 1.5, tube_top_y - 2), (first_x - 0.2, tube_top_y - 2))

    draw_circled_number(ax, first_x - 1.5, bottom_sensor_y, 7)
    draw_callout_line(ax, (first_x - 1.5, bottom_sensor_y), (first_x - 0.3, bottom_sensor_y))

    # Legend
    legend_items = [
        (COLORS['harness'], '1. Harness straps'),
        (COLORS['bungee'], '2. Bungee cords'),
        (COLORS['metal'], '3. Center mount (D-ring)'),
        (COLORS['carbon_fiber'], '4. Carbon fiber crossbar'),
        (COLORS['sensor'], '5. Top sensors (FG-3+)'),
        (COLORS['pvc'], '6. PVC drop tubes'),
        (COLORS['sensor'], '7. Bottom sensors (FG-3+)'),
    ]
    add_legend(ax, legend_items, loc='upper left')

    # Set view limits
    ax.set_xlim(-12, 14)
    ax.set_ylim(-3, 12)

    if output_path:
        save_figure(fig, output_path)

    return fig


def create_assembled_view(output_path=None):
    """
    Create an assembled view showing the complete gradiometer.

    Args:
        output_path: Optional path to save PNG

    Returns:
        matplotlib Figure
    """
    fig, ax = setup_figure(figsize=(14, 8), title='Pathfinder Gradiometer - Assembled View')

    cx = 0
    crossbar_y = 0

    # Draw crossbar
    draw_crossbar(ax, cx, crossbar_y, DIMENSIONS['crossbar_length'], DIMENSIONS['crossbar_od'])

    # Draw center mount
    mount_pos = draw_center_mount(ax, cx, crossbar_y + mm_to_units(DIMENSIONS['crossbar_od']/2))

    # Sensor positions
    positions = DIMENSIONS['sensor_positions']
    crossbar_half = mm_to_units(DIMENSIONS['crossbar_length']) / 2

    for pos in positions:
        sensor_x = cx - crossbar_half + mm_to_units(pos)
        crossbar_bottom = crossbar_y - mm_to_units(DIMENSIONS['crossbar_od']/2)

        # Drop tube
        tube_center = draw_drop_tube(ax, sensor_x, crossbar_bottom,
                                     DIMENSIONS['drop_tube_length'],
                                     DIMENSIONS['drop_tube_od'])

        # Top sensor (above crossbar)
        top_y = crossbar_y + mm_to_units(DIMENSIONS['crossbar_od']/2 + DIMENSIONS['sensor_length']/2 + 10)
        draw_sensor(ax, sensor_x, top_y,
                   DIMENSIONS['sensor_length'], DIMENSIONS['sensor_diameter'])

        # Bottom sensor
        tube_bottom = crossbar_bottom - mm_to_units(DIMENSIONS['drop_tube_length'])
        bottom_y = tube_bottom + mm_to_units(DIMENSIONS['sensor_length']/2)
        draw_sensor(ax, sensor_x, bottom_y,
                   DIMENSIONS['sensor_length'], DIMENSIONS['sensor_diameter'])

    # Dimensions
    draw_dimension_line(ax, (-crossbar_half, crossbar_y + 2),
                       (crossbar_half, crossbar_y + 2),
                       '2000 mm', offset=0.3)

    # Sensor spacing
    first_x = cx - crossbar_half + mm_to_units(positions[0])
    second_x = cx - crossbar_half + mm_to_units(positions[1])
    draw_dimension_line(ax, (first_x, crossbar_y - 8),
                       (second_x, crossbar_y - 8),
                       '500 mm', offset=-0.3)

    # Drop tube length
    tube_top = crossbar_y - mm_to_units(DIMENSIONS['crossbar_od']/2)
    tube_bottom = tube_top - mm_to_units(DIMENSIONS['drop_tube_length'])
    last_x = cx - crossbar_half + mm_to_units(positions[3])
    draw_dimension_line(ax, (last_x + 1, tube_top),
                       (last_x + 1, tube_bottom),
                       '500 mm', offset=0.5)

    # Legend
    legend_items = [
        (COLORS['carbon_fiber'], 'Carbon fiber crossbar'),
        (COLORS['pvc'], 'PVC drop tubes'),
        (COLORS['sensor'], 'FG-3+ fluxgate sensors'),
        (COLORS['metal'], 'Center mount'),
    ]
    add_legend(ax, legend_items, loc='upper left')

    # Set view limits
    ax.set_xlim(-12, 14)
    ax.set_ylim(-9, 4)

    if output_path:
        save_figure(fig, output_path)

    return fig


def create_side_profile(output_path=None):
    """
    Create a side profile view showing sensor heights and gradient distance.

    Args:
        output_path: Optional path to save PNG

    Returns:
        matplotlib Figure
    """
    fig, ax = setup_figure(figsize=(10, 10), title='Pathfinder Gradiometer - Side Profile')

    cx = 0
    ground_y = 0

    # Draw ground
    draw_ground_plane(ax, (-4, 4), ground_y)
    ax.text(3, -0.3, 'Ground Level', fontsize=9, ha='right', color=COLORS['ground'])

    # Bottom sensor height from ground
    bottom_height = mm_to_units(DIMENSIONS['bottom_sensor_height'])

    # Crossbar height (bottom sensor + drop tube)
    crossbar_height = bottom_height + mm_to_units(DIMENSIONS['drop_tube_length'])
    crossbar_y = ground_y + crossbar_height

    # Draw crossbar (side view - just a small section)
    crossbar_d = mm_to_units(DIMENSIONS['crossbar_od'])
    rect = Rectangle((cx - 0.5, crossbar_y - crossbar_d/2), 1.0, crossbar_d,
                     facecolor=COLORS['carbon_fiber'], edgecolor=COLORS['carbon_fiber_light'],
                     zorder=10)
    ax.add_patch(rect)

    # Drop tube
    tube_d = mm_to_units(DIMENSIONS['drop_tube_od'])
    tube_l = mm_to_units(DIMENSIONS['drop_tube_length'])
    draw_cylinder_gradient(ax, cx - tube_d/2, crossbar_y - crossbar_d/2 - tube_l,
                          tube_d, tube_l, COLORS['pvc_dark'], COLORS['pvc'],
                          horizontal=False, zorder=10)

    # Top sensor
    top_sensor_y = crossbar_y + crossbar_d/2 + mm_to_units(DIMENSIONS['sensor_length']/2 + 10)
    draw_sensor(ax, cx, top_sensor_y,
               DIMENSIONS['sensor_length'], DIMENSIONS['sensor_diameter'], zorder=15)
    ax.text(cx + 1, top_sensor_y, 'Top Sensor\n(Reference)', fontsize=9, va='center')

    # Bottom sensor
    bottom_sensor_y = ground_y + bottom_height
    draw_sensor(ax, cx, bottom_sensor_y,
               DIMENSIONS['sensor_length'], DIMENSIONS['sensor_diameter'], zorder=15)
    ax.text(cx + 1, bottom_sensor_y, 'Bottom Sensor\n(Signal)', fontsize=9, va='center')

    # Dimension: ground to bottom sensor
    draw_dimension_line(ax, (cx - 2, ground_y),
                       (cx - 2, bottom_sensor_y),
                       f'{DIMENSIONS["bottom_sensor_height"]} mm', offset=-0.5)

    # Dimension: gradient distance (sensor separation)
    gradient_dist = mm_to_units(DIMENSIONS['drop_tube_length'])  # Approximate
    draw_dimension_line(ax, (cx + 3, bottom_sensor_y),
                       (cx + 3, top_sensor_y),
                       '~500 mm\n(Gradient\nBaseline)', offset=0.5, text_offset=0.3)

    # Dimension: total height
    total_height = top_sensor_y + mm_to_units(DIMENSIONS['sensor_length']/2)
    draw_dimension_line(ax, (cx - 3.5, ground_y),
                       (cx - 3.5, total_height),
                       'Total Height', offset=-0.5)

    # Labels
    ax.text(cx, crossbar_y + 0.3, 'Crossbar', fontsize=9, ha='center', va='bottom')
    ax.text(cx - 0.8, crossbar_y - tube_l/2, 'Drop\nTube', fontsize=8, ha='right', va='center')

    # Set view limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 10)

    if output_path:
        save_figure(fig, output_path)

    return fig


def create_operator_view(output_path=None):
    """
    Create a view showing the gradiometer worn by an operator.

    Args:
        output_path: Optional path to save PNG

    Returns:
        matplotlib Figure
    """
    fig, ax = setup_figure(figsize=(12, 12), title='Pathfinder Gradiometer - Operator View')

    cx = 0
    ground_y = 0

    # Draw ground
    draw_ground_plane(ax, (-15, 15), ground_y)

    # Operator
    operator_height = DIMENSIONS['operator_height']
    draw_operator_silhouette(ax, cx, ground_y, operator_height)

    # Gradiometer position (waist height approximately)
    gradiometer_y = ground_y + mm_to_units(1100)  # About waist height

    # Draw harness
    harness_y = ground_y + mm_to_units(1200)
    draw_harness(ax, cx, harness_y - mm_to_units(150), zorder=6)

    # Bungee cords
    mount_top_y = gradiometer_y + mm_to_units(DIMENSIONS['crossbar_od']/2 + 30)
    draw_bungee(ax, (cx - 0.5, harness_y + 1), (cx - 0.3, mount_top_y))
    draw_bungee(ax, (cx + 0.5, harness_y + 1), (cx + 0.3, mount_top_y))

    # Draw center mount
    draw_center_mount(ax, cx, gradiometer_y + mm_to_units(DIMENSIONS['crossbar_od']/2))

    # Draw crossbar
    draw_crossbar(ax, cx, gradiometer_y, DIMENSIONS['crossbar_length'], DIMENSIONS['crossbar_od'])

    # Draw sensor pairs
    positions = DIMENSIONS['sensor_positions']
    crossbar_half = mm_to_units(DIMENSIONS['crossbar_length']) / 2

    for pos in positions:
        sensor_x = cx - crossbar_half + mm_to_units(pos)
        crossbar_bottom = gradiometer_y - mm_to_units(DIMENSIONS['crossbar_od']/2)

        # Drop tube
        draw_drop_tube(ax, sensor_x, crossbar_bottom,
                      DIMENSIONS['drop_tube_length'], DIMENSIONS['drop_tube_od'])

        # Top sensor
        top_y = gradiometer_y + mm_to_units(DIMENSIONS['crossbar_od']/2 + DIMENSIONS['sensor_length']/2 + 10)
        draw_sensor(ax, sensor_x, top_y,
                   DIMENSIONS['sensor_length'], DIMENSIONS['sensor_diameter'])

        # Bottom sensor
        tube_bottom = crossbar_bottom - mm_to_units(DIMENSIONS['drop_tube_length'])
        bottom_y = tube_bottom + mm_to_units(DIMENSIONS['sensor_length']/2)
        draw_sensor(ax, sensor_x, bottom_y,
                   DIMENSIONS['sensor_length'], DIMENSIONS['sensor_diameter'])

    # Bottom sensor height dimension
    first_x = cx - crossbar_half + mm_to_units(positions[0])
    bottom_sensor_y = gradiometer_y - mm_to_units(DIMENSIONS['crossbar_od']/2 +
                                                   DIMENSIONS['drop_tube_length'] -
                                                   DIMENSIONS['sensor_length']/2)

    draw_dimension_line(ax, (first_x - 2, ground_y),
                       (first_x - 2, bottom_sensor_y - mm_to_units(DIMENSIONS['sensor_length']/2)),
                       '15-20 cm', offset=-0.5)

    # Swath width dimension
    last_x = cx - crossbar_half + mm_to_units(positions[3])
    swath_y = ground_y - 0.5
    draw_dimension_line(ax, (first_x, swath_y),
                       (last_x, swath_y),
                       '1.5 m swath', offset=-0.3)

    # Annotations
    ax.text(cx, gradiometer_y + 4, 'Hands-free\noperation', fontsize=10,
           ha='center', va='bottom', style='italic')

    ax.text(cx + 12, gradiometer_y, 'Weight on\nshoulders/hips\nvia harness',
           fontsize=9, ha='left', va='center')

    # Set view limits
    ax.set_xlim(-14, 16)
    ax.set_ylim(-2, 20)

    if output_path:
        save_figure(fig, output_path)

    return fig


def render_all(output_dir='docs/assets/images/assembly'):
    """
    Render all assembly diagrams to the output directory.

    Args:
        output_dir: Directory to save PNG files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    views = [
        ('exploded', create_exploded_view),
        ('assembled', create_assembled_view),
        ('side_profile', create_side_profile),
        ('operator', create_operator_view),
    ]

    for name, func in views:
        output_path = os.path.join(output_dir, f'pathfinder_{name}.png')
        print(f'Rendering {name}...')
        func(output_path)
        print(f'  -> {output_path}')

    print('\nAll diagrams rendered!')


if __name__ == '__main__':
    render_all()
