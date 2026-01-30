"""
HIRT PDF Generator - Subsurface Diagrams Module

Functions for creating ground layers, probes, ray paths, and cross-section diagrams.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, Ellipse, FancyArrowPatch
)
import numpy as np
import random
from io import BytesIO

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
    'ground_tan': '#d4a373',
    'orange': '#c05621',
    'purple': '#6b46c1',
    'gray_dark': '#2d3748',
    'gray_med': '#718096',
    'gray_light': '#cbd5e0',
    'sky': '#e8f4f8',
    'ground_dark': '#654321',
    'tx_coil': '#2f855a',
    'rx_coil': '#3182ce',
    'ert_ring': '#c05621',
    'probe_body': '#2c5282',
    'connector': '#4a5568',
}


def draw_soil_texture(ax, xmin, xmax, ytop, ybottom, type='sand', density=0.5):
    """
    Draw realistic soil texture patterns.
    """
    width = xmax - xmin
    height = abs(ytop - ybottom)
    area = width * height
    
    if type == 'sand':
        # Random dots
        n_dots = int(50 * density * area)
        if n_dots > 0:
            x = np.random.uniform(xmin, xmax, n_dots)
            y = np.random.uniform(ybottom, ytop, n_dots)
            ax.scatter(x, y, s=0.5, c='#8c6239', alpha=0.4, marker='.', zorder=1)
            
    elif type == 'clay':
        # Horizontal dashes
        n_dashes = int(20 * density * area)
        if n_dashes > 0:
            x = np.random.uniform(xmin, xmax, n_dashes)
            y = np.random.uniform(ybottom, ytop, n_dashes)
            # Create short horizontal lines
            for i in range(n_dashes):
                ax.plot([x[i], x[i] + 0.2], [y[i], y[i]], color='#5d4037', lw=0.5, alpha=0.3, zorder=1)
                
    elif type == 'loam':
        # Mix of dots and dashes
        draw_soil_texture(ax, xmin, xmax, ytop, ybottom, 'sand', density * 0.5)
        draw_soil_texture(ax, xmin, xmax, ytop, ybottom, 'clay', density * 0.5)


def draw_probe(ax, x, top_y, length, components=None, show_label=True, label=None):
    """
    Draw a single probe with internal components.

    Args:
        ax: matplotlib Axes object
        x: X position of probe center
        top_y: Y position of probe top (usually 0 for ground surface)
        length: Total probe length (positive value, extends downward)
        components: List of dicts with keys:
            - 'depth': depth from top (positive)
            - 'type': 'tx', 'rx', 'ert', 'joint', 'tip'
            - 'size': optional size override
        show_label: Whether to show probe label
        label: Custom label text

    Returns:
        Dict with component positions for reference
    """
    probe_width = 0.16
    half_width = probe_width / 2

    # Main probe body
    ax.add_patch(Rectangle((x - half_width, top_y - length), probe_width, length,
                           color=COLORS['probe_body'], ec='black', lw=1, zorder=10))

    # Junction box at top
    ax.add_patch(Rectangle((x - 0.2, top_y - 0.1), 0.4, 0.3,
                           color=COLORS['connector'], ec='black', lw=1, zorder=11))

    positions = {}

    if components:
        for comp in components:
            depth = comp['depth']
            comp_type = comp['type']
            size = comp.get('size', 0.12)
            y_pos = top_y - depth

            if comp_type == 'tx':
                ax.add_patch(Circle((x, y_pos), size, color=COLORS['tx_coil'],
                                   ec='black', lw=1, zorder=12))
                positions['tx'] = positions.get('tx', []) + [(x, y_pos)]
            elif comp_type == 'rx':
                ax.add_patch(Circle((x, y_pos), size, color=COLORS['rx_coil'],
                                   ec='black', lw=1, zorder=12))
                positions['rx'] = positions.get('rx', []) + [(x, y_pos)]
            elif comp_type == 'ert':
                ax.add_patch(Rectangle((x - 0.15, y_pos - 0.05), 0.3, 0.1,
                                       color=COLORS['ert_ring'], ec='black', lw=0.5, zorder=12))
                positions['ert'] = positions.get('ert', []) + [(x, y_pos)]
            elif comp_type == 'joint':
                ax.add_patch(Rectangle((x - half_width, y_pos - 0.04), probe_width, 0.08,
                                       color=COLORS['gray_light'], ec='black', lw=0.5, zorder=12))
            elif comp_type == 'tip':
                # Pointed tip
                tip_y = top_y - length
                points = [(x - half_width, tip_y + 0.15),
                         (x + half_width, tip_y + 0.15),
                         (x, tip_y)]
                ax.add_patch(Polygon(points, color=COLORS['primary'],
                                    ec='black', lw=1, zorder=12))

    if show_label and label:
        ax.text(x, top_y + 0.35, label, ha='center', fontsize=9, fontweight='bold')

    return positions


def draw_probe_array(ax, positions, top_y=0, length=3.0, components=None, labels=None):
    """
    Draw an array of probes.

    Args:
        ax: matplotlib Axes object
        positions: List of x positions for probes
        top_y: Y position of ground surface
        length: Probe length
        components: Component spec (same for all probes) or list of specs
        labels: List of labels or None

    Returns:
        List of component position dicts
    """
    all_positions = []

    for i, x in enumerate(positions):
        label = labels[i] if labels else f'P{i+1}'
        comp = components[i] if isinstance(components, list) else components
        pos = draw_probe(ax, x, top_y, length, comp, show_label=True, label=label)
        all_positions.append(pos)

    return all_positions


def draw_ray_paths(ax, probe_positions, depths, color=None, alpha=0.5, lw=1.2):
    """
    Draw ray paths between probes at specified depths.

    Args:
        ax: matplotlib Axes object
        probe_positions: List of (x, top_y) tuples for each probe
        depths: List of depths where rays connect
        color: Line color (default: green)
        alpha: Line transparency
        lw: Line width
    """
    if color is None:
        color = COLORS['success']

    for depth in depths:
        for i, (x1, top1) in enumerate(probe_positions):
            for j, (x2, top2) in enumerate(probe_positions):
                if i < j:
                    y1 = top1 - depth
                    y2 = top2 - depth
                    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, lw=lw)


def draw_ray_matrix(ax, left_pos, right_pos, left_depths, right_depths, color=None, alpha=0.5):
    """
    Draw full ray path matrix between two probes.

    Args:
        ax: matplotlib Axes object
        left_pos: (x, top_y) for left probe
        right_pos: (x, top_y) for right probe
        left_depths: List of sensor depths on left probe
        right_depths: List of sensor depths on right probe
        color: Line color
        alpha: Transparency
    """
    if color is None:
        colors_list = plt.cm.Greens(np.linspace(0.4, 0.8, len(left_depths) * len(right_depths)))
    else:
        colors_list = [color] * (len(left_depths) * len(right_depths))

    idx = 0
    for d1 in left_depths:
        for d2 in right_depths:
            y1 = left_pos[1] - d1
            y2 = right_pos[1] - d2
            ax.plot([left_pos[0], right_pos[0]], [y1, y2],
                   color=colors_list[idx] if isinstance(colors_list, np.ndarray) else colors_list[idx],
                   alpha=alpha, lw=1.2)
            idx += 1


def draw_ground_layers(ax, xmin, xmax, layer_specs):
    """
    Draw multiple ground layers with customizable properties and textures.
    """
    for i, layer in enumerate(layer_specs):
        top = -layer.get('top', 0)
        bottom = -layer.get('bottom', 0)
        color = layer.get('color', COLORS['ground_tan'])
        alpha = layer.get('alpha', 0.3 + 0.1*i)
        texture = layer.get('texture', 'sand' if i == 0 else 'clay') # Default textures

        # Background fill
        ax.fill_between([xmin, xmax], [top, top], [bottom, bottom],
                       color=color, alpha=alpha, zorder=0)
        
        # Texture overlay
        draw_soil_texture(ax, xmin, xmax, top, bottom, texture)

        if 'label' in layer:
            lx, ly = layer.get('label_pos', ((xmin + xmax)/2, (top + bottom)/2))
            ax.text(lx, ly, layer['label'], ha='center', va='center',
                   fontsize=8, color=COLORS['gray_dark'], 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                   zorder=5)


def draw_target(ax, x, y, radius=0.5, color=None, label=None, label_pos=None):
    """
    Draw a target anomaly (e.g., UXB, burial).

    Args:
        ax: matplotlib Axes object
        x, y: Center position
        radius: Target radius
        color: Fill color (default: red)
        label: Optional label text
        label_pos: (x, y) for label or None for auto
    """
    if color is None:
        color = COLORS['warning']

    target = Circle((x, y), radius, color=color, ec='#9b2c2c', lw=2, zorder=15)
    ax.add_patch(target)

    if label:
        if label_pos is None:
            label_pos = (x + radius + 0.5, y)
        ax.annotate(label, (x, y), label_pos,
                   fontsize=8, ha='left', va='center',
                   arrowprops=dict(arrowstyle='->', color=color))


def draw_sensitivity_zone(ax, x, y, width, height, color=None, label=None):
    """
    Draw an elliptical sensitivity zone between probes.

    Args:
        ax: matplotlib Axes object
        x, y: Center of ellipse
        width, height: Ellipse dimensions
        color: Fill color (default: green)
        label: Optional label text
    """
    if color is None:
        color = COLORS['success']

    ellipse = Ellipse((x, y), width, height, color=color, alpha=0.15, zorder=5)
    ax.add_patch(ellipse)

    if label:
        ax.text(x, y, label, ha='center', va='center',
               fontsize=8, color='#276749', fontweight='bold')


def draw_crosshole_comparison(ax_surface, ax_crosshole, target_depth=3.5):
    """
    Draw side-by-side comparison of surface vs crosshole methods.

    Args:
        ax_surface: Axes for surface method panel
        ax_crosshole: Axes for crosshole method panel
        target_depth: Depth of target anomaly

    Returns:
        Tuple of (surface_buf, crosshole_buf) figure buffers
    """
    # Common setup
    for ax, title in [(ax_surface, '(a) Surface Method'),
                      (ax_crosshole, '(b) HIRT Crosshole Method')]:
        ax.set_xlim(-5, 5)
        ax.set_ylim(-6.5, 1)

        # Sky
        ax.fill_between([-5, 5], [0, 0], [1, 1], color=COLORS['sky'], alpha=0.5)

        # Ground layers
        for d1, d2, alpha in [(0, -1.5, 0.3), (-1.5, -3, 0.4),
                              (-3, -4.5, 0.5), (-4.5, -6.5, 0.6)]:
            ax.fill_between([-5, 5], [d1, d1], [d2, d2],
                           color=COLORS['ground_tan'], alpha=alpha)

        # Ground surface
        ax.axhline(0, color=COLORS['ground_dark'], linewidth=2.5)

        # Depth markers
        for d in [1, 2, 3, 4, 5, 6]:
            ax.axhline(-d, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
            ax.text(4.7, -d, f'{d}m', fontsize=7, va='center', color='gray')

        ax.set_title(title, fontsize=11, fontweight='bold', color=COLORS['primary'], pad=10)
        ax.set_yticks([0, -2, -4, -6])
        ax.set_yticklabels(['0', '2', '4', '6'])
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # Surface method specifics
    ax_surface.set_ylabel('Depth (m)', fontsize=9)
    ax_surface.text(-4.8, 0.15, 'Ground Surface', fontsize=8, color=COLORS['ground_dark'])

    # Surface sensors
    for x in [-3, -1, 1, 3]:
        ax_surface.add_patch(Polygon([[x-0.3, 0], [x+0.3, 0], [x, 0.4]],
                                     color=COLORS['secondary'], ec='black', lw=1))
    ax_surface.text(0, 0.65, 'Surface Sensors', ha='center', fontsize=9, fontweight='bold')

    # Sensitivity decay
    for radius, alpha in [(1.5, 0.3), (2.5, 0.2), (3.5, 0.15), (4.5, 0.1)]:
        theta = np.linspace(np.pi, 2*np.pi, 50)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        ax_surface.plot(x, y, color=COLORS['accent'], alpha=alpha, linewidth=1.5)

    # Target (weak signal)
    target = Circle((0, -target_depth), 0.5, color=COLORS['warning'], ec='#9b2c2c', lw=2)
    ax_surface.add_patch(target)
    ax_surface.annotate('Target\n(weak signal)', (0, -target_depth), (2.5, -target_depth),
                       fontsize=8, ha='left', va='center',
                       arrowprops=dict(arrowstyle='->', color=COLORS['warning']))

    ax_surface.text(-4.5, -1.2, 'HIGH\nSensitivity', fontsize=7, color=COLORS['secondary'], alpha=0.8)
    ax_surface.text(-4.5, -3, 'LOW\nSensitivity', fontsize=7, color=COLORS['secondary'], alpha=0.4)

    # Crosshole method specifics
    probe_x = [-2.5, 2.5]
    for x in probe_x:
        # Probe rod
        ax_crosshole.add_patch(Rectangle((x-0.15, -5), 0.3, 5,
                                         color=COLORS['probe_body'], ec='black', lw=1))
        # Junction box
        ax_crosshole.add_patch(Rectangle((x-0.35, -0.1), 0.7, 0.4,
                                         color=COLORS['connector'], ec='black', lw=1))

        # Sensors
        for depth, color in [(-1.5, COLORS['tx_coil']), (-3, COLORS['rx_coil']),
                            (-4.5, COLORS['rx_coil'])]:
            ax_crosshole.add_patch(Circle((x, depth), 0.2, color=color, ec='black', lw=1))

        # ERT rings
        for depth in [-2, -3.5, -4.8]:
            ax_crosshole.add_patch(Rectangle((x-0.25, depth-0.08), 0.5, 0.16,
                                            color=COLORS['ert_ring'], ec='black', lw=0.5))

    ax_crosshole.text(-2.5, 0.6, 'Probe 1', ha='center', fontsize=9, fontweight='bold')
    ax_crosshole.text(2.5, 0.6, 'Probe 2', ha='center', fontsize=9, fontweight='bold')

    # Ray paths
    ray_colors = plt.cm.Greens(np.linspace(0.4, 0.8, 9))
    depths_left = [-1.5, -1.5, -1.5, -3, -3, -3, -4.5, -4.5, -4.5]
    depths_right = [-1.5, -3, -4.5, -1.5, -3, -4.5, -1.5, -3, -4.5]
    for i, (d1, d2) in enumerate(zip(depths_left, depths_right)):
        ax_crosshole.plot([-2.3, 2.3], [d1, d2], color=ray_colors[i], lw=1.5, alpha=0.7)

    # Sensitivity zone
    ellipse = Ellipse((0, -3), 4, 3, color=COLORS['success'], alpha=0.15)
    ax_crosshole.add_patch(ellipse)
    ax_crosshole.text(0, -3, 'HIGH\nSensitivity\nZone', ha='center', va='center',
                     fontsize=8, color='#276749', fontweight='bold')

    # Target (strong signal)
    target = Circle((0, -target_depth), 0.5, color=COLORS['warning'], ec='#9b2c2c', lw=2)
    ax_crosshole.add_patch(target)
    ax_crosshole.annotate('Target\n(strong signal)', (0, -target_depth), (-3.5, -5.5),
                         fontsize=8, ha='center', va='center',
                         arrowprops=dict(arrowstyle='->', color=COLORS['warning']))

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['tx_coil'], label='TX Coil'),
        mpatches.Patch(color=COLORS['rx_coil'], label='RX Coil'),
        mpatches.Patch(color=COLORS['ert_ring'], label='ERT Ring'),
    ]
    ax_crosshole.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.9)


def create_crosshole_comparison_figure():
    """
    Create the complete crosshole comparison figure.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    draw_crosshole_comparison(ax1, ax2)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_crater_investigation_figure():
    """
    Create a bomb crater investigation diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 1.5)
    ax.set_aspect('equal')

    # Sky and ground
    ax.fill_between([-6, 6], [0, 0], [1.5, 1.5], color=COLORS['sky'], alpha=0.5)
    ax.fill_between([-6, 6], [0, 0], [-5, -5], color=COLORS['ground_tan'], alpha=0.3)
    ax.axhline(0, color=COLORS['ground_dark'], linewidth=2.5)

    # Crater shape with stochastic noise
    crater_x = np.linspace(-3, 3, 100)
    base_y = -0.5 - 3 * (1 - (crater_x/3)**2)
    # Add noise
    noise = np.random.normal(0, 0.05, size=len(crater_x))
    # Smooth noise at edges to match ground
    noise[0:10] = noise[0:10] * np.linspace(0, 1, 10)
    noise[-10:] = noise[-10:] * np.linspace(1, 0, 10)
    crater_y = base_y + noise
    
    # Fill crater
    ax.fill_between(crater_x, crater_y, 0, color=COLORS['gray_light'], alpha=0.5, zorder=1)
    ax.plot(crater_x, crater_y, color=COLORS['gray_dark'], linewidth=1.0, zorder=2)
    
    # Add texture to crater fill (disturbed soil dots)
    n_fill_dots = 200
    fill_x = np.random.uniform(-2.8, 2.8, n_fill_dots)
    fill_y = np.random.uniform(-3.5, -0.1, n_fill_dots)
    
    # Filter points inside crater parabola approx
    valid_mask = fill_y > (-0.5 - 3 * (1 - (fill_x/3)**2))
    ax.scatter(fill_x[valid_mask], fill_y[valid_mask], s=1, c='#4a5568', alpha=0.5, marker='.', zorder=1)

    ax.text(0, -1.2, 'Crater fill\n(disturbed soil)', fontsize=8, ha='center',
           color=COLORS['gray_dark'], fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.6, edgecolor='none'),
           zorder=5)

    # Probes around perimeter
    probe_x = [-4.5, -2.5, 0, 2.5, 4.5]
    for px in probe_x:
        draw_probe(ax, px, 0, 4.0, components=[
            {'depth': 1.0, 'type': 'tx'},
            {'depth': 2.0, 'type': 'rx'},
            {'depth': 3.0, 'type': 'rx'},
            {'depth': 1.5, 'type': 'ert'},
            {'depth': 2.5, 'type': 'ert'},
            {'depth': 3.5, 'type': 'ert'},
        ], show_label=False)

    # Ray paths
    for d in [1.0, 2.0, 3.0]:
        for i, p1 in enumerate(probe_x[:-1]):
            for p2 in probe_x[i+1:]:
                ax.plot([p1, p2], [-d, -d], color=COLORS['success'], alpha=0.3, lw=0.8)

    # Target UXB
    target = Circle((0, -3.5), 0.5, color=COLORS['warning'], ec='#9b2c2c', lw=2, zorder=20)
    ax.add_patch(target)
    ax.text(0, -3.5, 'UXB', fontsize=7, ha='center', va='center',
           color='white', fontweight='bold', zorder=21)

    # Annotations
    ax.annotate('Perimeter probes\n(5×4m array)', (-4.5, 0.8), fontsize=8, ha='center')
    ax.annotate('', xy=(0, -3.5), xytext=(2, -4.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['warning']))
    ax.text(2.2, -4.5, 'Target at 3.5m depth', fontsize=8, va='center')

    # Depth scale
    for d in [1, 2, 3, 4]:
        ax.axhline(-d, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.text(5.7, -d, f'{d}m', fontsize=7, va='center', color='gray')

    ax.set_title('Bomb Crater Investigation Configuration', fontsize=11,
                fontweight='bold', color=COLORS['primary'], pad=10)
    ax.set_ylabel('Depth (m)', fontsize=9)
    ax.set_yticks([0, -2, -4])
    ax.set_yticklabels(['0', '2', '4'])
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf
