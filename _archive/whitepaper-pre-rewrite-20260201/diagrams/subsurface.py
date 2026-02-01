"""
HIRT PDF Generator - Subsurface Diagrams Module

Functions for creating ground layers, probes, ray paths, and cross-section diagrams.
Enhanced with realistic soil textures, Fresnel zones, and improved target rendering.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, Ellipse, FancyArrowPatch
)
import matplotlib.colors as mcolors
import numpy as np
import random
from io import BytesIO

# Import utility functions from parent module
from . import (
    draw_soil_texture_advanced,
    draw_sphere_gradient,
    WONG_PALETTE,
    COLORS as PARENT_COLORS
)

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

# Color palette (Muted Academic) - Extended with Wong palette integration
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
    # Wong colorblind-safe colors for sensitivity visualization
    'fresnel_high': WONG_PALETTE['bluish_green'],
    'fresnel_med': WONG_PALETTE['sky_blue'],
    'fresnel_low': WONG_PALETTE['yellow'],
    'eddy_current': WONG_PALETTE['vermillion'],
    'anomaly_halo': WONG_PALETTE['reddish_purple'],
}


def draw_soil_texture(ax, xmin, xmax, ytop, ybottom, type='sand', density=0.5):
    """
    Draw realistic soil texture patterns.

    This is a legacy function maintained for backwards compatibility.
    For enhanced rendering, use draw_soil_texture_advanced() from the parent module.
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


def draw_gravel_inclusions(ax, xmin, xmax, ytop, ybottom, density=1.0, seed=None):
    """
    Draw scattered gravel/rock inclusions as small polygons.

    Args:
        ax: matplotlib Axes object
        xmin, xmax: Horizontal extent
        ytop, ybottom: Vertical extent
        density: Relative density multiplier
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    width = xmax - xmin
    height = abs(ytop - ybottom)
    n_gravel = int(width * height * 2 * density)

    for _ in range(n_gravel):
        # Random position
        gx = np.random.uniform(xmin, xmax)
        gy = np.random.uniform(min(ytop, ybottom), max(ytop, ybottom))

        # Random size and shape (irregular polygon)
        size = np.random.uniform(0.03, 0.12)
        n_vertices = np.random.randint(4, 8)
        angles = np.sort(np.random.uniform(0, 2*np.pi, n_vertices))
        radii = size * np.random.uniform(0.6, 1.0, n_vertices)

        vertices = [(gx + r * np.cos(a), gy + r * np.sin(a))
                    for r, a in zip(radii, angles)]

        # Random gray color
        gray = np.random.uniform(0.35, 0.65)
        gravel = Polygon(vertices, facecolor=(gray, gray, gray),
                        edgecolor=(gray*0.7, gray*0.7, gray*0.7),
                        linewidth=0.3, alpha=0.7, zorder=2)
        ax.add_patch(gravel)


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


def draw_fresnel_zone(ax, x1, y1, x2, y2, wavelength=0.5, zone_number=1, color=None, alpha=0.15):
    """
    Draw first Fresnel zone ellipse around a ray path.

    The Fresnel zone is the elliptical region where most signal energy propagates.
    Width = sqrt(n * wavelength * distance) where n is zone number.

    Args:
        ax: matplotlib Axes object
        x1, y1: Start point coordinates
        x2, y2: End point coordinates
        wavelength: Signal wavelength in meters (default 0.5m for typical EMI)
        zone_number: Fresnel zone number (1 = first zone)
        color: Fill color (default: based on sensitivity)
        alpha: Transparency

    Returns:
        Ellipse patch object
    """
    # Calculate ray path properties
    dx = x2 - x1
    dy = y2 - y1
    distance = np.sqrt(dx**2 + dy**2)
    angle = np.degrees(np.arctan2(dy, dx))

    # First Fresnel zone radius at midpoint
    # r = sqrt(n * lambda * d1 * d2 / (d1 + d2)) where d1 = d2 = distance/2
    fresnel_radius = np.sqrt(zone_number * wavelength * distance / 4)

    # Midpoint
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2

    # Default color based on zone number (sensitivity indicator)
    if color is None:
        if zone_number == 1:
            color = COLORS['fresnel_high']
        elif zone_number == 2:
            color = COLORS['fresnel_med']
        else:
            color = COLORS['fresnel_low']

    # Draw ellipse (width = distance, height = 2 * fresnel_radius)
    ellipse = Ellipse((mx, my), distance, 2 * fresnel_radius,
                      angle=angle, facecolor=color, edgecolor=color,
                      alpha=alpha, linewidth=0.5, linestyle='--', zorder=3)
    ax.add_patch(ellipse)
    return ellipse


def draw_ray_paths(ax, probe_positions, depths, color=None, alpha=0.5, lw=1.2,
                   show_fresnel=False, fresnel_wavelength=0.5, color_by_sensitivity=False):
    """
    Draw ray paths between probes at specified depths.

    Args:
        ax: matplotlib Axes object
        probe_positions: List of (x, top_y) tuples for each probe
        depths: List of depths where rays connect
        color: Line color (default: green)
        alpha: Line transparency
        lw: Line width
        show_fresnel: Whether to draw first Fresnel zone ellipses
        fresnel_wavelength: Wavelength for Fresnel zone calculation
        color_by_sensitivity: Color rays by geometric sensitivity contribution
    """
    if color is None:
        color = COLORS['success']

    # Calculate total number of rays for sensitivity normalization
    total_rays = 0
    ray_data = []
    for depth in depths:
        for i, (x1, top1) in enumerate(probe_positions):
            for j, (x2, top2) in enumerate(probe_positions):
                if i < j:
                    y1 = top1 - depth
                    y2 = top2 - depth
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    ray_data.append((x1, y1, x2, y2, distance, depth))
                    total_rays += 1

    # Normalize sensitivity by inverse distance (closer = more sensitive)
    if ray_data:
        max_dist = max(r[4] for r in ray_data)

    for x1, y1, x2, y2, distance, depth in ray_data:
        # Color by sensitivity if requested
        if color_by_sensitivity:
            # Sensitivity inversely proportional to distance
            sensitivity = 1.0 - (distance / max_dist) * 0.7
            ray_color = plt.cm.Greens(0.3 + 0.6 * sensitivity)
        else:
            ray_color = color

        ax.plot([x1, x2], [y1, y2], color=ray_color, alpha=alpha, lw=lw, zorder=4)

        # Draw Fresnel zone if requested
        if show_fresnel:
            draw_fresnel_zone(ax, x1, y1, x2, y2, wavelength=fresnel_wavelength,
                            alpha=0.08)


def draw_ray_matrix(ax, left_pos, right_pos, left_depths, right_depths, color=None, alpha=0.5,
                    show_fresnel=False, fresnel_wavelength=0.5, color_by_sensitivity=False):
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
        show_fresnel: Whether to draw first Fresnel zone ellipses
        fresnel_wavelength: Wavelength for Fresnel zone calculation
        color_by_sensitivity: Color rays by geometric sensitivity contribution
    """
    n_rays = len(left_depths) * len(right_depths)

    if color is None:
        colors_list = plt.cm.Greens(np.linspace(0.4, 0.8, n_rays))
    else:
        colors_list = [color] * n_rays

    # Pre-calculate ray data for sensitivity coloring
    ray_data = []
    for d1 in left_depths:
        for d2 in right_depths:
            y1 = left_pos[1] - d1
            y2 = right_pos[1] - d2
            x1, x2 = left_pos[0], right_pos[0]
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            ray_data.append((x1, y1, x2, y2, distance))

    max_dist = max(r[4] for r in ray_data) if ray_data else 1.0

    idx = 0
    for x1, y1, x2, y2, distance in ray_data:
        # Determine ray color
        if color_by_sensitivity:
            sensitivity = 1.0 - (distance / max_dist) * 0.6
            ray_color = plt.cm.Greens(0.3 + 0.6 * sensitivity)
        elif isinstance(colors_list, np.ndarray):
            ray_color = colors_list[idx]
        else:
            ray_color = colors_list[idx]

        ax.plot([x1, x2], [y1, y2], color=ray_color, alpha=alpha, lw=1.2, zorder=4)

        # Draw Fresnel zone if requested
        if show_fresnel:
            draw_fresnel_zone(ax, x1, y1, x2, y2, wavelength=fresnel_wavelength,
                            alpha=0.06)

        idx += 1


def draw_ground_layers(ax, xmin, xmax, layer_specs, use_advanced_texture=True,
                       moisture_gradient=True, include_gravel=True):
    """
    Draw multiple ground layers with customizable properties and textures.

    Args:
        ax: matplotlib Axes object
        xmin, xmax: Horizontal extent
        layer_specs: List of dicts with keys:
            - 'top': Top depth (positive, measured from surface)
            - 'bottom': Bottom depth
            - 'color': Fill color (optional)
            - 'alpha': Transparency (optional)
            - 'texture': 'sand', 'clay', 'loam', 'gravel', 'mixed' (optional)
            - 'label': Layer label text (optional)
            - 'label_pos': (x, y) position for label (optional)
            - 'moisture': Moisture level 0-1 for darkening (optional)
        use_advanced_texture: Use draw_soil_texture_advanced for realistic rendering
        moisture_gradient: Apply depth-dependent moisture darkening
        include_gravel: Add random gravel inclusions
    """
    for i, layer in enumerate(layer_specs):
        top = -layer.get('top', 0)
        bottom = -layer.get('bottom', 0)
        color = layer.get('color', COLORS['ground_tan'])
        alpha = layer.get('alpha', 0.3 + 0.1*i)
        texture = layer.get('texture', 'sand' if i == 0 else 'clay')
        moisture = layer.get('moisture', 0.3 + 0.15*i)  # Increases with depth

        if use_advanced_texture:
            # Map texture names to soil types
            soil_type_map = {
                'sand': 'sand',
                'clay': 'clay',
                'loam': 'mixed',
                'gravel': 'gravel',
                'mixed': 'mixed',
            }
            soil_type = soil_type_map.get(texture, 'mixed')

            # Use advanced texture with moisture gradient
            draw_soil_texture_advanced(ax, xmin, xmax, top, bottom,
                                       soil_type=soil_type,
                                       moisture_gradient=moisture_gradient,
                                       seed=42 + i)

            # Add gravel inclusions if requested
            if include_gravel and texture in ['gravel', 'mixed', 'loam']:
                density = 1.5 if texture == 'gravel' else 0.5
                draw_gravel_inclusions(ax, xmin, xmax, top, bottom,
                                      density=density, seed=100 + i)
        else:
            # Legacy rendering
            # Background fill with moisture-adjusted color
            if moisture_gradient:
                rgb = mcolors.to_rgb(color)
                darkened = tuple(max(0, c * (1.0 - moisture * 0.4)) for c in rgb)
                fill_color = darkened
            else:
                fill_color = color

            ax.fill_between([xmin, xmax], [top, top], [bottom, bottom],
                           color=fill_color, alpha=alpha, zorder=0)

            # Texture overlay
            draw_soil_texture(ax, xmin, xmax, top, bottom, texture)

        if 'label' in layer:
            lx, ly = layer.get('label_pos', ((xmin + xmax)/2, (top + bottom)/2))
            ax.text(lx, ly, layer['label'], ha='center', va='center',
                   fontsize=8, color=COLORS['gray_dark'],
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'),
                   zorder=5)


def draw_eddy_current_loops(ax, x, y, radius, n_loops=3, color=None):
    """
    Draw induced eddy current loops around a target as dashed ellipses.

    Eddy currents are induced in conductive targets by the TX magnetic field.
    They flow in concentric loops and generate secondary magnetic fields.

    Args:
        ax: matplotlib Axes object
        x, y: Target center position
        radius: Target radius
        n_loops: Number of current loops to draw
        color: Loop color (default: vermillion from Wong palette)
    """
    if color is None:
        color = COLORS['eddy_current']

    for i in range(1, n_loops + 1):
        loop_radius = radius * (0.4 + 0.25 * i)  # Increasing radii
        # Slightly elliptical to suggest 3D perspective
        ellipse = Ellipse((x, y), loop_radius * 2, loop_radius * 1.5,
                         facecolor='none', edgecolor=color,
                         linestyle='--', linewidth=1.0,
                         alpha=0.6 - 0.1*i, zorder=16)
        ax.add_patch(ellipse)

        # Add small arrow to indicate current direction
        arrow_angle = np.pi/4 + i * 0.3
        ax_pos = x + loop_radius * np.cos(arrow_angle)
        ay_pos = y + loop_radius * 0.75 * np.sin(arrow_angle)
        # Arrow tangent to ellipse
        ax.annotate('', xy=(ax_pos + 0.08, ay_pos + 0.06),
                   xytext=(ax_pos - 0.08, ay_pos - 0.06),
                   arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                   zorder=17)


def draw_anomaly_halo(ax, x, y, radius, halo_factor=2.5, color=None, n_rings=4):
    """
    Draw a disturbance/anomaly halo around a buried target.

    The halo represents the zone of soil disturbance and electromagnetic
    anomaly that extends beyond the physical object.

    Args:
        ax: matplotlib Axes object
        x, y: Target center position
        radius: Target radius
        halo_factor: Multiplier for halo extent (default: 2.5x radius)
        color: Halo color (default: reddish-purple from Wong palette)
        n_rings: Number of gradient rings
    """
    if color is None:
        color = COLORS['anomaly_halo']

    rgb = mcolors.to_rgb(color)

    # Draw concentric rings from outside in with increasing alpha
    for i in range(n_rings, 0, -1):
        ring_radius = radius * (1.0 + (halo_factor - 1.0) * i / n_rings)
        ring_alpha = 0.05 + 0.05 * (n_rings - i) / n_rings

        halo = Circle((x, y), ring_radius, facecolor=rgb,
                      edgecolor='none', alpha=ring_alpha, zorder=12)
        ax.add_patch(halo)

    # Add subtle dashed outline at maximum extent
    outer_ring = Circle((x, y), radius * halo_factor, facecolor='none',
                        edgecolor=color, linestyle=':', linewidth=0.5,
                        alpha=0.4, zorder=13)
    ax.add_patch(outer_ring)


def draw_target(ax, x, y, radius=0.5, color=None, label=None, label_pos=None,
                show_halo=False, show_eddy_currents=False, halo_factor=2.5):
    """
    Draw a target anomaly (e.g., UXB, burial) with optional enhancements.

    Args:
        ax: matplotlib Axes object
        x, y: Center position
        radius: Target radius
        color: Fill color (default: red)
        label: Optional label text
        label_pos: (x, y) for label or None for auto
        show_halo: Whether to draw anomaly disturbance halo
        show_eddy_currents: Whether to draw induced eddy current loops
        halo_factor: Size multiplier for halo (if enabled)
    """
    if color is None:
        color = COLORS['warning']

    # Draw halo first (lowest z-order of target elements)
    if show_halo:
        draw_anomaly_halo(ax, x, y, radius, halo_factor=halo_factor)

    # Draw eddy current loops
    if show_eddy_currents:
        draw_eddy_current_loops(ax, x, y, radius)

    # Main target body with gradient shading
    draw_sphere_gradient(ax, x, y, radius, color, light_offset=(-0.25, 0.3), n_rings=12)

    # Add edge ring for definition
    target_edge = Circle((x, y), radius, facecolor='none',
                         edgecolor='#9b2c2c', linewidth=2, zorder=18)
    ax.add_patch(target_edge)

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


def draw_crosshole_comparison(ax_surface, ax_crosshole, target_depth=3.5,
                               use_advanced_rendering=True, show_fresnel=True,
                               show_eddy_currents=True, show_halo=True):
    """
    Draw side-by-side comparison of surface vs crosshole methods.

    Args:
        ax_surface: Axes for surface method panel
        ax_crosshole: Axes for crosshole method panel
        target_depth: Depth of target anomaly
        use_advanced_rendering: Use enhanced soil textures and target rendering
        show_fresnel: Show Fresnel zone ellipses around ray paths
        show_eddy_currents: Show induced eddy current loops on targets
        show_halo: Show anomaly disturbance halos around targets

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

        # Ground layers - use advanced texture if enabled
        if use_advanced_rendering:
            layer_specs = [
                {'top': 0, 'bottom': 1.5, 'texture': 'sand', 'moisture': 0.2},
                {'top': 1.5, 'bottom': 3, 'texture': 'loam', 'moisture': 0.35},
                {'top': 3, 'bottom': 4.5, 'texture': 'clay', 'moisture': 0.5},
                {'top': 4.5, 'bottom': 6.5, 'texture': 'mixed', 'moisture': 0.65},
            ]
            draw_ground_layers(ax, -5, 5, layer_specs,
                              use_advanced_texture=True,
                              moisture_gradient=True,
                              include_gravel=True)
        else:
            # Legacy rendering
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

    # Surface sensors with gradient shading
    for x in [-3, -1, 1, 3]:
        if use_advanced_rendering:
            draw_sphere_gradient(ax_surface, x, 0.2, 0.25, COLORS['secondary'],
                                light_offset=(-0.3, 0.3), n_rings=10)
        else:
            ax_surface.add_patch(Polygon([[x-0.3, 0], [x+0.3, 0], [x, 0.4]],
                                         color=COLORS['secondary'], ec='black', lw=1))
    ax_surface.text(0, 0.65, 'Surface Sensors', ha='center', fontsize=9, fontweight='bold')

    # Sensitivity decay
    for radius, alpha in [(1.5, 0.3), (2.5, 0.2), (3.5, 0.15), (4.5, 0.1)]:
        theta = np.linspace(np.pi, 2*np.pi, 50)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        ax_surface.plot(x, y, color=COLORS['accent'], alpha=alpha, linewidth=1.5)

    # Target (weak signal) - enhanced rendering
    draw_target(ax_surface, 0, -target_depth, radius=0.5,
                show_halo=show_halo and use_advanced_rendering,
                show_eddy_currents=False,  # No eddy currents shown for weak surface detection
                halo_factor=2.0)
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

        # Sensors with sphere gradient rendering
        sensor_specs = [(-1.5, COLORS['tx_coil']), (-3, COLORS['rx_coil']),
                       (-4.5, COLORS['rx_coil'])]
        for depth, color in sensor_specs:
            if use_advanced_rendering:
                draw_sphere_gradient(ax_crosshole, x, depth, 0.2, color,
                                    light_offset=(-0.3, 0.3), n_rings=10)
            else:
                ax_crosshole.add_patch(Circle((x, depth), 0.2, color=color, ec='black', lw=1))

        # ERT rings
        for depth in [-2, -3.5, -4.8]:
            ax_crosshole.add_patch(Rectangle((x-0.25, depth-0.08), 0.5, 0.16,
                                            color=COLORS['ert_ring'], ec='black', lw=0.5))

    ax_crosshole.text(-2.5, 0.6, 'Probe 1', ha='center', fontsize=9, fontweight='bold')
    ax_crosshole.text(2.5, 0.6, 'Probe 2', ha='center', fontsize=9, fontweight='bold')

    # Ray paths with optional Fresnel zones
    depths_left = [1.5, 1.5, 1.5, 3, 3, 3, 4.5, 4.5, 4.5]
    depths_right = [1.5, 3, 4.5, 1.5, 3, 4.5, 1.5, 3, 4.5]

    if show_fresnel and use_advanced_rendering:
        # Draw ray matrix with Fresnel zones
        draw_ray_matrix(ax_crosshole, (-2.3, 0), (2.3, 0),
                       depths_left, depths_right,
                       alpha=0.7, show_fresnel=True, fresnel_wavelength=0.8,
                       color_by_sensitivity=True)
    else:
        # Legacy ray rendering
        ray_colors = plt.cm.Greens(np.linspace(0.4, 0.8, 9))
        for i, (d1, d2) in enumerate(zip(depths_left, depths_right)):
            ax_crosshole.plot([-2.3, 2.3], [-d1, -d2], color=ray_colors[i], lw=1.5, alpha=0.7)

    # Sensitivity zone
    ellipse = Ellipse((0, -3), 4, 3, color=COLORS['success'], alpha=0.15)
    ax_crosshole.add_patch(ellipse)
    ax_crosshole.text(0, -3, 'HIGH\nSensitivity\nZone', ha='center', va='center',
                     fontsize=8, color='#276749', fontweight='bold')

    # Target (strong signal) - enhanced with eddy currents and halo
    draw_target(ax_crosshole, 0, -target_depth, radius=0.5,
                show_halo=show_halo and use_advanced_rendering,
                show_eddy_currents=show_eddy_currents and use_advanced_rendering,
                halo_factor=2.5)
    ax_crosshole.annotate('Target\n(strong signal)', (0, -target_depth), (-3.5, -5.5),
                         fontsize=8, ha='center', va='center',
                         arrowprops=dict(arrowstyle='->', color=COLORS['warning']))

    # Legend with colorblind-friendly colors
    legend_elements = [
        mpatches.Patch(color=COLORS['tx_coil'], label='TX Coil'),
        mpatches.Patch(color=COLORS['rx_coil'], label='RX Coil'),
        mpatches.Patch(color=COLORS['ert_ring'], label='ERT Ring'),
    ]
    if show_eddy_currents and use_advanced_rendering:
        legend_elements.append(
            mpatches.Patch(facecolor='none', edgecolor=COLORS['eddy_current'],
                          linestyle='--', label='Eddy Currents'))
    if show_fresnel and use_advanced_rendering:
        legend_elements.append(
            mpatches.Patch(color=COLORS['fresnel_high'], alpha=0.3, label='Fresnel Zone'))

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


def create_crater_investigation_figure(use_advanced_rendering=True, show_fresnel=True,
                                       show_eddy_currents=True, show_halo=True):
    """
    Create a bomb crater investigation diagram.

    Args:
        use_advanced_rendering: Use enhanced soil textures and target rendering
        show_fresnel: Show Fresnel zone ellipses around ray paths
        show_eddy_currents: Show induced eddy current loops on targets
        show_halo: Show anomaly disturbance halos around targets

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 1.5)
    ax.set_aspect('equal')

    # Sky
    ax.fill_between([-6, 6], [0, 0], [1.5, 1.5], color=COLORS['sky'], alpha=0.5)

    # Ground with advanced texture
    if use_advanced_rendering:
        layer_specs = [
            {'top': 0, 'bottom': 2, 'texture': 'sand', 'moisture': 0.25},
            {'top': 2, 'bottom': 3.5, 'texture': 'loam', 'moisture': 0.4},
            {'top': 3.5, 'bottom': 5, 'texture': 'clay', 'moisture': 0.55},
        ]
        draw_ground_layers(ax, -6, 6, layer_specs,
                          use_advanced_texture=True,
                          moisture_gradient=True,
                          include_gravel=True)
    else:
        ax.fill_between([-6, 6], [0, 0], [-5, -5], color=COLORS['ground_tan'], alpha=0.3)

    ax.axhline(0, color=COLORS['ground_dark'], linewidth=2.5)

    # Crater shape with stochastic noise
    np.random.seed(42)  # Reproducible noise
    crater_x = np.linspace(-3, 3, 100)
    base_y = -0.5 - 3 * (1 - (crater_x/3)**2)
    # Add noise
    noise = np.random.normal(0, 0.05, size=len(crater_x))
    # Smooth noise at edges to match ground
    noise[0:10] = noise[0:10] * np.linspace(0, 1, 10)
    noise[-10:] = noise[-10:] * np.linspace(1, 0, 10)
    crater_y = base_y + noise

    # Fill crater with disturbed soil texture
    ax.fill_between(crater_x, crater_y, 0, color=COLORS['gray_light'], alpha=0.5, zorder=1)
    ax.plot(crater_x, crater_y, color=COLORS['gray_dark'], linewidth=1.0, zorder=2)

    # Add advanced texture to crater fill (disturbed soil)
    if use_advanced_rendering:
        # Add mixed debris in crater fill
        n_debris = 150
        debris_x = np.random.uniform(-2.5, 2.5, n_debris)
        debris_y = np.random.uniform(-3.2, -0.3, n_debris)

        # Filter points inside crater
        valid_mask = debris_y > (-0.5 - 3 * (1 - (debris_x/3)**2))

        for dx, dy, valid in zip(debris_x, debris_y, valid_mask):
            if valid:
                # Mix of gravel and disturbed soil fragments
                size = np.random.uniform(0.02, 0.08)
                gray = np.random.uniform(0.4, 0.7)
                if np.random.random() > 0.7:
                    # Larger fragment (polygon)
                    n_v = np.random.randint(3, 6)
                    angles = np.sort(np.random.uniform(0, 2*np.pi, n_v))
                    radii = size * np.random.uniform(0.5, 1.0, n_v)
                    verts = [(dx + r*np.cos(a), dy + r*np.sin(a)) for r, a in zip(radii, angles)]
                    ax.add_patch(Polygon(verts, facecolor=(gray, gray, gray),
                                        edgecolor='none', alpha=0.6, zorder=1))
                else:
                    # Small dot
                    ax.scatter([dx], [dy], s=2, c=[(gray, gray, gray)], alpha=0.5, zorder=1)
    else:
        # Legacy texture
        n_fill_dots = 200
        fill_x = np.random.uniform(-2.8, 2.8, n_fill_dots)
        fill_y = np.random.uniform(-3.5, -0.1, n_fill_dots)
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

    # Ray paths with optional Fresnel zones
    depths = [1.0, 2.0, 3.0]
    probe_positions = [(px, 0) for px in probe_x]

    if show_fresnel and use_advanced_rendering:
        draw_ray_paths(ax, probe_positions, depths,
                      alpha=0.4, lw=0.8,
                      show_fresnel=True, fresnel_wavelength=0.6,
                      color_by_sensitivity=True)
    else:
        for d in depths:
            for i, p1 in enumerate(probe_x[:-1]):
                for p2 in probe_x[i+1:]:
                    ax.plot([p1, p2], [-d, -d], color=COLORS['success'], alpha=0.3, lw=0.8)

    # Target UXB with enhanced rendering
    draw_target(ax, 0, -3.5, radius=0.5,
                show_halo=show_halo and use_advanced_rendering,
                show_eddy_currents=show_eddy_currents and use_advanced_rendering,
                halo_factor=2.5)
    ax.text(0, -3.5, 'UXB', fontsize=7, ha='center', va='center',
           color='white', fontweight='bold', zorder=21)

    # Annotations
    ax.annotate('Perimeter probes\n(5x4m array)', (-4.5, 0.8), fontsize=8, ha='center')
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
