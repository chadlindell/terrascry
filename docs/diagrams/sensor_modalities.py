"""
HIRT Technical Manual - Sensor Modalities Diagrams Module

Functions for creating sensor maturity framework, tilt error propagation,
and multi-modal fusion diagrams.
"""

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch
import numpy as np
from io import BytesIO

# Color palette matching the main module
COLOR_PALETTE = {
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
    'light_blue': '#ebf8ff',
    'light_green': '#c6f6d5',
    'light_orange': '#feebc8',
    'light_red': '#fff5f5',
    'light_purple': '#e9d8fd',
    # Maturity level colors
    'supported': '#38a169',       # Green - Supported Now
    'supported_bg': '#c6f6d5',
    'recommended': '#3182ce',     # Blue - Recommended Extension
    'recommended_bg': '#ebf8ff',
    'future': '#805ad5',          # Purple - Future Exploration
    'future_bg': '#e9d8fd',
}


def create_sensor_maturity_diagram():
    """
    Visual representation of the S/R/F sensor maturity classification.
    Shows each modality positioned at its maturity level on a staircase.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(5.5, 4.7, 'Sensor Maturity Framework', fontsize=13, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    # Three maturity level platforms (staircase going up left to right)
    levels = [
        ('F', 'Future\nExploration', COLOR_PALETTE['future_bg'], COLOR_PALETTE['future'],
         ['Moisture', 'pH', 'Gas', 'SP', 'DO', 'Pressure'],
         0.5, 0.8),
        ('R', 'Recommended\nExtension', COLOR_PALETTE['recommended_bg'], COLOR_PALETTE['recommended'],
         ['Accelerometer', 'Temperature'],
         3.8, 1.8),
        ('S', 'Supported\nNow', COLOR_PALETTE['supported_bg'], COLOR_PALETTE['supported'],
         ['MIT-3D', 'ERT-Lite'],
         7.3, 2.8),
    ]

    for label, title, bg_color, edge_color, sensors, x_start, y_base in levels:
        platform_w = 3.2
        platform_h = 1.5

        # Platform
        ax.add_patch(FancyBboxPatch((x_start, y_base), platform_w, platform_h,
                                     boxstyle="round,pad=0.05",
                                     facecolor=bg_color, edgecolor=edge_color,
                                     linewidth=2))

        # Level badge
        ax.add_patch(Circle((x_start + 0.4, y_base + platform_h - 0.3), 0.25,
                            facecolor=edge_color, edgecolor='white', linewidth=2))
        ax.text(x_start + 0.4, y_base + platform_h - 0.3, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

        # Title
        ax.text(x_start + platform_w/2 + 0.2, y_base + platform_h - 0.3, title,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=COLOR_PALETTE['gray_dark'])

        # Sensor chips
        chip_x = x_start + 0.3
        chip_y = y_base + 0.2
        for i, sensor in enumerate(sensors):
            col = i % 3
            row = i // 3
            cx = chip_x + col * 1.0
            cy = chip_y + row * 0.4

            ax.add_patch(FancyBboxPatch((cx, cy), 0.9, 0.3,
                                         boxstyle="round,pad=0.03",
                                         facecolor='white', edgecolor=edge_color,
                                         linewidth=1, alpha=0.9))
            ax.text(cx + 0.45, cy + 0.15, sensor, ha='center', va='center',
                    fontsize=7, color=COLOR_PALETTE['gray_dark'])

    # Arrows showing progression path
    ax.annotate('', xy=(3.6, 2.3), xytext=(3.0, 1.5),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['gray_med'],
                               lw=2, connectionstyle='arc3,rad=0.2'))
    ax.text(3.0, 1.9, 'Validate', fontsize=7, color=COLOR_PALETTE['gray_med'],
            rotation=35, ha='center')

    ax.annotate('', xy=(7.1, 3.3), xytext=(6.5, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['gray_med'],
                               lw=2, connectionstyle='arc3,rad=0.2'))
    ax.text(6.3, 2.9, 'Validate', fontsize=7, color=COLOR_PALETTE['gray_med'],
            rotation=35, ha='center')

    # Legend
    ax.text(0.5, 0.3, 'Maturity increases with validation and field testing',
            fontsize=8, style='italic', color=COLOR_PALETTE['gray_med'])

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf


def create_tilt_error_diagram():
    """
    Geometric sketch showing positional error from probe tilt.
    Vertical probe vs tilted probe at 5 degrees, with error annotation.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1.5, 4)
    ax.set_ylim(-3.5, 0.8)
    ax.set_aspect('equal')

    # Ground surface
    ax.fill_between([-1.5, 4], [0, 0], [-3.5, -3.5],
                    color=COLOR_PALETTE['ground_tan'], alpha=0.3)
    ax.axhline(0, color=COLOR_PALETTE['gray_dark'], linewidth=2)
    ax.text(-1.3, 0.15, 'Ground Surface', fontsize=8, color=COLOR_PALETTE['gray_dark'])

    # Depth markers
    for d in [1, 2, 3]:
        ax.axhline(-d, color=COLOR_PALETTE['gray_light'], linewidth=0.5, linestyle='--')
        ax.text(3.7, -d + 0.08, f'{d} m', fontsize=7, color=COLOR_PALETTE['gray_med'])

    # --- Vertical probe (ideal) ---
    probe_x = 0.5
    ax.plot([probe_x, probe_x], [0, -3.0], color=COLOR_PALETTE['secondary'],
            linewidth=4, solid_capstyle='round', zorder=5)
    # Electrode rings
    for depth in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ax.plot([probe_x - 0.12, probe_x + 0.12], [-depth, -depth],
                color=COLOR_PALETTE['orange'], linewidth=3, zorder=6)
    # Label
    ax.text(probe_x, 0.4, 'Vertical\n(assumed)', ha='center', fontsize=9,
            fontweight='bold', color=COLOR_PALETTE['secondary'])
    # Tip marker
    ax.plot(probe_x, -3.0, 'v', color=COLOR_PALETTE['secondary'], markersize=8, zorder=6)

    # --- Tilted probe (actual) ---
    tilt_deg = 5.0
    tilt_rad = np.radians(tilt_deg)
    tip_x = probe_x + 3.0 * np.sin(tilt_rad)  # ~0.26 m offset at 3m
    tip_y = -3.0 * np.cos(tilt_rad)

    # Offset for visual clarity
    offset = 2.0
    ax.plot([probe_x + offset, probe_x + offset + 3.0 * np.sin(tilt_rad)],
            [0, tip_y], color=COLOR_PALETTE['warning'],
            linewidth=4, solid_capstyle='round', zorder=5)
    # Electrode rings on tilted probe
    for depth in [0.5, 1.0, 1.5, 2.0, 2.5]:
        ring_x = probe_x + offset + depth * np.sin(tilt_rad)
        ring_y = -depth * np.cos(tilt_rad)
        # Perpendicular to probe axis
        dx = 0.12 * np.cos(tilt_rad)
        dy = 0.12 * np.sin(tilt_rad)
        ax.plot([ring_x - dx, ring_x + dx], [ring_y - dy, ring_y + dy],
                color=COLOR_PALETTE['orange'], linewidth=3, zorder=6)
    # Label
    ax.text(probe_x + offset, 0.4, 'Tilted 5 deg\n(actual)', ha='center', fontsize=9,
            fontweight='bold', color=COLOR_PALETTE['warning'])
    # Tip marker
    actual_tip_x = probe_x + offset + 3.0 * np.sin(tilt_rad)
    ax.plot(actual_tip_x, tip_y, 'v', color=COLOR_PALETTE['warning'], markersize=8, zorder=6)

    # --- Error annotation ---
    # Assumed tip position (directly below insertion point)
    assumed_tip_x = probe_x + offset
    assumed_tip_y = -3.0

    # Horizontal error line
    ax.annotate('', xy=(actual_tip_x, tip_y - 0.15),
                xytext=(assumed_tip_x, assumed_tip_y - 0.15),
                arrowprops=dict(arrowstyle='<->', color=COLOR_PALETTE['accent'],
                               lw=2))
    ax.text((actual_tip_x + assumed_tip_x) / 2, tip_y - 0.4,
            '26 cm error', ha='center', fontsize=10, fontweight='bold',
            color=COLOR_PALETTE['accent'],
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=COLOR_PALETTE['accent'], alpha=0.9))

    # Tilt angle arc
    angle_radius = 1.0
    theta = np.linspace(-np.pi/2, -np.pi/2 + tilt_rad, 20)
    arc_x = probe_x + offset + angle_radius * np.cos(theta)
    arc_y = angle_radius * np.sin(theta)
    ax.plot(arc_x, arc_y, color=COLOR_PALETTE['gray_dark'], linewidth=1.5)
    ax.text(probe_x + offset + 0.25, -0.7, '5 deg', fontsize=8,
            color=COLOR_PALETTE['gray_dark'])

    # Dashed vertical reference line for tilted probe
    ax.plot([probe_x + offset, probe_x + offset], [0, -3.2],
            color=COLOR_PALETTE['gray_light'], linewidth=1, linestyle=':')

    # Info box
    info_text = ('Uncorrected tilt causes\n'
                 'sensor ring displacement\n'
                 'from assumed positions,\n'
                 'distorting inversion results')
    ax.text(-1.3, -2.5, info_text, fontsize=8, color=COLOR_PALETTE['gray_dark'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_PALETTE['light_blue'],
                      edgecolor=COLOR_PALETTE['accent'], alpha=0.8),
            va='center')

    ax.set_xlabel('Lateral Position (m)', fontsize=9)
    ax.set_ylabel('Depth (m)', fontsize=9)
    ax.set_yticks([0, -1, -2, -3])
    ax.set_yticklabels(['0', '1', '2', '3'])

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf


def create_multimodal_fusion_diagram():
    """
    Block diagram showing how MIT-3D, ERT-Lite, and auxiliary sensors
    feed into preprocessing and joint inversion for 3D model output.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Title
    ax.text(5.5, 4.7, 'Multi-Modal Sensor Fusion Architecture', fontsize=13,
            fontweight='bold', ha='center', color=COLOR_PALETTE['primary'])

    # --- Input sensors (left column) ---
    sensors = [
        ('MIT-3D', 'Conductivity\nMapping', COLOR_PALETTE['supported_bg'],
         COLOR_PALETTE['supported']),
        ('ERT-Lite', 'Resistivity\nImaging', COLOR_PALETTE['supported_bg'],
         COLOR_PALETTE['supported']),
        ('Accelerometer', 'Geometry\nCorrection', COLOR_PALETTE['recommended_bg'],
         COLOR_PALETTE['recommended']),
        ('Temperature', 'Thermal\nAnomalies', COLOR_PALETTE['recommended_bg'],
         COLOR_PALETTE['recommended']),
    ]

    sensor_x = 0.3
    sensor_w = 2.0
    sensor_h = 0.8
    sensor_gap = 0.15
    total_h = len(sensors) * sensor_h + (len(sensors) - 1) * sensor_gap
    start_y = (4.2 - total_h) / 2 + 0.2

    sensor_centers = []
    for i, (name, desc, bg_color, edge_color) in enumerate(sensors):
        y = start_y + i * (sensor_h + sensor_gap)
        ax.add_patch(FancyBboxPatch((sensor_x, y), sensor_w, sensor_h,
                                     boxstyle="round,pad=0.05",
                                     facecolor=bg_color, edgecolor=edge_color,
                                     linewidth=2))
        ax.text(sensor_x + sensor_w/2, y + sensor_h/2 + 0.1, name,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=COLOR_PALETTE['primary'])
        ax.text(sensor_x + sensor_w/2, y + sensor_h/2 - 0.15, desc,
                ha='center', va='center', fontsize=7, color=COLOR_PALETTE['gray_dark'])
        sensor_centers.append((sensor_x + sensor_w, y + sensor_h/2))

    # --- Preprocessing block (middle) ---
    preproc_x = 3.5
    preproc_w = 2.0
    preproc_h = 2.0
    preproc_y = start_y + (total_h - preproc_h) / 2

    ax.add_patch(FancyBboxPatch((preproc_x, preproc_y), preproc_w, preproc_h,
                                 boxstyle="round,pad=0.05",
                                 facecolor=COLOR_PALETTE['light_orange'],
                                 edgecolor=COLOR_PALETTE['orange'], linewidth=2))
    ax.text(preproc_x + preproc_w/2, preproc_y + preproc_h - 0.25, 'Preprocessing',
            ha='center', fontsize=10, fontweight='bold', color=COLOR_PALETTE['primary'])
    preproc_items = ['Signal conditioning', 'Noise removal', 'Geometry correction',
                     'Calibration apply']
    for j, item in enumerate(preproc_items):
        ax.text(preproc_x + preproc_w/2, preproc_y + preproc_h - 0.6 - j * 0.3,
                item, ha='center', fontsize=7, color=COLOR_PALETTE['gray_dark'])

    # Arrows from sensors to preprocessing
    for sx, sy in sensor_centers:
        ax.annotate('', xy=(preproc_x, preproc_y + preproc_h/2),
                    xytext=(sx + 0.1, sy),
                    arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['gray_med'],
                                   lw=1.5, connectionstyle='arc3,rad=0.1'))

    # --- Joint Inversion block ---
    inv_x = 6.5
    inv_w = 2.0
    inv_h = 2.0
    inv_y = preproc_y

    ax.add_patch(FancyBboxPatch((inv_x, inv_y), inv_w, inv_h,
                                 boxstyle="round,pad=0.05",
                                 facecolor=COLOR_PALETTE['light_blue'],
                                 edgecolor=COLOR_PALETTE['accent'], linewidth=2))
    ax.text(inv_x + inv_w/2, inv_y + inv_h - 0.25, 'Joint Inversion',
            ha='center', fontsize=10, fontweight='bold', color=COLOR_PALETTE['primary'])
    inv_items = ['Structural coupling', 'Petrophysical links', 'Bayesian fusion',
                 'pyGIMLi / SimPEG']
    for j, item in enumerate(inv_items):
        ax.text(inv_x + inv_w/2, inv_y + inv_h - 0.6 - j * 0.3,
                item, ha='center', fontsize=7, color=COLOR_PALETTE['gray_dark'])

    # Arrow preprocessing -> inversion
    ax.annotate('', xy=(inv_x, inv_y + inv_h/2),
                xytext=(preproc_x + preproc_w + 0.1, preproc_y + preproc_h/2),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['orange'],
                               lw=2.5))

    # --- Output block ---
    out_x = 9.3
    out_w = 1.5
    out_h = 2.0
    out_y = inv_y

    ax.add_patch(FancyBboxPatch((out_x, out_y), out_w, out_h,
                                 boxstyle="round,pad=0.05",
                                 facecolor=COLOR_PALETTE['light_green'],
                                 edgecolor=COLOR_PALETTE['success'], linewidth=2))
    ax.text(out_x + out_w/2, out_y + out_h - 0.25, '3D Model',
            ha='center', fontsize=10, fontweight='bold', color=COLOR_PALETTE['primary'])
    out_items = ['Conductivity', 'Resistivity', 'Temperature', 'Confidence']
    for j, item in enumerate(out_items):
        ax.text(out_x + out_w/2, out_y + out_h - 0.6 - j * 0.3,
                item, ha='center', fontsize=7, color=COLOR_PALETTE['gray_dark'])

    # Arrow inversion -> output
    ax.annotate('', xy=(out_x, out_y + out_h/2),
                xytext=(inv_x + inv_w + 0.1, inv_y + inv_h/2),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['accent'],
                               lw=2.5))

    # Data flow label
    ax.text(5.5, 0.25, 'Each modality provides independent information; '
            'anomaly correlation across modalities increases detection confidence',
            ha='center', fontsize=8, style='italic', color=COLOR_PALETTE['gray_med'])

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf
