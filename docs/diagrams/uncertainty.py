"""
HIRT Diagrams - Uncertainty Visualization Module

Functions for creating diagrams that illustrate uncertainty and limitations
in geophysical tomographic imaging.

Diagrams:
- Non-uniqueness problem (multiple models producing same response)
- Resolution degradation with depth below probe tips
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, Ellipse, Polygon, FancyBboxPatch
import numpy as np
from io import BytesIO

# Import utility functions from parent module
from . import (
    COLORS, WONG_PALETTE, LINE_WEIGHTS,
    create_figure, save_figure_to_buffer, setup_axes_clean,
    draw_ground_surface, draw_ground_layers, draw_circle, draw_ellipse
)


def create_non_uniqueness_diagram():
    """
    Create diagram showing the non-uniqueness problem in geophysical inversion.

    Shows 3-4 different subsurface models that produce the same tomographic response:
    - Panel A: Single large object at depth
    - Panel B: Multiple small objects distributed
    - Panel C: Conductive layer + small target
    - Center/Top: The identical tomogram anomaly they all produce

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig = plt.figure(figsize=(12, 7))

    # Create grid layout: 2x2 for scenarios, plus top center for common tomogram
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.5, 1.5],
                          width_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)

    # Top center: Common tomogram output
    ax_tomo = fig.add_subplot(gs[0, :])

    # Bottom panels: Three different scenarios
    ax_scenario_a = fig.add_subplot(gs[1, 0])
    ax_scenario_b = fig.add_subplot(gs[1, 1])
    ax_scenario_c = fig.add_subplot(gs[1, 2])
    ax_scenario_d = fig.add_subplot(gs[2, 1])

    # === Common Tomogram (what inversion produces) ===
    ax_tomo.set_xlim(0, 10)
    ax_tomo.set_ylim(-4, 1)
    ax_tomo.set_aspect('equal')

    # Ground surface
    draw_ground_surface(ax_tomo, 0, 10, y=0)

    # Soil background
    draw_ground_layers(ax_tomo, 0, 10,
                       [(0, -1), (-1, -2), (-2, -3), (-3, -4)],
                       alphas=[0.15, 0.2, 0.25, 0.3])

    # Probes at left and right
    for x_probe in [2, 8]:
        ax_tomo.add_patch(Rectangle((x_probe - 0.1, -3), 0.2, 3,
                                    color=COLORS['probe_body'], zorder=5))

    # The tomogram anomaly (same for all scenarios below)
    # Draw as a blurred elliptical region representing high conductivity
    anomaly_x, anomaly_y = 5, -2.5
    anomaly_width, anomaly_height = 2.5, 1.2

    # Multiple overlapping ellipses for blur effect
    for i in range(4, 0, -1):
        alpha = 0.15 * (5 - i) / 4
        scale = 1 + 0.2 * i
        draw_ellipse(ax_tomo, anomaly_x, anomaly_y,
                    anomaly_width * scale, anomaly_height * scale,
                    color=COLORS['warning'], alpha=alpha)

    # Core anomaly
    draw_ellipse(ax_tomo, anomaly_x, anomaly_y,
                anomaly_width, anomaly_height,
                color=COLORS['warning'], alpha=0.5)

    # Contour lines around anomaly
    for scale in [1.2, 1.5, 1.8]:
        ell = Ellipse((anomaly_x, anomaly_y),
                     anomaly_width * scale, anomaly_height * scale,
                     fill=False, edgecolor=COLORS['warning'],
                     linewidth=1, linestyle='--', alpha=0.4)
        ax_tomo.add_patch(ell)

    # Label
    ax_tomo.text(anomaly_x, anomaly_y, 'ANOMALY', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['warning'],
                         edgecolor='none', alpha=0.8))

    ax_tomo.set_title('Measured Tomographic Response (Identical for All Scenarios Below)',
                     fontsize=12, fontweight='bold', color=COLORS['primary'], pad=15)
    ax_tomo.axis('off')

    # Arrow pointing down to scenarios
    ax_tomo.annotate('', xy=(5, -3.7), xytext=(5, -3.2),
                    arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['gray_dark']))
    ax_tomo.text(5, -3.0, 'Could represent ANY of these:', ha='center',
                fontsize=9, style='italic', color=COLORS['gray_dark'])

    # === Scenario A: Single Large Object ===
    ax_scenario_a.set_xlim(0, 10)
    ax_scenario_a.set_ylim(-4, 1)
    ax_scenario_a.set_aspect('equal')

    draw_ground_surface(ax_scenario_a, 0, 10, y=0)
    draw_ground_layers(ax_scenario_a, 0, 10,
                      [(0, -1), (-1, -2), (-2, -3), (-3, -4)],
                      alphas=[0.15, 0.2, 0.25, 0.3])

    # Probes
    for x_probe in [2, 8]:
        ax_scenario_a.add_patch(Rectangle((x_probe - 0.1, -3), 0.2, 3,
                                         color=COLORS['probe_body'], zorder=5))

    # Single large metallic object
    large_object = Circle((5, -2.5), 0.8, facecolor=COLORS['warning'],
                         edgecolor='darkred', linewidth=2, zorder=4)
    ax_scenario_a.add_patch(large_object)
    ax_scenario_a.text(5, -2.5, 'UXO', ha='center', va='center',
                      fontsize=9, fontweight='bold', color='white')

    ax_scenario_a.set_title('Scenario A:\nSingle Large Metal Object',
                           fontsize=10, fontweight='bold', color=COLORS['secondary'])
    ax_scenario_a.text(5, -3.7, '(e.g., 500 kg bomb at 2.5m depth)',
                      ha='center', fontsize=7, style='italic', color=COLORS['gray_dark'])
    ax_scenario_a.axis('off')

    # === Scenario B: Multiple Small Objects ===
    ax_scenario_b.set_xlim(0, 10)
    ax_scenario_b.set_ylim(-4, 1)
    ax_scenario_b.set_aspect('equal')

    draw_ground_surface(ax_scenario_b, 0, 10, y=0)
    draw_ground_layers(ax_scenario_b, 0, 10,
                      [(0, -1), (-1, -2), (-2, -3), (-3, -4)],
                      alphas=[0.15, 0.2, 0.25, 0.3])

    # Probes
    for x_probe in [2, 8]:
        ax_scenario_b.add_patch(Rectangle((x_probe - 0.1, -3), 0.2, 3,
                                         color=COLORS['probe_body'], zorder=5))

    # Multiple small objects distributed in space
    small_objects = [
        (4.5, -2.0, 0.35),
        (5.5, -2.3, 0.3),
        (4.8, -2.8, 0.4),
        (5.2, -2.5, 0.25),
        (5.6, -2.8, 0.3),
    ]

    for obj_x, obj_y, obj_r in small_objects:
        obj = Circle((obj_x, obj_y), obj_r, facecolor=COLORS['warning'],
                    edgecolor='darkred', linewidth=1.5, zorder=4)
        ax_scenario_b.add_patch(obj)

    ax_scenario_b.set_title('Scenario B:\nMultiple Small Metal Fragments',
                           fontsize=10, fontweight='bold', color=COLORS['secondary'])
    ax_scenario_b.text(5, -3.7, '(e.g., shrapnel cluster, 2-3m depth range)',
                      ha='center', fontsize=7, style='italic', color=COLORS['gray_dark'])
    ax_scenario_b.axis('off')

    # === Scenario C: Conductive Layer + Small Target ===
    ax_scenario_c.set_xlim(0, 10)
    ax_scenario_c.set_ylim(-4, 1)
    ax_scenario_c.set_aspect('equal')

    draw_ground_surface(ax_scenario_c, 0, 10, y=0)
    draw_ground_layers(ax_scenario_c, 0, 10,
                      [(0, -1), (-1, -2), (-2, -3), (-3, -4)],
                      alphas=[0.15, 0.2, 0.25, 0.3])

    # Probes
    for x_probe in [2, 8]:
        ax_scenario_c.add_patch(Rectangle((x_probe - 0.1, -3), 0.2, 3,
                                         color=COLORS['probe_body'], zorder=5))

    # Horizontal conductive layer (clay or saturated zone)
    layer = Rectangle((3, -2.8), 4, 0.6,
                     facecolor=COLORS['purple'], alpha=0.4,
                     edgecolor=COLORS['purple'], linewidth=1,
                     linestyle='--', zorder=2)
    ax_scenario_c.add_patch(layer)
    ax_scenario_c.text(5, -2.5, 'Conductive Layer\n(clay/saturated soil)',
                      ha='center', va='center', fontsize=7,
                      color=COLORS['purple'], fontweight='bold')

    # Small metal object embedded
    small_metal = Circle((5.2, -2.2), 0.3, facecolor=COLORS['warning'],
                        edgecolor='darkred', linewidth=2, zorder=4)
    ax_scenario_c.add_patch(small_metal)

    ax_scenario_c.set_title('Scenario C:\nConductive Layer + Small Metal',
                           fontsize=10, fontweight='bold', color=COLORS['secondary'])
    ax_scenario_c.text(5, -3.7, '(e.g., clay lens with artifact)',
                      ha='center', fontsize=7, style='italic', color=COLORS['gray_dark'])
    ax_scenario_c.axis('off')

    # === Scenario D: Disturbed Fill Only ===
    ax_scenario_d.set_xlim(0, 10)
    ax_scenario_d.set_ylim(-4, 1)
    ax_scenario_d.set_aspect('equal')

    draw_ground_surface(ax_scenario_d, 0, 10, y=0)
    draw_ground_layers(ax_scenario_d, 0, 10,
                      [(0, -1), (-1, -2), (-2, -3), (-3, -4)],
                      alphas=[0.15, 0.2, 0.25, 0.3])

    # Probes
    for x_probe in [2, 8]:
        ax_scenario_d.add_patch(Rectangle((x_probe - 0.1, -3), 0.2, 3,
                                         color=COLORS['probe_body'], zorder=5))

    # Large disturbed fill region (no metal, just soil contrast)
    fill_region = Ellipse((5, -2.5), 2.5, 1.8,
                         facecolor=COLORS['purple'], alpha=0.3,
                         edgecolor=COLORS['purple'], linewidth=2,
                         linestyle='--', zorder=2)
    ax_scenario_d.add_patch(fill_region)

    # Add texture to indicate disturbed soil
    np.random.seed(42)
    for _ in range(30):
        fx = np.random.uniform(4, 6)
        fy = np.random.uniform(-3.3, -1.7)
        if ((fx - 5)**2 / 1.25**2 + (fy + 2.5)**2 / 0.9**2) < 1:
            ax_scenario_d.plot(fx, fy, 'o', color=COLORS['purple'],
                             markersize=2, alpha=0.5)

    ax_scenario_d.text(5, -2.5, 'Disturbed Fill\n(no metal present)',
                      ha='center', va='center', fontsize=8,
                      color=COLORS['purple'], fontweight='bold')

    ax_scenario_d.set_title('Scenario D:\nBackfilled Crater (No Metal)',
                           fontsize=10, fontweight='bold', color=COLORS['secondary'])
    ax_scenario_d.text(5, -3.7, '(e.g., bomb crater, detonated/removed)',
                      ha='center', fontsize=7, style='italic', color=COLORS['gray_dark'])
    ax_scenario_d.axis('off')

    # Overall title
    fig.suptitle('Non-Uniqueness Problem: Multiple Subsurface Models, Same Tomographic Response',
                fontsize=14, fontweight='bold', color=COLORS['primary'], y=0.98)

    # Add note at bottom
    fig.text(0.5, 0.02,
            'The inversion cannot distinguish between these scenarios based solely on measured data.\n'
            'Mitigation: Use site history, geological context, multi-modal fusion, and depth constraints.',
            ha='center', fontsize=8, style='italic', color=COLORS['gray_dark'],
            bbox=dict(boxstyle='round,pad=0.8', facecolor=COLORS['light_bg'],
                     edgecolor=COLORS['gray_med'], linewidth=1))

    return save_figure_to_buffer(fig, dpi=200)


def create_resolution_degradation_diagram():
    """
    Create diagram showing sensitivity degradation below probe tips.

    Left panel: Cross-section with probes and sensitivity zones
    Right panel: Exponential decay curve with degradation thresholds

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, (ax_section, ax_curve) = plt.subplots(1, 2, figsize=(12, 6),
                                                gridspec_kw={'width_ratios': [1.2, 1]})

    # === Left Panel: Cross-section with sensitivity zones ===
    ax_section.set_xlim(-2, 8)
    ax_section.set_ylim(-5, 1)
    ax_section.set_aspect('equal')

    # Ground surface
    draw_ground_surface(ax_section, -2, 8, y=0)

    # Soil layers with increasing depth
    draw_ground_layers(ax_section, -2, 8,
                      [(0, -1), (-1, -2), (-2, -3), (-3, -4), (-4, -5)],
                      alphas=[0.15, 0.2, 0.25, 0.3, 0.35])

    # Two probes inserted to 3m depth
    probe_depth = -3
    probe_positions = [1.5, 4.5]

    for x_probe in probe_positions:
        # Probe body
        probe = Rectangle((x_probe - 0.12, probe_depth), 0.24, 3,
                         facecolor=COLORS['probe_body'],
                         edgecolor=COLORS['gray_dark'],
                         linewidth=1.5, zorder=10)
        ax_section.add_patch(probe)

        # Sensor positions along probe
        for sensor_y in [-0.5, -1.0, -1.5, -2.0, -2.5]:
            sensor = Circle((x_probe, sensor_y), 0.1,
                          facecolor=COLORS['success'],
                          edgecolor='white',
                          linewidth=1, zorder=11)
            ax_section.add_patch(sensor)

    # Sensitivity zones (colored regions)
    # High sensitivity: Between probe tips (green)
    high_sens = Rectangle((probe_positions[0], probe_depth),
                          probe_positions[1] - probe_positions[0], 3,
                          facecolor=COLORS['success'], alpha=0.25, zorder=1)
    ax_section.add_patch(high_sens)

    # Medium sensitivity: 0-0.5m below tips (yellow/orange)
    med_sens = Rectangle((probe_positions[0], probe_depth - 0.5),
                        probe_positions[1] - probe_positions[0], 0.5,
                        facecolor=WONG_PALETTE['yellow'], alpha=0.35, zorder=2)
    ax_section.add_patch(med_sens)

    # Low sensitivity: 0.5-1.0m below tips (orange/red)
    low_sens = Rectangle((probe_positions[0], probe_depth - 1.0),
                        probe_positions[1] - probe_positions[0], 0.5,
                        facecolor=COLORS['orange'], alpha=0.4, zorder=2)
    ax_section.add_patch(low_sens)

    # Very low sensitivity: >1.0m below tips (red)
    vlow_sens = Rectangle((probe_positions[0], probe_depth - 1.5),
                         probe_positions[1] - probe_positions[0], 0.5,
                         facecolor=COLORS['warning'], alpha=0.3, zorder=2)
    ax_section.add_patch(vlow_sens)

    # Probe tip depth marker
    ax_section.axhline(probe_depth, xmin=0.15, xmax=0.85,
                      color=COLORS['gray_dark'], linestyle='--',
                      linewidth=2, alpha=0.7, zorder=3)
    ax_section.text(7.5, probe_depth + 0.15, 'Probe Tip Depth (3m)',
                   fontsize=9, va='bottom', ha='right',
                   color=COLORS['gray_dark'], fontweight='bold')

    # Depth markers on right side
    for depth in [0, -1, -2, -3, -4, -5]:
        ax_section.axhline(depth, color='gray', linestyle=':',
                          alpha=0.3, linewidth=0.5, zorder=0)
        ax_section.text(7.7, depth, f'{-depth}m', fontsize=7,
                       va='center', color='gray')

    # Sensitivity zone labels
    ax_section.text(3, -1.5, 'HIGH\nSensitivity', ha='center', va='center',
                   fontsize=9, fontweight='bold', color=COLORS['success'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=COLORS['success'], linewidth=2, alpha=0.9))

    ax_section.text(3, -3.25, 'Moderate\n(~30% loss)', ha='center', va='center',
                   fontsize=7, color='#806600',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='#806600', linewidth=1, alpha=0.9))

    ax_section.text(3, -3.75, 'Weak\n(~60% loss)', ha='center', va='center',
                   fontsize=7, color=COLORS['orange'],
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor=COLORS['orange'], linewidth=1, alpha=0.9))

    ax_section.text(3, -4.25, 'Unreliable\n(>1m below tip)', ha='center', va='center',
                   fontsize=7, color=COLORS['warning'],
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor=COLORS['warning'], linewidth=1, alpha=0.9))

    ax_section.set_title('Sensitivity Distribution: Cross-Section View',
                        fontsize=11, fontweight='bold', color=COLORS['primary'])
    ax_section.axis('off')

    # === Right Panel: Exponential decay curve ===
    # Depth below probe tip (0 to 2m)
    depth_below_tip = np.linspace(0, 2.5, 100)

    # Exponential decay model: S = S0 * exp(-depth / decay_length)
    # decay_length chosen to give ~30% at 0.5m, ~10% at 1m
    decay_length = 0.5  # meters
    sensitivity = np.exp(-depth_below_tip / decay_length)

    ax_curve.plot(depth_below_tip, sensitivity * 100,
                 color=COLORS['primary'], linewidth=3, zorder=5)
    ax_curve.fill_between(depth_below_tip, 0, sensitivity * 100,
                         color=COLORS['accent'], alpha=0.2, zorder=2)

    # Threshold lines
    # 70% (30% degradation)
    ax_curve.axhline(70, color='#806600', linestyle='--',
                    linewidth=2, alpha=0.7, zorder=4)
    ax_curve.text(2.3, 72, '70%\n(30% loss)', fontsize=8,
                 va='bottom', ha='right', color='#806600',
                 fontweight='bold')

    # 40% (60% degradation)
    ax_curve.axhline(40, color=COLORS['orange'], linestyle='--',
                    linewidth=2, alpha=0.7, zorder=4)
    ax_curve.text(2.3, 42, '40%\n(60% loss)', fontsize=8,
                 va='bottom', ha='right', color=COLORS['orange'],
                 fontweight='bold')

    # 10% threshold (unreliable)
    ax_curve.axhline(10, color=COLORS['warning'], linestyle='--',
                    linewidth=2, alpha=0.7, zorder=4)
    ax_curve.text(2.3, 12, '10%\n(unreliable)', fontsize=8,
                 va='bottom', ha='right', color=COLORS['warning'],
                 fontweight='bold')

    # Depth markers for key thresholds
    depth_70 = -decay_length * np.log(0.7)
    depth_40 = -decay_length * np.log(0.4)
    depth_10 = -decay_length * np.log(0.1)

    for depth, percent, color in [(depth_70, 70, '#806600'),
                                   (depth_40, 40, COLORS['orange']),
                                   (depth_10, 10, COLORS['warning'])]:
        ax_curve.plot(depth, percent, 'o', color=color,
                     markersize=8, markeredgecolor='white',
                     markeredgewidth=1.5, zorder=6)
        ax_curve.axvline(depth, ymin=0, ymax=percent/100,
                        color=color, linestyle=':', alpha=0.4, zorder=3)
        ax_curve.text(depth, 3, f'{depth:.2f}m', ha='center',
                     fontsize=7, color=color, fontweight='bold')

    # Shaded regions matching left panel
    ax_curve.axvspan(0, 0.5, alpha=0.15, color=WONG_PALETTE['yellow'], zorder=1)
    ax_curve.axvspan(0.5, 1.0, alpha=0.15, color=COLORS['orange'], zorder=1)
    ax_curve.axvspan(1.0, 2.5, alpha=0.15, color=COLORS['warning'], zorder=1)

    # Labels and formatting
    ax_curve.set_xlabel('Depth Below Probe Tip (m)', fontsize=10, fontweight='bold')
    ax_curve.set_ylabel('Relative Sensitivity (%)', fontsize=10, fontweight='bold')
    ax_curve.set_title('Sensitivity vs. Depth Below Probe Tip',
                      fontsize=11, fontweight='bold', color=COLORS['primary'])

    ax_curve.set_xlim(0, 2.5)
    ax_curve.set_ylim(0, 105)
    ax_curve.grid(True, which='both', linestyle=':', alpha=0.4, zorder=0)

    ax_curve.spines['top'].set_visible(False)
    ax_curve.spines['right'].set_visible(False)

    # Add physics note
    ax_curve.text(0.98, 0.05,
                 'Model: exponential decay\nS = S₀ exp(-z/δ)\nδ ≈ 0.5m (modeled)',
                 transform=ax_curve.transAxes, fontsize=7,
                 ha='right', va='bottom', color=COLORS['gray_med'],
                 style='italic',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=COLORS['gray_light'], alpha=0.8))

    # Overall figure title
    fig.suptitle('Resolution Degradation Below Probe Tips',
                fontsize=13, fontweight='bold', color=COLORS['primary'], y=0.97)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Add warning note at bottom
    fig.text(0.5, 0.01,
            'WARNING: Do not extend inversions beyond probe depth. Results below tips are highly uncertain and often artifacts.',
            ha='center', fontsize=8, fontweight='bold', color=COLORS['warning'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe6e6',
                     edgecolor=COLORS['warning'], linewidth=2))

    return save_figure_to_buffer(fig, dpi=200)
