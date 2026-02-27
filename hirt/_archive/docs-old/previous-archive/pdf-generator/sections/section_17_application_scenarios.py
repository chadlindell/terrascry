#!/usr/bin/env python3
"""
HIRT Section 17: Application Scenarios - Publication-Quality PDF Generator

Comprehensive guide to deployment configurations for bomb craters,
woodland burials, wetland surveys, and configuration recommendations.

Run standalone: python section_17_application_scenarios.py
Output: output/sections/17-application-scenarios.pdf
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, Ellipse, FancyArrowPatch, Wedge
)
import numpy as np
from io import BytesIO

from lib.pdf_builder import SectionPDFBuilder, CONTENT_WIDTH
from lib.styles import COLOR_PALETTE, PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING
from lib.diagrams.subsurface import (
    create_crater_investigation_figure, draw_probe_array, draw_ray_paths, draw_target
)
from lib.diagrams.field_ops import create_deployment_scenarios

# Diagram color constants
COLORS = COLOR_PALETTE


# ============================================================================
# FIGURE 1: BOMB CRATER CROSS-SECTION WITH PROBES
# ============================================================================
def create_bomb_crater_diagram():
    """
    Detailed bomb crater cross-section showing probe deployment,
    crater fill zones, and ray path coverage.

    Returns:
        BytesIO buffer containing PNG image
    """
    fig, ax = plt.subplots(figsize=(10.67, 5))

    ax.set_xlim(-8, 8)
    ax.set_ylim(-5.5, 2)
    ax.set_aspect('equal')

    # Sky region
    ax.fill_between([-8, 8], [0.5, 0.5], [2, 2], color=COLORS['sky'], alpha=0.5)

    # Native soil background
    ax.fill_between([-8, 8], [0.5, 0.5], [-5.5, -5.5], color=COLORS['ground_tan'], alpha=0.4)

    # Crater rim uplift (ejecta)
    rim_left_x = np.array([-5, -4.5, -3.5, -3])
    rim_left_y = np.array([0.5, 0.7, 0.6, 0.5])
    ax.fill_between(rim_left_x, rim_left_y, 0.5, color='#b8a082', alpha=0.6)

    rim_right_x = np.array([3, 3.5, 4.5, 5])
    rim_right_y = np.array([0.5, 0.6, 0.7, 0.5])
    ax.fill_between(rim_right_x, rim_right_y, 0.5, color='#b8a082', alpha=0.6)

    # Crater bowl shape (parabolic profile)
    crater_x = np.linspace(-3.5, 3.5, 60)
    crater_y = 0.3 - 3.5 * (1 - (crater_x/3.5)**2)**0.8

    # Crater fill (disturbed soil - different layers)
    ax.fill_between(crater_x, crater_y, 0.3, color='#a0aec0', alpha=0.5)
    ax.plot(crater_x, crater_y, color=COLORS['gray_dark'], linewidth=2, linestyle='--',
            label='Crater boundary')

    # Secondary fill layer (more recent)
    crater_y2 = 0.2 - 1.5 * (1 - (crater_x/3.5)**2)**0.6
    ax.fill_between(crater_x, crater_y2, 0.2, color='#c0c8d0', alpha=0.4)

    # Water accumulation zone at bottom
    water_x = np.linspace(-1.5, 1.5, 30)
    water_y_top = -2.8 - 0.3 * (1 - (water_x/1.5)**2)
    ax.fill_between(water_x, water_y_top, -3.5, color='#90cdf4', alpha=0.4)
    ax.text(0, -3.2, 'Wet zone', fontsize=7, ha='center', color='#2b6cb0', style='italic')

    # Ground surface line
    surface_x = np.array([-8, -5, -4.5, -3.5, -3, 3, 3.5, 4.5, 5, 8])
    surface_y = np.array([0.5, 0.5, 0.7, 0.6, 0.5, 0.5, 0.6, 0.7, 0.5, 0.5])
    ax.plot(surface_x, surface_y, color=COLORS['ground_dark'], linewidth=2.5)
    ax.text(-7, 0.7, 'Ground Surface', fontsize=8, color=COLORS['ground_dark'])

    # Fill annotations
    ax.text(0, -0.8, 'Crater Fill\n(disturbed soil)', fontsize=8, ha='center',
            color=COLORS['gray_dark'], fontweight='bold')
    ax.text(-5.5, -1.5, 'Native\nSoil', fontsize=8, ha='center',
            color='#8b7355', style='italic')

    # Probes around perimeter (3m length, 1.5-2m spacing)
    probe_x = [-6, -4, -1.5, 1.5, 4, 6]
    probe_length = 3.5

    for i, px in enumerate(probe_x):
        # Calculate surface height at probe location
        if abs(px) <= 3:
            surface_h = 0.5
        elif abs(px) < 5:
            surface_h = 0.6
        else:
            surface_h = 0.5

        # Probe rod
        ax.add_patch(Rectangle((px-0.1, surface_h - probe_length), 0.2, probe_length,
                               color=COLORS['probe_body'], ec='black', lw=1, zorder=10))

        # Junction box
        ax.add_patch(Rectangle((px-0.2, surface_h), 0.4, 0.25,
                               color=COLORS['connector'], ec='black', lw=1, zorder=11))

        # ERT rings (at 0.5m, 1.5m, 2.5m from tip)
        for depth in [0.5, 1.5, 2.5]:
            y = surface_h - probe_length + depth
            ax.add_patch(Rectangle((px-0.15, y-0.04), 0.3, 0.08,
                                   color=COLORS['ert_ring'], ec='black', lw=0.5, zorder=12))

        # TX/RX coils
        for depth, coil_type in [(1.0, 'tx'), (2.0, 'rx'), (3.0, 'rx')]:
            y = surface_h - probe_length + depth
            color = COLORS['tx_coil'] if coil_type == 'tx' else COLORS['rx_coil']
            ax.add_patch(Circle((px, y), 0.12, color=color, ec='black', lw=0.5, zorder=12))

        # Probe label
        ax.text(px, surface_h + 0.5, f'P{i+1}', ha='center', fontsize=8, fontweight='bold')

    # Ray paths (showing measurement coverage)
    ray_depths = [1.5, 2.5, 3.0]
    for depth in ray_depths:
        for i, p1 in enumerate(probe_x[:-1]):
            for p2 in probe_x[i+1:]:
                if abs(p1 - p2) <= 5:  # Only nearby pairs
                    y1 = 0.5 - probe_length + depth
                    y2 = 0.5 - probe_length + depth
                    ax.plot([p1, p2], [y1, y2], color=COLORS['success'], alpha=0.25, lw=0.8)

    # Target UXB at depth
    target = Circle((0, -3.5), 0.45, color=COLORS['warning'], ec='#9b2c2c', lw=2, zorder=20)
    ax.add_patch(target)
    ax.text(0, -3.5, 'UXB', fontsize=7, ha='center', va='center',
            color='white', fontweight='bold', zorder=21)

    # Target annotation
    ax.annotate('Potential UXB\n(strong MIT response)', (0.5, -3.5), (3, -4.5),
                fontsize=8, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=1.5))

    # Depth scale
    for d in [1, 2, 3, 4, 5]:
        ax.axhline(0.5 - d, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.text(7.5, 0.5 - d, f'{d}m', fontsize=7, va='center', color='gray')

    # Dimension annotations
    ax.annotate('', xy=(-4, 1.5), xytext=(4, 1.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(0, 1.7, '~8m crater diameter', ha='center', fontsize=8)

    ax.annotate('', xy=(-6.8, -3.5), xytext=(-6.8, 0.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(-7.3, -1.5, '3.5m', fontsize=7, va='center', rotation=90)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['tx_coil'], label='TX Coil'),
        mpatches.Patch(color=COLORS['rx_coil'], label='RX Coil'),
        mpatches.Patch(color=COLORS['ert_ring'], label='ERT Ring'),
        mpatches.Patch(color='#a0aec0', alpha=0.5, label='Crater Fill'),
        mpatches.Patch(color=COLORS['warning'], label='Target (UXB)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9)

    ax.set_title('Bomb Crater Investigation: Perimeter Probe Deployment',
                fontsize=11, fontweight='bold', color=COLORS['primary'], pad=10)
    ax.set_ylabel('Depth (m)', fontsize=9)
    ax.set_yticks([0.5, -1.5, -3.5, -5.5])
    ax.set_yticklabels(['0', '2', '4', '6'])
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


# ============================================================================
# FIGURE 2: WOODLAND BURIAL SITE DEPLOYMENT
# ============================================================================
def create_woodland_burial_diagram():
    """
    Woodland burial site showing shallow probe array,
    disturbed soil detection, and tree root considerations.

    Returns:
        BytesIO buffer containing PNG image
    """
    fig, ax = plt.subplots(figsize=(9.23, 5))

    ax.set_xlim(-6, 6)
    ax.set_ylim(-3.5, 3)
    ax.set_aspect('equal')

    # Sky
    ax.fill_between([-6, 6], [0, 0], [3, 3], color=COLORS['sky'], alpha=0.5)

    # Ground
    ax.fill_between([-6, 6], [0, 0], [-3.5, -3.5], color=COLORS['ground_tan'], alpha=0.3)
    ax.axhline(0, color=COLORS['ground_dark'], linewidth=2.5)

    # Trees
    for tx in [-5, 5]:
        # Trunk
        ax.add_patch(Rectangle((tx-0.2, 0), 0.4, 1.8, color='#553c9a'))
        # Crown
        ax.add_patch(Circle((tx, 2.3), 0.7, color=COLORS['success'], alpha=0.8))
        # Root system (simplified)
        for angle in [-45, 0, 45]:
            rad = np.radians(angle - 90)
            root_len = 1.5
            end_x = tx + root_len * np.cos(rad)
            end_y = -root_len * np.sin(rad)
            ax.plot([tx, end_x], [-0.1, end_y], color='#553c9a', alpha=0.4, lw=2)

    # Burial shaft (disturbed soil zone)
    shaft_left = -1.0
    shaft_right = 1.0
    shaft_bottom = -1.8

    # Disturbed fill pattern
    ax.fill_between([shaft_left, shaft_right], [shaft_bottom, shaft_bottom], [-0.2, -0.2],
                   color=COLORS['purple'], alpha=0.3)
    ax.plot([shaft_left, shaft_left, shaft_right, shaft_right],
            [-0.2, shaft_bottom, shaft_bottom, -0.2],
            color='#553c9a', linewidth=2, linestyle='--')

    # Surface depression (subtle)
    surface_x = np.linspace(-1.5, 1.5, 30)
    surface_y = -0.08 * np.exp(-(surface_x**2)/0.8)
    ax.plot(surface_x, surface_y, color=COLORS['ground_dark'], linewidth=2.5)

    ax.text(0, -1.0, 'Disturbed\nSoil Zone', fontsize=8, ha='center', va='center',
            color='#553c9a', fontweight='bold')

    # Potential remains indication (subtle)
    ax.add_patch(Ellipse((0, -1.5), 1.0, 0.4, color='#e2e8f0', alpha=0.5))

    # Probes (1.6m length, 1m spacing)
    probe_x = [-3, -2, -1, 0, 1, 2, 3]
    probe_length = 1.8

    for i, px in enumerate(probe_x):
        # Probe rod
        ax.add_patch(Rectangle((px-0.08, -probe_length), 0.16, probe_length,
                               color=COLORS['probe_body'], ec='black', lw=1, zorder=10))

        # Junction box
        ax.add_patch(Rectangle((px-0.15, 0), 0.3, 0.2,
                               color=COLORS['connector'], ec='black', lw=1, zorder=11))

        # ERT rings (at 0.4m & 1.2m from tip)
        for depth in [0.4, 1.2]:
            y = -probe_length + depth
            ax.add_patch(Rectangle((px-0.12, y-0.03), 0.24, 0.06,
                                   color=COLORS['ert_ring'], ec='black', lw=0.5, zorder=12))

        # TX/RX coils
        for depth, coil_type in [(0.6, 'tx'), (1.0, 'rx'), (1.5, 'rx')]:
            y = -probe_length + depth
            color = COLORS['tx_coil'] if coil_type == 'tx' else COLORS['rx_coil']
            ax.add_patch(Circle((px, y), 0.08, color=color, ec='black', lw=0.5, zorder=12))

        # Probe label
        ax.text(px, 0.4, f'P{i+1}', ha='center', fontsize=7, fontweight='bold')

    # Dense ray paths through burial zone
    for d1 in [0.6, 1.0, 1.5]:
        for i in range(len(probe_x)-1):
            for j in range(i+1, min(i+3, len(probe_x))):  # Short baselines
                y = -probe_length + d1
                ax.plot([probe_x[i], probe_x[j]], [y, y],
                       color=COLORS['success'], alpha=0.3, lw=0.8)

    # Metallic artifact indication
    artifact = Circle((-0.3, -1.3), 0.1, color='#718096', ec='black', lw=1, zorder=15)
    ax.add_patch(artifact)
    ax.annotate('Metal\nartifact', (-0.3, -1.3), (-2.5, -2.5),
                fontsize=7, ha='center',
                arrowprops=dict(arrowstyle='->', color='#718096', lw=1))

    # ERT sensitivity zone
    sensitivity = Ellipse((0, -1.0), 4.5, 2.0, color=COLORS['success'], alpha=0.1)
    ax.add_patch(sensitivity)

    # Depth markers
    for d in [0.5, 1.0, 1.5, 2.0]:
        ax.axhline(-d, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.text(5.7, -d, f'{d}m', fontsize=7, va='center', color='gray')

    # Annotations
    ax.text(-5, -0.5, 'Root\nZone', fontsize=7, ha='center', color='#553c9a', alpha=0.7)
    ax.text(5, -0.5, 'Root\nZone', fontsize=7, ha='center', color='#553c9a', alpha=0.7)

    # Configuration note
    ax.text(0, 2.5, '1.6m probes | 1m spacing | 5-20 kHz MIT | ERT focus',
            ha='center', fontsize=8, color=COLORS['gray_dark'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title('Woodland Burial Site: Dense Array for Shallow Targets',
                fontsize=11, fontweight='bold', color=COLORS['primary'], pad=10)
    ax.set_ylabel('Depth (m)', fontsize=9)
    ax.set_yticks([0, -1, -2, -3])
    ax.set_yticklabels(['0', '1', '2', '3'])
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


# ============================================================================
# FIGURE 3: SWAMP MARGIN SURVEY CONFIGURATION
# ============================================================================
def create_swamp_margin_diagram():
    """
    Swamp/wetland margin survey showing shore-based deployment
    with long baselines toward deep target.

    Returns:
        BytesIO buffer containing PNG image
    """
    fig, ax = plt.subplots(figsize=(8.75, 5))

    ax.set_xlim(-7, 7)
    ax.set_ylim(-6, 2)
    ax.set_aspect('equal')

    # Sky
    ax.fill_between([-7, 7], [0.5, 0.5], [2, 2], color=COLORS['sky'], alpha=0.5)

    # Bank slopes
    left_bank_x = np.array([-7, -5, -4, -3.5])
    left_bank_y = np.array([0.5, 0.5, 0.3, -0.3])
    ax.fill_between(left_bank_x, left_bank_y, -6, color=COLORS['ground_tan'], alpha=0.5)

    right_bank_x = np.array([3.5, 4, 5, 7])
    right_bank_y = np.array([-0.3, 0.3, 0.5, 0.5])
    ax.fill_between(right_bank_x, right_bank_y, -6, color=COLORS['ground_tan'], alpha=0.5)

    # Native soil below water
    ax.fill_between([-3.5, 3.5], [-2, -2], [-6, -6], color=COLORS['ground_tan'], alpha=0.3)

    # Water body
    ax.fill_between([-3.5, 3.5], [0, 0], [-2, -2], color='#90cdf4', alpha=0.5)
    ax.text(0, -1, 'Water Body', fontsize=9, ha='center', color='#2b6cb0', fontweight='bold')

    # Organic layer beneath water
    ax.fill_between([-3.5, 3.5], [-2, -2], [-3, -3], color=COLORS['purple'], alpha=0.3)
    ax.text(0, -2.5, 'Organic sediment', fontsize=7, ha='center', color='#553c9a', style='italic')

    # Bank surface lines
    ax.plot(left_bank_x, left_bank_y, color=COLORS['ground_dark'], linewidth=2.5)
    ax.plot(right_bank_x, right_bank_y, color=COLORS['ground_dark'], linewidth=2.5)

    # Water surface
    water_x = np.linspace(-3.5, 3.5, 50)
    water_y = 0.05 * np.sin(water_x * 2)  # Subtle ripples
    ax.plot(water_x, water_y, color='#3182ce', linewidth=1.5)

    # Left margin probes
    left_probes = [-6, -5, -4.5]
    for i, px in enumerate(left_probes):
        # Calculate bank height
        surface_h = 0.5 if px <= -5 else 0.4
        probe_len = 3.0

        # Probe rod
        ax.add_patch(Rectangle((px-0.1, surface_h - probe_len), 0.2, probe_len,
                               color=COLORS['probe_body'], ec='black', lw=1, zorder=10))

        # Junction box
        ax.add_patch(Rectangle((px-0.18, surface_h), 0.36, 0.22,
                               color=COLORS['connector'], ec='black', lw=1, zorder=11))

        # Sensors
        for depth in [1.0, 2.0, 2.8]:
            y = surface_h - depth
            ax.add_patch(Circle((px, y), 0.1, color=COLORS['tx_coil'], ec='black', lw=0.5, zorder=12))

        ax.text(px, surface_h + 0.4, f'L{i+1}', ha='center', fontsize=7, fontweight='bold')

    # Right margin probes
    right_probes = [4.5, 5, 6]
    for i, px in enumerate(right_probes):
        surface_h = 0.5 if px >= 5 else 0.4
        probe_len = 3.0

        # Probe rod
        ax.add_patch(Rectangle((px-0.1, surface_h - probe_len), 0.2, probe_len,
                               color=COLORS['probe_body'], ec='black', lw=1, zorder=10))

        # Junction box
        ax.add_patch(Rectangle((px-0.18, surface_h), 0.36, 0.22,
                               color=COLORS['connector'], ec='black', lw=1, zorder=11))

        # Sensors
        for depth in [1.0, 2.0, 2.8]:
            y = surface_h - depth
            ax.add_patch(Circle((px, y), 0.1, color=COLORS['rx_coil'], ec='black', lw=0.5, zorder=12))

        ax.text(px, surface_h + 0.4, f'R{i+1}', ha='center', fontsize=7, fontweight='bold')

    # Long-baseline ray paths (diagonal across)
    for lp in left_probes:
        for rp in right_probes:
            for d1, d2 in [(1.0, 2.0), (2.0, 1.0), (2.0, 2.0), (2.8, 2.8)]:
                y1 = 0.5 - d1
                y2 = 0.5 - d2
                ax.plot([lp, rp], [y1, y2], color=COLORS['success'], alpha=0.2, lw=0.8)

    # Deep target
    target = Circle((0, -5), 0.5, color=COLORS['warning'], ec='#9b2c2c', lw=2, zorder=20)
    ax.add_patch(target)
    ax.annotate('Deep target\n(>5m)', (0, -5), (2.5, -5.5),
                fontsize=8, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=1.5))

    # Coverage zone indication
    coverage = Polygon([(-4.5, -2), (4.5, -2), (2, -5.5), (-2, -5.5)],
                       closed=True, fill=True, facecolor=COLORS['success'],
                       alpha=0.1, edgecolor=COLORS['success'], linestyle='--', lw=1)
    ax.add_patch(coverage)
    ax.text(0, -3.8, 'Interrogation\nVolume', fontsize=7, ha='center',
            color='#276749', style='italic')

    # Depth markers
    for d in [1, 2, 3, 4, 5]:
        ax.axhline(0.5 - d, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax.text(6.7, 0.5 - d, f'{d}m', fontsize=7, va='center', color='gray')

    # Annotations
    ax.text(-5.5, 1.2, 'Shore Access', fontsize=8, ha='center', color=COLORS['gray_dark'])
    ax.text(5.5, 1.2, 'Shore Access', fontsize=8, ha='center', color=COLORS['gray_dark'])

    # Configuration note
    ax.text(0, 1.5, '2-3m probes | 2-5 kHz (deep) | Extended baselines',
            ha='center', fontsize=8, color=COLORS['gray_dark'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title('Swamp/Margin Survey: Perimeter Deployment for Deep Targets',
                fontsize=11, fontweight='bold', color=COLORS['primary'], pad=10)
    ax.set_ylabel('Depth (m)', fontsize=9)
    ax.set_yticks([0.5, -1.5, -3.5, -5.5])
    ax.set_yticklabels(['0', '2', '4', '6'])
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


# ============================================================================
# FIGURE 4: CONFIGURATION COMPARISON MATRIX
# ============================================================================
def create_configuration_matrix():
    """
    Visual comparison matrix of configurations across scenarios.

    Returns:
        BytesIO buffer containing PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Configuration Comparison by Scenario', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Column headers
    headers = ['Parameter', 'Woods Burial', 'Bomb Crater', 'Swamp/Margin']
    x_positions = [0.5, 2.8, 5.1, 7.4]

    for x, h in zip(x_positions, headers):
        ax.add_patch(FancyBboxPatch((x, 5.0), 2.0, 0.5, boxstyle="round,pad=0.03",
                                    facecolor=COLORS['secondary'], edgecolor='black', lw=1))
        ax.text(x + 1.0, 5.25, h, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')

    # Data rows
    row_data = [
        ('Rod Length', '1.6 m', '3.0 m', '1.5-2 m'),
        ('Spacing', '1-1.5 m', '1.5-2 m', '2-3 m'),
        ('MIT Freq', '5-20 kHz', '2-20 kHz', '2-5 kHz'),
        ('Depth Target', '0-2 m', '0-4+ m', '>5 m'),
        ('Key Method', 'ERT + MIT', 'MIT + ERT', 'Low-freq MIT'),
        ('Probes', '12-16', '20-36', 'Variable'),
    ]

    colors_row = [COLORS['success'], COLORS['warning'], COLORS['accent']]

    y = 4.4
    for row in row_data:
        # Parameter name
        ax.add_patch(FancyBboxPatch((0.5, y), 2.0, 0.5, boxstyle="round,pad=0.02",
                                    facecolor='#e2e8f0', edgecolor='#a0aec0', lw=0.5))
        ax.text(1.5, y + 0.25, row[0], ha='center', va='center', fontsize=8,
                fontweight='bold', color=COLORS['gray_dark'])

        # Values for each scenario
        for i, (x, val) in enumerate(zip(x_positions[1:], row[1:])):
            ax.add_patch(FancyBboxPatch((x, y), 2.0, 0.5, boxstyle="round,pad=0.02",
                                        facecolor='white', edgecolor=colors_row[i], lw=1))
            ax.text(x + 1.0, y + 0.25, val, ha='center', va='center', fontsize=8)

        y -= 0.6

    # Visual indicators (icons)
    icon_y = 0.6

    # Woods - tree icon
    ax.add_patch(Rectangle((2.7, icon_y), 0.15, 0.4, color=COLORS['success']))
    ax.add_patch(Circle((2.775, icon_y + 0.55), 0.2, color=COLORS['success'], alpha=0.7))
    ax.text(3.5, icon_y + 0.2, 'Shallow\nDense', ha='center', fontsize=7, color=COLORS['gray_dark'])

    # Crater - bowl icon
    crater_x = np.linspace(4.9, 5.7, 20)
    crater_y = icon_y + 0.3 - 0.4 * (1 - ((crater_x - 5.3)/0.4)**2)
    ax.fill_between(crater_x, crater_y, icon_y - 0.1, color=COLORS['warning'], alpha=0.5)
    ax.text(5.8, icon_y + 0.2, 'Deep\nPerimeter', ha='center', fontsize=7, color=COLORS['gray_dark'])

    # Swamp - water icon
    wave_x = np.linspace(7.2, 8.0, 30)
    wave_y = icon_y + 0.3 + 0.05 * np.sin(wave_x * 8)
    ax.plot(wave_x, wave_y, color=COLORS['accent'], lw=2)
    ax.fill_between([7.2, 8.0], [icon_y, icon_y], [icon_y + 0.25, icon_y + 0.25],
                   color=COLORS['accent'], alpha=0.3)
    ax.text(8.3, icon_y + 0.2, 'Margin\nAccess', ha='center', fontsize=7, color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 5: TARGET CHARACTERIZATION EXAMPLES
# ============================================================================
def create_target_characterization():
    """
    Visual examples of MIT and ERT signatures for different target types.

    Returns:
        BytesIO buffer containing PNG image
    """
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))

    targets = [
        ('Large Metal\n(UXB)', 'MIT Strong', '#c53030', 'circle'),
        ('Small Metal\n(Artifacts)', 'MIT Moderate', '#ed8936', 'scatter'),
        ('Fill/Disturbed\nSoil', 'ERT Contrast', '#805ad5', 'gradient'),
        ('Water/Wet Zone', 'ERT Low-R', '#3182ce', 'low_r'),
        ('Void/Air', 'ERT High-R', '#718096', 'high_r'),
        ('Combined\nAnomaly', 'MIT + ERT', '#38a169', 'combined'),
    ]

    for ax, (name, response, color, pattern) in zip(axes.flatten(), targets):
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')

        # Background (soil)
        ax.fill_between([-3, 3], [-3, -3], [3, 3], color=COLORS['ground_tan'], alpha=0.2)

        # Response pattern
        if pattern == 'circle':
            # Strong concentric response
            for r, alpha in [(2.0, 0.15), (1.5, 0.25), (1.0, 0.4), (0.5, 0.6)]:
                ax.add_patch(Circle((0, 0), r, facecolor=color, alpha=alpha, edgecolor='none'))
            ax.add_patch(Circle((0, 0), 0.3, facecolor=color, edgecolor='black', lw=1))

        elif pattern == 'scatter':
            # Multiple small responses
            np.random.seed(42)
            for _ in range(5):
                cx, cy = np.random.uniform(-1.5, 1.5, 2)
                ax.add_patch(Circle((cx, cy), 0.3, facecolor=color, alpha=0.5))
                ax.add_patch(Circle((cx, cy), 0.6, facecolor=color, alpha=0.2))

        elif pattern == 'gradient':
            # Vertical gradient (disturbed zone)
            for i in range(10):
                y = -1.5 + i * 0.3
                alpha = 0.1 + 0.05 * (5 - abs(i - 5))
                ax.add_patch(Rectangle((-1.5, y), 3.0, 0.3, facecolor=color, alpha=alpha))
            ax.plot([-1.5, -1.5, 1.5, 1.5], [1.5, -1.5, -1.5, 1.5], '--', color=color, lw=1.5)

        elif pattern == 'low_r':
            # Low resistivity (conductive)
            theta = np.linspace(0, 2*np.pi, 100)
            for r, alpha in [(1.8, 0.2), (1.2, 0.4), (0.6, 0.6)]:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ax.fill(x, y, color=color, alpha=alpha)
            ax.text(0, -2.5, 'Low R', fontsize=7, ha='center', color=color)

        elif pattern == 'high_r':
            # High resistivity (void)
            ax.add_patch(Ellipse((0, 0), 2.0, 1.5, facecolor='white', edgecolor=color, lw=2))
            ax.add_patch(Ellipse((0, 0), 2.0, 1.5, facecolor=color, alpha=0.1))
            ax.text(0, -2.5, 'High R', fontsize=7, ha='center', color=color)

        elif pattern == 'combined':
            # Combined MIT + ERT response
            ax.add_patch(Rectangle((-1.5, -1.5), 3.0, 3.0, facecolor=COLORS['purple'], alpha=0.2,
                                   edgecolor=COLORS['purple'], linestyle='--', lw=1))
            for r, alpha in [(1.0, 0.3), (0.5, 0.5)]:
                ax.add_patch(Circle((0, 0), r, facecolor=color, alpha=alpha))
            ax.add_patch(Circle((0, 0), 0.25, facecolor=color, edgecolor='black', lw=1))

        ax.set_title(name, fontsize=9, fontweight='bold', color=COLORS['primary'], pad=5)
        ax.text(0, 2.5, response, fontsize=7, ha='center', color=color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.suptitle('Target Characterization: Expected Response Patterns',
                fontsize=11, fontweight='bold', color=COLORS['primary'], y=1.02)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 6: SITE ASSESSMENT DECISION DIAGRAM
# ============================================================================
def create_site_assessment_diagram():
    """
    Flowchart for site assessment and configuration selection.

    Returns:
        BytesIO buffer containing PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.7, 'Site Assessment Decision Flow', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Start node
    ax.add_patch(FancyBboxPatch((3.5, 6.8), 3, 0.6, boxstyle="round,pad=0.05",
                                facecolor=COLORS['success'], edgecolor='black', lw=1.5))
    ax.text(5, 7.1, 'Site Assessment', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    # First decision: Target Depth
    ax.add_patch(Polygon([(5, 6.5), (6.5, 5.8), (5, 5.1), (3.5, 5.8)],
                        facecolor='#ebf8ff', edgecolor=COLORS['secondary'], lw=1.5))
    ax.text(5, 5.8, 'Target\nDepth?', ha='center', va='center', fontsize=9)

    # Depth branches
    ax.annotate('', xy=(2.5, 5.0), xytext=(4.0, 5.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.5))
    ax.text(2.8, 5.4, '<2m', fontsize=8, color=COLORS['gray_dark'])

    ax.annotate('', xy=(5, 4.5), xytext=(5, 5.1),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.5))
    ax.text(5.2, 4.8, '2-4m', fontsize=8, color=COLORS['gray_dark'])

    ax.annotate('', xy=(7.5, 5.0), xytext=(6.0, 5.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.5))
    ax.text(6.8, 5.4, '>4m', fontsize=8, color=COLORS['gray_dark'])

    # Shallow config
    ax.add_patch(FancyBboxPatch((1.0, 4.2), 3, 0.8, boxstyle="round,pad=0.03",
                                facecolor=COLORS['success'], alpha=0.3,
                                edgecolor=COLORS['success'], lw=1.5))
    ax.text(2.5, 4.6, 'Shallow Config', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#276749')
    ax.text(2.5, 4.3, '1.5m probes, 1m spacing', ha='center', va='center', fontsize=7)

    # Medium config
    ax.add_patch(FancyBboxPatch((3.5, 3.7), 3, 0.8, boxstyle="round,pad=0.03",
                                facecolor=COLORS['warning'], alpha=0.3,
                                edgecolor=COLORS['warning'], lw=1.5))
    ax.text(5, 4.1, 'Standard Config', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#9b2c2c')
    ax.text(5, 3.8, '3m probes, 1.5-2m spacing', ha='center', va='center', fontsize=7)

    # Deep config
    ax.add_patch(FancyBboxPatch((6.0, 4.2), 3, 0.8, boxstyle="round,pad=0.03",
                                facecolor=COLORS['accent'], alpha=0.3,
                                edgecolor=COLORS['accent'], lw=1.5))
    ax.text(7.5, 4.6, 'Deep Config', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#2c5282')
    ax.text(7.5, 4.3, 'Extended baselines, 2-5 kHz', ha='center', va='center', fontsize=7)

    # Second decision: Access Conditions
    ax.add_patch(Polygon([(5, 3.3), (6.2, 2.6), (5, 1.9), (3.8, 2.6)],
                        facecolor='#ebf8ff', edgecolor=COLORS['secondary'], lw=1.5))
    ax.text(5, 2.6, 'Access?', ha='center', va='center', fontsize=9)

    # Access branches
    ax.annotate('', xy=(2.5, 2.0), xytext=(4.2, 2.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.5))
    ax.text(3.0, 2.4, 'Full', fontsize=8, color=COLORS['gray_dark'])

    ax.annotate('', xy=(7.5, 2.0), xytext=(5.8, 2.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.5))
    ax.text(6.7, 2.4, 'Limited', fontsize=8, color=COLORS['gray_dark'])

    # Full access
    ax.add_patch(FancyBboxPatch((1.0, 1.2), 3, 0.8, boxstyle="round,pad=0.03",
                                facecolor='#f0fff4', edgecolor='#276749', lw=1))
    ax.text(2.5, 1.6, 'Full Grid Array', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(2.5, 1.3, '5x5 or 6x6 pattern', ha='center', va='center', fontsize=7)

    # Limited access
    ax.add_patch(FancyBboxPatch((6.0, 1.2), 3, 0.8, boxstyle="round,pad=0.03",
                                facecolor='#fff5f5', edgecolor='#9b2c2c', lw=1))
    ax.text(7.5, 1.6, 'Perimeter Only', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(7.5, 1.3, 'Margin deployment', ha='center', va='center', fontsize=7)

    # Key considerations box
    ax.add_patch(FancyBboxPatch((0.3, 0.1), 4.4, 0.9, boxstyle="round,pad=0.03",
                                facecolor='#ebf8ff', edgecolor=COLORS['accent'], lw=1))
    ax.text(2.5, 0.85, 'Key Considerations:', ha='center', fontsize=8, fontweight='bold',
            color=COLORS['secondary'])
    ax.text(2.5, 0.55, 'UXO clearance | Soil type | Water table', ha='center', fontsize=7)
    ax.text(2.5, 0.3, 'Target type | Required resolution', ha='center', fontsize=7)

    # Frequency selection box
    ax.add_patch(FancyBboxPatch((5.3, 0.1), 4.4, 0.9, boxstyle="round,pad=0.03",
                                facecolor='#feebc8', edgecolor=COLORS['orange'], lw=1))
    ax.text(7.5, 0.85, 'MIT Frequency Selection:', ha='center', fontsize=8, fontweight='bold',
            color='#c05621')
    ax.text(7.5, 0.55, 'Shallow: 10-50 kHz | Deep: 2-10 kHz', ha='center', fontsize=7)
    ax.text(7.5, 0.3, 'Conductive soil: lower frequencies', ha='center', fontsize=7)

    # Connector arrows from configs to access decision
    for x in [2.5, 5, 7.5]:
        if x == 2.5:
            ax.plot([x, 4.5], [4.2, 3.3], color=COLORS['gray_light'], lw=1, linestyle=':')
        elif x == 5:
            ax.plot([x, x], [3.7, 3.3], color=COLORS['gray_light'], lw=1, linestyle=':')
        else:
            ax.plot([x, 5.5], [4.2, 3.3], color=COLORS['gray_light'], lw=1, linestyle=':')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# DOCUMENT BUILDER
# ============================================================================
def build_section_17():
    """Build the complete Section 17 PDF document."""

    print("Section 17: Application Scenarios")
    print("=" * 50)

    # Initialize builder
    builder = SectionPDFBuilder(
        section_num=17,
        title="Application Scenarios",
        output_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'output', 'sections')
    )

    # Generate figures
    print("Generating diagrams...")
    print("  - Bomb crater cross-section")
    fig_crater = create_bomb_crater_diagram()

    print("  - Woodland burial deployment")
    fig_burial = create_woodland_burial_diagram()

    print("  - Swamp margin survey")
    fig_swamp = create_swamp_margin_diagram()

    print("  - Configuration comparison matrix")
    fig_matrix = create_configuration_matrix()

    print("  - Target characterization examples")
    fig_targets = create_target_characterization()

    print("  - Site assessment diagram")
    fig_assessment = create_site_assessment_diagram()

    # Build document
    print("\nBuilding PDF document...")

    # Title block
    builder.add_title_block(
        subtitle="Deployment Configurations for Bomb Craters, Burials, and Wetland Surveys"
    )

    # === INTRODUCTION ===
    builder.add_section_header("17. Application Scenarios")

    builder.add_body_text(
        "The HIRT system's modular design enables optimization for diverse field scenarios. "
        "This section provides detailed configuration recommendations for three primary "
        "application categories: bomb crater investigation, woodland burial search, and "
        "wetland/margin surveys. Each scenario requires specific probe lengths, spacing, "
        "frequency selection, and deployment strategies to maximize detection probability "
        "while maintaining operational safety.",
        first_paragraph=True
    )

    builder.add_body_text(
        "Configuration selection depends on several key factors: target depth, soil "
        "conditions, site access constraints, and the nature of expected anomalies. "
        "The dual-channel MIT+ERT approach provides complementary information: MIT excels "
        "at detecting metallic objects (both ferrous and non-ferrous), while ERT maps "
        "soil disturbance patterns, moisture variations, and void structures."
    )

    # === SCENARIO 1: BOMB CRATER ===
    builder.add_section_header("17.1 Scenario 1: Bomb Crater Investigation", level=2)

    builder.add_body_text(
        "Bomb crater investigation represents the most demanding HIRT application, "
        "requiring deep penetration (3-4+ meters), comprehensive coverage of disturbed "
        "fill material, and detection of potentially large metallic masses. Typical "
        "crater dimensions range from 10-15 meters diameter with depths of 2-4 meters.",
        first_paragraph=True
    )

    # Bomb crater figure
    builder.add_figure(
        fig_crater,
        "Bomb crater investigation configuration showing perimeter probe deployment "
        "around crater margin. Six 3.5m probes at 2m spacing provide comprehensive "
        "coverage through crater fill. MIT ray paths (green) interrogate the full "
        "volume including potential UXB location at crater base. ERT rings map fill "
        "boundaries and wet zones."
    )

    builder.add_section_header("Configuration Parameters", level=3)

    builder.add_bullet_list([
        "<b>Rod Length:</b> 3.0 m (minimum; 3.5m preferred for full-depth coverage)",
        "<b>ERT Ring Positions:</b> 0.5 m, 1.5 m, 2.5 m from probe tip",
        "<b>Probe Spacing:</b> 1.5-2 m between probes",
        "<b>Section Size:</b> Covers full crater plus rim (may require multiple sections)",
        "<b>Probe Count:</b> 20-36 probes depending on crater dimensions",
    ])

    builder.add_section_header("Measurement Strategy", level=3)

    builder.add_bullet_list([
        "<b>MIT Emphasis:</b> 2-20 kHz frequency range for aluminum/steel mass detection",
        "<b>ERT Focus:</b> Map fill bowl geometry and identify wet pockets",
        "<b>Depth Target:</b> 0-4+ meters (full crater depth plus underlying zone)",
        "<b>Frequency Selection:</b> Lower frequencies (2-5 kHz) for maximum penetration",
        "<b>Integration Time:</b> 10-30 seconds per measurement for improved SNR at depth",
    ])

    builder.add_section_header("Deployment Considerations", level=3)

    builder.add_body_text(
        "Crater investigation presents unique challenges including uneven surface "
        "topography, loose fill material, and potential UXO hazards. Initial deployment "
        "should focus on the crater rim where stable soil provides secure probe anchoring.",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "<b>Rim-First Approach:</b> Deploy perimeter probes before attempting center insertion",
        "<b>Multiple Overlapping Sections:</b> Large craters may require 2-3 overlapping arrays",
        "<b>Deep Insertion Protocol:</b> Use pilot rods to verify probe path before full insertion",
        "<b>Extended Baselines:</b> ERT corner-to-corner measurements for deep investigation",
    ])

    builder.add_section_header("Expected Results", level=3)

    builder.add_body_text(
        "MIT measurements will show strong amplitude and phase anomalies from large "
        "metallic objects (engine blocks, landing gear, ordnance casings). ERT will "
        "map the crater boundary as a resistivity contrast between disturbed fill and "
        "native soil. Water accumulation at the crater base appears as low resistivity zones.",
        first_paragraph=True
    )

    # === SCENARIO 2: WOODLAND BURIAL ===
    builder.add_section_header("17.2 Scenario 2: Woodland Burials", level=2)

    builder.add_body_text(
        "Woodland burial search requires high-resolution detection of relatively shallow "
        "targets (0.5-2 m depth) in environments complicated by root systems and organic "
        "material. The primary detection mechanism is ERT mapping of disturbed soil "
        "zones (grave shafts), supplemented by MIT detection of metallic artifacts.",
        first_paragraph=True
    )

    # Woodland burial figure
    builder.add_figure(
        fig_burial,
        "Woodland burial site configuration showing dense probe array (7 probes, 1m spacing) "
        "for shallow target detection. Disturbed soil zone (grave shaft) shows distinct ERT "
        "contrast. Tree root systems require careful probe positioning. MIT detects small "
        "metallic artifacts (buckles, buttons) within burial context."
    )

    builder.add_section_header("Configuration Parameters", level=3)

    builder.add_bullet_list([
        "<b>Rod Length:</b> 1.6 m (sufficient for typical burial depths)",
        "<b>ERT Ring Positions:</b> 0.4 m and 1.2 m from probe tip",
        "<b>Probe Spacing:</b> 1-1.5 m (tight spacing for small targets)",
        "<b>Section Size:</b> 8x8 m standard search area",
        "<b>Probe Count:</b> 12-16 probes for comprehensive coverage",
    ])

    builder.add_section_header("Measurement Strategy", level=3)

    builder.add_bullet_list([
        "<b>ERT Emphasis:</b> Primary method for grave shaft detection",
        "<b>MIT Frequencies:</b> 5-20 kHz range (focus on small metallic artifacts)",
        "<b>Target Signatures:</b> Clusters of small anomalies suggest artifact concentrations",
        "<b>Depth Target:</b> 0-2 m (standard burial depth range)",
        "<b>Multiple Frequencies:</b> Sweep 5, 10, 20 kHz for sensitivity optimization",
    ])

    builder.add_section_header("Deployment Considerations", level=3)

    builder.add_body_text(
        "Woodland environments require sensitivity to site preservation. Use shallow "
        "insertion where possible and avoid major root systems. Document all surface "
        "features that may correlate with subsurface anomalies.",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "<b>Minimal Intrusion:</b> Shallow insertion protocol, narrow pilot holes",
        "<b>Root Avoidance:</b> Survey root positions before probe placement",
        "<b>Dense Measurement Pattern:</b> Short baselines for maximum near-surface resolution",
        "<b>Complementary Methods:</b> Coordinate with GPR surface survey if available",
    ])

    # === SCENARIO 3: SWAMP/MARGINS ===
    builder.add_section_header("17.3 Scenario 3: Swamp/Wetland Margins", level=2)

    builder.add_body_text(
        "Wetland and swamp margin surveys address the challenge of deep targets (>5 m) "
        "in areas with limited access. Probes must be deployed from accessible shore "
        "zones, using extended baselines to interrogate central volumes that cannot "
        "be directly instrumented.",
        first_paragraph=True
    )

    # Swamp margin figure
    builder.add_figure(
        fig_swamp,
        "Swamp/margin survey configuration showing shore-based probe deployment. Probes "
        "on opposing banks create diagonal ray paths through target volume beneath water "
        "body. Low-frequency MIT (2-5 kHz) provides deep penetration. Extended baselines "
        "enable detection of targets beyond direct probe reach."
    )

    builder.add_section_header("Configuration Parameters", level=3)

    builder.add_bullet_list([
        "<b>Rod Length:</b> Maximum feasible at margins (typically 1.5-2 m)",
        "<b>Probe Spacing:</b> 2-3 m (wider for challenging access)",
        "<b>Section Size:</b> Adapts to accessible areas",
        "<b>Probe Count:</b> Variable based on margin accessibility",
        "<b>Baseline Length:</b> Maximize TX-RX separation for depth penetration",
    ])

    builder.add_section_header("Measurement Strategy", level=3)

    builder.add_bullet_list([
        "<b>Low MIT Frequencies:</b> 2-5 kHz for maximum depth penetration",
        "<b>Extended Offsets:</b> Long TX-RX baselines essential for deep targets",
        "<b>ERT Configuration:</b> Wide injection pairs across water if possible",
        "<b>Shore-Based Probes:</b> Deploy from accessible margins only",
        "<b>Complementary Methods:</b> Consider seismic add-on for void detection",
    ])

    builder.add_section_header("Deployment Considerations", level=3)

    builder.add_body_text(
        "Safety is paramount in wetland environments. Ensure stable footing, assess "
        "water depth, and use appropriate personal protective equipment. The high "
        "conductivity of water affects both MIT and ERT measurements, requiring "
        "adjusted interpretation parameters.",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "<b>Access Limitations:</b> Work strictly from stable shore positions",
        "<b>Water Effects:</b> Account for high conductivity in data interpretation",
        "<b>Extended Integration:</b> Longer measurement times for weak signals",
        "<b>Marginal Detection:</b> Targets near detection limits require repeated measurements",
    ])

    # === CONFIGURATION COMPARISON ===
    builder.add_section_header("17.4 Scenario Comparison Matrix", level=2)

    builder.add_body_text(
        "The following comparison summarizes key configuration parameters across all "
        "three primary scenarios. Selection depends on target depth, soil conditions, "
        "and site access constraints.",
        first_paragraph=True
    )

    # Configuration matrix figure
    builder.add_figure(
        fig_matrix,
        "Configuration comparison matrix showing recommended parameters for each scenario. "
        "Woods burial uses shallow/dense configuration, bomb crater uses deep/perimeter "
        "approach, and swamp/margin requires margin access with extended baselines."
    )

    # Configuration comparison table
    config_data = [
        ['Scenario', 'Rod Length', 'Spacing', 'Frequencies', 'Depth Target', 'Key Method'],
        ['Woods Burials', '1.6 m', '1-1.5 m', '5-20 kHz', '0-2 m', 'ERT + MIT'],
        ['Bomb Crater', '3.0 m', '1.5-2 m', '2-20 kHz', '0-4+ m', 'MIT + ERT'],
        ['Swamp/Margins', '1.5-2 m', '2-3 m', '2-5 kHz', '>5 m', 'Low-freq MIT'],
    ]

    builder.add_table(
        config_data,
        col_widths=[CONTENT_WIDTH*0.18, CONTENT_WIDTH*0.13, CONTENT_WIDTH*0.13,
                    CONTENT_WIDTH*0.15, CONTENT_WIDTH*0.16, CONTENT_WIDTH*0.16],
        caption="Summary of configuration parameters by application scenario."
    )

    # === TARGET CHARACTERIZATION ===
    builder.add_section_header("17.5 Target Characterization", level=2)

    builder.add_body_text(
        "Understanding expected response patterns helps operators identify and classify "
        "anomalies during field operations. The dual-channel MIT+ERT approach provides "
        "complementary signatures for different target types.",
        first_paragraph=True
    )

    # Target characterization figure
    builder.add_figure(
        fig_targets,
        "Target characterization examples showing expected MIT and ERT response patterns. "
        "Large metal objects produce strong MIT anomalies; small artifacts show weaker "
        "clustered responses. ERT detects fill zones, wet areas, and voids through "
        "resistivity contrast. Combined anomalies suggest buried metallic objects in "
        "disturbed soil contexts."
    )

    builder.add_section_header("Interpretation Guidelines", level=3)

    builder.add_bullet_list([
        "<b>Strong MIT + ERT Contrast:</b> Large metallic object in disturbed context (e.g., UXB in crater)",
        "<b>MIT Only:</b> Metal object in undisturbed native soil",
        "<b>ERT Only:</b> Soil disturbance without metallic content (possible burial, cache)",
        "<b>Weak MIT Cluster:</b> Multiple small metallic artifacts (artifact scatter)",
        "<b>Low ERT Zone:</b> Water accumulation, saturated fill",
        "<b>High ERT Zone:</b> Air void, dry compacted material",
    ])

    # === SITE ASSESSMENT ===
    builder.add_section_header("17.6 Site Assessment and Configuration Selection", level=2)

    builder.add_body_text(
        "Proper site assessment is essential for selecting optimal configuration "
        "parameters. The decision flow considers target depth, access conditions, "
        "soil type, and safety requirements.",
        first_paragraph=True
    )

    # Site assessment figure
    builder.add_figure(
        fig_assessment,
        "Site assessment decision flow for configuration selection. Primary factors "
        "include estimated target depth and site access conditions. Secondary "
        "considerations include soil conductivity, water table depth, and required "
        "resolution. Frequency selection depends on depth requirements and soil "
        "conditions."
    )

    builder.add_section_header("Pre-Deployment Checklist", level=3)

    builder.add_numbered_list([
        "Conduct site safety assessment (UXO clearance if applicable)",
        "Estimate target depth range from historical records or surface features",
        "Evaluate access conditions and identify probe deployment zones",
        "Assess soil type and expected conductivity",
        "Determine required spatial resolution for target detection",
        "Select configuration parameters from scenario guidelines",
        "Plan grid layout and probe numbering scheme",
        "Verify all equipment functionality before deployment",
    ])

    # === GENERAL FIELD PROCEDURES ===
    builder.add_section_header("17.7 General Field Procedures", level=2)

    builder.add_section_header("Pre-Deployment", level=3)

    builder.add_bullet_list([
        "<b>Site Assessment:</b> Evaluate access, safety, and estimated target depth",
        "<b>Configuration Selection:</b> Choose rod length, spacing, and frequencies per scenario",
        "<b>Grid Planning:</b> Lay out probe positions with consistent numbering",
        "<b>Equipment Check:</b> Verify all probes, base hub, tools, and cables",
    ])

    builder.add_section_header("During Deployment", level=3)

    builder.add_bullet_list([
        "<b>Systematic Approach:</b> Follow 'set once, measure many' workflow",
        "<b>Quality Control:</b> Check reciprocity, repeat critical measurements",
        "<b>Documentation:</b> Record all conditions, anomalies, and deviations",
        "<b>Adaptation:</b> Adjust strategy based on initial results if necessary",
    ])

    builder.add_section_header("Post-Deployment", level=3)

    builder.add_bullet_list([
        "<b>Data Backup:</b> Secure all data immediately upon completion",
        "<b>Quick Analysis:</b> Generate preliminary plots to verify data quality",
        "<b>Equipment Care:</b> Clean, inspect, and repair probes as needed",
        "<b>Documentation:</b> Complete field notes and measurement logs",
    ])

    # === COST AND TIMELINE ===
    builder.add_section_header("17.8 Cost and Timeline Planning", level=2)

    builder.add_body_text(
        "Planning a HIRT deployment requires realistic assessment of equipment costs "
        "and build timeline. The following estimates assume standard components and "
        "moderate fabrication experience.",
        first_paragraph=True
    )

    # Cost table
    cost_data = [
        ['Component', 'Essential (12 probes)', 'Standard (20 probes)', 'Complete (24 probes)'],
        ['Probes', '$840-1,800', '$1,400-3,000', '$1,680-3,600'],
        ['Base Hub', '$200-500', '$200-500', '$200-500'],
        ['Tools/Supplies', '$200-400', '$200-400', '$200-400'],
        ['Total', '$1,200-2,700', '$1,800-3,900', '$2,100-4,500'],
    ]

    builder.add_table(
        cost_data,
        col_widths=[CONTENT_WIDTH*0.22, CONTENT_WIDTH*0.26, CONTENT_WIDTH*0.26, CONTENT_WIDTH*0.26],
        caption="Estimated costs for HIRT system configurations. Actual costs vary by "
                "supplier, quantity discounts, and component choices."
    )

    builder.add_section_header("Build Timeline", level=3)

    builder.add_bullet_list([
        "<b>Week 1-2 (Prototype):</b> Build 2 prototype probes, document procedures",
        "<b>Week 3 (Calibration):</b> Bench calibration, test target trials, design refinement",
        "<b>Week 4-5 (Scale-up):</b> Build 12-20 probes, assemble base hub, field shakedown",
        "<b>Week 6+ (Deployment):</b> First real section scans, procedure refinement",
    ])

    # === RECOMMENDATIONS ===
    builder.add_section_header("17.9 Recommendations", level=2)

    builder.add_section_header("Start Simple", level=3)

    builder.add_body_text(
        "Begin with the core MIT+ERT system on controlled test sites before attempting "
        "operational deployments. Build experience with probe handling, measurement "
        "protocols, and data interpretation before addressing complex field scenarios.",
        first_paragraph=True
    )

    builder.add_section_header("Add Selectively", level=3)

    builder.add_body_text(
        "Optional enhancements (borehole radar, seismic crosshole, magnetometer sweep) "
        "should be added based on demonstrated need rather than theoretical benefit. "
        "Each addition increases system complexity and field deployment time.",
        first_paragraph=True
    )

    builder.add_section_header("Maintain Modularity", level=3)

    builder.add_body_text(
        "The modular probe design enables component replacement and configuration "
        "adaptation. Maintain this modularity by documenting all modifications and "
        "ensuring backward compatibility with existing components.",
        first_paragraph=True
    )

    # Safety warning box
    builder.add_warning_box(
        "CRITICAL: UXO SITE SAFETY",
        [
            "Professional EOD clearance required before ANY probe insertion at suspected UXO sites",
            "Never hammer or drive probes - use soft insertion only",
            "Maintain appropriate exclusion zones during all operations",
            "Monitor groundwater conductivity - elevated levels may indicate corrosion risk",
            "Document all anomalies for EOD follow-up before excavation",
        ]
    )

    # Build PDF
    print("\nWriting PDF file...")
    output_path = builder.build()

    print(f"\nCompleted: {output_path}")
    return output_path


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    build_section_17()
