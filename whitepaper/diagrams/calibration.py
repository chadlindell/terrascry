"""
HIRT Whitepaper - Calibration Diagrams Module

Functions for creating calibration procedure diagrams, test setup visualizations,
and calibration data charts.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, Ellipse
import numpy as np
from io import BytesIO

# Color palette
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
    'light_green': '#c6f6d5',
    'light_blue': '#ebf8ff',
    'light_orange': '#feebc8',
    'light_red': '#fed7d7',
    'light_purple': '#e9d8fd',
    'ground_tan': '#d4a373',
}


def create_calibration_hierarchy():
    """
    Create calibration hierarchy pyramid diagram showing levels from
    component to array-level calibration.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Pyramid levels (bottom to top)
    levels = [
        (1, 'Component-Level', 'Coils, resistors, ADCs', COLORS['light_blue'], 8),
        (2, 'Subsystem-Level', 'TX chain, RX chain, ERT', COLORS['light_green'], 6),
        (3, 'System-Level', 'Complete probe in air/known', COLORS['light_orange'], 4),
        (4, 'Array-Level', 'Cross-calibration', COLORS['light_purple'], 2),
    ]

    y_base = 0.5
    for level_num, name, description, color, width in levels:
        y = y_base + (level_num - 1) * 1.2
        x_center = 5
        x_left = x_center - width / 2
        height = 1.0

        # Draw trapezoid/rectangle
        ax.add_patch(FancyBboxPatch((x_left, y), width, height,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', linewidth=1.5))

        # Level name
        ax.text(x_center, y + 0.65, name, ha='center', va='center',
                fontsize=10, fontweight='bold', color=COLORS['primary'])
        # Description
        ax.text(x_center, y + 0.3, description, ha='center', va='center',
                fontsize=8, color=COLORS['gray_dark'])

    # Arrows showing progression
    for i in range(3):
        y_start = y_base + i * 1.2 + 1.0
        y_end = y_start + 0.2
        ax.annotate('', xy=(5, y_end), xytext=(5, y_start),
                    arrowprops=dict(arrowstyle='->', color=COLORS['gray_med'], lw=2))

    # Side annotations
    ax.text(0.5, 2.5, 'Increasing\nComplexity', ha='center', va='center',
            fontsize=9, color=COLORS['gray_dark'], rotation=90)
    ax.text(9.5, 2.5, 'Increasing\nConfidence', ha='center', va='center',
            fontsize=9, color=COLORS['success'], rotation=-90)

    # Title
    ax.text(5, 5.8, 'Calibration Hierarchy', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_air_calibration_setup():
    """
    Create diagram showing air calibration test setup with probe
    isolated from ground effects and metal objects.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Background - test area
    ax.add_patch(FancyBboxPatch((0.5, 0.5), 9, 4.5,
                                boxstyle="round,pad=0.05",
                                facecolor='#f7fafc', edgecolor=COLORS['gray_med'],
                                linewidth=1, linestyle='--'))
    ax.text(5, 4.7, 'Calibration Test Area (Metal-Free Zone)', ha='center',
            fontsize=10, fontweight='bold', color=COLORS['primary'])

    # Non-conductive support stand
    stand_x = 3
    ax.add_patch(Rectangle((stand_x - 0.1, 0.8), 0.2, 2.5,
                           facecolor=COLORS['orange'], edgecolor='black', linewidth=1))
    ax.add_patch(Rectangle((stand_x - 0.5, 0.5), 1.0, 0.3,
                           facecolor=COLORS['orange'], edgecolor='black', linewidth=1))
    ax.text(stand_x, 0.3, 'Non-Metallic\nStand', ha='center', fontsize=7,
            color=COLORS['orange'])

    # Probe under test (horizontal orientation)
    probe_y = 3.3
    ax.add_patch(Rectangle((stand_x - 0.1, probe_y - 0.1), 3.5, 0.2,
                           facecolor=COLORS['secondary'], edgecolor='black', linewidth=1))

    # Coils on probe
    ax.add_patch(Ellipse((stand_x + 0.8, probe_y), 0.5, 0.4,
                         facecolor=COLORS['success'], edgecolor='black', linewidth=1))
    ax.text(stand_x + 0.8, probe_y - 0.4, 'TX', ha='center', fontsize=7,
            fontweight='bold', color=COLORS['success'])

    ax.add_patch(Ellipse((stand_x + 2.3, probe_y), 0.5, 0.4,
                         facecolor=COLORS['accent'], edgecolor='black', linewidth=1))
    ax.text(stand_x + 2.3, probe_y - 0.4, 'RX', ha='center', fontsize=7,
            fontweight='bold', color=COLORS['accent'])

    # ERT rings
    for x_offset in [1.5, 2.8]:
        ax.add_patch(Rectangle((stand_x + x_offset - 0.05, probe_y - 0.15), 0.1, 0.3,
                               facecolor=COLORS['orange'], edgecolor='black', linewidth=0.5))

    # Test equipment
    equip_x = 7.5
    equip_y = 2.5

    # LCR Meter
    ax.add_patch(FancyBboxPatch((equip_x - 0.8, equip_y + 0.8), 1.6, 1.0,
                                boxstyle="round,pad=0.02",
                                facecolor='#e2e8f0', edgecolor='black', linewidth=1))
    ax.text(equip_x, equip_y + 1.3, 'LCR Meter', ha='center', fontsize=8,
            fontweight='bold')
    ax.text(equip_x, equip_y + 1.0, '10 kHz', ha='center', fontsize=7,
            color=COLORS['gray_dark'])

    # Oscilloscope
    ax.add_patch(FancyBboxPatch((equip_x - 0.8, equip_y - 0.4), 1.6, 1.0,
                                boxstyle="round,pad=0.02",
                                facecolor='#e2e8f0', edgecolor='black', linewidth=1))
    ax.text(equip_x, equip_y + 0.1, 'Oscilloscope', ha='center', fontsize=8,
            fontweight='bold')

    # Connection lines
    ax.plot([stand_x + 3.4, equip_x - 0.8], [probe_y, equip_y + 1.3],
            color=COLORS['gray_dark'], linewidth=1.5, linestyle='-')
    ax.plot([stand_x + 3.4, equip_x - 0.8], [probe_y, equip_y + 0.1],
            color=COLORS['gray_dark'], linewidth=1.5, linestyle='-')

    # Distance markers
    ax.annotate('', xy=(1, 3.3), xytext=(1, 0.5),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray_med'], lw=1))
    ax.text(0.7, 1.9, '>1m', fontsize=8, ha='center', color=COLORS['gray_med'])

    # Exclusion zone
    ax.add_patch(Circle((1.5, 3.3), 0.8, facecolor='none',
                        edgecolor=COLORS['warning'], linewidth=1.5, linestyle='--'))
    ax.text(1.5, 2.3, 'No Metal\nWithin 1m', ha='center', fontsize=7,
            color=COLORS['warning'])

    # Requirements checklist
    checklist_x = 8.5
    checklist_items = [
        'Temp: 20-25C',
        'Warm-up: 15 min',
        'EMI: Low',
    ]
    ax.text(checklist_x, 4.5, 'Requirements:', fontsize=8, fontweight='bold',
            color=COLORS['primary'])
    for i, item in enumerate(checklist_items):
        ax.text(checklist_x, 4.1 - i * 0.35, f'\u2022 {item}', fontsize=7,
                color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_known_target_setup():
    """
    Create diagram showing calibration test setup with known targets
    (aluminum plate, steel bar) and probe pairs.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Ground surface
    ax.axhline(2.5, color='#654321', linewidth=2.5)
    ax.fill_between([0, 10], [2.5, 2.5], [0, 0], color=COLORS['ground_tan'], alpha=0.4)
    ax.text(0.3, 2.65, 'Ground Surface', fontsize=8, color='#654321')

    # Probe 1 (TX)
    probe1_x = 2
    ax.add_patch(Rectangle((probe1_x - 0.15, 0.3), 0.3, 2.2,
                           facecolor=COLORS['secondary'], edgecolor='black', linewidth=1))
    # Junction box
    ax.add_patch(Rectangle((probe1_x - 0.25, 2.5), 0.5, 0.3,
                           facecolor=COLORS['gray_dark'], edgecolor='black', linewidth=1))
    ax.text(probe1_x, 2.95, 'Probe 1\n(TX)', ha='center', fontsize=8, fontweight='bold')

    # TX Coils
    for y, label in [(1.8, 'TX'), (1.0, 'RX')]:
        color = COLORS['success'] if 'TX' in label else COLORS['accent']
        ax.add_patch(Circle((probe1_x, y), 0.15, facecolor=color, edgecolor='black'))

    # Probe 2 (RX)
    probe2_x = 6
    ax.add_patch(Rectangle((probe2_x - 0.15, 0.3), 0.3, 2.2,
                           facecolor=COLORS['secondary'], edgecolor='black', linewidth=1))
    # Junction box
    ax.add_patch(Rectangle((probe2_x - 0.25, 2.5), 0.5, 0.3,
                           facecolor=COLORS['gray_dark'], edgecolor='black', linewidth=1))
    ax.text(probe2_x, 2.95, 'Probe 2\n(RX)', ha='center', fontsize=8, fontweight='bold')

    # RX Coils
    for y, label in [(1.8, 'TX'), (1.0, 'RX')]:
        color = COLORS['success'] if 'TX' in label else COLORS['accent']
        ax.add_patch(Circle((probe2_x, y), 0.15, facecolor=color, edgecolor='black'))

    # Known target (aluminum plate) between probes
    target_x = 4
    target_y = 1.4
    ax.add_patch(Rectangle((target_x - 0.5, target_y - 0.3), 1.0, 0.6,
                           facecolor='#c0c0c0', edgecolor='#808080', linewidth=2))
    ax.text(target_x, target_y, 'Al', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['gray_dark'])
    ax.annotate('Known Target\n(Aluminum Plate)', (target_x, target_y - 0.3),
                (target_x, target_y - 0.9), ha='center', fontsize=8,
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1))

    # Signal ray paths
    for y1, y2, alpha in [(1.8, 1.8, 0.6), (1.8, 1.0, 0.4), (1.0, 1.8, 0.4), (1.0, 1.0, 0.6)]:
        ax.plot([probe1_x, probe2_x], [y1, y2],
                color=COLORS['success'], alpha=alpha, linewidth=1.5, linestyle='--')

    # Distance marker
    ax.annotate('', xy=(probe1_x, 0.15), xytext=(probe2_x, 0.15),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray_med'], lw=1.5))
    ax.text(4, 0.0, 'd = 1-3m (adjustable)', ha='center', fontsize=8,
            color=COLORS['gray_med'])

    # Test equipment panel
    equip_box_x = 8
    ax.add_patch(FancyBboxPatch((equip_box_x - 0.8, 3.5), 2.2, 2.2,
                                boxstyle="round,pad=0.05",
                                facecolor='#f7fafc', edgecolor=COLORS['gray_med'],
                                linewidth=1))
    ax.text(equip_box_x + 0.3, 5.5, 'Electronics Hub', ha='center',
            fontsize=9, fontweight='bold', color=COLORS['primary'])

    # Hub components
    components = [
        ('DDS + TX Driver', 4.9, COLORS['light_green']),
        ('RX Preamp', 4.4, COLORS['light_blue']),
        ('Lock-In Amp', 3.9, COLORS['light_purple']),
    ]
    for name, y, color in components:
        ax.add_patch(FancyBboxPatch((equip_box_x - 0.6, y), 1.8, 0.4,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', linewidth=0.5))
        ax.text(equip_box_x + 0.3, y + 0.2, name, ha='center', fontsize=7)

    # Cables to probes
    ax.plot([probe1_x + 0.25, equip_box_x - 0.8], [2.65, 4.9],
            color='black', linewidth=1)
    ax.plot([probe2_x + 0.25, equip_box_x - 0.8], [2.65, 4.4],
            color='black', linewidth=1)

    # Test parameters box
    ax.add_patch(FancyBboxPatch((0.3, 3.5), 2.4, 2.2,
                                boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=COLORS['accent'],
                                linewidth=1))
    ax.text(1.5, 5.5, 'Test Parameters', ha='center',
            fontsize=9, fontweight='bold', color=COLORS['primary'])
    params = [
        'Freq: 2, 5, 10, 20, 50 kHz',
        'Target: Al plate, Fe bar',
        'Distance: 1, 2, 3 m',
        'Measure: Amp, Phase',
    ]
    for i, param in enumerate(params):
        ax.text(0.5, 5.0 - i * 0.4, f'\u2022 {param}', fontsize=7,
                color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_calibration_data_chart():
    """
    Create calibration data visualization showing expected vs measured
    response curves for MIT system across frequencies.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # === Panel A: Amplitude Response ===
    ax1 = axes[0]

    frequencies = np.array([2, 5, 10, 20, 50])  # kHz

    # Simulated calibration data
    np.random.seed(42)
    baseline_amp = np.array([1.0, 0.95, 0.88, 0.78, 0.62])
    with_target_amp = np.array([0.75, 0.68, 0.58, 0.48, 0.35])
    error_bars = np.array([0.03, 0.03, 0.025, 0.025, 0.02])

    # Plot baseline
    ax1.errorbar(frequencies, baseline_amp, yerr=error_bars, fmt='o-',
                 color=COLORS['accent'], linewidth=2, markersize=8,
                 capsize=4, label='Air Baseline')

    # Plot with target
    ax1.errorbar(frequencies, with_target_amp, yerr=error_bars, fmt='s-',
                 color=COLORS['warning'], linewidth=2, markersize=8,
                 capsize=4, label='With Al Target')

    # Expected response (dashed)
    ax1.plot(frequencies, baseline_amp * 0.98, '--',
             color=COLORS['gray_med'], linewidth=1.5, label='Expected (model)')

    # Formatting
    ax1.set_xlabel('Frequency (kHz)', fontsize=10)
    ax1.set_ylabel('Normalized Amplitude', fontsize=10)
    ax1.set_title('(a) Amplitude Response', fontsize=11,
                  fontweight='bold', color=COLORS['primary'])
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.set_xlim(0, 55)
    ax1.set_ylim(0, 1.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Pass/fail zone
    ax1.axhspan(0.85, 1.15, alpha=0.1, color=COLORS['success'])
    ax1.text(3, 1.1, 'Baseline\nSpec Range', fontsize=7, color=COLORS['success'])

    # === Panel B: Phase Response ===
    ax2 = axes[1]

    # Simulated phase data
    baseline_phase = np.array([2, 5, 10, 18, 32])  # degrees
    with_target_phase = np.array([8, 15, 25, 38, 55])  # degrees
    phase_error = np.array([1, 1.5, 2, 2.5, 3])

    # Plot baseline
    ax2.errorbar(frequencies, baseline_phase, yerr=phase_error, fmt='o-',
                 color=COLORS['accent'], linewidth=2, markersize=8,
                 capsize=4, label='Air Baseline')

    # Plot with target
    ax2.errorbar(frequencies, with_target_phase, yerr=phase_error, fmt='s-',
                 color=COLORS['warning'], linewidth=2, markersize=8,
                 capsize=4, label='With Al Target')

    # Phase shift region
    for i, (f, p1, p2) in enumerate(zip(frequencies, baseline_phase, with_target_phase)):
        ax2.annotate('', xy=(f, p2), xytext=(f, p1),
                     arrowprops=dict(arrowstyle='->', color=COLORS['orange'],
                                     lw=1, alpha=0.5))

    ax2.text(35, 30, 'Phase\nShift', fontsize=8, color=COLORS['orange'],
             ha='center')

    # Formatting
    ax2.set_xlabel('Frequency (kHz)', fontsize=10)
    ax2.set_ylabel('Phase Angle (degrees)', fontsize=10)
    ax2.set_title('(b) Phase Response', fontsize=11,
                  fontweight='bold', color=COLORS['primary'])
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.set_xlim(0, 55)
    ax2.set_ylim(0, 70)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf
