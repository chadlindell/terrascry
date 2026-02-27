#!/usr/bin/env python3
"""
HIRT Section 09: Calibration - Publication-Quality PDF Generator

This module generates the calibration section of the HIRT whitepaper,
covering MIT and ERT calibration procedures, uncertainty analysis,
and field verification protocols.

Usage:
    python section_09_calibration.py

Output:
    output/sections/09-calibration.pdf
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, Ellipse
import numpy as np
from io import BytesIO

from lib.pdf_builder import SectionPDFBuilder
from lib.styles import CONTENT_WIDTH, PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING
from lib.diagrams.flowcharts import create_calibration_workflow

# Color palette for diagrams
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


# ============================================================================
# DIAGRAM 1: CALIBRATION HIERARCHY
# ============================================================================
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


# ============================================================================
# DIAGRAM 2: AIR CALIBRATION SETUP
# ============================================================================
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


# ============================================================================
# DIAGRAM 3: KNOWN-TARGET TEST SETUP
# ============================================================================
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


# ============================================================================
# DIAGRAM 4: CALIBRATION DATA CHART
# ============================================================================
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


# ============================================================================
# DOCUMENT BUILDER
# ============================================================================
def build_section_09():
    """Build Section 09: Calibration PDF document."""

    print("Generating Section 09: Calibration diagrams...")

    # Generate diagrams
    fig_workflow = create_calibration_workflow()
    fig_hierarchy = create_calibration_hierarchy()
    fig_air_setup = create_air_calibration_setup()
    fig_target_setup = create_known_target_setup()
    fig_data_chart = create_calibration_data_chart()

    print("Building PDF...")

    # Create PDF builder
    builder = SectionPDFBuilder(
        section_num=9,
        title="Calibration"
    )

    # Title block
    builder.add_title_block(
        subtitle="MIT and ERT Calibration Procedures for Field-Ready Probes"
    )

    # Overview
    builder.add_section_header("9.1 Overview")
    builder.add_body_text(
        "This section provides detailed calibration procedures for HIRT probes before "
        "field deployment. Calibration ensures accurate and consistent measurements "
        "across all probes in the array. The dual-channel nature of HIRT (MIT and ERT) "
        "requires calibration of both subsystems, along with system-level verification "
        "to ensure the complete measurement chain performs within specification.",
        first_paragraph=True
    )
    builder.add_body_text(
        "Proper calibration is essential for scientific data quality. Without it, "
        "probe-to-probe variations can introduce systematic errors that compromise "
        "tomographic reconstruction. The procedures outlined here establish traceability "
        "from individual components through to array-level cross-calibration."
    )

    # Calibration Principles
    builder.add_section_header("9.2 Calibration Principles")
    builder.add_section_header("Why Calibrate?", level=2)
    builder.add_numbered_list([
        "<b>Probe-to-probe consistency</b> - Ensure all probes give comparable readings under identical conditions",
        "<b>Accuracy</b> - Relate measurements to physical units (mH, ohms, volts)",
        "<b>Drift compensation</b> - Account for component aging and environmental effects",
        "<b>Quality assurance</b> - Verify proper assembly and function before deployment"
    ])

    # Calibration hierarchy figure
    builder.add_figure(
        fig_hierarchy,
        "Calibration hierarchy showing the progression from component-level "
        "measurements through array-level cross-calibration. Each level builds "
        "upon the previous, increasing overall system confidence.",
        width=CONTENT_WIDTH * 0.85
    )

    # Key parameters table
    builder.add_section_header("Key Parameters and Tolerances", level=2)
    param_data = [
        ['Parameter', 'Target', 'Tolerance'],
        ['Coil Inductance', '1-2 mH', '+/-10%'],
        ['Coil Q Factor', '>20', 'Minimum'],
        ['DDS Frequency', 'Commanded', '+/-1%'],
        ['TX Amplitude', 'Design spec', '+/-10%'],
        ['RX Gain', 'Design spec', '+/-10%'],
        ['ERT Current', 'Commanded', '+/-5%'],
        ['Reciprocity', 'A->B = B->A', '+/-5%'],
    ]
    builder.add_table(param_data, caption="Calibration parameters and acceptance tolerances")

    # Calibration Workflow
    builder.add_section_header("9.3 Calibration Workflow")
    builder.add_body_text(
        "The calibration workflow proceeds through three main phases: MIT calibration "
        "(air baseline, known resistor test, frequency sweep), ERT calibration (ring "
        "isolation, contact resistance, current source verification), and target response "
        "characterization (metal target tests, distance sweeps, lookup table generation).",
        first_paragraph=True
    )

    builder.add_figure(
        fig_workflow,
        "Complete calibration workflow showing the three-phase approach: MIT calibration "
        "(blue), ERT calibration (green), and target response characterization (orange). "
        "Checkmarks indicate verification points where data is recorded.",
        width=CONTENT_WIDTH * 0.9
    )

    # Required Equipment
    builder.add_section_header("9.4 Required Equipment")

    equip_data = [
        ['Equipment', 'Purpose', 'Notes'],
        ['LCR Meter', 'Coil measurements', 'Set to 10 kHz'],
        ['Oscilloscope', 'Signal verification', '50 MHz minimum'],
        ['Multimeter', 'V/I/R measurements', '6-digit preferred'],
        ['Function Generator', 'Signal injection', '1 Hz - 100 kHz'],
        ['Known Test Targets', 'Coupling verification', 'Al plate, steel bar'],
        ['Test Medium', 'ERT testing', 'Sand box, known resistivity'],
        ['Reference Probe', 'Cross-calibration', 'If available'],
    ]
    builder.add_table(equip_data, caption="Required calibration equipment")

    builder.add_section_header("Environment Requirements", level=2)
    builder.add_bullet_list([
        "Temperature: 20-25C (stable, within +/-2C during calibration)",
        "EMI: Minimize interference sources (turn off nearby equipment, use shielded area)",
        "Warm-up time: Allow 15-30 minutes for equipment stabilization",
        "Humidity: <80% RH to prevent condensation on probe surfaces"
    ])

    # Air Calibration Setup
    builder.add_section_header("9.5 Air Calibration Setup")
    builder.add_body_text(
        "Air calibration establishes the baseline response of the probe when isolated "
        "from all external influences. The probe is suspended on a non-metallic stand "
        "(wood, PVC, or fiberglass) at least 1 meter from any metal objects. This "
        "measurement provides the 'zero' reference for subsequent target detection.",
        first_paragraph=True
    )

    builder.add_figure(
        fig_air_setup,
        "Air calibration test setup showing probe suspended on non-metallic stand "
        "with test equipment connections. The 1-meter exclusion zone ensures no "
        "spurious coupling from nearby metal objects.",
        width=CONTENT_WIDTH * 0.9
    )

    # Coil calibration procedures
    builder.add_section_header("9.6 Coil Calibration Procedures")

    builder.add_section_header("TX Coil Inductance", level=2)
    builder.add_numbered_list([
        "Connect LCR meter to TX coil leads (ensure proper polarity)",
        "Set measurement frequency to 10 kHz",
        "Measure inductance (L) and record in mH",
        "Compare to specification: target 1-2 mH, tolerance +/-10%",
        "Record Pass/Fail status on calibration sheet"
    ])

    builder.add_section_header("TX Coil Q Factor", level=2)
    builder.add_numbered_list([
        "Using same LCR meter setup as inductance measurement",
        "Measure Q factor directly if meter supports it",
        "Alternatively, calculate: Q = (2 x pi x f x L) / R",
        "Target: Q > 20 (minimum acceptable value)",
        "Low Q indicates winding issues or core problems"
    ])

    builder.add_section_header("RX Coil Parameters", level=2)
    builder.add_body_text(
        "Repeat the inductance and Q factor measurements for the RX coil. The RX coil "
        "specifications should match the TX coil within tolerance to ensure balanced "
        "coupling characteristics. Record all measurements on the calibration sheet.",
        first_paragraph=True
    )

    # TX Calibration
    builder.add_section_header("9.7 TX Chain Calibration")

    builder.add_section_header("DDS Frequency Accuracy", level=2)
    builder.add_body_text(
        "Verify the DDS (Direct Digital Synthesis) generates correct frequencies "
        "across the operating range. Test at: 2, 5, 10, 20, and 50 kHz. Measure "
        "each commanded frequency with an oscilloscope and calculate frequency error.",
        first_paragraph=True
    )

    builder.add_equation("Error (%) = (f_actual - f_commanded) / f_commanded x 100")
    builder.add_body_text(
        "Expected accuracy: +/-1% or better. Larger errors indicate DDS programming "
        "issues or clock reference problems."
    )

    builder.add_section_header("TX Output Amplitude", level=2)
    builder.add_bullet_list([
        "DDS output: ~0.6 V RMS typical (before driver stage)",
        "Driver output: 1-5 V RMS (design dependent)",
        "Gain should be 2-5x across frequency range",
        "Check for flat frequency response (+/-1 dB)"
    ])

    # RX Calibration
    builder.add_section_header("9.8 RX Chain Calibration")

    builder.add_section_header("RX Chain Gain", level=2)
    builder.add_body_text(
        "Verify RX amplification chain gain by injecting a known test signal at the "
        "RX input and measuring output at each stage. A typical test signal is 1 mV "
        "at 10 kHz. Measure at preamp output, instrumentation amplifier output, and "
        "ADC input. Calculate gain at each stage and total system gain.",
        first_paragraph=True
    )

    builder.add_section_header("RX Noise Floor", level=2)
    builder.add_numbered_list([
        "Place probe in quiet environment (away from metal objects)",
        "Apply no input signal (or short input terminals)",
        "Measure output noise amplitude over 10-second window",
        "Record multiple measurements and calculate standard deviation",
        "Target: Noise floor < 1% of full scale"
    ])

    # Known Target Setup
    builder.add_section_header("9.9 Known-Target Test Setup")
    builder.add_body_text(
        "The known-target test verifies that the complete MIT system detects "
        "conductive targets with the expected amplitude and phase response. "
        "Two probes are set up 1-3 meters apart with a known target (aluminum "
        "plate or steel bar) placed between them. This test validates both the "
        "detection sensitivity and the multi-frequency response characteristics.",
        first_paragraph=True
    )

    builder.add_figure(
        fig_target_setup,
        "Known-target test configuration with two probes and calibration target. "
        "The aluminum plate provides a non-magnetic conductive target, allowing "
        "verification of eddy current detection without ferrous interference.",
        width=CONTENT_WIDTH * 0.9
    )

    builder.add_section_header("Expected Results", level=2)
    builder.add_bullet_list([
        "Amplitude reduction: 10-50% depending on target size and distance",
        "Phase lag: 5-30 degrees depending on frequency and target conductivity",
        "Lower frequencies show deeper sensitivity (skin depth effect)",
        "Response should scale predictably with target distance"
    ])

    # Calibration Data
    builder.add_section_header("9.10 Calibration Data Analysis")
    builder.add_body_text(
        "Calibration data should be plotted to verify system behavior matches "
        "expected models. The amplitude response shows the normalized signal "
        "strength versus frequency for both baseline (air) and with-target "
        "conditions. The phase response shows the phase angle change introduced "
        "by the target. Both measurements should fall within specified tolerances.",
        first_paragraph=True
    )

    builder.add_figure(
        fig_data_chart,
        "Representative calibration data showing (a) amplitude response and "
        "(b) phase response across the MIT operating frequency range. Error bars "
        "indicate measurement uncertainty. The phase shift between baseline and "
        "with-target conditions provides target characterization information.",
        width=CONTENT_WIDTH
    )

    # ERT Calibration
    builder.add_section_header("9.11 ERT Calibration")

    builder.add_section_header("Ring Isolation Verification", level=2)
    builder.add_body_text(
        "Verify electrical isolation between ERT rings to ensure independent "
        "measurements. Measure resistance between all ring pairs, between rings "
        "and ground, and between rings and the probe rod. All measurements should "
        "exceed 1 M-ohm. Lower values indicate contamination or insulation failure.",
        first_paragraph=True
    )

    builder.add_section_header("Current Source Calibration", level=2)
    builder.add_numbered_list([
        "Connect current source to test load (1 k-ohm precision resistor)",
        "Set commanded current level (start at 1 mA)",
        "Measure actual current with calibrated multimeter",
        "Calculate calibration factor: Cal = I_actual / I_commanded",
        "Test across full range: 0.5, 1.0, 1.5, 2.0 mA",
        "Record all values and average calibration factor"
    ])

    # Error Analysis
    builder.add_section_header("9.12 Error Analysis and Uncertainty Budget")
    builder.add_body_text(
        "Scientific data reporting requires characterizing not just the measured "
        "value, but its uncertainty. The HIRT system has both systematic errors "
        "(consistent offsets that can be removed via calibration) and random errors "
        "(unpredictable fluctuations reduced by averaging).",
        first_paragraph=True
    )

    uncertainty_data = [
        ['Source', 'Type', 'Magnitude', 'Distribution'],
        ['Reference Resistor', 'Systematic', '0.1%', 'Rectangular'],
        ['ADC Quantization', 'Random', '< 1 uV', 'Uniform'],
        ['Thermal Drift', 'Systematic', '50 ppm/C', 'Linear'],
        ['Probe Geometry', 'Random', '+/- 1 cm', 'Normal'],
        ['Combined Uncertainty', 'Mixed', '~1-2%', '(k=2)'],
    ]
    builder.add_table(uncertainty_data,
                      caption="Typical uncertainty budget for HIRT measurements")

    builder.add_body_text(
        "All final data products should be reported with 95% confidence intervals (k=2):"
    )
    builder.add_equation("Measurement = X +/- 2 sigma")

    # Field Quick-Check
    builder.add_section_header("9.13 Field Quick-Check Procedure")
    builder.add_body_text(
        "Before each field deployment, perform these abbreviated checks to verify "
        "system readiness. This procedure takes approximately 15-30 minutes and "
        "can identify common issues before they affect data collection.",
        first_paragraph=True
    )

    builder.add_info_box("Pre-Deployment Quick Check", [
        "Visual inspection: No visible damage, all connections secure",
        "Power check: System powers on, LEDs functioning, communication established",
        "Coil quick test: TX/RX connected (~1-2 mH), Q factor reasonable (>20)",
        "ERT quick test: Rings isolated (>1 M-ohm), no shorts to ground",
        "Coupling verification: Wave hand near coil and observe response"
    ])

    # Recalibration Schedule
    builder.add_section_header("9.14 Recalibration Schedule")

    schedule_data = [
        ['Trigger', 'Action'],
        ['Before each field deployment', 'Quick check'],
        ['After repairs or modifications', 'Full calibration'],
        ['If measurements become inconsistent', 'Full calibration'],
        ['Every 6 months minimum', 'Full calibration'],
        ['After environmental exposure', 'Full calibration'],
    ]
    builder.add_table(schedule_data, caption="Recalibration schedule triggers and actions")

    builder.add_body_text(
        "Calibration is considered valid if: all measurements are within specifications, "
        "no repairs or modifications have been made since calibration, system performance "
        "remains consistent, and the calibration validity period has not expired "
        "(typically 6 months)."
    )

    # Documentation
    builder.add_section_header("9.15 Documentation Requirements")
    builder.add_body_text(
        "Maintain complete calibration records for all probes in the system. "
        "Records should include probe identification, calibration date, calibrator name, "
        "all measured values with units, pass/fail status for each parameter, "
        "and recalibration due date. Store records in both digital and printed formats.",
        first_paragraph=True
    )

    builder.add_warning_box("Calibration Record Retention", [
        "All probes must have current calibration records",
        "Records must be available during field operations",
        "Out-of-tolerance conditions must be documented",
        "Probes failing calibration must not be deployed"
    ])

    # Build PDF
    output_path = builder.build()
    print(f"Section 09 PDF created: {output_path}")
    return output_path


if __name__ == "__main__":
    build_section_09()
