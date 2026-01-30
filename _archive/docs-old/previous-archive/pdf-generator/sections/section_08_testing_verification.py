#!/usr/bin/env python3
"""
HIRT Section 08: Testing and Verification - PDF Generator

Generates a publication-quality PDF document covering:
- Testing workflow and sequences
- Equipment setup and requirements
- Mechanical, electrical, and subsystem tests
- Pass/fail criteria and QC checklists
- Quantitative validation protocols

Usage:
    python section_08_testing_verification.py

Output:
    output/sections/08-testing-verification.pdf
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch
import numpy as np
from io import BytesIO

from lib.pdf_builder import SectionPDFBuilder
from lib.styles import CONTENT_WIDTH, PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING
from lib.diagrams import (
    COLORS, save_figure_to_buffer, create_figure, setup_axes_clean,
    draw_box, draw_arrow, draw_circle
)
from lib.diagrams.flowcharts import draw_process_box, draw_decision_diamond, draw_terminal, draw_flow_arrow


# =============================================================================
# DIAGRAM 1: Test Setup Diagram
# =============================================================================

def create_test_setup_diagram():
    """
    Create a diagram showing the test bench setup with all required equipment
    and their connections to the probe under test.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5.5, 6.7, 'Test Bench Setup', ha='center', fontsize=12,
            fontweight='bold', color=COLORS['primary'])

    # === Probe Under Test (center) ===
    # Probe body
    probe_x, probe_y = 5.5, 3.5
    ax.add_patch(FancyBboxPatch((probe_x - 0.4, probe_y - 1.5), 0.8, 3.0,
                                 boxstyle="round,pad=0.02",
                                 facecolor=COLORS['secondary'], edgecolor='black', lw=2))
    ax.text(probe_x, probe_y + 1.8, 'PROBE\nUNDER\nTEST', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')

    # ERT rings on probe
    for y_offset in [-0.8, 0, 0.8]:
        ax.add_patch(Rectangle((probe_x - 0.5, probe_y + y_offset - 0.1), 1.0, 0.2,
                               facecolor=COLORS['orange'], edgecolor='black', lw=1))

    # Junction box at top
    ax.add_patch(FancyBboxPatch((probe_x - 0.6, probe_y + 1.5), 1.2, 0.6,
                                 boxstyle="round,pad=0.02",
                                 facecolor=COLORS['gray_dark'], edgecolor='black', lw=1.5))
    ax.text(probe_x, probe_y + 1.8, 'JB', ha='center', va='center',
            fontsize=7, color='white', fontweight='bold')

    # === Test Equipment ===
    equipment = [
        (1.5, 5.5, 'Power Supply\n5V/12V', COLORS['light_bg'], 'Powers probe'),
        (1.5, 3.5, 'Digital\nMultimeter', COLORS['light_bg'], 'V, I, R'),
        (1.5, 1.5, 'LCR Meter\n10 kHz', COLORS['light_bg'], 'L, Q factor'),
        (9.5, 5.5, 'Oscilloscope\n50 MHz', COLORS['light_bg'], 'Waveforms'),
        (9.5, 3.5, 'Function Gen\n1Hz-100kHz', COLORS['light_bg'], 'Test signals'),
        (9.5, 1.5, 'Base Hub\n(Complete)', COLORS['accent'], 'System test'),
    ]

    for x, y, label, color, note in equipment:
        ax.add_patch(FancyBboxPatch((x - 1.0, y - 0.5), 2.0, 1.0,
                                     boxstyle="round,pad=0.03",
                                     facecolor=color, edgecolor='black', lw=1.5))
        ax.text(x, y + 0.1, label, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.text(x, y - 0.6, note, ha='center', va='top', fontsize=6,
                color=COLORS['gray_med'], style='italic')

    # === Connection Lines ===
    # Power supply to probe
    ax.annotate('', xy=(probe_x - 0.6, probe_y + 1.8), xytext=(2.5, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=2))
    ax.text(3.2, 4.9, 'Power', fontsize=7, color=COLORS['warning'], rotation=30)

    # DMM to probe (ERT rings)
    ax.annotate('', xy=(probe_x - 0.5, probe_y), xytext=(2.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))
    ax.text(3.2, 3.3, 'ERT', fontsize=7, color=COLORS['success'])

    # LCR to probe (coils)
    ax.annotate('', xy=(probe_x - 0.5, probe_y - 0.8), xytext=(2.5, 1.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))
    ax.text(3.2, 2.2, 'Coils', fontsize=7, color=COLORS['accent'], rotation=-30)

    # Oscilloscope to probe
    ax.annotate('', xy=(probe_x + 0.6, probe_y + 1.8), xytext=(8.5, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.5))
    ax.text(7.7, 5.0, 'Signal', fontsize=7, color=COLORS['purple'], rotation=-30)

    # Function generator to probe
    ax.annotate('', xy=(probe_x + 0.5, probe_y), xytext=(8.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.5))
    ax.text(7.7, 3.3, 'Inject', fontsize=7, color=COLORS['orange'])

    # Base hub to probe (thicker, bidirectional)
    ax.annotate('', xy=(probe_x + 0.6, probe_y - 0.8), xytext=(8.5, 1.5),
                arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=2.5))
    ax.text(7.5, 2.1, 'Full System', fontsize=7, color=COLORS['primary'],
            fontweight='bold', rotation=30)

    # === Environment Box ===
    env_box = FancyBboxPatch((3.5, 0.3), 4.0, 0.8, boxstyle="round,pad=0.02",
                              facecolor='#fff9e6', edgecolor=COLORS['orange'], lw=1)
    ax.add_patch(env_box)
    ax.text(5.5, 0.7, 'Test Environment: 20-25C, Low EMI, Well-Lit Workbench',
            ha='center', va='center', fontsize=8, color=COLORS['gray_dark'])

    # === Legend ===
    legend_items = [
        (COLORS['warning'], 'Power'),
        (COLORS['success'], 'ERT Test'),
        (COLORS['accent'], 'MIT Coil Test'),
        (COLORS['purple'], 'Signal Monitor'),
        (COLORS['primary'], 'System Test'),
    ]
    for i, (color, label) in enumerate(legend_items):
        ax.plot([0.3], [6.3 - i * 0.35], 'o', color=color, markersize=6)
        ax.text(0.5, 6.3 - i * 0.35, label, fontsize=7, va='center')

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# =============================================================================
# DIAGRAM 2: Testing Verification Workflow
# =============================================================================

def create_verification_workflow():
    """
    Create a diagram showing the complete testing verification workflow
    from mechanical testing through QC sign-off.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5.5, 7.7, 'Testing and Verification Workflow', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])

    # === Workflow stages (vertical flow with horizontal tests) ===
    stages = [
        ('1. MECHANICAL', 6.5, ['Rod Integrity', 'Probe Head', 'ERT Rings', 'Cable'], '#e3f2fd'),
        ('2. ELECTRICAL', 5.0, ['Power Supply', 'Continuity', 'Communication'], '#e8f5e9'),
        ('3. MIT SUBSYSTEM', 3.5, ['Coil Params', 'DDS Output', 'TX Driver', 'RX Chain', 'Coupling'], '#fff3e0'),
        ('4. ERT SUBSYSTEM', 2.0, ['Ring Isolation', 'Contact Resist', 'Meas Chain', 'Current Src'], '#fce4ec'),
        ('5. INTEGRATION', 0.5, ['Full System', 'Reciprocity', 'Repeatability'], '#f3e5f5'),
    ]

    for stage_name, y, tests, color in stages:
        # Stage header (left side)
        ax.add_patch(FancyBboxPatch((0.3, y - 0.35), 2.2, 0.7,
                                     boxstyle="round,pad=0.02",
                                     facecolor=color, edgecolor='black', lw=1.5))
        ax.text(1.4, y, stage_name, ha='center', va='center',
                fontsize=9, fontweight='bold', color=COLORS['primary'])

        # Test boxes
        test_width = (10.5 - 2.8) / len(tests) - 0.15
        for i, test in enumerate(tests):
            x = 2.8 + i * (test_width + 0.15)
            ax.add_patch(FancyBboxPatch((x, y - 0.25), test_width, 0.5,
                                         boxstyle="round,pad=0.02",
                                         facecolor='white', edgecolor=COLORS['gray_med'], lw=1))
            ax.text(x + test_width/2, y, test, ha='center', va='center', fontsize=7)

    # === Vertical flow arrows between stages ===
    for i in range(len(stages) - 1):
        y1 = stages[i][1] - 0.35
        y2 = stages[i + 1][1] + 0.35
        ax.annotate('', xy=(1.4, y2), xytext=(1.4, y1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))

    # === Pass/Fail decision points ===
    decision_y = [6.5, 5.0, 3.5, 2.0, 0.5]
    for y in decision_y:
        # Decision diamond at end of each stage
        diamond_x = 10.7
        diamond = Polygon([(diamond_x, y + 0.25), (diamond_x + 0.25, y),
                           (diamond_x, y - 0.25), (diamond_x - 0.25, y)],
                          facecolor='#fff9c4', edgecolor=COLORS['warning'], lw=1)
        ax.add_patch(diamond)
        ax.text(diamond_x, y, '?', ha='center', va='center', fontsize=8, fontweight='bold')

    # Final QC Sign-off at bottom
    ax.add_patch(FancyBboxPatch((4.0, -0.5), 3.0, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=COLORS['success'], edgecolor='black', lw=2))
    ax.text(5.5, -0.2, '6. QC SIGN-OFF', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    # Arrow to sign-off
    ax.annotate('', xy=(5.5, -0.5), xytext=(5.5, 0.15),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))

    # === Legend for pass/fail ===
    ax.add_patch(Rectangle((0.3, -0.7), 0.3, 0.2, facecolor=COLORS['success']))
    ax.text(0.75, -0.6, 'PASS', fontsize=7, va='center', color=COLORS['success'])
    ax.add_patch(Rectangle((1.5, -0.7), 0.3, 0.2, facecolor=COLORS['warning']))
    ax.text(1.95, -0.6, 'FAIL - Repair & Retest', fontsize=7, va='center', color=COLORS['warning'])
    ax.add_patch(Rectangle((3.2, -0.7), 0.3, 0.2, facecolor='#fff9c4'))
    ax.text(3.65, -0.6, 'CONDITIONAL', fontsize=7, va='center', color=COLORS['orange'])

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# =============================================================================
# DIAGRAM 3: Pass/Fail Criteria Visual
# =============================================================================

def create_pass_fail_criteria():
    """
    Create a visual representation of pass/fail criteria for key measurements.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle('Pass/Fail Criteria by Test Category', fontsize=12,
                 fontweight='bold', color=COLORS['primary'], y=0.98)

    # === Panel A: Voltage Rails ===
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis('off')
    ax1.set_title('(a) Power Supply Tolerance', fontsize=10, fontweight='bold', pad=5)

    # 3.3V rail gauge
    ax1.add_patch(Rectangle((1, 4), 3.5, 1.2, facecolor='#e8f5e9', edgecolor=COLORS['success'], lw=2))
    ax1.text(2.75, 5.5, '3.3V Rail', ha='center', fontsize=9, fontweight='bold')
    ax1.add_patch(Rectangle((1.3, 4.2), 0.9, 0.8, facecolor=COLORS['success']))
    ax1.text(1.75, 4.6, '3.2V', ha='center', fontsize=7, color='white')
    ax1.add_patch(Rectangle((2.3, 4.2), 0.9, 0.8, facecolor=COLORS['success']))
    ax1.text(2.75, 4.6, '3.3V', ha='center', fontsize=7, color='white', fontweight='bold')
    ax1.add_patch(Rectangle((3.3, 4.2), 0.9, 0.8, facecolor=COLORS['success']))
    ax1.text(3.75, 4.6, '3.4V', ha='center', fontsize=7, color='white')

    # 5V rail gauge
    ax1.add_patch(Rectangle((5.5, 4), 3.5, 1.2, facecolor='#e8f5e9', edgecolor=COLORS['success'], lw=2))
    ax1.text(7.25, 5.5, '5.0V Rail', ha='center', fontsize=9, fontweight='bold')
    ax1.add_patch(Rectangle((5.8, 4.2), 0.9, 0.8, facecolor=COLORS['success']))
    ax1.text(6.25, 4.6, '4.9V', ha='center', fontsize=7, color='white')
    ax1.add_patch(Rectangle((6.8, 4.2), 0.9, 0.8, facecolor=COLORS['success']))
    ax1.text(7.25, 4.6, '5.0V', ha='center', fontsize=7, color='white', fontweight='bold')
    ax1.add_patch(Rectangle((7.8, 4.2), 0.9, 0.8, facecolor=COLORS['success']))
    ax1.text(8.25, 4.6, '5.1V', ha='center', fontsize=7, color='white')

    ax1.text(5, 3.5, 'Tolerance: +/- 0.1V', ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['gray_med']))

    # Out of range indicator
    ax1.add_patch(Rectangle((1, 1.5), 2.5, 0.8, facecolor=COLORS['warning'], alpha=0.3))
    ax1.text(2.25, 1.9, 'Out of Range = FAIL', ha='center', fontsize=8, color=COLORS['warning'])
    ax1.add_patch(Rectangle((5.5, 1.5), 2.5, 0.8, facecolor=COLORS['success'], alpha=0.3))
    ax1.text(6.75, 1.9, 'In Range = PASS', ha='center', fontsize=8, color=COLORS['success'])

    # === Panel B: Coil Parameters ===
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    ax2.set_title('(b) MIT Coil Specifications', fontsize=10, fontweight='bold', pad=5)

    # Inductance bar
    ax2.add_patch(Rectangle((1, 4.5), 8, 0.6, facecolor='#e3f2fd', edgecolor=COLORS['accent'], lw=1))
    ax2.text(0.5, 4.8, 'L:', fontsize=8, fontweight='bold', ha='right')
    ax2.add_patch(Rectangle((2.5, 4.55), 4, 0.5, facecolor=COLORS['success'], alpha=0.7))
    ax2.text(1.3, 4.8, '0', fontsize=7, ha='center')
    ax2.text(2.5, 4.8, '1 mH', fontsize=7, ha='center', color='white', fontweight='bold')
    ax2.text(6.5, 4.8, '2 mH', fontsize=7, ha='center', color='white', fontweight='bold')
    ax2.text(8.7, 4.8, '3+', fontsize=7, ha='center')

    # Q factor bar
    ax2.add_patch(Rectangle((1, 3.3), 8, 0.6, facecolor='#e3f2fd', edgecolor=COLORS['accent'], lw=1))
    ax2.text(0.5, 3.6, 'Q:', fontsize=8, fontweight='bold', ha='right')
    ax2.add_patch(Rectangle((4.5, 3.35), 4.5, 0.5, facecolor=COLORS['success'], alpha=0.7))
    ax2.text(1.3, 3.6, '0', fontsize=7, ha='center')
    ax2.text(4.5, 3.6, '20', fontsize=7, ha='center', color='white', fontweight='bold')
    ax2.text(8.7, 3.6, '50+', fontsize=7, ha='center')

    # DC resistance bar
    ax2.add_patch(Rectangle((1, 2.1), 8, 0.6, facecolor='#e3f2fd', edgecolor=COLORS['accent'], lw=1))
    ax2.text(0.5, 2.4, 'R:', fontsize=8, fontweight='bold', ha='right')
    ax2.add_patch(Rectangle((1, 2.15), 4, 0.5, facecolor=COLORS['success'], alpha=0.7))
    ax2.text(5, 2.4, '<10 ohm', fontsize=7, ha='center', fontweight='bold')
    ax2.add_patch(Rectangle((5, 2.15), 4, 0.5, facecolor=COLORS['warning'], alpha=0.3))
    ax2.text(7.5, 2.4, '>10 ohm FAIL', fontsize=7, ha='center', color=COLORS['warning'])

    ax2.text(5, 1.3, 'Target: L=1-2mH, Q>20, R<10 ohm', ha='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['success']))

    # === Panel C: ERT Isolation ===
    ax3 = axes[1, 0]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6)
    ax3.axis('off')
    ax3.set_title('(c) ERT Ring Isolation', fontsize=10, fontweight='bold', pad=5)

    # Visual representation of isolation test
    ring_positions = [2.5, 5, 7.5]
    for i, x in enumerate(ring_positions):
        ax3.add_patch(Circle((x, 4), 0.4, facecolor=COLORS['orange'], edgecolor='black', lw=1.5))
        ax3.text(x, 4, f'R{i+1}', ha='center', va='center', fontsize=8, fontweight='bold')

    # Isolation lines between rings
    ax3.annotate('', xy=(2.9, 4), xytext=(4.6, 4),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['success'], lw=2))
    ax3.text(3.75, 4.5, '>1 M ohm', ha='center', fontsize=8, color=COLORS['success'], fontweight='bold')

    ax3.annotate('', xy=(5.4, 4), xytext=(7.1, 4),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['success'], lw=2))
    ax3.text(6.25, 4.5, '>1 M ohm', ha='center', fontsize=8, color=COLORS['success'], fontweight='bold')

    # Ground reference
    ax3.add_patch(Rectangle((4.5, 1.5), 1, 0.4, facecolor=COLORS['gray_dark']))
    ax3.text(5, 1.7, 'GND', ha='center', va='center', fontsize=7, color='white')
    for x in ring_positions:
        ax3.plot([x, x, 5], [4 - 0.4, 2.5, 2.5], '--', color=COLORS['success'], lw=1)
    ax3.text(5, 2.7, '>1 M ohm each', ha='center', fontsize=8, color=COLORS['success'])

    ax3.text(5, 0.8, 'PASS: All isolation >1 M ohm', ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor=COLORS['success']))

    # === Panel D: Frequency Accuracy ===
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 6)
    ax4.axis('off')
    ax4.set_title('(d) DDS Frequency Accuracy', fontsize=10, fontweight='bold', pad=5)

    # Target frequencies with tolerance bars
    freqs = [('2 kHz', 2, 0.02), ('5 kHz', 5, 0.05), ('10 kHz', 10, 0.1),
             ('20 kHz', 20, 0.2), ('50 kHz', 50, 0.5)]

    for i, (label, nominal, tol) in enumerate(freqs):
        y = 4.5 - i * 0.7
        # Nominal marker
        ax4.plot([5], [y], 'o', color=COLORS['primary'], markersize=8)
        ax4.text(4.3, y, label, ha='right', va='center', fontsize=8, fontweight='bold')

        # Tolerance range
        ax4.add_patch(Rectangle((3.5, y - 0.15), 3, 0.3, facecolor=COLORS['success'], alpha=0.3))
        ax4.text(8, y, f'+/-1%', ha='left', va='center', fontsize=7, color=COLORS['success'])

    ax4.text(5, 0.8, 'PASS: All frequencies within +/-1%', ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor=COLORS['success']))

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# =============================================================================
# DIAGRAM 4: Test Results Chart Template
# =============================================================================

def create_test_results_template():
    """
    Create a test results chart template showing how results should be recorded
    and visualized.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    # === Panel A: Summary Status Chart ===
    ax1 = axes[0]

    # Categories and sample data
    categories = ['Mechanical\n(4 tests)', 'Electrical\n(3 tests)', 'MIT\n(5 tests)',
                  'ERT\n(4 tests)', 'Integration\n(3 tests)', 'Environment\n(2 tests)']
    pass_counts = [4, 3, 4, 3, 2, 2]
    fail_counts = [0, 0, 1, 1, 1, 0]
    total_counts = [4, 3, 5, 4, 3, 2]

    x = np.arange(len(categories))
    width = 0.6

    # Stacked bars
    bars_pass = ax1.bar(x, pass_counts, width, label='PASS', color=COLORS['success'])
    bars_fail = ax1.bar(x, fail_counts, width, bottom=pass_counts, label='FAIL', color=COLORS['warning'])

    ax1.set_ylabel('Number of Tests', fontsize=10)
    ax1.set_title('(a) Test Results Summary', fontsize=11, fontweight='bold',
                  color=COLORS['primary'], pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=8)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 6)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add pass rate labels
    for i, (p, t) in enumerate(zip(pass_counts, total_counts)):
        rate = p / t * 100
        color = COLORS['success'] if rate == 100 else COLORS['warning']
        ax1.text(i, t + 0.2, f'{rate:.0f}%', ha='center', fontsize=9,
                 fontweight='bold', color=color)

    # === Panel B: Measurement Trend Chart ===
    ax2 = axes[1]

    # Sample measurement data (e.g., repeatability test)
    measurements = np.array([100.2, 99.8, 100.1, 100.0, 99.9, 100.3, 99.7, 100.1, 100.0, 99.8])
    mean_val = np.mean(measurements)
    std_val = np.std(measurements)

    x_meas = np.arange(1, len(measurements) + 1)

    ax2.plot(x_meas, measurements, 'o-', color=COLORS['accent'], lw=2, markersize=8)
    ax2.axhline(mean_val, color=COLORS['success'], lw=2, linestyle='-', label=f'Mean: {mean_val:.1f}')
    ax2.axhline(mean_val + std_val, color=COLORS['orange'], lw=1, linestyle='--', label=f'+1 Std: {mean_val+std_val:.1f}')
    ax2.axhline(mean_val - std_val, color=COLORS['orange'], lw=1, linestyle='--', label=f'-1 Std: {mean_val-std_val:.1f}')

    # 5% tolerance band
    ax2.fill_between([0.5, 10.5], [mean_val * 0.95] * 2, [mean_val * 1.05] * 2,
                      color=COLORS['success'], alpha=0.1, label='5% Tolerance')

    ax2.set_xlabel('Measurement Number', fontsize=10)
    ax2.set_ylabel('Value (normalized)', fontsize=10)
    ax2.set_title('(b) Repeatability Test Results', fontsize=11, fontweight='bold',
                  color=COLORS['primary'], pad=10)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(0.5, 10.5)
    ax2.set_ylim(98, 102)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # Add pass/fail annotation
    if std_val / mean_val * 100 < 5:
        status_text = f'PASS\nStd Dev: {std_val:.2f} ({std_val/mean_val*100:.1f}% < 5%)'
        status_color = COLORS['success']
    else:
        status_text = f'FAIL\nStd Dev: {std_val:.2f} ({std_val/mean_val*100:.1f}% >= 5%)'
        status_color = COLORS['warning']

    ax2.text(0.95, 0.05, status_text, transform=ax2.transAxes, fontsize=9,
             va='bottom', ha='right', fontweight='bold', color=status_color,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=status_color))

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# =============================================================================
# PDF DOCUMENT BUILDER
# =============================================================================

def build_section_08():
    """
    Build the complete Section 08: Testing and Verification PDF.

    Returns:
        Path to the generated PDF file
    """
    # Initialize builder
    builder = SectionPDFBuilder(
        section_num=8,
        title="Testing and Verification"
    )

    # Title block
    builder.add_title_block(
        subtitle="Quality Control Procedures for HIRT Probe Systems",
        version="2.0"
    )

    # === Section 8.1: Overview ===
    builder.add_section_header("8.1 Overview", level=1)
    builder.add_body_text(
        "This section provides comprehensive testing procedures and quality control "
        "checklists for HIRT probes before field deployment. All probes must pass these "
        "tests to ensure reliable field operation. The testing framework covers mechanical "
        "integrity, electrical functionality, subsystem performance, and system integration.",
        first_paragraph=True
    )

    builder.add_info_box("Testing Philosophy", [
        "All tests must PASS for field deployment approval",
        "CONDITIONAL status requires documented workarounds and supervisor approval",
        "FAIL status requires repair, component replacement, and complete re-test",
        "Quantitative measurements recorded for traceability and trend analysis"
    ])

    # === Section 8.2: Testing Workflow ===
    builder.add_section_header("8.2 Testing Workflow", level=1)
    builder.add_body_text(
        "The HIRT testing workflow follows a systematic progression from basic mechanical "
        "verification through complete system integration testing. Each stage must be completed "
        "successfully before proceeding to the next, ensuring that fundamental issues are "
        "identified and resolved before more complex testing begins.",
        first_paragraph=True
    )

    builder.add_subsection_header("8.2.1 Test Sequence")
    builder.add_numbered_list([
        "<b>Mechanical Testing</b> - Verify physical integrity of rod, probe head, ERT rings, and cables",
        "<b>Electrical Testing</b> - Verify power supply, continuity, and communication interfaces",
        "<b>MIT Subsystem Testing</b> - Verify coils, DDS, TX driver, RX chain, and coupling response",
        "<b>ERT Subsystem Testing</b> - Verify ring isolation, contact resistance, and measurement chain",
        "<b>System Integration Testing</b> - Verify complete system operation and data collection",
        "<b>Environmental Testing</b> - Verify temperature stability and waterproofing (as applicable)",
        "<b>QC Sign-off</b> - Final documentation review and deployment approval"
    ])

    # Add workflow diagram
    builder.add_figure(
        create_verification_workflow(),
        "Complete testing and verification workflow showing test categories, individual "
        "tests within each category, and the sequential progression from mechanical testing "
        "through QC sign-off. Each decision point requires all tests within the category to "
        "pass before proceeding."
    )

    # === Section 8.3: Pre-Testing Setup ===
    builder.add_section_header("8.3 Pre-Testing Setup", level=1)

    builder.add_subsection_header("8.3.1 Test Equipment Required")
    builder.add_table(
        [
            ["Equipment", "Purpose", "Minimum Specification"],
            ["Power Supply", "Probe power delivery", "5V or 12V, current-limited to 2A"],
            ["Digital Multimeter", "Resistance, voltage, current", "0.1% accuracy, 4.5 digit"],
            ["LCR Meter", "Coil inductance and Q factor", "10 kHz test frequency, 0.1% accuracy"],
            ["Oscilloscope", "Signal verification", "50 MHz bandwidth, 2-channel minimum"],
            ["Function Generator", "Signal injection testing", "1 Hz - 100 kHz, sine/square"],
            ["Base Hub", "System-level testing", "Complete calibrated unit"],
        ],
        caption="Test equipment requirements for HIRT probe verification"
    )

    # Add test setup diagram
    builder.add_figure(
        create_test_setup_diagram(),
        "Test bench setup diagram showing the probe under test (center) connected to "
        "various test instruments. Power supply provides regulated DC power, DMM measures "
        "ERT parameters, LCR meter characterizes coils, oscilloscope monitors signals, "
        "function generator injects test waveforms, and base hub enables full system testing."
    )

    builder.add_subsection_header("8.3.2 Test Environment Requirements")
    builder.add_bullet_list([
        "Clean, well-lit workbench with adequate space for equipment and probe",
        "Temperature: 20-25 degrees C (controlled room temperature)",
        "Minimize EMI sources: move phones, radios, and switching power supplies away",
        "Follow electrical safety procedures: use current-limited supplies, proper grounding",
        "Prepare test log template, probe registry entry, and calibration sheet before starting"
    ])

    # === Section 8.4: Mechanical Tests ===
    builder.add_section_header("8.4 Mechanical Tests", level=1)

    builder.add_subsection_header("8.4.1 Test M1: Rod Integrity")
    builder.add_body_text(
        "<b>Purpose:</b> Verify the probe rod is straight, undamaged, and properly assembled.",
        first_paragraph=True
    )
    builder.add_body_text("<b>Procedure:</b>")
    builder.add_numbered_list([
        "Inspect rod visually for cracks, bends, scratches, or other damage",
        "Measure rod length using tape measure (should match specification +/-5 mm)",
        "Roll rod on flat surface to check straightness - observe for wobble",
        "Verify thread engagement on all joints (multi-section probes)",
        "Perform gentle pull test on joints - no movement should occur"
    ])
    builder.add_body_text(
        "<b>Pass Criteria:</b> No visible damage, length within specification, rod straight, "
        "threads engaged properly, joints secure with no movement."
    )

    builder.add_subsection_header("8.4.2 Test M2: Probe Head Integrity")
    builder.add_body_text(
        "<b>Purpose:</b> Verify the probe head (junction box) is properly sealed and assembled.",
        first_paragraph=True
    )
    builder.add_numbered_list([
        "Inspect capsule for cracks, damage, or discoloration",
        "Verify cap seal is properly seated with no gaps",
        "Check cable gland seal - should be tight with no cable movement",
        "Test attachment to rod - should be secure with no rotation",
        "Shake gently to check for loose internal components"
    ])

    builder.add_subsection_header("8.4.3 Test M3: ERT Ring Mounting")
    builder.add_body_text(
        "<b>Purpose:</b> Verify ERT electrode rings are properly mounted and positioned.",
        first_paragraph=True
    )
    builder.add_table(
        [
            ["Ring", "Position from Tip", "Tolerance"],
            ["Ring 1", "0.5 m", "+/- 2 cm"],
            ["Ring 2", "1.5 m", "+/- 2 cm"],
            ["Ring 3", "2.5 m", "+/- 2 cm"],
        ],
        caption="ERT ring position specifications"
    )

    builder.add_subsection_header("8.4.4 Test M4: Cable Integrity")
    builder.add_body_text(
        "Visual inspection of cable for damage, kinks, or abrasion. Test continuity of all "
        "conductors end-to-end. Verify cable routing has no sharp bends that could damage "
        "insulation. Check strain relief at both ends. Gentle pull test to verify secure "
        "termination.",
        first_paragraph=True
    )

    # === Section 8.5: Electrical Tests ===
    builder.add_section_header("8.5 Electrical Tests", level=1)

    builder.add_subsection_header("8.5.1 Test E1: Power Supply Verification")
    builder.add_body_text(
        "<b>Purpose:</b> Verify power input and regulation circuits function correctly.",
        first_paragraph=True
    )
    builder.add_table(
        [
            ["Parameter", "Target", "Tolerance", "Status"],
            ["3.3V Rail", "3.3V", "+/- 0.1V", "[ ] Pass [ ] Fail"],
            ["5V Rail", "5.0V", "+/- 0.1V", "[ ] Pass [ ] Fail"],
            ["Current Draw", "< spec", "See datasheet", "[ ] Pass [ ] Fail"],
            ["Voltage Stability", "Stable", "No fluctuations", "[ ] Pass [ ] Fail"],
            ["Temperature", "< 50C", "No excessive heat", "[ ] Pass [ ] Fail"],
        ],
        caption="Power supply test criteria"
    )

    builder.add_subsection_header("8.5.2 Test E2: Continuity and Shorts")
    builder.add_body_text(
        "With power off, measure resistance between power and ground (should be >100 ohm). "
        "Check for shorts between all signal lines. Verify all connections are continuous "
        "end-to-end. Confirm shield connections are proper.",
        first_paragraph=True
    )

    builder.add_subsection_header("8.5.3 Test E3: Communication Interface")
    builder.add_body_text(
        "Connect probe to base hub. Send test command and verify response. Test data transfer "
        "with multiple packets. Check reliability over 100 transactions (must be >95% success rate).",
        first_paragraph=True
    )

    # === Section 8.6: MIT Subsystem Tests ===
    builder.add_section_header("8.6 MIT Subsystem Tests", level=1)

    builder.add_subsection_header("8.6.1 Test MIT1: Coil Parameters")
    builder.add_table(
        [
            ["Parameter", "TX Coil", "RX Coil", "Pass Criteria"],
            ["Inductance", "Measured: ___", "Measured: ___", "1-2 mH"],
            ["Q Factor", "Measured: ___", "Measured: ___", "> 20"],
            ["DC Resistance", "Measured: ___", "Measured: ___", "< 10 ohm"],
            ["Coil Isolation", "N/A", "N/A", "> 1 M ohm"],
        ],
        caption="MIT coil parameter measurements"
    )

    builder.add_subsection_header("8.6.2 Test MIT2: DDS Output")
    builder.add_body_text(
        "Configure DDS for 10 kHz test frequency. Measure output with oscilloscope. Verify "
        "frequency accuracy is within +/-1%. Check output amplitude stability. Measure THD "
        "(target: <1%). Test at 2, 5, 10, 20, and 50 kHz.",
        first_paragraph=True
    )

    builder.add_subsection_header("8.6.3 Test MIT3: TX Driver Output")
    builder.add_body_text(
        "Measure DDS output before driver. Measure driver output after amplification. "
        "Calculate gain and compare to design specification (+/-10%). Verify no clipping "
        "or distortion. Test across frequency range.",
        first_paragraph=True
    )

    builder.add_subsection_header("8.6.4 Test MIT4: RX Chain Response")
    builder.add_body_text(
        "Inject known test signal into RX input. Measure signal at each amplification stage. "
        "Calculate gain at each stage. Verify total system gain matches design (+/-10%). "
        "Measure noise floor (target: <1% of full scale).",
        first_paragraph=True
    )

    builder.add_subsection_header("8.6.5 Test MIT5: Coupling Test")
    builder.add_body_text(
        "Set up two probes 1-3 m apart. Place aluminum calibration target between probes. "
        "Configure TX on one probe, RX on other. Measure amplitude and phase response. "
        "Remove target and establish baseline. Expected response with target: 10-50% amplitude "
        "drop, 5-30 degree phase lag.",
        first_paragraph=True
    )

    # === Section 8.7: ERT Subsystem Tests ===
    builder.add_section_header("8.7 ERT Subsystem Tests", level=1)

    builder.add_subsection_header("8.7.1 Test ERT1: Ring Isolation")
    builder.add_body_text(
        "Measure resistance between adjacent rings (should be >1 M ohm). Measure each ring "
        "to ground and to probe rod body (all should be >1 M ohm). Repeat measurements after "
        "water exposure to verify sealing integrity.",
        first_paragraph=True
    )

    builder.add_subsection_header("8.7.2 Test ERT2: Contact Resistance")
    builder.add_body_text(
        "Insert probe into test medium (sand box with known resistivity). Inject test current "
        "(0.5-1 mA). Measure voltage between adjacent rings. Calculate contact resistance "
        "(target: <1000 ohm). Verify readings are stable with no drift.",
        first_paragraph=True
    )

    builder.add_subsection_header("8.7.3 Test ERT3: Measurement Chain")
    builder.add_body_text(
        "Apply known voltage to ERT input. Select each ring via multiplexer. Verify correct "
        "ring selection. Measure at amplifier output. Read ADC value and compare to expected "
        "(should be within +/-5%). Test all rings sequentially.",
        first_paragraph=True
    )

    builder.add_subsection_header("8.7.4 Test ERT4: Current Source")
    builder.add_body_text(
        "Connect to known test load resistance. Configure current levels: 0.5, 1.0, 1.5, 2.0 mA. "
        "Measure actual current at each setting. Verify accuracy is within +/-5%. Test stability "
        "over 1 minute at each level.",
        first_paragraph=True
    )

    # Add pass/fail criteria diagram
    builder.add_figure(
        create_pass_fail_criteria(),
        "Visual representation of pass/fail criteria for key test parameters: (a) Power supply "
        "voltage tolerance bands showing the +/-0.1V acceptable range for 3.3V and 5V rails; "
        "(b) MIT coil specification ranges for inductance, Q factor, and DC resistance; "
        "(c) ERT ring isolation requirements showing minimum 1 M ohm between rings and to ground; "
        "(d) DDS frequency accuracy tolerance of +/-1% across all test frequencies."
    )

    # === Section 8.8: System Integration Tests ===
    builder.add_section_header("8.8 System Integration Tests", level=1)

    builder.add_subsection_header("8.8.1 Test INT1: Full System Test")
    builder.add_body_text(
        "Power on complete probe system. Verify communication with base hub. Test MIT "
        "measurement acquisition. Test ERT measurement acquisition. Verify data collection "
        "and storage. Test probe synchronization with other units.",
        first_paragraph=True
    )

    builder.add_subsection_header("8.8.2 Test INT2: Reciprocity Test")
    builder.add_body_text(
        "Set up two probes (A and B) at fixed separation. Measure A transmitting to B. "
        "Measure B transmitting to A. Compare results - should match within 5%. Repeat "
        "for multiple probe pairs to verify consistency.",
        first_paragraph=True
    )

    builder.add_subsection_header("8.8.3 Test INT3: Repeatability Test")
    builder.add_body_text(
        "Set up fixed test configuration. Take measurement. Wait 1 minute. Take same "
        "measurement. Repeat 5-10 times. Calculate mean and standard deviation. "
        "Pass criteria: standard deviation <5% of mean. No systematic drift over time.",
        first_paragraph=True
    )

    # Add test results template
    builder.add_figure(
        create_test_results_template(),
        "Test results recording template: (a) Summary bar chart showing pass/fail counts "
        "by test category with pass rate percentages; (b) Repeatability test trend chart "
        "showing individual measurements, mean value, standard deviation bands, and 5% "
        "tolerance envelope. This format enables rapid identification of problematic areas "
        "and provides documented evidence for QC sign-off."
    )

    # === Section 8.9: Quantitative Validation Protocols ===
    builder.add_section_header("8.9 Quantitative Validation Protocols", level=1)
    builder.add_body_text(
        "Beyond functional testing, rigorous quantitative validation is required to ensure "
        "the HIRT system meets scientific publication standards.",
        first_paragraph=True
    )

    builder.add_subsection_header("8.9.1 ERT Accuracy Validation (NIST-Traceable)")
    builder.add_body_text(
        "Connect precision metal-film resistors (0.1% tolerance) to probe electrode inputs. "
        "Test values: 100 ohm, 1 k ohm, 10 k ohm. Inject currents at 0.5, 1.0, 2.0 mA. "
        "Record 50 samples for each combination. Calculate Mean Absolute Percentage Error (MAPE).",
        first_paragraph=True
    )
    builder.add_equation("MAPE = (1/n) * SUM( |R_measured - R_true| / R_true ) * 100%")
    builder.add_body_text("<b>Target:</b> MAPE < 1.0% across the dynamic range.")

    builder.add_subsection_header("8.9.2 MIT Sensitivity Validation (Standard Loop)")
    builder.add_body_text(
        "Construct Standard Calibration Loop: single turn of 14 AWG copper wire, 10 cm diameter, "
        "shorted. Place loop coaxially with TX/RX pair at distances of 0.5 m, 1.0 m, 1.5 m. "
        "Measure change in mutual impedance at 10 kHz. Compare to analytical dipole-loop solution.",
        first_paragraph=True
    )
    builder.add_body_text("<b>Target:</b> Measurement within +/-10% of theoretical prediction.")

    builder.add_subsection_header("8.9.3 Receiver Noise Characterization")
    builder.add_body_text(
        "Short RX inputs at Zone Hub. Record 10-second timeseries at 30 kSPS. Compute Power "
        "Spectral Density (PSD). Report input-referred noise in nV/sqrt(Hz) at 2, 10, 50 kHz.",
        first_paragraph=True
    )
    builder.add_body_text("<b>Target:</b> < 20 nV/sqrt(Hz) (comparable to commercial geophysical amplifiers).")

    # === Section 8.10: QC Checklist Summary ===
    builder.add_section_header("8.10 QC Checklist Summary", level=1)

    builder.add_warning_box("QUALITY CONTROL REQUIREMENTS", [
        "ALL mechanical checks must pass before electrical testing",
        "ALL electrical checks must pass before subsystem testing",
        "ALL subsystem checks must pass before integration testing",
        "FAIL on any critical test requires repair and COMPLETE re-test of that category",
        "CONDITIONAL status requires documented workarounds and supervisor approval"
    ])

    builder.add_table(
        [
            ["Category", "Tests", "Critical Items", "Typical Time"],
            ["Mechanical", "M1-M4", "Rod integrity, sealing", "30 min"],
            ["Electrical", "E1-E3", "Power rails, communication", "30 min"],
            ["MIT Subsystem", "MIT1-MIT5", "Coil params, coupling", "60 min"],
            ["ERT Subsystem", "ERT1-ERT4", "Ring isolation, accuracy", "45 min"],
            ["Integration", "INT1-INT3", "Full system, reciprocity", "45 min"],
            ["Environmental", "ENV1-ENV2", "Waterproofing", "60 min"],
        ],
        caption="QC test category summary"
    )

    builder.add_spacer()
    builder.add_body_text(
        "<b>Overall Status Determination:</b>",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "<b>PASS:</b> All tests passed - probe approved for field deployment",
        "<b>CONDITIONAL:</b> Minor issues with documented workarounds - requires supervisor approval",
        "<b>FAIL:</b> Critical issues found - requires repair, component replacement, and re-test"
    ])

    # === Section 8.11: Sign-Off and Documentation ===
    builder.add_section_header("8.11 Sign-Off and Documentation", level=1)
    builder.add_body_text(
        "Upon completion of all testing, the QC inspector must complete the sign-off form "
        "including probe ID, test date, inspector name, overall status, and any issues found "
        "with required actions. The completed test log, calibration sheet, and sign-off form "
        "must be filed with the probe registry before field deployment.",
        first_paragraph=True
    )

    builder.add_info_box("Documentation Requirements", [
        "Completed test log with all measurements recorded",
        "Calibration sheet with coil parameters and baseline values",
        "Sign-off form with inspector signature and date",
        "Probe registry entry updated with test status and deployment approval"
    ])

    builder.add_note(
        "For calibration procedures, see Section 9: Calibration. For troubleshooting "
        "guidance when tests fail, see Section 13: Troubleshooting."
    )

    # Build the PDF
    return builder.build()


def add_subsection_header(builder, text):
    """Helper to add subsection header with level 2."""
    builder.add_section_header(text, level=2)


# Monkey-patch the helper method
SectionPDFBuilder.add_subsection_header = lambda self, text: self.add_section_header(text, level=2)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Generating HIRT Section 08: Testing and Verification PDF...")
    output_path = build_section_08()
    print(f"Complete! Output: {output_path}")
