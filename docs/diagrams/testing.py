"""
HIRT Whitepaper - Testing and Verification Diagrams Module

Functions for creating test setup diagrams, verification workflows,
pass/fail criteria visualizations, and test results templates.
"""

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon
)
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
    'light_bg': '#f7fafc',
}


def create_test_setup_diagram():
    """
    Create a diagram showing the test bench setup with all required equipment
    and their connections to the probe under test.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(5.5, 6.7, 'Test Bench Setup', ha='center', fontsize=12,
            fontweight='bold', color=COLORS['primary'])

    # Probe Under Test (center)
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

    # Test Equipment
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

    # Connection Lines
    ax.annotate('', xy=(probe_x - 0.6, probe_y + 1.8), xytext=(2.5, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=2))
    ax.text(3.2, 4.9, 'Power', fontsize=7, color=COLORS['warning'], rotation=30)

    ax.annotate('', xy=(probe_x - 0.5, probe_y), xytext=(2.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))
    ax.text(3.2, 3.3, 'ERT', fontsize=7, color=COLORS['success'])

    ax.annotate('', xy=(probe_x - 0.5, probe_y - 0.8), xytext=(2.5, 1.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))
    ax.text(3.2, 2.2, 'Coils', fontsize=7, color=COLORS['accent'], rotation=-30)

    ax.annotate('', xy=(probe_x + 0.6, probe_y + 1.8), xytext=(8.5, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=1.5))
    ax.text(7.7, 5.0, 'Signal', fontsize=7, color=COLORS['purple'], rotation=-30)

    ax.annotate('', xy=(probe_x + 0.5, probe_y), xytext=(8.5, 3.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.5))
    ax.text(7.7, 3.3, 'Inject', fontsize=7, color=COLORS['orange'])

    ax.annotate('', xy=(probe_x + 0.6, probe_y - 0.8), xytext=(8.5, 1.5),
                arrowprops=dict(arrowstyle='<->', color=COLORS['primary'], lw=2.5))
    ax.text(7.5, 2.1, 'Full System', fontsize=7, color=COLORS['primary'],
            fontweight='bold', rotation=30)

    # Environment Box
    env_box = FancyBboxPatch((3.5, 0.3), 4.0, 0.8, boxstyle="round,pad=0.02",
                              facecolor='#fff9e6', edgecolor=COLORS['orange'], lw=1)
    ax.add_patch(env_box)
    ax.text(5.5, 0.7, 'Test Environment: 20-25C, Low EMI, Well-Lit Workbench',
            ha='center', va='center', fontsize=8, color=COLORS['gray_dark'])

    # Legend
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


def create_verification_workflow():
    """
    Create a diagram showing the complete testing verification workflow
    from mechanical testing through QC sign-off.
    """
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(5.5, 7.7, 'Testing and Verification Workflow', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Workflow stages
    stages = [
        ('1. MECHANICAL', 6.5, ['Rod Integrity', 'Probe Head', 'ERT Rings', 'Cable'], '#e3f2fd'),
        ('2. ELECTRICAL', 5.0, ['Power Supply', 'Continuity', 'Communication'], '#e8f5e9'),
        ('3. MIT SUBSYSTEM', 3.5, ['Coil Params', 'DDS Output', 'TX Driver', 'RX Chain', 'Coupling'], '#fff3e0'),
        ('4. ERT SUBSYSTEM', 2.0, ['Ring Isolation', 'Contact Resist', 'Meas Chain', 'Current Src'], '#fce4ec'),
        ('5. INTEGRATION', 0.5, ['Full System', 'Reciprocity', 'Repeatability'], '#f3e5f5'),
    ]

    for stage_name, y, tests, color in stages:
        ax.add_patch(FancyBboxPatch((0.3, y - 0.35), 2.2, 0.7,
                                     boxstyle="round,pad=0.02",
                                     facecolor=color, edgecolor='black', lw=1.5))
        ax.text(1.4, y, stage_name, ha='center', va='center',
                fontsize=9, fontweight='bold', color=COLORS['primary'])

        test_width = (10.5 - 2.8) / len(tests) - 0.15
        for i, test in enumerate(tests):
            x = 2.8 + i * (test_width + 0.15)
            ax.add_patch(FancyBboxPatch((x, y - 0.25), test_width, 0.5,
                                         boxstyle="round,pad=0.02",
                                         facecolor='white', edgecolor=COLORS['gray_med'], lw=1))
            ax.text(x + test_width/2, y, test, ha='center', va='center', fontsize=7)

    # Vertical flow arrows
    for i in range(len(stages) - 1):
        y1 = stages[i][1] - 0.35
        y2 = stages[i + 1][1] + 0.35
        ax.annotate('', xy=(1.4, y2), xytext=(1.4, y1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))

    # Pass/Fail decision points
    decision_y = [6.5, 5.0, 3.5, 2.0, 0.5]
    for y in decision_y:
        diamond_x = 10.7
        diamond = Polygon([(diamond_x, y + 0.25), (diamond_x + 0.25, y),
                           (diamond_x, y - 0.25), (diamond_x - 0.25, y)],
                          facecolor='#fff9c4', edgecolor=COLORS['warning'], lw=1)
        ax.add_patch(diamond)
        ax.text(diamond_x, y, '?', ha='center', va='center', fontsize=8, fontweight='bold')

    # Final QC Sign-off
    ax.add_patch(FancyBboxPatch((4.0, -0.5), 3.0, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=COLORS['success'], edgecolor='black', lw=2))
    ax.text(5.5, -0.2, '6. QC SIGN-OFF', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    ax.annotate('', xy=(5.5, -0.5), xytext=(5.5, 0.15),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))

    # Legend
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


def create_pass_fail_criteria():
    """
    Create a visual representation of pass/fail criteria for key measurements.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle('Pass/Fail Criteria by Test Category', fontsize=12,
                 fontweight='bold', color=COLORS['primary'], y=0.98)

    # Panel A: Voltage Rails
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

    ax1.add_patch(Rectangle((1, 1.5), 2.5, 0.8, facecolor=COLORS['warning'], alpha=0.3))
    ax1.text(2.25, 1.9, 'Out of Range = FAIL', ha='center', fontsize=8, color=COLORS['warning'])
    ax1.add_patch(Rectangle((5.5, 1.5), 2.5, 0.8, facecolor=COLORS['success'], alpha=0.3))
    ax1.text(6.75, 1.9, 'In Range = PASS', ha='center', fontsize=8, color=COLORS['success'])

    # Panel B: Coil Parameters
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    ax2.set_title('(b) MIT Coil Specifications', fontsize=10, fontweight='bold', pad=5)

    ax2.add_patch(Rectangle((1, 4.5), 8, 0.6, facecolor='#e3f2fd', edgecolor=COLORS['accent'], lw=1))
    ax2.text(0.5, 4.8, 'L:', fontsize=8, fontweight='bold', ha='right')
    ax2.add_patch(Rectangle((2.5, 4.55), 4, 0.5, facecolor=COLORS['success'], alpha=0.7))
    ax2.text(1.3, 4.8, '0', fontsize=7, ha='center')
    ax2.text(2.5, 4.8, '1 mH', fontsize=7, ha='center', color='white', fontweight='bold')
    ax2.text(6.5, 4.8, '2 mH', fontsize=7, ha='center', color='white', fontweight='bold')
    ax2.text(8.7, 4.8, '3+', fontsize=7, ha='center')

    ax2.add_patch(Rectangle((1, 3.3), 8, 0.6, facecolor='#e3f2fd', edgecolor=COLORS['accent'], lw=1))
    ax2.text(0.5, 3.6, 'Q:', fontsize=8, fontweight='bold', ha='right')
    ax2.add_patch(Rectangle((4.5, 3.35), 4.5, 0.5, facecolor=COLORS['success'], alpha=0.7))
    ax2.text(1.3, 3.6, '0', fontsize=7, ha='center')
    ax2.text(4.5, 3.6, '20', fontsize=7, ha='center', color='white', fontweight='bold')
    ax2.text(8.7, 3.6, '50+', fontsize=7, ha='center')

    ax2.add_patch(Rectangle((1, 2.1), 8, 0.6, facecolor='#e3f2fd', edgecolor=COLORS['accent'], lw=1))
    ax2.text(0.5, 2.4, 'R:', fontsize=8, fontweight='bold', ha='right')
    ax2.add_patch(Rectangle((1, 2.15), 4, 0.5, facecolor=COLORS['success'], alpha=0.7))
    ax2.text(5, 2.4, '<10 ohm', fontsize=7, ha='center', fontweight='bold')
    ax2.add_patch(Rectangle((5, 2.15), 4, 0.5, facecolor=COLORS['warning'], alpha=0.3))
    ax2.text(7.5, 2.4, '>10 ohm FAIL', fontsize=7, ha='center', color=COLORS['warning'])

    ax2.text(5, 1.3, 'Target: L=1-2mH, Q>20, R<10 ohm', ha='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['success']))

    # Panel C: ERT Isolation
    ax3 = axes[1, 0]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6)
    ax3.axis('off')
    ax3.set_title('(c) ERT Ring Isolation', fontsize=10, fontweight='bold', pad=5)

    ring_positions = [2.5, 5, 7.5]
    for i, x in enumerate(ring_positions):
        ax3.add_patch(Circle((x, 4), 0.4, facecolor=COLORS['orange'], edgecolor='black', lw=1.5))
        ax3.text(x, 4, f'R{i+1}', ha='center', va='center', fontsize=8, fontweight='bold')

    ax3.annotate('', xy=(2.9, 4), xytext=(4.6, 4),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['success'], lw=2))
    ax3.text(3.75, 4.5, '>1 M ohm', ha='center', fontsize=8, color=COLORS['success'], fontweight='bold')

    ax3.annotate('', xy=(5.4, 4), xytext=(7.1, 4),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['success'], lw=2))
    ax3.text(6.25, 4.5, '>1 M ohm', ha='center', fontsize=8, color=COLORS['success'], fontweight='bold')

    ax3.add_patch(Rectangle((4.5, 1.5), 1, 0.4, facecolor=COLORS['gray_dark']))
    ax3.text(5, 1.7, 'GND', ha='center', va='center', fontsize=7, color='white')
    for x in ring_positions:
        ax3.plot([x, x, 5], [4 - 0.4, 2.5, 2.5], '--', color=COLORS['success'], lw=1)
    ax3.text(5, 2.7, '>1 M ohm each', ha='center', fontsize=8, color=COLORS['success'])

    ax3.text(5, 0.8, 'PASS: All isolation >1 M ohm', ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor=COLORS['success']))

    # Panel D: Frequency Accuracy
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 6)
    ax4.axis('off')
    ax4.set_title('(d) DDS Frequency Accuracy', fontsize=10, fontweight='bold', pad=5)

    freqs = [('2 kHz', 2, 0.02), ('5 kHz', 5, 0.05), ('10 kHz', 10, 0.1),
             ('20 kHz', 20, 0.2), ('50 kHz', 50, 0.5)]

    for i, (label, nominal, tol) in enumerate(freqs):
        y = 4.5 - i * 0.7
        ax4.plot([5], [y], 'o', color=COLORS['primary'], markersize=8)
        ax4.text(4.3, y, label, ha='right', va='center', fontsize=8, fontweight='bold')
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


def create_test_results_template():
    """
    Create a test results chart template showing how results should be recorded.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    # Panel A: Summary Status Chart
    ax1 = axes[0]
    categories = ['Mechanical\n(4 tests)', 'Electrical\n(3 tests)', 'MIT\n(5 tests)',
                  'ERT\n(4 tests)', 'Integration\n(3 tests)', 'Environment\n(2 tests)']
    pass_counts = [4, 3, 4, 3, 2, 2]
    fail_counts = [0, 0, 1, 1, 1, 0]
    total_counts = [4, 3, 5, 4, 3, 2]

    x = np.arange(len(categories))
    width = 0.6

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

    for i, (p, t) in enumerate(zip(pass_counts, total_counts)):
        rate = p / t * 100
        color = COLORS['success'] if rate == 100 else COLORS['warning']
        ax1.text(i, t + 0.2, f'{rate:.0f}%', ha='center', fontsize=9,
                 fontweight='bold', color=color)

    # Panel B: Measurement Trend Chart
    ax2 = axes[1]
    measurements = np.array([100.2, 99.8, 100.1, 100.0, 99.9, 100.3, 99.7, 100.1, 100.0, 99.8])
    mean_val = np.mean(measurements)
    std_val = np.std(measurements)

    x_meas = np.arange(1, len(measurements) + 1)

    ax2.plot(x_meas, measurements, 'o-', color=COLORS['accent'], lw=2, markersize=8)
    ax2.axhline(mean_val, color=COLORS['success'], lw=2, linestyle='-', label=f'Mean: {mean_val:.1f}')
    ax2.axhline(mean_val + std_val, color=COLORS['orange'], lw=1, linestyle='--', label=f'+1 Std: {mean_val+std_val:.1f}')
    ax2.axhline(mean_val - std_val, color=COLORS['orange'], lw=1, linestyle='--', label=f'-1 Std: {mean_val-std_val:.1f}')

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

    if std_val / mean_val * 100 < 5:
        status_text = f'PASS\nStd Dev: {std_val:.2f} ({std_val/mean_val*100:.1f}% < 5%)'
        status_color = COLORS['success']
    else:
        status_text = f'FAIL\nStd Dev: {std_val:.2f} ({std_val/mean_val*100:.1f}% >= 5%)'
        status_color = COLORS['warning']

    ax2.text(5.5, 98.5, status_text, ha='center', fontsize=9, fontweight='bold',
             color=status_color, bbox=dict(boxstyle='round', facecolor='white',
                                           edgecolor=status_color, alpha=0.8))

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf
