"""
HIRT Technical Manual - Progressive Deployment Diagrams Module

Functions for creating progressive deployment workflow, depth extension,
and decision tree diagrams.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch, Arrow
)
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
    'sky': '#e8f4f8',
    'ground_dark': '#654321',
    'light_blue': '#ebf8ff',
    'light_green': '#c6f6d5',
    'light_orange': '#feebc8',
    'light_red': '#fff5f5',
    'light_purple': '#e9d8fd',
    'phase_a': '#c6f6d5',  # Light green
    'phase_b': '#feebc8',  # Light orange
    'phase_c': '#ebf8ff',  # Light blue
    'phase_d': '#e9d8fd',  # Light purple
}


def create_progressive_workflow():
    """
    Four-phase progressive deployment workflow diagram.
    Shows the progression from shallow survey through full depth.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(6, 5.7, 'Progressive Deployment Workflow', fontsize=14, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    # Phase boxes
    phases = [
        ('Phase A', 'Shallow Survey\n(0.5-1.0 m)', COLOR_PALETTE['phase_a'],
         'Initial coverage\nIdentify anomalies'),
        ('Phase B', 'Data Review\n(Analysis)', COLOR_PALETTE['phase_b'],
         'Assess results\nPlan extension'),
        ('Phase C', 'Selective Extend\n(1.5-2.5 m)', COLOR_PALETTE['phase_c'],
         'Target areas only\nConfirm anomalies'),
        ('Phase D', 'Full Depth\n(2.5-3.0 m)', COLOR_PALETTE['phase_d'],
         'Complete survey\nFinal data set'),
    ]

    box_width = 2.2
    box_height = 2.5
    spacing = 0.5
    start_x = 0.8

    for i, (title, subtitle, color, description) in enumerate(phases):
        x = start_x + i * (box_width + spacing)
        y = 2.0

        # Main box
        ax.add_patch(FancyBboxPatch((x, y), box_width, box_height,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor=COLOR_PALETTE['gray_dark'],
                                     linewidth=2))

        # Phase title
        ax.text(x + box_width/2, y + box_height - 0.3, title, ha='center', va='top',
                fontsize=11, fontweight='bold', color=COLOR_PALETTE['primary'])

        # Subtitle
        ax.text(x + box_width/2, y + box_height - 0.7, subtitle, ha='center', va='top',
                fontsize=9, color=COLOR_PALETTE['gray_dark'])

        # Horizontal line
        ax.plot([x + 0.2, x + box_width - 0.2], [y + box_height - 1.3, y + box_height - 1.3],
                color=COLOR_PALETTE['gray_light'], linewidth=1)

        # Description
        ax.text(x + box_width/2, y + 0.8, description, ha='center', va='center',
                fontsize=8, color=COLOR_PALETTE['gray_dark'])

        # Arrow to next phase
        if i < len(phases) - 1:
            arrow_x = x + box_width + 0.1
            ax.annotate('', xy=(arrow_x + spacing - 0.2, y + box_height/2),
                       xytext=(arrow_x, y + box_height/2),
                       arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['accent'],
                                      lw=2, mutation_scale=15))

    # Decision points below
    ax.text(6, 1.2, 'Decision Points', fontsize=10, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    decisions = [
        (2.0, 'Anomalies detected?'),
        (4.7, 'Extend needed?'),
        (7.4, 'Target confirmed?'),
        (10.1, 'Survey complete?'),
    ]

    for x, text in decisions:
        # Diamond shape
        diamond_size = 0.3
        diamond = Polygon([(x, 0.7 + diamond_size), (x + diamond_size, 0.7),
                          (x, 0.7 - diamond_size), (x - diamond_size, 0.7)],
                         facecolor=COLOR_PALETTE['light_orange'],
                         edgecolor=COLOR_PALETTE['orange'], linewidth=1.5)
        ax.add_patch(diamond)
        ax.text(x, 0.2, text, ha='center', va='top', fontsize=7,
                color=COLOR_PALETTE['gray_dark'])

    # Legend
    legend_y = 5.3
    ax.add_patch(Rectangle((0.5, legend_y - 0.15), 0.3, 0.2,
                           facecolor=COLOR_PALETTE['phase_a'], edgecolor='black'))
    ax.text(0.9, legend_y, 'Field Work', fontsize=8, va='center')

    ax.add_patch(Rectangle((2.5, legend_y - 0.15), 0.3, 0.2,
                           facecolor=COLOR_PALETTE['phase_b'], edgecolor='black'))
    ax.text(2.9, legend_y, 'Analysis', fontsize=8, va='center')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf


def create_depth_extension_procedure():
    """
    Step-by-step diagram showing in-place depth extension procedure.
    Four panels showing the extension sequence.
    """
    fig, axes = plt.subplots(1, 4, figsize=(12, 5))

    steps = [
        ('1. Initial Position', 'Probe at 1.0 m\nCable slack coiled\nJunction box accessible'),
        ('2. Disconnect Top', 'Remove upper segment\nProtect M12 connector\nKeep lower in place'),
        ('3. Insert Extension', 'Add 0.5 m segment\nReconnect upper\nFeed slack cable'),
        ('4. Push to Depth', 'Press to new depth\nVerify connectivity\nLog new position'),
    ]

    for ax, (title, description) in zip(axes, steps):
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 5)
        ax.axis('off')

        # Title
        ax.text(1.5, 4.8, title, ha='center', fontsize=10, fontweight='bold',
                color=COLOR_PALETTE['primary'])

        # Ground surface
        ax.fill_between([0, 3], [3.5, 3.5], [0, 0],
                       color=COLOR_PALETTE['ground_tan'], alpha=0.5)
        ax.plot([0, 3], [3.5, 3.5], color=COLOR_PALETTE['ground_dark'], linewidth=2)
        ax.text(0.1, 3.6, 'Surface', fontsize=7, color=COLOR_PALETTE['gray_dark'])

        # Probe (simplified)
        probe_color = COLOR_PALETTE['secondary']
        ax.add_patch(Rectangle((1.3, 1.5), 0.4, 2.0, facecolor=probe_color,
                               edgecolor=COLOR_PALETTE['gray_dark'], linewidth=1.5))

        # Description box
        ax.add_patch(FancyBboxPatch((0.2, 0.2), 2.6, 1.1, boxstyle="round,pad=0.05",
                                    facecolor=COLOR_PALETTE['light_bg'],
                                    edgecolor=COLOR_PALETTE['gray_light']))
        ax.text(1.5, 0.75, description, ha='center', va='center', fontsize=7,
                color=COLOR_PALETTE['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf


def create_deployment_decision_tree():
    """
    Decision tree for choosing between standard and progressive deployment.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.7, 'Progressive Deployment Decision Tree', fontsize=12, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    def draw_decision(x, y, text, size=0.5):
        """Draw a diamond decision node."""
        diamond = Polygon([(x, y + size), (x + size*1.5, y), (x, y - size), (x - size*1.5, y)],
                         facecolor=COLOR_PALETTE['light_blue'],
                         edgecolor=COLOR_PALETTE['accent'], linewidth=1.5)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                color=COLOR_PALETTE['gray_dark'])

    def draw_terminal(x, y, text, color):
        """Draw a terminal/result box."""
        ax.add_patch(FancyBboxPatch((x - 1.2, y - 0.3), 2.4, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor=COLOR_PALETTE['gray_dark']))
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')

    # Start
    ax.add_patch(FancyBboxPatch((3.8, 7.0), 2.4, 0.5, boxstyle="round,pad=0.1",
                                facecolor=COLOR_PALETTE['light_green'],
                                edgecolor=COLOR_PALETTE['success'], linewidth=2))
    ax.text(5, 7.25, 'Start Planning', ha='center', va='center', fontsize=9, fontweight='bold')

    # Decision 1: Large area?
    draw_decision(5, 6.0, 'Large area?\n(>25 probes)')
    ax.annotate('', xy=(5, 6.5), xytext=(5, 6.95),
               arrowprops=dict(arrowstyle='->', color='black'))

    # No path - Standard deployment
    ax.annotate('', xy=(2.5, 6.0), xytext=(3.5, 6.0),
               arrowprops=dict(arrowstyle='->', color='black'))
    ax.text(3.0, 6.15, 'No', fontsize=7)
    draw_terminal(1.3, 6.0, 'Standard\nFixed-Depth', COLOR_PALETTE['light_orange'])

    # Yes path
    ax.annotate('', xy=(5, 5.0), xytext=(5, 5.5),
               arrowprops=dict(arrowstyle='->', color='black'))
    ax.text(5.15, 5.3, 'Yes', fontsize=7)

    # Decision 2: Depth uncertainty?
    draw_decision(5, 4.5, 'Depth\nuncertainty?')

    # No path
    ax.annotate('', xy=(2.5, 4.5), xytext=(3.5, 4.5),
               arrowprops=dict(arrowstyle='->', color='black'))
    ax.text(3.0, 4.65, 'No', fontsize=7)
    draw_terminal(1.3, 4.5, 'Standard\nFixed-Depth', COLOR_PALETTE['light_orange'])

    # Yes path
    ax.annotate('', xy=(5, 3.5), xytext=(5, 4.0),
               arrowprops=dict(arrowstyle='->', color='black'))
    ax.text(5.15, 3.8, 'Yes', fontsize=7)

    # Decision 3: UXO concerns?
    draw_decision(5, 3.0, 'UXO\nconcerns?')

    # Yes path - Progressive with clearance
    ax.annotate('', xy=(7.5, 3.0), xytext=(6.5, 3.0),
               arrowprops=dict(arrowstyle='->', color='black'))
    ax.text(7.0, 3.15, 'Yes', fontsize=7)
    draw_terminal(8.7, 3.0, 'Progressive +\nClearance Protocol', COLOR_PALETTE['light_red'])

    # No path
    ax.annotate('', xy=(5, 2.0), xytext=(5, 2.5),
               arrowprops=dict(arrowstyle='->', color='black'))
    ax.text(5.15, 2.3, 'No', fontsize=7)

    # Decision 4: Soil variability?
    draw_decision(5, 1.5, 'Soil\nvariability?')

    # Yes path
    ax.annotate('', xy=(7.5, 1.5), xytext=(6.5, 1.5),
               arrowprops=dict(arrowstyle='->', color='black'))
    ax.text(7.0, 1.65, 'Yes', fontsize=7)
    draw_terminal(8.7, 1.5, 'Progressive\nRecommended', COLOR_PALETTE['light_purple'])

    # No path
    ax.annotate('', xy=(2.5, 1.5), xytext=(3.5, 1.5),
               arrowprops=dict(arrowstyle='->', color='black'))
    ax.text(3.0, 1.65, 'No', fontsize=7)
    draw_terminal(1.3, 1.5, 'Consider\nCost/Benefit', COLOR_PALETTE['light_bg'])

    # Legend
    ax.add_patch(Rectangle((0.3, 0.2), 9.4, 0.6, facecolor='white',
                           edgecolor=COLOR_PALETTE['gray_light']))
    ax.add_patch(Polygon([(1.0, 0.5), (1.3, 0.35), (1.0, 0.2), (0.7, 0.35)],
                        facecolor=COLOR_PALETTE['light_blue'], edgecolor=COLOR_PALETTE['accent']))
    ax.text(1.6, 0.5, 'Decision', fontsize=7, va='center')

    ax.add_patch(FancyBboxPatch((2.5, 0.3), 0.6, 0.3, boxstyle="round,pad=0.02",
                                facecolor=COLOR_PALETTE['light_orange'], edgecolor='black'))
    ax.text(3.3, 0.5, 'Standard', fontsize=7, va='center')

    ax.add_patch(FancyBboxPatch((4.5, 0.3), 0.6, 0.3, boxstyle="round,pad=0.02",
                                facecolor=COLOR_PALETTE['light_purple'], edgecolor='black'))
    ax.text(5.3, 0.5, 'Progressive', fontsize=7, va='center')

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf


def create_depth_profile_comparison():
    """
    Cross-section showing progressive depth profile vs fixed depth.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (title, depths) in zip(axes, [
        ('Fixed-Depth Deployment', [3.0, 3.0, 3.0, 3.0, 3.0]),
        ('Progressive Deployment', [1.0, 2.5, 3.0, 2.0, 1.0])
    ]):
        ax.set_xlim(0, 6)
        ax.set_ylim(-3.5, 1)
        ax.set_aspect('equal')

        # Title
        ax.set_title(title, fontsize=11, fontweight='bold', color=COLOR_PALETTE['primary'])

        # Surface
        ax.fill_between([0, 6], [0, 0], [-3.5, -3.5],
                       color=COLOR_PALETTE['ground_tan'], alpha=0.4)
        ax.axhline(0, color=COLOR_PALETTE['ground_dark'], linewidth=2)
        ax.text(0.1, 0.2, 'Surface', fontsize=8)

        # Depth markers
        for d in [1, 2, 3]:
            ax.axhline(-d, color=COLOR_PALETTE['gray_light'], linewidth=0.5, linestyle='--')
            ax.text(5.8, -d + 0.1, f'{d}m', fontsize=7, ha='right',
                   color=COLOR_PALETTE['gray_med'])

        # Probes
        probe_x = [1, 2, 3, 4, 5]
        for i, (x, depth) in enumerate(zip(probe_x, depths)):
            # Probe body
            ax.add_patch(Rectangle((x - 0.1, -depth), 0.2, depth,
                                   facecolor=COLOR_PALETTE['secondary'],
                                   edgecolor=COLOR_PALETTE['gray_dark'], linewidth=1))
            # Top cap
            ax.add_patch(Circle((x, 0), 0.15, facecolor=COLOR_PALETTE['accent'],
                                edgecolor=COLOR_PALETTE['gray_dark']))

        # Anomaly zone (same in both)
        anomaly = mpatches.Ellipse((3, -2.2), 1.5, 0.8, angle=0,
                                   facecolor=COLOR_PALETTE['warning'], alpha=0.4,
                                   edgecolor=COLOR_PALETTE['warning'], linewidth=2,
                                   linestyle='--')
        ax.add_patch(anomaly)
        ax.text(3, -2.2, 'Target', ha='center', va='center', fontsize=8,
               color='white', fontweight='bold')

        ax.set_xlabel('Position (m)', fontsize=9)
        ax.set_ylabel('Depth (m)', fontsize=9)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_yticks([0, -1, -2, -3])
        ax.set_yticklabels(['0', '1', '2', '3'])

    # Add annotation for progressive
    axes[1].annotate('Deeper where\nneeded', xy=(3, -2.8), xytext=(4.5, -3.2),
                    fontsize=8, ha='center',
                    arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['accent']))
    axes[1].annotate('Shallow where\nsufficient', xy=(1, -0.8), xytext=(0.3, -1.5),
                    fontsize=8, ha='center',
                    arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['accent']))

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf


def create_uxo_clearance_protocol():
    """
    Diagram showing the two-pass magnetometer clearance protocol for UXO sites.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'UXO Clearance Protocol: Two-Pass Procedure', fontsize=12, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    # Pass 1 box
    ax.add_patch(FancyBboxPatch((0.5, 2.5), 4, 2.8, boxstyle="round,pad=0.05",
                                facecolor=COLOR_PALETTE['light_blue'],
                                edgecolor=COLOR_PALETTE['accent'], linewidth=2))
    ax.text(2.5, 5.0, 'Pass 1: Magnetometer Sweep', ha='center', fontsize=10,
            fontweight='bold', color=COLOR_PALETTE['primary'])

    pass1_steps = [
        '1. Insert probe to current depth',
        '2. Mag sweep at depth + 0.5m',
        '3. Check reading < threshold',
        '4. If clear, proceed to insert',
        '5. If anomaly, STOP and mark',
    ]
    for i, step in enumerate(pass1_steps):
        ax.text(0.7, 4.5 - i * 0.35, step, fontsize=8, color=COLOR_PALETTE['gray_dark'])

    # Arrow between
    ax.annotate('', xy=(5.3, 4.0), xytext=(4.7, 4.0),
               arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['success'],
                              lw=3, mutation_scale=20))
    ax.text(5, 4.3, 'Clear?', fontsize=9, ha='center', color=COLOR_PALETTE['success'])

    # Pass 2 box
    ax.add_patch(FancyBboxPatch((5.5, 2.5), 4, 2.8, boxstyle="round,pad=0.05",
                                facecolor=COLOR_PALETTE['light_green'],
                                edgecolor=COLOR_PALETTE['success'], linewidth=2))
    ax.text(7.5, 5.0, 'Pass 2: Probe Extension', ha='center', fontsize=10,
            fontweight='bold', color=COLOR_PALETTE['primary'])

    pass2_steps = [
        '1. Add extension segment',
        '2. Push to new depth',
        '3. Verify probe response',
        '4. Log actual depth achieved',
        '5. Repeat for next increment',
    ]
    for i, step in enumerate(pass2_steps):
        ax.text(5.7, 4.5 - i * 0.35, step, fontsize=8, color=COLOR_PALETTE['gray_dark'])

    # Threshold table
    ax.add_patch(FancyBboxPatch((0.5, 0.3), 9, 1.8, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=COLOR_PALETTE['gray_med']))
    ax.text(5, 1.9, 'Magnetometer Thresholds by Depth', ha='center', fontsize=9,
            fontweight='bold', color=COLOR_PALETTE['primary'])

    # Table content
    headers = ['Depth Range', 'Max Reading', 'Action']
    ax.text(1.5, 1.5, headers[0], fontsize=8, fontweight='bold', ha='center')
    ax.text(4.0, 1.5, headers[1], fontsize=8, fontweight='bold', ha='center')
    ax.text(7.5, 1.5, headers[2], fontsize=8, fontweight='bold', ha='center')

    rows = [
        ('0 - 1.0 m', '100 nT', 'Proceed with caution'),
        ('1.0 - 2.0 m', '50 nT', 'Pause and assess'),
        ('> 2.0 m', '25 nT', 'Stop, consult EOD'),
    ]
    for i, (depth, reading, action) in enumerate(rows):
        y = 1.1 - i * 0.3
        ax.text(1.5, y, depth, fontsize=8, ha='center', color=COLOR_PALETTE['gray_dark'])
        ax.text(4.0, y, reading, fontsize=8, ha='center', color=COLOR_PALETTE['gray_dark'])
        ax.text(7.5, y, action, fontsize=8, ha='center', color=COLOR_PALETTE['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf
