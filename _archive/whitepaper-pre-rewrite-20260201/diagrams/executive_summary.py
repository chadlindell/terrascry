"""
HIRT Whitepaper - Executive Summary Diagrams Module

Functions for creating system overview, cost comparison, capabilities infographic,
depth investigation, and workflow diagrams.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch
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
    'tx_coil': '#38a169',
    'rx_coil': '#3182ce',
    'ert_ring': '#ed8936',
    'probe_body': '#2c5282',
    'connector': '#4a5568',
    'light_blue': '#ebf8ff',
    'light_green': '#c6f6d5',
    'light_orange': '#feebc8',
    'light_red': '#fff5f5',
}


def create_system_overview():
    """
    Simplified system overview showing the dual-channel MIT + ERT approach.
    Shows probes in ground with both sensing methods highlighted.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # === Title Banner ===
    ax.add_patch(FancyBboxPatch((0.2, 5.3), 9.6, 0.6, boxstyle="round,pad=0.02",
                                facecolor=COLOR_PALETTE['primary'], edgecolor='none'))
    ax.text(5, 5.6, 'HIRT: Dual-Channel Subsurface Imaging System',
            fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    # === Ground Cross-Section (Left Panel) ===
    # Sky
    ax.fill_between([0.3, 4.7], [4.2, 4.2], [5.0, 5.0], color=COLOR_PALETTE['sky'], alpha=0.5)

    # Ground layers
    for d1, d2, alpha in [(4.2, 3.4, 0.3), (3.4, 2.6, 0.4), (2.6, 1.8, 0.5), (1.8, 0.5, 0.6)]:
        ax.fill_between([0.3, 4.7], [d1, d1], [d2, d2],
                       color=COLOR_PALETTE['ground_tan'], alpha=alpha)

    # Ground surface line
    ax.plot([0.3, 4.7], [4.2, 4.2], color=COLOR_PALETTE['ground_dark'], linewidth=2.5)
    ax.text(2.5, 4.35, 'Ground Surface', fontsize=8, ha='center', color=COLOR_PALETTE['ground_dark'])

    # Probes (3 probes)
    probe_x = [1.0, 2.5, 4.0]
    for px in probe_x:
        # Probe rod
        ax.add_patch(Rectangle((px-0.08, 1.0), 0.16, 3.2,
                               color=COLOR_PALETTE['probe_body'], ec='black', lw=1))
        # Junction box at top
        ax.add_patch(Rectangle((px-0.18, 4.1), 0.36, 0.25,
                               color=COLOR_PALETTE['connector'], ec='black', lw=1))

        # MIT coils (TX/RX)
        for depth, color in [(3.5, COLOR_PALETTE['tx_coil']), (2.5, COLOR_PALETTE['rx_coil'])]:
            ax.add_patch(Circle((px, depth), 0.15, color=color, ec='black', lw=1))

        # ERT rings
        for depth in [3.0, 2.0, 1.2]:
            ax.add_patch(Rectangle((px-0.15, depth-0.06), 0.3, 0.12,
                                   color=COLOR_PALETTE['ert_ring'], ec='black', lw=0.5))

    # Ray paths between probes (MIT)
    for i, p1 in enumerate(probe_x[:-1]):
        for p2 in probe_x[i+1:]:
            for d1, d2 in [(3.5, 3.5), (3.5, 2.5), (2.5, 2.5), (2.5, 3.5)]:
                ax.plot([p1, p2], [d1, d2], color=COLOR_PALETTE['success'],
                       alpha=0.4, linewidth=1.2, linestyle='-')

    # ERT current paths (dashed)
    ax.plot([1.0, 4.0], [3.0, 3.0], color=COLOR_PALETTE['orange'],
           alpha=0.6, linewidth=1.5, linestyle='--')
    ax.plot([1.0, 4.0], [2.0, 2.0], color=COLOR_PALETTE['orange'],
           alpha=0.6, linewidth=1.5, linestyle='--')

    # Target
    target = Circle((2.5, 2.2), 0.35, color=COLOR_PALETTE['warning'], ec='#9b2c2c', lw=2)
    ax.add_patch(target)
    ax.text(2.5, 2.2, 'Target', fontsize=7, ha='center', va='center',
            color='white', fontweight='bold')

    # Depth scale
    for d, label in [(4.2, '0m'), (3.2, '1m'), (2.2, '2m'), (1.2, '3m')]:
        ax.text(0.15, d, label, fontsize=7, va='center', color='gray')

    # Legend for left panel
    legend_y = 0.25
    ax.add_patch(Circle((0.6, legend_y+0.15), 0.1, color=COLOR_PALETTE['tx_coil']))
    ax.text(0.8, legend_y+0.15, 'TX Coil', fontsize=7, va='center')
    ax.add_patch(Circle((1.8, legend_y+0.15), 0.1, color=COLOR_PALETTE['rx_coil']))
    ax.text(2.0, legend_y+0.15, 'RX Coil', fontsize=7, va='center')
    ax.add_patch(Rectangle((2.85, legend_y+0.05), 0.2, 0.12, color=COLOR_PALETTE['ert_ring']))
    ax.text(3.15, legend_y+0.15, 'ERT Ring', fontsize=7, va='center')

    # === MIT-3D Channel Box (Top Right) ===
    ax.add_patch(FancyBboxPatch((5.2, 3.5), 2.2, 1.6, boxstyle="round,pad=0.05",
                                facecolor='#c6f6d5', edgecolor=COLOR_PALETTE['success'], linewidth=2))
    ax.text(6.3, 4.9, 'MIT-3D Channel', fontsize=10, fontweight='bold',
            ha='center', color=COLOR_PALETTE['success'])

    mit_items = [
        'Low-frequency TX/RX coils',
        '2-50 kHz operation',
        'Eddy current detection',
        'Maps metal (incl. aluminum)',
    ]
    for i, item in enumerate(mit_items):
        ax.text(5.35, 4.55 - i*0.25, f'\u2022 {item}', fontsize=7, va='center')

    # === ERT-Lite Channel Box (Bottom Right) ===
    ax.add_patch(FancyBboxPatch((5.2, 1.6), 2.2, 1.6, boxstyle="round,pad=0.05",
                                facecolor='#feebc8', edgecolor=COLOR_PALETTE['orange'], linewidth=2))
    ax.text(6.3, 3.0, 'ERT-Lite Channel', fontsize=10, fontweight='bold',
            ha='center', color='#c05621')

    ert_items = [
        'Ring electrodes on rods',
        '0.5-2 mA injection',
        'Voltage measurement',
        'Maps soil disturbance',
    ]
    for i, item in enumerate(ert_items):
        ax.text(5.35, 2.65 - i*0.25, f'\u2022 {item}', fontsize=7, va='center')

    # === Key Benefits Box (Far Right) ===
    ax.add_patch(FancyBboxPatch((7.6, 1.6), 2.2, 3.5, boxstyle="round,pad=0.05",
                                facecolor=COLOR_PALETTE['light_bg'], edgecolor=COLOR_PALETTE['primary'], linewidth=2))
    ax.text(8.7, 4.9, 'Key Benefits', fontsize=10, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    benefits = [
        'True 3D tomographic',
        'imaging capability',
        '',
        '2-5x better resolution',
        'than surface methods',
        '',
        'Non-ferrous metal',
        'detection (aluminum)',
        '',
        '~10x less ground',
        'disturbance',
    ]
    for i, item in enumerate(benefits):
        ax.text(7.75, 4.55 - i*0.24, item, fontsize=7, va='center')

    # Connecting arrows
    ax.annotate('', xy=(5.1, 4.3), xytext=(4.8, 3.5),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['success'], lw=1.5))
    ax.annotate('', xy=(5.1, 2.4), xytext=(4.8, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['orange'], lw=1.5))

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_cost_comparison():
    """
    Professional cost comparison chart showing HIRT vs commercial systems.
    Includes bar chart with dramatic cost difference visualization.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                   gridspec_kw={'width_ratios': [1.2, 1]})

    # === Left Panel: Cost Bar Chart ===
    systems = ['Commercial\nCrosshole\n(Low End)', 'Commercial\nCrosshole\n(High End)',
               'HIRT\nComplete Kit\n(High)', 'HIRT\nComplete Kit\n(Low)']
    costs = [50000, 200000, 3900, 1800]
    colors = ['#e53e3e', '#9b2c2c', '#38a169', '#276749']

    # Create horizontal bar chart
    y_pos = np.arange(len(systems))
    bars = ax1.barh(y_pos, costs, color=colors, edgecolor='black', linewidth=1)

    # Add value labels
    for bar, cost in zip(bars, costs):
        width = bar.get_width()
        if width > 10000:
            ax1.text(width - 2000, bar.get_y() + bar.get_height()/2,
                    f'${cost:,}', ha='right', va='center', fontsize=9,
                    fontweight='bold', color='white')
        else:
            ax1.text(width + 2000, bar.get_y() + bar.get_height()/2,
                    f'${cost:,}', ha='left', va='center', fontsize=9, fontweight='bold')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(systems, fontsize=9)
    ax1.set_xlabel('System Cost (USD)', fontsize=10)
    ax1.set_xlim(0, 230000)
    ax1.set_title('Complete System Cost Comparison', fontsize=11, fontweight='bold',
                  color=COLOR_PALETTE['primary'], pad=10)

    # Add savings annotation
    ax1.annotate('95%+ Savings', xy=(50000, 2.5), xytext=(120000, 2.5),
                fontsize=10, fontweight='bold', color=COLOR_PALETTE['success'],
                arrowprops=dict(arrowstyle='<->', color=COLOR_PALETTE['success'], lw=2),
                va='center')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.grid(True, linestyle=':', alpha=0.5)
    ax1.set_axisbelow(True)

    # === Right Panel: Cost Breakdown ===
    categories = ['Probes\n(20-24)', 'Electronics\nHub', 'Cables &\nConnectors', 'Tools &\nMisc']
    low_costs = [1400, 300, 50, 50]
    high_costs = [3600, 200, 50, 50]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax2.bar(x - width/2, low_costs, width, label='Low-End Kit ($1,800)',
                    color=COLOR_PALETTE['success'], edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x + width/2, high_costs, width, label='High-End Kit ($3,900)',
                    color='#276749', edgecolor='black', linewidth=1)

    ax2.set_ylabel('Cost (USD)', fontsize=10)
    ax2.set_title('HIRT Kit Cost Breakdown', fontsize=11, fontweight='bold',
                  color=COLOR_PALETTE['primary'], pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=8)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_ylim(0, 4500)

    # Add bar labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'${int(height)}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'${int(height)}', ha='center', va='bottom', fontsize=7)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax2.set_axisbelow(True)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_capabilities_infographic():
    """
    Infographic highlighting key HIRT capabilities with icons and metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # === Title ===
    ax.text(5, 4.7, 'HIRT Key Capabilities', fontsize=14, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    # === Capability Cards ===
    card_data = [
        {
            'x': 0.3, 'y': 2.8, 'width': 2.2, 'height': 1.6,
            'title': 'TRUE TOMOGRAPHY',
            'metric': '3D',
            'desc': 'Full volumetric\nimaging vs 2D maps',
            'color': COLOR_PALETTE['primary'],
        },
        {
            'x': 2.7, 'y': 2.8, 'width': 2.2, 'height': 1.6,
            'title': 'RESOLUTION',
            'metric': '2-5x',
            'desc': 'Better than surface\nmethods at depth',
            'color': COLOR_PALETTE['success'],
        },
        {
            'x': 5.1, 'y': 2.8, 'width': 2.2, 'height': 1.6,
            'title': 'DEPTH RANGE',
            'metric': '3-6m',
            'desc': 'Deep target\ndetection',
            'color': COLOR_PALETTE['secondary'],
        },
        {
            'x': 7.5, 'y': 2.8, 'width': 2.2, 'height': 1.6,
            'title': 'COST SAVINGS',
            'metric': '95%+',
            'desc': 'vs commercial\nsystems',
            'color': COLOR_PALETTE['warning'],
        },
    ]

    for card in card_data:
        # Card background
        ax.add_patch(FancyBboxPatch((card['x'], card['y']), card['width'], card['height'],
                                    boxstyle="round,pad=0.03", facecolor='white',
                                    edgecolor=card['color'], linewidth=2))

        # Title
        ax.text(card['x'] + card['width']/2, card['y'] + card['height'] - 0.2,
               card['title'], fontsize=8, fontweight='bold', ha='center',
               color=card['color'])

        # Large metric
        ax.text(card['x'] + card['width']/2, card['y'] + card['height']/2 + 0.1,
               card['metric'], fontsize=22, fontweight='bold', ha='center',
               color=card['color'])

        # Description
        ax.text(card['x'] + card['width']/2, card['y'] + 0.25,
               card['desc'], fontsize=7, ha='center', va='bottom',
               color=COLOR_PALETTE['gray_dark'])

    # === Advantages Row ===
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 2.2, boxstyle="round,pad=0.03",
                                facecolor=COLOR_PALETTE['light_bg'],
                                edgecolor=COLOR_PALETTE['gray_med'], linewidth=1))

    ax.text(5, 2.3, 'Crosshole Geometry Advantages', fontsize=11, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    advantages = [
        ('True tomographic coverage\nthrough target volume', 0.6),
        ('Superior depth discrimination\n(3m vs 4m distinguishable)', 2.4),
        ('Non-ferrous metal detection\n(aluminum, brass, copper)', 4.2),
        ('3D localization rather\nthan pseudo-depth maps', 6.0),
        ('Minimal site disturbance\n(16mm probe diameter)', 7.8),
    ]

    for text, x in advantages:
        # Checkmark circle
        ax.add_patch(Circle((x + 0.15, 1.6), 0.15, color=COLOR_PALETTE['success']))
        ax.text(x + 0.15, 1.6, '\u2713', fontsize=10, ha='center', va='center',
               color='white', fontweight='bold')
        # Text
        ax.text(x + 0.4, 1.5, text, fontsize=8, va='center',
               color=COLOR_PALETTE['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_depth_investigation():
    """
    Visual representation of depth investigation ranges for different configurations.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    # Data from executive summary
    configs = [
        ('1.5m probes, 2m spacing', 2, 3, 'HIGH', COLOR_PALETTE['success']),
        ('3.0m probes, 2m spacing', 3, 5, 'MEDIUM', COLOR_PALETTE['secondary']),
        ('Edge cases (conductive)', 4, 6, 'LOW', COLOR_PALETTE['orange']),
    ]

    y_pos = np.arange(len(configs))

    # Draw depth bars
    for i, (label, min_d, max_d, conf, color) in enumerate(configs):
        # Range bar
        ax.barh(i, max_d - min_d, left=min_d, height=0.5,
               color=color, alpha=0.6, edgecolor=color, linewidth=2)

        # Min/max markers
        ax.plot([min_d], [i], '|', color=color, markersize=20, markeredgewidth=3)
        ax.plot([max_d], [i], '|', color=color, markersize=20, markeredgewidth=3)

        # Confidence label
        ax.text(max_d + 0.2, i, f'{conf}\nconfidence', fontsize=8, va='center',
               color=color, fontweight='bold')

        # Depth range text
        ax.text((min_d + max_d)/2, i, f'{min_d}-{max_d}m', fontsize=10,
               ha='center', va='center', fontweight='bold', color='white')

    # Surface method comparison line
    ax.axvline(1.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(1.5, 2.7, 'Typical surface\nmethod limit', fontsize=7,
           ha='center', va='bottom', color='gray')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c[0] for c in configs], fontsize=9)
    ax.set_xlabel('Depth of Investigation (m)', fontsize=10)
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.5, 3)
    ax.set_title('HIRT Depth of Investigation by Configuration', fontsize=11,
                fontweight='bold', color=COLOR_PALETTE['primary'], pad=10)

    # Add grid
    ax.xaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_workflow_diagram():
    """
    Diagram showing the optimal two-stage workflow: surface screening + HIRT follow-up.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # === Stage 1: Surface Screening ===
    ax.add_patch(FancyBboxPatch((0.2, 1.5), 3.5, 2.2, boxstyle="round,pad=0.05",
                                facecolor='#ebf8ff', edgecolor=COLOR_PALETTE['secondary'], linewidth=2))
    ax.text(2, 3.5, 'STAGE 1: Surface Screening', fontsize=10, fontweight='bold',
            ha='center', color=COLOR_PALETTE['secondary'])

    # Methods
    methods = ['Magnetometry', 'GPR', 'EM31/CMD']
    for i, method in enumerate(methods):
        ax.add_patch(Rectangle((0.4, 2.8 - i*0.5), 1.5, 0.35,
                               color=COLOR_PALETTE['gray_light'], ec='gray', lw=1))
        ax.text(1.15, 2.97 - i*0.5, method, fontsize=8, ha='center', va='center')

    # Benefits
    ax.text(2.5, 2.95, '\u2713 Rapid coverage', fontsize=7, color=COLOR_PALETTE['success'])
    ax.text(2.5, 2.55, '\u2713 Large areas', fontsize=7, color=COLOR_PALETTE['success'])
    ax.text(2.5, 2.15, '\u2713 Initial screening', fontsize=7, color=COLOR_PALETTE['success'])

    # Limitations
    ax.text(0.4, 1.7, 'Limited depth | Ambiguous at >1.5m | 2D pseudo-depth only',
           fontsize=6, color=COLOR_PALETTE['gray_dark'], style='italic')

    # === Arrow ===
    ax.annotate('', xy=(4.5, 2.6), xytext=(3.8, 2.6),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['primary'], lw=3))
    ax.text(4.15, 2.9, 'Anomalies\nidentified', fontsize=7, ha='center',
           color=COLOR_PALETTE['primary'])

    # === Stage 2: HIRT Follow-up ===
    ax.add_patch(FancyBboxPatch((4.6, 1.5), 3.5, 2.2, boxstyle="round,pad=0.05",
                                facecolor='#c6f6d5', edgecolor=COLOR_PALETTE['success'], linewidth=2))
    ax.text(6.35, 3.5, 'STAGE 2: HIRT Crosshole', fontsize=10, fontweight='bold',
            ha='center', color=COLOR_PALETTE['success'])

    # Channels
    ax.add_patch(Rectangle((4.8, 2.55), 1.3, 0.35,
                           color='#38a169', ec='#276749', lw=1))
    ax.text(5.45, 2.72, 'MIT-3D', fontsize=8, ha='center', va='center', color='white')
    ax.add_patch(Rectangle((4.8, 2.1), 1.3, 0.35,
                           color=COLOR_PALETTE['orange'], ec='#c05621', lw=1))
    ax.text(5.45, 2.27, 'ERT-Lite', fontsize=8, ha='center', va='center', color='white')

    # Benefits
    ax.text(6.3, 2.95, '\u2713 True 3D imaging', fontsize=7, color=COLOR_PALETTE['success'])
    ax.text(6.3, 2.55, '\u2713 3-6m depth', fontsize=7, color=COLOR_PALETTE['success'])
    ax.text(6.3, 2.15, '\u2713 Target characterization', fontsize=7, color=COLOR_PALETTE['success'])

    ax.text(4.8, 1.7, 'High-resolution | Aluminum detection | Soil disturbance',
           fontsize=6, color=COLOR_PALETTE['gray_dark'], style='italic')

    # === Result Box ===
    ax.annotate('', xy=(8.8, 2.6), xytext=(8.2, 2.6),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['primary'], lw=3))

    ax.add_patch(FancyBboxPatch((8.3, 1.5), 1.5, 2.2, boxstyle="round,pad=0.05",
                                facecolor=COLOR_PALETTE['primary'],
                                edgecolor=COLOR_PALETTE['primary'], linewidth=2))
    ax.text(9.05, 3.5, 'RESULT', fontsize=10, fontweight='bold', ha='center', color='white')
    ax.text(9.05, 2.7, '3D Model', fontsize=9, ha='center', color='white', fontweight='bold')
    ax.text(9.05, 2.3, 'Target', fontsize=8, ha='center', color='white')
    ax.text(9.05, 2.0, 'Location', fontsize=8, ha='center', color='white')
    ax.text(9.05, 1.7, '& Character', fontsize=8, ha='center', color='white')

    # === Bottom note ===
    ax.text(5, 0.9, 'Optimal Workflow: Leverage surface speed for screening, HIRT precision for characterization',
           fontsize=9, ha='center', color=COLOR_PALETTE['gray_dark'], style='italic')
    ax.text(5, 0.5, 'This two-stage approach minimizes deployment time while maximizing detection accuracy',
           fontsize=8, ha='center', color=COLOR_PALETTE['gray_med'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf
