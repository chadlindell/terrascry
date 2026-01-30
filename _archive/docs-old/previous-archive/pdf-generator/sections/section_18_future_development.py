#!/usr/bin/env python3
"""
HIRT Section 18: Future Development - PDF Generator

Generates a publication-quality PDF covering HIRT roadmap, planned features,
and research directions with professional diagrams.

Diagrams:
1. Development roadmap timeline
2. Feature priority matrix
3. Technology evolution diagram

Usage:
    python section_18_future_development.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch
import numpy as np
from io import BytesIO

from lib.pdf_builder import SectionPDFBuilder
from lib.styles import (
    CONTENT_WIDTH, PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING,
    LIGHT_BG, COLOR_PALETTE
)

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
    'light_yellow': '#fefcbf',
}


def create_development_roadmap_timeline():
    """
    Create development roadmap timeline showing software and hardware phases.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(6, 6.7, 'HIRT Development Roadmap', ha='center',
           fontsize=14, fontweight='bold', color=COLORS['primary'])

    # Timeline axis
    ax.axhline(y=3.5, xmin=0.08, xmax=0.92, color=COLORS['gray_dark'], linewidth=2)

    # Month markers
    months = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']
    for i, month in enumerate(months):
        x = 1 + i * 0.9
        ax.plot([x, x], [3.4, 3.6], color=COLORS['gray_dark'], linewidth=2)
        ax.text(x, 3.2, month, ha='center', fontsize=7, color=COLORS['gray_med'])

    # Phase 1: Basic Processing (Months 1-3) - Software
    phase1_box = FancyBboxPatch((0.8, 4.2), 2.7, 1.8, boxstyle="round,pad=0.03",
                                 facecolor=COLORS['light_blue'], edgecolor=COLORS['accent'],
                                 linewidth=2)
    ax.add_patch(phase1_box)
    ax.text(2.15, 5.7, 'Phase 1: Basic Processing', ha='center', fontsize=9,
           fontweight='bold', color=COLORS['accent'])
    ax.text(2.15, 5.3, 'Months 1-3', ha='center', fontsize=7, color=COLORS['gray_dark'])
    phase1_items = ['Data import/export', 'Basic QC tools', 'Simple visualization']
    for i, item in enumerate(phase1_items):
        ax.text(1.0, 5.0 - i * 0.3, f'\u2022 {item}', fontsize=7, va='center')

    # Phase 2: Inversion (Months 4-6) - Software
    phase2_box = FancyBboxPatch((3.7, 4.2), 2.7, 1.8, boxstyle="round,pad=0.03",
                                 facecolor=COLORS['light_green'], edgecolor=COLORS['success'],
                                 linewidth=2)
    ax.add_patch(phase2_box)
    ax.text(5.05, 5.7, 'Phase 2: Inversion', ha='center', fontsize=9,
           fontweight='bold', color=COLORS['success'])
    ax.text(5.05, 5.3, 'Months 4-6', ha='center', fontsize=7, color=COLORS['gray_dark'])
    phase2_items = ['Forward modeling', 'Inversion algorithms', 'Basic data fusion']
    for i, item in enumerate(phase2_items):
        ax.text(3.9, 5.0 - i * 0.3, f'\u2022 {item}', fontsize=7, va='center')

    # Phase 3: Advanced Features (Months 7-12) - Software
    phase3_box = FancyBboxPatch((6.6, 4.2), 4.5, 1.8, boxstyle="round,pad=0.03",
                                 facecolor=COLORS['light_purple'], edgecolor=COLORS['purple'],
                                 linewidth=2)
    ax.add_patch(phase3_box)
    ax.text(8.85, 5.7, 'Phase 3: Advanced Features', ha='center', fontsize=9,
           fontweight='bold', color=COLORS['purple'])
    ax.text(8.85, 5.3, 'Months 7-12', ha='center', fontsize=7, color=COLORS['gray_dark'])
    phase3_items = ['Advanced 3D visualization', 'Machine learning integration',
                   'User interface development', 'Complete documentation']
    for i, item in enumerate(phase3_items):
        ax.text(6.8, 5.0 - i * 0.25, f'\u2022 {item}', fontsize=7, va='center')

    # Hardware track label
    ax.text(0.3, 1.8, 'HARDWARE\nTRACK', ha='center', fontsize=8, fontweight='bold',
           color=COLORS['orange'], va='center')

    # Hardware improvements timeline (below main timeline)
    hw_phases = [
        (1.5, 'Current:\n16mm Modular', COLORS['light_green'], COLORS['success']),
        (4, 'Wireless\nProbes', COLORS['light_orange'], COLORS['orange']),
        (6.5, 'Higher\nChannel Count', COLORS['light_orange'], COLORS['orange']),
        (9, 'Real-time\nProcessing', COLORS['light_blue'], COLORS['accent']),
    ]

    for x, label, facecolor, edgecolor in hw_phases:
        hw_box = FancyBboxPatch((x - 1, 0.8), 2, 1.2, boxstyle="round,pad=0.02",
                                facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
        ax.add_patch(hw_box)
        ax.text(x, 1.4, label, ha='center', va='center', fontsize=7, fontweight='bold')
        # Connect to timeline
        ax.annotate('', xy=(x, 3.4), xytext=(x, 2.0),
                   arrowprops=dict(arrowstyle='->', color=edgecolor, lw=1.2))

    # Arrows between software phases
    ax.annotate('', xy=(3.7, 5.1), xytext=(3.5, 5.1),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.5))
    ax.annotate('', xy=(6.6, 5.1), xytext=(6.4, 5.1),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=1.5))

    # Software track label
    ax.text(0.3, 5.1, 'SOFTWARE\nTRACK', ha='center', fontsize=8, fontweight='bold',
           color=COLORS['accent'], va='center')

    # Milestone markers
    milestones = [
        (2.8, 'v0.1\nAlpha'),
        (5.5, 'v0.5\nBeta'),
        (10.9, 'v1.0\nRelease'),
    ]
    for x, label in milestones:
        ax.plot(x, 3.5, 'D', markersize=10, color=COLORS['warning'])
        ax.text(x, 3.0, label, ha='center', fontsize=7, fontweight='bold',
               color=COLORS['warning'])

    # Legend
    legend_y = 0.3
    ax.add_patch(Rectangle((7.5, legend_y - 0.1), 0.3, 0.2, facecolor=COLORS['light_blue'],
                           edgecolor=COLORS['accent']))
    ax.text(8.0, legend_y, 'Software Phase', fontsize=7, va='center')
    ax.add_patch(Rectangle((9.5, legend_y - 0.1), 0.3, 0.2, facecolor=COLORS['light_orange'],
                           edgecolor=COLORS['orange']))
    ax.text(10.0, legend_y, 'Hardware Phase', fontsize=7, va='center')
    ax.plot(11.2, legend_y, 'D', markersize=6, color=COLORS['warning'])
    ax.text(11.5, legend_y, 'Milestone', fontsize=7, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_feature_priority_matrix():
    """
    Create feature priority matrix showing impact vs effort for planned features.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Background quadrants
    ax.fill_between([0, 5], [5, 5], [10, 10], color=COLORS['light_green'], alpha=0.3)  # Quick Wins
    ax.fill_between([5, 10], [5, 5], [10, 10], color=COLORS['light_orange'], alpha=0.3)  # Major Projects
    ax.fill_between([0, 5], [0, 0], [5, 5], color=COLORS['light_blue'], alpha=0.3)  # Fill-ins
    ax.fill_between([5, 10], [0, 0], [5, 5], color=COLORS['light_red'], alpha=0.3)  # Avoid

    # Quadrant labels
    ax.text(2.5, 9.3, 'QUICK WINS', ha='center', fontsize=11, fontweight='bold',
           color=COLORS['success'])
    ax.text(2.5, 8.9, 'High Impact, Low Effort', ha='center', fontsize=8,
           color=COLORS['gray_dark'], style='italic')

    ax.text(7.5, 9.3, 'MAJOR PROJECTS', ha='center', fontsize=11, fontweight='bold',
           color=COLORS['orange'])
    ax.text(7.5, 8.9, 'High Impact, High Effort', ha='center', fontsize=8,
           color=COLORS['gray_dark'], style='italic')

    ax.text(2.5, 0.7, 'FILL-INS', ha='center', fontsize=11, fontweight='bold',
           color=COLORS['accent'])
    ax.text(2.5, 0.3, 'Low Impact, Low Effort', ha='center', fontsize=8,
           color=COLORS['gray_dark'], style='italic')

    ax.text(7.5, 0.7, 'AVOID', ha='center', fontsize=11, fontweight='bold',
           color=COLORS['warning'])
    ax.text(7.5, 0.3, 'Low Impact, High Effort', ha='center', fontsize=8,
           color=COLORS['gray_dark'], style='italic')

    # Center lines
    ax.axhline(y=5, color=COLORS['gray_dark'], linewidth=1.5, linestyle='--', alpha=0.5)
    ax.axvline(x=5, color=COLORS['gray_dark'], linewidth=1.5, linestyle='--', alpha=0.5)

    # Features as bubbles (x=effort, y=impact, size=priority)
    features = [
        # (effort, impact, size, label, color)
        # Quick Wins (low effort, high impact)
        (1.5, 8.5, 400, 'Data QA/QC', COLORS['success']),
        (3, 7.5, 350, 'CSV Import', COLORS['success']),
        (2, 6.5, 300, 'Basic Viz', COLORS['success']),
        (4, 8, 280, 'Reciprocity\nCheck', COLORS['success']),

        # Major Projects (high effort, high impact)
        (7, 8.5, 500, 'MIT\nInversion', COLORS['orange']),
        (8.5, 7.5, 480, 'ERT\nInversion', COLORS['orange']),
        (6.5, 6.5, 400, 'Data Fusion', COLORS['orange']),
        (8, 5.8, 350, '3D\nVisualization', COLORS['orange']),

        # Fill-ins (low effort, low impact)
        (2, 3.5, 200, 'GIS Export', COLORS['accent']),
        (3.5, 2.5, 180, 'File Format\nSupport', COLORS['accent']),
        (1.5, 1.8, 150, 'Config\nTemplates', COLORS['accent']),

        # Avoid/Defer (high effort, low impact)
        (7, 2.5, 220, 'Automated\nDeployment', COLORS['warning']),
        (8.5, 3.5, 200, 'Real-time\nInversion', COLORS['warning']),
    ]

    for effort, impact, size, label, color in features:
        ax.scatter(effort, impact, s=size, c=color, alpha=0.7, edgecolors='black', linewidth=1)
        ax.text(effort, impact, label, ha='center', va='center', fontsize=6,
               fontweight='bold', color='white',
               path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

    # Axes labels
    ax.set_xlabel('Implementation Effort', fontsize=11, fontweight='bold',
                 color=COLORS['primary'])
    ax.set_ylabel('User Impact', fontsize=11, fontweight='bold',
                 color=COLORS['primary'])

    # Custom tick labels
    ax.set_xticks([0, 2.5, 5, 7.5, 10])
    ax.set_xticklabels(['', 'Low', '', 'High', ''], fontsize=9)
    ax.set_yticks([0, 2.5, 5, 7.5, 10])
    ax.set_yticklabels(['', 'Low', '', 'High', ''], fontsize=9)

    # Title
    ax.set_title('Feature Priority Matrix\n', fontsize=14, fontweight='bold',
                color=COLORS['primary'])

    # Size legend
    ax.text(9.5, 4.3, 'Bubble size =\nPriority', ha='center', fontsize=7,
           color=COLORS['gray_dark'], style='italic')

    # Grid
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)

    # Border
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['gray_med'])
        spine.set_linewidth(1.5)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_technology_evolution_diagram():
    """
    Create technology evolution diagram showing HIRT design progression.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    # Title
    ax.text(5.5, 7.2, 'HIRT Technology Evolution', ha='center',
           fontsize=14, fontweight='bold', color=COLORS['primary'])

    # Evolution arrow (main timeline)
    arrow = FancyArrowPatch((0.5, 3.5), (10.5, 3.5),
                            arrowstyle='->', mutation_scale=25,
                            color=COLORS['gray_med'], linewidth=3)
    ax.add_patch(arrow)
    ax.text(5.5, 3.2, 'Evolution', ha='center', fontsize=9, style='italic',
           color=COLORS['gray_dark'])

    # Version boxes
    versions = [
        {
            'x': 1.5, 'y': 5.2, 'version': 'v1.0', 'name': 'Original Design',
            'status': 'DEPRECATED', 'status_color': COLORS['warning'],
            'specs': ['25mm OD rod', '50mm insertion hole', 'Active electronics\nin probe head',
                     'Large, heavy design'],
            'color': COLORS['light_red'], 'edge': COLORS['warning']
        },
        {
            'x': 5.5, 'y': 5.2, 'version': 'v2.0', 'name': 'Micro-Probe',
            'status': 'TRANSITIONAL', 'status_color': COLORS['orange'],
            'specs': ['12mm OD rod', 'Passive probes', 'Electronics at surface',
                     '~10x less intrusion'],
            'color': COLORS['light_orange'], 'edge': COLORS['orange']
        },
        {
            'x': 9.5, 'y': 5.2, 'version': 'v3.0', 'name': '16mm Modular',
            'status': 'CURRENT', 'status_color': COLORS['success'],
            'specs': ['16mm OD (increased\nfor strength)', 'Modular connectors',
                     'M12x1.75 threads', 'Flush-mount design'],
            'color': COLORS['light_green'], 'edge': COLORS['success']
        },
    ]

    for v in versions:
        # Main box
        box = FancyBboxPatch((v['x'] - 1.3, v['y'] - 1.6), 2.6, 2.8,
                             boxstyle="round,pad=0.03",
                             facecolor=v['color'], edgecolor=v['edge'], linewidth=2)
        ax.add_patch(box)

        # Version and name
        ax.text(v['x'], v['y'] + 0.9, v['version'], ha='center', fontsize=11,
               fontweight='bold', color=COLORS['primary'])
        ax.text(v['x'], v['y'] + 0.55, v['name'], ha='center', fontsize=9,
               fontweight='bold', color=COLORS['gray_dark'])

        # Status badge
        badge = FancyBboxPatch((v['x'] - 0.6, v['y'] + 0.1), 1.2, 0.3,
                               boxstyle="round,pad=0.02",
                               facecolor=v['status_color'], edgecolor='none')
        ax.add_patch(badge)
        ax.text(v['x'], v['y'] + 0.25, v['status'], ha='center', fontsize=7,
               fontweight='bold', color='white')

        # Specs
        for i, spec in enumerate(v['specs']):
            ax.text(v['x'], v['y'] - 0.25 - i * 0.35, f'\u2022 {spec}',
                   ha='center', fontsize=7, va='top')

        # Connect to timeline
        ax.plot([v['x'], v['x']], [v['y'] - 1.6, 3.5], '--',
               color=v['edge'], linewidth=1.5, alpha=0.6)
        ax.plot(v['x'], 3.5, 'o', markersize=10, color=v['edge'])

    # Evolution arrows between versions
    ax.annotate('', xy=(4, 5.2), xytext=(3, 5.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=2))
    ax.annotate('', xy=(8, 5.2), xytext=(7, 5.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=2))

    # Future directions box
    future_box = FancyBboxPatch((1, 0.3), 9, 1.8, boxstyle="round,pad=0.03",
                                 facecolor=COLORS['light_blue'], edgecolor=COLORS['accent'],
                                 linewidth=2, linestyle='--')
    ax.add_patch(future_box)
    ax.text(5.5, 1.85, 'Future Directions', ha='center', fontsize=10,
           fontweight='bold', color=COLORS['accent'])

    future_items = [
        ('Wireless Probes', 'LoRa/BLE communication'),
        ('Higher Channels', 'Modular expansion'),
        ('Real-time Processing', 'Field computing'),
        ('Automated Deployment', 'Robotic insertion'),
    ]

    for i, (title, desc) in enumerate(future_items):
        x = 1.8 + i * 2.3
        ax.text(x, 1.35, title, ha='center', fontsize=8, fontweight='bold',
               color=COLORS['secondary'])
        ax.text(x, 1.0, desc, ha='center', fontsize=7, color=COLORS['gray_dark'])

    # Key improvements callouts
    improvements = [
        (3.5, 6.3, 'Reduced\nIntrusion'),
        (7.5, 6.3, 'Improved\nStrength'),
    ]
    for x, y, text in improvements:
        ax.annotate(text, xy=(x, 5.5), xytext=(x, y),
                   fontsize=7, ha='center', va='bottom',
                   color=COLORS['success'], fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1))

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_software_architecture_diagram():
    """
    Create software architecture diagram showing processing pipeline layers.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    # Title
    ax.text(5, 7.2, 'Software Architecture Overview', ha='center',
           fontsize=13, fontweight='bold', color=COLORS['primary'])

    # Layer definitions (bottom to top)
    layers = [
        {
            'y': 0.8, 'height': 1.0, 'label': 'Data Layer',
            'color': COLORS['light_blue'], 'edge': COLORS['accent'],
            'items': ['CSV/HDF5 Import', 'Data Validation', 'File Management']
        },
        {
            'y': 2.0, 'height': 1.0, 'label': 'Processing Layer',
            'color': COLORS['light_green'], 'edge': COLORS['success'],
            'items': ['QA/QC', 'Filtering', 'Background Removal']
        },
        {
            'y': 3.2, 'height': 1.0, 'label': 'Inversion Layer',
            'color': COLORS['light_orange'], 'edge': COLORS['orange'],
            'items': ['MIT Inversion', 'ERT Inversion', 'Data Fusion']
        },
        {
            'y': 4.4, 'height': 1.0, 'label': 'Visualization Layer',
            'color': COLORS['light_purple'], 'edge': COLORS['purple'],
            'items': ['3D Rendering', 'Depth Slices', 'GIS Export']
        },
        {
            'y': 5.6, 'height': 1.0, 'label': 'User Interface',
            'color': COLORS['light_yellow'], 'edge': COLORS['orange'],
            'items': ['GUI Application', 'CLI Tools', 'API']
        },
    ]

    for layer in layers:
        # Layer box
        box = FancyBboxPatch((0.5, layer['y']), 9, layer['height'],
                             boxstyle="round,pad=0.02",
                             facecolor=layer['color'], edgecolor=layer['edge'],
                             linewidth=2)
        ax.add_patch(box)

        # Layer label
        ax.text(1.5, layer['y'] + layer['height']/2, layer['label'],
               ha='center', va='center', fontsize=10, fontweight='bold',
               color=layer['edge'])

        # Layer items
        for i, item in enumerate(layer['items']):
            x = 4 + i * 2
            item_box = FancyBboxPatch((x - 0.8, layer['y'] + 0.15), 1.6, 0.7,
                                      boxstyle="round,pad=0.02",
                                      facecolor='white', edgecolor=layer['edge'],
                                      linewidth=1)
            ax.add_patch(item_box)
            ax.text(x, layer['y'] + 0.5, item, ha='center', va='center',
                   fontsize=7, fontweight='bold')

    # Vertical connection arrows
    for i in range(len(layers) - 1):
        ax.annotate('', xy=(5, layers[i+1]['y']), xytext=(5, layers[i]['y'] + layers[i]['height']),
                   arrowprops=dict(arrowstyle='<->', color=COLORS['gray_dark'], lw=1.5))

    # External tools sidebar
    tools_box = FancyBboxPatch((0.2, 0.3), 2.2, 0.4, boxstyle="round,pad=0.02",
                               facecolor='white', edgecolor=COLORS['gray_med'], linewidth=1)
    ax.add_patch(tools_box)
    ax.text(1.3, 0.5, 'NumPy | SciPy | PyVista', ha='center', va='center',
           fontsize=7, color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def build_section_18_pdf():
    """
    Build the complete Section 18: Future Development PDF.

    Returns:
        Path to the generated PDF
    """
    builder = SectionPDFBuilder(
        section_num=18,
        title="Future Development"
    )

    # Title block
    builder.add_title_block(
        subtitle="HIRT Roadmap, Planned Features, and Research Directions",
        version="2.0"
    )

    # Overview
    builder.add_section_header("18.1 Overview")
    builder.add_body_text(
        "This document has focused on <b>hardware and field methods</b>. The next phase "
        "involves <b>software development</b> for data processing, inversion, and "
        "visualization, as well as continued hardware improvements.",
        first_paragraph=True
    )
    builder.add_body_text(
        "The software development is <b>explicitly separate</b> from the hardware/field "
        "guide presented in this document. The hardware system is designed to collect "
        "high-quality data that can be processed with standard or custom software tools."
    )

    # Development Roadmap
    builder.add_section_header("18.2 Development Roadmap")
    builder.add_body_text(
        "The HIRT project follows a phased development approach, with parallel tracks "
        "for software and hardware improvements. The timeline below illustrates the "
        "planned progression from basic data processing to advanced visualization and "
        "machine learning capabilities.",
        first_paragraph=True
    )

    # Figure 1: Development Roadmap Timeline
    roadmap_fig = create_development_roadmap_timeline()
    builder.add_figure(
        roadmap_fig,
        "Development roadmap showing software phases (top track) and hardware "
        "improvements (bottom track). Key milestones include Alpha (v0.1) at month 3, "
        "Beta (v0.5) at month 6, and full Release (v1.0) at month 12."
    )

    # Software Development Pipeline
    builder.add_section_header("18.3 Software Development Pipeline")
    builder.add_section_header("18.3.1 Data QA/QC (Quality Assurance/Quality Control)", level=2)
    builder.add_body_text(
        "The first stage of the processing pipeline involves comprehensive data "
        "validation and quality control:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "Import CSV data files with automatic format detection",
        "Check for missing or corrupted data entries",
        "Verify reciprocity (A to B approximately equals B to A)",
        "Remove outliers and flag suspicious measurements for review",
        "Generate data quality metrics and visualization reports"
    ])

    builder.add_section_header("18.3.2 MIT Inversion", level=2)
    builder.add_body_text(
        "Magnetic Induction Tomography inversion converts amplitude and phase "
        "measurements to conductivity distributions:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "Forward modeling: Predict measurements from conductivity model",
        "Inverse modeling: Reconstruct conductivity from measurements",
        "Multi-frequency handling for depth resolution",
        "Regularization for stable, physically meaningful solutions",
        "Uncertainty quantification for reliability assessment"
    ])

    builder.add_section_header("18.3.3 ERT Inversion", level=2)
    builder.add_body_text(
        "Electrical Resistivity Tomography inversion converts voltage/current "
        "measurements to resistivity volumes:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "DC resistivity forward solver (finite element/difference)",
        "Inversion algorithm with Gauss-Newton or Occam's method",
        "Electrode position and depth accounting",
        "Topography handling for non-flat surfaces",
        "Cross-borehole measurement geometry support"
    ])

    builder.add_section_header("18.3.4 Data Fusion", level=2)
    builder.add_body_text(
        "Combining MIT and ERT data provides complementary information:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "Co-registration of MIT and ERT volumes",
        "Unified 3D model generation",
        "Overlay capabilities with GPR, magnetometry, photogrammetry data",
        "Multi-parameter visualization and interpretation"
    ])

    # Feature Priority Matrix
    builder.add_section_header("18.4 Feature Priority Matrix")
    builder.add_body_text(
        "Features are prioritized based on user impact versus implementation effort. "
        "Quick wins (high impact, low effort) receive immediate attention, while major "
        "projects are scheduled according to available resources and dependencies.",
        first_paragraph=True
    )

    # Figure 2: Feature Priority Matrix
    priority_fig = create_feature_priority_matrix()
    builder.add_figure(
        priority_fig,
        "Feature priority matrix categorizing development items by implementation effort "
        "(x-axis) and user impact (y-axis). Bubble size indicates relative priority. "
        "Quick wins include data QA/QC and basic visualization; major projects include "
        "MIT/ERT inversion and 3D visualization."
    )

    # Software Architecture
    builder.add_section_header("18.5 Software Architecture")
    builder.add_section_header("18.5.1 Language and Library Considerations", level=2)
    builder.add_body_text(
        "The software stack is designed for scientific computing efficiency and "
        "accessibility:",
        first_paragraph=True
    )

    # Language options table
    lang_data = [
        ['Language', 'Strengths', 'Considerations'],
        ['Python', 'Scientific libraries, community support', 'Primary development language'],
        ['MATLAB', 'Powerful, well-documented', 'Requires license'],
        ['C++', 'High performance', 'Core algorithm optimization'],
        ['Hybrid', 'Best of both worlds', 'Python high-level, C++ intensive computation']
    ]
    builder.add_table(lang_data, caption="Language options for software development")

    builder.add_section_header("18.5.2 Key Libraries and Tools", level=2)
    tools_data = [
        ['Library/Tool', 'Purpose', 'Notes'],
        ['NumPy/SciPy', 'Numerical computing', 'Foundation for all calculations'],
        ['VTK/ParaView', '3D visualization', 'Professional rendering'],
        ['PyVista', 'Python 3D visualization', 'VTK wrapper for ease of use'],
        ['ResIPy/pyGIMLi', 'ERT inversion', 'Existing mature tools'],
        ['SimPEG', 'EM modeling and inversion', 'Comprehensive framework']
    ]
    builder.add_table(tools_data, caption="Recommended libraries and tools for HIRT software")

    # Technology Evolution
    builder.add_section_header("18.6 Hardware Evolution")
    builder.add_body_text(
        "The HIRT probe design has evolved through three major versions, each addressing "
        "lessons learned from field testing and user feedback. The current v3.0 16mm "
        "Modular design represents a balance between minimal site disturbance and "
        "mechanical durability.",
        first_paragraph=True
    )

    # Figure 3: Technology Evolution
    evolution_fig = create_technology_evolution_diagram()
    builder.add_figure(
        evolution_fig,
        "Technology evolution from v1.0 (25mm active probe) through v2.0 (12mm passive) "
        "to the current v3.0 (16mm modular). Key improvements include reduced intrusion, "
        "increased strength, and modular field serviceability. Future directions include "
        "wireless communication and real-time processing."
    )

    # Hardware Improvements
    builder.add_section_header("18.7 Planned Hardware Improvements")
    builder.add_section_header("18.7.1 Wireless Probes", level=2)
    builder.add_body_text(
        "Wireless communication would significantly reduce cable management complexity "
        "and enable faster deployment:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "LoRa or BLE communication protocols under evaluation",
        "Balance power consumption versus convenience",
        "Maintain data integrity and timing synchronization",
        "Consider field-replaceable battery modules"
    ])

    builder.add_section_header("18.7.2 Higher Channel Count", level=2)
    builder.add_bullet_list([
        "Increased probe density for finer resolution",
        "Modular expansion through daisy-chaining",
        "Enhanced multiplexing architecture",
        "Scalable from 4 to 16+ probes"
    ])

    builder.add_section_header("18.7.3 Real-time Processing", level=2)
    builder.add_bullet_list([
        "Process data during collection",
        "Enable adaptive survey strategies",
        "Immediate quality feedback in the field",
        "Requires edge computing capability (e.g., Raspberry Pi, Jetson)"
    ])

    # Forward Modeling and Validation
    builder.add_section_header("18.8 Forward Modeling and Validation")
    builder.add_body_text(
        "Before field deployment, the HIRT system response should be validated using "
        "synthetic models. Recommended tools include:",
        first_paragraph=True
    )

    tools_table = [
        ['Tool', 'Focus', 'Strengths', 'URL'],
        ['SimPEG', 'MIT + ERT', 'Full forward/inverse modeling, Python', 'simpeg.xyz'],
        ['pyGIMLi', 'ERT-focused', 'Excellent visualization, mature', 'pygimli.org'],
        ['Empymod', '1D layered EM', 'Fast, accurate for stratified media', 'empymod.emsig.xyz']
    ]
    builder.add_table(tools_table, caption="Recommended forward modeling tools")

    builder.add_section_header("18.8.1 Standard Validation Scenarios", level=2)
    builder.add_body_text(
        "Four standard scenarios are recommended for system validation:",
        first_paragraph=True
    )

    builder.add_numbered_list([
        "<b>Aluminum Bomb in Sandy Loam:</b> 1m diameter sphere at 3m depth in 0.1 S/m soil - validates MIT response to non-ferrous UXO",
        "<b>Grave Shaft (Disturbed Fill):</b> 0.8x0.5x1.5m prism with 0.05 S/m fill in 0.2 S/m clay - validates ERT contrast detection",
        "<b>Scattered Aircraft Debris:</b> Multiple fragments (0.2-1m) at 1-4m depths - validates MIT multi-target discrimination",
        "<b>Bomb Crater with Heterogeneous Fill:</b> 8m diameter, 5m deep crater with variable conductivity - tests combined MIT+ERT response"
    ])

    # Known Limitations
    builder.add_section_header("18.9 Known Limitations")
    builder.add_info_box("Technical Limitations", [
        "Smaller coil area results in ~19 dB SNR loss (compensated by longer integration)",
        "Survey time increases 5-10x compared to commercial systems",
        "Post-processing software required for 3D reconstruction",
        "Limited depth penetration in highly conductive soils"
    ])

    builder.add_spacer(8)

    builder.add_info_box("Current System Constraints", [
        "Electronics SNR adequate for detection but compromised for precise characterization",
        "Phase accuracy: +/-5 degrees (vs. commercial +/-0.5 degrees)",
        "Noise floor: ~100 nV (vs. commercial ~10 nV)"
    ])

    # Manufacturing Status
    builder.add_section_header("18.10 Manufacturing Status")
    builder.add_body_text(
        "The current release (16mm Modular Probe, released 2024-12-19) is ready for "
        "production printing. The design follows the philosophy of 'archaeologist brain "
        "first, engineer brain second' - minimizing site intrusion while maintaining "
        "field serviceability.",
        first_paragraph=True
    )

    status_data = [
        ['Component', 'Status', 'Notes'],
        ['Male Insert Plug', 'Ready', 'Tested, verified'],
        ['Female Sensor Module', 'Ready', 'Tested, verified'],
        ['Probe Tip', 'Ready', 'Tested, verified'],
        ['Top Cap', 'Ready', 'Tested, verified'],
        ['ERT Ring Collar', 'Ready', 'Tested, verified'],
        ['Rod Coupler', 'Ready', 'Tested, verified'],
        ['Base Hub Enclosure', 'In Progress', 'Backplane PCB design'],
        ['Cable Harness', 'Specified', 'Ready for fabrication']
    ]
    builder.add_table(status_data, caption="Component manufacturing status")

    # Machine Learning Opportunities
    builder.add_section_header("18.11 Machine Learning Opportunities")
    builder.add_body_text(
        "Machine learning presents opportunities for automated analysis and "
        "interpretation:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "<b>Anomaly detection:</b> Automatically identify targets of interest",
        "<b>Classification:</b> Distinguish metal vs. void vs. disturbed soil",
        "<b>Quality assessment:</b> Predict data quality from acquisition parameters",
        "<b>Parameter estimation:</b> Estimate target properties (size, depth, conductivity)"
    ])
    builder.add_note(
        "Machine learning integration requires labeled training data from both "
        "synthetic forward models and validated field measurements."
    )

    # Documentation Needs
    builder.add_section_header("18.12 Documentation Needs")
    builder.add_section_header("18.12.1 User Documentation", level=2)
    builder.add_bullet_list([
        "Processing workflow guide with step-by-step instructions",
        "Parameter selection guidelines for different survey types",
        "Interpretation guide with example datasets",
        "Troubleshooting common issues"
    ])

    builder.add_section_header("18.12.2 Technical Documentation", level=2)
    builder.add_bullet_list([
        "Algorithm descriptions with mathematical derivations",
        "Code documentation (docstrings, API reference)",
        "Validation studies with synthetic and field data",
        "Performance benchmarks for different hardware configurations"
    ])

    # Build the PDF
    return builder.build()


if __name__ == "__main__":
    print("Generating Section 18: Future Development PDF...")
    output_path = build_section_18_pdf()
    print(f"PDF generated successfully: {output_path}")
