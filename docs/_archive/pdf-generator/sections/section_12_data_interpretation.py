#!/usr/bin/env python3
"""
HIRT Section 12: Data Interpretation - Publication-Quality PDF Generator

Generates a professional PDF covering data interpretation for HIRT field data,
including depth of investigation, lateral resolution, method detection capabilities,
and combined interpretation strategies.

Diagrams included:
1. Data processing pipeline
2. Anomaly classification chart
3. Example tomogram visualization
4. Interpretation decision tree

Usage:
    python section_12_data_interpretation.py

Output:
    output/sections/12-data-interpretation.pdf
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
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import numpy as np
from io import BytesIO

from lib.pdf_builder import SectionPDFBuilder
from lib.styles import (
    CONTENT_WIDTH, PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING,
    LIGHT_BG, COLOR_PALETTE
)
from lib.diagrams.flowcharts import (
    draw_process_box, draw_decision_diamond, draw_terminal, draw_flow_arrow,
    create_data_processing_pipeline, COLORS
)


# =============================================================================
# DIAGRAM 1: Data Processing Pipeline (Extended Version)
# =============================================================================
def create_extended_data_processing_pipeline():
    """
    Create an extended data processing pipeline diagram showing all stages
    from raw data to final interpretation.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5.5, 6.2, 'HIRT Data Processing Pipeline', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Main processing stages (top row)
    main_stages = [
        (1.2, 4.5, 'Raw Field\nData', COLORS['light_blue'], 'ADC samples\nfrom probes'),
        (3.5, 4.5, 'Filter &\nAverage', COLORS['light_green'], 'Noise\nreduction'),
        (5.8, 4.5, 'Background\nRemoval', COLORS['light_green'], 'Anomaly\nextraction'),
        (8.1, 4.5, 'Reciprocity\nCheck', COLORS['light_orange'], 'QA/QC\nvalidation'),
    ]

    # Inversion and output stages (bottom row)
    inversion_stages = [
        (1.2, 2.0, 'Tomographic\nInversion', COLORS['light_purple'], 'SIRT/ART\nalgorithms'),
        (3.5, 2.0, 'MIT/ERT\nFusion', COLORS['light_green'], 'Multi-modal\nintegration'),
        (5.8, 2.0, '3D Model\nGeneration', COLORS['light_blue'], 'Voxel grid\nbuilding'),
        (8.1, 2.0, 'Visualization\n& Report', COLORS['light_orange'], 'Depth slices\nisosurfaces'),
    ]

    # Draw main stages
    for x, y, text, color, detail in main_stages:
        draw_process_box(ax, x, y, 1.8, 1.0, text, color)
        ax.text(x, y - 0.75, detail, ha='center', va='top', fontsize=6,
                color=COLORS['gray_dark'], style='italic')

    # Draw inversion stages
    for x, y, text, color, detail in inversion_stages:
        draw_process_box(ax, x, y, 1.8, 1.0, text, color)
        ax.text(x, y - 0.75, detail, ha='center', va='top', fontsize=6,
                color=COLORS['gray_dark'], style='italic')

    # Horizontal arrows (top row)
    for i in range(len(main_stages) - 1):
        draw_flow_arrow(ax, (main_stages[i][0] + 0.9, 4.5),
                        (main_stages[i+1][0] - 0.9, 4.5))

    # Vertical arrow from top to bottom
    draw_flow_arrow(ax, (8.1, 4.0), (8.1, 3.3))
    ax.annotate('', xy=(1.2, 2.5), xytext=(8.1, 3.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5,
                               connectionstyle='angle,angleA=0,angleB=90'))

    # Horizontal arrows (bottom row)
    for i in range(len(inversion_stages) - 1):
        draw_flow_arrow(ax, (inversion_stages[i][0] + 0.9, 2.0),
                        (inversion_stages[i+1][0] - 0.9, 2.0))

    # Final output arrow
    draw_flow_arrow(ax, (8.1 + 0.9, 2.0), (10.0, 2.0))
    draw_terminal(ax, 10.3, 2.0, 0.8, 0.5, 'END', COLORS['light_green'])

    # Input arrow
    draw_terminal(ax, 0.3, 4.5, 0.6, 0.5, 'IN', COLORS['light_blue'])
    draw_flow_arrow(ax, (0.6, 4.5), (1.2 - 0.9, 4.5))

    # Time estimates
    ax.text(2.35, 5.3, '~1-5s', fontsize=7, ha='center', color=COLORS['accent'])
    ax.text(4.65, 5.3, '~2-5s', fontsize=7, ha='center', color=COLORS['accent'])
    ax.text(6.95, 5.3, '~1-2s', fontsize=7, ha='center', color=COLORS['accent'])
    ax.text(2.35, 1.0, '~30-60s', fontsize=7, ha='center', color=COLORS['accent'])
    ax.text(4.65, 1.0, '~5-10s', fontsize=7, ha='center', color=COLORS['accent'])
    ax.text(6.95, 1.0, '~10-20s', fontsize=7, ha='center', color=COLORS['accent'])

    # Phase labels
    ax.text(0.2, 5.5, 'DATA\nACQUISITION', fontsize=7, va='center',
            fontweight='bold', color=COLORS['accent'], rotation=90)
    ax.text(0.2, 2.5, 'INVERSION\n& OUTPUT', fontsize=7, va='center',
            fontweight='bold', color=COLORS['purple'], rotation=90)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# =============================================================================
# DIAGRAM 2: Anomaly Classification Chart
# =============================================================================
def create_anomaly_classification_chart():
    """
    Create an anomaly classification chart showing MIT vs ERT response
    characteristics for different target types.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.7, 'Anomaly Classification: MIT vs ERT Response', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Create classification matrix
    categories = [
        ('Metal Objects', 'Strong', 'Weak/None', COLORS['warning'],
         ['Engine parts', 'Aircraft wreckage', 'Artifacts']),
        ('Conductive Soil', 'Moderate', 'Moderate', COLORS['orange'],
         ['Wet clay', 'Saline zones', 'Contamination']),
        ('Disturbed Fill', 'Weak', 'Strong', COLORS['purple'],
         ['Crater fill', 'Grave shafts', 'Backfill']),
        ('Voids/Air', 'None', 'Strong', COLORS['gray_light'],
         ['Cavities', 'Collapsed zones', 'Air pockets']),
        ('Moisture Zones', 'Weak', 'Strong', COLORS['accent'],
         ['Water table', 'Wet pockets', 'Seepage']),
    ]

    # Header row
    headers = ['Target Type', 'MIT Response', 'ERT Response', 'Examples']
    x_positions = [1.2, 3.5, 5.5, 8.0]

    for i, (header, x) in enumerate(zip(headers, x_positions)):
        ax.add_patch(FancyBboxPatch((x - 0.9, 5.8), 1.8 if i < 3 else 3.2, 0.5,
                                     boxstyle="round,pad=0.02",
                                     facecolor=COLORS['primary'],
                                     edgecolor='black', lw=1))
        ax.text(x, 6.05, header, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

    # Data rows
    for row_idx, (target_type, mit, ert, color, examples) in enumerate(categories):
        y = 5.2 - row_idx * 1.0

        # Target type cell
        ax.add_patch(FancyBboxPatch((0.3, y - 0.35), 1.8, 0.7,
                                     boxstyle="round,pad=0.02",
                                     facecolor=color, alpha=0.3,
                                     edgecolor=color, lw=1.5))
        ax.text(1.2, y, target_type, ha='center', va='center', fontsize=8,
                fontweight='bold', color=COLORS['gray_dark'])

        # MIT response cell with visual indicator
        ax.add_patch(FancyBboxPatch((2.6, y - 0.35), 1.8, 0.7,
                                     boxstyle="round,pad=0.02",
                                     facecolor='white', edgecolor=COLORS['gray_light'], lw=0.5))
        ax.text(3.5, y, mit, ha='center', va='center', fontsize=8)

        # Response level indicator for MIT
        mit_level = {'Strong': 0.9, 'Moderate': 0.5, 'Weak': 0.25, 'None': 0.0}[mit]
        ax.add_patch(Rectangle((2.7, y - 0.25), 1.6 * mit_level, 0.15,
                               facecolor=COLORS['success'], alpha=0.7))

        # ERT response cell with visual indicator
        ax.add_patch(FancyBboxPatch((4.6, y - 0.35), 1.8, 0.7,
                                     boxstyle="round,pad=0.02",
                                     facecolor='white', edgecolor=COLORS['gray_light'], lw=0.5))
        ax.text(5.5, y, ert, ha='center', va='center', fontsize=8)

        # Response level indicator for ERT
        ert_level = {'Strong': 0.9, 'Moderate': 0.5, 'Weak': 0.25, 'None': 0.0, 'Weak/None': 0.1}[ert]
        ax.add_patch(Rectangle((4.7, y - 0.25), 1.6 * ert_level, 0.15,
                               facecolor=COLORS['orange'], alpha=0.7))

        # Examples cell
        ax.add_patch(FancyBboxPatch((6.4, y - 0.35), 3.2, 0.7,
                                     boxstyle="round,pad=0.02",
                                     facecolor='white', edgecolor=COLORS['gray_light'], lw=0.5))
        examples_text = ', '.join(examples)
        ax.text(8.0, y, examples_text, ha='center', va='center', fontsize=7,
                style='italic', color=COLORS['gray_dark'])

    # Legend
    ax.add_patch(Rectangle((0.5, 0.3), 0.4, 0.15, facecolor=COLORS['success'], alpha=0.7))
    ax.text(1.0, 0.375, 'MIT Response Level', fontsize=7, va='center')
    ax.add_patch(Rectangle((3.5, 0.3), 0.4, 0.15, facecolor=COLORS['orange'], alpha=0.7))
    ax.text(4.0, 0.375, 'ERT Response Level', fontsize=7, va='center')

    # Key insight box
    ax.add_patch(FancyBboxPatch((6.0, 0.1), 3.8, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=COLORS['light_blue'],
                                 edgecolor=COLORS['accent'], lw=1.5))
    ax.text(7.9, 0.4, 'KEY: Combine MIT + ERT for complete picture',
            ha='center', va='center', fontsize=8, fontweight='bold',
            color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# =============================================================================
# DIAGRAM 3: Example Tomogram Visualization
# =============================================================================
def create_tomogram_visualization():
    """
    Create an example tomogram visualization showing a simulated HIRT
    scan result with MIT and ERT data combined.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig = plt.figure(figsize=(11, 6), constrained_layout=True)

    # Create grid for subplots
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], hspace=0.15, wspace=0.2)

    # MIT tomogram (top left)
    ax_mit = fig.add_subplot(gs[0, 0])

    # Create synthetic MIT data (metal target response)
    x = np.linspace(0, 5, 100)
    z = np.linspace(0, 4, 80)
    X, Z = np.meshgrid(x, z)

    # Simulate metal target anomaly at (2.5, 2.5)
    mit_data = np.zeros_like(X)
    target_x, target_z = 2.5, 2.5
    distance = np.sqrt((X - target_x)**2 + (Z - target_z)**2)
    mit_data = np.exp(-distance**2 / 0.3) * 100

    # Add some noise and background
    mit_data += np.random.randn(*mit_data.shape) * 2
    mit_data = np.clip(mit_data, 0, 100)

    im_mit = ax_mit.imshow(mit_data, extent=[0, 5, 4, 0], cmap='hot',
                           aspect='auto', vmin=0, vmax=100)
    ax_mit.set_xlabel('Distance (m)', fontsize=9)
    ax_mit.set_ylabel('Depth (m)', fontsize=9)
    ax_mit.set_title('MIT Conductivity (mS/m)', fontsize=10, fontweight='bold',
                     color=COLORS['primary'])

    # Mark target location
    ax_mit.plot(target_x, target_z, 'w+', markersize=12, markeredgewidth=2)
    ax_mit.annotate('Metal\nTarget', (target_x, target_z), (target_x + 1, target_z - 0.5),
                    fontsize=8, color='white', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    # Probe positions
    for probe_x in [0.5, 4.5]:
        ax_mit.axvline(probe_x, color='cyan', linestyle='--', alpha=0.5, lw=1)
        ax_mit.text(probe_x, 0.2, 'Probe', fontsize=7, ha='center', color='cyan')

    # ERT tomogram (top right)
    ax_ert = fig.add_subplot(gs[0, 1])

    # Simulate disturbed fill anomaly (bowl-shaped crater)
    ert_data = np.ones_like(X) * 50  # Background resistivity

    # Crater fill (lower resistivity)
    crater_center_x = 2.5
    for i, depth in enumerate(z):
        crater_width = max(0, 2.0 - 0.4 * depth)
        mask = np.abs(X[i, :] - crater_center_x) < crater_width
        if depth < 3.5:  # Fill depth
            ert_data[i, mask] = 30 + np.random.randn(np.sum(mask)) * 3

    # Add moisture pocket
    moisture_dist = np.sqrt((X - 2.8)**2 + (Z - 3.0)**2)
    ert_data -= np.exp(-moisture_dist**2 / 0.15) * 20
    ert_data = np.clip(ert_data, 10, 80)

    im_ert = ax_ert.imshow(ert_data, extent=[0, 5, 4, 0], cmap='viridis_r',
                           aspect='auto', vmin=10, vmax=80)
    ax_ert.set_xlabel('Distance (m)', fontsize=9)
    ax_ert.set_ylabel('Depth (m)', fontsize=9)
    ax_ert.set_title('ERT Resistivity (ohm-m)', fontsize=10, fontweight='bold',
                     color=COLORS['primary'])

    # Mark crater boundary
    crater_outline_x = [0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5]
    crater_outline_z = [0.5, 1.5, 2.5, 3.5, 2.5, 1.5, 0.5]
    ax_ert.plot(crater_outline_x, crater_outline_z, 'w--', lw=2, alpha=0.7)
    ax_ert.annotate('Fill\nBoundary', (3.5, 1.5), (4.2, 0.8),
                    fontsize=8, color='white', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    # Colorbars
    ax_cb = fig.add_subplot(gs[0, 2])
    ax_cb.axis('off')

    # Combined interpretation (bottom, spanning both columns)
    ax_combined = fig.add_subplot(gs[1, :2])

    # Create combined visualization
    combined_data = np.zeros((*mit_data.shape, 3))

    # Normalize MIT to red channel
    mit_norm = mit_data / 100
    combined_data[:, :, 0] = mit_norm * 0.8

    # Normalize ERT contrast to blue/green channels
    ert_norm = 1 - (ert_data - 10) / 70
    combined_data[:, :, 1] = ert_norm * 0.6
    combined_data[:, :, 2] = ert_norm * 0.8

    combined_data = np.clip(combined_data, 0, 1)

    ax_combined.imshow(combined_data, extent=[0, 5, 4, 0], aspect='auto')
    ax_combined.set_xlabel('Distance (m)', fontsize=9)
    ax_combined.set_ylabel('Depth (m)', fontsize=9)
    ax_combined.set_title('Combined MIT + ERT Interpretation', fontsize=10,
                          fontweight='bold', color=COLORS['primary'])

    # Annotations on combined view
    ax_combined.plot(target_x, target_z, 'w+', markersize=15, markeredgewidth=3)
    ax_combined.annotate('HIGH PRIORITY:\nMetal in Fill', (target_x, target_z),
                         (target_x + 1.2, target_z + 0.8),
                         fontsize=9, color='white', fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', color='white', lw=2))

    ax_combined.annotate('Disturbed\nFill Zone', (2.5, 1.5), (0.5, 0.8),
                         fontsize=8, color='white',
                         arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    # Legend box
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.7, label='MIT: High conductivity (metal)'),
        mpatches.Patch(facecolor='blue', alpha=0.7, label='ERT: Low resistivity (wet fill)'),
        mpatches.Patch(facecolor='purple', alpha=0.7, label='Combined: Target in disturbed zone'),
    ]
    ax_combined.legend(handles=legend_elements, loc='lower right', fontsize=7,
                       framealpha=0.9)

    # Main title
    fig.suptitle('Example Tomogram: WWII Crash Site Investigation',
                 fontsize=12, fontweight='bold', color=COLORS['primary'])

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# =============================================================================
# DIAGRAM 4: Interpretation Decision Tree
# =============================================================================
def create_interpretation_decision_tree():
    """
    Create a decision tree for interpreting HIRT survey results.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5.5, 8.7, 'Data Interpretation Decision Tree', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Start
    draw_terminal(ax, 5.5, 8.2, 2.5, 0.5, 'START: Data QC', COLORS['light_green'])

    # First decision: Data quality
    draw_decision_diamond(ax, 5.5, 7.0, 1.4, 'Reciprocity\nOK?', COLORS['light_blue'])
    draw_flow_arrow(ax, (5.5, 7.95), (5.5, 7.7))

    # Poor reciprocity path
    draw_flow_arrow(ax, (4.8, 7.0), (3.0, 7.0), 'No')
    draw_process_box(ax, 1.8, 7.0, 1.8, 0.8, 'Check\nconnections\n& reacquire', COLORS['light_red'])
    draw_flow_arrow(ax, (1.8, 7.4), (1.8, 8.2))
    ax.annotate('', xy=(5.5, 8.2), xytext=(2.7, 8.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Good data path
    draw_flow_arrow(ax, (5.5, 6.3), (5.5, 5.7), 'Yes')
    draw_decision_diamond(ax, 5.5, 5.2, 1.4, 'MIT\nAnomaly?', COLORS['light_blue'])

    # Strong MIT anomaly path
    draw_flow_arrow(ax, (6.2, 5.2), (8.0, 5.2), 'Strong')
    draw_decision_diamond(ax, 9.0, 5.2, 1.2, 'ERT\nContrast?', COLORS['light_blue'])

    # MIT + ERT both positive
    draw_flow_arrow(ax, (9.0, 4.6), (9.0, 3.8), 'Yes')
    draw_process_box(ax, 9.0, 3.3, 2.0, 0.8, 'Metal in\ndisturbed zone\n(HIGH priority)', COLORS['light_red'])

    # MIT only
    draw_flow_arrow(ax, (9.6, 5.2), (10.3, 5.2), 'No')
    draw_process_box(ax, 10.3, 5.2, 1.2, 0.6, 'Isolated\nmetal', COLORS['light_orange'])

    # Weak/No MIT path
    draw_flow_arrow(ax, (5.5, 4.5), (5.5, 3.8), 'Weak/None')
    draw_decision_diamond(ax, 5.5, 3.3, 1.4, 'ERT\nAnomaly?', COLORS['light_blue'])

    # ERT anomaly path
    draw_flow_arrow(ax, (6.2, 3.3), (7.5, 3.3), 'Yes')
    draw_decision_diamond(ax, 8.3, 3.3, 1.0, 'High\nor Low\nR?', COLORS['light_blue'])

    # High resistivity
    draw_flow_arrow(ax, (8.8, 3.3), (10.0, 3.3), 'High')
    draw_process_box(ax, 10.0, 3.3, 1.5, 0.6, 'Void or\ndry zone', COLORS['light_purple'])

    # Low resistivity
    draw_flow_arrow(ax, (8.3, 2.8), (8.3, 2.2), 'Low')
    draw_process_box(ax, 8.3, 1.7, 1.8, 0.7, 'Wet fill or\nclay layer', COLORS['light_blue'])

    # No ERT anomaly either
    draw_flow_arrow(ax, (5.5, 2.6), (5.5, 1.8), 'No')
    draw_process_box(ax, 5.5, 1.3, 2.2, 0.8, 'Homogeneous\nground\n(no targets)', COLORS['light_green'])

    # Output decision boxes
    ax.add_patch(FancyBboxPatch((0.5, 0.3), 10.0, 0.8,
                                 boxstyle="round,pad=0.02",
                                 facecolor=COLORS['light_blue'], alpha=0.3,
                                 edgecolor=COLORS['accent'], lw=1.5))

    # Priority indicators
    priority_items = [
        (1.5, 0.7, 'HIGH', COLORS['warning'], 'Metal + Disturbed'),
        (4.0, 0.7, 'MEDIUM', COLORS['orange'], 'Metal or Fill only'),
        (6.8, 0.7, 'LOW', COLORS['success'], 'Minor anomalies'),
        (9.2, 0.7, 'NONE', COLORS['gray_light'], 'No targets'),
    ]

    for x, y, label, color, desc in priority_items:
        ax.add_patch(Circle((x - 0.3, y), 0.15, facecolor=color, edgecolor='black', lw=1))
        ax.text(x, y, f'{label}:', fontsize=8, fontweight='bold', va='center')
        ax.text(x + 0.8, y, desc, fontsize=7, va='center', color=COLORS['gray_dark'])

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
def build_section_12_pdf():
    """
    Build the complete Section 12: Data Interpretation PDF document.

    Returns:
        Path to the generated PDF file
    """
    # Initialize builder
    builder = SectionPDFBuilder(
        section_num=12,
        title="Data Interpretation"
    )

    # Title block
    builder.add_title_block(
        subtitle="HIRT Field Data Analysis, Tomographic Inversion, and Anomaly Interpretation",
        version="2.0"
    )

    # Overview
    builder.add_section_header("12.1 Overview", level=1)
    builder.add_body_text(
        "This section provides comprehensive guidance on interpreting HIRT field data, "
        "including depth of investigation, lateral resolution, and what each measurement "
        "method detects. Effective interpretation requires understanding both the physics "
        "of the measurements and the geological/archaeological context of the survey site.",
        first_paragraph=True
    )

    # Data Processing Pipeline
    builder.add_section_header("12.2 Data Processing Pipeline", level=1)
    builder.add_body_text(
        "HIRT data processing follows a systematic pipeline from raw field acquisition "
        "through final interpretation. Each stage validates and refines the data to produce "
        "reliable subsurface images.",
        first_paragraph=True
    )

    # Add extended pipeline diagram
    pipeline_buf = create_extended_data_processing_pipeline()
    builder.add_figure(
        pipeline_buf,
        "HIRT data processing pipeline from raw field measurements through tomographic "
        "inversion to final 3D visualization. Processing times shown are typical for a "
        "4x4 probe grid survey."
    )

    builder.add_section_header("Processing Stages", level=2)
    builder.add_numbered_list([
        "<b>Data Acquisition:</b> Raw ADC samples collected from probe pairs",
        "<b>Filter and Average:</b> Noise reduction through digital filtering and stacking",
        "<b>Background Removal:</b> Subtract homogeneous response to isolate anomalies",
        "<b>Reciprocity Check:</b> Validate A-to-B equals B-to-A for quality assurance",
        "<b>Tomographic Inversion:</b> Reconstruct 3D conductivity/resistivity distribution",
        "<b>MIT/ERT Fusion:</b> Combine multi-modal data for comprehensive interpretation",
        "<b>3D Model Generation:</b> Build voxel grid from inverted data",
        "<b>Visualization:</b> Generate depth slices, isosurfaces, and reports"
    ])

    # Depth of Investigation
    builder.add_section_header("12.3 Depth of Investigation", level=1)
    builder.add_body_text(
        "The achievable investigation depth depends on several factors including probe "
        "depth, probe spacing, soil conductivity, and measurement frequency. Understanding "
        "these relationships is essential for survey planning and result interpretation.",
        first_paragraph=True
    )

    builder.add_section_header("Factors Affecting Depth", level=2)
    builder.add_bullet_list([
        "<b>Probe depth:</b> Deeper insertion enables deeper sensitivity",
        "<b>Probe spacing:</b> Wider spacing increases depth but reduces lateral resolution",
        "<b>Soil conductivity:</b> Lower conductivity enables deeper signal penetration",
        "<b>Frequency (MIT):</b> Lower frequency provides deeper sensitivity",
        "<b>Current geometry (ERT):</b> Longer baselines enable deeper investigation"
    ])

    # Depth claims table
    depth_data = [
        ['Configuration', 'MIT Depth', 'ERT Depth', 'Combined', 'Confidence'],
        ['1.5m probes, 1.5m spacing', '1.5-2.5m', '2-3m', '2-3m', 'HIGH'],
        ['1.5m probes, 2.0m spacing', '2-3m', '2-3m', '2-3m', 'HIGH'],
        ['3.0m probes, 2.0m spacing', '3-4m', '3-5m', '3-5m', 'MEDIUM'],
        ['3.0m probes, 2.5m spacing', '3-4m', '4-6m', '4-6m', 'LOW'],
    ]
    builder.add_table(depth_data, caption="Depth of investigation by configuration",
                      col_widths=[CONTENT_WIDTH*0.30, CONTENT_WIDTH*0.17, CONTENT_WIDTH*0.17,
                                  CONTENT_WIDTH*0.17, CONTENT_WIDTH*0.19])

    builder.add_note(
        "The commonly cited '3-6m' depth range represents favorable conditions only. "
        "For most field conditions, expect 2-4m typical depth, with up to 5-6m achievable "
        "in optimal soil conditions with longer probes and wider spacing."
    )

    # Lateral Resolution
    builder.add_section_header("12.4 Lateral Resolution", level=1)
    builder.add_body_text(
        "Lateral resolution approximately equals 0.5 to 1.5 times the probe spacing. "
        "Tighter spacing provides finer resolution but requires more survey time, while "
        "wider spacing offers faster coverage with coarser resolution.",
        first_paragraph=True
    )

    resolution_data = [
        ['Spacing', 'Lateral Resolution', 'Best Application'],
        ['1.0 m', '0.5-1.5 m', 'High-resolution burial/artifact search'],
        ['1.5 m', '0.75-2.25 m', 'Standard WWII crash investigation'],
        ['2.0 m', '1.0-3.0 m', 'Large feature reconnaissance'],
    ]
    builder.add_table(resolution_data, caption="Resolution by probe spacing",
                      col_widths=[CONTENT_WIDTH*0.15, CONTENT_WIDTH*0.25, CONTENT_WIDTH*0.60])

    # HIRT vs Surface Methods
    builder.add_section_header("12.5 HIRT vs Surface Methods", level=1)
    builder.add_body_text(
        "HIRT's crosshole geometry provides 2-5 times better resolution than surface "
        "methods at depths greater than 2m. This advantage increases with depth due to "
        "direct ray paths and elimination of surface clutter.",
        first_paragraph=True
    )

    comparison_data = [
        ['Method', 'Lateral Res.', 'Depth Res.', 'At 3m Depth'],
        ['Surface Magnetometry', '1-2m', 'Poor', '~2m lateral'],
        ['GPR (in sand)', '0.3-0.5m', '0.05-0.1m', 'Degrades to 1m+'],
        ['GPR (in clay)', 'Limited', 'Limited', 'Often fails'],
        ['Surface ERT (Wenner)', '~1x spacing', '~0.5x spacing', '~2-3m'],
        ['HIRT (1.5m spacing)', '0.75-1.5m', '0.5-0.75m', '~1m lateral'],
        ['HIRT (2m spacing)', '1-2m', '0.5-1m', '~1.5m lateral'],
    ]
    builder.add_table(comparison_data, caption="Resolution comparison: HIRT vs surface methods",
                      col_widths=[CONTENT_WIDTH*0.28, CONTENT_WIDTH*0.18, CONTENT_WIDTH*0.18, CONTENT_WIDTH*0.36])

    builder.add_info_box("Why HIRT Achieves Better Resolution", [
        "Direct ray paths through target volume (not down-and-back)",
        "No surface clutter from topography and cultural noise",
        "True 3D sampling enables genuine tomographic reconstruction",
        "Better depth discrimination between targets at different depths"
    ])

    # What Each Method Detects
    builder.add_section_header("12.6 Anomaly Classification", level=1)
    builder.add_body_text(
        "MIT and ERT respond differently to various subsurface features. Understanding "
        "these response characteristics enables accurate classification of detected anomalies.",
        first_paragraph=True
    )

    # Add anomaly classification chart
    anomaly_buf = create_anomaly_classification_chart()
    builder.add_figure(
        anomaly_buf,
        "Anomaly classification chart showing expected MIT and ERT response levels "
        "for common target types encountered in archaeological and forensic investigations."
    )

    builder.add_section_header("MIT (Magneto-Inductive Tomography) Detection", level=2)
    builder.add_bullet_list([
        "<b>Metal objects:</b> Strong response to aluminum, steel, and iron",
        "<b>Conductive regions:</b> Moderate response to saline water, clay layers",
        "<b>Eddy current anomalies:</b> Metallic wreckage produces characteristic signatures",
        "Phase lag indicates conductivity; amplitude indicates size/distance",
        "Higher frequencies provide better near-surface sensitivity"
    ])

    builder.add_section_header("ERT (Electrical Resistivity) Detection", level=2)
    builder.add_bullet_list([
        "<b>Disturbed fill:</b> Different compaction and moisture than native soil",
        "<b>Moisture variations:</b> Wet zones appear as low resistivity",
        "<b>Crater walls:</b> Clear boundaries between fill and native soil",
        "<b>Voids:</b> Air-filled spaces show very high resistivity",
        "Depth slices reveal layering and lateral extent of features"
    ])

    builder.add_page_break()

    # Example Tomogram
    builder.add_section_header("12.7 Example Tomogram Interpretation", level=1)
    builder.add_body_text(
        "The following example demonstrates combined MIT and ERT interpretation for a "
        "simulated WWII crash site investigation. Note how the complementary data streams "
        "provide more complete subsurface characterization than either method alone.",
        first_paragraph=True
    )

    # Add tomogram visualization
    tomogram_buf = create_tomogram_visualization()
    builder.add_figure(
        tomogram_buf,
        "Example tomogram showing MIT conductivity (top left), ERT resistivity (top right), "
        "and combined interpretation (bottom). The metal target within the disturbed fill "
        "zone is identified as a high-priority excavation target."
    )

    builder.add_section_header("Interpretation Notes", level=2)
    builder.add_bullet_list([
        "MIT clearly identifies the metal concentration at 2.5m depth",
        "ERT reveals the crater fill geometry and moisture distribution",
        "Combined view shows the target context within disturbed ground",
        "High-priority designation indicates both metal and disturbed fill present"
    ])

    # Combined Interpretation
    builder.add_section_header("12.8 Combined Interpretation Strategy", level=1)
    builder.add_body_text(
        "Effective HIRT interpretation combines MIT and ERT results with site context "
        "and historical information. The following decision tree provides systematic "
        "guidance for prioritizing investigation targets.",
        first_paragraph=True
    )

    # Add interpretation decision tree
    decision_tree_buf = create_interpretation_decision_tree()
    builder.add_figure(
        decision_tree_buf,
        "Data interpretation decision tree for prioritizing HIRT survey results. "
        "Begin with data quality verification, then evaluate MIT and ERT anomalies "
        "to assign investigation priority levels."
    )

    builder.add_section_header("Example Interpretation Scenarios", level=2)

    builder.add_section_header("Scenario 1: Bomb Crater Investigation", level=3)
    builder.add_bullet_list([
        "<b>MIT:</b> Metal parts detected near crater base (aluminum/steel fragments)",
        "<b>ERT:</b> Fill bowl geometry visible, wet pockets at base, clear crater walls",
        "<b>Interpretation:</b> Classic impact crater with retained metallic debris"
    ])

    builder.add_section_header("Scenario 2: Woods Burial Search", level=3)
    builder.add_bullet_list([
        "<b>MIT:</b> Small metallic clusters (buckles, dog tags, buttons)",
        "<b>ERT:</b> Rectangular disturbed zone with different moisture profile",
        "<b>Interpretation:</b> Possible grave shaft requiring careful investigation"
    ])

    builder.add_section_header("Scenario 3: Aircraft Wreckage", level=3)
    builder.add_bullet_list([
        "<b>MIT:</b> Large conductive masses (engine block, landing gear)",
        "<b>ERT:</b> Disturbed ground pattern, possible fuel contamination zones",
        "<b>Interpretation:</b> Significant wreckage concentration warranting excavation"
    ])

    # Data Quality Indicators
    builder.add_section_header("12.9 Data Quality Indicators", level=1)

    builder.add_section_header("Good Data Characteristics", level=2)
    builder.add_bullet_list([
        "Consistent reciprocity (A-to-B approximately equals B-to-A)",
        "Smooth spatial variations without erratic jumps",
        "Expected depth sensitivity matching configuration",
        "Stable baseline measurements over survey duration"
    ])

    builder.add_warning_box("Problematic Data Warning Signs", [
        "Poor reciprocity indicates coupling problems or calibration drift",
        "Noisy/spiky readings suggest connection issues or EMI",
        "No depth sensitivity may indicate inadequate spacing or frequency",
        "Inconsistent repeats require checking timebase and connectors"
    ])

    # Field Expectations
    builder.add_section_header("12.10 Field Expectations and Detection Limits", level=1)

    builder.add_section_header("Typical Anomaly Sizes", level=2)
    anomaly_sizes = [
        ['Target Type', 'Typical Size', 'Expected Response'],
        ['Large metal (engine)', '1-3 m', 'Strong MIT response'],
        ['Small metal (artifacts)', '0.1-0.5 m', 'Weaker MIT, requires tight spacing'],
        ['Grave shaft', '0.5-1.5 m wide', 'Clear ERT contrast'],
        ['Crater fill', '10-15 m diameter', 'ERT shows boundaries clearly'],
    ]
    builder.add_table(anomaly_sizes, caption="Typical anomaly sizes and expected responses")

    builder.add_section_header("Detection Limits", level=2)
    builder.add_bullet_list([
        "<b>MIT:</b> Can detect ~0.1m metal at 1-2m depth (size dependent)",
        "<b>ERT:</b> Can resolve ~0.5m features at 1-2m depth",
        "<b>Depth:</b> Practical limit 2-4m typical with 3m probes (5-6m in optimal conditions)"
    ])

    # Next Steps
    builder.add_section_header("12.11 Post-Survey Workflow", level=1)
    builder.add_numbered_list([
        "<b>QA/QC:</b> Verify data quality and reciprocity",
        "<b>Inversion:</b> Reconstruct 3D models using appropriate algorithms",
        "<b>Fusion:</b> Combine MIT and ERT results for complete picture",
        "<b>Visualization:</b> Generate depth slices and 3D isosurfaces",
        "<b>Interpretation:</b> Correlate anomalies with site context",
        "<b>Reporting:</b> Document findings with confidence assessments",
        "<b>Planning:</b> Recommend excavation priorities and methods"
    ])

    builder.add_info_box("Key Interpretation Principles", [
        "Always combine MIT and ERT for complete subsurface picture",
        "Consider site history and expected target characteristics",
        "Use confidence levels to guide excavation prioritization",
        "Document interpretation rationale for future reference"
    ])

    # === Mathematical Basis ===
    builder.add_page_break()
    builder.add_section_header("12.12 Mathematical Basis of Reconstruction", level=1)
    builder.add_body_text(
        "For scientific rigor, it is important to define the physical models governing the "
        "reconstruction algorithms. The inversion process solves the 'inverse problem': "
        "finding the subsurface property distribution that best explains the measured data.",
        first_paragraph=True
    )

    builder.add_section_header("Forward Models", level=2)
    builder.add_body_text(
        "<b>MIT (Quasi-Static Maxwell's Equations):</b> The secondary magnetic field B_s "
        "generated by eddy currents in a conductive medium is governed by:",
    )
    builder.add_equation("curl E = -i * omega * B")
    builder.add_equation("curl H = sigma * E + J_s")
    builder.add_body_text(
        "Where sigma is conductivity, omega is angular frequency, and J_s is the source current."
    )

    builder.add_body_text(
        "<b>ERT (Poisson's Equation):</b> The electric potential phi distribution due to a "
        "current source I in a medium of conductivity sigma is governed by:",
    )
    builder.add_equation("div (sigma * grad phi) = -I * delta(r - r_s)")

    builder.add_section_header("Inverse Problem Formulation", level=2)
    builder.add_body_text(
        "The reconstruction minimizes an objective function Phi comprising data mismatch "
        "and model roughness (regularization):"
    )
    builder.add_equation("Phi(m) = || d_obs - F(m) ||^2 + lambda * || R * m ||^2")
    builder.add_body_text(
        "Where d_obs is observed data, F(m) is the forward model output for model m, "
        "R is the roughness matrix, and lambda is the regularization parameter. "
        "This is typically solved using a Gauss-Newton iterative update:"
    )
    builder.add_equation("m_{k+1} = m_k - (J^T J + lambda R^T R)^-1 J^T (d_obs - F(m_k))")
    builder.add_body_text(
        "Where J is the Sensitivity Matrix (Jacobian) describing how data changes with model parameters."
    )

    # Build and return
    return builder.build()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("HIRT Section 12: Data Interpretation - PDF Generator")
    print("="*60)

    output_path = build_section_12_pdf()

    print("\n" + "="*60)
    print("PDF Generation Complete!")
    print(f"Output: {output_path}")
    print("="*60)
