#!/usr/bin/env python3
"""
HIRT Section 11: Data Recording - PDF Generator

Generates a publication-quality PDF for Section 11 of the HIRT whitepaper,
covering data formats, file structures, and logging procedures.

Usage:
    python section_11_data_recording.py

Output:
    output/sections/11-data-recording.pdf
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
from io import BytesIO

from lib.pdf_builder import SectionPDFBuilder
from lib.styles import (
    CONTENT_WIDTH, COLOR_PALETTE, PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING
)


# ============================================================================
# FIGURE 1: DATA FILE STRUCTURE DIAGRAM
# ============================================================================
def create_file_structure_diagram():
    """
    Create a diagram showing the HIRT data file organization and structure.
    Shows directory hierarchy and file naming conventions.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.7, 'HIRT Data File Organization', fontsize=12, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    # === Directory Tree (Left Side) ===
    tree_x = 0.5

    # Root folder
    ax.add_patch(FancyBboxPatch((tree_x, 6.5), 2.2, 0.5, boxstyle="round,pad=0.02",
                                facecolor=COLOR_PALETTE['accent'], edgecolor='black', linewidth=1))
    ax.text(tree_x + 1.1, 6.75, 'data/', fontsize=10, fontweight='bold',
            ha='center', va='center', color='white', family='monospace')

    # Date folder
    ax.plot([tree_x + 0.3, tree_x + 0.3, tree_x + 0.6], [6.5, 5.8, 5.8],
            color='#4a5568', linewidth=1.5)
    ax.add_patch(FancyBboxPatch((tree_x + 0.6, 5.55), 2.5, 0.5, boxstyle="round,pad=0.02",
                                facecolor=COLOR_PALETTE['secondary'], edgecolor='black', linewidth=1))
    ax.text(tree_x + 1.85, 5.8, '2024-03-15/', fontsize=9, fontweight='bold',
            ha='center', va='center', color='white', family='monospace')

    # MIT files
    file_y = 4.9
    for i, fname in enumerate(['MIT_S01_2024-03-15.csv', 'MIT_S02_2024-03-15.csv']):
        ax.plot([tree_x + 0.9, tree_x + 0.9, tree_x + 1.2], [5.55, file_y - i*0.55, file_y - i*0.55],
                color='#4a5568', linewidth=1.5)
        ax.add_patch(FancyBboxPatch((tree_x + 1.2, file_y - 0.2 - i*0.55), 3.0, 0.4,
                                    boxstyle="round,pad=0.02",
                                    facecolor=COLOR_PALETTE['success'], edgecolor='black', linewidth=1))
        ax.text(tree_x + 2.7, file_y - i*0.55, fname, fontsize=8,
                ha='center', va='center', color='white', family='monospace')

    # ERT files
    file_y = 3.5
    for i, fname in enumerate(['ERT_S01_2024-03-15.csv', 'ERT_S02_2024-03-15.csv']):
        ax.plot([tree_x + 0.9, tree_x + 0.9, tree_x + 1.2], [4.35, file_y - i*0.55, file_y - i*0.55],
                color='#4a5568', linewidth=1.5)
        ax.add_patch(FancyBboxPatch((tree_x + 1.2, file_y - 0.2 - i*0.55), 3.0, 0.4,
                                    boxstyle="round,pad=0.02",
                                    facecolor=COLOR_PALETTE['orange'], edgecolor='black', linewidth=1))
        ax.text(tree_x + 2.7, file_y - i*0.55, fname, fontsize=8,
                ha='center', va='center', color='white', family='monospace')

    # Registry file (at root level)
    ax.plot([tree_x + 0.3, tree_x + 0.3, tree_x + 0.6], [6.5, 2.0, 2.0],
            color='#4a5568', linewidth=1.5)
    ax.add_patch(FancyBboxPatch((tree_x + 0.6, 1.75), 2.8, 0.5, boxstyle="round,pad=0.02",
                                facecolor=COLOR_PALETTE['purple'], edgecolor='black', linewidth=1))
    ax.text(tree_x + 2.0, 2.0, 'probe_registry.csv', fontsize=9, fontweight='bold',
            ha='center', va='center', color='white', family='monospace')

    # Field log
    ax.plot([tree_x + 0.3, tree_x + 0.3, tree_x + 0.6], [2.0, 1.0, 1.0],
            color='#4a5568', linewidth=1.5)
    ax.add_patch(FancyBboxPatch((tree_x + 0.6, 0.75), 3.0, 0.5, boxstyle="round,pad=0.02",
                                facecolor=COLOR_PALETTE['gray_med'], edgecolor='black', linewidth=1))
    ax.text(tree_x + 2.1, 1.0, 'field_log_2024-03-15.txt', fontsize=9, fontweight='bold',
            ha='center', va='center', color='white', family='monospace')

    # === File Format Details (Right Side) ===
    detail_x = 5.5

    # MIT Record Format Box
    ax.add_patch(FancyBboxPatch((detail_x, 5.2), 4.3, 2.5, boxstyle="round,pad=0.05",
                                facecolor='#f0fff4', edgecolor=COLOR_PALETTE['success'], linewidth=2))
    ax.text(detail_x + 2.15, 7.4, 'MIT Record Fields', fontsize=10, fontweight='bold',
            ha='center', color=COLOR_PALETTE['success'])

    mit_fields = [
        ('timestamp', 'ISO 8601'),
        ('section_id', 'String (S01)'),
        ('zone_id', 'String (ZA)'),
        ('tx_probe_id', 'String (P01)'),
        ('rx_probe_id', 'String (P02)'),
        ('freq_hz', 'Hz'),
        ('amp', 'V or normalized'),
        ('phase_deg', 'Degrees'),
        ('tx_current_mA', 'mA'),
    ]

    for i, (field, fmt) in enumerate(mit_fields):
        y_pos = 7.0 - i * 0.2
        ax.text(detail_x + 0.15, y_pos, field, fontsize=7, family='monospace',
                color=COLOR_PALETTE['gray_dark'])
        ax.text(detail_x + 2.3, y_pos, fmt, fontsize=7, color=COLOR_PALETTE['text_muted'])

    # ERT Record Format Box
    ax.add_patch(FancyBboxPatch((detail_x, 2.3), 4.3, 2.6, boxstyle="round,pad=0.05",
                                facecolor='#fffaf0', edgecolor=COLOR_PALETTE['orange'], linewidth=2))
    ax.text(detail_x + 2.15, 4.6, 'ERT Record Fields', fontsize=10, fontweight='bold',
            ha='center', color=COLOR_PALETTE['orange'])

    ert_fields = [
        ('timestamp', 'ISO 8601'),
        ('section_id', 'String'),
        ('zone_id', 'String'),
        ('inject_pos_id', 'String (P01)'),
        ('inject_neg_id', 'String (P20)'),
        ('sense_id', 'String (P05)'),
        ('volt_mV', 'mV'),
        ('current_mA', 'mA'),
        ('polarity', '+1 or -1'),
        ('notes', 'Free text'),
    ]

    for i, (field, fmt) in enumerate(ert_fields):
        y_pos = 4.25 - i * 0.19
        ax.text(detail_x + 0.15, y_pos, field, fontsize=7, family='monospace',
                color=COLOR_PALETTE['gray_dark'])
        ax.text(detail_x + 2.3, y_pos, fmt, fontsize=7, color=COLOR_PALETTE['text_muted'])

    # Naming Convention Box
    ax.add_patch(FancyBboxPatch((detail_x, 0.3), 4.3, 1.7, boxstyle="round,pad=0.05",
                                facecolor=COLOR_PALETTE['light_bg'], edgecolor=COLOR_PALETTE['gray_med'], linewidth=1.5))
    ax.text(detail_x + 2.15, 1.75, 'File Naming Convention', fontsize=9, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    conventions = [
        ('MIT:', 'MIT_S{section}_{date}.csv'),
        ('ERT:', 'ERT_S{section}_{date}.csv'),
        ('Registry:', 'probe_registry.csv'),
        ('Log:', 'field_log_{date}.txt'),
    ]

    for i, (label, pattern) in enumerate(conventions):
        y_pos = 1.45 - i * 0.27
        ax.text(detail_x + 0.15, y_pos, label, fontsize=8, fontweight='bold',
                color=COLOR_PALETTE['secondary'])
        ax.text(detail_x + 1.0, y_pos, pattern, fontsize=8, family='monospace',
                color=COLOR_PALETTE['gray_dark'])

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLOR_PALETTE['success'], label='MIT Data'),
        mpatches.Patch(color=COLOR_PALETTE['orange'], label='ERT Data'),
        mpatches.Patch(color=COLOR_PALETTE['purple'], label='Probe Registry'),
        mpatches.Patch(color=COLOR_PALETTE['gray_med'], label='Field Log'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8,
              framealpha=0.9, bbox_to_anchor=(0.05, 0.02))

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 2: RECORDING WORKFLOW DIAGRAM
# ============================================================================
def create_recording_workflow_diagram():
    """
    Create a workflow diagram showing the data recording process
    from field setup through data storage and quality control.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(6, 6.7, 'HIRT Data Recording Workflow', fontsize=12, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    def draw_process_box(x, y, width, height, text, color, text_color='white'):
        """Helper to draw a process box with text."""
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.03",
                                    facecolor=color, edgecolor='black', linewidth=1.5))
        # Handle multi-line text
        lines = text.split('\n')
        line_height = 0.25 if len(lines) > 1 else 0
        for i, line in enumerate(lines):
            ax.text(x + width/2, y + height/2 + (len(lines)/2 - i - 0.5) * line_height,
                    line, fontsize=8, fontweight='bold', ha='center', va='center',
                    color=text_color)

    def draw_arrow(x1, y1, x2, y2, color='#4a5568'):
        """Helper to draw an arrow between points."""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # === Row 1: Field Setup ===
    row1_y = 5.3

    draw_process_box(0.3, row1_y, 2.0, 0.8, 'Site\nPreparation', COLOR_PALETTE['secondary'])
    draw_arrow(2.3, row1_y + 0.4, 2.8, row1_y + 0.4)

    draw_process_box(2.8, row1_y, 2.0, 0.8, 'Probe\nDeployment', COLOR_PALETTE['secondary'])
    draw_arrow(4.8, row1_y + 0.4, 5.3, row1_y + 0.4)

    draw_process_box(5.3, row1_y, 2.0, 0.8, 'System\nInitialization', COLOR_PALETTE['secondary'])
    draw_arrow(7.3, row1_y + 0.4, 7.8, row1_y + 0.4)

    draw_process_box(7.8, row1_y, 2.0, 0.8, 'Record\nMetadata', COLOR_PALETTE['accent'])
    draw_arrow(9.8, row1_y + 0.4, 10.3, row1_y + 0.4)

    draw_process_box(10.3, row1_y, 1.4, 0.8, 'Start\nLogging', COLOR_PALETTE['success'])

    # === Row 2: Data Acquisition Loop ===
    row2_y = 3.8

    # Loop indicator box
    ax.add_patch(FancyBboxPatch((0.1, row2_y - 0.3), 11.6, 1.4, boxstyle="round,pad=0.05",
                                facecolor='#f7fafc', edgecolor=COLOR_PALETTE['gray_med'],
                                linewidth=1, linestyle='--'))
    ax.text(0.3, row2_y + 0.85, 'Measurement Loop (per section)', fontsize=8,
            fontstyle='italic', color=COLOR_PALETTE['gray_dark'])

    draw_process_box(0.5, row2_y, 1.8, 0.8, 'Select\nProbe Pair', COLOR_PALETTE['primary'])
    draw_arrow(2.3, row2_y + 0.4, 2.6, row2_y + 0.4)

    draw_process_box(2.6, row2_y, 1.8, 0.8, 'MIT\nMeasurement', COLOR_PALETTE['success'])
    draw_arrow(4.4, row2_y + 0.4, 4.7, row2_y + 0.4)

    draw_process_box(4.7, row2_y, 1.8, 0.8, 'ERT\nMeasurement', COLOR_PALETTE['orange'])
    draw_arrow(6.5, row2_y + 0.4, 6.8, row2_y + 0.4)

    draw_process_box(6.8, row2_y, 1.6, 0.8, 'Record\nData', COLOR_PALETTE['accent'])
    draw_arrow(8.4, row2_y + 0.4, 8.7, row2_y + 0.4)

    # Decision diamond
    diamond_x, diamond_y = 9.5, row2_y + 0.4
    diamond = plt.Polygon([[diamond_x - 0.6, diamond_y],
                           [diamond_x, diamond_y + 0.4],
                           [diamond_x + 0.6, diamond_y],
                           [diamond_x, diamond_y - 0.4]],
                          facecolor=COLOR_PALETTE['warning'], edgecolor='black', linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y, 'More\npairs?', fontsize=7, ha='center', va='center',
            color='white', fontweight='bold')

    # Loop back arrow
    ax.annotate('', xy=(0.5, row2_y + 0.4), xytext=(diamond_x - 0.6, diamond_y),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['success'], lw=1.5,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(5, row2_y + 1.1, 'Yes', fontsize=7, color=COLOR_PALETTE['success'])

    # Exit arrow
    draw_arrow(diamond_x, diamond_y - 0.4, diamond_x, row2_y - 0.6, COLOR_PALETTE['warning'])
    ax.text(diamond_x + 0.2, row2_y - 0.15, 'No', fontsize=7, color=COLOR_PALETTE['warning'])

    # === Row 3: Data Storage ===
    row3_y = 1.8

    draw_process_box(1.0, row3_y, 2.0, 0.8, 'Quality\nCheck', COLOR_PALETTE['warning'])
    draw_arrow(3.0, row3_y + 0.4, 3.5, row3_y + 0.4)

    draw_process_box(3.5, row3_y, 2.0, 0.8, 'Add Field\nNotes', COLOR_PALETTE['gray_med'])
    draw_arrow(5.5, row3_y + 0.4, 6.0, row3_y + 0.4)

    # Storage options (parallel)
    draw_process_box(6.0, row3_y + 0.5, 1.8, 0.7, 'CSV\nExport', COLOR_PALETTE['secondary'])
    draw_process_box(6.0, row3_y - 0.4, 1.8, 0.7, 'Paper\nLog', COLOR_PALETTE['gray_dark'])

    draw_arrow(7.8, row3_y + 0.85, 8.5, row3_y + 0.4)
    draw_arrow(7.8, row3_y - 0.05, 8.5, row3_y + 0.4)

    draw_process_box(8.5, row3_y, 2.2, 0.8, 'Backup\nStorage', COLOR_PALETTE['primary'])

    # Connect row 2 to row 3
    draw_arrow(diamond_x, row2_y - 0.6, 2.0, row3_y + 0.8)

    # === Data Quality Notes Box ===
    ax.add_patch(FancyBboxPatch((0.3, 0.2), 4.5, 1.3, boxstyle="round,pad=0.05",
                                facecolor='#fff5f5', edgecolor=COLOR_PALETTE['warning'], linewidth=1.5))
    ax.text(2.55, 1.3, 'Quality Notes to Record', fontsize=9, fontweight='bold',
            ha='center', color=COLOR_PALETTE['warning'])

    quality_items = [
        'Soil moisture changes during survey',
        'Temperature variations',
        'Disturbances (vehicles, people)',
        'Equipment issues / anomalous readings',
    ]
    for i, item in enumerate(quality_items):
        ax.text(0.5, 1.0 - i * 0.2, f'* {item}', fontsize=7,
                color=COLOR_PALETTE['gray_dark'])

    # === Metadata Checklist Box ===
    ax.add_patch(FancyBboxPatch((5.2, 0.2), 3.2, 1.3, boxstyle="round,pad=0.05",
                                facecolor='#ebf8ff', edgecolor=COLOR_PALETTE['accent'], linewidth=1.5))
    ax.text(6.8, 1.3, 'Required Metadata', fontsize=9, fontweight='bold',
            ha='center', color=COLOR_PALETTE['accent'])

    metadata_items = [
        'Site name / GPS coordinates',
        'Survey date/time, team members',
        'Weather, soil type/moisture',
        'Probe spacing, insertion depths',
    ]
    for i, item in enumerate(metadata_items):
        ax.text(5.4, 1.0 - i * 0.2, f'* {item}', fontsize=7,
                color=COLOR_PALETTE['gray_dark'])

    # === Legend for process types ===
    ax.add_patch(FancyBboxPatch((8.8, 0.2), 2.9, 1.3, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor=COLOR_PALETTE['gray_med'], linewidth=1))
    ax.text(10.25, 1.3, 'Process Types', fontsize=9, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    legend_items = [
        (COLOR_PALETTE['secondary'], 'Setup'),
        (COLOR_PALETTE['success'], 'MIT Data'),
        (COLOR_PALETTE['orange'], 'ERT Data'),
        (COLOR_PALETTE['accent'], 'Recording'),
    ]
    for i, (color, label) in enumerate(legend_items):
        ax.add_patch(Rectangle((9.0, 1.0 - i * 0.2 - 0.05), 0.3, 0.15, facecolor=color))
        ax.text(9.4, 1.0 - i * 0.2, label, fontsize=7, va='center',
                color=COLOR_PALETTE['gray_dark'])

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
def build_section_11():
    """Build the Section 11: Data Recording PDF."""

    print("Generating Section 11: Data Recording...")

    # Create builder
    builder = SectionPDFBuilder(
        section_num=11,
        title="Data Recording"
    )

    # Generate figures
    print("  Creating file structure diagram...")
    fig_file_structure = create_file_structure_diagram()

    print("  Creating recording workflow diagram...")
    fig_workflow = create_recording_workflow_diagram()

    # === Title Block ===
    builder.add_title_block(
        subtitle="Data Formats, File Organization, and Logging Procedures"
    )

    # === Overview ===
    builder.add_section_header("11.1 Overview")
    builder.add_body_text(
        "This section specifies the data formats and organization for HIRT field measurements. "
        "Consistent data recording ensures reliable post-processing and long-term data management. "
        "The HIRT system generates two primary data types: Magnetic Induction Tomography (MIT)"
        "measurements and Electrical Resistivity Tomography (ERT) measurements, each requiring "
        "specific record formats to capture all relevant parameters.",
        first_paragraph=True
    )
    builder.add_body_text(
        "All data files use CSV (Comma-Separated Values) format for maximum compatibility with "
        "analysis software. Field logs supplement electronic records with contextual information "
        "that may affect data interpretation. The combination of structured electronic records "
        "and detailed field notes enables comprehensive quality assessment during post-processing."
    )

    # === File Structure Diagram ===
    builder.add_figure(
        fig_file_structure,
        "HIRT data file organization showing directory structure, file naming conventions, "
        "and record field definitions for MIT and ERT data types. Each survey date creates "
        "a separate subdirectory containing section-specific measurement files.",
        width=CONTENT_WIDTH
    )

    # === MIT Record Format ===
    builder.add_section_header("11.2 MIT Record Format")
    builder.add_body_text(
        "Each MIT measurement records the electromagnetic coupling between a transmitter-receiver "
        "probe pair. The record captures both the measurement geometry (which probes, which zone) "
        "and the signal parameters (amplitude, phase, frequency) needed for tomographic reconstruction.",
        first_paragraph=True
    )

    mit_table_data = [
        ['Field', 'Description', 'Units/Format'],
        ['timestamp', 'Measurement time', 'ISO 8601 or Unix'],
        ['section_id', 'Survey section identifier', 'String (e.g., "S01")'],
        ['zone_id', 'Zone Hub identifier', 'String (e.g., "ZA")'],
        ['tx_probe_id', 'Transmitting probe ID', 'String (e.g., "P01")'],
        ['rx_probe_id', 'Receiving probe ID', 'String (e.g., "P02")'],
        ['freq_hz', 'Measurement frequency', 'Hz'],
        ['amp', 'Signal amplitude', 'V or normalized'],
        ['phase_deg', 'Phase angle', 'Degrees'],
        ['tx_current_mA', 'TX coil current', 'mA'],
    ]
    builder.add_table(mit_table_data,
                      col_widths=[CONTENT_WIDTH*0.22, CONTENT_WIDTH*0.45, CONTENT_WIDTH*0.33],
                      caption="MIT record format fields")

    # === ERT Record Format ===
    builder.add_section_header("11.3 ERT Record Format")
    builder.add_body_text(
        "ERT measurements record the voltage response to injected current, capturing the "
        "four-electrode geometry (two injection electrodes, one sense electrode) and the "
        "resulting potential. Polarity reversal measurements are recorded separately to "
        "enable electrode polarization correction during processing.",
        first_paragraph=True
    )

    ert_table_data = [
        ['Field', 'Description', 'Units/Format'],
        ['timestamp', 'Measurement time', 'ISO 8601'],
        ['section_id', 'Survey section identifier', 'String'],
        ['zone_id', 'Zone Hub identifier', 'String'],
        ['inject_pos_id', 'Positive current probe ID', 'String'],
        ['inject_neg_id', 'Negative current probe ID', 'String'],
        ['sense_id', 'Voltage sensing probe ID', 'String'],
        ['volt_mV', 'Measured voltage', 'mV'],
        ['current_mA', 'Injected current', 'mA'],
        ['polarity', 'Current direction', '+1 or -1'],
        ['notes', 'Additional notes', 'Free text'],
    ]
    builder.add_table(ert_table_data,
                      col_widths=[CONTENT_WIDTH*0.22, CONTENT_WIDTH*0.45, CONTENT_WIDTH*0.33],
                      caption="ERT record format fields")

    builder.add_section_header("Example ERT Record", level=3)
    builder.add_code_block(
        "timestamp,section_id,inject_pos_probe_id,inject_neg_probe_id,sense_probe_id,volt_mV,current_mA,polarity,notes\n"
        "2024-03-15T10:45:12Z,S01,P01,P20,P05,12.5,1.2,+1,\n"
        "2024-03-15T10:45:13Z,S01,P01,P20,P06,8.3,1.2,+1,\n"
        "2024-03-15T10:45:14Z,S01,P01,P20,P05,-12.4,1.2,-1,reversed polarity"
    )

    # === Probe Registry ===
    builder.add_section_header("11.4 Probe Registry")
    builder.add_body_text(
        "Each probe in the HIRT system has a registry entry documenting its physical "
        "characteristics and calibration status. The probe registry is a shared file "
        "that applies across all surveys and enables consistent data interpretation "
        "even when probes are swapped between deployments.",
        first_paragraph=True
    )

    registry_table_data = [
        ['Field', 'Description', 'Units/Format'],
        ['probe_id', 'Unique probe identifier', 'String'],
        ['coil_L_mH', 'TX coil inductance', 'mH'],
        ['coil_Q', 'Coil Q factor', 'Dimensionless'],
        ['rx_gain_dB', 'RX amplifier gain', 'dB'],
        ['ring_depths_m', 'ERT ring depths', 'm (comma-separated)'],
        ['firmware_rev', 'Firmware version', 'String'],
        ['calibration_date', 'Last calibration date', 'YYYY-MM-DD'],
        ['notes', 'Additional notes', 'Free text'],
    ]
    builder.add_table(registry_table_data,
                      col_widths=[CONTENT_WIDTH*0.25, CONTENT_WIDTH*0.42, CONTENT_WIDTH*0.33],
                      caption="Probe registry fields")

    builder.add_section_header("Example Probe Registry", level=3)
    builder.add_code_block(
        "probe_id,coil_L_mH,coil_Q,rx_gain_dB,ring_depths_m,firmware_rev,calibration_date,notes\n"
        "P01,1.2,25,40,0.5,1.5,v1.2,2024-03-10,\n"
        "P02,1.15,28,40,0.5,1.5,v1.2,2024-03-10,\n"
        "P03,1.18,26,40,0.5,1.5,v1.2,2024-03-10,"
    )

    # === Recording Workflow ===
    builder.add_section_header("11.5 Recording Workflow")
    builder.add_body_text(
        "The HIRT data recording workflow follows a systematic process from site preparation "
        "through data backup. Each measurement loop captures MIT and ERT data for all probe "
        "pairs in a section before moving to the next section. Quality checks at multiple "
        "stages ensure data integrity before final storage.",
        first_paragraph=True
    )

    builder.add_figure(
        fig_workflow,
        "Data recording workflow from field setup through storage. The measurement loop "
        "iterates through all probe pairs within each section, recording both MIT and ERT "
        "data before advancing. Quality checks and field notes are captured before final "
        "backup to multiple storage locations.",
        width=CONTENT_WIDTH
    )

    # === Data Storage ===
    builder.add_section_header("11.6 Data Storage")

    builder.add_section_header("File Organization", level=3)
    builder.add_bullet_list([
        "<b>One CSV file per section</b> for MIT data",
        "<b>One CSV file per section</b> for ERT data",
        "<b>One registry file</b> for all probes (shared across surveys)",
        "<b>Paper log</b> for conditions and notes (backup)",
    ])

    builder.add_section_header("Naming Convention", level=3)
    builder.add_bullet_list([
        "MIT: <font face='Courier'>MIT_S{section_id}_{date}.csv</font>",
        "ERT: <font face='Courier'>ERT_S{section_id}_{date}.csv</font>",
        "Registry: <font face='Courier'>probe_registry.csv</font>",
        "Field Log: <font face='Courier'>field_log_{date}.txt</font>",
    ])

    # === Metadata ===
    builder.add_section_header("11.7 Metadata Requirements")

    builder.add_section_header("Site Information", level=3)
    builder.add_bullet_list([
        "Site name and GPS coordinates",
        "Survey date and time (start/end)",
        "Team members present",
        "Weather conditions (temperature, precipitation, wind)",
        "Soil type and estimated moisture content",
        "Site access notes and restrictions",
    ])

    builder.add_section_header("Measurement Parameters", level=3)
    builder.add_bullet_list([
        "Frequency list for MIT measurements",
        "Current levels for ERT measurements",
        "Probe spacing configuration",
        "Insertion depths achieved",
        "Grid coordinates and orientation",
    ])

    # === Data Quality Notes ===
    builder.add_section_header("11.8 Data Quality Notes")
    builder.add_body_text(
        "Field conditions that may affect data quality should be recorded in the paper log "
        "or the notes field of individual records. These annotations are critical for "
        "proper interpretation during post-processing and help identify measurements that "
        "may require special handling or exclusion from analysis.",
        first_paragraph=True
    )

    builder.add_info_box("Record in Paper Log or Notes Field", [
        "Soil moisture changes during survey",
        "Temperature variations (especially for long surveys)",
        "External disturbances (vehicles, people, machinery)",
        "Equipment issues or malfunctions",
        "Anomalous readings with contextual explanation",
        "Probe insertion difficulties or obstructions",
    ])

    # === Best Practices ===
    builder.add_section_header("11.9 Best Practices")

    builder.add_numbered_list([
        "<b>Verify file integrity</b> after each section by checking record counts match expected probe pair combinations.",
        "<b>Backup data</b> to at least two locations before leaving the field site.",
        "<b>Timestamp all entries</b> using ISO 8601 format for unambiguous date/time parsing.",
        "<b>Document probe positions</b> with photographs showing grid layout and reference markers.",
        "<b>Record baseline readings</b> before and after each survey session to detect drift.",
        "<b>Note environmental changes</b> including passing weather fronts, irrigation events, or traffic patterns.",
    ])

    builder.add_warning_box("Data Integrity Warning", [
        "Never modify raw data files after collection",
        "Create processed copies for any transformations",
        "Maintain chain of custody documentation for forensic applications",
        "Store original files in read-only archive locations",
    ])

    # === Software Compatibility Roadmap ===
    builder.add_section_header("11.10 Software Compatibility Roadmap")
    builder.add_body_text(
        "The CSV data formats described above are designed as raw intermediate storage. "
        "Future software tools will provide import scripts to convert these raw logs into "
        "standard formats for open-source inversion frameworks:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "<b>EIDORS (Matlab/Octave):</b> Import script will generate .mat structures for MIT-3D finite element reconstruction.",
        "<b>pyGIMLi (Python):</b> Converters will map ERT data to the Unified Data Format for coupled inversion.",
        "<b>ResIPy:</b> Direct import support for ERT .csv files is planned."
    ])

    # Build the PDF
    output_path = builder.build()
    print(f"  Output: {output_path}")
    return output_path


if __name__ == "__main__":
    build_section_11()
