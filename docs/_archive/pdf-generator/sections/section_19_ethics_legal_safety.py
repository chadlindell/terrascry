#!/usr/bin/env python3
"""
HIRT PDF Generator - Section 19: Ethics, Legal, and Safety

Generates a publication-quality PDF covering HIRT safety protocols,
legal considerations, and ethical guidelines for UXO sites.

Usage:
    python section_19_ethics_legal_safety.py

Output:
    ../output/sections/19-ethics-legal-safety.pdf
"""

import os
import sys
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon
import numpy as np

# Add parent directory to path for lib package imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from lib import SectionPDFBuilder, CONTENT_WIDTH, COLOR_PALETTE, WARNING, SUCCESS, ACCENT, PRIMARY
from lib.diagrams.flowcharts import create_safety_checklist_visual

# Color constants for diagrams
COLORS = COLOR_PALETTE


def create_safety_protocol_flowchart():
    """
    Create UXO site safety protocol decision flowchart.

    Shows the step-by-step decision process for safe operations at UXO sites.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    def draw_terminal(x, y, width, height, text, color):
        patch = FancyBboxPatch((x - width/2, y - height/2), width, height,
                               boxstyle="round,pad=0.3",
                               facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(patch)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
        return patch

    def draw_process(x, y, width, height, text, color):
        patch = FancyBboxPatch((x - width/2, y - height/2), width, height,
                               boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(patch)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, wrap=True)
        return patch

    def draw_decision(x, y, size, text, color):
        half = size / 2
        points = [(x, y + half), (x + half, y), (x, y - half), (x - half, y)]
        patch = Polygon(points, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(patch)
        ax.text(x, y, text, ha='center', va='center', fontsize=7)
        return patch

    def draw_arrow(start, end, label=None, label_offset=(0.1, 0)):
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        if label:
            mid_x = (start[0] + end[0]) / 2 + label_offset[0]
            mid_y = (start[1] + end[1]) / 2 + label_offset[1]
            ax.text(mid_x, mid_y, label, fontsize=7, fontweight='bold')

    # Title
    ax.text(5, 9.7, 'UXO Site Safety Protocol Flowchart', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Start
    draw_terminal(5, 9, 2.5, 0.5, 'SITE ARRIVAL', '#c6f6d5')

    # Decision 1: EOD Clearance
    draw_arrow((5, 8.75), (5, 8.2))
    draw_decision(5, 7.7, 1.4, 'EOD\nCleared?', '#ebf8ff')

    # No EOD -> Wait
    draw_arrow((4.3, 7.7), (2.5, 7.7), 'NO')
    draw_process(1.5, 7.7, 1.8, 0.8, 'Request EOD\nSweep', '#feebc8')
    draw_arrow((1.5, 7.3), (1.5, 6.5))
    draw_terminal(1.5, 6.1, 1.6, 0.5, 'WAIT', '#fed7d7')

    # Yes EOD -> Perimeter
    draw_arrow((5, 7), (5, 6.4), 'YES')
    draw_process(5, 6, 2.0, 0.7, 'Establish 100m\nExclusion Zone', '#ebf8ff')

    # Decision 2: Perimeter secure
    draw_arrow((5, 5.65), (5, 5.1))
    draw_decision(5, 4.6, 1.4, 'Perimeter\nSecure?', '#ebf8ff')

    # No -> Mark boundary
    draw_arrow((5.7, 4.6), (7.5, 4.6), 'NO')
    draw_process(8.5, 4.6, 1.8, 0.7, 'Mark\nBoundary', '#feebc8')
    draw_arrow((8.5, 4.25), (8.5, 3.7))
    draw_arrow((8.5, 3.7), (6.7, 3.7))

    # Yes -> Radio check
    draw_arrow((5, 3.9), (5, 3.4), 'YES')
    draw_process(5, 3, 2.0, 0.7, 'Radio Check &\nComms Test', '#ebf8ff')

    # Decision 3: Conductivity check
    draw_arrow((5, 2.65), (5, 2.1))
    draw_decision(5, 1.6, 1.4, 'Cond.\n<5500?', '#ebf8ff')

    # No -> Evacuate
    draw_arrow((5.7, 1.6), (7.5, 1.6), 'NO')
    draw_process(8.5, 1.6, 1.8, 0.7, 'EVACUATE\nConsult EOD', '#fed7d7')

    # Yes -> Proceed
    draw_arrow((5, 0.9), (5, 0.4), 'YES')
    draw_terminal(5, 0.1, 2.5, 0.5, 'SAFE TO PROCEED', '#c6f6d5')

    # Side note box
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 2.8, 2.0,
                                boxstyle="round,pad=0.02",
                                facecolor='#fff5f5', edgecolor=COLORS['warning'], linewidth=1.5))
    ax.text(1.7, 2.0, 'CRITICAL RULES', ha='center', fontsize=8, fontweight='bold', color=COLORS['warning'])
    ax.text(0.5, 1.6, '\u2022 NO hammering probes', fontsize=7, va='top')
    ax.text(0.5, 1.3, '\u2022 Soft insertion only', fontsize=7, va='top')
    ax.text(0.5, 1.0, '\u2022 Min personnel in zone', fontsize=7, va='top')
    ax.text(0.5, 0.7, '\u2022 Constant radio contact', fontsize=7, va='top')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_risk_assessment_matrix():
    """
    Create conductivity-based risk assessment matrix for UXO sites.

    Shows the traffic-light system for conductivity thresholds
    and corresponding actions.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(4.5, 5.7, 'Conductivity Risk Assessment Matrix', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Risk levels - color-coded boxes
    levels = [
        (1.5, 4.2, 'GREEN', '<3,000', 'SAFE', 'Normal operations\npermitted', '#c6f6d5', COLORS['success']),
        (4.5, 4.2, 'YELLOW', '3,000-5,500', 'CAUTION', 'Increased monitoring\nLimit insertion depth', '#fefcbf', COLORS['orange']),
        (7.5, 4.2, 'RED', '>5,500', 'HALT', 'STOP all insertion\nEvacuate to safe zone\nConsult EOD', '#fed7d7', COLORS['warning']),
    ]

    for x, y, status, threshold, action, description, bg_color, border_color in levels:
        # Main box
        box_width, box_height = 2.4, 2.8
        ax.add_patch(FancyBboxPatch((x - box_width/2, y - box_height/2), box_width, box_height,
                                    boxstyle="round,pad=0.02",
                                    facecolor=bg_color, edgecolor=border_color, linewidth=2))

        # Status header
        ax.add_patch(FancyBboxPatch((x - box_width/2 + 0.1, y + box_height/2 - 0.55),
                                    box_width - 0.2, 0.45,
                                    boxstyle="round,pad=0.02",
                                    facecolor=border_color, edgecolor='none'))
        ax.text(x, y + box_height/2 - 0.35, status, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

        # Threshold value
        ax.text(x, y + 0.5, f'{threshold}', ha='center', va='center',
               fontsize=11, fontweight='bold', color=border_color)
        ax.text(x, y + 0.1, 'uS/cm', ha='center', va='center',
               fontsize=8, color=COLORS['gray_dark'])

        # Action
        ax.text(x, y - 0.4, action, ha='center', va='center',
               fontsize=9, fontweight='bold', color=COLORS['gray_dark'])

        # Description
        ax.text(x, y - 1.0, description, ha='center', va='center',
               fontsize=7, color=COLORS['gray_dark'])

    # Legend / Scientific basis
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 8.4, 1.2,
                                boxstyle="round,pad=0.02",
                                facecolor='#f7fafc', edgecolor=COLORS['gray_med'], linewidth=1))
    ax.text(4.5, 1.25, 'Scientific Basis', ha='center', fontsize=9, fontweight='bold', color=COLORS['primary'])
    ax.text(4.5, 0.85, 'Field experience indicates conductivity >6,000 uS/cm at WWII bomb craters', ha='center', fontsize=8)
    ax.text(4.5, 0.55, 'signals chemical activation risk, potentially leading to spontaneous explosions.', ha='center', fontsize=8)

    # Arrows between levels
    ax.annotate('', xy=(2.9, 4.2), xytext=(2.45, 4.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_med'], lw=2))
    ax.annotate('', xy=(5.95, 4.2), xytext=(5.5, 4.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_med'], lw=2))

    ax.text(2.67, 4.4, 'Rising', fontsize=7, color=COLORS['gray_med'], ha='center')
    ax.text(5.72, 4.4, 'Rising', fontsize=7, color=COLORS['gray_med'], ha='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_rim_deployment_diagram():
    """
    Create rim-only deployment diagram for high-risk craters.

    Shows proper probe placement around crater perimeter
    avoiding the central exclusion zone.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(4, 5.7, 'Rim-Only Deployment Pattern', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['primary'])
    ax.text(4, 5.35, 'For High-Risk Craters with Suspected UXB', ha='center',
           fontsize=9, style='italic', color=COLORS['gray_dark'])

    # Crater outline
    crater_center = (4, 2.8)
    crater_radius = 1.8
    crater = Circle(crater_center, crater_radius, facecolor='#fef3c7',
                   edgecolor=COLORS['orange'], linewidth=2, linestyle='--')
    ax.add_patch(crater)

    # Exclusion zone (inner circle)
    exclusion_radius = 1.0
    exclusion = Circle(crater_center, exclusion_radius, facecolor='#fed7d7',
                      edgecolor=COLORS['warning'], linewidth=2)
    ax.add_patch(exclusion)
    ax.text(crater_center[0], crater_center[1], 'EXCLUSION\nZONE', ha='center', va='center',
           fontsize=8, fontweight='bold', color=COLORS['warning'])

    # Probe positions around rim (8 positions)
    n_probes = 8
    probe_radius = 2.3  # Distance from crater center
    for i in range(n_probes):
        angle = 2 * np.pi * i / n_probes - np.pi/2  # Start from top
        x = crater_center[0] + probe_radius * np.cos(angle)
        y = crater_center[1] + probe_radius * np.sin(angle)

        # Probe marker
        probe = Circle((x, y), 0.12, facecolor=COLORS['secondary'], edgecolor='black', linewidth=1)
        ax.add_patch(probe)
        ax.text(x, y, 'X', ha='center', va='center', fontsize=6, fontweight='bold', color='white')

    # Legend
    ax.add_patch(FancyBboxPatch((0.2, 0.2), 3.0, 1.4,
                                boxstyle="round,pad=0.02",
                                facecolor='white', edgecolor=COLORS['gray_med'], linewidth=1))
    ax.text(1.7, 1.35, 'Legend', ha='center', fontsize=9, fontweight='bold', color=COLORS['primary'])

    # Legend items
    probe_legend = Circle((0.5, 1.0), 0.1, facecolor=COLORS['secondary'], edgecolor='black')
    ax.add_patch(probe_legend)
    ax.text(0.75, 1.0, 'Probe positions', fontsize=7, va='center')

    ax.add_patch(Rectangle((0.4, 0.55), 0.2, 0.2, facecolor='#fed7d7', edgecolor=COLORS['warning']))
    ax.text(0.75, 0.65, 'No insertion zone', fontsize=7, va='center')

    # Benefits box
    ax.add_patch(FancyBboxPatch((4.8, 0.2), 2.9, 1.4,
                                boxstyle="round,pad=0.02",
                                facecolor='#c6f6d5', edgecolor=COLORS['success'], linewidth=1))
    ax.text(6.25, 1.35, 'Benefits', ha='center', fontsize=9, fontweight='bold', color=COLORS['success'])
    ax.text(5.0, 1.0, '\u2022 3D reconstruction still possible', fontsize=7, va='center')
    ax.text(5.0, 0.7, '\u2022 Safe standoff maintained', fontsize=7, va='center')
    ax.text(5.0, 0.4, '\u2022 Disturbance: ~12-15L (vs 150L)', fontsize=7, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def build_section_19():
    """
    Build Section 19: Ethics, Legal, and Safety PDF.

    Returns:
        Path to the generated PDF file
    """
    # Initialize builder
    builder = SectionPDFBuilder(
        section_num=19,
        title="Ethics, Legal, and Safety"
    )

    # Title block
    builder.add_title_block(
        subtitle="Safety Protocols for UXO Sites, Legal Compliance, and Ethical Guidelines",
        version="2.0"
    )

    # Introduction
    builder.add_section_header("19.1 Introduction", level=1)
    builder.add_body_text(
        "HIRT deployment at sensitive archaeological, forensic, or humanitarian demining sites "
        "requires strict adherence to safety protocols, legal requirements, and ethical guidelines. "
        "This section provides comprehensive guidance for safe and responsible operations, with "
        "particular emphasis on unexploded ordnance (UXO) site safety.",
        first_paragraph=True
    )

    # Human Remains Section
    builder.add_section_header("19.2 Human Remains Protocol", level=1)
    builder.add_body_text(
        "HIRT surveys at sites potentially containing human remains must be treated as "
        "<b>forensic/archaeological contexts</b>. The following requirements apply:",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "Obtain all <b>permits and permissions</b> before deployment",
        "Follow jurisdictional requirements (heritage boards, war graves authorities)",
        "Maintain proper chain of custody for any findings",
        "Document all activities thoroughly with photographs and written records",
        "Coordinate with relevant authorities before, during, and after surveys"
    ])

    # UXO Risk Section
    builder.add_section_header("19.3 UXO Risk Assessment", level=1)
    builder.add_warning_box("CRITICAL WARNING", [
        "WWII sites can contain live ordnance",
        "Standard insertion methodology is DANGEROUS at UXO sites",
        "Never drive or hammer probes until area is cleared by qualified EOD/UXO professionals"
    ])

    builder.add_section_header("19.3.1 Pre-Survey Requirements", level=2)
    builder.add_body_text(
        "Before any HIRT deployment at a suspected UXO site, the following requirements "
        "must be satisfied:",
        first_paragraph=True
    )

    builder.add_numbered_list([
        "Coordinate with explosive ordnance disposal (EOD) teams",
        "Obtain professional EOD sweep clearance for the survey area",
        "Follow established UXO clearance protocols for the jurisdiction",
        "Maintain safe standoff distances during all operations",
        "Establish clear emergency evacuation procedures"
    ])

    builder.add_section_header("19.3.2 Critical Problems with Standard Procedures", level=2)
    builder.add_body_text(
        "The standard HIRT insertion methodology (\"drive pilot rod to depth, wiggle to "
        "12-14 mm, remove\") presents unacceptable risks at UXO sites:",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "<b>Mechanical trigger risk:</b> Driving a steel rod into ground containing potential ordnance could trigger detonation",
        "<b>No risk analysis:</b> Standard procedures lack risk analysis for probe insertion near UXB",
        "<b>Electrical interaction:</b> ERT current injection (0.5-2 mA) could theoretically interact with sensitive fuzes",
        "<b>Transient hazards:</b> No overvoltage protection specified (12V/0.5 ohm = 24A transient on short)"
    ])

    # Safety Protocol Flowchart
    builder.add_figure(
        create_safety_protocol_flowchart(),
        "UXO site safety protocol decision flowchart. Each survey must complete all safety "
        "checks before proceeding with probe insertion."
    )

    # Required Safety Measures
    builder.add_section_header("19.3.3 Required Safety Measures for UXO Sites", level=2)

    safety_data = [
        ["Protocol", "Action"],
        ["Pre-survey EOD clearance", "Professional EOD sweep before ANY insertion"],
        ["Safe standoff perimeter", "Minimum 100m exclusion zone during insertion"],
        ["Soft insertion only", "Hand auger or water-jet only (NO hammering)"],
        ["Conductivity pre-check", "Check soil conductivity before deep insertion"],
        ["Personnel limits", "Minimum personnel in hazard zone"],
        ["Communications", "Constant radio contact with safe zone"]
    ]
    builder.add_table(
        safety_data,
        col_widths=[CONTENT_WIDTH * 0.35, CONTENT_WIDTH * 0.65],
        caption="Required safety measures for UXO site operations"
    )

    # Risk Assessment Matrix
    builder.add_section_header("19.4 Conductivity Threshold Monitoring", level=1)
    builder.add_body_text(
        "Research (Waga et al. 2026) indicates that conductivity >6,000 uS/cm at WWII bomb "
        "craters signals chemical activation risk, potentially leading to spontaneous explosions. "
        "HIRT's ERT-Lite system provides continuous conductivity monitoring capability.",
        first_paragraph=True
    )

    builder.add_figure(
        create_risk_assessment_matrix(),
        "Conductivity-based risk assessment matrix. Operations must halt immediately if "
        "conductivity exceeds 5,500 uS/cm."
    )

    builder.add_section_header("19.4.1 Safety Thresholds", level=2)

    threshold_data = [
        ["Conductivity", "Status", "Action"],
        ["<3,000 uS/cm", "GREEN (Safe)", "Normal operations"],
        ["3,000-5,500 uS/cm", "YELLOW (Caution)", "Increased monitoring, limit insertion depth"],
        [">5,500 uS/cm", "RED (Halt)", "STOP all insertion, evacuate to safe zone, consult EOD"]
    ]
    builder.add_table(
        threshold_data,
        col_widths=[CONTENT_WIDTH * 0.3, CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.45],
        caption="Conductivity safety thresholds for UXO site operations"
    )

    builder.add_section_header("19.4.2 Time-Lapse Monitoring Schedule", level=2)
    builder.add_body_text(
        "For extended UXO site investigations, implement the following monitoring schedule:",
        first_paragraph=True
    )

    monitoring_data = [
        ["Phase", "Timing", "Metrics", "Action"],
        ["Baseline", "Day 0", "Full MIT + ERT survey", "Establish reference values"],
        ["Early detection", "Day 7-14", "Conductivity only", "Alert if >10% change"],
        ["Long-term", "Monthly", "Full ERT", "Track redox boundary evolution"],
        ["Pre-excavation", "24-48h before", "Full survey", "Final safety check"]
    ]
    builder.add_table(
        monitoring_data,
        col_widths=[CONTENT_WIDTH * 0.2, CONTENT_WIDTH * 0.2, CONTENT_WIDTH * 0.3, CONTENT_WIDTH * 0.3],
        caption="Recommended time-lapse monitoring schedule for UXO sites"
    )

    # Rim-Only Deployment
    builder.add_section_header("19.5 Rim-Only Deployment", level=1)
    builder.add_body_text(
        "For craters with suspected UXB at center, deploy probes <b>around the perimeter only</b>. "
        "This approach maintains survey geometry while avoiding direct insertion over suspected ordnance.",
        first_paragraph=True
    )

    builder.add_figure(
        create_rim_deployment_diagram(),
        "Rim-only deployment pattern for high-risk craters. Probes are positioned around "
        "the perimeter, maintaining safe standoff from the central exclusion zone."
    )

    builder.add_info_box("HIRT Early Warning Capability", [
        "Volumetric 3D resistivity (vs. single-point measurements)",
        "Gradient detection identifies active redox boundaries",
        "Non-magnetic operation safe near UXB",
        "\"Set Once; Measure Many\" enables time-lapse without repeated insertion"
    ])

    # Minimal Intrusion
    builder.add_section_header("19.6 Minimal Intrusion Principles", level=1)
    builder.add_body_text(
        "HIRT operations should always prioritize minimal site disturbance. The following "
        "principles guide responsible deployment:",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "Prefer <b>rim/perimeter probing</b> over direct insertion into suspected areas",
        "Use <b>shallow depths</b> when possible to achieve survey objectives",
        "Employ <b>pilot holes</b> to minimize ground disturbance volume",
        "<b>Avoid inserting probes into suspected burial voids</b> without explicit authorization",
        "Document all probe insertion points and depths for site records"
    ])

    # Safety Checklist Visual
    builder.add_section_header("19.7 Field Safety Checklist", level=1)
    builder.add_body_text(
        "The following checklist must be completed before commencing any HIRT survey at "
        "a UXO-suspected site. All critical items (marked with a star) are mandatory.",
        first_paragraph=True
    )

    builder.add_figure(
        create_safety_checklist_visual(),
        "UXO site safety checklist. All critical items (marked with star) must be verified "
        "before operations commence."
    )

    # General Field Safety
    builder.add_section_header("19.8 General Field Safety", level=1)
    builder.add_body_text(
        "Beyond UXO-specific protocols, all HIRT field operations must adhere to general "
        "safety practices:",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "Maintain clear communication protocols between all team members",
        "Use appropriate personal protective equipment (PPE) for site conditions",
        "Have emergency contact information readily available and posted",
        "Follow local environmental regulations for equipment deployment",
        "Respect site boundaries and access restrictions at all times",
        "Conduct daily equipment safety checks before deployment",
        "Maintain first aid kit and emergency supplies on site"
    ])

    # Recommended Protocol Summary
    builder.add_section_header("19.9 Summary: Recommended Protocol", level=1)
    builder.add_body_text(
        "For UXO sites requiring time-lapse monitoring, implement the following protocol:",
        first_paragraph=True
    )

    builder.add_numbered_list([
        "Baseline ERT survey of crater perimeter (Day 0)",
        "Weekly monitoring of conductivity trends",
        "Alert threshold set at >10% change from baseline",
        "Halt excavation immediately at >5,500 uS/cm",
        "Consult EOD team for any anomalous readings"
    ])

    # References
    builder.add_references([
        "[1] OSCE Guidelines for Humanitarian Demining Operations (2024). Organization for "
        "Security and Co-operation in Europe.",
        "[2] IMAS 07.11: Land Release. International Mine Action Standards, 3rd Edition (2023).",
        "[3] ASTM D7128-18: Standard Guide for Using the Seismic-Reflection Method for "
        "Shallow Subsurface Investigation.",
        "[4] US Army Corps of Engineers (2018). Engineering and Design: Ordnance and Explosives "
        "Response. EM 200-1-15."
    ])

    # Build and return path
    return builder.build()


if __name__ == "__main__":
    output_path = build_section_19()
    print(f"\nSection 19 PDF generated successfully:")
    print(f"  {output_path}")
