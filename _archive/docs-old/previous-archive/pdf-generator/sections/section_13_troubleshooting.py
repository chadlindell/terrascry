#!/usr/bin/env python3
"""
HIRT Section 13: Troubleshooting - PDF Generator

Generates a publication-quality PDF covering diagnostic approaches
and solutions for common HIRT field operation issues.

Usage:
    python section_13_troubleshooting.py

Output:
    output/sections/13-troubleshooting.pdf
"""

import os
import sys
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from lib.pdf_builder import SectionPDFBuilder
from lib.diagrams.flowcharts import create_troubleshooting_flowchart

# Colors for diagrams
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


def draw_process_box(ax, x, y, width, height, text, color='white', fontsize=8):
    """Draw a process box (rectangle with rounded corners)."""
    patch = FancyBboxPatch((x - width/2, y - height/2), width, height,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(patch)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            wrap=True)
    return patch


def draw_decision_diamond(ax, x, y, size, text, color='white'):
    """Draw a decision diamond."""
    half = size / 2
    points = [(x, y + half), (x + half, y), (x, y - half), (x - half, y)]
    patch = Polygon(points, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(patch)
    ax.text(x, y, text, ha='center', va='center', fontsize=7)
    return patch


def draw_terminal(ax, x, y, width, height, text, color='white'):
    """Draw a terminal/start/end box (pill shape)."""
    patch = FancyBboxPatch((x - width/2, y - height/2), width, height,
                           boxstyle="round,pad=0.3",
                           facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(patch)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    return patch


def draw_flow_arrow(ax, start, end, label=None, color='black', label_offset=(0.1, 0)):
    """Draw a flow arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    if label:
        mid_x = (start[0] + end[0]) / 2 + label_offset[0]
        mid_y = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mid_x, mid_y, label, fontsize=7, color=color)


def create_diagnostic_decision_tree():
    """
    Create comprehensive diagnostic decision tree for HIRT troubleshooting.

    This diagram covers the complete diagnostic workflow from symptom
    identification through resolution.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(7, 9.7, 'HIRT Diagnostic Decision Tree', ha='center',
            fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Start terminal
    draw_terminal(ax, 7, 9, 2.5, 0.5, 'ISSUE DETECTED', COLORS['light_red'])

    # Level 1: Initial Assessment
    draw_decision_diamond(ax, 7, 7.8, 1.4, 'Power\nLED On?', COLORS['light_blue'])
    draw_flow_arrow(ax, (7, 8.75), (7, 8.5))

    # Power OFF path (left branch)
    draw_flow_arrow(ax, (6.3, 7.8), (4, 7.8), 'No')
    draw_process_box(ax, 2.5, 7.8, 2.2, 0.7, 'Check Battery\n>12.0V?', COLORS['light_orange'])
    draw_flow_arrow(ax, (2.5, 7.45), (2.5, 6.8))
    draw_decision_diamond(ax, 2.5, 6.3, 1.2, 'Battery\nOK?', COLORS['light_blue'])

    # Battery bad
    draw_flow_arrow(ax, (1.9, 6.3), (0.8, 6.3), 'No')
    draw_process_box(ax, 0.8, 5.5, 1.4, 0.6, 'Charge or\nReplace', COLORS['light_green'])

    # Battery OK, check fuse
    draw_flow_arrow(ax, (2.5, 5.7), (2.5, 5.0), 'Yes')
    draw_process_box(ax, 2.5, 4.5, 2.0, 0.7, 'Check Fuse\n& Connectors', COLORS['light_orange'])
    draw_flow_arrow(ax, (2.5, 4.15), (2.5, 3.5))
    draw_terminal(ax, 2.5, 3.0, 2.0, 0.5, 'Replace\nFuse/Cable', COLORS['light_green'])

    # Power ON path (continue down)
    draw_flow_arrow(ax, (7, 7.1), (7, 6.5), 'Yes')
    draw_decision_diamond(ax, 7, 6, 1.4, 'Data\nOutput?', COLORS['light_blue'])

    # No data path (right branch)
    draw_flow_arrow(ax, (7.7, 6), (9.5, 6), 'No')
    draw_process_box(ax, 11, 6, 2.2, 0.7, 'Check USB/\nCommunication', COLORS['light_orange'])
    draw_flow_arrow(ax, (11, 5.65), (11, 5.0))
    draw_decision_diamond(ax, 11, 4.5, 1.2, 'Comms\nOK?', COLORS['light_blue'])

    # Comms bad
    draw_flow_arrow(ax, (11.6, 4.5), (13, 4.5), 'No')
    draw_process_box(ax, 13, 3.8, 1.6, 0.6, 'Replace\nCable', COLORS['light_green'])

    # Comms OK but still no data
    draw_flow_arrow(ax, (11, 3.9), (11, 3.3), 'Yes')
    draw_process_box(ax, 11, 2.8, 2.0, 0.7, 'Check Hub\nFirmware', COLORS['light_orange'])

    # Data present path (continue down)
    draw_flow_arrow(ax, (7, 5.3), (7, 4.7), 'Yes')
    draw_decision_diamond(ax, 7, 4.2, 1.4, 'Valid\nReadings?', COLORS['light_blue'])

    # Invalid readings - branch to MIT/ERT
    draw_flow_arrow(ax, (6.3, 4.2), (5, 4.2), 'No')
    draw_decision_diamond(ax, 4, 4.2, 1.2, 'MIT or\nERT?', COLORS['light_blue'])

    # MIT Issue branch
    draw_flow_arrow(ax, (4, 3.6), (4, 2.8), 'MIT')
    draw_process_box(ax, 4, 2.3, 2.2, 0.7, 'MIT Diagnostics', COLORS['light_purple'])

    # MIT sub-issues
    draw_flow_arrow(ax, (4, 1.95), (4, 1.4))
    ax.add_patch(FancyBboxPatch((2.5, 0.3), 3.0, 1.0,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS['light_purple'], edgecolor=COLORS['purple'], lw=1))
    ax.text(4, 0.8, 'MIT Checks:', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(4, 0.45, '\u2022 Coil connections\n\u2022 TX level (saturation?)\n\u2022 Frequency setting',
            ha='center', va='center', fontsize=6)

    # ERT Issue branch
    draw_flow_arrow(ax, (4.6, 4.2), (5.5, 4.2))
    draw_flow_arrow(ax, (5.5, 4.2), (5.5, 2.8), 'ERT')
    draw_process_box(ax, 5.5, 2.3, 2.0, 0.7, 'ERT Diagnostics', COLORS['light_orange'])

    # ERT sub-issues
    draw_flow_arrow(ax, (6.5, 2.3), (8, 2.3))
    ax.add_patch(FancyBboxPatch((7.5, 1.3), 3.0, 1.3,
                                boxstyle="round,pad=0.02",
                                facecolor=COLORS['light_orange'], edgecolor=COLORS['orange'], lw=1))
    ax.text(9, 1.95, 'ERT Checks:', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(9, 1.4, '\u2022 Ring-soil contact\n\u2022 Pre-moisten hole\n\u2022 Current source\n\u2022 Saline gel application',
            ha='center', va='center', fontsize=6)

    # Valid readings - SUCCESS
    draw_flow_arrow(ax, (7, 3.5), (7, 2.8), 'Yes')
    draw_terminal(ax, 7, 2.3, 2.5, 0.5, 'SYSTEM OK', COLORS['light_green'])

    # Additional notes box
    ax.add_patch(FancyBboxPatch((10.5, 7.5), 3.0, 1.8,
                                boxstyle="round,pad=0.02",
                                facecolor='white', edgecolor=COLORS['gray_med'], lw=1))
    ax.text(12, 9.0, 'Quick Reference', ha='center', va='center', fontsize=9, fontweight='bold',
            color=COLORS['primary'])
    ax.text(12, 8.4, '\u2022 Battery min: 12.0V\n\u2022 Warmup: 15 min\n\u2022 Ground: single-point\n\u2022 MIT freq: 2-20 kHz',
            ha='center', va='center', fontsize=7)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def build_troubleshooting_pdf():
    """
    Build the complete Section 13: Troubleshooting PDF.

    Returns:
        Path to the generated PDF file
    """
    builder = SectionPDFBuilder(
        section_num=13,
        title="Troubleshooting"
    )

    # Title block
    builder.add_title_block(
        subtitle="Diagnostic Approaches and Solutions for HIRT Field Operations"
    )

    # Overview
    builder.add_section_header("Overview", level=1)
    builder.add_body_text(
        "This section provides diagnostic approaches and solutions for common issues "
        "encountered during HIRT field operations. Systematic troubleshooting ensures "
        "minimal downtime and maximum data quality during surveys.",
        first_paragraph=True
    )

    # 13.1 Diagnostic Approach
    builder.add_section_header("13.1 Diagnostic Approach", level=1)

    builder.add_section_header("System Check Procedure", level=2)
    builder.add_body_text(
        "When issues arise, follow this systematic check procedure:",
        first_paragraph=True
    )
    builder.add_numbered_list([
        "<b>Hub Power:</b> Verify the Central Hub power LED is green.",
        "<b>Zone Continuity:</b> Check continuity between Hub Trunk port and Probe tip (using DMM).",
        "<b>Trunk Seating:</b> Ensure all DB25/Trunk connectors are fully seated and screwed in.",
        "<b>MIT Baseline:</b> Verify TX to RX coupling on two probes in air before insertion."
    ])

    # Quick Reference Table
    builder.add_section_header("Quick Reference Table", level=2)
    quick_ref_data = [
        ["Symptom", "Likely Cause", "Fix"],
        ["RX saturation on nearby probes", "TX too strong", "Lower TX level; rotate coils to orthogonal"],
        ["Noisy MIT data", "Trunk shielding issue", "Check trunk cable ground; verify star ground"],
        ["Unstable ERT voltages", "Poor soil contact", "Pre-moisten hole; apply saline gel"],
        ["Zone Hub not detected", "Trunk disconnected", "Inspect DB25 pins; reseat Trunk Cable"],
        ["Inconsistent repeats", "Thermal drift", "Allow 15 min warmup; check Hub fans"]
    ]
    builder.add_table(quick_ref_data, caption="Common symptoms, causes, and solutions")

    # Add the diagnostic decision tree diagram
    builder.add_section_header("Diagnostic Decision Tree", level=2)
    builder.add_body_text(
        "The following decision tree provides a systematic approach to diagnosing "
        "HIRT system issues, from initial symptom identification through resolution.",
        first_paragraph=True
    )
    decision_tree_buf = create_diagnostic_decision_tree()
    builder.add_figure(
        decision_tree_buf,
        "Comprehensive diagnostic decision tree for HIRT troubleshooting. "
        "Follow the flowchart from the top to systematically identify and resolve issues."
    )

    # 13.2 MIT Troubleshooting
    builder.add_section_header("13.2 MIT Troubleshooting", level=1)

    builder.add_section_header("RX Saturation on Nearby Probes", level=2)
    builder.add_body_text(
        "<b>Symptoms:</b> Amplitude readings maxed out on nearby probes, phase readings "
        "erratic or stuck, effects only visible on probes close to TX.",
        first_paragraph=True
    )
    builder.add_body_text(
        "<b>Causes:</b> TX output too high for close-range measurements, direct magnetic "
        "coupling between TX and RX coils, insufficient separation or poor coil orientation."
    )
    builder.add_body_text("<b>Solutions:</b>")
    builder.add_numbered_list([
        "<b>Reduce TX level:</b> Lower DDS output or TX driver gain",
        "<b>Increase separation:</b> Use wider probe spacing for close pairs",
        "<b>Rotate coils:</b> Ensure TX and RX coils are orthogonal",
        "<b>Add attenuation:</b> Use lower gain on RX for nearby probes",
        "<b>Skip close pairs:</b> Do not measure TX to RX pairs less than 0.5 m apart"
    ])

    builder.add_section_header("Noisy MIT Data", level=2)
    builder.add_body_text(
        "<b>Symptoms:</b> High noise floor, erratic amplitude/phase readings, "
        "inconsistent measurements between sweeps.",
        first_paragraph=True
    )
    builder.add_body_text(
        "<b>Causes:</b> Poor cable shielding, ground loops (multiple ground paths), "
        "EMI from nearby sources, insufficient integration time."
    )
    builder.add_body_text("<b>Solutions:</b>")
    builder.add_numbered_list([
        "<b>Check shielding:</b> Verify all cables are properly shielded",
        "<b>Single-point ground:</b> Ensure only one ground connection at Hub",
        "<b>Twist pairs:</b> Use twisted-pair cables for signal lines",
        "<b>Increase integration:</b> Longer measurement time reduces noise",
        "<b>Check EMI sources:</b> Move away from power lines, vehicles, radios",
        "<b>Verify connections:</b> Check all connectors are tight and clean"
    ])

    builder.add_section_header("No Depth Sensitivity (MIT)", level=2)
    builder.add_body_text(
        "<b>Symptoms:</b> Measurements do not change with depth, all readings similar "
        "regardless of target depth, no response to known deep targets.",
        first_paragraph=True
    )
    builder.add_body_text("<b>Solutions:</b>")
    builder.add_numbered_list([
        "<b>Lower frequencies:</b> Use 2-5 kHz for MIT (deeper penetration)",
        "<b>Add longer offsets:</b> Include TX to RX pairs with >3 m spacing",
        "<b>Deeper probes:</b> Insert probes deeper if possible",
        "<b>Check targets:</b> Verify expected target depths are realistic for configuration"
    ])

    # 13.3 ERT Troubleshooting
    builder.add_section_header("13.3 ERT Troubleshooting", level=1)

    builder.add_section_header("Unstable ERT Voltages", level=2)
    builder.add_body_text(
        "<b>Symptoms:</b> Voltage readings drift or jump, inconsistent measurements, "
        "poor contact indicated in diagnostic output.",
        first_paragraph=True
    )
    builder.add_body_text(
        "<b>Causes:</b> Dry soil preventing good electrical contact, poor ring-to-soil "
        "contact, loose connections, polarization effects at electrodes."
    )
    builder.add_body_text("<b>Solutions:</b>")
    builder.add_numbered_list([
        "<b>Pre-moisten hole:</b> Add water to improve contact",
        "<b>Use saline gel:</b> Apply conductive gel around rings",
        "<b>Check ring contact:</b> Ensure rings are flush with soil",
        "<b>Reverse polarity:</b> Use AC or periodic polarity reversal to reduce polarization",
        "<b>Check connections:</b> Verify all wiring is secure",
        "<b>Increase current:</b> Slightly higher current may improve SNR"
    ])

    builder.add_section_header("ERT Contact Problems", level=2)
    builder.add_body_text(
        "<b>Symptoms:</b> High contact resistance, erratic voltage readings, "
        "some electrodes not responding.",
        first_paragraph=True
    )
    builder.add_body_text("<b>Solutions:</b>")
    builder.add_numbered_list([
        "Verify ring contact with soil",
        "Add water around probe (improve contact)",
        "Check ring-to-cable connection",
        "Clean rings if accessible",
        "Consider relocating probe if contact remains poor"
    ])

    builder.add_section_header("No Depth Sensitivity (ERT)", level=2)
    builder.add_body_text(
        "<b>Symptoms:</b> All measurements show surface effects only, no response "
        "to known deep features, uniform readings across array.",
        first_paragraph=True
    )
    builder.add_body_text("<b>Solutions:</b>")
    builder.add_numbered_list([
        "<b>Longer ERT baselines:</b> Corner-to-corner, edge-to-edge injections",
        "<b>Deeper probes:</b> Insert probes deeper if possible",
        "<b>Wider spacing:</b> Increase array dimensions"
    ])

    # Add the troubleshooting flowchart from the library
    builder.add_spacer(12)
    troubleshooting_flowchart = create_troubleshooting_flowchart()
    builder.add_figure(
        troubleshooting_flowchart,
        "Quick troubleshooting flowchart for rapid issue identification. "
        "This simplified guide helps identify the most common failure modes."
    )

    # 13.4 Power Issues
    builder.add_page_break()
    builder.add_section_header("13.4 Power Issues", level=1)

    builder.add_section_header("Probe Not Responding", level=2)
    builder.add_body_text(
        "<b>Symptoms:</b> No LED indicator, no communication with base hub, "
        "probe not detected in diagnostic scan.",
        first_paragraph=True
    )
    builder.add_body_text("<b>Solutions:</b>")
    builder.add_numbered_list([
        "Check cable connections (both ends)",
        "Test with multimeter (continuity)",
        "Try different port on base hub",
        "Swap to spare cable",
        "Restart probe (disconnect/reconnect power)",
        "Replace with spare probe if available"
    ])

    builder.add_section_header("Base Hub Not Powering On", level=2)
    builder.add_body_text(
        "<b>Symptoms:</b> No power LED, no response to power switch, "
        "system completely dead.",
        first_paragraph=True
    )
    builder.add_body_text(
        "<b>Causes:</b> Depleted battery, blown fuse, power switch failure."
    )
    builder.add_body_text("<b>Solutions:</b>")
    builder.add_numbered_list([
        "Check battery voltage (should be >12.0 V)",
        "Check fuse (replace if blown)",
        "Verify power switch operation",
        "Check internal connections if accessible",
        "Use backup power supply if available"
    ])

    # 13.5 Communication Issues
    builder.add_section_header("13.5 Communication Issues", level=1)

    builder.add_section_header("Lost Probe Communication", level=2)
    builder.add_body_text(
        "<b>Symptoms:</b> Probe was working, now unresponsive; intermittent "
        "connection; partial data received.",
        first_paragraph=True
    )
    builder.add_body_text("<b>Solutions:</b>")
    builder.add_numbered_list([
        "Check cable connections (both ends)",
        "Inspect cable for damage (kinks, cuts)",
        "Clean connectors with contact cleaner",
        "Swap to known-good cable",
        "Check for EMI sources nearby",
        "Restart probe and base hub"
    ])

    builder.add_section_header("Sync Problems", level=2)
    builder.add_body_text(
        "<b>Symptoms:</b> Timing errors in data, inconsistent measurements "
        "between probes, data corruption.",
        first_paragraph=True
    )
    builder.add_body_text(
        "<b>Causes:</b> Timebase distribution failure, clock drift, cable "
        "issues on sync line."
    )
    builder.add_body_text("<b>Solutions:</b>")
    builder.add_numbered_list([
        "Verify sync signal at each probe",
        "Check sync cable for damage",
        "Restart measurement system",
        "Re-initialize sync before continuing"
    ])

    # 13.6 Environmental Factors
    builder.add_section_header("13.6 Environmental Factors", level=1)

    builder.add_section_header("Temperature Effects", level=2)
    builder.add_body_text(
        "<b>Symptoms:</b> Readings drift over time, morning vs. afternoon "
        "differences, inconsistent calibration.",
        first_paragraph=True
    )
    builder.add_body_text("<b>Solutions:</b>")
    builder.add_numbered_list([
        "Allow system to stabilize (10-15 min warmup)",
        "Note temperature variations in field log",
        "Apply temperature compensation if available",
        "Take baseline measurements at current temperature",
        "Shield equipment from direct sun if possible"
    ])

    builder.add_section_header("Weather Impacts", level=2)
    builder.add_body_text(
        "<b>Symptoms:</b> Data quality degradation, increased noise, "
        "contact problems.",
        first_paragraph=True
    )
    builder.add_body_text("<b>Solutions:</b>")
    builder.add_numbered_list([
        "Protect connections from moisture",
        "Secure cables against wind movement",
        "Use weather covers for sensitive equipment",
        "Postpone measurements in severe conditions"
    ])

    # 13.7 Field Repairs
    builder.add_section_header("13.7 Field Repairs", level=1)

    builder.add_section_header("Field Repair Kit", level=2)
    builder.add_body_text(
        "Keep these items handy for field repairs:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "Spare cables (2-3)",
        "Spare probes (2-4 recommended)",
        "Electrical tape",
        "Multimeter",
        "Small screwdriver set",
        "Contact cleaner spray",
        "Zip ties",
        "Heat-shrink tubing",
        "Solder and iron (battery powered, optional)",
        "Notebook and pencil (for notes)",
        "Calibration sheet"
    ])

    builder.add_section_header("Emergency Cable Repair", level=2)
    builder.add_body_text(
        "If a cable is damaged in the field:",
        first_paragraph=True
    )
    builder.add_numbered_list([
        "<b>Locate the break</b> (continuity test)",
        "<b>Cut out damaged section</b> if possible",
        "<b>Strip and splice</b> wires",
        "<b>Insulate with tape</b> or heat-shrink",
        "<b>Test before use</b>"
    ])
    builder.add_note(
        "Note: Field-repaired cables should be replaced at first opportunity."
    )

    builder.add_section_header("Connector Cleaning", level=2)
    builder.add_numbered_list([
        "Apply contact cleaner to connector",
        "Wipe with clean cloth",
        "Allow to dry before reconnecting",
        "Apply thin layer of dielectric grease (optional, for moisture protection)"
    ])

    # 13.8 When to Abort Survey
    builder.add_section_header("13.8 When to Abort Survey", level=1)

    builder.add_warning_box(
        "Conditions Requiring Survey Abort",
        [
            "<b>Safety Issues:</b> Severe weather (lightning, high winds), site hazards discovered, equipment malfunction creating hazard",
            "<b>Data Quality:</b> >25% probes non-functional, persistent unusable noise, unable to achieve ground contact on majority of probes",
            "<b>Critical Failures:</b> Base hub failure, data logger failure, battery depletion with no backup",
            "<b>Practical Issues:</b> Time constraints preventing quality data, site access revoked"
        ]
    )

    builder.add_section_header("Before Aborting", level=2)
    builder.add_numbered_list([
        "Document all issues in field log",
        "Save all data collected (even partial)",
        "Note probe positions for potential return",
        "Photograph site conditions",
        "Extract probes safely if time permits",
        "Backup data immediately"
    ])

    builder.add_section_header("Partial Survey Options", level=2)
    builder.add_body_text(
        "If full abort is not necessary, consider these alternatives:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "<b>Reduce survey area</b> to functional probes",
        "<b>Simplify measurement protocol</b> (fewer frequencies, fewer patterns)",
        "<b>Focus on priority targets</b> only",
        "<b>Document limitations</b> for data interpretation"
    ])

    # 13.9 Prevention
    builder.add_section_header("13.9 Prevention", level=1)

    builder.add_section_header("Pre-Field Checks", level=2)
    builder.add_bullet_list([
        "All connectors tight and clean",
        "Cables tested for continuity",
        "Calibration up to date",
        "Spare parts available",
        "Field diagnostic kit packed",
        "Batteries charged",
        "Weather forecast reviewed"
    ])

    builder.add_section_header("During Survey", level=2)
    builder.add_bullet_list([
        "Monitor data quality in real-time",
        "Check reciprocity periodically",
        "Note any anomalies immediately",
        "Keep spare probes ready",
        "Protect equipment from weather"
    ])

    builder.add_section_header("Post-Survey", level=2)
    builder.add_bullet_list([
        "Inspect all equipment",
        "Note any issues for repair",
        "Update calibration records",
        "Clean and store properly",
        "Recharge all batteries",
        "Restock field repair kit"
    ])

    # Cross-reference note
    builder.add_spacer(12)
    builder.add_info_box(
        "Related Sections",
        [
            "For detailed operating procedures, see <b>Section 10: Field Operations</b>",
            "For data format specifications, see <b>Section 11: Data Recording</b>",
            "For calibration procedures, see <b>Section 9: Calibration</b>"
        ]
    )

    # Build and return path
    return builder.build()


def main():
    """Main entry point for standalone execution."""
    print("Generating HIRT Section 13: Troubleshooting PDF...")
    output_path = build_troubleshooting_pdf()
    print(f"PDF generated successfully: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
