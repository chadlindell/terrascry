#!/usr/bin/env python3
"""
HIRT Section 10: Field Operations - Publication-Quality PDF Generator

Generates a comprehensive PDF covering field deployment procedures, grid layouts,
measurement sequences, and operational workflows for the HIRT system.

Usage:
    python section_10_field_operations.py

Output:
    ../output/sections/10-field-operations.pdf
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pdf_builder import SectionPDFBuilder
from lib.styles import CONTENT_WIDTH
from lib.diagrams.field_ops import (
    create_grid_layout,
    create_measurement_sequence,
    create_ert_injection_patterns,
    create_deployment_scenarios,
    create_insertion_procedure,
)
from lib.diagrams.flowcharts import (
    create_soil_type_decision_tree,
    create_survey_workflow,
)


def build_section_10():
    """
    Build Section 10: Field Operations PDF.

    Returns:
        Path to the generated PDF file
    """
    builder = SectionPDFBuilder(
        section_num=10,
        title="Field Operations"
    )

    # === Title Block ===
    builder.add_title_block(
        subtitle="HIRT Field Deployment Procedures and Survey Protocols",
        version="2.0"
    )

    # === Overview ===
    builder.add_section_header("10.1 Overview")
    builder.add_body_text(
        "This section provides comprehensive procedures for deploying and operating "
        "the HIRT (Hybrid Inductive-Resistivity Tomography) system in the field, from "
        "pre-deployment planning through data backup. The HIRT system combines Magnetic "
        "Induction Tomography (MIT) and Electrical Resistivity Tomography (ERT) measurements "
        "using a modular array of subsurface probes to achieve high-resolution imaging of "
        "buried targets.",
        first_paragraph=True
    )

    # === Pre-Deployment Planning ===
    builder.add_section_header("10.2 Pre-Deployment Planning")

    builder.add_section_header("Site Assessment (Day Before)", level=2)
    builder.add_body_text(
        "Thorough preparation is critical to successful field operations. Before deployment, "
        "conduct a comprehensive site assessment covering the following aspects:",
        first_paragraph=True
    )

    builder.add_numbered_list([
        "<b>Review site conditions:</b> Evaluate soil type and moisture levels, identify "
        "presence of utilities or obstructions, and establish access routes and staging areas.",
        "<b>Check equipment:</b> Verify all probes are tested and calibrated, ensure base hub "
        "battery is charged (>80%), verify cable continuity, and confirm data logger/tablet "
        "is configured.",
        "<b>Verify permits and permissions:</b> Confirm site access authorization, excavation "
        "permits (if required), and archaeological survey approval.",
    ])

    builder.add_section_header("Equipment Checklist", level=2)

    # Equipment checklist table
    equipment_data = [
        ["Category", "Item", "Quantity", "Notes"],
        ["Essential", "Probes (tested)", "20-24", "Full functional test"],
        ["Essential", "Base hub/control unit", "1", "Battery >80%"],
        ["Essential", "Connection cables", "3-5 m each", "Verified continuity"],
        ["Essential", "Fiberglass survey stakes", "As needed", "For marking"],
        ["Essential", "Measuring tape", "30 m min", "Metric markings"],
        ["Essential", "Field tablet/laptop", "1", "Configured with software"],
        ["Essential", "Tool kit", "1", "Wrenches, screwdrivers, multimeter"],
        ["Optional", "GPS unit", "1", "Sub-meter accuracy preferred"],
        ["Optional", "Weather station", "1", "Temperature, humidity"],
        ["Optional", "Backup battery", "1", "12V compatible"],
    ]
    builder.add_table(
        equipment_data,
        col_widths=[CONTENT_WIDTH * 0.15, CONTENT_WIDTH * 0.35,
                    CONTENT_WIDTH * 0.2, CONTENT_WIDTH * 0.3],
        caption="Field equipment checklist for HIRT deployment"
    )

    # === Site Assessment and Grid Design ===
    builder.add_section_header("10.3 Site Assessment and Grid Design")

    builder.add_body_text(
        "Grid design is fundamental to survey success. The grid geometry determines spatial "
        "resolution, depth sensitivity, and survey efficiency. Standard configurations are "
        "provided below, with adjustments based on site-specific requirements.",
        first_paragraph=True
    )

    # Grid configuration table
    grid_data = [
        ["Configuration", "Grid Size", "Spacing", "Probes", "Application"],
        ["Standard", "10 x 10 m", "2.0 m", "20-24", "General surveys"],
        ["Small", "8 x 8 m", "1.5 m", "12-16", "Woodland/confined areas"],
        ["Large", "15 x 15 m", "2.5 m", "30-36", "Crater sites, open areas"],
        ["High Resolution", "6 x 6 m", "1.0 m", "36", "Detailed anomaly mapping"],
    ]
    builder.add_table(
        grid_data,
        col_widths=[CONTENT_WIDTH * 0.18, CONTENT_WIDTH * 0.16, CONTENT_WIDTH * 0.14,
                    CONTENT_WIDTH * 0.14, CONTENT_WIDTH * 0.38],
        caption="Standard grid configurations for different survey scenarios"
    )

    builder.add_spacer(8)

    # Figure 1: Grid Layout
    grid_buf = create_grid_layout(rows=5, cols=5, spacing=2.0, show_zones=True)
    builder.add_figure(
        grid_buf,
        "Standard 10x10 m grid layout with 2 m probe spacing showing zone groupings "
        "for sequential deployment. Probes are numbered 1-25 in row-major order.",
        width=CONTENT_WIDTH * 0.85,
        height=CONTENT_WIDTH * 0.68  # Maintain aspect ratio
    )

    builder.add_section_header("Grid Setup Steps", level=2)
    builder.add_numbered_list([
        "<b>Establish baseline:</b> Set reference point (0,0) with survey stake, align "
        "baseline with site grid (N-S or as specified), and mark corners with bright flags.",
        "<b>Mark probe positions:</b> Measure spacing intervals along baseline, extend "
        "perpendicular rows, and mark each position with small stake or flag.",
        "<b>Verify geometry:</b> Cross-measure diagonals (should match within 5 cm) and "
        "record any deviations in field notes.",
    ])

    # === Probe Installation ===
    builder.add_section_header("10.4 Probe Installation")

    builder.add_body_text(
        "Proper probe installation is critical for obtaining quality data. Insertion depth "
        "and technique vary with soil conditions and target depth requirements.",
        first_paragraph=True
    )

    # Insertion depth table
    depth_data = [
        ["Scenario", "Target Depth", "Insertion Depth", "Notes"],
        ["Woodland survey", "1-2 m", "1.5 m", "Standard for shallow targets"],
        ["Crater survey", "2-4 m", "3.0 m", "Extended depth for crater fill"],
        ["Reconnaissance", "0.5-1 m", "1.0 m", "Rapid screening mode"],
    ]
    builder.add_table(
        depth_data,
        col_widths=[CONTENT_WIDTH * 0.22, CONTENT_WIDTH * 0.2, CONTENT_WIDTH * 0.22,
                    CONTENT_WIDTH * 0.36],
        caption="Recommended insertion depths by survey scenario"
    )

    builder.add_spacer(8)

    # Figure 2: Insertion Procedure
    insertion_buf = create_insertion_procedure()
    builder.add_figure(
        insertion_buf,
        "Probe insertion procedure showing the four-step process: (1) pilot hole "
        "creation with hand auger, (2) first segment insertion, (3) segment connection "
        "at thread joint, and (4) completed installation with junction box and cable routing.",
        width=CONTENT_WIDTH * 0.95,
        height=CONTENT_WIDTH * 0.40  # 12:5 aspect ratio
    )

    # Figure 3: Soil Type Decision Tree
    soil_tree_buf = create_soil_type_decision_tree()
    builder.add_figure(
        soil_tree_buf,
        "Soil type insertion decision tree. The procedure adapts based on soil hardness, "
        "water table depth, and presence of obstructions. Key principle: never hammer probes; "
        "use maximum hand pressure only.",
        width=CONTENT_WIDTH * 0.80,
        height=CONTENT_WIDTH * 0.64  # 10:8 aspect ratio
    )

    builder.add_section_header("Insertion Methods by Soil Type", level=2)

    builder.add_section_header("Sandy/Loose Soil", level=3)
    builder.add_bullet_list([
        "May push probe directly without pilot hole",
        "Use water jet if available for deeper insertion",
        "Watch for collapse of hole walls during insertion",
    ])

    builder.add_section_header("Clay/Compact Soil", level=3)
    builder.add_bullet_list([
        "Pre-drill pilot hole using 18 mm hand auger",
        "May need to enlarge hole slightly for probe diameter",
        "Allow settling time (5-10 min) before measurement",
    ])

    builder.add_section_header("Rocky/Mixed Soil", level=3)
    builder.add_bullet_list([
        "Use auger with care to avoid damage",
        "Note rock contact locations in field log",
        "May need to relocate probe position slightly",
    ])

    builder.add_info_box("Cable Routing Guidelines", [
        "Route cables radially from base hub placed at grid center or edge",
        "Route cables along grid lines to avoid tangling",
        "Use cable clips or ties to secure at 1 m intervals",
        "Leave 0.5 m slack at each probe connection for strain relief",
    ])

    # === System Setup and Power-Up ===
    builder.add_section_header("10.5 System Setup and Power-Up")

    builder.add_body_text(
        "After probe installation, systematic setup and verification ensures reliable "
        "data acquisition. Follow the power-up sequence carefully to avoid equipment damage "
        "and verify all connections before beginning measurements.",
        first_paragraph=True
    )

    builder.add_section_header("Power-Up Sequence", level=2)
    builder.add_numbered_list([
        "<b>Connect all cables:</b> Verify each probe is connected to the base hub, "
        "check cable routing for kinks or tension points.",
        "<b>Power on base hub:</b> Turn main power switch ON, wait for initialization "
        "(10-15 seconds), and verify power LED is solid green.",
        "<b>Verify probe communication:</b> Run diagnostic scan, verify all probes report "
        "status, and note any non-responsive probes for troubleshooting.",
        "<b>Initialize measurement system:</b> Start field software, load site configuration, "
        "and verify probe array geometry matches field deployment.",
    ])

    # === MIT Measurement Protocol ===
    builder.add_section_header("10.6 MIT Measurement Protocol")

    builder.add_body_text(
        "Magnetic Induction Tomography measurements detect conductive anomalies through "
        "electromagnetic coupling. Each probe sequentially transmits while all others receive, "
        "building a complete response matrix at multiple frequencies.",
        first_paragraph=True
    )

    # Figure 4: Measurement Sequence Timeline
    sequence_buf = create_measurement_sequence()
    builder.add_figure(
        sequence_buf,
        "Measurement sequence timeline showing interleaved MIT scanning and ERT measurement "
        "cycles. MIT scans sweep through frequencies (2-50 kHz) while ERT measurements occur "
        "during multiplexer switching intervals.",
        width=CONTENT_WIDTH * 0.95,
        height=CONTENT_WIDTH * 0.48  # 10:5 aspect ratio
    )

    builder.add_section_header("Frequency Selection", level=2)

    freq_data = [
        ["Frequency", "Penetration", "Primary Use"],
        ["2 kHz", "Deep (3-4 m)", "Deep target detection, high conductivity"],
        ["5 kHz", "Medium-deep", "General subsurface mapping"],
        ["10 kHz", "Medium", "Balanced depth/resolution"],
        ["20 kHz", "Shallow-medium", "Enhanced resolution"],
        ["50 kHz", "Shallow (<1 m)", "Near-surface detail, small targets"],
    ]
    builder.add_table(
        freq_data,
        col_widths=[CONTENT_WIDTH * 0.2, CONTENT_WIDTH * 0.3, CONTENT_WIDTH * 0.5],
        caption="MIT frequency selection guide for different survey objectives"
    )

    builder.add_section_header("Timing", level=2)
    builder.add_bullet_list([
        "Full matrix measurement: all TX-RX pairs at single frequency (3-5 minutes)",
        "Multi-frequency sweep: complete matrix at all frequencies (30-45 minutes)",
        "Typical survey cycle: 10-15 minutes per complete scan at primary frequency",
    ])

    # === ERT Measurement Protocol ===
    builder.add_section_header("10.7 ERT Measurement Protocol")

    builder.add_body_text(
        "Electrical Resistivity Tomography measurements inject DC current across probe pairs "
        "while measuring voltage at all other electrodes. Multiple injection patterns provide "
        "complementary sensitivity for robust inversion.",
        first_paragraph=True
    )

    # Figure 5: ERT Injection Patterns
    ert_buf = create_ert_injection_patterns()
    builder.add_figure(
        ert_buf,
        "ERT injection pattern options. (a) Wenner-Alpha: symmetric ABMN spacing for uniform "
        "sensitivity. (b) Dipole-Dipole: separated current and voltage dipoles for lateral "
        "resolution. (c) Pole-Dipole: remote current electrode for deep penetration.",
        width=CONTENT_WIDTH * 0.95,
        height=CONTENT_WIDTH * 0.32  # 12:4 aspect ratio
    )

    builder.add_section_header("ERT Configuration Parameters", level=2)

    ert_config_data = [
        ["Parameter", "Standard Value", "Range", "Notes"],
        ["Current level", "1.0 mA", "0.5-2.0 mA", "Adjust for noise floor"],
        ["Integration time", "2 s", "1-5 s", "Longer for noisy sites"],
        ["Polarity reversal", "Every 1 s", "0.5-2 s", "Reduces electrode polarization"],
        ["Stacking", "4 cycles", "2-8", "More for weak signals"],
    ]
    builder.add_table(
        ert_config_data,
        col_widths=[CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.2, CONTENT_WIDTH * 0.2,
                    CONTENT_WIDTH * 0.35],
        caption="ERT measurement configuration parameters"
    )

    # === Set Once, Measure Many Workflow ===
    builder.add_section_header('10.8 "Set Once, Measure Many" Workflow')

    builder.add_body_text(
        "The HIRT operational philosophy emphasizes thorough probe placement followed by "
        "comprehensive measurement. This approach maximizes data quality and enables "
        "redundancy checks while minimizing handling of deployed equipment.",
        first_paragraph=True
    )

    # Figure 6: Survey Workflow
    workflow_buf = create_survey_workflow()
    builder.add_figure(
        workflow_buf,
        "Complete field survey workflow showing four phases: Planning (site reconnaissance, "
        "grid design, equipment check), Deployment (marking, insertion, cabling), Acquisition "
        "(system test, survey execution, QC checks), and Recovery (backup, extraction, restoration).",
        width=CONTENT_WIDTH * 0.90,
        height=CONTENT_WIDTH * 0.72  # 10:8 aspect ratio
    )

    builder.add_section_header("Workflow Steps", level=2)
    builder.add_numbered_list([
        "<b>Install All Probes:</b> Deploy complete array, mark with numbered flags, "
        "record GPS coordinates and insertion depths.",
        "<b>Background Scan:</b> Perform short MIT and ERT scan outside suspected zone "
        "to establish baseline/control measurements.",
        "<b>MIT Sweep:</b> Complete full TX-RX matrix at all frequencies before moving probes.",
        "<b>ERT Patterns:</b> Execute all injection patterns with multiple baselines for redundancy.",
        "<b>Quality Control:</b> Repeat 5-10% of TX-RX pairs, verify reciprocity (A-B "
        "approximately equals B-A), document any anomalies.",
        "<b>Extract and Move:</b> Carefully extract probes, shift to next section with "
        "one-column overlap for data continuity.",
    ])

    # === Quality Checks ===
    builder.add_section_header("10.9 Quality Checks")

    builder.add_body_text(
        "Real-time quality assessment during data acquisition prevents costly re-surveys. "
        "Monitor the following indicators throughout the measurement process:",
        first_paragraph=True
    )

    builder.add_section_header("Good Data Indicators", level=2)
    builder.add_bullet_list([
        "Consistent reciprocity: TX(A)-RX(B) matches TX(B)-RX(A) within 5%",
        "Smooth spatial variations without abrupt discontinuities",
        "Expected depth sensitivity decay with increasing offset",
        "Stable baseline measurements throughout survey",
    ])

    builder.add_warning_box("Problematic Data Indicators", [
        "Poor reciprocity (>10% difference) - check coupling, recalibrate",
        "Noisy/spiky readings - check connections, improve shielding",
        "No depth sensitivity - verify spacing, adjust frequency",
        "Inconsistent repeats - check timebase synchronization, connector integrity",
    ])

    # === Deployment Scenarios ===
    builder.add_section_header("10.10 Deployment Scenarios")

    builder.add_body_text(
        "Different site conditions require adapted deployment strategies. The following "
        "scenarios illustrate common configurations optimized for specific survey objectives.",
        first_paragraph=True
    )

    # Figure 7: Deployment Scenarios
    scenarios_buf = create_deployment_scenarios()
    builder.add_figure(
        scenarios_buf,
        "Deployment scenario configurations. (a) Square Array: uniform coverage for general "
        "surveys (25 probes). (b) Perimeter Only: ring deployment around known target zone "
        "for minimal disturbance (16 probes). (c) Dense Center: enhanced resolution over "
        "target with perimeter context (21 probes).",
        width=CONTENT_WIDTH * 0.95,
        height=CONTENT_WIDTH * 0.36  # 12:4.5 aspect ratio
    )

    builder.add_section_header("Minimal-Intrusion Variants", level=2)

    builder.add_section_header("Rim-Only Deployment", level=3)
    builder.add_bullet_list([
        "Place ring of probes around suspected feature edge",
        "Add select probes angled inward for cross-coverage",
        "Reduces ground disturbance in sensitive archaeological contexts",
        "Provides adequate coverage with proper geometric factor correction",
    ])

    builder.add_section_header("Shallow Mode", level=3)
    builder.add_bullet_list([
        "Insert probes to 1 m or less",
        "Use wider spacing (2-3 m) to compensate",
        "Rely on lower frequencies (2-5 kHz) for deeper field penetration",
        "Suitable for extremely sensitive sites or regulatory constraints",
    ])

    # === Time Estimates ===
    builder.add_section_header("10.11 Time Estimates")

    time_data = [
        ["Activity", "Duration", "Team Size", "Notes"],
        ["Setup (probe insertion)", "30-60 min", "2-3", "Varies with soil"],
        ["MIT sweep (all frequencies)", "30-45 min", "1", "Automated"],
        ["ERT patterns", "15-30 min", "1", "Multiple configurations"],
        ["QC checks", "10-15 min", "1", "Repeat measurements"],
        ["Extraction", "15-30 min", "2-3", "Care required"],
        ["<b>Total per section</b>", "<b>2-3 hours</b>", "2-3", "10x10 m, 20 probes"],
    ]
    builder.add_table(
        time_data,
        col_widths=[CONTENT_WIDTH * 0.35, CONTENT_WIDTH * 0.2, CONTENT_WIDTH * 0.15,
                    CONTENT_WIDTH * 0.3],
        caption="Time estimates for standard 10x10 m section survey"
    )

    # === Shutdown and Data Backup ===
    builder.add_section_header("10.12 Shutdown and Data Backup")

    builder.add_section_header("Safe Shutdown Procedure", level=2)
    builder.add_numbered_list([
        "<b>Complete final measurement:</b> Take final data set and verify data is saved.",
        "<b>Power down:</b> Stop measurement software, power off base hub, disconnect "
        "cables from probes.",
        "<b>Extract probes:</b> Pull gently with straight vertical motion, do not twist "
        "excessively, note any stuck probes.",
        "<b>Site restoration:</b> Fill probe holes as required by permit, remove all markers "
        "and equipment, photograph final site condition.",
    ])

    builder.add_section_header("Probe Extraction Tips", level=2)
    builder.add_info_box("Stuck Probe Recovery", [
        "Work probe back and forth gently with slow oscillation",
        "Add water around probe shaft to lubricate soil interface",
        "Use extraction handle tool if available",
        "Do not use excessive force - risk of probe damage",
        "For collapsed holes, allow water to soften soil before retry",
    ])

    builder.add_section_header("Data Backup Procedure", level=2)
    builder.add_numbered_list([
        "<b>Download data:</b> Connect tablet/laptop to base hub, download all raw data "
        "files, verify file sizes are non-zero.",
        "<b>Backup to multiple locations:</b> Copy to laptop hard drive, copy to USB drive, "
        "upload to cloud storage if connectivity available.",
        "<b>Data organization:</b> Use consistent naming (SITE_DATE_SCAN#.dat), create "
        "folder per site/day, include field notes file.",
        "<b>Verify backups:</b> Open files in viewer software, check data completeness, "
        "note any missing or corrupted files immediately.",
    ])

    # === Safety Reminders ===
    builder.add_section_header("10.13 Safety Reminders")

    builder.add_warning_box("Field Safety Checklist", [
        "Always inform someone of your field location and expected return time",
        "Stay hydrated and take regular breaks, especially in hot conditions",
        "Be aware of wildlife, terrain hazards, and site-specific dangers",
        "Follow all site-specific safety rules and permit requirements",
        "Keep first aid kit accessible and know location of nearest medical facility",
        "For UXO sites: EOD clearance required, soft insertion tools only, 100 m exclusion zone",
    ])

    # === Maintenance ===
    builder.add_section_header("10.14 Equipment Maintenance and Storage", level=1)
    
    builder.add_section_header("Post-Survey Cleaning", level=2)
    builder.add_bullet_list([
        "<b>Rods:</b> Wipe down with damp cloth to remove soil and clay. Do not use solvents.",
        "<b>Threads:</b> Clean M12 threads with a soft brush (toothbrush). Grit in threads causes seizing.",
        "<b>Connectors:</b> Inspect for dirt. Use compressed air or contact cleaner if needed.",
        "<b>Cables:</b> Wipe clean while coiling. Check for nicks in insulation."
    ])

    builder.add_section_header("Storage", level=2)
    builder.add_bullet_list([
        "<b>Batteries:</b> Store LiFePO4 batteries at 50-60% charge if unused for >1 month.",
        "<b>O-Rings:</b> Lightly grease O-rings with silicone grease to prevent drying.",
        "<b>Coiling:</b> Use 'over-under' coiling method to prevent cable twisting.",
        "<b>Environment:</b> Store in dry, cool location to prevent mold or thermal cycling damage."
    ])

    # === Cross-References ===
    builder.add_horizontal_rule()
    builder.add_note(
        "<i>For quick reference procedures, see Section 15: Quick Reference Card. "
        "For troubleshooting guidance, see Section 13: Troubleshooting. "
        "For calibration procedures, see Section 9: Calibration.</i>"
    )

    # Build and return
    return builder.build()


if __name__ == "__main__":
    output_path = build_section_10()
    print(f"\nSection 10 PDF generated successfully:")
    print(f"  {output_path}")
