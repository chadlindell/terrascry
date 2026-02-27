#!/usr/bin/env python3
"""
HIRT Section 02: Physics Theory - Publication-Quality PDF Generator

Covers the physics principles underlying the HIRT system:
- Electromagnetic skin depth
- MIT coil coupling (1/r^3 decay)
- ERT geometric factors
- Surface vs crosshole geometry advantages
- Multi-frequency response characteristics

Diagrams (7 total):
1. Skin depth vs. frequency curves (5 soil types)
2. TX/RX waveform comparison (amplitude/phase lag)
3. Surface vs crosshole ray path comparison
4. ERT geometric factor visualization
5. MIT 1/r^3 coupling decay curves
6. Multi-frequency response plot
7. Resolution vs depth comparison

Usage:
    python section_02_physics_theory.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pdf_builder import SectionPDFBuilder
from lib.styles import CONTENT_WIDTH
from lib.diagrams.physics import (
    create_skin_depth_plot,
    create_tx_rx_waveforms,
    create_coupling_decay_plot,
    create_ert_geometric_factor,
    create_frequency_response,
    create_ray_path_comparison,
    create_resolution_depth_plot,
)


def build_section_02():
    """Build Section 02: Physics Theory PDF."""

    print("=" * 60)
    print("HIRT Section 02: Physics Theory - PDF Generator")
    print("=" * 60)

    # Initialize builder
    builder = SectionPDFBuilder(
        section_num=2,
        title="Physics Theory"
    )

    # === TITLE BLOCK ===
    builder.add_title_block(
        subtitle="Electromagnetic and Galvanic Principles for Subsurface Imaging"
    )

    # === INTRODUCTION ===
    builder.add_section_header("2.1 Overview", level=1)
    builder.add_body_text(
        "The HIRT system combines two complementary geophysical sensing modalities: "
        "Magneto-Inductive Tomography (MIT-3D) and Electrical Resistivity Tomography "
        "(ERT-Lite). Understanding the physics underlying each method is essential for "
        "proper system deployment, frequency selection, and data interpretation. This "
        "section provides a practical, field-level treatment of the relevant physics "
        "principles.",
        first_paragraph=True
    )

    # === MIT-3D SECTION ===
    builder.add_section_header("2.2 MIT-3D (Low-Frequency Electromagnetic)", level=1)

    builder.add_section_header("2.2.1 Operating Principle", level=2)
    builder.add_body_text(
        "MIT-3D uses oscillating magnetic fields to detect conductive anomalies in the "
        "subsurface. A transmit (TX) coil drives a stable sinusoidal current at frequencies "
        "between 2-50 kHz, generating a primary magnetic field. When this field encounters "
        "conductive material&mdash;such as metal objects or high-conductivity soil zones&mdash;"
        "it induces eddy currents within the conductor.",
        first_paragraph=True
    )
    builder.add_body_text(
        "These eddy currents generate a secondary magnetic field that opposes the primary "
        "field. Receive (RX) coils positioned at known distances from the TX coil measure "
        "the combined field. The presence of conductive targets manifests as measurable "
        "changes in both the <b>amplitude</b> and <b>phase</b> of the received signal "
        "relative to the transmitted waveform."
    )

    # Generate and add TX/RX waveform figure
    print("  Generating TX/RX waveform diagram...")
    fig_waveforms = create_tx_rx_waveforms()
    builder.add_figure(
        fig_waveforms,
        "TX and RX waveform comparison at 10 kHz. The transmitted signal (top) induces "
        "a received signal (bottom) that is both attenuated and phase-shifted. The "
        "amplitude ratio and phase lag encode information about the conductivity and "
        "geometry of subsurface targets along the TX-RX path.",
        width=CONTENT_WIDTH * 0.95
    )

    builder.add_section_header("2.2.2 Frequency Selection", level=2)
    builder.add_body_text(
        "Operating frequency determines the trade-off between depth penetration and "
        "near-surface sensitivity:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "<b>Lower frequency (2-10 kHz)</b>: Deeper penetration, better for targets at 2.5-4+ meters",
        "<b>Higher frequency (20-50 kHz)</b>: Sharper sensitivity near probes, better for 0.5-1.5 meter targets",
        "<b>Multi-frequency sweep</b>: Recommended for unknown target depths; provides spectral discrimination"
    ])

    # Frequency selection table
    freq_table_data = [
        ['Target Depth', 'Recommended Frequencies', 'Integration Time'],
        ['0.5-1.5 m (shallow)', '20-50 kHz', '1-3 sec'],
        ['1.5-2.5 m (mid-range)', '10-20 kHz', '3-5 sec'],
        ['2.5-4 m (deep)', '2-10 kHz', '5-15 sec'],
        ['>4 m (very deep)', '2-5 kHz', '10-30 sec'],
    ]
    builder.add_table(
        freq_table_data,
        col_widths=[CONTENT_WIDTH * 0.3, CONTENT_WIDTH * 0.4, CONTENT_WIDTH * 0.3],
        caption="Target-dependent frequency selection guidelines. Higher frequencies "
                "provide sharper near-surface resolution while lower frequencies enable "
                "deeper penetration. Longer integration times improve SNR at all frequencies."
    )

    # === SKIN DEPTH SECTION ===
    builder.add_section_header("2.3 Electromagnetic Skin Depth", level=1)
    builder.add_body_text(
        "Electromagnetic skin depth (&delta;) defines how deeply alternating EM fields "
        "penetrate into conductive media before being attenuated to 1/e (~37%) of their "
        "surface value. This fundamental parameter is given by:",
        first_paragraph=True
    )

    # Skin depth equation
    builder.add_equation(
        "&delta; = &radic;(2 / &omega;&mu;&sigma;)"
    )
    builder.add_body_text(
        "where &omega; = 2&pi;f is the angular frequency, &mu; = &mu;<sub>0</sub>&mu;<sub>r</sub> "
        "is the magnetic permeability (typically &mu;<sub>0</sub> = 4&pi;&times;10<sup>-7</sup> H/m "
        "for non-magnetic soils), and &sigma; is the electrical conductivity in S/m."
    )

    # Skin depth table
    skin_depth_table = [
        ['Conductivity (S/m)', '2 kHz', '5 kHz', '10 kHz', '20 kHz', '50 kHz'],
        ['0.01 (dry sand)', '112 m', '71 m', '50 m', '35 m', '22.5 m'],
        ['0.1 (moist sand)', '35.6 m', '22.5 m', '15.9 m', '11.2 m', '7.1 m'],
        ['0.5 (wet clay)', '15.9 m', '10.1 m', '7.1 m', '5.0 m', '3.2 m'],
        ['1.0 (saturated clay)', '11.2 m', '7.1 m', '5.0 m', '3.6 m', '2.3 m'],
    ]
    builder.add_table(
        skin_depth_table,
        col_widths=[CONTENT_WIDTH * 0.24] + [CONTENT_WIDTH * 0.152] * 5,
        caption="Skin depth values for typical soil conductivities across the HIRT "
                "operating frequency range. Even in the most conductive soils, skin "
                "depth exceeds typical investigation depths at HIRT frequencies."
    )

    # Generate and add skin depth plot
    print("  Generating skin depth diagram...")
    fig_skin_depth = create_skin_depth_plot()
    builder.add_figure(
        fig_skin_depth,
        "Electromagnetic skin depth versus frequency for five soil types spanning "
        "the conductivity range from dry sand (0.001 S/m) to saline/contaminated "
        "soil (1.0 S/m). The green shaded region indicates the HIRT operating "
        "range (2-50 kHz). The dashed line marks 10 kHz, a typical operating frequency.",
        width=CONTENT_WIDTH * 0.95
    )

    builder.add_info_box(
        "KEY INSIGHT: Skin Depth vs. Coupling Geometry",
        [
            "Skin depth alone does NOT limit MIT investigation depth in most field conditions",
            "Even in saturated clay at 50 kHz, skin depth (2.3 m) exceeds typical probe spacing",
            "The practical depth limitation is coil coupling geometry (1/r^3 decay), not skin depth",
            "Effective MIT depth is approximately 1-2x probe spacing in near-field conditions"
        ]
    )

    # === COUPLING DECAY SECTION ===
    builder.add_section_header("2.4 MIT Coil Coupling and 1/r^3 Decay", level=1)
    builder.add_body_text(
        "In the near-field regime where HIRT operates, the magnetic field coupling between "
        "TX and RX coils decays as the cube of the separation distance. This 1/r<sup>3</sup> "
        "relationship is the fundamental limitation on MIT investigation depth&mdash;not "
        "electromagnetic skin depth.",
        first_paragraph=True
    )
    builder.add_body_text(
        "The magnetic dipole field strength at distance r from a coil is proportional to "
        "1/r<sup>3</sup> for the near-field component (which dominates at distances much "
        "less than the wavelength). For a round-trip measurement (TX to target to RX), "
        "the sensitivity can decay as fast as 1/r<sup>6</sup> depending on target geometry."
    )

    # Generate and add coupling decay plot
    print("  Generating coupling decay diagram...")
    fig_coupling = create_coupling_decay_plot()
    builder.add_figure(
        fig_coupling,
        "Magnetic coupling decay versus distance. The MIT near-field coupling (1/r^3) "
        "decays much faster than ERT geometric spreading (1/r^2). The green shaded "
        "region indicates typical HIRT probe spacing (1.5-3.5 m). Signal strength "
        "drops below practical detection thresholds beyond ~4-5 m separation.",
        width=CONTENT_WIDTH * 0.90
    )

    builder.add_body_text(
        "This rapid decay has important implications for system design:"
    )
    builder.add_bullet_list([
        "<b>Probe spacing</b>: Maximum useful TX-RX separation is ~3-4 m for typical targets",
        "<b>Array density</b>: Dense probe arrays (1-2 m spacing) provide better coverage than sparse arrays",
        "<b>Signal processing</b>: High dynamic range ADCs required to capture both strong and weak paths",
        "<b>Integration time</b>: Longer measurement integration needed for distant TX-RX pairs"
    ])

    # === ERT SECTION ===
    builder.add_section_header("2.5 ERT-Lite (Galvanic Method)", level=1)

    builder.add_section_header("2.5.1 Operating Principle", level=2)
    builder.add_body_text(
        "Electrical Resistivity Tomography injects electrical current through the subsurface "
        "and measures the resulting voltage distribution. Two electrodes inject a known "
        "current (typically 0.5-2 mA for safety), while other electrodes measure voltage "
        "differences. The ratio V/I, combined with the electrode geometry, yields the "
        "apparent resistivity &rho;<sub>a</sub> of the material between electrodes.",
        first_paragraph=True
    )
    builder.add_body_text(
        "Unlike MIT, ERT does not respond to metallic targets directly. Instead, it detects "
        "<b>resistivity contrasts</b> such as disturbed fill, moisture variations, compacted "
        "layers, voids, and the boundaries of burial pits. The combination of MIT (metal-"
        "sensitive) and ERT (structure-sensitive) provides comprehensive target characterization."
    )

    builder.add_section_header("2.5.2 Geometric Factor K", level=2)
    builder.add_body_text(
        "The geometric factor K converts measured V/I ratios to apparent resistivity:",
        first_paragraph=True
    )
    builder.add_equation(
        "&rho;<sub>a</sub> = K &times; (V / I)"
    )
    builder.add_body_text(
        "For HIRT's borehole electrode geometry, the geometric factor is approximately:"
    )
    builder.add_equation(
        "K &asymp; &pi; &times; L"
    )
    builder.add_body_text(
        "where L is the separation distance between current injection electrodes."
    )

    # Geometric factor table
    k_table_data = [
        ['Configuration', 'L (electrode separation)', 'K'],
        ['A(0.5m) to B(1.5m)', '1.0 m', '3.14 ohm-m'],
        ['A(0.5m) to C(2.5m)', '2.0 m', '6.28 ohm-m'],
        ['A(0.5m) to D(3.0m)', '2.5 m', '7.85 ohm-m'],
    ]
    builder.add_table(
        k_table_data,
        col_widths=[CONTENT_WIDTH * 0.35, CONTENT_WIDTH * 0.35, CONTENT_WIDTH * 0.30],
        caption="ERT geometric factors for HIRT ring electrode configurations. "
                "Larger electrode separations provide greater depth of investigation "
                "but lower resolution."
    )

    # Generate and add ERT geometric factor figure
    print("  Generating ERT geometric factor diagram...")
    fig_ert = create_ert_geometric_factor()
    builder.add_figure(
        fig_ert,
        "ERT Wenner array configuration showing current injection electrodes (C1, C2) "
        "and potential measurement electrodes (P1, P2). Red curves indicate current "
        "flow lines; blue dashed lines show equipotential surfaces. The green ellipse "
        "marks the zone of maximum sensitivity. The geometric factor K = 2&pi;a for "
        "equal electrode spacing a.",
        width=CONTENT_WIDTH * 0.95
    )

    builder.add_section_header("2.5.3 Depth of Investigation", level=2)
    builder.add_body_text(
        "The ERT depth of investigation (DOI) is approximately 1.5x the maximum electrode "
        "separation. For HIRT probe configurations:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "<b>1.5 m probes</b> (rings at 0.5m, 1.5m): DOI = 2-3 m",
        "<b>3.0 m probes</b> (rings at 0.5m, 1.5m, 2.5m): DOI = 3-5 m (edge cases to 6 m)"
    ])

    # === MULTI-FREQUENCY RESPONSE ===
    builder.add_section_header("2.6 Multi-Frequency Response", level=1)
    builder.add_body_text(
        "Different subsurface features exhibit characteristic frequency-dependent responses "
        "that enable target discrimination. Metal targets respond strongly at lower "
        "frequencies where skin depth into the metal is greater. Soil disturbances and "
        "moisture contrasts show broader, less frequency-dependent signatures.",
        first_paragraph=True
    )

    # Generate and add frequency response plot
    print("  Generating multi-frequency response diagram...")
    fig_freq = create_frequency_response()
    builder.add_figure(
        fig_freq,
        "Multi-frequency target discrimination. Metal targets (red) show strong response "
        "at lower frequencies with characteristic roll-off. Soil disturbances (purple, "
        "dashed) exhibit broader frequency response. The combined signal (gray, dotted) "
        "can be decomposed through multi-frequency analysis. The green region indicates "
        "the HIRT sweep range (2-50 kHz).",
        width=CONTENT_WIDTH * 0.95
    )

    builder.add_body_text(
        "Multi-frequency sweeps enable:"
    )
    builder.add_bullet_list([
        "<b>Target classification</b>: Metal vs. soil anomaly discrimination based on spectral signature",
        "<b>Depth estimation</b>: Lower frequencies see deeper; response vs. frequency constrains depth",
        "<b>Size estimation</b>: Larger metal objects have lower characteristic frequencies",
        "<b>Conductivity mapping</b>: Frequency response shape indicates soil conductivity distribution"
    ])

    # === CROSSHOLE ADVANTAGE ===
    builder.add_section_header("2.7 Why Crosshole Geometry Beats Surface Methods", level=1)
    builder.add_body_text(
        "HIRT's borehole/crosshole tomography provides fundamental physics advantages over "
        "surface geophysical methods for targets deeper than approximately 1.5 m. These "
        "advantages stem from the geometry of measurement ray paths.",
        first_paragraph=True
    )

    builder.add_section_header("2.7.1 Ray Path Geometry", level=2)
    builder.add_body_text(
        "<b>Surface methods</b> must send energy down to a target and receive the return "
        "signal&mdash;doubling the path length and exponentially increasing attenuation. "
        "Sensitivity decreases as 1/r<sup>2</sup> to 1/r<sup>4</sup> with depth.",
        first_paragraph=True
    )
    builder.add_body_text(
        "<b>Crosshole methods</b> send signals horizontally through the target volume. "
        "Energy travels directly between probes at depth, with sensitivity concentrated "
        "precisely where targets are located. This geometry provides 2-5x better resolution "
        "than surface methods at depths exceeding 2 m."
    )

    # Generate and add ray path comparison
    print("  Generating ray path comparison diagram...")
    fig_raypath = create_ray_path_comparison()
    builder.add_figure(
        fig_raypath,
        "Comparison of (a) surface and (b) crosshole ray path geometries. Surface "
        "sensors create curved, indirect paths with poor sensitivity at depth. Crosshole "
        "probes provide direct, straight paths through the target volume. The checkmark "
        "in (b) indicates confident target detection; the question mark in (a) indicates "
        "ambiguous response at depth.",
        width=CONTENT_WIDTH * 0.95
    )

    builder.add_section_header("2.7.2 Additional Crosshole Advantages", level=2)
    builder.add_body_text(
        "Beyond ray path geometry, crosshole methods offer several practical advantages:",
        first_paragraph=True
    )
    builder.add_bullet_list([
        "<b>No surface interference</b>: Measurements occur below near-surface heterogeneity, "
        "fill, roots, and cultural noise (fences, vehicles)",
        "<b>True 3D sampling</b>: Multiple ray paths at different angles enable tomographic "
        "reconstruction&mdash;not pseudo-depth estimation from diffraction patterns",
        "<b>Superior depth discrimination</b>: Targets at 3 m vs. 4 m depth are clearly "
        "distinguishable; surface methods show nearly identical responses"
    ])

    # Resolution comparison table
    resolution_table = [
        ['Method', 'Lateral Resolution', 'Depth Resolution', 'At 3m Depth'],
        ['Surface Magnetometry', '1-2 m', 'Poor', '~2 m'],
        ['GPR', '0.3-0.5 m (shallow)', '0.05-0.1 m (shallow)', '>1 m in clay'],
        ['Surface ERT (Wenner)', '~1x spacing', '~0.5x spacing', '2-3 m'],
        ['EM31/CMD', '1-2 m', 'Poor', '~2 m'],
        ['HIRT Crosshole', '0.5-1x spacing', '0.3-0.5x spacing', '0.75-1.5 m'],
    ]
    builder.add_table(
        resolution_table,
        col_widths=[CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.22, CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.28],
        caption="Resolution comparison across geophysical methods. HIRT crosshole "
                "geometry maintains superior resolution at depth where surface methods "
                "degrade significantly."
    )

    # Generate and add resolution vs depth plot
    print("  Generating resolution vs depth diagram...")
    fig_resolution = create_resolution_depth_plot()
    builder.add_figure(
        fig_resolution,
        "Spatial resolution versus investigation depth. Surface methods (dashed gray) "
        "show quadratic degradation of resolution with depth. HIRT crosshole methods "
        "(solid green) maintain approximately linear resolution scaling. The red dashed "
        "line indicates typical UXB diameter (~0.5 m). The orange shaded region marks "
        "typical UXB burial depths (2-4 m).",
        width=CONTENT_WIDTH * 0.90
    )

    # === DEPTH SUMMARY ===
    builder.add_section_header("2.8 Depth of Investigation Summary", level=1)
    builder.add_body_text(
        "The effective investigation depth depends on probe configuration, soil conditions, "
        "and the measurement technique (MIT vs. ERT). The following table summarizes "
        "depth capabilities with associated confidence levels:",
        first_paragraph=True
    )

    depth_summary_table = [
        ['Configuration', 'MIT Depth', 'ERT Depth', 'Combined Claim'],
        ['1.5 m probes, 2 m spacing', '2-3 m', '2-3 m', '2-3 m (HIGH confidence)'],
        ['3.0 m probes, 2 m spacing', '3-4 m', '3-5 m', '3-5 m (MEDIUM confidence)'],
        ['Edge cases (conductive soil)', '2-3 m', '4-6 m', 'Up to 6 m (LOW confidence)'],
    ]
    builder.add_table(
        depth_summary_table,
        col_widths=[CONTENT_WIDTH * 0.30, CONTENT_WIDTH * 0.18, CONTENT_WIDTH * 0.18, CONTENT_WIDTH * 0.34],
        caption="Depth of investigation summary by configuration and confidence level. "
                "Conservative claims are appropriate for field planning; extended depths "
                "may be achievable under favorable conditions."
    )

    builder.add_body_text(
        "<b>Rule of thumb:</b> With rods inserted to depth D, the sensitivity volume "
        "typically extends D to 1.5D below the surface. Actual depth depends on soil "
        "conductivity, probe spacing, measurement frequency (MIT), and current injection "
        "geometry (ERT)."
    )

    # === OPTIMAL WORKFLOW ===
    builder.add_section_header("2.9 When Crosshole Wins vs. Loses", level=1)
    builder.add_body_text(
        "The physics supports strategic use of both surface and crosshole methods. "
        "Understanding their respective strengths enables optimal workflow design.",
        first_paragraph=True
    )

    builder.add_body_text(
        "<b>HIRT crosshole geometry is SUPERIOR for:</b>"
    )
    builder.add_bullet_list([
        "Targets deeper than 1.5-2 m",
        "3D localization requirements",
        "Conductive soils where GPR fails",
        "Distinguishing multiple targets at similar depths",
        "Non-ferrous (aluminum) target detection"
    ])

    builder.add_body_text(
        "<b>Surface methods remain SUPERIOR for:</b>"
    )
    builder.add_bullet_list([
        "Rapid large-area screening (10x faster coverage)",
        "Shallow targets (<1 m) where GPR resolution excels",
        "Purely ferrous targets (magnetometry)",
        "Initial site characterization before targeted investigation"
    ])

    builder.add_info_box(
        "RECOMMENDED WORKFLOW",
        [
            "Stage 1: Surface screening (magnetometry, GPR, EM31) to identify anomalies quickly over large areas",
            "Stage 2: HIRT crosshole follow-up to characterize identified anomalies with superior 3D resolution",
            "This two-stage approach leverages the strengths of both methods while minimizing deployment time"
        ]
    )

    # === REFERENCES ===
    references = [
        "[1] Wait, J.R. (1951). The magnetic dipole over the horizontally stratified earth. "
        "Canadian Journal of Physics, 29(6), 577-592.",

        "[2] Ward, S.H., & Hohmann, G.W. (1988). Electromagnetic theory for geophysical "
        "applications. In Electromagnetic Methods in Applied Geophysics, Vol. 1, Theory. "
        "Society of Exploration Geophysicists.",

        "[3] Telford, W.M., Geldart, L.P., & Sheriff, R.E. (1990). Applied Geophysics, "
        "2nd Edition. Cambridge University Press.",

        "[4] Griffiths, D.H., & Barker, R.D. (1993). Two-dimensional resistivity imaging "
        "and modelling in areas of complex geology. Journal of Applied Geophysics, 29(3-4), 211-226.",

        "[5] Butler, D.K. (2001). Potential fields methods for location of unexploded ordnance. "
        "The Leading Edge, 20(8), 890-895.",

        "[6] Fernandez, J.P., et al. (2010). Realistic subsurface anomaly discrimination "
        "using electromagnetic induction and an SVM classifier. EURASIP Journal on Advances "
        "in Signal Processing, 2010, Article 6.",
    ]
    builder.add_references(references)

    # === BUILD PDF ===
    print("\nBuilding PDF document...")
    output_path = builder.build()

    print("=" * 60)
    print(f"SUCCESS: PDF generated at:")
    print(f"  {output_path}")
    print("=" * 60)

    return output_path


if __name__ == "__main__":
    build_section_02()
