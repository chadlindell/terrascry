#!/usr/bin/env python3
"""
HIRT PDF Generator - Section 14: Glossary

Generates a publication-quality PDF for the HIRT glossary section.
Contains technical terms, acronyms, and definitions organized by category.

Usage:
    python section_14_glossary.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.pdf_builder import SectionPDFBuilder
from lib.styles import CONTENT_WIDTH, PRIMARY, SECONDARY, ACCENT, get_styles

from reportlab.platypus import Table, TableStyle, Spacer, Paragraph
from reportlab.lib import colors
from reportlab.lib.colors import HexColor


class GlossaryPDFBuilder(SectionPDFBuilder):
    """
    Builds the Glossary section PDF with formatted definition lists.
    """

    def __init__(self):
        super().__init__(
            section_num=14,
            title="Glossary"
        )

    def add_definition_term(self, term, acronym=None):
        """
        Add a glossary term header.

        Args:
            term: The term being defined
            acronym: Optional acronym in parentheses
        """
        if acronym:
            text = f"<b>{acronym} ({term})</b>"
        else:
            text = f"<b>{term}</b>"

        self.story.append(Paragraph(text, self.styles['Subsection']))

    def add_definition(self, definition_items):
        """
        Add definition bullet points for a term.

        Args:
            definition_items: List of definition/description strings
        """
        for item in definition_items:
            self.story.append(Paragraph(
                f"\u2022  {item}",
                self.styles['BulletItem']
            ))

    def add_glossary_entry(self, term, definitions, acronym=None):
        """
        Add a complete glossary entry as a single dense paragraph.

        Args:
            term: The term being defined
            definitions: List of definition/description strings
            acronym: Optional acronym
        """
        # Format the term part
        if acronym:
            term_text = f"<b>{term} ({acronym})</b>"
        else:
            term_text = f"<b>{term}</b>"

        # Join definition items into a single string
        # If there are multiple items, join them with semicolons or periods
        def_text = "; ".join(definitions)
        if not def_text.endswith('.'):
            def_text += "."

        # Create the full paragraph: "Term: Definition."
        full_text = f"{term_text}: {def_text}"

        # Use a Hanging Indent style for glossary?
        # Standard Body style is fine for now, or maybe small hanging indent.
        # Let's use Body style but ensure density.
        self.story.append(Paragraph(full_text, self.styles['Body']))

    def add_category_header(self, text):
        """
        Add a category header (e.g., "A-D", "Measurement Terms").

        Args:
            text: Category header text
        """
        self.story.append(Paragraph(text, self.styles['Section']))


def build_glossary_pdf():
    """
    Build the complete Section 14: Glossary PDF.

    Returns:
        Path to the generated PDF file
    """
    builder = GlossaryPDFBuilder()

    # Title block
    builder.add_title_block(
        subtitle="Technical Terms, Acronyms, and Definitions",
        version="2.0"
    )

    # Introduction
    builder.add_body_text(
        "This glossary provides definitions for technical terms, acronyms, and "
        "specialized vocabulary used throughout the HIRT whitepaper. Terms are "
        "organized alphabetically within functional categories to facilitate "
        "quick reference during system assembly, calibration, and field deployment.",
        first_paragraph=True
    )

    builder.add_spacer(12)

    # ==========================================================================
    # ACRONYMS AND TERMS
    # ==========================================================================
    builder.add_section_header("Acronyms and Terms", level=1)

    # --- A-D ---
    builder.add_category_header("A-D")

    builder.add_glossary_entry(
        "Analog-to-Digital Converter",
        ["Converts analog voltage signals to digital values for processing"],
        acronym="ADC"
    )

    builder.add_glossary_entry(
        "Amplitude",
        ["Magnitude of a signal, typically measured in volts or normalized units"]
    )

    builder.add_glossary_entry(
        "Baseline",
        [
            "Distance/geometry between source and receiver probes",
            "Longer baselines provide deeper investigation depth"
        ]
    )

    builder.add_glossary_entry(
        "Bill of Materials",
        ["Complete list of components needed to build the system"],
        acronym="BOM"
    )

    builder.add_glossary_entry(
        "Crosshole",
        [
            "Measurement geometry where sensors are placed in separate boreholes/probes",
            "Provides true tomographic coverage through the volume"
        ]
    )

    # --- E-H ---
    builder.add_category_header("E-H")

    builder.add_glossary_entry(
        "Electrical Resistivity Tomography",
        [
            "Method using current injection and voltage measurement to map soil resistivity",
            "Detects moisture, disturbance, voids, and soil variations"
        ],
        acronym="ERT"
    )

    builder.add_glossary_entry(
        "Eddy Currents",
        [
            "Electrical currents induced in conductive materials by changing magnetic fields",
            "Cause attenuation and phase lag in MIT measurements"
        ]
    )

    builder.add_glossary_entry(
        "Ferrite Core",
        [
            "Magnetic material used in coils to increase inductance and efficiency",
            "Typically rod-shaped for probe applications"
        ]
    )

    builder.add_glossary_entry(
        "Frequency",
        [
            "Number of cycles per second, measured in Hz (Hertz)",
            "Lower frequencies penetrate deeper; higher frequencies provide better resolution"
        ]
    )

    # --- I-L ---
    builder.add_category_header("I-L")

    builder.add_glossary_entry(
        "Inductance",
        [
            "Property of a coil that resists changes in current",
            "Measured in Henries (H) or millihenries (mH)"
        ]
    )

    builder.add_glossary_entry(
        "Inversion",
        [
            "Mathematical process to reconstruct 3D property distribution from measurements",
            "Software step (not covered in this hardware guide)"
        ]
    )

    builder.add_glossary_entry(
        "Lock-in Detection",
        [
            "Technique to extract small signals at a known reference frequency with high SNR",
            "Can be analog (AD630) or digital (DSP-based)"
        ]
    )

    # --- M-P ---
    builder.add_category_header("M-P")

    builder.add_glossary_entry(
        "Magnetic Induction Tomography",
        [
            "Low-frequency electromagnetic method using coils",
            "Measures amplitude and phase changes caused by conductive objects"
        ],
        acronym="MIT"
    )

    builder.add_glossary_entry(
        "Microcontroller Unit",
        [
            "Small computer on a chip (e.g., ESP32)",
            "Located exclusively in the Central Hub; manages signal generation, "
            "multiplexing, and data acquisition"
        ],
        acronym="MCU"
    )

    builder.add_glossary_entry(
        "Probe",
        [
            "The passive sensor assembly inserted into the ground",
            "Contains only coils and electrodes; no active electronics are located downhole"
        ]
    )

    builder.add_glossary_entry(
        "Zone Hub",
        [
            "A local, passive breakout box that aggregates 4 probes into a single "
            "high-density trunk cable"
        ]
    )

    builder.add_glossary_entry(
        "Trunk Cable",
        [
            "A high-density shielded multi-core cable (e.g., DB25) that carries "
            "analog signals from a Zone Hub to the Central Hub"
        ]
    )

    # --- Q-T ---
    builder.add_category_header("Q-T")

    builder.add_glossary_entry(
        "Q Factor",
        [
            "Quality factor of a coil, indicates efficiency",
            "Higher Q = lower losses, better performance"
        ]
    )

    builder.add_glossary_entry(
        "Reciprocity",
        [
            "Principle that TX->RX measurement should equal RX->TX measurement",
            "Used for quality control"
        ]
    )

    builder.add_glossary_entry(
        "Resistivity",
        [
            "Property of material to resist electrical current flow",
            "Measured in ohm-meters (ohm-m)",
            "High resistivity: dry soil, voids",
            "Low resistivity: wet soil, clay, metal"
        ]
    )

    builder.add_glossary_entry(
        "Receive/Receiver",
        ["Receiving coil or probe that measures signals"],
        acronym="RX"
    )

    builder.add_glossary_entry(
        "Sensitivity Volume",
        [
            "The 3D region that contributes most to a particular measurement",
            "Depends on probe spacing, frequency, and soil properties"
        ]
    )

    # --- U-Z ---
    builder.add_category_header("U-Z")

    builder.add_glossary_entry(
        "Transmit/Transmitter",
        ["Transmitting coil or probe that generates signals"],
        acronym="TX"
    )

    builder.add_glossary_entry(
        "Tomography",
        [
            "Imaging method that reconstructs 3D structure from multiple measurements",
            "Similar to medical CT scanning"
        ]
    )

    builder.add_glossary_entry(
        "Unexploded Ordnance",
        [
            "Live explosive devices that may be present at WWII sites",
            "Requires EOD clearance before deployment"
        ],
        acronym="UXO"
    )

    # ==========================================================================
    # MEASUREMENT TERMS
    # ==========================================================================
    builder.add_section_header("Measurement Terms", level=1)

    builder.add_glossary_entry(
        "Apparent Resistivity",
        [
            "Calculated resistivity from voltage/current measurements",
            "May differ from true resistivity due to measurement geometry"
        ]
    )

    builder.add_glossary_entry(
        "Attenuation",
        [
            "Reduction in signal amplitude",
            "Indicates presence of conductive objects or losses"
        ]
    )

    builder.add_glossary_entry(
        "Common-Mode Rejection",
        [
            "Ability to reject signals common to both inputs",
            "Important for differential measurements"
        ]
    )

    builder.add_glossary_entry(
        "Signal-to-Noise Ratio",
        [
            "Ratio of signal strength to noise level",
            "Higher SNR = better data quality"
        ],
        acronym="SNR"
    )

    # ==========================================================================
    # FIELD TERMS
    # ==========================================================================
    builder.add_section_header("Field Terms", level=1)

    builder.add_glossary_entry(
        "Section",
        [
            "Grid area surveyed in one deployment cycle",
            "Typically 10x10 m, manageable by small team"
        ]
    )

    builder.add_glossary_entry(
        "Node",
        [
            "Probe insertion point in the grid",
            "Spacing determines resolution and depth"
        ]
    )

    builder.add_glossary_entry(
        "Pilot Rod",
        [
            "Metal rod used to create hole for probe insertion",
            "Removed before inserting sensor rod"
        ]
    )

    builder.add_glossary_entry(
        "Rim Deployment",
        [
            "Placing probes around perimeter rather than throughout area",
            "Reduces ground disturbance"
        ]
    )

    # Build and return
    return builder.build()


def main():
    """Main entry point for standalone execution."""
    print("Generating HIRT Section 14: Glossary PDF...")
    output_path = build_glossary_pdf()
    print(f"Successfully generated: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
