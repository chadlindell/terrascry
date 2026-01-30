"""
HIRT PDF Generator - Section Scripts

Each section module generates a standalone PDF for one whitepaper section.
All modules follow the same pattern:
    - Import SectionPDFBuilder from lib.pdf_builder
    - Define custom diagram functions as needed
    - Implement main() that builds the complete section PDF

Usage:
    python section_XX_name.py

Or use the build scripts:
    python build_all.py       # Build all sections
    python build_combined.py  # Build combined whitepaper
"""

# Section metadata for programmatic access
SECTIONS = {
    0: ("index", "Index"),
    1: ("executive_summary", "Executive Summary"),
    2: ("physics_theory", "Physics Theory"),
    3: ("system_architecture", "System Architecture"),
    4: ("bill_of_materials", "Bill of Materials"),
    5: ("mechanical_design", "Mechanical Design"),
    6: ("electronics_circuits", "Electronics & Circuits"),
    7: ("assembly_wiring", "Assembly & Wiring"),
    8: ("testing_verification", "Testing & Verification"),
    9: ("calibration", "Calibration"),
    10: ("field_operations", "Field Operations"),
    11: ("data_recording", "Data Recording"),
    12: ("data_interpretation", "Data Interpretation"),
    13: ("troubleshooting", "Troubleshooting"),
    14: ("glossary", "Glossary"),
    15: ("quick_reference", "Quick Reference"),
    16: ("field_checklists", "Field Checklists"),
    17: ("application_scenarios", "Application Scenarios"),
    18: ("future_development", "Future Development"),
    19: ("ethics_legal_safety", "Ethics, Legal, Safety"),
}


def get_section_info(section_num):
    """Get (slug, title) for a section number."""
    return SECTIONS.get(section_num, (None, None))


def get_all_sections():
    """Get list of (num, slug, title) for all sections."""
    return [(num, slug, title) for num, (slug, title) in sorted(SECTIONS.items())]
