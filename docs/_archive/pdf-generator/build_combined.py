#!/usr/bin/env python3
"""
HIRT Whitepaper - Build Combined PDF

This script creates a single combined PDF containing all 19 sections
with a cover page and table of contents.

Usage:
    python build_combined.py [--skip-sections]

Options:
    --skip-sections    Don't rebuild individual sections first
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from io import BytesIO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, KeepTogether
)
from reportlab.lib import colors

from PyPDF2 import PdfReader, PdfWriter

# Import shared styles
from lib.styles import (
    get_styles, PAGE_WIDTH, PAGE_HEIGHT, MARGIN, CONTENT_WIDTH,
    PRIMARY, SECONDARY, ACCENT
)

# Section definitions
SECTIONS = [
    (0, "index", "Index"),
    (1, "executive_summary", "Executive Summary"),
    (2, "physics_theory", "Physics Theory"),
    (3, "system_architecture", "System Architecture"),
    (4, "bill_of_materials", "Bill of Materials"),
    (5, "mechanical_design", "Mechanical Design"),
    (6, "electronics_circuits", "Electronics & Circuits"),
    (7, "assembly_wiring", "Assembly & Wiring"),
    (8, "testing_verification", "Testing & Verification"),
    (9, "calibration", "Calibration"),
    (10, "field_operations", "Field Operations"),
    (11, "data_recording", "Data Recording"),
    (12, "data_interpretation", "Data Interpretation"),
    (13, "troubleshooting", "Troubleshooting"),
    (14, "glossary", "Glossary"),
    (15, "quick_reference", "Quick Reference"),
    (16, "field_checklists", "Field Checklists"),
    (17, "application_scenarios", "Application Scenarios"),
    (18, "future_development", "Future Development"),
    (19, "ethics_legal_safety", "Ethics, Legal, Safety"),
]


def get_script_dir():
    """Get the directory containing this script."""
    return Path(__file__).parent.absolute()


def create_cover_page(output_path):
    """
    Create the cover page PDF.

    Returns:
        Path to the cover page PDF
    """
    styles = get_styles()

    # Custom styles for cover page
    cover_title = ParagraphStyle(
        'CoverTitle',
        fontName='Times-Bold',
        fontSize=28,
        textColor=PRIMARY,
        alignment=TA_CENTER,
        spaceAfter=20,
        leading=34
    )

    cover_subtitle = ParagraphStyle(
        'CoverSubtitle',
        fontName='Times-Roman',
        fontSize=14,
        textColor=HexColor('#4a5568'),
        alignment=TA_CENTER,
        spaceAfter=30,
        leading=18
    )

    cover_meta = ParagraphStyle(
        'CoverMeta',
        fontName='Times-Roman',
        fontSize=11,
        textColor=HexColor('#666666'),
        alignment=TA_CENTER,
        spaceAfter=8
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN
    )

    story = []

    # Top spacing
    story.append(Spacer(1, 2.5 * inch))

    # Title
    story.append(Paragraph(
        "HIRT",
        cover_title
    ))
    story.append(Paragraph(
        "Hybrid Impedance-Resistivity<br/>Tomography System",
        ParagraphStyle(
            'CoverTitle2',
            fontName='Times-Bold',
            fontSize=22,
            textColor=PRIMARY,
            alignment=TA_CENTER,
            spaceAfter=30,
            leading=28
        )
    ))

    # Subtitle
    story.append(Paragraph(
        "Complete Technical Whitepaper",
        cover_subtitle
    ))

    # Description
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        "A Low-Cost, High-Resolution Subsurface Imaging System for<br/>"
        "Archaeological, Forensic, and Humanitarian Demining Applications",
        ParagraphStyle(
            'CoverDesc',
            fontName='Times-Italic',
            fontSize=12,
            textColor=HexColor('#555555'),
            alignment=TA_CENTER,
            leading=16
        )
    ))

    # Bottom section
    story.append(Spacer(1, 2 * inch))

    # Horizontal line
    story.append(Table(
        [['']],
        colWidths=[CONTENT_WIDTH],
        style=TableStyle([
            ('LINEBELOW', (0, 0), (-1, -1), 1, PRIMARY),
        ])
    ))
    story.append(Spacer(1, 20))

    # Metadata
    story.append(Paragraph("HIRT Development Team", cover_meta))
    story.append(Paragraph(
        f"<b>Version:</b> 2.0 | <b>Date:</b> {datetime.now().strftime('%B %Y')}",
        cover_meta
    ))
    story.append(Spacer(1, 30))

    # Section count
    story.append(Paragraph(
        f"<i>This document contains {len(SECTIONS)} sections covering all aspects of<br/>"
        "HIRT system design, construction, operation, and applications.</i>",
        ParagraphStyle(
            'CoverNote',
            fontName='Times-Roman',
            fontSize=10,
            textColor=HexColor('#888888'),
            alignment=TA_CENTER,
            leading=13
        )
    ))

    doc.build(story)
    return output_path


def create_toc_page(output_path, section_pages):
    """
    Create the table of contents PDF.

    Args:
        output_path: Path for output PDF
        section_pages: Dict mapping section number to starting page

    Returns:
        Path to the TOC PDF
    """
    styles = get_styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN
    )

    story = []

    # Title
    story.append(Paragraph(
        "Table of Contents",
        ParagraphStyle(
            'TOCTitle',
            fontName='Times-Bold',
            fontSize=18,
            textColor=PRIMARY,
            alignment=TA_CENTER,
            spaceBefore=20,
            spaceAfter=30
        )
    ))

    # TOC entries
    toc_style = ParagraphStyle(
        'TOCEntry',
        fontName='Times-Roman',
        fontSize=11,
        textColor=HexColor('#333333'),
        alignment=TA_LEFT,
        spaceAfter=8,
        leading=14
    )

    for num, slug, title in SECTIONS:
        page_num = section_pages.get(num, "...")

        # Create dotted leader line effect
        entry_text = f"<b>Section {num:02d}:</b> {title}"

        # Simple table row for alignment
        toc_data = [[
            Paragraph(entry_text, toc_style),
            Paragraph(str(page_num), ParagraphStyle(
                'TOCPage',
                fontName='Times-Roman',
                fontSize=11,
                alignment=TA_LEFT
            ))
        ]]

        toc_table = Table(toc_data, colWidths=[CONTENT_WIDTH - 0.5*inch, 0.5*inch])
        toc_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(toc_table)

    doc.build(story)
    return output_path


def merge_pdfs(pdf_paths, output_path):
    """
    Merge multiple PDFs into one.

    Args:
        pdf_paths: List of paths to PDFs to merge
        output_path: Path for output merged PDF

    Returns:
        Total page count
    """
    writer = PdfWriter()
    total_pages = 0

    for pdf_path in pdf_paths:
        if pdf_path.exists():
            reader = PdfReader(str(pdf_path))
            for page in reader.pages:
                writer.add_page(page)
            total_pages += len(reader.pages)
        else:
            print(f"  Warning: Missing PDF: {pdf_path}")

    with open(output_path, 'wb') as output_file:
        writer.write(output_file)

    return total_pages


def build_combined(skip_sections=False):
    """
    Build the combined whitepaper PDF.

    Args:
        skip_sections: If True, don't rebuild individual sections
    """
    script_dir = get_script_dir()
    output_dir = script_dir / "output"
    sections_dir = output_dir / "sections"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    sections_dir.mkdir(exist_ok=True)

    # Step 1: Build individual sections if needed
    if not skip_sections:
        print("Building individual section PDFs...")
        result = subprocess.run(
            [sys.executable, str(script_dir / "build_all.py")],
            cwd=str(script_dir)
        )
        if result.returncode != 0:
            print("Warning: Some sections failed to build")

    # Step 2: Find actual PDF files and calculate page numbers
    print("\nCalculating page numbers...")
    section_pages = {}
    section_pdfs = {}  # Map section number to actual PDF path
    current_page = 3  # Start after cover + TOC

    # Scan for actual PDF files by section number prefix
    for pdf_file in sorted(sections_dir.glob("*.pdf")):
        # Extract section number from filename (format: NN-*.pdf)
        try:
            num = int(pdf_file.name[:2])
            section_pdfs[num] = pdf_file
        except ValueError:
            continue

    for num, slug, title in SECTIONS:
        pdf_path = section_pdfs.get(num)

        if pdf_path and pdf_path.exists():
            reader = PdfReader(str(pdf_path))
            section_pages[num] = current_page
            current_page += len(reader.pages)
            print(f"  Section {num:02d}: starts page {section_pages[num]}, {len(reader.pages)} pages")
        else:
            section_pages[num] = "?"
            print(f"  Section {num:02d}: PDF not found")

    # Step 3: Create cover and TOC
    print("\nCreating cover page...")
    cover_path = output_dir / "_cover.pdf"
    create_cover_page(cover_path)

    print("Creating table of contents...")
    toc_path = output_dir / "_toc.pdf"
    create_toc_page(toc_path, section_pages)

    # Step 4: Merge all PDFs
    print("\nMerging PDFs...")
    pdf_paths = [cover_path, toc_path]

    for num, slug, title in SECTIONS:
        pdf_path = section_pdfs.get(num)
        if pdf_path:
            pdf_paths.append(pdf_path)
        else:
            print(f"  Warning: No PDF found for section {num:02d}")

    combined_path = output_dir / "HIRT_Whitepaper_Complete.pdf"
    total_pages = merge_pdfs(pdf_paths, combined_path)

    # Clean up temporary files
    if cover_path.exists():
        cover_path.unlink()
    if toc_path.exists():
        toc_path.unlink()

    # Report results
    print("\n" + "=" * 60)
    print("COMBINED PDF COMPLETE")
    print("=" * 60)
    print(f"Output: {combined_path}")
    print(f"Total pages: {total_pages}")
    print(f"File size: {combined_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 60)

    return combined_path


def main():
    parser = argparse.ArgumentParser(description="Build Combined HIRT Whitepaper PDF")
    parser.add_argument("--skip-sections", action="store_true",
                       help="Don't rebuild individual sections first")
    args = parser.parse_args()

    print("=" * 60)
    print("HIRT WHITEPAPER - BUILD COMBINED PDF")
    print("=" * 60)

    try:
        build_combined(skip_sections=args.skip_sections)
    except ImportError as e:
        print(f"\nError: Missing required package: {e}")
        print("Install with: pip install PyPDF2 reportlab matplotlib")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
