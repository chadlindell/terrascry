"""
HIRT PDF Generator - Styles Module

Color palette, typography, and paragraph styles for publication-quality PDFs.
Extracted from create_hirt_intro.py for shared use across all section generators.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT

# === Page Setup ===
PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 0.75 * inch
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

# === Color Palette ===
PRIMARY = HexColor('#1a365d')      # Navy - main headings
SECONDARY = HexColor('#2c5282')    # Blue - subheadings, probes
ACCENT = HexColor('#3182ce')       # Light blue - RX coils, highlights
SUCCESS = HexColor('#38a169')      # Green - TX coils, positive
WARNING = HexColor('#c53030')      # Red - warnings, targets
LIGHT_BG = HexColor('#f7fafc')     # Light gray - table backgrounds
GROUND_TAN = HexColor('#d4a373')   # Tan - ground/soil

# Additional diagram colors
ORANGE = HexColor('#ed8936')       # ERT rings
PURPLE = HexColor('#805ad5')       # Disturbed soil
GRAY_DARK = HexColor('#4a5568')    # Dark gray - connectors
GRAY_MED = HexColor('#718096')     # Medium gray - borders
GRAY_LIGHT = HexColor('#a0aec0')   # Light gray - inactive

# Convenience dict for matplotlib (hex strings)
COLOR_PALETTE = {
    'primary': '#1a365d',
    'secondary': '#2c5282',
    'accent': '#3182ce',
    'success': '#38a169',
    'warning': '#c53030',
    'light_bg': '#f7fafc',
    'ground_tan': '#d4a373',
    'orange': '#ed8936',
    'purple': '#805ad5',
    'gray_dark': '#4a5568',
    'gray_med': '#718096',
    'gray_light': '#a0aec0',
    'sky': '#e8f4f8',
    'ground_dark': '#654321',
    'text_muted': '#666666',
    'text_secondary': '#555555',
    'tx_coil': '#38a169',
    'rx_coil': '#3182ce',
    'ert_ring': '#ed8936',
    'probe_body': '#2c5282',
    'connector': '#4a5568',
}


def get_styles():
    """
    Get the complete style dictionary for HIRT PDF documents.

    Returns a reportlab StyleSheet with all custom paragraph styles.
    Styles include:
        - Title, Subtitle, Author, Meta (title block)
        - AbstractLabel, Abstract, Keywords (abstract section)
        - Section, Subsection (headings)
        - Body, BodyFirst (main text)
        - Caption, TableNote (figures/tables)
        - Reference (bibliography)
        - Equation (math)
        - BulletItem (lists)
        - TOCEntry, TOCSection (table of contents)
    """
    styles = getSampleStyleSheet()

    # Modify existing Title style
    styles['Title'].fontName = 'Times-Bold'
    styles['Title'].fontSize = 24
    styles['Title'].textColor = PRIMARY
    styles['Title'].alignment = TA_CENTER
    styles['Title'].spaceAfter = 16
    styles['Title'].leading = 28

    styles.add(ParagraphStyle(
        'Subtitle', fontName='Times-Italic', fontSize=14, textColor=HexColor('#4a5568'),
        alignment=TA_CENTER, spaceAfter=20, leading=16
    ))
    styles.add(ParagraphStyle(
        'Author', fontName='Times-Roman', fontSize=12, alignment=TA_CENTER, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        'Meta', fontName='Times-Roman', fontSize=10, textColor=HexColor('#666'),
        alignment=TA_CENTER, spaceAfter=24
    ))
    styles.add(ParagraphStyle(
        'AbstractLabel', fontName='Times-Bold', fontSize=11, textColor=PRIMARY,
        spaceBefore=0, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        'Abstract', fontName='Times-Roman', fontSize=11, alignment=TA_JUSTIFY,
        leftIndent=24, rightIndent=24, leading=14, spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        'Keywords', fontName='Times-Roman', fontSize=10, textColor=HexColor('#555'),
        leftIndent=24, rightIndent=24, spaceAfter=18
    ))
    styles.add(ParagraphStyle(
        'Section', fontName='Times-Bold', fontSize=16, textColor=PRIMARY,
        spaceBefore=18, spaceAfter=10, keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'Subsection', fontName='Times-Bold', fontSize=13, textColor=SECONDARY,
        spaceBefore=14, spaceAfter=6, keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'Subsubsection', fontName='Times-Bold', fontSize=11, textColor=SECONDARY,
        spaceBefore=10, spaceAfter=4, keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'Body', fontName='Times-Roman', fontSize=11, alignment=TA_JUSTIFY,
        spaceAfter=10, leading=14, firstLineIndent=18
    ))
    styles.add(ParagraphStyle(
        'BodyFirst', fontName='Times-Roman', fontSize=11, alignment=TA_JUSTIFY,
        spaceAfter=10, leading=14, firstLineIndent=0
    ))
    styles.add(ParagraphStyle(
        'BodyNoIndent', fontName='Times-Roman', fontSize=11, alignment=TA_JUSTIFY,
        spaceAfter=10, leading=14, firstLineIndent=0
    ))
    styles.add(ParagraphStyle(
        'Caption', fontName='Times-Roman', fontSize=10, alignment=TA_JUSTIFY,
        spaceBefore=6, spaceAfter=14, leading=12
    ))
    styles.add(ParagraphStyle(
        'Reference', fontName='Times-Roman', fontSize=10, alignment=TA_JUSTIFY,
        leftIndent=18, firstLineIndent=-18, spaceAfter=4, leading=12
    ))
    styles.add(ParagraphStyle(
        'Equation', fontName='Times-Italic', fontSize=11, alignment=TA_CENTER,
        spaceBefore=10, spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        'BulletItem', fontName='Times-Roman', fontSize=11, alignment=TA_JUSTIFY,
        leftIndent=24, firstLineIndent=-12, spaceAfter=4, leading=14, bulletIndent=6
    ))
    styles.add(ParagraphStyle(
        'NumberedItem', fontName='Times-Roman', fontSize=11, alignment=TA_JUSTIFY,
        leftIndent=30, firstLineIndent=-18, spaceAfter=4, leading=14
    ))
    styles.add(ParagraphStyle(
        'TableNote', fontName='Times-Italic', fontSize=9, textColor=HexColor('#666'),
        alignment=TA_LEFT, spaceBefore=4, spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        'TableHeader', fontName='Times-Bold', fontSize=10, textColor=HexColor('#ffffff'),
        alignment=TA_LEFT
    ))
    styles.add(ParagraphStyle(
        'TableCell', fontName='Times-Roman', fontSize=10, alignment=TA_LEFT,
        leading=12
    ))
    # Override Code style if it exists, otherwise add it
    if 'Code' in styles.byName:
        styles.byName['Code'].fontName = 'Courier'
        styles.byName['Code'].fontSize = 8
        styles.byName['Code'].alignment = TA_LEFT
        styles.byName['Code'].leftIndent = 18
        styles.byName['Code'].spaceAfter = 8
        styles.byName['Code'].leading = 10
        styles.byName['Code'].backColor = HexColor('#f7fafc')
    else:
        styles.add(ParagraphStyle(
            'Code', fontName='Courier', fontSize=8, alignment=TA_LEFT,
            leftIndent=18, spaceAfter=8, leading=10, backColor=HexColor('#f7fafc')
        ))
    styles.add(ParagraphStyle(
        'Warning', fontName='Times-Bold', fontSize=10, textColor=WARNING,
        spaceBefore=8, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        'Note', fontName='Times-Italic', fontSize=9, textColor=SECONDARY,
        leftIndent=18, rightIndent=18, spaceBefore=6, spaceAfter=6, leading=11
    ))

    # Table of Contents styles
    styles.add(ParagraphStyle(
        'TOCTitle', fontName='Times-Bold', fontSize=14, textColor=PRIMARY,
        alignment=TA_CENTER, spaceBefore=12, spaceAfter=18
    ))
    styles.add(ParagraphStyle(
        'TOCEntry', fontName='Times-Roman', fontSize=10, alignment=TA_LEFT,
        leftIndent=20, spaceAfter=4, leading=14
    ))
    styles.add(ParagraphStyle(
        'TOCSection', fontName='Times-Bold', fontSize=10, textColor=PRIMARY,
        alignment=TA_LEFT, spaceBefore=8, spaceAfter=2, leading=14
    ))

    return styles


def get_table_style(header_color=None):
    """
    Get standard table style for HIRT documents.

    Args:
        header_color: HexColor for header background (default: PRIMARY)

    Returns:
        TableStyle object
    """
    from reportlab.platypus import TableStyle
    from reportlab.lib import colors

    if header_color is None:
        header_color = PRIMARY

    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_BG),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ])


def get_warning_table_style():
    """Get table style for warning/safety boxes."""
    from reportlab.platypus import TableStyle

    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), WARNING),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#fff5f5')),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#742a2a')),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('BOX', (0, 0), (-1, -1), 2, WARNING),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ])


def get_info_table_style():
    """Get table style for info/note boxes."""
    from reportlab.platypus import TableStyle

    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ebf8ff')),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2c5282')),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('BOX', (0, 0), (-1, -1), 2, ACCENT),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ])
