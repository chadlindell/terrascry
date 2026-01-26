#!/usr/bin/env python3
"""
HIRT Section 01: Executive Summary - Publication-Quality PDF Generator

This module generates a publication-quality PDF for the HIRT Executive Summary,
providing an overview of the system, key capabilities, target audience, and
cost comparison with commercial alternatives.

Diagrams included:
1. System Overview - Dual-channel approach (MIT + ERT)
2. Cost Comparison Chart - HIRT vs commercial systems
3. Key Capabilities Infographic - Main advantages visualization
4. Depth of Investigation Chart - Configuration-based depth ranges
5. Two-Stage Workflow - Surface screening + HIRT follow-up

Usage:
    python section_01_executive_summary.py

Output:
    output/sections/01-executive-summary.pdf
"""

import os
import sys

# Add parent directory to path for lib imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, Wedge, FancyArrowPatch
import matplotlib.patheffects as path_effects
import numpy as np
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    KeepTogether, HRFlowable, PageBreak
)
from reportlab.lib import colors
from datetime import datetime

# === Page Setup ===
PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 0.85 * inch
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

# === Color Palette ===
PRIMARY = HexColor('#1a365d')
SECONDARY = HexColor('#2c5282')
ACCENT = HexColor('#3182ce')
SUCCESS = HexColor('#38a169')
WARNING = HexColor('#c53030')
LIGHT_BG = HexColor('#f7fafc')

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
    'tx_coil': '#38a169',
    'rx_coil': '#3182ce',
    'ert_ring': '#ed8936',
    'probe_body': '#2c5282',
    'connector': '#4a5568',
}


# ============================================================================
# STYLES
# ============================================================================
def get_styles():
    """Get the complete style dictionary for HIRT PDF documents."""
    styles = getSampleStyleSheet()

    styles['Title'].fontName = 'Times-Bold'
    styles['Title'].fontSize = 20
    styles['Title'].textColor = PRIMARY
    styles['Title'].alignment = TA_CENTER
    styles['Title'].spaceAfter = 8
    styles['Title'].leading = 24

    styles.add(ParagraphStyle(
        'Subtitle', fontName='Times-Italic', fontSize=11, textColor=HexColor('#4a5568'),
        alignment=TA_CENTER, spaceAfter=14, leading=14
    ))
    styles.add(ParagraphStyle(
        'Meta', fontName='Times-Roman', fontSize=9, textColor=HexColor('#666'),
        alignment=TA_CENTER, spaceAfter=18
    ))
    styles.add(ParagraphStyle(
        'Section', fontName='Times-Bold', fontSize=13, textColor=PRIMARY,
        spaceBefore=12, spaceAfter=8, keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'Subsection', fontName='Times-Bold', fontSize=11, textColor=SECONDARY,
        spaceBefore=12, spaceAfter=6, keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'Subsubsection', fontName='Times-Bold', fontSize=10, textColor=SECONDARY,
        spaceBefore=8, spaceAfter=4, keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'Body', fontName='Times-Roman', fontSize=10, alignment=TA_JUSTIFY,
        spaceAfter=8, leading=13, firstLineIndent=18
    ))
    styles.add(ParagraphStyle(
        'BodyFirst', fontName='Times-Roman', fontSize=10, alignment=TA_JUSTIFY,
        spaceAfter=8, leading=13, firstLineIndent=0
    ))
    styles.add(ParagraphStyle(
        'Caption', fontName='Times-Roman', fontSize=9, alignment=TA_JUSTIFY,
        spaceBefore=6, spaceAfter=14, leading=12
    ))
    styles.add(ParagraphStyle(
        'BulletItem', fontName='Times-Roman', fontSize=10, alignment=TA_JUSTIFY,
        leftIndent=25, firstLineIndent=-12, spaceAfter=4, leading=12, bulletIndent=15
    ))
    styles.add(ParagraphStyle(
        'NumberedItem', fontName='Times-Roman', fontSize=10, alignment=TA_JUSTIFY,
        leftIndent=30, firstLineIndent=-18, spaceAfter=4, leading=12
    ))
    styles.add(ParagraphStyle(
        'Note', fontName='Times-Italic', fontSize=9, textColor=SECONDARY,
        leftIndent=18, rightIndent=18, spaceBefore=6, spaceAfter=6, leading=11
    ))

    return styles


def get_table_style(header_color=None):
    """Get standard table style for HIRT documents."""
    if header_color is None:
        header_color = PRIMARY

    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), header_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
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


# ============================================================================
# PDF BUILDER CLASS
# ============================================================================
class SectionPDFBuilder:
    """
    Base class for building HIRT section PDFs.

    Provides consistent styling, figure handling, and document structure
    across all whitepaper sections.
    """

    def __init__(self, section_num, title, output_dir=None):
        """Initialize the PDF builder."""
        self.section_num = section_num
        self.title = title
        self.story = []
        self.styles = get_styles()
        self.figure_count = 0
        self.table_count = 0

        # Set output directory
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, 'output', 'sections')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Output filename
        title_slug = title.lower().replace(' ', '-').replace('&', 'and')
        title_slug = ''.join(c for c in title_slug if c.isalnum() or c == '-')
        self.output_path = os.path.join(
            self.output_dir,
            f"{section_num:02d}-{title_slug}.pdf"
        )

    def add_title_block(self, subtitle=None, version="2.0", show_meta=True):
        """Add the section title block."""
        full_title = f"Section {self.section_num}: {self.title}"
        self.story.append(Paragraph(full_title, self.styles['Title']))

        if subtitle:
            self.story.append(Paragraph(subtitle, self.styles['Subtitle']))

        if show_meta:
            date_str = datetime.now().strftime("%B %Y")
            self.story.append(Paragraph(
                f"<b>Version:</b> {version} | <b>Date:</b> {date_str} | "
                f"<b>HIRT Whitepaper</b>",
                self.styles['Meta']
            ))

        self.story.append(HRFlowable(
            width="100%", thickness=1, color=PRIMARY, spaceAfter=12
        ))

    def add_section_header(self, text, level=1):
        """Add a section or subsection header."""
        style_name = {
            1: 'Section',
            2: 'Subsection',
            3: 'Subsubsection'
        }.get(level, 'Subsection')

        self.story.append(Paragraph(text, self.styles[style_name]))

    def add_body_text(self, text, first_paragraph=False, no_indent=False):
        """Add body text paragraph."""
        if no_indent or first_paragraph:
            style = self.styles['BodyFirst']
        else:
            style = self.styles['Body']

        self.story.append(Paragraph(text, style))

    def add_bullet_list(self, items):
        """Add a bulleted list."""
        for item in items:
            self.story.append(Paragraph(
                f"\u2022  {item}",
                self.styles['BulletItem']
            ))

    def add_numbered_list(self, items):
        """Add a numbered list."""
        for i, item in enumerate(items, 1):
            self.story.append(Paragraph(
                f"{i}.  {item}",
                self.styles['NumberedItem']
            ))

    def add_figure(self, figure_buffer, caption, width=None, height=None,
                   scale=0.9, aspect_ratio=None):
        """
        Add a figure with caption.

        Args:
            figure_buffer: BytesIO buffer containing PNG image
            caption: Caption text (auto-numbered as "Figure N.")
            width: Image width (default: CONTENT_WIDTH * scale)
            height: Image height (default: calculated from aspect_ratio)
            scale: Scale factor for width (default: 0.9)
            aspect_ratio: Height/width ratio (e.g., 0.5 for half as tall as wide)
        """
        self.figure_count += 1

        # Calculate width
        max_width = CONTENT_WIDTH * scale
        if width is None:
            width = max_width
        else:
            width = min(width, max_width)

        # Calculate height from aspect ratio if not provided
        if height is None and aspect_ratio is not None:
            height = width * aspect_ratio

        if height:
            img = Image(figure_buffer, width=width, height=height)
        else:
            img = Image(figure_buffer, width=width)

        full_caption = f"<b>Figure {self.figure_count}.</b> {caption}"
        caption_para = Paragraph(full_caption, self.styles['Caption'])

        self.story.append(Spacer(1, 8))
        self.story.append(img)
        self.story.append(caption_para)

    def add_table(self, data, col_widths=None, caption=None, header_color=None):
        """Add a table with optional caption."""
        self.table_count += 1

        if col_widths is None:
            num_cols = len(data[0]) if data else 1
            col_widths = [CONTENT_WIDTH / num_cols] * num_cols

        table = Table(data, colWidths=col_widths)
        table.setStyle(get_table_style(header_color))

        elements = [Spacer(1, 8), table]

        if caption:
            full_caption = f"<b>Table {self.table_count}.</b> {caption}"
            elements.append(Paragraph(full_caption, self.styles['Caption']))

        self.story.append(KeepTogether(elements))

    def add_warning_box(self, title, items):
        """Add a warning/safety box with red styling."""
        data = [[title]]
        for item in items:
            data.append([f"\u2022  {item}"])

        table = Table(data, colWidths=[CONTENT_WIDTH])
        table.setStyle(get_warning_table_style())

        self.story.append(Spacer(1, 8))
        self.story.append(table)
        self.story.append(Spacer(1, 8))

    def add_info_box(self, title, items):
        """Add an info/note box with blue styling."""
        data = [[title]]
        for item in items:
            data.append([f"\u2022  {item}"])

        table = Table(data, colWidths=[CONTENT_WIDTH])
        table.setStyle(get_info_table_style())

        self.story.append(Spacer(1, 8))
        self.story.append(table)
        self.story.append(Spacer(1, 8))

    def add_spacer(self, height=12):
        """Add vertical space."""
        self.story.append(Spacer(1, height))

    def add_page_break(self):
        """Add a page break."""
        self.story.append(PageBreak())

    def add_horizontal_rule(self, thickness=1, color=None):
        """Add a horizontal line."""
        if color is None:
            color = PRIMARY
        self.story.append(HRFlowable(
            width="100%", thickness=thickness, color=color, spaceAfter=12
        ))

    def build(self):
        """Build and save the PDF document."""
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN,
            bottomMargin=MARGIN
        )

        doc.build(self.story)
        print(f"PDF created: {self.output_path}")
        return self.output_path


# ============================================================================
# FIGURE 1: SYSTEM OVERVIEW - DUAL-CHANNEL APPROACH
# ============================================================================
def create_system_overview():
    """
    Simplified system overview showing the dual-channel MIT + ERT approach.
    Shows probes in ground with both sensing methods highlighted.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # === Title Banner ===
    ax.add_patch(FancyBboxPatch((0.2, 5.3), 9.6, 0.6, boxstyle="round,pad=0.02",
                                facecolor=COLOR_PALETTE['primary'], edgecolor='none'))
    ax.text(5, 5.6, 'HIRT: Dual-Channel Subsurface Imaging System',
            fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    # === Ground Cross-Section (Left Panel) ===
    # Sky
    ax.fill_between([0.3, 4.7], [4.2, 4.2], [5.0, 5.0], color=COLOR_PALETTE['sky'], alpha=0.5)

    # Ground layers
    for d1, d2, alpha in [(4.2, 3.4, 0.3), (3.4, 2.6, 0.4), (2.6, 1.8, 0.5), (1.8, 0.5, 0.6)]:
        ax.fill_between([0.3, 4.7], [d1, d1], [d2, d2],
                       color=COLOR_PALETTE['ground_tan'], alpha=alpha)

    # Ground surface line
    ax.plot([0.3, 4.7], [4.2, 4.2], color=COLOR_PALETTE['ground_dark'], linewidth=2.5)
    ax.text(2.5, 4.35, 'Ground Surface', fontsize=8, ha='center', color=COLOR_PALETTE['ground_dark'])

    # Probes (3 probes)
    probe_x = [1.0, 2.5, 4.0]
    for px in probe_x:
        # Probe rod
        ax.add_patch(Rectangle((px-0.08, 1.0), 0.16, 3.2,
                               color=COLOR_PALETTE['probe_body'], ec='black', lw=1))
        # Junction box at top
        ax.add_patch(Rectangle((px-0.18, 4.1), 0.36, 0.25,
                               color=COLOR_PALETTE['connector'], ec='black', lw=1))

        # MIT coils (TX/RX)
        for depth, color in [(3.5, COLOR_PALETTE['tx_coil']), (2.5, COLOR_PALETTE['rx_coil'])]:
            ax.add_patch(Circle((px, depth), 0.15, color=color, ec='black', lw=1))

        # ERT rings
        for depth in [3.0, 2.0, 1.2]:
            ax.add_patch(Rectangle((px-0.15, depth-0.06), 0.3, 0.12,
                                   color=COLOR_PALETTE['ert_ring'], ec='black', lw=0.5))

    # Ray paths between probes (MIT)
    for i, p1 in enumerate(probe_x[:-1]):
        for p2 in probe_x[i+1:]:
            for d1, d2 in [(3.5, 3.5), (3.5, 2.5), (2.5, 2.5), (2.5, 3.5)]:
                ax.plot([p1, p2], [d1, d2], color=COLOR_PALETTE['success'],
                       alpha=0.4, linewidth=1.2, linestyle='-')

    # ERT current paths (dashed)
    ax.plot([1.0, 4.0], [3.0, 3.0], color=COLOR_PALETTE['orange'],
           alpha=0.6, linewidth=1.5, linestyle='--')
    ax.plot([1.0, 4.0], [2.0, 2.0], color=COLOR_PALETTE['orange'],
           alpha=0.6, linewidth=1.5, linestyle='--')

    # Target
    target = Circle((2.5, 2.2), 0.35, color=COLOR_PALETTE['warning'], ec='#9b2c2c', lw=2)
    ax.add_patch(target)
    ax.text(2.5, 2.2, 'Target', fontsize=7, ha='center', va='center',
            color='white', fontweight='bold')

    # Depth scale
    for d, label in [(4.2, '0m'), (3.2, '1m'), (2.2, '2m'), (1.2, '3m')]:
        ax.text(0.15, d, label, fontsize=7, va='center', color='gray')

    # Legend for left panel
    legend_y = 0.25
    ax.add_patch(Circle((0.6, legend_y+0.15), 0.1, color=COLOR_PALETTE['tx_coil']))
    ax.text(0.8, legend_y+0.15, 'TX Coil', fontsize=7, va='center')
    ax.add_patch(Circle((1.8, legend_y+0.15), 0.1, color=COLOR_PALETTE['rx_coil']))
    ax.text(2.0, legend_y+0.15, 'RX Coil', fontsize=7, va='center')
    ax.add_patch(Rectangle((2.85, legend_y+0.05), 0.2, 0.12, color=COLOR_PALETTE['ert_ring']))
    ax.text(3.15, legend_y+0.15, 'ERT Ring', fontsize=7, va='center')

    # === MIT-3D Channel Box (Top Right) ===
    ax.add_patch(FancyBboxPatch((5.2, 3.5), 2.2, 1.6, boxstyle="round,pad=0.05",
                                facecolor='#c6f6d5', edgecolor=COLOR_PALETTE['success'], linewidth=2))
    ax.text(6.3, 4.9, 'MIT-3D Channel', fontsize=10, fontweight='bold',
            ha='center', color=COLOR_PALETTE['success'])

    mit_items = [
        'Low-frequency TX/RX coils',
        '2-50 kHz operation',
        'Eddy current detection',
        'Maps metal (incl. aluminum)',
    ]
    for i, item in enumerate(mit_items):
        ax.text(5.35, 4.55 - i*0.25, f'\u2022 {item}', fontsize=7, va='center')

    # === ERT-Lite Channel Box (Bottom Right) ===
    ax.add_patch(FancyBboxPatch((5.2, 1.6), 2.2, 1.6, boxstyle="round,pad=0.05",
                                facecolor='#feebc8', edgecolor=COLOR_PALETTE['orange'], linewidth=2))
    ax.text(6.3, 3.0, 'ERT-Lite Channel', fontsize=10, fontweight='bold',
            ha='center', color='#c05621')

    ert_items = [
        'Ring electrodes on rods',
        '0.5-2 mA injection',
        'Voltage measurement',
        'Maps soil disturbance',
    ]
    for i, item in enumerate(ert_items):
        ax.text(5.35, 2.65 - i*0.25, f'\u2022 {item}', fontsize=7, va='center')

    # === Key Benefits Box (Far Right) ===
    ax.add_patch(FancyBboxPatch((7.6, 1.6), 2.2, 3.5, boxstyle="round,pad=0.05",
                                facecolor=COLOR_PALETTE['light_bg'], edgecolor=COLOR_PALETTE['primary'], linewidth=2))
    ax.text(8.7, 4.9, 'Key Benefits', fontsize=10, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    benefits = [
        'True 3D tomographic',
        'imaging capability',
        '',
        '2-5x better resolution',
        'than surface methods',
        '',
        'Non-ferrous metal',
        'detection (aluminum)',
        '',
        '~10x less ground',
        'disturbance',
    ]
    for i, item in enumerate(benefits):
        ax.text(7.75, 4.55 - i*0.24, item, fontsize=7, va='center')

    # Connecting arrows
    ax.annotate('', xy=(5.1, 4.3), xytext=(4.8, 3.5),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['success'], lw=1.5))
    ax.annotate('', xy=(5.1, 2.4), xytext=(4.8, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['orange'], lw=1.5))

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 2: COST COMPARISON CHART
# ============================================================================
def create_cost_comparison():
    """
    Professional cost comparison chart showing HIRT vs commercial systems.
    Includes bar chart with dramatic cost difference visualization.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                   gridspec_kw={'width_ratios': [1.2, 1]})

    # === Left Panel: Cost Bar Chart ===
    systems = ['Commercial\nCrosshole\n(Low End)', 'Commercial\nCrosshole\n(High End)',
               'HIRT\nComplete Kit\n(High)', 'HIRT\nComplete Kit\n(Low)']
    costs = [50000, 200000, 3900, 1800]
    colors = ['#e53e3e', '#9b2c2c', '#38a169', '#276749']

    # Create horizontal bar chart
    y_pos = np.arange(len(systems))
    bars = ax1.barh(y_pos, costs, color=colors, edgecolor='black', linewidth=1)

    # Add value labels
    for bar, cost in zip(bars, costs):
        width = bar.get_width()
        if width > 10000:
            ax1.text(width - 2000, bar.get_y() + bar.get_height()/2,
                    f'${cost:,}', ha='right', va='center', fontsize=9,
                    fontweight='bold', color='white')
        else:
            ax1.text(width + 2000, bar.get_y() + bar.get_height()/2,
                    f'${cost:,}', ha='left', va='center', fontsize=9, fontweight='bold')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(systems, fontsize=9)
    ax1.set_xlabel('System Cost (USD)', fontsize=10)
    ax1.set_xlim(0, 230000)
    ax1.set_title('Complete System Cost Comparison', fontsize=11, fontweight='bold',
                  color=COLOR_PALETTE['primary'], pad=10)

    # Add savings annotation
    ax1.annotate('95%+ Savings', xy=(50000, 2.5), xytext=(120000, 2.5),
                fontsize=10, fontweight='bold', color=COLOR_PALETTE['success'],
                arrowprops=dict(arrowstyle='<->', color=COLOR_PALETTE['success'], lw=2),
                va='center')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.grid(True, linestyle=':', alpha=0.5)
    ax1.set_axisbelow(True)

    # === Right Panel: Cost Breakdown ===
    categories = ['Probes\n(20-24)', 'Electronics\nHub', 'Cables &\nConnectors', 'Tools &\nMisc']
    low_costs = [1400, 300, 50, 50]  # $70 x 20 probes
    high_costs = [3600, 200, 50, 50]  # $150 x 24 probes

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax2.bar(x - width/2, low_costs, width, label='Low-End Kit ($1,800)',
                    color=COLOR_PALETTE['success'], edgecolor='black', linewidth=1)
    bars2 = ax2.bar(x + width/2, high_costs, width, label='High-End Kit ($3,900)',
                    color='#276749', edgecolor='black', linewidth=1)

    ax2.set_ylabel('Cost (USD)', fontsize=10)
    ax2.set_title('HIRT Kit Cost Breakdown', fontsize=11, fontweight='bold',
                  color=COLOR_PALETTE['primary'], pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=8)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_ylim(0, 4500)

    # Add bar labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'${int(height)}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'${int(height)}', ha='center', va='bottom', fontsize=7)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax2.set_axisbelow(True)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 3: KEY CAPABILITIES INFOGRAPHIC
# ============================================================================
def create_capabilities_infographic():
    """
    Infographic highlighting key HIRT capabilities with icons and metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # === Title ===
    ax.text(5, 4.7, 'HIRT Key Capabilities', fontsize=14, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    # === Capability Cards ===
    card_data = [
        {
            'x': 0.3, 'y': 2.8, 'width': 2.2, 'height': 1.6,
            'title': 'TRUE TOMOGRAPHY',
            'metric': '3D',
            'desc': 'Full volumetric\nimaging vs 2D maps',
            'color': COLOR_PALETTE['primary'],
            'icon': 'cube'
        },
        {
            'x': 2.7, 'y': 2.8, 'width': 2.2, 'height': 1.6,
            'title': 'RESOLUTION',
            'metric': '2-5x',
            'desc': 'Better than surface\nmethods at depth',
            'color': COLOR_PALETTE['success'],
            'icon': 'target'
        },
        {
            'x': 5.1, 'y': 2.8, 'width': 2.2, 'height': 1.6,
            'title': 'DEPTH RANGE',
            'metric': '3-6m',
            'desc': 'Deep target\ndetection',
            'color': COLOR_PALETTE['secondary'],
            'icon': 'depth'
        },
        {
            'x': 7.5, 'y': 2.8, 'width': 2.2, 'height': 1.6,
            'title': 'COST SAVINGS',
            'metric': '95%+',
            'desc': 'vs commercial\nsystems',
            'color': COLOR_PALETTE['warning'],
            'icon': 'savings'
        },
    ]

    for card in card_data:
        # Card background
        ax.add_patch(FancyBboxPatch((card['x'], card['y']), card['width'], card['height'],
                                    boxstyle="round,pad=0.03", facecolor='white',
                                    edgecolor=card['color'], linewidth=2))

        # Title
        ax.text(card['x'] + card['width']/2, card['y'] + card['height'] - 0.2,
               card['title'], fontsize=8, fontweight='bold', ha='center',
               color=card['color'])

        # Large metric
        ax.text(card['x'] + card['width']/2, card['y'] + card['height']/2 + 0.1,
               card['metric'], fontsize=22, fontweight='bold', ha='center',
               color=card['color'])

        # Description
        ax.text(card['x'] + card['width']/2, card['y'] + 0.25,
               card['desc'], fontsize=7, ha='center', va='bottom',
               color=COLOR_PALETTE['gray_dark'])

    # === Advantages Row ===
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 2.2, boxstyle="round,pad=0.03",
                                facecolor=COLOR_PALETTE['light_bg'],
                                edgecolor=COLOR_PALETTE['gray_med'], linewidth=1))

    ax.text(5, 2.3, 'Crosshole Geometry Advantages', fontsize=11, fontweight='bold',
            ha='center', color=COLOR_PALETTE['primary'])

    advantages = [
        ('True tomographic coverage\nthrough target volume', 0.6),
        ('Superior depth discrimination\n(3m vs 4m distinguishable)', 2.4),
        ('Non-ferrous metal detection\n(aluminum, brass, copper)', 4.2),
        ('3D localization rather\nthan pseudo-depth maps', 6.0),
        ('Minimal site disturbance\n(16mm probe diameter)', 7.8),
    ]

    for text, x in advantages:
        # Checkmark circle
        ax.add_patch(Circle((x + 0.15, 1.6), 0.15, color=COLOR_PALETTE['success']))
        ax.text(x + 0.15, 1.6, '\u2713', fontsize=10, ha='center', va='center',
               color='white', fontweight='bold')
        # Text
        ax.text(x + 0.4, 1.5, text, fontsize=8, va='center',
               color=COLOR_PALETTE['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 4: DEPTH OF INVESTIGATION CHART
# ============================================================================
def create_depth_investigation():
    """
    Visual representation of depth investigation ranges for different configurations.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    # Data from executive summary
    configs = [
        ('1.5m probes, 2m spacing', 2, 3, 'HIGH', COLOR_PALETTE['success']),
        ('3.0m probes, 2m spacing', 3, 5, 'MEDIUM', COLOR_PALETTE['secondary']),
        ('Edge cases (conductive)', 4, 6, 'LOW', COLOR_PALETTE['orange']),
    ]

    y_pos = np.arange(len(configs))

    # Draw depth bars
    for i, (label, min_d, max_d, conf, color) in enumerate(configs):
        # Range bar
        ax.barh(i, max_d - min_d, left=min_d, height=0.5,
               color=color, alpha=0.6, edgecolor=color, linewidth=2)

        # Min/max markers
        ax.plot([min_d], [i], '|', color=color, markersize=20, markeredgewidth=3)
        ax.plot([max_d], [i], '|', color=color, markersize=20, markeredgewidth=3)

        # Confidence label
        ax.text(max_d + 0.2, i, f'{conf}\nconfidence', fontsize=8, va='center',
               color=color, fontweight='bold')

        # Depth range text
        ax.text((min_d + max_d)/2, i, f'{min_d}-{max_d}m', fontsize=10,
               ha='center', va='center', fontweight='bold', color='white')

    # Surface method comparison line
    ax.axvline(1.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(1.5, 2.7, 'Typical surface\nmethod limit', fontsize=7,
           ha='center', va='bottom', color='gray')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c[0] for c in configs], fontsize=9)
    ax.set_xlabel('Depth of Investigation (m)', fontsize=10)
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.5, 3)
    ax.set_title('HIRT Depth of Investigation by Configuration', fontsize=11,
                fontweight='bold', color=COLOR_PALETTE['primary'], pad=10)

    # Add grid
    ax.xaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 5: TWO-STAGE WORKFLOW
# ============================================================================
def create_workflow_diagram():
    """
    Diagram showing the optimal two-stage workflow: surface screening + HIRT follow-up.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # === Stage 1: Surface Screening ===
    ax.add_patch(FancyBboxPatch((0.2, 1.5), 3.5, 2.2, boxstyle="round,pad=0.05",
                                facecolor='#ebf8ff', edgecolor=COLOR_PALETTE['secondary'], linewidth=2))
    ax.text(2, 3.5, 'STAGE 1: Surface Screening', fontsize=10, fontweight='bold',
            ha='center', color=COLOR_PALETTE['secondary'])

    # Methods
    methods = ['Magnetometry', 'GPR', 'EM31/CMD']
    for i, method in enumerate(methods):
        ax.add_patch(Rectangle((0.4, 2.8 - i*0.5), 1.5, 0.35,
                               color=COLOR_PALETTE['gray_light'], ec='gray', lw=1))
        ax.text(1.15, 2.97 - i*0.5, method, fontsize=8, ha='center', va='center')

    # Benefits
    ax.text(2.5, 2.95, '\u2713 Rapid coverage', fontsize=7, color=COLOR_PALETTE['success'])
    ax.text(2.5, 2.55, '\u2713 Large areas', fontsize=7, color=COLOR_PALETTE['success'])
    ax.text(2.5, 2.15, '\u2713 Initial screening', fontsize=7, color=COLOR_PALETTE['success'])

    # Limitations
    ax.text(0.4, 1.7, 'Limited depth | Ambiguous at >1.5m | 2D pseudo-depth only',
           fontsize=6, color=COLOR_PALETTE['gray_dark'], style='italic')

    # === Arrow ===
    ax.annotate('', xy=(4.5, 2.6), xytext=(3.8, 2.6),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['primary'], lw=3))
    ax.text(4.15, 2.9, 'Anomalies\nidentified', fontsize=7, ha='center',
           color=COLOR_PALETTE['primary'])

    # === Stage 2: HIRT Follow-up ===
    ax.add_patch(FancyBboxPatch((4.6, 1.5), 3.5, 2.2, boxstyle="round,pad=0.05",
                                facecolor='#c6f6d5', edgecolor=COLOR_PALETTE['success'], linewidth=2))
    ax.text(6.35, 3.5, 'STAGE 2: HIRT Crosshole', fontsize=10, fontweight='bold',
            ha='center', color=COLOR_PALETTE['success'])

    # Channels
    ax.add_patch(Rectangle((4.8, 2.55), 1.3, 0.35,
                           color='#38a169', ec='#276749', lw=1))
    ax.text(5.45, 2.72, 'MIT-3D', fontsize=8, ha='center', va='center', color='white')
    ax.add_patch(Rectangle((4.8, 2.1), 1.3, 0.35,
                           color=COLOR_PALETTE['orange'], ec='#c05621', lw=1))
    ax.text(5.45, 2.27, 'ERT-Lite', fontsize=8, ha='center', va='center', color='white')

    # Benefits
    ax.text(6.3, 2.95, '\u2713 True 3D imaging', fontsize=7, color=COLOR_PALETTE['success'])
    ax.text(6.3, 2.55, '\u2713 3-6m depth', fontsize=7, color=COLOR_PALETTE['success'])
    ax.text(6.3, 2.15, '\u2713 Target characterization', fontsize=7, color=COLOR_PALETTE['success'])

    ax.text(4.8, 1.7, 'High-resolution | Aluminum detection | Soil disturbance',
           fontsize=6, color=COLOR_PALETTE['gray_dark'], style='italic')

    # === Result Box ===
    ax.annotate('', xy=(8.8, 2.6), xytext=(8.2, 2.6),
                arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['primary'], lw=3))

    ax.add_patch(FancyBboxPatch((8.3, 1.5), 1.5, 2.2, boxstyle="round,pad=0.05",
                                facecolor=COLOR_PALETTE['primary'],
                                edgecolor=COLOR_PALETTE['primary'], linewidth=2))
    ax.text(9.05, 3.5, 'RESULT', fontsize=10, fontweight='bold', ha='center', color='white')
    ax.text(9.05, 2.7, '3D Model', fontsize=9, ha='center', color='white', fontweight='bold')
    ax.text(9.05, 2.3, 'Target', fontsize=8, ha='center', color='white')
    ax.text(9.05, 2.0, 'Location', fontsize=8, ha='center', color='white')
    ax.text(9.05, 1.7, '& Character', fontsize=8, ha='center', color='white')

    # === Bottom note ===
    ax.text(5, 0.9, 'Optimal Workflow: Leverage surface speed for screening, HIRT precision for characterization',
           fontsize=9, ha='center', color=COLOR_PALETTE['gray_dark'], style='italic')
    ax.text(5, 0.5, 'This two-stage approach minimizes deployment time while maximizing detection accuracy',
           fontsize=8, ha='center', color=COLOR_PALETTE['gray_med'])

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
def build_section():
    """Build the complete Section 01: Executive Summary PDF."""

    print("Generating figures...")
    fig_overview = create_system_overview()
    fig_cost = create_cost_comparison()
    fig_capabilities = create_capabilities_infographic()
    fig_depth = create_depth_investigation()
    fig_workflow = create_workflow_diagram()

    print("Building PDF...")

    # Initialize the section builder
    builder = SectionPDFBuilder(
        section_num=1,
        title="Executive Summary"
    )

    # === Title Block ===
    builder.add_title_block(
        subtitle="HIRT Overview, Capabilities, Target Audience, and Cost Comparison"
    )

    # === What is HIRT? ===
    builder.add_section_header("1.1 What is HIRT?")
    builder.add_body_text(
        "HIRT (Hybrid Inductive-Resistivity Tomography) is a <b>dual-channel"
        "subsurface imaging system</b> designed for archaeological and forensic investigations. "
        "By placing sensors inside the ground and measuring through the volume using "
        "<b>crosshole geometry</b>, HIRT obtains true tomographic coverage for 3D "
        "reconstruction of subsurface features.",
        first_paragraph=True
    )

    # System overview figure (figsize 9x5 -> aspect ratio 5/9)
    builder.add_figure(
        fig_overview,
        "HIRT dual-channel system overview. The system combines MIT-3D (Magneto-Inductive "
        "Tomography) for metal detection with ERT-Lite (Electrical Resistivity) for soil "
        "disturbance mapping. Probes are inserted into pre-drilled 18-20mm holes, with "
        "ray paths traversing the target volume for true 3D tomographic imaging.",
        aspect_ratio=5/9
    )

    # === Why the Name HIRT ===
    builder.add_section_header("1.2 Why the Name")
    builder.add_body_text(
        "HIRT was created in a landscape shaped by war and unfinished consequences.",
        first_paragraph=True
    )
    builder.add_body_text(
        "The Silesian basin of modern-day Poland was the target of one of the most intensive "
        "Allied bombing campaigns of World War II, centered on the Blechhammer industrial complex. "
        "Tens of thousands of bombs fell here. Many exploded. Thousands did not. Today, an estimated "
        "4,000\u20136,000 unexploded bombs remain buried beneath forests, wetlands, and towns, still "
        "capable of detonation decades later. These UXOs continue to shape land use, infrastructure "
        "planning, and public safety across the region."
    )
    builder.add_body_text(
        "This same ground holds other histories. POW camps and forced labor sites. Jewish victims "
        "of the industrial death system tied to Blechhammer. Aircrews lost in the bombing campaigns. "
        "Civilians\u2014families and children\u2014killed not as targets, but by proximity. After the war "
        "came displacement, silence, and redevelopment layered over danger that was never fully removed."
    )
    builder.add_body_text(
        "My great-uncle, 1st Lt. Arthur Lindell, was shot down here. He and his crew are still missing, "
        "believed to lie in a filled bomb crater at the crash site. That search is personal\u2014but it "
        "exists within a much larger, unresolved problem shared by communities across Europe and beyond."
    )
    builder.add_body_text(
        "HIRT takes its name from this history of hurt and pain, and from a commitment to address "
        "what remains beneath the surface. It was designed to see without excavation, to reduce risk "
        "where danger still exists, and to help bring clarity\u2014whether the goal is safety, "
        "remembrance, or recovery."
    )

    builder.add_spacer(8)

    # === Dual-Channel Approach ===
    builder.add_section_header("1.3 Dual-Channel Approach", level=2)
    builder.add_body_text(
        "HIRT employs two complementary sensing modalities that together provide "
        "comprehensive subsurface characterization:",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "<b>MIT-3D (Magneto-Inductive Tomography):</b> Low-frequency TX/RX coils (2-50 kHz) "
        "measure amplitude and phase changes caused by eddy currents. Maps conductive metal "
        "masses including aluminum, which magnetometry cannot detect.",
        "<b>ERT-Lite (Electrical Resistivity):</b> Small ring electrodes inject tiny currents "
        "(0.5-2 mA) and measure voltage distribution. Maps soil resistivity variations from "
        "moisture, disturbance patterns, voids, and grave shafts."
    ])

    builder.add_info_box("Design Philosophy", [
        "Make each probe dual-role (TX & RX for MIT) plus ERT pickup",
        "Identical probes simplify logistics and improve data quality",
        "Modular, interchangeable components for field serviceability",
        "Minimal site disturbance (~10x less than traditional methods)"
    ])

    # === Primary Use Cases ===
    builder.add_section_header("1.4 Primary Use Cases")
    builder.add_body_text(
        "HIRT is optimized for three primary application scenarios, each with specific "
        "configuration recommendations:",
        first_paragraph=True
    )

    # Use cases table
    use_case_data = [
        ['Use Case', 'Typical Parameters', 'Key Method', 'Target Types'],
        ['Filled Bomb Crater', '10-15m dia, 0-4m depth\n3m probes, 1.5m spacing',
         'MIT + ERT combined', 'Metal parts + remains'],
        ['Woods Burials', '0.6-1.5m depth\n1.5m probes, 1m spacing',
         'ERT patterns + MIT', 'Grave shafts, artifacts'],
        ['Swamp/Wetland', '>5m targets, margins\nLow-freq MIT',
         'MIT from margins', 'Deep targets'],
    ]
    col_widths = [CONTENT_WIDTH * 0.2, CONTENT_WIDTH * 0.3,
                  CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.25]
    builder.add_table(use_case_data, col_widths=col_widths, caption="Primary HIRT use cases and configurations")

    # === System Capabilities ===
    builder.add_section_header("1.5 System Capabilities")

    # Capabilities infographic (figsize 10x4.5 -> aspect ratio 4.5/10)
    builder.add_figure(
        fig_capabilities,
        "Key HIRT capabilities infographic. The crosshole geometry provides true 3D "
        "tomographic imaging with 2-5x better resolution than surface methods at depths "
        "exceeding 2 meters. Cost savings of 95%+ make professional-grade subsurface "
        "imaging accessible to research institutions and humanitarian organizations.",
        aspect_ratio=4.5/10
    )

    builder.add_body_text(
        "Surface GPR and magnetometry are excellent screening tools but can yield ambiguous "
        "results in complex conditions. HIRT's crosshole geometry provides superior "
        "performance through direct measurement paths:",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "<b>True tomographic coverage</b> through the target volume (not surface extrapolation)",
        "<b>2-5x better resolution</b> than surface methods at depths exceeding 2 meters",
        "<b>Superior depth discrimination</b> - targets at 3m vs 4m are clearly distinguishable",
        "<b>Non-ferrous detection</b> - aluminum aircraft parts that magnetometry cannot sense",
        "<b>3D localization</b> with measured positions rather than estimated pseudo-depth"
    ])

    # Depth investigation figure (figsize 9x4 -> aspect ratio 4/9)
    builder.add_figure(
        fig_depth,
        "Depth of investigation ranges by HIRT configuration. The typical surface method "
        "limit (~1.5m) is shown for comparison. Confidence levels indicate the reliability "
        "of detection at different depths and soil conditions.",
        aspect_ratio=4/9
    )

    # === Limitations ===
    builder.add_section_header("1.6 Limitations and Non-Goals")

    builder.add_warning_box("Non-Goals", [
        "Producing final 3D visuals (software pipeline is a separate development phase)",
        "Replacing standard ethical/excavation practices or permits",
        "Replacing professional archaeological/forensic protocols"
    ])

    builder.add_body_text(
        "<b>When Surface Methods Remain Superior:</b> Rapid large-area screening (10x faster), "
        "shallow targets (<1m) where GPR resolution excels, purely ferrous targets "
        "(magnetometry), and initial site characterization before targeted investigation.",
        first_paragraph=True
    )

    builder.add_body_text(
        "<b>Technical Limitations:</b> Smaller coil area results in ~19 dB SNR loss compared "
        "to commercial systems (compensated by longer integration times). Survey time increases "
        "5-10x compared to commercial systems. Requires post-processing software for 3D "
        "reconstruction. Limited depth performance in highly conductive soils."
    )

    # === Optimal Workflow ===
    builder.add_section_header("1.7 Optimal Workflow")

    # Workflow diagram (figsize 10x3.5 -> aspect ratio 3.5/10)
    builder.add_figure(
        fig_workflow,
        "Two-stage optimal workflow combining surface screening methods with HIRT crosshole "
        "follow-up. This approach leverages the speed of surface methods for initial "
        "anomaly identification, then applies HIRT's superior resolution for detailed "
        "characterization of identified targets.",
        aspect_ratio=3.5/10
    )

    builder.add_body_text(
        "The physics supports a <b>two-stage approach</b> that leverages the strengths of "
        "both surface and crosshole methods:",
        first_paragraph=True
    )

    builder.add_numbered_list([
        "<b>Surface screening</b> (magnetometry, GPR, EM31): Identify anomalies quickly "
        "over large areas with rapid coverage rates.",
        "<b>HIRT crosshole follow-up</b>: Characterize identified anomalies with superior "
        "3D resolution and depth discrimination."
    ])

    # === Target Audience ===
    builder.add_section_header("1.8 Target Audience")
    builder.add_body_text(
        "This documentation and the HIRT system are designed for:",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "<b>Archaeologists</b> investigating WWII sites, burial locations, or crash sites",
        "<b>Forensic investigators</b> requiring non-destructive subsurface imaging",
        "<b>Geophysicists</b> interested in low-cost tomographic methods",
        "<b>DIY builders</b> seeking to construct field-deployable systems",
        "<b>Researchers</b> exploring crosshole electromagnetic/resistivity techniques"
    ])

    # === Cost Overview ===
    builder.add_section_header("1.9 Cost Overview")

    # Cost comparison (figsize 10x4 -> aspect ratio 4/10)
    builder.add_figure(
        fig_cost,
        "Cost comparison between HIRT and commercial crosshole systems. A complete HIRT "
        "starter kit costs $1,800-3,900 depending on component selection, representing "
        "95%+ savings compared to commercial systems ($50,000-200,000+). The primary cost "
        "driver is the number of probes (20-24 for standard deployment at $70-150 each).",
        aspect_ratio=4/10
    )

    # Cost summary table
    cost_data = [
        ['Component', 'Low-End', 'High-End', 'Notes'],
        ['Probes (20-24)', '$1,400', '$3,600', '$70-150 per probe'],
        ['Electronics Hub', '$300', '$200', 'DIY vs pre-built'],
        ['Cables & Connectors', '$50', '$50', 'Standard components'],
        ['Tools & Misc', '$50', '$50', 'Assembly tools'],
        ['Total', '$1,800', '$3,900', '95%+ vs commercial'],
    ]
    builder.add_table(cost_data, caption="HIRT system cost breakdown")

    builder.add_body_text(
        "This cost structure places professional-grade subsurface imaging capabilities "
        "within reach of university research groups, non-profit humanitarian organizations, "
        "and archaeological teams that could not otherwise afford crosshole tomography.",
        first_paragraph=True
    )

    # Build the PDF
    output_path = builder.build()
    print(f"\nSection 01 PDF generated successfully!")
    print(f"Output: {output_path}")

    return output_path


if __name__ == "__main__":
    build_section()
