"""
HIRT PDF Generator - PDF Builder Module

Base class for building section PDFs with consistent styling and layout.
"""

import os
import re
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    KeepTogether, HRFlowable, PageBreak, ListFlowable, ListItem, CondPageBreak
)
from reportlab.lib import colors
from reportlab.pdfgen import canvas  # Added for header/footer

from .styles import (
    get_styles, get_table_style, get_warning_table_style, get_info_table_style,
    PAGE_WIDTH, PAGE_HEIGHT, MARGIN, CONTENT_WIDTH,
    PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING, LIGHT_BG, COLOR_PALETTE
)


class SectionPDFBuilder:
    """
    Base class for building HIRT section PDFs.

    Provides consistent styling, figure handling, and document structure
    across all whitepaper sections.

    Usage:
        builder = SectionPDFBuilder(
            section_num=3,
            title="System Architecture",
            output_dir="/path/to/output"
        )
        builder.add_section_header("3.1 Overview")
        builder.add_body_text("The HIRT system...")
        builder.add_figure(my_figure_buffer, "Figure 1. System diagram")
        builder.build()
    """

    def __init__(self, section_num, title, output_dir=None):
        """
        Initialize the PDF builder.

        Args:
            section_num: Section number (0-19)
            title: Section title string
            output_dir: Output directory path (default: output/sections/)
        """
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

    def _wrap_cell_content(self, content, is_header=False):
        """Wrap table cell content in Paragraph if it contains HTML tags."""
        if isinstance(content, str) and re.search(r'<[^>]+>', content):
            style_name = 'TableHeader' if is_header else 'TableCell'
            return Paragraph(content, self.styles[style_name])
        return content

    def add_conditional_page_break(self, min_space_needed=2*inch):
        """Add page break only if remaining space is less than min_space_needed."""
        self.story.append(CondPageBreak(min_space_needed))

    def add_grouped_section(self, header_text, body_paragraphs, level=1):
        """Add section header + content as single KeepTogether unit."""
        style_name = {1: 'Section', 2: 'Subsection', 3: 'Subsubsection'}.get(level, 'Subsection')
        elements = [Paragraph(header_text, self.styles[style_name])]
        if isinstance(body_paragraphs, str):
            body_paragraphs = [body_paragraphs]
        for i, text in enumerate(body_paragraphs):
            style = self.styles['BodyFirst'] if i == 0 else self.styles['Body']
            elements.append(Paragraph(text, style))
        self.story.append(KeepTogether(elements))

    def add_title_block(self, subtitle=None, version="2.0", show_meta=True):
        """
        Add the section title block.

        Args:
            subtitle: Optional subtitle text
            version: Version string (default: "2.0")
            show_meta: Whether to show version/date metadata
        """
        # Main title
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

        # Horizontal rule
        self.story.append(HRFlowable(
            width="100%", thickness=1, color=PRIMARY, spaceAfter=12
        ))

    def add_section_header(self, text, level=1):
        """
        Add a section or subsection header.

        Args:
            text: Header text (can include number, e.g., "3.1 Overview")
            level: Header level (1=Section, 2=Subsection, 3=Subsubsection)
        """
        style_name = {
            1: 'Section',
            2: 'Subsection',
            3: 'Subsubsection'
        }.get(level, 'Subsection')

        self.story.append(Paragraph(text, self.styles[style_name]))

    def add_body_text(self, text, first_paragraph=False, no_indent=False):
        """
        Add body text paragraph.

        Args:
            text: Paragraph text (can include basic HTML: <b>, <i>, etc.)
            first_paragraph: If True, no first-line indent
            no_indent: If True, no indent regardless of position
        """
        if no_indent or first_paragraph:
            style = self.styles['BodyFirst']
        else:
            style = self.styles['Body']

        self.story.append(Paragraph(text, style))

    def add_bullet_list(self, items):
        """
        Add a bulleted list.

        Args:
            items: List of strings for bullet points
        """
        for item in items:
            # Using bullet character with BulletItem style
            self.story.append(Paragraph(
                f"\u2022  {item}",
                self.styles['BulletItem']
            ))

    def add_numbered_list(self, items):
        """
        Add a numbered list.

        Args:
            items: List of strings for numbered items
        """
        for i, item in enumerate(items, 1):
            self.story.append(Paragraph(
                f"{i}.  {item}",
                self.styles['NumberedItem']
            ))

    def add_figure(self, figure_buffer, caption, width=None, height=None):
        """
        Add a figure with caption, kept together on the same page.

        Args:
            figure_buffer: BytesIO buffer containing PNG image
            caption: Caption text (auto-numbered as "Figure N.")
            width: Image width (default: CONTENT_WIDTH * 0.97 for safety margin)
            height: Image height (default: proportional based on image aspect ratio)
        """
        self.figure_count += 1

        # Use slightly smaller default to ensure fit within frame
        if width is None:
            width = CONTENT_WIDTH * 0.97

        # Create image - let reportlab calculate height from aspect ratio
        # by reading the image dimensions first
        from PIL import Image as PILImage
        figure_buffer.seek(0)
        pil_img = PILImage.open(figure_buffer)
        img_width, img_height = pil_img.size
        aspect_ratio = img_height / img_width
        figure_buffer.seek(0)

        if height is None:
            height = width * aspect_ratio

        img = Image(figure_buffer, width=width, height=height)

        # Create caption with figure number
        full_caption = f"<b>Figure {self.figure_count}.</b> {caption}"
        caption_para = Paragraph(full_caption, self.styles['Caption'])

        # Keep figure and caption together
        self.story.append(Spacer(1, 8))
        self.story.append(KeepTogether([img, caption_para]))

    def add_table(self, data, col_widths=None, caption=None, header_color=None):
        """
        Add a table with optional caption.

        Args:
            data: 2D list of cell contents (first row is header)
            col_widths: List of column widths (default: equal distribution)
            caption: Optional table caption (auto-numbered)
            header_color: Header background color (default: PRIMARY)
        """
        self.table_count += 1

        # Calculate column widths
        if col_widths is None:
            num_cols = len(data[0]) if data else 1
            col_widths = [CONTENT_WIDTH / num_cols] * num_cols

        # Process data through _wrap_cell_content for HTML tag handling
        processed_data = []
        for row_idx, row in enumerate(data):
            processed_row = [self._wrap_cell_content(cell, row_idx == 0) for cell in row]
            processed_data.append(processed_row)

        # Create table
        table = Table(processed_data, colWidths=col_widths)
        table.setStyle(get_table_style(header_color))

        elements = [Spacer(1, 8), table]

        # Add caption if provided
        if caption:
            full_caption = f"<b>Table {self.table_count}.</b> {caption}"
            elements.append(Paragraph(full_caption, self.styles['Caption']))

        self.story.append(KeepTogether(elements))

    def add_warning_box(self, title, items):
        """
        Add a warning/safety box with red styling.

        Args:
            title: Box header text
            items: List of warning items
        """
        data = [[title]]
        for item in items:
            cell = self._wrap_cell_content(f"\u2022  {item}", False)
            data.append([cell])

        table = Table(data, colWidths=[CONTENT_WIDTH])
        table.setStyle(get_warning_table_style())
        self.story.append(KeepTogether([Spacer(1, 8), table, Spacer(1, 8)]))

    def add_info_box(self, title, items):
        """
        Add an info/note box with blue styling.

        Args:
            title: Box header text
            items: List of info items
        """
        data = [[title]]
        for item in items:
            cell = self._wrap_cell_content(f"\u2022  {item}", False)
            data.append([cell])

        table = Table(data, colWidths=[CONTENT_WIDTH])
        table.setStyle(get_info_table_style())
        self.story.append(KeepTogether([Spacer(1, 8), table, Spacer(1, 8)]))

    def add_equation(self, equation_text):
        """
        Add a centered equation.

        Args:
            equation_text: Equation text (can use HTML entities for symbols)
        """
        self.story.append(Paragraph(equation_text, self.styles['Equation']))

    def add_code_block(self, code_text):
        """
        Add a code block with monospace font.

        Args:
            code_text: Code text
        """
        # Replace newlines with <br/> for reportlab
        formatted = code_text.replace('\n', '<br/>')
        self.story.append(Paragraph(formatted, self.styles['Code']))

    def add_note(self, text):
        """
        Add an italicized note.

        Args:
            text: Note text
        """
        self.story.append(Paragraph(text, self.styles['Note']))

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

    def add_references(self, references):
        """
        Add a references section.

        Args:
            references: List of reference strings (should include [N] prefix)
        """
        self.add_section_header("References")
        for ref in references:
            self.story.append(Paragraph(ref, self.styles['Reference']))

    def _header_footer(self, canvas, doc):
        """
        Draw header and footer on each page.
        """
        canvas.saveState()
        
        # Header
        canvas.setFont('Times-Roman', 9)
        canvas.setFillColor(colors.HexColor('#666666'))
        header_text = f"HIRT Whitepaper - Section {self.section_num}: {self.title}"
        canvas.drawString(MARGIN, PAGE_HEIGHT - MARGIN + 10, header_text)
        
        date_str = datetime.now().strftime("%B %Y")
        canvas.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN + 10, date_str)
        
        canvas.setStrokeColor(colors.HexColor('#cccccc'))
        canvas.line(MARGIN, PAGE_HEIGHT - MARGIN + 4, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - MARGIN + 4)

        # Footer
        page_num = canvas.getPageNumber()
        canvas.setFont('Times-Roman', 10)
        canvas.drawCentredString(PAGE_WIDTH / 2, MARGIN - 20, f"Page {page_num}")
        
        canvas.restoreState()

    def build(self):
        """
        Build and save the PDF document.

        Returns:
            Path to the generated PDF file
        """
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN,
            bottomMargin=MARGIN
        )

        doc.build(
            self.story,
            onFirstPage=self._header_footer,
            onLaterPages=self._header_footer
        )
        print(f"PDF created: {self.output_path}")
        return self.output_path

    def get_output_path(self):
        """Get the output file path."""
        return self.output_path


def create_combined_pdf(section_builders, output_path, title="HIRT Whitepaper"):
    """
    Create a combined PDF from multiple section builders.

    Args:
        section_builders: List of SectionPDFBuilder instances (already populated)
        output_path: Path for the combined PDF
        title: Document title

    Returns:
        Path to the generated PDF
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN
    )

    combined_story = []
    styles = get_styles()

    # Title page
    combined_story.append(Spacer(1, 2 * inch))
    combined_story.append(Paragraph(
        "HIRT: Hybrid Impedance-Resistivity Tomography System",
        styles['Title']
    ))
    combined_story.append(Paragraph(
        "Complete Technical Whitepaper",
        styles['Subtitle']
    ))
    combined_story.append(Spacer(1, inch))
    combined_story.append(Paragraph(
        "A Low-Cost, High-Resolution Subsurface Imaging System for "
        "Archaeological, Forensic, and Humanitarian Demining Applications",
        styles['Abstract']
    ))
    combined_story.append(Spacer(1, 2 * inch))
    combined_story.append(Paragraph("HIRT Development Team", styles['Author']))
    date_str = datetime.now().strftime("%B %Y")
    combined_story.append(Paragraph(
        f"<b>Version:</b> 2.0 | <b>Date:</b> {date_str}",
        styles['Meta']
    ))
    combined_story.append(PageBreak())

    # Table of Contents
    combined_story.append(Paragraph("Table of Contents", styles['TOCTitle']))
    for builder in section_builders:
        toc_entry = f"{builder.section_num:02d}. {builder.title}"
        combined_story.append(Paragraph(toc_entry, styles['TOCSection']))
    combined_story.append(PageBreak())

    # Add all sections
    for i, builder in enumerate(section_builders):
        if i > 0:
            combined_story.append(PageBreak())
        combined_story.extend(builder.story)

    doc.build(combined_story)
    print(f"Combined PDF created: {output_path}")
    return output_path
