#!/usr/bin/env python3
"""
HIRT White Paper Generator
Compiles all markdown sections into a single professional PDF.
"""

import os
import re
import glob
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, 
    Table, TableStyle, KeepTogether
)

# Import diagram generators from the intro script if possible, 
# or redefine them here. For robustness, we will redefine a simple
# placeholder generator or reuse the logic if I can import it.
# To keep this script standalone and robust, I'll include the plotting logic.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from io import BytesIO

# === CONFIG ===
SOURCE_DIR = "docs/whitepaper/sections"
OUTPUT_FILENAME = "HIRT_Whitepaper_v2.0.pdf"

# === STYLES ===
def get_styles():
    styles = getSampleStyleSheet()
    
    # Define Unique Style Names to avoid conflicts with ReportLab defaults
    styles.add(ParagraphStyle(
        'WP_MainTitle', parent=styles['Title'], fontName='Times-Bold', fontSize=24,
        textColor=HexColor('#1a365d'), spaceAfter=24, alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        'WP_SectionHeader', parent=styles['Heading1'], fontName='Times-Bold', fontSize=16,
        textColor=HexColor('#1a365d'), spaceBefore=18, spaceAfter=12, keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'WP_SubHeader', parent=styles['Heading2'], fontName='Times-Bold', fontSize=13,
        textColor=HexColor('#2c5282'), spaceBefore=12, spaceAfter=6, keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'WP_BodyText', parent=styles['Normal'], fontName='Times-Roman', fontSize=10,
        leading=13, alignment=TA_JUSTIFY, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        'WP_CodeBlock', parent=styles['Code'], fontName='Courier', fontSize=8,
        backColor=HexColor('#f7fafc'), borderPadding=6, spaceAfter=10, leading=10
    ))
    styles.add(ParagraphStyle(
        'WP_Bullet', parent=styles['Normal'], fontName='Times-Roman', fontSize=10,
        leftIndent=20, bulletIndent=10, spaceAfter=4, leading=12
    ))
    styles.add(ParagraphStyle(
        'WP_Caption', parent=styles['Italic'], fontName='Times-Italic', fontSize=9,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=12
    ))
    
    return styles

# === DIAGRAM GENERATORS (Condensed) ===

def create_zone_diagram():
    """Generates the Zone Wiring Architecture Diagram"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    
    # Draw Main Hub
    ax.add_patch(FancyBboxPatch((0.1, 0.4), 0.2, 0.4, boxstyle="round,pad=0.02", fc='#ebf8ff', ec='#2c5282'))
    ax.text(0.2, 0.6, "MAIN\nHUB", ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw Trunk Cable
    ax.plot([0.32, 0.6], [0.6, 0.6], 'k-', lw=3, color='#4a5568')
    ax.text(0.46, 0.65, "Trunk (DB25)", ha='center', fontsize=8)
    
    # Draw Zone Box
    ax.add_patch(FancyBboxPatch((0.6, 0.4), 0.15, 0.4, boxstyle="round,pad=0.02", fc='#edf2f7', ec='#4a5568'))
    ax.text(0.675, 0.6, "ZONE\nBOX", ha='center', va='center', fontsize=9)
    
    # Draw Probes
    for i in range(4):
        y = 0.8 - (i * 0.1)
        ax.plot([0.77, 0.85], [y, y], 'k-', lw=1)
        ax.add_patch(Circle((0.87, y), 0.03, fc='#2c5282'))
        ax.text(0.92, y, f"P{i+1}", va='center', fontsize=8)
        
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

# === PARSER ===

def clean_markdown(text):
    """Basic markdown to HTML converter for Paragraphs"""
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Inline code
    text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)
    return text

def parse_markdown(file_path, styles):
    flowables = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    in_code_block = False
    code_buffer = []
    
    for line in lines:
        line = line.rstrip()
        
        if line.startswith('```'):
            if in_code_block:
                text = "<br/>".join(code_buffer).replace(" ", "&nbsp;")
                flowables.append(Paragraph(text, styles['WP_CodeBlock']))
                code_buffer = []
                in_code_block = False
            else:
                in_code_block = True
            continue
            
        if in_code_block:
            code_buffer.append(line)
            continue
            
        if not line.strip():
            continue
            
        # Headers
        if line.startswith('# '):
            flowables.append(PageBreak())
            flowables.append(Paragraph(line[2:], styles['WP_SectionHeader']))
        elif line.startswith('## '):
            flowables.append(Paragraph(line[3:], styles['WP_SubHeader']))
        elif line.startswith('### '):
            flowables.append(Paragraph(line[4:], styles['WP_SubHeader']))
            
        # Tables (Simple detection)
        elif "|" in line and "---" not in line:
            # Table handling would go here, for now treat as text or skip
            # Scientific papers need real tables, but ReportLab Table conversion is complex
            # We'll skip raw md table lines to avoid cluttering the body
            continue
            
        # Lists
        elif line.strip().startswith(('-', '*', '1.')):
            text = clean_markdown(line.strip().lstrip('-*1. '))
            flowables.append(Paragraph(f"• {text}", styles['WP_Bullet']))
            
        # Diagrams
        elif "Figure 3." in line and "Architecture" in line:
             img = Image(create_zone_diagram(), width=6*inch, height=3*inch)
             flowables.append(img)
             flowables.append(Paragraph("<b>Figure 3:</b> Scalable Zone Architecture", styles['WP_Caption']))
             
        # Normal Text
        else:
            flowables.append(Paragraph(clean_markdown(line), styles['WP_BodyText']))
            
    return flowables

# === MAIN ===

def build_whitepaper():
    print(f"Scanning {SOURCE_DIR}...")
    files = sorted(glob.glob(f"{SOURCE_DIR}/*.md"))
    
    if not files:
        print("No markdown files found!")
        return

    doc = SimpleDocTemplate(
        OUTPUT_FILENAME,
        pagesize=letter,
        rightMargin=inch, leftMargin=inch,
        topMargin=inch, bottomMargin=inch
    )
    
    styles = get_styles()
    story = []
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("HIRT Project", styles['WP_MainTitle']))
    story.append(Paragraph("Technical White Paper v2.0", styles['WP_MainTitle']))
    story.append(Paragraph("Hybrid Inductive-Resistivity Tomography", styles['WP_SubHeader']))
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("Date: January 2026", styles['WP_BodyText']))
    story.append(PageBreak())
    
    # Process Files
    for file_path in files:
        print(f"Processing {os.path.basename(file_path)}...")
        section_flowables = parse_markdown(file_path, styles)
        story.extend(section_flowables)
        
    print("Building PDF...")
    doc.build(story)
    print(f"Success! {OUTPUT_FILENAME} created.")

if __name__ == "__main__":
    build_whitepaper()
