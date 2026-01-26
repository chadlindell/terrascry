#!/usr/bin/env python3
"""
HIRT Narrative White Paper Generator
Renders the single-source narrative markdown into a professional PDF.
"""

import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
)

# Reuse diagram generators
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from io import BytesIO

SOURCE_FILE = "docs/whitepaper/HIRT_Scientific_Narrative_Full.md"
OUTPUT_FILENAME = "HIRT_Scientific_Whitepaper.pdf"

def get_styles():
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        'PaperTitle', parent=styles['Title'], fontName='Times-Bold', fontSize=24,
        textColor=HexColor('#1a365d'), spaceAfter=12, alignment=TA_CENTER, leading=28
    ))
    styles.add(ParagraphStyle(
        'PaperHeading1', parent=styles['Heading1'], fontName='Times-Bold', fontSize=14,
        textColor=HexColor('#1a365d'), spaceBefore=16, spaceAfter=8, keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'PaperHeading2', parent=styles['Heading2'], fontName='Times-BoldItalic', fontSize=12,
        textColor=HexColor('#2c5282'), spaceBefore=10, spaceAfter=4, keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        'PaperBody', parent=styles['Normal'], fontName='Times-Roman', fontSize=11,
        leading=15, alignment=TA_JUSTIFY, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        'PaperAbstract', parent=styles['Normal'], fontName='Times-Italic', fontSize=10,
        leading=13, alignment=TA_JUSTIFY, leftIndent=40, rightIndent=40, spaceAfter=20
    ))
    styles.add(ParagraphStyle(
        'PaperEquation', parent=styles['Normal'], fontName='Times-Italic', fontSize=11,
        alignment=TA_CENTER, spaceBefore=6, spaceAfter=12
    ))
    
    return styles

# --- Diagram Generation (Simplified Zone Arch) ---
def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    
    # Hub
    ax.add_patch(FancyBboxPatch((0.05, 0.2), 0.2, 0.6, boxstyle="round,pad=0.02", fc='#ebf8ff', ec='#2c5282'))
    ax.text(0.15, 0.5, "Central\nHub", ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Trunk
    ax.annotate('', xy=(0.4, 0.5), xytext=(0.25, 0.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.text(0.325, 0.55, "Trunk Cable", ha='center', fontsize=8)
    
    # Zone Box
    ax.add_patch(FancyBboxPatch((0.4, 0.3), 0.15, 0.4, boxstyle="round,pad=0.02", fc='#edf2f7', ec='#4a5568'))
    ax.text(0.475, 0.5, "Zone\nHub", ha='center', va='center', fontsize=9)
    
    # Probes
    for i in range(4):
        y = 0.65 - (i * 0.1)
        ax.plot([0.55, 0.7], [y, y], 'k-', lw=1)
        ax.add_patch(Rectangle((0.7, y-0.04), 0.02, 0.08, fc='#2c5282'))
        ax.text(0.75, y, f"Probe {i+1}", va='center', fontsize=8)

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    plt.close(fig)
    return buf

def parse_markdown(file_path, styles):
    flowables = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.rstrip()
        if not line: continue
        
        # Headers
        if line.startswith('# '):
            flowables.append(Paragraph(line[2:], styles['PaperTitle']))
        elif line.startswith('## '):
            flowables.append(Paragraph(line[3:], styles['PaperHeading1']))
        elif line.startswith('### '):
            flowables.append(Paragraph(line[4:], styles['PaperHeading2']))
            
        # Special Sections
        elif line.startswith('**Abstract**'):
            # The next lines until a header should be abstract. 
            # For simplicity, we assume the paragraph immediately following is abstract text.
            pass 
            
        # Equations (detect $$)
        elif line.strip().startswith('$$'):
            eq_text = line.replace('$$', '').strip()
            # Simple LaTeX-like replacement for PDF text
            eq_text = eq_text.replace('\rho_a', 'ρ_a').replace('\frac', '').replace('{V}{I}', 'V / I')
            flowables.append(Paragraph(eq_text, styles['PaperEquation']))
            
        # Images (Insertion Logic)
        elif "Zone Wiring Topology" in line:
             flowables.append(Paragraph(line, styles['PaperHeading2'])) # Print the header first
             img = Image(create_architecture_diagram(), width=6*inch, height=2.25*inch)
             flowables.append(img)
             flowables.append(Spacer(1, 12))
             
        # Normal Text
        else:
            # Check if this is the abstract paragraph (heuristic: follows Title, before Intro)
            if "The detection of unexploded ordnance" in line:
                flowables.append(Paragraph(line, styles['PaperAbstract']))
            else:
                # Basic Bold/Italic parsing
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
                flowables.append(Paragraph(text, styles['PaperBody']))
                
    return flowables

if __name__ == "__main__":
    doc = SimpleDocTemplate(
        OUTPUT_FILENAME,
        pagesize=letter,
        rightMargin=inch, leftMargin=inch,
        topMargin=inch, bottomMargin=inch
    )
    styles = get_styles()
    story = parse_markdown(SOURCE_FILE, styles)
    doc.build(story)
    print(f"Narrative PDF generated: {OUTPUT_FILENAME}")