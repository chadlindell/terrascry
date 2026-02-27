#!/usr/bin/env python3
"""
HIRT Project Introduction - Professional Scientific PDF Generator
Comprehensive document with detailed diagrams and historical context.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon, Wedge, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects
import numpy as np
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    KeepTogether, HRFlowable, PageBreak, ListFlowable, ListItem
)
from reportlab.lib import colors

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
GROUND_TAN = HexColor('#d4a373')

# === Styles ===
def get_styles():
    styles = getSampleStyleSheet()

    # Modify existing Title style
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
        'Author', fontName='Times-Roman', fontSize=11, alignment=TA_CENTER, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        'Meta', fontName='Times-Roman', fontSize=9, textColor=HexColor('#666'),
        alignment=TA_CENTER, spaceAfter=18
    ))
    styles.add(ParagraphStyle(
        'AbstractLabel', fontName='Times-Bold', fontSize=10, textColor=PRIMARY,
        spaceBefore=0, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        'Abstract', fontName='Times-Roman', fontSize=10, alignment=TA_JUSTIFY,
        leftIndent=18, rightIndent=18, leading=13, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        'Keywords', fontName='Times-Roman', fontSize=9, textColor=HexColor('#555'),
        leftIndent=18, rightIndent=18, spaceAfter=18
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
        'Reference', fontName='Times-Roman', fontSize=9, alignment=TA_JUSTIFY,
        leftIndent=18, firstLineIndent=-18, spaceAfter=3, leading=11
    ))
    styles.add(ParagraphStyle(
        'Equation', fontName='Times-Italic', fontSize=10, alignment=TA_CENTER,
        spaceBefore=8, spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        'BulletItem', fontName='Times-Roman', fontSize=10, alignment=TA_JUSTIFY,
        leftIndent=25, firstLineIndent=-12, spaceAfter=4, leading=12, bulletIndent=15
    ))
    styles.add(ParagraphStyle(
        'TableNote', fontName='Times-Italic', fontSize=8, textColor=HexColor('#666'),
        alignment=TA_LEFT, spaceBefore=2, spaceAfter=8
    ))

    return styles


# ============================================================================
# FIGURE 1: CROSSHOLE VS SURFACE COMPARISON
# ============================================================================
def create_crosshole_comparison():
    """
    Professional comparison of surface method vs HIRT crosshole method
    showing electromagnetic field patterns and sensitivity zones.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Common styling
    ground_color = '#d4a373'
    sky_color = '#e8f4f8'

    # === Panel A: Surface Method ===
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-6.5, 1)

    # Sky/air region
    ax1.fill_between([-5, 5], [0, 0], [1, 1], color=sky_color, alpha=0.5)

    # Ground layers with gradient
    for i, (d1, d2, alpha) in enumerate([
        (0, -1.5, 0.3), (-1.5, -3, 0.4), (-3, -4.5, 0.5), (-4.5, -6.5, 0.6)
    ]):
        ax1.fill_between([-5, 5], [d1, d1], [d2, d2],
                        color=ground_color, alpha=alpha)

    # Ground surface line
    ax1.axhline(0, color='#654321', linewidth=2.5)
    ax1.text(-4.8, 0.15, 'Ground Surface', fontsize=8, color='#654321')

    # Surface sensors
    sensor_x = [-3, -1, 1, 3]
    for x in sensor_x:
        ax1.add_patch(plt.Polygon([[x-0.3, 0], [x+0.3, 0], [x, 0.4]],
                                  color='#2c5282', ec='black', lw=1))
    ax1.text(0, 0.65, 'Surface Sensors', ha='center', fontsize=9, fontweight='bold')

    # Sensitivity decay pattern (concentric curves)
    for radius, alpha in [(1.5, 0.3), (2.5, 0.2), (3.5, 0.15), (4.5, 0.1)]:
        theta = np.linspace(np.pi, 2*np.pi, 50)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        ax1.plot(x, y, color='#3182ce', alpha=alpha, linewidth=1.5)

    # Signal paths (curved, showing down-and-back)
    for start_x in [-1, 1]:
        path_x = np.linspace(start_x, 0, 20)
        path_y = -2.5 * np.sin(np.linspace(0, np.pi, 20)) ** 0.5 - np.abs(path_x - start_x) * 0.3
        ax1.plot(path_x, path_y, 'b--', alpha=0.4, linewidth=1.5)
        ax1.plot(path_x, path_y - 0.3, 'b--', alpha=0.3, linewidth=1.5)

    # Target at depth
    target = plt.Circle((0, -3.5), 0.5, color='#e53e3e', ec='#9b2c2c', linewidth=2)
    ax1.add_patch(target)
    ax1.annotate('Target\n(weak signal)', (0, -3.5), (2.5, -3.5),
                fontsize=8, ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='#e53e3e'))

    # Attenuation zone labels
    ax1.text(-4.5, -1.2, 'HIGH\nSensitivity', fontsize=7, color='#2c5282', alpha=0.8)
    ax1.text(-4.5, -3, 'LOW\nSensitivity', fontsize=7, color='#2c5282', alpha=0.4)

    # Depth markers
    for d in [1, 2, 3, 4, 5, 6]:
        ax1.axhline(-d, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax1.text(4.7, -d, f'{d}m', fontsize=7, va='center', color='gray')

    ax1.set_title('(a) Surface Method', fontsize=11, fontweight='bold', color='#1a365d', pad=10)
    ax1.set_ylabel('Depth (m)', fontsize=9)
    ax1.set_yticks([0, -2, -4, -6])
    ax1.set_yticklabels(['0', '2', '4', '6'])
    ax1.set_xticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    # === Panel B: HIRT Crosshole Method ===
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-6.5, 1)

    # Sky/air region
    ax2.fill_between([-5, 5], [0, 0], [1, 1], color=sky_color, alpha=0.5)

    # Ground layers with gradient
    for i, (d1, d2, alpha) in enumerate([
        (0, -1.5, 0.3), (-1.5, -3, 0.4), (-3, -4.5, 0.5), (-4.5, -6.5, 0.6)
    ]):
        ax2.fill_between([-5, 5], [d1, d1], [d2, d2],
                        color=ground_color, alpha=alpha)

    # Ground surface line
    ax2.axhline(0, color='#654321', linewidth=2.5)

    # Probes
    probe_x = [-2.5, 2.5]
    probe_color = '#2c5282'
    for x in probe_x:
        # Probe rod
        ax2.add_patch(Rectangle((x-0.15, -5), 0.3, 5, color=probe_color, ec='black', lw=1))
        # Junction box at top
        ax2.add_patch(Rectangle((x-0.35, -0.1), 0.7, 0.4, color='#4a5568', ec='black', lw=1))

        # Coil positions (TX/RX)
        for depth, color, label in [(-1.5, '#38a169', 'TX'), (-3, '#3182ce', 'RX'), (-4.5, '#ed8936', 'RX')]:
            ax2.add_patch(Circle((x, depth), 0.2, color=color, ec='black', lw=1))

        # ERT rings
        for depth in [-2, -3.5, -4.8]:
            ax2.add_patch(Rectangle((x-0.25, depth-0.08), 0.5, 0.16,
                                   color='#ed8936', ec='black', lw=0.5))

    # Labels
    ax2.text(-2.5, 0.6, 'Probe 1', ha='center', fontsize=9, fontweight='bold')
    ax2.text(2.5, 0.6, 'Probe 2', ha='center', fontsize=9, fontweight='bold')

    # Ray paths between probes (showing dense coverage)
    ray_colors = plt.cm.Greens(np.linspace(0.4, 0.8, 9))
    ray_depths_left = [-1.5, -1.5, -1.5, -3, -3, -3, -4.5, -4.5, -4.5]
    ray_depths_right = [-1.5, -3, -4.5, -1.5, -3, -4.5, -1.5, -3, -4.5]
    for i, (d1, d2) in enumerate(zip(ray_depths_left, ray_depths_right)):
        ax2.plot([-2.3, 2.3], [d1, d2], color=ray_colors[i], linewidth=1.5, alpha=0.7)

    # Sensitivity ellipse (high sensitivity zone)
    ellipse = mpatches.Ellipse((0, -3), 4, 3, color='#38a169', alpha=0.15)
    ax2.add_patch(ellipse)
    ax2.text(0, -3, 'HIGH\nSensitivity\nZone', ha='center', va='center',
             fontsize=8, color='#276749', fontweight='bold')

    # Target at depth (strong signal)
    target = plt.Circle((0, -3.5), 0.5, color='#e53e3e', ec='#9b2c2c', linewidth=2)
    ax2.add_patch(target)
    ax2.annotate('Target\n(strong signal)', (0, -3.5), (-3.5, -5.5),
                fontsize=8, ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='#e53e3e'))

    # Depth markers
    for d in [1, 2, 3, 4, 5, 6]:
        ax2.axhline(-d, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        ax2.text(4.7, -d, f'{d}m', fontsize=7, va='center', color='gray')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#38a169', label='TX Coil'),
        mpatches.Patch(color='#3182ce', label='RX Coil'),
        mpatches.Patch(color='#ed8936', label='ERT Ring'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.9)

    ax2.set_title('(b) HIRT Crosshole Method', fontsize=11, fontweight='bold', color='#1a365d', pad=10)
    ax2.set_yticks([0, -2, -4, -6])
    ax2.set_yticklabels(['0', '2', '4', '6'])
    ax2.set_xticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 2: SYSTEM ARCHITECTURE
# ============================================================================
def create_system_architecture():
    """
    Detailed system architecture showing electronics hub, zone wiring, and probe design.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))  # Reduced height
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # === Central Electronics Hub ===
    hub_box = FancyBboxPatch((0.2, 3.2), 3.0, 2.6, boxstyle="round,pad=0.05",
                             facecolor='#ebf8ff', edgecolor='#2c5282', linewidth=2)
    ax.add_patch(hub_box)
    ax.text(1.7, 5.55, 'Central Electronics Hub', fontsize=10, fontweight='bold',
            ha='center', color='#1a365d')

    # Internal Hub Blocks (Simplified)
    ax.add_patch(FancyBboxPatch((0.4, 4.5), 1.0, 0.8, boxstyle="round,pad=0.02",
                                facecolor='#c6f6d5', edgecolor='#276749', linewidth=1))
    ax.text(0.9, 4.9, 'MIT TX/RX\n(Analog FE)', fontsize=7, ha='center', va='center')

    ax.add_patch(FancyBboxPatch((1.6, 4.5), 1.0, 0.8, boxstyle="round,pad=0.02",
                                facecolor='#feebc8', edgecolor='#c05621', linewidth=1))
    ax.text(2.1, 4.9, 'ERT Source\n& Measure', fontsize=7, ha='center', va='center')

    ax.add_patch(FancyBboxPatch((0.4, 3.5), 2.2, 0.8, boxstyle="round,pad=0.02",
                                facecolor='#e9d8fd', edgecolor='#6b46c1', linewidth=1))
    ax.text(1.5, 3.9, 'MCU + High-Density\nAnalog Mux Array', fontsize=7, ha='center', va='center')

    # === Zone Wiring Architecture ===
    
    # Trunk Cable (High Density)
    ax.annotate('', xy=(3.8, 4.0), xytext=(3.2, 4.0),
                arrowprops=dict(arrowstyle='-', color='#4a5568', lw=3))
    ax.text(3.5, 4.2, 'Trunk Cable\n(DB25)', fontsize=7, ha='center', color='#4a5568')

    # Zone Box
    zone_box = FancyBboxPatch((3.8, 3.5), 1.2, 1.0, boxstyle="round,pad=0.05",
                             facecolor='#edf2f7', edgecolor='#4a5568', linewidth=1.5)
    ax.add_patch(zone_box)
    ax.text(4.4, 4.0, 'Zone Box\n(Passive)', fontsize=8, fontweight='bold', ha='center')

    # Branch cables to probes
    for y_offset in [-0.2, 0.0, 0.2]:
        start = (5.0, 4.0 + y_offset)
        end = (5.8, 4.0 + y_offset*3)
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', lw=1)

    # === Probes (in Zone) ===
    probe_positions = [(5.8, 3.4), (5.8, 4.0), (5.8, 4.6)]
    for i, (px, py) in enumerate(probe_positions):
        # Probe rod top
        ax.add_patch(Rectangle((px-0.08, py-0.8), 0.16, 0.8,
                               color='#2c5282', ec='black', lw=1))
        # Junction Box
        ax.add_patch(Rectangle((px-0.15, py), 0.3, 0.15,
                               color='#4a5568', ec='black', lw=1))
        # Label
        ax.text(px, py+0.25, f'P{i+1}', fontsize=7, ha='center')

    # Ground line
    ax.axhline(3.0, xmin=0.55, xmax=0.65, color='#654321', linewidth=2)
    ax.text(5.8, 2.8, 'Ground', fontsize=7, ha='center', color='#654321')

    # === Probe Detail Inset (Right Side) ===
    inset_x = 7.0
    ax.add_patch(FancyBboxPatch((7.0, 0.2), 2.8, 5.5, boxstyle="round,pad=0.05",
                                facecolor='white', edgecolor='#718096', linewidth=1))
    ax.text(8.4, 5.5, 'Passive Micro-Probe Detail', fontsize=9, fontweight='bold',
            ha='center', color='#1a365d')

    # Large probe cross-section
    probe_cx = 8.4
    ax.add_patch(Rectangle((probe_cx-0.25, 0.5), 0.5, 4.5,
                           color='#2c5282', ec='black', lw=1.5, alpha=0.8))

    # Internal components with labels
    components = [
        (4.6, 'Junction Box (Passive)', '#4a5568', 0.3),
        (4.0, 'M12x1.75 Flush Joint', '#a0aec0', 0.1),
        (3.4, 'TX Coil (Internal)', '#38a169', 0.4),
        (2.8, 'M12x1.75 Flush Joint', '#a0aec0', 0.1),
        (2.0, 'RX Coil (Internal)', '#3182ce', 0.4),
        (1.2, 'ERT Ring (Flush)', '#ed8936', 0.1),
        (0.6, 'Probe Tip (16mm)', '#1a365d', 0.2),
    ]

    for y, label, color, height in components:
        ax.add_patch(Rectangle((probe_cx-0.25, y), 0.5, height,
                               color=color, ec='black', lw=0.5))
        ax.annotate(label, (probe_cx + 0.3, y + height/2), (probe_cx + 1.2, y + height/2),
                   fontsize=7, va='center',
                   arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Dimensions
    ax.annotate('', xy=(probe_cx-0.3, 5.15), xytext=(probe_cx+0.3, 5.15),
                arrowprops=dict(arrowstyle='<->', color='#718096', lw=1))
    ax.text(probe_cx, 5.25, '16mm OD', fontsize=8, ha='center', fontweight='bold')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 3: DEPTH COMPARISON CHART
# ============================================================================
def create_depth_comparison():
    """
    Professional horizontal bar chart comparing method depths with ranges.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Data: method name, min depth, typical depth, max depth, color, category
    methods = [
        ('Surface Magnetometry', 0.5, 1.2, 1.8, '#718096', 'Surface'),
        ('GPR (dry sand)', 1.0, 2.0, 3.5, '#718096', 'Surface'),
        ('GPR (moist clay)', 0.2, 0.4, 0.8, '#a0aec0', 'Surface'),
        ('Surface ERT', 1.0, 2.0, 3.5, '#718096', 'Surface'),
        ('EM31/CMD', 1.0, 1.5, 2.5, '#718096', 'Surface'),
        ('HIRT (1.5m probes)', 1.5, 2.5, 3.5, '#38a169', 'Crosshole'),
        ('HIRT (3.0m probes)', 2.5, 4.0, 5.5, '#276749', 'Crosshole'),
        ('Commercial Crosshole', 3.0, 5.0, 8.0, '#3182ce', 'Crosshole'),
    ]

    y_pos = np.arange(len(methods))

    # Draw range bars
    for i, (name, min_d, typ_d, max_d, color, cat) in enumerate(methods):
        # Full range bar (light)
        ax.barh(i, max_d - min_d, left=min_d, height=0.5,
                color=color, alpha=0.3, edgecolor='none')
        # Typical range marker
        ax.plot([typ_d], [i], 'k|', markersize=12, markeredgewidth=2)
        # Min/max markers
        ax.plot([min_d, max_d], [i, i], color=color, linewidth=2)
        ax.plot([min_d], [i], '|', color=color, markersize=8, markeredgewidth=2)
        ax.plot([max_d], [i], '|', color=color, markersize=8, markeredgewidth=2)

    # Depth zone highlighting
    ax.axvspan(0.8, 1.5, alpha=0.1, color='gray', label='Foundation depth')
    ax.axvspan(2.0, 4.0, alpha=0.15, color='#e53e3e', label='UXB burial range')

    # UXB depth line
    ax.axvline(3.0, color='#c53030', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(3.1, 7.5, 'Typical UXB\ndepth (3m)', fontsize=7, color='#c53030', va='top')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[0] for m in methods], fontsize=9)
    ax.set_xlabel('Effective Depth of Investigation (m)', fontsize=10)
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.6, len(methods) - 0.4)

    # Add grid
    ax.xaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    # Category separators
    ax.axhline(4.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.text(8.5, 2, 'Surface\nMethods', fontsize=8, va='center', ha='center',
            color='#718096', fontstyle='italic')
    ax.text(8.5, 6, 'Crosshole\nMethods', fontsize=8, va='center', ha='center',
            color='#276749', fontstyle='italic')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor='#e53e3e', alpha=0.15, label='UXB depth range'),
        Line2D([0], [0], color='black', marker='|', linestyle='None',
               markersize=10, markeredgewidth=2, label='Typical depth'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.9)

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
# FIGURE 4: APPLICATION SCENARIOS
# ============================================================================
def create_scenarios_diagram():
    """
    Three-panel diagram showing HIRT application scenarios with ray paths.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    scenarios = [
        ('(a) Bomb Crater', '#c53030'),
        ('(b) Woodland Burial', '#276749'),
        ('(c) Wetland Survey', '#2b6cb0'),
    ]

    for ax, (title, accent_color) in zip(axes, scenarios):
        ax.set_xlim(-4, 4)
        ax.set_ylim(-5, 1)
        ax.set_aspect('equal')

        # Ground surface
        ax.axhline(0, color='#654321', linewidth=2)
        ax.fill_between([-4, 4], [0, 0], [-5, -5], color='#d4a373', alpha=0.3)

        ax.set_title(title, fontsize=10, fontweight='bold', color='#1a365d', pad=8)
        ax.set_ylabel('Depth (m)', fontsize=8)
        ax.set_yticks([0, -2, -4])
        ax.set_yticklabels(['0', '2', '4'], fontsize=7)
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # === Panel A: Bomb Crater ===
    ax = axes[0]

    # Crater shape
    crater_x = np.linspace(-2.5, 2.5, 50)
    crater_y = -0.5 - 2.5 * (1 - (crater_x/2.5)**2)
    ax.fill_between(crater_x, crater_y, 0, color='#a0aec0', alpha=0.5)
    ax.plot(crater_x, crater_y, color='#4a5568', linewidth=1.5)
    ax.text(0, -1, 'Crater fill', fontsize=7, ha='center', color='#4a5568')

    # Probes around perimeter
    probe_x = [-3, -1.5, 0, 1.5, 3]
    for px in probe_x:
        ax.add_patch(Rectangle((px-0.08, -4), 0.16, 4, color='#2c5282'))
        # Sensors
        for dy in [1.0, 2.0, 3.0]:
            ax.add_patch(Circle((px, -dy), 0.12, color='#38a169'))

    # Ray paths through target
    for i, p1 in enumerate(probe_x[:-1]):
        for p2 in probe_x[i+1:]:
            ax.plot([p1, p2], [-2.5, -2.5], 'g-', alpha=0.3, linewidth=0.8)
            ax.plot([p1, p2], [-3.0, -3.0], 'g-', alpha=0.3, linewidth=0.8)

    # Target
    ax.add_patch(Circle((0, -3), 0.4, color='#e53e3e', ec='#9b2c2c', lw=1.5))
    ax.text(0, -3, 'UXB', fontsize=6, ha='center', va='center', color='white', fontweight='bold')
    ax.text(0, -4.5, '3m probes, 1.5m spacing', fontsize=7, ha='center', color='#4a5568')

    # === Panel B: Woodland Burial ===
    ax = axes[1]

    # Trees
    for tx in [-2.5, 2]:
        ax.add_patch(Rectangle((tx-0.1, 0), 0.2, 0.6, color='#553c9a'))
        ax.add_patch(Circle((tx, 0.9), 0.4, color='#38a169'))

    # Burial shaft
    ax.fill_between([-0.8, 0.8], [-1.5, -1.5], [-0.3, -0.3], color='#805ad5', alpha=0.4)
    ax.plot([-0.8, -0.8, 0.8, 0.8], [-0.3, -1.5, -1.5, -0.3], color='#553c9a', linewidth=1.5, linestyle='--')
    ax.text(0, -0.9, 'Disturbed\nsoil', fontsize=6, ha='center', va='center', color='#553c9a')

    # Probes
    probe_x = [-1.8, -0.6, 0.6, 1.8]
    for px in probe_x:
        ax.add_patch(Rectangle((px-0.08, -2), 0.16, 2, color='#2c5282'))
        for dy in [0.5, 1.2]:
            ax.add_patch(Circle((px, -dy), 0.1, color='#38a169'))

    # Dense ray paths
    for p1, p2 in [(-1.8, -0.6), (-0.6, 0.6), (0.6, 1.8), (-1.8, 0.6), (-0.6, 1.8)]:
        ax.plot([p1, p2], [-0.8, -0.8], 'g-', alpha=0.4, linewidth=0.8)
        ax.plot([p1, p2], [-1.2, -1.2], 'g-', alpha=0.4, linewidth=0.8)

    ax.text(0, -4.5, '1.5m probes, 1m spacing', fontsize=7, ha='center', color='#4a5568')

    # === Panel C: Wetland Survey ===
    ax = axes[2]

    # Water body
    ax.fill_between([-2.5, 2.5], [0, 0], [-1.5, -1.5], color='#90cdf4', alpha=0.5)
    ax.text(0, -0.7, 'Water', fontsize=7, ha='center', color='#2b6cb0')

    # Shore/margins
    ax.fill_between([-4, -2.5], [0, 0], [-0.5, 0], color='#d4a373', alpha=0.5)
    ax.fill_between([2.5, 4], [0, 0], [0, -0.5], color='#d4a373', alpha=0.5)

    # Organic layer
    ax.fill_between([-2.5, 2.5], [-1.5, -1.5], [-2.5, -2.5], color='#553c9a', alpha=0.3)
    ax.text(0, -2, 'Organic', fontsize=6, ha='center', color='#553c9a')

    # Probes from margins only
    for px in [-3.2, -2.8]:
        ax.add_patch(Rectangle((px-0.08, -3.5), 0.16, 3.5, color='#2c5282'))
        for dy in [1.0, 2.0, 3.0]:
            ax.add_patch(Circle((px, -dy), 0.1, color='#38a169'))
    for px in [2.8, 3.2]:
        ax.add_patch(Rectangle((px-0.08, -3.5), 0.16, 3.5, color='#2c5282'))
        for dy in [1.0, 2.0, 3.0]:
            ax.add_patch(Circle((px, -dy), 0.1, color='#38a169'))

    # Diagonal ray paths toward deep target
    ax.plot([-2.8, 2.8], [-2.0, -3.0], 'g-', alpha=0.4, linewidth=0.8)
    ax.plot([-3.2, 3.2], [-2.0, -3.0], 'g-', alpha=0.4, linewidth=0.8)
    ax.plot([-2.8, 2.8], [-3.0, -2.0], 'g-', alpha=0.4, linewidth=0.8)

    # Deep target
    ax.add_patch(Circle((0, -4), 0.35, color='#e53e3e', ec='#9b2c2c', lw=1.5))
    ax.annotate('Deep\ntarget', (0, -4), (0, -3.2), fontsize=6, ha='center',
                arrowprops=dict(arrowstyle='->', color='#e53e3e', lw=0.8))

    ax.text(0, -4.7, 'Perimeter deployment', fontsize=7, ha='center', color='#4a5568')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# FIGURE 5: HISTORICAL CONTEXT - BOMBING STATISTICS
# ============================================================================
def create_historical_figure():
    """
    Visual representation of bombing campaign statistics and global UXO crisis.
    """
    fig = plt.figure(figsize=(10, 4))

    # Create gridspec for different sized panels
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

    # === Left Panel: Kozle Basin Statistics ===
    ax1 = fig.add_subplot(gs[0])

    # Create visual representation of bombing scale
    categories = ['Bombs\nDropped', 'Mapped\nCraters', 'Est. UXBs', 'High-Risk\nSites']
    values = [39137, 6000, 5000, 180]
    colors = ['#e53e3e', '#ed8936', '#ecc94b', '#38a169']

    # Normalize for visualization (log scale effect)
    display_values = [np.log10(v) * 20 for v in values]

    bars = ax1.bar(categories, display_values, color=colors, edgecolor='black', linewidth=1)

    # Add actual values as labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Relative Scale (log)', fontsize=9)
    ax1.set_title('Kozle Basin, Poland (1944-45)', fontsize=11, fontweight='bold',
                  color='#1a365d', pad=10)
    ax1.set_ylim(0, 100)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.set_yticks([])

    # Add context note
    ax1.text(0.5, -0.15, 'Max density: 77 bombs/hectare | 10-15% failure rate',
             transform=ax1.transAxes, fontsize=7, ha='center', color='#4a5568',
             fontstyle='italic')

    # === Right Panel: Global UXO Crisis ===
    ax2 = fig.add_subplot(gs[1])

    # Countries with UXO casualties
    countries = ['Laos', 'Vietnam', 'Cambodia', 'Ukraine', 'Germany']
    casualties = [20000, 40000, 65000, 1379, 200]  # Approximate figures
    contaminated = [30, 19, 100, 23, 5]  # Percentage of land

    # Create grouped comparison
    x = np.arange(len(countries))
    width = 0.35

    bars1 = ax2.bar(x - width/2, [c/1000 for c in casualties], width,
                    label='Casualties (thousands)', color='#e53e3e', alpha=0.8)
    bars2 = ax2.bar(x + width/2, contaminated, width,
                    label='Land contaminated (%)', color='#ed8936', alpha=0.8)

    ax2.set_ylabel('Value', fontsize=9)
    ax2.set_title('Global UXO Crisis', fontsize=11, fontweight='bold',
                  color='#1a365d', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(countries, fontsize=8)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Note
    ax2.text(0.5, -0.15, '60+ countries affected | 100+ million people at risk',
             transform=ax2.transAxes, fontsize=7, ha='center', color='#4a5568',
             fontstyle='italic')

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
def build_document():
    """Build the complete PDF document."""

    print("Generating figures...")
    fig_crosshole = create_crosshole_comparison()
    fig_architecture = create_system_architecture()
    fig_depth = create_depth_comparison()
    fig_scenarios = create_scenarios_diagram()
    fig_historical = create_historical_figure()

    print("Building PDF...")
    styles = get_styles()

    doc = SimpleDocTemplate(
        "HIRT_Introduction.pdf",
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN
    )

    story = []

    # === TITLE BLOCK ===
    story.append(Paragraph(
        "HIRT: Hybrid Impedance-Resistivity Tomography System",
        styles['Title']
    ))
    story.append(Paragraph(
        "A Low-Cost, High-Resolution Subsurface Imaging System for Archaeological, "
        "Forensic, and Humanitarian Demining Applications",
        styles['Subtitle']
    ))
    story.append(Paragraph("HIRT Development Team", styles['Author']))
    story.append(Paragraph(
        "<b>Version:</b> 2.0 | <b>Date:</b> January 2026 | <b>Status:</b> Technical Introduction",
        styles['Meta']
    ))

    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=12))

    # === ABSTRACT ===
    story.append(Paragraph("<b>Abstract</b>", styles['AbstractLabel']))
    story.append(Paragraph(
        "Buried beneath the surface of former conflict zones lies a deadly legacy: millions of tons of "
        "unexploded ordnance (UXO) that threaten civilians and obstruct historical recovery. "
        "Current methods for detecting these deep targets are either dangerously primitive (manual probing) "
        "or prohibitively expensive (commercial geophysics). This document introduces the **HIRT "
        "(Hybrid Impedance-Resistivity Tomography)** system—a novel, low-cost solution designed to "
        "democratize high-resolution subsurface imaging. By combining two sensing technologies—"
        "Magneto-Inductive Tomography (MIT) and Electrical Resistivity Tomography (ERT)—into a "
        "passive probe array, HIRT creates 3D models of the subsurface similar to a medical CT scan. "
        "This system enables archaeological and humanitarian teams to 'see' metallic wreckage, filled craters, "
        "and clandestine burials at depths of 3–6 meters without excavation. Developed in response to the "
        "ongoing crisis in Poland's Kozle Basin, HIRT aims to provide a safe, scalable, and scientifically "
        "rigorous tool for reclaiming hazardous ground.",
        styles['Abstract']
    ))
    story.append(Paragraph(
        "<b>Keywords:</b> crosshole tomography, electrical resistivity, magneto-inductive, "
        "UXO detection, humanitarian demining, conflict archaeology, subsurface imaging",
        styles['Keywords']
    ))

    # === 1. INTRODUCTION ===
    story.append(Paragraph("1. Introduction: The Hidden Legacy of War", styles['Section']))
    story.append(Paragraph(
        "For millions of people worldwide, the war is not over. Long after peace treaties are signed, "
        "land remains contaminated with unexploded ordnance (UXO), transforming fields, forests, and "
        "construction sites into potential minefields. More than 60 countries grapple with this "
        "legacy, putting over 100 million people at risk. The challenge is not just removing these "
        "hazards, but *finding* them—often buried deep in soil that has shifted over decades.",
        styles['BodyFirst']
    ))
    story.append(Paragraph(
        "The legacy of World War II is particularly persistent. In Germany alone, approximately 2,000 tons of "
        "munitions are discovered annually. But as these devices age, they become more unstable, and their "
        "chemical fuses degrade. The window for safe removal is closing, yet the pace of discovery is "
        "limited by the tools available to field teams. To accelerate this work, we need technology that "
        "can look deeper and clearer than standard metal detectors, without the crushing cost of "
        "industrial geophysical services.",
        styles['Body']
    ))

    story.append(Paragraph("1.1 Historical Motivation: The Kozle Basin", styles['Subsection']))
    story.append(Paragraph(
        "The development of the HIRT system was directly inspired by the specific, urgent needs of the "
        "Kozle Basin in Poland. Once the site of the Third Reich's massive Blechhammer synthetic fuel "
        "plants, this region was the target of relentless Allied bombing campaigns in 1944–1945. "
        "Over 39,000 bombs were dropped, turning the landscape into a cratered moonscape [3].",
        styles['BodyFirst']
    ))
    story.append(Paragraph(
        "Today, thousands of these craters remain, many filled with sediment and concealing unexploded "
        "bombs. Recent research suggests 4,000–6,000 UXBs may still lie dormant in the basin. "
        "Standard surface scanners struggle here: the conductive clay soils block ground-penetrating radar "
        "(GPR), and the depth of the targets (3–6 meters) renders handheld detectors useless. "
        "The HIRT system was conceived to solve this specific problem: to penetrate conductive soil and "
        "identify deep anomalies without the risk of physical excavation.",
        styles['Body']
    ))

    # Historical figure
    fig1_img = Image(fig_historical, width=CONTENT_WIDTH, height=CONTENT_WIDTH * 4/10)
    fig1_caption = Paragraph(
        "<b>Figure 1.</b> The scale of the challenge. Left: Statistics from the Kozle Basin bombing campaign "
        "reveal the density of potential hazards [3]. Right: The global context of the UXO crisis, highlighting "
        "the need for scalable detection solutions.",
        styles['Caption']
    )
    story.append(Spacer(1, 8))
    story.append(KeepTogether([fig1_img, fig1_caption]))

    story.append(Paragraph("1.2 The Mission: Democratizing Subsurface Vision", styles['Subsection']))
    story.append(Paragraph(
        "Current humanitarian demining often relies on tools that have barely changed since the 1940s: "
        "prodding sticks and basic metal detectors. While effective for surface mines, they fail against "
        "deeply buried aircraft bombs. Conversely, the oil and gas industry uses sophisticated 'Crosshole Tomography' "
        "systems that offer incredible 3D resolution—but at a cost of $50,000 to $200,000 per unit.",
        styles['BodyFirst']
    ))
    story.append(Paragraph(
        "**The mission of the HIRT project is to bridge this gap.** By utilizing modern microcontrollers, "
        "3D printing, and open-source hardware, we have re-engineered the principles of crosshole tomography "
        "into a system that can be built for under $4,000. This places professional-grade imaging capabilities "
        "into the hands of university researchers, non-profits, and historical recovery teams who previously "
        "could only guess at what lay beneath the surface.",
        styles['Body']
    ))

    # === 2. PHYSICS AND MEASUREMENT PRINCIPLES ===
    story.append(Paragraph("2. Physics and Measurement Principles", styles['Section']))

    story.append(Paragraph("2.1 The Crosshole Geometry Advantage", styles['Subsection']))
    story.append(Paragraph(
        "The fundamental advantage of crosshole geometry lies in the physics of electromagnetic and electrical "
        "field propagation. Surface methods must send energy down to a target and receive the return "
        "signal&mdash;doubling the path length and exponentially increasing attenuation. In contrast, "
        "crosshole methods send signals horizontally through the target volume, with sensitivity concentrated "
        "precisely where targets are located. Figure 2 illustrates this geometric advantage.",
        styles['BodyFirst']
    ))

    # Crosshole comparison figure
    fig2_img = Image(fig_crosshole, width=CONTENT_WIDTH, height=CONTENT_WIDTH * 5/10)
    fig2_caption = Paragraph(
        "<b>Figure 2.</b> Comparison of (a) surface method versus (b) HIRT crosshole method. Surface sensors "
        "must contend with rapid sensitivity decay with depth (1/r<sup>2</sup> to 1/r<sup>4</sup> falloff) and "
        "near-surface interference zones. Crosshole probes provide direct ray paths through the target volume "
        "with high sensitivity maintained at the investigation depth. The green ellipse in (b) indicates the "
        "zone of maximum sensitivity between probes.",
        styles['Caption']
    )
    story.append(Spacer(1, 8))
    story.append(KeepTogether([fig2_img, fig2_caption]))

    story.append(Paragraph(
        "The electromagnetic skin depth (&delta;) determines the effective penetration of MIT signals:",
        styles['Body']
    ))
    story.append(Paragraph(
        "&delta; = &radic;(2 / &omega;&mu;&sigma;)",
        styles['Equation']
    ))
    story.append(Paragraph(
        "where &omega; = angular frequency, &mu; = permeability, and &sigma; = conductivity. At 10 kHz in "
        "typical soil (&sigma; = 0.01 S/m), skin depth exceeds 50 meters. However, the practical limitation "
        "is coil coupling geometry, which scales as 1/r<sup>3</sup> in near-field conditions. The effective "
        "MIT investigation depth is approximately 1-2x the probe spacing.",
        styles['Body']
    ))

    story.append(Paragraph("2.2 Dual-Channel Detection: MIT and ERT", styles['Subsection']))
    story.append(Paragraph(
        "HIRT employs two complementary sensing modalities. Magneto-Inductive Tomography (MIT) uses "
        "oscillating magnetic fields at 2-50 kHz to induce eddy currents in conductive targets, enabling "
        "detection of both ferrous and non-ferrous metals (including aluminum aircraft components that "
        "magnetometry cannot detect). Electrical Resistivity Tomography (ERT) injects small DC or "
        "low-frequency currents (0.5-2 mA) and measures voltage distribution, detecting resistivity "
        "contrasts from disturbed soil, moisture variations, voids, and grave shafts.",
        styles['BodyFirst']
    ))

    # MIT/ERT comparison table
    mit_ert_data = [
        ['Parameter', 'MIT-3D', 'ERT-Lite'],
        ['Operating Principle', 'TX coil magnetic field;\nRX measures eddy currents',
         'Current injection;\nvoltage measurement'],
        ['Frequency Range', '2-50 kHz', 'DC or 8-16 Hz'],
        ['Detection Targets', 'Metal (incl. aluminum),\nconductivity anomalies',
         'Disturbed fill, moisture,\nvoids, grave shafts'],
        ['Current Level', '10-50 mA', '0.5-2 mA'],
        ['Key Advantage', 'Non-ferrous metal detection', 'Soil disturbance patterns'],
        ['Resolution at 3m', '0.75-1.5 m lateral', '0.5-0.75 m vertical'],
    ]

    table = Table(mit_ert_data, colWidths=[CONTENT_WIDTH*0.25, CONTENT_WIDTH*0.375, CONTENT_WIDTH*0.375])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
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
    ]))
    table1_caption = Paragraph(
        "<b>Table 1.</b> Comparison of MIT and ERT measurement methods. The dual-channel approach "
        "enables detection of targets that would be missed by either method alone.",
        styles['Caption']
    )
    story.append(Spacer(1, 8))
    story.append(KeepTogether([table, table1_caption]))

    # === 3. SYSTEM ARCHITECTURE ===
    story.append(Paragraph("3. System Architecture", styles['Section']))
    story.append(Paragraph(
        "The HIRT system follows an 'archaeologist brain first, engineer brain second' design philosophy, "
        "prioritizing minimal ground disturbance while maintaining measurement quality. The architecture "
        "employs passive micro-probes with centralized electronics. To address scalability for large "
        "arrays (20-50 probes), the system utilizes a 'Zone Wiring Strategy' where groups of 4 probes "
        "connect to passive Zone Hubs, which then feed back to the main unit via high-density trunk cables. "
        "Figure 3 shows the overall architecture.",
        styles['BodyFirst']
    ))

    # Architecture figure
    fig3_img = Image(fig_architecture, width=CONTENT_WIDTH, height=CONTENT_WIDTH * 6/10)
    fig3_caption = Paragraph(
        "<b>Figure 3.</b> HIRT system architecture. Left: Central electronics hub connected to 'Zone Boxes' "
        "via high-density trunk cables. Each Zone Box aggregates 4 passive probes. Right: Detailed probe "
        "cross-section showing the flush 16mm OD profile, internal coils, and modular M12x1.75 threaded "
        "joints.",
        styles['Caption']
    )
    story.append(Spacer(1, 8))
    story.append(KeepTogether([fig3_img, fig3_caption]))

    story.append(Paragraph("3.1 Micro-Probe Design", styles['Subsection']))
    story.append(Paragraph(
        "Each probe is a 16mm outer diameter fiberglass rod containing only passive sensors. The modular design "
        "uses **flush M12x1.75 threaded connectors** allowing field assembly of segments without creating "
        "snag points during insertion. This smooth 16mm profile ensures the probe can be inserted and "
        "removed with minimal force (~10x less disturbance than traditional geophysical probes).",
        styles['BodyFirst']
    ))

    # Probe specifications table
    probe_data = [
        ['Specification', 'Value', 'Notes'],
        ['Rod Outer Diameter', '16 mm', '~10x less disturbance than traditional'],
        ['Required Hole Size', '18-20 mm', 'Hand auger or push-rod insertion'],
        ['Material', 'Fiberglass (G10)', 'Non-conductive, RF transparent'],
        ['Segment Lengths', '50, 100 cm', 'Modular field assembly'],
        ['Coil Inductance', '1-2 mH', '200-400 turns on ferrite core'],
        ['ERT Ring Positions', '0.5, 1.5, 2.5 m', 'From probe tip'],
        ['Weight', '~50-100 g/m', 'Lightweight for transport'],
    ]

    table = Table(probe_data, colWidths=[CONTENT_WIDTH*0.3, CONTENT_WIDTH*0.25, CONTENT_WIDTH*0.45])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
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
    ]))
    table2_caption = Paragraph(
        "<b>Table 2.</b> HIRT micro-probe physical specifications.",
        styles['Caption']
    )
    story.append(Spacer(1, 8))
    story.append(KeepTogether([table, table2_caption]))

    # === 4. PERFORMANCE COMPARISON ===
    story.append(Paragraph("4. Performance Comparison", styles['Section']))
    story.append(Paragraph(
        "The crosshole geometry provides significant performance advantages over surface methods, "
        "particularly for targets at depths exceeding 1.5-2 meters. Research on crosshole ERT and "
        "electromagnetic tomography demonstrates 2-5x better resolution than surface methods at these "
        "depths [6,7]. Figure 4 compares the effective depth of investigation across common geophysical "
        "methods.",
        styles['BodyFirst']
    ))

    # Depth comparison figure
    fig4_img = Image(fig_depth, width=CONTENT_WIDTH*0.95, height=CONTENT_WIDTH*0.95 * 4.5/8)
    fig4_caption = Paragraph(
        "<b>Figure 4.</b> Effective depth of investigation comparison across geophysical methods. "
        "Horizontal bars show the range from minimum (poor conditions) to maximum (optimal conditions) "
        "depth, with vertical markers indicating typical performance. The red shaded zone indicates "
        "typical UXB burial depths (2-4m). HIRT crosshole methods maintain high-confidence detection "
        "throughout this critical range, while surface methods show significant degradation.",
        styles['Caption']
    )
    story.append(Spacer(1, 8))
    story.append(KeepTogether([fig4_img, fig4_caption]))

    story.append(Paragraph("4.1 Cost-Effectiveness", styles['Subsection']))
    story.append(Paragraph(
        "A complete HIRT starter kit (25 probes, central electronics hub, cables, and tools) costs "
        "$1,800-3,900, representing a 95%+ reduction compared to commercial crosshole systems "
        "($50,000-200,000+). This cost structure places professional-grade subsurface imaging within "
        "reach of research institutions, humanitarian organizations, and archaeological teams that "
        "could not otherwise afford crosshole tomography capabilities.",
        styles['BodyFirst']
    ))

    # === 5. APPLICATION SCENARIOS ===
    story.append(Paragraph("5. Application Scenarios", styles['Section']))
    story.append(Paragraph(
        "HIRT is optimized for three primary deployment scenarios, each with specific configuration "
        "recommendations. The system's dual-channel MIT+ERT approach enables comprehensive target "
        "characterization across diverse site conditions. Figure 5 illustrates the deployment strategies.",
        styles['BodyFirst']
    ))

    # Scenarios figure
    fig5_img = Image(fig_scenarios, width=CONTENT_WIDTH, height=CONTENT_WIDTH * 4/11)
    fig5_caption = Paragraph(
        "<b>Figure 5.</b> Primary HIRT application scenarios: (a) Bomb crater investigation uses perimeter "
        "probes (3m depth, 1.5m spacing) around suspected UXB locations with dense ray path coverage through "
        "the crater fill; (b) woodland burial search uses shallow dense arrays (1.5m probes, 1m spacing) to "
        "detect disturbed soil signatures; (c) wetland surveys deploy probes from accessible margins with "
        "long-baseline diagonal ray paths toward deep targets. Green lines indicate MIT measurement paths.",
        styles['Caption']
    )
    story.append(Spacer(1, 8))
    story.append(KeepTogether([fig5_img, fig5_caption]))

    story.append(Paragraph(
        "<b>WWII Crash Sites and UXB Investigation:</b> For craters 10-15m diameter with targets at 2-4m "
        "depth, use 3.0m probes at 1.5-2.0m spacing in a 5x5 or 6x6 array. MIT frequencies should emphasize "
        "lower ranges (2-10 kHz) for depth penetration. The combined MIT+ERT approach detects both metallic "
        "debris (MIT) and the disturbed fill boundaries of the crater (ERT).",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>Woodland Burials and Forensic Sites:</b> For shallow targets (0.5-1.5m), use 1.5m probes at "
        "1.0-1.5m spacing. Higher MIT frequencies (10-50 kHz) provide better near-surface detail. ERT "
        "excels at detecting the disturbed grave shaft signature while MIT identifies metallic artifacts "
        "such as belt buckles, buttons, or personal effects.",
        styles['Body']
    ))
    story.append(Paragraph(
        "<b>Wetland and Difficult-Access Sites:</b> For targets exceeding 5m depth in areas with limited "
        "access, deploy probes from accessible margins. Use lowest MIT frequencies (2-5 kHz) with extended "
        "integration times (10-30 seconds). Diagonal ray paths from perimeter probes can interrogate "
        "central volumes that cannot be directly accessed.",
        styles['Body']
    ))

    # === 6. SAFETY CONSIDERATIONS ===
    story.append(Paragraph("6. Safety Considerations for UXO Sites", styles['Section']))

    # Safety warning box
    safety_data = [
        ['CRITICAL SAFETY REQUIREMENTS FOR UXO SITES'],
        ['*  Professional EOD clearance required before probe insertion'],
        ['*  No hammering or driving of pilot rods - soft insertion only'],
        ['*  Maintain 100m exclusion zone during insertion operations'],
        ['*  Perimeter-only deployment when UXB suspected at center'],
        ['*  Monitor groundwater conductivity: >5,500 uS/cm indicates elevated risk'],
    ]

    safety_table = Table(safety_data, colWidths=[CONTENT_WIDTH])
    safety_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), WARNING),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
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
    ]))
    story.append(Spacer(1, 8))
    story.append(safety_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Research by Waga et al. [3] documented a spontaneous UXB detonation in the Kozle Basin associated "
        "with elevated groundwater conductivity (>6,000 uS/cm) from industrial contamination. The "
        "electrochemical environment created by high-conductivity groundwater can accelerate corrosion of "
        "bomb casings and destabilize aging explosive fills. HIRT's ERT capability enables real-time "
        "monitoring of subsurface conductivity trends that may indicate elevated risk conditions.",
        styles['Body']
    ))

    # === 7. CONCLUSIONS ===
    story.append(Paragraph("7. Conclusions", styles['Section']))
    story.append(Paragraph(
        "The HIRT system addresses a critical gap between expensive commercial crosshole systems and "
        "limited-capability surface methods. By combining MIT and ERT sensing in a passive micro-probe "
        "architecture, HIRT achieves 2-5x better resolution than surface methods at depths exceeding "
        "2 meters, while maintaining costs accessible to research institutions and humanitarian organizations.",
        styles['BodyFirst']
    ))
    story.append(Paragraph(
        "The global unexploded ordnance crisis&mdash;affecting over 100 million people across 60+ "
        "countries&mdash;demands technological solutions that are both effective and deployable at scale. "
        "Traditional humanitarian demining methods clear only 10-20 square meters per day, a pace that "
        "would require centuries to address the existing backlog. HIRT offers a path toward more efficient "
        "detection and characterization of buried hazards, enabling prioritized clearance operations and "
        "reduced civilian casualties.",
        styles['Body']
    ))
    story.append(Paragraph(
        "Key advantages of the HIRT approach include: (1) true 3D tomographic imaging rather than "
        "pseudo-depth estimation; (2) detection of non-ferrous metals that magnetometry cannot sense; "
        "(3) soil disturbance mapping independent of metallic content; (4) minimal ground disturbance "
        "(~10x less than traditional methods); (5) modular, field-serviceable design suitable for remote "
        "deployment; and (6) cost reduction of 95%+ compared to commercial alternatives.",
        styles['Body']
    ))

    # === REFERENCES ===
    story.append(Paragraph("References", styles['Section']))

    references = [
        "[1] Landmine and Cluster Munition Monitor (2024). Annual Report: Global casualties from mines "
        "and explosive remnants of war 1999-2023. International Campaign to Ban Landmines.",

        "[2] Smithsonian Magazine (2016). There Are Still Thousands of Tons of Unexploded Bombs in "
        "Germany, Left Over From World War II. https://www.smithsonianmag.com/history/",

        "[3] Waga, J.M., Szypula, B., & Fajer, M. (2022). The Archaeology of Unexploded World War II "
        "Bomb Sites in the Kozle Basin, Southern Poland. International Journal of Historical Archaeology "
        "26: 1-23.",

        "[4] Auschwitz-Birkenau Memorial and Museum (2024). Blechhammer: Auschwitz IV Labor Camp. "
        "Historical documentation of the Blechhammer concentration camp complex.",

        "[5] Quod Library, University of Michigan (2023). Unexploded Ordnance (UXO) in Laos: The Legacy "
        "of the Secret War. Technology and Policy Analysis.",

        "[6] Butler, D.K. (2001). Potential fields methods for location of unexploded ordnance. "
        "The Leading Edge 20(8): 890-895.",

        "[7] Fernandez, J.P., et al. (2010). Realistic Subsurface Anomaly Discrimination Using "
        "Electromagnetic Induction and an SVM Classifier. EURASIP Journal on Advances in Signal Processing.",
    ]

    for ref in references:
        story.append(Paragraph(ref, styles['Reference']))

    # Build PDF
    doc.build(story)
    print("PDF created: HIRT_Introduction.pdf")


if __name__ == "__main__":
    build_document()
