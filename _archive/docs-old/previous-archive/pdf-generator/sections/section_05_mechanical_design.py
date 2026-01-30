#!/usr/bin/env python3
"""
HIRT Section 05: Mechanical Design - Publication-Quality PDF Generator

Generates a comprehensive PDF document covering mechanical aspects of the HIRT
probe system including probe construction, threading, coil mounting, and materials.

Usage:
    python section_05_mechanical_design.py

Output:
    output/sections/05-mechanical-design.pdf
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, Arc, Wedge,
    FancyArrowPatch, Ellipse, PathPatch
)
import numpy as np
from io import BytesIO

from lib.pdf_builder import SectionPDFBuilder
from lib.styles import (
    CONTENT_WIDTH, PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING,
    LIGHT_BG, ORANGE, GRAY_DARK, GRAY_MED, GRAY_LIGHT, COLOR_PALETTE
)
from lib.diagrams.mechanical import (
    create_probe_cross_section,
    create_exploded_assembly,
    create_thread_detail,
    create_coil_mounting_detail,
    create_disturbance_comparison,
    create_ert_ring_detail,
    COLORS
)


# ============================================================================
# ADDITIONAL DIAGRAM FUNCTIONS
# ============================================================================

def draw_dimension(ax, start, end, offset, text, fontsize=8, horizontal=True):
    """Draw a dimension line with arrows and centered text."""
    if horizontal:
        y = start[1] + offset
        ax.plot([start[0], start[0]], [start[1], y], 'k-', lw=0.5)
        ax.plot([end[0], end[0]], [end[1], y], 'k-', lw=0.5)
        ax.annotate('', xy=(end[0], y), xytext=(start[0], y),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
        ax.text((start[0] + end[0])/2, y + 0.05, text, ha='center',
               fontsize=fontsize, fontweight='bold')
    else:
        x = start[0] + offset
        ax.plot([start[0], x], [start[1], start[1]], 'k-', lw=0.5)
        ax.plot([end[0], x], [end[1], end[1]], 'k-', lw=0.5)
        ax.annotate('', xy=(x, end[1]), xytext=(x, start[1]),
                   arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
        ax.text(x + 0.1, (start[1] + end[1])/2, text, va='center',
               fontsize=fontsize, fontweight='bold', rotation=90)


def create_material_specs_diagram():
    """
    Create material specifications comparison diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.5, 'Material Selection Guide', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Materials data
    materials = [
        ('Fiberglass (G10)', 'Preferred', 7, [
            'Non-conductive', 'RF transparent', 'High strength',
            'Moderate cost', 'Excellent durability'
        ], COLORS['success']),
        ('Carbon Fiber', 'Premium', 5.2, [
            'Very high strength', 'Lightweight',
            'Conductive (!)', 'Expensive'
        ], COLORS['accent']),
        ('PVC', 'Budget', 3.4, [
            'Low cost', 'Easy to machine',
            'Lower strength', 'Shallow use only'
        ], COLORS['orange']),
        ('Metal (Al/Steel)', 'Avoid', 1.6, [
            'Strong', 'Durable',
            'EM interference', 'Not recommended'
        ], COLORS['warning']),
    ]

    for name, rec, y, props, color in materials:
        # Material box
        ax.add_patch(FancyBboxPatch((0.5, y - 0.5), 2.5, 1.0,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='black',
                                    lw=1.5, alpha=0.3))
        ax.text(1.75, y, name, ha='center', va='center',
               fontsize=10, fontweight='bold', color=COLORS['primary'])

        # Recommendation badge
        badge_color = COLORS['success'] if rec == 'Preferred' else (
            COLORS['accent'] if rec == 'Premium' else (
            COLORS['orange'] if rec == 'Budget' else COLORS['warning']))
        ax.add_patch(FancyBboxPatch((3.2, y - 0.25), 1.2, 0.5,
                                    boxstyle="round,pad=0.02",
                                    facecolor=badge_color, edgecolor='black',
                                    lw=1, alpha=0.8))
        ax.text(3.8, y, rec, ha='center', va='center',
               fontsize=8, fontweight='bold', color='white')

        # Properties
        for i, prop in enumerate(props):
            x_pos = 5.0 + (i % 3) * 1.8
            y_off = 0.15 if i < 3 else -0.25
            ax.text(x_pos, y + y_off, f'- {prop}', fontsize=7,
                   va='center', color=COLORS['gray_dark'])

    # Legend
    ax.text(0.5, 0.8, 'Legend:', fontsize=9, fontweight='bold')
    for i, (label, color) in enumerate([
        ('Preferred', COLORS['success']),
        ('Premium', COLORS['accent']),
        ('Budget', COLORS['orange']),
        ('Avoid', COLORS['warning'])
    ]):
        ax.add_patch(Rectangle((1.8 + i*2, 0.5), 0.3, 0.3,
                               facecolor=color, edgecolor='black', lw=0.5))
        ax.text(2.2 + i*2, 0.65, label, fontsize=7, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_modular_segment_connection():
    """
    Create modular segment connection detail diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(4, 7.5, 'Modular Flush-Mount Connector System', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    # === Left side: Assembled view ===
    ax.text(1.5, 6.8, '(a) Assembled', ha='center', fontsize=10, fontweight='bold')

    # Upper rod segment
    ax.add_patch(Rectangle((0.7, 4.5), 1.6, 2.0,
                           facecolor=COLORS['fiberglass'], edgecolor='black', lw=1.5))
    ax.text(1.5, 5.5, 'Rod\nSegment 1', ha='center', va='center', fontsize=8)

    # Connection zone (flush)
    ax.add_patch(Rectangle((0.7, 3.8), 1.6, 0.7,
                           facecolor=COLORS['gray_light'], edgecolor='black', lw=1))
    ax.text(1.5, 4.15, 'Flush\nJoint', ha='center', va='center', fontsize=7)

    # Lower rod segment
    ax.add_patch(Rectangle((0.7, 1.8), 1.6, 2.0,
                           facecolor=COLORS['fiberglass'], edgecolor='black', lw=1.5))
    ax.text(1.5, 2.8, 'Rod\nSegment 2', ha='center', va='center', fontsize=8)

    # OD dimension
    draw_dimension(ax, (0.7, 1.5), (2.3, 1.5), -0.3, '16mm OD')

    # Flush callout
    ax.annotate('No snag points\n(flush profile)', xy=(2.3, 4.15),
               xytext=(3.5, 4.15), fontsize=8, ha='left',
               arrowprops=dict(arrowstyle='->', color=COLORS['success']))

    # === Right side: Exploded view ===
    ax.text(6.5, 6.8, '(b) Exploded', ha='center', fontsize=10, fontweight='bold')

    # Upper rod with male insert
    ax.add_patch(Rectangle((5.5, 5.5), 2.0, 1.0,
                           facecolor=COLORS['fiberglass'], edgecolor='black', lw=1.5))
    ax.add_patch(Rectangle((6.0, 4.7), 1.0, 0.8,
                           facecolor=COLORS['gray_med'], edgecolor='black', lw=1))
    ax.text(6.5, 6.0, 'Rod 1', ha='center', va='center', fontsize=8)
    ax.text(6.5, 5.1, 'Male\nInsert', ha='center', va='center', fontsize=7,
           color='white')

    # Thread visualization on male
    for i in range(4):
        y = 4.75 + i * 0.15
        ax.plot([6.95, 7.05, 6.95], [y, y + 0.075, y + 0.15],
               'k-', lw=0.8)

    # Arrow showing connection
    ax.annotate('', xy=(6.5, 4.0), xytext=(6.5, 4.6),
               arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=2))
    ax.text(7.3, 4.3, 'M12x1.75\nThread', fontsize=8, va='center')

    # Lower rod with female insert
    ax.add_patch(Rectangle((5.5, 1.8), 2.0, 1.5,
                           facecolor=COLORS['fiberglass'], edgecolor='black', lw=1.5))
    ax.add_patch(Rectangle((6.0, 3.3), 1.0, 0.5,
                           facecolor=COLORS['gray_med'], edgecolor='black', lw=1))
    # Hollow center for female
    ax.add_patch(Rectangle((6.15, 3.35), 0.7, 0.45,
                           facecolor='white', edgecolor='black', lw=0.5))
    ax.text(6.5, 2.5, 'Rod 2', ha='center', va='center', fontsize=8)
    ax.text(6.5, 3.55, 'Female', ha='center', va='center', fontsize=6)

    # Wiring channel
    ax.add_patch(Circle((6.5, 5.0), 0.15, facecolor='white',
                       edgecolor=COLORS['copper'], lw=1.5))
    ax.add_patch(Circle((6.5, 2.5), 0.15, facecolor='white',
                       edgecolor=COLORS['copper'], lw=1.5))
    ax.annotate('6mm wiring\nchannel', xy=(6.5, 2.5), xytext=(8.0, 2.0),
               fontsize=7, ha='left',
               arrowprops=dict(arrowstyle='->', color=COLORS['copper']))

    # Specifications box
    specs_text = [
        'Thread: M12x1.75 ISO',
        'Engagement: 12-15mm',
        'Male OD: 12.2mm (print)',
        'Female ID: 10.5mm (print)',
        'Post-process: Tap/Die'
    ]
    ax.add_patch(FancyBboxPatch((0.3, -0.5), 4.0, 1.3, boxstyle="round,pad=0.05",
                                facecolor=COLORS['light_bg'] if 'light_bg' in COLORS else '#f7fafc',
                                edgecolor=COLORS['gray_med'], lw=1))
    ax.text(2.3, 0.6, 'Specifications:', fontsize=8, fontweight='bold',
           ha='center', color=COLORS['primary'])
    for i, spec in enumerate(specs_text):
        ax.text(0.5, 0.3 - i*0.2, f'- {spec}', fontsize=7)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_junction_box_diagram():
    """
    Create junction box design detail diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # === Left: External view ===
    ax1.set_xlim(-1.5, 3.5)
    ax1.set_ylim(-1, 4)
    ax1.set_aspect('equal')
    ax1.axis('off')

    ax1.text(1, 3.7, '(a) External View', ha='center', fontsize=10, fontweight='bold')

    # Cable gland
    ax1.add_patch(FancyBboxPatch((0.6, 2.8), 0.8, 0.5, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['gray_dark'], edgecolor='black', lw=1))
    ax1.text(1, 3.05, 'Cable\nGland', ha='center', va='center',
            fontsize=6, color='white')

    # Main body
    ax1.add_patch(FancyBboxPatch((0.2, 1.0), 1.6, 1.8, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['gray_dark'], edgecolor='black', lw=1.5))
    ax1.text(1, 1.9, 'Junction\nBox Body\n(PETG)', ha='center', va='center',
            fontsize=8, color='white')

    # Thread region
    ax1.add_patch(Rectangle((0.6, 0.5), 0.8, 0.5,
                            facecolor=COLORS['gray_med'], edgecolor='black', lw=1))
    ax1.text(1, 0.75, 'Thread', ha='center', va='center', fontsize=7)

    # Rod connection
    ax1.add_patch(Rectangle((0.6, -0.3), 0.8, 0.8,
                            facecolor=COLORS['fiberglass'], edgecolor='black', lw=1))
    ax1.text(1, 0.1, 'Rod', ha='center', va='center', fontsize=7)

    # Dimensions
    draw_dimension(ax1, (0.2, 0.3), (1.8, 0.3), -0.4, '25mm')
    draw_dimension(ax1, (2.0, 1.0), (2.0, 2.8), 0.3, '35mm', horizontal=False)

    # === Right: Cutaway view ===
    ax2.set_xlim(-1.5, 4.5)
    ax2.set_ylim(-1, 4)
    ax2.set_aspect('equal')
    ax2.axis('off')

    ax2.text(1.5, 3.7, '(b) Internal Cutaway', ha='center', fontsize=10, fontweight='bold')

    # Outer shell (cutaway)
    ax2.add_patch(Rectangle((0, 0.8), 0.2, 2.2,
                            facecolor=COLORS['gray_dark'], edgecolor='black', lw=1))
    ax2.add_patch(Rectangle((2.6, 0.8), 0.2, 2.2,
                            facecolor=COLORS['gray_dark'], edgecolor='black', lw=1))
    ax2.add_patch(Rectangle((0, 2.8), 2.8, 0.2,
                            facecolor=COLORS['gray_dark'], edgecolor='black', lw=1))

    # Internal cavity
    ax2.add_patch(Rectangle((0.2, 0.8), 2.4, 2.0,
                            facecolor='white', edgecolor=COLORS['gray_light'], lw=0.5))

    # Terminal block
    ax2.add_patch(FancyBboxPatch((0.5, 1.5), 1.8, 1.0, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['accent'], edgecolor='black', lw=1))
    ax2.text(1.4, 2.0, 'Terminal\nBlock', ha='center', va='center',
            fontsize=8, color='white')

    # Connection points on terminal block
    for x in [0.7, 1.1, 1.5, 1.9]:
        ax2.add_patch(Circle((x, 1.7), 0.08, facecolor=COLORS['steel'],
                            edgecolor='black', lw=0.5))

    # Wire routing
    ax2.plot([1.4, 1.4], [1.5, 0.5], color=COLORS['copper'], lw=2)

    # Thread
    ax2.add_patch(Rectangle((0.8, 0.3), 1.2, 0.5,
                            facecolor=COLORS['gray_light'], edgecolor='black', lw=0.5))
    for i in range(4):
        y = 0.35 + i * 0.1
        ax2.plot([0.8, 0.7, 0.8], [y, y + 0.05, y + 0.1], 'k-', lw=0.5)
        ax2.plot([2.0, 2.1, 2.0], [y, y + 0.05, y + 0.1], 'k-', lw=0.5)

    # Labels
    ax2.annotate('MIT coil leads (2x2)', xy=(0.7, 1.7), xytext=(3.2, 2.5),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))
    ax2.annotate('ERT electrode leads (3-4)', xy=(1.5, 1.7), xytext=(3.2, 1.8),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))
    ax2.annotate('M12x1.75\nInternal', xy=(1.4, 0.55), xytext=(3.2, 0.5),
                fontsize=7, ha='left',
                arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

    # Key features box
    ax2.add_patch(FancyBboxPatch((3.0, -0.5), 1.3, 0.8, boxstyle="round,pad=0.03",
                                 facecolor='#e8f4f8', edgecolor=COLORS['accent'], lw=1))
    ax2.text(3.65, 0.1, 'Key Features:', fontsize=7, fontweight='bold', ha='center')
    ax2.text(3.65, -0.1, 'Passive only', fontsize=6, ha='center')
    ax2.text(3.65, -0.25, 'Weatherproof', fontsize=6, ha='center')
    ax2.text(3.65, -0.4, 'Field serviceable', fontsize=6, ha='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_tool_requirements_diagram():
    """
    Create tool requirements diagram for probe assembly.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.5, 'Tool Requirements for Probe Assembly', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Categories
    categories = [
        ('Thread Cutting', 5.2, [
            ('M12x1.75 Tap', 'Female threads'),
            ('M12x1.75 Die', 'Male threads'),
            ('Tap wrench', 'T-handle style'),
            ('Cutting oil', 'Reduces friction'),
        ], COLORS['success']),
        ('Assembly', 3.4, [
            ('Epoxy (2-part)', 'Bonding inserts'),
            ('CA glue (thin)', 'Thread hardening'),
            ('Heat shrink', 'Wire protection'),
            ('Multimeter', 'Continuity check'),
        ], COLORS['accent']),
        ('Finishing', 1.6, [
            ('Flush cutters', 'Brim removal'),
            ('Fine sandpaper', '220-400 grit'),
            ('Deburring tool', 'Thread cleanup'),
            ('Calipers', 'Dimension check'),
        ], COLORS['orange']),
    ]

    for cat_name, y_base, tools, color in categories:
        # Category header
        ax.add_patch(FancyBboxPatch((0.3, y_base + 0.3), 2.2, 0.5,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', lw=1))
        ax.text(1.4, y_base + 0.55, cat_name, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

        # Tools
        for i, (tool, desc) in enumerate(tools):
            x = 2.8 + i * 1.8
            ax.add_patch(Rectangle((x, y_base + 0.1), 1.6, 0.7,
                                   facecolor='white', edgecolor=color, lw=1))
            ax.text(x + 0.8, y_base + 0.55, tool, ha='center', va='center',
                   fontsize=8, fontweight='bold')
            ax.text(x + 0.8, y_base + 0.25, desc, ha='center', va='center',
                   fontsize=6, color=COLORS['gray_dark'])

    # 3D Printing section
    ax.add_patch(FancyBboxPatch((0.3, 0.2), 9.4, 1.0, boxstyle="round,pad=0.03",
                                facecolor='#e8f4f8', edgecolor=COLORS['primary'], lw=1.5))
    ax.text(5, 1.0, '3D Printer Settings (Bambu Lab / FDM)', ha='center',
           fontsize=9, fontweight='bold', color=COLORS['primary'])

    settings = [
        'Material: PETG/ASA',
        'Layer: 0.12mm',
        'Infill: 100%',
        'Walls: 6 loops',
        'Supports: OFF',
        'Speed: 50mm/s (outer)'
    ]
    for i, setting in enumerate(settings):
        ax.text(0.8 + i * 1.55, 0.55, setting, fontsize=7, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_assembly_sequence_diagram():
    """
    Create assembly sequence diagram showing bottom-to-top order.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(6.67, 10))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(2, 10.5, 'Assembly Sequence (Bottom to Top)', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    # Components from bottom to top with step numbers
    components = [
        (0.5, 'Probe Tip', COLORS['primary'], '(pointed, screws on)'),
        (2.0, 'Rod Section 1', COLORS['fiberglass'], '(1.5m with RX coil)'),
        (3.5, 'ERT Ring #1', COLORS['orange'], '(epoxy bond at 0.5m)'),
        (5.0, 'Coupler', COLORS['gray_light'], '(M12 threads both ends)'),
        (6.5, 'Rod Section 2', COLORS['fiberglass'], '(1.5m with TX coil)'),
        (8.0, 'ERT Ring #2 & #3', COLORS['orange'], '(at 1.5m, 2.5m)'),
        (9.5, 'Junction Box', COLORS['gray_dark'], '(at surface, passive)'),
    ]

    for i, (y, name, color, desc) in enumerate(components):
        # Step number
        ax.add_patch(Circle((-0.5, y), 0.3, facecolor=COLORS['secondary'],
                           edgecolor='black', lw=1))
        ax.text(-0.5, y, str(i+1), ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

        # Component bar
        ax.add_patch(FancyBboxPatch((0.3, y - 0.3), 2.0, 0.6, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', lw=1.5))
        ax.text(1.3, y, name, ha='center', va='center',
               fontsize=9, fontweight='bold',
               color='white' if color in [COLORS['primary'], COLORS['gray_dark'], COLORS['secondary']] else 'black')

        # Description
        ax.text(2.6, y, desc, fontsize=8, va='center', color=COLORS['gray_dark'])

        # Connecting arrow (except last)
        if i < len(components) - 1:
            ax.annotate('', xy=(1.3, y + 0.6), xytext=(1.3, y + 1.1),
                       arrowprops=dict(arrowstyle='->', color=COLORS['gray_med'], lw=1.5))

    # Ground level indicator
    ax.axhline(9.0, xmin=0.1, xmax=0.7, color='#654321', lw=3)
    ax.text(-0.8, 9.0, 'Ground\nLevel', fontsize=8, ha='center', va='center',
           color='#654321')

    # Key points box
    ax.add_patch(FancyBboxPatch((-1.8, -0.5), 7.4, 1.2, boxstyle="round,pad=0.03",
                                facecolor='#fff5f5', edgecolor=COLORS['warning'], lw=1))
    ax.text(1.9, 0.5, 'Key Points:', fontsize=9, fontweight='bold',
           color=COLORS['warning'])
    ax.text(1.9, 0.15, '- All parts screw together (modular)   - Tip points DOWN   - Wires route through center',
           fontsize=7, color=COLORS['gray_dark'])
    ax.text(1.9, -0.15, '- Junction box stays at surface       - ERT collars bond with epoxy',
           fontsize=7, color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_insertion_methods_diagram():
    """
    Create insertion methods comparison diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    methods = [
        ('(a) Hand Auger', axes[0, 0], 'Best for most soils'),
        ('(b) Pilot Rod', axes[0, 1], 'Clay/compact soils'),
        ('(c) Direct Push', axes[1, 0], 'Sandy soils only'),
        ('(d) Water-Jet', axes[1, 1], 'Excellent for sand'),
    ]

    for title, ax, desc in methods:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-3, 1)
        ax.set_aspect('equal')

        # Ground
        ax.axhline(0, color='#654321', lw=2)
        ax.fill_between([-2, 2], [0, 0], [-3, -3], color='#d4a373', alpha=0.4)

        ax.set_title(title, fontsize=10, fontweight='bold', color=COLORS['primary'])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.text(0, -2.8, desc, ha='center', fontsize=8, style='italic',
               color=COLORS['gray_dark'])

    # (a) Hand Auger
    ax = axes[0, 0]
    # Auger tool
    ax.add_patch(Rectangle((-0.15, -1.5), 0.3, 2.0,
                           facecolor=COLORS['gray_med'], edgecolor='black', lw=1))
    ax.add_patch(Circle((0, -1.5), 0.3, facecolor=COLORS['gray_dark'],
                       edgecolor='black', lw=1))
    # Spiral suggestion
    for i in range(3):
        y = -0.5 - i * 0.3
        ax.plot([-0.15, -0.3, -0.15], [y, y - 0.1, y - 0.2],
               color=COLORS['gray_dark'], lw=1)
    # Hole
    ax.add_patch(Rectangle((-0.25, -2.2), 0.5, 0.7,
                           facecolor='white', edgecolor=COLORS['gray_light'], lw=1, alpha=0.5))
    ax.text(1.2, -0.8, '10-20mm\nauger', fontsize=7, va='center')

    # (b) Pilot Rod
    ax = axes[0, 1]
    # Steel rod
    ax.add_patch(Rectangle((-0.1, -2.0), 0.2, 2.5,
                           facecolor=COLORS['steel'], edgecolor='black', lw=1))
    ax.text(0, 0.7, 'Steel', ha='center', fontsize=7, color=COLORS['steel'])
    # Warning
    ax.add_patch(FancyBboxPatch((0.5, -1.5), 1.3, 0.5, boxstyle="round,pad=0.02",
                                facecolor='#fff5f5', edgecolor=COLORS['warning'], lw=1))
    ax.text(1.15, -1.25, 'Remove\nbefore survey!', ha='center', fontsize=6,
           color=COLORS['warning'])
    ax.annotate('', xy=(0.15, -1.5), xytext=(0.5, -1.25),
               arrowprops=dict(arrowstyle='->', color=COLORS['warning']))

    # (c) Direct Push
    ax = axes[1, 0]
    # Probe
    ax.add_patch(Rectangle((-0.12, -1.8), 0.24, 2.0,
                           facecolor=COLORS['secondary'], edgecolor='black', lw=1))
    # Tip
    tip_points = [(-0.12, -1.8), (0.12, -1.8), (0, -2.1)]
    ax.add_patch(Polygon(tip_points, facecolor=COLORS['primary'],
                        edgecolor='black', lw=1))
    # Force arrow
    ax.annotate('', xy=(0, 0), xytext=(0, 0.6),
               arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))
    ax.text(0, 0.75, 'Push', ha='center', fontsize=8, fontweight='bold',
           color=COLORS['success'])
    ax.text(0.8, -1.0, 'Sand/loam\nonly', fontsize=7, va='center')

    # (d) Water-Jet
    ax = axes[1, 1]
    # Water lance
    ax.add_patch(Rectangle((-0.08, -1.5), 0.16, 1.8,
                           facecolor=COLORS['accent'], edgecolor='black', lw=1))
    # Water spray effect
    for angle in [-30, 0, 30]:
        rad = np.radians(angle - 90)
        dx = 0.5 * np.cos(rad)
        dy = 0.5 * np.sin(rad)
        ax.annotate('', xy=(dx, -1.5 + dy), xytext=(0, -1.5),
                   arrowprops=dict(arrowstyle='->', color='#90cdf4', lw=1))
    # Fluidized zone
    ax.add_patch(Ellipse((0, -2.0), 0.6, 0.8,
                        facecolor='#90cdf4', edgecolor='none', alpha=0.4))
    ax.text(1.0, -1.8, 'Fluidized\nsand', fontsize=7, va='center',
           color=COLORS['accent'])
    ax.text(-1.5, -1.0, 'Water\nlance', fontsize=7, va='center')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_advantages_diagram():
    """
    Create micro-probe advantages infographic.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.5, 'Micro-Probe Design Advantages', ha='center',
           fontsize=12, fontweight='bold', color=COLORS['primary'])

    # Central probe image
    ax.add_patch(Rectangle((4.5, 2), 1.0, 4.5,
                           facecolor=COLORS['secondary'], edgecolor='black', lw=2))
    ax.add_patch(Polygon([(4.5, 2), (5.5, 2), (5, 1.5)],
                        facecolor=COLORS['primary'], edgecolor='black', lw=2))
    ax.text(5, 3.5, '16mm\nProbe', ha='center', va='center',
           fontsize=9, fontweight='bold', color='white', rotation=90)

    # Advantage callouts arranged around the probe
    advantages = [
        (1.5, 6.5, '10x Less Disturbance', '~3 cm^2 vs 20 cm^2 hole', COLORS['success']),
        (8.5, 6.5, 'Modular Design', 'Field-serviceable segments', COLORS['accent']),
        (1.5, 5.0, '50% Lighter', '50-100 g/m vs 200+ g/m', COLORS['success']),
        (8.5, 5.0, 'M12 Threads', 'Standard, robust connection', COLORS['accent']),
        (1.5, 3.5, 'Passive Probes', 'No electronics downhole', COLORS['success']),
        (8.5, 3.5, 'Flush Profile', 'No snag points', COLORS['accent']),
        (1.5, 2.0, '70% Cost Reduction', '$40-60 vs $130-180', COLORS['success']),
        (8.5, 2.0, 'Better Reliability', 'Simpler = more robust', COLORS['accent']),
    ]

    for x, y, title, desc, color in advantages:
        # Connector line
        probe_x = 4.5 if x < 5 else 5.5
        ax.plot([x, probe_x], [y, y], color=color, lw=1, alpha=0.5)

        # Advantage box
        ax.add_patch(FancyBboxPatch((x - 1.3, y - 0.4), 2.6, 0.8,
                                    boxstyle="round,pad=0.03",
                                    facecolor=color, edgecolor='black', lw=1, alpha=0.2))
        ax.add_patch(FancyBboxPatch((x - 1.3, y - 0.4), 2.6, 0.8,
                                    boxstyle="round,pad=0.03",
                                    facecolor='none', edgecolor=color, lw=1.5))
        ax.text(x, y + 0.1, title, ha='center', va='center',
               fontsize=8, fontweight='bold', color=COLORS['primary'])
        ax.text(x, y - 0.2, desc, ha='center', va='center',
               fontsize=6, color=COLORS['gray_dark'])

    # Summary box at bottom
    ax.add_patch(FancyBboxPatch((0.5, 0.2), 9.0, 0.9, boxstyle="round,pad=0.03",
                                facecolor=COLORS['primary'], edgecolor='black', lw=2, alpha=0.1))
    ax.text(5, 0.65, 'Result: Professional-grade subsurface imaging at 95% lower cost',
           ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# MAIN DOCUMENT BUILDER
# ============================================================================

def main():
    """Generate the Section 05: Mechanical Design PDF."""

    print("Generating Section 05: Mechanical Design PDF...")
    print("=" * 60)

    # Initialize builder
    builder = SectionPDFBuilder(
        section_num=5,
        title="Mechanical Design"
    )

    # === TITLE BLOCK ===
    builder.add_title_block(
        subtitle="Probe Construction, Threading, Coil Mounting, and Materials"
    )

    # === OVERVIEW ===
    builder.add_section_header("5.1 Overview", level=1)
    builder.add_body_text(
        "This section consolidates all mechanical design specifications for the HIRT probe "
        "system, including the micro-probe architecture, rod specifications, component drawings, "
        "and manufacturing procedures for 3D printed parts. The design philosophy prioritizes "
        "minimal ground disturbance while maintaining robust measurement capability.",
        first_paragraph=True
    )

    # === DESIGN PHILOSOPHY ===
    builder.add_section_header("5.2 Design Philosophy", level=1)
    builder.add_body_text(
        "<b>\"Archaeologist brain first, engineer brain second\"</b> - This guiding principle "
        "shapes every aspect of the HIRT mechanical design. The goal is to create the smallest "
        "possible hole while retaining enough physics (coil area, electrode contact) for quality "
        "signals. The constraint leads to many thin, gentle holes rather than few large ones, "
        "resulting in approximately 10x less disturbance than conventional designs.",
        first_paragraph=True
    )

    builder.add_section_header("5.2.1 Core Design Principles", level=2)
    builder.add_bullet_list([
        "<b>No big electronics at the tip:</b> PCBs, ADCs, MCUs stay at surface. Only passive components downhole.",
        "<b>Probes are mostly passive:</b> Downhole contains only coils and electrodes plus thin wiring. "
        "All 'smart' electronics reside in junction box above ground.",
        "<b>Rod diameter standard: 16mm OD:</b> Robust hiking pole standard allowing strong M12 threads. "
        "Hole size of 18-20mm is much less destructive than traditional 50mm."
    ])

    # Figure 1: Probe Assembly Exploded View
    print("  Generating Figure 1: Probe Assembly Exploded View...")
    fig1_buf = create_exploded_assembly()
    builder.add_figure(
        fig1_buf,
        "Probe assembly exploded view showing the modular construction with M12x1.75 threaded "
        "connections. Components from bottom: probe tip, rod segments with integrated coils, "
        "ERT ring collars, and surface junction box. All parts screw together for field assembly.",
        height=CONTENT_WIDTH * 0.75
    )

    # === DISTURBANCE COMPARISON ===
    builder.add_section_header("5.2.2 Disturbance Comparison", level=2)
    builder.add_body_text(
        "The micro-probe design achieves dramatic reduction in soil disturbance compared to "
        "traditional geophysical probes. A 16mm rod requires only an 18-20mm hole, with "
        "cross-sectional area of approximately 2.5-3.0 cm^2. At 3m depth, this creates only "
        "0.75-1.0 liters of displacement per hole. In contrast, traditional 25mm rods requiring "
        "50mm holes displace approximately 6 liters per hole - a 7-10x increase in disturbance.",
        first_paragraph=True
    )

    # Figure 2: Disturbance Comparison
    print("  Generating Figure 2: Disturbance Comparison...")
    fig2_buf = create_disturbance_comparison()
    builder.add_figure(
        fig2_buf,
        "Soil disturbance comparison between HIRT micro-probe (16mm) and traditional probe (25mm). "
        "The dashed lines indicate the disturbed zone extent. The 16mm design achieves approximately "
        "60% less soil disturbance, critical for archaeological site preservation.",
        height=CONTENT_WIDTH * 0.45
    )

    # === KEY DIMENSIONS ===
    builder.add_section_header("5.3 Key Dimensions", level=1)

    # Dimensions table
    dimensions_data = [
        ['Component', 'Dimension', 'Notes'],
        ['Rod OD', '16 mm', 'Increased from 12mm for strength'],
        ['Rod ID', '12-13 mm', 'Standard pultruded tube'],
        ['Wall Thickness', '1.5-2.0 mm', 'Structural requirement'],
        ['Segment Length', '50, 100 cm', 'Defines sensor spacing'],
        ['Hole Size', '18-20 mm', 'For 16mm rod insertion'],
        ['Total Length', '2.5-3.0 m', 'Multiple segments assembled'],
    ]
    builder.add_table(
        dimensions_data,
        col_widths=[CONTENT_WIDTH*0.3, CONTENT_WIDTH*0.25, CONTENT_WIDTH*0.45],
        caption="Key dimensional specifications for the HIRT micro-probe system."
    )

    # === ROD SEGMENTS AND COUPLERS ===
    builder.add_section_header("5.4 Rod Segments and Couplers", level=1)

    builder.add_section_header("5.4.1 Rod Material Selection", level=2)
    builder.add_body_text(
        "Material selection is critical for probe performance. The rod must be non-conductive "
        "to avoid interference with electromagnetic measurements, strong enough to withstand "
        "insertion forces, and durable for repeated field use. Fiberglass (G10) emerges as the "
        "preferred material due to its combination of high strength, non-conductivity, RF "
        "transparency, and moderate cost.",
        first_paragraph=True
    )

    # Figure 3: Material Specifications
    print("  Generating Figure 3: Material Specifications...")
    fig3_buf = create_material_specs_diagram()
    builder.add_figure(
        fig3_buf,
        "Material selection guide for probe rod construction. Fiberglass (G10) is the preferred "
        "material offering the best balance of strength, non-conductivity, and cost. Metal rods "
        "must be avoided as they interfere with electromagnetic measurements.",
        height=CONTENT_WIDTH * 0.55
    )

    # Fiberglass specs table
    fg_data = [
        ['Parameter', 'Specification'],
        ['Outer Diameter (OD)', '16 mm (approx. 5/8")'],
        ['Inner Diameter (ID)', '12-13 mm'],
        ['Wall Thickness', '~1.5-2.0 mm'],
        ['Material', 'Fiberglass (non-conductive, RF transparent)'],
        ['Segment Lengths', '50 cm, 100 cm (spacers)'],
    ]
    builder.add_table(
        fg_data,
        col_widths=[CONTENT_WIDTH*0.4, CONTENT_WIDTH*0.6],
        caption="Fiberglass rod specifications."
    )

    # === MODULAR DESIGN ===
    builder.add_section_header("5.4.2 Modular Connector System", level=2)
    builder.add_body_text(
        "The system uses a 2-part connector system permanently epoxied into rod ends to create "
        "a screw-together stack. The male insert provides the threaded portion while the female "
        "insert/sensor module receives the thread and houses sensors. This flush-mount design "
        "ensures a smooth 16mm OD profile throughout, preventing snag points during insertion "
        "and extraction.",
        first_paragraph=True
    )

    # Figure 4: Modular Segment Connection
    print("  Generating Figure 4: Modular Segment Connection...")
    fig4_buf = create_modular_segment_connection()
    builder.add_figure(
        fig4_buf,
        "Modular flush-mount connector system showing (a) assembled view with smooth profile "
        "and (b) exploded view revealing male/female insert architecture. The 6mm central "
        "channel accommodates wiring. M12x1.75 threads provide robust mechanical connection.",
        height=CONTENT_WIDTH * 0.65
    )

    # === THREAD SPECIFICATIONS ===
    builder.add_section_header("5.4.3 Thread Specifications", level=2)
    builder.add_body_text(
        "All threaded connections use M12x1.75 ISO standard metric threads. This provides "
        "adequate strength for field assembly while remaining manufacturable via 3D printing "
        "with post-processing. The 'chunky' thread profile improves printability while "
        "maintaining engagement strength.",
        first_paragraph=True
    )

    # Figure 5: Thread Profile Detail
    print("  Generating Figure 5: Thread Profile Detail...")
    fig5_buf = create_thread_detail()
    builder.add_figure(
        fig5_buf,
        "M12x1.75 thread profile detail showing (a) male thread external profile and "
        "(b) female thread cross-section. Pitch of 1.75mm provides good engagement with "
        "12-15mm thread length. Male threads print at 12.2mm OD for die cutting; female "
        "threads print at 10.5mm ID for tapping.",
        height=CONTENT_WIDTH * 0.45
    )

    # Thread specs table
    thread_data = [
        ['Parameter', 'Value'],
        ['Thread Type', 'M12x1.75 ISO Standard'],
        ['Pitch', '1.75 mm'],
        ['Major Diameter', '12.0 mm'],
        ['Engagement Length', '12-15 mm'],
        ['Print Oversize (Male)', '12.2 mm (for Die cutting)'],
        ['Print Undersize (Female)', '10.5 mm hole (for Tapping)'],
    ]
    builder.add_table(
        thread_data,
        col_widths=[CONTENT_WIDTH*0.4, CONTENT_WIDTH*0.6],
        caption="Thread specifications for modular connections."
    )

    # === COIL MOUNTING ===
    builder.add_section_header("5.5 Coil Mounting and Ferrite Cores", level=1)

    builder.add_body_text(
        "The MIT sensing coils are wound onto ferrite rod cores positioned along the probe "
        "body. This configuration keeps coils internal to the 16mm profile, adding only "
        "1-2mm to the rod OD with potting. Coils are positioned orthogonally (90 degree "
        "separation) to minimize direct coupling between TX and RX elements.",
        first_paragraph=True
    )

    # Figure 6: Coil Mounting Detail
    print("  Generating Figure 6: Coil Mounting Detail...")
    fig6_buf = create_coil_mounting_detail()
    builder.add_figure(
        fig6_buf,
        "Coil assembly detail showing ferrite rod core (100mm x 6mm) with copper wire "
        "windings (200-400 turns of 0.2mm enameled wire). Support brackets maintain "
        "alignment within the probe cavity. Lead wires route to the surface junction box.",
        height=CONTENT_WIDTH * 0.55
    )

    # Coil specs tables
    ferrite_data = [
        ['Parameter', 'Specification'],
        ['Diameter', '6-8 mm'],
        ['Length', '40-80 mm'],
        ['Material', 'MnZn ferrite'],
        ['Quantity', '1-2 per probe (TX/RX)'],
    ]
    builder.add_table(
        ferrite_data,
        col_widths=[CONTENT_WIDTH*0.4, CONTENT_WIDTH*0.6],
        caption="Ferrite core specifications."
    )

    coil_data = [
        ['Parameter', 'Specification'],
        ['Wire Gauge', '34-38 AWG (fine wire)'],
        ['Turns', '200-400 turns'],
        ['Target Inductance', '1-2 mH'],
        ['Target Q Factor', '>20'],
    ]
    builder.add_table(
        coil_data,
        col_widths=[CONTENT_WIDTH*0.4, CONTENT_WIDTH*0.6],
        caption="Coil winding specifications."
    )

    # === ERT RING ELECTRODES ===
    builder.add_section_header("5.6 ERT Ring Electrodes", level=1)

    builder.add_body_text(
        "Electrical Resistivity Tomography (ERT) electrodes are implemented as narrow "
        "stainless steel or copper ring bands mounted on 3D-printed insulating collars. "
        "The rings are flush-mounted to maintain the smooth probe profile. Multiple rings "
        "(typically 2-3 per probe) enable various measurement configurations.",
        first_paragraph=True
    )

    # Figure 7: ERT Ring Detail
    print("  Generating Figure 7: ERT Ring Detail...")
    fig7_buf = create_ert_ring_detail()
    builder.add_figure(
        fig7_buf,
        "ERT ring collar assembly showing stainless steel ring (316L) flush-mounted on "
        "insulating collar. Spring-loaded contact pins ensure reliable electrical connection. "
        "Ring dimensions: 5mm width x 16mm OD, approximately 50mm collar length.",
        height=CONTENT_WIDTH * 0.45
    )

    # ERT specs
    ert_data = [
        ['Parameter', 'Value'],
        ['Ring Material', 'Stainless steel (316L) or copper'],
        ['Ring Width', '3-5 mm (narrow bands)'],
        ['Ring Thickness', '0.5-1 mm'],
        ['Diameter', 'Match rod OD (16 mm)'],
        ['Quantity', '2-3 rings per probe'],
        ['Minimum Spacing', '0.3 m between rings'],
    ]
    builder.add_table(
        ert_data,
        col_widths=[CONTENT_WIDTH*0.4, CONTENT_WIDTH*0.6],
        caption="ERT ring electrode specifications."
    )

    # Ring positions
    pos_data = [
        ['Position', 'Distance from Tip'],
        ['Ring A (Upper)', '0.5 m'],
        ['Ring B (Mid)', '1.5 m'],
        ['Ring C (Deep)', '2.5-3.0 m (optional)'],
    ]
    builder.add_table(
        pos_data,
        col_widths=[CONTENT_WIDTH*0.4, CONTENT_WIDTH*0.6],
        caption="Standard ERT ring mounting positions."
    )

    # === JUNCTION BOX ===
    builder.add_section_header("5.7 Junction Box Design", level=1)

    builder.add_body_text(
        "The surface junction box serves as the termination point for all probe wiring. "
        "It contains no active electronics - only passive connections via a terminal block. "
        "This keeps the probe lightweight and simple while enabling field serviceability. "
        "All 'smart' electronics reside in the central hub at the surface.",
        first_paragraph=True
    )

    # Figure 8: Junction Box Design
    print("  Generating Figure 8: Junction Box Design...")
    fig8_buf = create_junction_box_diagram()
    builder.add_figure(
        fig8_buf,
        "Junction box design showing (a) external view with cable gland and weatherproof "
        "body, and (b) internal cutaway revealing terminal block for MIT coil and ERT "
        "electrode connections. M12x1.75 internal thread mates with probe rod.",
        height=CONTENT_WIDTH * 0.45
    )

    # Junction box specs
    jbox_data = [
        ['Dimension', 'Value'],
        ['Diameter', '25 mm'],
        ['Height', '35 mm'],
        ['Thread', 'Internal M12x1.75 (bottom)'],
        ['Material', 'PETG or ABS (weatherproof)'],
        ['Features', 'Terminal block mount, cable gland'],
    ]
    builder.add_table(
        jbox_data,
        col_widths=[CONTENT_WIDTH*0.4, CONTENT_WIDTH*0.6],
        caption="Surface junction box specifications."
    )

    # === PROBE CROSS-SECTION ===
    builder.add_section_header("5.8 Probe Cross-Section Detail", level=1)

    builder.add_body_text(
        "The complete probe assembly integrates all components into a unified 16mm OD "
        "profile. The cross-section reveals the internal arrangement of coils, ERT rings, "
        "threaded joints, and wiring passages. This dense packing enables full sensing "
        "capability while minimizing ground disturbance.",
        first_paragraph=True
    )

    # Figure 9: Probe Cross-Section
    print("  Generating Figure 9: Probe Cross-Section...")
    fig9_buf = create_probe_cross_section()
    builder.add_figure(
        fig9_buf,
        "Detailed probe cross-section showing complete internal architecture. From bottom: "
        "tapered tip, ERT ring positions, RX coil with ferrite core, M12 threaded joints, "
        "TX coil assembly, and junction box connection. Standard 50cm segment shown.",
        width=CONTENT_WIDTH * 0.4
    )

    # === MANUFACTURING ===
    builder.add_section_header("5.9 Manufacturing Procedures", level=1)

    builder.add_section_header("5.9.1 3D Printing Guide", level=2)
    builder.add_body_text(
        "All plastic components are designed for FDM 3D printing with specific settings "
        "optimized for thread quality and structural integrity. PETG or ASA material is "
        "required for impact resistance and UV stability. Critical: threads require "
        "post-processing with tap and die tools.",
        first_paragraph=True
    )

    # Figure 10: Tool Requirements
    print("  Generating Figure 10: Tool Requirements...")
    fig10_buf = create_tool_requirements_diagram()
    builder.add_figure(
        fig10_buf,
        "Tool requirements for probe assembly including thread cutting tools (M12x1.75 "
        "tap and die), assembly supplies (epoxy, heat shrink), and finishing tools. "
        "3D printer settings shown for Bambu Lab / FDM machines.",
        height=CONTENT_WIDTH * 0.55
    )

    # Print settings table
    print_data = [
        ['Setting', 'Value', 'Notes'],
        ['Material', 'PETG or ASA', 'Required for impact/UV'],
        ['Layer Height', '0.12mm', 'Critical for threads'],
        ['Infill', '100% (Solid)', 'Critical for strength'],
        ['Walls', '6 Loops', 'Solid threaded regions'],
        ['Supports', 'DISABLED', 'Use built-in scaffolding'],
        ['Speed', '50mm/s outer wall', 'Quality over speed'],
    ]
    builder.add_table(
        print_data,
        col_widths=[CONTENT_WIDTH*0.25, CONTENT_WIDTH*0.3, CONTENT_WIDTH*0.45],
        caption="Recommended 3D printer settings for probe components."
    )

    builder.add_section_header("5.9.2 Thread Post-Processing", level=2)
    builder.add_info_box("Thread Cutting Procedure", [
        "Male threads: Print at 12.2mm OD, cut with M12x1.75 Die",
        "Female threads: Print at 10.5mm hole, cut with M12x1.75 Tap",
        "Always use cutting oil to reduce friction",
        "Cut slowly, back out frequently to clear chips",
        "Test fit before epoxy assembly"
    ])

    # === INSERTION METHODS ===
    builder.add_section_header("5.10 Insertion Methods", level=1)

    builder.add_body_text(
        "Four primary insertion methods accommodate different soil conditions. Method "
        "selection depends on soil type, target depth, and available equipment. The "
        "pilot rod method requires complete removal of metal before measurements.",
        first_paragraph=True
    )

    # Figure 11: Insertion Methods (using new diagram)
    print("  Generating Figure 11: Insertion Methods...")
    fig11_buf = create_insertion_methods_diagram()
    builder.add_figure(
        fig11_buf,
        "Probe insertion methods: (a) Hand auger for most soils using 10-20mm auger; "
        "(b) Pilot rod for compact soils - must remove metal before survey; "
        "(c) Direct push in sandy soils only; (d) Water-jet method for excellent sand "
        "penetration with minimal disturbance.",
        height=CONTENT_WIDTH * 0.75
    )

    # === ASSEMBLY SEQUENCE ===
    builder.add_section_header("5.11 Assembly Sequence", level=1)

    builder.add_body_text(
        "Probe assembly follows a bottom-to-top sequence with all components screwing "
        "together via M12 threads. The modular design enables field assembly and allows "
        "replacement of individual segments without rebuilding the entire probe.",
        first_paragraph=True
    )

    # Figure 12: Assembly Sequence
    print("  Generating Figure 12: Assembly Sequence...")
    fig12_buf = create_assembly_sequence_diagram()
    builder.add_figure(
        fig12_buf,
        "Assembly sequence from bottom to top: (1) Probe tip pointing down, (2) Bottom "
        "rod segment with RX coil, (3) First ERT ring at 0.5m, (4) Coupler joint, "
        "(5) Top rod segment with TX coil, (6) Additional ERT rings, (7) Junction box "
        "at surface level.",
        width=CONTENT_WIDTH * 0.55
    )

    # === ADVANTAGES ===
    builder.add_section_header("5.12 Advantages of Micro-Probe Design", level=1)

    # Figure 13: Advantages
    print("  Generating Figure 13: Advantages Infographic...")
    fig13_buf = create_advantages_diagram()
    builder.add_figure(
        fig13_buf,
        "Summary of micro-probe design advantages including 10x less disturbance, "
        "modular field-serviceable construction, passive probes with no downhole "
        "electronics, and 70% cost reduction compared to traditional designs.",
        height=CONTENT_WIDTH * 0.65
    )

    builder.add_bullet_list([
        "<b>Strength:</b> 16mm OD allows for robust M12 threads",
        "<b>Modularity:</b> Sensor spacing determined by rod segment length",
        "<b>Manufacturability:</b> Sensors built into printed parts, not glued onto rod",
        "<b>Smooth Profile:</b> Flush connections prevent snagging during insertion/extraction",
        "<b>Field Serviceable:</b> Replace individual segments without rebuilding entire probe",
        "<b>Simpler Assembly:</b> No electronics in probe",
        "<b>Lighter Weight:</b> ~50-100 g per meter (vs 200-250 g)",
        "<b>Easier Insertion:</b> Smaller diameter, less force needed",
        "<b>Lower Cost:</b> ~$40-60 per probe (vs $130-180)",
        "<b>Better Reliability:</b> Passive probes more robust",
        "<b>Minimal Intrusion:</b> ~10x less disturbance than 25mm design"
    ])

    # === REFERENCES ===
    builder.add_references([
        "[1] HIRT Development Team (2026). HIRT Whitepaper: Hybrid Impedance-Resistivity "
        "Tomography System. Section 7: Assembly and Wiring.",
        "[2] HIRT Development Team (2026). HIRT Whitepaper: Hybrid Impedance-Resistivity "
        "Tomography System. Section 6: Electronics and Circuits.",
        "[3] ISO 261:1998. ISO general purpose metric screw threads - General plan.",
        "[4] ASTM D2584. Standard Test Method for Ignition Loss of Cured Reinforced Resins "
        "(Fiberglass specifications).",
    ])

    # Build PDF
    output_path = builder.build()

    print("=" * 60)
    print(f"PDF generated successfully: {output_path}")
    print(f"Total figures: {builder.figure_count}")
    print(f"Total tables: {builder.table_count}")

    return output_path


if __name__ == "__main__":
    main()
