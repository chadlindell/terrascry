"""
HIRT PDF Generator - Mechanical Diagrams Module

Functions for creating CAD-style views, exploded assemblies, and
mechanical detail drawings.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, Arc, Wedge,
    FancyArrowPatch, Ellipse, PathPatch
)
from matplotlib.path import Path
import numpy as np
from io import BytesIO

# Color palette
COLORS = {
    'primary': '#1a365d',
    'secondary': '#2c5282',
    'accent': '#3182ce',
    'success': '#38a169',
    'warning': '#c53030',
    'orange': '#ed8936',
    'purple': '#805ad5',
    'gray_dark': '#4a5568',
    'gray_med': '#718096',
    'gray_light': '#a0aec0',
    'fiberglass': '#e8d4b8',
    'ferrite': '#333333',
    'copper': '#b87333',
    'steel': '#71797E',
}


def draw_dimension(ax, start, end, offset, text, fontsize=8, horizontal=True):
    """
    Draw a dimension line with arrows and centered text.

    Args:
        ax: matplotlib Axes object
        start: Start point (x, y)
        end: End point (x, y)
        offset: Perpendicular offset for dimension line
        text: Dimension text
        fontsize: Text font size
        horizontal: True if dimension is horizontal
    """
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


def draw_thread_profile(ax, x, y, length, pitch=1.75, od=12, side='right'):
    """
    Draw a thread profile cross-section.

    Args:
        ax: matplotlib Axes object
        x, y: Start position
        length: Thread length
        pitch: Thread pitch in mm
        od: Outer diameter in mm
        side: 'right' or 'left' for which side to draw
    """
    scale = 0.1  # Scale factor for display
    n_threads = int(length / (pitch * scale))

    thread_height = pitch * 0.65 * scale
    thread_points = []

    for i in range(n_threads):
        base_y = y - i * pitch * scale
        if side == 'right':
            thread_points.extend([
                (x, base_y),
                (x + thread_height, base_y - pitch * scale * 0.25),
                (x, base_y - pitch * scale * 0.5),
            ])
        else:
            thread_points.extend([
                (x, base_y),
                (x - thread_height, base_y - pitch * scale * 0.25),
                (x, base_y - pitch * scale * 0.5),
            ])

    if thread_points:
        xs, ys = zip(*thread_points)
        ax.plot(xs, ys, 'k-', lw=0.8)


def create_probe_cross_section():
    """
    Create detailed probe cross-section diagram with 3D-like shading.
    """
    fig, ax = plt.subplots(figsize=(6, 10)) # Wider canvas to prevent stretching
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # Main probe body (16mm OD = 0.8 display units at scale)
    probe_width = 0.8
    probe_height = 10

    # Shading helper
    def draw_shaded_cyl(x, y, w, h, color, alpha=1.0):
        # Base
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='black', lw=1, alpha=alpha))
        # Highlight (left)
        ax.add_patch(Rectangle((x + w*0.1, y), w*0.2, h, facecolor='white', alpha=0.3, edgecolor='none'))
        # Shadow (right)
        ax.add_patch(Rectangle((x + w*0.7, y), w*0.2, h, facecolor='black', alpha=0.2, edgecolor='none'))

    # Outer tube (fiberglass) - Shaded
    draw_shaded_cyl(-probe_width/2, 0, probe_width, probe_height, COLORS['fiberglass'])

    # Inner cavity (cutaway view)
    inner_width = 0.5
    ax.add_patch(Rectangle((-inner_width/2, 0.3), inner_width, probe_height - 0.6,
                           facecolor='white', edgecolor=COLORS['gray_light'], lw=0.5))

    # Components from bottom to top

    # Tip (pointed)
    tip_points = [(-probe_width/2, 0.3), (probe_width/2, 0.3), (0, -0.2)]
    ax.add_patch(Polygon(tip_points, facecolor=COLORS['primary'], edgecolor='black', lw=1))
    ax.text(1.5, 0.1, 'Probe Tip\n(Tapered)', fontsize=10, va='center', fontweight='bold')

    # ERT Ring 1 (bottom) - Shaded
    draw_shaded_cyl(-probe_width/2 - 0.05, 1.0, probe_width + 0.1, 0.15, COLORS['orange'])
    ax.text(1.5, 1.07, 'ERT Ring #1\n(Stainless)', fontsize=10, va='center')

    # Thread joint 1
    draw_shaded_cyl(-probe_width/2, 2.0, probe_width, 0.3, COLORS['gray_light'])
    ax.text(1.5, 2.15, 'M12x1.75\nFlush Joint', fontsize=9, va='center', style='italic')

    # RX Coil (lower)
    coil_y = 3.5
    # Coil winding - hatch pattern
    ax.add_patch(Rectangle((-inner_width/2 + 0.05, coil_y - 0.4), inner_width - 0.1, 0.8,
                           facecolor=COLORS['accent'], edgecolor='black', lw=1, hatch='///'))
    # Ferrite core
    ax.add_patch(Rectangle((-0.08, coil_y - 0.35), 0.16, 0.7,
                           facecolor=COLORS['ferrite'], edgecolor='black', lw=0.5))
    ax.text(1.5, coil_y, 'RX Coil\n(200T, 1mH)', fontsize=10, va='center', fontweight='bold')

    # ERT Ring 2 (middle)
    draw_shaded_cyl(-probe_width/2 - 0.05, 4.5, probe_width + 0.1, 0.15, COLORS['orange'])
    ax.text(1.5, 4.57, 'ERT Ring #2', fontsize=10, va='center')

    # Thread joint 2
    draw_shaded_cyl(-probe_width/2, 5.5, probe_width, 0.3, COLORS['gray_light'])

    # TX Coil
    coil_y = 7.0
    ax.add_patch(Rectangle((-inner_width/2 + 0.05, coil_y - 0.4), inner_width - 0.1, 0.8,
                           facecolor=COLORS['success'], edgecolor='black', lw=1, hatch='///'))
    # Ferrite core
    ax.add_patch(Rectangle((-0.08, coil_y - 0.35), 0.16, 0.7,
                           facecolor=COLORS['ferrite'], edgecolor='black', lw=0.5))
    ax.text(1.5, coil_y, 'TX Coil\n(400T, 2mH)', fontsize=10, va='center', fontweight='bold')

    # ERT Ring 3 (top)
    draw_shaded_cyl(-probe_width/2 - 0.05, 8.0, probe_width + 0.1, 0.15, COLORS['orange'])

    # Thread joint 3 (top)
    draw_shaded_cyl(-probe_width/2, 9.0, probe_width, 0.3, COLORS['gray_light'])

    # Junction box
    draw_shaded_cyl(-0.6, 9.5, 1.2, 0.8, COLORS['gray_dark'])
    ax.text(0, 9.9, 'Junction\nBox', ha='center', va='center',
           fontsize=8, color='white', fontweight='bold')
    ax.text(1.5, 9.9, '12-Pin\nConnector', fontsize=10, va='center')

    # Wiring harness (internal)
    ax.plot([0, 0], [1.0, 9.5], color=COLORS['copper'], lw=2, alpha=0.6, linestyle=':')

    # Dimensions
    draw_dimension(ax, (-probe_width/2, -0.5), (probe_width/2, -0.5),
                  -0.3, '16mm OD')

    draw_dimension(ax, (-1.5, 0), (-1.5, 10), 0.2, '50cm\nSegment',
                  horizontal=False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_exploded_assembly():
    """
    Create exploded view of probe assembly.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-2, 8)
    ax.set_ylim(-1, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # Exploded components with vertical separation
    components = [
        # (x, y, width, height, color, label)
        (2, 0.5, 1.5, 0.4, COLORS['primary'], 'Probe Tip'),
        (2, 1.5, 1.5, 3.0, COLORS['fiberglass'], 'Bottom Segment\n(with RX coil)'),
        (2, 5.0, 1.5, 3.0, COLORS['fiberglass'], 'Top Segment\n(with TX coil)'),
        (2, 8.5, 1.8, 0.6, COLORS['gray_dark'], 'Junction Box'),
    ]

    for x, y, w, h, color, label in components:
        ax.add_patch(FancyBboxPatch((x - w/2, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='black', lw=1.5))
        ax.text(x + w/2 + 0.5, y + h/2, label, fontsize=9, va='center')

    # Thread detail callouts
    ax.annotate('M12x1.75\nMale Thread', xy=(2, 4.4), xytext=(5, 4.5),
               fontsize=8, ha='left',
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))
    ax.annotate('M12x1.75\nFemale Thread', xy=(2, 5.1), xytext=(5, 5.8),
               fontsize=8, ha='left',
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

    # Internal component callouts
    ax.add_patch(Circle((2, 2.5), 0.25, facecolor=COLORS['accent'],
                       edgecolor='black', lw=1))
    ax.annotate('RX Coil\n(1mH)', xy=(2, 2.5), xytext=(-0.5, 2.5),
               fontsize=8, ha='right',
               arrowprops=dict(arrowstyle='->', color=COLORS['accent']))

    ax.add_patch(Circle((2, 6.5), 0.25, facecolor=COLORS['success'],
                       edgecolor='black', lw=1))
    ax.annotate('TX Coil\n(2mH)', xy=(2, 6.5), xytext=(-0.5, 6.5),
               fontsize=8, ha='right',
               arrowprops=dict(arrowstyle='->', color=COLORS['success']))

    # ERT rings
    for y in [1.2, 3.2, 5.8, 7.8]:
        ax.add_patch(Rectangle((1.15, y), 1.7, 0.12,
                               facecolor=COLORS['orange'], edgecolor='black', lw=0.5))
    ax.annotate('ERT Rings\n(4 per probe)', xy=(1.15, 3.2), xytext=(-0.5, 3.5),
               fontsize=8, ha='right',
               arrowprops=dict(arrowstyle='->', color=COLORS['orange']))

    # Assembly arrows
    ax.annotate('', xy=(2, 4.6), xytext=(2, 4.9),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=2))
    ax.annotate('', xy=(2, 1.0), xytext=(2, 1.4),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=2))
    ax.annotate('', xy=(2, 8.1), xytext=(2, 8.4),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark'], lw=2))

    # Title
    ax.text(3, -0.5, 'Probe Assembly - Exploded View', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_thread_detail():
    """
    Create detailed thread profile diagram.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # === Male Thread ===
    ax1.set_xlim(-1, 3)
    ax1.set_ylim(-0.5, 3)
    ax1.set_aspect('equal')

    # Thread body
    ax1.add_patch(Rectangle((0, 0), 0.8, 2.5,
                            facecolor=COLORS['fiberglass'], edgecolor='black', lw=1))

    # Thread profile (simplified zigzag)
    pitch = 0.175  # 1.75mm at scale
    n_threads = int(1.5 / pitch)
    for i in range(n_threads):
        y_base = 0.5 + i * pitch
        # Right side threads
        ax1.plot([0.8, 0.9, 0.8], [y_base, y_base + pitch/2, y_base + pitch],
                'k-', lw=1)

    # Dimensions
    ax1.annotate('', xy=(0.8, 0.5), xytext=(0.9, 0.5 + pitch/2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['warning'], lw=1))
    ax1.text(1.1, 0.6, 'P=1.75mm', fontsize=8, color=COLORS['warning'])

    ax1.annotate('', xy=(0, -0.3), xytext=(0.8, -0.3),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax1.text(0.4, -0.45, 'M12', fontsize=9, ha='center', fontweight='bold')

    ax1.set_title('(a) Male Thread Profile', fontsize=10, fontweight='bold',
                 color=COLORS['primary'])
    ax1.axis('off')

    # === Female Thread (Cross-section) ===
    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-0.5, 3)
    ax2.set_aspect('equal')

    # Outer body
    ax2.add_patch(Rectangle((0, 0), 1.2, 2.5,
                            facecolor=COLORS['fiberglass'], edgecolor='black', lw=1))

    # Inner bore
    ax2.add_patch(Rectangle((0.2, 0), 0.8, 2.5,
                            facecolor='white', edgecolor='black', lw=0.5))

    # Thread profile (internal)
    for i in range(n_threads):
        y_base = 0.5 + i * pitch
        # Left side threads (internal)
        ax2.plot([0.2, 0.1, 0.2], [y_base, y_base + pitch/2, y_base + pitch],
                'k-', lw=1)
        # Right side threads (internal)
        ax2.plot([1.0, 1.1, 1.0], [y_base, y_base + pitch/2, y_base + pitch],
                'k-', lw=1)

    # Dimensions
    ax2.annotate('', xy=(0.2, -0.2), xytext=(1.0, -0.2),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax2.text(0.6, -0.35, 'ID 10.6mm', fontsize=8, ha='center')

    ax2.annotate('', xy=(0, -0.2), xytext=(1.2, -0.2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray_med'], lw=1))
    ax2.text(0.6, -0.5, '(OD 16mm)', fontsize=7, ha='center', color=COLORS['gray_med'])

    ax2.set_title('(b) Female Thread (Section)', fontsize=10, fontweight='bold',
                 color=COLORS['primary'])
    ax2.axis('off')

    # Overall title
    fig.suptitle('M12x1.75 Thread Profile Detail', fontsize=11, fontweight='bold',
                y=0.98, color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_coil_mounting_detail():
    """
    Create coil mounting and ferrite core detail.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-2, 6)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Probe body cross-section
    ax.add_patch(Rectangle((0, 0), 4, 4,
                           facecolor=COLORS['fiberglass'], edgecolor='black', lw=1.5))
    ax.add_patch(Rectangle((0.3, 0.3), 3.4, 3.4,
                           facecolor='white', edgecolor=COLORS['gray_light'], lw=0.5))

    # Ferrite core (rod)
    ax.add_patch(Rectangle((1.7, 0.8), 0.6, 2.4,
                           facecolor=COLORS['ferrite'], edgecolor='black', lw=1))
    ax.text(2, 4.3, 'Ferrite Rod\n(100mm x 6mm)', ha='center', fontsize=8)

    # Coil windings (stylized)
    for y in np.linspace(1.0, 2.8, 12):
        # Left side windings
        ax.add_patch(Circle((1.4, y), 0.08, facecolor=COLORS['copper'],
                           edgecolor='black', lw=0.3))
        # Right side windings
        ax.add_patch(Circle((2.6, y), 0.08, facecolor=COLORS['copper'],
                           edgecolor='black', lw=0.3))

    # Winding annotation
    ax.annotate('Copper Wire\n(0.2mm enameled)\n200-400 turns', xy=(2.6, 2.0),
               xytext=(4.5, 2.5), fontsize=8, ha='left',
               arrowprops=dict(arrowstyle='->', color=COLORS['copper']))

    # Mounting supports
    ax.add_patch(Rectangle((0.5, 0.8), 0.6, 0.3,
                           facecolor=COLORS['gray_light'], edgecolor='black', lw=0.5))
    ax.add_patch(Rectangle((2.9, 0.8), 0.6, 0.3,
                           facecolor=COLORS['gray_light'], edgecolor='black', lw=0.5))
    ax.add_patch(Rectangle((0.5, 2.9), 0.6, 0.3,
                           facecolor=COLORS['gray_light'], edgecolor='black', lw=0.5))
    ax.add_patch(Rectangle((2.9, 2.9), 0.6, 0.3,
                           facecolor=COLORS['gray_light'], edgecolor='black', lw=0.5))
    ax.annotate('Support\nBrackets', xy=(0.8, 0.95), xytext=(-1.0, 1.0),
               fontsize=8, ha='right',
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

    # Lead wires
    ax.plot([1.4, 1.4, 0.5], [3.1, 3.5, 3.5], color=COLORS['copper'], lw=1.5)
    ax.plot([2.6, 2.6, 3.5], [3.1, 3.5, 3.5], color=COLORS['copper'], lw=1.5)
    ax.text(0.3, 3.7, 'Lead\nWires', fontsize=7, ha='center')

    # Dimensions
    draw_dimension(ax, (0, -0.3), (4, -0.3), -0.3, '16mm (OD)')
    draw_dimension(ax, (1.7, -0.3), (2.3, -0.3), -0.6, '6mm core', fontsize=7)

    # Title
    ax.text(2, 4.8, 'Coil Assembly Detail', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_disturbance_comparison():
    """
    Create soil disturbance comparison diagram.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6)) # Widened from default to ensure circles stay round
    
    for ax, title, rod_dia, hole_dia, vol in [
        (ax1, 'HIRT Micro-Probe', 16, 20, '~1.0 L'),
        (ax2, 'Standard Probe', 25, 50, '~6.0 L')
    ]:
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_aspect('equal') # Critical for circles
        ax.axis('off')
        
        # Ground surface
        ax.add_patch(Rectangle((-40, -40), 80, 80, facecolor='#d4a373', alpha=0.3))
        
        # Disturbed zone
        dist_rad = hole_dia / 2
        ax.add_patch(Circle((0, 0), dist_rad, facecolor='#805ad5', alpha=0.4))
        ax.add_patch(Circle((0, 0), dist_rad, fill=False, edgecolor='#805ad5', linestyle='--', lw=1))
        
        # Rod
        rod_rad = rod_dia / 2
        ax.add_patch(Circle((0, 0), rod_rad, facecolor='#2c5282', edgecolor='black', lw=1.5))
        
        # Labels
        ax.text(0, 32, title, ha='center', fontsize=11, fontweight='bold', color='#1a365d')
        ax.text(0, -32, f'Hole: {hole_dia}mm', ha='center', fontsize=10)
        ax.text(0, -38, f'Vol: {vol}', ha='center', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_ert_ring_detail():
    """
    Create ERT ring collar assembly detail.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Probe body (cross-section)
    ax.add_patch(Rectangle((0, 0.5), 5, 2,
                           facecolor=COLORS['fiberglass'], edgecolor='black', lw=1.5))

    # ERT ring (external collar)
    ring_inner = 0.4
    ax.add_patch(Rectangle((-0.1, 1.0), 5.2, 1,
                           facecolor=COLORS['orange'], edgecolor='black', lw=1))
    ax.add_patch(Rectangle((0, 1.1), 5, 0.8,
                           facecolor=COLORS['fiberglass'], edgecolor='none'))

    # Contact points
    for x in [0.5, 2.5, 4.5]:
        ax.add_patch(Circle((x, 1.5), 0.15, facecolor=COLORS['steel'],
                           edgecolor='black', lw=0.5))
    ax.annotate('Spring-loaded\ncontact pins', xy=(0.5, 1.5), xytext=(-0.5, 2.8),
               fontsize=8, ha='center',
               arrowprops=dict(arrowstyle='->', color=COLORS['gray_dark']))

    # Wire connection
    ax.plot([4.5, 4.5, 5.5], [1.5, 2.8, 2.8], color=COLORS['copper'], lw=1.5)
    ax.text(5.7, 2.8, 'To harness', fontsize=8, va='center')

    # Dimensions
    draw_dimension(ax, (0, 0.2), (5, 0.2), -0.3, '~50mm length')
    draw_dimension(ax, (-0.1, 1.0), (-0.1, 2.0), -0.5, '10mm\nheight', horizontal=False)

    # Material callouts
    ax.text(2.5, 3.2, 'Stainless Steel Ring\n(316L, flush mount)',
           ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor=COLORS['orange'], alpha=0.3))

    # Title
    ax.text(2.5, -0.5, 'ERT Ring Assembly Detail', ha='center',
           fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf
