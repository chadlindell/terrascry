#!/usr/bin/env python3
"""
HIRT Section 06: Electronics and Circuits - Publication-Quality PDF Generator

This module generates a comprehensive PDF covering all HIRT electronics:
- MIT TX/RX signal chains
- ERT current source (Howland pump)
- Lock-in detection
- Power distribution
- Signal processing and filtering
- Multiplexer topology
- Ground loop prevention

Usage:
    python section_06_electronics_circuits.py

Output:
    output/sections/06-electronics-circuits.pdf
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch, Arc, Wedge
)
import numpy as np
from io import BytesIO

from lib.pdf_builder import SectionPDFBuilder
from lib.styles import CONTENT_WIDTH, COLOR_PALETTE
from lib.diagrams.circuits import (
    create_mit_tx_chain,
    create_mit_rx_chain,
    create_ert_current_source,
    create_system_block_diagram,
    create_power_distribution,
    draw_block,
    draw_connection,
    draw_opamp,
    draw_resistor,
    draw_capacitor,
    draw_ground_symbol,
    COLORS
)


# ============================================================================
# CUSTOM DIAGRAMS FOR SECTION 06
# ============================================================================

def create_lock_in_detection_diagram():
    """
    Create detailed lock-in detection block diagram showing
    digital and analog lock-in approaches.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 4.7, 'Lock-In Detection (Digital Implementation)',
           fontsize=12, fontweight='bold', ha='center', color=COLORS['primary'])

    # Input signal path
    draw_block(ax, 1.0, 3.0, 1.2, 0.7, 'RX\nSignal', COLORS['light_green'], fontsize=8)

    # Multiplier blocks (I and Q channels)
    draw_block(ax, 3.0, 3.8, 1.0, 0.6, 'Multiply', COLORS['light_blue'], fontsize=8)
    draw_block(ax, 3.0, 2.2, 1.0, 0.6, 'Multiply', COLORS['light_blue'], fontsize=8)

    # Reference signal generators
    draw_block(ax, 3.0, 1.0, 1.2, 0.6, 'sin(wt)\nRef', COLORS['light_purple'], fontsize=7)
    draw_block(ax, 3.0, 4.6, 1.2, 0.6, 'cos(wt)\nRef', COLORS['light_purple'], fontsize=7)

    # Low-pass filters
    draw_block(ax, 5.0, 3.8, 1.2, 0.6, 'LP Filter', COLORS['light_orange'], fontsize=8)
    draw_block(ax, 5.0, 2.2, 1.2, 0.6, 'LP Filter', COLORS['light_orange'], fontsize=8)

    # Integration blocks
    draw_block(ax, 7.0, 3.8, 1.0, 0.6, 'Integrate', COLORS['light_green'], fontsize=7)
    draw_block(ax, 7.0, 2.2, 1.0, 0.6, 'Integrate', COLORS['light_green'], fontsize=7)

    # Output computation
    draw_block(ax, 9.0, 3.0, 1.6, 1.4, 'Compute\nA = sqrt(I^2+Q^2)\nPhi = atan2(Q,I)',
              COLORS['light_purple'], fontsize=7)

    # Connections - main signal path
    ax.plot([1.6, 2.0], [3.0, 3.0], 'k-', lw=1.5)
    ax.plot([2.0, 2.0], [2.2, 3.8], 'k-', lw=1.5)
    draw_connection(ax, (2.0, 3.8), (2.5, 3.8), color=COLORS['gray_dark'])
    draw_connection(ax, (2.0, 2.2), (2.5, 2.2), color=COLORS['gray_dark'])

    # Reference connections
    ax.plot([3.0, 3.0], [4.3, 4.1], 'k-', lw=1)
    ax.plot([3.0, 3.0], [1.3, 1.9], 'k-', lw=1)

    # Through filters
    draw_connection(ax, (3.5, 3.8), (4.4, 3.8), color=COLORS['gray_dark'])
    draw_connection(ax, (3.5, 2.2), (4.4, 2.2), color=COLORS['gray_dark'])

    # Through integrators
    draw_connection(ax, (5.6, 3.8), (6.5, 3.8), color=COLORS['gray_dark'])
    draw_connection(ax, (5.6, 2.2), (6.5, 2.2), color=COLORS['gray_dark'])

    # To output
    ax.plot([7.5, 8.0], [3.8, 3.8], 'k-', lw=1)
    ax.plot([8.0, 8.0], [3.8, 3.4], 'k-', lw=1)
    draw_connection(ax, (8.0, 3.4), (8.2, 3.4), color=COLORS['gray_dark'])

    ax.plot([7.5, 8.0], [2.2, 2.2], 'k-', lw=1)
    ax.plot([8.0, 8.0], [2.2, 2.6], 'k-', lw=1)
    draw_connection(ax, (8.0, 2.6), (8.2, 2.6), color=COLORS['gray_dark'])

    # Channel labels
    ax.text(1.8, 4.1, 'I-channel', fontsize=7, color=COLORS['gray_dark'])
    ax.text(1.8, 1.9, 'Q-channel', fontsize=7, color=COLORS['gray_dark'])

    # Output labels
    ax.text(9.0, 1.8, 'Amplitude &\nPhase Output', fontsize=8, ha='center',
           color=COLORS['success'], fontweight='bold')

    # Annotations
    ax.text(5, 0.4, 'Digital lock-in extracts signal at reference frequency,\n'
           'rejecting noise at other frequencies (SNR improvement > 40 dB)',
           fontsize=8, ha='center', style='italic', color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_signal_level_progression():
    """
    Create signal level progression chart showing voltage levels
    through the MIT RX chain.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Stages and their signal levels (in dBV)
    stages = ['RX Coil', 'Preamp\n(G=10)', 'Inst Amp\n(G=100)', 'Filter', 'ADC\nInput']
    levels_dbv = [-100, -80, -60, -58, -58]  # dBV
    levels_v = ['1 uV', '10 uV', '1 mV', '0.8 mV', '0.8 mV']
    noise_floor = [-120, -110, -90, -92, -92]  # dBV noise floor

    x = np.arange(len(stages))
    width = 0.35

    # Signal bars
    bars1 = ax.bar(x - width/2, [l + 120 for l in levels_dbv], width,
                   label='Signal Level', color=COLORS['success'], alpha=0.8)

    # Noise floor bars
    bars2 = ax.bar(x + width/2, [n + 120 for n in noise_floor], width,
                   label='Noise Floor', color=COLORS['warning'], alpha=0.5)

    # Add voltage labels on signal bars
    for bar, v_label in zip(bars1, levels_v):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               v_label, ha='center', va='bottom', fontsize=8, fontweight='bold')

    # SNR annotation line
    ax.annotate('', xy=(4.2, 62), xytext=(4.2, 28),
               arrowprops=dict(arrowstyle='<->', color=COLORS['accent'], lw=2))
    ax.text(4.4, 45, 'SNR\n~34 dB', fontsize=8, color=COLORS['accent'], fontweight='bold')

    ax.set_ylabel('Signal Level (dBV + 120)', fontsize=10)
    ax.set_title('MIT RX Signal Level Progression', fontsize=12,
                fontweight='bold', color=COLORS['primary'])
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, 80)

    # Grid
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    # Secondary y-axis with actual dBV
    ax2 = ax.twinx()
    ax2.set_ylim(-120, -40)
    ax2.set_ylabel('Actual Level (dBV)', fontsize=10)

    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_adc_interface_diagram():
    """
    Create ADC interface diagram showing connections to ADS1256.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(4.5, 4.7, 'ADC Interface (ADS1256)', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # ADC chip block
    adc_box = FancyBboxPatch((3.0, 1.5), 3.0, 2.5, boxstyle="round,pad=0.05",
                             facecolor=COLORS['light_purple'], edgecolor=COLORS['secondary'],
                             linewidth=2)
    ax.add_patch(adc_box)
    ax.text(4.5, 3.5, 'ADS1256', fontsize=11, fontweight='bold', ha='center')
    ax.text(4.5, 3.0, '24-bit ADC\n30 kSPS\nPGA 1-64x', fontsize=8, ha='center')

    # Input side labels
    inputs = [
        (2.8, 3.8, 'AIN0+', 'MIT RX'),
        (2.8, 3.4, 'AIN0-', ''),
        (2.8, 2.8, 'AIN1+', 'ERT V+'),
        (2.8, 2.4, 'AIN1-', 'ERT V-'),
        (2.8, 1.8, 'AINCOM', 'AGND'),
    ]

    for x, y, label, src in inputs:
        ax.text(x, y, label, fontsize=7, ha='right', va='center')
        ax.plot([x + 0.05, x + 0.2], [y, y], 'k-', lw=1)
        if src:
            ax.text(0.5, y, src, fontsize=8, ha='left', va='center', color=COLORS['gray_dark'])
            ax.plot([1.3, 2.8], [y, y], 'k-', lw=1)

    # Output side labels (SPI)
    outputs = [
        (6.2, 3.6, 'SCLK'),
        (6.2, 3.2, 'DIN (MOSI)'),
        (6.2, 2.8, 'DOUT (MISO)'),
        (6.2, 2.4, 'CS'),
        (6.2, 2.0, 'DRDY'),
    ]

    for x, y, label in outputs:
        ax.text(x, y, label, fontsize=7, ha='left', va='center')
        ax.plot([x - 0.05, x - 0.2], [y, y], 'k-', lw=1)

    # MCU block
    draw_block(ax, 8.0, 2.8, 1.4, 1.8, 'MCU\n(ESP32)\nSPI\nMaster', COLORS['light_blue'], fontsize=7)

    # SPI connections
    for y in [3.6, 3.2, 2.8, 2.4, 2.0]:
        ax.plot([6.5, 7.3], [y, y], 'b-', lw=1)

    # Power connections
    ax.text(4.5, 1.3, '+3.3V', fontsize=8, ha='center', color='red')
    ax.plot([4.5, 4.5], [1.5, 1.1], 'r-', lw=1.5)

    # Reference voltage
    draw_block(ax, 1.0, 1.2, 1.0, 0.6, 'Vref\n2.5V', COLORS['light_orange'], fontsize=7)
    ax.plot([1.5, 3.0], [1.2, 1.8], 'k--', lw=1)
    ax.text(2.0, 1.6, 'VREFP', fontsize=7, color=COLORS['gray_dark'])

    # Annotations
    ax.text(4.5, 0.5, 'Differential inputs | Internal MUX | PGA for gain adjustment',
           fontsize=8, ha='center', style='italic', color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_multiplexer_switching_diagram():
    """
    Create multiplexer switching topology diagram showing
    how probes are selected for TX/RX operations.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'Multiplexer Switching Topology', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # TX Multiplexer section
    draw_block(ax, 1.5, 4.5, 1.4, 0.7, 'TX\nDriver', COLORS['light_green'], fontsize=8)

    # TX MUX
    mux_tx = FancyBboxPatch((2.5, 3.5), 1.5, 1.8, boxstyle="round,pad=0.03",
                            facecolor=COLORS['light_blue'], edgecolor=COLORS['secondary'],
                            linewidth=1.5)
    ax.add_patch(mux_tx)
    ax.text(3.25, 5.1, 'TX MUX', fontsize=9, ha='center', fontweight='bold')
    ax.text(3.25, 4.5, 'CD4051\n8:1', fontsize=8, ha='center')

    # TX MUX outputs
    tx_outputs = ['TX1', 'TX2', 'TX3', '...', 'TX8']
    for i, label in enumerate(tx_outputs):
        y = 5.0 - i * 0.35
        ax.plot([4.0, 4.5], [y, y], 'k-', lw=1)
        ax.text(4.6, y, label, fontsize=7, va='center')

    # RX Multiplexer section
    draw_block(ax, 1.5, 1.5, 1.4, 0.7, 'RX\nPreamp', COLORS['light_orange'], fontsize=8)

    # RX MUX
    mux_rx = FancyBboxPatch((2.5, 0.7), 1.5, 1.8, boxstyle="round,pad=0.03",
                            facecolor=COLORS['light_blue'], edgecolor=COLORS['secondary'],
                            linewidth=1.5)
    ax.add_patch(mux_rx)
    ax.text(3.25, 2.3, 'RX MUX', fontsize=9, ha='center', fontweight='bold')
    ax.text(3.25, 1.7, 'CD4051\n8:1', fontsize=8, ha='center')

    # RX MUX outputs
    rx_outputs = ['RX1', 'RX2', 'RX3', '...', 'RX8']
    for i, label in enumerate(rx_outputs):
        y = 2.2 - i * 0.35
        ax.plot([4.0, 4.5], [y, y], 'k-', lw=1)
        ax.text(4.6, y, label, fontsize=7, va='center')

    # Connections to driver/preamp
    draw_connection(ax, (2.2, 4.5), (2.5, 4.5), color=COLORS['gray_dark'])
    draw_connection(ax, (2.5, 1.5), (2.2, 1.5), color=COLORS['gray_dark'])

    # Probe array representation
    probe_box = FancyBboxPatch((5.5, 0.5), 3.5, 5.0, boxstyle="round,pad=0.05",
                               facecolor='white', edgecolor=COLORS['gray_med'],
                               linewidth=1, linestyle='--')
    ax.add_patch(probe_box)
    ax.text(7.25, 5.3, 'Probe Array (8-24 probes)', fontsize=9, ha='center',
           fontweight='bold', color=COLORS['gray_dark'])

    # Individual probes
    probe_y = [4.5, 3.7, 2.9, 2.1, 1.3]
    probe_labels = ['Probe 1', 'Probe 2', 'Probe 3', '...', 'Probe N']
    for i, (y, label) in enumerate(zip(probe_y, probe_labels)):
        draw_block(ax, 7.25, y, 1.8, 0.5, label, COLORS['light_green'], fontsize=7)
        # TX connection
        ax.plot([5.0, 5.5], [5.0 - i*0.35, y + 0.1], 'g-', lw=0.8, alpha=0.6)
        # RX connection
        ax.plot([5.0, 5.5], [2.2 - i*0.35, y - 0.1], 'b-', lw=0.8, alpha=0.6)

    # MCU control
    draw_block(ax, 1.5, 3.0, 1.2, 0.5, 'MCU\nGPIO', COLORS['light_purple'], fontsize=7)
    ax.plot([2.1, 2.5], [3.2, 4.0], 'k--', lw=1)
    ax.plot([2.1, 2.5], [2.8, 2.0], 'k--', lw=1)
    ax.text(2.0, 3.6, 'A0-A2', fontsize=6, color=COLORS['gray_dark'])

    # Annotations
    ax.text(5, 0.2, 'Sequential TX/RX switching allows full tomographic measurement matrix',
           fontsize=8, ha='center', style='italic', color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_noise_filtering_diagram():
    """
    Create noise filtering stages diagram showing the
    filtering at each stage of the signal chain.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 3.7, 'Noise Filtering Stages', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Stage 1: Input filter
    draw_block(ax, 1.0, 2.0, 1.4, 1.0, 'Input\nFilter\n(RC LP)', COLORS['light_green'], fontsize=7)
    ax.text(1.0, 1.3, 'fc = 100 kHz', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Stage 2: Preamp with high-pass
    draw_block(ax, 3.0, 2.0, 1.4, 1.0, 'Preamp\n+\nHP Filter', COLORS['light_blue'], fontsize=7)
    ax.text(3.0, 1.3, 'fc = 100 Hz', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Stage 3: Band-pass filter
    draw_block(ax, 5.0, 2.0, 1.4, 1.0, 'Active\nBand-Pass\nFilter', COLORS['light_orange'], fontsize=7)
    ax.text(5.0, 1.3, '1-50 kHz', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Stage 4: Anti-aliasing
    draw_block(ax, 7.0, 2.0, 1.4, 1.0, 'Anti-Alias\nFilter\n(4th order)', COLORS['light_purple'], fontsize=7)
    ax.text(7.0, 1.3, 'fc = 15 kHz', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Stage 5: Digital filter (in DSP)
    draw_block(ax, 9.0, 2.0, 1.2, 1.0, 'Digital\nFilter\n(DSP)', COLORS['light_green'], fontsize=7)
    ax.text(9.0, 1.3, 'Matched', fontsize=7, ha='center', color=COLORS['gray_dark'])

    # Connections
    connections = [(1.7, 2.3), (3.7, 4.3), (5.7, 6.3), (7.7, 8.4)]
    for start_x, end_x in connections:
        draw_connection(ax, (start_x, 2.0), (end_x, 2.0), color=COLORS['gray_dark'])

    # Noise sources labeled above
    noise_sources = [
        (1.0, 'EMI\n60 Hz'),
        (3.0, 'DC\nOffset'),
        (5.0, 'Wideband\nNoise'),
        (7.0, 'Aliasing'),
        (9.0, 'Quantization'),
    ]
    for x, label in noise_sources:
        ax.text(x, 2.9, label, fontsize=6, ha='center', color=COLORS['warning'],
               fontweight='bold')
        ax.annotate('', xy=(x, 2.5), xytext=(x, 2.7),
                   arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=1))

    # Rejection labels below
    ax.text(5, 0.6, 'Each stage targets specific noise sources:\n'
           'Total noise rejection > 60 dB in measurement band',
           fontsize=8, ha='center', style='italic', color=COLORS['gray_dark'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_ground_loop_prevention():
    """
    Create ground loop prevention diagram showing proper
    grounding techniques.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # === Left Panel: Ground Loop Problem ===
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 4.5)
    ax1.set_aspect('equal')
    ax1.axis('off')

    ax1.set_title('(a) Ground Loop Problem', fontsize=10, fontweight='bold',
                 color=COLORS['warning'], pad=10)

    # Two circuit blocks
    draw_block(ax1, 1.0, 3.0, 1.2, 0.8, 'Circuit\nA', COLORS['light_red'], fontsize=8)
    draw_block(ax1, 4.0, 3.0, 1.2, 0.8, 'Circuit\nB', COLORS['light_red'], fontsize=8)

    # Signal connection
    ax1.plot([1.6, 3.4], [3.0, 3.0], 'k-', lw=1.5)
    ax1.text(2.5, 3.2, 'Signal', fontsize=7, ha='center')

    # Multiple ground paths (the problem)
    # Path 1 - direct
    ax1.plot([1.0, 1.0], [2.6, 1.5], 'r-', lw=2)
    ax1.plot([4.0, 4.0], [2.6, 1.5], 'r-', lw=2)
    ax1.plot([1.0, 4.0], [1.5, 1.5], 'r-', lw=2)

    # Path 2 - through chassis/earth
    ax1.plot([1.0, 1.0], [1.5, 0.8], 'r--', lw=1.5)
    ax1.plot([4.0, 4.0], [1.5, 0.8], 'r--', lw=1.5)
    ax1.plot([1.0, 4.0], [0.8, 0.8], 'r--', lw=1.5)

    # Ground symbols
    draw_ground_symbol(ax1, 1.0, 0.5)
    draw_ground_symbol(ax1, 4.0, 0.5)

    # Loop current indicator
    ax1.annotate('', xy=(2.0, 1.2), xytext=(3.0, 1.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=2))
    ax1.text(2.5, 0.9, 'Loop\nCurrent', fontsize=7, ha='center', color=COLORS['warning'])

    # Noise injection
    ax1.text(2.5, 1.8, 'EMI pickup', fontsize=7, ha='center',
            color=COLORS['warning'], style='italic')

    # === Right Panel: Star Ground Solution ===
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 4.5)
    ax2.set_aspect('equal')
    ax2.axis('off')

    ax2.set_title('(b) Star Ground Solution', fontsize=10, fontweight='bold',
                 color=COLORS['success'], pad=10)

    # Two circuit blocks
    draw_block(ax2, 1.0, 3.0, 1.2, 0.8, 'Circuit\nA', COLORS['light_green'], fontsize=8)
    draw_block(ax2, 4.0, 3.0, 1.2, 0.8, 'Circuit\nB', COLORS['light_green'], fontsize=8)

    # Signal connection with shield
    ax2.plot([1.6, 3.4], [3.0, 3.0], 'k-', lw=1.5)
    ax2.plot([1.6, 3.4], [3.15, 3.15], 'g--', lw=1)
    ax2.plot([1.6, 3.4], [2.85, 2.85], 'g--', lw=1)
    ax2.text(2.5, 3.4, 'Shielded Signal', fontsize=7, ha='center')

    # Star ground point
    star_x, star_y = 2.5, 1.2
    ax2.add_patch(Circle((star_x, star_y), 0.15, facecolor=COLORS['success'],
                        edgecolor='black', lw=2))
    ax2.text(star_x, 0.7, 'Star Ground\nPoint', fontsize=7, ha='center',
            fontweight='bold', color=COLORS['success'])

    # Individual ground connections (star pattern)
    ax2.plot([1.0, star_x], [2.6, star_y + 0.15], 'g-', lw=2)
    ax2.plot([4.0, star_x], [2.6, star_y + 0.15], 'g-', lw=2)
    ax2.plot([2.5, star_x], [2.85, star_y + 0.15], 'g-', lw=1)  # Shield ground

    # Single earth ground
    ax2.plot([star_x, star_x], [star_y - 0.15, 0.5], 'g-', lw=2)
    draw_ground_symbol(ax2, star_x, 0.3)

    # Annotations
    ax2.text(0.5, 4.2, 'No loop!', fontsize=8, color=COLORS['success'], fontweight='bold')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_ert_polarity_reversal():
    """
    Create ERT polarity reversal circuit diagram showing
    the relay-based switching for electrode polarization reduction.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(4, 4.7, 'ERT Polarity Reversal Circuit', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # Current source
    draw_block(ax, 1.5, 2.5, 1.4, 1.0, 'Current\nSource\n(1 mA)', COLORS['light_orange'], fontsize=8)

    # DPDT Relay
    relay_box = FancyBboxPatch((3.0, 1.5), 2.0, 2.0, boxstyle="round,pad=0.05",
                               facecolor=COLORS['light_blue'], edgecolor=COLORS['secondary'],
                               linewidth=1.5)
    ax.add_patch(relay_box)
    ax.text(4.0, 3.3, 'DPDT Relay', fontsize=9, ha='center', fontweight='bold')
    ax.text(4.0, 2.9, '(G5V-2)', fontsize=7, ha='center')

    # Relay internal connections (simplified)
    # Common contacts
    ax.plot([3.2, 3.5], [2.5, 2.5], 'k-', lw=1.5)
    ax.plot([3.2, 3.5], [2.0, 2.0], 'k-', lw=1.5)

    # Switching contacts (shown in one position)
    ax.plot([3.5, 4.0], [2.5, 2.7], 'k-', lw=1.5)
    ax.plot([3.5, 4.0], [2.0, 1.8], 'k-', lw=1.5)
    ax.add_patch(Circle((3.5, 2.5), 0.05, facecolor='black'))
    ax.add_patch(Circle((3.5, 2.0), 0.05, facecolor='black'))

    # NC/NO contacts
    ax.add_patch(Circle((4.0, 2.7), 0.05, facecolor='black'))
    ax.add_patch(Circle((4.0, 2.3), 0.05, facecolor='black'))
    ax.add_patch(Circle((4.0, 2.0), 0.05, facecolor='black'))
    ax.add_patch(Circle((4.0, 1.8), 0.05, facecolor='black'))

    # Cross-wiring
    ax.plot([4.0, 4.8], [2.7, 2.7], 'b-', lw=1)
    ax.plot([4.0, 4.8], [1.8, 1.8], 'b-', lw=1)
    ax.plot([4.0, 4.5], [2.3, 2.3], 'r-', lw=1)
    ax.plot([4.5, 4.5], [2.3, 1.8], 'r-', lw=1)
    ax.plot([4.0, 4.6], [2.0, 2.0], 'r-', lw=1)
    ax.plot([4.6, 4.6], [2.0, 2.7], 'r-', lw=1)

    # Input from current source
    draw_connection(ax, (2.2, 2.7), (3.2, 2.7), color='black')
    ax.plot([3.2, 3.2], [2.7, 2.5], 'k-', lw=1)
    draw_connection(ax, (2.2, 2.3), (3.2, 2.3), color='black')
    ax.plot([3.2, 3.2], [2.3, 2.0], 'k-', lw=1)

    # Output to electrodes
    draw_block(ax, 6.5, 2.7, 1.0, 0.5, 'Ring A', COLORS['light_green'], fontsize=8)
    draw_block(ax, 6.5, 1.8, 1.0, 0.5, 'Ring B', COLORS['light_green'], fontsize=8)

    ax.plot([4.8, 6.0], [2.7, 2.7], 'k-', lw=1)
    ax.plot([4.8, 6.0], [1.8, 1.8], 'k-', lw=1)

    # MCU control
    draw_block(ax, 4.0, 0.8, 1.0, 0.5, 'MCU', COLORS['light_purple'], fontsize=8)
    ax.plot([4.0, 4.0], [1.05, 1.5], 'k--', lw=1)
    ax.text(4.2, 1.3, 'Control', fontsize=7, color=COLORS['gray_dark'])

    # Timing annotation
    ax.text(4, 4.2, 'Reversal frequency: 0.5 Hz (every 2 seconds)',
           fontsize=8, ha='center', style='italic', color=COLORS['gray_dark'])

    # Polarity indicators
    ax.text(2.5, 3.0, 'I+', fontsize=9, color='red', fontweight='bold')
    ax.text(2.5, 2.0, 'I-', fontsize=9, color='blue', fontweight='bold')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_component_location_diagram():
    """
    Create PCB component location diagram showing the
    recommended layout for the electronics hub.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(5, 5.7, 'PCB Layout Guidelines', fontsize=12,
           fontweight='bold', ha='center', color=COLORS['primary'])

    # PCB outline
    pcb = FancyBboxPatch((0.5, 0.5), 9.0, 5.0, boxstyle="round,pad=0.02",
                         facecolor='#c6f6d5', edgecolor='black', linewidth=2)
    ax.add_patch(pcb)

    # Analog section (shielded)
    analog_box = FancyBboxPatch((0.8, 2.5), 3.5, 2.7, boxstyle="round,pad=0.02",
                                facecolor='#ebf8ff', edgecolor=COLORS['accent'],
                                linewidth=1.5, linestyle='--')
    ax.add_patch(analog_box)
    ax.text(2.55, 5.0, 'ANALOG SECTION', fontsize=8, ha='center',
           fontweight='bold', color=COLORS['accent'])
    ax.text(2.55, 4.7, '(Shielded)', fontsize=7, ha='center', color=COLORS['accent'])

    # Analog components
    draw_block(ax, 1.5, 4.0, 1.0, 0.6, 'Preamp\nAD620', COLORS['light_green'], fontsize=7)
    draw_block(ax, 3.2, 4.0, 1.0, 0.6, 'Inst Amp\nINA128', COLORS['light_green'], fontsize=7)
    draw_block(ax, 1.5, 3.0, 1.0, 0.6, 'Vref\n2.5V', COLORS['light_orange'], fontsize=7)
    draw_block(ax, 3.2, 3.0, 1.0, 0.6, 'ERT\nSource', COLORS['light_orange'], fontsize=7)

    # Digital section
    digital_box = FancyBboxPatch((4.5, 2.5), 3.5, 2.7, boxstyle="round,pad=0.02",
                                 facecolor='#e9d8fd', edgecolor=COLORS['purple'],
                                 linewidth=1.5, linestyle='--')
    ax.add_patch(digital_box)
    ax.text(6.25, 5.0, 'DIGITAL SECTION', fontsize=8, ha='center',
           fontweight='bold', color=COLORS['purple'])

    # Digital components
    draw_block(ax, 5.3, 4.0, 1.2, 0.6, 'MCU\nESP32', COLORS['light_purple'], fontsize=7)
    draw_block(ax, 7.0, 4.0, 1.0, 0.6, 'ADC\nADS1256', COLORS['light_blue'], fontsize=7)
    draw_block(ax, 5.3, 3.0, 1.0, 0.6, 'DDS\nAD9833', COLORS['light_blue'], fontsize=7)
    draw_block(ax, 7.0, 3.0, 1.0, 0.6, 'MUX\nCD4051', COLORS['light_blue'], fontsize=7)

    # Power section
    power_box = FancyBboxPatch((8.2, 2.5), 1.0, 2.7, boxstyle="round,pad=0.02",
                               facecolor='#fed7d7', edgecolor=COLORS['warning'],
                               linewidth=1.5, linestyle='--')
    ax.add_patch(power_box)
    ax.text(8.7, 5.0, 'PWR', fontsize=7, ha='center',
           fontweight='bold', color=COLORS['warning'])
    draw_block(ax, 8.7, 4.0, 0.6, 0.5, '5V', COLORS['light_red'], fontsize=7)
    draw_block(ax, 8.7, 3.3, 0.6, 0.5, '3.3V', COLORS['light_red'], fontsize=7)

    # Ground plane indication
    ax.fill_between([0.8, 8.8], [0.7, 0.7], [2.3, 2.3], color=COLORS['gray_light'], alpha=0.5)
    ax.text(4.8, 1.5, 'GROUND PLANE (Bottom Layer)', fontsize=9, ha='center',
           fontweight='bold', color=COLORS['gray_dark'])

    # Star ground point
    ax.add_patch(Circle((4.3, 1.0), 0.15, facecolor=COLORS['success'],
                        edgecolor='black', lw=2))
    ax.text(4.3, 0.6, 'Star GND', fontsize=7, ha='center', color=COLORS['success'])

    # Separation line between analog/digital grounds
    ax.plot([4.3, 4.3], [1.0, 2.3], 'g--', lw=2)
    ax.text(4.1, 1.8, 'A/D\nSplit', fontsize=6, ha='right', color=COLORS['success'])

    # Connectors
    draw_block(ax, 1.5, 1.5, 1.2, 0.5, 'Input\nConnector', COLORS['gray_light'], fontsize=7)
    draw_block(ax, 7.5, 1.5, 1.2, 0.5, 'USB\nPort', COLORS['gray_light'], fontsize=7)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


# ============================================================================
# MAIN PDF GENERATION
# ============================================================================

def main():
    """Generate the complete Section 06 PDF."""

    print("Generating Section 06: Electronics and Circuits PDF...")
    print("-" * 50)

    # Initialize the PDF builder
    builder = SectionPDFBuilder(
        section_num=6,
        title="Electronics & Circuits"
    )

    # ========================================================================
    # TITLE BLOCK
    # ========================================================================
    builder.add_title_block(
        subtitle="MIT TX/RX Chains, ERT Circuits, Power Distribution, and Signal Processing"
    )

    # ========================================================================
    # INTRODUCTION
    # ========================================================================
    builder.add_section_header("6.1 Overview", level=1)

    builder.add_body_text(
        "This section consolidates all circuit designs for the HIRT system, including "
        "the MIT (Magneto-Inductive Tomography) circuits, ERT (Electrical Resistivity "
        "Tomography) circuits, and base hub electronics. The design uses <b>centralized "
        "electronics with passive probes</b>: all active components reside in the surface "
        "hub, while probes contain only coils and electrodes.",
        first_paragraph=True
    )

    builder.add_body_text(
        "This architecture offers significant advantages: lower per-probe cost, easier "
        "maintenance, better reliability (passive probes are more robust), centralized "
        "firmware updates, and simplified probe construction. The trade-off is increased "
        "cabling complexity for the multi-probe harness."
    )

    # ========================================================================
    # SYSTEM BLOCK DIAGRAM
    # ========================================================================
    builder.add_section_header("6.2 System Block Diagram", level=1)

    print("  Generating system block diagram...")
    fig_system = create_system_block_diagram()
    builder.add_figure(
        fig_system,
        "Complete HIRT system architecture showing the central electronics hub, "
        "zone wiring strategy, and passive probe connections. All signal processing "
        "occurs in the hub; probes are purely passive sensors.",
        height=CONTENT_WIDTH * 0.65
    )

    builder.add_body_text(
        "The central hub contains: (1) MCU for control and DSP, (2) MIT transmitter "
        "chain (DDS + power driver), (3) MIT receiver chain (preamp + instrumentation "
        "amplifier + ADC), (4) ERT current source and voltage measurement, and "
        "(5) high-density analog multiplexers. These connect via DB25 trunk cables "
        "to passive **Zone Hubs**, which then distribute signals to individual probes.",
        first_paragraph=True
    )

    # ========================================================================
    # MIT TRANSMITTER CHAIN
    # ========================================================================
    builder.add_section_header("6.3 MIT Transmit (TX) Chain", level=1)

    print("  Generating MIT TX chain diagram...")
    fig_tx = create_mit_tx_chain()
    builder.add_figure(
        fig_tx,
        "MIT transmitter signal chain from MCU through DDS, filtering, and power "
        "driver to the TX coil. Operating frequency range is 2-50 kHz.",
        height=CONTENT_WIDTH * 0.3
    )

    builder.add_section_header("6.3.1 DDS Sine Generator", level=2)

    builder.add_body_text(
        "The transmit chain begins with a Direct Digital Synthesis (DDS) generator, "
        "the AD9833. This IC generates precise sine waves from 0.1 Hz to 12.5 MHz "
        "with 28-bit frequency resolution. For HIRT, the operating range is 2-50 kHz, "
        "with lower frequencies (2-10 kHz) preferred for deeper penetration.",
        first_paragraph=True
    )

    # DDS specifications table
    dds_data = [
        ['Parameter', 'Specification'],
        ['Part Number', 'AD9833BRMZ'],
        ['Frequency Range', '0.1 Hz to 12.5 MHz'],
        ['Output Level', '0.6 V RMS sine wave'],
        ['Interface', 'SPI (10 MHz max)'],
        ['Resolution', '28-bit frequency, 12-bit phase'],
        ['Power Supply', '2.3-5.5V, 3 mA typical'],
    ]
    builder.add_table(dds_data, caption="AD9833 DDS Generator Specifications")

    builder.add_section_header("6.3.2 TX Power Driver", level=2)

    builder.add_body_text(
        "The DDS output is amplified by an OPA454 or OPA2277 operational amplifier "
        "configured as a non-inverting amplifier. The gain is set by external resistors "
        "to provide 2-10x amplification, delivering 10-50 mA into the TX coil at "
        "0-5 V RMS. The driver bandwidth (2.5 MHz) easily accommodates the operating "
        "frequency range.",
        first_paragraph=True
    )

    # TX driver specifications
    tx_driver_data = [
        ['Parameter', 'OPA454', 'OPA2277'],
        ['Gain (configurable)', '2-10x', '2-10x'],
        ['Output Current', '+/-2.5 A peak', '+/-20 mA'],
        ['Bandwidth', '2.5 MHz', '1 MHz'],
        ['Slew Rate', '19 V/us', '0.8 V/us'],
        ['Power Supply', '+/-5V to +/-40V', '+/-2V to +/-18V'],
    ]
    builder.add_table(tx_driver_data, caption="TX Driver Op-Amp Options")

    # ========================================================================
    # MIT RECEIVER CHAIN
    # ========================================================================
    builder.add_section_header("6.4 MIT Receive (RX) Chain", level=1)

    print("  Generating MIT RX chain diagram...")
    fig_rx = create_mit_rx_chain()
    builder.add_figure(
        fig_rx,
        "MIT receiver signal chain from RX coil through preamplifier, instrumentation "
        "amplifier, band-pass filter, and ADC. Total gain is approximately 1000x (60 dB). "
        "Lock-in detection extracts the signal at the reference frequency.",
        height=CONTENT_WIDTH * 0.35
    )

    builder.add_section_header("6.4.1 RX Preamplifier (AD620)", level=2)

    builder.add_body_text(
        "The receive coil output (microvolts to millivolts) connects to an AD620 "
        "instrumentation amplifier. This device provides excellent common-mode "
        "rejection (100 dB minimum) and low noise (9 nV/sqrt(Hz)). Gain is set by "
        "a single resistor R_G according to: G = (49.4k / R_G) + 1.",
        first_paragraph=True
    )

    # Gain setting table
    gain_data = [
        ['Target Gain', 'R_G Value', 'Notes'],
        ['G = 10', '5.49 k-ohm', 'Standard setting'],
        ['G = 100', '499 ohm', 'High sensitivity'],
        ['G = 1000', '49.9 ohm', 'Maximum gain'],
    ]
    builder.add_table(gain_data, caption="AD620 Gain Resistor Selection")

    builder.add_section_header("6.4.2 Instrumentation Amplifier (INA128)", level=2)

    builder.add_body_text(
        "A second gain stage uses the INA128, providing additional amplification "
        "(typically 10-100x) and further common-mode rejection. The combined gain "
        "of both stages can reach 10,000x (80 dB), sufficient to amplify microvolt "
        "signals to ADC input levels.",
        first_paragraph=True
    )

    builder.add_section_header("6.4.3 Signal Level Progression", level=2)

    print("  Generating signal level progression chart...")
    fig_signal = create_signal_level_progression()
    builder.add_figure(
        fig_signal,
        "Signal level progression through the MIT RX chain. The signal starts at "
        "approximately 1 uV at the coil and is amplified to 0.8 mV at the ADC input. "
        "The noise floor also rises but at a slower rate, maintaining adequate SNR.",
        height=CONTENT_WIDTH * 0.45
    )

    # ========================================================================
    # LOCK-IN DETECTION
    # ========================================================================
    builder.add_section_header("6.5 Lock-In Detection", level=1)

    builder.add_body_text(
        "Lock-in detection is essential for extracting weak MIT signals from noise. "
        "The HIRT system uses <b>digital lock-in detection</b> implemented in the MCU, "
        "which offers flexibility, software configurability, and no analog drift. "
        "The technique multiplies the received signal by reference sine and cosine "
        "waveforms at the excitation frequency, then low-pass filters to extract "
        "the in-phase (I) and quadrature (Q) components.",
        first_paragraph=True
    )

    print("  Generating lock-in detection diagram...")
    fig_lockin = create_lock_in_detection_diagram()
    builder.add_figure(
        fig_lockin,
        "Digital lock-in detection block diagram. The RX signal is multiplied by "
        "reference sine (Q-channel) and cosine (I-channel) signals, then filtered "
        "and integrated. Amplitude A = sqrt(I^2 + Q^2) and phase Phi = atan2(Q,I) "
        "are computed from the I/Q outputs.",
        height=CONTENT_WIDTH * 0.5
    )

    builder.add_body_text(
        "The lock-in algorithm provides exceptional noise rejection (> 40 dB improvement) "
        "by responding only to signals at the exact reference frequency. This allows "
        "detection of MIT signals buried deep in noise, critical for weak responses "
        "from distant or low-conductivity targets."
    )

    # Lock-in code example
    builder.add_section_header("6.5.1 Digital Lock-In Algorithm", level=2)

    builder.add_code_block(
        "// Simplified digital lock-in implementation\n"
        "float I_sum = 0, Q_sum = 0;\n"
        "for (int i = 0; i < N_samples; i++) {\n"
        "    float sample = read_adc();\n"
        "    float ref_I = sin(2 * PI * f * i / Fs);\n"
        "    float ref_Q = cos(2 * PI * f * i / Fs);\n"
        "    I_sum += sample * ref_I;\n"
        "    Q_sum += sample * ref_Q;\n"
        "}\n"
        "float amplitude = sqrt(I_sum*I_sum + Q_sum*Q_sum) / N_samples;\n"
        "float phase = atan2(Q_sum, I_sum);"
    )

    # ========================================================================
    # ERT CURRENT SOURCE
    # ========================================================================
    builder.add_section_header("6.6 ERT Current Source (Howland Pump)", level=1)

    builder.add_body_text(
        "The ERT subsystem injects controlled current into the ground through electrode "
        "rings and measures the resulting voltage distribution. The current source uses "
        "a <b>Howland current pump</b> topology, which provides high output impedance "
        "(>1 M-ohm) and stable current regardless of load variations.",
        first_paragraph=True
    )

    print("  Generating ERT current source schematic...")
    fig_ert = create_ert_current_source()
    builder.add_figure(
        fig_ert,
        "Howland current pump circuit for ERT measurements. When R1=R2=R3=R4, "
        "the output current is I_out = V_in / R_sense, independent of load impedance. "
        "A precision 2.5V reference (REF5025) provides stable input voltage.",
        height=CONTENT_WIDTH * 0.55
    )

    # ERT specifications
    ert_spec_data = [
        ['Parameter', 'Specification'],
        ['Output Current', '0.5-2 mA (adjustable)'],
        ['Current Accuracy', '+/- 5%'],
        ['Compliance Voltage', '+/- 10V minimum'],
        ['Output Impedance', '> 1 M-ohm'],
        ['Load Range', '100 ohm - 10 k-ohm'],
        ['Polarity Reversal', 'Programmable (0.5 Hz)'],
    ]
    builder.add_table(ert_spec_data, caption="ERT Current Source Specifications")

    builder.add_section_header("6.6.1 Polarity Reversal", level=2)

    builder.add_body_text(
        "Electrode polarization causes measurement drift in DC systems. To mitigate this, "
        "HIRT reverses the current polarity every 2 seconds using a DPDT relay (G5V-2-H1) "
        "or solid-state switch (ADG1219). Positive and negative measurements are averaged "
        "to cancel polarization effects.",
        first_paragraph=True
    )

    print("  Generating ERT polarity reversal diagram...")
    fig_polarity = create_ert_polarity_reversal()
    builder.add_figure(
        fig_polarity,
        "ERT polarity reversal circuit using a DPDT relay. The relay alternates "
        "current direction between Ring A and Ring B every 2 seconds, eliminating "
        "electrode polarization artifacts from the measurement.",
        height=CONTENT_WIDTH * 0.5
    )

    # ========================================================================
    # ADC INTERFACE
    # ========================================================================
    builder.add_section_header("6.7 ADC Interface", level=1)

    print("  Generating ADC interface diagram...")
    fig_adc = create_adc_interface_diagram()
    builder.add_figure(
        fig_adc,
        "ADS1256 ADC interface showing differential analog inputs for MIT and ERT "
        "signals, SPI connection to MCU, and reference voltage. The 24-bit resolution "
        "and internal PGA provide excellent dynamic range.",
        height=CONTENT_WIDTH * 0.5
    )

    # ADC specifications
    adc_data = [
        ['Parameter', 'ADS1256 Specification'],
        ['Resolution', '24 bits'],
        ['Sample Rate', '30 kSPS maximum'],
        ['Noise', '0.6 uV RMS (at 100 SPS)'],
        ['Interface', 'SPI'],
        ['Internal PGA', '1, 2, 4, 8, 16, 32, 64x'],
        ['Input Channels', '8 single-ended or 4 differential'],
    ]
    builder.add_table(adc_data, caption="ADS1256 ADC Specifications")

    # ========================================================================
    # MULTIPLEXER TOPOLOGY
    # ========================================================================
    builder.add_section_header("6.8 Multiplexer Switching", level=1)

    builder.add_body_text(
        "With up to 24 probes in a typical array, the HIRT system requires a scalable "
        "multiplexing strategy. The design uses cascaded CD4051 8:1 analog multiplexers "
        "controlled by MCU GPIO pins. Each TX and RX signal path has its own multiplexer "
        "chain, allowing independent selection of transmit and receive probes.",
        first_paragraph=True
    )

    print("  Generating multiplexer switching diagram...")
    fig_mux = create_multiplexer_switching_diagram()
    builder.add_figure(
        fig_mux,
        "Multiplexer switching topology for probe selection. Separate TX and RX "
        "multiplexers allow any TX-RX probe combination to be measured. MCU GPIO "
        "controls multiplexer address lines (A0-A2).",
        height=CONTENT_WIDTH * 0.6
    )

    # ========================================================================
    # POWER DISTRIBUTION
    # ========================================================================
    builder.add_section_header("6.9 Power Distribution", level=1)

    print("  Generating power distribution diagram...")
    fig_power = create_power_distribution()
    builder.add_figure(
        fig_power,
        "Power distribution from 12V battery through regulators to various subsystems. "
        "Total system current is approximately 200 mA, providing several hours of "
        "operation from a 3S LiPo battery.",
        height=CONTENT_WIDTH * 0.5
    )

    # Power rail table
    power_data = [
        ['Rail', 'Voltage', 'Current', 'Purpose'],
        ['+12V (Battery)', '11.1-12.6V', '200 mA total', 'System input'],
        ['+5V (LDO)', '5.0V', '50 mA', 'Power driver, references'],
        ['+3.3V (LDO)', '3.3V', '100 mA', 'MCU, ADC, digital'],
        ['-5V (Charge pump)', '-5.0V', '10 mA', 'Op-amp negative rail'],
    ]
    builder.add_table(power_data, caption="Power Rail Specifications")

    # ========================================================================
    # NOISE FILTERING
    # ========================================================================
    builder.add_section_header("6.10 Noise Filtering Stages", level=1)

    print("  Generating noise filtering diagram...")
    fig_filter = create_noise_filtering_diagram()
    builder.add_figure(
        fig_filter,
        "Cascaded noise filtering stages in the signal chain. Each stage targets "
        "specific noise sources: EMI, DC offsets, wideband noise, aliasing, and "
        "quantization noise. Combined rejection exceeds 60 dB.",
        height=CONTENT_WIDTH * 0.4
    )

    builder.add_body_text(
        "Noise reduction is critical for MIT measurements where signal levels can be "
        "in the microvolt range. The filtering strategy employs multiple stages:",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "<b>Input RC filter (fc=100kHz):</b> Blocks RF interference and high-frequency EMI",
        "<b>High-pass filter (fc=100Hz):</b> Removes DC offsets and 60Hz pickup",
        "<b>Active band-pass (1-50kHz):</b> Passes only the measurement band",
        "<b>Anti-aliasing filter (fc=15kHz):</b> Prevents aliasing in ADC sampling",
        "<b>Digital matched filter:</b> Optimizes SNR in post-processing",
    ])

    # ========================================================================
    # GROUND LOOP PREVENTION
    # ========================================================================
    builder.add_section_header("6.11 Shielding and Ground Loop Prevention", level=1)

    print("  Generating ground loop prevention diagram...")
    fig_ground = create_ground_loop_prevention()
    builder.add_figure(
        fig_ground,
        "Ground loop prevention techniques. (a) shows the problem: multiple ground "
        "paths create loops that pick up EMI. (b) shows the solution: star grounding "
        "with a single connection point eliminates loops.",
        height=CONTENT_WIDTH * 0.45
    )

    builder.add_body_text(
        "Proper grounding is essential for achieving low-noise measurements. The HIRT "
        "design follows these best practices:",
        first_paragraph=True
    )

    builder.add_bullet_list([
        "<b>Star grounding:</b> All ground returns connect at a single point near the ADC",
        "<b>Separate ground planes:</b> Analog and digital grounds are split on the PCB",
        "<b>Shielded cables:</b> All signal cables use twisted-pair with overall shield",
        "<b>Single-point shield termination:</b> Shields grounded at hub end only",
        "<b>Minimum loop area:</b> Signal and return paths routed together",
    ])

    # ========================================================================
    # PCB LAYOUT GUIDELINES
    # ========================================================================
    builder.add_section_header("6.12 PCB Layout Guidelines", level=1)

    print("  Generating PCB layout diagram...")
    fig_pcb = create_component_location_diagram()
    builder.add_figure(
        fig_pcb,
        "Recommended PCB layout showing component placement zones. The analog section "
        "(shielded) is physically separated from digital circuits. Ground plane spans "
        "the bottom layer with analog/digital split connected at a star point.",
        height=CONTENT_WIDTH * 0.6
    )

    builder.add_section_header("6.12.1 Layout Rules", level=2)

    builder.add_numbered_list([
        "Place bypass capacitors (100nF) within 5mm of each IC power pin",
        "Route analog signals with minimum trace length; avoid crossing digital signals",
        "Use differential pairs for RX coil signals with matched length",
        "Keep TX and RX signal paths physically separated (>10mm)",
        "Provide shielding (copper pour) around sensitive analog circuits",
        "Use wide traces (>20 mil) for power distribution",
        "Connect analog and digital ground planes at a single star point near the ADC",
    ])

    # ========================================================================
    # COMPONENT SUMMARY
    # ========================================================================
    builder.add_section_header("6.13 Key Component Summary", level=1)

    # Component table
    component_data = [
        ['Component', 'Part Number', 'Function', 'Package'],
        ['DDS', 'AD9833BRMZ', 'Signal generator', 'MSOP-10'],
        ['TX Driver', 'OPA454AIDDAR', 'Coil driver', 'SOIC-8'],
        ['Preamp', 'AD620ARZ', 'RX preamplifier', 'SOIC-8'],
        ['Inst Amp', 'INA128PAG4', 'Differential amp', 'DIP-8'],
        ['ADC', 'ADS1256IDBR', '24-bit ADC', 'SSOP-28'],
        ['Multiplexer', 'CD4051BE', '8-channel mux', 'DIP-16'],
        ['MCU', 'ESP32-WROOM-32', 'Controller', 'Module'],
        ['V-Reference', 'REF5025AIDGKR', '2.5V reference', 'SOIC-8'],
        ['LDO', 'AMS1117-3.3', '3.3V regulator', 'SOT-223'],
    ]
    builder.add_table(component_data, caption="Key IC Components")

    # ========================================================================
    # CONNECTOR PINOUTS
    # ========================================================================
    builder.add_section_header("6.14 Connector Pinouts (Probe-to-Zone)", level=1)

    pinout_data = [
        ['Pin', 'Signal', 'Description'],
        ['1', 'TX+', 'To probe TX coil'],
        ['2', 'TX-', 'Return path'],
        ['3', 'RX+', 'Differential RX'],
        ['4', 'RX-', 'Differential RX return'],
        ['5', 'Guard', 'Analog ground'],
        ['6', 'Ring A', 'Upper ERT electrode'],
        ['7', 'Ring B', 'Mid ERT electrode'],
        ['8', 'Ring C', 'Deep electrode'],
        ['9', 'ID Sense', 'Auto-ID resistor'],
        ['10-11', 'Spare', 'Reserved'],
        ['12', 'Shield', 'Cable shield clamp'],
    ]
    builder.add_table(pinout_data, caption="12-pin Probe Connector Pinout")

    # ========================================================================
    # SAFETY CONSIDERATIONS
    # ========================================================================
    builder.add_section_header("6.15 Safety Considerations", level=1)

    builder.add_warning_box(
        "ELECTRICAL SAFETY",
        [
            "Maximum output current limited to 5 mA by design",
            "Compliance voltage restricted to +/-12V (safe for soil contact)",
            "Include 10 mA fast-blow fuse on ERT output",
            "Use opto-isolated relay control for polarity switching",
            "Ensure proper earth ground connection for safety",
        ]
    )

    # ========================================================================
    # TROUBLESHOOTING
    # ========================================================================
    builder.add_section_header("6.16 Troubleshooting Guide", level=1)

    trouble_data = [
        ['Symptom', 'Likely Cause', 'Solution'],
        ['No TX output', 'DDS not initialized', 'Check SPI connection, verify clock'],
        ['Weak RX signal', 'Gain too low', 'Verify R_G resistor, check amplifier rails'],
        ['Noisy readings', 'Ground loops', 'Implement star grounding, shield cables'],
        ['No ERT current', 'Open circuit', 'Check electrode contact, verify relay state'],
        ['Current drift', 'Reference unstable', 'Verify Vref output, add bypass caps'],
        ['ADC errors', 'SPI timing', 'Reduce SPI clock speed, check DRDY timing'],
    ]
    builder.add_table(trouble_data, caption="Common Issues and Solutions")

    # ========================================================================
    # REFERENCES
    # ========================================================================
    builder.add_references([
        "[1] Analog Devices. AD9833 Programmable Waveform Generator Data Sheet. "
        "Rev. F, 2019.",

        "[2] Texas Instruments. OPA454 High-Voltage, High-Current Operational Amplifier. "
        "SBOS328D, 2018.",

        "[3] Analog Devices. AD620 Low Cost, Low Power Instrumentation Amplifier. "
        "Rev. H, 2011.",

        "[4] Texas Instruments. INA128 Precision, Low Power Instrumentation Amplifiers. "
        "SBOS051B, 2005.",

        "[5] Texas Instruments. ADS1256 Very Low Noise, 24-Bit Analog-to-Digital "
        "Converter. SBAS288K, 2013.",

        "[6] Horowitz, P. and Hill, W. The Art of Electronics, 3rd Edition. "
        "Cambridge University Press, 2015. Chapter 5: Precision Circuits.",

        "[7] Ott, H.W. Electromagnetic Compatibility Engineering. Wiley, 2009. "
        "Chapter 3: Grounding.",
    ])

    # ========================================================================
    # BUILD PDF
    # ========================================================================
    print("-" * 50)
    output_path = builder.build()
    print(f"PDF generated successfully: {output_path}")
    print(f"Total figures: {builder.figure_count}")
    print(f"Total tables: {builder.table_count}")

    return output_path


if __name__ == "__main__":
    main()
