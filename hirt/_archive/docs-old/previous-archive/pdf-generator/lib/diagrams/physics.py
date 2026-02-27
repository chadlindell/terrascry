"""
HIRT PDF Generator - Physics Diagrams Module

Functions for creating waveforms, frequency plots, skin depth curves,
and other physics-related visualizations.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon
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
}

# Soil conductivity values (S/m)
SOIL_CONDUCTIVITY = {
    'dry_sand': 0.001,
    'moist_sand': 0.01,
    'clay': 0.1,
    'saturated_clay': 0.5,
    'saline': 1.0,
}

# Soil labels and colors for plots
SOIL_TYPES = [
    ('Dry Sand', 0.001, '#d4a373'),
    ('Moist Sand', 0.01, '#ed8936'),
    ('Clay', 0.1, '#805ad5'),
    ('Saturated Clay', 0.5, '#2c5282'),
    ('Saline/Contaminated', 1.0, '#c53030'),
]


def calculate_skin_depth(frequency, conductivity, mu_r=1.0):
    """
    Calculate electromagnetic skin depth.

    delta = sqrt(2 / (omega * mu * sigma))

    Args:
        frequency: Frequency in Hz
        conductivity: Electrical conductivity in S/m
        mu_r: Relative permeability (default: 1.0)

    Returns:
        Skin depth in meters
    """
    mu_0 = 4 * np.pi * 1e-7  # H/m
    mu = mu_r * mu_0
    omega = 2 * np.pi * frequency
    return np.sqrt(2 / (omega * mu * conductivity))


def create_skin_depth_plot():
    """
    Create skin depth vs frequency curves for different soil types.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    frequencies = np.logspace(1, 5, 100)  # 10 Hz to 100 kHz

    for label, sigma, color in SOIL_TYPES:
        depths = [calculate_skin_depth(f, sigma) for f in frequencies]
        ax.loglog(frequencies/1000, depths, label=f'{label} (σ={sigma} S/m)',
                 color=color, linewidth=2)

    # HIRT operating range
    ax.axvspan(2, 50, alpha=0.15, color=COLORS['success'], label='HIRT Range')
    ax.axvline(10, color=COLORS['success'], linestyle='--', alpha=0.7)
    ax.text(10, 0.3, '10 kHz\n(typical)', fontsize=8, ha='center',
           color=COLORS['success'])

    # Probe spacing reference
    ax.axhline(3, color=COLORS['gray_med'], linestyle=':', alpha=0.7)
    ax.text(0.015, 3.3, 'Typical probe spacing (3m)', fontsize=8,
           color=COLORS['gray_med'])

    ax.set_xlabel('Frequency (kHz)', fontsize=10)
    ax.set_ylabel('Skin Depth (m)', fontsize=10)
    ax.set_title('Electromagnetic Skin Depth vs. Frequency', fontsize=11,
                fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0.1, 1000)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_tx_rx_waveforms():
    """
    Create TX and RX waveform comparison showing amplitude and phase.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    # Time axis (2 cycles at 10 kHz)
    t = np.linspace(0, 0.2, 1000)  # ms
    f = 10  # kHz (so period = 0.1 ms)

    # TX signal
    tx = np.sin(2 * np.pi * f * t)

    # RX signal (attenuated and phase shifted)
    amplitude = 0.3
    phase_shift = np.pi / 6  # 30 degrees
    rx = amplitude * np.sin(2 * np.pi * f * t - phase_shift)

    # Plot TX
    ax1.plot(t, tx, color=COLORS['success'], linewidth=2, label='TX Signal')
    ax1.fill_between(t, tx, alpha=0.2, color=COLORS['success'])
    ax1.set_ylabel('TX Amplitude (V)', fontsize=9)
    ax1.set_ylim(-1.3, 1.3)
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title('MIT Transmit and Receive Waveforms (10 kHz)', fontsize=11,
                 fontweight='bold', color=COLORS['primary'])

    # Plot RX
    ax2.plot(t, rx, color=COLORS['accent'], linewidth=2, label='RX Signal')
    ax2.fill_between(t, rx, alpha=0.2, color=COLORS['accent'])
    ax2.set_ylabel('RX Amplitude (mV)', fontsize=9)
    ax2.set_xlabel('Time (ms)', fontsize=9)
    ax2.set_ylim(-0.5, 0.5)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)

    # Phase shift annotation
    phase_time = phase_shift / (2 * np.pi * f)
    ax2.annotate('', xy=(0.025 + phase_time, 0.3), xytext=(0.025, 0.3),
                arrowprops=dict(arrowstyle='<->', color=COLORS['orange'], lw=1.5))
    ax2.text(0.025 + phase_time/2, 0.38, 'Phase\nLag', fontsize=7, ha='center',
            color=COLORS['orange'])

    # Amplitude annotation
    ax2.annotate('', xy=(0.075, amplitude), xytext=(0.075, 0),
                arrowprops=dict(arrowstyle='<->', color=COLORS['purple'], lw=1.5))
    ax2.text(0.085, amplitude/2, 'Amplitude\nRatio', fontsize=7, va='center',
            color=COLORS['purple'])

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_coupling_decay_plot():
    """
    Create 1/r³ coupling decay curves for MIT.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    r = np.linspace(0.5, 5, 100)

    # Different coupling models
    coupling_1r3 = 1 / r**3  # Near-field MIT
    coupling_1r2 = 1 / r**2  # Intermediate
    coupling_1r = 1 / r      # Far-field

    # Normalize to 1 at r=1
    ax.semilogy(r, coupling_1r3 / coupling_1r3[np.argmin(np.abs(r-1))],
               label='1/r³ (MIT near-field)', color=COLORS['success'], lw=2)
    ax.semilogy(r, coupling_1r2 / coupling_1r2[np.argmin(np.abs(r-1))],
               label='1/r² (ERT geometric)', color=COLORS['accent'], lw=2, linestyle='--')
    ax.semilogy(r, coupling_1r / coupling_1r[np.argmin(np.abs(r-1))],
               label='1/r (reference)', color=COLORS['gray_med'], lw=1.5, linestyle=':')

    # Practical range
    ax.axvspan(1.5, 3.5, alpha=0.15, color=COLORS['success'], label='Typical spacing')

    # Detection threshold
    ax.axhline(0.01, color=COLORS['warning'], linestyle='--', alpha=0.7)
    ax.text(4.5, 0.012, 'Detection\nThreshold', fontsize=8, ha='center',
           color=COLORS['warning'])

    ax.set_xlabel('Distance (m)', fontsize=10)
    ax.set_ylabel('Relative Signal Strength', fontsize=10)
    ax.set_title('Magnetic Coupling vs. Distance', fontsize=11,
                fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.set_xlim(0.5, 5)
    ax.set_ylim(0.001, 10)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_ert_geometric_factor():
    """
    Create ERT geometric factor visualization.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(-2, 10)
    ax.set_ylim(-3, 2)
    ax.set_aspect('equal')

    # Ground surface
    ax.axhline(0, color='#654321', linewidth=2.5)
    ax.fill_between([-2, 10], [0, 0], [-3, -3], color='#d4a373', alpha=0.3)

    # Electrodes (Wenner array)
    electrode_x = [0, 2, 4, 6]
    labels = ['C1', 'P1', 'P2', 'C2']
    colors_el = [COLORS['warning'], COLORS['accent'], COLORS['accent'], COLORS['warning']]

    for x, label, color in zip(electrode_x, labels, colors_el):
        ax.add_patch(Circle((x, 0), 0.15, color=color, ec='black', lw=1, zorder=10))
        ax.text(x, 0.5, label, ha='center', fontsize=9, fontweight='bold')

    # Current flow lines
    theta = np.linspace(-np.pi, 0, 50)
    for r_scale in [1, 1.5, 2, 2.5]:
        # From C1
        x1 = 0 + r_scale * np.cos(theta)
        y1 = r_scale * np.sin(theta)
        ax.plot(x1, y1, 'r-', alpha=0.3, lw=1)
        # To C2
        x2 = 6 + r_scale * np.cos(theta + np.pi)
        y2 = r_scale * np.sin(theta + np.pi)
        ax.plot(x2, y2, 'r-', alpha=0.3, lw=1)

    # Equipotential lines
    for x_offset in [1, 2, 3, 4, 5]:
        y_eq = np.linspace(-2.5, 0, 20)
        ax.plot([x_offset]*len(y_eq), y_eq, 'b:', alpha=0.3, lw=1)

    # Sensitivity region
    from matplotlib.patches import Ellipse
    ellipse = Ellipse((3, -1), 4, 2, color=COLORS['success'], alpha=0.15)
    ax.add_patch(ellipse)
    ax.text(3, -1, 'High\nSensitivity', ha='center', va='center',
           fontsize=8, color='#276749')

    # Annotations
    ax.annotate('Current\nInjection', xy=(0, 0), xytext=(-1.5, 1.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['warning']),
               fontsize=8, ha='center')
    ax.annotate('Voltage\nMeasurement', xy=(3, 0), xytext=(3, 1.2),
               arrowprops=dict(arrowstyle='->', color=COLORS['accent']),
               fontsize=8, ha='center')

    # Equation
    ax.text(8, -1, 'K = 2πa\n(Wenner)', fontsize=9, ha='center',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['gray_med']))

    ax.set_title('ERT Wenner Array - Geometric Factor', fontsize=11,
                fontweight='bold', color=COLORS['primary'])
    ax.set_xlabel('Distance (m)', fontsize=9)
    ax.set_ylabel('Depth (m)', fontsize=9)
    ax.set_yticks([0, -1, -2])
    ax.set_yticklabels(['0', '1', '2'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_frequency_response():
    """
    Create multi-frequency response plot showing target signatures.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    frequencies = np.logspace(2, 5, 100)  # 100 Hz to 100 kHz

    # Simulated responses for different targets
    # Metal (high conductivity) - responds at lower frequencies
    metal_response = 1 / (1 + (frequencies / 5000)**2) * 0.8

    # Soil disturbance (low conductivity) - broader response
    soil_response = 0.3 * np.exp(-(np.log10(frequencies) - 3.5)**2 / 0.5)

    # Combined
    combined = metal_response + soil_response

    ax.semilogx(frequencies/1000, metal_response, label='Metal Target',
               color=COLORS['warning'], lw=2)
    ax.semilogx(frequencies/1000, soil_response, label='Soil Disturbance',
               color=COLORS['purple'], lw=2, linestyle='--')
    ax.semilogx(frequencies/1000, combined, label='Combined Response',
               color=COLORS['gray_dark'], lw=2, linestyle=':')

    # HIRT sweep range
    ax.axvspan(2, 50, alpha=0.15, color=COLORS['success'])
    ax.text(10, 0.85, 'HIRT\nSweep Range', fontsize=8, ha='center',
           color=COLORS['success'])

    ax.set_xlabel('Frequency (kHz)', fontsize=10)
    ax.set_ylabel('Relative Response', fontsize=10)
    ax.set_title('Multi-Frequency Target Discrimination', fontsize=11,
                fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_ray_path_comparison():
    """
    Create surface vs crosshole ray path comparison.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # === Surface Method ===
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-4, 1)

    # Ground
    ax1.axhline(0, color='#654321', lw=2)
    ax1.fill_between([-3, 3], [0, 0], [-4, -4], color='#d4a373', alpha=0.3)

    # Surface sensors
    for x in [-2, -1, 0, 1, 2]:
        ax1.add_patch(Polygon([[x-0.2, 0], [x+0.2, 0], [x, 0.3]],
                             color=COLORS['secondary']))

    # Curved ray paths (semicircular approximation)
    for r in [1, 1.5, 2, 2.5, 3]:
        theta = np.linspace(0, np.pi, 50)
        x = r * np.cos(theta)
        y = -r * np.sin(theta) * 0.6
        ax1.plot(x, y, color=COLORS['accent'], alpha=0.3, lw=1)

    # Target
    ax1.add_patch(Circle((0, -2.5), 0.3, color=COLORS['warning']))
    ax1.text(0, -2.5, '?', ha='center', va='center', color='white', fontweight='bold')

    ax1.set_title('(a) Surface Ray Paths', fontsize=10, fontweight='bold',
                 color=COLORS['primary'])
    ax1.set_ylabel('Depth (m)', fontsize=9)
    ax1.set_yticks([0, -2, -4])
    ax1.set_yticklabels(['0', '2', '4'])
    ax1.set_xticks([])
    ax1.text(0, -3.5, 'Low sensitivity at depth', fontsize=8, ha='center',
            color=COLORS['gray_dark'], style='italic')

    # === Crosshole Method ===
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-4, 1)

    # Ground
    ax2.axhline(0, color='#654321', lw=2)
    ax2.fill_between([-3, 3], [0, 0], [-4, -4], color='#d4a373', alpha=0.3)

    # Probes
    for x in [-1.5, 1.5]:
        ax2.add_patch(Rectangle((x-0.1, -3), 0.2, 3, color=COLORS['secondary']))
        # Sensors
        for y in [-0.8, -1.6, -2.4]:
            ax2.add_patch(Circle((x, y), 0.12, color=COLORS['success']))

    # Straight ray paths (crosshole)
    for y1 in [-0.8, -1.6, -2.4]:
        for y2 in [-0.8, -1.6, -2.4]:
            ax2.plot([-1.5, 1.5], [y1, y2], color=COLORS['success'], alpha=0.4, lw=1)

    # Target (clearly intersected)
    ax2.add_patch(Circle((0, -2), 0.3, color=COLORS['warning']))
    ax2.text(0, -2, '\u2713', ha='center', va='center', color='white',
            fontweight='bold', fontsize=12)

    ax2.set_title('(b) Crosshole Ray Paths', fontsize=10, fontweight='bold',
                 color=COLORS['primary'])
    ax2.set_yticks([0, -2, -4])
    ax2.set_yticklabels(['0', '2', '4'])
    ax2.set_xticks([])
    ax2.text(0, -3.5, 'Direct paths through target', fontsize=8, ha='center',
            color=COLORS['success'], style='italic')

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_resolution_depth_plot():
    """
    Create resolution vs depth comparison plot.

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    depths = np.linspace(0.5, 6, 50)

    # Resolution models (smaller = better)
    surface_res = 0.3 + 0.5 * depths + 0.1 * depths**2
    crosshole_res = 0.2 + 0.15 * depths

    ax.plot(depths, surface_res, label='Surface Methods', color=COLORS['gray_med'],
           lw=2, linestyle='--')
    ax.plot(depths, crosshole_res, label='HIRT Crosshole', color=COLORS['success'],
           lw=2)

    # Target size reference
    ax.axhline(0.5, color=COLORS['warning'], linestyle=':', alpha=0.7)
    ax.text(5.5, 0.55, 'Typical UXB\ndiameter', fontsize=8, ha='center',
           color=COLORS['warning'])

    # Practical depth range
    ax.axvspan(2, 4, alpha=0.15, color=COLORS['orange'])
    ax.text(3, 2.5, 'UXB\nBurial\nRange', fontsize=8, ha='center',
           color=COLORS['orange'])

    ax.set_xlabel('Investigation Depth (m)', fontsize=10)
    ax.set_ylabel('Minimum Resolvable Feature (m)', fontsize=10)
    ax.set_title('Spatial Resolution vs. Depth', fontsize=11,
                fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_xlim(0.5, 6)
    ax.set_ylim(0, 3)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf
