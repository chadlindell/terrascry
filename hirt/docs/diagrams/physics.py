"""
HIRT PDF Generator - Physics Diagrams Module

Functions for creating waveforms, frequency plots, skin depth curves,
and other physics-related visualizations.

Enhanced with:
- Uncertainty bands for soil conductivity variations
- Temperature correction overlays
- Near-field/far-field regime boundaries
- Realistic noise visualization (pink noise + 50/60Hz pickup)
- Lock-in detection effects
- Physically accurate equipotential contours
- Sensitivity kernel visualization
- Coil orientation effects
- SNR threshold intersections
"""

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, Ellipse
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import numpy as np
from io import BytesIO

# Import utility functions from parent module
from . import WONG_PALETTE, draw_sphere_gradient, get_colorblind_cmap

# Color palette - use WONG_PALETTE for colorblind-friendly defaults
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

# Colorblind-safe colors from Wong palette for scientific plots
WONG_COLORS = {
    'orange': WONG_PALETTE['orange'],
    'sky_blue': WONG_PALETTE['sky_blue'],
    'bluish_green': WONG_PALETTE['bluish_green'],
    'yellow': WONG_PALETTE['yellow'],
    'blue': WONG_PALETTE['blue'],
    'vermillion': WONG_PALETTE['vermillion'],
    'reddish_purple': WONG_PALETTE['reddish_purple'],
}

# Soil conductivity values (S/m)
SOIL_CONDUCTIVITY = {
    'dry_sand': 0.001,
    'moist_sand': 0.01,
    'clay': 0.1,
    'saturated_clay': 0.5,
    'saline': 1.0,
}

# Soil labels and colors for plots (using Wong palette for colorblind safety)
SOIL_TYPES = [
    ('Dry Sand', 0.001, WONG_PALETTE['orange']),
    ('Moist Sand', 0.01, WONG_PALETTE['sky_blue']),
    ('Clay', 0.1, WONG_PALETTE['bluish_green']),
    ('Saturated Clay', 0.5, WONG_PALETTE['blue']),
    ('Saline/Contaminated', 1.0, WONG_PALETTE['vermillion']),
]

# Temperature coefficients for conductivity correction
# Conductivity increases ~2% per degree C above 25C reference
TEMP_COEFFICIENT = 0.02  # per degree Celsius

# Near-field / far-field transition (in skin depths)
NEAR_FIELD_LIMIT = 0.3  # r/delta < 0.3 is near-field
FAR_FIELD_LIMIT = 3.0   # r/delta > 3 is far-field


def generate_pink_noise(n_samples, amplitude=1.0, seed=None):
    """
    Generate pink (1/f) noise using the Voss-McCartney algorithm.

    Args:
        n_samples: Number of samples to generate
        amplitude: Peak amplitude of noise
        seed: Random seed for reproducibility

    Returns:
        numpy array of pink noise samples
    """
    if seed is not None:
        np.random.seed(seed)

    # Voss-McCartney algorithm for 1/f noise
    n_rows = 16  # Number of random sources
    array = np.zeros((n_rows, n_samples))

    for i in range(n_rows):
        period = 2 ** i
        for j in range(0, n_samples, period):
            value = np.random.uniform(-1, 1)
            array[i, j:min(j + period, n_samples)] = value

    pink = np.sum(array, axis=0) / n_rows
    return amplitude * pink / np.max(np.abs(pink))


def generate_mains_pickup(t, amplitude=0.05, freq=50, harmonics=[1, 3, 5]):
    """
    Generate realistic mains pickup (50/60 Hz) with harmonics.

    Args:
        t: Time array in seconds
        amplitude: Base amplitude of mains pickup
        freq: Mains frequency (50 Hz Europe, 60 Hz Americas)
        harmonics: List of harmonic numbers to include

    Returns:
        numpy array of mains pickup signal
    """
    signal = np.zeros_like(t)
    for h in harmonics:
        # Harmonic amplitude decreases with order
        harm_amp = amplitude / h
        # Add slight random phase for realism
        phase = np.random.uniform(0, 2 * np.pi)
        signal += harm_amp * np.sin(2 * np.pi * freq * h * t + phase)
    return signal


def calculate_temperature_correction(sigma_ref, temp, temp_ref=25.0):
    """
    Calculate temperature-corrected conductivity.

    Args:
        sigma_ref: Reference conductivity at temp_ref
        temp: Actual temperature in Celsius
        temp_ref: Reference temperature (default 25C)

    Returns:
        Temperature-corrected conductivity
    """
    return sigma_ref * (1 + TEMP_COEFFICIENT * (temp - temp_ref))


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


def create_skin_depth_plot(show_uncertainty=True, show_temp_correction=False,
                            show_regime_boundary=True, temp_range=None):
    """
    Create skin depth vs frequency curves for different soil types.

    Enhanced with:
    - Uncertainty bands for soil conductivity variations (+/- 50%)
    - Optional temperature correction overlay
    - Near-field vs far-field regime boundary markers

    Args:
        show_uncertainty: Show +/- 50% conductivity uncertainty bands
        show_temp_correction: Show temperature correction overlay
        show_regime_boundary: Mark near-field/far-field boundary
        temp_range: Tuple of (min_temp, max_temp) in Celsius for temp overlay

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    frequencies = np.logspace(1, 5, 100)  # 10 Hz to 100 kHz

    # Plot each soil type with uncertainty bands
    for label, sigma, color in SOIL_TYPES:
        depths = np.array([calculate_skin_depth(f, sigma) for f in frequencies])

        # Main line
        ax.loglog(frequencies/1000, depths, label=f'{label} (σ={sigma} S/m)',
                 color=color, linewidth=2, zorder=3)

        # Uncertainty bands: +/- 50% conductivity variation
        if show_uncertainty:
            depths_low = np.array([calculate_skin_depth(f, sigma * 0.5) for f in frequencies])
            depths_high = np.array([calculate_skin_depth(f, sigma * 1.5) for f in frequencies])
            ax.fill_between(frequencies/1000, depths_low, depths_high,
                           color=color, alpha=0.15, zorder=1)

    # Temperature correction overlay (if enabled)
    if show_temp_correction and temp_range is not None:
        temp_min, temp_max = temp_range
        # Show effect on middle soil type (Clay)
        sigma_ref = 0.1  # Clay
        color = WONG_PALETTE['reddish_purple']

        # Cold temperature (lower conductivity, larger skin depth)
        depths_cold = np.array([
            calculate_skin_depth(f, calculate_temperature_correction(sigma_ref, temp_min))
            for f in frequencies
        ])
        # Hot temperature (higher conductivity, smaller skin depth)
        depths_hot = np.array([
            calculate_skin_depth(f, calculate_temperature_correction(sigma_ref, temp_max))
            for f in frequencies
        ])

        ax.fill_between(frequencies/1000, depths_cold, depths_hot,
                       color=color, alpha=0.2, hatch='///',
                       label=f'Temp effect ({temp_min}-{temp_max}°C)')
        ax.loglog(frequencies/1000, depths_cold, color=color, linewidth=1,
                 linestyle=':', alpha=0.7)
        ax.loglog(frequencies/1000, depths_hot, color=color, linewidth=1,
                 linestyle=':', alpha=0.7)

    # Near-field / far-field regime boundary
    if show_regime_boundary:
        # For typical probe spacing of 3m, mark where skin depth equals spacing
        probe_spacing = 3.0  # meters

        # Near-field boundary: delta = probe_spacing / NEAR_FIELD_LIMIT
        near_field_depth = probe_spacing / NEAR_FIELD_LIMIT
        # Far-field boundary: delta = probe_spacing / FAR_FIELD_LIMIT
        far_field_depth = probe_spacing / FAR_FIELD_LIMIT

        # Shade near-field region (large skin depth relative to spacing)
        ax.axhspan(near_field_depth, 1000, alpha=0.08, color=WONG_PALETTE['yellow'],
                  zorder=0)
        ax.text(0.012, near_field_depth * 1.5, 'NEAR-FIELD\n(quasi-static)',
               fontsize=7, color='#806600', style='italic', va='bottom')

        # Mark far-field region
        ax.axhspan(0.1, far_field_depth, alpha=0.08, color=WONG_PALETTE['blue'],
                  zorder=0)
        ax.text(0.012, far_field_depth * 0.5, 'FAR-FIELD\n(wave propagation)',
               fontsize=7, color=WONG_PALETTE['blue'], style='italic', va='top')

        # Transition zone
        ax.axhline(near_field_depth, color='#806600', linestyle='-.', alpha=0.5, linewidth=1)
        ax.axhline(far_field_depth, color=WONG_PALETTE['blue'], linestyle='-.', alpha=0.5, linewidth=1)

    # HIRT operating range
    ax.axvspan(2, 50, alpha=0.15, color=COLORS['success'], label='HIRT Range', zorder=2)
    ax.axvline(10, color=COLORS['success'], linestyle='--', alpha=0.7, zorder=2)
    ax.text(10, 0.15, '10 kHz\n(typical)', fontsize=8, ha='center',
           color=COLORS['success'])

    # Probe spacing reference
    ax.axhline(3, color=COLORS['gray_med'], linestyle=':', alpha=0.7, zorder=2)
    ax.text(70, 3.3, 'Probe spacing (3m)', fontsize=8, ha='right',
           color=COLORS['gray_med'])

    ax.set_xlabel('Frequency (kHz)', fontsize=10)
    ax.set_ylabel('Skin Depth (m)', fontsize=10)
    ax.set_title('Electromagnetic Skin Depth vs. Frequency', fontsize=11,
                fontweight='bold', color=COLORS['primary'])

    # Enhanced legend with uncertainty note
    legend = ax.legend(loc='upper right', fontsize=7, framealpha=0.9, ncol=1)
    if show_uncertainty:
        ax.text(0.98, 0.02, 'Shaded bands: ±50% conductivity uncertainty',
               transform=ax.transAxes, fontsize=6, ha='right', va='bottom',
               color=COLORS['gray_med'], style='italic')

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


def create_tx_rx_waveforms(show_noise=True, show_lockin=True, mains_freq=50,
                           noise_amplitude=0.08, seed=42):
    """
    Create TX and RX waveform comparison showing amplitude and phase.

    Enhanced with:
    - Realistic noise visualization (pink noise + mains pickup)
    - Lock-in detection effect demonstration
    - Colorblind-friendly colors from Wong palette

    Args:
        show_noise: Show realistic noise on RX signal
        show_lockin: Show lock-in detection filtering effect
        mains_freq: Mains frequency (50 Hz or 60 Hz)
        noise_amplitude: Relative amplitude of noise
        seed: Random seed for reproducibility

    Returns:
        BytesIO buffer containing the PNG image
    """
    np.random.seed(seed)

    # Determine subplot configuration based on options
    if show_lockin:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                                            gridspec_kw={'height_ratios': [1, 1.2, 0.8]})
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        ax3 = None

    # Time axis (2 cycles at 10 kHz, high resolution for noise)
    n_samples = 2000
    t = np.linspace(0, 0.2, n_samples)  # ms
    t_seconds = t / 1000  # Convert to seconds for noise functions
    f = 10  # kHz (so period = 0.1 ms)

    # TX signal (clean sinusoid)
    tx = np.sin(2 * np.pi * f * t)

    # RX signal (attenuated and phase shifted)
    amplitude = 0.3
    phase_shift = np.pi / 6  # 30 degrees
    rx_clean = amplitude * np.sin(2 * np.pi * f * t - phase_shift)

    # Add realistic noise if enabled
    if show_noise:
        # Pink (1/f) noise component
        pink = generate_pink_noise(n_samples, amplitude=noise_amplitude * 0.5, seed=seed)

        # Mains pickup (50/60 Hz with harmonics)
        # Note: at this time scale (0.2ms), mains appears as slow drift
        mains = generate_mains_pickup(t_seconds, amplitude=noise_amplitude * 0.3,
                                      freq=mains_freq, harmonics=[1, 3, 5])

        # White noise floor
        white = np.random.normal(0, noise_amplitude * 0.2, n_samples)

        # Combined noise
        total_noise = pink + mains + white
        rx_noisy = rx_clean + total_noise
    else:
        rx_noisy = rx_clean
        total_noise = np.zeros_like(rx_clean)

    # Use colorblind-friendly colors
    tx_color = WONG_PALETTE['bluish_green']
    rx_color = WONG_PALETTE['blue']
    noise_color = WONG_PALETTE['vermillion']
    lockin_color = WONG_PALETTE['orange']

    # Plot TX (clean reference signal)
    ax1.plot(t, tx, color=tx_color, linewidth=2, label='TX Reference')
    ax1.fill_between(t, tx, alpha=0.15, color=tx_color)
    ax1.set_ylabel('TX Amplitude (V)', fontsize=9)
    ax1.set_ylim(-1.4, 1.4)
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title('MIT Transmit and Receive Waveforms (10 kHz)', fontsize=11,
                 fontweight='bold', color=COLORS['primary'])

    # Plot RX with noise
    if show_noise:
        # Plot noisy signal
        ax2.plot(t, rx_noisy, color=rx_color, linewidth=1, alpha=0.7,
                label='RX Signal (with noise)')
        # Overlay clean signal for comparison
        ax2.plot(t, rx_clean, color=rx_color, linewidth=2, linestyle='--',
                alpha=0.5, label='True signal')
        # Show noise envelope
        noise_std = np.std(total_noise)
        ax2.fill_between(t, rx_clean - 2*noise_std, rx_clean + 2*noise_std,
                        color=noise_color, alpha=0.1, label='Noise envelope (2s)')
    else:
        ax2.plot(t, rx_clean, color=rx_color, linewidth=2, label='RX Signal')
        ax2.fill_between(t, rx_clean, alpha=0.2, color=rx_color)

    ax2.set_ylabel('RX Amplitude (mV)', fontsize=9)
    ax2.set_ylim(-0.6, 0.6)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax2.legend(loc='upper right', fontsize=7, ncol=2)

    # Phase shift annotation
    phase_time = phase_shift / (2 * np.pi * f)
    ax2.annotate('', xy=(0.025 + phase_time, 0.45), xytext=(0.025, 0.45),
                arrowprops=dict(arrowstyle='<->', color=WONG_PALETTE['reddish_purple'], lw=1.5))
    ax2.text(0.025 + phase_time/2, 0.52, 'Phase\nLag', fontsize=7, ha='center',
            color=WONG_PALETTE['reddish_purple'])

    # Amplitude annotation
    ax2.annotate('', xy=(0.075, amplitude), xytext=(0.075, 0),
                arrowprops=dict(arrowstyle='<->', color=WONG_PALETTE['yellow'], lw=1.5))
    ax2.text(0.085, amplitude/2, 'Amplitude\nRatio', fontsize=7, va='center',
            color='#806600')

    # Lock-in detection panel (if enabled)
    if show_lockin and ax3 is not None:
        # Lock-in amplifier simulation:
        # Multiply RX by reference, then low-pass filter

        # In-phase component (multiply by cos)
        ref_I = np.cos(2 * np.pi * f * t)
        # Quadrature component (multiply by sin)
        ref_Q = np.sin(2 * np.pi * f * t)

        # Demodulated signals
        demod_I = rx_noisy * ref_I
        demod_Q = rx_noisy * ref_Q

        # Simple moving average as low-pass filter (in reality, time constant >> period)
        window = int(n_samples / 20)  # Average over ~1/20th of display

        # Use numpy convolution for low-pass filter
        kernel = np.ones(window) / window
        filtered_I = np.convolve(demod_I, kernel, mode='same')
        filtered_Q = np.convolve(demod_Q, kernel, mode='same')

        # Recovered amplitude and phase
        recovered_amp = 2 * np.sqrt(filtered_I**2 + filtered_Q**2)

        # Plot lock-in output
        ax3.axhline(amplitude, color=COLORS['gray_med'], linestyle='--',
                   alpha=0.7, label='True amplitude')

        ax3.plot(t, recovered_amp, color=lockin_color, linewidth=2,
                label='Lock-in amplitude')
        ax3.fill_between(t, recovered_amp, alpha=0.2, color=lockin_color)

        ax3.set_ylabel('Lock-in Output', fontsize=9)
        ax3.set_xlabel('Time (ms)', fontsize=9)
        ax3.set_ylim(0, 0.5)
        ax3.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax3.legend(loc='upper right', fontsize=7)

        # Add annotation explaining lock-in benefit
        noise_rejection_db = 20 * np.log10(amplitude / noise_amplitude) if noise_amplitude > 0 else 0
        ax3.text(0.02, 0.95, f'Lock-in detection: noise rejection ~{noise_rejection_db:.0f} dB',
                transform=ax3.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Set x-axis label on bottom plot
    if ax3 is None:
        ax2.set_xlabel('Time (ms)', fontsize=9)

    # Add noise composition annotation if noise is shown
    if show_noise:
        noise_text = f'Noise: pink (1/f) + {mains_freq}Hz mains + thermal'
        ax2.text(0.02, 0.02, noise_text, transform=ax2.transAxes, fontsize=6,
                va='bottom', color=COLORS['gray_med'], style='italic')

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_coupling_decay_plot(show_coil_orientations=True, show_snr_intersection=True,
                                snr_threshold_db=20, use_sphere_markers=True):
    """
    Create 1/r^3 coupling decay curves for MIT.

    Enhanced with:
    - Coil orientation effects (coaxial, coplanar, perpendicular)
    - SNR threshold intersection markers
    - 3D sphere markers for coil positions using draw_sphere_gradient

    Args:
        show_coil_orientations: Show different coil orientation coupling curves
        show_snr_intersection: Mark where signal crosses SNR threshold
        snr_threshold_db: SNR threshold in dB (default 20 dB)
        use_sphere_markers: Use 3D sphere markers for coil positions

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    r = np.linspace(0.5, 5, 200)

    # Base coupling (normalized to 1 at r=1m)
    ref_idx = np.argmin(np.abs(r - 1))

    # Coil orientation coupling factors (relative to coaxial)
    # Reference: Near-field magnetic dipole coupling theory
    # Coaxial (z-z): Full coupling, 1/r^3
    # Coplanar (x-x or y-y): Half coupling with sign reversal
    # Perpendicular (x-z): Zero coupling at boresight, varies with angle

    if show_coil_orientations:
        # Coaxial configuration (maximum coupling)
        coupling_coaxial = 1 / r**3
        coupling_coaxial = coupling_coaxial / coupling_coaxial[ref_idx]

        # Coplanar configuration (half magnitude, opposite sign)
        coupling_coplanar = 0.5 / r**3
        coupling_coplanar = coupling_coplanar / (1 / r[ref_idx]**3)  # Normalize to coaxial at r=1

        # Perpendicular configuration (typical off-axis, ~30% of coaxial)
        coupling_perp = 0.3 / r**3
        coupling_perp = coupling_perp / (1 / r[ref_idx]**3)

        # Plot with colorblind-friendly colors
        ax.semilogy(r, coupling_coaxial, label='Coaxial (TX||RX)',
                   color=WONG_PALETTE['blue'], lw=2.5, zorder=3)
        ax.semilogy(r, coupling_coplanar, label='Coplanar (TX, RX in plane)',
                   color=WONG_PALETTE['orange'], lw=2, linestyle='--', zorder=3)
        ax.semilogy(r, coupling_perp, label='Perpendicular (TX perp RX)',
                   color=WONG_PALETTE['bluish_green'], lw=2, linestyle=':', zorder=3)

        main_coupling = coupling_coaxial
    else:
        # Original single-curve behavior
        coupling_1r3 = 1 / r**3  # Near-field MIT
        coupling_1r3 = coupling_1r3 / coupling_1r3[ref_idx]
        ax.semilogy(r, coupling_1r3, label='1/r^3 (MIT near-field)',
                   color=WONG_PALETTE['blue'], lw=2)
        main_coupling = coupling_1r3

    # Also show ERT geometric factor for comparison
    coupling_1r2 = 1 / r**2
    coupling_1r2 = coupling_1r2 / coupling_1r2[ref_idx]
    ax.semilogy(r, coupling_1r2, label='1/r^2 (ERT geometric)',
               color=COLORS['gray_med'], lw=1.5, linestyle='-.', alpha=0.6)

    # Practical range
    ax.axvspan(1.5, 3.5, alpha=0.12, color=WONG_PALETTE['bluish_green'],
              label='Typical spacing', zorder=1)

    # SNR threshold with intersection marking
    snr_linear = 10 ** (-snr_threshold_db / 20)  # Convert dB to linear
    ax.axhline(snr_linear, color=WONG_PALETTE['vermillion'], linestyle='--',
              alpha=0.8, lw=1.5, zorder=2)
    ax.text(4.7, snr_linear * 1.3, f'SNR = {snr_threshold_db} dB\nThreshold',
           fontsize=8, ha='center', color=WONG_PALETTE['vermillion'])

    if show_snr_intersection:
        # Find intersection points for each orientation
        def find_intersection(coupling, threshold, r_array):
            """Find r where coupling crosses threshold."""
            crossings = np.where(np.diff(np.sign(coupling - threshold)))[0]
            if len(crossings) > 0:
                # Linear interpolation for precise crossing
                idx = crossings[0]
                r1, r2 = r_array[idx], r_array[idx + 1]
                c1, c2 = coupling[idx], coupling[idx + 1]
                r_cross = r1 + (threshold - c1) * (r2 - r1) / (c2 - c1)
                return r_cross
            return None

        if show_coil_orientations:
            intersections = [
                ('Coaxial', find_intersection(coupling_coaxial, snr_linear, r), WONG_PALETTE['blue']),
                ('Coplanar', find_intersection(coupling_coplanar, snr_linear, r), WONG_PALETTE['orange']),
                ('Perp', find_intersection(coupling_perp, snr_linear, r), WONG_PALETTE['bluish_green']),
            ]
        else:
            intersections = [
                ('MIT', find_intersection(main_coupling, snr_linear, r), WONG_PALETTE['blue']),
            ]

        # Mark intersection points
        for name, r_int, color in intersections:
            if r_int is not None and r_int < 5:
                if use_sphere_markers:
                    # Use 3D sphere marker from utility functions
                    draw_sphere_gradient(ax, r_int, snr_linear, radius=0.08,
                                        base_color=color, n_rings=10)
                else:
                    ax.plot(r_int, snr_linear, 'o', color=color, markersize=10,
                           markeredgecolor='white', markeredgewidth=1.5, zorder=5)

                # Add range annotation
                ax.annotate(f'{r_int:.1f}m', xy=(r_int, snr_linear),
                           xytext=(r_int, snr_linear * 3),
                           fontsize=7, ha='center', color=color,
                           arrowprops=dict(arrowstyle='-', color=color, alpha=0.5))

    # Add coil orientation diagram inset
    if show_coil_orientations:
        # Create inset axes for orientation legend
        inset_ax = fig.add_axes([0.15, 0.15, 0.2, 0.25])
        inset_ax.set_xlim(-1.5, 1.5)
        inset_ax.set_ylim(-1.2, 1.2)
        inset_ax.set_aspect('equal')
        inset_ax.axis('off')

        # Draw simplified coil orientation diagrams
        # Coaxial: both coils as horizontal lines
        inset_ax.plot([-0.8, -0.4], [0.7, 0.7], color=WONG_PALETTE['blue'], lw=3)
        inset_ax.plot([0.4, 0.8], [0.7, 0.7], color=WONG_PALETTE['blue'], lw=3)
        inset_ax.text(0, 0.7, 'Coaxial', fontsize=6, ha='center', va='bottom')

        # Coplanar: both coils as vertical lines in same plane
        inset_ax.plot([-0.6, -0.6], [0.1, 0.4], color=WONG_PALETTE['orange'], lw=3)
        inset_ax.plot([0.6, 0.6], [0.1, 0.4], color=WONG_PALETTE['orange'], lw=3)
        inset_ax.text(0, 0.25, 'Coplanar', fontsize=6, ha='center', va='center')

        # Perpendicular: one horizontal, one vertical
        inset_ax.plot([-0.8, -0.4], [-0.5, -0.5], color=WONG_PALETTE['bluish_green'], lw=3)
        inset_ax.plot([0.6, 0.6], [-0.7, -0.3], color=WONG_PALETTE['bluish_green'], lw=3)
        inset_ax.text(0, -0.5, 'Perp', fontsize=6, ha='center', va='center')

        inset_ax.set_title('Orientations', fontsize=7, pad=2)
        inset_ax.patch.set_alpha(0.9)
        inset_ax.patch.set_facecolor('white')

    ax.set_xlabel('Distance (m)', fontsize=10)
    ax.set_ylabel('Relative Signal Strength', fontsize=10)
    ax.set_title('Magnetic Coupling vs. Distance', fontsize=11,
                fontweight='bold', color=COLORS['primary'])
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.set_xlim(0.5, 5)
    ax.set_ylim(0.001, 10)

    # Add physics note
    ax.text(0.02, 0.02, 'Near-field magnetic dipole coupling (r << wavelength)',
           transform=ax.transAxes, fontsize=6, color=COLORS['gray_med'],
           style='italic', va='bottom')

    # Use constrained_layout=False since we have manual inset axes
    if not show_coil_orientations:
        plt.tight_layout()
    else:
        # Adjust main axes to make room for inset
        plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def create_ert_geometric_factor(show_equipotentials=True, show_sensitivity_kernel=True,
                                 array_type='wenner', use_sphere_markers=True):
    """
    Create ERT geometric factor visualization.

    Enhanced with:
    - Physically accurate equipotential contours (from point source solution)
    - Sensitivity kernel visualization (Frechet derivative)
    - Support for different array types
    - 3D sphere markers for electrodes using draw_sphere_gradient

    Args:
        show_equipotentials: Show physically accurate equipotential lines
        show_sensitivity_kernel: Show sensitivity kernel as colormap
        array_type: 'wenner', 'schlumberger', or 'dipole_dipole'
        use_sphere_markers: Use 3D sphere markers for electrodes

    Returns:
        BytesIO buffer containing the PNG image
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-2, 10)
    ax.set_ylim(-4, 2)
    ax.set_aspect('equal')

    # Ground surface
    ax.axhline(0, color='#654321', linewidth=2.5, zorder=5)

    # Electrode positions based on array type
    if array_type == 'wenner':
        # Wenner: equal spacing a between all electrodes
        a = 2  # spacing
        electrode_x = [0, a, 2*a, 3*a]  # C1, P1, P2, C2
        labels = ['C1', 'P1', 'P2', 'C2']
        k_factor = 2 * np.pi * a
        k_text = f'K = 2pa = {k_factor:.1f}'
    elif array_type == 'schlumberger':
        # Schlumberger: AB >> MN
        electrode_x = [0, 2.5, 3.5, 6]
        labels = ['C1', 'P1', 'P2', 'C2']
        k_factor = np.pi * 3 * 2.5 / 1  # pi * L * s / a where L=(AB)/2, s=(MN)/2
        k_text = f'K = pLs/a ~ {k_factor:.1f}'
    else:  # dipole_dipole
        electrode_x = [0, 1, 4, 5]
        labels = ['C1', 'C2', 'P1', 'P2']
        k_text = 'K = pn(n+1)(n+2)a'

    colors_el = [WONG_PALETTE['vermillion'], WONG_PALETTE['blue'],
                WONG_PALETTE['blue'], WONG_PALETTE['vermillion']]

    # Calculate and plot sensitivity kernel
    if show_sensitivity_kernel:
        # Create grid for sensitivity calculation
        x_grid = np.linspace(-1, 8, 100)
        z_grid = np.linspace(-3.5, 0, 80)
        X, Z = np.meshgrid(x_grid, z_grid)

        # Sensitivity kernel for 4-electrode array
        # S = (grad_phi_A * grad_phi_M) / I^2
        # For point sources at surface, this simplifies to geometric factors

        C1_x, P1_x, P2_x, C2_x = electrode_x

        def point_potential_grad(x, z, source_x):
            """Gradient of potential from point source at surface."""
            dx = x - source_x
            r = np.sqrt(dx**2 + z**2)
            r = np.maximum(r, 0.1)  # Avoid singularity
            # Gradient components (radial from source)
            grad_x = dx / (2 * np.pi * r**3)
            grad_z = z / (2 * np.pi * r**3)
            return grad_x, grad_z

        # Current electrode gradients (C1 source, C2 sink)
        gx_c1, gz_c1 = point_potential_grad(X, Z, C1_x)
        gx_c2, gz_c2 = point_potential_grad(X, Z, C2_x)
        gx_A = gx_c1 - gx_c2
        gz_A = gz_c1 - gz_c2

        # Potential electrode gradients (P1-P2 dipole)
        gx_p1, gz_p1 = point_potential_grad(X, Z, P1_x)
        gx_p2, gz_p2 = point_potential_grad(X, Z, P2_x)
        gx_M = gx_p1 - gx_p2
        gz_M = gz_p1 - gz_p2

        # Sensitivity (Frechet kernel) = dot product of gradients
        sensitivity = gx_A * gx_M + gz_A * gz_M

        # Normalize and apply log scaling for visualization
        sens_max = np.percentile(np.abs(sensitivity), 99)
        sensitivity_norm = sensitivity / sens_max

        # Plot sensitivity kernel with diverging colormap
        levels = np.linspace(-1, 1, 21)
        cmap = plt.cm.RdBu_r
        contourf = ax.contourf(X, Z, sensitivity_norm, levels=levels, cmap=cmap,
                              alpha=0.6, extend='both', zorder=1)

        # Add colorbar
        cbar = plt.colorbar(contourf, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('Sensitivity (normalized)', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    else:
        # Simple soil fill if no sensitivity kernel
        ax.fill_between([-2, 10], [0, 0], [-4, -4], color='#d4a373', alpha=0.3)

    # Physically accurate equipotential contours
    if show_equipotentials:
        # Create finer grid for equipotentials
        x_eq = np.linspace(-1, 8, 150)
        z_eq = np.linspace(-3.5, 0, 100)
        X_eq, Z_eq = np.meshgrid(x_eq, z_eq)

        C1_x, P1_x, P2_x, C2_x = electrode_x

        # Potential from two point sources (C1 positive, C2 negative)
        def point_potential(x, z, source_x):
            r = np.sqrt((x - source_x)**2 + z**2)
            r = np.maximum(r, 0.1)
            return 1 / (2 * np.pi * r)

        V = point_potential(X_eq, Z_eq, C1_x) - point_potential(X_eq, Z_eq, C2_x)

        # Plot equipotential contours
        levels_eq = np.linspace(-0.5, 0.5, 15)
        contour = ax.contour(X_eq, Z_eq, V, levels=levels_eq,
                            colors=WONG_PALETTE['blue'], linewidths=0.8,
                            alpha=0.5, linestyles='-', zorder=2)

        # Also plot current flow lines (perpendicular to equipotentials)
        # These are streamlines of the gradient field
        theta_vals = np.linspace(-np.pi * 0.95, -np.pi * 0.05, 8)
        for r_start in [0.3, 0.6, 1.0, 1.5]:
            for theta in theta_vals:
                # Start from C1
                x_start = C1_x + r_start * np.cos(theta)
                z_start = r_start * np.sin(theta)
                if z_start < 0:  # Only below surface
                    # Trace current line (simplified arc)
                    t_param = np.linspace(0, 1, 30)
                    x_curr = x_start + (C2_x - C1_x) * t_param * (1 - np.exp(-2*t_param))
                    z_curr = z_start * np.sin(np.pi * t_param)
                    ax.plot(x_curr, z_curr, color=WONG_PALETTE['vermillion'],
                           alpha=0.25, lw=0.8, zorder=2)

    # Draw electrodes
    for x, label, color in zip(electrode_x, labels, colors_el):
        if use_sphere_markers:
            draw_sphere_gradient(ax, x, 0, radius=0.2, base_color=color, n_rings=12)
        else:
            ax.add_patch(Circle((x, 0), 0.15, color=color, ec='black', lw=1, zorder=10))
        ax.text(x, 0.6, label, ha='center', fontsize=9, fontweight='bold', zorder=11)

    # Annotations
    ax.annotate('Current\nInjection', xy=(electrode_x[0], 0), xytext=(-1.5, 1.4),
               arrowprops=dict(arrowstyle='->', color=WONG_PALETTE['vermillion']),
               fontsize=8, ha='center', zorder=11)
    ax.annotate('Voltage\nMeasurement', xy=((electrode_x[1] + electrode_x[2])/2, 0),
               xytext=((electrode_x[1] + electrode_x[2])/2, 1.4),
               arrowprops=dict(arrowstyle='->', color=WONG_PALETTE['blue']),
               fontsize=8, ha='center', zorder=11)

    # Equation box
    array_name = array_type.replace('_', '-').title()
    ax.text(8.5, -2.5, f'{array_name}\n{k_text}', fontsize=9, ha='center',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['gray_med'],
                    alpha=0.9), zorder=11)

    # Add sensitivity interpretation note
    if show_sensitivity_kernel:
        ax.text(0.02, 0.02, 'Red: positive sensitivity | Blue: negative sensitivity',
               transform=ax.transAxes, fontsize=6, color=COLORS['gray_med'],
               style='italic', va='bottom')

    ax.set_title(f'ERT {array_name} Array - Geometric Factor & Sensitivity', fontsize=11,
                fontweight='bold', color=COLORS['primary'])
    ax.set_xlabel('Distance (m)', fontsize=9)
    ax.set_ylabel('Depth (m)', fontsize=9)
    ax.set_yticks([0, -1, -2, -3])
    ax.set_yticklabels(['0', '1', '2', '3'])

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


def create_illumination_angle_comparison():
    """
    Create illumination angle comparison showing surface vs crosshole angular coverage.

    Shows side-by-side comparison of ray path angles with polar coverage plots at bottom.

    Returns:
        BytesIO buffer containing the PNG image
    """
    # Create figure with 4 subplots: 2 main diagrams + 2 polar plots
    fig = plt.figure(figsize=(10, 7))

    # Top row: main diagram panels
    ax_surface = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    ax_crosshole = plt.subplot2grid((3, 2), (0, 1), rowspan=2)

    # Bottom row: polar plots
    ax_polar_surface = plt.subplot2grid((3, 2), (2, 0), projection='polar')
    ax_polar_crosshole = plt.subplot2grid((3, 2), (2, 1), projection='polar')

    # ===== LEFT PANEL: Surface Methods =====
    ax_surface.set_xlim(-1, 6)
    ax_surface.set_ylim(-3.5, 1)
    ax_surface.set_aspect('equal')
    ax_surface.axis('off')

    # Ground surface
    ax_surface.axhline(0, color='#654321', linewidth=2.5, zorder=5)
    ax_surface.fill_between([-1, 6], [0, 0], [-3.5, -3.5],
                            color='#d4a373', alpha=0.3, zorder=1)

    # Surface sensors (7 sensors in a line)
    sensor_x_positions = np.linspace(0.5, 4.5, 7)
    for x in sensor_x_positions:
        # Triangle sensor markers
        triangle = Polygon([[x-0.15, 0.05], [x+0.15, 0.05], [x, 0.35]],
                          facecolor=COLORS['secondary'], edgecolor='black',
                          linewidth=0.5, zorder=10)
        ax_surface.add_patch(triangle)

    # Target at 2.5 m depth
    target_x = 2.5
    target_depth = -2.5
    target_radius = 0.25
    ax_surface.add_patch(Circle((target_x, target_depth), target_radius,
                                facecolor=COLORS['warning'], edgecolor='black',
                                linewidth=1, zorder=5))

    # Ray paths from sensors to target (color-coded by angle)
    ray_angles = []
    for x in sensor_x_positions:
        dx = target_x - x
        dy = target_depth - 0
        angle = np.arctan2(-dy, dx)  # Angle from horizontal (0 = horizontal right)
        ray_angles.append(angle)

        # Color by angle: steep rays (close to vertical) are red,
        # shallower rays are orange/yellow
        angle_deg = np.degrees(angle)
        if angle_deg > 80:
            ray_color = WONG_PALETTE['vermillion']  # Steep (red)
        elif angle_deg > 70:
            ray_color = WONG_PALETTE['orange']  # Moderate (orange)
        else:
            ray_color = WONG_PALETTE['yellow']  # Shallow (yellow)

        # Draw ray path (curved to suggest indirect path)
        t = np.linspace(0, 1, 30)
        # Parabolic path (curves outward slightly)
        curve_factor = 0.3 * np.abs(dx) * np.sin(np.pi * t)
        ray_x = x + dx * t
        ray_y = 0 + dy * t - curve_factor
        ax_surface.plot(ray_x, ray_y, color=ray_color, alpha=0.5,
                       linewidth=1.5, zorder=3)

    # Annotation
    ax_surface.text(2.5, -3.2, 'All rays from above - poor vertical resolution',
                    fontsize=8, ha='center', style='italic',
                    color=COLORS['gray_dark'])

    # Title
    ax_surface.text(2.5, 0.8, '(a) Surface Methods',
                    fontsize=10, ha='center', fontweight='bold',
                    color=COLORS['primary'])

    # ===== RIGHT PANEL: Crosshole Methods =====
    ax_crosshole.set_xlim(-1, 6)
    ax_crosshole.set_ylim(-3.5, 1)
    ax_crosshole.set_aspect('equal')
    ax_crosshole.axis('off')

    # Ground surface
    ax_crosshole.axhline(0, color='#654321', linewidth=2.5, zorder=5)
    ax_crosshole.fill_between([-1, 6], [0, 0], [-3.5, -3.5],
                              color='#d4a373', alpha=0.3, zorder=1)

    # Probe positions (3 probes at 2 m spacing)
    probe_positions = [1, 3, 5]
    probe_depth = -3.0
    sensor_depths = [-0.5, -1.5, -2.5]  # Sensors at multiple depths

    # Draw probes
    for px in probe_positions:
        # Probe body
        ax_crosshole.add_patch(Rectangle((px - 0.08, probe_depth), 0.16,
                                         abs(probe_depth),
                                         facecolor=COLORS['secondary'],
                                         edgecolor='black', linewidth=0.5,
                                         zorder=4))

        # Sensor markers on probe
        for sy in sensor_depths:
            ax_crosshole.add_patch(Circle((px, sy), 0.1,
                                          facecolor=COLORS['success'],
                                          edgecolor='black', linewidth=0.5,
                                          zorder=6))

    # Target at 2.5 m depth
    target_x_ch = 2.5
    target_depth_ch = -2.5
    ax_crosshole.add_patch(Circle((target_x_ch, target_depth_ch), target_radius,
                                   facecolor=COLORS['warning'], edgecolor='black',
                                   linewidth=1, zorder=5))

    # Ray paths between probe sensors (color-coded by direction)
    ch_ray_angles = []
    for i, px1 in enumerate(probe_positions):
        for j, px2 in enumerate(probe_positions):
            if i >= j:  # Only draw rays in one direction
                continue
            for sy1 in sensor_depths:
                for sy2 in sensor_depths:
                    # Calculate angle of ray
                    dx = px2 - px1
                    dy = sy2 - sy1
                    angle = np.arctan2(dy, dx)
                    ch_ray_angles.append(angle)

                    # Color by direction
                    if np.abs(dy) < 0.3:  # Horizontal
                        ray_color = WONG_PALETTE['bluish_green']
                        alpha = 0.4
                    elif dy > 0:  # Ascending
                        ray_color = WONG_PALETTE['blue']
                        alpha = 0.35
                    else:  # Descending
                        ray_color = WONG_PALETTE['reddish_purple']
                        alpha = 0.35

                    # Draw straight ray path
                    ax_crosshole.plot([px1, px2], [sy1, sy2],
                                      color=ray_color, alpha=alpha,
                                      linewidth=1.2, zorder=2)

    # Annotation
    ax_crosshole.text(2.5, -3.2, 'Rays from all directions - isotropic 3D resolution',
                      fontsize=8, ha='center', style='italic',
                      color=COLORS['success'])

    # Title
    ax_crosshole.text(2.5, 0.8, '(b) HIRT Crosshole',
                      fontsize=10, ha='center', fontweight='bold',
                      color=COLORS['primary'])

    # ===== POLAR PLOTS =====

    # Surface angular coverage (narrow wedge)
    ax_polar_surface.set_theta_zero_location('E')  # 0 degrees at right (horizontal)
    ax_polar_surface.set_theta_direction(1)  # Counter-clockwise

    # Convert ray angles to polar convention
    surface_angles = np.array(ray_angles)
    # Create wedge showing coverage (60-90 degrees from horizontal = 0-30 deg from vertical)
    wedge_angles = np.linspace(np.radians(60), np.radians(90), 50)
    wedge_r = np.ones_like(wedge_angles)
    ax_polar_surface.fill_between(wedge_angles, 0, wedge_r,
                                   color=WONG_PALETTE['vermillion'], alpha=0.3)
    ax_polar_surface.plot(wedge_angles, wedge_r, color=WONG_PALETTE['vermillion'],
                          linewidth=2)

    ax_polar_surface.set_ylim(0, 1)
    ax_polar_surface.set_yticks([])
    ax_polar_surface.set_title('Surface Coverage\n(narrow wedge)',
                               fontsize=8, pad=10, color=COLORS['gray_dark'])
    ax_polar_surface.grid(True, alpha=0.3)

    # Crosshole angular coverage (nearly full circle)
    ax_polar_crosshole.set_theta_zero_location('E')
    ax_polar_crosshole.set_theta_direction(1)

    # Show full coverage: -45 to +135 degrees (180 deg horizontal + vertical range)
    coverage_angles = np.linspace(np.radians(-45), np.radians(135), 100)
    coverage_r = np.ones_like(coverage_angles)
    ax_polar_crosshole.fill_between(coverage_angles, 0, coverage_r,
                                     color=WONG_PALETTE['bluish_green'], alpha=0.3)
    ax_polar_crosshole.plot(coverage_angles, coverage_r,
                            color=WONG_PALETTE['bluish_green'], linewidth=2)

    ax_polar_crosshole.set_ylim(0, 1)
    ax_polar_crosshole.set_yticks([])
    ax_polar_crosshole.set_title('Crosshole Coverage\n(near-full circle)',
                                 fontsize=8, pad=10, color=COLORS['gray_dark'])
    ax_polar_crosshole.grid(True, alpha=0.3)

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf
