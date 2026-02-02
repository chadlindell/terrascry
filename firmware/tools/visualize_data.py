#!/usr/bin/env python3
"""
Pathfinder Data Visualization Tool

Quick visualization of gradiometer CSV data from Pathfinder surveys.
Generates simple plots to verify data quality in the field.

Usage:
    python visualize_data.py PATH0001.CSV
    python visualize_data.py PATH0001.CSV --pair 1
    python visualize_data.py PATH0001.CSV --map

Dependencies:
    pip install pandas matplotlib
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_pathfinder_data(filepath):
    """Load Pathfinder CSV data."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def plot_time_series(df, pair=None):
    """Plot gradient vs time for one or all pairs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if pair is not None:
        # Plot single pair
        col = f'g{pair}_grad'
        if col not in df.columns:
            print(f"Error: Pair {pair} not found in data")
            return
        ax.plot(df['timestamp'] / 1000, df[col], label=f'Pair {pair}', linewidth=0.5)
        ax.set_title(f'Pathfinder Gradiometer - Pair {pair}')
    else:
        # Plot all pairs
        for i in range(1, 5):
            col = f'g{i}_grad'
            if col in df.columns:
                ax.plot(df['timestamp'] / 1000, df[col], label=f'Pair {i}', linewidth=0.5, alpha=0.7)
        ax.set_title('Pathfinder Gradiometer - All Pairs')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Gradient (ADC counts)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_map(df):
    """Plot gradient values on map (GPS coordinates)."""
    # Filter to valid GPS coordinates
    df_gps = df[(df['lat'] != 0) & (df['lon'] != 0)].copy()

    if len(df_gps) == 0:
        print("No valid GPS coordinates found in data")
        return

    print(f"Plotting {len(df_gps)} GPS-tagged samples")

    # Create 2x2 subplot for each pair
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Pathfinder Spatial Gradient Map')

    for i, ax in enumerate(axes.flat, start=1):
        col = f'g{i}_grad'
        if col not in df_gps.columns:
            continue

        scatter = ax.scatter(
            df_gps['lon'],
            df_gps['lat'],
            c=df_gps[col],
            cmap='RdBu_r',
            s=10,
            alpha=0.6
        )
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Pair {i}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Gradient (ADC)')

    plt.tight_layout()
    plt.show()


def plot_histogram(df):
    """Plot gradient distribution histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Pathfinder Gradient Distributions')

    for i, ax in enumerate(axes.flat, start=1):
        col = f'g{i}_grad'
        if col not in df.columns:
            continue

        ax.hist(df[col], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Gradient (ADC counts)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Pair {i}')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean = df[col].mean()
        std = df[col].std()
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f}')
        ax.axvline(mean + std, color='orange', linestyle=':', linewidth=1, label=f'±1σ: {std:.1f}')
        ax.axvline(mean - std, color='orange', linestyle=':', linewidth=1)
        ax.legend()

    plt.tight_layout()
    plt.show()


def print_statistics(df):
    """Print data quality statistics."""
    print("\n" + "="*60)
    print("PATHFINDER DATA STATISTICS")
    print("="*60)

    # GPS lock percentage
    gps_valid = ((df['lat'] != 0) & (df['lon'] != 0)).sum()
    gps_pct = 100 * gps_valid / len(df)
    print(f"\nGPS Lock: {gps_valid}/{len(df)} samples ({gps_pct:.1f}%)")

    if gps_valid > 0:
        df_gps = df[(df['lat'] != 0) & (df['lon'] != 0)]
        print(f"  Lat range: {df_gps['lat'].min():.6f} to {df_gps['lat'].max():.6f}")
        print(f"  Lon range: {df_gps['lon'].min():.6f} to {df_gps['lon'].max():.6f}")

    # Gradient statistics
    print(f"\nGradient Statistics (ADC counts):")
    print(f"{'Pair':<8} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
    print("-" * 48)

    for i in range(1, 5):
        col = f'g{i}_grad'
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"{i:<8} {mean:<10.1f} {std:<10.1f} {min_val:<10} {max_val:<10}")

    # Survey duration
    duration_sec = df['timestamp'].max() / 1000
    duration_min = duration_sec / 60
    print(f"\nSurvey Duration: {duration_min:.1f} minutes ({duration_sec:.0f} seconds)")
    print(f"Sample Rate: {len(df) / duration_sec:.1f} Hz (average)")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize Pathfinder gradiometer data')
    parser.add_argument('csvfile', type=str, help='Path to CSV data file (e.g., PATH0001.CSV)')
    parser.add_argument('--pair', type=int, choices=[1, 2, 3, 4],
                        help='Plot specific pair (1-4) instead of all')
    parser.add_argument('--map', action='store_true',
                        help='Generate spatial map plot (requires GPS data)')
    parser.add_argument('--hist', action='store_true',
                        help='Show gradient distribution histograms')
    parser.add_argument('--stats-only', action='store_true',
                        help='Print statistics without plotting')

    args = parser.parse_args()

    # Check file exists
    if not Path(args.csvfile).exists():
        print(f"Error: File not found: {args.csvfile}")
        sys.exit(1)

    # Load data
    df = load_pathfinder_data(args.csvfile)

    # Print statistics
    print_statistics(df)

    # Generate plots
    if args.stats_only:
        return

    if args.map:
        plot_map(df)
    elif args.hist:
        plot_histogram(df)
    else:
        plot_time_series(df, pair=args.pair)


if __name__ == '__main__':
    main()
