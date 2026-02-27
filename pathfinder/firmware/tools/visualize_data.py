#!/usr/bin/env python3
"""
Pathfinder Data Visualization Tool

Quick visualization of gradiometer CSV data from Pathfinder surveys.
Automatically detects number of sensor pairs from CSV columns.

Usage:
    python visualize_data.py PATH0001.CSV
    python visualize_data.py PATH0001.CSV --pair 1
    python visualize_data.py PATH0001.CSV --map
    python visualize_data.py PATH0001.CSV --geojson output.geojson
    python visualize_data.py PATH0001.CSV --calibrate offsets.json -o corrected.csv

Dependencies:
    pip install pandas matplotlib
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def detect_pairs(df):
    """Detect number of sensor pairs from column names."""
    n = 0
    for i in range(1, 5):
        if f'g{i}_grad' in df.columns:
            n = i
    return n


def load_pathfinder_data(filepath):
    """Load Pathfinder CSV data, skipping comment lines starting with #."""
    try:
        df = pd.read_csv(filepath, comment='#')
        n_pairs = detect_pairs(df)
        has_gps_quality = 'fix_quality' in df.columns
        extras = []
        if has_gps_quality:
            extras.append("GPS quality")
        print(f"Loaded {len(df)} samples from {filepath} "
              f"({n_pairs} pair{'s' if n_pairs != 1 else ''}"
              f"{', ' + ', '.join(extras) if extras else ''})")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def _subplot_layout(n):
    """Return (rows, cols) for n subplots."""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    return 2, 2  # 3 or 4


def plot_time_series(df, pair=None):
    """Plot gradient vs time for one or all pairs."""
    n_pairs = detect_pairs(df)
    fig, ax = plt.subplots(figsize=(12, 6))

    if pair is not None:
        col = f'g{pair}_grad'
        if col not in df.columns:
            print(f"Error: Pair {pair} not found in data (file has {n_pairs} pairs)")
            return
        ax.plot(df['timestamp'] / 1000, df[col], label=f'Pair {pair}', linewidth=0.5)
        ax.set_title(f'Pathfinder Gradiometer - Pair {pair}')
    else:
        for i in range(1, n_pairs + 1):
            col = f'g{i}_grad'
            ax.plot(df['timestamp'] / 1000, df[col], label=f'Pair {i}', linewidth=0.5, alpha=0.7)
        ax.set_title(f'Pathfinder Gradiometer - {n_pairs} Pair{"s" if n_pairs != 1 else ""}')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Gradient (ADC counts)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_map(df):
    """Plot gradient values on map (GPS coordinates)."""
    df_gps = df[(df['lat'] != 0) & (df['lon'] != 0)].copy()
    n_pairs = detect_pairs(df_gps)

    if len(df_gps) == 0:
        print("No valid GPS coordinates found in data")
        return

    print(f"Plotting {len(df_gps)} GPS-tagged samples")

    rows, cols = _subplot_layout(n_pairs)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    fig.suptitle('Pathfinder Spatial Gradient Map')

    for idx in range(n_pairs):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        col = f'g{idx + 1}_grad'

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
        ax.set_title(f'Pair {idx + 1}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Gradient (ADC)')

    # Hide unused subplots (e.g., 3 pairs in 2x2 grid)
    for idx in range(n_pairs, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_histogram(df):
    """Plot gradient distribution histograms."""
    n_pairs = detect_pairs(df)
    rows, cols = _subplot_layout(n_pairs)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    fig.suptitle('Pathfinder Gradient Distributions')

    for idx in range(n_pairs):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        col = f'g{idx + 1}_grad'

        ax.hist(df[col], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Gradient (ADC counts)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Pair {idx + 1}')
        ax.grid(True, alpha=0.3)

        mean = df[col].mean()
        std = df[col].std()
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f}')
        ax.axvline(mean + std, color='orange', linestyle=':', linewidth=1, label=f'\u00b11\u03c3: {std:.1f}')
        ax.axvline(mean - std, color='orange', linestyle=':', linewidth=1)
        ax.legend()

    for idx in range(n_pairs, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout()
    plt.show()


def print_statistics(df):
    """Print data quality statistics."""
    n_pairs = detect_pairs(df)

    print("\n" + "=" * 60)
    print("PATHFINDER DATA STATISTICS")
    print("=" * 60)
    print(f"Sensor pairs: {n_pairs}")

    # GPS lock percentage
    gps_valid = ((df['lat'] != 0) & (df['lon'] != 0)).sum()
    gps_pct = 100 * gps_valid / len(df)
    print(f"\nGPS Lock: {gps_valid}/{len(df)} samples ({gps_pct:.1f}%)")

    if gps_valid > 0:
        df_gps = df[(df['lat'] != 0) & (df['lon'] != 0)]
        print(f"  Lat range: {df_gps['lat'].min():.6f} to {df_gps['lat'].max():.6f}")
        print(f"  Lon range: {df_gps['lon'].min():.6f} to {df_gps['lon'].max():.6f}")
        if 'hdop' in df_gps.columns:
            print(f"  HDOP range: {df_gps['hdop'].min():.1f} to {df_gps['hdop'].max():.1f}")
        if 'altitude' in df_gps.columns:
            alt_valid = df_gps[df_gps['altitude'] != 0]
            if len(alt_valid) > 0:
                print(f"  Altitude range: {alt_valid['altitude'].min():.1f} to {alt_valid['altitude'].max():.1f} m")

    # Gradient statistics
    print(f"\nGradient Statistics (ADC counts):")
    print(f"{'Pair':<8} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
    print("-" * 48)

    for i in range(1, n_pairs + 1):
        col = f'g{i}_grad'
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
    print("=" * 60 + "\n")


def export_geojson(df, output_path):
    """Export GPS-tagged gradient data as GeoJSON FeatureCollection."""
    df_gps = df[(df['lat'] != 0) & (df['lon'] != 0)].copy()
    n_pairs = detect_pairs(df_gps)

    if len(df_gps) == 0:
        print("No valid GPS coordinates found - cannot export GeoJSON")
        return

    features = []
    for _, row in df_gps.iterrows():
        properties = {"timestamp_ms": int(row['timestamp'])}
        if 'fix_quality' in row.index:
            properties['fix_quality'] = int(row['fix_quality'])
        if 'hdop' in row.index:
            properties['hdop'] = float(row['hdop'])
        if 'altitude' in row.index:
            properties['altitude_m'] = float(row['altitude'])
        for i in range(1, n_pairs + 1):
            properties[f'pair{i}_top'] = int(row[f'g{i}_top'])
            properties[f'pair{i}_bot'] = int(row[f'g{i}_bot'])
            properties[f'pair{i}_grad'] = int(row[f'g{i}_grad'])

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['lon']), float(row['lat'])]
            },
            "properties": properties
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"Exported {len(features)} GPS-tagged samples to {output_path}")


def apply_calibration(df, cal_path, output_path):
    """Apply calibration offsets and gains, write corrected CSV."""
    with open(cal_path) as f:
        cal = json.load(f)

    n_pairs = detect_pairs(df)
    df_out = df.copy()
    channels = cal.get("channels", {})

    for key, params in channels.items():
        if key not in df_out.columns:
            print(f"Warning: calibration channel '{key}' not found in data, skipping")
            continue
        offset = params.get("offset", 0)
        gain = params.get("gain", 1.0)
        if gain == 0:
            print(f"Warning: gain=0 for '{key}', skipping to avoid division by zero")
            continue
        df_out[key] = ((df_out[key] - offset) / gain).round().astype(int)

    # Recompute gradients from corrected values
    for i in range(1, n_pairs + 1):
        top_col = f'g{i}_top'
        bot_col = f'g{i}_bot'
        grad_col = f'g{i}_grad'
        if top_col in df_out.columns and bot_col in df_out.columns:
            df_out[grad_col] = df_out[bot_col] - df_out[top_col]

    df_out.to_csv(output_path, index=False)
    print(f"Calibrated data written to {output_path} ({len(df_out)} samples)")


def main():
    parser = argparse.ArgumentParser(description='Visualize Pathfinder gradiometer data')
    parser.add_argument('csvfile', type=str, help='Path to CSV data file (e.g., PATH0001.CSV)')
    parser.add_argument('--pair', type=int, choices=[1, 2, 3, 4],
                        help='Plot specific pair (1-4)')
    parser.add_argument('--map', action='store_true',
                        help='Generate spatial map plot (requires GPS data)')
    parser.add_argument('--hist', action='store_true',
                        help='Show gradient distribution histograms')
    parser.add_argument('--stats-only', action='store_true',
                        help='Print statistics without plotting')
    parser.add_argument('--geojson', type=str, metavar='OUTPUT',
                        help='Export GPS-tagged data as GeoJSON file')
    parser.add_argument('--calibrate', type=str, metavar='CAL_JSON',
                        help='Apply calibration from JSON file (use with -o)')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file for --calibrate (corrected CSV)')

    args = parser.parse_args()

    if not Path(args.csvfile).exists():
        print(f"Error: File not found: {args.csvfile}")
        sys.exit(1)

    df = load_pathfinder_data(args.csvfile)

    if args.geojson:
        export_geojson(df, args.geojson)
        return

    if args.calibrate:
        if not args.output:
            print("Error: --calibrate requires -o/--output for the corrected CSV path")
            sys.exit(1)
        if not Path(args.calibrate).exists():
            print(f"Error: Calibration file not found: {args.calibrate}")
            sys.exit(1)
        apply_calibration(df, args.calibrate, args.output)
        return

    print_statistics(df)

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
