#!/usr/bin/env python3
"""
HIRT Whitepaper - Build All Section PDFs

This script builds all 19 individual section PDFs by running each section generator.
Run from the pdf-generator directory.

Usage:
    python build_all.py [--parallel] [--section N]

Options:
    --parallel    Run section generators in parallel (faster but uses more memory)
    --section N   Only build section N (0-19)
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Section definitions
SECTIONS = [
    (0, "index", "Index"),
    (1, "executive_summary", "Executive Summary"),
    (2, "physics_theory", "Physics Theory"),
    (3, "system_architecture", "System Architecture"),
    (4, "bill_of_materials", "Bill of Materials"),
    (5, "mechanical_design", "Mechanical Design"),
    (6, "electronics_circuits", "Electronics & Circuits"),
    (7, "assembly_wiring", "Assembly & Wiring"),
    (8, "testing_verification", "Testing & Verification"),
    (9, "calibration", "Calibration"),
    (10, "field_operations", "Field Operations"),
    (11, "data_recording", "Data Recording"),
    (12, "data_interpretation", "Data Interpretation"),
    (13, "troubleshooting", "Troubleshooting"),
    (14, "glossary", "Glossary"),
    (15, "quick_reference", "Quick Reference"),
    (16, "field_checklists", "Field Checklists"),
    (17, "application_scenarios", "Application Scenarios"),
    (18, "future_development", "Future Development"),
    (19, "ethics_legal_safety", "Ethics, Legal, Safety"),
]


def get_script_dir():
    """Get the directory containing this script."""
    return Path(__file__).parent.absolute()


def build_section(section_num, section_slug, section_title):
    """
    Build a single section PDF.

    Returns:
        Tuple of (section_num, success, message)
    """
    script_dir = get_script_dir()
    section_script = script_dir / "sections" / f"section_{section_num:02d}_{section_slug}.py"

    if not section_script.exists():
        return (section_num, False, f"Script not found: {section_script}")

    try:
        result = subprocess.run(
            [sys.executable, str(section_script)],
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            return (section_num, True, f"Built section {section_num:02d}: {section_title}")
        else:
            error_msg = result.stderr[:500] if result.stderr else "Unknown error"
            return (section_num, False, f"Failed section {section_num:02d}: {error_msg}")

    except subprocess.TimeoutExpired:
        return (section_num, False, f"Timeout building section {section_num:02d}")
    except Exception as e:
        return (section_num, False, f"Error building section {section_num:02d}: {str(e)}")


def build_all_sequential():
    """Build all sections sequentially."""
    results = []
    total = len(SECTIONS)

    for i, (num, slug, title) in enumerate(SECTIONS):
        print(f"[{i+1}/{total}] Building section {num:02d}: {title}...")
        result = build_section(num, slug, title)
        results.append(result)

        if result[1]:
            print(f"  ✓ {result[2]}")
        else:
            print(f"  ✗ {result[2]}")

    return results


def build_all_parallel(max_workers=4):
    """Build all sections in parallel."""
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(build_section, num, slug, title): (num, title)
            for num, slug, title in SECTIONS
        }

        for future in as_completed(futures):
            num, title = futures[future]
            try:
                result = future.result()
                results.append(result)

                if result[1]:
                    print(f"✓ Section {num:02d}: {title}")
                else:
                    print(f"✗ Section {num:02d}: {result[2]}")

            except Exception as e:
                results.append((num, False, str(e)))
                print(f"✗ Section {num:02d}: {e}")

    return sorted(results, key=lambda x: x[0])


def print_summary(results):
    """Print build summary."""
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print("\n" + "=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    print(f"Total sections: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed sections:")
        for num, _, msg in failed:
            print(f"  - Section {num:02d}: {msg[:60]}...")

    # Check output directory
    script_dir = get_script_dir()
    output_dir = script_dir / "output" / "sections"
    if output_dir.exists():
        pdfs = list(output_dir.glob("*.pdf"))
        print(f"\nPDF files in output/sections/: {len(pdfs)}")
        for pdf in sorted(pdfs):
            size_kb = pdf.stat().st_size / 1024
            print(f"  - {pdf.name} ({size_kb:.1f} KB)")

    print("=" * 60)

    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(description="Build HIRT Whitepaper Section PDFs")
    parser.add_argument("--parallel", action="store_true", help="Build in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--section", type=int, help="Only build specific section (0-19)")
    args = parser.parse_args()

    print("=" * 60)
    print("HIRT WHITEPAPER - BUILD ALL SECTIONS")
    print("=" * 60)

    start_time = time.time()

    if args.section is not None:
        # Build single section
        if 0 <= args.section <= 19:
            num, slug, title = SECTIONS[args.section]
            print(f"Building section {num:02d}: {title}...")
            result = build_section(num, slug, title)
            results = [result]
            if result[1]:
                print(f"✓ {result[2]}")
            else:
                print(f"✗ {result[2]}")
        else:
            print(f"Invalid section number: {args.section}")
            sys.exit(1)
    elif args.parallel:
        print(f"Building {len(SECTIONS)} sections in parallel (workers={args.workers})...\n")
        results = build_all_parallel(args.workers)
    else:
        print(f"Building {len(SECTIONS)} sections sequentially...\n")
        results = build_all_sequential()

    elapsed = time.time() - start_time
    print(f"\nBuild completed in {elapsed:.1f} seconds")

    success = print_summary(results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
