#!/usr/bin/env python3
"""
Convert PDF pages to PNG images for visual inspection.
Uses PyMuPDF (fitz) for high-quality rendering.
"""

import os
import sys
from pathlib import Path

import fitz  # PyMuPDF


def convert_pdf_to_images(pdf_path, output_dir, dpi=150):
    """
    Convert each page of a PDF to a PNG image.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save images
        dpi: Resolution for rendering (default: 150)

    Returns:
        List of output image paths
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open PDF
    doc = fitz.open(str(pdf_path))
    pdf_name = pdf_path.stem

    image_paths = []
    zoom = dpi / 72  # 72 is default PDF resolution
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)

        # Output filename: pdfname_page_NN.png
        output_path = output_dir / f"{pdf_name}_page_{page_num + 1:02d}.png"
        pix.save(str(output_path))
        image_paths.append(output_path)
        print(f"  Converted: {output_path.name}")

    doc.close()
    return image_paths


def convert_all_sections(sections_dir, output_dir):
    """Convert all section PDFs to images."""
    sections_dir = Path(sections_dir)
    output_dir = Path(output_dir)

    all_images = {}

    for pdf_file in sorted(sections_dir.glob("*.pdf")):
        print(f"\nProcessing: {pdf_file.name}")
        section_output = output_dir / pdf_file.stem
        images = convert_pdf_to_images(pdf_file, section_output)
        all_images[pdf_file.name] = images

    return all_images


def main():
    script_dir = Path(__file__).parent
    sections_dir = script_dir / "output" / "sections"
    combined_pdf = script_dir / "output" / "HIRT_Whitepaper_Complete.pdf"
    inspection_dir = script_dir / "output" / "inspection"

    print("=" * 60)
    print("CONVERTING PDFs TO IMAGES FOR INSPECTION")
    print("=" * 60)

    # Convert individual sections
    # print("\n--- Converting Individual Sections ---")
    # convert_all_sections(sections_dir, inspection_dir / "sections")

    # Convert combined PDF
    print("\n--- Converting Combined Whitepaper ---")
    combined_output = inspection_dir / "combined"
    convert_pdf_to_images(combined_pdf, combined_output)

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print(f"Images saved to: {inspection_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
