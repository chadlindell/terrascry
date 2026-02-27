#!/bin/bash
# Quick script to render OpenSCAD files to STL for 3D printing

if [ -z "$1" ]; then
    echo "Usage: $0 <filename.scad>"
    echo "Example: $0 micro_probe_head.scad"
    exit 1
fi

INPUT="$1"
BASENAME=$(basename "$INPUT" .scad)
OUTPUT="${BASENAME}.stl"

echo "Rendering $INPUT to $OUTPUT..."
openscad -o "$OUTPUT" --export-format binstl "$INPUT"

if [ $? -eq 0 ]; then
    echo "✓ Render complete: $OUTPUT"
    ls -lh "$OUTPUT"

    # Copy to home directory for easy slicer access
    cp "$OUTPUT" ~/"$OUTPUT"
    echo "✓ Also copied to ~/$OUTPUT for slicer access"
else
    echo "✗ Render failed"
    exit 1
fi
