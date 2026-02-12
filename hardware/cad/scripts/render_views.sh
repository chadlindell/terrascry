#!/bin/bash
# Render OpenSCAD views to PNG for documentation
#
# Usage:
#   ./render_views.sh all           # Render all views
#   ./render_views.sh assembled     # Render single view
#   ./render_views.sh --stl         # Export STL file
#
# Requirements:
#   - OpenSCAD 2021.01+
#   - xvfb-run (for headless rendering on Linux)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCAD_FILE="$SCRIPT_DIR/../openscad/pathfinder_frame.scad"
OUTPUT_DIR="$SCRIPT_DIR/../../../docs/assets/images/assembly"
STL_DIR="$SCRIPT_DIR/../stl"

# Image dimensions
WIDTH=1200
HEIGHT=900

# Check for OpenSCAD
if ! command -v openscad &> /dev/null; then
    echo "Error: OpenSCAD not found. Please install OpenSCAD."
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$STL_DIR"

# Camera settings for each view (translate, rotate, distance)
# Format: "eye_x,eye_y,eye_z,center_x,center_y,center_z,vp_d"
declare -A CAMERAS
CAMERAS["assembled"]="0,0,0,55,25,25,4000"
CAMERAS["exploded"]="0,0,200,55,25,25,5000"
CAMERAS["cross_section"]="0,0,0,90,0,25,3000"
CAMERAS["deployment"]="0,500,800,45,30,15,5000"
CAMERAS["single_tube"]="0,0,0,45,25,25,1200"

# Render a single view
render_view() {
    local view="$1"
    local output_file="$OUTPUT_DIR/pathfinder_${view}.png"
    local camera="${CAMERAS[$view]}"

    echo "Rendering $view view..."

    # Parse camera settings
    IFS=',' read -r cx cy cz rx ry rz dist <<< "$camera"

    # Use xvfb-run for headless rendering if available
    if command -v xvfb-run &> /dev/null && [ -z "$DISPLAY" ]; then
        RENDER_CMD="xvfb-run -a openscad"
    else
        RENDER_CMD="openscad"
    fi

    $RENDER_CMD \
        -o "$output_file" \
        -D "part=\"$view\"" \
        --imgsize=$WIDTH,$HEIGHT \
        --camera="$cx,$cy,$cz,$rx,$ry,$rz,$dist" \
        --autocenter \
        --viewall \
        --colorscheme="Tomorrow Night" \
        "$SCAD_FILE"

    if [ -f "$output_file" ]; then
        echo "  -> $output_file"
    else
        echo "  ERROR: Failed to create $output_file"
        return 1
    fi
}

# Export STL file
export_stl() {
    local output_file="$STL_DIR/pathfinder_frame.stl"

    echo "Exporting STL..."

    if command -v xvfb-run &> /dev/null && [ -z "$DISPLAY" ]; then
        RENDER_CMD="xvfb-run -a openscad"
    else
        RENDER_CMD="openscad"
    fi

    $RENDER_CMD \
        -o "$output_file" \
        -D "part=\"assembled\"" \
        "$SCAD_FILE"

    if [ -f "$output_file" ]; then
        echo "  -> $output_file"
    else
        echo "  ERROR: Failed to create $output_file"
        return 1
    fi
}

# Print usage
usage() {
    echo "Usage: $0 [view|all|--stl]"
    echo ""
    echo "Views:"
    echo "  assembled      - Complete trapeze system (isometric)"
    echo "  exploded       - Components separated with gaps"
    echo "  cross_section  - Half-section showing tube interiors"
    echo "  deployment     - With operator silhouette"
    echo "  single_tube    - Single sensor pair detail"
    echo "  all            - Render all views"
    echo ""
    echo "Options:"
    echo "  --stl          - Export assembled model as STL"
    echo "  --help         - Show this help"
}

# Main
case "${1:-all}" in
    all)
        echo "Rendering all views..."
        for view in "${!CAMERAS[@]}"; do
            render_view "$view"
        done
        echo ""
        echo "All renders complete!"
        echo "Output directory: $OUTPUT_DIR"
        ;;
    --stl)
        export_stl
        ;;
    --help|-h)
        usage
        ;;
    assembled|exploded|cross_section|deployment|single_tube)
        render_view "$1"
        ;;
    *)
        echo "Unknown view: $1"
        usage
        exit 1
        ;;
esac
