#!/bin/bash
# =============================================================================
# HIRT OpenSCAD Visualization Renderer
# =============================================================================
#
# Renders documentation-quality PNG images from OpenSCAD visualization modules.
# Uses xvfb-run for headless operation on servers without display.
#
# Usage:
#   ./render_views.sh              # Render all views
#   ./render_views.sh exploded     # Render specific view
#   ./render_views.sh --help       # Show help
#
# Output:
#   docs/assets/images/assembly/probe_exploded.png
#   docs/assets/images/assembly/probe_assembled.png
#   docs/assets/images/assembly/probe_cross_section.png
#   docs/assets/images/assembly/deployment_array.png
#   docs/assets/images/assembly/component_callout.png
#
# =============================================================================

set -e

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SCAD_FILE="$PROJECT_ROOT/hardware/cad/openscad/modular_flush_connector.scad"
OUTPUT_DIR="$PROJECT_ROOT/docs/assets/images/assembly"

# Render settings
WIDTH=1200
HEIGHT=900
COLORSCHEME="Tomorrow"

# Check for OpenSCAD
if ! command -v openscad &> /dev/null; then
    echo "ERROR: OpenSCAD not found. Please install OpenSCAD."
    exit 1
fi

# Check for xvfb-run (optional, for headless)
USE_XVFB=false
if command -v xvfb-run &> /dev/null; then
    USE_XVFB=true
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to render a single view
render_view() {
    local part_name="$1"
    local output_name="$2"
    local camera="$3"

    echo "Rendering: $output_name..."

    local output_file="$OUTPUT_DIR/${output_name}.png"
    local cmd="openscad"

    # Add xvfb-run for headless if available and no display
    if [ "$USE_XVFB" = true ] && [ -z "$DISPLAY" ]; then
        cmd="xvfb-run -a $cmd"
    fi

    $cmd \
        -o "$output_file" \
        -D "part=\"$part_name\"" \
        --imgsize=$WIDTH,$HEIGHT \
        --camera="$camera" \
        --colorscheme="$COLORSCHEME" \
        --projection=p \
        "$SCAD_FILE" 2>/dev/null || {
            echo "  WARNING: Render failed for $output_name (OpenSCAD may need display)"
            return 1
        }

    echo "  Created: $output_file"
}

# Camera angles for different views
# Format: translateX,translateY,translateZ,rotX,rotY,rotZ,distance

# Exploded view - isometric angle from above
CAM_EXPLODED="0,0,150,55,0,45,600"

# Assembled view - similar but closer
CAM_ASSEMBLED="0,0,80,55,0,45,300"

# Cross section - side view to show internal features
CAM_CROSS_SECTION="0,0,60,90,0,0,250"

# Deployment array - bird's eye view tilted
CAM_DEPLOYMENT="0,0,50,70,0,30,500"

# Component callout - close-up isometric
CAM_CALLOUT="0,0,70,60,0,35,200"

show_help() {
    echo "HIRT OpenSCAD Visualization Renderer"
    echo ""
    echo "Usage: $0 [view_name|all|--help]"
    echo ""
    echo "Available views:"
    echo "  exploded       - Exploded view with parts separated (20mm gaps)"
    echo "  assembled      - Tight assembly showing complete probe"
    echo "  cross_section  - Half-section showing internal details"
    echo "  deployment     - 4 probes with ground plane and hub"
    echo "  callout        - Component detail with annotation markers"
    echo "  all            - Render all views (default)"
    echo ""
    echo "Output directory: $OUTPUT_DIR"
}

render_all() {
    echo "=== HIRT Visualization Render ==="
    echo "Output: $OUTPUT_DIR"
    echo ""

    render_view "exploded_view" "probe_exploded" "$CAM_EXPLODED"
    render_view "isometric_assembled" "probe_assembled" "$CAM_ASSEMBLED"
    render_view "cross_section" "probe_cross_section" "$CAM_CROSS_SECTION"
    render_view "deployment_array" "deployment_array" "$CAM_DEPLOYMENT"
    render_view "component_callout" "component_callout" "$CAM_CALLOUT"

    echo ""
    echo "=== Render complete ==="
    ls -la "$OUTPUT_DIR"/*.png 2>/dev/null || echo "No PNG files generated"
}

# Main
case "${1:-all}" in
    --help|-h)
        show_help
        ;;
    exploded)
        render_view "exploded_view" "probe_exploded" "$CAM_EXPLODED"
        ;;
    assembled)
        render_view "isometric_assembled" "probe_assembled" "$CAM_ASSEMBLED"
        ;;
    cross_section)
        render_view "cross_section" "probe_cross_section" "$CAM_CROSS_SECTION"
        ;;
    deployment)
        render_view "deployment_array" "deployment_array" "$CAM_DEPLOYMENT"
        ;;
    callout)
        render_view "component_callout" "component_callout" "$CAM_CALLOUT"
        ;;
    all)
        render_all
        ;;
    *)
        echo "Unknown view: $1"
        echo "Run '$0 --help' for available options."
        exit 1
        ;;
esac
