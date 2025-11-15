/*
 * HIRT Rod Coupler
 * 
 * Threaded coupler for joining rod sections
 * 
 * Specifications:
 * - Length: 75mm
 * - Outer Diameter: 30mm
 * - Thread: M/F (matches rod threads)
 * - Material: Glass-filled nylon or PETG
 * 
 * Features:
 * - Female threads on both ends
 * - Smooth center section for grip
 * - O-ring grooves for sealing
 */

// Parameters
coupler_length = 75;         // Total length (mm)
coupler_od = 30;             // Outer diameter (mm)
thread_id = 25;              // Thread inner diameter (matches rod OD)
thread_engagement = 25;      // Thread engagement length per side (mm)
wall_thickness = 3;          // Wall thickness (mm)
o_ring_groove_width = 2;     // O-ring groove width (mm)
o_ring_groove_depth = 1.5;   // O-ring groove depth (mm)
o_ring_position = 2;         // Distance from end to O-ring groove (mm)

// Calculated dimensions
center_section_length = coupler_length - 2 * thread_engagement;
thread_od = thread_id + 2 * wall_thickness;

// Main coupler body
module coupler_body() {
    difference() {
        // Outer cylinder
        cylinder(h = coupler_length, d = coupler_od, $fn = 100);
        
        // Internal thread cavities (simplified - use tap for actual threads)
        // Left side
        translate([0, 0, -0.5])
            cylinder(h = thread_engagement + 0.5, d = thread_id, $fn = 100);
        
        // Right side
        translate([0, 0, coupler_length - thread_engagement])
            cylinder(h = thread_engagement + 0.5, d = thread_id, $fn = 100);
        
        // Center section (smooth, no threads)
        translate([0, 0, thread_engagement - 0.5])
            cylinder(h = center_section_length + 1, d = thread_id + 1, $fn = 100);
    }
}

// O-ring grooves
module o_ring_grooves() {
    // Left O-ring groove
    translate([0, 0, o_ring_position])
        rotate_extrude($fn = 100)
            translate([thread_id/2, 0, 0])
                square([o_ring_groove_depth, o_ring_groove_width]);
    
    // Right O-ring groove
    translate([0, 0, coupler_length - o_ring_position - o_ring_groove_width])
        rotate_extrude($fn = 100)
            translate([thread_id/2, 0, 0])
                square([o_ring_groove_depth, o_ring_groove_width]);
}

// Complete coupler
module coupler() {
    difference() {
        coupler_body();
        o_ring_grooves();
    }
}

// Export
coupler();

// For visualization with O-rings (uncomment to see):
// module coupler_with_o_rings() {
//     coupler();
//     // O-rings (for visualization only)
//     color("black") {
//         translate([0, 0, o_ring_position + o_ring_groove_width/2])
//             rotate_extrude($fn = 100)
//                 translate([thread_id/2 + o_ring_groove_depth/2, 0, 0])
//                     circle(d = o_ring_groove_width, $fn = 50);
//         translate([0, 0, coupler_length - o_ring_position - o_ring_groove_width/2])
//             rotate_extrude($fn = 100)
//                 translate([thread_id/2 + o_ring_groove_depth/2, 0, 0])
//                     circle(d = o_ring_groove_width, $fn = 50);
//     }
// }
// coupler_with_o_rings();

