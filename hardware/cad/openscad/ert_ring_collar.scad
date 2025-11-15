/*
 * HIRT ERT Ring Mounting Collar
 * 
 * Insulating collar for mounting ERT (Electrical Resistivity) rings
 * 
 * Specifications:
 * - Inner Diameter: 25mm (matches rod OD)
 * - Width: 12mm
 * - Material: PETG or ABS (non-conductive)
 * 
 * Features:
 * - Insulating collar for ERT ring
 * - Groove for ring mounting
 * - Wire routing channel
 */

// Parameters
collar_id = 25;              // Inner diameter (matches rod OD)
collar_od = 28;             // Outer diameter (mm)
collar_width = 12;           // Width of collar (mm)
ring_groove_width = 0.5;    // Groove for ring (mm)
ring_groove_depth = 0.2;    // Groove depth (mm)
wire_channel_width = 3;     // Wire routing channel width (mm)
wire_channel_depth = 2;     // Wire channel depth (mm)

// Main collar
module collar() {
    difference() {
        // Outer cylinder
        cylinder(h = collar_width, d = collar_od, $fn = 100);
        
        // Inner hole (fits over rod)
        translate([0, 0, -0.5])
            cylinder(h = collar_width + 1, d = collar_id, $fn = 100);
        
        // Ring groove (for ring to sit in)
        translate([0, 0, collar_width/2])
            rotate_extrude($fn = 100)
                translate([collar_od/2 - ring_groove_depth, 0, 0])
                    square([ring_groove_depth, ring_groove_width]);
        
        // Wire routing channel (slot for wire)
        translate([collar_od/2 - wire_channel_depth, -wire_channel_width/2, -0.5])
            cube([wire_channel_depth + 1, wire_channel_width, collar_width + 1]);
    }
}

// Export
collar();

// For visualization with ring (uncomment to see):
// module collar_with_ring() {
//     collar();
//     // ERT ring (for visualization only)
//     color("silver") {
//         translate([0, 0, collar_width/2])
//             rotate_extrude($fn = 100)
//                 translate([collar_od/2, 0, 0])
//                     square([0.5, 12]);
//     }
// }
// collar_with_ring();

