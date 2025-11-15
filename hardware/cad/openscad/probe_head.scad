/*
 * HIRT Probe Head (Nose Capsule)
 * 
 * 3D printable probe head for electronics pod
 * 
 * Specifications:
 * - Outer Diameter: 30mm
 * - Length: 100mm
 * - Wall Thickness: 2mm
 * - Material: PETG or ABS
 * 
 * Features:
 * - Threaded connection to rod (top)
 * - Removable cap (bottom)
 * - Cable gland mount (side)
 * - Internal space for electronics and coils
 */

// Parameters (customize as needed)
probe_head_od = 30;          // Outer diameter (mm)
probe_head_length = 100;    // Total length (mm)
wall_thickness = 2;          // Wall thickness (mm)
thread_od = 25;             // Thread outer diameter (matches rod)
thread_length = 20;          // Thread engagement length (mm)
thread_pitch = 2.5;          // Thread pitch (mm) - 4 TPI equivalent
cable_gland_dia = 8;         // Cable gland hole diameter (mm)
cable_gland_height = 15;     // Height of cable gland mount (mm)

// Calculated dimensions
probe_head_id = probe_head_od - 2 * wall_thickness;
cap_thickness = 3;           // Cap wall thickness (mm)
cap_thread_length = 10;      // Cap thread length (mm)

// Main body
module probe_head_body() {
    difference() {
        // Outer cylinder
        cylinder(h = probe_head_length, d = probe_head_od, $fn = 100);
        
        // Inner cavity
        translate([0, 0, cap_thickness])
            cylinder(h = probe_head_length - cap_thickness, d = probe_head_id, $fn = 100);
        
        // Cable gland hole
        translate([probe_head_od/2, 0, cable_gland_height])
            rotate([0, 90, 0])
                cylinder(h = wall_thickness + 1, d = cable_gland_dia, $fn = 50);
    }
}

// Threaded section (top, connects to rod)
module top_thread() {
    difference() {
        // Threaded cylinder
        cylinder(h = thread_length, d = thread_od + 0.5, $fn = 100);
        
        // Internal thread (simplified - use tap for actual threads)
        translate([0, 0, -0.5])
            cylinder(h = thread_length + 1, d = thread_od - 1, $fn = 100);
    }
}

// Cap (removable bottom)
module cap() {
    difference() {
        // Cap body
        cylinder(h = cap_thickness, d = probe_head_od, $fn = 100);
        
        // Internal thread for cap (simplified)
        translate([0, 0, -0.5])
            cylinder(h = cap_thickness + 1, d = probe_head_id - 0.5, $fn = 100);
    }
    
    // Cap thread section
    translate([0, 0, cap_thickness])
        difference() {
            cylinder(h = cap_thread_length, d = probe_head_id - 0.2, $fn = 100);
            translate([0, 0, -0.5])
                cylinder(h = cap_thread_length + 1, d = probe_head_id - 1, $fn = 100);
        }
}

// Assembly view (comment out for separate parts)
module assembly() {
    // Main body
    probe_head_body();
    
    // Top thread
    translate([0, 0, probe_head_length])
        top_thread();
    
    // Cap (shown separated)
    translate([0, 0, -cap_thickness - cap_thread_length - 5])
        cap();
}

// Export individual parts
// Uncomment the part you want to export:

// Main body only
probe_head_body();

// Top thread only (for separate printing if needed)
// translate([0, 0, probe_head_length]) top_thread();

// Cap only
// cap();

// Full assembly (for visualization)
// assembly();

