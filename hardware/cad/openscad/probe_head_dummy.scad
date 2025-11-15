/*
 * HIRT Probe Head - Dummy/Test Version
 * 
 * Simplified probe head for fit testing and prototyping
 * 
 * This is a simplified version without threads for quick testing
 * Use for:
 * - Fit testing with electronics
 * - Coil placement testing
 * - Size verification
 * - Prototype assembly
 */

// Parameters (same as full version)
probe_head_od = 30;
probe_head_length = 100;
wall_thickness = 2;
cable_gland_dia = 8;
cable_gland_height = 15;

// Calculated
probe_head_id = probe_head_od - 2 * wall_thickness;

// Simplified body (no threads, open top and bottom)
module dummy_probe_head() {
    difference() {
        // Outer cylinder
        cylinder(h = probe_head_length, d = probe_head_od, $fn = 100);
        
        // Inner cavity
        translate([0, 0, -0.5])
            cylinder(h = probe_head_length + 1, d = probe_head_id, $fn = 100);
        
        // Cable gland hole
        translate([probe_head_od/2, 0, cable_gland_height])
            rotate([0, 90, 0])
                cylinder(h = wall_thickness + 1, d = cable_gland_dia, $fn = 50);
        
        // Top opening (for easy access)
        translate([0, 0, probe_head_length - 5])
            cylinder(h = 6, d = probe_head_id + 2, $fn = 100);
        
        // Bottom opening (for coil insertion)
        translate([0, 0, -1])
            cylinder(h = 6, d = probe_head_id + 2, $fn = 100);
    }
}

// Export
dummy_probe_head();

