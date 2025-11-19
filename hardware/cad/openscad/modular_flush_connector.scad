/*
 * HIRT Modular Flush Connector System
 * Target Rod: 16mm OD x 12mm ID Fiberglass Tube
 * 
 * Features:
 * - Flush 16mm OD match
 * - M12 threads (robust for printing)
 * - Center pass-through for wiring
 * - O-ring seal groove
 * - Printing Optimizations: Chamfers, flat bases, breakaway support tabs
 */

// --- Parameters ---
ROD_OD = 16.0;
ROD_ID = 12.0;         // Inner diameter of fiberglass tube
ROD_INSERT_DEPTH = 20.0; // How far it glues into the tube

THREAD_DIA = 10.0;      // M10 Thread (Core dia for printing tolerance)
THREAD_LEN = 15.0;
WIRE_HOLE_DIA = 6.0;    // Center wiring channel

SENSOR_BODY_LEN = 60.0; // Length of the exposed sensor section (Female side)
FLANGE_THICKNESS = 2.0; // Stop flange to butt against tube end

// Render Control
part = "all"; // "all", "male_array", "female_array"

$fn = 64; // Resolution

// --- Modules ---

module chamfer_cylinder(d, h, chamfer=1.0) {
    union() {
        translate([0,0,chamfer]) cylinder(d=d, h=h-2*chamfer);
        cylinder(d1=d-2*chamfer, d2=d, h=chamfer);
        translate([0,0,h-chamfer]) cylinder(d1=d, d2=d-2*chamfer, h=chamfer);
    }
}

module male_insert() {
    difference() {
        union() {
            // 1. Insert Section (Goes inside rod)
            cylinder(d=ROD_ID - 0.2, h=ROD_INSERT_DEPTH - 1);
            translate([0,0,ROD_INSERT_DEPTH-1]) cylinder(d1=ROD_ID-0.2, d2=ROD_OD, h=1); 
            
            // 2. Stop Flange
            translate([0,0,ROD_INSERT_DEPTH])
                cylinder(d=ROD_OD, h=FLANGE_THICKNESS);
                
            // 3. Threaded Section
            translate([0,0,ROD_INSERT_DEPTH + FLANGE_THICKNESS])
                cylinder(d=THREAD_DIA, h=THREAD_LEN);
        }
        
        // Center Wire Hole
        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN + 2);
            
        // O-Ring Groove
        translate([0,0,ROD_INSERT_DEPTH + FLANGE_THICKNESS])
            difference() {
                cylinder(d=THREAD_DIA + 2.5, h=1);
                cylinder(d=THREAD_DIA + 0.5, h=1);
            }
    }
}

module female_sensor_module() {
    difference() {
        union() {
            // 1. Insert Section
            cylinder(d=ROD_ID - 0.2, h=ROD_INSERT_DEPTH); 
            
            // 2. Sensor Body
            translate([0,0,ROD_INSERT_DEPTH])
                cylinder(d=ROD_OD, h=SENSOR_BODY_LEN);
        }
        
        // Center Wire Hole
        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=ROD_INSERT_DEPTH + SENSOR_BODY_LEN + 2);
            
        // Threaded Socket
        translate([0,0,ROD_INSERT_DEPTH + SENSOR_BODY_LEN - THREAD_LEN])
            cylinder(d=THREAD_DIA + 0.5, h=THREAD_LEN + 1);
            
        // --- Sensor Cutouts ---
        translate([0,0,ROD_INSERT_DEPTH + 10])
            difference() { cylinder(d=ROD_OD+0.1, h=4); cylinder(d=ROD_OD-1, h=4); }
            
        translate([0,0,ROD_INSERT_DEPTH + SENSOR_BODY_LEN - 14])
             difference() { cylinder(d=ROD_OD+0.1, h=4); cylinder(d=ROD_OD-1, h=4); }
            
        translate([0,0,ROD_INSERT_DEPTH + 20])
             difference() { cylinder(d=ROD_OD+0.1, h=20); cylinder(d=ROD_OD-2, h=20); }
    }
}

// --- Array Modules for Printing ---

module stabilization_grid(spacing, z_height) {
    tab_width = 1.2;  // Width of the connector
    tab_height = 0.4; // Layer height (1-2 layers)
    
    translate([0, 0, z_height]) {
        // Horizontal connectors
        translate([spacing/2, 0, 0]) cube([spacing, tab_width, tab_height], center=true);
        translate([spacing/2, spacing, 0]) cube([spacing, tab_width, tab_height], center=true);
        
        // Vertical connectors
        translate([0, spacing/2, 0]) cube([tab_width, spacing, tab_height], center=true);
        translate([spacing, spacing/2, 0]) cube([tab_width, spacing, tab_height], center=true);
        
        // Diagonal X cross (optional, keeps square)
        // usually plain grid is enough for 2x2
    }
}

module print_array_male_insert() {
    spacing = ROD_OD + 8; // 24mm spacing (8mm gap)
    
    for (i = [0:1]) {
        for (j = [0:1]) {
            translate([i*spacing, j*spacing, 0])
                male_insert();
        }
    }
    
    // Stabilization tabs
    stabilization_grid(spacing, 0.2); // Base layer
    stabilization_grid(spacing, 30);  // Top of thread area (approx)
}

module print_array_female_sensor() {
    spacing = ROD_OD + 8; // 24mm spacing
    
    for (i = [0:1]) {
        for (j = [0:1]) {
            translate([i*spacing, j*spacing, 0])
                female_sensor_module();
        }
    }
    
    // Stabilization tabs
    stabilization_grid(spacing, 0.2); // Base layer
    stabilization_grid(spacing, 40);  // Mid-point
    stabilization_grid(spacing, 75);  // Near top
}

// --- Render Logic ---

if (part == "male_array") {
    print_array_male_insert();
} else if (part == "female_array") {
    print_array_female_sensor();
} else {
    // Default View
    translate([-30, -30, 0]) print_array_male_insert();
    translate([30, -30, 0]) print_array_female_sensor();
}
