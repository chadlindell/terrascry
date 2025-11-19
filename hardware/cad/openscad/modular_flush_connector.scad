/*
 * HIRT Modular Flush Connector System
 * Target Rod: 16mm OD x 12mm ID Fiberglass Tube
 * 
 * Features:
 * - Flush 16mm OD match
 * - M12 threads (12mm nominal) - Sized for Die/Tap cutting
 * - Center pass-through for wiring
 * - O-ring seal groove
 * - Printing Optimizations: Chamfers, flat bases, breakaway support tabs
 */

// --- Parameters ---
ROD_OD = 16.0;
ROD_ID = 12.0;         // Inner diameter of fiberglass tube
ROD_INSERT_DEPTH = 20.0; // How far it glues into the tube

// THREADING DIMENSIONS FOR TAP/DIE
// M12x1.75 Thread
// Male Pin: Printed slightly oversize (12.2mm) to ensure Die cuts clean full threads
// Female Hole: Printed slightly undersize (10.5mm) for Tap drill diameter (standard M12x1.75 drill is 10.2mm, using 10.5 for easier plastic tapping)

THREAD_DIA_MALE = 12.2;  
THREAD_DIA_FEMALE = 10.5; 

THREAD_LEN = 15.0;
WIRE_HOLE_DIA = 6.0;    // Center wiring channel

SENSOR_BODY_LEN = 60.0; // Length of the exposed sensor section (Female side)
FLANGE_THICKNESS = 2.0; // Stop flange to butt against tube end

// Render Control
part = "all"; // "all", "male_array", "female_array", "mixed_array"

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
                
            // 3. Threaded Section (Male stud for Die cutting)
            translate([0,0,ROD_INSERT_DEPTH + FLANGE_THICKNESS])
                cylinder(d=THREAD_DIA_MALE, h=THREAD_LEN);
        }
        
        // Center Wire Hole
        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN + 2);
            
        // O-Ring Groove
        translate([0,0,ROD_INSERT_DEPTH + FLANGE_THICKNESS])
            difference() {
                cylinder(d=THREAD_DIA_MALE + 2.5, h=1);
                cylinder(d=THREAD_DIA_MALE + 0.5, h=1);
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
            
        // Threaded Socket (Female hole for Tapping)
        translate([0,0,ROD_INSERT_DEPTH + SENSOR_BODY_LEN - THREAD_LEN])
            cylinder(d=THREAD_DIA_FEMALE, h=THREAD_LEN + 1);
            
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

module print_array_mixed() {
    // 2 Male + 2 Female
    spacing = ROD_OD + 8;
    
    // Row 0: Male Inserts (Shorter)
    translate([0, 0, 0]) male_insert();
    translate([spacing, 0, 0]) male_insert();
    
    // Row 1: Female Sensors (Taller)
    translate([0, spacing, 0]) female_sensor_module();
    translate([spacing, spacing, 0]) female_sensor_module();
    
    // Stabilization - Complex mixed height handling
    
    // Base Layer (All connected)
    stabilization_grid(spacing, 0.2);
    
    // Mid Layer (Z=30) - Connects tops of Male to mids of Female
    stabilization_grid(spacing, 30);
    
    // Top Layer (Z=75) - Connects tops of Female only (Back row)
    tab_width = 1.2;
    tab_height = 0.4;
    translate([0, 0, 75]) {
        // Connect the two female parts at the top
         translate([spacing/2, spacing, 0]) cube([spacing, tab_width, tab_height], center=true);
    }
}

// --- Render Logic ---

if (part == "male_array") {
    print_array_male_insert();
} else if (part == "female_array") {
    print_array_female_sensor();
} else if (part == "mixed_array") {
    print_array_mixed();
} else {
    // Default View
    translate([-30, -30, 0]) print_array_male_insert();
    translate([30, -30, 0]) print_array_female_sensor();
}
