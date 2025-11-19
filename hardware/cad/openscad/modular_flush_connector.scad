/*
 * HIRT Modular Flush Connector System
 * Target Rod: 16mm OD x 12mm ID Fiberglass Tube
 * 
 * Features:
 * - Flush 16mm OD match
 * - M12 threads (12mm nominal) - Sized for Die/Tap cutting
 * - Center pass-through for wiring
 * - O-ring seal groove
 * - Printing Optimizations: SUPER BRIM and rigid scaffolding
 */

// --- Parameters ---
ROD_OD = 16.0;
ROD_ID = 12.0;         // Inner diameter of fiberglass tube
ROD_INSERT_DEPTH = 20.0; // How far it glues into the tube

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

// --- Stability Enhancements ---

module super_brim(spacing, radius) {
    // A solid 0.28mm sheet connecting the bases
    // Acts as a super-raft to prevent lifting
    hull() {
        translate([0, 0, 0]) cylinder(r=radius, h=0.28);
        translate([spacing, 0, 0]) cylinder(r=radius, h=0.28);
        translate([0, spacing, 0]) cylinder(r=radius, h=0.28);
        translate([spacing, spacing, 0]) cylinder(r=radius, h=0.28);
    }
}

module rigid_scaffolding(spacing, z_height) {
    // Much thicker tabs that won't flex
    beam_width = 2.5;
    beam_height = 1.2; 
    
    translate([0, 0, z_height]) {
        // Box frame
        translate([spacing/2, 0, 0]) cube([spacing, beam_width, beam_height], center=true);
        translate([spacing/2, spacing, 0]) cube([spacing, beam_width, beam_height], center=true);
        
        translate([0, spacing/2, 0]) cube([beam_width, spacing, beam_height], center=true);
        translate([spacing, spacing/2, 0]) cube([beam_width, spacing, beam_height], center=true);
        
        // Cross bracing
        rotate([0,0,45]) translate([spacing*0.707, 0, 0]) cube([spacing*1.4, beam_width, beam_height], center=true);
    }
}

module print_array_mixed() {
    // 2 Male + 2 Female
    spacing = ROD_OD + 8; // 24mm
    brim_rad = 12.0; // Radius of brim around each part center
    
    // Parts
    translate([0, 0, 0]) male_insert();
    translate([spacing, 0, 0]) male_insert();
    
    translate([0, spacing, 0]) female_sensor_module();
    translate([spacing, spacing, 0]) female_sensor_module();
    
    // 1. SUPER BRIM (Base Stability)
    super_brim(spacing, brim_rad);
    
    // 2. Rigid Scaffolding (Mid-height)
    rigid_scaffolding(spacing, 30);
    
    // 3. Top Scaffolding (High up for female parts)
    // Connects the two tall parts
    translate([spacing/2, spacing, 75]) 
        cube([spacing, 3.0, 1.5], center=true); // Thick bar at top
}

// --- Render Logic ---

if (part == "male_array") {
    // Only mixed array updated with super brim for now as requested
    print_array_mixed(); 
} else if (part == "female_array") {
    print_array_mixed();
} else if (part == "mixed_array") {
    print_array_mixed();
} else {
    print_array_mixed();
}
