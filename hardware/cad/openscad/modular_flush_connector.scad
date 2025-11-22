/*
 * HIRT Modular Flush Connector System
 * Target Rod: 16mm OD x 12mm ID Fiberglass Tube
 * 
 * Features:
 * - Flush 16mm OD match
 * - M12 threads (FULLY PRINTED) - "Chunky" Trapezoidal Profile for Printability
 * - Center pass-through for wiring
 * - O-ring seal groove
 * - Printing Optimizations: SUPER BRIM and rigid scaffolding
 */

// --- Parameters ---
ROD_OD = 16.0;
ROD_ID = 12.0;         // Inner diameter of fiberglass tube
ROD_INSERT_DEPTH = 20.0; // How far it glues into the tube

THREAD_DIA_MAJOR = 11.8; // Slight undersize for male fit (clearance)
THREAD_DIA_MINOR = 10.4; // Female hole size
THREAD_PITCH = 1.75;

THREAD_LEN = 15.0;
WIRE_HOLE_DIA = 6.0;    // Center wiring channel

SENSOR_BODY_LEN = 70.0; // Length of the exposed sensor section (Female side)
THREAD_HOLE_DEPTH = 25.0; // Depth of thread hole

FLANGE_THICKNESS = 2.0; // Stop flange to butt against tube end

// Render Control
part = "all"; // "all", "male_array", "female_array", "mixed_array"

$fn = 64; // Resolution

// --- ROBUST CHUNKY THREAD MODULE ---
module simple_thread(od, len, pitch, internal=false) {
    // Generates a robust trapezoidal thread (Acme-like) 
    // that is much easier to print than sharp ISO threads.
    
    // Thread Dimensions
    // Ensure deep overlap with core to prevent "paper thin" non-manifold issues
    
    // Base Radius (Core)
    // Male: Core is smaller (thread adds to OD)
    // Female: Core is the Hole diameter (thread subtracts from solid)
    
    // For M12x1.75
    // Tooth Height approx 1mm
    t_height = 0.6 * pitch; 
    
    base_r = internal ? (od/2) : (od/2 - t_height);
    
    // Clearance
    clearance = 0.15; // Radial clearance
    adj_base_r = internal ? base_r + clearance : base_r - clearance;
    
    // Tooth Geometry (Trapezoid)
    // Root width (wide base for strength)
    root_w = 0.75 * pitch; 
    // Tip width (no sharp edges)
    tip_w = 0.25 * pitch;
    
    // Overlap into cylinder (anchoring)
    overlap = 0.5; 
    
    turns = len / pitch;
    
    union() {
        // Core cylinder
        cylinder(r=adj_base_r, h=len, $fn=$fn);
        
        // Spiral thread tooth
        // We use linear_extrude with twist on a offset 2D profile
        linear_extrude(height=len, twist=-360*turns, slices=turns*30, convexity=10)
            polygon(points=[
                [adj_base_r - overlap, -root_w/2], // Bottom Inner (Anchor)
                [adj_base_r + t_height, -tip_w/2], // Bottom Outer (Tip)
                [adj_base_r + t_height, tip_w/2],  // Top Outer (Tip)
                [adj_base_r - overlap, root_w/2]   // Top Inner (Anchor)
            ]);
    }
}

module iso_thread_12(len, internal=false) {
    // M12x1.75 Optimized
    pitch = 1.75;
    od = 12.0;
    
    simple_thread(od, len, pitch, internal);
}


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
                
            // 3. Threaded Section (PRINTED THREAD)
            translate([0,0,ROD_INSERT_DEPTH + FLANGE_THICKNESS])
                iso_thread_12(THREAD_LEN, internal=false);
        }
        
        // Center Wire Hole
        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN + 2);
            
        // O-Ring Groove
        translate([0,0,ROD_INSERT_DEPTH + FLANGE_THICKNESS])
            difference() {
                cylinder(d=12.0 + 2.5, h=1);
                cylinder(d=12.0 + 0.5, h=1);
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
            
        // Threaded Socket (PRINTED THREAD)
        translate([0,0,ROD_INSERT_DEPTH + SENSOR_BODY_LEN - THREAD_HOLE_DEPTH])
            // We difference the thread shape
            iso_thread_12(THREAD_HOLE_DEPTH + 1, internal=true);
            
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
    difference() {
        hull() {
            translate([0, 0, 0]) cylinder(r=radius, h=0.28);
            translate([spacing, 0, 0]) cylinder(r=radius, h=0.28);
            translate([0, spacing, 0]) cylinder(r=radius, h=0.28);
            translate([spacing, spacing, 0]) cylinder(r=radius, h=0.28);
        }
        // CLEANOUT
        translate([0, 0, -1]) cylinder(d=WIRE_HOLE_DIA, h=2);
        translate([spacing, 0, -1]) cylinder(d=WIRE_HOLE_DIA, h=2);
        translate([0, spacing, -1]) cylinder(d=WIRE_HOLE_DIA, h=2);
        translate([spacing, spacing, -1]) cylinder(d=WIRE_HOLE_DIA, h=2);
    }
}

module rigid_scaffolding(spacing, z_height, parts_height_config="mixed") {
    // z_height is target height
    // parts_height_config helps decide if we need to avoid threads
    
    // For mixed array:
    // Male parts are at (0,0) and (spacing,0). Thread starts at ~22mm.
    // Female parts are at (0,spacing) and (spacing,spacing). No threads on OD.
    
    beam_width = 2.5;
    beam_height = 1.2; 
    
    // CHECK: If this is the mid-scaffolding (z=30), it hits the male threads!
    // Male insert: 20mm insert + 2mm flange = 22mm height. Then 15mm thread starts.
    // So threads are from Z=22 to Z=37.
    // Scaffolding at Z=30 cuts right through them.
    
    // FIX: Lower scaffolding to Z=20 (Flange level)
    // The calling module controls Z.
    
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
    brim_rad = 12.0; 
    
    union() {
        translate([0, 0, 0]) male_insert();
        translate([spacing, 0, 0]) male_insert();
        
        translate([0, spacing, 0]) female_sensor_module();
        translate([spacing, spacing, 0]) female_sensor_module();
    }
    
    super_brim(spacing, brim_rad);
    
    // Scaffolding logic
    difference() {
        union() {
            // LOWERED mid-scaffolding to Z=20mm to grab the FLANGE, not the thread.
            // Flange is at Z=20 to 22. Scaffolding centered at 20 might hit the rod insert section?
            // Rod insert is Z=0 to 20.
            // Flange is Z=20 to 22.
            // Thread is Z=22+.
            // Safest place: Z=18 (High up on the insert section) or Z=21 (On the flange).
            // Z=21 is best (Solid flange).
            rigid_scaffolding(spacing, 21.0);
            
            // Top bar remains high for female parts
            translate([spacing/2, spacing, 85]) 
                cube([spacing, 3.0, 1.5], center=true); 
        }
        
        cutout_d = ROD_OD - 1.0; 
        
        translate([0, 0, -10]) cylinder(d=cutout_d, h=200);
        translate([spacing, 0, -10]) cylinder(d=cutout_d, h=200);
        translate([0, spacing, -10]) cylinder(d=cutout_d, h=200);
        translate([spacing, spacing, -10]) cylinder(d=cutout_d, h=200);
    }
}

// --- Render Logic ---

if (part == "male_array") {
    print_array_mixed(); 
} else if (part == "female_array") {
    print_array_mixed();
} else if (part == "mixed_array") {
    print_array_mixed();
} else {
    print_array_mixed();
}
