/*
 * HIRT Modular Flush Connector System
 * Target Rod: 16mm OD x 12mm ID Fiberglass Tube
 * 
 * Features:
 * - Flush 16mm OD match
 * - M12 threads (FULLY PRINTED) - ISO Profile M12x1.75
 * - Center pass-through for wiring
 * - O-ring seal groove
 * - Printing Optimizations: SUPER BRIM and rigid scaffolding
 */

// --- Parameters ---
ROD_OD = 16.0;
ROD_ID = 12.0;         // Inner diameter of fiberglass tube
ROD_INSERT_DEPTH = 20.0; // How far it glues into the tube

// THREADING DIMENSIONS (PRINTED)
// Pitch = 1.75mm
// Major Dia = 12mm

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

$fn = 64; // Resolution for cylinders

// --- Thread Module (Linear Extrude Twist Method) ---
module printed_thread(od, len, pitch, internal=false) {
    // Generates a printable metric-like thread using twisted extrusion
    // This is robust for 3D printing and boolean operations
    
    major_r = od / 2;
    // Triangle height for M12x1.75 is approx 1mm
    t_h = 0.866 * pitch; 
    
    // Effective radius adjustment for internal/external
    eff_r = internal ? major_r : major_r - t_h/8;
    
    // Twist calculation
    turns = len / pitch;
    twist_angle = -360 * turns;
    
    intersection() {
        // Crop top/bottom to length
        cylinder(r=od/2 + 2, h=len, $fn=$fn);
        
        // The twisted thread coil
        linear_extrude(height=len, twist=twist_angle, slices=turns*30, convexity=10)
            projection(cut=true)
            rotate([90,0,0])
            translate([0, 0, 0]) // Center at origin for rotation
            // Generate profile at distance
            translate([eff_r, 0, 0]) 
            rotate([0, 0, 90]) // Face outward
            polygon(points=[
                [-pitch/2, 0],
                [0, -t_h], // Point inwards? No, extrude twist rotates around Z
                [pitch/2, 0]
            ]);
            // This projection hack is tricky. 
            // Better method: Polygon offset from center rotated.
    }
}

// --- ROBUST THREAD MODULE (Polyhedron-free, simple geometry) ---
module simple_thread(od, len, pitch, internal=false) {
    // Create a spiral using linear_extrude on a 2D shape offset from center
    
    // Profile: 60 deg triangle
    t_h = 0.866 * pitch;
    
    // Determine base radius
    // For Male: od is the peaks. Base is od/2 - t_h
    // For Female: od is the valleys. Base is od/2
    
    base_r = internal ? (od/2) : (od/2 - t_h);
    tip_r = internal ? (od/2 - t_h) : (od/2);
    
    // Adjust for clearance
    clearance = 0.15;
    adj_base_r = internal ? base_r + clearance : base_r - clearance;
    
    turns = len / pitch;
    
    union() {
        // Core cylinder
        cylinder(r=adj_base_r, h=len, $fn=$fn);
        
        // Spiral thread
        linear_extrude(height=len, twist=-360*turns, slices=turns*20, convexity=10)
            translate([adj_base_r, 0, 0])
            circle(r=t_h/1.8, $fn=3); // Triangle approximated by 3-sided circle (quick & dirty thread)
            // Actually, using a circle makes a "Knuckle Thread" which is stronger for printing
            // and much smoother to screw in.
            // ISO threads are sharp and prone to stress concentrations.
            // We will use a round profile thread (Knuckle) for printed parts.
            // Ideally M12x1.75 Knuckle.
    }
}

module iso_thread_12(len, internal=false) {
    // Hardcoded M12x1.75 optimized for printing
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

module rigid_scaffolding(spacing, z_height) {
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
            rigid_scaffolding(spacing, 30);
            translate([spacing/2, spacing, 85]) 
                cube([spacing, 3.0, 1.5], center=true); 
        }
        
        // SUBTRACT PARTS from Scaffolding
        // FIX: Subtracting ROD_OD (approx) ensures scaffolding only touches OUTSIDE
        // and does NOT penetrate to the inner hole.
        // ROD_OD is 16. We subtract 15mm cylinders.
        // This means scaffolding penetrates 0.5mm into the 16mm shell.
        // Inner hole (6mm) is safe.
        
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
