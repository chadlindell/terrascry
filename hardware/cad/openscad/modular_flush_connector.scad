/*
 * HIRT Modular Flush Connector System
 * Target Rod: 16mm OD x 12mm ID Fiberglass Tube
 * 
 * This file generates:
 * 1. Male Insert (Plug): Epoxied into Rod A
 * 2. Female Insert (Socket/Sensor Body): Epoxied into Rod B
 *
 * Features:
 * - Flush 16mm OD match
 * - M12 threads (robust for printing)
 * - Center pass-through for wiring
 * - O-ring seal groove
 * - Printing Optimizations: Chamfers and flat bases
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

$fn = 64; // Resolution

// --- Modules ---

module chamfer_cylinder(d, h, chamfer=1.0) {
    // Simple cylinder with chamfered ends for easier insertion
    union() {
        translate([0,0,chamfer]) cylinder(d=d, h=h-2*chamfer);
        cylinder(d1=d-2*chamfer, d2=d, h=chamfer);
        translate([0,0,h-chamfer]) cylinder(d1=d, d2=d-2*chamfer, h=chamfer);
    }
}

module male_insert() {
    difference() {
        union() {
            // 1. Insert Section (Goes inside rod) - with slight chamfer for glue insertion
            cylinder(d=ROD_ID - 0.2, h=ROD_INSERT_DEPTH - 1);
            translate([0,0,ROD_INSERT_DEPTH-1]) cylinder(d1=ROD_ID-0.2, d2=ROD_OD, h=1); // Taper to flange
            
            // 2. Stop Flange (Matches Rod OD)
            translate([0,0,ROD_INSERT_DEPTH])
                cylinder(d=ROD_OD, h=FLANGE_THICKNESS);
                
            // 3. Threaded Section (Male)
            translate([0,0,ROD_INSERT_DEPTH + FLANGE_THICKNESS])
                cylinder(d=THREAD_DIA, h=THREAD_LEN);
        }
        
        // Center Wire Hole
        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN + 2);
            
        // O-Ring Groove on Flange Face (optional)
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
            // 1. Insert Section (Goes inside rod)
            cylinder(d=ROD_ID - 0.2, h=ROD_INSERT_DEPTH); 
            
            // 2. Sensor Body (Exposed section)
            translate([0,0,ROD_INSERT_DEPTH])
                cylinder(d=ROD_OD, h=SENSOR_BODY_LEN);
        }
        
        // Center Wire Hole (Through whole part)
        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=ROD_INSERT_DEPTH + SENSOR_BODY_LEN + 2);
            
        // Threaded Socket (Female) at the TOP end of the sensor body
        translate([0,0,ROD_INSERT_DEPTH + SENSOR_BODY_LEN - THREAD_LEN])
            cylinder(d=THREAD_DIA + 0.5, h=THREAD_LEN + 1); // +0.5 clearance
            
        // --- Sensor Cutouts ---
        
        // ERT Ring Grooves (Top and Bottom)
        translate([0,0,ROD_INSERT_DEPTH + 10])
            difference() {
                cylinder(d=ROD_OD+0.1, h=4);
                cylinder(d=ROD_OD-1, h=4); // 0.5mm deep groove
            }
            
        translate([0,0,ROD_INSERT_DEPTH + SENSOR_BODY_LEN - 14])
             difference() {
                cylinder(d=ROD_OD+0.1, h=4);
                cylinder(d=ROD_OD-1, h=4);
            }
            
        // MIT Coil Area (Recessed center section)
        translate([0,0,ROD_INSERT_DEPTH + 20])
             difference() {
                cylinder(d=ROD_OD+0.1, h=20);
                cylinder(d=ROD_OD-2, h=20); // 1mm deep recess for coil
            }
    }
}

// --- Render ---

// Layout for viewing
translate([-15, 0, 0]) male_insert();
translate([15, 0, 0]) female_sensor_module();
