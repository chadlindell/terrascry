/*
 * HIRT Modular Flush Connector System v2.0
 * Target Rod: 16mm OD x 12mm ID Fiberglass Tube
 *
 * Thread Library: Uses threads-scad by rcolyer for robust polyhedron mesh
 * https://github.com/rcolyer/threads-scad
 *
 * =============================================================================
 * ARCHITECTURE: Independent Sensor Bodies
 * =============================================================================
 *
 * This system separates sensors from fiberglass rod connectors for true modularity.
 *
 * Fiberglass Section:
 *     [Male Cap] ======== Fiberglass Tube ======== [Male Cap]
 *     (epoxied)           (50cm or 100cm)          (epoxied)
 *
 * Independent Sensor Body:
 *     [Female Socket] ==== Sensor Body ==== [Female Socket]
 *     (threaded)          (70mm, 16mm OD)    (threaded)
 *
 * Assembly Example (1.5m probe):
 *     [Junction Cap]
 *          |
 *     [Male Cap]----[50cm Fiberglass]----[Male Cap]
 *          |                                  |
 *     [Sensor Body - Dual Female]
 *          |                                  |
 *     [Male Cap]----[100cm Fiberglass]---[Male Cap]
 *          |
 *     [Sensor Body - Dual Female]
 *          |
 *     [Probe Tip]
 *
 * =============================================================================
 * COMPONENTS
 * =============================================================================
 *
 * | Component          | Description                    | Threads              |
 * |--------------------|--------------------------------|----------------------|
 * | male_rod_cap       | Epoxies into fiberglass end    | External M12 male    |
 * | sensor_body_dual   | Independent sensor unit        | Internal M12 (BOTH)  |
 * | probe_tip          | Bottom terminator              | Internal M12 (top)   |
 * | junction_cap       | Top terminator, cable exit     | Internal M12 (bottom)|
 *
 * =============================================================================
 * FEATURES
 * =============================================================================
 * - Flush 16mm OD match to fiberglass tube
 * - M12 threads (FULLY PRINTED) - "Chunky" Trapezoidal Profile
 * - Center 6mm pass-through for wiring
 * - O-ring seal grooves
 * - ERT ring groove and MIT coil zone on sensor bodies
 * - Radial wire entry holes for sensor connections
 * - Printing Optimizations: SUPER BRIM and rigid scaffolding
 */

// =============================================================================
// EXTERNAL LIBRARIES
// =============================================================================
use <threads/threads.scad>

// =============================================================================
// PARAMETERS
// =============================================================================

// --- Rod Dimensions ---
ROD_OD = 16.0;           // Fiberglass tube outer diameter
ROD_ID = 12.0;           // Fiberglass tube inner diameter
ROD_INSERT_DEPTH = 20.0; // How far male cap glues into tube

// --- Thread Parameters ---
THREAD_DIA_MAJOR = 11.8; // Slight undersize for male fit (clearance)
THREAD_DIA_MINOR = 10.4; // Female hole size (for blanks/tapping)
THREAD_PITCH = 1.75;     // M12 thread pitch
THREAD_LEN = 15.0;       // External thread length
THREAD_HOLE_DEPTH = 25.0; // Female socket depth

// --- General Dimensions ---
WIRE_HOLE_DIA = 6.0;     // Center wiring channel
FLANGE_THICKNESS = 2.0;  // Stop flange thickness

// --- Sensor Body Parameters ---
SENSOR_BODY_LEN = 70.0;      // Total sensor body length
SENSOR_WALL_THICK = 2.0;     // Minimum wall at thread area

// --- Sensor Feature Parameters ---
ERT_GROOVE_WIDTH = 4.0;      // Width of ERT ring groove
ERT_GROOVE_DEPTH = 1.0;      // Depth of ERT ring groove (into 16mm OD)
MIT_COIL_ZONE = 15.0;        // Length of coil winding area
WIRE_ENTRY_DIA = 1.5;        // Radial holes for coil/ring wires

// --- Probe Tip Parameters ---
TIP_LENGTH = 30.0;           // Total probe tip length
TIP_POINT_DIA = 8.0;         // Diameter at tip point
TIP_TAPER_LEN = 15.0;        // Length of tapered section

// --- Junction Cap Parameters ---
JUNCTION_CAP_LEN = 35.0;     // Total junction cap length
CABLE_GLAND_BOSS_DIA = 20.0; // Boss diameter for cable gland
CABLE_GLAND_THREAD_DIA = 12.5; // M12 or PG7 gland thread hole
CABLE_GLAND_BOSS_LEN = 15.0; // Length of cable gland boss

// --- Render Control ---
// Options: "all", "male_cap_array", "sensor_body_array", "mixed_system_array",
//          "male_cap_single", "sensor_body_single", "probe_tip_single",
//          "junction_cap_single", "complete_system", "blanks_array"
//          Add "_oversize" suffix for die/tap chasing version (e.g., "mixed_system_array_oversize")
//
// Visualization options (for documentation renders):
//          "exploded_view"      - Parts separated with 20mm gaps
//          "cross_section"      - Half-section showing internal wire channel
//          "isometric_assembled" - Tight assembly for "what you're building"
//          "deployment_array"   - 4 probes with ground plane and hub
//          "component_callout"  - Single sensor body with annotation markers
part = "all";

// Thread oversize for die/tap chasing (0 = print-ready, 0.3 = chase with die/tap)
THREAD_OVERSIZE = 0;

$fn = 64; // Resolution

// =============================================================================
// THREAD MODULES
// =============================================================================

module simple_thread(od, len, pitch, internal=false, oversize=0) {
    // Uses threads-scad library for robust polyhedron mesh generation
    // This avoids degenerate triangles that occur with linear_extrude + twist
    // oversize: extra material for tap/die chasing (use 0.3-0.5 for chase-ready)
    //
    // The threads-scad library generates threads as a single polyhedron using
    // list comprehensions, producing valid, smooth meshes for FDM slicers.

    // Base tolerance for threads-scad
    base_tolerance = 0.4;

    if (internal) {
        // Female thread (for subtraction via difference())
        // ScrewThread creates geometry to subtract - leaves internal threads in the part
        // Uses same approach as threads-scad ScrewHole: slightly oversized thread
        extra = 0.5;  // Extra length to extend beyond body surfaces

        // For print-ready: use standard clearance for FDM fit
        // For tap-ready (oversize>0): subtract less material, leaving more for tap
        thread_dia = 1.01*od + 1.25*base_tolerance - oversize;

        // Just the thread geometry - no core cylinder (would fill in thread valleys)
        translate([0, 0, -extra/2])
            ScrewThread(thread_dia, len + extra, pitch, tolerance=base_tolerance);
    } else {
        // Male thread (external)
        // For die-ready: ADD material (larger diameter) so die can cut to final size
        tolerance = base_tolerance - oversize;
        ScrewThread(od + oversize, len, pitch, tolerance=tolerance);
    }
}

module iso_thread_12(len, internal=false, oversize=0) {
    // M12x1.75 Optimized
    // oversize: 0 for print-ready, 0.3 for die/tap chasing
    simple_thread(12.0, len, 1.75, internal, oversize);
}


// =============================================================================
// UTILITY MODULES
// =============================================================================

module chamfer_cylinder(d, h, chamfer=1.0) {
    union() {
        translate([0,0,chamfer]) cylinder(d=d, h=h-2*chamfer);
        cylinder(d1=d-2*chamfer, d2=d, h=chamfer);
        translate([0,0,h-chamfer]) cylinder(d1=d, d2=d-2*chamfer, h=chamfer);
    }
}

module o_ring_groove(inner_dia, groove_width=2.0, groove_depth=1.5) {
    // O-ring groove for sealing
    difference() {
        cylinder(d=inner_dia + groove_depth*2, h=groove_width);
        translate([0,0,-0.1])
            cylinder(d=inner_dia, h=groove_width + 0.2);
    }
}


// =============================================================================
// MAIN COMPONENTS
// =============================================================================

// -----------------------------------------------------------------------------
// MALE ROD CAP (formerly male_insert)
// Epoxies into fiberglass tube end, external M12 male thread
// -----------------------------------------------------------------------------
module male_rod_cap() {
    difference() {
        union() {
            // 1. Insert Section (Goes inside fiberglass tube)
            cylinder(d=ROD_ID - 0.2, h=ROD_INSERT_DEPTH - 1);
            translate([0,0,ROD_INSERT_DEPTH-1])
                cylinder(d1=ROD_ID-0.2, d2=ROD_OD, h=1);

            // 2. Stop Flange (butts against tube end)
            translate([0,0,ROD_INSERT_DEPTH])
                cylinder(d=ROD_OD, h=FLANGE_THICKNESS);

            // 3. Threaded Section (External M12 male)
            translate([0,0,ROD_INSERT_DEPTH + FLANGE_THICKNESS])
                iso_thread_12(THREAD_LEN, internal=false, oversize=THREAD_OVERSIZE);
        }

        // Center Wire Hole (through entire part)
        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN + 2);

        // O-Ring Groove at thread base
        translate([0,0,ROD_INSERT_DEPTH + FLANGE_THICKNESS])
            difference() {
                cylinder(d=12.0 + 2.5, h=1.5);
                cylinder(d=12.0 + 0.5, h=1.5);
            }
    }
}

// Alias for backwards compatibility
module male_insert() {
    male_rod_cap();
}

// -----------------------------------------------------------------------------
// MALE ROD CAP BLANK (for tap & die)
// -----------------------------------------------------------------------------
module male_rod_cap_blank() {
    difference() {
        union() {
            // 1. Insert Section
            cylinder(d=ROD_ID - 0.2, h=ROD_INSERT_DEPTH - 1);
            translate([0,0,ROD_INSERT_DEPTH-1])
                cylinder(d1=ROD_ID-0.2, d2=ROD_OD, h=1);

            // 2. Stop Flange
            translate([0,0,ROD_INSERT_DEPTH])
                cylinder(d=ROD_OD, h=FLANGE_THICKNESS);

            // 3. Blank Stud (11.9mm for easy die start)
            translate([0,0,ROD_INSERT_DEPTH + FLANGE_THICKNESS])
                cylinder(d=11.9, h=THREAD_LEN);
        }

        // Center Wire Hole
        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN + 2);

        // O-Ring Groove
        translate([0,0,ROD_INSERT_DEPTH + FLANGE_THICKNESS])
            difference() {
                cylinder(d=12.0 + 2.5, h=1.5);
                cylinder(d=12.0 + 0.5, h=1.5);
            }
    }
}

// Alias for backwards compatibility
module male_insert_blank() {
    male_rod_cap_blank();
}


// -----------------------------------------------------------------------------
// SENSOR BODY DUAL (NEW - Independent sensor with female threads on BOTH ends)
// 16mm OD, 70mm length, female M12 sockets on both ends
// Includes ERT ring groove, MIT coil zone, and radial wire entry holes
// -----------------------------------------------------------------------------
module sensor_body_dual() {
    // Calculate positions
    // Total: 70mm = 25mm thread + 20mm sensor zone + 25mm thread
    sensor_zone_len = SENSOR_BODY_LEN - 2*THREAD_HOLE_DEPTH; // 20mm
    sensor_zone_start = THREAD_HOLE_DEPTH; // 25mm from bottom

    // Sensor feature positions (within sensor zone)
    ert_center = sensor_zone_start + sensor_zone_len/4;  // ERT ring at 1/4 of sensor zone
    mit_center = sensor_zone_start + sensor_zone_len*3/4; // MIT coil at 3/4 of sensor zone

    difference() {
        union() {
            // Main cylindrical body (flush 16mm OD)
            cylinder(d=ROD_OD, h=SENSOR_BODY_LEN);

            // Orientation flat indicator - embedded into cylinder for solid union
            // Extends from inside the cylinder to create a bump that won't float
            translate([ROD_OD/2 - 1.5, 0, SENSOR_BODY_LEN/2])
                cube([2, 3, 5], center=true);
        }

        // --- BOTTOM Female Thread Socket ---
        translate([0,0,-0.1])
            iso_thread_12(THREAD_HOLE_DEPTH + 0.1, internal=true, oversize=THREAD_OVERSIZE);

        // --- TOP Female Thread Socket ---
        translate([0,0,SENSOR_BODY_LEN - THREAD_HOLE_DEPTH])
            iso_thread_12(THREAD_HOLE_DEPTH + 0.1, internal=true, oversize=THREAD_OVERSIZE);

        // --- Center Wire Channel (6mm through entire length) ---
        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=SENSOR_BODY_LEN + 2);

        // --- ERT Ring Groove ---
        // Circumferential groove for stainless steel ring
        translate([0,0,ert_center - ERT_GROOVE_WIDTH/2])
            difference() {
                cylinder(d=ROD_OD + 0.1, h=ERT_GROOVE_WIDTH);
                cylinder(d=ROD_OD - 2*ERT_GROOVE_DEPTH, h=ERT_GROOVE_WIDTH);
            }

        // --- MIT Coil Zone ---
        // Shallow recess for winding coil (prevents slipping)
        translate([0,0,mit_center - MIT_COIL_ZONE/2])
            difference() {
                cylinder(d=ROD_OD + 0.1, h=MIT_COIL_ZONE);
                cylinder(d=ROD_OD - 0.6, h=MIT_COIL_ZONE); // 0.3mm deep recess
            }

        // --- Radial Wire Entry Holes ---
        // These connect outer surface to center 6mm channel

        // Hole 1: ERT ring wire (at ERT groove center)
        translate([0,0,ert_center])
            rotate([90,0,0])
                cylinder(d=WIRE_ENTRY_DIA, h=ROD_OD, center=true);

        // Hole 2: MIT coil start wire (at coil zone edge)
        translate([0,0,mit_center - MIT_COIL_ZONE/2 + 2])
            rotate([90,0,60])  // Offset 60 degrees from ERT hole
                cylinder(d=WIRE_ENTRY_DIA, h=ROD_OD, center=true);

        // Hole 3: MIT coil end wire (at opposite coil zone edge)
        translate([0,0,mit_center + MIT_COIL_ZONE/2 - 2])
            rotate([90,0,120]) // Offset 120 degrees from ERT hole
                cylinder(d=WIRE_ENTRY_DIA, h=ROD_OD, center=true);

        // --- O-Ring Grooves at thread interfaces ---
        // Bottom
        translate([0,0,THREAD_HOLE_DEPTH - 2])
            difference() {
                cylinder(d=THREAD_DIA_MINOR + 3, h=1.5);
                cylinder(d=THREAD_DIA_MINOR + 0.5, h=1.5);
            }
        // Top
        translate([0,0,SENSOR_BODY_LEN - THREAD_HOLE_DEPTH + 0.5])
            difference() {
                cylinder(d=THREAD_DIA_MINOR + 3, h=1.5);
                cylinder(d=THREAD_DIA_MINOR + 0.5, h=1.5);
            }
    }
}

// -----------------------------------------------------------------------------
// SENSOR BODY DUAL BLANK (for tap & die)
// -----------------------------------------------------------------------------
module sensor_body_dual_blank() {
    sensor_zone_len = SENSOR_BODY_LEN - 2*THREAD_HOLE_DEPTH;
    sensor_zone_start = THREAD_HOLE_DEPTH;
    ert_center = sensor_zone_start + sensor_zone_len/4;
    mit_center = sensor_zone_start + sensor_zone_len*3/4;

    difference() {
        union() {
            cylinder(d=ROD_OD, h=SENSOR_BODY_LEN);

            // Orientation flat indicator - embedded into cylinder for solid union
            translate([ROD_OD/2 - 1.5, 0, SENSOR_BODY_LEN/2])
                cube([2, 3, 5], center=true);
        }

        // Bottom blank hole (10.4mm for tapping)
        translate([0,0,-0.1])
            cylinder(d=THREAD_DIA_MINOR, h=THREAD_HOLE_DEPTH + 0.1);

        // Top blank hole (10.4mm for tapping)
        translate([0,0,SENSOR_BODY_LEN - THREAD_HOLE_DEPTH])
            cylinder(d=THREAD_DIA_MINOR, h=THREAD_HOLE_DEPTH + 0.1);

        // Center Wire Channel
        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=SENSOR_BODY_LEN + 2);

        // ERT Ring Groove
        translate([0,0,ert_center - ERT_GROOVE_WIDTH/2])
            difference() {
                cylinder(d=ROD_OD + 0.1, h=ERT_GROOVE_WIDTH);
                cylinder(d=ROD_OD - 2*ERT_GROOVE_DEPTH, h=ERT_GROOVE_WIDTH);
            }

        // MIT Coil Zone
        translate([0,0,mit_center - MIT_COIL_ZONE/2])
            difference() {
                cylinder(d=ROD_OD + 0.1, h=MIT_COIL_ZONE);
                cylinder(d=ROD_OD - 0.6, h=MIT_COIL_ZONE);
            }

        // Radial Wire Entry Holes
        translate([0,0,ert_center])
            rotate([90,0,0])
                cylinder(d=WIRE_ENTRY_DIA, h=ROD_OD, center=true);

        translate([0,0,mit_center - MIT_COIL_ZONE/2 + 2])
            rotate([90,0,60])
                cylinder(d=WIRE_ENTRY_DIA, h=ROD_OD, center=true);

        translate([0,0,mit_center + MIT_COIL_ZONE/2 - 2])
            rotate([90,0,120])
                cylinder(d=WIRE_ENTRY_DIA, h=ROD_OD, center=true);
    }
}


// -----------------------------------------------------------------------------
// PROBE TIP (NEW - Bottom terminator with tapered point)
// Single female thread socket at top, tapered point at bottom
// -----------------------------------------------------------------------------
module probe_tip() {
    // Thread socket must fit within the cylindrical body section only
    // Body starts at TIP_TAPER_LEN, so max socket depth = TIP_LENGTH - TIP_TAPER_LEN
    body_len = TIP_LENGTH - TIP_TAPER_LEN;  // 15mm
    socket_depth = min(THREAD_HOLE_DEPTH, body_len - 2);  // Max 13mm, leave 2mm solid

    difference() {
        union() {
            // Tapered tip section (conical)
            cylinder(d1=TIP_POINT_DIA, d2=ROD_OD, h=TIP_TAPER_LEN);

            // Cylindrical body section
            translate([0,0,TIP_TAPER_LEN])
                cylinder(d=ROD_OD, h=body_len);
        }

        // Female thread socket at top (contained within body section)
        translate([0,0,TIP_LENGTH - socket_depth])
            iso_thread_12(socket_depth + 0.1, internal=true, oversize=THREAD_OVERSIZE);

        // Center wire channel (BLIND - stops before tip)
        // Extends from thread bottom down to just above taper
        translate([0,0,TIP_TAPER_LEN + 2])
            cylinder(d=WIRE_HOLE_DIA, h=body_len - 2 + 1);

        // Wire termination cavity (larger space at blind end for wire ends)
        translate([0,0,TIP_TAPER_LEN + 0.5])
            cylinder(d=WIRE_HOLE_DIA + 2, h=3);

        // O-Ring groove at thread interface (inside the socket, near top)
        translate([0,0,TIP_LENGTH - 3])
            difference() {
                cylinder(d=THREAD_DIA_MINOR + 3, h=1.5);
                cylinder(d=THREAD_DIA_MINOR + 0.5, h=1.5);
            }
    }
}

// -----------------------------------------------------------------------------
// PROBE TIP BLANK (for tap & die)
// -----------------------------------------------------------------------------
module probe_tip_blank() {
    body_len = TIP_LENGTH - TIP_TAPER_LEN;
    socket_depth = min(THREAD_HOLE_DEPTH, body_len - 2);

    difference() {
        union() {
            cylinder(d1=TIP_POINT_DIA, d2=ROD_OD, h=TIP_TAPER_LEN);
            translate([0,0,TIP_TAPER_LEN])
                cylinder(d=ROD_OD, h=body_len);
        }

        // Blank hole for tapping (contained within body section)
        translate([0,0,TIP_LENGTH - socket_depth])
            cylinder(d=THREAD_DIA_MINOR, h=socket_depth + 0.1);

        // Center wire channel (blind)
        translate([0,0,TIP_TAPER_LEN + 2])
            cylinder(d=WIRE_HOLE_DIA, h=body_len - 2 + 1);

        // Wire termination cavity
        translate([0,0,TIP_TAPER_LEN + 0.5])
            cylinder(d=WIRE_HOLE_DIA + 2, h=3);
    }
}


// -----------------------------------------------------------------------------
// JUNCTION CAP (NEW - Top terminator with cable exit)
// Single female thread socket at bottom, cable gland boss at top
// -----------------------------------------------------------------------------
module junction_cap() {
    socket_depth = THREAD_HOLE_DEPTH;
    body_len = JUNCTION_CAP_LEN - CABLE_GLAND_BOSS_LEN;

    difference() {
        union() {
            // Main body (16mm OD, matches rod)
            cylinder(d=ROD_OD, h=body_len);

            // Cable gland boss (larger diameter)
            translate([0,0,body_len])
                cylinder(d=CABLE_GLAND_BOSS_DIA, h=CABLE_GLAND_BOSS_LEN);

            // Transition cone
            translate([0,0,body_len - 3])
                cylinder(d1=ROD_OD, d2=CABLE_GLAND_BOSS_DIA, h=3);
        }

        // Female thread socket at bottom
        translate([0,0,-0.1])
            iso_thread_12(socket_depth + 0.1, internal=true, oversize=THREAD_OVERSIZE);

        // Center wire channel (transitions to cable exit)
        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=body_len + 2);

        // Wire routing chamber (larger space for splices)
        translate([0,0,socket_depth])
            cylinder(d=WIRE_HOLE_DIA + 4, h=body_len - socket_depth - 2);

        // Cable gland thread hole (M12 or PG7)
        translate([0,0,body_len - 1])
            cylinder(d=CABLE_GLAND_THREAD_DIA, h=CABLE_GLAND_BOSS_LEN + 2);

        // O-Ring groove at thread interface
        translate([0,0,socket_depth - 2])
            difference() {
                cylinder(d=THREAD_DIA_MINOR + 3, h=1.5);
                cylinder(d=THREAD_DIA_MINOR + 0.5, h=1.5);
            }

        // Weather seal groove at cable gland
        translate([0,0,JUNCTION_CAP_LEN - 3])
            difference() {
                cylinder(d=CABLE_GLAND_THREAD_DIA + 3, h=2);
                cylinder(d=CABLE_GLAND_THREAD_DIA + 0.5, h=2);
            }
    }
}

// -----------------------------------------------------------------------------
// JUNCTION CAP BLANK (for tap & die)
// -----------------------------------------------------------------------------
module junction_cap_blank() {
    socket_depth = THREAD_HOLE_DEPTH;
    body_len = JUNCTION_CAP_LEN - CABLE_GLAND_BOSS_LEN;

    difference() {
        union() {
            cylinder(d=ROD_OD, h=body_len);
            translate([0,0,body_len])
                cylinder(d=CABLE_GLAND_BOSS_DIA, h=CABLE_GLAND_BOSS_LEN);
            translate([0,0,body_len - 3])
                cylinder(d1=ROD_OD, d2=CABLE_GLAND_BOSS_DIA, h=3);
        }

        // Blank hole for tapping
        translate([0,0,-0.1])
            cylinder(d=THREAD_DIA_MINOR, h=socket_depth + 0.1);

        // Center wire channel
        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=body_len + 2);

        // Wire routing chamber
        translate([0,0,socket_depth])
            cylinder(d=WIRE_HOLE_DIA + 4, h=body_len - socket_depth - 2);

        // Cable gland thread hole
        translate([0,0,body_len - 1])
            cylinder(d=CABLE_GLAND_THREAD_DIA, h=CABLE_GLAND_BOSS_LEN + 2);
    }
}


// =============================================================================
// LEGACY MODULES (Deprecated - kept for reference)
// =============================================================================

// -----------------------------------------------------------------------------
// FEMALE SENSOR MODULE (DEPRECATED)
// Old design with rod insert + single female socket
// Replaced by sensor_body_dual() for true modularity
// -----------------------------------------------------------------------------
module female_sensor_module() {
    echo("WARNING: female_sensor_module() is deprecated. Use sensor_body_dual() instead.");
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
        translate([0,0,ROD_INSERT_DEPTH + SENSOR_BODY_LEN - THREAD_HOLE_DEPTH])
            iso_thread_12(THREAD_HOLE_DEPTH + 1, internal=true);

        // Sensor Cutouts (legacy)
        translate([0,0,ROD_INSERT_DEPTH + 10])
            difference() { cylinder(d=ROD_OD+0.1, h=4); cylinder(d=ROD_OD-1, h=4); }

        translate([0,0,ROD_INSERT_DEPTH + SENSOR_BODY_LEN - 14])
             difference() { cylinder(d=ROD_OD+0.1, h=4); cylinder(d=ROD_OD-1, h=4); }

        translate([0,0,ROD_INSERT_DEPTH + 20])
             difference() { cylinder(d=ROD_OD+0.1, h=20); cylinder(d=ROD_OD-2, h=20); }
    }
}

module female_sensor_module_blank() {
    echo("WARNING: female_sensor_module_blank() is deprecated. Use sensor_body_dual_blank() instead.");
    difference() {
        union() {
            cylinder(d=ROD_ID - 0.2, h=ROD_INSERT_DEPTH);
            translate([0,0,ROD_INSERT_DEPTH])
                cylinder(d=ROD_OD, h=SENSOR_BODY_LEN);
        }

        translate([0,0,-1])
            cylinder(d=WIRE_HOLE_DIA, h=ROD_INSERT_DEPTH + SENSOR_BODY_LEN + 2);

        translate([0,0,ROD_INSERT_DEPTH + SENSOR_BODY_LEN - THREAD_HOLE_DEPTH])
            cylinder(d=10.4, h=THREAD_HOLE_DEPTH + 1);

        translate([0,0,ROD_INSERT_DEPTH + 10])
            difference() { cylinder(d=ROD_OD+0.1, h=4); cylinder(d=ROD_OD-1, h=4); }

        translate([0,0,ROD_INSERT_DEPTH + SENSOR_BODY_LEN - 14])
             difference() { cylinder(d=ROD_OD+0.1, h=4); cylinder(d=ROD_OD-1, h=4); }

        translate([0,0,ROD_INSERT_DEPTH + 20])
             difference() { cylinder(d=ROD_OD+0.1, h=20); cylinder(d=ROD_OD-2, h=20); }
    }
}


// =============================================================================
// PRINT ARRAY MODULES
// =============================================================================

// --- Stability Enhancements ---

module super_brim(positions, radius) {
    // A solid 0.28mm sheet connecting the bases at given positions
    difference() {
        hull() {
            for (pos = positions) {
                translate([pos[0], pos[1], 0]) cylinder(r=radius, h=0.28);
            }
        }
        // Wire hole cleanout
        for (pos = positions) {
            translate([pos[0], pos[1], -1]) cylinder(d=WIRE_HOLE_DIA, h=2);
        }
    }
}

module rigid_scaffolding(positions, z_height) {
    beam_width = 2.5;
    beam_height = 1.2;

    // Calculate bounding box
    min_x = min([for (p = positions) p[0]]);
    max_x = max([for (p = positions) p[0]]);
    min_y = min([for (p = positions) p[1]]);
    max_y = max([for (p = positions) p[1]]);

    translate([0, 0, z_height]) {
        // Perimeter frame
        if (max_x > min_x) {
            translate([(min_x + max_x)/2, min_y, 0])
                cube([max_x - min_x + beam_width, beam_width, beam_height], center=true);
            translate([(min_x + max_x)/2, max_y, 0])
                cube([max_x - min_x + beam_width, beam_width, beam_height], center=true);
        }
        if (max_y > min_y) {
            translate([min_x, (min_y + max_y)/2, 0])
                cube([beam_width, max_y - min_y + beam_width, beam_height], center=true);
            translate([max_x, (min_y + max_y)/2, 0])
                cube([beam_width, max_y - min_y + beam_width, beam_height], center=true);
        }
    }
}

// -----------------------------------------------------------------------------
// MALE CAP ARRAY (4x male rod caps)
// -----------------------------------------------------------------------------
module male_cap_array(use_blanks=false) {
    spacing = ROD_OD + 8; // 24mm
    brim_rad = 12.0;

    positions = [
        [0, 0],
        [spacing, 0],
        [0, spacing],
        [spacing, spacing]
    ];

    union() {
        for (pos = positions) {
            translate([pos[0], pos[1], 0])
                if (use_blanks) male_rod_cap_blank();
                else male_rod_cap();
        }
    }

    super_brim(positions, brim_rad);

    difference() {
        rigid_scaffolding(positions, 21.0);
        for (pos = positions) {
            translate([pos[0], pos[1], -10]) cylinder(d=ROD_OD-1, h=200);
        }
    }
}

// -----------------------------------------------------------------------------
// MALE CAP ARRAY 16x (4x4 grid for batch production)
// -----------------------------------------------------------------------------
module male_cap_array_16x(use_blanks=false) {
    spacing = ROD_OD + 8; // 24mm
    brim_rad = 12.0;

    positions = [
        for (y = [0:3])
            for (x = [0:3])
                [x * spacing, y * spacing]
    ];

    union() {
        for (pos = positions) {
            translate([pos[0], pos[1], 0])
                if (use_blanks) male_rod_cap_blank();
                else male_rod_cap();
        }
    }

    super_brim(positions, brim_rad);

    difference() {
        rigid_scaffolding(positions, 21.0);
        for (pos = positions) {
            translate([pos[0], pos[1], -10]) cylinder(d=ROD_OD-1, h=200);
        }
    }
}

// -----------------------------------------------------------------------------
// SENSOR BODY ARRAY (4x dual-female sensor bodies)
// -----------------------------------------------------------------------------
module sensor_body_array(use_blanks=false) {
    spacing = ROD_OD + 8; // 24mm
    brim_rad = 12.0;

    positions = [
        [0, 0],
        [spacing, 0],
        [0, spacing],
        [spacing, spacing]
    ];

    union() {
        for (pos = positions) {
            translate([pos[0], pos[1], 0])
                if (use_blanks) sensor_body_dual_blank();
                else sensor_body_dual();
        }
    }

    super_brim(positions, brim_rad);

    difference() {
        union() {
            rigid_scaffolding(positions, 21.0);
            rigid_scaffolding(positions, SENSOR_BODY_LEN - 10);
        }
        for (pos = positions) {
            translate([pos[0], pos[1], -10]) cylinder(d=ROD_OD-1, h=200);
        }
    }
}

// -----------------------------------------------------------------------------
// PROBE TIP ARRAY (4x probe tips)
// -----------------------------------------------------------------------------
module probe_tip_array(use_blanks=false) {
    spacing = ROD_OD + 8; // 24mm
    brim_rad = 12.0;

    positions = [
        [0, 0],
        [spacing, 0],
        [0, spacing],
        [spacing, spacing]
    ];

    union() {
        for (pos = positions) {
            translate([pos[0], pos[1], 0])
                if (use_blanks) probe_tip_blank();
                else probe_tip();
        }
    }

    super_brim(positions, brim_rad);

    difference() {
        rigid_scaffolding(positions, TIP_LENGTH - 5);
        for (pos = positions) {
            translate([pos[0], pos[1], -10]) cylinder(d=ROD_OD-1, h=200);
        }
    }
}

// -----------------------------------------------------------------------------
// JUNCTION CAP ARRAY (4x junction caps)
// -----------------------------------------------------------------------------
module junction_cap_array(use_blanks=false) {
    spacing = CABLE_GLAND_BOSS_DIA + 6; // 26mm (larger due to boss)
    brim_rad = 14.0;

    positions = [
        [0, 0],
        [spacing, 0],
        [0, spacing],
        [spacing, spacing]
    ];

    union() {
        for (pos = positions) {
            translate([pos[0], pos[1], 0])
                if (use_blanks) junction_cap_blank();
                else junction_cap();
        }
    }

    super_brim(positions, brim_rad);

    difference() {
        rigid_scaffolding(positions, JUNCTION_CAP_LEN - 10);
        for (pos = positions) {
            translate([pos[0], pos[1], -10]) cylinder(d=ROD_OD-1, h=200);
        }
    }
}

// -----------------------------------------------------------------------------
// MIXED SYSTEM ARRAY (2 caps + 2 sensors + 1 tip + 1 junction)
// Complete set for a basic probe
// -----------------------------------------------------------------------------
module mixed_system_array(use_blanks=false) {
    spacing = ROD_OD + 8; // 24mm
    brim_rad = 12.0;

    // Layout: 3x2 grid
    // Row 1: Male Cap, Male Cap, Probe Tip
    // Row 2: Sensor, Sensor, Junction Cap

    union() {
        // Male caps (position 0,0 and 1,0)
        translate([0, 0, 0])
            if (use_blanks) male_rod_cap_blank(); else male_rod_cap();
        translate([spacing, 0, 0])
            if (use_blanks) male_rod_cap_blank(); else male_rod_cap();

        // Probe tip (position 2,0)
        translate([spacing*2, 0, 0])
            if (use_blanks) probe_tip_blank(); else probe_tip();

        // Sensor bodies (position 0,1 and 1,1)
        translate([0, spacing, 0])
            if (use_blanks) sensor_body_dual_blank(); else sensor_body_dual();
        translate([spacing, spacing, 0])
            if (use_blanks) sensor_body_dual_blank(); else sensor_body_dual();

        // Junction cap (position 2,1)
        translate([spacing*2, spacing, 0])
            if (use_blanks) junction_cap_blank(); else junction_cap();
    }

    // Super brim for all positions
    positions = [
        [0, 0], [spacing, 0], [spacing*2, 0],
        [0, spacing], [spacing, spacing], [spacing*2, spacing]
    ];
    super_brim(positions, brim_rad);

    // Scaffolding
    difference() {
        union() {
            rigid_scaffolding(positions, 21.0);
            // Higher scaffolding for taller parts
            rigid_scaffolding([[0, spacing], [spacing, spacing]], SENSOR_BODY_LEN - 10);
        }
        for (pos = positions) {
            translate([pos[0], pos[1], -10]) cylinder(d=ROD_OD-1, h=200);
        }
    }
}

// -----------------------------------------------------------------------------
// LEGACY: Mixed array (2 male + 2 female) for backwards compatibility
// -----------------------------------------------------------------------------
module print_array_mixed(use_blanks=false) {
    spacing = ROD_OD + 8;
    brim_rad = 12.0;

    positions = [
        [0, 0], [spacing, 0],
        [0, spacing], [spacing, spacing]
    ];

    union() {
        if (use_blanks) {
            translate([0, 0, 0]) male_insert_blank();
            translate([spacing, 0, 0]) male_insert_blank();
            translate([0, spacing, 0]) female_sensor_module_blank();
            translate([spacing, spacing, 0]) female_sensor_module_blank();
        } else {
            translate([0, 0, 0]) male_insert();
            translate([spacing, 0, 0]) male_insert();
            translate([0, spacing, 0]) female_sensor_module();
            translate([spacing, spacing, 0]) female_sensor_module();
        }
    }

    super_brim(positions, brim_rad);

    difference() {
        union() {
            rigid_scaffolding(positions, 21.0);
            translate([spacing/2, spacing, 85])
                cube([spacing, 3.0, 1.5], center=true);
        }

        cutout_d = ROD_OD - 1.0;
        for (pos = positions) {
            translate([pos[0], pos[1], -10]) cylinder(d=cutout_d, h=200);
        }
    }
}


// =============================================================================
// VISUALIZATION MODULES (for documentation renders)
// =============================================================================

// -----------------------------------------------------------------------------
// EXPLODED VIEW (20mm gaps for clear part separation)
// Shows all components with vertical alignment and larger gaps
// -----------------------------------------------------------------------------
module exploded_view() {
    gap = 20;  // Larger gap for clear visualization

    // Calculate component heights
    cap_height = ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN;

    // Bottom to top stack (probe tip at bottom, junction cap at top)
    // Probe tip (at origin)
    color("DarkSlateGray")
    translate([0, 0, 0]) probe_tip();

    // Male cap (inverted, above tip)
    color("SteelBlue")
    translate([0, 0, TIP_LENGTH + gap])
        rotate([180, 0, 0])
            translate([0, 0, -cap_height])
                male_rod_cap();

    // Sensor body 1
    color("ForestGreen")
    translate([0, 0, TIP_LENGTH + gap + cap_height + gap])
        sensor_body_dual();

    // Male cap (above sensor 1)
    color("SteelBlue")
    translate([0, 0, TIP_LENGTH + gap + cap_height + gap + SENSOR_BODY_LEN + gap])
        male_rod_cap();

    // Male cap (inverted, above previous)
    color("SteelBlue")
    translate([0, 0, TIP_LENGTH + gap + cap_height + gap + SENSOR_BODY_LEN + gap + cap_height + gap])
        rotate([180, 0, 0])
            translate([0, 0, -cap_height])
                male_rod_cap();

    // Sensor body 2
    color("ForestGreen")
    translate([0, 0, TIP_LENGTH + gap + cap_height + gap + SENSOR_BODY_LEN + gap + cap_height + gap + cap_height + gap])
        sensor_body_dual();

    // Male cap (above sensor 2)
    color("SteelBlue")
    translate([0, 0, TIP_LENGTH + gap + cap_height + gap + SENSOR_BODY_LEN + gap + cap_height + gap + cap_height + gap + SENSOR_BODY_LEN + gap])
        male_rod_cap();

    // Junction cap (at top)
    color("DimGray")
    translate([0, 0, TIP_LENGTH + gap + cap_height + gap + SENSOR_BODY_LEN + gap + cap_height + gap + cap_height + gap + SENSOR_BODY_LEN + gap + cap_height + gap])
        junction_cap();
}

// -----------------------------------------------------------------------------
// CROSS SECTION VIEW (half-section showing internal details)
// Cuts through center to show wire channel, threads, and internal features
// -----------------------------------------------------------------------------
module cross_section_view() {
    // Assembled probe (same as complete_system but with smaller gaps)
    gap = 2;  // Small gap to show thread engagement area
    cap_height = ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN;

    difference() {
        union() {
            // Probe tip (at origin)
            color("DarkSlateGray")
            translate([0, 0, 0]) probe_tip();

            // Male cap (inverted)
            color("SteelBlue")
            translate([0, 0, TIP_LENGTH + gap])
                rotate([180, 0, 0])
                    translate([0, 0, -cap_height])
                        male_rod_cap();

            // Sensor body
            color("ForestGreen")
            translate([0, 0, TIP_LENGTH + gap + cap_height + gap])
                sensor_body_dual();

            // Male cap
            color("SteelBlue")
            translate([0, 0, TIP_LENGTH + gap + cap_height + gap + SENSOR_BODY_LEN + gap])
                male_rod_cap();

            // Junction cap
            color("DimGray")
            translate([0, 0, TIP_LENGTH + gap + cap_height + gap + SENSOR_BODY_LEN + gap + cap_height + gap])
                junction_cap();
        }

        // Cutting plane - remove half
        translate([0, -50, -10])
            cube([100, 50, 300]);
    }
}

// -----------------------------------------------------------------------------
// ISOMETRIC ASSEMBLED VIEW (tight assembly for "what you're building")
// Camera-ready view showing complete assembled probe
// -----------------------------------------------------------------------------
module isometric_assembled() {
    // Tight assembly with realistic thread engagement
    engagement = 10;  // Thread engagement depth
    cap_height = ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN;

    // Calculate Z positions accounting for thread engagement
    z_tip = 0;
    z_cap1 = TIP_LENGTH - engagement;
    z_sensor1 = z_cap1 + cap_height - engagement;
    z_cap2 = z_sensor1 + SENSOR_BODY_LEN - engagement;
    z_cap3 = z_cap2 + cap_height - engagement;
    z_sensor2 = z_cap3 + cap_height - engagement;
    z_cap4 = z_sensor2 + SENSOR_BODY_LEN - engagement;
    z_junction = z_cap4 + cap_height - engagement;

    // Bottom to top stack
    color("DarkSlateGray") translate([0, 0, z_tip]) probe_tip();
    color("SteelBlue") translate([0, 0, z_cap1]) rotate([180, 0, 0]) translate([0, 0, -cap_height]) male_rod_cap();
    color("ForestGreen") translate([0, 0, z_sensor1]) sensor_body_dual();
    color("SteelBlue") translate([0, 0, z_cap2]) male_rod_cap();
    color("SteelBlue") translate([0, 0, z_cap3]) rotate([180, 0, 0]) translate([0, 0, -cap_height]) male_rod_cap();
    color("ForestGreen") translate([0, 0, z_sensor2]) sensor_body_dual();
    color("SteelBlue") translate([0, 0, z_cap4]) male_rod_cap();
    color("DimGray") translate([0, 0, z_junction]) junction_cap();
}

// -----------------------------------------------------------------------------
// DEPLOYMENT ARRAY (4 probes with ground plane)
// Shows typical field deployment configuration
// -----------------------------------------------------------------------------
module deployment_array() {
    probe_spacing = 100;  // 100mm = 10cm grid spacing (scaled)
    ground_size = 300;
    probe_depth = 150;  // Visual probe depth below ground

    // Ground plane
    color("SaddleBrown", 0.7)
    translate([-ground_size/2, -ground_size/2, 0])
        cube([ground_size, ground_size, 5]);

    // Surface texture (grass suggestion)
    color("DarkOliveGreen", 0.5)
    translate([-ground_size/2, -ground_size/2, 5])
        cube([ground_size, ground_size, 1]);

    // 2x2 grid of probes
    for (x = [-probe_spacing/2, probe_spacing/2]) {
        for (y = [-probe_spacing/2, probe_spacing/2]) {
            translate([x, y, -probe_depth]) {
                // Simplified probe representation for array view
                color("Navy")
                cylinder(d=ROD_OD, h=probe_depth + 30, $fn=32);

                // Junction cap at surface
                color("DimGray")
                translate([0, 0, probe_depth])
                    junction_cap();
            }
        }
    }

    // Cable suggestion (from probes to hub)
    color("Black", 0.6)
    for (x = [-probe_spacing/2, probe_spacing/2]) {
        for (y = [-probe_spacing/2, probe_spacing/2]) {
            translate([x, y, 6]) {
                // Cable running to center
                hull() {
                    cylinder(d=3, h=2, $fn=16);
                    translate([-x*0.8, -y*0.8, 0]) cylinder(d=3, h=2, $fn=16);
                }
            }
        }
    }

    // Central hub suggestion
    color("Gray")
    translate([0, 0, 6])
        cube([40, 30, 15], center=true);

    color("DarkGray")
    translate([0, 0, 14])
        cube([35, 25, 5], center=true);
}

// -----------------------------------------------------------------------------
// COMPONENT CALLOUT VIEW (single sensor body with annotation points)
// For creating annotated diagrams in post-processing
// -----------------------------------------------------------------------------
module component_callout_view() {
    // Large sensor body for detail view
    scale([2, 2, 2]) {
        sensor_body_dual();

        // Callout marker points (small spheres at key features)
        // These can be used as anchor points for annotations

        // Top socket
        color("Red") translate([ROD_OD/2 + 2, 0, SENSOR_BODY_LEN - 5]) sphere(r=1, $fn=16);

        // Bottom socket
        color("Red") translate([ROD_OD/2 + 2, 0, 5]) sphere(r=1, $fn=16);

        // ERT groove (at 1/4 position in sensor zone)
        sensor_zone_start = THREAD_HOLE_DEPTH;
        sensor_zone_len = SENSOR_BODY_LEN - 2*THREAD_HOLE_DEPTH;
        ert_center = sensor_zone_start + sensor_zone_len/4;
        color("Orange") translate([ROD_OD/2 + 2, 0, ert_center]) sphere(r=1, $fn=16);

        // MIT zone (at 3/4 position in sensor zone)
        mit_center = sensor_zone_start + sensor_zone_len*3/4;
        color("Green") translate([ROD_OD/2 + 2, 0, mit_center]) sphere(r=1, $fn=16);

        // Center channel
        color("Blue") translate([0, ROD_OD/2 + 2, SENSOR_BODY_LEN/2]) sphere(r=1, $fn=16);
    }
}


// =============================================================================
// RENDER LOGIC
// =============================================================================

if (part == "male_cap_array") {
    male_cap_array(false);
} else if (part == "male_cap_array_blanks") {
    male_cap_array(true);
} else if (part == "male_cap_array_16x") {
    male_cap_array_16x(false);
} else if (part == "male_cap_array_16x_blanks") {
    male_cap_array_16x(true);
} else if (part == "sensor_body_array") {
    sensor_body_array(false);
} else if (part == "sensor_body_array_no_scaffold") {
    // Test version without brim/scaffolding to isolate floating region issue
    spacing = ROD_OD + 8;
    for (pos = [[0,0], [spacing,0], [0,spacing], [spacing,spacing]]) {
        translate([pos[0], pos[1], 0]) sensor_body_dual();
    }
} else if (part == "sensor_body_array_blanks") {
    sensor_body_array(true);
} else if (part == "mixed_system_array") {
    mixed_system_array(false);
} else if (part == "mixed_system_array_blanks") {
    mixed_system_array(true);
} else if (part == "male_cap_single") {
    male_rod_cap();
} else if (part == "male_cap_single_blank") {
    male_rod_cap_blank();
} else if (part == "sensor_body_single") {
    sensor_body_dual();
} else if (part == "sensor_body_single_blank") {
    sensor_body_dual_blank();
} else if (part == "probe_tip_array") {
    probe_tip_array(false);
} else if (part == "probe_tip_array_blanks") {
    probe_tip_array(true);
} else if (part == "probe_tip_single") {
    probe_tip();
} else if (part == "probe_tip_single_blank") {
    probe_tip_blank();
} else if (part == "junction_cap_array") {
    junction_cap_array(false);
} else if (part == "junction_cap_array_blanks") {
    junction_cap_array(true);
} else if (part == "junction_cap_single") {
    junction_cap();
} else if (part == "junction_cap_single_blank") {
    junction_cap_blank();
} else if (part == "complete_system") {
    // Display assembled system (for visualization)
    translate([0, 0, 0]) junction_cap();
    translate([0, 0, -JUNCTION_CAP_LEN - 5]) male_rod_cap();
    translate([0, 0, -JUNCTION_CAP_LEN - 5 - (ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN) - 5])
        sensor_body_dual();
    translate([0, 0, -JUNCTION_CAP_LEN - 5 - (ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN) - 5 - SENSOR_BODY_LEN - 5])
        rotate([180,0,0]) male_rod_cap();
    translate([0, 0, -JUNCTION_CAP_LEN - 5 - (ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN) - 5 - SENSOR_BODY_LEN - 5 - (ROD_INSERT_DEPTH + FLANGE_THICKNESS + THREAD_LEN) - 5])
        rotate([180,0,0]) probe_tip();
// Visualization modules for documentation
} else if (part == "exploded_view") {
    exploded_view();
} else if (part == "cross_section") {
    cross_section_view();
} else if (part == "isometric_assembled") {
    isometric_assembled();
} else if (part == "deployment_array") {
    deployment_array();
} else if (part == "component_callout") {
    component_callout_view();
// Legacy render options
} else if (part == "male_array") {
    print_array_mixed(false);
} else if (part == "female_array") {
    print_array_mixed(false);
} else if (part == "mixed_array") {
    print_array_mixed(false);
} else if (part == "mixed_array_blanks") {
    print_array_mixed(true);
} else {
    // Default: show new mixed system array
    mixed_system_array(false);
}
