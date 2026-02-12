// Pathfinder Gradiometer Frame - Parametric Model
// Trapeze-style 4-channel fluxgate gradiometer frame
//
// Usage:
//   openscad -D 'part="assembled"' pathfinder_frame.scad -o pathfinder.stl
//
// Render modes (set 'part' variable):
//   "assembled"     - Complete trapeze system
//   "exploded"      - Components separated with gaps
//   "cross_section" - Half-section showing tube interiors
//   "deployment"    - With operator silhouette
//   "single_tube"   - Single sensor pair detail

/* [Render Mode] */
part = "assembled"; // ["assembled", "exploded", "cross_section", "deployment", "single_tube"]

/* [Crossbar Parameters] */
CROSSBAR_LENGTH = 2000;        // mm - Total length
CROSSBAR_OD = 25;              // mm - Outer diameter (carbon fiber tube)
CROSSBAR_WALL = 2;             // mm - Wall thickness
CROSSBAR_ID = CROSSBAR_OD - 2 * CROSSBAR_WALL;  // Inner diameter

/* [Sensor Positions] */
// Positions from left end (50cm spacing)
SENSOR_POSITIONS = [250, 750, 1250, 1750];  // mm
CENTER_MOUNT = 1000;           // mm - Center harness attachment point

/* [Drop Tube Parameters] */
DROP_TUBE_LENGTH = 500;        // mm - PVC tube length
DROP_TUBE_OD = 21.3;           // mm - 3/4" Schedule 40 PVC
DROP_TUBE_WALL = 2.7;          // mm - Schedule 40 wall
DROP_TUBE_ID = DROP_TUBE_OD - 2 * DROP_TUBE_WALL;

/* [Sensor Parameters] */
BOTTOM_SENSOR_HEIGHT = 175;    // mm from ground (15-20cm range)
SENSOR_DIAMETER = 16;          // mm - FG-3+ approximate diameter
SENSOR_LENGTH = 80;            // mm - FG-3+ approximate length

/* [Harness Parameters] */
D_RING_OD = 30;                // mm
D_RING_THICKNESS = 4;          // mm
BUNGEE_LENGTH = 400;           // mm - Approximate bungee length
SPREADER_BAR_LENGTH = 200;     // mm

/* [Visualization] */
EXPLODE_GAP = 100;             // mm - Gap between exploded parts
$fn = 32;                      // Circle resolution

// Colors
COLOR_CARBON = [0.15, 0.15, 0.15];   // Dark gray for carbon fiber
COLOR_PVC = [0.9, 0.9, 0.9];         // Light gray for PVC
COLOR_SENSOR = [0.2, 0.4, 0.6];      // Steel blue for sensors
COLOR_HARNESS = [0.6, 0.3, 0.1];     // Brown for straps
COLOR_METAL = [0.7, 0.7, 0.75];      // Silver for hardware

// ============================================================
// COMPONENT MODULES
// ============================================================

// Carbon fiber crossbar with mounting holes
module crossbar() {
    color(COLOR_CARBON)
    difference() {
        // Main tube
        rotate([0, 90, 0])
        cylinder(d=CROSSBAR_OD, h=CROSSBAR_LENGTH, center=true);

        // Hollow interior
        rotate([0, 90, 0])
        cylinder(d=CROSSBAR_ID, h=CROSSBAR_LENGTH + 2, center=true);

        // Mounting holes for drop tubes
        for (pos = SENSOR_POSITIONS) {
            translate([pos - CROSSBAR_LENGTH/2, 0, 0])
            rotate([0, 0, 0])
            cylinder(d=5, h=CROSSBAR_OD + 2, center=true);
        }

        // Center mount hole
        translate([CENTER_MOUNT - CROSSBAR_LENGTH/2, 0, CROSSBAR_OD/2])
        cylinder(d=8, h=10, center=true);
    }
}

// End cap for crossbar
module end_cap() {
    color(COLOR_CARBON)
    difference() {
        union() {
            // Outer cap
            cylinder(d=CROSSBAR_OD, h=10);
            // Insert portion
            translate([0, 0, -15])
            cylinder(d=CROSSBAR_ID - 0.5, h=15);
        }
        // Hollow for lighter weight
        translate([0, 0, 3])
        cylinder(d=CROSSBAR_ID - 4, h=10);
    }
}

// PVC drop tube with sensor mounts
module drop_tube() {
    color(COLOR_PVC)
    difference() {
        // Main tube
        cylinder(d=DROP_TUBE_OD, h=DROP_TUBE_LENGTH, center=true);

        // Hollow interior
        cylinder(d=DROP_TUBE_ID, h=DROP_TUBE_LENGTH + 2, center=true);

        // Top mounting hole
        translate([0, 0, DROP_TUBE_LENGTH/2 - 20])
        rotate([90, 0, 0])
        cylinder(d=5, h=DROP_TUBE_OD + 2, center=true);

        // Bottom sensor slot
        translate([0, 0, -DROP_TUBE_LENGTH/2 + 20])
        cube([DROP_TUBE_ID + 2, 5, 40], center=true);
    }
}

// Simplified FG-3+ fluxgate sensor representation
module fluxgate_sensor() {
    color(COLOR_SENSOR)
    union() {
        // Main sensor body
        cylinder(d=SENSOR_DIAMETER, h=SENSOR_LENGTH, center=true);

        // End caps
        translate([0, 0, SENSOR_LENGTH/2])
        cylinder(d=SENSOR_DIAMETER + 2, h=5, center=true);
        translate([0, 0, -SENSOR_LENGTH/2])
        cylinder(d=SENSOR_DIAMETER + 2, h=5, center=true);

        // Cable exit
        translate([0, 0, -SENSOR_LENGTH/2 - 10])
        cylinder(d=4, h=20, center=true);
    }
}

// Single sensor pair (top + bottom with drop tube)
module sensor_pair(exploded=false) {
    explode_offset = exploded ? EXPLODE_GAP : 0;

    // Drop tube
    translate([0, 0, -DROP_TUBE_LENGTH/2 - CROSSBAR_OD/2 - explode_offset])
    drop_tube();

    // Top sensor (on crossbar)
    translate([0, 0, CROSSBAR_OD/2 + SENSOR_LENGTH/2 + (exploded ? EXPLODE_GAP : 5)])
    rotate([0, 0, 0])
    fluxgate_sensor();

    // Bottom sensor (at tube end)
    translate([0, 0, -DROP_TUBE_LENGTH - CROSSBAR_OD/2 + SENSOR_LENGTH/2 - (exploded ? EXPLODE_GAP*2 : 0)])
    fluxgate_sensor();
}

// Center mount D-ring for harness attachment
module center_mount() {
    color(COLOR_METAL) {
        // Base plate
        translate([0, 0, CROSSBAR_OD/2])
        difference() {
            cylinder(d=40, h=3);
            translate([0, 0, -1])
            cylinder(d=8, h=5);
        }

        // D-ring
        translate([0, 0, CROSSBAR_OD/2 + 3]) {
            difference() {
                union() {
                    // Straight part
                    translate([0, -D_RING_OD/2 + D_RING_THICKNESS, D_RING_OD/2])
                    rotate([0, 90, 0])
                    cylinder(d=D_RING_THICKNESS, h=D_RING_OD - D_RING_THICKNESS, center=true);

                    // Curved part
                    translate([0, D_RING_THICKNESS/2, D_RING_OD/2])
                    rotate([0, 90, 0])
                    rotate_extrude(angle=180)
                    translate([D_RING_OD/2 - D_RING_THICKNESS/2, 0, 0])
                    circle(d=D_RING_THICKNESS);
                }
            }
        }
    }
}

// Spreader bar for harness
module spreader_bar() {
    color(COLOR_METAL)
    rotate([0, 90, 0])
    cylinder(d=10, h=SPREADER_BAR_LENGTH, center=true);
}

// Bungee cord segment
module bungee_cord(length) {
    color([0.2, 0.2, 0.2])
    cylinder(d=6, h=length, center=true);
}

// Simplified harness shoulder straps
module harness_straps() {
    color(COLOR_HARNESS) {
        // Left strap
        translate([-100, 0, 0])
        cube([50, 5, 300], center=true);

        // Right strap
        translate([100, 0, 0])
        cube([50, 5, 300], center=true);

        // Back panel
        translate([0, 0, 100])
        cube([250, 5, 150], center=true);
    }
}

// Human operator silhouette for scale
module operator_silhouette() {
    color([0.3, 0.3, 0.3, 0.5]) {
        // Torso
        translate([0, 0, 1200])
        cylinder(d1=400, d2=350, h=500);

        // Head
        translate([0, 0, 1750])
        sphere(d=200);

        // Legs
        for (x = [-100, 100]) {
            translate([x, 0, 0])
            cylinder(d=150, h=900);
        }
    }
}

// Mounting bolt
module mounting_bolt() {
    color(COLOR_METAL) {
        // Head
        cylinder(d=10, h=4);
        // Shaft
        translate([0, 0, -56])
        cylinder(d=5, h=60);
    }
}

// Sensor clip (3D printed part)
module sensor_clip() {
    color([0.2, 0.6, 0.2])
    difference() {
        cube([25, 20, 15], center=true);
        // Sensor hole
        rotate([90, 0, 0])
        cylinder(d=SENSOR_DIAMETER + 1, h=25, center=true);
        // Mounting hole
        cylinder(d=5, h=20, center=true);
    }
}

// ============================================================
// ASSEMBLY MODULES
// ============================================================

// Complete assembled frame
module assembled_frame() {
    // Crossbar
    crossbar();

    // End caps
    translate([-CROSSBAR_LENGTH/2 - 5, 0, 0])
    rotate([0, -90, 0])
    end_cap();

    translate([CROSSBAR_LENGTH/2 + 5, 0, 0])
    rotate([0, 90, 0])
    end_cap();

    // Sensor pairs at each position
    for (pos = SENSOR_POSITIONS) {
        translate([pos - CROSSBAR_LENGTH/2, 0, 0])
        sensor_pair(exploded=false);
    }

    // Center mount
    translate([CENTER_MOUNT - CROSSBAR_LENGTH/2, 0, 0])
    center_mount();
}

// Exploded view with gaps between components
module exploded_frame() {
    // Crossbar (raised)
    translate([0, 0, EXPLODE_GAP * 2])
    crossbar();

    // End caps (offset further)
    translate([-CROSSBAR_LENGTH/2 - EXPLODE_GAP - 5, 0, EXPLODE_GAP * 2])
    rotate([0, -90, 0])
    end_cap();

    translate([CROSSBAR_LENGTH/2 + EXPLODE_GAP + 5, 0, EXPLODE_GAP * 2])
    rotate([0, 90, 0])
    end_cap();

    // Sensor pairs exploded
    for (pos = SENSOR_POSITIONS) {
        translate([pos - CROSSBAR_LENGTH/2, 0, EXPLODE_GAP * 2])
        sensor_pair(exploded=true);
    }

    // Center mount (raised)
    translate([CENTER_MOUNT - CROSSBAR_LENGTH/2, 0, EXPLODE_GAP * 3])
    center_mount();

    // Spreader bar
    translate([0, EXPLODE_GAP, EXPLODE_GAP * 4 + 100])
    spreader_bar();
}

// Cross-section view (cut in half)
module cross_section_frame() {
    difference() {
        assembled_frame();
        // Cut plane
        translate([0, -500, 0])
        cube([CROSSBAR_LENGTH + 200, 1000, 2000], center=true);
    }
}

// Deployment view with operator
module deployment_view() {
    // Frame at operating height
    translate([0, 200, 1100])
    rotate([10, 0, 0])
    assembled_frame();

    // Operator silhouette
    operator_silhouette();

    // Harness
    translate([0, 50, 1400])
    harness_straps();

    // Bungee cords
    translate([0, 150, 1250])
    rotate([20, 0, 0])
    bungee_cord(200);

    // Ground plane
    color([0.4, 0.3, 0.2, 0.3])
    translate([0, 0, -5])
    cube([3000, 2000, 10], center=true);
}

// Single sensor pair detail view
module single_tube_view() {
    // Drop tube section
    drop_tube();

    // Top sensor
    translate([0, 0, DROP_TUBE_LENGTH/2 + CROSSBAR_OD/2 + SENSOR_LENGTH/2 + 5])
    fluxgate_sensor();

    // Bottom sensor
    translate([0, 0, -DROP_TUBE_LENGTH/2 + SENSOR_LENGTH/2 - 10])
    fluxgate_sensor();

    // Crossbar section
    translate([0, 0, DROP_TUBE_LENGTH/2 + CROSSBAR_OD/2])
    rotate([0, 90, 0])
    color(COLOR_CARBON)
    difference() {
        cylinder(d=CROSSBAR_OD, h=100, center=true);
        cylinder(d=CROSSBAR_ID, h=102, center=true);
    }

    // Sensor clips
    translate([0, 0, DROP_TUBE_LENGTH/2 + CROSSBAR_OD + SENSOR_LENGTH + 10])
    sensor_clip();

    // Mounting bolt
    translate([0, 0, DROP_TUBE_LENGTH/2 + CROSSBAR_OD/2 + 5])
    mounting_bolt();

    // Dimension annotation helper - vertical line
    color([1, 0, 0])
    translate([40, 0, 0])
    cylinder(d=1, h=DROP_TUBE_LENGTH, center=true);
}

// ============================================================
// MAIN RENDER SELECTION
// ============================================================

if (part == "assembled") {
    assembled_frame();
} else if (part == "exploded") {
    exploded_frame();
} else if (part == "cross_section") {
    cross_section_frame();
} else if (part == "deployment") {
    deployment_view();
} else if (part == "single_tube") {
    single_tube_view();
} else {
    // Default to assembled
    assembled_frame();
}
