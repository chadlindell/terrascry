# Contributing to Pathfinder

Pathfinder is an open-source handheld fluxgate gradiometer for rapid geophysical reconnaissance. We welcome contributions from researchers, engineers, makers, and field practitioners. This guide covers everything you need to get started.

## 1. Getting Started

### Prerequisites

- **PlatformIO CLI** or **VS Code with the PlatformIO extension** (firmware builds)
- **Python 3.10+** (visualization tools)
- **Git**
- **OpenSCAD** (optional, only needed for CAD contributions)

### Clone and Build

```bash
git clone https://github.com/pathfinder-project/pathfinder.git
cd pathfinder
```

Build the firmware:

```bash
cd firmware && pio run
```

Run the host-side test suite (no hardware needed):

```bash
cd firmware && pio test -e native
```

Install the visualization tools:

```bash
pip install -r firmware/tools/requirements.txt
```

Verify the tools work with the included sample data:

```bash
python firmware/tools/visualize_data.py firmware/tools/example_data.csv
```

## 2. Project Structure

```
Pathfinder/
├── README.md              # Project overview, features, status
├── VISION.md              # Design philosophy and goals
├── CLAUDE.md              # AI assistant context and style guide
├── LICENSE                # MIT (firmware/software), CERN-OHL-S v2 (hardware)
│
├── firmware/              # PlatformIO project (Arduino Nano)
│   ├── platformio.ini     # Build environments and library deps
│   ├── src/main.cpp       # Main firmware (acquisition loop, logging, GPS)
│   ├── include/config.h   # All user-configurable parameters
│   ├── test/              # Unity test suites (run on host via native env)
│   │   ├── test_csv/      # CSV format validation tests
│   │   └── test_gradient/ # Gradient computation tests
│   └── tools/             # Python post-processing utilities
│       ├── visualize_data.py    # Time series and spatial map plots
│       ├── requirements.txt     # Python dependencies (pandas, matplotlib)
│       └── example_data.csv     # Sample data for testing tools
│
├── hardware/              # Physical design files
│   ├── bom/               # Bill of materials (pathfinder-bom.csv)
│   ├── cad/               # OpenSCAD sources, rendered STLs
│   │   ├── openscad/      # Parametric source files
│   │   └── stl/           # Pre-rendered meshes
│   └── schematics/        # Circuit documentation (main-board.md)
│
├── docs/                  # Project documentation
│   ├── design-concept.md  # Technical design rationale
│   └── platform-variants.md # Handheld, backpack, and drone configs
│
└── research/              # Background research and references
    └── sensor-selection.md
```

### Which Files to Modify for Common Changes

| Change | Files to edit |
|--------|--------------|
| Tune sample rate, GPS baud, ADC gain | `firmware/include/config.h` |
| Add a new sensor or data column | `firmware/src/main.cpp` (struct + writeCSVHeader + logReading) |
| Add a new build environment | `firmware/platformio.ini` |
| Change visualization or add a plot type | `firmware/tools/visualize_data.py` |
| Update component list or pricing | `hardware/bom/pathfinder-bom.csv` and `hardware/bom/README.md` |
| Modify frame or mount geometry | `hardware/cad/openscad/*.scad`, then re-render STLs |

### config.h vs main.cpp

`config.h` holds every value a builder might want to change without understanding the firmware internals: pin assignments, sample rates, baud rates, feature toggles, and thresholds. If you are adding a new tunable parameter, put it in `config.h` with a descriptive comment and a sensible default. The logic that uses those parameters lives in `main.cpp`.

## 3. Development Workflow

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/add-temperature-logging
   ```
   Use prefixes: `feature/`, `fix/`, `docs/`, `hardware/`.

2. **Make your changes**, following the style guidelines below.

3. **Run tests** before committing:
   ```bash
   cd firmware && pio test -e native
   ```

4. **Write descriptive commit messages:**
   ```
   Add temperature column to CSV output

   Adds DS18B20 one-wire temperature reading to the GradiometerReading
   struct and CSV log. Temperature helps identify thermal drift in
   fluxgate sensors during long surveys.

   Fixes #42
   ```
   First line: imperative mood, under 50 characters. Body: explain *why*, not just *what*. Reference issues when applicable.

5. **Create a pull request** with a description of what changed, why, and how to test it. If the change affects hardware or wiring, note that explicitly.

## 4. Firmware Development

### Adding a New Sensor

To add an I2C or SPI device to the firmware:

1. **Add the library** to `platformio.ini` under `[common] lib_deps`.
2. **Add a feature toggle** in `config.h`:
   ```cpp
   #ifndef ENABLE_TEMPERATURE
   #define ENABLE_TEMPERATURE  0
   #endif
   ```
3. **Guard all new code** with `#if ENABLE_TEMPERATURE` blocks in `main.cpp`.
4. **Add the global object and setup function** following the pattern of `setupADCs()` or `setupRTC()`.
5. **Extend `GradiometerReading`** with the new data fields.
6. **Update `writeCSVHeader()`** and **`logReading()`** to include the new columns.
7. **Update `printDebugInfo()`** so the serial monitor shows the new data.

### Modifying Data Output

The CSV format is generated at runtime. To add a new column:

1. Add the field to the `GradiometerReading` struct.
2. Add the column header in `writeCSVHeader()` using `logFile.print(F("..."))`.
3. Write the value in `logReading()` at the corresponding position.
4. Add a test case in `test/test_csv/test_csv.cpp` that verifies the new column count.
5. Update `firmware/tools/visualize_data.py` if the new column should appear in plots.

All three locations (struct, header, log) must stay in sync. If they diverge, the CSV will be malformed.

### Platform Variants

Pathfinder supports multiple hardware configurations through build environments in `platformio.ini`:

| Environment | Platform | Pairs | GPS | Beeper | Use case |
|-------------|----------|-------|-----|--------|----------|
| `nanoatmega328` | Handheld | 4 | NEO-6M (9600 baud) | Yes | Default walking survey |
| `nano_drone` | Drone | 2 | ZED-F9P (115200 baud) | No | UAV-mounted survey |
| `native` | Host | N/A | N/A | N/A | Unit tests only |

To add a new platform variant:

1. Add a `PLATFORM_*` define in `config.h` with appropriate defaults.
2. Add a `[env:your_variant]` section in `platformio.ini` with the correct `build_flags`.
3. Update the platform selection logic in `config.h` (the `#ifndef` chain at the top).
4. Update `docs/platform-variants.md` with the new configuration.

### Testing

Tests use the **Unity** framework and run on the host machine via PlatformIO's `native` environment. No Arduino hardware is needed.

Each test suite lives in its own subdirectory under `firmware/test/`:

```
test/
├── test_csv/test_csv.cpp           # CSV format validation
└── test_gradient/test_gradient.cpp # Gradient math verification
```

To write a new test:

1. Create a directory: `firmware/test/test_yourfeature/`
2. Create `test_yourfeature.cpp` with Unity test functions:
   ```cpp
   #include <unity.h>

   void test_something(void) {
       TEST_ASSERT_EQUAL_INT(expected, actual);
   }

   int main(int argc, char **argv) {
       UNITY_BEGIN();
       RUN_TEST(test_something);
       return UNITY_END();
   }
   ```
3. Run with `pio test -e native`. PlatformIO auto-discovers test directories matching `test_*`.

Extract the logic you want to test into pure functions that do not depend on Arduino hardware (no `Serial`, no `Wire`, no `digitalWrite`). The existing tests demonstrate this pattern: `compute_gradient()` and `build_csv_header()` are standalone C functions that mirror the firmware logic.

### Build Environments

- **`nanoatmega328`** (default): Full handheld firmware. Builds for ATmega328P. Upload with `pio run -t upload`.
- **`nano_drone`**: Drone variant. Same MCU, different build flags (2 pairs, fast GPS, no beeper).
- **`native`**: Host-side compilation for running tests. No Arduino libraries, no hardware access. Use `pio test -e native` to run.

## 5. Code Style

### C++ (Firmware)

- Follow Arduino conventions. Use `camelCase` for functions and variables, `UPPER_CASE` for `#define` constants.
- Use `#if FEATURE` guards for optional features, not `#ifdef`. This catches typos where a feature macro is defined but set to `0` -- `#ifdef` would still enable it.
  ```cpp
  // Correct:
  #if ENABLE_BEEPER
  updateBeeper();
  #endif

  // Wrong:
  #ifdef ENABLE_BEEPER
  updateBeeper();
  #endif
  ```
- Use the `F()` macro for string literals in `Serial.print()` and `logFile.print()`. This stores strings in flash instead of SRAM, which matters on the ATmega328's 2 KB of RAM.
  ```cpp
  Serial.println(F("SD card ready"));   // Correct: stored in flash
  Serial.println("SD card ready");      // Wrong: wastes 14 bytes of SRAM
  ```
- Keep functions short and focused. The existing code is organized into clear sections (setup, acquisition, UI). Follow the same pattern.
- Comments should explain *why*, not *what*. The code should be self-documenting for *what*.
  ```cpp
  // Good: explains a non-obvious design decision
  // Flush every 10 samples to limit data loss on power failure
  // without wearing out the SD card with per-sample writes
  if (sampleCount % SD_FLUSH_INTERVAL == 0) {
      logFile.flush();
  }

  // Bad: restates the code
  // Flush the log file
  logFile.flush();
  ```

### Python (Tools)

- Follow PEP 8. Use a formatter like `black` or `ruff format` if available.
- Add type hints to function signatures.
- Use numpy-style docstrings for public functions:
  ```python
  def load_survey(path: str, n_pairs: int = 4) -> pd.DataFrame:
      """Load a Pathfinder CSV file into a DataFrame.

      Parameters
      ----------
      path : str
          Path to the CSV file.
      n_pairs : int, optional
          Number of gradiometer pairs (default 4).

      Returns
      -------
      pd.DataFrame
          Survey data with gradient columns computed.
      """
  ```

## 6. Documentation

### Performance Claims

Follow the CLAUDE.md performance claims framework. Every specification must be tagged with its validation status:

| Qualifier | Meaning | Example |
|-----------|---------|---------|
| **(Measured)** | Bench or field tested | "Bench tests show 50 nT noise floor (Measured)" |
| **(Modeled)** | Theoretical estimate or simulation | "Detection depth estimated at 1.5 m (Modeled)" |
| **(Target)** | Design goal, not yet validated | "System weight <1.5 kg (Target)" |

Do not present targets as facts. If a number has not been measured, say so.

### Writing Style

- Use active voice: "The firmware logs readings at 10 Hz" not "Readings are logged at 10 Hz."
- Use first person plural for design decisions: "We selected the ADS1115 because..."
- Keep documentation close to the code it describes. Inline comments for implementation details, markdown files for architecture and concepts.
- Update documentation when changing behavior. A feature that works differently than the docs describe is a bug.

## 7. Hardware Contributions

Hardware changes affect multiple files. When modifying hardware, update all of the following:

### Bill of Materials

Edit `hardware/bom/pathfinder-bom.csv` with the component change (add, remove, or substitute). Then update `hardware/bom/README.md` to reflect any cost or sourcing changes. If the change affects the total build cost mentioned in `README.md`, update that too.

### CAD / Mechanical

Edit the parametric source in `hardware/cad/openscad/`, then re-render the STL:

```bash
openscad -o hardware/cad/stl/output.stl hardware/cad/openscad/source.scad
```

Commit both the `.scad` source and the `.stl` output. The STLs are checked in so that builders without OpenSCAD can still print parts.

### Schematics

The circuit is documented as a text description in `hardware/schematics/main-board.md`. If you change wiring, pin assignments, or add a component, update this file. If the change affects `config.h` pin definitions, update those too.

## 8. Testing Requirements

### Firmware

- All changes must pass the existing test suite: `pio test -e native`.
- New features should include tests where feasible. Extract testable logic into pure functions that do not depend on Arduino hardware.
- If a change affects CSV output format, add or update tests in `test/test_csv/`.
- If a change affects gradient computation or sensor math, add or update tests in `test/test_gradient/`.

### Visualization Tools

- Test changes against `firmware/tools/example_data.csv` to verify plots render correctly.
- If you add new command-line options, test them with and without the `--map` flag.

### Hardware Testing

- Test firmware changes on actual hardware before merging if at all possible.
- If you do not have hardware, say so in the pull request. Another contributor can test for you.
- For hardware design changes (BOM, wiring, mechanical), describe how you validated the change (bench test, simulation, datasheet analysis).

## 9. Related Projects

Pathfinder is part of a family of open-source geophysical tools. These projects complement each other but do not share code dependencies.

| Project | Purpose | Relationship |
|---------|---------|-------------|
| **HIRT** | Crosshole subsurface tomography | Pathfinder screens; HIRT investigates. Data formats are independent. |
| **GeoSim** | Shared simulation engine | Coordinate transforms and synthetic data generation. |

When making changes that affect cross-project workflows (data formats, coordinate conventions, file naming):

1. Submit PRs to all affected repositories.
2. Reference the related PRs in each description so reviewers can see the full picture.
3. Coordinate merge order if there are format dependencies.

## License

By contributing, you agree that your contributions will be licensed under the project's existing licenses: MIT for firmware and software, CERN-OHL-S v2.0 for hardware designs. See [LICENSE](../LICENSE) for details.
