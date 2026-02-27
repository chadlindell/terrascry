/**
 * Pathfinder Gradiometer - Configuration Header
 *
 * User-configurable parameters for the fluxgate gradiometer.
 * Modify these values to customize system behavior without editing main code.
 *
 * License: MIT (see LICENSE in project root)
 */

#ifndef CONFIG_H
#define CONFIG_H

// ============================================================================
// FIRMWARE VERSION
// ============================================================================

#define FIRMWARE_VERSION    "1.4.0"
#define FIRMWARE_DATE       "2026-02-18"

// ============================================================================
// PLATFORM SELECTION
// ============================================================================
// Uncomment ONE platform define, or set via build_flags in platformio.ini.
// Defaults to handheld if none specified.

#ifndef PLATFORM_HANDHELD
#ifndef PLATFORM_BACKPACK
#ifndef PLATFORM_DRONE
#define PLATFORM_HANDHELD   1
#endif
#endif
#endif

// ============================================================================
// SENSOR PAIR COUNT
// ============================================================================
// Number of gradiometer pairs (1-4). Each pair = 2 sensors (top + bottom).
//   1 pair  = 2 sensors, 1x ADS1115 (channels 0-1)
//   2 pairs = 4 sensors, 1x ADS1115 (channels 0-3)
//   3 pairs = 6 sensors, 2x ADS1115
//   4 pairs = 8 sensors, 2x ADS1115 (full)
//
// Platform defaults:
//   Handheld: 4 pairs (1.5m swath)
//   Backpack: 4 pairs
//   Drone:    2 pairs (weight limited)
#ifndef NUM_SENSOR_PAIRS
  #ifdef PLATFORM_DRONE
    #define NUM_SENSOR_PAIRS  2
  #else
    #define NUM_SENSOR_PAIRS  4
  #endif
#endif

// Validate at compile time
#if NUM_SENSOR_PAIRS < 1 || NUM_SENSOR_PAIRS > 4
  #error "NUM_SENSOR_PAIRS must be 1-4"
#endif

// Derived: does this configuration need the second ADS1115?
#define NEEDS_ADC2  (NUM_SENSOR_PAIRS > 2)

// ============================================================================
// PLATFORM-DERIVED DEFAULTS
// ============================================================================

#ifdef PLATFORM_DRONE
  #ifndef ENABLE_BEEPER
  #define ENABLE_BEEPER       0
  #endif
  #ifndef SAMPLE_RATE_HZ
  #define SAMPLE_RATE_HZ      20
  #endif
  #ifndef GPS_LOG_QUALITY
  #define GPS_LOG_QUALITY     1
  #endif
#endif

// Beeper enabled by default on handheld/backpack
#ifndef ENABLE_BEEPER
#define ENABLE_BEEPER         1
#endif

// Log GPS fix quality and HDOP (recommended for RTK, optional otherwise)
#ifndef GPS_LOG_QUALITY
#define GPS_LOG_QUALITY       0
#endif

// ============================================================================
// WATCHDOG TIMER (opt-in)
// ============================================================================
// Set to 1 to enable hardware watchdog (4-second timeout).
// WARNING: Some Arduino Nano bootloaders do not handle watchdog resets
// correctly, causing a boot loop. Test with your specific board first.
#ifndef ENABLE_WATCHDOG
#define ENABLE_WATCHDOG     0
#endif

// ============================================================================
// REAL-TIME CLOCK (opt-in)
// ============================================================================
// Set to 1 to enable DS3231 RTC for ISO 8601 absolute timestamps.
// When disabled, timestamps use millis() (relative ms since boot).
// Required for correlation with HIRT or other absolute-time systems.
#ifndef ENABLE_RTC
#define ENABLE_RTC          0
#endif

// DS3231 I2C address (fixed by hardware)
#define DS3231_I2C_ADDR     0x68

// ============================================================================
// HARDWARE PIN ASSIGNMENTS
// ============================================================================

// I2C pins (fixed on Arduino Nano)
// SDA: A4
// SCL: A5

// ADS1115 I2C Addresses
#define ADS1115_ADDR_1  0x48  // First ADC (pairs 1-2)
#define ADS1115_ADDR_2  0x49  // Second ADC (pairs 3-4), only used if NEEDS_ADC2

// GPS UART pins
#define GPS_RX_PIN      4     // Connect to GPS TX
#define GPS_TX_PIN      3     // Connect to GPS RX (not used for NEO-6M)

// SD Card SPI pins (hardware SPI)
#define SD_CS_PIN       10    // Chip Select
// MOSI: 11 (fixed)
// MISO: 12 (fixed)
// SCK:  13 (fixed)

// Pace beeper and status LED
#define BEEPER_PIN      9     // Piezo buzzer or speaker
#define STATUS_LED_PIN  2     // Status indicator LED

// ============================================================================
// ACQUISITION PARAMETERS
// ============================================================================

// Sample rate (Hz) - how many readings per second
// Typical walking pace: 10-20 Hz is sufficient
// Maximum: ~50 Hz with dual ADS1115 at 860 SPS setting
#ifndef SAMPLE_RATE_HZ
#define SAMPLE_RATE_HZ  10
#endif

// GPS baud rate (NEO-6M default is 9600; ZED-F9P uses 115200)
#ifndef GPS_BAUD
#define GPS_BAUD        9600
#endif

// ADS1115 gain setting
// Options: GAIN_TWOTHIRDS, GAIN_ONE, GAIN_TWO, GAIN_FOUR, GAIN_EIGHT, GAIN_SIXTEEN
// GAIN_ONE = +/- 4.096V range (most common for fluxgate sensors)
#define ADC_GAIN        adsGain_t::GAIN_ONE

// ADS1115 data rate (samples per second per channel)
// Options: 8, 16, 32, 64, 128, 250, 475, 860
// Higher = faster updates but more noise
#define ADC_DATA_RATE   128

// ADC saturation detection threshold (16-bit ADC max is 32767)
// Readings above this value indicate the sensor is likely saturated
#define ADC_SATURATION_THRESHOLD  32000

// ============================================================================
// CHANNEL MAPPING
// ============================================================================
// Each pair maps to one ADS1115 and two adjacent channels (top, bottom).
// Pairs 1-2 use ADC1 (0x48), pairs 3-4 use ADC2 (0x49).
// This layout is fixed by the hardware wiring. Only NUM_SENSOR_PAIRS
// determines how many pairs are actually read.
//
//   Pair 0 (label "1"): ADC1, channels 0 (top), 1 (bot)
//   Pair 1 (label "2"): ADC1, channels 2 (top), 3 (bot)
//   Pair 2 (label "3"): ADC2, channels 0 (top), 1 (bot)
//   Pair 3 (label "4"): ADC2, channels 2 (top), 3 (bot)
//
// The arrays below are indexed 0-3 (max). Only [0..NUM_SENSOR_PAIRS-1] are used.

#define MAX_PAIRS 4

// Which ADC module (1 or 2) serves each pair
static const uint8_t PAIR_ADC[MAX_PAIRS]     = {1, 1, 2, 2};
// ADS1115 channel for the top sensor of each pair
static const uint8_t PAIR_TOP_CH[MAX_PAIRS]  = {0, 2, 0, 2};
// ADS1115 channel for the bottom sensor of each pair
static const uint8_t PAIR_BOT_CH[MAX_PAIRS]  = {1, 3, 1, 3};

// ============================================================================
// PACE BEEPER SETTINGS
// ============================================================================

// Beep interval (milliseconds)
// 1000 ms = 1 beep per second (standard for ~1 m/s walking pace)
#define BEEP_INTERVAL_MS    1000

// Beep duration (milliseconds)
#define BEEP_DURATION_MS    50

// Beep frequency (Hz) - only used if PWM beeper
#define BEEP_FREQUENCY_HZ   2000

// ============================================================================
// SD CARD LOGGING
// ============================================================================

// Filename pattern: "PATHXXXX.CSV" where XXXX increments automatically
#define LOG_FILE_PREFIX     "PATH"
#define LOG_FILE_EXTENSION  ".CSV"

// NOTE: CSV header is generated at runtime in createLogFile() based on
// NUM_SENSOR_PAIRS and GPS_LOG_QUALITY. No CSV_HEADER macro needed.

// Flush buffer to SD card every N samples (prevents data loss on power failure)
#define SD_FLUSH_INTERVAL   10

// Number of consecutive SD write failures before attempting file re-open
#define SD_RETRY_THRESHOLD  5

// ============================================================================
// STATUS LED BLINK PATTERNS (in milliseconds)
// ============================================================================

#define BLINK_STARTUP       100   // Fast blink during initialization
#define BLINK_NO_GPS        500   // Medium blink when GPS not locked
#define BLINK_LOGGING       2000  // Slow blink during normal logging
#define BLINK_ERROR         100   // Fast blink on error

// ============================================================================
// DEBUG SETTINGS
// ============================================================================

// Enable serial debug output (disable for battery efficiency)
#ifndef SERIAL_DEBUG
#define SERIAL_DEBUG        1
#endif

// Serial baud rate for debug output
#define SERIAL_BAUD         115200

// Print debug info every N samples
#define DEBUG_PRINT_INTERVAL 10

#endif // CONFIG_H
