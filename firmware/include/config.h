/**
 * Pathfinder Gradiometer - Configuration Header
 *
 * User-configurable parameters for the 4-pair fluxgate gradiometer.
 * Modify these values to customize system behavior without editing main code.
 */

#ifndef CONFIG_H
#define CONFIG_H

// ============================================================================
// HARDWARE PIN ASSIGNMENTS
// ============================================================================

// I2C pins (fixed on Arduino Nano)
// SDA: A4
// SCL: A5

// ADS1115 I2C Addresses
#define ADS1115_ADDR_1  0x48  // First ADC (channels 0-3: pairs 1-2)
#define ADS1115_ADDR_2  0x49  // Second ADC (channels 0-3: pairs 3-4)

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
#define SAMPLE_RATE_HZ  10

// GPS baud rate (NEO-6M default is 9600)
#define GPS_BAUD        9600

// ADS1115 gain setting
// Options: GAIN_TWOTHIRDS, GAIN_ONE, GAIN_TWO, GAIN_FOUR, GAIN_EIGHT, GAIN_SIXTEEN
// GAIN_ONE = +/- 4.096V range (most common for fluxgate sensors)
#define ADC_GAIN        adsGain_t::GAIN_ONE

// ADS1115 data rate (samples per second per channel)
// Options: 8, 16, 32, 64, 128, 250, 475, 860
// Higher = faster updates but more noise
#define ADC_DATA_RATE   128

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

// CSV header row
#define CSV_HEADER "timestamp,lat,lon,"\
                   "g1_top,g1_bot,g1_grad,"\
                   "g2_top,g2_bot,g2_grad,"\
                   "g3_top,g3_bot,g3_grad,"\
                   "g4_top,g4_bot,g4_grad"

// Flush buffer to SD card every N samples (prevents data loss on power failure)
#define SD_FLUSH_INTERVAL   10

// ============================================================================
// STATUS LED BLINK PATTERNS (in milliseconds)
// ============================================================================

#define BLINK_STARTUP       100   // Fast blink during initialization
#define BLINK_NO_GPS        500   // Medium blink when GPS not locked
#define BLINK_LOGGING       2000  // Slow blink during normal logging
#define BLINK_ERROR         100   // Fast blink on error

// ============================================================================
// CHANNEL MAPPING
// ============================================================================

// ADS1115 Module 1 (0x48): Pairs 1 and 2
#define PAIR1_TOP_ADC       1
#define PAIR1_TOP_CHANNEL   0
#define PAIR1_BOT_ADC       1
#define PAIR1_BOT_CHANNEL   1

#define PAIR2_TOP_ADC       1
#define PAIR2_TOP_CHANNEL   2
#define PAIR2_BOT_ADC       1
#define PAIR2_BOT_CHANNEL   3

// ADS1115 Module 2 (0x49): Pairs 3 and 4
#define PAIR3_TOP_ADC       2
#define PAIR3_TOP_CHANNEL   0
#define PAIR3_BOT_ADC       2
#define PAIR3_BOT_CHANNEL   1

#define PAIR4_TOP_ADC       2
#define PAIR4_TOP_CHANNEL   2
#define PAIR4_BOT_ADC       2
#define PAIR4_BOT_CHANNEL   3

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
