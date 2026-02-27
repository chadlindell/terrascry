/**
 * HIRT Firmware - Configuration Header
 *
 * User-configurable parameters for the Hybrid Inductive-Resistivity
 * Tomography system. Modify these values to customize behavior without
 * editing main code.
 *
 * Target MCU: ESP32-WROOM-32
 * Hardware Rev: 2.3
 */

#ifndef CONFIG_H
#define CONFIG_H

// ============================================================================
// FIRMWARE VERSION
// ============================================================================

#ifndef FIRMWARE_VERSION
#define FIRMWARE_VERSION    "0.1.0"
#endif
#ifndef FIRMWARE_DATE
#define FIRMWARE_DATE       "2026-02-18"
#endif

// ============================================================================
// SPI BUS CONFIGURATION (ESP32 default VSPI)
// ============================================================================

#define SPI_MOSI_PIN        23
#define SPI_MISO_PIN        19
#define SPI_SCLK_PIN        18

// ============================================================================
// I2C BUS CONFIGURATION (ESP32 default)
// ============================================================================

#define I2C_SDA_PIN         21
#define I2C_SCL_PIN         22

// ============================================================================
// AD9833 DDS SIGNAL GENERATOR
// ============================================================================
// 3-wire SPI interface. Shares SCLK and SDATA with main SPI bus;
// FSYNC acts as chip select.

#define DDS_FSYNC_PIN       5     // Frame sync (chip select)
#define DDS_SDATA_PIN       SPI_MOSI_PIN  // Serial data (shared MOSI)
#define DDS_SCLK_PIN        SPI_SCLK_PIN  // Serial clock (shared SCLK)
#define DDS_MCLK_HZ        25000000      // Master clock frequency (25 MHz crystal)

// ============================================================================
// AD7124-8 ADC (24-bit Sigma-Delta)
// ============================================================================
// SPI interface on the shared bus. CS is dedicated.

#define ADC_CS_PIN          15    // ADC chip select
#define ADC_DRDY_PIN        4     // Data ready interrupt (optional)
#define ADC_SPI_SPEED       4000000  // 4 MHz SPI clock

// AD7124-8 configuration
#define ADC_VREF_MV         2500.0f  // Internal reference voltage (mV)
#define ADC_RESOLUTION_BITS 24
#define ADC_GAIN            1        // PGA gain setting (1, 2, 4, 8, 16, 32, 64, 128)
#define ADC_DATA_RATE       19200    // Output data rate (Hz) for lock-in sampling

// ============================================================================
// CD4051 ANALOG MULTIPLEXER (TX and RX switching)
// ============================================================================
// Two CD4051s are used: one for TX probe selection, one for RX probe selection.
// Address lines A, B, C select one of 8 channels. INH disables output.
// For >8 probes, multiple mux stages are daisy-chained.

// TX multiplexer
#define MUX_TX_A_PIN        25
#define MUX_TX_B_PIN        26
#define MUX_TX_C_PIN        27
#define MUX_TX_INH_PIN      14    // Inhibit (active high = output disabled)

// RX multiplexer
#define MUX_RX_A_PIN        32
#define MUX_RX_B_PIN        33
#define MUX_RX_C_PIN        34
#define MUX_RX_INH_PIN      35    // Inhibit

// ============================================================================
// ERT RELAY / SWITCH PINS
// ============================================================================
// Controls current injection electrode switching for resistivity measurements.
// Active-low relay drivers.

#define ERT_RELAY_A_PIN     12    // Current injection positive electrode
#define ERT_RELAY_B_PIN     13    // Current injection negative electrode
#define ERT_POLARITY_PIN    2     // Polarity reversal relay

// ============================================================================
// PROBE CONFIGURATION
// ============================================================================

// Number of probes in the array (configurable at compile time)
#ifndef NUM_PROBES
#define NUM_PROBES          24
#endif

// Maximum supported probes (hardware limit of mux stages)
#define MAX_PROBES          32

// Validate at compile time
#if NUM_PROBES < 1 || NUM_PROBES > MAX_PROBES
  #error "NUM_PROBES must be 1-32"
#endif

// ============================================================================
// MIT (Magneto-Inductive Tomography) PARAMETERS
// ============================================================================

// Transmit frequencies (Hz) - swept during MIT acquisition
#define NUM_TX_FREQUENCIES  5
static const uint32_t TX_FREQUENCIES[NUM_TX_FREQUENCIES] = {
    2000, 5000, 10000, 20000, 50000
};

// TX coil current target (mA)
#define TX_CURRENT_MA       10.0f

// TX stabilization time after frequency change (ms)
#define MIT_TX_SETTLE_MS    10

// Lock-in amplifier parameters
#define LOCKIN_NUM_CYCLES   16    // Number of signal cycles to average
#define LOCKIN_SAMPLE_RATE  192000  // ADC sample rate for lock-in (Hz)

// ============================================================================
// ERT (Electrical Resistivity Tomography) PARAMETERS
// ============================================================================

// Injection current (mA) - target for Howland current source
#define ERT_CURRENT_MA      1.0f

// Current stabilization time (ms)
#define ERT_SETTLE_MS       50

// Number of stacking cycles (repeat and average for noise reduction)
#define ERT_STACK_COUNT     4

// Enable polarity reversal (recommended for electrode polarization removal)
#define ERT_POLARITY_REVERSAL   1

// ============================================================================
// SD CARD LOGGING
// ============================================================================

#define SD_CS_PIN           16    // SD card chip select (dedicated SPI CS)

// File naming
#define MIT_FILE_PREFIX     "MIT_"
#define ERT_FILE_PREFIX     "ERT_"
#define LOG_FILE_EXTENSION  ".csv"

// Flush interval (number of readings between SD flushes)
#define SD_FLUSH_INTERVAL   10

// ============================================================================
// MQTT CONFIGURATION (optional wireless telemetry)
// ============================================================================

// Set to 1 to enable MQTT data publishing via WiFi
#ifndef ENABLE_MQTT
#define ENABLE_MQTT         0
#endif

#if ENABLE_MQTT
#define MQTT_BROKER         "192.168.4.1"
#define MQTT_PORT           1883
#define MQTT_TOPIC_PREFIX   "hirt/"
#define MQTT_CLIENT_ID      "hirt-hub"
#define MQTT_KEEPALIVE_S    60

// WiFi credentials (override via build flags for security)
#ifndef WIFI_SSID
#define WIFI_SSID           "HIRT_AP"
#endif
#ifndef WIFI_PASS
#define WIFI_PASS           "changeme"
#endif
#endif  // ENABLE_MQTT

// ============================================================================
// RTC CONFIGURATION (DS3231)
// ============================================================================

#ifndef ENABLE_RTC
#define ENABLE_RTC          1
#endif

#define DS3231_I2C_ADDR     0x68

// ============================================================================
// DEBUG / SERIAL SETTINGS
// ============================================================================

#ifndef SERIAL_DEBUG
#define SERIAL_DEBUG        1
#endif

#define SERIAL_BAUD         115200

// Print debug summary every N measurements
#define DEBUG_PRINT_INTERVAL  10

// ============================================================================
// SYSTEM TIMING
// ============================================================================

// Watchdog timeout (seconds)
#define WATCHDOG_TIMEOUT_S  2

// Communication timeout for probe response (ms)
#define COMM_TIMEOUT_MS     100

// Communication retry count
#define COMM_RETRY_COUNT    3

#endif // CONFIG_H
