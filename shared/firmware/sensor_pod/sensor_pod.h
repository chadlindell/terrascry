/**
 * TERRASCRY Shared Sensor Pod Driver
 *
 * Manages the shared sensor pod (ZED-F9P + BNO055 + BMP390 + DS3231)
 * connected via PCA9615 differential I2C on Bus 1.
 *
 * Used by both Pathfinder and HIRT firmware.
 *
 * I2C Bus 1 addresses:
 *   ZED-F9P  0x42  RTK GPS
 *   BNO055   0x29  9-axis IMU
 *   BMP390   0x77  Barometric pressure
 *   DS3231   0x68  Real-time clock
 *
 * Connection: M8 8-pin connector → PCA9615 → Cat5 STP (1-2m)
 */

#ifndef TERRASCRY_SENSOR_POD_H
#define TERRASCRY_SENSOR_POD_H

#include <Wire.h>
#include <stdint.h>
#include <stdbool.h>

// I2C addresses (Bus 1)
#define POD_I2C_ADDR_GPS     0x42  // ZED-F9P
#define POD_I2C_ADDR_IMU     0x29  // BNO055
#define POD_I2C_ADDR_BARO    0x77  // BMP390
#define POD_I2C_ADDR_RTC     0x68  // DS3231

// I2C Bus 1 pins (ESP32)
// NOTE: GPIO 32/33 for UART2, GPIO 16/17 for I2C Bus 1
// Do NOT use GPIO 16/17 for UART — conflicts with I2C Bus 1
#define POD_I2C_SDA  16
#define POD_I2C_SCL  17

// GPS fix types
typedef enum {
    GPS_FIX_NONE       = 0,
    GPS_FIX_AUTONOMOUS = 1,
    GPS_FIX_DGPS       = 2,
    GPS_FIX_RTK_FLOAT  = 4,
    GPS_FIX_RTK_FIXED  = 5
} gps_fix_type_t;

// Sensor pod state (shared between Pathfinder and HIRT)
typedef struct {
    bool present;              // Pod detected on I2C bus
    // GPS
    uint8_t fix_type;          // gps_fix_type_t
    double latitude;           // WGS84 degrees
    double longitude;          // WGS84 degrees
    float altitude_m;          // Above ellipsoid
    float hdop;
    uint8_t satellites;
    // IMU (BNO055, IMUPLUS mode — magnetometer disabled)
    float pitch_deg;
    float roll_deg;
    float heading_deg;         // From gyro integration, not magnetometer
    // Barometer
    float pressure_pa;
    float baro_altitude_m;     // Barometric altitude
    float temperature_c;       // BMP390 internal temp
    // RTC
    uint32_t rtc_unix;         // Unix timestamp from DS3231
} sensor_pod_state_t;

/**
 * Initialize I2C Bus 1 and scan for sensor pod.
 * Returns true if at least ZED-F9P is detected.
 * Graceful degradation: missing sensors are flagged but pod still usable.
 */
bool sensor_pod_init(TwoWire *bus1);

/**
 * Periodic re-scan for hot-plugged pod.
 * Call from main loop at ~1 Hz.
 */
bool sensor_pod_detect(TwoWire *bus1);

/**
 * Read all pod sensors into state struct.
 * Non-responsive sensors are skipped (values unchanged).
 */
void sensor_pod_read(TwoWire *bus1, sensor_pod_state_t *state);

/**
 * Read GPS only (for HIRT probe position recording).
 * Blocks until fix_type >= GPS_FIX_AUTONOMOUS or timeout_ms expires.
 * Returns true if valid fix obtained.
 */
bool sensor_pod_read_gps_blocking(TwoWire *bus1, sensor_pod_state_t *state, uint32_t timeout_ms);

#endif // TERRASCRY_SENSOR_POD_H
