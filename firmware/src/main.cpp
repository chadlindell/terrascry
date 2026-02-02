/**
 * Pathfinder Gradiometer Firmware
 *
 * 4-pair fluxgate gradiometer with GPS logging for archaeological/forensic reconnaissance.
 *
 * Hardware:
 * - Arduino Nano
 * - 2x ADS1115 16-bit ADC (I2C addresses 0x48, 0x49)
 * - 8x Fluxgate sensors (4 pairs: top/bottom)
 * - NEO-6M GPS module
 * - SD card module (SPI)
 * - Piezo beeper for pace marking
 *
 * Data Format:
 * CSV with columns: timestamp,lat,lon,g1_top,g1_bot,g1_grad,...(4 pairs)
 *
 * Author: Pathfinder Project
 * License: Open Source (specify your license)
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <TinyGPSPlus.h>
#include <SdFat.h>
#include <SoftwareSerial.h>
#include "config.h"

// ============================================================================
// GLOBAL OBJECTS
// ============================================================================

// ADC objects for two ADS1115 modules
Adafruit_ADS1115 ads1;  // Address 0x48 - Pairs 1 and 2
Adafruit_ADS1115 ads2;  // Address 0x49 - Pairs 3 and 4

// GPS objects
SoftwareSerial gpsSerial(GPS_RX_PIN, GPS_TX_PIN);
TinyGPSPlus gps;

// SD card objects
SdFat sd;
SdFile logFile;

// ============================================================================
// GLOBAL STATE
// ============================================================================

struct GradiometerReading {
    uint32_t timestamp_ms;
    double latitude;
    double longitude;
    int16_t g1_top, g1_bot, g1_grad;
    int16_t g2_top, g2_bot, g2_grad;
    int16_t g3_top, g3_bot, g3_grad;
    int16_t g4_top, g4_bot, g4_grad;
};

// Timing state
unsigned long lastSampleTime = 0;
unsigned long lastBeepTime = 0;
unsigned long lastBlinkTime = 0;
unsigned long lastFlushTime = 0;
unsigned long sampleCount = 0;

// System state
bool sdCardReady = false;
bool gpsLocked = false;
bool ledState = false;
char logFileName[16];

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

void setupPins();
void setupADCs();
void setupGPS();
void setupSD();
void createLogFile();
void readGradiometers(GradiometerReading &reading);
void logReading(const GradiometerReading &reading);
void updateBeeper();
void updateStatusLED();
void printDebugInfo(const GradiometerReading &reading);

// ============================================================================
// SETUP
// ============================================================================

void setup() {
    // Initialize serial for debugging
    #if SERIAL_DEBUG
    Serial.begin(SERIAL_BAUD);
    while (!Serial && millis() < 3000); // Wait up to 3 seconds for serial
    Serial.println(F("Pathfinder Gradiometer Starting..."));
    #endif

    // Setup hardware
    setupPins();
    setupADCs();
    setupGPS();
    setupSD();
    createLogFile();

    #if SERIAL_DEBUG
    Serial.println(F("Initialization complete. Starting acquisition..."));
    Serial.print(F("Sample rate: "));
    Serial.print(SAMPLE_RATE_HZ);
    Serial.println(F(" Hz"));
    Serial.print(F("Log file: "));
    Serial.println(logFileName);
    #endif

    // Initial beep to signal ready
    digitalWrite(BEEPER_PIN, HIGH);
    delay(200);
    digitalWrite(BEEPER_PIN, LOW);
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
    unsigned long currentTime = millis();

    // Update GPS data (call frequently to process NMEA sentences)
    while (gpsSerial.available() > 0) {
        gps.encode(gpsSerial.read());
    }

    // Check if GPS has valid fix
    gpsLocked = gps.location.isValid() && gps.location.age() < 2000;

    // Acquire data at configured sample rate
    unsigned long sampleInterval = 1000 / SAMPLE_RATE_HZ;
    if (currentTime - lastSampleTime >= sampleInterval) {
        lastSampleTime = currentTime;

        // Read all gradiometer channels
        GradiometerReading reading;
        readGradiometers(reading);

        // Get GPS coordinates (or 0,0 if no fix)
        if (gpsLocked) {
            reading.latitude = gps.location.lat();
            reading.longitude = gps.location.lng();
        } else {
            reading.latitude = 0.0;
            reading.longitude = 0.0;
        }

        // Log to SD card
        if (sdCardReady) {
            logReading(reading);
            sampleCount++;

            // Flush to SD card periodically
            if (sampleCount % SD_FLUSH_INTERVAL == 0) {
                logFile.flush();
            }
        }

        // Print debug info
        #if SERIAL_DEBUG
        if (sampleCount % DEBUG_PRINT_INTERVAL == 0) {
            printDebugInfo(reading);
        }
        #endif
    }

    // Update pace beeper
    updateBeeper();

    // Update status LED
    updateStatusLED();
}

// ============================================================================
// HARDWARE SETUP FUNCTIONS
// ============================================================================

void setupPins() {
    pinMode(BEEPER_PIN, OUTPUT);
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(BEEPER_PIN, LOW);
    digitalWrite(STATUS_LED_PIN, LOW);

    #if SERIAL_DEBUG
    Serial.println(F("Pins configured"));
    #endif
}

void setupADCs() {
    // Initialize I2C
    Wire.begin();

    // Configure ADS1115 module 1
    if (!ads1.begin(ADS1115_ADDR_1)) {
        #if SERIAL_DEBUG
        Serial.println(F("ERROR: ADS1115 #1 (0x48) not found!"));
        #endif
        while (1) {
            // Blink error pattern
            digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
            delay(BLINK_ERROR);
        }
    }

    // Configure ADS1115 module 2
    if (!ads2.begin(ADS1115_ADDR_2)) {
        #if SERIAL_DEBUG
        Serial.println(F("ERROR: ADS1115 #2 (0x49) not found!"));
        #endif
        while (1) {
            // Blink error pattern
            digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
            delay(BLINK_ERROR);
        }
    }

    // Set gain and data rate
    ads1.setGain(ADC_GAIN);
    ads1.setDataRate(ADC_DATA_RATE);
    ads2.setGain(ADC_GAIN);
    ads2.setDataRate(ADC_DATA_RATE);

    #if SERIAL_DEBUG
    Serial.println(F("ADS1115 modules configured"));
    #endif
}

void setupGPS() {
    gpsSerial.begin(GPS_BAUD);

    #if SERIAL_DEBUG
    Serial.print(F("GPS initialized at "));
    Serial.print(GPS_BAUD);
    Serial.println(F(" baud"));
    Serial.println(F("Waiting for GPS fix..."));
    #endif
}

void setupSD() {
    pinMode(SD_CS_PIN, OUTPUT);

    if (!sd.begin(SD_CS_PIN, SD_SCK_MHZ(4))) {
        #if SERIAL_DEBUG
        Serial.println(F("ERROR: SD card initialization failed!"));
        #endif
        sdCardReady = false;
        return;
    }

    sdCardReady = true;

    #if SERIAL_DEBUG
    Serial.println(F("SD card ready"));
    #endif
}

void createLogFile() {
    if (!sdCardReady) return;

    // Find next available filename: PATH0001.CSV, PATH0002.CSV, etc.
    for (uint16_t i = 1; i < 10000; i++) {
        snprintf(logFileName, sizeof(logFileName), "%s%04d%s",
                 LOG_FILE_PREFIX, i, LOG_FILE_EXTENSION);

        if (!sd.exists(logFileName)) {
            // File doesn't exist, use this name
            if (logFile.open(logFileName, O_CREAT | O_WRITE | O_EXCL)) {
                // Write CSV header
                logFile.println(F(CSV_HEADER));
                logFile.flush();

                #if SERIAL_DEBUG
                Serial.print(F("Created log file: "));
                Serial.println(logFileName);
                #endif
                return;
            }
        }
    }

    #if SERIAL_DEBUG
    Serial.println(F("ERROR: Could not create log file"));
    #endif
    sdCardReady = false;
}

// ============================================================================
// DATA ACQUISITION FUNCTIONS
// ============================================================================

void readGradiometers(GradiometerReading &reading) {
    reading.timestamp_ms = millis();

    // Read pair 1 (ADS1 channels 0 and 1)
    reading.g1_top = ads1.readADC_SingleEnded(PAIR1_TOP_CHANNEL);
    reading.g1_bot = ads1.readADC_SingleEnded(PAIR1_BOT_CHANNEL);
    reading.g1_grad = reading.g1_bot - reading.g1_top;

    // Read pair 2 (ADS1 channels 2 and 3)
    reading.g2_top = ads1.readADC_SingleEnded(PAIR2_TOP_CHANNEL);
    reading.g2_bot = ads1.readADC_SingleEnded(PAIR2_BOT_CHANNEL);
    reading.g2_grad = reading.g2_bot - reading.g2_top;

    // Read pair 3 (ADS2 channels 0 and 1)
    reading.g3_top = ads2.readADC_SingleEnded(PAIR3_TOP_CHANNEL);
    reading.g3_bot = ads2.readADC_SingleEnded(PAIR3_BOT_CHANNEL);
    reading.g3_grad = reading.g3_bot - reading.g3_top;

    // Read pair 4 (ADS2 channels 2 and 3)
    reading.g4_top = ads2.readADC_SingleEnded(PAIR4_TOP_CHANNEL);
    reading.g4_bot = ads2.readADC_SingleEnded(PAIR4_BOT_CHANNEL);
    reading.g4_grad = reading.g4_bot - reading.g4_top;
}

void logReading(const GradiometerReading &reading) {
    if (!sdCardReady) return;

    // Format: timestamp,lat,lon,g1_top,g1_bot,g1_grad,...
    logFile.print(reading.timestamp_ms);
    logFile.print(',');
    logFile.print(reading.latitude, 7);  // 7 decimal places (~1 cm precision)
    logFile.print(',');
    logFile.print(reading.longitude, 7);
    logFile.print(',');

    // Pair 1
    logFile.print(reading.g1_top);
    logFile.print(',');
    logFile.print(reading.g1_bot);
    logFile.print(',');
    logFile.print(reading.g1_grad);
    logFile.print(',');

    // Pair 2
    logFile.print(reading.g2_top);
    logFile.print(',');
    logFile.print(reading.g2_bot);
    logFile.print(',');
    logFile.print(reading.g2_grad);
    logFile.print(',');

    // Pair 3
    logFile.print(reading.g3_top);
    logFile.print(',');
    logFile.print(reading.g3_bot);
    logFile.print(',');
    logFile.print(reading.g3_grad);
    logFile.print(',');

    // Pair 4
    logFile.print(reading.g4_top);
    logFile.print(',');
    logFile.print(reading.g4_bot);
    logFile.print(',');
    logFile.println(reading.g4_grad);
}

// ============================================================================
// USER INTERFACE FUNCTIONS
// ============================================================================

void updateBeeper() {
    unsigned long currentTime = millis();

    // Generate beep at configured interval
    if (currentTime - lastBeepTime >= BEEP_INTERVAL_MS) {
        lastBeepTime = currentTime;
        digitalWrite(BEEPER_PIN, HIGH);
        delay(BEEP_DURATION_MS);
        digitalWrite(BEEPER_PIN, LOW);
    }
}

void updateStatusLED() {
    unsigned long currentTime = millis();
    unsigned long blinkInterval;

    // Select blink pattern based on system state
    if (!sdCardReady) {
        blinkInterval = BLINK_ERROR;
    } else if (!gpsLocked) {
        blinkInterval = BLINK_NO_GPS;
    } else {
        blinkInterval = BLINK_LOGGING;
    }

    // Toggle LED at appropriate rate
    if (currentTime - lastBlinkTime >= blinkInterval) {
        lastBlinkTime = currentTime;
        ledState = !ledState;
        digitalWrite(STATUS_LED_PIN, ledState);
    }
}

void printDebugInfo(const GradiometerReading &reading) {
    #if SERIAL_DEBUG
    Serial.print(F("Sample: "));
    Serial.print(sampleCount);
    Serial.print(F(" | GPS: "));
    if (gpsLocked) {
        Serial.print(reading.latitude, 6);
        Serial.print(F(","));
        Serial.print(reading.longitude, 6);
    } else {
        Serial.print(F("NO FIX"));
    }
    Serial.print(F(" | G1: "));
    Serial.print(reading.g1_grad);
    Serial.print(F(" | G2: "));
    Serial.print(reading.g2_grad);
    Serial.print(F(" | G3: "));
    Serial.print(reading.g3_grad);
    Serial.print(F(" | G4: "));
    Serial.println(reading.g4_grad);
    #endif
}
