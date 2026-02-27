/**
 * Pathfinder Gradiometer Firmware
 *
 * Modular fluxgate gradiometer with GPS logging for archaeological/forensic
 * reconnaissance. Supports 1-4 sensor pairs across handheld, backpack,
 * and drone platforms.
 *
 * Hardware:
 * - Arduino Nano
 * - 1-2x ADS1115 16-bit ADC (I2C addresses 0x48, 0x49)
 * - 2-8 Fluxgate sensors (1-4 pairs: top/bottom)
 * - NEO-6M or ZED-F9P GPS module
 * - SD card module (SPI)
 * - Piezo beeper for pace marking (handheld/backpack only)
 *
 * Data Format:
 * CSV with columns: timestamp,lat,lon,[fix_quality,hdop,altitude,]
 *                   g1_top,g1_bot,g1_grad,...(NUM_SENSOR_PAIRS pairs)
 *
 * Author: Pathfinder Project
 * License: MIT (see LICENSE in project root)
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <TinyGPSPlus.h>
#include <SdFat.h>
#include <SoftwareSerial.h>
#include "config.h"

#if ENABLE_WATCHDOG
#include <avr/wdt.h>
#endif

// ============================================================================
// GLOBAL OBJECTS
// ============================================================================

// ADC objects
Adafruit_ADS1115 ads1;  // Address 0x48 - Pairs 1 and 2
#if NEEDS_ADC2
Adafruit_ADS1115 ads2;  // Address 0x49 - Pairs 3 and 4
#endif

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
    int16_t top[NUM_SENSOR_PAIRS];
    int16_t bot[NUM_SENSOR_PAIRS];
    int16_t grad[NUM_SENSOR_PAIRS];
};

// Timing state
unsigned long lastSampleTime = 0;
unsigned long lastBeepTime = 0;
unsigned long lastBlinkTime = 0;
unsigned long sampleCount = 0;

// Non-blocking beeper state
bool beeping = false;
unsigned long beepStartTime = 0;

// System state
bool sdCardReady = false;
bool gpsLocked = false;
bool ledState = false;
bool adc1Ready = false;
#if NEEDS_ADC2
bool adc2Ready = false;
#endif
char logFileName[16];

// Error tracking
uint16_t sdWriteErrors = 0;
uint16_t adcSaturationCount = 0;

// ============================================================================
// FUNCTION DECLARATIONS
// ============================================================================

void setupPins();
void setupADCs();
void setupGPS();
void setupSD();
void createLogFile();
void writeCSVHeader();
void readGradiometers(GradiometerReading &reading);
void logReading(const GradiometerReading &reading);
void updateBeeper();
void updateStatusLED();
void printDebugInfo(const GradiometerReading &reading);
void checkSaturation(int16_t value);
void reopenLogFile();
bool anyAdcReady();

// ============================================================================
// HELPER
// ============================================================================

bool anyAdcReady() {
    #if NEEDS_ADC2
    return adc1Ready || adc2Ready;
    #else
    return adc1Ready;
    #endif
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
    #if ENABLE_WATCHDOG
    wdt_disable();
    #endif

    #if SERIAL_DEBUG
    Serial.begin(SERIAL_BAUD);
    while (!Serial && millis() < 3000);
    Serial.println(F("Pathfinder Gradiometer v" FIRMWARE_VERSION " (" FIRMWARE_DATE ")"));
    #ifdef PLATFORM_DRONE
    Serial.println(F("Platform: DRONE"));
    #elif defined(PLATFORM_BACKPACK)
    Serial.println(F("Platform: BACKPACK"));
    #else
    Serial.println(F("Platform: HANDHELD"));
    #endif
    Serial.print(F("Sensor pairs: "));
    Serial.println(NUM_SENSOR_PAIRS);
    #if GPS_LOG_QUALITY
    Serial.println(F("GPS quality logging: ON"));
    #endif
    #endif

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
    if (sdCardReady) {
        Serial.print(F("Log file: "));
        Serial.println(logFileName);
    }
    Serial.print(F("ADC1: "));
    Serial.println(adc1Ready ? F("OK") : F("FAILED"));
    #if NEEDS_ADC2
    Serial.print(F("ADC2: "));
    Serial.println(adc2Ready ? F("OK") : F("FAILED"));
    #endif
    #endif

    #if ENABLE_BEEPER
    digitalWrite(BEEPER_PIN, HIGH);
    delay(200);
    digitalWrite(BEEPER_PIN, LOW);
    #endif

    #if ENABLE_WATCHDOG
    wdt_enable(WDTO_4S);
    #endif
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
    #if ENABLE_WATCHDOG
    wdt_reset();
    #endif

    unsigned long currentTime = millis();

    // Update GPS data (call frequently to process NMEA sentences)
    while (gpsSerial.available() > 0) {
        gps.encode(gpsSerial.read());
    }

    gpsLocked = gps.location.isValid() && gps.location.age() < 2000;

    // Acquire data at configured sample rate
    unsigned long sampleInterval = 1000 / SAMPLE_RATE_HZ;
    if (currentTime - lastSampleTime >= sampleInterval) {
        lastSampleTime = currentTime;

        GradiometerReading reading;
        readGradiometers(reading);

        if (gpsLocked) {
            reading.latitude = gps.location.lat();
            reading.longitude = gps.location.lng();
        } else {
            reading.latitude = 0.0;
            reading.longitude = 0.0;
        }

        if (sdCardReady) {
            logReading(reading);
            sampleCount++;

            if (sampleCount % SD_FLUSH_INTERVAL == 0) {
                logFile.flush();
            }
        }

        #if SERIAL_DEBUG
        if (sampleCount % DEBUG_PRINT_INTERVAL == 0) {
            printDebugInfo(reading);
        }
        #endif
    }

    #if ENABLE_BEEPER
    updateBeeper();
    #endif

    updateStatusLED();
}

// ============================================================================
// HARDWARE SETUP FUNCTIONS
// ============================================================================

void setupPins() {
    #if ENABLE_BEEPER
    pinMode(BEEPER_PIN, OUTPUT);
    digitalWrite(BEEPER_PIN, LOW);
    #endif
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);

    #if SERIAL_DEBUG
    Serial.println(F("Pins configured"));
    #endif
}

void setupADCs() {
    Wire.begin();

    // ADC1 serves pairs 1-2 (always needed)
    if (ads1.begin(ADS1115_ADDR_1)) {
        ads1.setGain(ADC_GAIN);
        ads1.setDataRate(ADC_DATA_RATE);
        adc1Ready = true;
        #if SERIAL_DEBUG
        Serial.println(F("ADS1115 #1 (0x48) OK"));
        #endif
    } else {
        adc1Ready = false;
        #if SERIAL_DEBUG
        Serial.println(F("WARNING: ADS1115 #1 (0x48) not found"));
        #endif
    }

    // ADC2 serves pairs 3-4 (only needed for 3+ pairs)
    #if NEEDS_ADC2
    if (ads2.begin(ADS1115_ADDR_2)) {
        ads2.setGain(ADC_GAIN);
        ads2.setDataRate(ADC_DATA_RATE);
        adc2Ready = true;
        #if SERIAL_DEBUG
        Serial.println(F("ADS1115 #2 (0x49) OK"));
        #endif
    } else {
        adc2Ready = false;
        #if SERIAL_DEBUG
        Serial.println(F("WARNING: ADS1115 #2 (0x49) not found"));
        #endif
    }
    #endif

    if (!anyAdcReady()) {
        #if SERIAL_DEBUG
        Serial.println(F("ERROR: No ADCs found! GPS/diagnostic mode only."));
        #endif
    }
}

void setupGPS() {
    gpsSerial.begin(GPS_BAUD);

    #if SERIAL_DEBUG
    Serial.print(F("GPS initialized at "));
    Serial.print(GPS_BAUD);
    Serial.println(F(" baud"));
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

void writeCSVHeader() {
    // Build header dynamically based on NUM_SENSOR_PAIRS and GPS_LOG_QUALITY
    logFile.print(F("timestamp,lat,lon"));

    #if GPS_LOG_QUALITY
    logFile.print(F(",fix_quality,hdop,altitude"));
    #endif

    for (uint8_t i = 0; i < NUM_SENSOR_PAIRS; i++) {
        logFile.print(F(",g"));
        logFile.print(i + 1);
        logFile.print(F("_top,g"));
        logFile.print(i + 1);
        logFile.print(F("_bot,g"));
        logFile.print(i + 1);
        logFile.print(F("_grad"));
    }
    logFile.println();
}

void createLogFile() {
    if (!sdCardReady) return;

    for (uint16_t i = 1; i < 10000; i++) {
        snprintf(logFileName, sizeof(logFileName), "%s%04d%s",
                 LOG_FILE_PREFIX, i, LOG_FILE_EXTENSION);

        if (!sd.exists(logFileName)) {
            if (logFile.open(logFileName, O_CREAT | O_WRITE | O_EXCL)) {
                // Firmware version comment
                logFile.print(F("# Pathfinder v" FIRMWARE_VERSION
                                " (" FIRMWARE_DATE ") pairs="));
                logFile.println(NUM_SENSOR_PAIRS);
                // CSV header
                writeCSVHeader();
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

void checkSaturation(int16_t value) {
    if (value > ADC_SATURATION_THRESHOLD || value < -ADC_SATURATION_THRESHOLD) {
        adcSaturationCount++;
        #if SERIAL_DEBUG
        if (adcSaturationCount % 10 == 1) {
            Serial.print(F("WARNING: ADC saturation (count="));
            Serial.print(adcSaturationCount);
            Serial.println(')');
        }
        #endif
    }
}

void readGradiometers(GradiometerReading &reading) {
    reading.timestamp_ms = millis();

    for (uint8_t i = 0; i < NUM_SENSOR_PAIRS; i++) {
        // Determine which ADC and whether it's ready
        bool adcOk;
        Adafruit_ADS1115 *adc;

        if (PAIR_ADC[i] == 1) {
            adc = &ads1;
            adcOk = adc1Ready;
        } else {
            #if NEEDS_ADC2
            adc = &ads2;
            adcOk = adc2Ready;
            #else
            adcOk = false;
            adc = &ads1; // unused, but avoids uninitialized warning
            #endif
        }

        if (adcOk) {
            reading.top[i]  = adc->readADC_SingleEnded(PAIR_TOP_CH[i]);
            reading.bot[i]  = adc->readADC_SingleEnded(PAIR_BOT_CH[i]);
            reading.grad[i] = reading.bot[i] - reading.top[i];
            checkSaturation(reading.top[i]);
            checkSaturation(reading.bot[i]);
        } else {
            reading.top[i] = 0;
            reading.bot[i] = 0;
            reading.grad[i] = 0;
        }
    }
}

void reopenLogFile() {
    logFile.close();
    if (logFile.open(logFileName, O_WRITE | O_AT_END)) {
        sdCardReady = true;
        sdWriteErrors = 0;
        #if SERIAL_DEBUG
        Serial.println(F("SD card: file re-opened"));
        #endif
    } else {
        sdCardReady = false;
        #if SERIAL_DEBUG
        Serial.println(F("SD card: re-open failed"));
        #endif
    }
}

void logReading(const GradiometerReading &reading) {
    if (!sdCardReady) return;

    logFile.print(reading.timestamp_ms);
    logFile.print(',');
    logFile.print(reading.latitude, 7);
    logFile.print(',');
    logFile.print(reading.longitude, 7);

    #if GPS_LOG_QUALITY
    logFile.print(',');
    logFile.print(gps.location.isValid() ? 1 : 0);
    logFile.print(',');
    logFile.print(gps.hdop.isValid() ? gps.hdop.hdop() : 99.9, 1);
    logFile.print(',');
    logFile.print(gps.altitude.isValid() ? gps.altitude.meters() : 0.0, 1);
    #endif

    for (uint8_t i = 0; i < NUM_SENSOR_PAIRS; i++) {
        logFile.print(',');
        logFile.print(reading.top[i]);
        logFile.print(',');
        logFile.print(reading.bot[i]);
        logFile.print(',');
        logFile.print(reading.grad[i]);
    }

    if (!logFile.println()) {
        sdWriteErrors++;
        #if SERIAL_DEBUG
        Serial.print(F("SD write error #"));
        Serial.println(sdWriteErrors);
        #endif
        if (sdWriteErrors >= SD_RETRY_THRESHOLD) {
            reopenLogFile();
        }
    } else {
        if (sdWriteErrors > 0) sdWriteErrors = 0;
    }
}

// ============================================================================
// USER INTERFACE FUNCTIONS
// ============================================================================

void updateBeeper() {
    unsigned long currentTime = millis();

    if (beeping) {
        if (currentTime - beepStartTime >= BEEP_DURATION_MS) {
            digitalWrite(BEEPER_PIN, LOW);
            beeping = false;
        }
        return;
    }

    if (currentTime - lastBeepTime >= BEEP_INTERVAL_MS) {
        lastBeepTime = currentTime;
        beepStartTime = currentTime;
        digitalWrite(BEEPER_PIN, HIGH);
        beeping = true;
    }
}

void updateStatusLED() {
    unsigned long currentTime = millis();
    unsigned long blinkInterval;

    if (!anyAdcReady()) {
        blinkInterval = BLINK_ERROR / 2;
    } else if (!sdCardReady) {
        blinkInterval = BLINK_ERROR;
    } else if (!gpsLocked) {
        blinkInterval = BLINK_NO_GPS;
    } else {
        blinkInterval = BLINK_LOGGING;
    }

    if (currentTime - lastBlinkTime >= blinkInterval) {
        lastBlinkTime = currentTime;
        ledState = !ledState;
        digitalWrite(STATUS_LED_PIN, ledState);
    }
}

void printDebugInfo(const GradiometerReading &reading) {
    #if SERIAL_DEBUG
    Serial.print(F("S:"));
    Serial.print(sampleCount);
    Serial.print(F(" GPS:"));
    if (gpsLocked) {
        Serial.print(reading.latitude, 6);
        Serial.print(',');
        Serial.print(reading.longitude, 6);
    } else {
        Serial.print(F("--"));
    }

    for (uint8_t i = 0; i < NUM_SENSOR_PAIRS; i++) {
        Serial.print(F(" G"));
        Serial.print(i + 1);
        Serial.print(':');
        Serial.print(reading.grad[i]);
    }

    if (sdWriteErrors > 0) {
        Serial.print(F(" SD_ERR:"));
        Serial.print(sdWriteErrors);
    }
    if (adcSaturationCount > 0) {
        Serial.print(F(" SAT:"));
        Serial.print(adcSaturationCount);
    }
    Serial.println();
    #endif
}
