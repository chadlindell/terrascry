/**
 * TERRASCRY SD Card CSV Logger
 *
 * Common SD card logging for both Pathfinder and HIRT.
 * Writes CSV files matching the schemas defined in
 * shared/protocols/csv_schemas.md.
 *
 * Features:
 * - Auto-incrementing filenames (SURV0001.CSV, SURV0002.CSV, ...)
 * - Atomic writes with flush after each row
 * - Header row written on file creation
 * - Configurable flush interval
 */

#ifndef TERRASCRY_SD_LOGGER_H
#define TERRASCRY_SD_LOGGER_H

#include <SD.h>
#include <stdint.h>
#include <stdbool.h>

// SD card CS pin (ESP32)
#define SD_CS_PIN  5

// Maximum filename length
#define SD_MAX_FILENAME  32

// Flush interval (rows between forced flush)
#define SD_FLUSH_INTERVAL  10

typedef struct {
    File file;
    char filename[SD_MAX_FILENAME];
    uint32_t rows_written;
    uint32_t rows_since_flush;
    bool is_open;
} sd_logger_t;

/**
 * Initialize SD card and create a new survey file.
 * Filename format: {prefix}NNNN.CSV where NNNN auto-increments.
 * Writes CSV header row.
 * Returns true on success.
 */
bool sd_logger_init(sd_logger_t *logger, const char *prefix, const char *header);

/**
 * Write a pre-formatted CSV row. Caller is responsible for
 * formatting the row string (use snprintf, not String concatenation).
 * Auto-flushes every SD_FLUSH_INTERVAL rows.
 */
bool sd_logger_write_row(sd_logger_t *logger, const char *row);

/**
 * Force flush buffered data to SD card.
 */
void sd_logger_flush(sd_logger_t *logger);

/**
 * Close the current file.
 */
void sd_logger_close(sd_logger_t *logger);

#endif // TERRASCRY_SD_LOGGER_H
