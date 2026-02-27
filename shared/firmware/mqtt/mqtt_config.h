/**
 * TERRASCRY MQTT Configuration
 *
 * Shared MQTT topic definitions and client configuration
 * for ESP32 firmware. Uses esp-mqtt library (NOT PubSubClient)
 * for deterministic timing compatibility with TDM.
 *
 * Message serialization: snprintf (NOT ArduinoJson) for
 * deterministic memory usage at 10 Hz.
 */

#ifndef TERRASCRY_MQTT_CONFIG_H
#define TERRASCRY_MQTT_CONFIG_H

// MQTT broker defaults (Jetson mDNS)
#define TERRASCRY_MQTT_BROKER     "terrascry-jetson.local"
#define TERRASCRY_MQTT_PORT       1883
#define TERRASCRY_MQTT_KEEPALIVE  60

// Topic prefix
#define TERRASCRY_TOPIC_PREFIX    "terrascry"

// --- Pathfinder topics ---
#define TOPIC_PF_RAW              TERRASCRY_TOPIC_PREFIX "/pathfinder/data/raw"
#define TOPIC_PF_CORRECTED        TERRASCRY_TOPIC_PREFIX "/pathfinder/data/corrected"
#define TOPIC_PF_EMI              TERRASCRY_TOPIC_PREFIX "/pathfinder/data/emi"
#define TOPIC_PF_THERMAL          TERRASCRY_TOPIC_PREFIX "/pathfinder/data/thermal"
#define TOPIC_PF_ANOMALY          TERRASCRY_TOPIC_PREFIX "/pathfinder/anomaly/detected"
#define TOPIC_PF_STATUS           TERRASCRY_TOPIC_PREFIX "/status/pathfinder"

// --- HIRT topics ---
#define TOPIC_HIRT_MIT_RAW        TERRASCRY_TOPIC_PREFIX "/hirt/data/mit/raw"
#define TOPIC_HIRT_ERT_RAW        TERRASCRY_TOPIC_PREFIX "/hirt/data/ert/raw"
#define TOPIC_HIRT_MODEL          TERRASCRY_TOPIC_PREFIX "/hirt/model/update"
#define TOPIC_HIRT_PROBE_ORIENT   TERRASCRY_TOPIC_PREFIX "/hirt/probe/orientation"
#define TOPIC_HIRT_PROBE_POS      TERRASCRY_TOPIC_PREFIX "/hirt/probe/position"
#define TOPIC_HIRT_STATUS         TERRASCRY_TOPIC_PREFIX "/status/hirt"

// QoS levels
#define QOS_RAW_DATA   0  // At most once — high rate, loss tolerable
#define QOS_CORRECTED  1  // At least once — logged to disk
#define QOS_ANOMALY    2  // Exactly once — critical notification
#define QOS_STATUS     1  // At least once

// Ring buffer size (TDM task → MQTT publisher task)
// 100 messages = 10 seconds at 10 Hz
#define MQTT_RING_BUFFER_SIZE  100

// Maximum JSON message size (bytes)
// Pathfinder raw message is ~250 bytes
#define MQTT_MAX_MESSAGE_SIZE  512

#endif // TERRASCRY_MQTT_CONFIG_H
