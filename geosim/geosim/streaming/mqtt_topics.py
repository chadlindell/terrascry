"""
MQTT topic definitions for the GeoSim instrument streaming protocol.

Topic hierarchy:
    geosim/{instrument}/data/raw          - Raw sensor readings
    geosim/{instrument}/data/corrected    - After tilt/orientation correction
    geosim/{instrument}/anomaly/detected  - Real-time anomaly flags
    geosim/{instrument}/model/update      - Latest model output
    geosim/{instrument}/probe/orientation - IMU/inclinometer readings
    geosim/status/{instrument}            - System health
"""

# Topic templates
TOPIC_PREFIX = "geosim"

# Pathfinder topics
PATHFINDER_RAW = f"{TOPIC_PREFIX}/pathfinder/data/raw"
PATHFINDER_CORRECTED = f"{TOPIC_PREFIX}/pathfinder/data/corrected"
PATHFINDER_ANOMALY = f"{TOPIC_PREFIX}/pathfinder/anomaly/detected"
PATHFINDER_STATUS = f"{TOPIC_PREFIX}/status/pathfinder"

# HIRT topics
HIRT_MIT_RAW = f"{TOPIC_PREFIX}/hirt/data/mit/raw"
HIRT_ERT_RAW = f"{TOPIC_PREFIX}/hirt/data/ert/raw"
HIRT_MODEL_UPDATE = f"{TOPIC_PREFIX}/hirt/model/update"
HIRT_PROBE_ORIENTATION = f"{TOPIC_PREFIX}/hirt/probe/orientation"
HIRT_STATUS = f"{TOPIC_PREFIX}/status/hirt"

# System-wide
SYSTEM_STATUS = f"{TOPIC_PREFIX}/status/+"
