#!/usr/bin/env bash
# Integration test: verify Godot can connect to GeoSim physics server.
#
# Usage:
#   ./tools/test_godot_integration.sh [scenario_path]
#
# This script:
#   1. Starts the GeoSim ZeroMQ server with a scenario
#   2. Sends test queries via the Python test client
#   3. Shuts down the server
#   4. Reports pass/fail

set -euo pipefail

SCENARIO="${1:-scenarios/single-ferrous-target.json}"
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[test] Shutting down server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== GeoSim Godot Integration Test ==="
echo "[test] Scenario: $SCENARIO"

# 1. Start the server
echo "[test] Starting GeoSim server..."
geosim-server --scenario "$SCENARIO" --verbose &
SERVER_PID=$!
sleep 1

# Check server is running
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[FAIL] Server failed to start"
    exit 1
fi
echo "[test] Server running (PID $SERVER_PID)"

# 2. Run test queries via Python
echo "[test] Running test queries..."
python3 -c "
import zmq, json, sys, time

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect('tcp://127.0.0.1:5555')
sock.setsockopt(zmq.RCVTIMEO, 5000)

tests_passed = 0
tests_failed = 0

def test(name, command, params=None):
    global tests_passed, tests_failed
    try:
        req = {'command': command}
        if params:
            req['params'] = params
        sock.send_json(req)
        resp = sock.recv_json()
        if resp.get('status') == 'ok':
            print(f'  PASS: {name}')
            tests_passed += 1
        else:
            print(f'  FAIL: {name} - {resp.get(\"message\", \"unknown\")}')
            tests_failed += 1
    except Exception as e:
        print(f'  FAIL: {name} - {e}')
        tests_failed += 1

# Test ping
test('ping', 'ping')

# Test scenario info
test('get_scenario_info', 'get_scenario_info')

# Test single-point field query
test('query_field (single)', 'query_field', {'positions': [[10.0, 10.0, 0.3]]})

# Test single-point gradient query
test('query_gradient (single)', 'query_gradient', {
    'positions': [[10.0, 10.0, 0.175]],
    'sensor_separation': 0.35,
    'component': 2,
})

# Test batch query (simulates Godot querying at walk speed)
positions = [[x, 10.0, 0.175] for x in range(5, 16)]
test('query_field (batch 11 pts)', 'query_field', {'positions': positions})
test('query_gradient (batch 11 pts)', 'query_gradient', {
    'positions': positions,
    'sensor_separation': 0.35,
    'component': 2,
})

# Shutdown
test('shutdown', 'shutdown')

print(f'\nResults: {tests_passed} passed, {tests_failed} failed')
sock.close()
ctx.term()
sys.exit(1 if tests_failed > 0 else 0)
"

RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo "[PASS] All integration tests passed"
else
    echo "[FAIL] Some integration tests failed"
fi

exit $RESULT
