#!/usr/bin/env python3
"""Standalone ZeroMQ test client for the GeoSim physics server.

Usage:
    # Terminal 1: start the server
    geosim-server --scenario scenarios/scattered-debris.json -v

    # Terminal 2: run this client
    python tools/test_client.py

This exercises the full protocol and measures latency.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import zmq


def main():
    server_addr = sys.argv[1] if len(sys.argv) > 1 else "tcp://127.0.0.1:5555"

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(server_addr)

    print(f"Connected to {server_addr}\n")

    def send(command: str, params: dict | None = None) -> dict:
        msg = {"command": command}
        if params:
            msg["params"] = params
        t0 = time.perf_counter()
        sock.send_json(msg)
        resp = sock.recv_json()
        dt = (time.perf_counter() - t0) * 1000
        return resp, dt

    # 1. Ping
    resp, dt = send("ping")
    print(f"[{dt:.2f}ms] ping → {resp['data']['message']}")

    # 2. Load scenario (if not pre-loaded)
    resp, dt = send("get_scenario_info")
    if resp["status"] == "error":
        print("No scenario loaded. Loading scattered-debris.json...")
        resp, dt = send("load_scenario", {"path": "scenarios/scattered-debris.json"})
        print(f"[{dt:.2f}ms] load_scenario → {resp['data']}")
    else:
        print(f"[{dt:.2f}ms] Scenario: {resp['data']['name']}")

    # 3. Scenario info
    resp, dt = send("get_scenario_info")
    info = resp["data"]
    print(f"[{dt:.2f}ms] {info['name']}: {info['n_sources']} sources")
    print(f"  Terrain: x={info['terrain']['x_extent']}, y={info['terrain']['y_extent']}")

    # 4. Single-point field query
    resp, dt = send("query_field", {"positions": [[5.0, 5.0, 0.3]]})
    B = np.array(resp["data"]["B"][0])
    print(f"\n[{dt:.2f}ms] Field at (5,5,0.3): B = [{B[0]:.3e}, {B[1]:.3e}, {B[2]:.3e}] T")
    print(f"  |B| = {np.linalg.norm(B):.3e} T = {np.linalg.norm(B)*1e9:.2f} nT")

    # 5. Gradient query
    resp, dt = send("query_gradient", {
        "positions": [[5.0, 5.0, 0.175]],
        "sensor_separation": 0.35,
        "component": 2,
    })
    grad = resp["data"]["gradient"][0]
    print(f"[{dt:.2f}ms] Gradient at (5,5): {grad:.3e} T/m = {grad*1e9:.2f} nT/m")

    # 6. Grid scan — simulate what Godot would do
    print("\nGrid scan (50x50 = 2500 points)...")
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)
    positions = np.column_stack([X.ravel(), Y.ravel(), np.full(2500, 0.3)]).tolist()

    resp, dt = send("query_field", {"positions": positions})
    B = np.array(resp["data"]["B"])
    B_mag = np.linalg.norm(B, axis=1) * 1e9
    print(f"[{dt:.2f}ms] 2500-point grid: max |B| = {B_mag.max():.1f} nT")

    # 7. Latency benchmark
    print("\nLatency benchmark (100 single-point queries)...")
    times = []
    for _ in range(100):
        _, dt = send("query_field", {"positions": [[5.0, 5.0, 0.3]]})
        times.append(dt)

    times = np.array(times)
    print(f"  Median: {np.median(times):.3f} ms")
    print(f"  Mean:   {np.mean(times):.3f} ms")
    print(f"  P95:    {np.percentile(times, 95):.3f} ms")
    print(f"  P99:    {np.percentile(times, 99):.3f} ms")
    print(f"  Max:    {np.max(times):.3f} ms")

    print("\nAll tests passed!")
    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
