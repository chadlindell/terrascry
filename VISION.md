# GeoSim Vision

## Purpose

Physics-realistic simulation engine for HIRT and Pathfinder geophysical instruments. The physics engine **is** the product — visualization layers are consumers, not the core.

## Goals

1. **Validate instrument designs** against known scenarios with analytically solvable physics
2. **Generate synthetic sensor data** that exactly matches real instrument output formats
3. **Train operators** with interactive simulations before expensive field deployments
4. **Demonstrate the technology** in classrooms, presentations, and publications

## Architecture Principle

**Scenario files are the single source of truth.** A JSON file defines what's buried where. Every frontend (notebooks, Godot, web) and the physics engine itself reads from the same source. No physics lives in visualization code.

## Non-Goals

- This is not a game engine. Physics accuracy is non-negotiable.
- This is not a replacement for SimPEG or pyGIMLi. It wraps them.
- This is not a data processing pipeline. It generates data, not interprets it.

## Success Criteria

Phase 1: Simulated Pathfinder CSV → fed into existing `visualize_data.py` → produces correct anomaly map with targets at right positions and amplitudes matching published detection depths.

Phase 2: SimPEG/pyGIMLi forward models reproduce HIRT's 4 documented validation scenarios.
