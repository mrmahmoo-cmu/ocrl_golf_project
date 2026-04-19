"""
Course 1: "Gentle Dogleg"

A ~39m course that bends slightly to the right with a hill
on the fairway and sand bunkers. The optimizer must choose
aim direction and swing strength to reach the offset hole.

Features:
  - Rounded hill on the fairway
  - Sand bunker blocking the straight path
  - Greenside sand bunker
  - Dogleg bends right toward hole
  - Mixed terrain: tee, fairway, sand, green, rough

Distance: ~39m (straight line from tee to hole)
Hole:     (38, 8)
Goal:     Hole in one
"""

import numpy as np
from pathlib import Path

# ── Course geometry ────────────────────────────────────────────────
HOLE_POS = np.array([38.0, 8.0])
BALL_START = np.array([0.05, 0.0, 0.0214])

# Path to the course-specific MuJoCo model
MODEL_PATH = Path(__file__).parent.parent / "models" / "course1.xml"

# ── Terrain zones ──────────────────────────────────────────────────
# Order matters: first match wins.
TERRAIN_ZONES = [
    (-2.0,   3.0,  -2.5,  2.5, "tee"),
    (23.0,  28.0,  -1.0,  3.0, "sand"),
    (33.0,  37.0,   9.0, 12.0, "sand"),
    (32.0,  44.0,   3.0, 13.0, "green"),
    ( 3.0,  25.0,  -4.5,  4.5, "fairway"),
    (25.0,  33.0,  -2.0,  9.0, "fairway"),
    (33.0,  38.0,   0.0,  8.0, "fairway"),
]

# ── Derived quantities ─────────────────────────────────────────────
STRAIGHT_LINE_DIST = np.linalg.norm(HOLE_POS - BALL_START[:2])
AIM_ANGLE_TO_HOLE = np.arctan2(HOLE_POS[1], HOLE_POS[0])
HOLE_RADIUS = 0.1