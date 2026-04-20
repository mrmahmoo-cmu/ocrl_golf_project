"""
Course 1: "Straight Shot"

A simple ~35m straight course with one sand bunker guarding
the approach. No dogleg, no wind, no terrain landing penalties.
The easiest course — demonstrates basic CMA-ES optimization.

Features:
  - Straight fairway from tee to green
  - One sand bunker near the hole
  - No wind
  - No terrain landing penalties

Distance: ~35m
Hole:     (35, 0)
Goal:     Hole in one
"""

import numpy as np
from pathlib import Path

HOLE_POS = np.array([35.0, 0.0])
BALL_START = np.array([0.05, 0.0, 0.0214])
MODEL_PATH = Path(__file__).parent.parent / "models" / "course1.xml"

TERRAIN_ZONES = [
    (-2.0,   2.5,  -2.0,  2.0, "tee"),
    (30.0,  34.0,   1.0,  4.0, "sand"),
    (30.0,  40.0,  -4.0,  4.0, "green"),
    ( 2.0,  30.0,  -4.0,  4.0, "fairway"),
]

STRAIGHT_LINE_DIST = np.linalg.norm(HOLE_POS - BALL_START[:2])
AIM_ANGLE_TO_HOLE = np.arctan2(HOLE_POS[1], HOLE_POS[0])
HOLE_RADIUS = 0.1