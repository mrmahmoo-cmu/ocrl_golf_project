"""
Course 2: "Gentle Dogleg"

A ~39m course with a dogleg right, hill, sand bunkers, crosswind,
and landed-terrain penalties. More challenging than Course 1.

Features:
  - Rounded hill on the fairway
  - Sand bunker blocking the straight path
  - Greenside sand bunker
  - Dogleg bends right toward hole
  - Crosswind: 1.5 m/s from -Y (pushes ball toward +Y)
  - Landed-terrain penalties discourage rough/sand landings

Distance: ~39m
Hole:     (38, 8)
Goal:     Hole in one
"""

import numpy as np
from pathlib import Path

HOLE_POS = np.array([38.0, 8.0])
BALL_START = np.array([0.05, 0.0, 0.0214])
MODEL_PATH = Path(__file__).parent.parent / "models" / "course2.xml"

# Crosswind: 1.5 m/s pushing toward +Y
WIND = np.array([0.0, 1.5, 0.0])

TERRAIN_ZONES = [
    (-2.0,   3.0,  -2.5,  2.5, "tee"),
    (23.0,  28.0,  -1.0,  3.0, "sand"),
    (33.0,  37.0,   9.0, 12.0, "sand"),
    (32.0,  44.0,   3.0, 13.0, "green"),
    ( 3.0,  25.0,  -4.5,  4.5, "fairway"),
    (25.0,  33.0,  -2.0,  9.0, "fairway"),
    (33.0,  38.0,   0.0,  8.0, "fairway"),
]

STRAIGHT_LINE_DIST = np.linalg.norm(HOLE_POS - BALL_START[:2])
AIM_ANGLE_TO_HOLE = np.arctan2(HOLE_POS[1], HOLE_POS[0])
HOLE_RADIUS = 0.1