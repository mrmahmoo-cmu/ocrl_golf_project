"""
Course 1: "Straight Shot"

Simple 35m straight course, one sand bunker. No wind.
Optimizes 3 variables: launch speed, vertical angle, horizontal angle.
"""

import numpy as np
from pathlib import Path

HOLE_POS = np.array([35.0, 0.0])
MODEL_PATH = Path(__file__).parent.parent / "models" / "course1.xml"

TERRAIN_ZONES = [
    (-2.0,   2.5,  -2.0,  2.0, "tee"),
    (30.0,  34.0,   1.0,  4.0, "sand"),
    (30.0,  40.0,  -4.0,  4.0, "green"),
    ( 2.0,  30.0,  -4.0,  4.0, "fairway"),
]

STRAIGHT_LINE_DIST = np.linalg.norm(HOLE_POS - np.array([0.05, 0.0]))
AIM_ANGLE_TO_HOLE = np.arctan2(HOLE_POS[1], HOLE_POS[0])
HOLE_RADIUS = 0.054