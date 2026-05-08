"""
Course 2: "Gentle Dogleg"

39m dogleg course with hill, bunkers, crosswind, and
landed-terrain penalties. Optimizes 3 variables.
"""

import numpy as np
from pathlib import Path

HOLE_POS = np.array([38.0, 8.0])
MODEL_PATH = Path(__file__).parent.parent / "models" / "course2.xml"

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

STRAIGHT_LINE_DIST = np.linalg.norm(HOLE_POS - np.array([0.05, 0.0]))
AIM_ANGLE_TO_HOLE = np.arctan2(HOLE_POS[1], HOLE_POS[0])
HOLE_RADIUS = 0.054