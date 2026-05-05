"""
Course 3: "White Dogwood"

Long multi-stroke course inspired by Augusta National Hole 11.
Features a left-to-right fairway shape, approach hazards, pond near
the green, bunkers, and a longer distance requiring multiple shots.
"""

import numpy as np
from pathlib import Path

HOLE_POS = np.array([116.0, 16.0])
MODEL_PATH = Path(__file__).parent.parent / "models" / "course3.xml"

# Start with no wind. Add wind later after the course works.
WIND = np.array([0.0, 0.0, 0.0])

TERRAIN_ZONES = [
    # Tee
    (-2.0,    3.0,  -2.5,   2.5, "tee"),

    # Water / pond near green
    # Put water before green/fairway so it gets detected first.
    (101.0, 121.0,   4.2,  13.8, "water"),

    # Sand bunkers
    (63.8,   70.2,  -4.3,  -0.7, "sand"),
    (118.0, 124.0,  18.7,  22.3, "sand"),

    # Green
    (111.0, 121.0,  12.0,  20.0, "green"),

    # Fairway sections
    (2.0,    42.0,  -4.2,   4.2, "fairway"),
    (34.0,   70.0,  -1.3,   8.3, "fairway"),
    (62.0,   94.0,   2.3,  12.7, "fairway"),
    (88.0,  114.0,   8.0,  16.0, "fairway"),
]

STRAIGHT_LINE_DIST = np.linalg.norm(HOLE_POS - np.array([0.05, 0.0]))
AIM_ANGLE_TO_HOLE = np.arctan2(HOLE_POS[1], HOLE_POS[0])
HOLE_RADIUS = 0.1