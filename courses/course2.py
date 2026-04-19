"""
Course 2: "The Ridge"

A longer course (~48m) with a sharp double-bend dogleg,
two hills, three sand bunkers, and a crosswind.
More challenging than Course 1 but still achievable as a hole-in-one.

Features:
  - Two ellipsoid hills on the fairway
  - Three sand bunkers at strategic positions
  - Sharp dogleg right requiring ~23 deg aim
  - Crosswind: 2 m/s from -Y (pushes ball toward +Y)
  - Narrow approach corridor between hill and bunker

Distance: ~46m (straight line from tee to hole)
Hole:     (42, 18)
Goal:     Hole in one
"""

import numpy as np
from pathlib import Path

# ── Course geometry ────────────────────────────────────────────────
HOLE_POS = np.array([42.0, 18.0])
BALL_START = np.array([0.05, 0.0, 0.0214])

MODEL_PATH = Path(__file__).parent.parent / "models" / "course2.xml"

# ── Wind ───────────────────────────────────────────────────────────
# 2 m/s crosswind from -Y direction (pushes ball toward +Y)
# This helps shots aimed right but hurts shots aimed straight
WIND = np.array([0.0, 2.0, 0.0])

# ── Terrain zones ──────────────────────────────────────────────────
TERRAIN_ZONES = [
    # Tee box
    (-2.0,   3.0,  -2.5,  2.5, "tee"),

    # Sand bunkers (checked before fairway)
    (25.5,  30.5,  -3.0,  1.0, "sand"),     # bunker 1: end of straight
    (24.0,  28.0,   3.5,  8.5, "sand"),     # bunker 2: inside dogleg
    (37.0,  41.0,  14.5, 17.5, "sand"),     # bunker 3: greenside

    # Green
    (37.0,  47.0,  13.5, 22.5, "green"),

    # Fairway sections
    ( 3.0,  28.0,  -4.5,  4.5, "fairway"),   # straight section
    (25.0,  35.0,  -0.5, 12.5, "fairway"),   # first bend
    (32.5,  40.5,   9.0, 19.0, "fairway"),   # second bend / approach
]

# ── Derived quantities ─────────────────────────────────────────────
STRAIGHT_LINE_DIST = np.linalg.norm(HOLE_POS - BALL_START[:2])
AIM_ANGLE_TO_HOLE = np.arctan2(HOLE_POS[1], HOLE_POS[0])
HOLE_RADIUS = 0.054
