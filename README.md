# Golf Swing Optimization — MuJoCo Environment

## Project Structure

```
golf_project/
├── models/
│   └── arm_club_ball.xml      # 2-DOF arm + club + ball + ground
├── envs/
│   └── golf_env.py            # GolfSwingEnv wrapper (obs/action interface)
├── scripts/
│   └── test_basic_swing.py    # Verification: open-loop swing → contact → launch
├── courses/                   # (Phase 3?)
└── README.md
```

## Quick Start

```bash
pip install mujoco gymnasium
cd golf_project
python scripts/test_basic_swing.py
```

Expected output: contact at ~0.17s, ball launch at ~22 m/s, all checks will pass

## Model Details

**Arm (2-DOF, swing plane = XZ):**
- Base fixed at (0.52, 0, 1.4)
- Elbow hinge (Y-axis): ±180°, torque limit ±150 Nm
- Wrist hinge (Y-axis): ±150°, torque limit ±60 Nm
- Segments: upper arm 0.35m, forearm 0.30m, club shaft 0.80m, club head (box)

**Ball:**
- Radius 0.0214m (standard golf ball), mass 0.0459 kg
- Free joint (6-DOF), initial position (0.15, 0, 0.0214)

**Contact tuning:**
- Club-ball pair: solref=[0.0005, 1.0], solimp=[0.95, 0.99, 0.001]
- Very stiff, short-duration impact to try to produce a realistic impulse

**Coordinate convention:**
- +X = toward hole, +Z = up

## Phase Roadmap

### Phase 1 — Arm + Swing
- MJCF model with 2-DOF arm, club, ball
- Contact detection and ball launch
- Environment wrapper with clean API
- Open-loop test demonstrating full pipeline

### Phase 2 — Ball Flight + Terrain
- Air resistance (quadratic drag on ball)
- Terrain heightfields with friction zones (sand, fairway, rough, OOB)
- Ground deceleration model (surface-dependent)

### Phase 3 — Courses + Optimization?
