# Golf Swing Optimization — MuJoCo Environment

## Project Structure

```
golf_project/
├── models/
│   └── arm_club_ball.xml      # MJCF model: 2-DOF arm + club + ball + ground
├── envs/
│   └── golf_env.py            # GolfSwingEnv wrapper (obs/action interface)
├── scripts/
│   └── test_basic_swing.py    # Verification: open-loop swing → contact → launch
├── courses/                   # (Phase 3: terrain heightfields)
└── README.md
```

## Quick Start

```bash
pip install mujoco gymnasium
cd golf_project
python scripts/test_basic_swing.py
```

Expected output: contact at ~0.17s, ball launch at ~22 m/s, all checks PASS.

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
- Very stiff, short-duration impact to produce realistic impulse

**Coordinate convention:**
- +X = toward hole, +Z = up
- Joint θ=0 → arm hangs down, θ=+π/2 → arm horizontal backward (backswing start)

## Environment API

```python
from envs.golf_env import GolfSwingEnv

env = GolfSwingEnv(ctrl_dt=0.001, max_time=5.0, hole_pos=(5.0, 0.0))
obs = env.reset(elbow_init=np.pi/2, wrist_init=0.0)

# obs dict keys: elbow_pos, elbow_vel, wrist_pos, wrist_vel,
#                club_head_pos, club_head_vel, ball_pos, ball_vel

obs, info = env.step([elbow_torque, wrist_torque])
# info keys: time, contact, ball_launched, ball_launch_vel,
#            ball_speed, dist_to_hole, ball_stopped, done

flat = env.get_flat_obs()  # 14-dim vector for RL
```

## Phase Roadmap

### Phase 1 — Arm + Swing (DONE)
- [x] MJCF model with 2-DOF arm, club, ball
- [x] Contact detection and ball launch
- [x] Environment wrapper with clean API
- [x] Open-loop test demonstrating full pipeline

### Phase 2 — Ball Flight + Terrain
- [ ] Air resistance (quadratic drag on ball)
- [ ] Terrain heightfields with friction zones (sand, fairway, rough, OOB)
- [ ] Ground deceleration model (surface-dependent)

### Phase 3 — Courses + Optimization
- [ ] Course 1: short flat, aim for hole-in-one
- [ ] Course 2: short with slope, hole-in-one
- [ ] Course 3: longer, minimize strokes
- [ ] Trajectory optimization (direct collocation / shooting)
- [ ] RL baseline (PPO/SAC via Gymnasium wrapper)

## Known Limitations (current)
- No air resistance → ball travels unrealistically far
- No bounce/roll physics → ball slides after landing
- Flat ground only → no terrain variation yet
- Open-loop torque only → no optimization loop yet
