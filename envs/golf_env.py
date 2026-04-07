"""
Golf Swing MuJoCo Environment — Phase 2 (geometry corrected)

Swing geometry:
  MuJoCo Y-hinge: θ>0 puts arm on -X side (backswing)
  Negative torque → θ decreases → arm swings down → club moves +X at impact
  Base is directly above ball. Hole is in +X direction.

Features:
  - Aerodynamic drag on the ball (quadratic, no lift/spin)
  - Terrain zone detection (fairway, green, sand, rough/OOB)
  - Rolling deceleration (surface-dependent braking force)
  - Hole completion: ball stops when it enters hole area
"""

import numpy as np
import mujoco
from pathlib import Path

# ── Aerodynamic constants ──────────────────────────────────────────
AIR_DENSITY = 1.225
BALL_RADIUS = 0.0214
BALL_AREA = np.pi * BALL_RADIUS ** 2
CD_GOLF_BALL = 0.25
DRAG_K = 0.5 * AIR_DENSITY * CD_GOLF_BALL * BALL_AREA

# ── Terrain types and rolling deceleration ─────────────────────────
TERRAIN_FAIRWAY = "fairway"
TERRAIN_GREEN   = "green"
TERRAIN_SAND    = "sand"
TERRAIN_ROUGH   = "rough"
TERRAIN_TEE     = "tee"

ROLLING_DECEL = {
    TERRAIN_FAIRWAY: 1.5,
    TERRAIN_GREEN:   0.8,
    TERRAIN_SAND:    8.0,
    TERRAIN_ROUGH:   3.5,
    TERRAIN_TEE:     1.5,
}

# ── Hole dimensions ────────────────────────────────────────────────
HOLE_RADIUS = 0.054   # standard golf hole radius (108mm diameter)


class TerrainMap:
    """
    Axis-aligned zone map. First matching zone wins; default is rough.
    """
    DEFAULT_ZONES = [
        (-1.0,   2.0,  -2.0,  2.0, TERRAIN_TEE),
        (35.0,  40.0,  -2.0,  2.0, TERRAIN_SAND),
        (40.0,  55.0,  -5.0,  5.0, TERRAIN_GREEN),
        ( 2.0,  50.0,  -5.0,  5.0, TERRAIN_FAIRWAY),
    ]

    def __init__(self, zones=None):
        self.zones = zones if zones is not None else self.DEFAULT_ZONES

    def classify(self, x, y):
        for x_min, x_max, y_min, y_max, ttype in self.zones:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return ttype
        return TERRAIN_ROUGH

    def deceleration(self, x, y):
        return ROLLING_DECEL[self.classify(x, y)]


class GolfSwingEnv:
    MODEL_PATH = Path(__file__).parent.parent / "models" / "arm_club_ball.xml"

    SENS_ELBOW_POS = 0
    SENS_ELBOW_VEL = 1
    SENS_WRIST_POS = 2
    SENS_WRIST_VEL = 3
    SENS_CLUB_POS = slice(4, 7)
    SENS_CLUB_QUAT = slice(7, 11)
    SENS_CLUB_VEL = slice(11, 14)
    SENS_BALL_POS = slice(14, 17)
    SENS_BALL_VEL = slice(17, 20)

    GROUND_Z_THRESH = 0.05

    def __init__(self, ctrl_dt=0.005, max_time=10.0, hole_pos=None,
                 terrain_zones=None, enable_drag=True, enable_rolling_decel=True):
        self.model = mujoco.MjModel.from_xml_path(str(self.MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        self.sim_dt = self.model.opt.timestep
        self.ctrl_dt = ctrl_dt
        self.ctrl_substeps = int(round(ctrl_dt / self.sim_dt))
        self.max_steps = int(max_time / ctrl_dt)

        self.enable_drag = enable_drag
        self.enable_rolling_decel = enable_rolling_decel

        if hole_pos is not None:
            self.hole_pos = np.array(hole_pos, dtype=np.float64)
        else:
            hole_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hole")
            self.hole_pos = self.model.body_pos[hole_body_id][:2].copy()

        self.terrain = TerrainMap(terrain_zones)

        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.club_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "club_head")

        self.elbow_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_torque")
        self.wrist_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wrist_torque")

        # Ball free joint qvel indices (freejoint = 6 DOF: 3 translational + 3 rotational)
        ball_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qvel_start = self.model.jnt_dofadr[ball_jnt_id]

        self.step_count = 0
        self.ball_launched = False
        self.ball_launch_vel = None
        self.contact_detected = False
        self.ball_in_hole = False

    def reset(self, elbow_init=None, wrist_init=None):
        mujoco.mj_resetData(self.model, self.data)

        if elbow_init is None:
            elbow_init = 3 * np.pi / 4
        if wrist_init is None:
            wrist_init = 0.0

        elbow_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")
        wrist_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wrist")
        self.data.qpos[elbow_jnt] = elbow_init
        self.data.qpos[wrist_jnt] = wrist_init

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.ball_launched = False
        self.ball_launch_vel = None
        self.contact_detected = False
        self.ball_in_hole = False

        return self.get_obs()

    def _freeze_ball(self):
        """Zero out ball velocity to freeze it in place (when it enters the hole)."""
        qv = self.ball_qvel_start
        self.data.qvel[qv:qv+6] = 0.0
        self.data.xfrc_applied[self.ball_body_id, :] = 0.0

    def _check_hole(self):
        """
        Check if ball is within the hole area.
        Counts as holed if the ball's XY position is within HOLE_RADIUS
        of the hole center AND the ball is at ground level.
        """
        if self.ball_in_hole:
            return True

        ball_pos = self.data.xpos[self.ball_body_id]
        ball_xy = ball_pos[:2]
        dist = np.linalg.norm(ball_xy - self.hole_pos)

        if dist < HOLE_RADIUS and ball_pos[2] < self.GROUND_Z_THRESH:
            self.ball_in_hole = True
            self._freeze_ball()
            return True

        return False

    def _apply_ball_forces(self):
        # If ball is in hole, keep it frozen
        if self.ball_in_hole:
            self._freeze_ball()
            return

        ball_pos = self.data.xpos[self.ball_body_id]
        ball_vel = self.data.cvel[self.ball_body_id][3:]

        speed = np.linalg.norm(ball_vel)
        if speed < 1e-6:
            return

        vel_dir = ball_vel / speed

        if self.enable_drag:
            drag_force = -DRAG_K * speed ** 2 * vel_dir
            self.data.xfrc_applied[self.ball_body_id, :3] = drag_force
        else:
            self.data.xfrc_applied[self.ball_body_id, :3] = 0.0

        if self.enable_rolling_decel and ball_pos[2] < self.GROUND_Z_THRESH:
            horiz_vel = ball_vel[:2].copy()
            horiz_speed = np.linalg.norm(horiz_vel)
            if horiz_speed > 0.01:
                horiz_dir = horiz_vel / horiz_speed
                decel = self.terrain.deceleration(ball_pos[0], ball_pos[1])
                ball_mass = self.model.body_mass[self.ball_body_id]
                brake_force = -decel * ball_mass * np.append(horiz_dir, 0.0)
                self.data.xfrc_applied[self.ball_body_id, :3] += brake_force[:3]

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.data.ctrl[self.elbow_act] = action[0]
        self.data.ctrl[self.wrist_act] = action[1]

        contact_this_step = False
        for _ in range(self.ctrl_substeps):
            self._apply_ball_forces()
            self._check_hole()
            mujoco.mj_step(self.model, self.data)

            if not self.contact_detected:
                for i in range(self.data.ncon):
                    c = self.data.contact[i]
                    geoms = {c.geom1, c.geom2}
                    if self.ball_geom_id in geoms and self.club_geom_id in geoms:
                        self.contact_detected = True
                        contact_this_step = True
                        break

        self.step_count += 1
        obs = self.get_obs()

        ball_vel = obs["ball_vel"]
        ball_speed = np.linalg.norm(ball_vel)
        if self.contact_detected and not self.ball_launched and ball_speed > 0.5:
            self.ball_launched = True
            self.ball_launch_vel = ball_vel.copy()

        done = self.step_count >= self.max_steps
        ball_xy = obs["ball_pos"][:2]
        dist_to_hole = np.linalg.norm(ball_xy - self.hole_pos)

        # Ball stopped naturally (not in hole)
        ball_stopped = self.ball_launched and not self.ball_in_hole and ball_speed < 0.01
        if ball_stopped:
            done = True

        # Ball in hole = immediate completion
        if self.ball_in_hole:
            done = True

        terrain_type = self.terrain.classify(ball_xy[0], ball_xy[1])

        info = {
            "time": self.data.time,
            "contact": contact_this_step,
            "ball_launched": self.ball_launched,
            "ball_launch_vel": self.ball_launch_vel,
            "ball_speed": ball_speed,
            "dist_to_hole": dist_to_hole,
            "ball_stopped": ball_stopped,
            "in_hole": self.ball_in_hole,
            "terrain": terrain_type,
            "done": done,
        }

        return obs, info

    def get_obs(self):
        s = self.data.sensordata
        return {
            "elbow_pos": s[self.SENS_ELBOW_POS],
            "elbow_vel": s[self.SENS_ELBOW_VEL],
            "wrist_pos": s[self.SENS_WRIST_POS],
            "wrist_vel": s[self.SENS_WRIST_VEL],
            "club_head_pos": s[self.SENS_CLUB_POS].copy(),
            "club_head_quat": s[self.SENS_CLUB_QUAT].copy(),
            "club_head_vel": s[self.SENS_CLUB_VEL].copy(),
            "ball_pos": s[self.SENS_BALL_POS].copy(),
            "ball_vel": s[self.SENS_BALL_VEL].copy(),
        }

    def get_flat_obs(self):
        obs = self.get_obs()
        return np.concatenate([
            [obs["elbow_pos"], obs["elbow_vel"]],
            [obs["wrist_pos"], obs["wrist_vel"]],
            obs["club_head_pos"],
            obs["club_head_vel"],
            obs["ball_pos"],
            obs["ball_vel"],
        ])

    @property
    def obs_dim(self):
        return 14

    @property
    def act_dim(self):
        return 2