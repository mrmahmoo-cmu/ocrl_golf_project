"""
Golf Swing MuJoCo Environment — Phase 3

Supports:
  - Custom model paths (for course-specific XMLs)
  - Optional aim joint (turntable for aiming direction)
  - Aerodynamic drag with wind (relative velocity model)
  - Slope-aware rolling deceleration using contact normals
  - Terrain zone detection (fairway, green, sand, rough/OOB)
  - Hole completion (ball freezes when it enters hole area)
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
HOLE_RADIUS = 0.1


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
    DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "arm_club_ball.xml"

    def __init__(self, ctrl_dt=0.005, max_time=10.0, hole_pos=None,
                 terrain_zones=None, enable_drag=True, enable_rolling_decel=True,
                 model_path=None, wind=None):
        """
        Args:
            wind: Wind velocity as [wx, wy, wz] in m/s, or None for no wind.
                  Example: [3.0, 1.0, 0.0] = 3 m/s in +X, 1 m/s in +Y.
                  Drag is computed using ball velocity relative to the air,
                  so a tailwind reduces drag and a headwind increases it.
        """
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            self.model_path = self.DEFAULT_MODEL_PATH

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.sim_dt = self.model.opt.timestep
        self.ctrl_dt = ctrl_dt
        self.ctrl_substeps = int(round(ctrl_dt / self.sim_dt))
        self.max_steps = int(max_time / ctrl_dt)

        self.enable_drag = enable_drag
        self.enable_rolling_decel = enable_rolling_decel

        # Wind velocity (3D vector, m/s)
        if wind is not None:
            self.wind = np.array(wind, dtype=np.float64)
        else:
            self.wind = np.zeros(3)

        # Hole position
        if hole_pos is not None:
            self.hole_pos = np.array(hole_pos, dtype=np.float64)
        else:
            hole_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hole")
            self.hole_pos = self.model.body_pos[hole_body_id][:2].copy()

        # Terrain
        self.terrain = TerrainMap(terrain_zones)

        # Body/geom IDs
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.club_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "club_head")

        # Actuator IDs
        self.elbow_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_torque")
        self.wrist_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wrist_torque")

        # Ball free joint
        ball_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qvel_start = self.model.jnt_dofadr[ball_jnt_id]

        # Detect optional aim joint
        aim_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "aim")
        self.has_aim = aim_jnt_id >= 0
        self.aim_jnt_qpos = self.model.jnt_qposadr[aim_jnt_id] if self.has_aim else None

        # Joint qpos addresses
        self.elbow_jnt_qpos = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")]
        self.wrist_jnt_qpos = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wrist")]

        # Sensor indices
        aim_sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "aim_pos")
        if aim_sensor_id >= 0:
            self.SENS_AIM_POS = 0
            self.SENS_ELBOW_POS = 1
            self.SENS_ELBOW_VEL = 2
            self.SENS_WRIST_POS = 3
            self.SENS_WRIST_VEL = 4
            self.SENS_CLUB_POS = slice(5, 8)
            self.SENS_CLUB_QUAT = slice(8, 12)
            self.SENS_CLUB_VEL = slice(12, 15)
            self.SENS_BALL_POS = slice(15, 18)
            self.SENS_BALL_VEL = slice(18, 21)
        else:
            self.SENS_AIM_POS = None
            self.SENS_ELBOW_POS = 0
            self.SENS_ELBOW_VEL = 1
            self.SENS_WRIST_POS = 2
            self.SENS_WRIST_VEL = 3
            self.SENS_CLUB_POS = slice(4, 7)
            self.SENS_CLUB_QUAT = slice(7, 11)
            self.SENS_CLUB_VEL = slice(11, 14)
            self.SENS_BALL_POS = slice(14, 17)
            self.SENS_BALL_VEL = slice(17, 20)

        # State tracking
        self.step_count = 0
        self.ball_launched = False
        self.ball_launch_vel = None
        self.contact_detected = False
        self.ball_in_hole = False
        self.ball_landed = False
        self.ball_landed_terrain = None

    def reset(self, elbow_init=None, wrist_init=None, aim_init=None):
        mujoco.mj_resetData(self.model, self.data)

        if elbow_init is None:
            elbow_init = 3 * np.pi / 4
        if wrist_init is None:
            wrist_init = 0.0

        if self.has_aim and aim_init is not None:
            self.data.qpos[self.aim_jnt_qpos] = aim_init

        self.data.qpos[self.elbow_jnt_qpos] = elbow_init
        self.data.qpos[self.wrist_jnt_qpos] = wrist_init

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.ball_launched = False
        self.ball_launch_vel = None
        self.contact_detected = False
        self.ball_in_hole = False
        self.ball_landed = False
        self.ball_landed_terrain = None

        return self.get_obs()

    def _get_ground_contact_info(self):
        """
        Check if ball is on ground and return contact normal if so.

        Returns:
            (on_ground, normal): on_ground is bool, normal is 3D unit vector
            pointing away from the surface (upward on flat ground).
            If not on ground, normal is [0, 0, 1] (default upward).
        """
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.ball_geom_id or c.geom2 == self.ball_geom_id:
                other = c.geom2 if c.geom1 == self.ball_geom_id else c.geom1
                if other != self.club_geom_id:
                    # Contact normal from MuJoCo (points from geom2 to geom1)
                    normal = c.frame[:3].copy()
                    # Ensure normal points upward (away from ground)
                    if c.geom1 == self.ball_geom_id:
                        # normal points from geom2 (ground) to geom1 (ball) = upward
                        pass
                    else:
                        # normal points from ball to ground = downward, flip it
                        normal = -normal
                    # Ensure upward direction
                    if normal[2] < 0:
                        normal = -normal
                    return True, normal
        return False, np.array([0.0, 0.0, 1.0])

    def _freeze_ball(self):
        qv = self.ball_qvel_start
        self.data.qvel[qv:qv+6] = 0.0
        self.data.xfrc_applied[self.ball_body_id, :] = 0.0

    def _check_hole(self):
        if self.ball_in_hole:
            return True
        ball_pos = self.data.xpos[self.ball_body_id]
        dist_xy = np.linalg.norm(ball_pos[:2] - self.hole_pos)
        if dist_xy < HOLE_RADIUS and self.ball_landed:
            self.ball_in_hole = True
            self._freeze_ball()
            return True
        return False

    def _apply_ball_forces(self):
        if self.ball_in_hole:
            self._freeze_ball()
            return

        ball_pos = self.data.xpos[self.ball_body_id]
        ball_vel = self.data.cvel[self.ball_body_id][3:]

        # Check current ground contact and get surface normal
        on_ground_now, surface_normal = self._get_ground_contact_info()

        # Once the ball touches ground after launch, it stays "landed"
        # This prevents flickering from noisy contact detection
        if on_ground_now and self.ball_launched:
            self.ball_landed = True
            self.ball_landed_terrain = self.terrain.classify(ball_pos[0], ball_pos[1])

        # ── Aerodynamic drag (with wind) ──
        relative_vel = ball_vel - self.wind
        rel_speed = np.linalg.norm(relative_vel)
        if self.enable_drag and rel_speed > 1e-6:
            rel_dir = relative_vel / rel_speed
            drag_force = -DRAG_K * rel_speed ** 2 * rel_dir
            self.data.xfrc_applied[self.ball_body_id, :3] = drag_force
        else:
            self.data.xfrc_applied[self.ball_body_id, :3] = 0.0

        # ── Rolling deceleration ──
        # Applied continuously once the ball has landed (no flickering)
        # Uses contact normal when available, falls back to vertical
        if self.enable_rolling_decel and self.ball_landed:
            speed = np.linalg.norm(ball_vel)
            if speed > 0.01:
                # Use current contact normal if touching, else vertical
                normal = surface_normal if on_ground_now else np.array([0.0, 0.0, 1.0])

                # Project velocity onto surface plane
                v_dot_n = np.dot(ball_vel, normal)
                v_surface = ball_vel - v_dot_n * normal
                v_surface_speed = np.linalg.norm(v_surface)

                if v_surface_speed > 0.01:
                    v_surface_dir = v_surface / v_surface_speed
                    cos_slope = abs(normal[2])
                    decel = self.terrain.deceleration(ball_pos[0], ball_pos[1])
                    ball_mass = self.model.body_mass[self.ball_body_id]
                    brake_force = -decel * ball_mass * cos_slope * v_surface_dir
                    self.data.xfrc_applied[self.ball_body_id, :3] += brake_force

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

        ball_stopped = self.ball_launched and not self.ball_in_hole and ball_speed < 0.01
        if ball_stopped:
            done = True
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
            "landed_terrain": self.ball_landed_terrain,
            "done": done,
        }
        return obs, info

    def get_obs(self):
        s = self.data.sensordata
        obs = {
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
        if self.SENS_AIM_POS is not None:
            obs["aim_pos"] = s[self.SENS_AIM_POS]
        return obs

    def get_flat_obs(self):
        obs = self.get_obs()
        parts = [
            [obs["elbow_pos"], obs["elbow_vel"]],
            [obs["wrist_pos"], obs["wrist_vel"]],
            obs["club_head_pos"],
            obs["club_head_vel"],
            obs["ball_pos"],
            obs["ball_vel"],
        ]
        if "aim_pos" in obs:
            parts.insert(0, [obs["aim_pos"]])
        return np.concatenate(parts)

    @property
    def obs_dim(self):
        return 15 if self.has_aim else 14

    @property
    def act_dim(self):
        return 2