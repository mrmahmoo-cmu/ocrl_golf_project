"""
Golf Environment — Direct Launch
"""

import numpy as np
import mujoco
from pathlib import Path

# Aerodynamic constants
AIR_DENSITY = 1.225
BALL_RADIUS = 0.0214
BALL_AREA = np.pi * BALL_RADIUS ** 2
CD_GOLF_BALL = 0.25
DRAG_K = 0.5 * AIR_DENSITY * CD_GOLF_BALL * BALL_AREA

# Terrain types and rolling deceleration
TERRAIN_FAIRWAY = "fairway"
TERRAIN_GREEN   = "green"
TERRAIN_SAND    = "sand"
TERRAIN_ROUGH   = "rough"
TERRAIN_TEE     = "tee"
TERRAIN_WATER  = "water"    # Bashar added water for course 3

ROLLING_DECEL = {
    TERRAIN_FAIRWAY: 2.0,
    TERRAIN_GREEN:   0.8,
    TERRAIN_SAND:    8.0,
    TERRAIN_ROUGH:   5.0,
    TERRAIN_TEE:     2.0,
    TERRAIN_WATER: 20.0,        # Bashar added for water (for physics)
}

# Hole dimensions
HOLE_RADIUS = 0.054


class TerrainMap:
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


class GolfEnv:
    def __init__(self, ctrl_dt=0.002, max_time=15.0, hole_pos=None,
                 terrain_zones=None, enable_drag=True, enable_rolling_decel=True,
                 model_path=None, wind=None):
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            self.model_path = Path(__file__).parent.parent / "models" / "course1.xml"

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.sim_dt = self.model.opt.timestep
        self.ctrl_dt = ctrl_dt
        self.ctrl_substeps = int(round(ctrl_dt / self.sim_dt))
        self.max_steps = int(max_time / ctrl_dt)

        self.enable_drag = enable_drag
        self.enable_rolling_decel = enable_rolling_decel

        # Wind
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

        # Ball IDs
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

        # Ball free joint
        ball_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qpos_start = self.model.jnt_qposadr[ball_jnt_id]
        self.ball_qvel_start = self.model.jnt_dofadr[ball_jnt_id]

        # Sensor indices (ball only — no arm sensors)
        self.SENS_BALL_POS = slice(0, 3)
        self.SENS_BALL_VEL = slice(3, 6)

        # State tracking
        self.step_count = 0
        self.ball_launched = False
        self.ball_launch_vel = None
        self.ball_in_hole = False
        self.ball_landed = False
        self.ball_landed_terrain = None

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.ball_launched = False
        self.ball_launch_vel = None
        self.ball_in_hole = False
        self.ball_landed = False
        self.ball_landed_terrain = None

    def launch_ball(self, speed, vert_angle, horiz_angle):
        """
        Set the ball's initial velocity directly.

        Args:
            speed: Launch speed in m/s
            vert_angle: Vertical angle in radians (0 = horizontal, pi/4 = 45 deg up)
            horiz_angle: Horizontal angle in radians (0 = +X, positive = toward +Y)
        """
        vx = speed * np.cos(vert_angle) * np.cos(horiz_angle)
        vy = speed * np.cos(vert_angle) * np.sin(horiz_angle)
        vz = speed * np.sin(vert_angle)

        qv = self.ball_qvel_start
        self.data.qvel[qv]     = vx
        self.data.qvel[qv + 1] = vy
        self.data.qvel[qv + 2] = vz

        self.ball_launched = True
        self.ball_launch_vel = np.array([vx, vy, vz])

    def _get_ground_contact_info(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.ball_geom_id or c.geom2 == self.ball_geom_id:
                normal = c.frame[:3].copy()
                if c.geom1 == self.ball_geom_id:
                    pass
                else:
                    normal = -normal
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
        on_ground_now, _ = self._get_ground_contact_info()
        if dist_xy < HOLE_RADIUS and on_ground_now and self.ball_landed:
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

        on_ground_now, surface_normal = self._get_ground_contact_info()

        if on_ground_now and self.ball_launched:
            if not self.ball_landed:
                self.ball_landed = True
                self.ball_landed_terrain = self.terrain.classify(ball_pos[0], ball_pos[1])

        # Drag (wind-relative)
        relative_vel = ball_vel - self.wind
        rel_speed = np.linalg.norm(relative_vel)
        if self.enable_drag and rel_speed > 1e-6:
            rel_dir = relative_vel / rel_speed
            drag_force = -DRAG_K * rel_speed ** 2 * rel_dir
            self.data.xfrc_applied[self.ball_body_id, :3] = drag_force
        else:
            self.data.xfrc_applied[self.ball_body_id, :3] = 0.0

        # Rolling deceleration (slope-sensitive)
        if self.enable_rolling_decel and self.ball_landed:
            speed = np.linalg.norm(ball_vel)
            if speed > 0.01:
                normal = surface_normal if on_ground_now else np.array([0.0, 0.0, 1.0])
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

    def step(self):
        """Advance physics by one ctrl_dt. No action needed — ball is free-flying."""
        for _ in range(self.ctrl_substeps):
            self._apply_ball_forces()
            self._check_hole()
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        s = self.data.sensordata
        ball_pos = s[self.SENS_BALL_POS].copy()
        ball_vel = s[self.SENS_BALL_VEL].copy()
        ball_speed = np.linalg.norm(ball_vel)

        done = self.step_count >= self.max_steps
        ball_xy = ball_pos[:2]
        dist_to_hole = np.linalg.norm(ball_xy - self.hole_pos)

        ball_stopped = self.ball_launched and not self.ball_in_hole and ball_speed < 0.01
        if ball_stopped:
            done = True
        if self.ball_in_hole:
            done = True

        terrain_type = self.terrain.classify(ball_xy[0], ball_xy[1])

        info = {
            "time": self.data.time,
            "ball_pos": ball_pos,
            "ball_vel": ball_vel,
            "ball_speed": ball_speed,
            "dist_to_hole": dist_to_hole,
            "ball_stopped": ball_stopped,
            "in_hole": self.ball_in_hole,
            "terrain": terrain_type,
            "landed_terrain": self.ball_landed_terrain,
            "done": done,
        }
        return info