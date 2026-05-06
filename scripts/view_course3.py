"""
View Course 3 in the MuJoCo viewer.

This does not optimize anything.
It only loads models/course3.xml so we can inspect the course visually.
"""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer

from courses.course3 import MODEL_PATH


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    print("Course 3 loaded in MuJoCo viewer.")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Starting camera view: roughly centered on the full course
        viewer.cam.lookat[:] = [60.0, 8.0, 0.0]
        viewer.cam.distance = 130.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -45.0

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()