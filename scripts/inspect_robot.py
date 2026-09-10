# SPDX-License-Identifier: BSD-3-Clause
"""Inspect the M20 USD asset and print its joint/body names.

This is useful to verify that the joint-name regular expressions used in the
action/actuator configuration (``.*_hipx_joint``, ``.*_wheel_joint``, ...) match
the names that IsaacLab actually resolves from the USD articulation.

Example
-------

.. code-block:: bash

    ./isaaclab.sh -p scripts/inspect_robot.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect the M20 articulation asset.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from M20_rl.assets import M20_CFG, M20_URDF_PATH


def design_scene():
    """Design the scene with a ground plane and the M20 robot."""
    # ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    # robot
    robot = Articulation(M20_CFG.replace(prim_path="/World/Robot"))
    return robot


def run_inspector(robot: Articulation) -> None:
    """Print joint and body names and step the simulation briefly."""
    print(f"[INFO] URDF path: {M20_URDF_PATH}")
    print(f"[INFO] num_joints = {robot.num_joints}")
    print("[INFO] Joint names:")
    for i, name in enumerate(robot.data.joint_names):
        print(f"  [{i:02d}] {name}")
    print(f"[INFO] num_bodies = {robot.num_bodies}")
    print("[INFO] Body names:")
    for name in robot.data.body_names:
        print(f"  - {name}")
    print("[INFO] Stepping the simulation for 10 steps to check for errors...")
    for _ in range(10):
        robot.write_root_pose_to_sim()
        robot.write_joint_state_to_sim()
        robot.reset()
        simulation_app.update()


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 0.0, 1.0], target=[0.0, 0.0, 0.3])
    robot = design_scene()
    sim.reset()
    run_inspector(robot)


if __name__ == "__main__":
    main()
    simulation_app.close()
