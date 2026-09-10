# SPDX-License-Identifier: BSD-3-Clause
"""Configuration for M20 velocity-tracking locomotion on flat terrain.

This is the *primary* training environment for the M20 robot: a flat plane with
no height scanner and no terrain curriculum. It is simpler and converges faster
than the rough-terrain variant, making it a good starting point.
"""

from isaaclab.utils import configclass

from .rough_env_cfg import M20RoughEnvCfg


@configclass
class M20FlatEnvCfg(M20RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # a wheeled robot must stay flat to keep its wheels in contact
        self.rewards.flat_orientation_l2.weight = -2.5

        # change terrain to a flat plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scanner and no height-scan observation
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # drop the terrain-aware height target (flat terrain has a fixed target)
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class M20FlatEnvCfg_PLAY(M20FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable observation noise for deterministic playback
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
