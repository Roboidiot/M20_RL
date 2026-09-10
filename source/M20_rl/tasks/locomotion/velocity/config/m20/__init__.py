# SPDX-License-Identifier: BSD-3-Clause
"""Register the M20 velocity-tracking locomotion environments with gymnasium."""

import gymnasium as gym

from . import agents


##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Rough-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:M20RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:M20RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-M20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:M20RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:M20RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:M20FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:M20FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-M20-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:M20FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:M20FlatPPORunnerCfg",
    },
)
