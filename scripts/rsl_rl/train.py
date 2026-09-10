# SPDX-License-Identifier: BSD-3-Clause
"""Train an RL agent for the M20 robot with RSL-RL.

Example
-------

.. code-block:: bash

    ./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-M20-v0
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent for M20 with RSL-RL.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-M20-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import M20_rl.tasks  # noqa: F401  (registers the M20 gym environments)


def main():
    """Train with RSL-RL."""
    # parse the environment and agent configurations from the gym registry
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    # apply command-line overrides
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.resume:
        agent_cfg.resume = True
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify the logging directory
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # resolve the resume checkpoint before creating a fresh log dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # create the environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap around the environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create the runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # load a checkpoint if resuming
    if agent_cfg.resume:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
