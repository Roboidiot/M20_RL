"""Gym-registered manager-based RL tasks for the M20 robot.

Importing :mod:`M20_rl.tasks.locomotion.velocity` registers the following
gymnasium environments (see ``config/m20/__init__.py``):

* ``Isaac-Velocity-Rough-M20-v0``      -- rough-terrain locomotion (training).
* ``Isaac-Velocity-Rough-M20-Play-v0`` -- rough-terrain locomotion (play).
* ``Isaac-Velocity-Flat-M20-v0``       -- flat-terrain locomotion (training).
* ``Isaac-Velocity-Flat-M20-Play-v0``  -- flat-terrain locomotion (play).
"""

from . import locomotion  # noqa: F401
