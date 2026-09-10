"""Setup script for the M20_rl package.

This package provides IsaacLab tasks and scripts for training the M20
wheel-legged quadruped robot. Install it in *editable* mode inside your
Isaac Sim / IsaacLab Python environment:

    ./isaaclab.sh -p -m pip install -e .

or, if you use a standalone conda environment:

    pip install -e .
"""

from setuptools import find_packages, setup

setup(
    name="M20_rl",
    version="0.1.0",
    description=(
        "IsaacLab 2.3 reinforcement-learning training project for the M20 "
        "wheel-legged quadruped robot."
    ),
    author="Roboidiot",
    license="BSD-3-Clause",
    python_requires=">=3.10",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    include_package_data=True,
    # IsaacLab, isaaclab_tasks, isaaclab_rl and isaaclab_assets are provided by
    # the Isaac Sim / IsaacLab environment and are intentionally not listed here.
    install_requires=[],
)
