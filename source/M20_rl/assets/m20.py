"""Articulation configuration for the M20 wheel-legged quadruped robot.

The M20 robot is a wheel-legged quadruped. Each of its four legs has four
joints:

* ``hipx`` -- hip roll (revolute, axis ``-x``).
* ``hipy`` -- hip pitch (revolute, axis ``-y``).
* ``knee`` -- knee pitch (revolute, axis ``-y``).
* ``wheel`` -- wheel (continuous, axis ``-y``).

This gives 16 actuated joints in total. The hip/knee joints are driven in
*position* mode (implicit PD actuator with non-zero stiffness), while the wheel
joints are driven in *velocity* mode (implicit actuator with ``stiffness=0`` and
a non-zero ``damping`` -- the damping converts the commanded joint velocity into
torque).

The robot is loaded from the URDF file (``M20.urdf``), which IsaacLab converts
to a USD articulation at runtime via the built-in URDF importer. This is more
robust than the bundled USD, which uses the ``IsaacRobotAPI`` format that
IsaacLab's articulation importer does not resolve into PhysX joints.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Absolute paths to the robot files (resolved relative to this source file so
# that the asset loads regardless of the current working directory).
_M20_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "M20")
M20_URDF_PATH = os.path.join(_M20_DIR, "urdf", "M20.urdf")
M20_USD_PATH = os.path.join(_M20_DIR, "M20_usd", "M20.usd")


M20_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=M20_URDF_PATH,
        # The base link is free-floating (mobile robot).
        fix_base=False,
        # Merge the fixed sensor joints (camera/lidar) into their parent link.
        merge_fixed_joints=True,
        make_instanceable=True,
        force_usd_conversion=False,
        # No conversion-time joint drive: the runtime actuators (ImplicitActuatorCfg
        # below) set the per-joint stiffness/damping themselves.
        joint_drive=None,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Base hovering ~2 cm above the nominal wheel-contact height; the robot
        # settles onto its wheels at the start of the episode.
        pos=(0.0, 0.0, 0.6),
        rot=(1.0, 0.0, 0.0, 0.0),
        # Default pose is the URDF zero configuration (legs fully extended).
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Legs (hipx / hipy / knee): position-controlled implicit PD actuators.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hipx_joint", ".*_hipy_joint", ".*_knee_joint"],
            stiffness=40.0,
            damping=1.0,
            effort_limit_sim=30.0,
            velocity_limit_sim=20.0,
        ),
        # Wheels: velocity-controlled implicit actuators (stiffness=0 -> the
        # damping term tracks the commanded joint velocity).
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=0.0,
            damping=1.5,
            effort_limit_sim=20.0,
            velocity_limit_sim=30.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
