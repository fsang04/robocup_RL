# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Booster T1 humanoid robot.

The following configurations are available:

* :obj:`T1_CFG`: Booster T1 robot (locomotion variant, arms/head fixed)

"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

_T1_USD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/Robots/Booster/T1/T1_locomotion.usd"))

T1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_T1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.98),
        joint_pos={
            "Left_Hip_Yaw": 0.0,
            "Left_Hip_Roll": 0.0,
            "Left_Hip_Pitch": -0.20,
            "Left_Knee_Pitch": 0.42,
            "Left_Ankle_Pitch": -0.23,
            "Left_Ankle_Roll": 0.0,
            "Right_Hip_Yaw": 0.0,
            "Right_Hip_Roll": 0.0,
            "Right_Hip_Pitch": -0.20,
            "Right_Knee_Pitch": 0.42,
            "Right_Ankle_Pitch": -0.23,
            "Right_Ankle_Roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Yaw",
                ".*_Hip_Roll",
                ".*_Hip_Pitch",
                ".*_Knee_Pitch",
            ],
            effort_limit_sim=300,
            stiffness={
                ".*_Hip_Yaw": 150.0,
                ".*_Hip_Roll": 150.0,
                ".*_Hip_Pitch": 200.0,
                ".*_Knee_Pitch": 200.0,
            },
            damping={
                ".*_Hip_Yaw": 5.0,
                ".*_Hip_Roll": 5.0,
                ".*_Hip_Pitch": 5.0,
                ".*_Knee_Pitch": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
            effort_limit_sim=100,
            stiffness=20.0,
            damping=4.0,
        ),
    },
)
"""Configuration for the Booster T1 humanoid robot (locomotion variant).

Arms, head, and neck joints are fixed. Only the 12 leg joints are actuated.
"""
