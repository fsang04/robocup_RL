# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab_tasks.manager_based.locomotion.velocity.config.t1.rough_env_cfg import T1Rewards
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from .flat_env_cfg import T1FlatEnvCfg


@configclass
class T1GetUpRewards(T1Rewards):
    """Rewards for the get=up task."""
    upright_posture_bonus = RewTerm(
        func=mdp.upright_posture_bonus,
        weight=1.0,
        params={"threshold": 0.8},
    )
    trunk_height = RewTerm(
        func=mdp.trunk_height_reward,
        weight=10.0,
        params={"target_height": 0.98, "std": 0.15},
    )
    both_feet_contact = RewTerm(
        func=mdp.both_feet_contact_reward,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_link")},
    )
    knee_straight = RewTerm(
        func=mdp.knee_straight_reward,
        weight=0.25,
        params={"std": 0.3, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Knee_Pitch"])},
    )
    keyframe_tracking = RewTerm(
        func=mdp.KeyframeJointTracking,
        weight=2.0,
        params={
            "keyframe_file": os.path.join(os.path.dirname(__file__), "keyframes", "get_up_front.yaml"),
            "keyframe_file_back": os.path.join(os.path.dirname(__file__), "keyframes", "get_up_back.yaml"),
            # Per-keyframe advance thresholds (sum of squared joint errors).
            # Floors are computed from joints commanded past URDF limits — the robot
            # can never get below these values so thresholds must be set above them.
            # front KF floors: [0.0, 0.0, 6.54, 0.0, 0.0]
            "advance_threshold":      [18.0, 30.0, 8.0, 1.0, 1.0], # front: front: KF0/KF1 just above initial error
            # back KF floors:  [19.46, 16.30, 0.03, 0.0]
            "advance_threshold_back": [110.0, 18.0, 1.0, 1.0], # back: back: KF0 requires extreme tuck, initial error ~104
        },
    )
    arm_ground_contact = RewTerm(
        func=mdp.arm_ground_contact_reward,
        weight=0.5,
        params={
            # AL3/AR3 = forearms; left_hand_link/right_hand_link = distal hand links
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["AL3", "AR3", "left_hand_link", "right_hand_link"]),
            "height_threshold": 0.5,  # only active while trunk is still low
        },
    )
    getup_success_bonus = RewTerm( # success reward: give +200 when getup_success termination fires
        func=mdp.is_terminated_term,
        weight=200.0,
        params={"term_keys": "getup_success"},
    )


@configclass
class T1GetUpEnvCfg(T1FlatEnvCfg):
    """Environment config for the get-up task."""

    rewards: T1GetUpRewards = T1GetUpRewards()
    # observations: T1GetUpObservations = T1GetUpObservations()

    def __post_init__(self):
        super().__post_init__()

        # spawn robot in random fallen pose
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14),
                "roll": (-3.14, 3.14),   # random fall direction
                "pitch": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        # do not reset robot joints by scale? (default in velocity_env_cfg)
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0) # start in default fallen pose

        # disable termination reset
        self.terminations.base_contact = None

        # Rewards -- adjust / zero-out inherited locomotion rewards as needed
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -3.0 # penalize non-upright positions
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_pos_limits.weight = 0.0 # keyframes want to hit joint limits 
        self.rewards.dof_torques_l2.weight = -1.0e-6
        self.rewards.joint_deviation_hip.weight = 0.0 # keyframes use hip roll/yaw
        self.rewards.termination_penalty.weight = 0.0 # zero out original negative termination penalty

        # Commands: zero out velocity commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # Terminations: add success term for reaching final keyframe and standing position
        self.terminations.getup_success = DoneTerm(
            func=mdp.GetUpSuccess,
            params={
                "min_upright_steps": 60,
                "upright_threshold": 0.85,
                "knee_straight_threshold": 0.3,
                "foot_contact_threshold": 1.0,
            },
            time_out=False, # do not terminate on time out, to get better learning signal
        )


class T1GetUpEnvCfg_PLAY(T1GetUpEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None