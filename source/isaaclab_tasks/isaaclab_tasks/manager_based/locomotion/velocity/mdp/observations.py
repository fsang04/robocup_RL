# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def ball_pos_relative(env, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Ball position in the robot local frame (x=forward, y=left, z=up)."""
    robot = env.scene["robot"]
    ball = env.scene[ball_cfg.name]
    ball_pos_w = ball.data.root_link_pos_w[:, :3]
    robot_pos_w = robot.data.root_pos_w
    # rotate world-frame offset into robot's heading frame (strips roll/pitch, keeps yaw)
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), ball_pos_w - robot_pos_w)

def ball_vel_relative(env, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")) -> torch.Tensor:
    """Ball velocity in the robot local frame (x=forward, y=left, z=up)."""
    robot = env.scene["robot"]
    ball = env.scene[ball_cfg.name]
    ball_vel_w = ball.data.root_vel_w[:, :3]
    robot_vel_w = robot.data.root_vel_w[:, :3]
    # rotate world-frame velocity difference into robot's heading frame (strips roll/pitch, keeps yaw)
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), ball_vel_w - robot_vel_w)


def goal_pos_relative_to_ball(
    env, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"), goal_pos: tuple = (10.0, 0.0, 0.0)
) -> torch.Tensor:
    """Goal position relative to the ball, expressed in the robot's yaw-aligned local frame.

    Gives the robot a direction to kick: positive x means the goal is ahead of the ball
    along the robot's forward axis.
    """
    robot = env.scene["robot"]
    ball = env.scene[ball_cfg.name]
    goal_w = torch.tensor(goal_pos, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    ball_pos_w = ball.data.root_link_pos_w[:, :3]
    # rotate goal-from-ball offset into robot's heading frame for a consistent observation frame
    return quat_apply_inverse(yaw_quat(robot.data.root_quat_w), goal_w - ball_pos_w)

