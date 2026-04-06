# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned
    robot frame using an exponential kernel.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)

##
# Terms added for kicking policy training, which inherits from flat env rewards
##

def both_feet_in_air(
    env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*foot_link"]),
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    Returns 1.0 when neither foot has contact force above threshold, 0.0 otherwise.
    Use w/ negative weight to discourage jumping
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids: list[int] | slice = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    # max contact force over history for each foot: shape (N, num_feet)
    foot_forces = contact_sensor.data.net_forces_w_history[:, :, body_ids, :].norm(dim=-1).max(dim=1)[0]
    # True where a foot has contact above threshold
    foot_in_contact = foot_forces > threshold  # (N, num_feet)
    # penalty fires when NO foot is in contact
    both_in_air = ~foot_in_contact.any(dim=1)
    return both_in_air.float()


def approach_ball_reward(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    std: float = 1.0,
) -> torch.Tensor:
    """Reward approaching the ball using an exponential kernel.

    Always returns [0, 1] — never negative — so a successful kick sending the ball far away
    does not punish the robot. Returns ~1.0 when adjacent, falls toward 0 at large distances.
    """
    asset = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    distance = torch.norm(asset.data.root_pos_w - ball.data.root_link_pos_w[:, :3], dim=1)
    return torch.exp(-distance / std)


# added 04/05
def swing_foot_contact_near_ball(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="right_foot_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    threshold: float = 1.0,
    proximity: float = 0.5,
) -> torch.Tensor:
    """Penalize kicking foot ground contact, gated on the robot being close to the ball.

    Fires only within `proximity` metres of the ball so the penalty doesn't discourage normal
    walking — it only kicks in when the robot should be swinging through to kick.
    Use with a negative weight.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_forces = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )  # (N, num_selected_bodies)
    in_contact = (foot_forces > threshold).any(dim=-1).float()  # (N,)

    asset = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    dist = torch.norm(asset.data.root_pos_w - ball.data.root_link_pos_w[:, :3], dim=1)
    near_ball = (dist < proximity).float()

    return in_contact * near_ball

 
# added 04/05
class BallProgressReward(ManagerTermBase):
    """Reward cumulative ball displacement toward the goal since episode start.

    Unlike ``ball_vel_toward_target``, this fires continuously — even when the ball is
    stationary — giving a persistent signal for how far the ball has been advanced.
    The reward is clamped at zero so backward ball movement is never penalised.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        ball_cfg: SceneEntityCfg = cfg.params.get("ball_cfg", SceneEntityCfg("ball"))
        self._ball_name = ball_cfg.name
        self._ball_init_pos = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        ball = self._env.scene[self._ball_name]
        self._ball_init_pos[env_ids] = ball.data.root_link_pos_w[env_ids, :3].clone()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
        target_pos: tuple = (5.0, 0.0, 0.0),
    ) -> torch.Tensor:
        ball = env.scene[ball_cfg.name]
        target = torch.as_tensor(target_pos, device=env.device, dtype=torch.float32)
        goal_dir = target / (target.norm() + 1e-6)  # unit vector toward goal
        displacement = ball.data.root_link_pos_w[:, :3] - self._ball_init_pos
        progress = torch.sum(displacement * goal_dir.unsqueeze(0), dim=1)
        return progress.clamp(min=0.0)


def ball_vel_toward_target(
    env,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    target_pos: tuple = (5.0, 0.0, 0.0),
    ball_speed_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward the ball's velocity toward the target, only when ball is actively moving.

    Reward only fires during/after a kick, not while ball is stationary or drifting slowly
    """
    ball = env.scene[ball_cfg.name]
    target = torch.as_tensor(target_pos, device=env.device, dtype=torch.float32)
    ball_to_target = target - ball.data.root_link_pos_w[:, :3]

    ball_vel = ball.data.root_link_vel_w[:, :3]
    ball_speed = ball_vel.norm(dim=1)

    # cosine component of ball velocity toward target, clamped so deflections don't penalise
    vel_toward_target = (torch.sum(ball_vel * ball_to_target, dim=1) / (
        ball_to_target.norm(dim=1) + 1e-6
    )).clamp(min=0.0)

    # only reward during/immediately after a kick (so ball drifting slowly towards goal isn't rewarded)
    ball_is_moving = (ball_speed > ball_speed_threshold).float()

    # only reward when robot is close enough to ball
    robot = env.scene["robot"]
    robot_to_ball = torch.norm(robot.data.root_pos_w - ball.data.root_link_pos_w[:, :3], dim=1)
    near_ball = (robot_to_ball < 0.5).float() # when less than 0.5 meter away

    return vel_toward_target * ball_is_moving * near_ball