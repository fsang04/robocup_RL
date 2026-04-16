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


def robot_behind_ball_alignment(
    env, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_pos: tuple = (5.0, 0.0, 0.0),
    proximity_threshold: float = 1.5,
) -> torch.Tensor:
    """Reward the robot for being behind the ball relative to the goal direction.
    Only active when the robot is within proximity_threshold of the ball.
    """
    robot = env.scene[asset_cfg.name]
    ball = env.scene[ball_cfg.name]
    goal_w = torch.tensor(goal_pos, device=env.device)

    ball_pos = ball.data.root_link_pos_w[:, :3]
    robot_pos = robot.data.root_pos_w

    ball_to_goal = goal_w - ball_pos                        # (N, 3)
    ball_to_robot = robot_pos - ball_pos                    # (N, 3)

    # ideal approach direction is opposite to ball→goal
    ideal = -ball_to_goal / (ball_to_goal.norm(dim=1, keepdim=True) + 1e-6)
    actual = ball_to_robot / (ball_to_robot.norm(dim=1, keepdim=True) + 1e-6)

    alignment = torch.sum(ideal * actual, dim=1).clamp(min=0.0)  # [0, 1]

    # only reward when close enough to the ball to matter
    dist = ball_to_robot.norm(dim=1)
    active = (dist < proximity_threshold).float()

    return alignment * active


# added 04/05
def swing_foot_contact_near_ball(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="right_foot_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    threshold: float = 1.0,
    proximity: float = 0.5,
    ball_kick_speed: float = 0.3,
) -> torch.Tensor:
    """Penalize kicking foot *ground* contact near the ball.

    Check whether ball is already moving: if ball_speed > ball_kick_speed, foot just kicked the ball
    and contact should NOT be penalised.
    Only penalise when the ball is still stationary.
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

    # suppress penalty when ball is moving (kick contact)
    ball_speed = ball.data.root_link_vel_w[:, :3].norm(dim=1)
    ball_stationary = (ball_speed < ball_kick_speed).float()

    dist = torch.norm(asset.data.root_pos_w - ball.data.root_link_pos_w[:, :3], dim=1)
    near_ball = (dist < proximity).float()

    return in_contact * ball_stationary * near_ball

 
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
    ball_speed_threshold: float = 0.1,
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
    # robot_to_ball = torch.norm(robot.data.root_pos_w - ball.data.root_link_pos_w[:, :3], dim=1)
    # near_ball = (robot_to_ball < 0.5).float() # when less than 0.5 meter away

    return vel_toward_target * ball_is_moving # * near_ball


def arm_ground_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.5,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward arm/elbow contact with the ground when trunk is low (get-up phase).
    Activates when some part of arm body is in contact with the ground, AND the trunk height
    is below height_threshold. Rewards positively to encourage pushing with arms up off ground.
    
    Body names to use for T1: AL2, AL3, left_hand_link, AR2, AR3, right_hand_link
    (AL3/AR3 are forearms; left_hand_link/right_hand_link are the distal links).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    arm_forces = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
    )  # (N, num_arm_bodies)
    arm_in_contact = (arm_forces > force_threshold).any(dim=1).float()

    asset = env.scene[asset_cfg.name]
    is_low = (asset.data.root_pos_w[:, 2] < height_threshold).float()

    return arm_in_contact * is_low


def upright_posture_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture."""
    asset = env.scene[asset_cfg.name]
    up_proj = -asset.data.projected_gravity_b[:, 2]
    return (up_proj > threshold).float()


def trunk_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.98,
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward trunk height being close to desired standing height.

    Sitting upright gives ~0.4m trunk height; standing gives ~0.98m.
    """
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    return torch.exp(-torch.square(height - target_height) / std**2)


def both_feet_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*foot_link"),
    threshold: float = 1.0,
    min_trunk_height: float = 0.6,   # only activate when height is above certain threshold
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    foot_forces = (
        contact_sensor.data.net_forces_w_history[:, :, body_ids, :]
        .norm(dim=-1).max(dim=1)[0]
    )
    asset = env.scene[asset_cfg.name]   # need to pass asset_cfg too

    # only reward both feet contact is trunk is at a certain height
    trunk_high_enough = (asset.data.root_pos_w[:, 2] > min_trunk_height).float()
    return (foot_forces > threshold).all(dim=1).float() * trunk_high_enough



def knee_straight_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_Knee_Pitch"]),
) -> torch.Tensor:
    """Reward knees being near fully extended (0 rad = straight) using an exponential kernel."""
    asset = env.scene[asset_cfg.name]
    knee_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (N, 2)
    error = torch.sum(torch.square(knee_pos), dim=1)
    return torch.exp(-error / std**2)


# map joint names from keyframe YAML to Isaac joint names used in booster.py
# YAMLs is written for one symmetrical side of the robot: roll/yaw joints require opposite signs
_KEYFRAME_SYMMETRIC_SAME = {"Hip_Pitch", "Knee_Pitch", "Ankle_Pitch", "Shoulder_Pitch", "Elbow_Pitch"}
_KEYFRAME_SYMMETRIC_MIRROR = {"Hip_Roll", "Hip_Yaw", "Ankle_Roll", "Shoulder_Roll", "Elbow_Yaw"}
_KEYFRAME_SINGLE = {"Waist"}  # specifically allow reward to track waist so that it stays at 0
_KEYFRAME_IGNORED = {"Head_yaw", "Head_pitch"}  # name mismatch: YAML uses Head_yaw, Isaac uses AAHead_yaw
_DEG_TO_RAD = 3.14159265358979 / 180.0


def _expand_keyframe(motor_positions: dict) -> dict:
    """Expand symmetric YAML motor_positions to {isaac_joint_name: radians}."""
    result = {}
    for name, deg in motor_positions.items():
        if name in _KEYFRAME_IGNORED:
            continue
        rad = deg * _DEG_TO_RAD
        if name in _KEYFRAME_SYMMETRIC_SAME:
            # original controller applies -position to both sides for these joints
            result[f"Left_{name}"] = -rad
            result[f"Right_{name}"] = -rad
        elif name in _KEYFRAME_SYMMETRIC_MIRROR:
            result[f"Left_{name}"] = rad
            result[f"Right_{name}"] = -rad
        elif name in _KEYFRAME_SINGLE:
            result[name] = rad
    return result


class KeyframeJointTracking(ManagerTermBase):
    """Reward joint positions matching a pose-gated keyframe sequence.

    Loads one or two YAML keyframe files (front and optionally back) once at
    init. At the first step of each episode, fall direction is detected from
    projected_gravity_b[:, 0].
    Each env advances to the next keyframe only once its joints are within
    advance_threshold (sum of squared errors) of the current target.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        import yaml

        def _load_sequence(path):
            with open(str(path), "r") as f:
                data = yaml.safe_load(f)

            joint_names = env.scene["robot"].joint_names
            targets = []
            gravity_targets = []

            for kf in data["keyframes"]:
                expanded = _expand_keyframe(kf["motor_positions"])
                target = torch.zeros(len(joint_names), device=env.device)

                for isaac_name, rad_val in expanded.items():
                    if isaac_name in joint_names:
                        target[joint_names.index(isaac_name)] = rad_val
                targets.append(target)
                
                # load target gravity direction for keyframe
                g = kf.get("target_gravity", [0.0, 0.0, -1.0])
                g_t = torch.tensor(g, dtype=torch.float32, device=env.device)
                gravity_targets.append(g_t / (g_t.norm() + 1e-6))

            return torch.stack(targets, dim=0), torch.stack(gravity_targets, dim=0)

        self._targets_front, self._gravity_front = _load_sequence(cfg.params["keyframe_file"])

        keyframe_file_back = cfg.params.get("keyframe_file_back")
        if keyframe_file_back:
            self._targets_back, self._gravity_back = _load_sequence(keyframe_file_back)
        else:
            self._targets_back, self._gravity_back = self._targets_front, self._gravity_front

        # build per-keyframe threshold tensors — accepts a scalar or a list
        def _to_threshold_tensor(val, num_keyframes):
            if isinstance(val, (int, float)):
                return torch.full((num_keyframes,), float(val), device=env.device)
            t = torch.tensor(val, dtype=torch.float32, device=env.device)
            assert len(t) == num_keyframes, (
                f"advance_threshold list length {len(t)} != num keyframes {num_keyframes}"
            )
            return t

        raw_front = cfg.params.get("advance_threshold", 2.0)
        raw_back  = cfg.params.get("advance_threshold_back", cfg.params.get("advance_threshold", 2.0))
        self._thresholds_front = _to_threshold_tensor(raw_front, len(self._targets_front))
        self._thresholds_back  = _to_threshold_tensor(raw_back,  len(self._targets_back))

        # per-env current keyframe index and fall-direction flag
        self._kf_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._use_back = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._kf_idx[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        keyframe_file: str,
        keyframe_file_back: str | None = None,
        advance_threshold: float | list = 2.0,
        advance_threshold_back: float | list | None = None,
    ) -> torch.Tensor:
        asset = env.scene["robot"]
        joint_pos = asset.data.joint_pos  # (N, num_joints)

        # on the first step of each episode detect fall direction and reset index
        new_episode = env.episode_length_buf == 1
        if new_episode.any():
            face_up = asset.data.projected_gravity_b[:, 0] < 0
            self._use_back[new_episode] = face_up[new_episode]
            self._kf_idx[new_episode] = 0

        # clamp index to valid range for each sequence
        max_front = len(self._targets_front) - 1
        max_back  = len(self._targets_back) - 1
        idx_front = self._kf_idx.clamp(max=max_front)
        idx_back  = self._kf_idx.clamp(max=max_back)

        # look up current target pose for front/back
        targets_front = self._targets_front[idx_front]  # (N, num_joints)
        targets_back  = self._targets_back[idx_back]
        targets = torch.where(self._use_back.unsqueeze(1), targets_back, targets_front)

        # look up current target gravity direction for front/back
        gravity_front = self._gravity_front[idx_front]  # (N, 3)
        gravity_back  = self._gravity_back[idx_back]
        gravity_targets = torch.where(self._use_back.unsqueeze(1), gravity_back, gravity_front)

        # sum of squares used only for pose-gating
        sq_error = torch.sum(torch.square(joint_pos - targets), dim=1)  # (N,)

        # look up the per-keyframe threshold for each env's current index
        thresh_front = self._thresholds_front[idx_front]  # (N,)
        thresh_back  = self._thresholds_back[idx_back]    # (N,)
        threshold = torch.where(self._use_back, thresh_back, thresh_front)

        # advance envs that are within their current keyframe's threshold
        max_idx = torch.where(self._use_back,
                              torch.full_like(self._kf_idx, max_back),
                              torch.full_like(self._kf_idx, max_front))
        advance = (sq_error < threshold) & (self._kf_idx < max_idx)
        self._kf_idx[advance] += 1

        # from paper: reward = joint angle score * gravity score
        # joint angle score: clip(1 - ||delta_q||_2 / pi, 0, 1)  — 1=perfect, 0=pi error
        import math
        joint_error_norm = torch.sqrt(sq_error + 1e-8)
        jae_score = torch.clamp(1.0 - joint_error_norm / math.pi, 0.0, 1.0)

        # gravity score: clip(1 - angle / (pi/2), 0, 1)  — 1=aligned, 0=pi/2 off
        g_actual = asset.data.projected_gravity_b  # (N, 3)
        g_actual_n = g_actual / (torch.norm(g_actual, dim=1, keepdim=True) + 1e-6)
        cos_sim = torch.clamp(torch.sum(g_actual_n * gravity_targets, dim=1), -1.0, 1.0)
        angle = torch.acos(cos_sim)
        ge_score = torch.clamp(1.0 - angle / (math.pi / 2), 0.0, 1.0)

        return jae_score * ge_score