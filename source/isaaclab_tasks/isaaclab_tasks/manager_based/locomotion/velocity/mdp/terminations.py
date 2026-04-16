# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.sensors import ContactSensor


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


# Add a success termination for getup to reward reaching final keyframe and standing up
class GetUpSuccess(ManagerTermBase):
    """Terminate (successfully) when robot is upright, stable, feet on ground, and knees extended."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._upright_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

        # resolve knee joint indices once at init
        asset = env.scene["robot"]
        joint_names = asset.joint_names
        self._knee_ids = [joint_names.index(n) for n in ("Left_Knee_Pitch", "Right_Knee_Pitch")]

        # resolve foot body ids directly from the contact sensor
        contact_sensor = cast(ContactSensor, env.scene.sensors["contact_forces"])
        self._foot_body_ids = contact_sensor.find_bodies(".*foot_link")[0]

    def reset(self, env_ids):
        self._upright_steps[env_ids] = 0

    def __call__(
        self,
        env,
        min_upright_steps: int = 60,
        upright_threshold: float = 0.85,
        knee_straight_threshold: float = 0.3,
        foot_contact_threshold: float = 1.0,
    ) -> torch.Tensor:
        asset = env.scene["robot"]

        # trunk upright check
        up_proj = -asset.data.projected_gravity_b[:, 2]
        is_upright = up_proj > upright_threshold

        # low velocity check
        low_vel = asset.data.root_lin_vel_b.norm(dim=1) < 0.3

        # both knees near fully extended (0 rad); abs value so works for +/- deviations
        knee_pos = asset.data.joint_pos[:, self._knee_ids]  # (N, 2)
        knees_straight = (knee_pos.abs() < knee_straight_threshold).all(dim=1)

        # both feet in contact with ground
        contact_sensor = cast(ContactSensor, env.scene.sensors["contact_forces"])
        foot_forces = (
            contact_sensor.data.net_forces_w_history[:, :, self._foot_body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
        )  # (N, num_feet)
        both_feet_contact = (foot_forces > foot_contact_threshold).all(dim=1)

        # final success termination check: upright trunk, low vel, and standing (straight knees + feet touching ground)
        stable_and_upright = is_upright & low_vel & knees_straight & both_feet_contact
        self._upright_steps[stable_and_upright] += 1
        self._upright_steps[~stable_and_upright] = 0

        return self._upright_steps >= min_upright_steps
