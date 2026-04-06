'''
This script loads a pretrained walking checkpoint and pads the first layer weights to account
for the 9 new observations added in the kicking task:
  - ball_pos_relative      (3)
  - ball_vel_relative      (3)
  - goal_pos_relative_to_ball (3)

Original walking obs dim: 48  →  kicking obs dim: 57
'''

import torch

ckpt = torch.load("logs/rsl_rl/t1_kicking/pretrained_walking/model_1499.pt", map_location="cpu")

# Pad the first layer weight with 9 zero columns (for the 9 new kicking obs)
w = ckpt["actor_state_dict"]["mlp.0.weight"]  # [256, 48]
pad = torch.zeros(w.shape[0], 9)
ckpt["actor_state_dict"]["mlp.0.weight"] = torch.cat([w, pad], dim=1)  # [256, 57]

# Pad the critic's first layer as well (critic also takes obs as input)
w_critic = ckpt["critic_state_dict"]["mlp.0.weight"]  # [256, 48]
pad_critic = torch.zeros(w_critic.shape[0], 9)
ckpt["critic_state_dict"]["mlp.0.weight"] = torch.cat([w_critic, pad_critic], dim=1)  # [256, 57]

# Drop optimizer state — it has stale shapes from the old obs size and will cause errors
ckpt["optimizer_state_dict"] = {}

torch.save(ckpt, "logs/rsl_rl/t1_kicking/pretrained_walking/model_1499_padded.pt")
print("Saved padded checkpoint.")
