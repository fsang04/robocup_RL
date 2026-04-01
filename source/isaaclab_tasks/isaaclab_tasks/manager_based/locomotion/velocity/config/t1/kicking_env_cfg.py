'''
Define unified dense rewards such that policy learns:
approach -> position itself close to ball -> kick
# Approach
ball_approach = RewTerm(func=mdp.neg_distance_to_ball, weight=0.5)

# Contact quality
ball_velocity_toward_goal = RewTerm(func=mdp.ball_vel_toward_target, weight=2.0)

# Stability (already in your base cfg)
# lin_vel_z_l2, ang_vel_xy_l2, flat_orientation_l2

# Penalize kicking leg contacting ground while swinging
swing_foot_contact = RewTerm(func=mdp.undesired_contacts, weight=-0.5, ...)
'''

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .flat_env_cfg import T1FlatEnvCfg

@configclass
class T1KickingEnvCfg(T1FlatEnvCfg):
