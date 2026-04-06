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
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab_tasks.manager_based.locomotion.velocity.config.t1.rough_env_cfg import T1Rewards
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, MySceneCfg, RewardsCfg, ObservationsCfg

from .flat_env_cfg import T1FlatEnvCfg


@configclass
class KickingSceneCfg(MySceneCfg):
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.41),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.11)),
    )


@configclass
class T1KickingRewards(T1Rewards):
    """Reward terms for the MDP."""

    # Rewards for kicking task
    ball_approach = RewTerm(
        func=mdp.approach_ball_reward,
        weight=2.0,
        params={"std": 1.0},
    )
    ball_velocity_toward_goal = RewTerm(
        func=mdp.ball_vel_toward_target, weight=2.0
    )
    swing_foot_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="right_foot_link"),
            "threshold": 1.0,
        },
    )
    both_feet_in_air = RewTerm(
        func=mdp.both_feet_in_air,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*foot_link"]),
            "threshold": 1.0,
        },
    )
    ball_progress = RewTerm(
        func=mdp.BallProgressReward,
        weight=1.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "target_pos": (10.0, 0.0, 0.0),
        },
    )

    # add reward for aligning towards goal (orientation towards ball->goal)


@configclass
class T1KickingObservations(ObservationsCfg):
    """Specific kicking observation specifications."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        ball_pos = ObsTerm(func=mdp.ball_pos_relative)
        ball_vel = ObsTerm(func=mdp.ball_vel_relative)
        goal_pos_from_ball = ObsTerm(func=mdp.goal_pos_relative_to_ball)

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class T1KickingEnvCfg(T1FlatEnvCfg):
    # override LocomotionVelocityRoughEnvCfg's scene config with KickingSceneCfg 
    scene: KickingSceneCfg = KickingSceneCfg(num_envs=4096, env_spacing=2.5)

    # override LocomotionVelocityRoughEnvCfg's T1Rewards with T1KickingRewards
    rewards: T1KickingRewards = T1KickingRewards() 
    observations: T1KickingObservations = T1KickingObservations()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Rewards -- adjust / zero-out inherited locomotion rewards as needed
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.feet_air_time.weight = 0.25 
        self.rewards.track_lin_vel_xy_exp.weight = 0.1
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.joint_deviation_hip.weight = -0.5 # stronger to keep legs from crossing
        self.rewards.feet_slide.weight = -0.2 # add foot sliding penalty to encourage stable footing during kick 
        self.rewards.track_ang_vel_z_exp.weight = 0.0 # zero out 

        # Commands -- adjust command ranges so doesn't conflict with ball approach
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0) # less turning needed


class T1KickingEnvCfg_PLAY(T1KickingEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None