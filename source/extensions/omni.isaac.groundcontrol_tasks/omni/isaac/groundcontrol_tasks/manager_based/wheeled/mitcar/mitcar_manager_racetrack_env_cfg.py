import torch

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.utils import configclass

################
# REWARDS
################

def track_progress_rate(env):
    '''Estimate track progress by positive z-axis angular velocity around the environment'''
    root_ang_vel = mdp.root_ang_vel_w(env)
    progress_rate = root_ang_vel[..., 2]
    return progress_rate


def upright_penalty(env, thresh_deg):
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    up_dot = torch.rad2deg(torch.arccos(up_dot))
    penalty = torch.where(up_dot > thresh_deg, up_dot - thresh_deg, 0.)
    return penalty

def forward_vel_rew(env):
    lin_vel = mdp.base_lin_vel(env)
    return lin_vel[..., 0]

def falling_penalty(env):
    pos = mdp.root_pos_w(env)
    return torch.where(pos[..., 2] < 0.1, 100.0, 0.0)

def forward_wheel_spin(env):
    joint_vel = mdp.joint_vel(env)
    return torch.sum(joint_vel[..., [0,1,4,5]], dim=-1)

@configclass
class RacetrackRewardsCfg:

    """Reward terms for the MDP."""
    # (1) Progress around track
    # progress = RewTerm(
    #     func=track_progress_rate,
    #     weight=1.0,
    # )

    # (2) Upright
    upright = RewTerm(
        func=upright_penalty,
        weight=-1.0,
        params={"thresh_deg": 30.},
    )

    # (3) Forward velocity
    forward_vel = RewTerm(
        func=forward_vel_rew,
        weight=20.,
    )

    # (4) Falling penalty
    falling_penalty = RewTerm(
        func=falling_penalty,
        weight=-10.0,
    )

    # (5) Forward wheel spin
    forward_wheel_spin = RewTerm(
        func=forward_wheel_spin,
        weight=1.,
    )


@configclass
class RacetrackTerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.},
    )
    # (3) Stuck TODO
    # stuck = DoneTerm(
    #     func=mdp.root_lin_vel_below_threshold,
    #     params={"threshold": 0.01},
    # )

@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""
    pass

#####################################
############## EVENTS ###############
#####################################

from omni.isaac.lab.managers import EventTermCfg as EventTerm
from .utils import reset_root_state_from_terrain_points

@configclass
class RacetrackEventsCfg:
    """Configuration for the events."""
    reset_root_state_from_terrain_points = EventTerm(
        func=reset_root_state_from_terrain_points,
        mode="reset",
    )


##################################################
############## ENVIRONMENT CONFIGS ###############
##################################################

from . import common
from .common.rl_env import MITCarRLCommonCfg
from .common.scenes import MITCarRacetrackSceneCfg

####### RL Environment #######

@configclass
class MITCarRacetrackRLEnvCfg(MITCarRLCommonCfg):
    """Configuration for the cartpole environment."""

    seed: int = 42

    # Reset config
    events: RacetrackEventsCfg = RacetrackEventsCfg()

    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RacetrackRewardsCfg = RacetrackRewardsCfg()
    terminations: RacetrackTerminationsCfg = RacetrackTerminationsCfg()
    # No command generator
    commands: common.NoCommandsCfg = common.NoCommandsCfg()


    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # viewer settings
        self.viewer.eye = [11., -20.0, 14.0]
        self.viewer.lookat = [2.0, 0.0, 0.]

        # Terminations config
        self.episode_length_s = 30
        # self.max_episode_length = 512

        # Scene settings
        self.scene:MITCarRacetrackSceneCfg = MITCarRacetrackSceneCfg(
                num_envs=self.num_envs, env_spacing=self.env_spacing,
            )

        # HACK to gain control of ordering of joints through JOINT_NAMES
        # observation terms (order preserved)
        # self.observations.joint_pos_rel.func = lambda x : self.observations.joint_pos_rel.func(x, joint_order_asset_cfg)
        # self.observations.joint_vel_rel.func = lambda x : self.observations.joint_vel_rel.func(x, joint_order_asset_cfg)
        # self.observations.pos_w.func = lambda x : self.observations.pos_w.func(x, joint_order_asset_cfg)
        # self.observations.base_lin_vel.func = lambda x : self.observations.base_lin_vel.func(x, joint_order_asset_cfg)
        # self.observations.lin_vel_w.func = lambda x : self.observations.lin_vel_w.func(x, joint_order_asset_cfg)
        # self.observations.base_ang_vel.func = lambda x : self.observations.base_ang_vel.func(x, joint_order_asset_cfg)


####### Base Environment #######

@configclass
class NoTerminationsCfg:
    pass

@configclass
class MITCarRacetrackPlayEnvCfg(MITCarRacetrackRLEnvCfg):

    # play_duration_s: float = 60.
    terminations = NoTerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # self.episode_length_s = self.play_duration_s

