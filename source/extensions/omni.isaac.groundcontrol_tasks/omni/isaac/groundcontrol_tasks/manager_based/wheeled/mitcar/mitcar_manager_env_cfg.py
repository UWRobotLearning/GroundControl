import torch
import torch.nn as nn

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.utils import configclass

################
# REWARDS
################

def dist2goal(env, target):
    root_pos = mdp.root_pos_w(env)
    # target = env.scene.env_origins + torch.tensor(target, dtype=root_pos.dtype, device=root_pos.device)
    target = torch.tensor(target, dtype=root_pos.dtype, device=root_pos.device)
    dist = torch.norm(target - root_pos, dim=-1)
    return dist


def upright_penalty(env, thresh_deg):
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    up_dot = torch.rad2deg(torch.arccos(up_dot))
    penalty = torch.where(up_dot > thresh_deg, up_dot - thresh_deg, 0.)
    return penalty


@configclass
class RewardsCfg:

    """Reward terms for the MDP."""
    # (1) Primary task: Distance to target
    dist2goal = RewTerm(
        func=dist2goal,
        weight=-1.0,
        params={"target": [3., 2., 0.]},
    )

    # (2) Upright
    upright = RewTerm(
        func=upright_penalty,
        weight=-1.0,
        params={"thresh_deg": 30.},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )

@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""
    pass

##################################################
############## ENVIRONMENT CONFIGS ###############
##################################################

from . import common
from .common.rl_env import MITCarRLCommonCfg
from .common.scenes import MITCarRoughSceneCfg, MITCarFlatSceneCfg, MITCarBaseSceneCfg

####### RL Environment #######

@configclass
class MITCarRLEnvCfg(MITCarRLCommonCfg):
    """Configuration for the cartpole environment."""

    seed: int = 42

    # # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # No command generator
    commands: common.NoCommandsCfg = common.NoCommandsCfg()


    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # viewer settings
        self.viewer.eye = [11., 0.0, 14.0]
        self.viewer.lookat = [2.0, 0.0, 0.]

        # termination settings
        self.episode_length_s = 5
        # self.max_episode_length = 512

        # Scene settings
        self.scene = MITCarRoughSceneCfg(
                num_envs=self.num_envs, env_spacing=self.env_spacing,
            )

        # Set seed for terrain generator
        self.scene.terrain.terrain_generator = self.scene.terrain.terrain_generator.replace(seed=self.seed)

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
    """No terminations for the MDP."""
    pass

@configclass
class MITCarPlayEnvCfg(MITCarRLEnvCfg):
    """Configuration for the cartpole environment."""

    terminations = NoTerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # viewer settings
        self.viewer.eye = [11., 0.0, 14.0]
        self.viewer.lookat = [2.0, 0.0, 0.]

        # Scene settings
        self.scene:MITCarBaseSceneCfg = MITCarFlatSceneCfg(
                num_envs=self.num_envs, env_spacing=self.env_spacing
            )


####### IRL #######

def _compute_learned_reward(env, model):
    obs = env.observation_manager.compute()['reward']
    rew = model(obs).squeeze(dim=-1)
    return rew

@configclass
class LearnedRewardsCfg:
    """Learned reward terms for the MDP."""
    reward:RewTerm = None


def root_2dpos_w(env):
    return mdp.root_pos_w(env)[..., :2]


@configclass
class MITCarIRLEnvCfg(MITCarRLEnvCfg):

    rew_model: nn.Module = None

    def __post_init__(self):
        super().__post_init__()
        self.rewards = LearnedRewardsCfg()
        rewterm = lambda env: _compute_learned_reward(env, self.rew_model)
        self.rewards.reward = RewTerm(
            func=rewterm,
            weight=1.0,
        )


###########################################
############## ENVIRONMENTS ###############
###########################################

# TODO
# from wheeled_gym.tasks.wrappers.irl_wrapper import IRLWrapper
# class MITCarIRLWrapper(IRLWrapper):
#     """MIT Car environment for IRL."""

#     # def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
#     def __init__(self, env):
#         super().__init__(env)
#         self._episode_length_s_persistent = self.cfg.episode_length_s

#     def irl_mode(self, ep_steps):
#         super().irl_mode()
#         self.env.cfg.episode_length_s = ep_steps

#     def rl_mode(self):
#         super().rl_mode()
#         self.env.cfg.episode_length_s = self._episode_length_s_persistent

#     # TODO: Use ClipActionWrapper to clip actions (which uses env.action_space.lower/upper
#     # which should be set in the config)
#     def step(self, action):
#         action = torch.clip(action, -1., 1.)
#         obs, rew, term, trunc, info = super().step(action)

#         return obs, rew, term, trunc, info
