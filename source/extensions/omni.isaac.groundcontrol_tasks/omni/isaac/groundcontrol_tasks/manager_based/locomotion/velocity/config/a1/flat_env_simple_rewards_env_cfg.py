# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from .rough_env_cfg import UnitreeA1RoughEnvCfg
# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV


@configclass
class UnitreeA1FlatEnvSimpleRewardsCfg(UnitreeA1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards that are not task related
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.dof_pos_limits.weight = 0.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class UnitreeA1FlatEnvSimpleRewardsCfg_PLAY(UnitreeA1FlatEnvSimpleRewardsCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None