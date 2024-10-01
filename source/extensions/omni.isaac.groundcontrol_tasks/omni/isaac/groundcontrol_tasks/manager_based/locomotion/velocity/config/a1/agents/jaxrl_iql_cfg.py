# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV

from omni.isaac.groundcontrol_tasks.utils.wrappers.jaxrl import (
    IQLRunnerConfig,
    IQLPolicyConfig,
    IQLAlgorithmConfig,
)

@configclass
class UnitreeA1FlatIQLRunnerCfg(IQLRunnerConfig):
    experiment_name = "unitree_a1_flat"
    log_interval = 1_000
    save_interval = 100_000
    eval_episodes = 1
    eval_interval = 10_000
    checkpoint_model = True
    save_video = True
    video_interval = 1_000
    max_iterations = 1_000_000
    batch_size = 256

    ## Only used for offline to online
    num_pretraining_steps = 1_000_000
    replay_buffer_size = 2_000_000

    algorithm = IQLAlgorithmConfig(
        actor_lr = 1e-3,
        critic_lr = 3e-4,
        value_lr = 3e-4,
        discount = 0.99,
        tau = 0.005,
        expectile = 0.8,
        temperature = 0.1,
        policy = IQLPolicyConfig(
            hidden_dims = (128, 128, 128),
            actor_weight_decay = None,
            critic_weight_decay = None,
            value_weight_decay = None,
            critic_layer_norm = False,
            value_layer_norm = False,
            num_qs = 2,
            num_min_qs = None,
            num_vs = 1,
            num_min_vs = None,
            use_tanh_normal = False,
            state_dependent_std = False,
        )
    )


# @configclass
# class UnitreeA1FlatIQLRunnerCfg(UnitreeA1RoughIQLRunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         self.max_iterations = 1500
#         self.experiment_name = "unitree_a1_flat"
#         self.policy.actor_hidden_dims = [128, 128, 128]
#         self.policy.critic_hidden_dims = [128, 128, 128]
