# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV

from omni.isaac.groundcontrol_tasks.utils.wrappers.jaxrl import (
    TD3RunnerConfig,
    TD3PolicyConfig,
    TD3AlgorithmConfig,
)

@configclass
class UnitreeA1FlatTD3RunnerCfg(TD3RunnerConfig):
    experiment_name = "unitree_a1_flat"
    log_interval = 1000
    save_interval = 100000
    eval_episodes = 1
    eval_interval = 10000#5000
    checkpoint_model = True
    checkpoint_buffer = True
    save_video = True
    video_interval = 1000
    max_iterations = int(1e6)
    batch_size = 256
    start_training = 10000

    algorithm = TD3AlgorithmConfig(
        algorithm_name = "td3",
        actor_lr = 3e-4,
        critic_lr = 3e-4,
        discount = 0.99,
        tau = 0.005,
        exploration_noise = 0.1,
        target_policy_noise = 0.2,
        target_policy_noise_clip = 0.5,
        actor_delay = 2,
        policy = TD3PolicyConfig(
            hidden_dims = (128, 128, 128),
            critic_layer_norm = False,
            critic_dropout_rate = None,
            num_qs = 2,
            num_min_qs = None,
        )
    )


# @configclass
# class UnitreeA1FlatTD3RunnerCfg(UnitreeA1RoughTD3RunnerCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         self.max_iterations = 1500
#         self.experiment_name = "unitree_a1_flat"
#         self.policy.actor_hidden_dims = [128, 128, 128]
#         self.policy.critic_hidden_dims = [128, 128, 128]
