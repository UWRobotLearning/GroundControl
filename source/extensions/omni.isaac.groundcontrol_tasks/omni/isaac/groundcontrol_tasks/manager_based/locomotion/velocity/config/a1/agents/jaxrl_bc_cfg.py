# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV

from omni.isaac.groundcontrol_tasks.utils.wrappers.jaxrl import (
    BCRunnerConfig,
    BCPolicyConfig,
    BCAlgorithmConfig,
)

@configclass
class UnitreeA1FlatBCRunnerCfg(BCRunnerConfig):
    experiment_name = "unitree_a1_flat"
    log_interval = 1000
    save_interval = 100000
    eval_episodes = 1
    eval_interval = 10000#5000
    checkpoint_model = True
    save_video = True
    video_interval = 1000
    max_iterations = int(1e6)
    batch_size = 256

    algorithm = BCAlgorithmConfig(
        actor_lr = 3e-4,
        warmup_steps = 1000,
        decay_steps = 1000000,
        weight_decay = None,
        distr = "normal",
        policy = BCPolicyConfig(
            hidden_dims = (128, 128, 128),
            dropout_rate = None,
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
