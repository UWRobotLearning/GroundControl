# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV

from omni.isaac.groundcontrol_tasks.utils.wrappers.jaxrl import (
    RLPDSACRunnerConfig,
    RLPDREDQRunnerConfig,
    RLPDDroQRunnerConfig,
    RLPDSACPolicyConfig,
    RLPDSACAlgorithmConfig,
)

@configclass
class UnitreeA1FlatRLPDSACRunnerCfg(RLPDSACRunnerConfig):
    experiment_name = "unitree_a1_flat"
    log_interval = 1000
    save_interval = 10000
    eval_episodes = 3
    eval_interval = 10000
    checkpoint_model = True
    checkpoint_buffer = True
    save_video = True
    video_interval = 1000
    max_iterations = int(1e5)
    batch_size = 256
    start_training = 1000
    utd_ratio = 1
    reset_param_interval = None
    offline_ratio = 0.5

    algorithm = RLPDSACAlgorithmConfig(
        algorithm_name = "rlpd_sac",
        actor_lr = 3e-4,
        critic_lr = 3e-4,
        temp_lr = 3e-4,
        discount = 0.99,
        tau = 0.005,
        init_temperature = 1.0,
        target_entropy = None,
        backup_entropy = True,
        policy = RLPDSACPolicyConfig(
            hidden_dims = (128, 128, 128),
            num_qs = 2,
            num_min_qs = None,
            critic_weight_decay = None,
            critic_dropout_rate = None,
            critic_layer_norm = False,
            use_critic_resnet = False,
            use_pnorm = False,
        )
    )

@configclass
class UnitreeA1FlatRLPDDREDQRunnerCfg(RLPDREDQRunnerConfig):
    experiment_name = "unitree_a1_flat"
    log_interval = 1000
    save_interval = 10000
    eval_episodes = 3
    eval_interval = 10000
    checkpoint_model = True
    checkpoint_buffer = True
    save_video = True
    video_interval = 1000
    max_iterations = int(1e5)
    batch_size = 256
    start_training = 1000
    utd_ratio = 20
    reset_param_interval = None
    offline_ratio = 0.5

    algorithm = RLPDSACAlgorithmConfig(
        algorithm_name = "rlpd_redq",
        actor_lr = 3e-4,
        critic_lr = 3e-4,
        temp_lr = 3e-4,
        discount = 0.99,
        tau = 0.005,
        init_temperature = 1.0,
        target_entropy = None,
        backup_entropy = True,
        policy = RLPDSACPolicyConfig(
            hidden_dims = (128, 128, 128),
            num_qs = 10,
            num_min_qs = 2,
            critic_weight_decay = None,
            critic_dropout_rate = None,
            critic_layer_norm = True,
            use_critic_resnet = False,
            use_pnorm = False,
        )
    )


@configclass
class UnitreeA1FlatRLPDDroQRunnerCfg(RLPDDroQRunnerConfig):
    experiment_name = "unitree_a1_flat"
    log_interval = 1000
    save_interval = 10000
    eval_episodes = 3
    eval_interval = 10000
    checkpoint_model = True
    checkpoint_buffer = True
    save_video = True
    video_interval = 1000
    max_iterations = int(1e5)
    batch_size = 256
    start_training = 1000
    utd_ratio = 20
    reset_param_interval = None
    offline_ratio = 0.5
    algorithm = RLPDSACAlgorithmConfig(
        algorithm_name = "rlpd_droq",
        actor_lr = 3e-4,
        critic_lr = 3e-4,
        temp_lr = 3e-4,
        discount = 0.99,
        tau = 0.005,
        init_temperature = 0.1,
        target_entropy = None,
        backup_entropy = True,
        policy = RLPDSACPolicyConfig(
            hidden_dims = (128, 128, 128),
            num_qs = 2,
            num_min_qs = None,
            critic_weight_decay = None,
            critic_dropout_rate = 0.01,
            critic_layer_norm = True,
            use_critic_resnet = False,
            use_pnorm = False,
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
