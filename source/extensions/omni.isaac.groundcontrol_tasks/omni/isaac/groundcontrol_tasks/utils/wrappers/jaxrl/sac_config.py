# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab.utils import configclass

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV

# from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional


## Config for family of SAC algorithms. Includes SAC, REDQ, DroQ, RLPD-SAC, RLPD-DroQ, RLPD-REDQ
## TODO: The proper way to do this seems to be to define the interface here and then populate it within each agent
## withn each "robot" folder. For now, I'm gonna keep this populated with "default" values but should clean this up
## in the future.

@configclass
class SACPolicyConfig:
    hidden_dims: Tuple[int] = (128, 128, 128)
    num_qs: int = 2
    num_min_qs: Optional[int] = None  ## Used for REDQ/DroQ
    critic_layer_norm: bool = False  ## Used in RLPD

    critic_weight_decay: Optional[float] = None
    critic_dropout_rate: Optional[float] = None  ## Used in DroQ
    use_pnorm: bool = False  ## Unsure why it's here, seems to always be False. Leaving it here for completeness
    use_critic_resnet: bool = False

    # def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
    #     return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return self.to_dict()

@configclass
class SACAlgorithmConfig:
    algorithm_name: str = "sac"
    actor_lr: float = 3e-4  
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    init_temperature: float = 1.0
    target_entropy: Optional[float] = None
    backup_entropy: bool = True

    policy: SACPolicyConfig = SACPolicyConfig()

    # def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
    #     return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return self.to_dict()


@configclass
class SACRunnerConfig:
    algorithm: SACAlgorithmConfig = SACAlgorithmConfig()

    seed: int = 42
    env_name: str = "HalfCheetah-v4"
    experiment_name: str = "sac"  # Used for wandb and project folder logging
    run_name: Optional[str] = None  # If not None, used to save logs in dir date_run_name
    save_dir: str = "checkpoints"
    log_interval: int = 1000
    save_interval: int = 100000
    eval_episodes: int = 10  # Not currently used
    eval_interval: int = 5000  # Not currently used
    checkpoint_model: bool = True
    checkpoint_buffer: bool = True
    save_video: bool = False
    # wandb: bool = True
    video_interval: int = 20
    tqdm: bool = True
    episode_buffer_len: int = 100  # Window of previous episodes to consider for average rewards/lengths

    max_iterations: int = int(1e6)
    start_training: int = 1000
    # offline_ratio: float = 0.5  # This is used in hybrid rl only
    batch_size: int = 256
    utd_ratio: int = 1
    reset_param_interval: Optional[int] = None #int(1e4)  # Not currently integrated

    # def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
    #     return flatten_config_dataclass(self, '' if use_prefix else None)

    resume: bool = MISSING
    load_run: str = MISSING
    checkpoint: str = MISSING
    logger: str = MISSING
    log_project_name: str = MISSING
    wandb_project: str = MISSING
    neptune_project: str = MISSING

    def to_dict(self) -> Dict[str, Any]:
        return self.to_dict()


@configclass
class REDQRunnerConfig(SACRunnerConfig):
    algorithm: SACAlgorithmConfig = SACAlgorithmConfig(
        algorithm_name="redq",
        policy=SACPolicyConfig(
            num_qs=10,  ## This is what makes it REDQ
            num_min_qs=2,  ## This is what makes it REDQ
            critic_layer_norm=True,  ## REDQ paper doesn't have this, but leaving this on for now.
        )
    )
    utd_ratio = 20

@configclass
class DroQRunnerConfig(SACRunnerConfig):
    algorithm: SACAlgorithmConfig = SACAlgorithmConfig(
        algorithm_name="droq",
        init_temperature=0.1,
        policy=SACPolicyConfig(
            critic_dropout_rate=0.01,  ## This is what makes it DroQ
            critic_layer_norm=True,  ## Also part of DroQ paper
        )
    )
    utd_ratio = 20
