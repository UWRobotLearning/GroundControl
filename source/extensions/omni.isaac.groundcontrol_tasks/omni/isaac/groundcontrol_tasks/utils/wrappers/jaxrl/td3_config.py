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

@configclass
class TD3PolicyConfig:
    hidden_dims: Tuple[int] = (128, 128, 128)
    num_qs: int = 2
    num_min_qs: Optional[int] = None
    critic_layer_norm: bool = False
    critic_dropout_rate: Optional[float] = None

    # def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
    #     return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return self.to_dict()

@configclass
class TD3AlgorithmConfig:
    algorithm_name: str = "td3"
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    exploration_noise: float = 0.1
    target_policy_noise: float = 0.2
    target_policy_noise_clip: float = 0.5
    actor_delay: int = 2

    policy: TD3PolicyConfig = TD3PolicyConfig()

    # def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
    #     return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return self.to_dict()


@configclass
class TD3RunnerConfig:
    algorithm: TD3AlgorithmConfig = TD3AlgorithmConfig()

    seed: int = 42
    env_name: str = MISSING #"HalfCheetah-v4"
    experiment_name: str = "td3"  # Used for wandb and project folder logging
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
    start_training: int = 10000
    batch_size: int = 256
    utd_ratio: int = 1
    reset_param_interval: Optional[int] = None #int(1e4)  # Not currently integrated

    resume: bool = MISSING
    load_run: str = MISSING
    checkpoint: str = MISSING
    logger: str = MISSING
    log_project_name: str = MISSING
    wandb_project: str = MISSING
    neptune_project: str = MISSING

    # def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
    #     return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return self.to_dict()