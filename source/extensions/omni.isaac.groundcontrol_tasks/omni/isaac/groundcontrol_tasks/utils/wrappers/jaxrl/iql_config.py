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

## TODO: The proper way to do this seems to be to define the interface here and then populate it within each agent
## withn each "robot" folder. For now, I'm gonna keep this populated with "default" values but should clean this up
## in the future.

@configclass
class IQLPolicyConfig:
    hidden_dims: Tuple[int] = (128, 128, 128)
    actor_weight_decay: Optional[float] = None
    critic_weight_decay: Optional[float] = None
    value_weight_decay: Optional[float] = None
    critic_layer_norm: bool = False
    value_layer_norm: bool = False
    num_qs: int = 2
    num_min_qs: Optional[int] = None
    num_vs: int = 1
    num_min_vs: Optional[int] = None
    use_tanh_normal: bool = True
    state_dependent_std: bool = False

    # def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
    #     return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return self.to_dict()

@configclass
class IQLAlgorithmConfig:
    algorithm_name: str = "iql"
    actor_lr: float = 1e-3
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.8
    temperature: float = 0.1

    policy: IQLPolicyConfig = IQLPolicyConfig()

    # def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
    #     return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return self.to_dict()


@configclass
class IQLRunnerConfig:
    algorithm: IQLAlgorithmConfig = IQLAlgorithmConfig()

    seed: int = 42
    env_name: str = MISSING #"HalfCheetah-v4"
    dataset_path: str = MISSING #"buffer.npz"
    experiment_name: str = "iql"  # Used for wandb and project folder logging
    run_name: Optional[str] = None  # If not None, used to save logs in dir date_run_name
    save_dir: str = "checkpoints"
    log_interval: int = 1000
    save_interval: int = 100000
    eval_episodes: int = 10  # Not currently used
    eval_interval: int = 5000  # Not currently used
    checkpoint_model: bool = True
    save_video: bool = False
    wandb: bool = True
    video_interval: int = 20
    tqdm: bool = True
    episode_buffer_len: int = 100  # Window of previous episodes to consider for average rewards/lengths
    filter_percentile: Optional[float] = None
    filter_threshold: Optional[float] = None
    clip_to_eps: bool = True  ## Replace [-1, 1] with [-1-eps, 1+eps] in dataset
    eps: float = 1e-5
    
    max_iterations: int = int(1e6)
    batch_size: int = 256

    resume: bool = MISSING
    load_run: str = MISSING
    checkpoint: str = MISSING
    logger: str = MISSING
    log_project_name: str = MISSING
    wandb_project: str = MISSING
    neptune_project: str = MISSING

    num_pretraining_steps: int = MISSING  ## Used for offline to online
    replay_buffer_size: int = MISSING  ## Used for offline to online


    # def get_flat_config(self, use_prefix: bool = True) -> Dict[str, Any]:
    #     return flatten_config_dataclass(self, '' if use_prefix else None)

    def to_dict(self) -> Dict[str, Any]:
        return self.to_dict()