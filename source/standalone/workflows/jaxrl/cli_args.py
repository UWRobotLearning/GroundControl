# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV

if TYPE_CHECKING:
    # from omni.isaac.groundcontrol_tasks.utils.wrappers.jaxrl import RslRlOnPolicyRunnerCfg  ## TODO: Replace PPO config with JaxRL SAC config
    from typing import Union
    from omni.isaac.groundcontrol_tasks.utils.wrappers.jaxrl import SACRunnerConfig, TD3RunnerConfig, IQLRunnerConfig, BCRunnerConfig

## TODO: For now, only handle IQL
## TODO: I think one good way to handle this is to have a custom type that includes all of the JAXRL configs, with the same
## parameters to be populated by the add_jaxrl_args function. 

def add_jaxrl_args(parser: argparse.ArgumentParser):
    """Add JaxRL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    ALGORITHM_CHOICES = [
        "iql",
        "sac",
        "td3",
        "bc",
        "redq",
        "droq",
        "rlpd_sac",
        "rlpd_redq",
        "rlpd_droq",
        # Add more algorithms here
    ]

    # create a new argument group
    arg_group = parser.add_argument_group("jaxrl", description="Arguments for JaxRL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    ## TODO: Need to change these to match the configs in the runner policy configs
    arg_group.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset to use for training.")
    arg_group.add_argument(
        "--algorithm", type=str, default=None, choices=ALGORITHM_CHOICES, help="Algorithm to use for training."
    )


# def parse_jaxrl_cfg(task_name: str, args_cli: argparse.Namespace) -> IQLRunnerConfig:
#     """Parse configuration for JaxRL agent based on inputs.

#     Args:
#         task_name: The name of the environment.
#         args_cli: The command line arguments.

#     Returns:
#         The parsed configuration for RSL-RL agent based on inputs.
#     """
#     from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry

#     # load the default configuration
#     jaxrl_cfg: IQLRunnerConfig = load_cfg_from_registry(task_name, "jaxrl_cfg_entry_point")
#     jaxrl_cfg = update_jaxrl_cfg(jaxrl_cfg, args_cli)
#     return jaxrl_cfg

def parse_jaxrl_cfg(task_name: str, args_cli: argparse.Namespace) -> Union[IQLRunnerConfig, BCRunnerConfig, SACRunnerConfig, TD3RunnerConfig]:
    """Parse configuration for JaxRL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for JaxRL agent based on inputs.
    """
    from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry

    # Determine which configuration to load based on the algorithm
    if args_cli.algorithm.lower() == "iql":
        cfg_entry_point = "jaxrl_iql_cfg_entry_point"
    elif args_cli.algorithm.lower() == "bc":
        cfg_entry_point = "jaxrl_bc_cfg_entry_point"
    elif args_cli.algorithm.lower() == "sac":
        cfg_entry_point = "jaxrl_sac_cfg_entry_point"
    elif args_cli.algorithm.lower() == "td3":
        cfg_entry_point = "jaxrl_td3_cfg_entry_point"
    else:
        raise ValueError(f"Unknown algorithm: {args_cli.algorithm}")

    # Load the default configuration
    jaxrl_cfg = load_cfg_from_registry(task_name, cfg_entry_point)
    jaxrl_cfg = update_jaxrl_cfg(jaxrl_cfg, args_cli)
    return jaxrl_cfg


def update_jaxrl_cfg(agent_cfg: Union[IQLRunnerConfig, BCRunnerConfig, SACRunnerConfig, TD3RunnerConfig], args_cli: argparse.Namespace):
    """Update configuration for JaxRL agent based on inputs.

    Args:
        agent_cfg: The configuration for JaxRL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for JaxRL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if args_cli.seed is not None:
        agent_cfg.seed = int(args_cli.seed)
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    if args_cli.dataset_path is not None:
        agent_cfg.dataset_path = args_cli.dataset_path


    return agent_cfg