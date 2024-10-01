# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with JaxRL

Since the current JaxRL does wrapper not support buffers living on GPU directly,
we recommend using num_envs=1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an online RL agent with JaxRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to the checkpoint directory to load.")
parser.add_argument("--checkpoint_step", type=int, default=None, help="Step of the checkpoint to load.")

# append JaxRL cli arguments
cli_args.add_jaxrl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
from datetime import datetime

from omni.isaac.lab.envs import (
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
# from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV
from omni.isaac.groundcontrol_tasks.utils.wrappers.jaxrl import JaxrlEnvWrapper
from jaxrl.agents import SACLearner, TD3Learner, IQLLearner, BCLearner
from typing import Dict, Any, Optional, Union
import orbax.checkpoint as ocp

## TODO: Put somewhere else
def flatten_config(cfg: Dict[str, Any], prefix: Optional[str] = '') -> Dict[str, Any]:
    flat_config = {}
    for key, value in cfg.items():
        if prefix is None:
            new_prefix = None
            flat_key = key
        else:
            new_prefix = f"{prefix}{key}." if prefix else f"{key}."
            flat_key = new_prefix[:-1]
        
        if isinstance(value, dict):
            flat_config.update(flatten_config(value, new_prefix))
        else:
            flat_config[flat_key] = value
    return flat_config

def to_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # This function is not needed for regular dictionaries as they are already in dict form
    # But we'll keep it for consistency, and it can be used for deep copying
    return {k: v for k, v in cfg.items()}

def get_flat_config(cfg: Dict[str, Any], use_prefix: bool = True) -> Dict[str, Any]:
    return flatten_config(cfg, '' if use_prefix else None)

def get_jaxrl_entry_point(algorithm: str = "sac"):
    if algorithm.lower() == "sac":
        return "jaxrl_sac_cfg_entry_point"
    elif algorithm.lower() == "redq":
        return "jaxrl_redq_cfg_entry_point"
    elif algorithm.lower() == "droq":
        return "jaxrl_droq_cfg_entry_point"
    elif algorithm.lower() == "td3":
        return "jaxrl_td3_cfg_entry_point"
    elif algorithm.lower() == "iql":
        return "jaxrl_iql_cfg_entry_point"
    elif algorithm.lower() == "bc":
        return "jaxrl_bc_cfg_entry_point"
    elif algorithm.lower() == "rlpd_sac":
        return "jaxrl_rlpd_sac_cfg_entry_point"
    elif algorithm.lower() == "rlpd_redq":
        return "jaxrl_rlpd_redq_cfg_entry_point"
    elif algorithm.lower() == "rlpd_droq":
        return "jaxrl_rlpd_droq_cfg_entry_point"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def get_learner(learner_name: str, seed: int, observation_space, action_space, **kwargs):
    if (
        (learner_name.lower() == "sac") or
        (learner_name.lower() == "redq") or
        (learner_name.lower() == "droq") or
        (learner_name.lower() == "rlpd_sac") or
        (learner_name.lower() == "rlpd_redq") or
        (learner_name.lower() == "rlpd_droq")
        # You can continue adding more conditions as needed
    ):
        return SACLearner.create(seed, observation_space, action_space, **kwargs)
    elif learner_name.lower() == "td3":
        return TD3Learner.create(seed, observation_space, action_space, **kwargs)
    elif learner_name.lower() == "bc":
        return BCLearner.create(seed, observation_space, action_space, **kwargs)
    elif learner_name.lower() == "iql":
        return IQLLearner.create(seed, observation_space, action_space, **kwargs)
    else:
        raise ValueError(f"Unknown learner: {learner_name}")
    
def load_jax_model(checkpoint_dir, step=0):
    """
    Load a JAX model saved with Orbax checkpointer.

    Args:
    checkpoint_dir (str): Directory where the checkpoint was saved.
    step (int): The step number of the checkpoint to load. Default is 0.

    Returns:
    The loaded model.
    """

    # Initialize the checkpointer
    options = ocp.CheckpointManagerOptions()
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir, 
        orbax_checkpointer, 
        options
    )

    # Get the latest checkpoint if step is not specified
    if step is None:
        step = checkpoint_manager.latest_step()

    # Load the checkpoint
    loaded_model = checkpoint_manager.restore(step)

    print(f"Loaded model from step {step}")
    return loaded_model

@hydra_task_config(args_cli.task, get_jaxrl_entry_point(args_cli.algorithm))
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    """Train with JaxRL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_jaxrl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # assert args_cli.num_envs == 1, "num_envs must be 1 for online training. Parallel environments not supported yet."

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = args_cli.seed
    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "jaxrl", agent_cfg.algorithm.algorithm_name, agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    # wrap around environment for stable baselines
    env = JaxrlEnvWrapper(env)

    kwargs = get_flat_config(agent_cfg.algorithm.to_dict(), use_prefix=False)
    algorithm_name = kwargs.pop('algorithm_name', 'sac')  # Default to SAC if not specified

    # create agent from jaxrl
    agent = get_learner(algorithm_name, agent_cfg.seed, env.observation_space, env.action_space, **kwargs)

    # load the checkpoint
    pretrained_agent: Union[SACLearner, TD3Learner, IQLLearner, BCLearner] = load_jax_model(args_cli.checkpoint_dir, 
                                                                                            step=args_cli.checkpoint_step)
    
    agent = agent.initialize_pretrained_model(pretrained_agent)

    
    ## It'd be nice to export the policy to onnx/jit similar to rsl_rl

    observation, _ = env.reset()
    # simulate environment
    while simulation_app.is_running():
        # agent stepping
        normalized_action = agent.eval_actions(observation)
        # env stepping
        observation, _, _, _, _ = env.step(normalized_action)
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


'''
Run with:

python source/standalone/workflows/jaxrl/play.py --task Isaac-Velocity-Flat-Unitree-A1-Play-v0 --num_envs 64 --algorithm rlpd_redq --checkpoint_dir /home/mateo/projects/GroundControl/logs/jaxrl/rlpd_redq/unitree_a1_flat/2024-10-01_14-20-02/checkpoints --checkpoint_step 30000

'''
