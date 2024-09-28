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
parser = argparse.ArgumentParser(description="Train an offline RL agent with JaxRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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
from jaxrl.agents import IQLLearner
from jaxrl.data import load_replay_buffer, ReplayBuffer
from typing import Dict, Any, Optional
import tqdm
import jax
import wandb
import torch

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

def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    return_queue = np.zeros((num_episodes, env.num_envs))
    length_queue = np.zeros((num_episodes, env.num_envs))
    for i in range(num_episodes):
        observation, _ = env.reset()
        done = np.full(env.num_envs, False)
        while not done.all():
            action = agent.eval_actions(observation)
            assert not np.isnan(action).any(), "NaN in action"
            observation, rewards, terminated, truncated, info = env.step(action)
            done = terminated | truncated | done
            return_queue[i, ~done] += rewards[~done]
            length_queue[i, ~done] += 1
    return {"return": np.mean(return_queue), "length": np.mean(length_queue)}

def get_jaxrl_entry_point(algorithm: str = "iql"):
    if algorithm.lower() == "iql":
        return "jaxrl_iql_cfg_entry_point"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def get_learner(learner_name: str, seed: int, observation_space, action_space, **kwargs):
    if learner_name.lower() == "iql":
        return IQLLearner.create(seed, observation_space, action_space, **kwargs)
    else:
        raise ValueError(f"Unknown learner: {learner_name}")
    

'''
For reference:

class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)
'''
def expand_buffer_capacity(replay_buffer, new_buffer_capacity, observation_space, action_space):
    """Expand the buffer capacity to new_buffer_capacity"""
    assert replay_buffer._capacity <= new_buffer_capacity, "Buffer capacity is already greater than or equal to new_buffer_capacity"

    expanded_replay_buffer = ReplayBuffer(
        observation_space,
        action_space,
        new_buffer_capacity,
        observation_space,
    )

    expanded_replay_buffer.dataset_dict["observations"][:replay_buffer._size] = replay_buffer.dataset_dict["observations"]
    expanded_replay_buffer.dataset_dict["next_observations"][:replay_buffer._size] = replay_buffer.dataset_dict["next_observations"]
    expanded_replay_buffer.dataset_dict["actions"][:replay_buffer._size] = replay_buffer.dataset_dict["actions"]
    expanded_replay_buffer.dataset_dict["rewards"][:replay_buffer._size] = replay_buffer.dataset_dict["rewards"]
    expanded_replay_buffer.dataset_dict["masks"][:replay_buffer._size] = replay_buffer.dataset_dict["masks"]
    expanded_replay_buffer.dataset_dict["dones"][:replay_buffer._size] = replay_buffer.dataset_dict["dones"]

    expanded_replay_buffer._size = replay_buffer._size
    expanded_replay_buffer._insert_index = replay_buffer._size

    return expanded_replay_buffer

@hydra_task_config(args_cli.task, get_jaxrl_entry_point(args_cli.algorithm))
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    """Train with JaxRL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_jaxrl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    assert args_cli.num_envs == 1, "num_envs must be 1 for JaxRL offline to online training"

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

    wandb.init(project="gc_jaxrl")
    wandb.config.update(agent_cfg.to_dict())

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

    # if "normalize_input" in agent_cfg:
    #     env = VecNormalize(
    #         env,
    #         training=True,
    #         norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
    #         norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
    #         clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
    #         gamma=agent_cfg["gamma"],
    #         clip_reward=np.inf,
    #     )

    # configure the logger
    ## TODO: Add logger for wandb similar to rsl_rl

    replay_buffer, obs_space, action_space = load_replay_buffer(agent_cfg.dataset_path)
    if agent_cfg.clip_to_eps:
        lim = 1 - agent_cfg.eps
        replay_buffer.dataset_dict["actions"].clip(-lim, lim)

    ## TODO: Choice of buffer size can be either agent_cfg.replay_buffer_size or agent_cfg.num_pretraining_steps + agent_cfg.max_iterations
    replay_buffer = expand_buffer_capacity(replay_buffer, agent_cfg.num_pretraining_steps + agent_cfg.max_iterations, obs_space, action_space)
    # replay_buffer = expand_buffer_capacity(replay_buffer, agent_cfg.replay_buffer_size, obs_space, action_space)

    kwargs = get_flat_config(agent_cfg.algorithm.to_dict(), use_prefix=False)
    algorithm_name = kwargs.pop('algorithm_name', 'iql')  # Default to BC if not specified

    # create agent from stable baselines
    agent = get_learner(algorithm_name, agent_cfg.seed, env.observation_space, env.action_space, **kwargs)

    observation, _ = env.reset()
    done = False
    # for i in tqdm.tqdm(
    #     range(1 - agent_cfg.num_pretraining_steps, agent_cfg.max_iterations + 1), smoothing=0.1, disable=not agent_cfg.tqdm
    # ):
    for i in tqdm.tqdm(
        range(1,  agent_cfg.num_pretraining_steps + agent_cfg.max_iterations + 1), smoothing=0.1, disable=not agent_cfg.tqdm
    ):
        # if i > 1:
        if i > agent_cfg.num_pretraining_steps:
            ## Online training
            normalized_action, agent = agent.sample_actions(observation)
            if agent_cfg.clip_to_eps:
                lim = 1 - agent_cfg.eps
                normalized_action.clip(-lim, lim)
            next_observation, reward, terminated, truncated, info = env.step(normalized_action)
            done = terminated | truncated

            if not terminated:
                mask = 1.0
            else:
                mask = 0.0

            if env.num_envs == 1:
                replay_buffer.insert(
                    dict(
                    observations=observation.squeeze(),
                    actions=normalized_action.squeeze(),
                    rewards=reward.squeeze(),
                    masks=mask,
                    dones=done,
                    next_observations=next_observation.squeeze(),
                    )
                ),
            else:
                raise NotImplementedError("Parallel environments not supported yet.")
            observation = next_observation

            if done:
                observation, _ = env.reset()
                done = False
                for k, v in info[0]["episode"].items():
                    decode = {"r": "return", "l": "length", "t": "time"}
                    if k in decode:
                        metric = f"training/{decode[k]}"
                    else:
                        metric = k
                    wandb.log({metric: v}, step=i)

        batch = replay_buffer.sample(agent_cfg.batch_size)
        agent, info = agent.update(batch)

        if i % agent_cfg.log_interval == 0:
            info = jax.device_get(info)
            wandb.log(info, step=i)

        if i % agent_cfg.eval_interval == 0:
            eval_info = evaluate(agent, env, num_episodes=agent_cfg.eval_episodes)
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)

        # if i % agent_cfg.save_interval == 0 and i > 0:
        #     if agent_cfg.checkpoint_model:
        #         try:
        #             checkpoint_manager.save(step=i, args=ocp.args.StandardSave(agent))
        #         except:
        #             print("Could not save model checkpoint.")


    # callbacks for agent
    # checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2)
    # train the agent
    # agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
    # save the final model
    # agent.save(os.path.join(log_dir, "model"))
    ## TODO: Save the model. It's a good idea to have an agent.save() function defined in jaxrl as part of the Agent class.

    ## TODO: It'd be nice to have this functionality in jaxrl:
    # # write git state to logs
    # runner.add_git_repo_to_log(__file__)
    # # save resume path before creating a new log_dir
    # if agent_cfg.resume:
    #     # get path to previous checkpoint
    #     resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    #     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    #     # load previously trained model
    #     runner.load(resume_path)

    # # set seed of the environment
    # env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
