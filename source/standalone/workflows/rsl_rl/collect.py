# def collect(collect_cfg, env, policy, obs_processor=lambda x: x, action_processor=lambda x: x, nn_normalized_actions=True):  ## TODO: task_cfg could be replaced by collect config
#     '''
#     if nn_normalized_actions=True, this means that the actions returned by policy(obs) will be in the range [-1, 1]
#     '''
#     log.info("Collecting")
    
#     ## Initialize Replay Buffer
#     replay_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=collect_cfg.dataset_size, next_observation_space=env.observation_space, observation_labels=env.observation_labels)
#     ## Reset environment
#     env.reset()
#     # import ipdb;ipdb.set_trace()
#     obs = env.get_observations(1)
#     critic_obs = env.get_critic_observations(collect_cfg.history_length)
#     extra_obs = env.get_extra_observations(collect_cfg.history_length)

#     obs_processed = obs_processor(obs.detach())

#     action_scaler = RescaleActionAsymmetric(env.action_space, -1, 1, center_action=np.zeros(env.action_space.shape))

#     current_time = time.time()
#     while len(replay_buffer) < collect_cfg.dataset_size:
#         raw_actions = policy(obs_processed)
#         if nn_normalized_actions:  ## This means action came from an SAC++ policy
#             unnormalized_actions = action_scaler.transform_action(raw_actions, use_torch=True)
#             # unnormalized_actions = unnormalize_action(raw_actions, env.action_space)
#         else:
#             unnormalized_actions = clip_action(raw_actions, env.action_space)

#         if torch.is_tensor(unnormalized_actions):
#             unnormalized_actions = unnormalized_actions.detach()
#         actions_processed = action_processor(unnormalized_actions)  ## This is here to interface between pytorch and numpy, to make sure actions are always torch tensors when they go into env
#         new_obs, new_critic_obs, rewards, dones, infos, *_ = env.step(actions_processed)
#         ## Hacky override to get a specific amount of steps:
#         new_obs = env.get_observations(1)
#         new_critic_obs = env.get_critic_observations(collect_cfg.history_length)
#         ## end of hacky override
#         new_extra_obs = env.get_extra_observations(collect_cfg.history_length)
#         duration = time.time() - current_time
#         if duration < env.dt:
#             time.sleep(env.dt - duration)
#         current_time = time.time()

#         print(f"replay_buffer size: {len(replay_buffer)}")
#         buffer_observations = torch.concatenate([critic_obs, extra_obs], dim=-1)  ## Note: critic_obs contains obs, but does not get processed, so it's always a PyTorch tensor
#         buffer_new_observations = torch.concatenate([new_critic_obs, new_extra_obs], dim=-1)
#         normalized_actions = action_scaler.inverse_transform_action(actions_processed, use_torch=True)
#         # normalized_actions = normalize_action(actions_processed, env.action_space)
#         insert_batch_into_replay_buffer(replay_buffer, buffer_observations, normalized_actions, rewards, dones, buffer_new_observations, infos, env.observation_labels)
#         obs = new_obs
#         critic_obs = new_critic_obs
#         extra_obs = new_extra_obs

#         obs_processed = obs_processor(obs.detach())

#     log.info("Done collecting")
#     return replay_buffer


# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect data from a checkpoint of an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect data from a checkpoint of an RL agent from RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV
import omni.isaac.groundcontrol_tasks
from jaxrl.data import ReplayBuffer
from jaxrl.data import save_replay_buffer
from omni.isaac.groundcontrol_tasks.utils.wrappers.jaxrl.rescale_action_asymmetric import RescaleActionAsymmetric
from typing import Union
import numpy as np

def clip_action(action: Union[np.array, torch.Tensor], action_space: gym.spaces.Box):
    '''
    Clips actions to [low, high] given by action_space
    '''
    low, high = action_space.low, action_space.high
    if isinstance(action, torch.Tensor):
        low = torch.from_numpy(low).to(action.device)
        high = torch.from_numpy(high).to(action.device)
        return torch.clip(action, low, high)
    return np.clip(action, low, high)

def insert_batch_into_replay_buffer(replay_buffer, observations, actions, rewards, dones, next_observations, infos):
    ## Obtain mask
    if 'time_outs' not in infos:  ## No episode was terminated, so should just take into account dones (should all be False)
         masks = torch.logical_not(dones).float()
    else:  ## There was an episode terminated. Masks should be 1 if episode is *not* done or episode was terminated due to timeout, should be 0 if episode was terminated due to MDP end condition.
         masks = torch.logical_or(torch.logical_not(dones), infos["time_outs"]).float()

    ## Convert data to numpy
    observations = observations.cpu().detach().numpy()
    actions = actions.cpu().detach().numpy()
    rewards = rewards.cpu().detach().numpy()
    dones = dones.cpu().detach().numpy()
    next_observations = next_observations.cpu().detach().numpy()
    masks = masks.cpu().detach().numpy()

    for i in range(observations.shape[0]):
        replay_buffer.insert(
            dict(observations=observations[i],
                 actions=actions[i],
                 rewards=rewards[i],
                 masks=masks[i],
                 dones=dones[i],
                 next_observations=next_observations[i],
                 )
        )

def main():
    """Collect with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "collect"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Make Replay Buffer
    lower_limit = env.scene['robot'].data.joint_limits[0, :, 0].cpu().detach().numpy()
    upper_limit = env.scene['robot'].data.joint_limits[0, :, 1].cpu().detach().numpy()
    action_space = gym.spaces.Box(low=lower_limit, high=upper_limit, shape=(env.action_space.shape[-1],))
    observation_space = env.observation_space['policy']
    observation_space = gym.spaces.Box(low=observation_space.low[0], high=observation_space.high[0], shape=(observation_space.shape[-1],))
    env.action_space = action_space
    dataset_size = int(1e6)
    replay_buffer = ReplayBuffer(observation_space, action_space, capacity=dataset_size, next_observation_space=observation_space)

     ## Action scaler
    action_scaler = RescaleActionAsymmetric(action_space, -1, 1, center_action=np.zeros(action_space.shape))

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

   


    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # reset environment
    obs, extras = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running() and len(replay_buffer) < dataset_size:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            ## Actions are unnormalized here, and unclipped
            unnormalized_actions = clip_action(actions, env.action_space)

            # env stepping
            new_obs, rew, dones, extras = env.step(actions)

            ## Save into buffer
            normalized_actions = action_scaler.inverse_transform_action(unnormalized_actions, use_torch=True)
            insert_batch_into_replay_buffer(replay_buffer, obs, normalized_actions, rew, dones, new_obs, extras)
            print(f"replay_buffer size: {len(replay_buffer)}")

            obs = new_obs

        # if args_cli.video:
        #     timestep += 1
        #     # Exit the play loop after recording one video
        #     if timestep == args_cli.video_length:
        #         break

    replay_buffer_path = os.path.abspath("./expert_ppo_buffer.npz")
    print(f"Saving replay buffer to {replay_buffer_path}")
    save_replay_buffer(replay_buffer, replay_buffer_path, observation_space, action_space)
    print("Done saving replay buffer")
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
