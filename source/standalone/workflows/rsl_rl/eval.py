# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

#import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.utils.math import quat_apply
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
import omni.isaac.core.utils.stage as stage_utils

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV

from omni.isaac.groundcontrol_tasks.manager_based.locomotion.velocity.config.a1.rough_env_cfg import UnitreeA1RoughEnvCfg_EVAL
import omni.isaac.groundcontrol_tasks

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    args_cli.task = "Isaac-Velocity-Rough-Unitree-A1-Eval-v0"
    env_cfg = UnitreeA1RoughEnvCfg_EVAL()
    env_cfg.sim.device = args_cli.device
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # Set the render mode for all env instances
    render_mode = "rgb_array" if args_cli.video else None

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    env = ManagerBasedRLEnv(env_cfg, render_mode=render_mode)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    
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

    print("Evaluation started.")
    # run everything in inference mode
    with torch.inference_mode():

        cur_env = env.env
        cur_terrain = cur_env.scene.terrain
        num_envs = cur_env.scene.num_envs
        command_term = cur_env.command_manager.get_term('base_velocity')

        metrics = []

        for i in range(len(env_cfg.terrain_sequence_args)):
            #print(f"Running terrain: {terrain_cfg.terrain_generator.sub_terrains['eval_terrain'].function.__name__}")
            #stage_utils.clear_stage(lambda x: True)
            
            # terrain_cfg.num_envs = env_cfg.scene.num_envs
            # terrain_cfg.env_spacing = env_cfg.scene.env_spacing
            # env_cfg.scene.terrain = terrain_cfg
            # cur_env.scene = InteractiveScene(env_cfg.scene)
            
            # reset environment
            obs, _ = env.get_observations()
            timestep = 0
            
            cur_terrain.terrain_types[:] = i
            cur_terrain.env_origins[:] = cur_terrain.terrain_origins[0, i]
            cur_terrain.env_origins[:, 2] = 0

            env.reset()

            new_goal = torch.tensor(cur_env.scene['robot'].data.root_pos_w[0, :2].T, dtype=torch.float, device=obs.device)
            new_goal[0] += env_cfg.scene.terrain.terrain_generator.size[0]
            command_term.set_goal(new_goal)
            # simulate environment
            time_to_reach = torch.full((num_envs,), torch.inf, device=obs.device)
            min_dist = torch.full((num_envs,), torch.inf, device=obs.device)
            total_reward = torch.zeros((num_envs,), device=obs.device)
            while timestep < cur_env.max_episode_length:
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, rew, done, _ = env.step(actions)
                total_reward += rew
                timestep += 1
                min_dist = torch.min(min_dist, command_term.goal_distance)
                finished_envs = command_term.goal_reached.nonzero(as_tuple=False)
                if len(finished_envs) > 0:
                    for env_i in finished_envs:
                        time_to_reach[env_i] = min(time_to_reach[env_i], timestep * cur_env.step_dt)
                if not simulation_app.is_running():
                    break
            # Collect episode metrics
            # Clip close enough distances
            min_dist[min_dist < command_term.cfg.goal_reach_threshold] = 0

            reached_goal = torch.mean(command_term.goal_reached.float()).item()
            distance_quartiles = torch.quantile(min_dist, torch.tensor([0.25, 0.5, 0.75], device=obs.device)).tolist()
            avg_time_to_reach = torch.mean(time_to_reach[time_to_reach < torch.inf]).item()
            metrics.append({
                'terrain': env_cfg.terrain_sequence_args[i][0],
                'reached_goal': reached_goal,
                'distance_quartiles': distance_quartiles,
                'avg_time_to_reach': avg_time_to_reach,
                'total_reward': total_reward.mean().item()
            })
            command_term.goal_reached[:] = False
            command_term.goal_distance[:] = torch.full((command_term.num_envs,), torch.inf, device=obs.device)

    # Print all metrics at the end
    print("Final Metrics:")
    for metric in metrics:
        print("Terrain: ", metric['terrain'])
        print("Percent Reached Goal: ", metric['reached_goal'] * 100, "%")
        print("Distance Quartiles (25%, 50%, 75%): ", metric['distance_quartiles'])
        print("Average Time to Reach: ", metric['avg_time_to_reach'], "s")
        print("Total Reward: ", metric['total_reward'])
        print("***************")

    # Close the environment (and simulation)
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
