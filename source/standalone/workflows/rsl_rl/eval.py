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
    for i in range(len(env_cfg.terrain_sequence_args)):
        #print(f"Running terrain: {terrain_cfg.terrain_generator.sub_terrains['eval_terrain'].function.__name__}")
        #stage_utils.clear_stage(lambda x: True)
        
        cur_env = env.env
        
        # terrain_cfg.num_envs = env_cfg.scene.num_envs
        # terrain_cfg.env_spacing = env_cfg.scene.env_spacing
        # env_cfg.scene.terrain = terrain_cfg
        # cur_env.scene = InteractiveScene(env_cfg.scene)
        
        num_envs = cur_env.scene.num_envs
        # reset environment
        obs, _ = env.get_observations()
        timestep = 0
        cur_env.scene.terrain.terrain_types[:] = i
        cur_env.reset()
        # run everything in inference mode
        with torch.inference_mode():
            # simulate environment
            while timestep < cur_env.max_episode_length:
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, _, _  = env.step(actions)
                #print(cur_env.scene["robot"].data.default_root_state[0, :])
                timestep += 1
                if args_cli.video:
                    # Exit the play loop after recording one video
                    if timestep == args_cli.video_length:
                        break

            # reset the environment
            env.reset()
    # Close the environment (and simulation)
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
