"""
This script tests the OffroadCarEnv in envs.
It also finds the joint_ids order returned by ObsTerm funcs

example usage:

python test/test_rl.py --headless
"""
###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

import argparse
from ..utils.app_startup import startup, add_all_wheeled_gym_args, add_rsl_rl_args

parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")

overrides = {
    "rl_max_iterations": 1024,
    "env_name": "Isaac-MITCarRacetrack-v0",
    "num_envs": 1024,
    "video": True,
    "log_every": 5,
    "video_interval": 1000,
    "video_length": 1200,
    # "no_wandb": True,
}
add_all_wheeled_gym_args(parser, overrides)
add_rsl_rl_args(parser)

def _args_cb(args):
    args.save_interval = args.log_every
    args.rl_no_log = args.no_log

simulation_app, args_cli = startup(parser=parser, prelaunch_callback=_args_cb)

#####################
###### LOGGING ######
#####################

import gymnasium as gym
import os
from datetime import datetime
import torch

if not args_cli.no_wandb:
    import wandb
    run = wandb.init(
        project="IRL",
    )
    wandb_name = wandb.run.name
    run_name = wandb_name
else:
    import random
    run_name = f"bfirl-local-{random.randint(0, 1e7)}"

log_dir = os.path.join(args_cli.log_dir, f'{run_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
# dump the configuration into log-directory
# dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
# dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
model_save_path = os.path.join(log_dir, "models")

############################
#### CREATE ENVIRONMENT ####
############################

import wheeled_gym.tasks # register envs to gym
from wheeled_gym.train.utils.utils import default_isaac_cfg
from wheeled_gym.utils.data_processing import load_from_sb3
from wheeled_gym.tasks.wrappers.clip_action import ClipAction
from wheeled_gym import WHEELED_GYM_LOGS_DIR

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils import update_class_from_dict
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
from omni.isaac.lab_tasks.utils import get_checkpoint_path

from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

####### FETCH CONFIGS #######
isaac_cfg = default_isaac_cfg(
    device=args_cli.device,
    num_envs=args_cli.num_envs,
    use_fabric=not args_cli.disable_fabric
)
env_cfg = parse_env_cfg(
    args_cli.env_name, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)
update_class_from_dict(env_cfg, isaac_cfg)

env = gym.make(args_cli.env_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
n_timesteps = args_cli.rl_max_iterations * args_cli.agent_n_steps * env.scene.num_envs

####### INSTANTIATE ENV #######
# env.action_space.low = torch.tensor(-1., device=args_cli.device)
# env.action_space.high = torch.tensor(1., device=args_cli.device)
env.action_space.low = -1.
env.action_space.high = 1.
env = ClipAction(env)

if args_cli.video:
    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos"),
        "step_trigger": lambda step: step % args_cli.video_interval*args_cli.num_envs == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

env = Sb3VecEnvWrapper(env)

#### CREATE AGENT (FACTORY) ####
env.seed(args_cli.seed)

policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))
checkpoint_callback = CheckpointCallback(save_freq=args_cli.checkpoint_every,
                                            save_path=os.path.join(model_save_path),
                                            name_prefix="model")
callbacklist = [checkpoint_callback]
if not args_cli.no_wandb:
    callbacklist.append(WandbCallback())
callbacklist = CallbackList(callbacklist)

if args_cli.load_run:
    run_path = os.path.join(WHEELED_GYM_LOGS_DIR, args_cli.load_run)
    resume_path = get_checkpoint_path(WHEELED_GYM_LOGS_DIR, run_dir=args_cli.load_run, 
                                      other_dirs=["models"], checkpoint="model_.*")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = load_from_sb3(resume_path, policy_type=args_cli.rl_algo_type)
    runner.env = env
else:
    runner = PPO("MlpPolicy", env,
                device=args_cli.device,
                policy_kwargs=policy_kwargs)

try:
    runner.learn(total_timesteps=n_timesteps, callback=callbacklist, progress_bar=True)
except KeyboardInterrupt:
    runner.save(os.path.join(model_save_path, f"{run_name}_interrupted"))

runner.save(os.path.join(model_save_path, f"{run_name}_done"))

run.finish()
