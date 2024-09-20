"""
This script tests the OffroadCarEnv in envs.
It also finds the joint_ids order returned by ObsTerm funcs

example usage:

python train/train_rsl_rl.py --headless

loading a previous run:

python train/train_rsl_rl.py --load-run <run_name>
"""
###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

import argparse
from utils.app_startup import startup, add_all_wheeled_gym_args, add_rsl_rl_args

parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")

overrides = {
    "rl_max_iterations": 4096,
    "env_name": "Isaac-MITCar-v0",
    "num_envs": 1024,
    "video": True,
    "log_every": 5,
    "video_interval": 5000,
    "video_length": 1000,
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
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

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
model_save_path = os.path.join(log_dir, "models")

if not args_cli.no_log:
    paths = [model_save_path]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

############################
#### CREATE ENVIRONMENT ####
############################

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils import update_class_from_dict
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.utils import get_checkpoint_path

import omni.isaac.groundcontrol_tasks
from omni.isaac.groundcontrol_tasks.utils.wrappers.torch_clip_action import ClipAction
from omni.isaac.groundcontrol_tasks.utils.runners.rslrl_runner import OnPolicyRunner

from utils import WHEELED_LAB_LOGS_DIR
from utils.args import parse_rsl_rl_cfg, default_isaac_cfg

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

####### INSTANTIATE ENV #######
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

env = RslRlVecEnvWrapper(env)

#### CREATE AGENT (FACTORY) ####
agent_cfg = parse_rsl_rl_cfg(args_cli.env_name, args_cli)

# dump the configuration into log-directory
if not args_cli.no_log:
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

runner_log_dir = None if args_cli.test_mode else log_dir
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=runner_log_dir, device=args_cli.device)

if args_cli.load_run:
    # get path to previous checkpoint
    resume_path = get_checkpoint_path(WHEELED_LAB_LOGS_DIR, run_dir=args_cli.load_run,
                                      other_dirs=["models"], checkpoint="model_.*")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner.load(resume_path)

env.seed(agent_cfg.seed)

runner.learn(num_learning_iterations=args_cli.rl_max_iterations)

if not args_cli.no_wandb:
    run.finish()
