"""
Boilerplate code for starting up IsaacLab backend
"""

import argparse
from . import WHEELED_LAB_LOGS_DIR


defaults = {
    "no_log": False,
    "log_dir": WHEELED_LAB_LOGS_DIR,
    "log_every": 10,
    "video": False,
    "video_length": 500,
    "video_interval": 1e4,
    "no_checkpoints": False,
    "checkpoint_every": 1e4,
    "no_wandb": False,
    "rl_max_iterations": 1024,
    "irl_max_iterations": 1024,
    "agent_n_steps": 256,
    "cpu": False,
    "device": "cuda:0",
    "disable_fabric": False,
    "num_envs": 256,
    "env_name": "Isaac-MITCar-v0",
    "rl_algo_type": "PPO",
    "rl_algo_lib": "sb3",
    "seed": 42,
    "rl_no_log": True,
    "logger": "wandb",
    "load-run": None,

    "test_mode": False,
}


def add_logging_args(parser, default_overrides={}):
    defaults.update(default_overrides)
    parser.add_argument("--no-log", action="store_true", default=defaults["no_log"] , help="Disable logging")
    parser.add_argument("--rl-no-log", action="store_true", default=defaults["rl_no_log"] , help="Disable logging for RL algorithm")
    parser.add_argument("--log-dir", type=str, default=defaults["log_dir"] , help="Directory for logging.")
    parser.add_argument("--log-every", type=int, default=defaults["log_every"] , help="Log every n updates.")
    parser.add_argument("--video", action="store_true", default=defaults["video"], help="Record videos during training.")
    parser.add_argument("--video-length", type=int, default=defaults["video_length"], help="Length of the recorded video (in steps).")
    parser.add_argument("--video-interval", type=int, default=defaults["video_interval"], help="Interval between video recordings (in steps).")
    parser.add_argument("--no-checkpoints", action="store_true", default=defaults["no_checkpoints"], help="Save model checkpoints.")
    parser.add_argument("--checkpoint-every", type=int, default=defaults["checkpoint_every"], help="Save model checkpoints every n steps.")
    parser.add_argument("--no-wandb", action="store_true", default=defaults["no_wandb"], help="Disable wandb logging.")
    parser.add_argument("--test-mode", action="store_true", default=defaults["test_mode"] , help="Disable logging; Disable wandb; Disable video recording; Disable checkpoints.")


def add_train_args(parser, default_overrides={}):
    defaults.update(default_overrides)
    parser.add_argument("--seed", type=int, default=defaults["seed"], help="Seed for training")
    parser.add_argument("--rl-max-iterations", type=int, default=defaults["rl_max_iterations"], help="RL rl_algo training iterations.")
    parser.add_argument("--irl-max-iterations", type=int, default=defaults["irl_max_iterations"], help="IRL rl_algo training iterations.")
    parser.add_argument("--agent-n-steps", type=int, default=defaults["agent_n_steps"], help="Agent max steps")
    parser.add_argument("--rl-algo-lib", type=str, default=defaults["rl_algo_lib"], help="library for rl_algo [sb3|rsl]")
    parser.add_argument("--rl-algo-type", type=str, default=defaults["rl_algo_type"], help="type of rl_algo [SAC|PPO|manual]")
    parser.add_argument("--load-run", type=str, default=defaults['load-run'], help="Name of run to load from logs dir.")


def add_env_args(parser, default_overrides={}):
    '''
    Add standard environment arguments to the parser.
        parser: argparse.ArgumentParser
            Argument parser to add arguments to.
    '''
    defaults.update(default_overrides)
    # parser.add_argument("--cpu", action="store_true", default=defaults["cpu"], help="Use CPU pipeline.") # Deprecated due to support in IsaacLab v1.0.0
    # parser.add_argument("--device", default="cuda:0", help="Device [cpu|cuda:0].") # Deprecated due to support in IsaacLab v1.0.0
    parser.add_argument(
        "--disable_fabric", action="store_true", default=defaults["disable_fabric"], help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument('-ne', "--num-envs", type=int, default=defaults["num_envs"], help="Number of environments to simulate.")
    parser.add_argument('-en', "--env-name", type=str, default=defaults["env_name"], help="Name of the task.")


def add_rsl_rl_args(parser: argparse.ArgumentParser, default_overrides={}):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    defaults.update(default_overrides)
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
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
        "--logger", type=str, default=defaults["logger"], choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )


def add_all_wheeled_gym_args(parser, default_overrides={}):
    add_logging_args(parser, default_overrides)
    add_train_args(parser, default_overrides)
    add_env_args(parser, default_overrides)

def startup(parser=None, prelaunch_callback=None, import_gym_envs=True):
    from omni.isaac.lab.app import AppLauncher
    '''
    Startup IsaacLab backend. Imports wheeled_gym environments optionally.
    Args:
        parser: argparse.ArgumentParser, optional, default=None
            Argument parser to add arguments to.
        prelaunch(args): function to be executed right before launching the app, optional, default=None
    Returns:
        simulation_app: omni.isaac.dynamic_control.DynamicControl, omni.isaac.dynamic_control._dynamic_control.DynamicControl
            Simulation app instance.
        args_cli: argparse.Namespace
            Parsed command line arguments.
    '''

    if parser is None:
        parser = argparse.ArgumentParser(description="Used Boilerplate Starter.")

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    if prelaunch_callback is not None:
        prelaunch_callback(args_cli)

    if "test_mode" in args_cli and args_cli.test_mode:
        args_cli.no_log = True
        args_cli.rl_no_log = True
        args_cli.no_wandb = True
        args_cli.video = False
        args_cli.no_checkpoints = True

    if args_cli.video:
        args_cli.enable_cameras = True

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    return simulation_app, args_cli
