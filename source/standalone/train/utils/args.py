import argparse

def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace):
    """
    Parse configuration for RSL-RL agent based on inputs.
    Needs rsl_rl_cfg_entry_point to load Agent Config

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from omni.isaac.lab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")

    # override the default configuration with CLI arguments
    if args_cli.seed is not None:
        rslrl_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        rslrl_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        rslrl_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        rslrl_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        rslrl_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        rslrl_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if rslrl_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        rslrl_cfg.wandb_project = args_cli.log_project_name
        rslrl_cfg.neptune_project = args_cli.log_project_name

    # wheeled_gym naming convention overrides
    rslrl_cfg.max_iterations = args_cli.rl_max_iterations
    rslrl_cfg.num_steps_per_env = args_cli.agent_n_steps
    if args_cli.no_log:
        rslrl_cfg.log_dir = None

    rslrl_cfg.rl_no_log = args_cli.rl_no_log

    return rslrl_cfg


def default_isaac_cfg(
        device: str = "cuda:0", num_envs: int | None = None, use_fabric: bool | None = None
        ):
    default_cfg = {"sim": {"physx": dict()}, "scene": dict()}

    # simulation device
    default_cfg["sim"]["device"] = device

    # disable fabric to read/write through USD
    if use_fabric is not None:
        default_cfg["sim"]["use_fabric"] = use_fabric

    # number of environments
    if num_envs is not None:
        default_cfg["scene"]["num_envs"] = num_envs

    return default_cfg