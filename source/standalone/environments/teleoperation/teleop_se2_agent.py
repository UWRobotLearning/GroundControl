# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import carb

from omni.isaac.lab_tasks.utils import parse_env_cfg

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV
from omni.isaac.groundcontrol.devices import Se2Gamepad, Se2Keyboard, Se2SpaceMouse
import omni.isaac.groundcontrol_tasks  # noqa: F401



def pre_process_actions(delta_pose: torch.Tensor) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # TODO: It looks like the raw actions from the keyboard and gamepad have the correct signs, but the observed
    # behavior of Spot is inverted. We need to see if this applies to other robots as well. If it does, then
    # we need to fix the command processing in the base envs. Note that changing this will also affect the command
    # generation for curriculum, so be careful.
    delta_pose[:, 1:] = -delta_pose[:, 1:]
    # ^TODO: temp fix

    return delta_pose 


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # modify configuration
    env_cfg.terminations.time_out = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        carb.log_warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se2Keyboard(
            v_x_sensitivity=args_cli.sensitivity,
            v_y_sensitivity=args_cli.sensitivity,
            omega_z_sensitivity=args_cli.sensitivity,
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se2SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.005 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se2Gamepad(
            v_x_sensitivity=args_cli.sensitivity,
            v_y_sensitivity=args_cli.sensitivity,
            omega_z_sensitivity=args_cli.sensitivity,
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', gamepad.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # get keyboard command
            delta_pose = teleop_interface.advance()
            delta_pose = delta_pose.astype("float32")
            # convert to torch
            delta_pose = torch.tensor(delta_pose, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            # pre-process actions
            actions = pre_process_actions(delta_pose)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
