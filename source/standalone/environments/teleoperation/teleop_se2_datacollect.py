# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher
from carb.input import GamepadInput

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--num_demos", type=int, default=5, help="Number of episodes to store in the dataset.")
parser.add_argument("--filename", type=str, default="hdf_dataset", help="Basename of output file.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import contextlib
import gymnasium as gym
import os
import torch
from dataclasses import dataclass

from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.utils.data_collector import RobomimicDataCollector
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV

import omni.isaac.groundcontrol_tasks  # noqa: F401
from omni.isaac.groundcontrol.devices import Se2Gamepad, Se2Keyboard, Se2SpaceMouse


def pre_process_actions(delta_pose: torch.Tensor) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # TODO: It looks like the raw actions from the keyboard and gamepad have the correct signs, but the observed
    # behavior of Spot is inverted. We need to see if this applies to other robots as well. If it does, then
    # we need to fix the command processing in the base envs. Note that changing this will also affect the command
    # generation for curriculum, so be careful.
    delta_pose[:, 1:] = -delta_pose[:, 1:]
    # ^TODO: temp fix

    return delta_pose 


@dataclass
class DataRecordState:
    is_recording: bool = True 
    has_flushed: bool = True
    is_finished: bool = False
    demo_num: int = 0

def toggle_data_collection(state: DataRecordState):
    print(f"===== Toggling Data Collect to {state.is_recording} =====")
    state.is_recording = not state.is_recording
    if state.is_recording:
        state.demo_num += 1

def reset_env_from_teleop_input(env):
    print("===== Resetting Environment =====")
    env.reset()
    #TODO: using logger

def finish_data_collection(state: DataRecordState):
    print("===== Finishing Data Collection and Closing Env =====")
    state.is_finished = True
    #TODO: using logger


def main():
    """Collect demonstrations from the environment using teleop interfaces."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # modify configuration such that the environment runs indefinitely
    # until goal is reached
    env_cfg.terminations.time_out = None
    # set the resampling time range to large number to avoid resampling
    #TODO: remove this
    #env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.observations.perception.concatenate_terms = False

    # add termination condition for reaching the goal otherwise the environment won't reset
    # TODO: fix the termination condition
    env_cfg.terminations.time_out = DoneTerm(func=mdp.time_out)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # create teleoperation controller
    #TODO: remove duplicate code between teleop and datacollect
    toggle_key, reset_key, done_key = None, None, None
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se2Keyboard(
            v_x_sensitivity=args_cli.sensitivity,
            v_y_sensitivity=args_cli.sensitivity,
            omega_z_sensitivity=args_cli.sensitivity,
        )
        toggle_key = "T"
        reset_key = "R"
        done_key = "E"
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
        toggle_key = GamepadInput.DPAD_UP
        reset_key = GamepadInput.DPAD_DOWN
        done_key = GamepadInput.DPAD_RIGHT
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', gamepad.")



    # print helper
    print(teleop_interface)

    # specify directory for logging experiments
    log_dir = os.path.join("./logs/robomimic", args_cli.task)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)

    # create data-collector
    collector_interface = RobomimicDataCollector(
        env_name=args_cli.task,
        directory_path=log_dir,
        filename=args_cli.filename,
        num_demos=args_cli.num_demos,
        flush_freq=env.num_envs,
        env_config={"teleop_device": args_cli.teleop_device},
    )

    # add teleoperation key for toggling data collection
    record_state = DataRecordState() 
    teleop_interface.add_callback(toggle_key, lambda : toggle_data_collection(record_state))
    # add teleoperation key for env reset
    teleop_interface.add_callback(reset_key, lambda : reset_env_from_teleop_input(env))
    # add teleoperation key for finishing collection 
    teleop_interface.add_callback(done_key, lambda : finish_data_collection(record_state))

    # reset environment
    obs_dict, _ = env.reset()

    #/// TODO: remove the block below so that logging low level observations can be defined in the high level policy
    #ll_obs_manager = env.action_manager.get_term("pre_trained_policy_action")._low_level_obs_manager
    #for group_name in ll_obs_manager._group_obs_term_names:
    #    group_term_names = ll_obs_manager._group_obs_term_names[group_name]
    #    # buffer to store obs per group
    #    group_obs = dict.fromkeys(group_term_names, None)
    #    # read attributes for each term
    #    obs_terms = zip(group_term_names, ll_obs_manager._group_obs_term_cfgs[group_name])
    #    for name, term_cfg in obs_terms:
    #        # compute term's value
    #        obs: torch.Tensor = term_cfg.func(ll_obs_manager._env, **term_cfg.params).clone()
    #        # apply post-processing
    #        if term_cfg.modifiers is not None:
    #            for modifier in term_cfg.modifiers:
    #                obs = modifier.func(obs, **modifier.params)
    #        if term_cfg.noise:
    #            obs = term_cfg.noise.func(obs, term_cfg.noise)
    #        if term_cfg.clip:
    #            obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
    #        if term_cfg.scale:
    #            obs = obs.mul_(term_cfg.scale)
    #        # add value to list
    #        group_obs[name] = obs
    # ///
    # reset interfaces
    teleop_interface.reset()
    collector_interface.reset()

    # simulate environment -- run everything in inference mode
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while not collector_interface.is_stopped():
            # get keyboard command
            delta_pose = teleop_interface.advance()
            # convert to torch
            delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=env.device).repeat(env.num_envs, 1)
            # compute actions based on environment
            actions = pre_process_actions(delta_pose)

            # TODO: Deal with the case when reset is triggered by teleoperation device.
            #   The observations need to be recollected.
            # store signals before stepping
            # -- obs
            if record_state.is_recording:
                for key, value in obs_dict["policy"].items():
                    collector_interface.add(f"obs/{key}", value)
                for key, value in obs_dict["env_state"].items():
                    collector_interface.add(f"obs/{key}", value)
                for key, value in obs_dict["perception"].items():
                    collector_interface.add(f"next_obs/{key}", value)
                # -- actions
                collector_interface.add("actions", actions)

            # perform action on environment
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated
            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

            # robomimic only cares about policy observations
            # store signals from the environment
            # -- next_obs
            if record_state.is_recording:
                for key, value in obs_dict["policy"].items():
                    collector_interface.add(f"next_obs/{key}", value)
                # -- rewards
                collector_interface.add("rewards", rewards)
                # -- dones
                collector_interface.add("dones", dones)
                # -- demo_num

                # -- is success label
                #TODO: change this to check if robot makes it to goal
                collector_interface.add("success", torch.tensor([record_state.is_recording], dtype=torch.bool, device=env.device))

            # flush data from collector for successful environments
            # reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            # reset_env_ids[:] = 1
            if record_state.is_recording:
                flush_env_ids = torch.tensor([], dtype=torch.long, device=env.device) 
                record_state.has_flushed = False
            else:
                if not record_state.has_flushed:
                    record_state.has_flushed = True
                    flush_env_ids = torch.tensor([0], dtype=torch.long, device=env.device) 
                    collector_interface.flush(flush_env_ids)
                

            # check if enough data is collected
            if record_state.is_finished:
                break
            if collector_interface.is_stopped():
                break

    # close the simulator
    collector_interface.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
