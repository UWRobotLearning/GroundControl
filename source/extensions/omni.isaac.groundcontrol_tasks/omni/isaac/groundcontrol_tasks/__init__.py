# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

ISAACLAB_TASKS_METADATA = toml.load(os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##

import gymnasium as gym

###### WHEELED ######

from .manager_based.wheeled.mitcar.mitcar_manager_env_cfg import (
    MITCarRLEnvCfg, MITCarPlayEnvCfg, MITCarIRLEnvCfg
)

import omni.isaac.groundcontrol_tasks.manager_based.wheeled.mitcar.agents as agents

gym.register(
    id='Isaac-MITCar-v0',
    entry_point='omni.isaac.lab.envs:ManagerBasedRLEnv',
    kwargs={
        "env_cfg_entry_point":MITCarRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MITCarPPORunnerCfg",
    }
)

gym.register(
    id='Isaac-MITCarPlay-v0',
    entry_point='omni.isaac.lab.envs:ManagerBasedRLEnv',
    kwargs={
        "env_cfg_entry_point":MITCarPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MITCarPPORunnerCfg",
    }
)

gym.register(
    id='Isaac-MITCarIRL-v0',
    entry_point='omni.isaac.lab.envs:ManagerBasedRLEnv',
    kwargs={
        "env_cfg_entry_point":MITCarIRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MITCarPPORunnerCfg",
    }
)

########################################
############ RACETRACK ENVS ############
########################################

from .manager_based.wheeled.mitcar.mitcar_manager_racetrack_env_cfg import (
    MITCarRacetrackRLEnvCfg, MITCarRacetrackPlayEnvCfg
)

gym.register(
    id='Isaac-MITCarRacetrack-v0',
    entry_point='omni.isaac.lab.envs:ManagerBasedRLEnv',
    kwargs={
        "env_cfg_entry_point":MITCarRacetrackRLEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MITCarPPORunnerCfg",
    }
)

gym.register(
    id='Isaac-MITCarRacetrackPlay-v0',
    entry_point='omni.isaac.lab.envs:ManagerBasedRLEnv',
    kwargs={
        "env_cfg_entry_point":MITCarRacetrackPlayEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MITCarPPORunnerCfg",
    }
)

from .utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
