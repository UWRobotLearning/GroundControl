# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2024-, The Ground Control Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Spot-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.SpotFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotFlatPPORunnerCfg",
        "jaxrl_cfg_entry_point": f"{agents.__name__}.jaxrl_sac_cfg:SpotFlatPPORunnerCfg"  ## TODO: Replace with JaxRL SAC config
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Spot-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.SpotFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Spot-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.SpotRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Spot-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.SpotRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotRoughPPORunnerCfg",
    },
)
