# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg, slope_env_cfg, flat_env_simple_rewards_env_cfg, slope_env_simple_rewards_env_cfg
    
##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-A1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeA1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "jaxrl_bc_cfg_entry_point": f"{agents.__name__}.jaxrl_bc_cfg:UnitreeA1FlatBCRunnerCfg",
        "jaxrl_iql_cfg_entry_point": f"{agents.__name__}.jaxrl_iql_cfg:UnitreeA1FlatIQLRunnerCfg",
        "jaxrl_td3_cfg_entry_point": f"{agents.__name__}.jaxrl_td3_cfg:UnitreeA1FlatTD3RunnerCfg",
        "jaxrl_sac_cfg_entry_point": f"{agents.__name__}.jaxrl_sac_cfg:UnitreeA1FlatSACRunnerCfg",
        "jaxrl_redq_cfg_entry_point": f"{agents.__name__}.jaxrl_sac_cfg:UnitreeA1FlatREDQRunnerCfg",
        "jaxrl_droq_cfg_entry_point": f"{agents.__name__}.jaxrl_sac_cfg:UnitreeA1FlatDroQRunnerCfg",
        "jaxrl_rlpd_sac_cfg_entry_point": f"{agents.__name__}.jaxrl_rlpd_cfg:UnitreeA1FlatRLPDSACRunnerCfg",
        "jaxrl_rlpd_redq_cfg_entry_point": f"{agents.__name__}.jaxrl_rlpd_cfg:UnitreeA1FlatRLPDDREDQRunnerCfg",
        "jaxrl_rlpd_droq_cfg_entry_point": f"{agents.__name__}.jaxrl_rlpd_cfg:UnitreeA1FlatRLPDDroQRunnerCfg",
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sbx_ppo_cfg_entry_point": f"{agents.__name__}:sbx_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "sbx_sac_cfg_entry_point": f"{agents.__name__}:sbx_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Unitree-A1-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.UnitreeA1FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sbx_cfg_entry_point": f"{agents.__name__}:sbx_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-A1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeA1RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sbx_cfg_entry_point": f"{agents.__name__}:sbx_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-A1-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeA1RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sbx_cfg_entry_point": f"{agents.__name__}:sbx_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-A1-Eval-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.UnitreeA1RoughEnvCfg_EVAL,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Slope-Unitree-A1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": slope_env_cfg.UnitreeA1SlopeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1SlopePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Slope-Unitree-A1-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": slope_env_cfg.UnitreeA1SlopeEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1SlopePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Simple-Rewards-Unitree-A1-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_simple_rewards_env_cfg.UnitreeA1FlatEnvSimpleRewardsCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "jaxrl_bc_cfg_entry_point": f"{agents.__name__}.jaxrl_bc_cfg:UnitreeA1FlatBCRunnerCfg",
        "jaxrl_iql_cfg_entry_point": f"{agents.__name__}.jaxrl_iql_cfg:UnitreeA1FlatIQLRunnerCfg",
        "jaxrl_td3_cfg_entry_point": f"{agents.__name__}.jaxrl_td3_cfg:UnitreeA1FlatTD3RunnerCfg",
        "jaxrl_sac_cfg_entry_point": f"{agents.__name__}.jaxrl_sac_cfg:UnitreeA1FlatSACRunnerCfg",
        "jaxrl_redq_cfg_entry_point": f"{agents.__name__}.jaxrl_sac_cfg:UnitreeA1FlatREDQRunnerCfg",
        "jaxrl_droq_cfg_entry_point": f"{agents.__name__}.jaxrl_sac_cfg:UnitreeA1FlatDroQRunnerCfg",
        "jaxrl_rlpd_sac_cfg_entry_point": f"{agents.__name__}.jaxrl_rlpd_cfg:UnitreeA1FlatRLPDSACRunnerCfg",
        "jaxrl_rlpd_redq_cfg_entry_point": f"{agents.__name__}.jaxrl_rlpd_cfg:UnitreeA1FlatRLPDDREDQRunnerCfg",
        "jaxrl_rlpd_droq_cfg_entry_point": f"{agents.__name__}.jaxrl_rlpd_cfg:UnitreeA1FlatRLPDDroQRunnerCfg",
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sbx_ppo_cfg_entry_point": f"{agents.__name__}:sbx_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "sbx_sac_cfg_entry_point": f"{agents.__name__}:sbx_sac_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Simple-Rewards-Unitree-A1-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_simple_rewards_env_cfg.UnitreeA1FlatEnvSimpleRewardsCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sbx_cfg_entry_point": f"{agents.__name__}:sbx_ppo_cfg.yaml",
    },
)