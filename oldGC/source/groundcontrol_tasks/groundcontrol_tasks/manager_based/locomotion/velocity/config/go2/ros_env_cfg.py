# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import RayCasterCfg, patterns


# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV
import groundcontrol_tasks.manager_based.locomotion.velocity.config.go2.mdp as go2_mdp
import groundcontrol_tasks.manager_based.locomotion.velocity.mdp as mdp
from groundcontrol_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from amrl_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

class MySceneCfg(SceneEntityCfg):
    