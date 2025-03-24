# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.envs import ViewerCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg, SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.sensors import RayCasterCfg, patterns


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
    