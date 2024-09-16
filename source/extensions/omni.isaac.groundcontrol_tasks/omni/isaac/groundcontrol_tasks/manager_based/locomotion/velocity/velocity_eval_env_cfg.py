# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Tuple
from omni.isaac.lab.utils import configclass
import omni.isaac.lab.terrains as default_terrains
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, SubTerrainBaseCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV

import omni.isaac.groundcontrol.terrains.eval_terrains_cfg as eval_terrains


def eval_terrain_gen(eval_terrain_cfg, static_friction=1.0, dynamic_friction=1.0):
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(10.0, 80.0),
            border_width=20.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "eval_terrain": eval_terrain_cfg(proportion=1.0)
            }
        ),
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

@configclass
class LocomotionVelocityEvalEnvCfg(LocomotionVelocityRoughEnvCfg):

    terrain_sequence_args: List[Tuple[SubTerrainBaseCfg, float, float]] = [
        (default_terrains.MeshPlaneTerrainCfg, 1.0, 1.0),
        (default_terrains.MeshPlaneTerrainCfg, 0.5, 0.5),
        (eval_terrains.HfSlopeTerrainCfg, 1.0, 1.0),
        (eval_terrains.HfInvertedSlopeTerrainCfg, 1.0, 1.0),
        (eval_terrains.HfObstacleTerrainCfg, 1.0, 1.0),
    ]

    def get_terrain_cfgs(self):
        return [eval_terrain_gen(*terrain_args) for terrain_args in self.terrain_sequence_args]

    def __post_init__(self):
        super().__post_init__()

        # Take commands to be only forward (no lateral/angular velocity), 2 m/s
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 2.0)