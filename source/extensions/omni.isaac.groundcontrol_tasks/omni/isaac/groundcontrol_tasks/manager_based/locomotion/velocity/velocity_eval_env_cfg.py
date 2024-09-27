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
import omni.isaac.groundcontrol_tasks.manager_based.locomotion.velocity.mdp as mdp


# def eval_terrain_gen(eval_terrain_cfg, static_friction=1.0, dynamic_friction=1.0):
#     return TerrainImporterCfg(
#         prim_path="/World/ground",
#         terrain_type="generator",
#         terrain_generator=TerrainGeneratorCfg(
#             size=(10.0, 80.0),
#             border_width=20.0,
#             num_rows=1,
#             num_cols=1,
#             horizontal_scale=0.1,
#             vertical_scale=0.005,
#             slope_threshold=0.75,
#             use_cache=False,
#             sub_terrains={
#                 "eval_terrain": eval_terrain_cfg(proportion=1.0)
#             }
#         ),
#         max_init_terrain_level=1,
#         collision_group=-1,
#         physics_material=sim_utils.RigidBodyMaterialCfg(
#             friction_combine_mode="multiply",
#             restitution_combine_mode="multiply",
#             static_friction=static_friction,
#             dynamic_friction=dynamic_friction,
#         ),
#         visual_material=sim_utils.MdlFileCfg(
#             mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
#             project_uvw=True,
#             texture_scale=(0.25, 0.25),
#         ),
#         debug_vis=False,
#     )

@configclass
class LocomotionVelocityEvalEnvCfg(LocomotionVelocityRoughEnvCfg):

    terrain_sequence_args = [
        ("flat", default_terrains.MeshPlaneTerrainCfg, {}),
        #("flat_2", default_terrains.MeshPlaneTerrainCfg, {}),
        ("up_slope", eval_terrains.HfSlopeTerrainCfg, {}),
        ("down_slope", eval_terrains.HfInvertedSlopeTerrainCfg, {}),
        ("obstacle", eval_terrains.HfObstacleTerrainCfg, {}),
        ("wave", default_terrains.HfWaveTerrainCfg, {"amplitude_range": (0.1, 0.5)}),
    ]

    # def get_terrain_cfgs(self):
    #     return [eval_terrain_gen(*terrain_args) for terrain_args in self.terrain_sequence_args]

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 15.0

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                size=(20.0, 10.0),
                border_width=20.0,
                num_rows=1,
                num_cols=5,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                curriculum=True,
                sub_terrains={args[0]: args[1](proportion=1.0, **args[2]) for args in self.terrain_sequence_args}
            ),
            max_init_terrain_level=1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1,
                dynamic_friction=1,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        self.terminations.base_contact = None
        self.curriculum.terrain_levels = None

        # Take commands to be only forward (no lateral/angular velocity), 2 m/s
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_x = (2.0, 2.0)

        self.commands.base_velocity = mdp.GoalVelocityCommandCfg(
            asset_name="robot",
            heading_control_stiffness=0.8,
            debug_vis=True,
            goal_position=(10.0, 10.0)
        )

        self.events.reset_base.params = {
            "pose_range": {"x": (-12.0, -12.0), "y": (-0.1, 0.1), "yaw": (-0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }