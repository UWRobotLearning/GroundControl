# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from .rough_env_cfg import UnitreeA1RoughEnvCfg
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.terrains.height_field import HfInvertedPyramidSlopedTerrainCfg
from omni.isaac.lab.envs import ViewerCfg
# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV
import omni.isaac.groundcontrol.terrains.eval_terrains_cfg as eval_terrains


@configclass
class UnitreeA1SlopeEnvSimpleRewardsCfg(UnitreeA1RoughEnvCfg):

    # terrain_sequence_args = [
    #     ("flat", default_terrains.MeshPlaneTerrainCfg, {}),
    #     ("up_slope", eval_terrains.HfSlopeTerrainCfg, {}),
    #     ("down_slope", eval_terrains.HfInvertedSlopeTerrainCfg, {}),
    #     ("obstacle", eval_terrains.HfObstacleTerrainCfg, {}),
    #     ("wave", default_terrains.HfWaveTerrainCfg, {"amplitude_range": (0.1, 0.5)}),
    # ]
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                size=(20.0, 10.0),
                border_width=20.0,
                num_rows=10,
                num_cols=20,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                curriculum=True,
                # sub_terrains={args[0]: args[1](proportion=1.0, **args[2]) for args in self.terrain_sequence_args},
                sub_terrains = {
                    "up_slope": HfInvertedPyramidSlopedTerrainCfg(proportion=1., platform_width=1., slope_range=(0.2, 0.2)),
                }
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

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        self.viewer = ViewerCfg(eye=(10.5, 10.5, 10.5), origin_type="asset_root", env_index=0, asset_name="robot")

        # override rewards that are not task related
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.dof_pos_limits.weight = 0.0


class UnitreeA1SlopeEnvSimpleRewardsCfg_PLAY(UnitreeA1SlopeEnvSimpleRewardsCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.viewer = ViewerCfg(eye=(10.5, 10.5, 10.5), origin_type="world")
