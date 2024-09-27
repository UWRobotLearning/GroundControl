# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
import omni.isaac.lab.terrains as default_terrains
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
# from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG  # isort: skip
# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV
from omni.isaac.groundcontrol_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from omni.isaac.groundcontrol_tasks.manager_based.locomotion.velocity.velocity_eval_env_cfg import LocomotionVelocityEvalEnvCfg
from omni.isaac.groundcontrol_assets.unitree import UNITREE_A1_CFG  # isort: skip
import omni.isaac.groundcontrol.terrains.eval_terrains_cfg as eval_terrains
import omni.isaac.groundcontrol_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class UnitreeA1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"


@configclass
class UnitreeA1RoughEnvCfg_PLAY(UnitreeA1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

# @configclass
# class UnitreeA1RoughEnvCfg_EVAL(LocomotionVelocityEvalEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         #self.scene.env_spacing = 2.5
        
#         self.scene.robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
#         self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"

#         # reduce action scale
#         self.actions.joint_pos.scale = 0.25

#         # event
#         self.events.push_robot = None
#         self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
#         self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
#         self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

#         # rewards
#         self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
#         self.rewards.feet_air_time.weight = 0.01
#         self.rewards.undesired_contacts = None
#         self.rewards.dof_torques_l2.weight = -0.0002
#         self.rewards.track_lin_vel_xy_exp.weight = 1.5
#         self.rewards.track_ang_vel_z_exp.weight = 0.75
#         self.rewards.dof_acc_l2.weight = -2.5e-7

#         # terminations
#         #self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"

#         # disable randomization for play
#         self.observations.policy.enable_corruption = False
#         # remove random pushing event
#         self.events.base_external_force_torque = None
#         self.events.push_robot = None



'''
Mateo's doodles

Can you have UnitreeA1RoughEnvCfg_EVAL inherit from UnitreeA1RoughEnvCfg_PLAY?
'''

@configclass
class UnitreeA1RoughEnvCfg_EVAL(UnitreeA1RoughEnvCfg):

    terrain_sequence_args = [
        ("flat", default_terrains.MeshPlaneTerrainCfg, {}),
        ("up_slope", eval_terrains.HfSlopeTerrainCfg, {}),
        ("down_slope", eval_terrains.HfInvertedSlopeTerrainCfg, {}),
        ("obstacle", eval_terrains.HfObstacleTerrainCfg, {}),
        ("wave", default_terrains.HfWaveTerrainCfg, {"amplitude_range": (0.1, 0.5)}),
    ]
        
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 50
        # disable randomization for eval
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

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