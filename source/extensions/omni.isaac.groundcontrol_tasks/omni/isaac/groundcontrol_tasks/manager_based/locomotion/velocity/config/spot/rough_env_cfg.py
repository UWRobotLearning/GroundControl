# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.envs import ViewerCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg, SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.sensors import RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from omni.isaac.lab_assets.spot import SPOT_CFG  # isort: skip
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV
import omni.isaac.groundcontrol_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import omni.isaac.groundcontrol_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.groundcontrol_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, MySceneCfg
from omni.isaac.groundcontrol_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import SpotFlatEnvCfg


@configclass
class SpotObservationsRoughCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotCurriculumRoughCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


## Mateo doodles below
'''
Gonna try to initialize a Spot rough locomotion policy based off of a) the Anymal D rough locomotion code, and b) the 
Spot Flat env given that there may be some Spot specific things.

Gonna compare the observations, rewards, etc between Anymal, A1, Spot

    # Basic settings
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()

    # MDP setting
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()
    curriculum: SpotCurriculumCfg = SpotCurriculumCfg()

'''

@configclass
class SpotRoughEnvCfg(SpotFlatEnvCfg):
    # Basic Settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: SpotObservationsRoughCfg = SpotObservationsRoughCfg()

    # MDP Settings
    curriculum: SpotCurriculumRoughCfg = SpotCurriculumRoughCfg()

    def __post_init__(self):
        # post init of parent
        
        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # switch robot to Spot-d
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body"  


        # ## REMOVE

        # super().__post_init__()



        # # change terrains to rough
        # # ground terrain
        # self.scene.terrain = TerrainImporterCfg(
        #     prim_path="/World/ground",
        #     terrain_type="generator",
        #     terrain_generator=ROUGH_TERRAINS_CFG,
        #     max_init_terrain_level=5,
        #     collision_group=-1,
        #     physics_material=sim_utils.RigidBodyMaterialCfg(
        #         friction_combine_mode="multiply",
        #         restitution_combine_mode="multiply",
        #         static_friction=1.0,
        #         dynamic_friction=1.0,
        #     ),
        #     visual_material=sim_utils.MdlFileCfg(
        #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        #         project_uvw=True,
        #         texture_scale=(0.25, 0.25),
        #     ),
        #     debug_vis=True,
        # )

        # # Activate height scan
        # self.scene.height_scanner = RayCasterCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/body",
        #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 50.0)),
        #     attach_yaw_only=True,
        #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        #     debug_vis=False,
        #     mesh_prim_paths=["/World/ground"],
        # )

        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # # this generates terrains with increasing difficulty and is useful for training
        # self.scene.terrain.terrain_generator.curriculum = True
        # # if getattr(self.curriculum, "terrain_levels", None) is not None:
        # #     if self.scene.terrain.terrain_generator is not None:
        # #         self.scene.terrain.terrain_generator.curriculum = True
        # # else:
        # #     if self.scene.terrain.terrain_generator is not None:
        # #         self.scene.terrain.terrain_generator.curriculum = False



@configclass
class SpotRoughEnvCfg_PLAY(SpotRoughEnvCfg):
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