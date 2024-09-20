import torch

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.assets import AssetBaseCfg, ArticulationCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.groundcontrol_tasks.manager_based.wheeled.terrains import rough, racetrack
from omni.isaac.groundcontrol_assets.mitcar import MITCAR_CFG

################
#### SCENES ####
################

FLAT_TERRAIN_CFG = TerrainImporterCfg(
                        prim_path="/World/ground",
                        terrain_type="plane",
                        debug_vis=False,
                    )


@configclass
class MITCarBaseSceneCfg(InteractiveSceneCfg):

    """Configuration for a MIT car Scene"""

    # Distant Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Mesh
    robot: ArticulationCfg = MITCAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class MITCarFlatSceneCfg(MITCarBaseSceneCfg):
    """Configuration for a MIT car Scene with flat terrain"""
    terrain = FLAT_TERRAIN_CFG


@configclass
class MITCarRoughSceneCfg(MITCarBaseSceneCfg):
    """Configuration for a MIT car Scene with rough terrain"""
    terrain = rough.ROUGH_TERRAIN_CFG


@configclass
class MITCarRacetrackSceneCfg(MITCarBaseSceneCfg):
    """Configuration for a MIT car Scene with racetrack terrain"""

    terrain = racetrack.RacetrackTerrainImporterCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # Set initial state of the robot
        self.robot.init_state = self.robot.init_state.replace(
                pos=(0.0, 0.0, 0.5)
            )
