
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from . import eval_terrains

@configclass
class HfSlopeTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a sloped height field terrain."""

    function = eval_terrains.slope_terrain

    slope_range: tuple[float, float] = (0.3, 0.3)
    inverted: bool = False

@configclass
class HfInvertedSlopeTerrainCfg(HfSlopeTerrainCfg):
    """Configuration for an inverted sloped height field terrain."""

    inverted: bool = True

@configclass
class HfObstacleTerrainCfg(HfTerrainBaseCfg):
    """Configuration for an obstacle height field terrain."""

    function = eval_terrains.obstacle_terrain

    hurdle_height_range: tuple[float, float] = (0.1, 0.1)
    hurdle_width: float = 0.5
    hurdle_gap: float = 5.0
    inverted: bool = False