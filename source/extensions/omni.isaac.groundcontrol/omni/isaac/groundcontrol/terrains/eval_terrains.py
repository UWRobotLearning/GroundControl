from __future__ import annotations
import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from omni.isaac.lab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import eval_terrains_cfg

@height_field_to_mesh
def slope_terrain(difficulty: float, cfg: eval_terrains_cfg.HfSlopeTerrainCfg) -> np.ndarray:
    """Generate a terrain with a simple slope.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- height
    height_max = int(slope * cfg.size[0] / cfg.vertical_scale)

    # create the height field
    y = np.linspace(0, height_max, width_pixels)
    if cfg.inverted:
        y = y[::-1]
    hf_raw = np.tile(y, (length_pixels,1)).T

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)

@height_field_to_mesh
def obstacle_terrain(difficulty: float, cfg: eval_terrains_cfg.HfObstacleTerrainCfg) -> np.ndarray:
    """Generate a terrain with equally spaced obstacles in the y direction.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    if cfg.inverted:
        hurdle_height = -cfg.hurdle_height_range[0] - difficulty * (cfg.hurdle_height_range[1] - cfg.hurdle_height_range[0])
    else:
        hurdle_height = cfg.hurdle_height_range[0] + difficulty * (cfg.hurdle_height_range[1] - cfg.hurdle_height_range[0])

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    hurdle_width_pixels = int(cfg.hurdle_width / cfg.horizontal_scale)
    hurdle_gap_pixels = int(cfg.hurdle_gap / cfg.horizontal_scale)
    # -- height
    hurdle_height_pixels = int(hurdle_height / cfg.vertical_scale)

    # create the height field
    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)
    hurdle_mask = np.arange(width_pixels) % (hurdle_gap_pixels + hurdle_width_pixels) < hurdle_width_pixels
    hf_raw[hurdle_mask, :] = hurdle_height_pixels

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


