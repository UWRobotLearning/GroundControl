# Copyright (c) 2022-2025, The GroundControl Project Developers.

import numpy as np
import trimesh

def object_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshObjTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray] | tuple[list[trimesh.Trimesh], np.ndarray]:
    mesh: trimesh.Trimesh = trimesh.load(cfg.obj_path)
    xy_scale = cfg.size / (mesh.bounds[1] - mesh.bounds[0])[:2]
    # set the height scale to the average between length and width scale to perserve as much original shap as possible
    height_scale = (xy_scale[0] + xy_scale[1]) / 2
    xyz_scale = np.array([*xy_scale, height_scale])
    mesh.apply_scale(xyz_scale)
    translation = -mesh.bounds[0]
    mesh.apply_translation(translation)
    
    extend = mesh.bounds[1] - mesh.bounds[0]
    origin = (*((extend[:2]) / 2), mesh.bounds[1][2] / 2)
    
    if isinstance(cfg.spawn_origin_path, str):
        spawning_option = np.load(cfg.spawn_origin_path, allow_pickle=True)
        spawning_option *= xyz_scale
        spawning_option += translation
        # insert the center of the terrain as the first indices
        # the rest of the indices represents the spawining locations
        return [mesh], np.insert(spawning_option, 0, origin, axis=0)
    else:
        return [mesh], np.array(origin)