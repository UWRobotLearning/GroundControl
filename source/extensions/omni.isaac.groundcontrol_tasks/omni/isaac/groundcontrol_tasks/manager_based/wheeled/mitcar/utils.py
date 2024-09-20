import torch

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.terrains import TerrainImporter


def root_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root position in the simulation world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


def root_lin_vel_below_threshold(env: ManagerBasedEnv, threshold: float) -> torch.Tensor:
    """Check if the root linear velocity is below the threshold.

    Args:
        env: The environment.
        threshold: The threshold value.

    Returns:
        A boolean tensor indicating if the root linear velocity is below the threshold.
    """
    return mdp.root_lin_vel_w(env) < threshold


@torch.jit.script
def _f(euler_xyz: torch.Tensor) -> torch.Tensor:
    return math_utils.quat_from_euler_xyz(euler_xyz[0], euler_xyz[1], euler_xyz[2])

__f = torch.vmap(_f)
def quat_from_euler_xyz_vect(euler_xyz: torch.Tensor) -> torch.Tensor:
    """Convert Euler XYZ angles to quaternions.

    Args:
        euler_xyz: A tensor of Euler XYZ angles in radians with shape ``(N, 3)``.

    Returns:
        A tensor of quaternions with shape ``(N, 4)``.
    """
    return __f(euler_xyz)


def reset_root_state_from_terrain_points(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # valid_posns_and_rots: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state by sampling a random valid point from the config.

    This function samples a random valid pose(based on flat patches) from the terrain and sets the root state
    of the asset to this position. The function also samples random velocities from the given ranges and sets them
    into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of pose ranges for each axis. The keys of the dictionary are ``roll``,
      ``pitch``, and ``yaw``. The position is sampled from the flat patches of the terrain.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.

    Note:
        The function expects the terrain to have valid flat patches under the key "init_pos". The flat patches
        are used to sample the random pose for the robot.

    Raises:
        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
    """
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # obtain all flat patches corresponding to the valid poses
    # valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    valid_poses = terrain.cfg.valid_init_poses
    if valid_poses is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain_points' requires 'valid_init_poses' in the TerrainImporterCfg."
        )
    # Tensorizes the valid poses
    # TODO move to constructor of terrain importer
    posns = torch.stack(list(map(lambda x: torch.tensor(x.pos, device=env.device), valid_poses)))
    oris = list(map(lambda x: torch.deg2rad(torch.tensor(x.rot_euler_xyz_deg, device=env.device)), valid_poses))
    oris = torch.stack([math_utils.quat_from_euler_xyz(*ori) for ori in oris])

    # sample random valid poses
    ids = torch.randint(0, len(valid_poses), size=(len(env_ids),), device=env.device)

    positions = posns[ids]
    positions += asset.data.default_root_state[env_ids, :3]
    orientations = oris[ids]

    # sample random orientations (TODO)
    # range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    # ranges = torch.tensor(range_list, device=asset.device)
    # rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    # orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # sample random velocities (TODO)
    # range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    # ranges = torch.tensor(range_list, device=asset.device)
    # rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    # velocities = asset.data.default_root_state[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    # asset.write_root_velocity_to_sim(velocities, env_ids=env_ids) TODO
