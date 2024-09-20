import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.groundcontrol_assets import GROUNDCONTROL_ASSETS_DATA_DIR

RACETRACK_TERRAIN_CFG = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{GROUNDCONTROL_ASSETS_DATA_DIR}/Props/terrain/racetrack-terrain.usd",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.5,
            dynamic_friction=1.5,
        ),
        debug_vis=False,
    )

@configclass
class RacetrackTerrainImporterCfg(TerrainImporterCfg):

    @configclass
    class InitialPoseCfg:
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot_euler_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    height = 0.0
    valid_init_poses = [
        InitialPoseCfg(
                pos=(12.0, 1.27, height),
                rot_euler_xyz_deg=(0., 0., 135.0)
            ),
        InitialPoseCfg(
                pos=(-5.33, 3.3, height),
                rot_euler_xyz_deg=(0., 0., 180.0),
            ),
        InitialPoseCfg(
                pos=(-8.7, -7.27, height),
            ),
        InitialPoseCfg(
                pos=(0., 0., height),
            ),
    ]
    prim_path="/World/ground"
    terrain_type="usd"
    usd_path=f"{GROUNDCONTROL_ASSETS_DATA_DIR}/Props/terrain/racetrack-terrain.usd",
    # usd_path=f"/home/tyler/Research/GroundControl/source/extensions/omni.isaac.groundcontrol_assets/data/Props/terrain/racetrack-terrain.usd",
    collision_group=-1
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.5,
        dynamic_friction=1.5,
    )
    debug_vis=False