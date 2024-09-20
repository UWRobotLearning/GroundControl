from omni.isaac.lab.envs import mdp
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from . import JOINT_NAMES

@configclass
class EventCfg:
    """Configuration for events."""

    # on startup
    # add_pole_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("chassis"),
    #         "mass_distribution_params": (0.1, 0.5),
    #         "operation": "add",
    #     },
    # )

    # on reset
    reset_car_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES),
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        },
    )

    reset_car_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=['chassis']),
            "pose_range": {},
            "velocity_range": {},
        },
    )
