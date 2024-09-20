import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.utils import configclass

from . import JOINT_NAMES

@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=250.
    )