
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm

from .. import utils

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        root_pos_w = ObsTerm(func=utils.root_pos_w) # position in simulation world frame
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

