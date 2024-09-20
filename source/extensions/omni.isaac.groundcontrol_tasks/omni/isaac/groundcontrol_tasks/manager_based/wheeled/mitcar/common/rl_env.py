from dataclasses import field

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg

from . import ObservationsCfg, ActionsCfg, EventCfg

@configclass
class MITCarRLCommonCfg(ManagerBasedRLEnvCfg):
    """
    Common configuration for the MIT Car environment.
    Includes the basic settings:
    - Observations
    - Actions
    - Events
    as well as the number of environments and the spacing between them.

    Also sets sim dt and decimation.
    """

    num_envs: int = field(default=5)
    env_spacing: float = field(default=0.05)

    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):

        self.sim.dt = 0.025  # sim step every 25ms = 40Hz
        self.decimation = 4  # env step every 4 sim steps: 40Hz / 4 = 10Hz
        self.sim.render_interval = self.decimation
