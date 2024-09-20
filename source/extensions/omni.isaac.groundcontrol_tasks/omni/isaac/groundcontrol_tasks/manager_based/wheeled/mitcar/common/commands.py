from omni.isaac.lab.envs import mdp
from omni.isaac.lab.utils import configclass

@configclass
class NoCommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()
