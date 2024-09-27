# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple
import math
from dataclasses import MISSING

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from .commands import GoalVelocityCommand


@configclass
class GoalVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = GoalVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    
    following_velocity: float = 2.0

    goal_position: Tuple[float, float] = (0.0, 0.0)

    goal_reach_threshold: float = 0.5

    heading_control_stiffness: float = MISSING

    def __post_init__(self):
        """Post initialization."""
        # set the resampling time range to infinity to avoid resampling
        self.resampling_time_range = (10 ** 8, 10 ** 8)

