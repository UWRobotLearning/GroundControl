# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2022-2025, The Ground Control Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the locomotion environments."""

# Some mdp functionality is taken from IsaacLab, for example, these:
from isaaclab.envs.mdp import *  # noqa: F401, F403

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV
# This first import is important to include since
# we also now have groundcontrol specific mdp terms
from groundcontrol.envs.mdp import *  # noqa: F401, F403
from .pre_trained_policy_action import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403