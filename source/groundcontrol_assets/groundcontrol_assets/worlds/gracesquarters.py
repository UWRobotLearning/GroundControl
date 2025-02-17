# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2022-2025, The GroundControl Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the GracesQuarters USD-based terrain and map.

"""

import os
# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV
from groundcontrol_assets import GROUNDCONTROL_ASSETS_DATA_DIR

def is_missing_world_USD_file():
    """Check if the world USD file is missing."""
    graces_quarters_path = os.path.join(GROUNDCONTROL_ASSETS_DATA_DIR, "Worlds", "Collected_GQ_lite", "GQ_lite.usd")
    return not os.path.exists(graces_quarters_path)