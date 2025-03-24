# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# Copyright (c) 2022-2024, The Ground Control Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing the core framework."""

import os
import toml

# Conveniences to other module directories via relative paths
GROUNDCONTROL_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""


GROUNDCONTROL_METADATA = toml.load(os.path.join(GROUNDCONTROL_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = GROUNDCONTROL_METADATA["package"]["version"]
