# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the lidar sensor."""

from dataclasses import MISSING
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.sensors.camera import CameraCfg
from omni.isaac.lab.sensors.camera import TiledCameraCfg
from .imu_cfg import ImuROSCfg

@configclass
class CameraROSCfg(CameraCfg):
    """Configuration for the camera sensor."""

    sensor_name: str = MISSING

    message_type: str = MISSING

    topic_name: str = MISSING

    imu: ImuROSCfg = None

@configclass
class TiledCameraROSCfg(TiledCameraCfg):
    """Configuration for the camera sensor."""

    sensor_name: str = MISSING

    message_type: str = MISSING

    topic_name: str = MISSING

    imu: ImuROSCfg = MISSING