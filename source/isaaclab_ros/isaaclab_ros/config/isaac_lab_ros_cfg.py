# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the lidar sensor."""

from dataclasses import MISSING
from omni.isaac.lab.utils import configclass

from .camera_cfg import CameraROSCfg
from .imu_cfg import ImuROSCfg
from .lidar_cfg import LidarROSCfg
@configclass
class IsaacLabRosCfg:
    """Configuration for ROS2 nodes."""

    # Name of the robot, used for namespacing topics
    name: str = MISSING
    # Type of robot being used in simulation
    platform: str = MISSING
    # Cfg for Camera sensor
    #TODO: Figure out sensors
    #sensors: [] = MISSING
    camera: CameraROSCfg = MISSING
    # Cfg for IMU sensor
    imu: ImuROSCfg = MISSING
    # Cfgs for the lidar sensor
    lidar: LidarROSCfg = MISSING
    ## Cfg for the camera sensors
    # camera: CameraCfg = MISSING