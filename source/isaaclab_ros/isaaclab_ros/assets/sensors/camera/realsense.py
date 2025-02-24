# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for Velodyne LiDAR sensors.

This file is a copy from IsaacLab and is here 
to serve as an example sensor set up.
"""

from isaaclab.sim.spawners.sensors import PinholeCameraCfg
from isaaclab_ros.config import CameraROSCfg, ImuROSCfg

##
# Configuration
##

REALSENSE_D455_ROS_CFG = CameraROSCfg(
    sensor_name="camera",
    message_type="sensor_msgs/image",
    topic_name="/image",
    spawn=PinholeCameraCfg(
        clipping_range=(0.01, 1000000.0), 
        focal_length=1.88, 
        focus_distance=0.5, 
        horizontal_aperture=3.896, 
        vertical_aperture=2.453,
    ),
    data_types=["rgb", "depth"],
    width=1280,
    height=720,
    debug_vis=True,
    imu=ImuROSCfg(
        message_type="sensor_msgs/imu",
        topic_name="realsense_d455/imu",
    ),
)
"""Configuration for Realsense D455 RGBD Camera for ROS2 as a :class:`CameraROSCfg`.

Reference: https://www.intelrealsense.com/download/21345/?tmstv=1697035582
"""
