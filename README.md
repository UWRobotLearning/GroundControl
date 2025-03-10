## Installing IsaacLab
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installation-using-isaac-sim-pip

## Installing GroundControl

```bash
# GroundControl sits on top of IsaacLab, and is a spearate set of packages.
# Activate the conda environment that was created via the IsaacLab setup.
conda activate IsaacLab

git clone https://github.com/UWRobotLearning/GroundControl.git
cd source
pip install -e groundcontrol
pip install -e groundcontrol_assets
pip install -e groundcontrol_tasks
pip install -e isaaclab_ros

```

### Run Example GQ Environment

Prior to running the simulator, add `"isaacsim.ros2.bridge" = {}` to `IsaacLab/apps/isaaclab.python.kit` under `[dependencies]`.

In source/groundcontrol_tasks/groundcontrol_tasks/manager_based/navigation/config/go2/navigation_env_cfg.py, update the path to the GQ USD file under:
```bash
spawn=sim_utils.UsdFileCfg( ....
```

```bash

#Assuming this python is tied to isaac-sim, otherwise see Isaac-Sim / IsaacLab docs:

#Download assets
python scripts/update_assets.py
python scripts/environments/teleoperation/teleop_se2_agent_ROS2.py --task Isaac-Navigation-Flat-Go2-Play-v0 --num_envs 1 
```


### VSCode Debug 
Create a .vscode/launch.json by following these steps:

https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/vs_code.html

```
  {
      "name": "Python: Teleop GroundControl",
      "type": "debugpy",
      "request": "launch",
      "args" : ["--task", "Isaac-Navigation-Flat-Go2-Play-v0", "--num_envs", "1"],
      "program": "${workspaceFolder}/scripts/environments/teleoperation/teleop_se2_agent_ROS2.py",
      "console": "integratedTerminal"
  }
```

### Go2.usd

The simulated Go2 uses action graphs to publish ROS2 topics needed to interface with the autonomy stack along with a front depth camera topic (not needed to run the basic stack). 

The minimum topics needed to interface with the stack are:

```
/clock
/go2/imu/data
/go2/lidar_points
/tf
/tf_static
```

The front camera topics are:

```
/go2/realsense_front/color/camera_info
/go2/realsense_front/color/image_raw
/go2/realsense_front/depth/image_rect_raw
```

### Isaaclab_ros
The isaaclab_ros package is needed to publish a subscriber node to translate cmd_vel from the autonomy stack to base_commands for the simulated Go2.

Currently, there is a bug in the IsaacSim odometry action graph that also requires IsaacLab_ros to publish the odometry node. This will be fixed in a future release.
