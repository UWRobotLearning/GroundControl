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

In source/groundcontrol_tasks/groundcontrol_tasks/manager_based/navigation/config/go2/navigation_env_cfg.py, update the path to the GQ USD file under:
```bash
spawn=sim_utils.UsdFileCfg( ....
```

```bash
# Assuming this python is tied to isaac-sim, otherwise see Isaac-Sim / IsaacLab docs:

# Download assets
python scripts/update_assets.py
python scripts/environments/teleoperation/teleop_se2_agent_ROS2.py --task Isaac-Navigation-Flat-Go2-Play-v0 --num_envs 1 



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
