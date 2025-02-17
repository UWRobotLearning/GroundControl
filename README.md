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
```

### Run Teleop Example
```bash
# Assuming this python is tied to isaac-sim, otherwise see Isaac-Sim / IsaacLab docs:
python source/standalone/environments/teleoperation/teleop_se2_agent.py --task Isaac-Navigation-Flat-Spot-Play-v0 --num_envs 1 --teleop_device keyboard
```



### VSCode Debug 
Create a .vscode/launch.json by following these steps:
  {
      "name": "Python: Teleop GroundControl",
      "type": "debugpy",
      "request": "launch",
      "args" : ["--task", "Isaac-Navigation-Flat-Spot-Play-v0", "--num_envs", "1", "--teleop_device", "keyboard", "--sensitivity", "2"],
      "program": "${workspaceFolder}/scripts/environments/teleoperation/teleop_se2_agent.py",
      "console": "integratedTerminal"
  }
```
