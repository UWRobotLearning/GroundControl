## Installing IsaacLab
https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html

## Installing GroundControl

```bash
# GroundControl sits on top of IsaacLab, and is a spearate set of packages.
# Activate the conda environment that was created via the IsaacLab setup.
conda activate IsaacLab

git clone git@github.com:UWRobotLearning/GroundControl.git
cd source/extensions
pip install -e omni.isaac.groundcontrol
pip install -e omni.isaac.groundcontrol_assets
pip install -e omni.isaac.groundcontrol_tasks

# Download the Policy and make sure the path is set up correctly in:
https://github.com/UWRobotLearning/GroundControl/blob/425d5c45e2ee17107748571ec5bf12a4d81df53e/source/extensions/omni.isaac.groundcontrol_tasks/omni/isaac/groundcontrol_tasks/manager_based/navigation/config/spot/navigation_env_cfg.py#L54

# Assuming this python is tied to isaac-sim, otherwise see Isaac-Sim / IsaacLab docs:
python /source/standalone/environments/teleoperation/teleop_se2_agent.py --task Isaac-Navigation-Flat-Spot-Play-v0 --num_envs 1 --teleop_device keyboard
```
