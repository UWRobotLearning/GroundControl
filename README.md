## Installing IsaacLab
https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html

## Installing GroundControl

```bash
conda activate IsaacLab
git clone git@github.com:UWRobotLearning/GroundControl.git
cd source/extensions
pip install -e omni.isaac.groundcontrol omni.isaac.groundcontrol_assets omni.isaac.groundcontrol_tasks

# Assuming this python is tied to isaac-sim, otherwise see Isaac-Sim / IsaacLab docs:
python /source/standalone/environments/teleoperation/teleop_se2_agent.py --task Isaac-Navigation-Flat-Spot-Play-v0 --num_envs 1 --teleop_device keyboard
```
