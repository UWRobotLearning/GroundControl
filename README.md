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


# Installation Instructions for Isaac Lab, GroundControl, and JaxRL

### Isaac Lab
```bash
conda create -n gc_jaxrl python=3.10
conda activate gc_jaxrl

## Install with cuda 12.1, but you could change this (but then you'd also have to change jax versions)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

pip install --upgrade pip

## Install Isaac Sim packages necessary for Isaac Lab
pip install isaacsim
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com

## To verify Isaac Sim installation, run: 
isaacsim
isaacsim omni.isaac.sim.python.kit

## Go to your Isaac Lab directory (if you don't have it yet, clone it with `git clone git@github.com:isaac-sim/IsaacLab.git`)
cd ~/projects/IsaacLab  ## Or wherever this lives

## Install Isaac Lab
./isaaclab.sh --install

## Verify installation
python source/standalone/tutorials/00_sim/create_empty.py
```

### GroundControl

```bash
## In case you haven't cloned the GroundControl repo yet, run `git clone git@github.com:UWRobotLearning/GroundControl.git`

cd ~/projects/GroundControl ## Or wherever this is saved
cd source/extensions
pip install -e omni.isaac.groundcontrol
pip install -e omni.isaac.groundcontrol_assets
pip install -e omni.isaac.groundcontrol_tasks
```

### Jax

```bash
## Check which version of cuda is installed with torch
pip list | grep "torch"  ## This will show a version like 2.4.0+cu121, which means you have cuda 12.1

## If it doesn't show a +cu version run the following
pip list | grep cu ## Check, for instance, the version of nvidia-cublas-cu12, which should show something like 12.1.3.1

## Check your cudnn version
pip list | grep cudnn  ## For me this returns "nvidia-cudnn-cu12               9.1.0.70" wich means I have cudnn 9.1.0

## Go to https://storage.googleapis.com/jax-releases/jax_cuda_releases.html and find a version with the right cuda and cudnn, 
## for me that is cuda12/jaxlib-0.4.29+cuda12.cudnn91-cp310-cp310-manylinux2014_aarch64.whl. Install it like this
pip install jax==0.4.29 jaxlib==0.4.29+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

## Fixes issues related to jaxlib.xla_extension.XlaRuntimeError: NOT_FOUND: Couldn't find a suitable version of ptxas.
conda install -c nvidia cuda-nvcc
```

Then, to verify that you have GPU capabilities in both torch and jax, in the command line run:

```bash
python
> import jax
> jax.local_devices()  ## You should see your GPU devices. If not, you haven’t installed the right version
> import torch
> torch.cuda.device_count() ## You should see a number greater than 0
> exit()
```

### jaxrl
```bash
cd ~/projects
## In case you don't have it yet, clone git@github.com:mateoguaman/jaxrl.git
cd ~/projects/jaxrl
pip install -e . 

## Verify installation
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=HalfCheetah-v4  --start_training 10000 --max_steps 1000000 --config=configs/rlpd_config.py
```

### Installing SB3 and SBX

Stable Baselines 3 should already be installed with IsaacLab. In order to install sbx as a pip package, you can follow the instructions here: https://github.com/araffin/sbx. Alternatively, if you want to modify sbx, you can install it locally as follows:

```bash
# Assuming you are in a directory like ~/projects/GroundControl, it is advisable to install sbx in ~/projects for cleaner version control.
conda activate IsaacLab # Or the name of your conda environment
cd ~/projects ## Or wherever you want it installed
git clone git@github.com:araffin/sbx.git
cd sbx
pip install -e .
```

# Instructions for running the code

## Running pre-training with PPO

To run pre-training with PPO, run:
```bash
python source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless --video 
```

## Running data collection of PPO expert policy

To run data collection of PPO expert policy, run:
```bash
python source/standalone/workflows/rsl_rl/collect.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 64 --headless
```

## Running fine-tuning with PPO on slope environment

To run fine-tuning with PPO on slope environment, run:
```bash
# If --resume is True, there is a hardcoded path to load the model from in the train.py file. Need to fix this. 
# Add --num_envs 1 if you want to try fine-tuning in the single-agent setting.
python source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Slope-Unitree-A1-v0 --resume True --video 
```

## Running offline training with BC or IQL

To run offline training with BC or IQL, run:
```bash
# BC (Replace dataset_path with your own). To avoid rendering, add --headless.
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python source/standalone/workflows/jaxrl/train_offline.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 64 --dataset_path /home/mateo/projects/GroundControl/expert_ppo_buffer.npz --algorithm bc --video

# IQL
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python source/standalone/workflows/jaxrl/train_offline.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 64 --dataset_path /home/mateo/projects/GroundControl/expert_ppo_buffer.npz --algorithm iql --video
```

## Running online training with {SAC, REDQ, DroQ, TD3}

To run online training with SAC, REDQ, DroQ, or TD3, run:
```bash
# SAC. To avoid rendering, add --headless.
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python source/standalone/workflows/jaxrl/train_online.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 1 --algorithm sac --video

# REDQ
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python source/standalone/workflows/jaxrl/train_online.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 1 --algorithm redq --video

# DroQ
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python source/standalone/workflows/jaxrl/train_online.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 1 --algorithm droq --video

# TD3
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python source/standalone/workflows/jaxrl/train_online.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 1 --algorithm td3 --video
```

## Running hybrid RL with {RLPD-SAC, RLPD-REDQ, RLPD-DroQ}

To run hybrid RL with RLPD-SAC, RLPD-REDQ, or RLPD-DroQ, run:
```bash
# RLPD-SAC. To avoid rendering, add --headless.
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python source/standalone/workflows/jaxrl/train_hybrid.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 1 --algorithm rlpd_sac --dataset_path /home/mateo/projects/GroundControl/expert_ppo_buffer.npz --video

# RLPD-REDQ
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python source/standalone/workflows/jaxrl/train_hybrid.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 1 --algorithm rlpd_redq --dataset_path /home/mateo/projects/GroundControl/expert_ppo_buffer.npz --video

# RLPD-DroQ
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python source/standalone/workflows/jaxrl/train_hybrid.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 1 --algorithm rlpd_droq --dataset_path /home/mateo/projects/GroundControl/expert_ppo_buffer.npz --video
```

## Running offline to online with IQL

To run offline to online with IQL, run:
```bash
# To avoid rendering, add --headless. Change dataset_path to your own.
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python source/standalone/workflows/jaxrl/train_offline_to_online.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --num_envs 1 --dataset_path /home/mateo/projects/GroundControl/expert_ppo_buffer.npz --algorithm iql --video
```

## Playing a policy from a checkpoint

To play a policy from a checkpoint, run:
```bash
# Make sure to change the env, algorithm, checkpoint_dir, and checkpoint_step to the correct values.
python source/standalone/workflows/jaxrl/play.py --task Isaac-Velocity-Flat-Unitree-A1-Play-v0 --num_envs 64 --algorithm rlpd_redq --checkpoint_dir /home/mateo/projects/GroundControl/logs/jaxrl/rlpd_redq/unitree_a1_flat/2024-10-01_14-20-02/checkpoints --checkpoint_step 30000
```

