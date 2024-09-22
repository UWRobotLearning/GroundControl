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


## Installation Instructions for Isaac Lab, GroundControl, and JaxRL

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
pip install -r requirements.txt 

## Verify installation
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false python train_online.py --env_name=HalfCheetah-v4  --start_training 10000 --max_steps 1000000 --config=configs/rlpd_config.py
```