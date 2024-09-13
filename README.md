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
```

### Download the Policy and make sure the path is set up correctly in:
https://github.com/UWRobotLearning/GroundControl/blob/425d5c45e2ee17107748571ec5bf12a4d81df53e/source/extensions/omni.isaac.groundcontrol_tasks/omni/isaac/groundcontrol_tasks/manager_based/navigation/config/spot/navigation_env_cfg.py#L54


### Run Teleop Example
```bash
# Assuming this python is tied to isaac-sim, otherwise see Isaac-Sim / IsaacLab docs:
python source/standalone/environments/teleoperation/teleop_se2_agent.py --task Isaac-Navigation-Flat-Spot-Play-v0 --num_envs 1 --teleop_device keyboard
```

## JAX installation

Note that the default IsaacLab installation installs Torch with some version of cuda and cudnn, for ecample, cuda 12, cudnn 8.9. To install JAX with GPU capabilities, we need to install JAX and more importantly jaxlib with matching versions. To see which cuda version is used, run `pip list` and look for torch. The version should look something like: `2.2.2+cu121`. This tells us we are using cuda 12.1. To see which cudnn version is installed, look for `nvidia-cudnn-cu12`. It should show something like 8.9.2.26, which tells us we have cudnn 8.9. Now, go to https://storage.googleapis.com/jax-releases/jax_cuda_releases.html and look for the latest version of jaxlib that matches your cuda version and cudnn version. In this example, that is `jaxlib==0.4.28+cuda12.cudnn89`. Go ahead and install jax as follows:

```bash
pip install jax==0.4.28 jaxlib==0.4.28+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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

If you get this error: `jaxlib.xla_extension.XlaRuntimeError: NOT_FOUND: Couldn't find a suitable version of ptxas.`, you may be able to fix it by running: 

```bash
conda install -c nvidia cuda-nvcc. See https://github.com/google/jax/discussions/6843.
```

## Installing SB3 and SBX

Stable Baselines 3 should already be installed with IsaacLab. In order to install sbx as a pip package, you can follow the instructions here: https://github.com/araffin/sbx. Alternatively, if you want to modify sbx, you can install it locally as follows:

```bash
# Assuming you are in a directory like ~/projects/GroundControl, it is advisable to install sbx in ~/projects for cleaner version control.
conda activate IsaacLab # Or the name of your conda environment
cd ~/projects ## Or wherever you want it installed
git clone git@github.com:araffin/sbx.git
cd sbx
pip install -e .
```

