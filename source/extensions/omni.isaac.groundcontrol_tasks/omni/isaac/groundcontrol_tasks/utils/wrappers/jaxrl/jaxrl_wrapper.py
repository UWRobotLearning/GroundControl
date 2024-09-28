# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure a :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv` instance to JaxRL environment.

The following example shows how to wrap an environment for Stable-Baselines3:

.. code-block:: python

    from omni.isaac.groundcontrol_tasks.utils.wrappers.jaxrl import JaxrlEnvWrapper

    env = JaxrlEnvWrapper(env)

"""

# needed to import for allowing type-hinting: torch.Tensor | dict[str, torch.Tensor]
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn  # noqa: F401
import jax
import jax.numpy as jnp
from typing import Any

# from stable_baselines3.common.utils import constant_fn
# from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv
from .rescale_action_asymmetric import RescaleActionAsymmetric
"""
Configuration Parser.
"""


# def process_sb3_cfg(cfg: dict) -> dict:
#     """Convert simple YAML types to Stable-Baselines classes/components.

#     Args:
#         cfg: A configuration dictionary.

#     Returns:
#         A dictionary containing the converted configuration.

#     Reference:
#         https://github.com/DLR-RM/rl-baselines3-zoo/blob/0e5eb145faefa33e7d79c7f8c179788574b20da5/utils/exp_manager.py#L358
#     """

#     def update_dict(hyperparams: dict[str, Any]) -> dict[str, Any]:
#         for key, value in hyperparams.items():
#             if isinstance(value, dict):
#                 update_dict(value)
#             else:
#                 if key in ["policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"]:
#                     hyperparams[key] = eval(value)
#                 elif key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
#                     if isinstance(value, str):
#                         _, initial_value = value.split("_")
#                         initial_value = float(initial_value)
#                         hyperparams[key] = lambda progress_remaining: progress_remaining * initial_value
#                     elif isinstance(value, (float, int)):
#                         # Negative value: ignore (ex: for clipping)
#                         if value < 0:
#                             continue
#                         hyperparams[key] = constant_fn(float(value))
#                     else:
#                         raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")

#         return hyperparams

#     # parse agent configuration and convert to classes
#     return update_dict(cfg)


"""
Vectorized environment wrapper.
"""


class JaxrlEnvWrapper(gym.Env):
    """Wraps around Isaac Lab environment for JaxRL.

    For now, we assume that we only have one environment. The wrapper will return a gym.Env environment,
    with the main change being that the actions will probably be rescaled. Most of the algorithms in jaxrl are 
    off-policy algorithms, and most of them return an action that is in the range [-1, 1]. This action should
    be re-scaled to the action space of the environment we are dealing with. Currently, Isaac Lab does not set action
    spaces correctly, so we will hard code it.

    We also add monitoring functionality that computes the un-discounted episode
    return and length. This information is added to the info dicts under key `episode`.

    In contrast to the Isaac Lab environment, jaxrl expect the following:

    1. numpy datatype for MDP signals
    2. Action range in [-1, 1]

    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env: ManagerBasedRLEnv = env
        # collect common information
        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode

        # obtain gym spaces
        # note: stable-baselines3 does not like when we have unbounded action space so
        #   we set it to some high value here. Maybe this is not general but something to think about.
        self.observation_space = self.unwrapped.single_observation_space["policy"]
        self.full_observation_space = self.unwrapped.observation_space["policy"]
        action_space = self.unwrapped.single_action_space
        self.full_action_space = self.unwrapped.action_space
        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            ## TODO: This needs to be set to be the proper joint limits of the USD. We may also need to have an action 
            ## space that represents the [-1, 1] action space that jaxrl expects
            # action_space = gym.spaces.Box(low=-100, high=100, shape=action_space.shape)
            lower_limit = self.env.scene['robot'].data.joint_limits[0, :, 0].cpu().detach().numpy()
            upper_limit = self.env.scene['robot'].data.joint_limits[0, :, 1].cpu().detach().numpy()
            self.action_space = gym.spaces.Box(low=lower_limit, high=upper_limit, shape=action_space.shape)
            # Reshape lower_limit and upper_limit
            lower_limit = lower_limit.reshape(1, -1).repeat(self.num_envs, axis=0)
            upper_limit = upper_limit.reshape(1, -1).repeat(self.num_envs, axis=0)
            ## This is useful for sampling actions for parallel environments
            self.full_action_space = gym.spaces.Box(
                low=lower_limit,
                high=upper_limit,
                shape=self.full_action_space.shape,
                dtype=self.full_action_space.dtype
            )
        # add buffer for logging episodic information
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)

        ## TODO: Support symmetric action scaler as well
        self.action_scaler = RescaleActionAsymmetric(self.action_space, low=-1, high=1, center_action=np.zeros(self.action_space.shape))

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_episode_rewards(self) -> list[float]:
        """Returns the rewards of all the episodes."""
        return self._ep_rew_buf.cpu().tolist()

    def get_episode_lengths(self) -> list[int]:
        """Returns the number of time-steps of all the episodes."""
        return self._ep_len_buf.cpu().tolist()

    """
    Operations - MDP
    """

    def seed(self, seed: int | None = None) -> list[int | None]:  # noqa: D102
        return [self.unwrapped.seed(seed)] * self.unwrapped.num_envs

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> np.ndarray | dict[str, np.ndarray]:  # noqa: D102
        obs_dict, info = self.env.reset(seed=seed, options=options)
        # reset episodic information buffers
        self._ep_rew_buf.zero_()
        self._ep_len_buf.zero_()
        # convert data types to numpy depending on backend
        return self._process_obs(obs_dict), info

    # def step_async(self, actions):  # noqa: D102
    #     # convert input to numpy array
    #     if not isinstance(actions, torch.Tensor):
    #         actions = np.asarray(actions)
    #         actions = torch.from_numpy(actions).to(device=self.sim_device, dtype=torch.float32)
    #     else:
    #         actions = actions.to(device=self.sim_device, dtype=torch.float32)
    #     # convert to tensor
    #     self._async_actions = actions

    # def step_wait(self) -> VecEnvStepReturn:  # noqa: D102
    #     # record step information
    #     obs_dict, rew, terminated, truncated, extras = self.env.step(self._async_actions)
    #     # update episode un-discounted return and length
    #     self._ep_rew_buf += rew
    #     self._ep_len_buf += 1
    #     # compute reset ids
    #     dones = terminated | truncated
    #     reset_ids = (dones > 0).nonzero(as_tuple=False)

    #     # convert data types to numpy depending on backend
    #     # note: ManagerBasedRLEnv uses torch backend (by default).
    #     obs = self._process_obs(obs_dict)
    #     rew = rew.detach().cpu().numpy()
    #     terminated = terminated.detach().cpu().numpy()
    #     truncated = truncated.detach().cpu().numpy()
    #     dones = dones.detach().cpu().numpy()
    #     # convert extra information to list of dicts
    #     infos = self._process_extras(obs, terminated, truncated, extras, reset_ids)

    #     # reset info for terminated environments
    #     self._ep_rew_buf[reset_ids] = 0
    #     self._ep_len_buf[reset_ids] = 0

    #     return obs, rew, dones, infos
    
    def step(self, normalized_actions): 
        assert normalized_actions.all() <= 1.0 and normalized_actions.all() >= -1.0, "Input actions should be normalized, i.e. in the range [-1, 1]"
        # Convert action to unnormalized values
        unnormalized_actions = self.action_scaler.transform_action(normalized_actions, use_torch=torch.is_tensor(normalized_actions))

        # Convert actions to tensor
        if not isinstance(unnormalized_actions, torch.Tensor):
            unnormalized_actions = np.asarray(unnormalized_actions)
            unnormalized_actions = torch.from_numpy(unnormalized_actions).to(device=self.sim_device, dtype=torch.float32)
        else:
            unnormalized_actions = unnormalized_actions.to(device=self.sim_device, dtype=torch.float32)

        # Step the environment
        obs_dict, rew, terminated, truncated, extras = self.env.step(unnormalized_actions)

        # Update episode un-discounted return and length
        self._ep_rew_buf += rew
        self._ep_len_buf += 1

        # Compute done and reset ids
        dones = terminated | truncated
        reset_ids = (dones > 0).nonzero(as_tuple=False)

        # convert data types to numpy depending on backend
        # note: ManagerBasedRLEnv uses torch backend (by default).
        obs = self._process_obs(obs_dict)
        rew = rew.detach().cpu().numpy()
        terminated = terminated.detach().cpu().numpy()
        truncated = truncated.detach().cpu().numpy()
        dones = dones.detach().cpu().numpy()

        # Convert extra information to list of dicts
        infos = self._process_extras(obs, terminated, truncated, extras, reset_ids)
        ## TODO: Check if this makes sense

        # reset info for terminated environments
        self._ep_rew_buf[reset_ids] = 0
        self._ep_len_buf[reset_ids] = 0

        # # Squeeze outputs if num_envs is 1
        # if self.num_envs == 1:
        #     obs = self._squeeze_output(obs)
        #     rew = np.squeeze(rew)
        #     terminated = np.squeeze(terminated)
        #     truncated = np.squeeze(truncated)
        #     infos = infos[0] if infos else {}

        # return obs, reward, terminated, truncated, extras
        return obs, rew, terminated, truncated, infos

    def _squeeze_output(self, obs):
        """Squeeze observation output when num_envs is 1."""
        if isinstance(obs, dict):
            return {k: np.squeeze(v) for k, v in obs.items()}
        else:
            return np.squeeze(obs)

    def close(self):  # noqa: D102
        self.env.close()

    def get_attr(self, attr_name, indices=None):  # noqa: D102
        # resolve indices
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)
        # obtain attribute value
        attr_val = getattr(self.env, attr_name)
        # return the value
        if not isinstance(attr_val, torch.Tensor):
            return [attr_val] * num_indices
        else:
            return attr_val[indices].detach().cpu().numpy()

    def set_attr(self, attr_name, value, indices=None):  # noqa: D102
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):  # noqa: D102
        if method_name == "render":
            # gymnasium does not support changing render mode at runtime
            return self.env.render()
        else:
            # this isn't properly implemented but it is not necessary.
            # mostly done for completeness.
            env_method = getattr(self.env, method_name)
            return env_method(*method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):  # noqa: D102
        raise NotImplementedError("Checking if environment is wrapped is not supported.")

    def get_images(self):  # noqa: D102
        raise NotImplementedError("Getting images is not supported.")

    """
    Helper functions.
    """

    def _process_obs(self, obs_dict: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""
        # Sb3 doesn't support asymmetric observation spaces, so we only use "policy"
        obs = obs_dict["policy"]
        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):
            for key, value in obs.items():
                obs[key] = value.detach().cpu().numpy()
        elif isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")
        return obs

    def _process_extras(
        self, obs: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, extras: dict, reset_ids: np.ndarray
    ) -> list[dict[str, Any]]:
        """Convert miscellaneous information into dictionary for each sub-environment."""
        # create empty list of dictionaries to fill
        infos: list[dict[str, Any]] = [dict.fromkeys(extras.keys()) for _ in range(self.num_envs)]
        # fill-in information for each sub-environment
        # note: This loop becomes slow when number of environments is large.
        for idx in range(self.num_envs):
            # fill-in episode monitoring info
            if idx in reset_ids:
                infos[idx]["episode"] = dict()
                infos[idx]["episode"]["r"] = float(self._ep_rew_buf[idx])
                infos[idx]["episode"]["l"] = float(self._ep_len_buf[idx])
            else:
                infos[idx]["episode"] = None
            # fill-in bootstrap information
            infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
            # fill-in information from extras
            for key, value in extras.items():
                # 1. remap extra episodes information safely
                # 2. for others just store their values
                if key == "log":
                    # only log this data for episodes that are terminated
                    if infos[idx]["episode"] is not None:
                        for sub_key, sub_value in value.items():
                            infos[idx]["episode"][sub_key] = sub_value
                else:
                    infos[idx][key] = value[idx]
            # add information about terminal observation separately
            # if idx in reset_ids:
            #     # extract terminal observations
            #     if isinstance(obs, dict):
            #         terminal_obs = dict.fromkeys(obs.keys())
            #         for key, value in obs.items():
            #             terminal_obs[key] = value[idx]
            #     else:
            #         terminal_obs = obs[idx]
            #     # add info to dict
            #     infos[idx]["terminal_observation"] = terminal_obs
            # else:
            #     infos[idx]["terminal_observation"] = None
        # return list of dictionaries
        return infos