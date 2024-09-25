import gymnasium as gym
import numpy as np
import torch
from typing import Optional, Union

class RescaleActionAsymmetric:
    """
    A standalone class that rescales actions asymmetrically.
    Supports both NumPy arrays (without batch dimension) and PyTorch tensors (with or without batch dimension).
    """

    def __init__(
        self, 
        original_space,
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        center_action: Optional[np.ndarray] = None
    ):
        """
        Initialize the RescaleActionAsymmetric class.

        Args:
            original_space: The original action space, assumed to be a gym/gymnasium box space.
            low (Union[float, np.ndarray]): The lower bound(s) of the new action space.
            high (Union[float, np.ndarray]): The upper bound(s) of the new action space.
            center_action (Optional[np.ndarray]): The center of the action space. If None, it will be calculated.
        """
        self.original_space = original_space
        self.low = low
        self.high = high
        self._center_action = None
        if center_action is not None:
            self.center_action = center_action  # This will call the setter

    @property
    def center_action(self) -> np.ndarray:
        if self._center_action is not None:
            return self._center_action
        center = (self.original_space.high + self.original_space.low) / 2.0
        return center

    @center_action.setter
    def center_action(self, value: Optional[np.ndarray]):
        if value is not None:
            assert value.ndim == 1, "Center action must be a 1-dimensional NumPy array"
        self._center_action = value

    @property
    def action_space(self):
        return type(self.original_space)(
            low=self.low,
            high=self.high,
            shape=self.original_space.shape,
            dtype=self.original_space.dtype
        )

    def transform_action(self, action: Union[np.ndarray, torch.Tensor], use_torch: bool = False) -> Union[np.ndarray, torch.Tensor]:
        '''Transform an action from the new space to the original space. It "unnormalizes" the action to the original space.'''
        if use_torch:
            return self._transform_action_torch(action)
        else:
            return self._transform_action_numpy(action)

    def _transform_action_numpy(self, action: np.ndarray) -> np.ndarray:
        assert action.ndim == 1, "NumPy action must be 1-dimensional"
        new_center = (self.high + self.low) / 2.0
        new_delta_action = action - new_center

        new_delta_high = self.high - new_center
        new_delta_low = new_center - self.low

        center_action = self.center_action

        delta_center = np.where(
            new_delta_action < 0,
            new_delta_action * ((center_action - self.original_space.low) / new_delta_low),
            new_delta_action * ((self.original_space.high - center_action) / new_delta_high)
        )
        
        return (center_action + delta_center)

    def _transform_action_torch(self, action: torch.Tensor) -> torch.Tensor:
        is_batched = action.ndim == 2
        device = action.device
        high = torch.Tensor([self.high]).to(device).repeat(self.original_space.shape[0]) if (isinstance(self.high, float) or isinstance(self.high, int)) else self.high
        low = torch.Tensor([self.low]).to(device).repeat(self.original_space.shape[0]) if (isinstance(self.low, float) or isinstance(self.low, int)) else self.low
        new_center = (high + low) / 2.0
        new_delta_action = action - new_center

        new_delta_high = high - new_center
        new_delta_low = new_center - low

        center_action = torch.Tensor(self.center_action).to(device)

        if is_batched:
            new_delta_high = new_delta_high.expand_as(action)
            new_delta_low = new_delta_low.expand_as(action)
            center_action = center_action.expand_as(action)

        delta_center = torch.where(
            new_delta_action < 0,
            new_delta_action * ((center_action - torch.Tensor(self.original_space.low).to(device)) / new_delta_low),
            new_delta_action * ((torch.Tensor(self.original_space.high).to(device) - center_action) / new_delta_high)
        )
        
        return (center_action + delta_center)

    def inverse_transform_action(self, action: Union[np.ndarray, torch.Tensor], use_torch: bool = False) -> Union[np.ndarray, torch.Tensor]:
        '''Transform an action from the original space to the new space. It "normalizes" the action to the new space.'''
        if use_torch:
            return self._inverse_transform_action_torch(action)
        else:
            return self._inverse_transform_action_numpy(action)

    def _inverse_transform_action_numpy(self, action: np.ndarray) -> np.ndarray:
        assert action.ndim == 1, "NumPy action must be 1-dimensional"
        new_center = (self.high + self.low) / 2.0
        delta_center = action - self.center_action

        new_delta_high = self.high - new_center
        new_delta_low = new_center - self.low

        new_delta_center = np.where(
            delta_center < 0,
            delta_center * (new_delta_low / (self.center_action - self.original_space.low)),
            delta_center * (new_delta_high / (self.original_space.high - self.center_action))
        )
        
        return new_center + new_delta_center

    def _inverse_transform_action_torch(self, action: torch.Tensor) -> torch.Tensor:
        is_batched = action.ndim == 2
        device = action.device
        high = torch.Tensor([self.high]).to(device).repeat(self.original_space.shape[0]) if (isinstance(self.high, float) or isinstance(self.high, int)) else self.high
        low = torch.Tensor([self.low]).to(device).repeat(self.original_space.shape[0]) if (isinstance(self.low, float) or isinstance(self.low, int)) else self.low
        new_center = (high + low) / 2.0
        delta_center = action - torch.Tensor(self.center_action).to(device)

        new_delta_high = high - new_center
        new_delta_low = new_center - low

        if is_batched:
            new_delta_high = new_delta_high.expand_as(action)
            new_delta_low = new_delta_low.expand_as(action)
            center_action = torch.Tensor(self.center_action).to(device).expand_as(action)
        else:
            center_action = torch.Tensor(self.center_action).to(device)

        new_delta_center = torch.where(
            delta_center < 0,
            delta_center * (new_delta_low / (center_action - torch.Tensor(self.original_space.low).to(device))),
            delta_center * (new_delta_high / (torch.Tensor(self.original_space.high).to(device) - center_action))
        )
        
        return new_center + new_delta_center
    
class RescaleActionAsymmetricWrapper(gym.Wrapper):
    """
    A Gymnasium wrapper that uses RescaleActionAsymmetric to rescale actions.
    """

    def __init__(
        self,
        env: gym.Env,
        low: Union[float, np.ndarray] = -1.0,
        high: Union[float, np.ndarray] = 1.0,
        center_action: Optional[np.ndarray] = None
    ):
        """
        Initialize the RescaleActionAsymmetricWrapper.

        Args:
            env (gym.Env): The environment to wrap.
            low (Union[float, np.ndarray]): The lower bound(s) of the new action space.
            high (Union[float, np.ndarray]): The upper bound(s) of the new action space.
            center_action (Optional[np.ndarray]): The center of the action space. If None, it will be calculated.
        """
        super().__init__(env)
        self.rescaler = RescaleActionAsymmetric(
            original_space=env.action_space,
            low=low,
            high=high,
            center_action=center_action
        )
        
        # Update the action space to reflect the new bounds
        self.action_space = self.rescaler.action_space

    def step(self, action):
        """
        Step the environment with a rescaled action.

        Args:
            action (Union[np.ndarray, torch.Tensor]): The action in the new space.

        Returns:
            The step return from the underlying environment.
        """
        use_torch = isinstance(action, torch.Tensor)
        rescaled_action = self.rescaler.transform_action(action, use_torch=use_torch)
        return self.env.step(rescaled_action)

    def reset(self, **kwargs):
        """
        Reset the environment.

        Returns:
            The reset return from the underlying environment.
        """
        return self.env.reset(**kwargs)

    def render(self, mode='human'):
        """
        Render the environment.

        Args:
            mode (str): The render mode.

        Returns:
            The render return from the underlying environment.
        """
        return self.env.render(mode)