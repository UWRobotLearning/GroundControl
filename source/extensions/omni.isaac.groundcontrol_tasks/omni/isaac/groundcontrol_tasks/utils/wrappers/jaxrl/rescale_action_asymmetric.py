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
        is_batched = action.ndim == 2
        if not is_batched:
            assert action.ndim == 1, "NumPy action must be 1-dimensional or 2-dimensional (batch)"
            action = action[np.newaxis, :]

        new_center = (self.high + self.low) / 2.0
        new_delta_action = action - new_center

        new_delta_high = self.high - new_center
        new_delta_low = new_center - self.low

        center_action = self.center_action

        if is_batched:
            new_delta_high = np.expand_dims(new_delta_high, 0)
            new_delta_low = np.expand_dims(new_delta_low, 0)
            center_action = np.expand_dims(center_action, 0)

        delta_center = np.where(
            new_delta_action < 0,
            new_delta_action * ((center_action - self.original_space.low) / new_delta_low),
            new_delta_action * ((self.original_space.high - center_action) / new_delta_high)
        )
        
        result = center_action + delta_center

        if not is_batched:
            result = result.squeeze(0)

        return result

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
        is_batched = action.ndim == 2
        if not is_batched:
            assert action.ndim == 1, "NumPy action must be 1-dimensional or 2-dimensional (batch)"
            action = action[np.newaxis, :]

        new_center = (self.high + self.low) / 2.0
        delta_center = action - self.center_action

        new_delta_high = self.high - new_center
        new_delta_low = new_center - self.low

        if is_batched:
            new_delta_high = np.expand_dims(new_delta_high, 0)
            new_delta_low = np.expand_dims(new_delta_low, 0)
            center_action = np.expand_dims(self.center_action, 0)
        else:
            center_action = self.center_action

        new_delta_center = np.where(
            delta_center < 0,
            delta_center * (new_delta_low / (center_action - self.original_space.low)),
            delta_center * (new_delta_high / (self.original_space.high - center_action))
        )
        
        result = new_center + new_delta_center

        if not is_batched:
            result = result.squeeze(0)

        return result

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
    

if __name__ == "__main__":
    import sys

    def print_test_result(test_name: str, passed: bool, error_msg: str = ""):
        status = "PASSED" if passed else "FAILED"
        print(f"Test {test_name}: {status}")
        if not passed and error_msg:
            print(f"    Error: {error_msg}")

    # Create a 12-dimensional gym Box action space with random limits between -2 and 2
    dim = 12
    np.random.seed(42)  # For reproducibility
    low = np.random.uniform(-5, 0, dim)
    high = np.random.uniform(0, 5, dim)
    # Ensure that for each dimension, low < high
    for i in range(dim):
        if low[i] > high[i]:
            low[i], high[i] = high[i], low[i]

    action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    rescaler = RescaleActionAsymmetric(original_space=action_space, low=-1, high=1, center_action=np.zeros(dim))

    # Define test cases
    test_cases = [
        "Transform and Inverse Transform - NumPy Single Dimensional",
        "Transform and Inverse Transform - NumPy Batched",
        "Transform and Inverse Transform - Torch Single Dimensional",
        "Transform and Inverse Transform - Torch Batched",
        "Center Action Automatic Calculation",
        "Center Action Setter",
        "Center Action Invalid Dimension",
        "Transform Invalid Dimension - NumPy",
        "Inverse Transform Invalid Dimension - NumPy",
        "Transform Invalid Dimension - Torch",
        "Inverse Transform Invalid Dimension - Torch",
        "Transform Action Range - NumPy",
        "Transform Action Range - Torch",
        "Round Trip - NumPy Batched",
        "Round Trip - Torch Batched",
        "No Batch Transform - NumPy Single Dimension",
        "No Batch Transform - Torch Single Dimension"
    ]

    # Execute tests
    print("Running RescaleActionAsymmetric Tests...\n")

    # 1. Transform and Inverse Transform - NumPy Single Dimensional
    try:
        action = np.random.uniform(low=rescaler.low, high=rescaler.high)
        transformed = rescaler.transform_action(action, use_torch=False)
        inverse_transformed = rescaler.inverse_transform_action(transformed, use_torch=False)
        np.testing.assert_allclose(action, inverse_transformed, rtol=1e-5, atol=1e-8)
        print_test_result(test_cases[0], True)
    except Exception as e:
        print_test_result(test_cases[0], False, str(e))

    # 2. Transform and Inverse Transform - NumPy Batched
    try:
        actions = np.random.uniform(low=rescaler.low, high=rescaler.high, size=(3, dim))
        transformed = rescaler.transform_action(actions, use_torch=False)
        inverse_transformed = rescaler.inverse_transform_action(transformed, use_torch=False)
        np.testing.assert_allclose(actions, inverse_transformed, rtol=1e-5, atol=1e-8)
        print_test_result(test_cases[1], True)
    except Exception as e:
        print_test_result(test_cases[1], False, str(e))

    # 3. Transform and Inverse Transform - Torch Single Dimensional
    try:
        action = torch.tensor(np.random.uniform(low=rescaler.low, high=rescaler.high), dtype=torch.float32)
        transformed = rescaler.transform_action(action, use_torch=True)
        inverse_transformed = rescaler.inverse_transform_action(transformed, use_torch=True)
        np.testing.assert_allclose(action.numpy(), inverse_transformed.numpy(), rtol=1e-5, atol=1e-8)
        print_test_result(test_cases[2], True)
    except Exception as e:
        print_test_result(test_cases[2], False, str(e))

    # 4. Transform and Inverse Transform - Torch Batched
    try:
        actions = torch.tensor(np.random.uniform(low=rescaler.low, high=rescaler.high, size=(3, dim)), dtype=torch.float32)
        transformed = rescaler.transform_action(actions, use_torch=True)
        inverse_transformed = rescaler.inverse_transform_action(transformed, use_torch=True)
        np.testing.assert_allclose(actions.numpy(), inverse_transformed.numpy(), rtol=1e-5, atol=1e-8)
        print_test_result(test_cases[3], True)
    except Exception as e:
        print_test_result(test_cases[3], False, str(e))

    # 5. Center Action Automatic Calculation
    try:
        expected_center = (action_space.high + action_space.low) / 2.0
        np.testing.assert_array_equal(rescaler.center_action, expected_center)
        print_test_result(test_cases[4], True)
    except Exception as e:
        print_test_result(test_cases[4], False, str(e))

    # 6. Center Action Setter
    try:
        new_center = np.random.uniform(low=rescaler.low, high=rescaler.high)
        rescaler.center_action = new_center
        np.testing.assert_array_equal(rescaler.center_action, new_center)
        print_test_result(test_cases[5], True)
    except Exception as e:
        print_test_result(test_cases[5], False, str(e))

    # 7. Center Action Invalid Dimension
    try:
        rescaler.center_action = np.random.uniform(low=-1, high=1, size=(dim, 1))  # Invalid dimension
        print_test_result(test_cases[6], False, "AssertionError was not raised for invalid center_action dimension.")
    except AssertionError as e:
        print_test_result(test_cases[6], True)
    except Exception as e:
        print_test_result(test_cases[6], False, f"Unexpected exception: {str(e)}")

    # 8. Transform Invalid Dimension - NumPy
    try:
        invalid_action = np.random.uniform(low=rescaler.low, high=rescaler.high, size=(1, 1, dim))  # 3D array
        rescaler.transform_action(invalid_action, use_torch=False)
        print_test_result(test_cases[7], False, "AssertionError was not raised for invalid NumPy action dimension.")
    except AssertionError as e:
        print_test_result(test_cases[7], True)
    except Exception as e:
        print_test_result(test_cases[7], False, f"Unexpected exception: {str(e)}")

    # 9. Inverse Transform Invalid Dimension - NumPy
    try:
        invalid_action = np.random.uniform(low=rescaler.low, high=rescaler.high, size=(1, 1, dim))  # 3D array
        rescaler.inverse_transform_action(invalid_action, use_torch=False)
        print_test_result(test_cases[8], False, "AssertionError was not raised for invalid NumPy inverse action dimension.")
    except AssertionError as e:
        print_test_result(test_cases[8], True)
    except Exception as e:
        print_test_result(test_cases[8], False, f"Unexpected exception: {str(e)}")

    # 10. Transform Invalid Dimension - Torch
    try:
        invalid_action = torch.rand(1, 1, dim)  # 3D tensor
        rescaler.transform_action(invalid_action, use_torch=True)
        print_test_result(test_cases[9], False, "RuntimeError was not raised for invalid Torch action dimension.")
    except RuntimeError as e:
        print_test_result(test_cases[9], True)
    except Exception as e:
        print_test_result(test_cases[9], False, f"Unexpected exception: {str(e)}")

    # 11. Inverse Transform Invalid Dimension - Torch
    try:
        invalid_action = torch.rand(1, 1, dim)  # 3D tensor
        rescaler.inverse_transform_action(invalid_action, use_torch=True)
        print_test_result(test_cases[10], False, "RuntimeError was not raised for invalid Torch inverse action dimension.")
    except RuntimeError as e:
        print_test_result(test_cases[10], True)
    except Exception as e:
        print_test_result(test_cases[10], False, f"Unexpected exception: {str(e)}")

    # 12. Transform Action Range - NumPy
    try:
        action = np.array(rescaler.high)  # Edge of new space
        transformed = rescaler.transform_action(action, use_torch=False)
        assert np.all(transformed >= rescaler.original_space.low), "Transformed action below original space."
        assert np.all(transformed <= rescaler.original_space.high), "Transformed action above original space."
        print_test_result(test_cases[11], True)
    except Exception as e:
        print_test_result(test_cases[11], False, str(e))

    # 13. Transform Action Range - Torch
    try:
        action = torch.tensor(rescaler.high, dtype=torch.float32)  # Edge of new space
        transformed = rescaler.transform_action(action, use_torch=True)
        assert torch.all(transformed >= torch.tensor(rescaler.original_space.low, dtype=torch.float32)), "Transformed Torch action below original space."
        assert torch.all(transformed <= torch.tensor(rescaler.original_space.high, dtype=torch.float32)), "Transformed Torch action above original space."
        print_test_result(test_cases[12], True)
    except Exception as e:
        print_test_result(test_cases[12], False, str(e))

    # 14. Round Trip - NumPy Batched
    try:
        actions = np.random.uniform(low=rescaler.low, high=rescaler.high, size=(10, dim))
        transformed = rescaler.transform_action(actions, use_torch=False)
        inverse_transformed = rescaler.inverse_transform_action(transformed, use_torch=False)
        np.testing.assert_allclose(actions, inverse_transformed, rtol=1e-5, atol=1e-8)
        print_test_result(test_cases[13], True)
    except Exception as e:
        print_test_result(test_cases[13], False, str(e))

    # 15. Round Trip - Torch Batched
    try:
        actions = torch.rand(10, dim) * (rescaler.high - rescaler.low) + rescaler.low
        transformed = rescaler.transform_action(actions, use_torch=True)
        inverse_transformed = rescaler.inverse_transform_action(transformed, use_torch=True)
        np.testing.assert_allclose(actions.numpy(), inverse_transformed.numpy(), rtol=1e-5, atol=1e-8)
        print_test_result(test_cases[14], True)
    except Exception as e:
        print_test_result(test_cases[14], False, str(e))

    # 16. No Batch Transform - NumPy Single Dimension
    try:
        action = np.array([0.0] * dim)
        transformed = rescaler.transform_action(action, use_torch=False)
        expected = rescaler.center_action
        np.testing.assert_allclose(transformed, expected, rtol=1e-5, atol=1e-8)
        print_test_result(test_cases[15], True)
    except Exception as e:
        print_test_result(test_cases[15], False, str(e))

    # 17. No Batch Transform - Torch Single Dimension
    try:
        action = torch.tensor([0.0] * dim, dtype=torch.float32)
        transformed = rescaler.transform_action(action, use_torch=True)
        expected = torch.tensor(rescaler.center_action, dtype=torch.float32)
        np.testing.assert_allclose(transformed.numpy(), expected.numpy(), rtol=1e-5, atol=1e-8)
        print_test_result(test_cases[16], True)
    except Exception as e:
        print_test_result(test_cases[16], False, str(e))

    print("\nAll tests completed.")