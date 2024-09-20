import torch

import gymnasium as gym

class ClipAction(gym.ActionWrapper):
    """ Adapted from https://github.com/openai/gym/blob/master/gym/wrappers/clip_action.py
    Clip the continuous action within the valid :class:`Box` observation space bound.

    Example:
        >>> import gym
        >>> env = gym.make('Bipedal-Walker-v3')
        >>> env = ClipAction(env)
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env.step(np.array([5.0, 2.0, -10.0, 0.0]))
        # Executes the action np.array([1.0, 1.0, -1.0, 0]) in the base environment
    """

    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        # assert isinstance(env.action_space, Box)
        super().__init__(env)

    def action(self, action):
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return torch.clip(action, min=self.action_space.low, max=self.action_space.high)