import random

class RandomAgent:
    """
    A baseline agent that randomly selects an action (edge) to traverse.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def predict(self, obs, deterministic=False):
        # Just pick a random action from the action space
        action = self.action_space.sample()
        return action, None