import gym
from gym.spaces import Box, Dict
import numpy as np


def compute_metrics(allowed_actions, action_space_range):
    interval_lengths = [np.ptp(interval) for interval in allowed_actions]

    total_interval_length = np.sum(interval_lengths)
    fraction_allowed_actions = np.divide(total_interval_length, action_space_range)
    maximum_interval_length = np.max(interval_lengths) if len(interval_lengths) > 0 else 0.0
    minimum_interval_length = np.min(interval_lengths) if len(interval_lengths) > 0 else 0.0
    average_interval_length = total_interval_length / len(allowed_actions) if len(allowed_actions) > 0 else 0.0
    interval_variance = np.var(interval_lengths) if len(interval_lengths) > 0 else 0.0
    return fraction_allowed_actions, total_interval_length, len(allowed_actions), average_interval_length, minimum_interval_length, maximum_interval_length, interval_variance


class APITranslationWrapper(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert isinstance(env.observation_space, Dict)
        assert isinstance(env.action_space, Box)

        self.action_space_width = env.action_space.high[0] - env.action_space.low[0]
        self.allowed_actions = [[env.action_space.low, env.action_space.high]]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['invalid'] = not np.any(
            [interval[0] <= action <= interval[1] for interval in self.allowed_actions])
        info['fraction_allowed_actions'], info['allowed_interval_length'], info['number_intervals'], info[
            'interval_avg'], info['interval_min'], info['interval_max'], info['interval_variance'] = compute_metrics(
            self.allowed_actions, self.action_space_width)

        self.allowed_actions = obs['allowed_actions']

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        assert isinstance(obs, dict)
        assert 'observation' in obs
        assert 'allowed_actions' in obs

        self.allowed_actions = obs['allowed_actions']

        return obs
