import gym
import numpy as np
from gym.spaces import Dict, Box

from src.wrapper.api_translation import compute_metrics


class ParametrizedDiscretizationWrapper(gym.Wrapper):

    def __init__(self, env, number_of_bins: int):
        gym.Wrapper.__init__(self, env)
        assert isinstance(env.observation_space, Dict)
        assert isinstance(env.action_space, Box)

        self.number_of_bins = number_of_bins
        self.action_space_low = env.action_space.low[0]
        self.action_space_range = env.action_space.high - env.action_space.low
        self.bin_width = self.action_space_range / self.number_of_bins

        self.allowed_actions = np.ones(self.number_of_bins)
        self.continuous_allowed_actions = [[env.action_space.low, env.action_space.high]]

        # New observation space
        self.observation_space = Dict({'observation': env.observation_space['observation'],
                                       'allowed_actions': Box(low=0.0, high=1.0, shape=(self.number_of_bins,))})

    def update_allowed_actions(self, allowed_actions):
        allowed_actions = np.subtract(allowed_actions, self.action_space_low)
        discrete_allowed_actions = np.zeros(self.number_of_bins)
        for allowed_subset in allowed_actions:
            lower_bound = int(allowed_subset[0] / self.bin_width) + 1
            upper_bound = int(allowed_subset[1] / self.bin_width) - 1
            discrete_allowed_actions[range(lower_bound, upper_bound + 1)] = 1.0
            if allowed_subset[0] == 0.0:
                discrete_allowed_actions[0] = 1.0
        return discrete_allowed_actions

    def step(self, action):
        invalid = not np.any(
            [index * action_bin * self.bin_width <= np.subtract(action, self.action_space_low) <= action_bin * (
                        (index * self.bin_width) + self.bin_width
                        ) for index, action_bin in enumerate(
                self.allowed_actions)])
        obs, reward, done, info = self.env.step(action)

        info['fraction_allowed_actions'], info['allowed_interval_length'], info['number_intervals'], info[
            'interval_avg'], info['interval_min'], info['interval_max'], info['interval_variance'] = compute_metrics(
            self.continuous_allowed_actions, self.action_space_range)

        self.continuous_allowed_actions = obs['allowed_actions']
        self.allowed_actions = self.update_allowed_actions(obs['allowed_actions'])
        info['invalid'] = invalid

        return {'observation': obs['observation'],
                'allowed_actions': self.allowed_actions}, reward, done, info

    def reset(self):
        obs = self.env.reset()
        assert isinstance(obs, dict)
        assert 'observation' in obs
        assert 'allowed_actions' in obs

        self.continuous_allowed_actions = obs['allowed_actions']
        self.allowed_actions = self.update_allowed_actions(obs['allowed_actions'])

        return {'observation': obs['observation'],
                'allowed_actions': self.allowed_actions}
