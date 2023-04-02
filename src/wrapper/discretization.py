import gym
import numpy as np
from gym.spaces import MultiBinary, Discrete, Dict, Box

from src.wrapper.api_translation import compute_metrics


class DiscretizationWrapper(gym.Wrapper):

    def __init__(self, env, number_of_actions: int):
        gym.Wrapper.__init__(self, env)
        assert number_of_actions > 2, 'At least two distinct actions are required'
        assert isinstance(env.observation_space, Dict)
        assert isinstance(env.action_space, Box)

        self.number_of_actions = number_of_actions
        self.action_space_range = env.action_space.high[0] - env.action_space.low[0]
        self.discrete_actions = ((env.action_space.high - env.action_space.low) * np.arange(self.number_of_actions)
                                 ) / (self.number_of_actions - 1) + env.action_space.low
        self.continuous_allowed_actions = [[env.action_space.low, env.action_space.high]]
        self.allowed_actions = np.ones(self.number_of_actions)

        # New observation and action space
        self.observation_space = Dict({'observation': env.observation_space['observation'],
                                       'allowed_actions': Box(low=0.0, high=1.0, shape=(self.number_of_actions,))})
        self.action_space = Discrete(self.number_of_actions)

    def update_allowed_actions(self, allowed_actions):
        discrete_allowed_actions = np.zeros(self.number_of_actions)
        for allowed_subset in allowed_actions:
            lower_bound = np.argmax(self.discrete_actions[(allowed_subset[0] - self.discrete_actions) > 0.0]
                                    ) + 1 if allowed_subset[0] > self.discrete_actions[0] else 0
            upper_bound = np.argmax(self.discrete_actions[(allowed_subset[1] - self.discrete_actions) >= 0.0])
            discrete_allowed_actions[range(lower_bound, upper_bound + 1)] = 1.0
        return discrete_allowed_actions

    def step(self, action):
        assert action < self.number_of_actions

        invalid = self.allowed_actions[action] == 0.0

        action = [self.discrete_actions[action]]
        obs, reward, done, info = self.env.step(action)

        info['fraction_allowed_actions'], info['allowed_interval_length'], info['number_intervals'], info[
            'interval_avg'], info['interval_min'], info['interval_max'], info['interval_variance'] = compute_metrics(
            self.continuous_allowed_actions, self.action_space_range)

        self.continuous_allowed_actions = obs['allowed_actions']
        self.allowed_actions = self.update_allowed_actions(obs['allowed_actions'])
        info['invalid'] = invalid
        info['atomic_action'] = action

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
