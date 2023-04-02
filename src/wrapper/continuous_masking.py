from gym.spaces import Box
import numpy as np

from src.wrapper.api_translation import APITranslationWrapper, compute_metrics


class ContinuousMaskingWrapper(APITranslationWrapper):

    def __init__(self, env):
        APITranslationWrapper.__init__(self, env)
        assert isinstance(env.action_space, Box)

        self.action_space_min = env.action_space.low[0]
        self.action_space_max = env.action_space.high[0]
        self.action_space_range = env.action_space.high[0] - self.action_space_min

    def project(self, action):
        new_action = (action - np.min(self.action_space_min)) * (
            np.sum([np.ptp(interval) for interval in self.allowed_actions]) / self.action_space_range)
        if action >= self.action_space_max:
            return [np.max(self.allowed_actions)]
        for interval in self.allowed_actions:
            if (new_action - np.ptp(interval)) <= 0:
                return new_action + np.min(interval)
            new_action -= np.ptp(interval)

    def step(self, action):
        if len(self.allowed_actions) > 0:
            action = self.project(action)

        obs, reward, done, info = self.env.step(action)

        info['fraction_allowed_actions'], info['allowed_interval_length'], info['number_intervals'], info[
            'interval_avg'], info['interval_min'], info['interval_max'], info['interval_variance'] = compute_metrics(
            self.allowed_actions, self.action_space_width)

        self.allowed_actions = obs['allowed_actions']
        info['Executed'] = action
        info['invalid'] = False

        return obs, reward, done, info
