import numpy as np

from src.wrapper.api_translation import APITranslationWrapper, compute_metrics


class EuclideanProjectionWrapper(APITranslationWrapper):

    def __init__(self, env):
        APITranslationWrapper.__init__(self, env)

    def is_allowed(self, action):
        for subset in self.allowed_actions:
            if subset[0] <= action <= subset[1]:
                return True
        return False

    def get_closest_allowed_action(self, action):
        allowed_actions = np.array(self.allowed_actions, dtype=np.float32).reshape(1, -1)[0]
        return np.array([allowed_actions[np.argmin(np.abs(allowed_actions - action))]], dtype=np.float32)

    def step(self, action):
        invalid = False
        if not len(self.allowed_actions) == 0 and not self.is_allowed(action):
            invalid = True
            action = self.get_closest_allowed_action(action)

        obs, reward, done, info = self.env.step(action)

        info['fraction_allowed_actions'], info['allowed_interval_length'], info['number_intervals'], info[
            'interval_avg'], info['interval_min'], info['interval_max'], info['interval_variance'] = compute_metrics(
            self.allowed_actions, self.action_space_width)

        self.allowed_actions = obs['allowed_actions']

        info['Executed'] = action
        info['invalid'] = invalid

        return obs, reward, done, info
