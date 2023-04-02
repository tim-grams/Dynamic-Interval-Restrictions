import numpy as np

from src.wrapper.api_translation import APITranslationWrapper, compute_metrics


class RandomReplacementWrapper(APITranslationWrapper):

    def __init__(self, env):
        APITranslationWrapper.__init__(self, env)

    def is_allowed(self, action):
        for subset in self.allowed_actions:
            if subset[0] <= action <= subset[1]:
                return True
        return False

    def sample_allowed_action(self):
        length_of_allowed_space = np.sum(np.ptp(self.allowed_actions, axis=1))
        random_number = np.random.random() * length_of_allowed_space
        for subset in self.allowed_actions:
            subset_range = np.ptp(subset)
            if random_number > subset_range:
                random_number -= subset_range
            else:
                return np.array([subset[0] + random_number], dtype=np.float32)

    def step(self, action):
        invalid = False
        if not len(self.allowed_actions) == 0 and not self.is_allowed(action):
            invalid = True
            action = self.sample_allowed_action()

        obs, reward, done, info = self.env.step(action)

        info['fraction_allowed_actions'], info['allowed_interval_length'], info['number_intervals'], info[
            'interval_avg'], info['interval_min'], info['interval_max'], info['interval_variance'] = compute_metrics(
            self.allowed_actions, self.action_space_width)

        self.allowed_actions = obs['allowed_actions']
        info['Executed'] = action
        info['invalid'] = invalid

        return obs, reward, done, info
