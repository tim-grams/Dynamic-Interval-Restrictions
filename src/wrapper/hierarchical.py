import gym
import numpy as np
from gym.spaces import Box, Dict
from ray.rllib import MultiAgentEnv
from ray.rllib.env.apis.task_settable_env import TaskType

from configs.base_config import MAX_PADDING_LEN
from src.wrapper.api_translation import compute_metrics


def pad_allowed_actions(obs: dict) -> dict:
    obs['allowed_actions'] = np.concatenate([
        obs['allowed_actions'],
        np.zeros((MAX_PADDING_LEN - len(obs['allowed_actions']), 2))
    ]) if len(obs['allowed_actions']) > 0 else np.ones((MAX_PADDING_LEN, 2))
    return obs


class HierarchicalWrapper(MultiAgentEnv):

    def __init__(self, env: gym.Env):
        assert isinstance(env.observation_space, Dict)
        assert isinstance(env.action_space, Box)
        super().__init__()

        self._skip_env_checking = True
        self._agent_ids = ['interval_level_agent', 'action_level_agent']
        self.flat_env = env

        self.action_space_min = env.action_space.low[0]
        self.action_space_range = env.action_space.high - env.action_space.low
        self.last_interval = None
        self.last_reward = 0.0
        self.observation = None
        self.full_space = [env.action_space.low[0], env.action_space.high[0]]
        self.allowed_actions = [self.full_space]

    def scale(self, action, interval):
        return (action - self.action_space_min) * (interval[1] - interval[0]) / self.action_space_range + interval[0]

    def action_level_policy(self, action):
        action = self.scale(action[0], self.last_interval)

        obs, reward, done, truncated, info = self.flat_env.step(action)
        info['invalid'] = not np.any(
            [interval[0] <= action <= interval[1] for interval in self.allowed_actions])
        info['fraction_allowed_actions'], info['allowed_interval_length'], info['number_intervals'] = compute_metrics(
            self.allowed_actions, self.action_space_range)

        self.observation = obs['observation']
        self.allowed_actions = obs['allowed_actions']
        self.last_reward = reward

        obs = {'interval_level_agent': {'observation': pad_allowed_actions(obs),
                                        'allowed_actions': np.concatenate(
                                            [np.ones(len(self.allowed_actions)), np.zeros(
                                                MAX_PADDING_LEN - len(self.allowed_actions))], dtype=np.float32)}}
        reward = {'interval_level_agent': self.last_reward}
        done = {'__all__': done or truncated}
        info = {'interval_level_agent': info}

        if done['__all__']:
            obs['action_level_agent'] = {'observation': self.observation,
                                         'interval': self.allowed_actions[0] if len(
                                             self.allowed_actions) > 0 else self.full_space}
            reward['action_level_agent'] = self.last_reward

        return obs, reward, done, info

    def interval_level_policy(self, action):
        self.last_interval = self.allowed_actions[action] if len(self.allowed_actions) > 0 else self.full_space

        obs = {'action_level_agent': {'observation': self.observation,
                                      'interval': self.last_interval}}
        reward = {'action_level_agent': self.last_reward}
        done = {'__all__': False}
        info = {'action_level_agent': {}}

        return obs, reward, done, info

    def step(self, action_dict):
        if "interval_level_agent" in action_dict:
            value = self.interval_level_policy(action_dict["interval_level_agent"])
        else:
            value = self.action_level_policy(list(action_dict.values())[0])
        return value

    def reset(self):
        obs = self.flat_env.reset()
        assert isinstance(obs, dict)
        assert 'observation' in obs
        assert 'allowed_actions' in obs

        self.observation = obs['observation']
        self.allowed_actions = obs['allowed_actions']

        return {'interval_level_agent': {'observation': pad_allowed_actions(obs),
                                         'allowed_actions': np.concatenate([np.ones(len(self.allowed_actions)),
                                                                            np.zeros(8 - len(self.allowed_actions))],
                                                                           dtype=np.float32)}}

    def set_task(self, task: TaskType) -> None:
        assert hasattr(self.flat_env, 'set_task')

        self.flat_env.set_task(task)

    def get_task(self) -> TaskType:
        assert hasattr(self.flat_env, 'get_task')

        return self.flat_env.get_task()
