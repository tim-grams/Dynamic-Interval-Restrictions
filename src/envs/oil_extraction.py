from typing import Optional

import os
import gym
from gym.error import DependencyNotInstalled
from gym.spaces import Dict, Box, Discrete
import numpy as np
from matplotlib import pyplot as plt
from ray.rllib.utils.spaces.repeated import Repeated


class Pump:

    def __init__(self, location, duration, effectiveness, length):
        self.duration = duration
        self.effectiveness = effectiveness
        self.location = location
        self.length = length

    def step(self):
        self.duration -= 1

        return self.duration != 0


class OilField(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array', 'jupyter'],
        'render_fps': 1,
    }

    def __init__(self, env_config: dict = None, render_mode: Optional[str] = None):
        assert 'LENGTH' in env_config
        assert 'LENGTH_PUMP' in env_config
        assert 'LENGTH_PUMP_STD' in env_config
        assert 'LENGTH_PUMP_MAX' in env_config
        assert 'LENGTH_PUMP_MIN' in env_config
        assert 'STEPS_PER_EPISODE' in env_config
        assert 'EFFECTIVENESS_MEAN' in env_config
        assert 'EFFECTIVENESS_STD' in env_config
        assert 'EFFECTIVENESS_MAX' in env_config
        assert 'EFFECTIVENESS_MIN' in env_config
        assert 'DURATION_MEAN' in env_config
        assert 'DURATION_MIN' in env_config
        assert 'DURATION_MAX' in env_config
        assert 'GAUSSIAN_NOISE' in env_config
        assert 'valid_action_reward' in env_config
        assert 'invalid_action_penalty' in env_config
        assert 'END_ON_COLLISION' in env_config

        # Initialization
        self.LENGTH = env_config['LENGTH']
        self.LENGTH_PUMP = env_config['LENGTH_PUMP']
        self.LENGTH_PUMP_STD = env_config['LENGTH_PUMP_STD']
        self.LENGTH_PUMP_MAX = env_config['LENGTH_PUMP_MAX']
        self.LENGTH_PUMP_MIN = env_config['LENGTH_PUMP_MIN']
        self.STEPS_PER_EPISODE = env_config['STEPS_PER_EPISODE']
        self.EFFECTIVENESS_MEAN = env_config['EFFECTIVENESS_MEAN']
        self.EFFECTIVENESS_STD = env_config['EFFECTIVENESS_STD']
        self.EFFECTIVENESS_MAX = env_config['EFFECTIVENESS_MAX']
        self.EFFECTIVENESS_MIN = env_config['EFFECTIVENESS_MIN']
        self.DURATION_MIN = env_config['DURATION_MIN']
        self.DURATION_MEAN = env_config['DURATION_MEAN']
        self.DURATION_MAX = env_config['DURATION_MAX']
        self.GAUSSIAN_NOISE = env_config['GAUSSIAN_NOISE']
        self.reward_function = env_config['valid_action_reward']
        self.penalty_function = env_config['invalid_action_penalty']
        self.END_ON_COLLISION = env_config['END_ON_COLLISION']

        self.state = []
        self.last_reward = 0.0
        self.next_state = []
        self.next_effectiveness = 0.0
        self.next_duration = 0.0
        self.next_length = 0.0
        self.allowed_actions = []
        self.current_step = 0

        self.window_scale = 25
        self.window = None
        self.clock = None
        self.render_mode = render_mode
        self.reward_function_coordinates = None
        self.reward_range = 0.0

        if 'RANDOM_SEED' in env_config:
            self.seed(env_config['RANDOM_SEED'])

        # Observation and Action Space
        self.observation_space = Dict({
            'observation': Dict({'effectiveness': Box(low=self.EFFECTIVENESS_MIN,
                                                      high=self.EFFECTIVENESS_MAX, shape=(1,),
                                                      dtype=np.float32),
                                 'duration': Discrete(self.DURATION_MAX + 1),
                                 'length': Box(low=self.LENGTH_PUMP_MIN,
                                               high=self.LENGTH_PUMP_MAX, shape=(1,),
                                               dtype=np.float32)}),
            'allowed_actions': Repeated(Box(low=0.0, high=self.LENGTH,
                                            shape=(2,), dtype=np.float32), max_len=self.DURATION_MAX + 1)})
        self.action_space = Box(low=0.0, high=self.LENGTH, shape=(1,))

    def place_pump(self, action):
        collision_depths = np.clip(np.concatenate((
            np.array([(placement.length / 2 + self.next_length / 2
                       ) - np.abs(action[0] - placement.location[0]
                                  ) for placement in self.next_state]),
            np.array([self.next_length / 2 - action[0], action[0] - self.LENGTH + self.next_length / 2])
        )), 0.0, self.LENGTH)
        if np.any(collision_depths > 0.0):
            return np.max(collision_depths)
        self.next_state.append(Pump(action, self.next_duration, self.next_effectiveness, self.next_length))
        return 0.0

    def get_reward(self, collision_depth):
        if collision_depth > 0.0:
            reward = self.penalty_function(collision_depth)
        else:
            reward = np.sum(
                [self.reward_function(pump.location) * pump.effectiveness for pump in self.next_state]
            ) + np.random.normal(0.0, self.GAUSSIAN_NOISE)

        return reward

    def draw_pump_specification(self):
        self.next_effectiveness = np.max([np.min([np.random.normal(self.EFFECTIVENESS_MEAN, self.EFFECTIVENESS_STD),
                                                  self.EFFECTIVENESS_MAX]), self.EFFECTIVENESS_MIN])
        self.next_duration = np.max([np.min([np.random.poisson(self.DURATION_MEAN), self.DURATION_MAX]),
                                     self.DURATION_MIN])
        self.next_length = np.max([np.min([np.random.normal(self.LENGTH_PUMP, self.LENGTH_PUMP_STD),
                                           self.LENGTH_PUMP_MAX]), self.LENGTH_PUMP_MIN])

    def get_allowed_actions(self):
        valid_space = [[self.next_length / 2, self.LENGTH - self.next_length / 2]]

        for pump in self.next_state:
            lower_limit = pump.location[0] - pump.length / 2 - self.next_length / 2
            upper_limit = pump.location[0] + pump.length / 2 + self.next_length / 2
            for key, subset in enumerate(valid_space):
                if lower_limit > subset[0] and upper_limit < subset[1]:
                    valid_space.append([subset[0], lower_limit])
                    valid_space.append([upper_limit, subset[1]])
                    valid_space.pop(key)
                elif lower_limit <= subset[0] and upper_limit >= subset[1]:
                    valid_space.pop(key)
                else:
                    if subset[0] < upper_limit < subset[1]:
                        subset[0] = upper_limit
                    if subset[0] < lower_limit < subset[1]:
                        subset[1] = lower_limit

        return np.array(valid_space, dtype=np.float32)

    def step(self, action):
        collision_depth = self.place_pump(action)
        self.last_reward = self.get_reward(collision_depth)
        self.state = self.next_state.copy()

        self.draw_pump_specification()
        self.next_state = [pump for pump in self.next_state if pump.step()]
        self.allowed_actions = self.get_allowed_actions()

        self.current_step += 1
        observation = {'effectiveness': np.array([self.next_effectiveness], dtype=np.float32),
                       'duration': self.next_duration,
                       'length': np.array([self.next_length], dtype=np.float32)}
        done = (collision_depth > 0.0) if self.END_ON_COLLISION else False
        truncated = self.current_step >= self.STEPS_PER_EPISODE
        return {'observation': observation,
                'allowed_actions': self.allowed_actions
                }, self.last_reward, done or truncated, {}

    def reset(self):
        self.next_state = []
        self.state = []
        self.current_step = 0
        self.draw_pump_specification()
        self.allowed_actions = self.get_allowed_actions()

        return {'observation': {'effectiveness': np.array([self.next_effectiveness], dtype=np.float32),
                                'duration': self.next_duration,
                                'length': np.array([self.next_length], dtype=np.float32)},
                'allowed_actions': self.allowed_actions}

    def seed(self, seed: int = None):
        np.random.seed(seed)

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled('Pygame is not installed, run `pip install pygame`')

        if self.reward_function_coordinates is None:
            self.reward_function_coordinates = np.array(list(zip(list(np.linspace(0.0, self.LENGTH,
                                                                                  600) * self.window_scale),
                                                                 list(map(
                                                                     lambda x: self.reward_function(
                                                                         x) * self.window_scale,
                                                                     np.linspace(0.0, self.LENGTH, 600)))
                                                                 )))
            self.reward_range = np.max(self.reward_function_coordinates[:, 1]
                                       ) - np.min(self.reward_function_coordinates[:, 1])
            self.reward_function_coordinates[:, 1] += self.reward_range

        window_width = self.LENGTH * self.window_scale
        window_height = self.reward_range + 7.0 * self.window_scale
        if self.window is None:
            pygame.init()
            pygame.font.init()
            if self.render_mode == 'human':
                pygame.display.init()
                self.window = pygame.display.set_mode((window_width, window_height))
                pygame.display.set_caption('Oil Field')
            else:
                self.window = pygame.Surface((window_width, window_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((window_width, window_height))
        canvas.fill((232, 232, 232))

        pygame.draw.aalines(canvas, (0, 0, 0), False, self.reward_function_coordinates)

        canvas = pygame.transform.flip(canvas, False, True)
        self.window.blit(canvas, (0, 0))

        pumps = [pygame.image.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'pump.png')
                                   ) for pump in self.state]
        for index, pump in enumerate(pumps):
            pump_length = self.state[index].length * self.window_scale
            pump_location_width = self.state[index].location[0] * self.window_scale - pump_length / 2
            pump_location_height = window_height - (self.reward_function(
                self.state[index].location[0]) * self.window_scale + self.reward_range + pump_length)

            self.window.blit(pygame.transform.scale(pump, (pump_length, pump_length)), (pump_location_width,
                                                                                        pump_location_height))

        font = pygame.font.SysFont('Arial', 15)
        text_canvas = font.render(
            f'Reward: {np.round(self.last_reward, 2)}',
            True, (0, 0, 0))
        allowed_actions_canvas = font.render(
            f'Allowed Actions: {[(np.round(subset[0], 2), np.round(subset[1], 2)) for subset in self.allowed_actions]}',
            True, (0, 0, 0))

        self.window.blit(text_canvas, (self.window_scale / 4, self.window_scale / 4))
        self.window.blit(allowed_actions_canvas, (self.window_scale / 4, 4 * self.window_scale / 4))

        if self.render_mode == 'human':
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()
        elif self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )
        elif self.render_mode == 'jupyter':
            plt.imshow(np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            ))
            plt.axis('off')
            plt.show()

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
