import gym
import numpy as np
from gym.spaces import Box, Dict
from ray.rllib.utils.spaces.repeated import Repeated


class Car:

    def __init__(self, pos, vel, car_behind, car_front, config: dict = None):
        assert 'KD' in config
        assert 'KP' in config
        assert 'KB' in config
        assert 'DT' in config

        self.KD = config['KD']
        self.KP = config['KP']
        self.KB = config['KB']
        self.DT = config['DT']

        self.pos = pos
        self.vel = vel
        self.accel = 0
        self.car_behind = car_behind
        self.car_front = car_front

    def step(self, action, act_noise):
        a = 1
        action = np.random.normal(action, act_noise)
        accel = a * action - self.KD * self.vel
        self.vel = self.vel + accel * self.DT
        self.pos = 0.5 * accel * self.DT ** 2 + self.vel * self.DT + self.pos
        self.accel = accel

    def get_state(self):
        return [self.pos, self.vel, self.accel]

    def update_order(self, car_behind, car_front):
        self.car_front = car_front
        self.car_behind = car_behind

    def get_next_action(self, vel_des):
        action = self.KP * (vel_des - self.vel)
        if self.car_front:
            if self.car_front.pos - self.pos < 6.0:
                action = action - self.KB * (self.car_front.pos - self.pos)

        return action

    def get_next_action_last_car(self, vel_des):
        action = self.KP * (vel_des - self.vel)
        if self.car_front:
            if self.car_front.car_front.pos - self.pos < 12.0:
                action = action - self.KB * (self.car_front.car_front.pos - self.pos) * 0.5

        return action

    def get_action_for_position(self, desired_position):
        return (1.5 * self.DT ** 2 * self.KD * self.vel - self.DT * self.vel - self.pos + desired_position) / (
                1.5 * self.DT ** 2)


class FuelSaving(gym.Env):

    def __init__(self, env_config: dict = None):
        assert 'STEPS_PER_EPISODE' in env_config
        assert 'AGENT_PROFILE' in env_config
        assert 'OTHERS_PROFILE' in env_config
        assert 'DT' in env_config
        assert 'KD' in env_config['AGENT_PROFILE'] and 'KD' in env_config['OTHERS_PROFILE']
        assert 'KP' in env_config['AGENT_PROFILE'] and 'KP' in env_config['OTHERS_PROFILE']
        assert 'KB' in env_config['AGENT_PROFILE'] and 'KB' in env_config['OTHERS_PROFILE']
        assert 'MINIMUM_DISTANCE' in env_config
        assert 'PENALTY_COEFFICIENT' in env_config
        assert 'COLLISION_REWARD' in env_config

        self.STEPS_PER_EPISODE = env_config['STEPS_PER_EPISODE']
        self.DT = env_config['DT']
        self.agent_config = {'KP': env_config['AGENT_PROFILE']['KP'],
                             'KB': env_config['AGENT_PROFILE']['KB'],
                             'KD': env_config['AGENT_PROFILE']['KD'],
                             'DT': self.DT}
        self.others_config = {'KP': env_config['OTHERS_PROFILE']['KP'],
                              'KB': env_config['OTHERS_PROFILE']['KB'],
                              'KD': env_config['OTHERS_PROFILE']['KD'],
                              'DT': self.DT}
        self.MINIMUM_DISTANCE = env_config['MINIMUM_DISTANCE']
        self.PENALTY_COEFFICIENT = env_config['PENALTY_COEFFICIENT']
        self.COLLISION_REWARD = env_config['COLLISION_REWARD']
        self.NUMBER_OF_CARS = 5
        self.current_step = 0
        self.time = 0
        self.distances = [0.0, 0.0]

        if 'RANDOM_SEED' in env_config:
            self.seed(env_config['RANDOM_SEED'])

        self.car_1 = Car(34.0, 30.0, None, None, self.others_config)
        self.car_2 = Car(28.0, 30.0, None, self.car_1, self.others_config)
        self.car_3 = Car(22.0, 30.0, None, self.car_2, self.others_config)
        self.car_4 = Car(16.0, 30.0, None, self.car_3, self.agent_config)
        self.car_5 = Car(10.0, 30.0, None, self.car_4, self.others_config)
        self.car_1.update_order(self.car_2, None)
        self.car_2.update_order(self.car_3, self.car_1)
        self.car_3.update_order(self.car_4, self.car_2)
        self.car_4.update_order(self.car_5, self.car_3)
        self.car_5.update_order(None, self.car_4)

        self.all_states = np.zeros((5, 3))
        self.all_states = self.get_all_states()

        # Observation and Action Space
        self.observation_space = Dict({'observation': Box(low=0.0, high=100000.0, shape=(15,), dtype=np.float32),
                                       'allowed_actions': Repeated(Box(low=-1000.0, high=1000.0, shape=(2,)),
                                                                   max_len=1)})
        self.action_space = Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32)

    def get_all_states(self):
        self.all_states[0, :] = self.car_1.get_state()
        self.all_states[1, :] = self.car_2.get_state()
        self.all_states[2, :] = self.car_3.get_state()
        self.all_states[3, :] = self.car_4.get_state()
        self.all_states[4, :] = self.car_5.get_state()
        return self.all_states

    def get_reward(self, action):
        state = self.car_4.get_state()
        if action > 0:
            reward = -np.abs(state[1]) * action
        else:
            reward = 0
        if (self.car_4.car_front.pos - self.car_4.pos) < 2.99:
            reward = reward - np.abs((self.PENALTY_COEFFICIENT * 500) / (self.car_4.car_front.pos - self.car_4.pos))

        if (self.car_4.pos - self.car_4.car_behind.pos) < 2.99:
            reward = reward - np.abs((self.PENALTY_COEFFICIENT * 500) / (self.car_4.pos - self.car_4.car_behind.pos))

        return reward

    def get_allowed_actions(self):
        self.car_5.step(self.car_5.get_next_action_last_car(30), 0.0)
        self.car_3.step(self.car_3.get_next_action(30), 0.0)
        self.car_2.step(self.car_2.get_next_action(30), 0.0)
        self.car_1.step(self.car_1.get_next_action(30 - 10 * np.sin(0.1 * self.time)), 0.0)

        allowed_actions_3_meter = np.clip([self.car_4.get_action_for_position(self.car_5.pos + self.MINIMUM_DISTANCE),
                                           self.car_4.get_action_for_position(self.car_3.pos - self.MINIMUM_DISTANCE)],
                                          -100, 100)

        if allowed_actions_3_meter[0] == allowed_actions_3_meter[1] == -100:
            allowed_actions_3_meter[0] = -100
            allowed_actions_3_meter[1] = -99
        elif allowed_actions_3_meter[0] == allowed_actions_3_meter[1] == 100:
            allowed_actions_3_meter[0] = 99
            allowed_actions_3_meter[1] = 100

        return np.array([allowed_actions_3_meter], dtype=np.float32)

    def step(self, action):
        action = action[0]
        self.car_4.step(action, 0)

        states = self.all_states.reshape(1, -1)[0]

        self.distances = [self.car_4.car_front.pos - self.car_4.pos, self.car_4.pos - self.car_4.car_behind.pos]
        if self.distances[0] <= 0.0 or self.distances[1] <= 0.0:
            return {'observation': np.array(states, dtype=np.float32),
                    'allowed_actions': self.get_allowed_actions()}, self.COLLISION_REWARD, True, False, {}

        reward = self.get_reward(action)

        self.time += self.DT
        self.current_step += 1

        if self.current_step == self.STEPS_PER_EPISODE:
            self.current_step = 0
            return {'observation': np.array(states, dtype=np.float32),
                    'allowed_actions': self.get_allowed_actions()}, reward, True, {}
        else:
            return {'observation': np.array(states, dtype=np.float32),
                    'allowed_actions': self.get_allowed_actions()}, reward, False, {}

    def reset(self):
        self.time = 0
        self.car_1 = Car(34.0, 30.0, None, None, self.others_config)
        self.car_2 = Car(28.0, 30.0, None, self.car_1, self.others_config)
        self.car_3 = Car(22.0, 30.0, None, self.car_2, self.others_config)
        self.car_4 = Car(16.0, 30.0, None, self.car_3, self.agent_config)
        self.car_5 = Car(10.0, 30.0, None, self.car_4, self.others_config)
        self.car_1.update_order(self.car_2, None)
        self.car_2.update_order(self.car_3, self.car_1)
        self.car_3.update_order(self.car_4, self.car_2)
        self.car_4.update_order(self.car_5, self.car_3)
        self.car_5.update_order(None, self.car_4)

        self.distances = [self.car_4.car_front.pos - self.car_4.pos, self.car_4.pos - self.car_4.car_behind.pos]

        self.all_states = np.zeros((5, 3))
        self.all_states = self.get_all_states()
        states = self.all_states.reshape(1, -1)[0]

        return {'observation': np.array(states, dtype=np.float32),
                'allowed_actions': self.get_allowed_actions()}

    def seed(self, seed: int = None):
        np.random.seed(seed)

    def render(self):
        print(f'Distances: {self.distances}')
