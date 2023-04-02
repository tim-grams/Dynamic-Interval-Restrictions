import os

import numpy as np
import pandas as pd
from ray.tune import Trainable
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from src.envs.obstacle_avoidance import ObstacleAvoidance, generate_obstacles
from src.utils.utils import save_or_extend_dataframe
from src.wrapper.api_translation import compute_metrics, APITranslationWrapper
from src.wrapper.continuous_masking import ContinuousMaskingWrapper
from src.wrapper.discretization import DiscretizationWrapper
from src.wrapper.euclidean_projection import EuclideanProjectionWrapper
from src.wrapper.parametrized_discretization import ParametrizedDiscretizationWrapper
from src.wrapper.random_replacement import RandomReplacementWrapper


def evaluate(env_name: str, env_config, agent: Trainable, name: str, seeds: list, obstacles: list,
             results_path: str = None, episode_results_path: str = None,
             spread: list = None, render: bool = True, record: str = None, alg_num: int = 0):
    """ Runs an agent without exploration in an evaluation environment

    Args:
        env_name (str): Text
        env_config (str): Config of the evaluation environment
        agent (Trainable): Rllib agent
        name (str): Text
        seeds (str): Evaluation environment seeds
        obstacles (list): Numbers of obstacles to put on the map
        results_path (str): Path to store the step results
        episode_results_path (str): Path to store the episode results
        spread (list): Covariance matrix for the obstacle placements
        render (bool): Whether to visualize the evaluation runs
        record (str): Path to store videos of the evaluation runs. Do not save if None
        alg_num (int): Unique identifier for a training run in the evaluation
    """
    if spread is None:
        spread = [[4.0, 0.0], [0.0, 4.0]]

    base_map = env_config['LEVELS'][1]['OBSTACLES']

    for seed in seeds:
        for obstacle_count in obstacles:
            print(f'Evaluation seed {seed} with {obstacle_count} obstacles')

            env_config['LEVELS'][1]['OBSTACLES'] = {
                **base_map,
                **generate_obstacles(env_config['LEVELS'][1]['HEIGHT'],
                                     env_config['LEVELS'][1]['WIDTH'],
                                     obstacle_count,
                                     spread,
                                     mean_size_obstacle=1.0,
                                     sigma_size_obstacle=0.25,
                                     range_size_obstacle=0.5,
                                     seed=seed, uniform=False,
                                     forbidden_circles=[[1.0, 1.0, 0.4], [12.0, 12.0, 0.5]]
                                     )
            }
            avg_obstacle_size = np.mean(
                [max(obstacle['coordinates'][:, 1]) - min(obstacle['coordinates'][:, 1]
                                                          ) for obstacle in
                 env_config['LEVELS'][1]['OBSTACLES'].values()])

            env = ObstacleAvoidance(env_config, render_mode='rgb_array')
            if env_name == 'warehouse-euclidean':
                env = EuclideanProjectionWrapper(env)
            elif env_name == 'warehouse-random':
                env = RandomReplacementWrapper(env)
            elif env_name == 'warehouse-masking':
                env = ContinuousMaskingWrapper(env)
            elif env_name == 'warehouse-discretization':
                env = DiscretizationWrapper(env, 9)
            elif env_name == 'warehouse-p_discretization':
                env = ParametrizedDiscretizationWrapper(env, 9)
            else:
                env = APITranslationWrapper(env)

            metrics = {
                'action': [],
                'reward': [],
                'step': [],
                'fraction_allowed_actions': [],
                'allowed_interval_length': [],
                'number_intervals': [],
                'interval_avg': [],
                'interval_min': [],
                'interval_max': [],
                'interval_variance': []
            }
            cumulative_reward = 0.0

            if not render and record is not None:
                recorder = VideoRecorder(env, os.path.join(record,
                                                           f'{name}_{env_name}_{seed}_{obstacle_count}_{alg_num}.mp4'))

            obs = env.reset()
            current_step = 0
            while True:
                action = agent.compute_single_action(obs, explore=False)
                obs, reward, done, info = env.step(action)
                current_step += 1

                fraction_allowed_actions = info['fraction_allowed_actions']
                allowed_interval_length = info['allowed_interval_length']
                number_intervals = info['number_intervals']
                interval_avg = info['interval_avg']
                interval_min = info['interval_min']
                interval_max = info['interval_max']
                interval_variance = info['interval_variance']

                if render:
                    env.render()
                elif record is not None:
                    recorder.capture_frame()

                if 'atomic_action' in info:
                    metrics['action'].append(info['atomic_action'][0])
                else:
                    metrics['action'].append(action[0])

                metrics['fraction_allowed_actions'].append(fraction_allowed_actions)
                metrics['allowed_interval_length'].append(allowed_interval_length)
                metrics['number_intervals'].append(number_intervals)
                metrics['interval_avg'].append(interval_avg)
                metrics['interval_min'].append(interval_min)
                metrics['interval_max'].append(interval_max)
                metrics['interval_variance'].append(interval_variance)

                metrics['reward'].append(reward)
                metrics['step'].append(current_step)
                cumulative_reward += reward

                if done:
                    break

            env.close()
            if not render and record is not None:
                recorder.close()

            if episode_results_path is not None:
                save_or_extend_dataframe(pd.DataFrame({'seed': [seed],
                                                       'obstacles': [obstacle_count],
                                                       'solved': [info['solved']],
                                                       'steps': [current_step],
                                                       'reward': [cumulative_reward],
                                                       'trajectory': [str(env.trajectory)],
                                                       'fraction_allowed_actions_avg': [
                                                           sum(metrics['fraction_allowed_actions']) / current_step],
                                                       'interval_avg': [
                                                           sum(metrics['interval_avg']) / current_step
                                                       ],
                                                       'interval_min': [
                                                           sum(metrics['interval_min']) / current_step
                                                       ],
                                                       'interval_max': [
                                                           sum(metrics['interval_max']) / current_step
                                                       ],
                                                       'interval_variance': [
                                                           sum(metrics['interval_variance']) / current_step
                                                       ],
                                                       'obstacle_size': [avg_obstacle_size],
                                                       'algorithm': [name],
                                                       'num': [alg_num]}),
                                         episode_results_path)
                print(f' * Episode saved to {episode_results_path}')

            if results_path is not None:
                current_results = pd.DataFrame({**metrics,
                                                'algorithm': name,
                                                'num': alg_num})
                current_results['seed'] = seed
                current_results['obstacles'] = obstacle_count
                save_or_extend_dataframe(current_results, results_path)
                print(f' * STEPS saved to {results_path}')
