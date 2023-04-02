import numpy as np
import ray
from ray import tune
from ray.rllib.env import EnvContext
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
import os
import dill as pickle
import pandas as pd
from gym.spaces import Box, Dict, Discrete, MultiBinary
from datetime import datetime
import random

from ray.rllib.policy.policy import PolicySpec
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import Repeater, BasicVariantGenerator
from ray.tune.search.hyperopt import HyperOptSearch
from ray.rllib.utils import try_import_torch

from configs.envs.oil_field_config import oil_field_config
from configs.envs.fuel_saving_config import fuel_saving_config
from configs.base_config import training_config, MAX_PADDING_LEN, evaluation_config
from configs.models.test.hierarchical_config import hierarchical_config
from src.algorithms.models.discrete_mask_model import DiscreteMaskModel
from src.algorithms.models.hierarchical_models import IntervalLevelModel
from src.envs.fuel_saving import FuelSaving
from src.envs.oil_extraction import OilField
from src.envs.obstacle_avoidance import ObstacleAvoidance
from src.evaluation.evaluation import evaluate
from src.utils.utils import load_agent, make_dir_if_not_exists, load_pickle, register_environments, get_algorithm
from src.envs.custom_metrics_callback import CustomMetricsCallback

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

register_environments(training_config['NUM_DISCRETIZATION_BINS'])

torch, nn = try_import_torch()


def curriculum_fn(train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext) -> TaskType:
    current_task = task_settable_env.get_task()
    if train_results['custom_metrics']['solved_mean'] > 0.7:
        print('Raising difficulty!')
        current_task = current_task + 1
    return current_task


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id.startswith('action_level_'):
        return 'action_level_agent'
    else:
        return 'interval_level_agent'


def train_agent(algorithm,
                environment,
                default_hyperparameters: dict,
                search_space: dict = None):
    """ Train an agent

    Args:
        algorithm (str): Name of the algorithm
        environment (str): Name of the environment
        default_hyperparameters (dict): Default hyperparameter
        search_space (dict): Search space for the algorithm
    """
    assert algorithm in [
        'PPO', 'PPO-MASKED', 'DDPG', 'TD3', 'DQN', 'MPS-TD3', 'PAM', 'DQN-MASKED'
    ], 'This algorithm is not supported.'

    if environment.startswith('fuel-saving'):
        config = fuel_saving_config
        env = FuelSaving(config)
    elif environment.startswith('oilfield'):
        config = oil_field_config
        env = OilField(config)
    else:
        config = training_config['ENV_CONFIG']
        env = ObstacleAvoidance(config)

    make_dir_if_not_exists(training_config['TRAINING_RESULTS_PATH'])
    results_full_path = os.path.join(training_config['TRAINING_RESULTS_PATH'], f'{training_config["NAME"]}.pkl')
    if os.path.exists(results_full_path):
        results = [pd.read_pickle(results_full_path)]
    else:
        results = []

    if os.path.exists(training_config['EXPERIMENTS_PATH']):
        experiments = pd.read_pickle(training_config['EXPERIMENTS_PATH'])
    else:
        experiments = pd.DataFrame(columns=['algorithm_name', 'config', 'model'])

    run_config = {
        'env': environment,
        'env_config': config,
        'env_task_fn': curriculum_fn if environment.startswith('obstacle_avoidance') else None,
        'framework': 'torch',
        'num_gpus': training_config['NUM_GPUS'],
        'num_workers': training_config['NUM_WORKERS'],
        'callbacks': CustomMetricsCallback,
        **default_hyperparameters
    }

    if algorithm == 'PPO-MASKED':
        run_config['model'] = {
            'custom_model': DiscreteMaskModel
        }

    if environment.endswith('hierarchical'):
        assert isinstance(env.action_space, Box)

        run_config['multiagent'] = {
            'policies': {
                'action_level_agent':
                    PolicySpec(
                        policy_class=None,
                        observation_space=Dict({'observation': env.observation_space['observation'],
                                                'interval': Box(low=env.action_space.low[0],
                                                                high=env.action_space.high[0], shape=(2,),
                                                                dtype=np.float32)}),
                        action_space=env.action_space,
                        config=None
                    ),
                'interval_level_agent':
                    PolicySpec(
                        policy_class=None,
                        observation_space=Dict(
                            {'observation': Dict({'observation': env.observation_space['observation'],
                                                  'allowed_actions': Box(low=env.action_space.low[0],
                                                                         high=env.action_space.high[0],
                                                                         shape=(MAX_PADDING_LEN, 2),
                                                                         dtype=np.float32)}),
                             'allowed_actions': MultiBinary(8)}),
                        action_space=Discrete(8),
                        config={'model': {
                            'custom_model': IntervalLevelModel,
                            'custom_model_config': hierarchical_config
                        }}
                    )
            },
            'policy_mapping_fn': policy_mapping_fn
        }

    # Replace default hyperparameters with search space
    if search_space is not None:
        for key, value in search_space.items():
            run_config[key] = value

    ray.init()
    timestamp = datetime.now().strftime('%m.%d.%Y_%H:%M:%S')
    algorithm_class = get_algorithm(algorithm)
    for seed in training_config['SEEDS']:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        if search_space is None:
            run_config['env_config']['RANDOM_SEED'] = seed
            run_config['seed'] = seed
        if search_space is not None:
            search_alg = HyperOptSearch(metric='episode_reward_mean', mode='max')
            search_alg = Repeater(searcher=search_alg, repeat=4)
        else:
            search_alg = BasicVariantGenerator()
        async_hyperband = ASHAScheduler(metric='episode_reward_mean', mode='max', max_t=20000,
                                        grace_period=10, reduction_factor=2)
        analysis = ray.tune.run(algorithm_class,
                                name=training_config['NAME'],
                                config=run_config,
                                stop={'num_env_steps_sampled': training_config['TRAINING_ITERATIONS']},
                                num_samples=training_config['NUM_SAMPLES'] if search_space is not None else 1,
                                scheduler=async_hyperband,
                                search_alg=search_alg,
                                log_to_file=False,
                                local_dir=training_config['LOCAL_DIR'],
                                checkpoint_at_end=True,
                                verbose=training_config['VERBOSE'],
                                raise_on_failed_trial=False)

        best_trial = analysis.get_best_trial('episode_reward_mean', mode='max')
        try:
            checkpoints = analysis.get_trial_checkpoints_paths(
                trial=best_trial,
                metric='episode_reward_mean')
            print(f' * Path to best model: {checkpoints[0][0]}')
        except Exception:
            print('No checkpoint found!')

        try:
            make_dir_if_not_exists(training_config['CONFIGURATION_PATH'])
            configs_full_path = os.path.join(training_config['CONFIGURATION_PATH'], f'{algorithm}_{timestamp}.pkl')
            pickle.dump({'algorithm': algorithm, 'config': best_trial.config}, open(configs_full_path, "wb"))
            print(f' * Configuration saved to {configs_full_path}')
        except Exception:
            print('Best config in trial not found!')

        if search_space is None:
            all_trials = analysis.trial_dataframes
            dataframe_of_results = all_trials[os.path.dirname(checkpoints[0][0])]
            dataframe_of_results['algorithm_name'] = algorithm
            dataframe_of_results['algorithm'] = training_config['ALGORITHM_NAME']
            results.append(dataframe_of_results)
            pd.concat(results).reset_index(drop=True).to_pickle(results_full_path)
            print(f" * Results saved to {training_config['TRAINING_RESULTS_PATH']}")

            experiments = pd.concat([experiments, pd.DataFrame({'algorithm_name': [training_config['ALGORITHM_NAME']],
                                                                'config': [configs_full_path],
                                                                'model': [checkpoints[0][0]]})])
            experiments.to_pickle(training_config['EXPERIMENTS_PATH'])

        if search_space is not None:
            make_dir_if_not_exists(training_config['HPO_RESULTS_PATH'])
            hpo_results_full_path = os.path.join(training_config['HPO_RESULTS_PATH'], f'{algorithm}_{timestamp}.pkl')
            pickle.dump(analysis.results_df, open(hpo_results_full_path, "wb"))
            print(f' * Best trial config: {best_trial.config}')
            print(f' * Best trial episode reward mean: {best_trial.last_result["episode_reward_mean"]}')
            print(f' * All trials saved to {hpo_results_full_path}')


def test_agent():
    """ Test an agent
    """
    experiments = pd.read_pickle(training_config['EXPERIMENTS_PATH'])
    experiments = experiments[experiments['algorithm_name'] == evaluation_config['ALGORITHM_NAME']]

    alg_num = 0
    for index, experiment in experiments.iterrows():
        config = load_pickle(experiment['config'])
        config['config']['num_gpus'] = 0
        agent = load_agent(config, experiment['model'])

        evaluate(config['config']['env'],
                 evaluation_config['ENV_CONFIG'],
                 agent,
                 evaluation_config['ALGORITHM_NAME'],
                 evaluation_config['SEEDS'],
                 evaluation_config['OBSTACLES'],
                 evaluation_config['EPISODE_RESULTS_PATH'],
                 evaluation_config['STEP_DATA_PATH'], render=False, record=evaluation_config['RECORD'],
                 alg_num=alg_num)
        print(f' * Episode finished')
        alg_num += 1
