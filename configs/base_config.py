from configs.envs.obstacle_avoidance.experiment_1 import experiment_1_setup
from configs.models.experiment_1.ppo_random_config import ppo_random_config

MAX_PADDING_LEN = 8

training_config = {
    'NAME': 'models',
    'HYPERPARAMETERS': ppo_random_config,
    'ALGORITHM_NAME': 'PPO-Random',
    'ENV_CONFIG': experiment_1_setup,
    'TRAINING_ITERATIONS': 50000,
    'LOCAL_DIR': 'results/experiment_1',
    'NUM_GPUS': 0,
    'NUM_WORKERS': 2,
    'NUM_SAMPLES': 1200,
    'SEEDS': [40, 41, 42, 43, 44, 45],
    'VERBOSE': 2,
    'NUM_DISCRETIZATION_BINS': 9,
    'HPO_RESULTS_PATH': 'results/experiment_1/hpo',
    'TRAINING_RESULTS_PATH': 'results/experiment_1',
    'CONFIGURATION_PATH': 'results/experiment_1/configurations',
    'EXPERIMENTS_PATH': 'results/experiment_1/experiments.pkl'
}

evaluation_config = {
    'ALGORITHM_NAME': 'MPS-TD3',
    'ENV_CONFIG': experiment_1_setup,
    'SEEDS': list(range(0, 40)),
    'OBSTACLES': [14],
    'EXPERIMENTS_PATH': 'results/experiment_1/experiments.pkl',
    'EPISODE_RESULTS_PATH': 'results/experiment_1/complex_evaluation/episode_results.pkl',
    'STEP_DATA_PATH': 'results/experiment_1/complex_evaluation/step_results.pkl',
    'RECORD': None
}
