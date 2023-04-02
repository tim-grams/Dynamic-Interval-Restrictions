from ray import tune

from configs.search_spaces.dqn_space import dqn_search_space
from src.algorithms.exploration.exploration import DynamicIntervalEpsilonGreedy

pam_search_space = {
    **dqn_search_space,
    'lr': 0.0,
    'hiddens': [],
    'dueling': tune.choice([True, False]),
    'double_q': tune.choice([True, False]),
    'exploration_config': {
        'type': DynamicIntervalEpsilonGreedy,
        "initial_epsilon": tune.uniform(lower=0.1, upper=0.8),
        "final_epsilon": tune.uniform(lower=0.005, upper=0.1),
        "epsilon_timesteps": tune.randint(lower=1000, upper=50000)
    },
    'replay_buffer_config': {
        'capacity': tune.randint(lower=200, upper=20000),
        'prioritized_replay_alpha': tune.uniform(lower=0.3, upper=0.7),
        'prioritized_replay_beta': tune.uniform(lower=0.2, upper=0.5)
    },
    'q_hiddens': tune.choice(
        ([256, 256, 128], [256, 128, 64], [128, 64, 64], [128, 128], [256, 128], [256, 256], [512, 256],
         [256, 512], [256], [512], [128], [64])),
    'p_hiddens': tune.choice(
        ([256, 256, 128], [256, 128, 64], [128, 64, 64], [128, 128], [256, 128], [256, 256], [512, 256],
         [256, 512], [256], [512], [128], [64])),
    'q_lr': tune.uniform(lower=5e-08, upper=1e-3),
    'p_lr': tune.uniform(lower=5e-08, upper=1e-3),
    'tau': tune.uniform(lower=1e-3, upper=0.2),
}
